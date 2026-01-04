
import numpy as np
import copy
import torch
import logging
from drone_env.drone import DroneEnv
from models.predictive_policy import Chebyshev

# Import the new helper if available
try:
    from drone_env.drone_cython import update_target_trajectory_from_params
    _HAS_CYTHON_HELPER = True
except ImportError:
    _HAS_CYTHON_HELPER = False
    logging.warning("update_target_trajectory_from_params not found in Cython ext. Falling back to what?")

class RRTController:
    """
    Simulates forward using a shooting method (variant of RRT) to find the control sequence
    (Chebyshev coefficients) that minimizes deviation from the Optimal/Oracle trajectory
    over a short horizon (0.5s).
    """
    def __init__(self, main_env, oracle, horizon_steps=10, num_samples=50):
        self.main_env = main_env
        self.oracle = oracle
        self.horizon_steps = horizon_steps
        self.num_samples = num_samples
        self.dt = 0.05

        # Action space: 4 dims. Degree: 2 (Quadratic) -> 3 coeffs.
        self.action_dim = 4
        self.degree = 2
        self.cheb_future = Chebyshev(horizon_steps, self.degree, device='cpu')

        self.num_main_agents = main_env.num_agents
        self.total_sim_agents = self.num_main_agents * self.num_samples

        logging.info(f"Initializing RRT Planner with {self.total_sim_agents} simulation slots...")

        # Create Sim Env with sufficient buffer
        # We need to handle indices up to episode_length of the main env + horizon.
        # Main env max step is 100.
        self.max_sim_steps = main_env.episode_length + horizon_steps + 10
        self.sim_env = DroneEnv(num_agents=self.total_sim_agents, episode_length=self.max_sim_steps)
        self.sim_env.reset_all_envs()

    def plan(self, current_state_dict, current_obs, current_traj_params, t_start):
        """
        Plans for all agents in parallel.
        Returns: Best Action Coefficients (N, 12)
        """
        # 1. Get Oracle Solution (The Mean / Guide)
        oracle_actions, oracle_planned_pos = self.oracle.compute_trajectory(
            current_traj_params, t_start, self.horizon_steps, current_state_dict
        )

        # ---------------------------------------------------------------------
        # SCANNING BEHAVIOR (Prepare Overrides)
        # ---------------------------------------------------------------------
        # Check Confidence
        conf = current_obs[:, 307]
        lost_mask = conf < 0.1 # Boolean array (num_agents,)

        # We will use this to override the FINAL selection.
        # But we also modify oracle_actions so that if we WERE to use RRT,
        # it would start from scanning. But RRT might drift away.
        # So we force the result for lost agents.

        # Calculate Scanning Coeffs
        scanning_coeffs = None

        if np.any(lost_mask):
            # Calculate Hover Thrust
            g = 9.81
            masses = current_state_dict['masses']
            thrust_coeffs = current_state_dict['thrust_coeffs']
            hover_thrust = (g * masses) / (20.0 * thrust_coeffs)
            hover_thrust = np.clip(hover_thrust, 0.0, 1.0)

            # Generate Scanning Yaw Rate
            t_grid = t_start + np.arange(self.horizon_steps) * self.dt # (Steps,)
            scan_yaw_rate = np.sin(t_grid) + t_grid * np.cos(t_grid)
            scan_yaw_rate = np.clip(scan_yaw_rate, -2.0, 2.0)

            # Create Scanning Actions tensor
            # (N, 4, Steps)
            scan_actions = np.zeros_like(oracle_actions)

            # Fill for all agents (we mask later) or just fill relevant?
            # Easier to fill all row-wise using broadcasting
            scan_actions[:, 0, :] = hover_thrust[:, np.newaxis] # Constant thrust
            scan_actions[:, 1, :] = 0.0
            scan_actions[:, 2, :] = 0.0
            scan_actions[:, 3, :] = scan_yaw_rate[np.newaxis, :] # Same pattern for all

            # Convert to Coeffs
            scan_actions_torch = torch.from_numpy(scan_actions).float()
            scanning_coeffs_all = self.cheb_future.fit(scan_actions_torch) # (N, 4, 3)
            scanning_coeffs = scanning_coeffs_all

        # ---------------------------------------------------------------------

        # Convert Oracle Actions to Coeffs
        oracle_actions_torch = torch.from_numpy(oracle_actions).float() # (N, 4, Steps)
        oracle_coeffs = self.cheb_future.fit(oracle_actions_torch) # (N, 4, 3)

        # 2. Sampling (Perturbation)
        coeffs_expanded = oracle_coeffs.unsqueeze(1).repeat(1, self.num_samples, 1, 1)

        noise = torch.randn_like(coeffs_expanded) * 0.5
        noise[:, :, 0, :] *= 0.2 # Less noise on thrust

        # Keep the first sample as pure Oracle
        noise[:, 0, :, :] = 0.0

        sampled_coeffs = coeffs_expanded + noise

        # Flatten to (N*K, 4, 3)
        sampled_coeffs_flat = sampled_coeffs.view(-1, 4, 3)

        # 3. Decode to Controls
        sampled_controls = self.cheb_future.evaluate(sampled_coeffs_flat)

        # Clip Controls
        sampled_controls[:, 0, :] = torch.clamp(sampled_controls[:, 0, :], 0.0, 1.0)
        sampled_controls[:, 1:, :] = torch.clamp(sampled_controls[:, 1:, :], -10.0, 10.0)

        sampled_controls_np = sampled_controls.numpy()

        # 4. Initialize Sim Env State
        self._sync_state(current_state_dict, current_traj_params)

        # 5. Rollout
        rollout_pos = []

        for step in range(self.horizon_steps):
            # Get actions for this step: (N*K, 4)
            step_actions = sampled_controls_np[:, :, step]

            # Apply to Sim Env
            self.sim_env.data_dictionary['actions'][:] = step_actions.reshape(-1)

            # Step
            step_args = self._get_step_args(self.sim_env)
            self.sim_env.step_function(**step_args)

            # Record Pos
            px = self.sim_env.data_dictionary['pos_x'].copy()
            py = self.sim_env.data_dictionary['pos_y'].copy()
            pz = self.sim_env.data_dictionary['pos_z'].copy()
            rollout_pos.append(np.stack([px, py, pz], axis=1))

        rollout_pos = np.stack(rollout_pos, axis=0) # (Steps, N*K, 3)

        # 6. Evaluate Cost
        rollout_pos_reshaped = rollout_pos.reshape(self.horizon_steps, self.num_main_agents, self.num_samples, 3)
        rollout_pos_reshaped = rollout_pos_reshaped.transpose(1, 2, 0, 3) # (N, K, Steps, 3)

        target_pos = oracle_planned_pos[:, np.newaxis, :, :]

        error = rollout_pos_reshaped - target_pos
        mse = np.mean(np.sum(error**2, axis=3), axis=2) # (N, K)

        # 7. Select Best
        best_indices = np.argmin(mse, axis=1) # (N,)

        best_coeffs = []
        for i in range(self.num_main_agents):
            if lost_mask[i]:
                # Force Scanning Coeffs
                best_coeffs.append(scanning_coeffs[i])
            else:
                idx = best_indices[i]
                best_coeffs.append(sampled_coeffs[i, idx])

        best_coeffs = torch.stack(best_coeffs) # (N, 4, 3)

        return best_coeffs.view(self.num_main_agents, -1)

    def _sync_state(self, main_state, traj_params):
        """
        Copies N states to N*K states in sim_env
        """
        K = self.num_samples
        N = self.num_main_agents

        # Keys to copy
        keys = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                'roll', 'pitch', 'yaw', 'masses', 'drag_coeffs', 'thrust_coeffs']

        for k in keys:
            val = main_state[k] # (N,)
            expanded = np.tile(val[:, np.newaxis], (1, K)).reshape(-1)
            self.sim_env.data_dictionary[k][:] = expanded

        # Traj Params (10, N) -> (10, N*K)
        tp = traj_params # (10, N)
        tp_expanded = np.repeat(tp, K, axis=1) # (10, N*K)
        self.sim_env.data_dictionary['traj_params'][:] = tp_expanded

        # Sync Step Counts (t)
        # main_env.step_counts is likely (N,) or (1,) depending on implementation details
        # We need to set sim_env.step_counts (N*K,) to the same value.
        # Just grab the first value.
        current_step = self.main_env.data_dictionary['step_counts'][0]
        self.sim_env.data_dictionary['step_counts'][:] = current_step

        # UPDATE TARGET TRAJECTORY BUFFER
        # Since traj_params might be different (for different agents),
        # we must regenerate the target trajectory buffer for the sim env.
        # (Although here traj_params are just expanded, so we could optimize if we knew that,
        # but for robustness we recompute).
        if _HAS_CYTHON_HELPER:
             # This runs in parallel in C++
             # target_trajectory shape: (episode_length, num_agents, 3)
             target_traj = self.sim_env.data_dictionary['target_trajectory']
             tp = self.sim_env.data_dictionary['traj_params']
             num_agents = self.total_sim_agents
             steps = target_traj.shape[0]

             update_target_trajectory_from_params(tp, target_traj, num_agents, steps)
        else:
             # Slow Python fallback (should not happen if compiled)
             # Not implementing full fallback for speed reasons, assume cython works
             pass

    def _get_step_args(self, env):
        # Helper
        step_kwargs = env.get_step_function_kwargs()
        step_args = {}
        for k, v in step_kwargs.items():
            if v in env.data_dictionary:
                step_args[k] = env.data_dictionary[v]
            elif k == "num_agents":
                step_args[k] = env.num_agents
            elif k == "episode_length":
                step_args[k] = env.episode_length
        return step_args
