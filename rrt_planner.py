
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
    logging.warning("update_target_trajectory_from_params not found in Cython ext. Falling back to python.")

class AggressiveOracle:
    """
    Aggressive Oracle Specification (v1.2) - Trajectory Optimizer
    Generates "expert" reference trajectories prioritizing high-speed interception
    and avoiding local minima (hovering) through aggressive potential field shaping.
    """
    def __init__(self, main_env, horizon_steps=10, iterations=5):
        self.main_env = main_env
        self.horizon_steps = horizon_steps # T = 10 steps (0.5s)
        self.iterations = iterations # <5 iterations recommended
        self.dt = 0.05

        # Action space: 4 dims. Degree: 4 (Quartic) -> 5 coeffs.
        self.action_dim = 4
        self.degree = 4
        self.num_params = self.action_dim * (self.degree + 1) # 20 parameters

        self.cheb_future = Chebyshev(horizon_steps, self.degree, device='cpu')

        self.num_main_agents = main_env.num_agents

        # Finite Difference Setup
        # We need 1 base + num_params perturbations per agent
        self.k_sims = 1 + self.num_params # 21
        self.total_sim_agents = self.num_main_agents * self.k_sims

        logging.info(f"Initializing AggressiveOracle with {self.total_sim_agents} sim slots ({self.iterations} iters)...")

        # Create Sim Env
        self.max_sim_steps = main_env.episode_length + horizon_steps + 10
        self.sim_env = DroneEnv(num_agents=self.total_sim_agents, episode_length=self.max_sim_steps)
        self.sim_env.reset_all_envs()

        # Perturbation scale for Finite Difference
        self.epsilon = 0.01

        # Optimization Step Size
        self.learning_rate = 0.05

        # Weights & Constants
        self.W_near = 2.0
        self.W_far = 4.0
        self.W_shark = 10.0
        self.W_anchor = 10.0
        self.W_jerk = 2.5
        self.V_min = 1.0
        self.W_terminal_vel = 2.0 # New weight for terminal velocity matching

        # Persistent Coeffs for Warm Start (num_agents, num_params)
        # Initialize with None
        self.previous_coeffs = None

    def plan(self, current_state_dict, current_obs, current_traj_params, t_start):
        """
        Plans using Gradient Descent on the Cost Function.
        """
        # Auto-reset warm start if we are at the beginning of an episode
        if t_start < self.dt:
             self.previous_coeffs = None

        # 1. Initialization / Warm Start
        if self.previous_coeffs is None:
            # Cold Start: Initialize with zero actions (Hover/Fall) or a guess?
            # Better to init with hover thrust.

            # Hover Thrust Guess
            g = 9.81
            masses = current_state_dict['masses']
            thrust_coeffs = current_state_dict['thrust_coeffs']
            hover_thrust = (g * masses) / (20.0 * thrust_coeffs)
            hover_thrust = np.clip(hover_thrust, 0.0, 1.0)

            init_actions = np.zeros((self.num_main_agents, 4, self.horizon_steps))
            init_actions[:, 0, :] = hover_thrust[:, np.newaxis]

            # Fit Chebyshev
            init_actions_torch = torch.from_numpy(init_actions).float()
            current_coeffs = self.cheb_future.fit(init_actions_torch).view(self.num_main_agents, self.num_params)
        else:
            # Warm Start: Reuse previous coefficients
            current_coeffs = self.previous_coeffs.clone()

        # ---------------------------------------------------------------------
        # OPTIMIZATION LOOP
        # ---------------------------------------------------------------------

        # Target Trajectory State for Cost Calculation
        target_pos_horizon, target_vel_horizon = self._compute_target_trajectory(current_traj_params, t_start, self.horizon_steps)
        # Shape: (N, Steps, 3)

        target_pos_horizon_torch = torch.from_numpy(target_pos_horizon).float()
        target_vel_horizon_torch = torch.from_numpy(target_vel_horizon).float()

        # Terminal Target (for Anchor)
        # target_pos_T = target_pos_horizon_torch[:, -1, :] # (N, 3) # Unused var

        for itr in range(self.iterations):
            # 1. Expand Coeffs (N, K, num_params)
            # Slot 0: Base
            # Slot 1..N: Base + epsilon * e_j

            coeffs_base = current_coeffs.clone() # (N, num_params)
            coeffs_expanded = coeffs_base.unsqueeze(1).repeat(1, self.k_sims, 1) # (N, K, num_params)

            # Apply perturbations
            perturbations = torch.eye(self.num_params) * self.epsilon
            # Broadcast
            coeffs_expanded[:, 1:, :] += perturbations.unsqueeze(0)

            # 2. Simulate
            # Flatten to (N*K, num_params) -> (N*K, 4, degree+1)
            sim_coeffs = coeffs_expanded.view(-1, 4, self.degree + 1)
            sim_controls = self.cheb_future.evaluate(sim_coeffs)

            # Clip Controls (Actuator Constraints)
            sim_controls[:, 0, :] = torch.clamp(sim_controls[:, 0, :], 0.0, 1.0) # Thrust
            sim_controls[:, 1:, :] = torch.clamp(sim_controls[:, 1:, :], -10.0, 10.0) # Rates

            sim_controls_np = sim_controls.numpy()

            # Sync Sim Env
            self._sync_state(current_state_dict, current_traj_params)

            # Rollout
            rollout_pos = []
            rollout_vel = [] # Needed for Shark

            # We also need controls for Smoothness cost (J_jerk)
            # controls are sim_controls (already known)

            for step in range(self.horizon_steps):
                step_actions = sim_controls_np[:, :, step]
                self.sim_env.data_dictionary['actions'][:] = step_actions.reshape(-1)

                step_args = self._get_step_args(self.sim_env)
                self.sim_env.step_function(**step_args)

                px = self.sim_env.data_dictionary['pos_x'].copy()
                py = self.sim_env.data_dictionary['pos_y'].copy()
                pz = self.sim_env.data_dictionary['pos_z'].copy()
                rollout_pos.append(np.stack([px, py, pz], axis=1))

                vx = self.sim_env.data_dictionary['vel_x'].copy()
                vy = self.sim_env.data_dictionary['vel_y'].copy()
                vz = self.sim_env.data_dictionary['vel_z'].copy()
                rollout_vel.append(np.stack([vx, vy, vz], axis=1))

            # Pos: (Steps, N*K, 3) -> (N, K, Steps, 3)
            rollout_pos = np.stack(rollout_pos, axis=0).transpose(1, 0, 2)
            rollout_pos = rollout_pos.reshape(self.num_main_agents, self.k_sims, self.horizon_steps, 3)

            # Vel: (N, K, Steps, 3)
            rollout_vel = np.stack(rollout_vel, axis=0).transpose(1, 0, 2)
            rollout_vel = rollout_vel.reshape(self.num_main_agents, self.k_sims, self.horizon_steps, 3)

            # Convert to Torch
            P_t = torch.from_numpy(rollout_pos).float() # (N, K, S, 3)
            V_t = torch.from_numpy(rollout_vel).float() # (N, K, S, 3)
            U_t = sim_controls.view(self.num_main_agents, self.k_sims, 4, self.horizon_steps).permute(0, 1, 3, 2) # (N, K, S, 4)

            # Target Pos needs expansion: (N, S, 3) -> (N, K, S, 3)
            P_target = target_pos_horizon_torch.unsqueeze(1).expand(-1, self.k_sims, -1, -1)
            V_target = target_vel_horizon_torch.unsqueeze(1).expand(-1, self.k_sims, -1, -1)

            # -----------------------------------------------------------------
            # COST CALCULATION
            # -----------------------------------------------------------------

            # A. Gravity Well (Pos Cost)
            # J_pos = ||Pt - Ptarget|| * (W_near + W_far * log(1 + ||Pt - Ptarget||))
            diff = P_t - P_target
            dist = torch.norm(diff, dim=3) # (N, K, S)
            J_pos = dist * (self.W_near + self.W_far * torch.log(1 + dist))

            # B. Shark Constraint (Velocity Floor)
            # Normalized direction to target
            dist_safe = dist + 1e-6
            dir_to_target = diff / dist_safe.unsqueeze(3) # (N, K, S, 3)

            # V_closing = sum(V_t * (-dir), dim=3)
            V_closing = torch.sum(V_t * (-dir_to_target), dim=3)

            J_shark = self.W_shark * torch.relu(self.V_min - V_closing)

            # C. Terminal Anchor
            dist_T = dist[:, :, -1]
            J_anchor = self.W_anchor * dist_T

            # D. Smoothness (Jerk)
            if self.horizon_steps > 1:
                u_diff = U_t[:, :, 1:, :] - U_t[:, :, :-1, :]
                jerk_sq = torch.sum(u_diff**2, dim=3) # Sum over control dims -> (N, K, S-1)
                J_smooth = self.W_jerk * torch.sum(jerk_sq, dim=2) # Sum over time
            else:
                J_smooth = torch.zeros((self.num_main_agents, self.k_sims), device=dist.device)

            # E. Terminal Velocity Cost
            # Penalize velocity difference at the last step
            V_diff_T = V_t[:, :, -1, :] - V_target[:, :, -1, :]
            V_dist_T = torch.norm(V_diff_T, dim=2)
            J_terminal_vel = self.W_terminal_vel * V_dist_T

            # Total Cost
            # Sum instantaneous costs over time
            J_inst = torch.sum(J_pos + J_shark, dim=2) # (N, K)

            J_total = J_inst + J_anchor + J_smooth + J_terminal_vel # (N, K)

            # -----------------------------------------------------------------
            # GRADIENT ESTIMATION & UPDATE
            # -----------------------------------------------------------------

            # We have J_total for Base (idx 0) and Perturbed (idx 1..20)
            J_base = J_total[:, 0] # (N,)
            J_perturbed = J_total[:, 1:] # (N, num_params)

            # Gradient approx: dJ/dTheta_i = (J(theta + eps*e_i) - J(theta)) / eps
            grad_J = (J_perturbed - J_base.unsqueeze(1)) / self.epsilon # (N, num_params)

            # Clamp gradients to avoid explosions
            grad_J = torch.clamp(grad_J, -10.0, 10.0)

            # Update: Theta = Theta - lr * grad
            current_coeffs = current_coeffs - self.learning_rate * grad_J

        # Save for next time
        self.previous_coeffs = current_coeffs.detach()

        return current_coeffs

    def _compute_target_trajectory(self, traj_params, t_start, steps):
        """
        Computes Target Position for the horizon analytically.
        Matches OracleController logic.
        """
        t_out = np.arange(steps) * self.dt + t_start
        # t_out shape: (steps,)

        # Params: (10, N)
        params = traj_params[:, :, np.newaxis] # (10, N, 1)
        t = t_out[np.newaxis, :] # (1, steps)

        # Convert to steps for formula (assuming params are defined per step or sec?
        # In train_jules it converts: t_steps = t / dt)
        t_in_steps = t / self.dt

        freq_scale = 1.0 / self.dt

        Ax, Fx, Px = params[0], params[1], params[2]
        Ay, Fy, Py = params[3], params[4], params[5]
        Az, Fz, Pz, Oz = params[6], params[7], params[8], params[9]

        x = Ax * np.sin(Fx * t_in_steps + Px)
        y = Ay * np.sin(Fy * t_in_steps + Py)
        z = Oz + Az * np.sin(Fz * t_in_steps + Pz)

        # Velocity
        vx = Ax * Fx * freq_scale * np.cos(Fx * t_in_steps + Px)
        vy = Ay * Fy * freq_scale * np.cos(Fy * t_in_steps + Py)
        vz = Az * Fz * freq_scale * np.cos(Fz * t_in_steps + Pz)

        # (N, Steps, 3)
        pos = np.stack([x, y, z], axis=2)
        vel = np.stack([vx, vy, vz], axis=2)
        return pos, vel

    def _sync_state(self, main_state, traj_params):
        """
        Copies N states to N*K states in sim_env
        """
        K = self.k_sims
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

        # Sync Step Counts
        current_step = self.main_env.data_dictionary['step_counts'][0]
        self.sim_env.data_dictionary['step_counts'][:] = current_step

        # UPDATE TARGET TRAJECTORY BUFFER
        if _HAS_CYTHON_HELPER:
             target_traj = self.sim_env.data_dictionary['target_trajectory']
             tp = self.sim_env.data_dictionary['traj_params']
             num_agents = self.total_sim_agents
             steps = target_traj.shape[0]
             update_target_trajectory_from_params(tp, target_traj, num_agents, steps)

    def _get_step_args(self, env):
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
