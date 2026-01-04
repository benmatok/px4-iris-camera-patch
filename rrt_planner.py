
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

class GradientController:
    """
    Refines the trajectory using Gradient Descent (Levenberg-Marquardt) on the
    Chebyshev coefficients, using the simulation as the forward model and
    Finite Differences for gradient estimation.
    """
    def __init__(self, main_env, oracle, horizon_steps=10, iterations=3):
        self.main_env = main_env
        self.oracle = oracle
        self.horizon_steps = horizon_steps
        self.iterations = iterations
        self.dt = 0.05

        # Action space: 4 dims. Degree: 2 (Quadratic) -> 3 coeffs.
        self.action_dim = 4
        self.degree = 2
        self.num_params = self.action_dim * 3 # 12 parameters

        self.cheb_future = Chebyshev(horizon_steps, self.degree, device='cpu')

        self.num_main_agents = main_env.num_agents

        # We need 1 base + num_params perturbations per agent
        self.k_sims = 1 + self.num_params # 13
        self.total_sim_agents = self.num_main_agents * self.k_sims

        logging.info(f"Initializing GradientController with {self.total_sim_agents} sim slots ({self.iterations} iters)...")

        # Create Sim Env
        self.max_sim_steps = main_env.episode_length + horizon_steps + 10
        self.sim_env = DroneEnv(num_agents=self.total_sim_agents, episode_length=self.max_sim_steps)
        self.sim_env.reset_all_envs()

        # Perturbation scale for Finite Difference
        self.epsilon = 0.01

        # Damping for LM
        self.lambda_damping = 0.1

    def plan(self, current_state_dict, current_obs, current_traj_params, t_start):
        """
        Plans using Gradient Refinement.
        """
        # 1. Oracle Initialization
        oracle_actions, oracle_planned_pos, oracle_planned_att = self.oracle.compute_trajectory(
            current_traj_params, t_start, self.horizon_steps, current_state_dict
        )

        # Oracle Coeffs: (N, 4, 3)
        oracle_actions_torch = torch.from_numpy(oracle_actions).float()
        current_coeffs = self.cheb_future.fit(oracle_actions_torch) # (N, 4, 3)
        current_coeffs = current_coeffs.view(self.num_main_agents, 12)

        # ---------------------------------------------------------------------
        # SCANNING CHECK
        # ---------------------------------------------------------------------
        conf = current_obs[:, 307]
        lost_mask = conf < 0.1
        scanning_coeffs = None

        if np.any(lost_mask):
             # Calculate Scanning Coeffs (Same as RRT)
            g = 9.81
            masses = current_state_dict['masses']
            thrust_coeffs = current_state_dict['thrust_coeffs']
            hover_thrust = (g * masses) / (20.0 * thrust_coeffs)
            hover_thrust = np.clip(hover_thrust, 0.0, 1.0)

            t_grid = t_start + np.arange(self.horizon_steps) * self.dt
            scan_yaw_rate = np.sin(t_grid) + t_grid * np.cos(t_grid)
            scan_yaw_rate = np.clip(scan_yaw_rate, -2.0, 2.0)

            scan_actions = np.zeros_like(oracle_actions)
            scan_actions[:, 0, :] = hover_thrust[:, np.newaxis]
            scan_actions[:, 1, :] = 0.0
            scan_actions[:, 2, :] = 0.0
            scan_actions[:, 3, :] = scan_yaw_rate[np.newaxis, :]

            scan_actions_torch = torch.from_numpy(scan_actions).float()
            scanning_coeffs = self.cheb_future.fit(scan_actions_torch).view(self.num_main_agents, 12)

        # ---------------------------------------------------------------------
        # OPTIMIZATION LOOP
        # ---------------------------------------------------------------------
        # Target Trajectory for residuals:
        # 1. Position: Oracle Planned Pos (N, Steps, 3)
        # 2. Attitude: Oracle Planned Att (N, Steps, 3)
        # Combine: (N, Steps, 6) -> (N, Steps*6)

        # Weight for Attitude Residuals
        w_att = 5.0

        Y_pos_tgt = torch.from_numpy(oracle_planned_pos).float() # (N, S, 3)
        Y_att_tgt = torch.from_numpy(oracle_planned_att).float() # (N, S, 3)

        # Flatten
        Y_pos_tgt_flat = Y_pos_tgt.reshape(self.num_main_agents, -1)
        Y_att_tgt_flat = Y_att_tgt.reshape(self.num_main_agents, -1) * w_att

        Y_target = torch.cat([Y_pos_tgt_flat, Y_att_tgt_flat], dim=1) # (N, S*6)

        for itr in range(self.iterations):
            # 1. Expand Coeffs (N, 13, 12)
            # Slot 0: Base
            # Slot 1..12: Base + epsilon * e_j

            coeffs_base = current_coeffs.clone() # (N, 12)
            coeffs_expanded = coeffs_base.unsqueeze(1).repeat(1, self.k_sims, 1) # (N, 13, 12)

            # Apply perturbations
            # We want coeffs_expanded[i, j+1, j] += epsilon
            # Create identity perturbation matrix (12, 12) scaled by eps
            perturbations = torch.eye(12) * self.epsilon
            # Broadcast to (N, 12, 12)
            coeffs_expanded[:, 1:, :] += perturbations.unsqueeze(0)

            # 2. Simulate
            # Flatten to (N*13, 12) -> (N*13, 4, 3)
            sim_coeffs = coeffs_expanded.view(-1, 4, 3)
            sim_controls = self.cheb_future.evaluate(sim_coeffs)

            # Clip
            sim_controls[:, 0, :] = torch.clamp(sim_controls[:, 0, :], 0.0, 1.0)
            sim_controls[:, 1:, :] = torch.clamp(sim_controls[:, 1:, :], -10.0, 10.0)

            sim_controls_np = sim_controls.numpy()

            # Sync Sim Env
            self._sync_state(current_state_dict, current_traj_params)

            # Rollout
            rollout_pos = []
            rollout_att = []

            for step in range(self.horizon_steps):
                step_actions = sim_controls_np[:, :, step]
                self.sim_env.data_dictionary['actions'][:] = step_actions.reshape(-1)

                step_args = self._get_step_args(self.sim_env)
                self.sim_env.step_function(**step_args)

                px = self.sim_env.data_dictionary['pos_x'].copy()
                py = self.sim_env.data_dictionary['pos_y'].copy()
                pz = self.sim_env.data_dictionary['pos_z'].copy()
                rollout_pos.append(np.stack([px, py, pz], axis=1))

                r = self.sim_env.data_dictionary['roll'].copy()
                p = self.sim_env.data_dictionary['pitch'].copy()
                y = self.sim_env.data_dictionary['yaw'].copy()
                rollout_att.append(np.stack([r, p, y], axis=1))

            # Pos: (Steps, N*13, 3) -> (N*13, Steps*3)
            rollout_pos = np.stack(rollout_pos, axis=0)
            rollout_pos = rollout_pos.transpose(1, 0, 2).reshape(self.total_sim_agents, -1)

            # Att: (Steps, N*13, 3) -> (N*13, Steps*3)
            rollout_att = np.stack(rollout_att, axis=0)
            rollout_att = rollout_att.transpose(1, 0, 2).reshape(self.total_sim_agents, -1)

            # Combine to Y_sim (matching Y_target structure)
            # Need to process Att to match Target Att (Yaw wrapping) before simple subtraction?
            # Actually, we do residuals = Y_base - Y_target.
            # Y_sim should just be the raw values.
            # However, for Jacobian estimation (perturbation), raw values are fine as long as perturbation is small.
            # But for the Base Residual, we need to handle wrapping.

            # Let's assemble Y_sim as concatenation first
            Y_pos_sim_t = torch.from_numpy(rollout_pos).float() # (N*13, S*3)
            Y_att_sim_t = torch.from_numpy(rollout_att).float() * w_att # (N*13, S*3)

            # Reshape to (N, 13, M)
            Y_pos_sim = Y_pos_sim_t.view(self.num_main_agents, self.k_sims, -1)
            Y_att_sim = Y_att_sim_t.view(self.num_main_agents, self.k_sims, -1)

            # 3. Compute Jacobian & Residuals

            # Position Part
            Y_pos_base = Y_pos_sim[:, 0, :]
            res_pos = Y_pos_base - Y_pos_tgt_flat

            # Attitude Part
            # We need to handle Yaw wrapping for Residuals
            # Y_att_sim structure: [r0, p0, y0, r1, p1, y1, ...]
            # We can just subtract and then wrap the specific indices?
            # Or simpler: Unpack, wrap, Repack.

            # Target Att (unflattened for math): (N, S, 3)
            # Base Att (unflattened): (N, S, 3)
            att_base_raw = Y_att_sim[:, 0, :].view(self.num_main_agents, self.horizon_steps, 3) / w_att
            att_tgt_raw = Y_att_tgt # (N, S, 3)

            diff_att = att_base_raw - att_tgt_raw
            # Wrap Yaw (Index 2)
            diff_att[:, :, 2] = (diff_att[:, :, 2] + np.pi) % (2 * np.pi) - np.pi

            # Flatten and Apply Weight
            res_att = diff_att.view(self.num_main_agents, -1) * w_att

            # Combined Residuals
            residuals = torch.cat([res_pos, res_att], dim=1) # (N, M_pos + M_att)

            # Jacobian Calculation
            # J = (Y_perturbed - Y_base) / epsilon

            # For Attitude Jacobian, simple difference is usually fine for small epsilon,
            # assuming we don't cross the wrap boundary with epsilon.
            # Epsilon is small (0.01).

            Y_sim_combined = torch.cat([Y_pos_sim, Y_att_sim], dim=2)

            Y_base_combined = Y_sim_combined[:, 0, :]
            Y_perturbed_combined = Y_sim_combined[:, 1:, :]

            # Note: residuals calculated above use wrapped yaw difference.
            # The Jacobian uses direct difference of perturbed vs base.
            # This is correct because locally J = d(Obs)/d(Action).

            J = (Y_perturbed_combined - Y_base_combined.unsqueeze(1)) / self.epsilon

            # Jacobian: J = (Y_perturbed - Y_base) / epsilon
            # Y_perturbed: (N, 12, M) -> indices 1..12
            Y_perturbed = Y_sim[:, 1:, :]

            J = (Y_perturbed - Y_base.unsqueeze(1)) / self.epsilon # (N, 12, M)
            # We need J in shape (N, M, 12) for standard notation J*delta = -r
            J = J.transpose(1, 2) # (N, M, 12)

            # 4. Levenberg-Marquardt Update
            # delta = -(J^T J + lambda I)^-1 J^T r

            # J_T: (N, 12, M)
            J_T = J.transpose(1, 2)

            # J^T J: (N, 12, 12)
            JTJ = torch.bmm(J_T, J)

            # Damping
            diag_idx = torch.arange(12)
            JTJ[:, diag_idx, diag_idx] += self.lambda_damping

            # J^T r: (N, 12, 1)
            # r: (N, M) -> (N, M, 1)
            JTr = torch.bmm(J_T, residuals.unsqueeze(2))

            # Solve linear system
            # Use torch.linalg.solve(A, B) -> X
            try:
                # delta: (N, 12, 1)
                delta = torch.linalg.solve(JTJ, -JTr)
                delta = delta.squeeze(2) # (N, 12)

                # Update
                current_coeffs = current_coeffs + delta

                # Check convergence? (Optional, skipping for fixed iter)

            except RuntimeError as e:
                logging.warning(f"LM Solve failed: {e}. Skipping update.")
                break

        # ---------------------------------------------------------------------
        # FINALIZE
        # ---------------------------------------------------------------------
        # Apply lost mask
        if np.any(lost_mask):
            # Replace coeffs for lost agents
            current_coeffs[lost_mask] = scanning_coeffs[lost_mask]

        return current_coeffs


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
