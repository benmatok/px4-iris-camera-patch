import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

# Try to import step_cpu for GDPC
try:
    from drone_env.drone import step_cpu
    HAS_SIM = True
except ImportError:
    step_cpu = None
    HAS_SIM = False
    logger.warning("Could not import step_cpu from drone_env.drone. GDPC mode will not work.")

class DPCFlightController:
    def __init__(self, dt=0.05, mode='PID'):
        self.dt = dt
        self.mode = mode

        # PID Gains
        self.k_yaw = 2.0
        self.k_roll = 2.0

        # Velocity Control Gains
        self.kp_vz = 2.0
        self.thrust_hover = 0.6

        # FOE Bias Gains
        self.k_foe_yaw = 2.0
        self.k_foe_pitch = 1.0

        # State Variables
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.flight_phase = "DIVE" # DIVE, BRAKE, APPROACH

        # Blind Mode Estimation
        self.est_pz = 100.0 # Initial assumption
        self.est_vz = 0.0

        # GDPC / Estimator State
        self.estimated_params = {
            'mass': 1.0,
            'drag_coeff': 0.1,
            'thrust_coeff': 1.0
        }

        # GDPC Simulation Buffer (Allocated on first use)
        self.gdpc_buffer = None
        self.num_ghosts = 20
        self.horizon = 20

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.est_pz = 100.0
        self.est_vz = 0.0
        self.flight_phase = "DIVE"
        logger.info(f"DPCFlightController reset. Mode: {self.mode}")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, extra_yaw_rate=0.0, foe_uv=None):
        if self.mode == 'GDPC' and HAS_SIM:
            return self.compute_action_gdpc(state_obs, target_cmd, tracking_uv, foe_uv, extra_yaw_rate)
        return self._compute_heuristic_action(state_obs, target_cmd, tracking_uv, extra_yaw_rate, foe_uv)

    def _compute_heuristic_action(self, state_obs, target_cmd, tracking_uv=None, extra_yaw_rate=0.0, foe_uv=None):
        # Unpack State
        pz = state_obs.get('pz')
        vz = state_obs.get('vz')
        roll = state_obs['roll']
        pitch = state_obs['pitch']

        # Blind Mode Logic
        if pz is None:
            th = self.last_action['thrust']
            g = 9.81
            T_max_accel = 20.0
            accel_z = (th * T_max_accel) * (math.cos(roll) * math.cos(pitch)) - g
            accel_z -= 0.1 * self.est_vz
            self.est_vz += accel_z * self.dt
            self.est_pz += self.est_vz * self.dt
            if self.est_pz < 0.0:
                self.est_pz = 0.0
                self.est_vz = 0.0
            pz = self.est_pz
            vz = self.est_vz
        else:
            self.est_pz = pz
            self.est_vz = vz if vz is not None else 0.0

        goal_z = target_cmd[2]
        dx = target_cmd[0]
        dy = target_cmd[1]
        dist_est = math.sqrt(dx*dx + dy*dy)
        alt_est = max(0.1, pz - goal_z)
        dist_3d = math.sqrt(dist_est*dist_est + alt_est*alt_est)

        # Visual Distance Refinement
        if tracking_uv:
             _, v = tracking_uv
             tilt_rad = math.radians(30.0)
             angle_from_axis = math.atan(v)
             depression = angle_from_axis - (pitch + tilt_rad)
             if depression > 0.1:
                  dist_visual = alt_est / math.tan(depression)
                  dist_est = dist_visual
                  # Update 3D distance estimate roughly
                  dist_3d = math.sqrt(dist_est*dist_est + alt_est*alt_est)

        # --- STATE MACHINE ---
        v_total = math.sqrt(state_obs.get('vx', 0.0)**2 + state_obs.get('vy', 0.0)**2 + state_obs.get('vz', 0.0)**2)

        # State Transitions
        depression_angle = math.atan2(alt_est, dist_est)
        brake_dist = max(15.0, v_total * 2.5)

        if self.flight_phase == "DESCEND":
            # Exit DESCEND when angle is shallow enough (< 50 deg) - Hysteresis
            if depression_angle < math.radians(50.0):
                self.flight_phase = "DIVE"
                logger.info(f"Switching to DIVE phase (Angle: {math.degrees(depression_angle):.1f})")

        elif self.flight_phase == "DIVE":
            # Check for steep angle entry
            if depression_angle > math.radians(65.0):
                self.flight_phase = "DESCEND"
                logger.info(f"Switching to DESCEND phase (Angle: {math.degrees(depression_angle):.1f})")

            # Transition to BRAKE if close or too fast
            if dist_3d < brake_dist or v_total > 20.0:
                self.flight_phase = "BRAKE"
                logger.info(f"Switching to BRAKE phase (Dist3D: {dist_3d:.1f}, V: {v_total:.1f})")
        elif self.flight_phase == "BRAKE":
            # Transition to APPROACH when slow enough
            if v_total < 5.0:
                self.flight_phase = "APPROACH"
                logger.info(f"Switching to APPROACH phase (Dist: {dist_est:.1f}, V: {v_total:.1f})")

            # Reset to DIVE if distance is large (e.g. false trigger or lost track)
            if dist_3d > brake_dist + 20.0:
                self.flight_phase = "DIVE"
                logger.info(f"Resetting to DIVE phase (Dist3D: {dist_3d:.1f})")

        elif self.flight_phase == "APPROACH":
             # Reset to DIVE if distance is large
            if dist_3d > 30.0:
                self.flight_phase = "DIVE"
                logger.info(f"Resetting to DIVE phase (Dist3D: {dist_3d:.1f})")

        # Control Logic per State
        gamma_ref = 0.0
        thrust_cmd = 0.5
        speed_limit = 10.0

        los = -math.atan2(alt_est, dist_est)

        if self.flight_phase == "DESCEND":
            # Slow Vertical Descent to reduce angle
            # Pitch 0 to minimize horizontal drift, Low Thrust to drop
            gamma_ref = math.radians(0.0)
            thrust_cmd = 0.45

        elif self.flight_phase == "DIVE":
            # Aggressive Dive
            # Bias dive angle to gain speed, but fade out near ground/shallow angles
            # Use depression angle to scale bias (prevent diving into ground on long approaches)
            deg_depression = math.degrees(depression_angle)
            dive_bias = min(12.0, deg_depression * 0.7)

            # Also fade by altitude (double safety)
            if alt_est < 20.0:
                 dive_bias *= (alt_est / 20.0)

            gamma_ref = los - math.radians(dive_bias)
            speed_limit = 20.0 # Allow speed
            thrust_cmd = 0.6 # Base thrust

        elif self.flight_phase == "BRAKE":
            # Braking Maneuver: Pitch Up + Low Thrust
            gamma_ref = math.radians(15.0)
            speed_limit = 0.0 # We want to stop
            thrust_cmd = 0.6 # Maintain lift (Hover)

        elif self.flight_phase == "APPROACH":
            # Precision Tracking (Linear LOS)
            gamma_ref = los
            speed_limit = 8.0 # Slow approach
            thrust_cmd = 0.5

        gamma_ref = max(gamma_ref, math.radians(-85.0))

        # Ground Safety Clamping for Gamma
        # Don't command a dive that intersects the ground closer than 10m ahead
        if alt_est < 20.0:
             # If shallow approach, flare early (20m lookahead) to avoid sagging
             if depression_angle < math.radians(30.0):
                 lookahead = 20.0
             else:
                 # If steep approach, aim closer (allow steeper dive)
                 lookahead = max(5.0, dist_est)

             safety_gamma = -math.atan2(alt_est, lookahead)
             gamma_ref = max(gamma_ref, safety_gamma)

        # 3. Speed Control (Thrust) - active in DIVE, APPROACH and DESCEND
        # In BRAKE, we override thrust manually above.
        if self.flight_phase != "BRAKE" and self.flight_phase != "DESCEND" and vz is not None:
            vz_cmd = speed_limit * math.sin(gamma_ref)
            vz_err = vz_cmd - vz
            thrust_cmd = self.thrust_hover + self.kp_vz * vz_err
            thrust_cmd = max(0.1, min(1.0, thrust_cmd))

        # 4. Pitch Control
        pitch_cmd = gamma_ref

        # Pitch Rate Loop
        k_pitch_ang = 5.0
        pitch_rate_cmd = k_pitch_ang * (pitch - pitch_cmd)

        # 1. Yaw Control
        yaw_rate_cmd = 0.0
        if tracking_uv:
            u, v = tracking_uv
            if foe_uv:
                foe_u, foe_v = foe_uv
                yaw_rate_cmd = self.k_foe_yaw * (u - foe_u)
            else:
                yaw_rate_cmd = self.k_yaw * u
        else:
            yaw_rate_cmd = extra_yaw_rate

        # 5. Roll Control
        roll_rate_cmd = self.k_roll * (0.0 - roll)

        action = {
            'thrust': thrust_cmd,
            'roll_rate': roll_rate_cmd,
            'pitch_rate': pitch_rate_cmd,
            'yaw_rate': yaw_rate_cmd
        }
        self.last_action = action

        # Ghost Paths (Viz)
        ghost_paths = []
        path = []
        sim_z = pz
        if vz is not None:
            sim_vz = vz
        else:
            sim_vz = 5.0 * math.sin(gamma_ref)

        for i in range(20):
             sim_z += sim_vz * self.dt
             path.append({'px': 0.0, 'py': 0.0, 'pz': sim_z})
        ghost_paths.append(path)

        return action, ghost_paths

    def _init_gdpc_buffer(self, num_agents):
        """Allocate buffers for vectorized simulation."""
        self.gdpc_buffer = {
            "pos_x": np.zeros(num_agents, dtype=np.float32),
            "pos_y": np.zeros(num_agents, dtype=np.float32),
            "pos_z": np.zeros(num_agents, dtype=np.float32),
            "vel_x": np.zeros(num_agents, dtype=np.float32),
            "vel_y": np.zeros(num_agents, dtype=np.float32),
            "vel_z": np.zeros(num_agents, dtype=np.float32),
            "roll": np.zeros(num_agents, dtype=np.float32),
            "pitch": np.zeros(num_agents, dtype=np.float32),
            "yaw": np.zeros(num_agents, dtype=np.float32),
            "ang_vel_x": np.zeros(num_agents, dtype=np.float32),
            "ang_vel_y": np.zeros(num_agents, dtype=np.float32),
            "ang_vel_z": np.zeros(num_agents, dtype=np.float32),
            "masses": np.ones(num_agents, dtype=np.float32),
            "drag_coeffs": np.zeros(num_agents, dtype=np.float32),
            "thrust_coeffs": np.ones(num_agents, dtype=np.float32),
            "target_vx": np.zeros(num_agents, dtype=np.float32),
            "target_vy": np.zeros(num_agents, dtype=np.float32),
            "target_vz": np.zeros(num_agents, dtype=np.float32),
            "target_yaw_rate": np.zeros(num_agents, dtype=np.float32),
            "vt_x": np.zeros(num_agents, dtype=np.float32),
            "vt_y": np.zeros(num_agents, dtype=np.float32),
            "vt_z": np.zeros(num_agents, dtype=np.float32),
            "traj_params": np.zeros((10, num_agents), dtype=np.float32),
            "target_trajectory": np.zeros((101, num_agents, 3), dtype=np.float32),
            "pos_history": np.zeros((100, num_agents, 3), dtype=np.float32),
            "observations": np.zeros((num_agents, 302), dtype=np.float32),
            "rewards": np.zeros(num_agents, dtype=np.float32),
            "reward_components": np.zeros((num_agents, 8), dtype=np.float32),
            "done_flags": np.zeros(num_agents, dtype=np.float32),
            "step_counts": np.zeros(num_agents, dtype=np.int32),
            "actions": np.zeros(num_agents * 4, dtype=np.float32),
            "env_ids": np.zeros(num_agents, dtype=np.int32)
        }

    def compute_action_gdpc(self, state_obs, target_cmd, tracking_uv, foe_uv, extra_yaw_rate=0.0):
        """
        Sampling-Based MPC (Gradient-Free GDPC).
        Generates K random action sequences, simulates them using step_cpu,
        and picks the first action of the best sequence.
        """
        # Call Heuristic first to get baseline action (using internal method to avoid recursion)
        heuristic_action_dict, _ = self._compute_heuristic_action(state_obs, target_cmd, tracking_uv, extra_yaw_rate, foe_uv)

        h_thrust = heuristic_action_dict['thrust']
        h_roll = heuristic_action_dict['roll_rate']
        h_pitch = heuristic_action_dict['pitch_rate']
        h_yaw = heuristic_action_dict['yaw_rate']

        num_ghosts = self.num_ghosts
        horizon = self.horizon

        if self.gdpc_buffer is None:
            self._init_gdpc_buffer(num_ghosts)

        buf = self.gdpc_buffer

        # 1. Initialize State from Observation
        px0 = 0.0
        py0 = 0.0
        pz0 = state_obs.get('pz')

        # In GDPC mode, we might need estimating PZ too if missing?
        # state_obs is dict. If 'pz' is missing, it returns None.
        if pz0 is None:
             pz0 = self.est_pz

        vx0 = state_obs.get('vx', 0.0)
        vy0 = state_obs.get('vy', 0.0)
        vz0 = state_obs.get('vz')
        if vz0 is None:
             vz0 = self.est_vz

        roll0 = state_obs.get('roll', 0.0)
        pitch0 = state_obs.get('pitch', 0.0)
        yaw0 = state_obs.get('yaw', 0.0)

        wx0 = state_obs.get('wx', 0.0)
        wy0 = state_obs.get('wy', 0.0)
        wz0 = state_obs.get('wz', 0.0)

        # Set Buffer Initial State
        buf['pos_x'][:] = px0
        buf['pos_y'][:] = py0
        buf['pos_z'][:] = pz0
        buf['vel_x'][:] = vx0
        buf['vel_y'][:] = vy0
        buf['vel_z'][:] = vz0
        buf['roll'][:] = roll0
        buf['pitch'][:] = pitch0
        buf['yaw'][:] = yaw0
        buf['ang_vel_x'][:] = wx0
        buf['ang_vel_y'][:] = wy0
        buf['ang_vel_z'][:] = wz0

        buf['masses'][:] = self.estimated_params['mass']
        buf['drag_coeffs'][:] = self.estimated_params['drag_coeff']
        buf['thrust_coeffs'][:] = self.estimated_params['thrust_coeff']

        buf['step_counts'][:] = 0
        buf['done_flags'][:] = 0.0

        # Target Position
        tgt_rel_x = target_cmd[0]
        tgt_rel_y = target_cmd[1]
        tgt_abs_z = target_cmd[2]

        target_pos = np.array([tgt_rel_x, tgt_rel_y, tgt_abs_z])

        # 2. Sample Action Sequences
        noise_thrust = 0.2
        noise_rate = 1.0

        action_seqs = np.zeros((horizon, num_ghosts, 4), dtype=np.float32)

        # Ghost 0: Heuristic
        action_seqs[:, 0, 0] = h_thrust
        action_seqs[:, 0, 1] = h_roll
        action_seqs[:, 0, 2] = h_pitch
        action_seqs[:, 0, 3] = h_yaw

        # Ghosts 1..N: Noisy Heuristic
        action_seqs[:, 1:, 0] = h_thrust + np.random.randn(horizon, num_ghosts-1) * noise_thrust
        action_seqs[:, 1:, 1] = h_roll + np.random.randn(horizon, num_ghosts-1) * noise_rate
        action_seqs[:, 1:, 2] = h_pitch + np.random.randn(horizon, num_ghosts-1) * noise_rate
        action_seqs[:, 1:, 3] = h_yaw + np.random.randn(horizon, num_ghosts-1) * noise_rate

        action_seqs[:, :, 0] = np.clip(action_seqs[:, :, 0], 0.0, 1.0)
        action_seqs[:, :, 1:] = np.clip(action_seqs[:, :, 1:], -10.0, 10.0)

        # 3. Rollout
        costs = np.zeros(num_ghosts, dtype=np.float32)
        ghost_paths_viz = [[] for _ in range(num_ghosts)]

        for t in range(horizon):
            acts_t = action_seqs[t].reshape(-1)

            step_cpu(
                buf['pos_x'], buf['pos_y'], buf['pos_z'],
                buf['vel_x'], buf['vel_y'], buf['vel_z'],
                buf['roll'], buf['pitch'], buf['yaw'],
                buf['ang_vel_x'], buf['ang_vel_y'], buf['ang_vel_z'],
                buf['masses'], buf['drag_coeffs'], buf['thrust_coeffs'],
                buf['target_vx'], buf['target_vy'], buf['target_vz'], buf['target_yaw_rate'],
                buf['vt_x'], buf['vt_y'], buf['vt_z'],
                buf['traj_params'],
                buf['target_trajectory'],
                buf['pos_history'],
                buf['observations'],
                buf['rewards'],
                buf['reward_components'],
                buf['done_flags'],
                buf['step_counts'],
                acts_t,
                num_ghosts,
                100,
                buf['env_ids']
            )

            # Calculate Cost
            dx = buf['pos_x'] - target_pos[0]
            dy = buf['pos_y'] - target_pos[1]
            dz = buf['pos_z'] - target_pos[2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)

            # Penalties
            ground_pen = np.where(buf['pos_z'] < 0.5, 1000.0, 0.0)

            step_cost = dist + ground_pen
            if t == horizon - 1:
                step_cost += dist * 5.0

            costs += step_cost

            # Viz (Ghost 0)
            p_pt = {'px': float(buf['pos_x'][0]), 'py': float(buf['pos_y'][0]), 'pz': float(buf['pos_z'][0])}
            ghost_paths_viz[0].append(p_pt)

        # 4. Select Best
        best_idx = np.argmin(costs)
        best_action_seq = action_seqs[:, best_idx, :]
        best_act = best_action_seq[0]

        action = {
            'thrust': float(best_act[0]),
            'roll_rate': float(best_act[1]),
            'pitch_rate': float(best_act[2]),
            'yaw_rate': float(best_act[3])
        }

        self.last_action = action
        return action, [ghost_paths_viz[0]]
