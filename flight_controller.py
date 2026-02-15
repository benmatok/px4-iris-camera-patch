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

        # Trajectory Parameters
        self.dive_angle_start = -60.0
        self.dive_angle_end = -15.0

        # Velocity Control Gains
        self.kp_vz = 2.0
        self.thrust_hover = 0.6

        # Pitch Control Gains
        self.kp_pitch_v = 1.0
        self.pitch_bias_max = math.radians(15.0)

        # FOE Bias Gains
        self.k_foe_yaw = 2.0
        self.k_foe_pitch = 1.0

        # State Variables
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

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
        self.num_ghosts = 20 # Number of parallel rollouts
        self.horizon = 20    # Steps to look ahead

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.est_pz = 100.0
        self.est_vz = 0.0
        # Reset estimator? Maybe keep learned params.
        logger.info(f"DPCFlightController reset. Mode: {self.mode}")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, extra_yaw_rate=0.0, foe_uv=None):
        """
        Compute control action based on state and target.
        """

        if self.mode == 'GDPC' and HAS_SIM:
            return self.compute_action_gdpc(state_obs, target_cmd, tracking_uv, foe_uv, extra_yaw_rate)

        # Default PID / Heuristic Control
        return self._compute_heuristic_action(state_obs, target_cmd, tracking_uv, extra_yaw_rate, foe_uv)

    def _compute_heuristic_action(self, state_obs, target_cmd, tracking_uv=None, extra_yaw_rate=0.0, foe_uv=None):
        """
        Internal method for PID/Heuristic control.
        """
        # Unpack State
        pz = state_obs.get('pz')
        vz = state_obs.get('vz')
        roll = state_obs['roll']
        pitch = state_obs['pitch']

        # Blind Mode Altitude Estimation
        if pz is None:
            # Estimate Vz based on last action or simple kinematics
            # Vz ~ Speed * sin(Pitch) ?
            # Better: Integrate thrust and gravity? Too complex without mass.
            # Simple heuristic: Vz ~ (Thrust - Hover) * Gain?
            # Or assume we follow velocity commands perfectly?
            # Let's use the commanded vz from previous step if available, or current kinematic projection.

            # Current Kinematic Projection:
            # Assume constant forward speed approx 10m/s if diving? Or hover?
            # If thrust is high, we accelerate.

            # Let's update est_pz based on a simple model.
            # If we don't have vz, we use est_vz.

            # Simple Gravity/Thrust Model for Vz
            # az = (Thrust / Mass) * cos(roll) * cos(pitch) - g
            # This is Z-Up.
            # Thrust 0.5 ~ Hover (1g).
            # Thrust 0.0 ~ -1g (Freefall).
            # Thrust 1.0 ~ +1g (Climb).
            # Normalised Thrust [0,1].

            th = self.last_action['thrust']
            # Assume T=0.6 is Hover (approx).
            # Accel = (th - 0.6) * k_accel.
            # If Pitch is -90 (Nose Down), Thrust accelerates X, not Z?
            # Z-accel depends on Pitch.
            # az_body = Thrust.
            # az_world = Thrust * sin(pitch)? No.
            # Z is Up.
            # Body Z points Up (relative to body).
            # If Pitch=0, Body Z = World Z.
            # If Pitch=-90, Body Z = World X. Body X = World -Z.
            # Thrust is along Body Z? Usually Body Z is up in quadcopter frame.
            # So Thrust vector is R * [0,0,T].
            # az_world = T * (cos(phi)cos(theta)) - g.

            # We need to integrate this.
            g = 9.81
            T_max_accel = 20.0 # From drone.py (20 * thrust_coeff)
            accel_z = (th * T_max_accel) * (math.cos(roll) * math.cos(pitch)) - g

            # Damping/Drag
            accel_z -= 0.1 * self.est_vz

            self.est_vz += accel_z * self.dt
            self.est_pz += self.est_vz * self.dt

            # Floor
            if self.est_pz < 0.0:
                self.est_pz = 0.0
                self.est_vz = 0.0

            pz = self.est_pz
            vz = self.est_vz
        else:
            # Sync estimator
            self.est_pz = pz
            self.est_vz = vz if vz is not None else 0.0

        goal_z = target_cmd[2]

        # 1. Yaw Control
        yaw_rate_cmd = 0.0
        if tracking_uv:
            u, v = tracking_uv
            # FOE Bias Logic for Yaw
            if foe_uv:
                foe_u, foe_v = foe_uv
                # We want Target U to align with FOE U (Flight Direction)
                # Error = u - foe_u
                yaw_rate_cmd = self.k_foe_yaw * (u - foe_u)
            else:
                # Standard centering
                yaw_rate_cmd = self.k_yaw * u
        else:
            yaw_rate_cmd = extra_yaw_rate

        # 2. Estimate Distance and Compute Adaptive Gamma Ref
        dx = target_cmd[0]
        dy = target_cmd[1]
        dist_est = math.sqrt(dx*dx + dy*dy)
        alt_est = max(0.1, pz - goal_z)

        if tracking_uv:
             _, v = tracking_uv
             tilt_rad = math.radians(30.0)
             angle_from_axis = math.atan(v)
             depression = angle_from_axis - (pitch + tilt_rad)
             if depression > 0.1:
                  dist_visual = alt_est / math.tan(depression)
                  dist_est = dist_visual

        # Guidance Logic: Biased LOS (Flare Strategy)
        # We use a parabolic pitch strategy: Aggressive dive then Linear approach.
        los = -math.atan2(alt_est, dist_est)

        # Aggressive Phase: Dive significantly below LOS
        gamma_ref = los - math.radians(10.0)
        gamma_ref = max(gamma_ref, math.radians(-85.0))

        # Linear Phase: Switch to pure LOS when close to target (e.g. < 50m)
        if dist_est < 50.0:
             gamma_ref = los

        # 3. Speed Control (Thrust)
        if vz is not None:
            # Closed-Loop Speed Control
            # Adaptive speed limit based on distance to prevent overshoot
            speed_limit = max(5.0, min(15.0, dist_est * 0.3))

            vz_cmd = speed_limit * math.sin(gamma_ref)
            vz_err = vz_cmd - vz
            thrust_cmd = self.thrust_hover + self.kp_vz * vz_err
            thrust_cmd = max(0.1, min(1.0, thrust_cmd))
        else:
            thrust_cmd = self.thrust_hover
            vz_err = 0.0

        # 4. Pitch Control
        pitch_cmd = gamma_ref

        # Velocity-Based Pitch Bias (Air Brake)
        # If we are falling too fast (vz_err > 0), pitch up to use drag/lift
        if vz_err > 0.0:
             pitch_brake = 0.1 * vz_err # 0.1 rad per m/s error
             pitch_brake = min(pitch_brake, 0.5) # Cap at ~30 deg correction
             pitch_cmd += pitch_brake

        # FOE Bias Logic for Pitch
        if tracking_uv and foe_uv:
             # Target V vs FOE V
             # If Target V > FOE V (Target is "Below" FOE in image)
             # We need to dive steeper -> Reduce Pitch
             _, v = tracking_uv
             _, foe_v = foe_uv

             pitch_bias = -self.k_foe_pitch * (v - foe_v)
             pitch_bias = max(-0.5, min(0.5, pitch_bias))
             pitch_cmd += pitch_bias

        # Pitch Rate Loop
        k_pitch_ang = 5.0
        pitch_rate_cmd = k_pitch_ang * (pitch - pitch_cmd)

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
