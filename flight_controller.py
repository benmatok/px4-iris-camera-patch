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
        self.k_yaw = 3.0 # Stronger Yaw for precise tracking
        self.k_roll = 4.0 # Stronger Roll response for crab
        self.k_pitch = 4.0

        # Velocity Control Gains
        self.kp_vz = 2.0
        self.thrust_hover = 0.6

        # FOE Bias Gains
        self.k_foe_yaw = 2.0
        self.k_foe_pitch = 1.0

        # State Variables
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.flight_phase = "DIVE" # DIVE, BRAKE, APPROACH, LAND

        # Blind Mode Estimation
        self.est_pz = -100.0 # Initial assumption (NED Z, Altitude 100 = -100)
        self.est_vz = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0
        self.last_pz_obs = None # For discrete differentiation
        self.last_dist_3d = None # For calculating closing rate

        # Visual RER State
        self.last_radius = None
        self.rer_smooth = 0.0

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
        self.est_pz = -100.0
        self.est_vz = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0
        self.last_pz_obs = None
        self.last_dist_3d = None
        self.last_radius = None
        self.rer_smooth = 0.0
        self.flight_phase = "DIVE"
        logger.info(f"DPCFlightController reset. Mode: {self.mode}")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, extra_yaw_rate=0.0, foe_uv=None, tracking_radius=None):
        if self.mode == 'GDPC' and HAS_SIM:
            return self.compute_action_gdpc(state_obs, target_cmd, tracking_uv, foe_uv, extra_yaw_rate)
        return self._compute_heuristic_action(state_obs, target_cmd, tracking_uv, extra_yaw_rate, foe_uv, tracking_radius)

    def _compute_heuristic_action(self, state_obs, target_cmd, tracking_uv=None, extra_yaw_rate=0.0, foe_uv=None, tracking_radius=None):
        # Unpack State
        # Convert Z/Vz to ENU (Up+) for internal logic to match control expectations (Thrust increases Z).
        # Input 'pz' and 'vz' are NED (Down+).
        pz_ned = state_obs.get('pz')
        vz_ned = state_obs.get('vz')

        pz = -pz_ned if pz_ned is not None else None
        vz = -vz_ned if vz_ned is not None else None

        roll = state_obs['roll']
        pitch = state_obs['pitch']
        yaw = state_obs['yaw']

        # --- Blind Mode Velocity Estimation (NO SENSORS PERMITTED) ---
        if vz is None:
            if pz is not None and self.last_pz_obs is not None:
                # pz is ENU here
                raw_vz = (pz - self.last_pz_obs) / self.dt
                alpha_vz = 0.5
                self.est_vz = (1.0 - alpha_vz) * self.est_vz + alpha_vz * raw_vz
            elif pz is not None:
                 self.est_vz = 0.0

            self.last_pz_obs = pz
            obs_vz = self.est_vz
        else:
            obs_vz = vz
            self.est_vz = vz
            self.last_pz_obs = pz

        obs_vx = state_obs.get('vx', 0.0)
        obs_vy = state_obs.get('vy', 0.0)

        # We assume X/Y are standard (ENU or NED doesn't matter if we just use magnitude,
        # but Yaw direction matters. Yaw is NED. X/Y are NED. This is fine.)

        # We need to use obs_vz (ENU) to estimate horizontal speed.
        # Dive: Pitch < -0.1 (Nose Down).
        # ENU Vz is Negative.
        # tan(pitch) is Negative.
        # V_xy = V_z / tan(pitch) = (- / -) = +. Correct.

        is_blind_xy = (abs(obs_vx) < 0.001 and abs(obs_vy) < 0.001 and abs(obs_vz) > 0.1)

        if is_blind_xy:
            if pitch < -0.1: # Coordinated Dive (Nose Down)
                 v_xy_est = obs_vz / math.tan(pitch)
                 v_xy_est = max(0.0, min(30.0, v_xy_est))

                 alpha = 0.1
                 vx_new = v_xy_est * math.cos(yaw)
                 vy_new = v_xy_est * math.sin(yaw)
                 self.est_vx = (1.0 - alpha) * self.est_vx + alpha * vx_new
                 self.est_vy = (1.0 - alpha) * self.est_vy + alpha * vy_new
            else:
                 decay = 0.98
                 self.est_vx *= decay
                 self.est_vy *= decay

            est_vx = self.est_vx
            est_vy = self.est_vy
        else:
            est_vx = obs_vx
            est_vy = obs_vy
            self.est_vx = obs_vx
            self.est_vy = obs_vy

        v_total = math.sqrt(est_vx**2 + est_vy**2 + obs_vz**2)

        # --- Target Logic ---
        # No more "Side Offset" virtual targets.
        # We target the TRUE target (0,0 relative) and inject ROLL bias for spiral.
        dx = target_cmd[0]
        dy = target_cmd[1]
        goal_z = target_cmd[2]

        dist_est = math.sqrt(dx*dx + dy*dy)

        # pz is ENU (Altitude). goal_z is NED (Relative or Absolute).
        # target_cmd[2] is Absolute NED Z (0.0).
        # We want Altitude above target.
        # Target Z (ENU) = -target_cmd[2] = 0.0.
        # alt_est = pz - Target_Z_ENU = pz.
        alt_est = max(0.1, pz - (-goal_z))

        dist_3d = math.sqrt(dist_est*dist_est + alt_est*alt_est)

        # --- RER (Looming) Estimation ---
        # Primary: Visual Expansion (Observable)
        # Secondary: Velocity / Distance (Estimator Fallback)

        rer_visual = 0.0
        if tracking_radius is not None and self.last_radius is not None:
             # RER = (r_t - r_prev) / (r_prev * dt)
             # Avoid div by zero
             if self.last_radius > 0.001:
                 raw_rer = (tracking_radius - self.last_radius) / (self.last_radius * self.dt)
                 # Constrain logical RER (0 to 10.0)
                 raw_rer = max(0.0, min(10.0, raw_rer))

                 # Smooth it
                 alpha_rer = 0.2
                 self.rer_smooth = (1.0 - alpha_rer) * self.rer_smooth + alpha_rer * raw_rer
                 rer_visual = self.rer_smooth

        if tracking_radius is not None:
             self.last_radius = tracking_radius
        else:
             # Decay radius if tracking lost
             if self.last_radius:
                 self.last_radius *= 0.95

        # Estimator Fallback
        rer_est = v_total / max(1.0, dist_3d)

        # Select RER source
        # Use visual if available and sensible (> 0.05), else fallback
        # If we are very close, visual might blow up or be erratic, but usually it's the gold standard for "Time to Contact"
        if tracking_radius is not None:
             rer = rer_visual
        else:
             rer = rer_est

        # Visual Distance Refinement
        if tracking_uv:
             _, v = tracking_uv
             tilt_rad = math.radians(30.0)
             angle_from_axis = math.atan(v)
             depression = angle_from_axis - (pitch + tilt_rad)
             if depression > 0.1:
                  dist_visual = alt_est / math.tan(depression)
                  dist_est = dist_visual
                  dist_3d = math.sqrt(dist_est*dist_est + alt_est*alt_est)
                  # If using estimate fallback, update it
                  if tracking_radius is None:
                      rer = v_total / max(1.0, dist_3d)

        # --- STATE MACHINE ---
        depression_angle = math.atan2(alt_est, dist_est)

        # Simple Logic: Only DIVE and LAND (Spiral/Brake)
        # We use RER to determine "intensity" of spiral.

        # Check for LAND transition (Critical Proximity)
        # pz is ENU (Altitude).
        # Trigger earlier to allow spiral to develop
        if self.flight_phase != "LAND" and dist_3d < 25.0:
            self.flight_phase = "LAND"
            logger.info(f"Switching to LAND phase (Dist3D: {dist_3d:.1f}, RER: {rer:.2f})")

        # --- Control Logic (CRAB SPIRAL) ---
        gamma_ref = 0.0
        thrust_cmd = 0.5
        speed_limit = 15.0 # Max speed

        los = -math.atan2(alt_est, dist_est)

        # 1. YAW Control (The Eye)
        # Pure Pursuit: Keep target centered horizontally.
        yaw_rate_cmd = 0.0
        if tracking_uv:
            u, v = tracking_uv
            # u is normalized horizontal error (-1 to 1?)
            # Usually u is tangent(angle).
            # Simple PID on u
            yaw_rate_cmd = self.k_yaw * u
        else:
            # Blind Mode: Point at estimated target
            target_yaw = math.atan2(dy, dx)
            yaw_err = target_yaw - yaw
            yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
            yaw_rate_cmd = self.k_yaw * yaw_err

        # 2. PITCH Control (The Engine)
        # Pitch Forward to close distance.
        # Constant pitch or scaled by distance?
        # Let's use Gamma Ref logic but separated.

        # Base Gamma: Aim at target
        gamma_base = los

        # If LANDING, we stop pitching forward (Pitch 0 or Flare)
        if self.flight_phase == "LAND":
            # Terminal Braking
            # Pitch 0 (Stop pushing) or Pitch Up (Flare)
            # Roll Max (Hard side slip) logic below.

            # Reduce speed target
            speed_limit = 2.0

            # Flare Pitch:
            # We want to arrest descent.
            # Gamma Ref = +5 deg (Climb/Level)
            gamma_ref = math.radians(5.0)

        else:
            # DIVE / APPROACH
            # Pitch down to target
            # Add bias for speed?
            # Standard LOS guidance for pitch
            gamma_ref = los - math.radians(5.0) # 5 deg attack angle

        # Safety Clamp (Don't hit ground early)
        if alt_est < 30.0 and self.flight_phase != "LAND":
             safety_gamma = -math.atan2(alt_est, 30.0) # Lookahead 30m
             gamma_ref = max(gamma_ref, safety_gamma)

        # 3. ROLL Control (The Brake / Looming Spiral)
        # Inject Roll based on RER (Looming)
        # If RER is high (close), Roll Right (positive?) to spiral.
        # Normalized RER:
        # Far: 15m/s / 100m = 0.15
        # Close: 15m/s / 20m = 0.75
        # Very Close: 15m/s / 10m = 1.5

        # Tuning Thresholds
        rer_start = 0.2
        rer_max = 0.8 # Reach max bank sooner
        max_bank = math.radians(50.0) # Increase max bank

        roll_cmd_rad = 0.0

        # Override safety during spiral entry to allow descent
        if self.flight_phase == "DIVE" and rer > rer_start:
             gamma_ref = los # Ignore safety clamp to spiral down

        if self.flight_phase == "LAND":
            # Terminal: HARD BANK
            roll_cmd_rad = max_bank
            thrust_cmd = 0.8 # High thrust to maintain lift in bank
        else:
            # Proportional Spiral
            # Add time factor? No, RER is dynamic.
            if rer > rer_start:
                # Scale from 0 to max_bank
                ratio = min(1.0, (rer - rer_start) / (rer_max - rer_start))
                roll_cmd_rad = ratio * max_bank

            # Reduce thrust slightly during spiral entry to drop altitude if needed?
            # Or keep it up to maintain speed for RER?
            thrust_cmd = 0.6

        # 4. Thrust Logic (Maintain Speed / Altitude)
        # Vertical speed control via Gamma Ref
        vz_cmd = speed_limit * math.sin(gamma_ref)
        vz_err = vz_cmd - obs_vz
        thrust_base = self.thrust_hover + self.kp_vz * vz_err

        # Mix thrust:
        # If banking hard, we need more total thrust to maintain vertical component
        # T_total * cos(phi) = T_vertical
        # T_total = T_vertical / cos(phi)

        if self.flight_phase == "LAND":
             thrust_cmd = 0.7 # Fixed high thrust for landing flare/bank
        else:
             thrust_cmd = thrust_base / max(0.5, math.cos(roll_cmd_rad))
             thrust_cmd = max(0.1, min(1.0, thrust_cmd))

        # Pitch Loop
        # Pitch Command from Gamma Ref
        pitch_cmd = gamma_ref
        pitch_rate_cmd = self.k_pitch * (pitch - pitch_cmd)

        # Roll Loop
        # Command Roll Angle
        roll_rate_cmd = self.k_roll * (roll_cmd_rad - roll)

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
        sim_x = 0.0
        sim_y = 0.0
        sim_z = pz

        sim_vz = obs_vz
        sim_vx = est_vx
        sim_vy = est_vy

        if vz is None and abs(sim_vz) < 0.1 and abs(gamma_ref) > 0.1:
             sim_vz = 5.0 * math.sin(gamma_ref)
             v_xy_est = sim_vz / math.tan(pitch) if abs(pitch) > 0.1 else 0.0
             v_xy_est = max(0.0, min(30.0, v_xy_est))
             sim_vx = v_xy_est * math.cos(yaw)
             sim_vy = v_xy_est * math.sin(yaw)

        for i in range(20):
             sim_x += sim_vx * self.dt
             sim_y += sim_vy * self.dt
             sim_z += sim_vz * self.dt
             path.append({'px': sim_x, 'py': sim_y, 'pz': sim_z})
        ghost_paths.append(path)

        return action, ghost_paths

    def _init_gdpc_buffer(self, num_agents):
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

        px0 = 0.0
        py0 = 0.0
        pz0 = state_obs.get('pz')

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

        tgt_rel_x = target_cmd[0]
        tgt_rel_y = target_cmd[1]
        tgt_abs_z = target_cmd[2]

        target_pos = np.array([tgt_rel_x, tgt_rel_y, tgt_abs_z])

        noise_thrust = 0.2
        noise_rate = 1.0

        action_seqs = np.zeros((horizon, num_ghosts, 4), dtype=np.float32)

        action_seqs[:, 0, 0] = h_thrust
        action_seqs[:, 0, 1] = h_roll
        action_seqs[:, 0, 2] = h_pitch
        action_seqs[:, 0, 3] = h_yaw

        action_seqs[:, 1:, 0] = h_thrust + np.random.randn(horizon, num_ghosts-1) * noise_thrust
        action_seqs[:, 1:, 1] = h_roll + np.random.randn(horizon, num_ghosts-1) * noise_rate
        action_seqs[:, 1:, 2] = h_pitch + np.random.randn(horizon, num_ghosts-1) * noise_rate
        action_seqs[:, 1:, 3] = h_yaw + np.random.randn(horizon, num_ghosts-1) * noise_rate

        action_seqs[:, :, 0] = np.clip(action_seqs[:, :, 0], 0.0, 1.0)
        action_seqs[:, :, 1:] = np.clip(action_seqs[:, :, 1:], -10.0, 10.0)

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

            dx = buf['pos_x'] - target_pos[0]
            dy = buf['pos_y'] - target_pos[1]
            dz = buf['pos_z'] - target_pos[2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)

            ground_pen = np.where(buf['pos_z'] < 0.5, 1000.0, 0.0)

            step_cost = dist + ground_pen
            if t == horizon - 1:
                step_cost += dist * 5.0

            costs += step_cost

            p_pt = {'px': float(buf['pos_x'][0]), 'py': float(buf['pos_y'][0]), 'pz': float(buf['pos_z'][0])}
            ghost_paths_viz[0].append(p_pt)

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
