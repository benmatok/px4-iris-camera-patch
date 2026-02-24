import numpy as np
import logging
import math
from flight_config import FlightConfig
from flight_controller_gdpc import GDPCOptimizer
from vision.spline_smoother import SplineSmoother
import time

logger = logging.getLogger(__name__)

class DPCFlightController:
    """
    Standard ENU (East-North-Up) Flight Controller.
    """
    def __init__(self, dt=0.05, mode='PID', config: FlightConfig = None):
        self.dt = dt
        self.mode = mode
        self.config = config or FlightConfig()

        # State Variables
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

        # Blind Mode Estimation
        self.est_pz = 100.0
        self.est_vz = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0

        self.last_tracking_size = None
        self.last_tracking_v = None
        self.rer_smoothed = 0.0

        # Cruise / Dive State
        self.dive_initiated = False

        # Final Mode State
        self.final_mode = False
        self.locked_pitch = 0.0

        # GDPC
        self.gdpc = GDPCOptimizer(self.config)

        # Smoother
        # Minimal lag configuration
        self.smoother = SplineSmoother(window_size=3, smoothing_factor=0.01)
        self.start_time = time.time()

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.gdpc.reset()
        self.smoother.reset()
        self.start_time = time.time()
        self.est_pz = 100.0
        self.est_vz = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0
        self.last_tracking_size = None
        self.last_tracking_v = None
        self.rer_smoothed = 0.0

        self.dive_initiated = False
        self.final_mode = False
        self.locked_pitch = 0.0
        logger.info(f"DPCFlightController reset.")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, tracking_size=None, extra_yaw_rate=0.0, foe_uv=None, velocity_est=None, position_est=None, velocity_reliable=False):
        if velocity_est:
             v_mag = math.sqrt(velocity_est['vx']**2 + velocity_est['vy']**2 + velocity_est['vz']**2)
             # logger.debug(f"CTRL INPUT: V_est={v_mag:.2f} ({velocity_est}), Reliable={velocity_reliable}")

        # Unpack State
        obs_pz = state_obs.get('pz')
        obs_vz = state_obs.get('vz')
        roll = state_obs['roll']
        pitch = state_obs['pitch']
        yaw = state_obs['yaw']

        ctrl = self.config.control
        vis = self.config.vision

        # Blind Mode Logic
        if obs_pz is None:
            # Full Blind Integration
            th = self.last_action['thrust']
            g = 9.81
            T_max_accel = 20.0
            accel_z = (th * T_max_accel) * (math.cos(roll) * math.cos(pitch)) - g
            accel_z -= 0.1 * self.est_vz
            self.est_vz += accel_z * self.dt
            self.est_pz += self.est_vz * self.dt
            pz = self.est_pz
            vz = self.est_vz
        else:
            self.est_pz = obs_pz
            if obs_vz is None:
                # Estimate VZ
                th = self.last_action['thrust']
                g = 9.81
                T_max_accel = 20.0
                accel_z = (th * T_max_accel) * (math.cos(roll) * math.cos(pitch)) - g
                accel_z -= 0.1 * self.est_vz
                self.est_vz += accel_z * self.dt
            else:
                self.est_vz = obs_vz

            pz = self.est_pz
            vz = self.est_vz

        # --- Visual RER / TTC Calculation ---
        raw_rer = 0.0
        if tracking_size is not None and self.last_tracking_size is not None and self.last_tracking_size > 0.001:
            raw_rer = (tracking_size - self.last_tracking_size) / (self.last_tracking_size * self.dt)

        alpha_rer = vis.rer_smoothing_alpha
        self.rer_smoothed = (1.0 - alpha_rer) * self.rer_smoothed + alpha_rer * raw_rer
        self.last_tracking_size = tracking_size

        # GDPC Logic (Overrides Heuristic if VIO Lock is reliable)
        # Check velocity reliability against limit
        vel_mag = 0.0
        if velocity_est:
             vel_mag = math.sqrt(velocity_est['vx']**2 + velocity_est['vy']**2 + velocity_est['vz']**2)

        is_reliable = velocity_reliable
        if vel_mag > ctrl.velocity_limit:
             logger.warning(f"VIO Velocity Exceeds Limit ({vel_mag:.2f} > {ctrl.velocity_limit}). Treating as unreliable.")
             is_reliable = False

        # Spline Smoothing
        if position_est and is_reliable:
            # VIO NED Pos -> ENU
            px_enu = position_est['py'] # East
            py_enu = position_est['px'] # North
            pz_enu = -position_est['pz'] # Up

            # Update Smoother (even if we don't use it for primary control, to keep state)
            t_now = time.time() - self.start_time
            self.smoother.update(t_now, [px_enu, py_enu, pz_enu])

            # Use Raw VIO Velocity for minimal lag if reliable
            # Spline is used as backup or for smoothing visualization if needed
            # But user requested 0 lag, so we prefer raw VIO which is already filtered by EKF.

            vx_enu = velocity_est['vy']
            vy_enu = velocity_est['vx']
            vz_enu = -velocity_est['vz']

            self.est_vx = vx_enu
            self.est_vy = vy_enu
            self.est_vz = vz_enu

        elif velocity_est and is_reliable:
             # Fallback to Raw VIO if Position not provided (shouldn't happen with updated call sites)
             vx_enu = velocity_est['vy']
             vy_enu = velocity_est['vx']
             vz_enu = -velocity_est['vz']

             alpha_vel = ctrl.velocity_smoothing_alpha
             self.est_vx = alpha_vel * vx_enu + (1-alpha_vel) * self.est_vx
             self.est_vy = alpha_vel * vy_enu + (1-alpha_vel) * self.est_vy
             self.est_vz = alpha_vel * vz_enu + (1-alpha_vel) * self.est_vz

        # if is_reliable and hasattr(self.config, 'gdpc'):
             # GDPC DISABLED PER REQUEST - Switched to Heuristic FOE Controller

        # ... (Rest of the Heuristic Controller) ...
        # 2. Tracking Control (Visual Servoing)
        pitch_track = 0.0
        yaw_rate_cmd = 0.0
        thrust_track = 0.6

        if tracking_uv:
            u, v = tracking_uv
            v_dot = 0.0
            if self.last_tracking_v is not None:
                v_dot = (v - self.last_tracking_v) / self.dt
            self.last_tracking_v = v

            if not self.dive_initiated:
                trigger_rer = self.rer_smoothed > ctrl.dive_trigger_rer
                trigger_v_pos = v > ctrl.dive_trigger_v_threshold

                if trigger_rer or trigger_v_pos:
                    logger.info(f"DIVE INITIATED! RER={self.rer_smoothed:.2f}, v={v:.2f}")
                    self.dive_initiated = True

            # Use FOE if available for Yaw Control
            # If FOE missing, estimate from Velocity State
            calc_foe = foe_uv
            if not calc_foe and velocity_est:
                # Estimate FOE from VIO/GT Velocity
                # V_world (ENU) -> V_body -> V_cam -> FOE (u, v)

                # 1. World (ENU) to Body
                # R_b2w = Rz(y) * Ry(p) * Rx(r)
                # V_b = R_b2w.T @ V_w
                cr = math.cos(roll); sr = math.sin(roll)
                cp = math.cos(pitch); sp = math.sin(pitch)
                cy = math.cos(yaw); sy = math.sin(yaw)

                # R_b2w construction
                r11 = cy * cp
                r12 = cy * sp * sr - sy * cr
                r13 = cy * sp * cr + sy * sr
                r21 = sy * cp
                r22 = sy * sp * sr + cy * cr
                r23 = sy * sp * cr - cy * sr
                r31 = -sp
                r32 = cp * sr
                r33 = cp * cr

                R_b2w = np.array([
                    [r11, r12, r13],
                    [r21, r22, r23],
                    [r31, r32, r33]
                ])

                v_w = np.array([self.est_vx, self.est_vy, self.est_vz])
                v_b = R_b2w.T @ v_w

                # 2. Body to Camera
                # Cam Tilt 30 deg UP (pitch +30 relative to body).
                # R_c2b = R_y(tilt). (Camera Frame: Z forward, X right, Y down)
                # Wait. Projector says:
                # Base: Xc=Yb(Right), Yc=Zb(Down), Zc=Xb(Fwd)
                # Then Tilt (Pitch).

                # Let's use Projector Logic manually:
                # R_c2b_base:
                # [0, 0, 1]
                # [1, 0, 0]
                # [0, 1, 0]
                # Body X -> Cam Z. Body Y -> Cam X. Body Z -> Cam Y.

                # Tilt 30 deg. (Positive Pitch).
                theta = np.radians(self.config.camera.tilt_deg)
                ct, st = math.cos(theta), math.sin(theta)
                R_tilt = np.array([
                    [ct, 0, st],
                    [0, 1, 0],
                    [-st, 0, ct]
                ])

                R_base = np.array([
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]
                ], dtype=np.float32)

                R_c2b = R_tilt @ R_base

                # V_c = R_c2b.T @ V_b
                v_c = R_c2b.T @ v_b

                # 3. Project to Normalized Image Plane
                # u = x/z, v = y/z
                if v_c[2] > 0.1: # Moving forward relative to camera
                     u_foe = v_c[0] / v_c[2]
                     v_foe = v_c[1] / v_c[2]
                     calc_foe = (u_foe, v_foe)
                     # logger.debug(f"Est FOE: {calc_foe}")

            if calc_foe:
                 # Steer Target (u, v) towards FOE (foe_u, foe_v)
                 # We want u_target = u_foe.
                 # Error = u_target - u_foe.
                 u_err = u - calc_foe[0]
                 yaw_rate_cmd = -ctrl.k_yaw * u_err
            else:
                 # Fallback to centering
                 yaw_rate_cmd = -ctrl.k_yaw * u

            yaw_rate_cmd = max(-1.5, min(1.5, yaw_rate_cmd))

            if not self.dive_initiated:
                vz_err = 0.0 - self.est_vz
                pitch_track = ctrl.cruise_pitch_gain * vz_err
                pitch_track = max(-1.0, min(1.0, pitch_track))
                thrust_track = 0.55
            else:
                if calc_foe:
                    v_err = v - calc_foe[1]
                    # If target is BELOW FOE (v > v_foe), we need to Pitch Down (Negative Rate).
                    # Sim Pitch: Neg = Nose Down.
                    # Controller Pitch Rate: Neg = Nose Down.
                    # If v_err > 0, Target is "below" FOE (in image, +v is down).
                    # We are aiming "above" the target.
                    # We need to pitch down.
                    pitch_track = -ctrl.k_pitch * v_err
                else:
                     # Fallback logic based on Speed
                     pitch_bias = ctrl.pitch_bias_intercept + ctrl.pitch_bias_slope * pitch
                     pitch_bias = max(ctrl.pitch_bias_min, min(ctrl.pitch_bias_max, pitch_bias))

                     # SPEED CONTROL FALLBACK
                     # If too slow -> Pitch Down (Neg)
                     # If too fast -> Pitch Up (Pos)
                     v_target = ctrl.velocity_limit - 1.0
                     v_current = 0.0
                     if velocity_est:
                          v_current = math.sqrt(velocity_est['vx']**2 + velocity_est['vy']**2 + velocity_est['vz']**2)

                     # v_current - v_target.
                     # If fast (positive diff), want Pos Pitch (Up).
                     pitch_track = +ctrl.k_pitch * (v_current - v_target) + pitch_bias

                # Keep Safety Clamps
                if pitch < -0.3 and pitch_track < 0.0:
                     pitch_track = 0.0

                pitch_track = max(-2.5, min(2.5, pitch_track))

                thrust_base = ctrl.thrust_base_intercept + ctrl.thrust_base_slope * pitch
                thrust_base = max(ctrl.thrust_min, min(ctrl.thrust_max, thrust_base))

                effective_k_rer = ctrl.k_rer * (abs(pitch) ** 2)
                thrust_correction = max(0.0, effective_k_rer * (self.rer_smoothed - ctrl.rer_target))
                thrust_track = thrust_base - thrust_correction

                # Speed governor via Thrust
                # v_target = 5.0 (approx)
                # If speed estimation available (v_mag_est)
                if velocity_est:
                     v_est = math.sqrt(velocity_est['vx']**2 + velocity_est['vy']**2 + velocity_est['vz']**2)
                     if v_est < 5.0:
                          thrust_track += 0.1
                     elif v_est > 6.0:
                          thrust_track -= 0.1

                thrust_track = max(0.1, min(0.95, thrust_track))

                if self.rer_smoothed > ctrl.rer_target + ctrl.flare_threshold_offset:
                    flare_pitch = ctrl.flare_gain * (self.rer_smoothed - (ctrl.rer_target + ctrl.flare_threshold_offset))
                    pitch_track += flare_pitch

        else:
            yaw_rate_cmd = 0.0
            pitch_track = 0.0
            thrust_track = 0.55

        if tracking_uv and not self.final_mode and self.dive_initiated:
            u, v = tracking_uv
            if v < ctrl.final_mode_v_threshold_low or v > ctrl.final_mode_v_threshold_high:
                self.final_mode = True

        pitch_rate_cmd = pitch_track
        thrust_cmd = thrust_track

        if self.final_mode and tracking_uv:
            u, v = tracking_uv
            yaw_rate_cmd = -ctrl.k_yaw * u * ctrl.final_mode_yaw_gain
            yaw_rate_cmd = max(-ctrl.final_mode_yaw_limit, min(ctrl.final_mode_yaw_limit, yaw_rate_cmd))

            if v < 0.1:
                target_pitch_final = ctrl.final_mode_undershoot_pitch_target
                pitch_rate_cmd = ctrl.final_mode_undershoot_pitch_gain * (target_pitch_final - pitch)
                thrust_cmd = ctrl.final_mode_undershoot_thrust_base - ctrl.final_mode_undershoot_thrust_gain * v
            else:
                target_pitch_final = ctrl.final_mode_overshoot_pitch_target
                pitch_rate_cmd = ctrl.final_mode_overshoot_pitch_gain * (target_pitch_final - pitch)
                thrust_cmd = ctrl.final_mode_overshoot_thrust_base - ctrl.final_mode_overshoot_thrust_gain * (v - ctrl.final_mode_overshoot_v_target)
            thrust_cmd = max(0.1, min(0.95, thrust_cmd))

        elif not tracking_uv:
            # Blind Recovery Mode
            # If tracking is lost and we don't have reliable velocity, level out.
            if not is_reliable:
                 # Command Pitch to 0
                 pitch_rate_cmd = 1.0 * (0.0 - pitch) # Proportional P-gain
                 pitch_rate_cmd = max(-1.0, min(1.0, pitch_rate_cmd))
                 thrust_cmd = 0.55 # Hover thrust
                 logger.info(f"BLIND RECOVERY: Pitching Level. Pitch={pitch:.2f}")
            else:
                 # If we have reliable velocity but lost tracking, hold current course or drift?
                 # Default to level for safety.
                 pitch_rate_cmd = 1.0 * (0.0 - pitch)
                 thrust_cmd = 0.5

        # Enforce Speed Limit (Hard Active Braking)
        if velocity_est:
             v_mag_est = math.sqrt(velocity_est['vx']**2 + velocity_est['vy']**2 + velocity_est['vz']**2)

             # Sanity check: Only use VIO for braking if it is believable (< 15.0 m/s)
             if v_mag_est < 15.0:
                  limit = 6.0 # Hard limit as requested

                  # If velocity exceeds limit, override control completely
                  if v_mag_est > limit:
                      # Max Braking
                      thrust_cmd = 0.0 # Cut throttle

                      # Pitch Up (Flare) proportional to excess speed
                      excess = v_mag_est - limit
                      pitch_brake = 1.0 * excess # Strong gain

                      # Add to existing command but prioritize braking
                      pitch_rate_cmd = max(pitch_rate_cmd, pitch_brake)

                      # Clamp to prevent stalls/loops
                      pitch_rate_cmd = min(2.0, pitch_rate_cmd)

                      logger.info(f"HARD LIMIT ACTIVATED: v={v_mag_est:.2f} > {limit}. Braking pitch={pitch_brake:.2f}")
             elif v_mag_est > 15.0:
                  logger.warning(f"VIO Divergence Detected (v={v_mag_est:.2f} > 15.0). Ignoring for control.")

        # Safety Clamps
        # Sim Pitch Positive = Nose Up.
        # Max Pitch Up = 0.2 rad (~11 deg) to avoid stalling/backward flight.
        # Max Pitch Down = -1.0 rad (~57 deg) to avoid overspeed.

        # If Pitch > 0.0 (Horizon). Prevent further Nose Up to avoid backward flight.
        # pitch_rate_cmd Positive = Nose Up command.
        if pitch > 0.5 and pitch_rate_cmd > 0.0:
             pitch_rate_cmd = 0.0

        # If Pitch < -1.0 (Too Nose Down)
        # We want Nose Up (Positive rate).
        # Safety Floor / Auto-Flare (Heuristic)
        # If Altitude is low (< 10m) and descending fast (< -2 m/s), force Pitch Up.
        # This prevents ground impact short of target.
        if obs_pz is not None and obs_pz < 10.0 and self.est_vz < -2.0:
             # Calculate required acceleration to stop at z=0 (or z=1.0 safety)
             # v^2 = 2 * a * d  => a = v^2 / 2d
             dist_to_floor = obs_pz - 1.0
             if dist_to_floor > 0.1:
                 req_accel = (self.est_vz**2) / (2.0 * dist_to_floor)
                 # Converting accel to pitch effort approx (1g = 9.81)
                 # Pitch ~ accel / 20.0 (Max Thrust Accel) roughly?
                 # Just add a bias proportional to urgency.
                 flare_bias = 0.1 * req_accel
                 pitch_rate_cmd += flare_bias
                 # Also increase thrust to arrest descent
                 thrust_cmd = max(thrust_cmd, 0.6)
                 logger.debug(f"SAFETY FLOOR: pz={obs_pz:.1f}, vz={self.est_vz:.1f}, flare={flare_bias:.2f}")

        if pitch < -1.0 and pitch_rate_cmd < 0.0:
             pitch_rate_cmd = 0.0

        roll_rate_cmd = 4.0 * (0.0 - roll)

        action = {
            'thrust': thrust_cmd,
            'roll_rate': roll_rate_cmd,
            'pitch_rate': -pitch_rate_cmd,
            'yaw_rate': -yaw_rate_cmd
        }
        self.last_action = action

        # Ghost Paths (Viz)
        sim_vx = self.est_vx
        sim_vy = self.est_vy
        sim_vz = vz # Using Baro VZ or Estimated

        # Blind Path Prediction
        ghost_paths = []
        path = []
        # Return RELATIVE trajectory (consistent with GDPC)
        sim_x = 0.0; sim_y = 0.0; sim_z = 0.0
        for i in range(20):
             sim_x += sim_vx * self.dt
             sim_y += sim_vy * self.dt
             sim_z += sim_vz * self.dt
             path.append({'px': sim_x, 'py': sim_y, 'pz': sim_z})
        ghost_paths.append(path)

        return action, ghost_paths
