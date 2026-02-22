import numpy as np
import logging
import math
from flight_config import FlightConfig
from flight_controller_gdpc import GDPCOptimizer

logger = logging.getLogger(__name__)

class TargetPredictor:
    def __init__(self, history_len=10, poly_degree=2):
        self.history_len = history_len
        self.poly_degree = poly_degree
        self.times = []
        self.u_hist = []
        self.v_hist = []

        # Last fitted model
        self.u_coeffs = None
        self.v_coeffs = None
        self.last_update_time = 0.0

    def update(self, t, u, v):
        self.times.append(t)
        self.u_hist.append(u)
        self.v_hist.append(v)

        if len(self.times) > self.history_len:
            self.times.pop(0)
            self.u_hist.pop(0)
            self.v_hist.pop(0)

        self.last_update_time = t

        if len(self.times) >= self.poly_degree + 1:
            try:
                # Fit polynomial to recent history
                # Shift time to be relative to last update (t=0 at last update)
                t_rel = np.array(self.times) - self.times[-1]
                self.u_coeffs = np.polyfit(t_rel, self.u_hist, self.poly_degree)
                self.v_coeffs = np.polyfit(t_rel, self.v_hist, self.poly_degree)
            except Exception as e:
                logger.warning(f"Polyfit failed: {e}")
                self.u_coeffs = None
                self.v_coeffs = None

    def predict(self, t_query):
        if self.u_coeffs is None or self.v_coeffs is None:
            return None

        # t_query is absolute time. We need relative time from last update.
        dt = t_query - self.last_update_time

        # Evaluate polynomial
        u_pred = np.polyval(self.u_coeffs, dt)
        v_pred = np.polyval(self.v_coeffs, dt)

        return (u_pred, v_pred)

    def reset(self):
        self.times = []
        self.u_hist = []
        self.v_hist = []
        self.u_coeffs = None
        self.v_coeffs = None
        self.last_update_time = 0.0


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
        self.dive_trigger_time = 0.0

        # Final Mode State
        self.final_mode = False
        self.locked_pitch = 0.0

        # Target Prediction
        self.predictor = TargetPredictor(history_len=20, poly_degree=2)
        self.current_time = 0.0

        # GDPC
        self.gdpc = GDPCOptimizer(self.config)

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.gdpc.reset()
        self.est_pz = 100.0
        self.est_vz = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0
        self.last_tracking_size = None
        self.last_tracking_v = None
        self.rer_smoothed = 0.0

        self.dive_initiated = False
        self.dive_trigger_time = 0.0
        self.final_mode = False
        self.locked_pitch = 0.0

        self.predictor.reset()
        self.current_time = 0.0

        logger.info(f"DPCFlightController reset.")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, tracking_size=None, extra_yaw_rate=0.0, foe_uv=None, velocity_est=None, velocity_reliable=False):
        self.current_time += self.dt

        # Unpack State
        obs_pz = state_obs.get('pz')
        obs_vz = state_obs.get('vz')
        roll = state_obs['roll']
        pitch = state_obs['pitch']
        yaw = state_obs['yaw']

        ctrl = self.config.control
        vis = self.config.vision

        # Update Prediction Model
        if tracking_uv:
            u, v = tracking_uv
            self.predictor.update(self.current_time, u, v)

        # Use Predicted Target if Actual is Missing (or for smoothing)
        target_uv = tracking_uv
        if target_uv is None and self.dive_initiated:
             pred = self.predictor.predict(self.current_time)
             if pred:
                 target_uv = pred
                 # Clamp to screen bounds roughly (normalized 0-1?) No, normalized usually centered or 0-1?
                 # Projector uses normalized u=x/z.
                 # Let's trust prediction for now.

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
        # DISABLED to force FOE-based control as requested
        if False and velocity_reliable and velocity_est and hasattr(self.config, 'gdpc'):
             # Update State Estimate (ENU) for Ghost Path Consistency
             # VIO is NED. Convert to ENU.
             # NED North (vx) -> ENU North (vy)
             # NED East (vy) -> ENU East (vx)
             # NED Down (vz) -> ENU Up (-vz)

             vx_enu = velocity_est['vy'] # East (VIO vy) -> ENU vx
             vy_enu = velocity_est['vx'] # North (VIO vx) -> ENU vy
             vz_enu = -velocity_est['vz'] # Down (VIO vz) -> ENU -vz

             alpha_vel = ctrl.velocity_smoothing_alpha
             self.est_vx = alpha_vel * vx_enu + (1-alpha_vel) * self.est_vx
             self.est_vy = alpha_vel * vy_enu + (1-alpha_vel) * self.est_vy
             self.est_vz = alpha_vel * vz_enu + (1-alpha_vel) * self.est_vz

             # Construct Relative ENU State for GDPC
             # We ignore absolute position to satisfy "local coordinates" requirement.
             # Drone is at 0,0,0. Target is at [dx, dy, dz] (Relative Vector).
             gdpc_state = {
                 'px': 0.0,
                 'py': 0.0,
                 'pz': 0.0,
                 'vx': self.est_vx,
                 'vy': self.est_vy,
                 'vz': self.est_vz,
                 'roll': roll,
                 'pitch': pitch,
                 'yaw': yaw,
                 'wx': state_obs.get('wx', 0.0),
                 'wy': state_obs.get('wy', 0.0),
                 'wz': state_obs.get('wz', 0.0)
             }

             # Target is simply the relative command
             target_pos_relative = target_cmd

             action, traj_enu = self.gdpc.compute_action(gdpc_state, target_pos_relative)

             # Apply Action
             self.last_action = {
                 'thrust': action['thrust'],
                 'roll_rate': action['roll_rate'],
                'pitch_rate': action['pitch_rate'],
                 'yaw_rate': action['yaw_rate']
             }

             # Ghost Paths (Already ENU)
             ghost_paths = []
             path = []
             for i in range(len(traj_enu)):
                 path.append({'px': traj_enu[i, 0], 'py': traj_enu[i, 1], 'pz': traj_enu[i, 2]})
             ghost_paths.append(path)

             final_action = {
                'thrust': action['thrust'],
                'roll_rate': action['roll_rate'],
                'pitch_rate': action['pitch_rate'],
                'yaw_rate': action['yaw_rate']
             }
             return final_action, ghost_paths

        # ... (Rest of the Heuristic Controller) ...
        # 2. Tracking Control (Visual Servoing)
        pitch_track = 0.0
        yaw_rate_cmd = 0.0
        thrust_track = 0.6

        # Use Predicted Target UV if available
        # But we also need 'v' for dive logic
        u, v = (0.0, 0.0)
        has_target = False

        if target_uv:
            u, v = target_uv
            has_target = True

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
                    self.dive_trigger_time = self.current_time

            yaw_rate_cmd = -ctrl.k_yaw * u
            yaw_rate_cmd = max(-1.5, min(1.5, yaw_rate_cmd))

            if not self.dive_initiated:
                vz_err = 0.0 - self.est_vz
                pitch_track = ctrl.cruise_pitch_gain * vz_err
                pitch_track = max(-1.0, min(1.0, pitch_track))
                thrust_track = 0.55
            else:
                # Tent Function for Pitch Bias (Flare)
                dt_dive = self.current_time - self.dive_trigger_time
                tent_bias = 0.0
                if dt_dive < ctrl.tent_duration:
                    if dt_dive < ctrl.tent_peak_time:
                        tent_bias = ctrl.tent_peak_pitch_rate * (dt_dive / ctrl.tent_peak_time)
                    else:
                        ramp_down_duration = ctrl.tent_duration - ctrl.tent_peak_time
                        if ramp_down_duration > 0:
                            progress = (dt_dive - ctrl.tent_peak_time) / ramp_down_duration
                            tent_bias = ctrl.tent_peak_pitch_rate * (1.0 - progress)

                pitch_bias = ctrl.pitch_bias_intercept + ctrl.pitch_bias_slope * pitch
                pitch_bias = max(ctrl.pitch_bias_min, min(ctrl.pitch_bias_max, pitch_bias))

                if pitch < ctrl.v_target_pitch_threshold:
                     v_target = ctrl.v_target_intercept + ctrl.v_target_slope_steep * (pitch - ctrl.v_target_pitch_threshold)
                else:
                     v_target = ctrl.v_target_intercept + ctrl.v_target_slope_shallow * (pitch - ctrl.v_target_pitch_threshold)
                v_target = max(0.1, v_target)

                # Add Tent Bias to pitch tracking
                pitch_track = -ctrl.k_pitch * (v - v_target) + pitch_bias + tent_bias
                pitch_track = max(-2.5, min(2.5, pitch_track))

                thrust_base = ctrl.thrust_base_intercept + ctrl.thrust_base_slope * pitch
                thrust_base = max(ctrl.thrust_min, min(ctrl.thrust_max, thrust_base))

                effective_k_rer = ctrl.k_rer * (abs(pitch) ** 2)
                thrust_correction = max(0.0, effective_k_rer * (self.rer_smoothed - ctrl.rer_target))
                thrust_track = thrust_base - thrust_correction

                if v < v_target - 0.2:
                    thrust_track += -(v - v_target) * 2.0
                thrust_track = max(0.1, min(0.95, thrust_track))

                if self.rer_smoothed > ctrl.rer_target + ctrl.flare_threshold_offset:
                    flare_pitch = ctrl.flare_gain * (self.rer_smoothed - (ctrl.rer_target + ctrl.flare_threshold_offset))
                    pitch_track += flare_pitch

        else:
            yaw_rate_cmd = 0.0
            pitch_track = 0.0
            thrust_track = 0.55

        # FOE Override for Yaw (Horizontal Intercept)
        # Use target_uv (Predicted or Real)
        if has_target and foe_uv:
            u, v = target_uv
            foe_u, foe_v = foe_uv
            # Steer Velocity (FOE) to Target (u)
            # Yaw Rate ~ -(Target - FOE)
            yaw_rate_cmd = -ctrl.k_yaw * (u - foe_u)
            yaw_rate_cmd = max(-1.5, min(1.5, yaw_rate_cmd))

        # FOE Override for Pitch (Vertical Intercept) - Only in Dive
        if has_target and foe_uv and self.dive_initiated:
            u, v = target_uv
            foe_u, foe_v = foe_uv

            # Recalculate Tent Bias
            dt_dive = self.current_time - self.dive_trigger_time
            tent_bias = 0.0
            if dt_dive < ctrl.tent_duration:
                if dt_dive < ctrl.tent_peak_time:
                    tent_bias = ctrl.tent_peak_pitch_rate * (dt_dive / ctrl.tent_peak_time)
                else:
                    ramp_down_duration = ctrl.tent_duration - ctrl.tent_peak_time
                    if ramp_down_duration > 0:
                        progress = (dt_dive - ctrl.tent_peak_time) / ramp_down_duration
                        tent_bias = ctrl.tent_peak_pitch_rate * (1.0 - progress)

            # Steer Velocity (FOE) to Target (v) with Tent Bias
            aim_offset = ctrl.aim_offset
            base_cmd = ctrl.k_pitch * (foe_v - (v + aim_offset))
            pitch_track = base_cmd + tent_bias
            pitch_track = max(-2.5, min(2.5, pitch_track))

        if has_target and not self.final_mode and self.dive_initiated:
            u, v = target_uv
            if v < ctrl.final_mode_v_threshold_low or v > ctrl.final_mode_v_threshold_high:
                self.final_mode = True

        pitch_rate_cmd = pitch_track
        thrust_cmd = thrust_track

        if self.final_mode and has_target:
            u, v = target_uv
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

        elif not has_target:
            pitch_rate_cmd = 0.0
            thrust_cmd = 0.5

        # Speed Limiting (Drag Simulation)
        if velocity_reliable and velocity_est:
             vx, vy, vz = velocity_est['vx'], velocity_est['vy'], velocity_est['vz']
             speed = math.sqrt(vx**2 + vy**2 + vz**2)

             limit = ctrl.velocity_limit
             if speed > limit:
                 # Brake
                 excess = speed - limit

                 # Reduce Thrust
                 thrust_cmd = max(ctrl.thrust_min, thrust_cmd - 0.05 * excess)

                 # Pitch Up (Flare)
                 brake_rate = ctrl.braking_pitch_gain * excess
                 brake_rate = min(brake_rate, ctrl.max_braking_pitch_rate)

                 pitch_rate_cmd += brake_rate

        if pitch < -1.45 and pitch_rate_cmd < 0.0:
             pitch_rate_cmd = 0.0
        if pitch > 0.8 and pitch_rate_cmd > 0.0:
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
        sim_x = 0.0; sim_y = 0.0; sim_z = pz
        for i in range(20):
             sim_x += sim_vx * self.dt
             sim_y += sim_vy * self.dt
             sim_z += sim_vz * self.dt
             path.append({'px': sim_x, 'py': sim_y, 'pz': sim_z})
        ghost_paths.append(path)

        return action, ghost_paths
