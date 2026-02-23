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

        if is_reliable and hasattr(self.config, 'gdpc'):
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

             action, trajs_enu = self.gdpc.compute_action(gdpc_state, target_pos_relative)

             self.last_action = {
                 'thrust': action['thrust'],
                 'roll_rate': action['roll_rate'],
                 'pitch_rate': action['pitch_rate'],
                 'yaw_rate': action['yaw_rate']
             }

             # Ghost Paths (Already ENU)
             ghost_paths = []

             # trajs_enu is a list of numpy arrays (Ensemble)
             if isinstance(trajs_enu, list):
                 for traj in trajs_enu:
                     path = []
                     for i in range(len(traj)):
                         path.append({'px': traj[i, 0], 'py': traj[i, 1], 'pz': traj[i, 2]})
                     ghost_paths.append(path)
             else:
                 # Legacy single trajectory
                 path = []
                 for i in range(len(trajs_enu)):
                     path.append({'px': trajs_enu[i, 0], 'py': trajs_enu[i, 1], 'pz': trajs_enu[i, 2]})
                 ghost_paths.append(path)

             # We need to return action in the dict format expected by DPCFlightController's invoker
             # BUT, DPCFlightController modifies action at the end of this method (lines 284+).
             # It sets self.last_action, but returns `action` constructed from local variables.

             # We should assign to local variables `thrust_cmd`, `roll_rate_cmd`, etc.
             # And skip the rest of the visual servoing logic.
             # Or just return immediately?
             # Returning immediately is safer to avoid interference.

             # BUT: The end of the method does the inversion:
             # action = { ..., 'pitch_rate': -pitch_rate_cmd, ... }
             # So we must manually do the inversion if we return early, OR assign to vars and let it flow.

             # Let's return early but format it correctly.

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

            yaw_rate_cmd = -ctrl.k_yaw * u
            yaw_rate_cmd = max(-1.5, min(1.5, yaw_rate_cmd))

            if not self.dive_initiated:
                vz_err = 0.0 - self.est_vz
                pitch_track = ctrl.cruise_pitch_gain * vz_err
                pitch_track = max(-1.0, min(1.0, pitch_track))
                thrust_track = 0.55
            else:
                pitch_bias = ctrl.pitch_bias_intercept + ctrl.pitch_bias_slope * pitch
                pitch_bias = max(ctrl.pitch_bias_min, min(ctrl.pitch_bias_max, pitch_bias))

                # Simplified Speed Control
                # Target speed slightly below limit to allow headroom
                v_target = ctrl.velocity_limit - 1.0

                # Positive Pitch Rate (Nose Up) slows down.
                # If v > v_target, term (v-v_target) is positive.
                # We want Pitch Up. So sign must be POSITIVE.
                pitch_track = +ctrl.k_pitch * (v - v_target) + pitch_bias

                # Limit Dive Angle (Sim Pitch Positive = Nose Down)
                # If Sim Pitch > 0.3 (17 deg), don't pitch down further (Negative Rate)
                # Wait, pitch_track is Rate. Positive = Nose Up.
                # If pitch_track < 0 (Nose Down) and pitch > 0.3 (Already Steep)
                # Clamp to 0.
                # BUT, earlier I said Sim Pitch Negative = Nose Down?
                # "Scenario 6 Start: pitch = -39.5 deg".
                # So Negative is Nose Down.
                # If pitch < -0.3 (-17 deg). And pitch_track < 0 (Nose Down rate).
                # Clamp.
                if pitch < -0.3 and pitch_track < 0.0:
                     pitch_track = 0.0

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
            pitch_rate_cmd = 0.0
            thrust_cmd = 0.5

        # Enforce Speed Limit (Smooth Active Braking)
        if velocity_est:
             v_mag_est = math.sqrt(velocity_est['vx']**2 + velocity_est['vy']**2 + velocity_est['vz']**2)

             # Sanity check: Only use VIO for braking if it is believable (< 15.0 m/s)
             # If VIO diverges (e.g. 100 m/s), ignoring it is safer than panic braking.
             if v_mag_est < 15.0:
                  limit = self.config.control.velocity_limit

                  # Soft Thrust Reduction
                  if v_mag_est > limit - 2.0:
                      ratio = (v_mag_est - (limit - 2.0)) / 2.0
                      ratio = max(0.0, min(1.0, ratio))
                      thrust_cmd = 0.1 + (thrust_cmd - 0.1) * (1.0 - ratio)

                  # Pitch Braking (If exceeding limit)
                  if v_mag_est > limit:
                      excess = v_mag_est - limit
                      # Gentle pitch up (Nose Up) to airbrake
                      pitch_brake = 0.5 * excess
                      pitch_rate_cmd += pitch_brake

                      # Clamp to prevent violent maneuvers
                      pitch_rate_cmd = max(-1.0, min(1.0, pitch_rate_cmd))

                      logger.debug(f"SMOOTH BRAKING: v={v_mag_est:.2f}, brake={pitch_brake:.2f}, thrust={thrust_cmd:.2f}")
             elif v_mag_est > 15.0:
                  logger.warning(f"VIO Divergence Detected (v={v_mag_est:.2f} > 15.0). Ignoring for control.")

        # Safety Clamps
        # Sim Pitch Positive = Nose Up.
        # Max Pitch Up = 0.2 rad (~11 deg) to avoid stalling/backward flight.
        # Max Pitch Down = -1.0 rad (~57 deg) to avoid overspeed.

        # If Pitch > 0.0 (Horizon). Prevent further Nose Up to avoid backward flight.
        # pitch_rate_cmd Positive = Nose Up command.
        if pitch > 0.0 and pitch_rate_cmd > 0.0:
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
