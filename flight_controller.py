import numpy as np
import logging
import math
from flight_config import FlightConfig

logger = logging.getLogger(__name__)

class DPCFlightController:
    """
    Standard ENU (East-North-Up) Flight Controller.

    Coordinate System:
    - Position (px, py, pz): ENU. pz is Altitude (Positive Up).
    - Velocity (vx, vy, vz): ENU. vz is Vertical Speed (Positive Up).
    - Attitude (roll, pitch, yaw): radians.
        - Yaw: Counter-Clockwise from East (Standard Math).
        - Pitch: Positive Nose Up (Standard Aerospace/ENU).
        - Roll: Positive Right Wing Down.

    Control Outputs (Rates):
    - thrust: 0.0 to 1.0 (Unitless).
    - roll_rate, pitch_rate, yaw_rate: rad/s.
      - Positive Pitch Rate -> Pitch Angle Increases (Nose Up).
      - Positive Yaw Rate -> Yaw Angle Increases (Turn Left).
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
        self.rer_smoothed = 0.0

        # Final Mode State
        self.final_mode = False
        self.locked_pitch = 0.0

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.est_pz = 100.0
        self.est_vz = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0
        self.last_tracking_size = None
        self.rer_smoothed = 0.0

        self.final_mode = False
        self.locked_pitch = 0.0
        logger.info(f"DPCFlightController reset.")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, tracking_size=None, extra_yaw_rate=0.0, foe_uv=None):
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
            # Calculate raw RER (Relative Expansion Rate)
            # RER = size_dot / size
            raw_rer = (tracking_size - self.last_tracking_size) / (self.last_tracking_size * self.dt)

        # Exponential Moving Average for RER smoothing
        alpha_rer = vis.rer_smoothing_alpha
        self.rer_smoothed = (1.0 - alpha_rer) * self.rer_smoothed + alpha_rer * raw_rer

        self.last_tracking_size = tracking_size

        # --- Control Logic ---

        # 2. Tracking Control (Visual Servoing)
        pitch_track = 0.0
        yaw_rate_cmd = 0.0
        thrust_track = 0.6 # Base Trim Thrust

        if tracking_uv:
            u, v = tracking_uv

            # Yaw Logic (ENU: +Yaw is Left)
            yaw_rate_cmd = -ctrl.k_yaw * u
            yaw_rate_cmd = max(-1.5, min(1.5, yaw_rate_cmd))

            # Pitch Logic (ENU: +Pitch is Nose Up)
            # Aggressive Adaptive Pitch Bias
            # Steep (-1.2) -> Bias 0.0 (Aim Straight)
            # Shallow (-0.6) -> Bias 0.2 (Aim High)
            pitch_bias = ctrl.pitch_bias_intercept + ctrl.pitch_bias_slope * pitch
            # Clamp limits reasonable for bias
            pitch_bias = max(ctrl.pitch_bias_min, min(ctrl.pitch_bias_max, pitch_bias))

            # --- Camera Tilt Compensation ---
            # Adaptive v_target based on pitch ("Tent" function).
            # Peak at pitch = -1.2 (Medium dive) -> v_target = 0.27 (Aim higher to extend range)
            # Steep (< -1.2) -> Reduce v_target to aim down (0.17 at -1.5)
            # Shallow (> -1.2) -> Reduce v_target to prevent float (0.19 at -0.5)
            if pitch < ctrl.v_target_pitch_threshold:
                 v_target = ctrl.v_target_intercept + ctrl.v_target_slope_steep * (pitch - ctrl.v_target_pitch_threshold)
            else:
                 v_target = ctrl.v_target_intercept + ctrl.v_target_slope_shallow * (pitch - ctrl.v_target_pitch_threshold)

            v_target = max(0.1, v_target)

            pitch_track = -ctrl.k_pitch * (v - v_target) + pitch_bias
            pitch_track = max(-2.5, min(2.5, pitch_track))

            # --- Constant RER Thrust Strategy ---
            rer_target = ctrl.rer_target # Target RER (TTC ~ 2.8s)

            # Pitch-Dependent Base Thrust
            # Steep Dive (-1.5) -> Low Thrust (0.15) to maintain steep glide slope.
            # Level/Shallow (0.0) -> High Thrust (0.6) to maintain lift/speed.
            thrust_base = ctrl.thrust_base_intercept + ctrl.thrust_base_slope * pitch
            thrust_base = max(ctrl.thrust_min, min(ctrl.thrust_max, thrust_base))

            # One-Sided RER Feedback (Brake Only)
            k_rer = ctrl.k_rer
            thrust_correction = max(0.0, k_rer * (self.rer_smoothed - rer_target))

            thrust_track = thrust_base + thrust_correction

            # Additional Heuristic: If we need to pull up significantly (Target High), Boost Thrust
            if v < v_target - 0.2:
                thrust_track += -(v - v_target) * 2.0

            thrust_track = max(0.1, min(0.95, thrust_track))

            # Flare Logic: Pitch Up if Collision Imminent (RER High)
            # Delayed Flare (Threshold +0.25)
            if self.rer_smoothed > rer_target + ctrl.flare_threshold_offset:
                flare_pitch = ctrl.flare_gain * (self.rer_smoothed - (rer_target + ctrl.flare_threshold_offset))
                pitch_track += flare_pitch

        else:
            # Blind / Lost Tracking
            yaw_rate_cmd = 0.0
            pitch_track = 0.0
            thrust_track = 0.55

        # --- Stage 3: Finale (Docking) Logic ---
        # Trigger: Target moves high in frame (v < -0.1) OR Low in frame (Overshoot v > 0.4).
        # Only enter if tracking is valid.
        if tracking_uv and not self.final_mode:
            u, v = tracking_uv
            if v < ctrl.final_mode_v_threshold_low or v > ctrl.final_mode_v_threshold_high:
                logger.info(f"Entering Final Mode! v={v:.2f}, RER={self.rer_smoothed:.2f}")
                self.final_mode = True

        # 4. Final Mode Execution
        if self.final_mode and tracking_uv:
            u, v = tracking_uv

            # Yaw: Slides to center X (Dampened)
            yaw_rate_cmd = -ctrl.k_yaw * u * ctrl.final_mode_yaw_gain
            yaw_rate_cmd = max(-ctrl.final_mode_yaw_limit, min(ctrl.final_mode_yaw_limit, yaw_rate_cmd))

            # Split Logic for Undershoot vs Overshoot
            if v < 0.1:
                # Undershoot / Recovery (Target High) -> Level & Power
                target_pitch_final = ctrl.final_mode_undershoot_pitch_target
                pitch_rate_cmd = ctrl.final_mode_undershoot_pitch_gain * (target_pitch_final - pitch)

                # Thrust: Modulates to center Y (v=0)
                # Gain 2.0 to climb
                thrust_cmd = ctrl.final_mode_undershoot_thrust_base - ctrl.final_mode_undershoot_thrust_gain * v
            else:
                # Overshoot / Sink (Target Low) -> Nose Down & Gentle Sink
                # Keep nose down (-5 deg) to maintain view/momentum
                target_pitch_final = ctrl.final_mode_overshoot_pitch_target
                pitch_rate_cmd = ctrl.final_mode_overshoot_pitch_gain * (target_pitch_final - pitch)

                # Thrust: Modulates to gently sink towards ideal v=0.24
                # If v=0.4, error=0.16. Thrust = 0.55 - 1.0 * 0.16 = 0.39.
                # Not a hard cut to 0.1
                thrust_cmd = ctrl.final_mode_overshoot_thrust_base - ctrl.final_mode_overshoot_thrust_gain * (v - ctrl.final_mode_overshoot_v_target)

            thrust_cmd = max(0.1, min(0.95, thrust_cmd))

        elif tracking_uv:
            pitch_rate_cmd = pitch_track
            thrust_cmd = thrust_track
        else:
            # Blind / Lost Tracking
            pitch_rate_cmd = 0.0
            thrust_cmd = 0.5

        # 5. Prevent Inverted Flight AND Stall
        if pitch < -1.45 and pitch_rate_cmd < 0.0:
             pitch_rate_cmd = 0.0

        if pitch > 0.8 and pitch_rate_cmd > 0.0:
             pitch_rate_cmd = 0.0

        # 6. Roll Control
        roll_rate_cmd = 4.0 * (0.0 - roll)

        # NOTE: The Sim Interface (PyGhostModel) uses Inverted Pitch/Yaw logic internally.
        action = {
            'thrust': thrust_cmd,
            'roll_rate': roll_rate_cmd,
            'pitch_rate': -pitch_rate_cmd,
            'yaw_rate': -yaw_rate_cmd
        }
        self.last_action = action

        # --- Ghost Paths (Viz) ---
        sim_vx = self.est_vx
        sim_vy = self.est_vy
        sim_vz = vz

        if abs(pitch) > 0.1:
             v_xy_est = abs(sim_vz / math.tan(pitch))
             v_xy_est = min(20.0, v_xy_est)
             sim_vx = 0.9 * sim_vx + 0.1 * (v_xy_est * math.cos(yaw))
             sim_vy = 0.9 * sim_vy + 0.1 * (v_xy_est * math.sin(yaw))
             self.est_vx = sim_vx
             self.est_vy = sim_vy
        else:
             self.est_vx *= 0.95
             self.est_vy *= 0.95

        ghost_paths = []
        path = []
        sim_x = 0.0; sim_y = 0.0; sim_z = pz

        for i in range(20):
             sim_x += self.est_vx * self.dt
             sim_y += self.est_vy * self.dt
             sim_z += sim_vz * self.dt
             path.append({'px': sim_x, 'py': sim_y, 'pz': sim_z})
        ghost_paths.append(path)

        return action, ghost_paths
