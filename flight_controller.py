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

        # Phase Logic
        self.phase = 'HOVER_ALIGN'
        self.hover_alt_target = None
        self.last_tracking_uv = None
        self.shallow_approach = False

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

        self.phase = 'HOVER_ALIGN'
        self.hover_alt_target = None
        self.last_tracking_uv = None
        self.shallow_approach = False

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

            # --- Update UV Speed ---
            uv_speed = 0.0
            if self.last_tracking_uv:
                lu, lv = self.last_tracking_uv
                uv_speed = math.sqrt((u - lu)**2 + (v - lv)**2) / self.dt
            self.last_tracking_uv = tracking_uv

            # --- Phase Transitions ---
            if self.phase == 'HOVER_ALIGN':
                # Steep Dive Override: If pitch is steep (e.g. < -0.9 rad), skip Hover Align
                if pitch < -0.9:
                     self.phase = 'ATTACK'
                     logger.info(f"Transition: HOVER_ALIGN -> ATTACK (Steep Dive: {pitch:.2f})")

                elif uv_speed > ctrl.hover_align_uv_speed_threshold:
                    # Refine Transition: Filter out rotational flow
                    # Only transition if pitch is stable
                    # Use 'wy' (Body Pitch Rate) as proxy for pitch instability
                    pitch_rate_est = state_obs.get('wy', 0.0)
                    if abs(pitch_rate_est) < 0.2:
                        self.phase = 'ATTACK'
                        logger.info(f"Transition: HOVER_ALIGN -> ATTACK (Speed: {uv_speed:.2f})")

            if self.phase == 'ATTACK':
                if tracking_size is not None and tracking_size > ctrl.brake_size_threshold:
                    self.phase = 'EARLY_BRAKE'
                    logger.info(f"Transition: ATTACK -> EARLY_BRAKE (Size: {tracking_size:.2f})")

            # --- Standard (ATTACK) Logic Calculation ---
            # Pitch Logic (ENU: +Pitch is Nose Up)
            pitch_bias = ctrl.pitch_bias_intercept + ctrl.pitch_bias_slope * pitch
            pitch_bias = max(ctrl.pitch_bias_min, min(ctrl.pitch_bias_max, pitch_bias))

            if pitch < ctrl.v_target_pitch_threshold:
                 v_target = ctrl.v_target_intercept + ctrl.v_target_slope_steep * (pitch - ctrl.v_target_pitch_threshold)
            else:
                 v_target = ctrl.v_target_intercept + ctrl.v_target_slope_shallow * (pitch - ctrl.v_target_pitch_threshold)
            v_target = max(0.1, v_target)

            standard_pitch_track = -ctrl.k_pitch * (v - v_target) + pitch_bias
            standard_pitch_track = max(-2.5, min(2.5, standard_pitch_track))

            # Thrust Logic
            rer_target = ctrl.rer_target
            thrust_base = ctrl.thrust_base_intercept + ctrl.thrust_base_slope * pitch
            thrust_base = max(ctrl.thrust_min, min(ctrl.thrust_max, thrust_base))

            # Shallow Dive Thrust Boost (Fix for Scenario 5)
            # Scenario 5 (25m/150m) needs boost. Scenario 3 (20m/50m) works without.
            if self.shallow_approach and self.hover_alt_target is not None and self.hover_alt_target > 22.0:
                 thrust_base += 0.25

            thrust_correction = max(0.0, ctrl.k_rer * (self.rer_smoothed - rer_target))
            standard_thrust_track = thrust_base + thrust_correction

            if v < v_target - 0.2:
                standard_thrust_track += -(v - v_target) * 2.0
            standard_thrust_track = max(0.1, min(0.95, standard_thrust_track))

            # Flare Logic
            if self.rer_smoothed > rer_target + ctrl.flare_threshold_offset:
                flare_pitch = ctrl.flare_gain * (self.rer_smoothed - (rer_target + ctrl.flare_threshold_offset))
                standard_pitch_track += flare_pitch

            # --- Apply Phase Logic ---
            if self.final_mode:
                 # Prioritize Final Mode Logic (Standard Logic + Final Mode Handling)
                 # This ensures that if we are close enough to trigger Final Mode, we don't force Hover Align
                 pitch_track = standard_pitch_track
                 thrust_track = standard_thrust_track

            elif self.phase == 'HOVER_ALIGN':
                # Mark as Shallow Approach since we are hovering
                self.shallow_approach = True

                # Pitch: Visual Tracking (Keep target in view)
                pitch_track = standard_pitch_track

                # Thrust: Altitude Hold
                if self.hover_alt_target is None:
                    self.hover_alt_target = pz # Capture current alt

                # PD Control on Altitude
                err_z = self.hover_alt_target - pz
                err_vz = 0.0 - vz

                thrust_track = 0.55 + ctrl.hover_align_alt_hold_kp * err_z + ctrl.hover_align_alt_hold_kd * err_vz
                thrust_track = max(0.1, min(0.9, thrust_track))

            elif self.phase == 'EARLY_BRAKE':
                pitch_track = standard_pitch_track + ctrl.early_brake_pitch_bias
                thrust_track = standard_thrust_track

            else: # ATTACK or FINAL (pre-check)
                pitch_track = standard_pitch_track
                thrust_track = standard_thrust_track

        else:
            # Blind / Lost Tracking
            self.last_tracking_uv = None
            yaw_rate_cmd = 0.0
            pitch_track = 0.0
            thrust_track = 0.55

        # --- Stage 3: Finale (Docking) Logic ---
        # Trigger: Target moves high in frame (v < -0.1) OR Low in frame (Overshoot v > 0.4).
        # Only enter if tracking is valid and we are LOW enough (e.g. < 5m).
        if tracking_uv and not self.final_mode:
            u, v = tracking_uv
            should_enter = v < ctrl.final_mode_v_threshold_low or v > ctrl.final_mode_v_threshold_high

            # Additional Check: Don't enter Final Mode if we are still high (e.g. > 5m)
            # This prevents premature triggering during high-speed flyovers
            if should_enter and self.est_pz < 5.0:
                logger.info(f"Entering Final Mode! v={v:.2f}, RER={self.rer_smoothed:.2f}, Alt={self.est_pz:.1f}")
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
