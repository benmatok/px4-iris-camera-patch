import numpy as np
import logging
import math

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
    def __init__(self, dt=0.05, mode='PID'):
        self.dt = dt
        self.mode = mode

        # Gains
        self.k_yaw = 2.5
        self.k_pitch = 4.0 # Increased for tighter tracking
        self.k_rer = 1.0

        # State Variables
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

        # Blind Mode Estimation
        self.est_pz = 100.0
        self.est_vz = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0

        self.last_tracking_size = None

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        self.est_pz = 100.0
        self.est_vz = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0
        self.last_tracking_size = None
        logger.info(f"DPCFlightController reset.")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, tracking_size=None, extra_yaw_rate=0.0, foe_uv=None):
        # Unpack State
        obs_pz = state_obs.get('pz')
        obs_vz = state_obs.get('vz')
        roll = state_obs['roll']
        pitch = state_obs['pitch']
        yaw = state_obs['yaw']

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
            if self.est_pz < 0.0: self.est_pz = 0.0; self.est_vz = 0.0
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
        ttc_visual = 100.0
        rer = 0.0

        if tracking_size is not None and self.last_tracking_size is not None and self.last_tracking_size > 0.001:
            rer = (tracking_size - self.last_tracking_size) / (self.last_tracking_size * self.dt)
            if rer > 0.01:
                 ttc_visual = 1.0 / rer

        self.last_tracking_size = tracking_size

        # --- Control Logic ---

        # 1. Weights based on Altitude (pz)
        # Flare starts late (5m)
        w_safety = max(0.0, min(1.0, (5.0 - pz) / 4.0))

        # 2. Tracking Control (Visual Servoing)
        pitch_track = 0.0
        yaw_rate_cmd = 0.0
        thrust_track = 0.6 # Base Cruise Thrust

        # Visual Braking
        visual_brake_active = False
        if ttc_visual < 2.5:
             visual_brake_active = True

        if tracking_uv:
            u, v = tracking_uv

            # Yaw Logic (ENU: +Yaw is Left)
            yaw_rate_cmd = -self.k_yaw * u
            yaw_rate_cmd = max(-1.5, min(1.5, yaw_rate_cmd))

            # Pitch Logic (ENU: +Pitch is Nose Up)
            # Add bias to aim above target (compensate gravity)
            # Adaptive Bias: Steep dives (Scen 1) need less bias to stay on target.
            # Shallow glides (Scen 2) need more bias to prevent undershoot.
            pitch_bias = 0.15
            if pitch < -0.8:
                # Ramp down bias from 0.15 at -0.8 to 0.05 at -1.2
                factor = max(0.0, min(1.0, (pitch - (-1.2)) / 0.4))
                pitch_bias = 0.05 + 0.1 * factor

            pitch_track = -self.k_pitch * v + pitch_bias
            pitch_track = max(-2.5, min(2.5, pitch_track))

            # Thrust Modulation

            # Case 1: Target High (v < 0). Need to Pull Up. Boost Thrust significantly.
            if v < 0:
                thrust_track += -v * 2.0 # Strong boost (Increased from 1.2)

            # Case 2: Steep Dive (Pitch < -0.5) and Target Low (v > 0). Reduce Thrust to Drop.
            # But don't reduce too much to avoid stalling/losing authority.
            if pitch < -0.5 and v >= 0:
                scale = min(1.0, max(0.0, (-0.5 - pitch) / 1.0))
                thrust_track -= scale * 0.3 # Reduced reduction (was 0.45) to keep speed

            # Visual Braking Boost
            if visual_brake_active:
                brake_factor = min(1.0, max(0.0, (2.5 - ttc_visual) / 2.0))
                thrust_track = max(thrust_track, 0.6 + 0.3 * brake_factor)

                if ttc_visual < 1.0:
                     pitch_track = max(pitch_track, 0.5)

            thrust_track = max(0.15, min(0.95, thrust_track))

        else:
            # Blind / Lost Tracking
            yaw_rate_cmd = 0.0
            pitch_track = 0.0
            thrust_track = 0.55

        # 3. Safety Control (Flare)
        target_pitch_safety = math.radians(5.0)
        pitch_safety = 3.0 * (target_pitch_safety - pitch)

        thrust_safety = 0.75

        # 4. Blending
        if tracking_uv:
            # If tracking, use visual inputs heavily until very close
            # w_safety modulates based on altitude (starts at 5m)
            pitch_rate_cmd = (1.0 - w_safety) * pitch_track + w_safety * pitch_safety
            thrust_cmd = (1.0 - w_safety) * thrust_track + w_safety * thrust_safety
        else:
            if pz < 5.0:
                 pitch_rate_cmd = 0.5
                 thrust_cmd = 0.6
            else:
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
        # (+Rate -> -Angle change).
        # We computed rates for Standard ENU (+Rate -> +Angle).
        # So we must negate Pitch and Yaw rates here.

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

        # Simple drag/lift model for estimation
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
