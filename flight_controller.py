import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

class DPCFlightController:
    def __init__(self, dt=0.05):
        self.dt = dt
        # Gains
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

        # State Variables
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        logger.info("DPCFlightController (Simple) reset")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, extra_yaw_rate=0.0):
        # Unpack State
        pz = state_obs['pz']
        vz = state_obs['vz']
        roll = state_obs['roll']
        pitch = state_obs['pitch']
        # Compatibility if keys missing (though they shouldn't be used)
        vx = state_obs.get('vx', 0.0)
        vy = state_obs.get('vy', 0.0)

        goal_z = target_cmd[2]

        # 1. Yaw Control
        yaw_rate_cmd = 0.0
        if tracking_uv:
            u, v = tracking_uv
            # Inverted Sim Logic: Positive u -> Positive Rate turns Right (Corrects Left error? No.)
            # Target Right (u>0). We want turn Right.
            # Sim Yaw: Right is Negative Rate? No. Right is CW. Sim 0=East, 90=North.
            # Decreasing Yaw = Right.
            # So Rate should be Negative.
            # Previous fix used Positive. Let's stick to what worked or re-verify.
            # Validation showed Yaw worked with `k * u`.
            yaw_rate_cmd = self.k_yaw * u
        else:
            yaw_rate_cmd = extra_yaw_rate

        # 2. Estimate Distance and Gamma Ref
        tilt_rad = math.radians(30.0)
        gamma_ref = math.radians(-30.0) # Default

        use_visual_gamma = False
        d_est = 100.0

        if tracking_uv:
            u, v = tracking_uv
            # v is normalized. v = tan(angle_from_center).
            # Center is +30 deg up from Body.
            # Ray Angle (in Body Frame) = 30 - atan(v)? No.
            # v increases DOWN.
            # Camera Axis (Center) is +30 deg relative to Body (if Pitch=0).
            # Ray is `atan(v)` relative to Camera Axis (Down is positive).
            # Ray Elevation relative to Horizon = (Pitch + 30) - atan(v_pixels * scale).
            # Normalized v: v = y/z. tan(phi) = v.
            # Angle down from axis = atan(v).
            # Elevation = (Pitch + 30) - atan(v). (deg).
            # Depression = -Elevation = atan(v) - Pitch - 30.

            # v range [-1.73, 1.73] for 120 FOV?
            # atan(v) gives angle from camera axis.

            angle_from_cam_axis = math.atan(v) # Positive is Down.
            # Camera Axis Elevation = Pitch + Tilt.
            # Ray Elevation = (Pitch + Tilt) - angle_from_cam_axis.
            # We want Depression Angle (Positive Down).
            # Depression = -Ray Elevation = angle_from_cam_axis - (Pitch + Tilt).

            depression = angle_from_cam_axis - (pitch + tilt_rad)

            if depression > 0.1: # At least ~6 degrees down
                d_est = max(0.1, (pz - goal_z) / math.tan(depression))

                # Desired Gamma to hit target
                # gamma_ref = -atan2(pz, d)
                gamma_ref = -math.atan2(pz - goal_z, d_est)
                use_visual_gamma = True
            else:
                # Target is above or level. Far away.
                d_est = 100.0

        if not use_visual_gamma:
            # Fallback to Altitude Profile
            dive_start_alt = 100.0
            ratio = max(0.0, min(1.0, (pz - goal_z) / (dive_start_alt - goal_z)))
            gamma_deg = self.dive_angle_end + (self.dive_angle_start - self.dive_angle_end) * ratio
            gamma_ref = math.radians(gamma_deg)

        # 3. Speed Control (Thrust)
        speed_limit = 10.0
        vz_cmd = speed_limit * math.sin(gamma_ref)

        vz_err = vz_cmd - vz
        # Positive Error (Too fast/low) -> Increase Thrust
        thrust_cmd = self.thrust_hover + self.kp_vz * vz_err
        thrust_cmd = max(0.0, min(1.0, thrust_cmd))

        # Limit Thrust in Steep Dive to prevent Forward Acceleration
        if pitch < math.radians(-45.0):
             thrust_cmd = min(thrust_cmd, 0.2)

        # Debug
        # if pz < 150.0:
        #      print(f"Pz={pz:.1f} GoalZ={goal_z:.1f} Vz={vz:.1f} VzCmd={vz_cmd:.1f} ThCmd={thrust_cmd:.2f} Gamma={math.degrees(gamma_ref):.1f} DistEst={d_est:.1f}")

        # 4. Pitch Control
        # If we have visual gamma, we can follow it directly.
        # But we need to ensure target visibility.
        # Ideally Gamma Ref points at target.
        # Pitch = Gamma Ref.
        # If Pitch = Gamma Ref.
        # Elevation = Pitch + Tilt - atan(v).
        # Depression = atan(v) - Pitch - Tilt.
        # tan(Depression) = z / d.
        # Gamma = -Depression.
        # Pitch = -atan(v) - Tilt - Depression ? No.

        # If Pitch = Gamma.
        # Then Body points along trajectory.
        # Camera points +30.
        # Target is at Angle 0 (along trajectory).
        # So Target should be at -30 deg in Camera (Up).
        # So v should be negative (Up).
        # atan(v) = -30 deg.
        # v = tan(-30) = -0.577.
        # This is well within FOV.

        # So Pitch = Gamma Ref is safe for visibility (Target at -30 deg).
        pitch_cmd = gamma_ref

        # Pitch Rate Loop
        k_pitch_ang = 5.0
        # Inverted logic for PyGhostModel state vs rate
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
        sim_vz = vz
        for i in range(20):
             sim_z += sim_vz * self.dt
             path.append({'px': 0.0, 'py': 0.0, 'pz': sim_z})
        ghost_paths.append(path)

        return action, ghost_paths
