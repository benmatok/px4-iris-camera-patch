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
        pz = state_obs.get('pz', 100.0) # Default if missing
        vz = state_obs.get('vz')        # None if missing (Blind Mode)
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
            # Inverted Sim Logic: Positive u -> Positive Rate turns Right
            yaw_rate_cmd = self.k_yaw * u
        else:
            yaw_rate_cmd = extra_yaw_rate

        # 2. Estimate Distance and Compute Adaptive Gamma Ref
        # Use target_cmd (from MissionManager/Tracker) to get relative position
        # target_cmd is [rel_x, rel_y, abs_z_target] (Sim Frame)
        dx = target_cmd[0]
        dy = target_cmd[1]
        dist_est = math.sqrt(dx*dx + dy*dy)
        alt_est = max(0.1, pz - goal_z)

        # Improved Distance Estimation (Visual Angle)
        # If tracking, we use the visual angle (measurable) + altitude (measurable)
        # to estimate ground distance, rather than relying on the potentially perfect
        # 3D coordinates from the simulation harness (unmeasurable in reality).
        if tracking_uv:
             _, v = tracking_uv
             # Camera Tilt is fixed at 30 deg up
             tilt_rad = math.radians(30.0)

             # v is normalized tangent (y/z) in camera frame (Y-Down).
             # Positive v = Down from Camera Axis.
             angle_from_axis = math.atan(v)

             # Depression Angle relative to Horizon
             # Camera Axis Elevation = Pitch + Tilt
             # Ray Elevation = Axis Elevation - Angle From Axis
             # Depression = -Ray Elevation = Angle From Axis - (Pitch + Tilt)
             depression = angle_from_axis - (pitch + tilt_rad)

             if depression > 0.1: # Minimum angle to avoid singularity/horizon issues
                  dist_visual = alt_est / math.tan(depression)
                  # Blend or overwrite? Overwrite for realism.
                  dist_est = dist_visual
             # If depression is small/negative (looking up/level), fallback to state estimate (dist_est)
             # or max range? Fallback to State Estimate is safer for control stability.

        # Guidance Logic: Biased LOS (Flare Strategy)
        # We fly steeper than LOS to approach from below, creating a parabolic flare.
        los = -math.atan2(alt_est, dist_est)
        gamma_ref = los - math.radians(15.0)
        gamma_ref = max(gamma_ref, math.radians(-85.0))

        # Terminal Phase Safety: If close, fly directly at target to ensure hit
        if dist_est < 30.0:
             gamma_ref = los

        # 3. Speed Control (Thrust)
        if vz is not None:
            # Closed-Loop Speed Control
            speed_limit = 10.0
            vz_cmd = speed_limit * math.sin(gamma_ref)

            vz_err = vz_cmd - vz
            # Positive Error (Too fast/low) -> Increase Thrust
            thrust_cmd = self.thrust_hover + self.kp_vz * vz_err
            thrust_cmd = max(0.0, min(1.0, thrust_cmd))
        else:
            # Blind Mode (Open-Loop Thrust)
            # Default to hover thrust to maintain forward momentum without overspeeding
            thrust_cmd = self.thrust_hover

        # Limit Thrust in Steep Dive to prevent Forward Acceleration
        # Use Command gamma_ref instead of current pitch to anticipate
        if gamma_ref < math.radians(-45.0) or pitch < math.radians(-45.0):
             thrust_cmd = min(thrust_cmd, 0.35)

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
        if vz is not None:
            sim_vz = vz
        else:
            # Blind Mode: Assume nominal dive speed for viz
            sim_vz = 5.0 * math.sin(gamma_ref)

        for i in range(20):
             sim_z += sim_vz * self.dt
             path.append({'px': 0.0, 'py': 0.0, 'pz': sim_z})
        ghost_paths.append(path)

        return action, ghost_paths
