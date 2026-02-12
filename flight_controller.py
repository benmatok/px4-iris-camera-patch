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
        # Start steep (e.g., -60 deg) and end shallow (-15 deg).
        self.dive_angle_start = -60.0
        self.dive_angle_end = -15.0

        # Velocity Control Gains (Thrust Only)
        self.kp_vz = 2.0 # Thrust (normalized) per m/s error
        self.thrust_hover = 0.6

        # Pitch Control Gains (Visual Servoing)
        self.kp_pitch_v = 1.0 # Pitch angle (rad) per unit v error
        self.pitch_bias_max = math.radians(15.0)

        # State Variables
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

    def reset(self):
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
        logger.info("DPCFlightController (Simple) reset")

    def compute_action(self, state_obs, target_cmd, tracking_uv=None, extra_yaw_rate=0.0):
        """
        Simple Controller Implementation using ONLY Tracking and Height (pz, vz derived).
        No Vx, Vy used.
        """
        # Unpack State
        pz = state_obs['pz']
        vz = state_obs['vz']
        # vx, vy are NOT available
        roll = state_obs['roll']
        pitch = state_obs['pitch']

        # Goal Z
        goal_z = target_cmd[2]

        # 1. Yaw Control (Visual Servoing)
        yaw_rate_cmd = 0.0
        if tracking_uv:
            u, v = tracking_uv
            # u is normalized horizontal coordinate (Right > 0).
            # Yaw Left (Positive Rate) to correct Positive u.
            yaw_rate_cmd = self.k_yaw * (-u)
        else:
            yaw_rate_cmd = extra_yaw_rate

        # 2. Trajectory Profile (Flight Path Angle gamma_ref)
        # Based on Altitude `pz`.
        # High Altitude (e.g. > 50m): Steep Dive.
        # Low Altitude (Goal Z): Shallow Dive.

        # Calculate progress ratio (0 at goal, 1 at high alt)
        # Assume start dive at 100m.
        dive_start_alt = 100.0
        ratio = max(0.0, min(1.0, (pz - goal_z) / (dive_start_alt - goal_z)))

        # Gamma Ref (Degrees)
        gamma_deg = self.dive_angle_end + (self.dive_angle_start - self.dive_angle_end) * ratio
        gamma_ref = math.radians(gamma_deg) # Negative (Down)

        # 3. Speed Control (Thrust)
        # Limit Vz (Descent Rate) based on Gamma Ref and Speed Limit (10m/s).
        # Vz_target = -Speed * sin(-gamma_ref).
        # Actually Vz_target = Speed * sin(gamma_ref). (gamma is negative).
        # Vz_target is negative.

        speed_limit = 10.0
        vz_cmd = speed_limit * math.sin(gamma_ref)

        # Thrust Loop
        vz_err = vz_cmd - vz
        # If vz > vz_cmd (e.g. -2 > -8), we are too slow. Error > 0.
        # Need to drop faster -> Decrease Thrust.
        # If vz < vz_cmd (e.g. -12 < -8), too fast. Error < 0.
        # Need to brake -> Increase Thrust.
        # So Thrust ~ -Error.

        thrust_cmd = self.thrust_hover - self.kp_vz * vz_err
        thrust_cmd = max(0.0, min(1.0, thrust_cmd))

        # 4. Pitch Control (Visual Servoing for Trajectory)
        # We want to pitch such that we fly the trajectory.
        # Ideally Pitch ~ Gamma Ref (Low Alpha).
        # But we must keep target in view.
        # Basic Command: Set Pitch = Gamma Ref.
        pitch_cmd = gamma_ref

        # Visual Correction:
        # Check if target `v` is acceptable.
        # If we Pitch = Gamma Ref, where is the target?
        # Target Angle (Geo) = atan2(dz, dist).
        # Since we don't know dist, we rely on tracking.
        # If we are flying pure pursuit, v = 0.
        # If we want to shallow out (Pitch Up), v increases (Target drops in image).
        # We want to bias Pitch Up as we get lower.

        # Let's enforce Pitch = Gamma Ref directly (Open Loop Trajectory Shape).
        # But modify it to ensure Target is visible (Closed Loop Safety).

        if tracking_uv:
            u, v = tracking_uv
            # Calculate Pitch Correction to center target?
            # No, if we center target, we fly straight line (collision).
            # We WANT to fly a curved path.
            # So we allow `v` to be non-zero.
            # Just ensure it's in FOV.
            # FOV Vertical is +/- 45 deg (90 total? Or 60 total? Projector default is 120 FOV... Horizontal? Diagonal?)
            # Projector default: fov_deg=120. Likely Horizontal.
            # Assume ~45 deg Vertical half-angle.
            # Normalized v range [-1.73, 1.73] for 120 FOV?
            # Tan(60) = 1.73. So v=+/-1.73 corresponds to +/- 60 deg.

            # If v > 1.0 (approx 45 deg down in image), we are losing target bottom.
            # Pitch Down to bring it up.
            if v > 1.0:
                pitch_correction = -0.5 * (v - 1.0)
                pitch_cmd += pitch_correction

            # If v < -1.0 (approx 45 deg up in image), we are losing target top.
            # Pitch Up to bring it down.
            if v < -1.0:
                pitch_correction = -0.5 * (v + 1.0) # v+1 is neg. correction is pos.
                pitch_cmd += pitch_correction

        # Pitch Rate Loop
        k_pitch_ang = 5.0
        pitch_rate_cmd = k_pitch_ang * (pitch_cmd - pitch)

        # 5. Roll Control (Stabilize)
        roll_rate_cmd = self.k_roll * (0.0 - roll)

        action = {
            'thrust': thrust_cmd,
            'roll_rate': roll_rate_cmd,
            'pitch_rate': pitch_rate_cmd,
            'yaw_rate': yaw_rate_cmd
        }
        self.last_action = action

        # Ghost Paths (Viz) - Simple projection
        ghost_paths = []
        path = []
        sim_x, sim_y, sim_z = 0, 0, pz
        sim_vz = vz
        # Project vertical only since we don't know horizontal velocity
        for i in range(20):
             sim_z += sim_vz * self.dt
             path.append({'px': 0.0, 'py': 0.0, 'pz': sim_z})
        ghost_paths.append(path)

        return action, ghost_paths
