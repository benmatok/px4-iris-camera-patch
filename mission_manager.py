import logging
import math
from ghost_dpc.target_estimator import TargetEstimator

logger = logging.getLogger(__name__)

class MissionManager:
    def __init__(self, target_alt=50.0):
        self.target_alt = target_alt
        self.estimator = TargetEstimator()
        self.reset()

    def reset(self, target_alt=None):
        self.state = "TAKEOFF"
        if target_alt is not None:
            self.target_alt = target_alt
        self.dpc_target = [0.0, 0.0, self.target_alt] # Z-Up Target
        self.estimator.reset()
        logger.info(f"MissionManager reset to TAKEOFF with Target Alt: {self.target_alt}")

    def update(self, drone_state_sim, detection_result):
        """
        Updates the mission state and determines the current target.

        Args:
            drone_state_sim: dict (px, py, pz... in Z-Up Sim Frame)
            detection_result: (center_pixel, target_world_pos) from tracker

        Returns:
            mission_state: str
            dpc_target: list [x, y, z] (Z-Up)
            extra_yaw: float (yaw rate override)
        """
        center, target_wp = detection_result
        current_alt = drone_state_sim['pz'] # Sim Z is Up (Altitude)

        # Get Velocities for Estimator
        vx = drone_state_sim.get('vx', 0.0)
        vy = drone_state_sim.get('vy', 0.0)
        vz = drone_state_sim.get('vz', 0.0)

        # Update Estimator
        self.estimator.predict([vx, vy, vz])
        if target_wp is not None:
            self.estimator.update(target_wp)

        est_rel_pos = self.estimator.get_estimate()

        extra_yaw = 0.0

        if self.state == "TAKEOFF":
            # Hold position (hover) at TARGET_ALT
            # Target is [Current X, Current Y, TARGET_ALT] in Z-Up
            self.dpc_target = [drone_state_sim['px'], drone_state_sim['py'], self.target_alt]

            if center is not None:
                self.state = "HOMING"
            # Transition to SCAN if close to altitude
            elif current_alt >= self.target_alt - 5.0:
                self.state = "SCAN"

        elif self.state == "SCAN":
            # Spin to find target
            extra_yaw = math.radians(15.0)

            # Maintain hover at current position but adjust to target altitude
            # This prevents the stale target issue causing dives
            self.dpc_target = [drone_state_sim['px'], drone_state_sim['py'], self.target_alt]

            if center is not None:
                self.state = "HOMING"
                extra_yaw = 0.0

        elif self.state == "HOMING":
            # Use Estimator for Target Position
            # dpc_target is [RelX, RelY, AbsZ]

            # Compute Absolute Target Z based on Estimate
            # est_rel_pos[2] is Relative Z (Target - Drone)
            # AbsTargetZ = DroneZ + RelZ
            # We want to fly 2.0m above Target
            target_abs_z = current_alt + est_rel_pos[2] + 2.0

            # If target_abs_z < 2.0, clamp it to 2.0 (Safety floor)
            if target_abs_z < 2.0: target_abs_z = 2.0

            self.dpc_target = [est_rel_pos[0], est_rel_pos[1], target_abs_z]

            if self.estimator.is_lost and self.estimator.time_since_last_seen > 2.0:
                logger.warning("Target Lost > 2.0s. Switching to LOST_RECOVERY.")
                self.state = "LOST_RECOVERY"

        elif self.state == "LOST_RECOVERY":
            # Climb and Rotate
            # Calculate Yaw to Last Known Position
            # est_rel_pos is [RelX, RelY, RelZ] in Sim Frame
            rel_x = est_rel_pos[0]
            rel_y = est_rel_pos[1]

            # Yaw to Target
            yaw_to_target = math.atan2(rel_y, rel_x)
            current_yaw = drone_state_sim.get('yaw', 0.0)

            # Normalize diff
            yaw_diff = yaw_to_target - current_yaw
            yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi

            # If diff is large, rotate fast towards it. If small, spin slowly to scan.
            # Threshold: 20 degrees (approx 0.35 rad)
            if abs(yaw_diff) > 0.35:
                 # Proportional Yaw Control towards target
                 extra_yaw = 2.0 * yaw_diff
                 # Clamp
                 extra_yaw = max(-1.0, min(1.0, extra_yaw))
            else:
                 # Scan rotation
                 extra_yaw = math.radians(20.0)

            # Target Altitude: Climb to 100m for better view (Dive Logic)
            # Was: max(current_alt + 5.0, 20.0)
            # Now: Aim for 100.0 directly if possible, or climb
            recovery_alt = 100.0

            # Hover over Estimated Target Position
            # dpc_target[0, 1] are Relative Setpoints
            self.dpc_target = [est_rel_pos[0], est_rel_pos[1], recovery_alt]

            if center is not None:
                logger.info("Target Re-acquired! Switching to HOMING.")
                self.state = "HOMING"
                extra_yaw = 0.0

        return self.state, self.dpc_target, extra_yaw
