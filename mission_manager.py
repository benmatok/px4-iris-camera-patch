import logging
import math

logger = logging.getLogger(__name__)

class MissionManager:
    def __init__(self, target_alt=50.0):
        self.target_alt = target_alt
        self.reset()

    def reset(self):
        self.state = "TAKEOFF"
        self.dpc_target = [0.0, 0.0, self.target_alt] # Z-Up Target
        logger.info("MissionManager reset to TAKEOFF")

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

        extra_yaw = 0.0

        if self.state == "TAKEOFF":
            # Hold position (hover) at TARGET_ALT
            # Target is [Current X, Current Y, TARGET_ALT] in Z-Up
            self.dpc_target = [drone_state_sim['px'], drone_state_sim['py'], self.target_alt]

            # Transition to SCAN if close to altitude
            if current_alt >= self.target_alt - 5.0:
                self.state = "SCAN"

        elif self.state == "SCAN":
            extra_yaw = math.radians(15.0)

            if center is not None:
                self.state = "HOMING"

        elif self.state == "HOMING":
            if target_wp:
                # Target is found and localized
                # Fly to 2m above target
                self.dpc_target = [target_wp[0], target_wp[1], 2.0]

        return self.state, self.dpc_target, extra_yaw
