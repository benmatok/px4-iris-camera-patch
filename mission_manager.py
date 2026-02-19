import logging
import math
import time
from flight_config import FlightConfig

logger = logging.getLogger(__name__)

class MissionManager:
    def __init__(self, target_alt=None, enable_staircase=None, config: FlightConfig = None):
        self.config = config or FlightConfig()

        # Priority: explicit arg > config > default (handled by config default)
        self.target_alt = target_alt if target_alt is not None else self.config.mission.target_alt
        self.enable_staircase = enable_staircase if enable_staircase is not None else self.config.mission.enable_staircase

        self.staircase_start_time = 0.0
        self.reset()

    def reset(self, target_alt=None):
        self.state = "TAKEOFF"
        if target_alt is not None:
            self.target_alt = target_alt

        self.dpc_target = [0.0, 0.0, self.target_alt] # Z-Up Target
        self.staircase_target_z = 0.0
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

        mis = self.config.mission

        extra_yaw = 0.0

        if self.state == "TAKEOFF":
            self.dpc_target = [0.0, 0.0, self.target_alt]

            if center is not None:
                self.state = "HOMING"
            elif current_alt >= self.target_alt - 5.0:
                self.state = "SCAN"

        elif self.state == "SCAN":
            extra_yaw = math.radians(15.0)
            self.dpc_target = [0.0, 0.0, self.target_alt]

            if center is not None:
                self.state = "HOMING"
                extra_yaw = 0.0

        elif self.state == "HOMING":
            if target_wp:
                # Calculate Lateral Distance
                lat_dist = math.sqrt(target_wp[0]**2 + target_wp[1]**2)

                # Check for "Too High and Steep" Condition (Staircase Trigger)
                # Alt > 20m AND Dist < 10m (Steep)
                if self.enable_staircase and current_alt > mis.staircase_trigger_alt and lat_dist < mis.staircase_trigger_dist:
                    self.state = "STAIRCASE_DESCEND"
                    self.staircase_target_z = current_alt - mis.staircase_drop
                    if self.staircase_target_z < 5.0:
                         self.staircase_target_z = 5.0 # Floor for safety during staircasing
                    logger.info(f"Triggering STAIRCASE_DESCEND. Alt: {current_alt}, Dist: {lat_dist}, TargetZ: {self.staircase_target_z}")

                    # Target: Relative XY = 0 (Hover), Absolute Z = Target
                    # In HOMING, dpc_target[0,1] is relative vector to target.
                    # If we switch to Hover relative to *Drone*, we set dpc_target[0,1] = 0.
                    # But we want to hover *over the target*?
                    # "Drop 20m then stabilize and reacquire".
                    # If we hover over drone, we drift away from target if moving.
                    # Let's try to maintain XY tracking but force Z drop.
                    self.dpc_target = [target_wp[0], target_wp[1], self.staircase_target_z]
                else:
                    # Normal Homing (Intercept)
                    # Fly to 0.0m (Collision allowed)
                    # Target is at Z=0 World. Relative Z = 0 - current_alt = -current_alt
                    # Add Safety Buffer: Aim for Z=5.0m to avoid ground impact during approach
                    self.dpc_target = [target_wp[0], target_wp[1], -current_alt + 5.0]
            else:
                # Lost Tracking
                dt = 0.05
                vx = drone_state_sim.get('vx', 0.0)
                vy = drone_state_sim.get('vy', 0.0)
                vz = drone_state_sim.get('vz', 0.0)
                self.dpc_target[0] -= vx * dt
                self.dpc_target[1] -= vy * dt
                self.dpc_target[2] -= vz * dt

        elif self.state == "STAIRCASE_DESCEND":
            # Continue tracking XY, but force Z
            if target_wp:
                self.dpc_target = [target_wp[0], target_wp[1], self.staircase_target_z]
            else:
                # Dead reckon
                dt = 0.05
                vx = drone_state_sim.get('vx', 0.0)
                vy = drone_state_sim.get('vy', 0.0)
                self.dpc_target[0] -= vx * dt
                self.dpc_target[1] -= vy * dt

            # Check completion
            if abs(current_alt - self.staircase_target_z) < 2.0:
                self.state = "STAIRCASE_STABILIZE"
                self.staircase_start_time = time.time()
                logger.info("STAIRCASE_STABILIZE. Holding position.")

        elif self.state == "STAIRCASE_STABILIZE":
            # Hold current XY and Z
            # If we see target, track it but hold Z? Or assume stabilized?
            # Instructions: "Stabilize and reacquire target"
            if target_wp:
                 self.dpc_target = [target_wp[0], target_wp[1], self.staircase_target_z]
            else:
                 # Drift
                 dt = 0.05
                 vx = drone_state_sim.get('vx', 0.0)
                 vy = drone_state_sim.get('vy', 0.0)
                 self.dpc_target[0] -= vx * dt
                 self.dpc_target[1] -= vy * dt

            # Timer
            if time.time() - self.staircase_start_time > 2.0:
                if center is not None:
                    logger.info("Staircase complete. Reacquired. Resume HOMING.")
                    self.state = "HOMING"
                else:
                    logger.info("Staircase complete. Target lost. Resume SCAN.")
                    self.state = "SCAN"

        return self.state, self.dpc_target, extra_yaw
