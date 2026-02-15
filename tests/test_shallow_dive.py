
import unittest
import numpy as np
import sys
import os
import math
import logging

# Add root to path
sys.path.append(os.getcwd())

from sim_interface import SimDroneInterface
from flight_controller import DPCFlightController
from vision.projection import Projector

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestShallowDive")

class TestShallowDive(unittest.TestCase):
    def test_shallow_dive_performance(self):
        # 1. Setup
        # Target at [50, 0, 0] Sim Frame
        target_pos = [50.0, 0.0, 0.0]

        # Start at Alt=50, Dist=150
        # Drone Pos: [-100, 0, 50]
        start_pos = [-100.0, 0.0, 50.0]

        # Initialize Components
        projector = Projector(width=640, height=480, fov_deg=120.0, tilt_deg=30.0)
        sim = SimDroneInterface(projector)

        # Calculate initial orientation (Face target)
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]
        dz = target_pos[2] - start_pos[2] # -50

        yaw = np.arctan2(dy, dx) # 0
        dist_xy = np.sqrt(dx*dx + dy*dy) # 150
        pitch_vec = np.arctan2(dz, dist_xy) # atan(-50/150) = -18.4 deg

        camera_tilt = np.deg2rad(30.0)
        initial_pitch = pitch_vec - camera_tilt

        sim.reset_to_scenario("Blind Dive",
                              pos_x=start_pos[0],
                              pos_y=start_pos[1],
                              pos_z=start_pos[2],
                              pitch=initial_pitch,
                              yaw=yaw)

        controller = DPCFlightController(dt=0.05, mode='PID')

        # Simulation Loop
        max_steps = 1000 # 50 seconds
        crashed = False
        success = False
        min_dist = 9999.0

        logger.info(f"Starting Simulation: Pos={start_pos}, Target={target_pos}")

        for step in range(max_steps):
            s = sim.get_state()

            # Check Termination
            curr_pos = np.array([s['px'], s['py'], s['pz']])
            t_pos = np.array(target_pos)
            dist = np.linalg.norm(curr_pos - t_pos)
            min_dist = min(min_dist, dist)

            if s['pz'] <= 0.1:
                # If we crash, check distance
                if dist < 5.0:
                    logger.info(f"Landed/Crashed Close to Target at Step {step}, Dist={dist:.2f}")
                    success = True
                else:
                    logger.error(f"CRASHED at Step {step}, Dist={dist:.2f}, Pos={curr_pos}")
                    crashed = True
                break

            if dist < 2.0:
                logger.info(f"SUCCESS at Step {step}, Dist={dist:.2f}")
                success = True
                break

            # Perception (Assume Perfect Tracking)
            rel_x = target_pos[0] - s['px']
            rel_y = target_pos[1] - s['py']
            target_cmd = [rel_x, rel_y, 0.0]

            # State Obs
            state_obs = {
                'pz': s['pz'],
                'vz': s['vz'],
                'roll': s['roll'],
                'pitch': s['pitch'],
                'yaw': s['yaw'],
                'wx': s['wx'],
                'wy': s['wy'],
                'wz': s['wz']
            }

            # Perfect Tracking UV (Center)
            # We assume centered because Yaw=0
            tracking_norm = (0.0, 0.0)

            # Compute Action
            action_out, _ = controller.compute_action(
                state_obs,
                target_cmd,
                tracking_uv=tracking_norm,
                extra_yaw_rate=0.0
            )

            # Apply Action
            sim_action = np.array([
                action_out['thrust'],
                action_out['roll_rate'],
                action_out['pitch_rate'],
                action_out['yaw_rate']
            ])

            sim.step(sim_action)

        self.assertTrue(success, f"Test Failed: Crashed={crashed}, Min Dist={min_dist:.2f}")
        logger.info(f"Test Passed with Min Dist: {min_dist:.2f}")

if __name__ == "__main__":
    unittest.main()
