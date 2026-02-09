import numpy as np
import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_interface import SimDroneInterface
from flight_controller import DPCFlightController
from mission_manager import MissionManager
from visual_tracker import VisualTracker
from vision.projection import Projector
from ghost_dpc.ghost_dpc import PyGhostModel, PyDPCSolver

class TestWebSimParity(unittest.TestCase):
    def test_blind_dive_parity(self):
        print("\n--- Testing Parity between Web Logic (Bypass) and Pure Sim ---")

        # 1. Run Web Logic Replica (Bypass Vision)
        # Using Default Init (1.0kg, 1.0 Thrust, 0.2 Tau)
        projector = Projector(width=640, height=480, fov_deg=110.0, tilt_deg=-45.0)
        sim = SimDroneInterface(projector)

        # Verify Defaults
        self.assertAlmostEqual(sim.masses[0], 1.0)
        self.assertAlmostEqual(sim.thrust_coeffs[0], 1.0)

        # Force Drag to 0.1 for deterministic comparison
        sim.dd['drag_coeffs'][0] = 0.1

        controller = DPCFlightController(dt=0.05)
        # Verify Controller Defaults
        self.assertEqual(controller.models_config[0]['mass'], 1.0)
        self.assertEqual(controller.models_config[0]['thrust_coeff'], 1.0)
        self.assertEqual(controller.models_config[0]['drag_coeff'], 0.1)
        self.assertEqual(controller.models_config[0]['tau'], 0.2)

        target_pos_sim_world = [50.0, 0.0, 0.0]
        drone_pos = [0.0, 0.0, 100.0]

        # Orientation Logic
        dx = target_pos_sim_world[0] - drone_pos[0]
        dy = target_pos_sim_world[1] - drone_pos[1]
        dz = target_pos_sim_world[2] - drone_pos[2]
        yaw = np.arctan2(dy, dx)
        dist_xy = np.sqrt(dx*dx + dy*dy)
        pitch_vec = np.arctan2(dz, dist_xy)
        camera_tilt = np.deg2rad(-45.0)
        pitch = pitch_vec - camera_tilt

        sim.reset_to_scenario("Blind Dive", pos_x=drone_pos[0], pos_y=drone_pos[1], pos_z=drone_pos[2], pitch=pitch, yaw=yaw)

        path_web = []
        for i in range(100):
            s = sim.get_state()
            path_web.append([s['px'], s['py'], s['pz']])

            # Bypass Vision Logic (Perfect Relative Target)
            rel_x = target_pos_sim_world[0] - s['px']
            rel_y = target_pos_sim_world[1] - s['py']
            # Target NED Z is Down. Target Z (Abs) - Drone Z (Abs) = dz_up. dz_ned = -dz_up.
            rel_z_ned = -(target_pos_sim_world[2] - s['pz'])
            target_wp = [rel_x, rel_y, rel_z_ned]

            dpc_target = [rel_x, rel_y, target_pos_sim_world[2]]

            state_obs = {
                'pz': s['pz'], 'vz': s['vz'],
                'roll': s['roll'], 'pitch': s['pitch'], 'yaw': s['yaw'],
                'wx': s['wx'], 'wy': s['wy'], 'wz': s['wz']
            }

            action_out, _ = controller.compute_action(
                state_obs, dpc_target, raw_target_rel_ned=target_wp, extra_yaw_rate=0.0
            )

            sim_action = np.array([
                action_out['thrust'], action_out['roll_rate'], action_out['pitch_rate'], action_out['yaw_rate']
            ])

            sim.step(sim_action)

        path_web = np.array(path_web)

        # 2. Run Pure Sim Logic
        # Matches Web Logic Params (1.0kg, 0.1 Drag, 1.0 Thrust)
        # Note: PyGhostModel uses tau=0.1 by default in Sim loop,
        # but Controller uses tau=0.2 in its model.
        # This tests if Web Logic (DroneEnv + Controller) matches Pure Sim (PyGhostModel + Controller).
        # Assuming PyGhostModel Plant matches DroneEnv (substeps=1).

        model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0, tau=0.1) # Plant Tau 0.1
        solver = PyDPCSolver()

        state = {
            'px': drone_pos[0], 'py': drone_pos[1], 'pz': drone_pos[2],
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'roll': 0.0, 'pitch': pitch, 'yaw': yaw,
            'wx': 0.0, 'wy': 0.0, 'wz': 0.0
        }
        action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

        dt = 0.05
        path_sim = []

        # Config for Solver (Use Default 0.1 Tau for Ideal Sim)
        # We compare Web Logic (Tuned Tau=0.2) against Pure Sim (Ideal Tau=0.1)
        # Because we want Web to match the Ideal Sim behavior.
        models_config = [{'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.0, 'tau': 0.1}]
        weights = [1.0]

        for i in range(100):
            path_sim.append([state['px'], state['py'], state['pz']])
            action = solver.solve(state, target_pos_sim_world, action, models_config, weights, dt)
            state = model.step(state, action, dt)

        path_sim = np.array(path_sim)

        # 3. Compare
        print(f"Web Final: {path_web[-1]}")
        print(f"Sim Final: {path_sim[-1]}")
        final_err = np.linalg.norm(path_web[-1] - path_sim[-1])
        print(f"Final Error: {final_err:.4f} m")

        # Assert low error
        self.assertLess(final_err, 1.0, "Web Logic differs significantly from Sim Logic!")

if __name__ == "__main__":
    unittest.main()
