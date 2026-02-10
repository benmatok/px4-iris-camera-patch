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
        print("\n--- Testing Binary Exact Parity between Web Logic (SimDroneInterface) and Pure Sim ---")

        # 1. Run Web Logic Replica (Bypass Vision)
        projector = Projector(width=640, height=480, fov_deg=110.0, tilt_deg=-45.0)
        sim = SimDroneInterface(projector)

        # Verify Model Init
        self.assertEqual(sim.mass, 1.0)
        self.assertEqual(sim.thrust_coeff, 1.0)
        self.assertEqual(sim.tau, 0.1)
        self.assertEqual(sim.drag_coeff, 0.1)

        controller = DPCFlightController(dt=0.05)
        # Verify Controller Defaults
        self.assertEqual(controller.models_config[0]['mass'], 1.0)
        self.assertEqual(controller.models_config[0]['thrust_coeff'], 1.0)
        self.assertEqual(controller.models_config[0]['drag_coeff'], 0.1)
        self.assertEqual(controller.models_config[0]['tau'], 0.1)

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

        # Explicitly set initial state to match "Pure Sim" perfectly
        # PyGhostModel uses 0.0 for initial velocities/rates unless specified
        # SimDroneInterface.reset_to_scenario sets them to 0.0
        # Check initial state
        # Check initial state
        s0 = sim.get_state()

        path_web = []
        for i in range(100):
            s = sim.get_state()
            path_web.append([s['px'], s['py'], s['pz']])

            # Bypass Vision Logic (Perfect Relative Target)
            # NED: X-North, Y-East, Z-Down
            # Sim: X-East, Y-North, Z-Up

            # Rel Sim X (East)
            rel_sim_x = target_pos_sim_world[0] - s['px']
            # Rel Sim Y (North)
            rel_sim_y = target_pos_sim_world[1] - s['py']
            # Rel Sim Z (Up)
            rel_sim_z = target_pos_sim_world[2] - s['pz']

            # Map to NED
            rel_ned_x = rel_sim_y # North
            rel_ned_y = rel_sim_x # East
            rel_ned_z = -rel_sim_z # Down

            target_wp = [rel_ned_x, rel_ned_y, rel_ned_z]

            dpc_target = [rel_sim_x, rel_sim_y, target_pos_sim_world[2]]

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

        # 2. Run Pure Sim Logic (using DPCFlightController)
        # Matches Web Logic Params (1.0kg, 0.1 Drag, 1.0 Thrust, 0.1 Tau)
        model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0, tau=0.1)
        sim_controller = DPCFlightController(dt=0.05)

        state = {
            'px': drone_pos[0], 'py': drone_pos[1], 'pz': drone_pos[2],
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'roll': 0.0, 'pitch': pitch, 'yaw': yaw,
            'wx': 0.0, 'wy': 0.0, 'wz': 0.0
        }

        dt = 0.05
        path_sim = []

        for i in range(100):
            path_sim.append([state['px'], state['py'], state['pz']])

            # Replicate Inputs
            # Rel Sim X (East)
            rel_sim_x = target_pos_sim_world[0] - state['px']
            # Rel Sim Y (North)
            rel_sim_y = target_pos_sim_world[1] - state['py']
            # Rel Sim Z (Up)
            rel_sim_z = target_pos_sim_world[2] - state['pz']

            # Map to NED
            rel_ned_x = rel_sim_y # North
            rel_ned_y = rel_sim_x # East
            rel_ned_z = -rel_sim_z # Down

            target_wp = [rel_ned_x, rel_ned_y, rel_ned_z]

            dpc_target = [rel_sim_x, rel_sim_y, target_pos_sim_world[2]]

            state_obs = {
                'pz': state['pz'], 'vz': state['vz'],
                'roll': state['roll'], 'pitch': state['pitch'], 'yaw': state['yaw'],
                'wx': state['wx'], 'wy': state['wy'], 'wz': state['wz']
            }

            action_out, _ = sim_controller.compute_action(
                state_obs, dpc_target, raw_target_rel_ned=target_wp, extra_yaw_rate=0.0
            )

            state = model.step(state, action_out, dt)

        path_sim = np.array(path_sim)

        # 3. Compare
        print(f"Web Final: {path_web[-1]}")
        print(f"Sim Final: {path_sim[-1]}")
        final_err = np.linalg.norm(path_web[-1] - path_sim[-1])
        print(f"Final Error: {final_err:.8f} m")

        # Assert Binary Exactness (Floating Point Tolerance)
        # Should be extremely close to 0.0
        self.assertLess(final_err, 1e-4, "Web Logic is not binary exact with Sim Logic!")

if __name__ == "__main__":
    unittest.main()
