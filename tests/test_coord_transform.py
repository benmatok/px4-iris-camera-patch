
import math
import unittest
import numpy as np

class TestCoordinateTransforms(unittest.TestCase):
    def sim_to_ned(self, sim_state):
        # Maps Sim (ENU) to NED (Right-Handed)
        ned = sim_state.copy()
        ned['px'] = sim_state['py']
        ned['py'] = sim_state['px']
        ned['pz'] = -sim_state['pz']

        ned['vx'] = sim_state['vy']
        ned['vy'] = sim_state['vx']
        ned['vz'] = -sim_state['vz']

        ned['roll'] = sim_state['roll']
        ned['pitch'] = sim_state['pitch']
        ned['yaw'] = (math.pi / 2.0) - sim_state['yaw']

        # Normalize Yaw
        ned['yaw'] = (ned['yaw'] + math.pi) % (2 * math.pi) - math.pi

        return ned

    def sim_pos_to_ned_pos(self, sim_pos):
        # [x, y, z] -> [y, x, -z]
        return [sim_pos[1], sim_pos[0], -sim_pos[2]]

    def ned_rel_to_sim_rel(self, ned_rel):
        # [dx, dy, dz] NED -> [dy, dx, -dz] Sim
        return [ned_rel[1], ned_rel[0], -ned_rel[2]]

    def test_sim_to_ned_yaw_east(self):
        # Sim Yaw 0 (East)
        sim_state = {
            'px': 10, 'py': 20, 'pz': 30,
            'vx': 1, 'vy': 2, 'vz': 3,
            'roll': 0.1, 'pitch': 0.2, 'yaw': 0.0
        }
        ned = self.sim_to_ned(sim_state)

        # Check Pos
        self.assertEqual(ned['px'], 20) # North (Sim Y)
        self.assertEqual(ned['py'], 10) # East (Sim X)
        self.assertEqual(ned['pz'], -30) # Down

        # Check Vel
        self.assertEqual(ned['vx'], 2)
        self.assertEqual(ned['vy'], 1)
        self.assertEqual(ned['vz'], -3)

        # Check Yaw
        # Sim 0 (East) -> NED pi/2 (East)
        expected_yaw = math.pi / 2.0
        self.assertAlmostEqual(ned['yaw'], expected_yaw)

    def test_sim_to_ned_yaw_north(self):
        # Sim Yaw pi/2 (North)
        sim_state = {
            'px': 0, 'py': 0, 'pz': 0,
            'vx': 0, 'vy': 0, 'vz': 0,
            'roll': 0, 'pitch': 0, 'yaw': math.pi / 2.0
        }
        ned = self.sim_to_ned(sim_state)

        # Sim pi/2 (North) -> NED 0 (North)
        # pi/2 - pi/2 = 0
        self.assertAlmostEqual(ned['yaw'], 0.0)

    def test_sim_to_ned_yaw_south(self):
        # Sim Yaw 3pi/2 (South) (-pi/2)
        sim_state = {
            'px': 0, 'py': 0, 'pz': 0,
            'vx': 0, 'vy': 0, 'vz': 0,
            'roll': 0, 'pitch': 0, 'yaw': 3 * math.pi / 2.0
        }
        ned = self.sim_to_ned(sim_state)

        # Sim 3pi/2 -> NED pi/2 - 3pi/2 = -pi (South)
        self.assertAlmostEqual(abs(ned['yaw']), math.pi)

    def test_sim_pos_to_ned_pos(self):
        sim_pos = [10, 20, 30]
        ned_pos = self.sim_pos_to_ned_pos(sim_pos)
        self.assertEqual(ned_pos, [20, 10, -30])

    def test_ned_rel_to_sim_rel(self):
        ned_rel = [20, 10, -30] # North=20, East=10, Down=-30 (Up=30)
        sim_rel = self.ned_rel_to_sim_rel(ned_rel)
        # Sim: East=10, North=20, Up=30
        self.assertEqual(sim_rel, [10, 20, 30])

if __name__ == '__main__':
    unittest.main()
