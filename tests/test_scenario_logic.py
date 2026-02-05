import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add root to path
sys.path.append(os.getcwd())

# Mock DroneEnv modules to avoid import errors
sys.modules['drone_env'] = MagicMock()
sys.modules['drone_env.drone'] = MagicMock()

# Now import
from sim_interface import SimDroneInterface
from vision.projection import Projector
from visual_tracker import VisualTracker

class TestScenarioLogic(unittest.TestCase):
    def test_reset_scenario(self):
        proj = MagicMock()

        # Mock Data Dictionary
        dd = {
            'pos_x': [0.0], 'pos_y': [0.0], 'pos_z': [0.0],
            'vel_x': [0.0], 'vel_y': [0.0], 'vel_z': [0.0],
            'roll': [0.0], 'pitch': [0.0], 'yaw': [0.0],
            'masses': [1.0], 'thrust_coeffs': [1.0],
            'actions': np.zeros(4),
            'done_flags': [0.0]
        }

        mock_env = MagicMock()
        mock_env.data_dictionary = dd

        # Patch where it is imported in sim_interface
        with patch('sim_interface.DroneEnv', return_value=mock_env):
            sim = SimDroneInterface(proj)
            sim.reset_to_scenario("Blind Dive")

            s = sim.get_state()
            self.assertEqual(s['pz'], 100.0)
            self.assertEqual(s['px'], 0.0)

    def test_projector_size(self):
        proj = Projector(width=640, height=480, fov_deg=90.0) # fov 90 -> fx = 320

        drone_state = {'px':0,'py':0,'pz':0, 'roll':0,'pitch':0,'yaw':0}

        # Test point 10m in front
        # Body X=10 -> Cam Z=10.
        u, v, r = proj.project_point_with_size(10.0, 0.0, 0.0, drone_state, object_radius=1.0)

        self.assertAlmostEqual(u, 320.0, delta=1.0)
        self.assertAlmostEqual(v, 240.0, delta=1.0)
        self.assertAlmostEqual(r, 32.0, delta=1.0)

    def test_tracker_radius(self):
        # Mock detector output
        class MockDetector:
            def detect(self, img):
                return (320, 240), 314.159, (300, 220, 40, 40)

        proj = MagicMock()
        tracker = VisualTracker(proj)
        tracker.detector = MockDetector()

        c, wp, r = tracker.process(np.zeros((10,10,3)), {})

        self.assertAlmostEqual(r, 10.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
