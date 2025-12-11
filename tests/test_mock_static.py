
import sys
from unittest.mock import MagicMock

# Mock PyCUDA
sys.modules["pycuda"] = MagicMock()
sys.modules["pycuda.compiler"] = MagicMock()
sys.modules["pycuda.driver"] = MagicMock()

# Mock WarpDrive
wd_mock = MagicMock()
sys.modules["warp_drive"] = wd_mock
sys.modules["warp_drive.environments"] = wd_mock
sys.modules["warp_drive.environments.cuda_env_state"] = wd_mock

# Create a mock for CUDAEnvironmentState that DroneEnv inherits from
class MockCUDAEnvironmentState:
    def __init__(self, **kwargs):
        pass

wd_mock.CUDAEnvironmentState = MockCUDAEnvironmentState

import unittest
import numpy as np
# Now import the env
from drone_env.drone import DroneEnv

class TestDroneEnvStatic(unittest.TestCase):
    def test_source_code_structure(self):
        """
        Verifies that the CUDA source code contains the expected logic
        for the new task reframing.
        """
        env = DroneEnv(num_agents=1)
        from drone_env import drone
        source_code = drone._DRONE_CUDA_SOURCE

        # Checks for Dynamics Randomization
        self.assertIn("float *masses", source_code)
        self.assertIn("float *thrust_coeffs", source_code)
        self.assertIn("float mass = masses[idx];", source_code)

        # Checks for Task Reframe (Commands)
        self.assertIn("float *target_vx", source_code)
        self.assertIn("float *target_yaw_rate", source_code)

        # Checks for Observation Logic (Body Frame)
        self.assertIn("float vx_b =", source_code) # Body velocity calculation
        self.assertIn("float vy_b =", source_code)

        # Checks for Reward Logic
        self.assertIn("float v_err_sq =", source_code)
        self.assertIn("reward += 1.0f * expf(-2.0f * v_err_sq);", source_code)

    def test_all_command_cases_present(self):
        """
        Verifies that the reset logic includes cases for all required commands:
        Forward, Backward, Up, Down, Rotate.
        """
        env = DroneEnv(num_agents=1)
        from drone_env import drone
        source_code = drone._DRONE_CUDA_SOURCE

        # Check for command logic keywords/comments or assignments
        # We look for the assignment blocks corresponding to the command distributions
        self.assertIn("tvx = 1.0f;", source_code) # Forward
        self.assertIn("tvx = -1.0f;", source_code) # Backward
        self.assertIn("tvz = 1.0f;", source_code) # Up
        self.assertIn("tvz = -1.0f;", source_code) # Down
        self.assertIn("tyr = 1.0f;", source_code) # Rotate Left
        self.assertIn("tyr = -1.0f;", source_code) # Rotate Right

    def test_observation_space_size(self):
        env = DroneEnv(num_agents=1)
        obs_dim = env.get_observation_space()[1]
        self.assertEqual(obs_dim, 79)

if __name__ == '__main__':
    unittest.main()
