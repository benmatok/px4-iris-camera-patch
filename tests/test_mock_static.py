
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

        # Checks for Reward Logic (calculating errors)
        self.assertIn("float v_err_sq =", source_code)

        # Check History Logic
        self.assertIn("float *imu_history", source_code)
        # Check shifting loop
        self.assertIn("imu_history[hist_start + i] = imu_history[hist_start + i + 6];", source_code)

    def test_observation_space_size(self):
        env = DroneEnv(num_agents=1)
        obs_dim = env.get_observation_space()[1]
        # New size is 184
        self.assertEqual(obs_dim, 184)

if __name__ == '__main__':
    unittest.main()
