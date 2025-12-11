
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

        # Check Substeps
        self.assertIn("const int substeps = 10;", source_code)

        # Check History Logic
        self.assertIn("float *imu_history", source_code)
        self.assertIn("imu_history[hist_start + i] = imu_history[hist_start + i + 60];", source_code)

    def test_observation_space_size(self):
        env = DroneEnv(num_agents=1)
        obs_dim = env.get_observation_space()[1]
        # New size is 1804
        self.assertEqual(obs_dim, 1804)

if __name__ == '__main__':
    unittest.main()
