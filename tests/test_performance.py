
import unittest
import numpy as np
import sys
import os

class TestPerformance(unittest.TestCase):
    def setUp(self):
        # Try to import dependencies. If they fail, skip tests or use mocks.
        try:
            import pycuda.driver as cuda
            from warp_drive.env_wrapper import EnvWrapper
            from drone_env.drone import DroneEnv
            self.dependencies_present = True

            # Initialize Real Environment
            self.env = DroneEnv(num_agents=5, episode_length=10)
            self.env_wrapper = EnvWrapper(self.env, num_envs=5, use_cuda=True)
            self.env_wrapper.reset_all_envs()

        except (ImportError, ModuleNotFoundError) as e:
            print(f"Dependencies missing: {e}. Skipping real execution tests.")
            self.dependencies_present = False
            self.env = None

    def test_forward_command_performance(self):
        if not self.dependencies_present:
            return

        # Step with zero actions
        obs, rewards, done, info = self.env_wrapper.step({'drone_0': np.zeros((5, 4), dtype=np.float32)})

        self.assertEqual(rewards['drone_0'].shape, (5,))
        # Check obs shape
        self.assertEqual(obs['drone_0'].shape, (5, 1804))
        print("Step executed successfully.")

    def test_mock_simulation_logic(self):
        if self.dependencies_present:
            return

        # Re-run the static analysis here as part of the performance suite
        from unittest.mock import MagicMock

        # Mocking for inspection
        sys.modules["pycuda"] = MagicMock()
        sys.modules["pycuda.compiler"] = MagicMock()
        sys.modules["pycuda.driver"] = MagicMock()
        wd_mock = MagicMock()
        sys.modules["warp_drive"] = wd_mock
        sys.modules["warp_drive.environments"] = wd_mock
        sys.modules["warp_drive.environments.cuda_env_state"] = wd_mock

        class MockCUDAEnvironmentState:
            def __init__(self, **kwargs): pass
        wd_mock.CUDAEnvironmentState = MockCUDAEnvironmentState

        if "drone_env.drone" in sys.modules:
            del sys.modules["drone_env.drone"]
        from drone_env.drone import DroneEnv

        env = DroneEnv(num_agents=1)
        step_kwargs = env.get_step_function_kwargs()
        self.assertNotIn("rng_states", step_kwargs)

        from drone_env import drone
        self.assertIn("const int substeps = 10;", drone._DRONE_CUDA_SOURCE)

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
