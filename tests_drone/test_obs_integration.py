import unittest
import numpy as np
from drone_env.drone import DroneEnv

class TestObsIntegration(unittest.TestCase):
    def test_obs_shape(self):
        env = DroneEnv(num_agents=2, episode_length=100, use_cuda=False)
        obs_shape = env.get_observation_space()
        self.assertEqual(obs_shape, (2, 308))

        data = env.get_data_dictionary()
        self.assertEqual(data["observations"]["shape"], (2, 308))
        self.assertEqual(data["vt_x"]["shape"], (2,))

    def test_step_updates_obs(self):
        env = DroneEnv(num_agents=1, episode_length=100, use_cuda=False)
        # Manually init state
        data = {name: np.zeros(spec["shape"], dtype=spec["dtype"])
                for name, spec in env.get_data_dictionary().items()}

        # Reset
        kwargs = {k: data[v] for k, v in env.get_reset_function_kwargs().items() if v in data}
        kwargs["reset_indices"] = np.array([0], dtype=np.int32)
        kwargs["num_agents"] = env.num_agents

        env.reset_function(**kwargs)

        obs = data["observations"][0]
        self.assertEqual(obs.shape, (308,))

        # Run one step
        actions = np.zeros((1, 4), dtype=np.float32)

        step_kwargs = {k: data[v] for k, v in env.get_step_function_kwargs().items() if v in data}
        step_kwargs["actions"] = actions.flatten()
        step_kwargs["num_agents"] = env.num_agents
        step_kwargs["episode_length"] = env.episode_length
        step_kwargs["env_ids"] = np.array([0], dtype=np.int32)

        env.step_function(**step_kwargs)

        # Check if virtual target moved
        t = data["step_counts"][0]
        self.assertEqual(t, 1)

        vt_x = data["vt_x"][0]
        vt_y = data["vt_y"][0]
        vt_z = data["vt_z"][0]

        # Calculate expected based on randomized params
        # traj_params are now randomized in reset.
        # Shape: (10, num_agents)
        tp = data["traj_params"] # Shape (10, 1)
        # 0:Ax, 1:Fx, 2:Px
        # access via [idx, agent_idx]
        expected_x = tp[0, 0] * np.sin(tp[1, 0] * 1.0 + tp[2, 0])
        self.assertAlmostEqual(vt_x, expected_x, places=5)

        # Check tracker features in obs [304:308]
        u = obs[304]
        v = obs[305]
        size = obs[306]
        conf = obs[307]

        print(f"Step 1: VT=({vt_x:.2f}, {vt_y:.2f}, {vt_z:.2f}) Obs=({u:.2f}, {v:.2f}, {size:.2f}, {conf:.2f})")
        self.assertNotEqual(size, 0.0)

        # Test large angular rate -> low confidence
        # NOTE: With delay logic, the action applied at Step 2 is determined by `delays`.
        # If `delays[0] > 0`, the high roll rate won't be applied immediately.
        # We need to force `delays` to 0 for this test.
        data["delays"][0] = 0

        actions[0, 1] = 10.0 # High roll rate
        step_kwargs["actions"] = actions.flatten()

        # Also need to clear action_buffer or ensure it's not overriding if delay=0?
        # If delay=0, effective action is the current one.
        # But wait, logic is:
        # Shift buffer. Insert new at 0. Read from buffer[delay].
        # So yes, if delay=0, we read from index 0 which is new action.

        env.step_function(**step_kwargs)
        obs = data["observations"][0]
        conf_high_motion = obs[307]

        print(f"Step 2 (High Motion): Conf={conf_high_motion:.4f}")
        self.assertLess(conf_high_motion, 0.1)

if __name__ == "__main__":
    unittest.main()
