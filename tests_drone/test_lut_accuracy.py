import unittest
import numpy as np
from drone_env.drone import step_cpu
try:
    from drone_env.drone_cython import step_cython
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

class TestLUTAccuracy(unittest.TestCase):
    def setUp(self):
        if not HAS_CYTHON:
            self.skipTest("Cython extension not available")

        self.num_agents = 8  # Multiple of 8 to trigger AVX path in Cython
        self.episode_length = 100

        # Initialize shared data
        self.pos_x = np.zeros(self.num_agents, dtype=np.float32)
        self.pos_y = np.zeros(self.num_agents, dtype=np.float32)
        self.pos_z = np.zeros(self.num_agents, dtype=np.float32)
        self.vel_x = np.zeros(self.num_agents, dtype=np.float32)
        self.vel_y = np.zeros(self.num_agents, dtype=np.float32)
        self.vel_z = np.zeros(self.num_agents, dtype=np.float32)
        self.roll = np.zeros(self.num_agents, dtype=np.float32)
        self.pitch = np.zeros(self.num_agents, dtype=np.float32)
        self.yaw = np.zeros(self.num_agents, dtype=np.float32)

        self.masses = np.ones(self.num_agents, dtype=np.float32)
        self.drag_coeffs = np.ones(self.num_agents, dtype=np.float32) * 0.1
        self.thrust_coeffs = np.ones(self.num_agents, dtype=np.float32)

        self.target_vx = np.zeros(self.num_agents, dtype=np.float32)
        self.target_vy = np.zeros(self.num_agents, dtype=np.float32)
        self.target_vz = np.zeros(self.num_agents, dtype=np.float32)
        self.target_yaw_rate = np.zeros(self.num_agents, dtype=np.float32)

        self.vt_x = np.zeros(self.num_agents, dtype=np.float32)
        self.vt_y = np.zeros(self.num_agents, dtype=np.float32)
        self.vt_z = np.zeros(self.num_agents, dtype=np.float32)

        self.traj_params = np.zeros((10, self.num_agents), dtype=np.float32)
        # Set some non-trivial trajectory parameters
        self.traj_params[0, :] = 3.0 # Ax
        self.traj_params[1, :] = 0.1 # Fx
        self.traj_params[3, :] = 3.0 # Ay
        self.traj_params[4, :] = 0.1 # Fy

        self.pos_history = np.zeros((self.episode_length, self.num_agents, 3), dtype=np.float32)
        self.observations = np.zeros((self.num_agents, 608), dtype=np.float32)
        self.rewards = np.zeros(self.num_agents, dtype=np.float32)
        self.reward_components = np.zeros((self.num_agents, 8), dtype=np.float32)
        self.done_flags = np.zeros(self.num_agents, dtype=np.float32)
        self.step_counts = np.zeros(1, dtype=np.int32)
        self.actions = np.zeros(self.num_agents * 4, dtype=np.float32)
        self.env_ids = np.zeros(self.num_agents, dtype=np.int32)

        # Set initial positions to exercise terrain LUT
        # Terrain is 5 * sin(0.1*x) * cos(0.1*y)
        self.pos_x[:] = np.linspace(0, 20, self.num_agents, dtype=np.float32)
        self.pos_y[:] = np.linspace(0, 20, self.num_agents, dtype=np.float32)
        self.pos_z[:] = 10.0 # Above terrain

        # Actions: Zero rotation to prevent tumbling and diverging state
        # Thrust 0.5 (approx hover/climb depending on gravity/mass)
        # Mass=1.0, G=9.8. MaxThrust=20.0. 0.5*20 = 10.0 > 9.8. Slow climb.
        self.actions[:] = 0.0
        self.actions[0::4] = 0.5 # Set thrust to 0.5 for all agents

    def test_lut_vs_numpy_accuracy(self):
        # Deep copy state for numpy and cython to ensure they start identical
        args_cpu = {
            'pos_x': self.pos_x.copy(), 'pos_y': self.pos_y.copy(), 'pos_z': self.pos_z.copy(),
            'vel_x': self.vel_x.copy(), 'vel_y': self.vel_y.copy(), 'vel_z': self.vel_z.copy(),
            'roll': self.roll.copy(), 'pitch': self.pitch.copy(), 'yaw': self.yaw.copy(),
            'masses': self.masses.copy(), 'drag_coeffs': self.drag_coeffs.copy(), 'thrust_coeffs': self.thrust_coeffs.copy(),
            'target_vx': self.target_vx.copy(), 'target_vy': self.target_vy.copy(), 'target_vz': self.target_vz.copy(),
            'target_yaw_rate': self.target_yaw_rate.copy(),
            'vt_x': self.vt_x.copy(), 'vt_y': self.vt_y.copy(), 'vt_z': self.vt_z.copy(),
            'traj_params': self.traj_params.copy(),
            'pos_history': self.pos_history.copy(),
            'observations': self.observations.copy(),
            'rewards': self.rewards.copy(),
            'reward_components': self.reward_components.copy(),
            'done_flags': self.done_flags.copy(),
            'step_counts': self.step_counts.copy(),
            'actions': self.actions.copy(),
            'num_agents': self.num_agents,
            'episode_length': self.episode_length,
            'env_ids': self.env_ids.copy()
        }

        args_cy = {
            'pos_x': self.pos_x.copy(), 'pos_y': self.pos_y.copy(), 'pos_z': self.pos_z.copy(),
            'vel_x': self.vel_x.copy(), 'vel_y': self.vel_y.copy(), 'vel_z': self.vel_z.copy(),
            'roll': self.roll.copy(), 'pitch': self.pitch.copy(), 'yaw': self.yaw.copy(),
            'masses': self.masses.copy(), 'drag_coeffs': self.drag_coeffs.copy(), 'thrust_coeffs': self.thrust_coeffs.copy(),
            'target_vx': self.target_vx.copy(), 'target_vy': self.target_vy.copy(), 'target_vz': self.target_vz.copy(),
            'target_yaw_rate': self.target_yaw_rate.copy(),
            'vt_x': self.vt_x.copy(), 'vt_y': self.vt_y.copy(), 'vt_z': self.vt_z.copy(),
            'traj_params': self.traj_params.copy(),
            'pos_history': self.pos_history.copy(),
            'observations': self.observations.copy(),
            'rewards': self.rewards.copy(),
            'reward_components': self.reward_components.copy(),
            'done_flags': self.done_flags.copy(),
            'step_counts': self.step_counts.copy(),
            'actions': self.actions.copy(),
            'num_agents': self.num_agents,
            'episode_length': self.episode_length,
            'env_ids': self.env_ids.copy()
        }

        # Run 100 steps
        for _ in range(100):
            step_cpu(**args_cpu)
            step_cython(**args_cy)

        # Verify Positions (Terrain interaction check)
        np.testing.assert_allclose(args_cpu['pos_x'], args_cy['pos_x'], rtol=1e-2, atol=1e-2, err_msg="PosX mismatch")
        np.testing.assert_allclose(args_cpu['pos_y'], args_cy['pos_y'], rtol=1e-2, atol=1e-2, err_msg="PosY mismatch")
        np.testing.assert_allclose(args_cpu['pos_z'], args_cy['pos_z'], rtol=1e-2, atol=1e-2, err_msg="PosZ mismatch")

        # Verify Rewards (Check rcp approximation accuracy)
        # Note: We use slightly loose tolerance due to rcp approximations in AVX
        np.testing.assert_allclose(args_cpu['rewards'], args_cy['rewards'], rtol=1e-1, atol=1e-1, err_msg="Reward mismatch")

if __name__ == '__main__':
    unittest.main()
