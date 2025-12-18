import unittest
import numpy as np
try:
    from drone_env.drone_cython import step_cython
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

class TestPhysicsValidation(unittest.TestCase):
    def terrain_height(self, x, y):
        return 5.0 * np.sin(0.1 * x) * np.cos(0.1 * y)

    def step_naive(self,
        pos_x, pos_y, pos_z,
        vel_x, vel_y, vel_z,
        roll, pitch, yaw,
        masses, drag_coeffs, thrust_coeffs,
        target_vx, target_vy, target_vz, target_yaw_rate,
        actions,
        episode_length, t
    ):
        """
        Naive Python implementation of the drone physics step.
        """
        dt = 0.01
        g = 9.81
        substeps = 10

        # Parse actions
        thrust_cmd = actions[0::4]
        roll_rate_cmd = actions[1::4]
        pitch_rate_cmd = actions[2::4]
        yaw_rate_cmd = actions[3::4]

        for s in range(substeps):
            # Dynamics
            roll += roll_rate_cmd * dt
            pitch += pitch_rate_cmd * dt
            yaw += yaw_rate_cmd * dt

            max_thrust = 20.0 * thrust_coeffs
            thrust_force = thrust_cmd * max_thrust

            sr = np.sin(roll)
            cr = np.cos(roll)
            sp = np.sin(pitch)
            cp = np.cos(pitch)
            sy = np.sin(yaw)
            cy = np.cos(yaw)

            ax_thrust = thrust_force * (cy * sp * cr + sy * sr) / masses
            ay_thrust = thrust_force * (sy * sp * cr - cy * sr) / masses
            az_thrust = thrust_force * (cp * cr) / masses

            az_gravity = -g

            ax_drag = -drag_coeffs * vel_x
            ay_drag = -drag_coeffs * vel_y
            az_drag = -drag_coeffs * vel_z

            ax = ax_thrust + ax_drag
            ay = ay_thrust + ay_drag
            az = az_thrust + az_gravity + az_drag

            vel_x += ax * dt
            vel_y += ay * dt
            vel_z += az * dt

            pos_x += vel_x * dt
            pos_y += vel_y * dt
            pos_z += vel_z * dt

            # Terrain Collision
            terr_z = self.terrain_height(pos_x, pos_y)
            mask = pos_z < terr_z

            pos_z[mask] = terr_z[mask]
            vel_x[mask] = 0.0
            vel_y[mask] = 0.0
            vel_z[mask] = 0.0

        return pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw

    @unittest.skipUnless(HAS_CYTHON, "Cython module not found")
    def test_validation(self):
        print("Running Naive Validation...")
        num_agents = 10
        episode_length = 100

        # Init data
        pos_x = np.zeros(num_agents, dtype=np.float32)
        pos_y = np.zeros(num_agents, dtype=np.float32)
        pos_z = np.ones(num_agents, dtype=np.float32) * 10.0
        vel_x = np.zeros(num_agents, dtype=np.float32)
        vel_y = np.zeros(num_agents, dtype=np.float32)
        vel_z = np.zeros(num_agents, dtype=np.float32)
        roll = np.zeros(num_agents, dtype=np.float32)
        pitch = np.zeros(num_agents, dtype=np.float32)
        yaw = np.zeros(num_agents, dtype=np.float32)

        masses = np.ones(num_agents, dtype=np.float32)
        drag_coeffs = np.ones(num_agents, dtype=np.float32) * 0.1
        thrust_coeffs = np.ones(num_agents, dtype=np.float32)

        target_vx = np.zeros(num_agents, dtype=np.float32)
        target_vy = np.zeros(num_agents, dtype=np.float32)
        target_vz = np.zeros(num_agents, dtype=np.float32)
        target_yaw_rate = np.zeros(num_agents, dtype=np.float32)

        pos_history = np.zeros(num_agents * episode_length * 3, dtype=np.float32)
        observations = np.zeros((num_agents, 1804), dtype=np.float32)
        rewards = np.zeros(num_agents, dtype=np.float32)
        done_flags = np.zeros(num_agents, dtype=np.float32)
        step_counts = np.zeros(1, dtype=np.int32)
        actions = np.ones(num_agents * 4, dtype=np.float32) * 0.1 # Small action

        # Copies for Naive
        n_pos_x = pos_x.copy()
        n_pos_y = pos_y.copy()
        n_pos_z = pos_z.copy()
        n_vel_x = vel_x.copy()
        n_vel_y = vel_y.copy()
        n_vel_z = vel_z.copy()
        n_roll = roll.copy()
        n_pitch = pitch.copy()
        n_yaw = yaw.copy()

        # Run Naive
        n_pos_x, n_pos_y, n_pos_z, n_vel_x, n_vel_y, n_vel_z, n_roll, n_pitch, n_yaw = self.step_naive(
            n_pos_x, n_pos_y, n_pos_z, n_vel_x, n_vel_y, n_vel_z, n_roll, n_pitch, n_yaw,
            masses, drag_coeffs, thrust_coeffs,
            target_vx, target_vy, target_vz, target_yaw_rate,
            actions, episode_length, 0
        )

        # Run Optimized
        env_ids = np.array([0], dtype=np.int32)
        step_cython(
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,
            masses, drag_coeffs, thrust_coeffs, target_vx, target_vy, target_vz, target_yaw_rate,
            pos_history, observations, rewards, done_flags, step_counts, actions,
            num_agents, episode_length, env_ids
        )

        # Compare
        tol = 1e-4

        np.testing.assert_allclose(pos_x, n_pos_x, atol=tol, err_msg="Position X mismatch")
        np.testing.assert_allclose(pos_y, n_pos_y, atol=tol, err_msg="Position Y mismatch")
        np.testing.assert_allclose(pos_z, n_pos_z, atol=tol, err_msg="Position Z mismatch")
        np.testing.assert_allclose(vel_x, n_vel_x, atol=tol, err_msg="Velocity X mismatch")
        np.testing.assert_allclose(vel_y, n_vel_y, atol=tol, err_msg="Velocity Y mismatch")
        np.testing.assert_allclose(vel_z, n_vel_z, atol=tol, err_msg="Velocity Z mismatch")
        np.testing.assert_allclose(roll, n_roll, atol=tol, err_msg="Roll mismatch")
        np.testing.assert_allclose(pitch, n_pitch, atol=tol, err_msg="Pitch mismatch")
        np.testing.assert_allclose(yaw, n_yaw, atol=tol, err_msg="Yaw mismatch")

if __name__ == "__main__":
    unittest.main()
