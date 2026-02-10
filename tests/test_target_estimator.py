import unittest
import numpy as np
from ghost_dpc.target_estimator import TargetEstimator
from mission_manager import MissionManager

class TestTargetEstimator(unittest.TestCase):
    def test_predict_update(self):
        est = TargetEstimator(dt=0.1)

        # Test Initial State
        self.assertTrue(est.is_lost)
        self.assertTrue(np.allclose(est.get_estimate(), [0, 0, 0]))

        # Test Update
        meas = [10.0, 5.0, -2.0]
        est.update(meas)
        self.assertFalse(est.is_lost)
        # Should be close to measurement (Kalman Gain might dampen it slightly, but initial cov is high)
        # P0=10, R=0.5 -> K ~ 10/10.5 ~ 0.95. So very close.
        est_pos = est.get_estimate()
        self.assertTrue(np.allclose(est_pos, meas, atol=1.0))

        # Test Predict (Dead Reckoning)
        # Drone moving at [1, 0, 0]
        # Target static relative to world -> Relative pos should decrease by [0.1, 0, 0]
        drone_vel = [1.0, 0.0, 0.0]
        est.predict(drone_vel)

        est_pos_next = est.get_estimate()
        expected = est_pos - np.array([0.1, 0.0, 0.0])
        self.assertTrue(np.allclose(est_pos_next, expected))

        # Test Time tracking
        self.assertAlmostEqual(est.time_since_last_seen, 0.1)

        # Test Lost Logic
        for _ in range(10): # 1.0s
            est.predict([0,0,0])

        self.assertTrue(est.time_since_last_seen > 0.5)
        self.assertTrue(est.is_lost)

class TestMissionManager(unittest.TestCase):
    def test_state_transitions(self):
        mm = MissionManager(target_alt=10.0)

        # 1. TAKEOFF -> SCAN
        # Drone at 0m. Target 10m.
        state, target, yaw = mm.update({'px':0, 'py':0, 'pz':0, 'vx':0, 'vy':0, 'vz':0}, (None, None))
        self.assertEqual(state, "TAKEOFF")

        # Drone at 9m (close to 10m)
        # Transition happens here
        state, target, yaw = mm.update({'px':0, 'py':0, 'pz':9.0, 'vx':0, 'vy':0, 'vz':0}, (None, None))
        self.assertEqual(state, "SCAN")
        # Yaw is 0 because transition logic happens in first block, SCAN logic in second block (skipped)
        self.assertEqual(yaw, 0.0)

        # Next frame, SCAN logic runs
        state, target, yaw = mm.update({'px':0, 'py':0, 'pz':9.0, 'vx':0, 'vy':0, 'vz':0}, (None, None))
        self.assertEqual(state, "SCAN")
        self.assertNotEqual(yaw, 0.0) # Now spinning

        # 2. SCAN -> HOMING
        # Detect target
        center = (320, 240)
        target_wp = [5.0, 5.0, -9.0] # Relative Pos
        state, target, yaw = mm.update({'px':0, 'py':0, 'pz':9.0, 'vx':0, 'vy':0, 'vz':0}, (center, target_wp))
        self.assertEqual(state, "HOMING")
        self.assertEqual(yaw, 0.0)

        # Target logic runs in the NEXT frame for HOMING (state machine delay)
        # So in this frame, target is still [px, py, target_alt] from SCAN block

        # Run one more frame to see HOMING target logic
        state, target, yaw = mm.update({'px':0, 'py':0, 'pz':9.0, 'vx':0, 'vy':0, 'vz':0}, (center, target_wp))
        self.assertEqual(state, "HOMING")

        # Target Alt Check: Current(9) + Rel(-9) + 2 = 2.0.
        # dpc_target should be [5.0, 5.0, 2.0]
        # Kalman filter will be very close to measurement after 2 updates.
        self.assertTrue(np.allclose(target, [5.0, 5.0, 2.0], atol=1.0))

        # 3. HOMING -> LOST_RECOVERY
        # Lose target for > 2s
        # Simulate 2.1s (dt=0.05 -> 42 steps)

        for _ in range(50):
            state, target, yaw = mm.update({'px':0, 'py':0, 'pz':9.0, 'vx':0, 'vy':0, 'vz':0}, (None, None))

        self.assertEqual(state, "LOST_RECOVERY")

        # 4. LOST_RECOVERY Behavior
        # Should climb and rotate
        # First frame of LOST_RECOVERY sets state, logic runs NEXT frame?
        # In HOMING block, we set state to LOST_RECOVERY.
        # Next frame LOST_RECOVERY block runs.

        state, target, yaw = mm.update({'px':0, 'py':0, 'pz':9.0, 'vx':0, 'vy':0, 'vz':0}, (None, None))
        self.assertEqual(state, "LOST_RECOVERY")
        self.assertNotEqual(yaw, 0.0)
        # Target Alt should be max(9+5, 20) = 20.0 (Recovery Floor)
        self.assertEqual(target[2], 20.0)

        # If we are at 20m.
        state, target, yaw = mm.update({'px':0, 'py':0, 'pz':20.0, 'vx':0, 'vy':0, 'vz':0}, (None, None))
        # Target Alt = max(20+5, 20) = 25.0
        self.assertEqual(target[2], 25.0)

        # 5. LOST_RECOVERY -> HOMING
        state, target, yaw = mm.update({'px':0, 'py':0, 'pz':20.0, 'vx':0, 'vy':0, 'vz':0}, (center, target_wp))
        self.assertEqual(state, "HOMING")
        self.assertEqual(yaw, 0.0)

if __name__ == '__main__':
    unittest.main()
