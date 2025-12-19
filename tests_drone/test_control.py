import unittest
import numpy as np
import sys
import os

# Add parent dir to path to import control.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import control
from control import (
    PIDController, get_rotation_body_to_world, estimate_body_velocities,
    velocity_to_attitude, calc_pursuit_velocities, DroneState, PursuitState,
    FrameState, InputState, ManualMode, AngleMode, ButtonState, DualSenseConfig,
    handle_input_logic
)

class TestPIDController(unittest.TestCase):
    def test_update(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        output = pid.update(10.0)
        self.assertEqual(output, 10.0)
        output = pid.update(5.0)
        self.assertEqual(output, 5.0)

    def test_integral(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        pid.update(1.0) # integral = 1
        output = pid.update(1.0) # integral = 2
        self.assertEqual(output, 2.0)

    def test_derivative(self):
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
        pid.update(1.0) # error=1, prev=0 -> d=1
        output = pid.update(2.0) # error=2, prev=1 -> d=1
        self.assertEqual(output, 1.0)

    def test_output_limit(self):
        pid = PIDController(kp=100.0, output_limit=10.0)
        output = pid.update(1.0)
        self.assertEqual(output, 10.0)
        output = pid.update(-1.0)
        self.assertEqual(output, -10.0)

    def test_reset(self):
        pid = PIDController(ki=1.0)
        pid.update(10.0)
        pid.reset()
        self.assertEqual(pid.integral, 0.0)
        self.assertEqual(pid.previous_error, 0.0)

class TestMathHelpers(unittest.TestCase):
    def test_get_rotation_body_to_world(self):
        # Identity
        R = get_rotation_body_to_world(0, 0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

        # Yaw 90 deg (z-axis rotation)
        R_yaw = get_rotation_body_to_world(0, 0, np.pi/2)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(R_yaw, expected)

    def test_estimate_body_velocities(self):
        # Hover state
        vx, vy, vz = estimate_body_velocities(0.25, (0,0,0), 2.8, 0.25)
        # g = 9.81
        # effective_g = (0.25/0.25)*9.81 = 9.81
        # vx = 9.81 * tan(0) = 0
        # vy = 9.81 * tan(0) = 0
        # vz = 9.81 * 1 * 1 = 9.81
        self.assertAlmostEqual(vx, 0.0)
        self.assertAlmostEqual(vy, 0.0)
        self.assertAlmostEqual(vz, 9.81)

class TestLogic(unittest.TestCase):
    def test_calc_pursuit_velocities_no_bbox(self):
        ps = PursuitState()
        ds = DroneState()
        vx, vy, vz, yaw = calc_pursuit_velocities(ps, ds, None, 100, 100)
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertEqual(vz, 0.0)
        self.assertEqual(yaw, 0.0)

    def test_handle_input_logic(self):
        # Test toggle stabilization
        inp = InputState()
        btn = ButtonState()
        fs = FrameState()
        ps = PursuitState()
        cfg = DualSenseConfig()
        mode = ManualMode()

        fs.show_stabilized = False
        inp.circle = True # Pressed

        handle_input_logic(inp, btn, fs, ps, cfg, mode)

        self.assertTrue(fs.show_stabilized)
        self.assertTrue(btn.prev_o_state)

        # Release button, state should persist
        inp.circle = False
        handle_input_logic(inp, btn, fs, ps, cfg, mode)
        self.assertTrue(fs.show_stabilized)
        self.assertFalse(btn.prev_o_state)

if __name__ == '__main__':
    unittest.main()
