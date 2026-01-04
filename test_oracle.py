
import numpy as np
from train_jules import OracleController
from drone_env.drone import DroneEnv

def test_oracle_physics():
    # Setup Env with 1 agent
    env = DroneEnv(num_agents=1, episode_length=50)
    env.reset_all_envs()

    # Force heavy mass and high drag
    mass = 2.0
    drag = 0.5 # High drag
    thrust_coeff = 1.0

    env.data_dictionary['masses'][0] = mass
    env.data_dictionary['drag_coeffs'][0] = drag
    env.data_dictionary['thrust_coeffs'][0] = thrust_coeff

    # Set Start State = (0,0,10), Vel=0
    env.data_dictionary['pos_x'][0] = 0.0
    env.data_dictionary['pos_y'][0] = 0.0
    env.data_dictionary['pos_z'][0] = 10.0
    env.data_dictionary['vel_x'][0] = 0.0
    env.data_dictionary['vel_y'][0] = 0.0
    env.data_dictionary['vel_z'][0] = 0.0

    # Set Trajectory to stay at (0,0,10)
    # We can't easily force the trajectory params to be constant pos without reverse engineering lissajous.
    # But we can ask Oracle to plan a path from (0,0,10) to (0,0,10).

    oracle = OracleController(num_agents=1)

    # Mock current state
    current_state = {
        'pos_x': np.array([0.0]),
        'pos_y': np.array([0.0]),
        'pos_z': np.array([10.0]),
        'vel_x': np.array([0.0]),
        'vel_y': np.array([0.0]),
        'vel_z': np.array([0.0]),
        'masses': np.array([mass]),
        'drag_coeffs': np.array([drag]),
        'thrust_coeffs': np.array([thrust_coeff])
    }

    # We want to test if the calculated Thrust compensates for Gravity.
    # a_target = 0.
    # F_thrust = m(0 + k*0 + g) = m*g.
    # thrust_cmd = m*g / (20.0 * thrust_coeff)
    # expected = 2.0 * 9.81 / 20.0 = 19.62 / 20.0 = 0.981.

    # Hack: We need to bypass `compute_trajectory` which relies on traj_params for the Target.
    # Let's just use the internal logic or create a dummy traj_params that yields 0,0,10.
    # If we assume P0, V0, A0 = 0 (at 10m), and Pf, Vf, Af = 0 (at 10m), then the quintic is constant.
    # And we just need to see what `actions` are returned.

    # Let's subclass or monkeypatch for this test to force the target state to be static.

    # Actually, `compute_trajectory` takes `traj_params`.
    # Let's construct `traj_params` such that target is fixed at (0,0,10).
    # Ax=0, Fx=0, Px=0 -> x=0
    # Ay=0, Fy=0, Py=0 -> y=0
    # Az=0, Fz=0, Pz=0, Oz=10 -> z=10.

    tp = np.zeros((10, 1), dtype=np.float32)
    tp[9, 0] = 10.0 # Oz

    actions, planned_pos, planned_att = oracle.compute_trajectory(tp, 0.0, 1, current_state)

    thrust = actions[0, 0, 0]
    print(f"Calculated Thrust: {thrust}")

    expected = (mass * 9.81) / (20.0 * thrust_coeff)
    print(f"Expected Thrust: {expected}")

    if abs(thrust - expected) > 1e-3:
        print("FAIL: Thrust mismatch.")
    else:
        print("PASS: Static Hover Thrust correct.")

    # Test Moving Case (Drag Compensation)
    # Moving UP at 1 m/s.
    # Drag opposes motion -> Downward force.
    # Thrust needs to overcome Gravity + Drag.
    # F = m(g + k*v).
    # v = 1.0 (z).
    # F = 2.0 * (9.81 + 0.5 * 1.0) = 2.0 * 10.31 = 20.62.
    # Max Thrust = 20.0.
    # This should CLIPPING (20.62 > 20.0). Output 1.0.

    current_state['vel_z'] = np.array([1.0])
    actions, _, _ = oracle.compute_trajectory(tp, 0.0, 1, current_state)
    thrust = actions[0, 0, 0]
    print(f"Moving Up (High Drag) Thrust: {thrust}. Expected: 1.0 (Clipped)")

    # Test Moving DOWN at 1 m/s.
    # Drag opposes motion -> Upward force.
    # Thrust needs to overcome Gravity - Drag.
    # F = m(g - k*|v|) -> m(g + k*v) where v is -1.
    # F = 2.0 * (9.81 + 0.5 * (-1.0)) = 2.0 * 9.31 = 18.62.
    # cmd = 18.62 / 20.0 = 0.931.

    current_state['vel_z'] = np.array([-1.0])
    actions, _, _ = oracle.compute_trajectory(tp, 0.0, 1, current_state)
    thrust = actions[0, 0, 0]
    expected_down = 18.62 / 20.0
    print(f"Moving Down Thrust: {thrust}. Expected: {expected_down}")

    if abs(thrust - expected_down) > 1e-3:
        print("FAIL: Drag compensation mismatch.")
    else:
        print("PASS: Drag compensation correct.")

if __name__ == "__main__":
    test_oracle_physics()
