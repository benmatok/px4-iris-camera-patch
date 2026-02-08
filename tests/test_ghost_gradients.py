import numpy as np
import sys
import os

# Ensure we can import ghost_dpc
import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ghost_dpc.ghost_dpc import PyGhostModel

def test_gradients():
    print("Testing Analytical Gradients vs Numerical Gradients...")

    # Init Model
    model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0, wind_x=0.0, wind_y=0.0)
    dt = 0.05

    # Random State
    state = {
        'px': 0.0, 'py': 0.0, 'pz': 10.0,
        'vx': 1.0, 'vy': -1.0, 'vz': 0.0,
        'roll': 0.1, 'pitch': -0.1, 'yaw': 0.5
    }

    # Random Action
    # Thrust 0.6, Rates small
    action = {
        'thrust': 0.6,
        'roll_rate': 0.2,
        'pitch_rate': -0.3,
        'yaw_rate': 0.1
    }

    # 1. Analytical Jacobian
    J_ana, J_mass_ana = model.get_gradients(state, action, dt) # (9, 4), (9,)

    # 2. Numerical Jacobian (Action)
    J_num = np.zeros((9, 4), dtype=np.float32)
    epsilon = 1e-2

    action_keys = ['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate']
    state_keys = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw']

    # Baseline
    next_s_base = model.step(state, action, dt)
    base_vec = np.array([next_s_base[k] for k in state_keys])

    for j, key in enumerate(action_keys):
        # Perturb
        act_p = action.copy()
        act_p[key] += epsilon

        next_s_p = model.step(state, act_p, dt)
        p_vec = np.array([next_s_p[k] for k in state_keys])

        # Finite Diff
        grad = (p_vec - base_vec) / epsilon
        J_num[:, j] = grad

    # Compare Action Gradients
    diff = np.abs(J_ana - J_num)
    max_diff = np.max(diff)

    print(f"Max Action Gradient Difference: {max_diff}")

    if max_diff > 1e-4:
        print("FAIL: Action Gradient mismatch!")
        print("Analytical:\n", J_ana)
        print("Numerical:\n", J_num)
        print("Diff:\n", diff)
        sys.exit(1)

    # 3. Numerical Gradient (Mass) - Centered Difference
    # Re-init model with perturbed mass
    mass_eps = 1e-2

    # Plus
    model_p = PyGhostModel(mass=1.0 + mass_eps, drag=0.1, thrust_coeff=1.0)
    next_s_p = model_p.step(state, action, dt)
    vec_p = np.array([next_s_p[k] for k in state_keys])

    # Minus
    model_m = PyGhostModel(mass=1.0 - mass_eps, drag=0.1, thrust_coeff=1.0)
    next_s_m = model_m.step(state, action, dt)
    vec_m = np.array([next_s_m[k] for k in state_keys])

    grad_mass_num = (vec_p - vec_m) / (2 * mass_eps)

    diff_m = np.abs(J_mass_ana - grad_mass_num)
    max_diff_m = np.max(diff_m)

    print(f"Max Mass Gradient Difference: {max_diff_m}")
    if max_diff_m > 1e-4:
        print("FAIL: Mass Gradient mismatch!")
        print("Analytical:\n", J_mass_ana)
        print("Numerical:\n", grad_mass_num)
        print("Diff:\n", diff_m)
        sys.exit(1)

    print("PASS: All Gradients match.")

if __name__ == "__main__":
    test_gradients()
