import numpy as np
import sys
import os

import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ghost_dpc.ghost_dpc import PyGhostEstimator, PyGhostModel

def test_estimator():
    print("Testing Ghost Estimator...")

    # 1. Models
    # Light, Nominal, Heavy
    models = [
        {'mass': 0.5, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # 0
        {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # 1
        {'mass': 1.5, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}  # 2
    ]

    estimator = PyGhostEstimator(models)

    # Check Init
    probs = estimator.get_probabilities()
    print("Init Probs:", probs)
    assert np.allclose(probs, [0.333, 0.333, 0.333], atol=1e-2)

    # 2. Simulation (Real World = Nominal)
    real_model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0)

    state = {
        'px': 0.0, 'py': 0.0, 'pz': 10.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
    }
    action = {
        'thrust': 0.5, # Hoverish
        'roll_rate': 0.0,
        'pitch_rate': 0.0,
        'yaw_rate': 0.0
    }
    dt = 0.05

    # Run loop
    for i in range(20):
        # Step Real World
        next_s = real_model.step(state, action, dt)

        # Calculate Acceleration (Real)
        # a = (v_next - v_curr) / dt
        ax = (next_s['vx'] - state['vx']) / dt
        ay = (next_s['vy'] - state['vy']) / dt
        az = (next_s['vz'] - state['vz']) / dt

        # Update Estimator
        estimator.update(state, action, [ax, ay, az], dt)

        state = next_s

    probs = estimator.get_probabilities()
    print("After 20 steps (Real=Nominal):", probs)

    best_idx = np.argmax(probs)
    if best_idx != 1:
        print(f"FAIL: Expected index 1, got {best_idx}")
        sys.exit(1)

    if probs[1] < 0.9:
        print(f"FAIL: Confidence too low: {probs[1]}")
        sys.exit(1)

    # 3. Switch Real World to Heavy
    print("Switching Real World to Heavy (1.5kg)...")
    real_model = PyGhostModel(mass=1.5, drag=0.1, thrust_coeff=1.0)

    for i in range(30):
        next_s = real_model.step(state, action, dt)
        ax = (next_s['vx'] - state['vx']) / dt
        ay = (next_s['vy'] - state['vy']) / dt
        az = (next_s['vz'] - state['vz']) / dt

        estimator.update(state, action, [ax, ay, az], dt)
        state = next_s

    probs = estimator.get_probabilities()
    print("After 30 steps (Real=Heavy):", probs)

    best_idx = np.argmax(probs)
    if best_idx != 2:
        print(f"FAIL: Expected index 2, got {best_idx}")
        sys.exit(1)

    print("PASS: Estimator converged correctly.")

if __name__ == "__main__":
    test_estimator()
