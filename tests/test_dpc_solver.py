import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from ghost_dpc.ghost_dpc import PyDPCSolver, PyGhostModel

def test_solver():
    print("Testing DPC Solver...")

    solver = PyDPCSolver()

    # Model
    model_dict = {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}
    models = [model_dict]
    weights = [1.0]

    # State: Hovering at (0,0,10)
    state = {
        'px': 0.0, 'py': 0.0, 'pz': 10.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
    }

    # Target: Above at (0,0,15)
    target = [0.0, 0.0, 15.0]

    # Init Guess: Hover (0.5)
    # Mass=1, MaxT=20. Hover = 9.81/20 = 0.49.
    init_action = {
        'thrust': 0.49,
        'roll_rate': 0.0,
        'pitch_rate': 0.0,
        'yaw_rate': 0.0
    }
    dt = 0.05

    # Solve
    opt_action = solver.solve(state, target, init_action, models, weights, dt)

    print("Init Thrust:", init_action['thrust'])
    print("Opt Thrust:", opt_action['thrust'])

    if opt_action['thrust'] > 0.55:
        print("PASS: Thrust increased to climb.")
    else:
        print("FAIL: Thrust did not increase significantly.")
        sys.exit(1)

    # Test lateral move
    # Target at (5, 0, 10). Drone at (0, 0, 10).
    # Needs to pitch down (pitch > 0) to accelerate +X.
    # Note: Body X is forward.
    # Positive Pitch is Nose Down?
    # In step:
    # r31 = cy*sp*cr + sy*sr.
    # If yaw=0, roll=0. r31 = sp.
    # ax = F/m * r31 = F/m * sin(pitch).
    # If pitch > 0, sin(pitch) > 0 -> ax > 0.
    # So Positive Pitch = Forward Acceleration.

    target_lat = [5.0, 0.0, 10.0]
    opt_action_lat = solver.solve(state, target_lat, init_action, models, weights, dt)

    print("Opt Pitch Rate:", opt_action_lat['pitch_rate'])

    # We assume it should pitch forward (positive rate).
    if opt_action_lat['pitch_rate'] > 0.1:
        print("PASS: Pitch rate positive to move forward.")
    else:
        print("FAIL: Pitch rate not positive.")
        sys.exit(1)

if __name__ == "__main__":
    test_solver()
