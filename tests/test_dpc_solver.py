import numpy as np
import sys
import os

import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ghost_dpc.ghost_dpc import PyDPCSolver, PyGhostModel

def test_solver():
    print("Testing DPC Solver...")

    solver = PyDPCSolver()

    # Model
    model_dict = {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}
    models = [model_dict]
    weights = [1.0]

    # --- TEST 1: Dive Scenario (Start High, Target Low) ---
    print("\n--- TEST 1: Dive Scenario ---")
    # px = -30 ensures Target (at 0,0) is Forward (+X relative to Drone)
    state_dive = {
        'px': -30.0, 'py': 0.0, 'pz': 100.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
    }
    init_action = {
        'thrust': 0.49, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0
    }
    dt = 0.05

    # Target 2.0 (Implicit). Dive Cost should be ACTIVE because 100 > 2 + 1.
    history_dive = [state_dive]
    opt_dive, _ = solver.solve(history_dive, init_action, models, weights, dt, goal_z=2.0)

    print("Init Thrust:", init_action['thrust'])
    print("Opt Thrust:", opt_dive['thrust'])
    print("Opt Pitch Rate:", opt_dive['pitch_rate'])

    # Expect Thrust Decrease (Dive) OR Pitch Active
    if opt_dive['thrust'] < 0.49 or abs(opt_dive['pitch_rate']) > 0.0:
        print("PASS: Dive reaction detected.")
    else:
        print("FAIL: No reaction in dive scenario.")
        sys.exit(1)

    # --- TEST 2: Climb Scenario (Start Low, Target High) ---
    print("\n--- TEST 2: Climb Scenario ---")
    state_climb = {
        'px': 0.0, 'py': 0.0, 'pz': 10.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
    }
    # Target 15.0. Climb. Dive Cost should be INACTIVE because 10 !> 15 + 1.
    history_climb = [state_climb]
    opt_climb, _ = solver.solve(history_climb, init_action, models, weights, dt, goal_z=15.0)

    print("Init Thrust:", init_action['thrust'])
    print("Opt Thrust:", opt_climb['thrust'])

    # Expect Thrust Increase (Climb)
    if opt_climb['thrust'] > 0.50:
        print("PASS: Thrust increased to climb.")
    else:
        print("FAIL: Thrust did not increase significantly.")
        sys.exit(1)

    # --- TEST 3: Lateral Move (Maintain Altitude) ---
    print("\n--- TEST 3: Lateral Move ---")
    # Target at (20, 0, 0). Drone at (0, 0, 10). Goal Z = 10.
    # Dive Cost should be INACTIVE because 10 !> 10 + 1.
    # Re-use state_climb (which is at 10m).
    opt_lat, _ = solver.solve(history_climb, init_action, models, weights, dt, target_pos_rel_xy=[20.0, 0.0], goal_z=10.0)

    print("Opt Pitch Rate:", opt_lat['pitch_rate'])

    # Expect Pitch Forward (Positive = Nose Down)
    if opt_lat['pitch_rate'] > 0.05:
        print("PASS: Pitch rate positive (Nose Down) to move forward.")
    else:
        print("FAIL: Pitch rate not positive (forward).")
        sys.exit(1)

if __name__ == "__main__":
    test_solver()
