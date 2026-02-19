
import sys
import os
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flight_controller_gdpc import GDPCOptimizer
from flight_config import FlightConfig

logging.basicConfig(level=logging.INFO)

def main():
    print("--- Debugging GDPC Plan ---")
    config = FlightConfig()
    optimizer = GDPCOptimizer(config)

    # Scenario 2 State (Approx)
    # Pos: (0, 0, 50). Vel: (0,0,0).
    # Pitch: -60 deg (Nose Down). Yaw: 0 (East).
    # Sim Frame.

    state_obs = {
        'px': 0.0, 'py': 0.0, 'pz': 50.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0,
        'pitch': np.radians(-60.0), # Nose Down
        'yaw': 0.0, # East
        'wx': 0.0, 'wy': 0.0, 'wz': 0.0
    }

    # Target (Relative Sim Frame)
    # East 75m.
    target_pos = np.array([75.0, 0.0, 0.0]) # Relative X, Y, Z?
    # Relative Z: Target at 0. Drone at 50. Rel = -50.
    target_pos = np.array([75.0, 0.0, -50.0])

    print(f"Initial State: {state_obs}")
    print(f"Target (Rel): {target_pos}")

    # Compute Action
    action, traj = optimizer.compute_action(state_obs, target_pos)

    print("\nPredicted Trajectory (First 5 steps):")
    print(f"{'Step':<5} {'PX':<10} {'PY':<10} {'PZ':<10} {'VX':<10} {'VY':<10}")

    for t in range(5):
        # traj: px, py, pz, vx, vy, vz ...
        p = traj[t]
        print(f"{t:<5} {p[0]:<10.2f} {p[1]:<10.2f} {p[2]:<10.2f} {p[3]:<10.2f} {p[4]:<10.2f}")

    final = traj[-1]
    print(f"\nFinal State (t={len(traj)}):")
    print(f"Pos: ({final[0]:.2f}, {final[1]:.2f}, {final[2]:.2f})")

    # Check Direction
    dx = final[0] - state_obs['px']
    dy = final[1] - state_obs['py']

    if abs(dy) > abs(dx):
        print("\n[FAIL] Prediction moves mainly in Y (North)!")
    else:
        print("\n[PASS] Prediction moves mainly in X (East).")

if __name__ == "__main__":
    main()
