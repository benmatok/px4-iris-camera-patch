
import sys
import os
import numpy as np
import logging
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules to avoid dependencies not relevant for viz logic
sys.modules['fastapi'] = MagicMock()
sys.modules['fastapi.staticfiles'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()

from theshow import TheShow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugVizFrame")

def main():
    print("--- Debugging Visualization Frame ---")

    # 1. Instantiate TheShow
    try:
        app = TheShow()
    except Exception as e:
        print(f"Init failed (expected if dependencies missing): {e}")
        # We might need to mock more if init fails hard.
        # But let's try to verify the logic method directly if possible.
        # `compute_step` is complex.
        # Let's inspect the logic by reading code or subclassing.
        return

    # 2. Mock State: Drone Moving East
    # Sim Frame: X=East, Y=North.
    # Pos: 0, 0, 100.
    # Vel: 10, 0, 0. (East)

    app.sim.state = {
        'px': 0.0, 'py': 0.0, 'pz': 100.0,
        'vx': 10.0, 'vy': 0.0, 'vz': 0.0, # Moving East
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, # Facing East (Yaw=0)
        'wx': 0.0, 'wy': 0.0, 'wz': 0.0
    }

    # 3. Mock Ghost Path (Prediction matches Reality)
    # Ghost is Relative Sim Frame.
    # Path: (0,0,0) -> (10,0,0) -> (20,0,0) ...
    path = []
    for i in range(10):
        path.append({
            'px': float(i * 10.0), # East
            'py': 0.0,
            'pz': 0.0
        })

    # Inject into Controller (Mock return)
    app.controller.compute_action = MagicMock(return_value=(
        {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0},
        [path] # ghost_paths
    ))

    # 4. Run compute_step (or just the viz logic part)
    # We can invoke compute_step.
    # It calls sim.get_state, tracker.process, etc.
    # We need to mock tracker.
    app.tracker.process = MagicMock(return_value=(None, None, 0.0))
    app.msckf.initialized = True # Skip init
    app.msckf.propagate = MagicMock()
    app.msckf.augment_state = MagicMock()
    app.msckf.update_features = MagicMock()
    app.msckf.update_height = MagicMock()
    app.msckf.get_velocity = MagicMock(return_value=np.array([0.0, 10.0, 0.0])) # NED: North=0, East=10.
    app.msckf.is_reliable = MagicMock(return_value=True)
    app.feature_tracker.update = MagicMock(return_value=(None, []))

    payload = app.compute_step()

    # 5. Analyze Payload
    # 'drone' state (NED)
    drone_ned = payload['drone']
    print("\nDrone Viz (NED):")
    print(f"  PX (North): {drone_ned['px']:.2f}")
    print(f"  PY (East):  {drone_ned['py']:.2f}")

    # 'ghosts' state (NED List)
    ghosts = payload['ghosts'][0]
    print("\nGhost Path Viz (NED):")
    print(f"{'Step':<5} {'PX (North)':<15} {'PY (East)':<15}")
    for i, p in enumerate(ghosts):
        print(f"{i:<5} {p['px']:<15.2f} {p['py']:<15.2f}")

    # Check Direction
    # Drone moves East. Sim X+. NED Y+.
    # Ghost moves East. Sim X+. NED Y+.

    d_ghost_x = ghosts[-1]['px'] - ghosts[0]['px'] # North change
    d_ghost_y = ghosts[-1]['py'] - ghosts[0]['py'] # East change

    print(f"\nGhost Displacement: North={d_ghost_x:.2f}, East={d_ghost_y:.2f}")

    if abs(d_ghost_y) > abs(d_ghost_x):
        print("[PASS] Ghost moves East (Parallel to Drone).")
    else:
        print("[FAIL] Ghost moves North (Perpendicular to Drone)!")

if __name__ == "__main__":
    main()
