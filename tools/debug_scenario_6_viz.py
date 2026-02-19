
import sys
import os
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.validate_dive_tracking import DiveValidator
from flight_config import FlightConfig

logging.basicConfig(level=logging.INFO)

def main():
    print("--- Debugging Scenario 6 Viz ---")
    config = FlightConfig()

    # Scenario 6: Alt 25, Dist 150
    validator = DiveValidator(
        use_ground_truth=True,
        use_blind_mode=True,
        init_alt=25.0,
        init_dist=150.0,
        config=config
    )

    # Run short
    hist = validator.run(duration=5.0)

    print("\n--- Ghost Path Gap Analysis ---")
    print(f"{'Time':<6} {'Act Pos (Sim)':<25} {'Ghost Start (Abs)':<25} {'Gap (m)':<10}")

    pos = hist['drone_pos']
    ghosts = hist['ghost_paths']
    reliable = hist['vel_reliable']

    max_gap = 0.0

    for i in range(0, len(hist['t']), 10):
        if i >= len(ghosts): break
        gps = ghosts[i]
        if not gps: continue
        gp = gps[0]

        curr_pos = pos[i] # [px, py, pz] Sim

        # Reconstruct Abs Start of Ghost
        # Validate logic:
        # abs_x = curr_pos[0] + px

        start_px = gp[0]['px']
        start_py = gp[0]['py']
        start_pz = gp[0]['pz']

        rel = reliable[i]

        abs_x = curr_pos[0] + start_px
        abs_y = curr_pos[1] + start_py

        if rel:
            abs_z = curr_pos[2] + start_pz
        else:
            # Heuristic logic in validate:
            # start_z_val = gp[0]['pz']
            # rel_z = start_pz - start_z_val = 0
            # abs_z = curr_pos[2] + 0
            abs_z = curr_pos[2]

        ghost_start_abs = np.array([abs_x, abs_y, abs_z])
        curr_pos_arr = np.array(curr_pos)

        gap = np.linalg.norm(ghost_start_abs - curr_pos_arr)
        if gap > max_gap: max_gap = gap

        print(f"{hist['t'][i]:<6.2f} {str(np.round(curr_pos, 2)):<25} {str(np.round(ghost_start_abs, 2)):<25} {gap:<10.2f}")

    print(f"\nMax Gap: {max_gap:.2f} m")

    if max_gap > 2.0:
        print("[FAIL] Ghost Path does not start at Drone Position!")
    else:
        print("[PASS] Ghost Path starts near Drone Position.")

if __name__ == "__main__":
    main()
