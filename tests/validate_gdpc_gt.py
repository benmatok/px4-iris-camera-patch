import sys
import os
import logging
import numpy as np
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.validate_dive_tracking import DiveValidator
from flight_config import FlightConfig

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GDPC_GT_Validator")

def run_scenarios_gt(target_id=None):
    print("--- Running Scenarios with Ground Truth State for Controller ---")
    config = FlightConfig()

    scenarios = [
        {"id": 1, "alt": 100.0, "dist": 50.0},
        {"id": 2, "alt": 50.0,  "dist": 75.0},
        {"id": 3, "alt": 20.0,  "dist": 50.0},
        {"id": 4, "alt": 60.0,  "dist": 80.0},
        {"id": 5, "alt": 50.0,  "dist": 50.0},
        {"id": 6, "alt": 25.0,  "dist": 150.0}
    ]

    if target_id is not None:
        scenarios = [s for s in scenarios if s['id'] == target_id]

    all_passed = True
    results = []

    print("-" * 70)
    print(f"{'ID':<4} {'Alt':<6} {'Dist':<6} {'Result':<10} {'Final Dist':<12} {'Time':<6}")
    print("-" * 70)

    for sc in scenarios:
        sid = sc['id']
        alt = sc['alt']
        dist = sc['dist']

        # Instantiate Validator with control_use_gt=True
        validator = DiveValidator(
            use_ground_truth=True,
            use_blind_mode=True,
            init_alt=alt,
            init_dist=dist,
            config=config,
            control_use_gt=True
        )

        # Run for 40 seconds
        hist = validator.run(duration=40.0)

        final_dist = hist['dist'][-1]
        min_dist = min(hist['dist'])

        # Success criteria
        success = min_dist < 2.0

        res_str = "PASS" if success else "FAIL"
        if not success:
            all_passed = False

        results.append({
            "id": sid,
            "alt": alt,
            "dist": dist,
            "result": res_str,
            "final_dist": min_dist
        })

        print(f"{sid:<4} {alt:<6.1f} {dist:<6.1f} {res_str:<10} {min_dist:<12.2f} 40.0s")

    print("-" * 70)

    if all_passed:
        print("All GDPC-GT scenarios passed.")
    else:
        print("Some GDPC-GT scenarios failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, help="Scenario ID to run")
    args = parser.parse_args()
    run_scenarios_gt(args.id)
