import sys
import os
import logging
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.validate_dive_tracking import DiveValidator
from flight_config import FlightConfig

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScenarioValidator")

def run_scenarios():
    config = FlightConfig()
    # You can modify config here if needed, e.g.:
    # config.control.k_yaw = 2.5

    scenarios = [
        {"id": 1, "alt": 100.0, "dist": 50.0},
        {"id": 2, "alt": 50.0,  "dist": 75.0},
        {"id": 3, "alt": 20.0,  "dist": 50.0},
        {"id": 4, "alt": 60.0,  "dist": 80.0},
        {"id": 5, "alt": 50.0,  "dist": 50.0},
        {"id": 6, "alt": 25.0,  "dist": 150.0}
    ]

    all_passed = True
    results = []

    print("-" * 60)
    print(f"{'ID':<4} {'Alt':<6} {'Dist':<6} {'Result':<10} {'Final Dist':<12} {'Time':<6}")
    print("-" * 60)

    for sc in scenarios:
        sid = sc['id']
        alt = sc['alt']
        dist = sc['dist']

        # Instantiate Validator matching TheShow configuration
        # use_ground_truth=True (Perfect Tracking)
        # use_blind_mode=True (No VZ sensor)
        validator = DiveValidator(
            use_ground_truth=True,
            use_blind_mode=True,
            init_alt=alt,
            init_dist=dist,
            config=config
        )

        # Run for 25 seconds
        hist = validator.run(duration=25.0)

        # GENERATE PLOTS
        from tests.validate_dive_tracking import plot_results

        # Pass the correct target position for the plot
        target_pos = [dist, 0.0, 0.0]

        plot_filename = f"scenario_{sid}.png"
        plot_results(hist, hist, hist, filename=plot_filename, target_pos=target_pos)
        logger.info(f"Generated plot: {plot_filename}")

        final_dist = hist['dist'][-1]
        min_dist = min(hist['dist'])

        # Check Success
        # Standard: Distance < 5.0m (Collision/Flyby)
        success = min_dist < 5.0

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

        print(f"{sid:<4} {alt:<6.1f} {dist:<6.1f} {res_str:<10} {min_dist:<12.2f} 25.0s")

    print("-" * 60)

    if all_passed:
        print("All scenarios passed.")
        sys.exit(0)
    else:
        print("Some scenarios failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_scenarios()
