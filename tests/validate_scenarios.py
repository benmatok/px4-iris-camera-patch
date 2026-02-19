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
        # {"id": 1, "alt": 100.0, "dist": 50.0},
        # {"id": 2, "alt": 50.0,  "dist": 75.0},
        # {"id": 3, "alt": 20.0,  "dist": 50.0},
        {"id": 4, "alt": 60.0,  "dist": 80.0},
        # {"id": 5, "alt": 50.0,  "dist": 50.0},
        # {"id": 6, "alt": 25.0,  "dist": 150.0}
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

        # Run for 40 seconds to accommodate 5 m/s speed limit
        hist = validator.run(duration=40.0)

        # GENERATE PLOTS
        from tests.validate_dive_tracking import plot_results

        # Pass the correct target position for the plot
        target_pos = [dist, 0.0, 0.0]

        plot_filename = f"scenario_{sid}.png"
        plot_results(hist, hist, hist, filename=plot_filename, target_pos=target_pos)
        logger.info(f"Generated plot: {plot_filename}")

        # Prediction Error Analysis
        if 'ghost_paths' in hist:
            pred_errors = []
            pos = np.array(hist['drone_pos'])
            for i in range(len(hist['t'])):
                if i >= len(hist['ghost_paths']): break
                gps = hist['ghost_paths'][i]
                if not gps: continue
                gp = gps[0]

                start_z = gp[0]['pz']
                reliable = hist['vel_reliable'][i]

                step_errors = []
                for k, p in enumerate(gp):
                    future_idx = i + k + 1
                    if future_idx >= len(pos): break

                    actual_rel = pos[future_idx] - pos[i]
                    pred_rel_x = p['px']
                    pred_rel_y = p['py']
                    if reliable:
                        pred_rel_z = p['pz']
                    else:
                        pred_rel_z = p['pz'] - start_z

                    pred_rel = np.array([pred_rel_x, pred_rel_y, pred_rel_z])
                    dist = np.linalg.norm(pred_rel - actual_rel)
                    step_errors.append(dist)

                if step_errors:
                    pred_errors.append(np.mean(step_errors))

            if pred_errors:
                avg_pred_err = np.mean(pred_errors)
                max_pred_err = np.max(pred_errors)
                print(f"Prediction Error: Mean={avg_pred_err:.4f}m Max={max_pred_err:.4f}m")
            else:
                 print("Prediction Error: No Data")

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

        print(f"{sid:<4} {alt:<6.1f} {dist:<6.1f} {res_str:<10} {min_dist:<12.2f} 40.0s")

    print("-" * 60)

    if all_passed:
        print("All scenarios passed.")
        sys.exit(0)
    else:
        print("Some scenarios failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_scenarios()
