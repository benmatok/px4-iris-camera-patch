import sys
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from dataclasses import asdict

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.validate_dive_tracking import DiveValidator
from flight_config import FlightConfig

def run_sensitivity_analysis():
    print("Starting Sensitivity Analysis...")

    # Baseline Config
    base_config = FlightConfig()

    # Parameters to vary (Perturbation Factor applied to Sim Config only)
    # The Controller Config stays at Baseline (1.0)
    # This simulates "Real World Mismatch" vs "Model Assumptions"

    params = {
        'mass': [0.9, 1.0, 1.1], # Reduced set
        'drag_coeff': [0.5, 1.0, 2.0], # Wide range
        'thrust_coeff': [1.0], # Skip for speed
    }

    results = {}

    # 1. Mass Sensitivity
    print("\n--- Testing Mass Sensitivity ---")
    results['mass'] = []
    for factor in params['mass']:
        sim_conf = copy.deepcopy(base_config)
        sim_conf.physics.mass *= factor

        # Controller thinks mass is base_config.mass (1.0 * base)
        ctrl_conf = copy.deepcopy(base_config)

        validator = DiveValidator(
            use_ground_truth=True, # Use GT for detection to isolate control/dynamics error
            use_blind_mode=True,   # Use VIO for state est
            init_alt=50.0,
            init_dist=150.0,
            sim_config=sim_conf,
            ctrl_config=ctrl_conf
        )

        # Run short simulation (enough to see divergence or lag)
        hist = validator.run(duration=2.5)
        hist = validator.run(duration=2.5)
        hist = validator.run(duration=2.5)

        # Metrics
        final_dist = hist['dist'][-1]

        # Prediction Error
        pred_errors = []
        if 'ghost_paths' in hist:
            for i in range(len(hist['t'])):
                if i >= len(hist['ghost_paths']): break
                gps = hist['ghost_paths'][i]
                if not gps: continue
                gp = gps[0] # First path

                # Check horizon error (last point) vs actual future
                horizon = len(gp)
                future_idx = i + horizon
                if future_idx < len(hist['drone_pos']):
                    pred_pos = np.array([gp[-1]['px'], gp[-1]['py'], gp[-1]['pz']])
                    # Relative prediction
                    start_pos = np.array(hist['drone_pos'][i])

                    # Convert GP to Abs
                    pred_abs = start_pos + pred_pos # GP is relative ENU [px, py, pz]

                    actual_abs = np.array(hist['drone_pos'][future_idx])

                    # Note: GP Z is relative to start Z?
                    # VIO reliable: yes. Unreliable: Z is rel.
                    # We assume reliable here for most parts.

                    err = np.linalg.norm(pred_abs - actual_abs)
                    pred_errors.append(err)

        mean_pred_err = np.mean(pred_errors) if pred_errors else 0.0

        print(f"Mass Factor {factor:.2f}: Dist={final_dist:.2f}m, PredErr={mean_pred_err:.2f}m")
        results['mass'].append({
            'factor': factor,
            'dist': final_dist,
            'pred_err': mean_pred_err
        })

    # 2. Drag Sensitivity
    print("\n--- Testing Drag Sensitivity ---")
    results['drag'] = []
    for factor in params['drag_coeff']:
        sim_conf = copy.deepcopy(base_config)
        sim_conf.physics.drag_coeff *= factor

        ctrl_conf = copy.deepcopy(base_config)

        validator = DiveValidator(
            use_ground_truth=True,
            use_blind_mode=True,
            init_alt=50.0,
            init_dist=150.0,
            sim_config=sim_conf,
            ctrl_config=ctrl_conf
        )

        hist = validator.run(duration=5.0)
        final_dist = hist['dist'][-1]

        # Simple Pred Error Calc (Last Step Horizon)
        # Just use final distance as proxy for "Did it work?"
        # But prediction error is what user asked about "what drives error".

        # Recalc pred error
        pred_errors = []
        if 'ghost_paths' in hist:
            for i in range(len(hist['t'])):
                if i >= len(hist['ghost_paths']): break
                gps = hist['ghost_paths'][i]
                if not gps: continue
                gp = gps[0]
                horizon = len(gp)
                future_idx = i + horizon
                if future_idx < len(hist['drone_pos']):
                    pred_pos = np.array([gp[-1]['px'], gp[-1]['py'], gp[-1]['pz']])
                    start_pos = np.array(hist['drone_pos'][i])
                    pred_abs = start_pos + pred_pos
                    actual_abs = np.array(hist['drone_pos'][future_idx])
                    err = np.linalg.norm(pred_abs - actual_abs)
                    pred_errors.append(err)
        mean_pred_err = np.mean(pred_errors) if pred_errors else 0.0

        print(f"Drag Factor {factor:.2f}: Dist={final_dist:.2f}m, PredErr={mean_pred_err:.2f}m")
        results['drag'].append({
            'factor': factor,
            'dist': final_dist,
            'pred_err': mean_pred_err
        })

    # 3. Thrust Sensitivity
    print("\n--- Testing Thrust Sensitivity ---")
    results['thrust'] = []
    for factor in params['thrust_coeff']:
        sim_conf = copy.deepcopy(base_config)
        sim_conf.physics.thrust_coeff *= factor

        ctrl_conf = copy.deepcopy(base_config)

        validator = DiveValidator(
            use_ground_truth=True,
            use_blind_mode=True,
            init_alt=50.0,
            init_dist=150.0,
            sim_config=sim_conf,
            ctrl_config=ctrl_conf
        )

        hist = validator.run(duration=5.0)
        final_dist = hist['dist'][-1]

        # Pred Err
        pred_errors = []
        if 'ghost_paths' in hist:
            for i in range(len(hist['t'])):
                if i >= len(hist['ghost_paths']): break
                gps = hist['ghost_paths'][i]
                if not gps: continue
                gp = gps[0]
                horizon = len(gp)
                future_idx = i + horizon
                if future_idx < len(hist['drone_pos']):
                    pred_pos = np.array([gp[-1]['px'], gp[-1]['py'], gp[-1]['pz']])
                    start_pos = np.array(hist['drone_pos'][i])
                    pred_abs = start_pos + pred_pos
                    actual_abs = np.array(hist['drone_pos'][future_idx])
                    err = np.linalg.norm(pred_abs - actual_abs)
                    pred_errors.append(err)
        mean_pred_err = np.mean(pred_errors) if pred_errors else 0.0

        print(f"Thrust Factor {factor:.2f}: Dist={final_dist:.2f}m, PredErr={mean_pred_err:.2f}m")
        results['thrust'].append({
            'factor': factor,
            'dist': final_dist,
            'pred_err': mean_pred_err
        })

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Mass
    mass_res = results['mass']
    axs[0].plot([r['factor'] for r in mass_res], [r['pred_err'] for r in mass_res], 'b-o', label='Pred Error')
    axs[0].set_title('Sensitivity to Mass Mismatch (Sim/Model)')
    axs[0].set_xlabel('Mass Factor (Sim)')
    axs[0].set_ylabel('Mean Prediction Error (m)')
    axs[0].grid(True)

    # Drag
    drag_res = results['drag']
    axs[1].plot([r['factor'] for r in drag_res], [r['pred_err'] for r in drag_res], 'r-o', label='Pred Error')
    axs[1].set_title('Sensitivity to Drag Mismatch')
    axs[1].set_xlabel('Drag Factor (Sim)')
    axs[1].set_ylabel('Mean Prediction Error (m)')
    axs[1].grid(True)

    # Thrust
    thrust_res = results['thrust']
    axs[2].plot([r['factor'] for r in thrust_res], [r['pred_err'] for r in thrust_res], 'g-o', label='Pred Error')
    axs[2].set_title('Sensitivity to Thrust Efficiency Mismatch')
    axs[2].set_xlabel('Thrust Factor (Sim)')
    axs[2].set_ylabel('Mean Prediction Error (m)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('sensitivity_results.png')
    print("Saved sensitivity_results.png")

if __name__ == "__main__":
    run_sensitivity_analysis()
