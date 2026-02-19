
import sys
import os
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.validate_dive_tracking import DiveValidator
from flight_config import FlightConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugScenario4")

def main():
    print("--- Debugging Scenario 4 (60m Alt, 80m Dist) ---")
    config = FlightConfig()

    # Init Validator
    validator = DiveValidator(
        use_ground_truth=True,
        use_blind_mode=True,
        init_alt=60.0,
        init_dist=80.0,
        config=config
    )

    # Run Step-by-Step and Log Divergence
    # We want to compare "What GDPC Predicted" vs "What Happened".

    steps = 400 # 20s
    dt = 0.05

    for i in range(steps):
        # 1. Get State
        s = validator.sim.get_state()

        # 2. Run Step (calls controller)
        # We need to hack `validator.run` logic or call `controller.compute_action`?
        # `DiveValidator.run` does everything. Let's assume we use `run` but break it down?
        # No, `run` is a loop.
        # Let's write a mini-loop here.

        # Capture Control Input
        # ... actually `DiveValidator.run` captures `ghost_paths`.
        # So we can just run `validator.run(duration=20.0)` and inspect the history.
        pass

    hist = validator.run(duration=10.0) # Short run to see start

    # Analyze Divergence
    # Compare hist['drone_pos'] (Actual) with hist['ghost_paths'][t] (Prediction)

    print("\n--- Divergence Analysis ---")
    print(f"{'Time':<6} {'Act Pos (N,E,D)':<25} {'Pred Pos (t+1)':<25} {'Error':<10}")

    for i in range(0, len(hist['t']), 10): # Every 0.5s
        t = hist['t'][i]
        act_pos = hist['drone_pos'][i]

        # Actual NED (approx, since s is Sim ENU)
        # s px (East), py (North), pz (Up)
        # Act Pos from history is Sim Frame [E, N, U].

        if i >= len(hist['ghost_paths']): break
        gps = hist['ghost_paths'][i]
        if not gps: continue
        gp = gps[0] # First path

        # Prediction at t+1 (First step of plan)
        # gp is Relative Sim Frame.
        pred_rel = gp[0] # px, py, pz

        # Reconstruct Absolute Prediction
        # Ghost starts at current state.
        # So pred_abs = current + pred_rel?
        # Wait. `NumPyGhostModel` rollout returns absolute state if initialized with absolute state?
        # `GDPCOptimizer` initializes with `gdpc_state`.
        # `gdpc_state` has `px=0, py=0, pz=0`.
        # So `gp` IS Relative displacement from current position.

        pred_abs_x = act_pos[0] + pred_rel['px']
        pred_abs_y = act_pos[1] + pred_rel['py']

        # Z is tricky.
        # If `vel_reliable` (Scenario 4 usually implies reliable VIO if speed < 40), then Z is relative.
        # If not, Z is Absolute Estimate?
        # `flight_controller.py`:
        # `ghost_paths` constructed from `traj_enu`.
        # `traj_enu` from `rollout`.
        # `rollout` initialized with `gdpc_state`.
        # `gdpc_state` has `pz=0`.
        # So `gp` is Relative Z.

        # But `validate_dive_tracking` handles reliable vs heuristic Z differently.
        # In `compute_action`:
        # GDPC logic: `path.append({'px': traj_enu...})`.
        # `traj_enu` starts at 0.
        # So `gp` is Relative.

        pred_abs_z = act_pos[2] + pred_rel['pz']

        # Next Actual Step (t+1)
        if i+1 < len(hist['drone_pos']):
            next_act = hist['drone_pos'][i+1]
            err_x = next_act[0] - pred_abs_x
            err_y = next_act[1] - pred_abs_y
            err_z = next_act[2] - pred_abs_z
            error = np.sqrt(err_x**2 + err_y**2 + err_z**2)

            print(f"{t:<6.2f} ({act_pos[1]:.1f}, {act_pos[0]:.1f}, {-act_pos[2]:.1f})   ({pred_abs_y:.1f}, {pred_abs_x:.1f}, {-pred_abs_z:.1f})   {error:.4f}")

            # Log detailed direction mismatch
            # Velocity Vector Actual
            v_act = np.array(next_act) - np.array(act_pos)
            # Velocity Vector Pred
            v_pred = np.array([pred_rel['px'], pred_rel['py'], pred_rel['pz']])

            # Angle between them?
            norm_act = np.linalg.norm(v_act)
            norm_pred = np.linalg.norm(v_pred)
            if norm_act > 0.01 and norm_pred > 0.01:
                dot = np.dot(v_act, v_pred)
                cos_angle = dot / (norm_act * norm_pred)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                if angle > 45.0:
                    print(f"  [WARNING] Large Angle Divergence: {angle:.1f} deg")
                    print(f"  Act Vel: {v_act}")
                    print(f"  Pred Vel: {v_pred}")

if __name__ == "__main__":
    main()
