
import sys
import os
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_interface import SimDroneInterface, PyGhostModel
from flight_controller_gdpc import NumPyGhostModel
from vision.projection import Projector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComparePhysics")

def compare_models(steps=20, dt=0.05):
    # 1. Initialize Ground Truth (Sim)
    proj = Projector(640, 480, 120.0, 30.0)
    sim = SimDroneInterface(proj)

    # 2. Initialize Prediction Model (GDPC)
    # Defaults from flight_controller_gdpc.py: mass=1.0, drag_coeff=0.1, thrust_coeff=1.0, tau=0.1
    # Defaults from sim_interface.py: mass=1.0, drag_coeff=0.1, thrust_coeff=1.0, tau=0.1
    pred_model = NumPyGhostModel()

    # 3. Set Initial State
    # Start at Hover: thrust=0.5 (approx), orientation flat
    # Sim Interface Reset:
    sim.reset_to_scenario("Blind Dive", pos_x=0.0, pos_y=0.0, pos_z=100.0, pitch=0.0, yaw=0.0)

    # Extract Initial State Dict
    s0 = sim.get_state()

    # 4. Define Action Sequence
    # Test a mix of inputs: Thrust up, Roll right, Pitch down
    actions = []
    for i in range(steps):
        # Action: [thrust, roll_rate, pitch_rate, yaw_rate]
        if i < 5:
            a = [0.6, 0.0, 0.0, 0.0] # Thrust Up
        elif i < 10:
            a = [0.55, 1.0, 0.0, 0.0] # Roll Right
        elif i < 15:
            a = [0.55, 0.0, -0.5, 0.0] # Pitch Up (Nose Up)
        else:
            a = [0.5, 0.0, 0.0, 0.5] # Yaw
        actions.append(np.array(a))

    # 5. Run Comparison
    sim_state = s0.copy()
    pred_state = s0.copy()

    print(f"{'Step':<5} {'Pos Err':<10} {'Vel Err':<10} {'Att Err (deg)':<15}")
    print("-" * 50)

    errors = {'pos': [], 'vel': [], 'att': []}

    # We need to manually step the NumPyGhostModel one by one to compare,
    # but it has a `rollout` function. Let's use rollout for the full sequence
    # and compare at the end, OR step manually if we want step-by-step.
    # NumPyGhostModel doesn't expose a single 'step' function easily (it's inside rollout loop).
    # Let's use rollout for the whole sequence from start state.

    # Prepare rollout input
    # rollout takes state_dict and action_seq (H, 4)
    action_seq = np.array(actions)

    # Run GDPC Prediction
    traj_pred = pred_model.rollout(s0, action_seq, dt=dt)

    # Run Sim Step-by-Step
    for t in range(steps):
        # Apply Action to Sim
        sim.step(actions[t])
        s_sim = sim.get_state()

        # Get Corresponding Prediction
        # traj_pred[t] is state AFTER step t (0-indexed)
        # indices: 0-2 pos, 3-5 vel, 6-8 att
        p_pred = traj_pred[t, 0:3]
        v_pred = traj_pred[t, 3:6]
        att_pred = traj_pred[t, 6:9]

        p_sim = np.array([s_sim['px'], s_sim['py'], s_sim['pz']])
        v_sim = np.array([s_sim['vx'], s_sim['vy'], s_sim['vz']])
        att_sim = np.array([s_sim['roll'], s_sim['pitch'], s_sim['yaw']])

        # Errors
        pos_err = np.linalg.norm(p_pred - p_sim)
        vel_err = np.linalg.norm(v_pred - v_sim)

        # Attitude Error (Simple Euclidean on RPY for small angles)
        # Be careful with wrap-around, but for short tests it's okay.
        att_diff = att_pred - att_sim
        # Normalize
        att_diff = (att_diff + np.pi) % (2 * np.pi) - np.pi
        att_err = np.linalg.norm(att_diff)

        errors['pos'].append(pos_err)
        errors['vel'].append(vel_err)
        errors['att'].append(att_err)

        print(f"{t:<5} {pos_err:<10.4f} {vel_err:<10.4f} {np.degrees(att_err):<15.4f}")

    print("-" * 50)
    print(f"Final Pos Error: {errors['pos'][-1]:.4f} m")
    print(f"Final Vel Error: {errors['vel'][-1]:.4f} m/s")

    if errors['pos'][-1] > 0.1:
        print("\n[FAIL] Significant Position Drift Detected!")
    else:
        print("\n[PASS] Models Match closely.")

if __name__ == "__main__":
    compare_models()
