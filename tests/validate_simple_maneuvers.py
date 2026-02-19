
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_interface import SimDroneInterface
from flight_controller_gdpc import NumPyGhostModel
from flight_config import FlightConfig
from vision.projection import Projector

def run_test_case(name, init_state, action_seq, duration, dt=0.05):
    print(f"--- Running Test: {name} ---")

    # Setup
    config = FlightConfig()
    phy = config.physics

    # 1. Sim (Ground Truth)
    proj = Projector(640, 480, 120.0, 30.0)
    sim = SimDroneInterface(proj, config=config)

    # Force Init State
    # Reset to blank then override
    sim.reset_to_scenario("Blind Dive")
    sim.state.update(init_state)

    # 2. Prediction Model
    pred_model = NumPyGhostModel(
        mass=phy.mass,
        drag_coeff=phy.drag_coeff,
        thrust_coeff=phy.thrust_coeff,
        tau=phy.tau,
        g=phy.g,
        max_thrust_base=phy.max_thrust_base
    )

    # Run Prediction (Open Loop Rollout)
    pred_traj = pred_model.rollout(sim.state, action_seq, dt=dt)

    # Run Sim (Step-by-Step)
    sim_traj = []

    for t, u in enumerate(action_seq):
        # Log State Before Step
        s = sim.get_state()
        sim_traj.append([s['px'], s['py'], s['pz'], s['vx'], s['vy'], s['vz']])

        # Step
        # Sim expects action array: [thrust, roll_rate, pitch_rate, yaw_rate]
        sim.step(u)

    sim_traj = np.array(sim_traj)

    # Compare
    # pred_traj includes state at t=1..H. sim_traj includes t=0..H-1?
    # pred_model.rollout returns state AFTER applying action.
    # sim_traj should be logged AFTER applying action to match?
    # Let's adjust sim loop to log AFTER.

    sim.state.update(init_state) # Reset again
    sim_traj_after = []
    for t, u in enumerate(action_seq):
        sim.step(u)
        s = sim.get_state()
        sim_traj_after.append([s['px'], s['py'], s['pz'], s['vx'], s['vy'], s['vz']])

    sim_traj = np.array(sim_traj_after)

    # Calculate Errors
    pos_err = np.linalg.norm(sim_traj[:, 0:3] - pred_traj[:, 0:3], axis=1)
    vel_err = np.linalg.norm(sim_traj[:, 3:6] - pred_traj[:, 3:6], axis=1)

    mean_pos_err = np.mean(pos_err)
    max_pos_err = np.max(pos_err)

    print(f"Mean Pos Error: {mean_pos_err:.6f} m")
    print(f"Max Pos Error:  {max_pos_err:.6f} m")

    return sim_traj, pred_traj, pos_err

def main():
    dt = 0.05
    duration = 2.0
    steps = int(duration / dt)

    # Hover Thrust calculation
    # g=9.81, max=20.0. T = 9.81/20.0 = 0.4905
    hover_thrust = 9.81 / 20.0

    # Case 1: Perfect Hover
    init_hover = {
        'px': 0.0, 'py': 0.0, 'pz': 100.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'wx': 0.0, 'wy': 0.0, 'wz': 0.0
    }
    action_hover = np.zeros((steps, 4))
    action_hover[:, 0] = hover_thrust

    s_hov, p_hov, e_hov = run_test_case("Hover", init_hover, action_hover, duration)

    # Case 2: Full Throttle Up
    action_up = np.zeros((steps, 4))
    action_up[:, 0] = 1.0
    s_up, p_up, e_up = run_test_case("Max Thrust Up", init_hover, action_up, duration)

    # Case 3: Free Fall (Zero Thrust)
    action_drop = np.zeros((steps, 4))
    action_drop[:, 0] = 0.0
    s_drop, p_drop, e_drop = run_test_case("Free Fall", init_hover, action_drop, duration)

    # Case 4: Forward Acceleration (Pitch Down)
    # Pitch down 30 deg approx? No, rate control.
    # Apply negative pitch rate for 0.5s, then neutral.
    action_fwd = np.zeros((steps, 4))
    action_fwd[:, 0] = hover_thrust # Maintain Z roughly
    action_fwd[0:10, 2] = 1.0 # Pitch Rate +1.0 (Nose Up? or Down?)
    # ENU: Pitch + is Nose Up.
    # We want Forward. Forward is +Y (North).
    # Nose Down = Pitch -.
    # Action: Pitch Rate -.
    action_fwd[0:10, 2] = -1.0

    s_fwd, p_fwd, e_fwd = run_test_case("Forward Accel", init_hover, action_fwd, duration)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    def plot_case(ax, s, p, title):
        ax.set_title(title)
        ax.plot(s[:, 0], s[:, 2], 'b-', label='Sim (True)') # X-Z view?
        ax.plot(p[:, 0], p[:, 2], 'r--', label='Pred')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.legend()
        ax.grid(True)

    # Hover: Plot Z vs Time
    axs[0, 0].set_title("Hover Z vs Time")
    t = np.arange(steps) * dt
    axs[0, 0].plot(t, s_hov[:, 2], 'b-', label='Sim')
    axs[0, 0].plot(t, p_hov[:, 2], 'r--', label='Pred')
    axs[0, 0].set_ylim(99, 101)

    # Max Up: Z vs Time
    axs[0, 1].set_title("Max Up Z vs Time")
    axs[0, 1].plot(t, s_up[:, 2], 'b-')
    axs[0, 1].plot(t, p_up[:, 2], 'r--')

    # Drop: Z vs Time
    axs[1, 0].set_title("Free Fall Z vs Time")
    axs[1, 0].plot(t, s_drop[:, 2], 'b-')
    axs[1, 0].plot(t, p_drop[:, 2], 'r--')

    # Fwd: Y vs Z (Side view)
    axs[1, 1].set_title("Forward Flight (Y-Z)")
    axs[1, 1].plot(s_fwd[:, 1], s_fwd[:, 2], 'b-', label='Sim')
    axs[1, 1].plot(p_fwd[:, 1], p_fwd[:, 2], 'r--', label='Pred')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("validation_simple_maneuvers.png")
    print("Saved plot to validation_simple_maneuvers.png")

if __name__ == "__main__":
    main()
