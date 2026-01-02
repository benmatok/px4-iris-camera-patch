import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
import cython_drone_sim
from jules_core import JulesPolicy, expand_action_trajectory
from visualization import Visualizer
import os

# --- 1. POLYNOMIAL BRIDGE ---
def coeffs_to_controls(flat_coeffs, duration, dt):
    """Maps 16 coeffs -> Control signals for the Simulation"""
    coeffs = flat_coeffs.reshape(4, 4) # (Controls, PolyOrder)
    steps = int(duration / dt)
    # Ensure at least one step
    if steps < 1: steps = 1
    t = np.linspace(-1, 1, steps)

    # Chebyshev Basis (Order 4)
    # T0 = 1
    # T1 = t
    # T2 = 2t^2-1
    # T3 = 4t^3-3t
    T = np.stack([
        np.ones_like(t),
        t,
        2*t**2-1,
        4*t**3-3*t
    ])

    # Result: (Steps, 4_Controls)
    return (coeffs @ T).T

# --- 2. HEURISTICS & CONSTRAINTS (USER CONFIG) ---
def apply_heuristics(state, control, target_pos):
    """
    Calculates soft penalties for specific flight behaviors.
    """
    cost = 0.0

    # H1: Orientation - Face the target?
    # cost += heading_error * 5.0

    # H2: Smoothness - Penalize jerking the throttle
    # cost += abs(control[0] - prev_control[0]) * 10.0

    return cost

def get_warm_start(initial_state, target_pos):
    """
    Provides a 'good guess' to speed up the solver.
    """
    guess = np.zeros((4, 4))

    # Heuristic: Counter-Gravity Thrust
    # T0 (Average Thrust) should be approx Gravity (~0.5 normalized)
    guess[0, 0] = 0.5

    # Heuristic: Pitch towards target
    # initial_state: [x, y, z, vx, vy, vz, r, p, y, ...]
    dx = target_pos[0] - initial_state[0]
    if abs(dx) > 0.5:
        guess[2, 0] = -0.1 * np.sign(dx) # Constant pitch

    return guess.flatten()

# --- 3. THE COST FUNCTION ---
def oracle_loss(flat_coeffs, init_state, target_pos):
    controls = coeffs_to_controls(flat_coeffs, duration=2.0, dt=0.05)

    # Reset Sim
    sim = cython_drone_sim.DroneSim()
    sim.set_state(init_state)

    total_cost = 0.0

    for u in controls:
        # Actuator Clipping (Physical constraint)
        u = np.clip(u, -1, 1) # Normalization range often -1 to 1 or 0 to 1 depending on env.
        # In DroneEnv, actions are typically clipped inside step, but here we clip for solver stability.
        # DroneEnv expects thrust 0..1? Let's check.
        # DroneEnv: thrust_force = thrust_cmd * max_thrust.
        # Usually thrust_cmd is expected 0..1 or -1..1.
        # Let's assume 0..1 for thrust, -1..1 for rates.
        # But `coeffs_to_controls` produces arbitrary values.
        # Let's clip to sensible ranges.
        u[0] = np.clip(u[0], 0.0, 1.0) # Thrust
        u[1:] = np.clip(u[1:], -1.0, 1.0) # Rates

        sim.step(u)
        state = sim.get_state()

        # A. Trajectory Tracking
        dist = np.linalg.norm(state[0:3] - target_pos)
        total_cost += dist

        # B. Apply Heuristics
        total_cost += apply_heuristics(state, u, target_pos)

        # C. Crash Penalty
        if state[2] < 0.1: return 1e6

    # D. Regularize High-Order Coeffs (Force Smoothness)
    # We punish T3 (cubic) and T4 strongly to prefer simple arcs
    coeffs = flat_coeffs.reshape(4, 4)
    reg = np.sum(coeffs[:, 2:]**2) * 10.0

    return total_cost + reg

# --- 4. THE SOLVER ---
def solve_perfect_trajectory(init_state, target_pos):
    guess = get_warm_start(init_state, target_pos)

    res = minimize(
        oracle_loss,
        guess,
        args=(init_state, target_pos),
        method='L-BFGS-B',
        bounds=[(-10, 10)]*16,
        options={'maxiter': 50}
    )
    return res.x if res.success else None

# --- Helper: Simulate Trajectory ---
def simulate_trajectory(init_state, coeffs, duration=2.0, dt=0.05):
    """
    Runs the simulation given the coefficients and returns the state trajectory.
    """
    controls = coeffs_to_controls(coeffs, duration, dt)
    sim = cython_drone_sim.DroneSim()
    sim.set_state(init_state)

    states = []
    # Add initial state
    states.append(sim.get_state()[0:3])

    for u in controls:
        # Clip actions as in loss
        u[0] = np.clip(u[0], 0.0, 1.0)
        u[1:] = np.clip(u[1:], -1.0, 1.0)

        sim.step(u)
        states.append(sim.get_state()[0:3]) # Store position

    return np.array(states)


# --- 5. TRAINING LOOP ---
def train_jules():
    model = JulesPolicy()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    vis = Visualizer(output_dir="visualizations")

    print("Starting training loop...")

    # We'll run fewer iterations for the demo to ensure completion
    for i in range(1, 1001):
        # 1. Generate Problem
        s0 = np.random.randn(12)
        s0[2] = abs(s0[2]) + 5.0 # Start above ground
        s0[3:6] *= 0.1 # Low initial velocity
        s0[6:] *= 0.1 # Low initial rotation

        target = np.array([0,0,10]) # Hover target at 10m

        # 2. Get Teacher's Answer (The "Perfect" Coeffs)
        # This is expensive, so in real RL this runs in parallel/offline.
        teacher_coeffs = solve_perfect_trajectory(s0, target)
        if teacher_coeffs is None:
            continue

        # 3. Train Student
        # (Mocking history buffer for example)
        # In a real loop, we would step the environment and build history.
        # Here we just pass a dummy history as per user's simplified loop.
        history = torch.zeros(1, 12, 40)
        state_tensor = torch.tensor(s0).float().unsqueeze(0)
        label = torch.tensor(teacher_coeffs).float().unsqueeze(0)

        pred = model(state_tensor, history)
        loss = loss_fn(pred, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % 10 == 0:
            print(f"Iter {i}, Loss: {loss.item():.4f}")
            vis.log_reward(i, -loss.item()) # MSE as negative reward equivalent

        # Visualization
        if i % 100 == 0:
            print(f"Generating visualization for iter {i}...")

            # Re-simulate Teacher
            teacher_traj = simulate_trajectory(s0, teacher_coeffs)

            # Simulate Student
            student_coeffs = pred.detach().numpy().flatten()
            student_traj = simulate_trajectory(s0, student_coeffs)

            # Pad to same length if necessary
            # They should be same length if duration/dt is constant

            # log_trajectory expects (num_agents, T, 3)
            # We treat student as "trajectory" and teacher as "target"
            vis.log_trajectory(
                i,
                trajectories=student_traj[np.newaxis, :, :],
                targets=teacher_traj[np.newaxis, :, :]
            )

            # Generate GIF
            vis.generate_trajectory_gif()

    # Save Model
    torch.save(model.state_dict(), "jules_model.pth")
    print("Training complete. Model saved.")
    vis.plot_rewards()

if __name__ == "__main__":
    train_jules()
