import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

sys.path.append(os.getcwd())
from ghost_dpc.ghost_dpc import PyGhostModel, PyGhostEstimator, PyDPCSolver

class AnalysisController:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.models = [
            {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # Nominal
            {'mass': 1.5, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # Heavy
            {'mass': 0.5, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # Light
            {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 0.7}, # Weak Motors
            {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.3}, # Strong Motors
        ]
        self.estimator = PyGhostEstimator(self.models)
        self.solver = PyDPCSolver()
        self.last_action = {
            'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0
        }

    def control(self, state, measured_accel, target_pos, solver_dt=None):
        if solver_dt is None:
            solver_dt = self.dt

        self.estimator.update(state, self.last_action, measured_accel, self.dt)
        weighted_model = self.estimator.get_weighted_model()

        # Simplified Solver Setup (similar to run_ghost_validation)
        base = weighted_model
        solver_models = []
        solver_weights = []

        # 1. Optimistic
        m1 = base.copy(); m1['wind_x'] = 0.0; m1['wind_y'] = 0.0
        solver_models.append(m1); solver_weights.append(0.2)

        # 2. Wind A
        m2 = base.copy(); m2['wind_x'] = 10.0
        solver_models.append(m2); solver_weights.append(0.2)

        # 3. Wind B
        m3 = base.copy(); m3['wind_x'] = -10.0
        solver_models.append(m3); solver_weights.append(0.2)

        # 4. High Drag
        m4 = base.copy(); m4['drag_coeff'] = base['drag_coeff'] * 1.5
        solver_models.append(m4); solver_weights.append(0.2)

        # 5. Low Drag
        m5 = base.copy(); m5['drag_coeff'] = base['drag_coeff'] * 0.5
        solver_models.append(m5); solver_weights.append(0.2)

        opt_action = self.solver.solve(state, target_pos, self.last_action, solver_models, solver_weights, solver_dt)
        self.last_action = opt_action

        return opt_action, weighted_model

def simulate_prediction(start_state, action, model_params, steps, dt):
    # Simulate a trajectory using the model and constant action
    model = PyGhostModel(
        model_params['mass'],
        model_params['drag_coeff'],
        model_params['thrust_coeff'],
        model_params.get('wind_x', 0.0),
        model_params.get('wind_y', 0.0)
    )

    traj = {'px': [], 'py': [], 'pz': []}
    state = start_state.copy()

    for _ in range(steps):
        state = model.step(state, action, dt)
        traj['px'].append(state['px'])
        traj['py'].append(state['py'])
        traj['pz'].append(state['pz'])

    return traj

def run_analysis_scenario(name, duration_sec=5.0):
    print(f"Running Analysis Scenario: {name}")
    dt = 0.05
    steps = int(duration_sec / dt)
    controller = AnalysisController(dt=dt)

    real_model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0)
    state = {
        'px': 0.0, 'py': 0.0, 'pz': 10.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
    }
    action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}
    target_pos = [0.0, 0.0, 10.0]

    # Special setups
    if name == "Blind Dive":
        real_model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0, wind_y=0.0)
        target_pos = [50.0, 0.0, 0.0]
        state['pz'] = 100.0
        state['px'] = 0.0
    elif name == "Wind Gusts":
        real_model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0, wind_x=0.0)
        target_pos = [0.0, 0.0, 10.0]

    history = {
        't': [],
        'pos_x': [], 'pos_z': [], 'vel_z': [],
        'pred_traj_x': [], 'pred_traj_z': [], # List of lists
        'est_mass': [], 'est_drag': [], 'est_wind_x': []
    }

    # Prediction horizon for analysis (e.g., 1.0s = 20 steps)
    pred_steps = 20

    for i in range(steps):
        t = i * dt

        # Scenario Dynamics
        if name == "Wind Gusts":
            wind = 0.0
            if t >= 2.0 and t < 4.0:
                wind = 8.0
            real_model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0, wind_x=wind)

        # Step Real World
        next_s = real_model.step(state, action, dt)

        # IMU
        ax = (next_s['vx'] - state['vx']) / dt
        ay = (next_s['vy'] - state['vy']) / dt
        az = (next_s['vz'] - state['vz']) / dt

        # Control
        action, weighted_model = controller.control(state, [ax, ay, az], target_pos)

        # Generate Prediction (using the weighted model and the chosen action)
        pred_traj = simulate_prediction(state, action, weighted_model, pred_steps, dt)

        # Store
        history['t'].append(t)
        history['pos_x'].append(state['px'])
        history['pos_z'].append(state['pz'])
        history['vel_z'].append(state['vz'])
        history['pred_traj_x'].append(pred_traj['px'])
        history['pred_traj_z'].append(pred_traj['pz'])

        history['est_mass'].append(weighted_model['mass'])
        history['est_drag'].append(weighted_model['drag_coeff'])
        history['est_wind_x'].append(weighted_model['wind_x'])

        state = next_s

    return history

def analyze_and_plot(history, name):
    # Calculate N-step prediction error
    pred_steps = 20
    errors = []
    times = []

    for i in range(len(history['t']) - pred_steps):
        t = history['t'][i]

        pred_x = history['pred_traj_x'][i][-1]
        pred_z = history['pred_traj_z'][i][-1]

        act_x = history['pos_x'][i + pred_steps]
        act_z = history['pos_z'][i + pred_steps]

        dist = math.sqrt((pred_x - act_x)**2 + (pred_z - act_z)**2)
        errors.append(dist)
        times.append(t)

    avg_error = np.mean(errors) if errors else 0.0
    print(f"Scenario {name}: Avg 1.0s Prediction Error: {avg_error:.4f} m")

    # Plotting Trajectories
    plt.figure(figsize=(18, 12))

    # 1. Trajectory
    plt.subplot(2, 2, 1)
    plt.plot(history['pos_x'], history['pos_z'], 'k-', label='Actual Path', linewidth=2)
    for i in range(0, len(history['t']), 10):
        traj_x = history['pred_traj_x'][i]
        traj_z = history['pred_traj_z'][i]
        curr_x = history['pos_x'][i]
        curr_z = history['pos_z'][i]
        plt.plot([curr_x] + traj_x, [curr_z] + traj_z, 'r-', alpha=0.3)
    plt.title(f"{name}: Actual vs Predicted")
    plt.xlabel("X (m)"); plt.ylabel("Z (m)")
    plt.legend(['Actual', 'Predicted'])
    plt.grid(True)

    # 2. Prediction Error
    plt.subplot(2, 2, 2)
    plt.plot(times, errors, 'b-')
    plt.title(f"{name}: 1.0s Prediction Error")
    plt.xlabel("Time (s)"); plt.ylabel("Error (m)")
    plt.grid(True)

    # 3. Vertical Velocity
    plt.subplot(2, 2, 3)
    plt.plot(history['t'], history['vel_z'], 'r-', label='Vertical Vel (vz)')
    # Plot limit if applicable
    plt.axhline(-12.0, color='k', linestyle='--', label='Limit (-12m/s)')
    plt.title("Vertical Velocity Profile")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity Z (m/s)")
    plt.legend()
    plt.grid(True)

    # 4. Parameter Evolution
    plt.subplot(2, 2, 4)
    plt.plot(history['t'], history['est_mass'], label='Mass')
    plt.plot(history['t'], history['est_drag'], label='Drag')
    plt.plot(history['t'], history['est_wind_x'], label='WindX')
    plt.title("Parameter Estimates")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    filename = f"analysis_pred_{name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    hist1 = run_analysis_scenario("Blind Dive", duration_sec=8.0)
    analyze_and_plot(hist1, "Blind Dive")

    hist2 = run_analysis_scenario("Wind Gusts", duration_sec=6.0)
    analyze_and_plot(hist2, "Wind Gusts")
