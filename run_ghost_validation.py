import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.append(os.getcwd())
from ghost_dpc.ghost_dpc import PyGhostModel, PyGhostEstimator, PyDPCSolver

class GhostController:
    def __init__(self, dt=0.05):
        self.dt = dt
        # 1. Initialize Estimator with "Uncertainty Lattice"
        # 5 Models: Nominal, Heavy, Light, Strong, Weak
        self.models = [
            {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # Nominal
            {'mass': 1.5, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # Heavy
            {'mass': 0.5, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # Light
            {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 0.7}, # Weak Motors
            {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.3}, # Strong Motors
        ]
        self.estimator = PyGhostEstimator(self.models)

        # 2. Solver
        self.solver = PyDPCSolver()

        self.last_action = {
            'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0
        }

    def control(self, state, measured_accel, target_pos, solver_dt=None):
        if solver_dt is None:
            solver_dt = self.dt

        # 1. Update Estimator
        self.estimator.update(state, self.last_action, measured_accel, self.dt)

        # 2. Get Beliefs
        probs = self.estimator.get_probabilities()
        weighted_model = self.estimator.get_weighted_model()

        # 3. Create Robust Hypotheses for Solver
        # Task 3.1 says "3 parallel lateral wind hypotheses".
        # We assume the weighted model parameters, but inject wind variants.
        # W0 = (0,0), WL = (0, 5), WR = (0, -5)?
        # Or better: Use the weighted model as base, and fork it.

        # Base Model
        base = weighted_model

        solver_models = []
        solver_weights = []

        # Hypothesis 1: Optimistic (Weighted Avg, No Wind)
        m1 = base.copy()
        m1['wind_x'] = 0.0
        m1['wind_y'] = 0.0
        solver_models.append(m1)
        solver_weights.append(0.2)

        # Hypothesis 2: Headwind/Crosswind A
        m2 = base.copy()
        m2['wind_x'] = 10.0 # Unmodeled wind (Widened to +/- 10)
        solver_models.append(m2)
        solver_weights.append(0.2)

        # Hypothesis 3: Headwind/Crosswind B
        m3 = base.copy()
        m3['wind_x'] = -10.0
        solver_models.append(m3)
        solver_weights.append(0.2)

        # Hypothesis 4: High Drag (Distance Ghost Short)
        m4 = base.copy()
        m4['drag_coeff'] = base['drag_coeff'] * 1.5
        solver_models.append(m4)
        solver_weights.append(0.2)

        # Hypothesis 5: Low Drag (Distance Ghost Long)
        m5 = base.copy()
        m5['drag_coeff'] = base['drag_coeff'] * 0.5
        solver_models.append(m5)
        solver_weights.append(0.2)

        # 4. Solve
        opt_action = self.solver.solve(state, target_pos, self.last_action, solver_models, solver_weights, solver_dt)

        self.last_action = opt_action
        return opt_action, probs, weighted_model

def run_scenario(name, duration_sec=5.0):
    print(f"Running Scenario: {name}")
    dt = 0.05
    steps = int(duration_sec / dt)

    # Init Controller
    controller = GhostController(dt=dt)

    # Init Simulation (Real World)
    real_model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0)

    state = {
        'px': 0.0, 'py': 0.0, 'pz': 10.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
    }
    action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

    # Data Logging
    history = {
        't': [], 'mass_est': [], 'mass_true': [],
        'mass_raw_est': [],
        'wind_true_x': [], 'thrust_coeff_true': [],
        'thrust_cmd': [], 'alt': [],
        'probs': [],
        'pos_x': [], 'pos_y': [], 'target_x': [], 'target_y': [],
        'obs_mass': [], 'obs_drag': []
    }

    target_pos = [0.0, 0.0, 10.0]

    for i in range(steps):
        t = i * dt

        # --- SCENARIO LOGIC ---
        if name == "Payload Drop":
            # Drop mass at t=2.0
            if t >= 2.0:
                real_model = PyGhostModel(mass=0.5, drag=0.1, thrust_coeff=1.0)
            target_pos = [0.0, 0.0, 10.0] # Hold position

        elif name == "Dying Battery":
            # Linear decay of thrust coeff from 1.0 to 0.7 starting t=0
            # Decay over 10s. But we run 5s? Task says 10s.
            # Let's run longer for this one.
            tc = 1.0 - 0.3 * (t / 10.0)
            if tc < 0.7: tc = 0.7
            real_model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=tc)
            target_pos = [0.0, 0.0, 10.0]

        elif name == "Blind Dive":
            # Target 45 deg down. Unmodeled Crosswind 10m/s.
            # Start High (50m). Target at (50, 0, 10). Safety First.
            # Crosswind Y = 10.
            real_model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0, wind_y=10.0)
            target_pos = [50.0, 0.0, 10.0]
            # Override Initial State in run_scenario?
            # run_scenario inits at (0,0,10). We need to change that.
            # We'll handle it by teleporting if t==0.
            if t == 0:
                state['pz'] = 50.0

        elif name == "Wind Gusts":
            # 0-2s: 0 wind. 2-4s: 8.0 wind_x. 4-6s: 0 wind.
            wind = 0.0
            if t >= 2.0 and t < 4.0:
                wind = 8.0
            real_model = PyGhostModel(mass=1.0, drag=0.1, thrust_coeff=1.0, wind_x=wind)
            target_pos = [0.0, 0.0, 10.0]

        elif name == "Heavy Configuration":
            # Constant Mass 1.5kg
            real_model = PyGhostModel(mass=1.5, drag=0.1, thrust_coeff=1.0)
            target_pos = [0.0, 0.0, 10.0]

        elif name == "Unmodeled Drag":
            # Constant Drag 0.5
            real_model = PyGhostModel(mass=1.0, drag=0.5, thrust_coeff=1.0)
            target_pos = [10.0, 0.0, 10.0] # Move to make drag apparent

        # --- STEP SIMULATION ---
        next_s = real_model.step(state, action, dt)

        # IMU (Kinematic Accel)
        ax = (next_s['vx'] - state['vx']) / dt
        ay = (next_s['vy'] - state['vy']) / dt
        az = (next_s['vz'] - state['vz']) / dt

        # --- CONTROLLER ---
        # Optimize Blind Dive by using a coarser time step for the solver (Lookahead 0.1 * 20 = 2.0s)
        s_dt = dt
        if name == "Blind Dive":
            s_dt = 0.1

        action, probs, est_model = controller.control(state, [ax, ay, az], target_pos, solver_dt=s_dt)

        # --- LOGGING ---
        history['t'].append(t)
        history['mass_est'].append(est_model['mass'])
        history['mass_true'].append(real_model.mass)
        history['wind_true_x'].append(real_model.wind_x)
        history['thrust_coeff_true'].append(real_model.thrust_coeff)

        # Get Obs Scores and Raw Est from estimator history
        # Since update() appends to history, the last element is current
        est_hist = controller.estimator.get_history()
        raw_est = est_hist['raw_estimates'][-1]
        obs_sc = est_hist['observability_scores'][-1]

        history['mass_raw_est'].append(raw_est['mass'])
        history['obs_mass'].append(obs_sc['mass'])
        history['obs_drag'].append(obs_sc['drag_coeff'])

        history['thrust_cmd'].append(action['thrust'])
        history['alt'].append(state['pz'])
        history['probs'].append(probs)
        history['pos_x'].append(state['px'])
        history['pos_y'].append(state['py'])
        history['target_x'].append(target_pos[0])
        history['target_y'].append(target_pos[1])

        state = next_s

    return history

def plot_payload_drop(hist):
    plt.figure(figsize=(10, 5))
    plt.plot(hist['t'], hist['mass_true'], 'k--', label='True Mass')
    plt.plot(hist['t'], hist['mass_est'], 'r-', label='Estimated Mass')
    plt.title("Scenario A: Payload Drop (Mass Estimation)")
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (kg)")
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_payload_drop.png")
    print("Saved validation_payload_drop.png")

def plot_dying_battery(hist):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)', color='tab:blue')
    ax1.plot(hist['t'], hist['alt'], color='tab:blue', label='Altitude')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(8, 12)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Throttle Cmd', color='tab:red')
    ax2.plot(hist['t'], hist['thrust_cmd'], color='tab:red', linestyle='--', label='Throttle')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 1)

    plt.title("Scenario B: Dying Battery (Throttle Compensation)")
    fig.tight_layout()
    plt.savefig("validation_dying_battery.png")
    print("Saved validation_dying_battery.png")

def plot_blind_dive(hist):
    plt.figure(figsize=(8, 8))
    plt.plot(hist['pos_x'], hist['pos_y'], 'b-', label='Drone Path')
    plt.plot(hist['target_x'][0], hist['target_y'][0], 'rx', label='Target') # Stationary target?
    # In Blind Dive, target is fixed at (10, 0, 0).
    plt.title("Scenario C: Blind Dive (Top View - Wind Compensation)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("validation_blind_dive.png")
    print("Saved validation_blind_dive.png")

def plot_wind_gusts(hist):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position X (m)', color='tab:blue')
    ax1.plot(hist['t'], hist['pos_x'], color='tab:blue', label='Pos X')
    ax1.plot(hist['t'], hist['target_x'], 'r--', label='Target X')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Wind X (m/s)', color='gray')
    ax2.plot(hist['t'], hist['wind_true_x'], color='gray', linestyle='--', label='True Wind')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')

    plt.title("Scenario D: Wind Gusts (Position Holding)")
    fig.tight_layout()
    plt.savefig("validation_wind_gusts.png")
    print("Saved validation_wind_gusts.png")

def plot_heavy_conf(hist):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mass (kg)', color='k')
    ax1.plot(hist['t'], hist['mass_true'], 'k--', label='True Mass')
    ax1.plot(hist['t'], hist['mass_est'], 'r-', label='Stable Est')
    ax1.plot(hist['t'], hist['mass_raw_est'], 'g:', label='Raw Est')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.legend(loc='center left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Observability Score', color='b')
    ax2.fill_between(hist['t'], 0, hist['obs_mass'], color='b', alpha=0.1, label='Obs Score')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim(0, 1.1)

    plt.title("Scenario E: Heavy Configuration (Observability Gating)")
    fig.tight_layout()
    plt.savefig("validation_heavy_conf.png")
    print("Saved validation_heavy_conf.png")

def plot_unmodeled_drag(hist):
    plt.figure(figsize=(10, 5))
    plt.plot(hist['t'], hist['pos_x'], 'b-', label='Pos X')
    plt.plot(hist['t'], hist['target_x'], 'r--', label='Target X')
    plt.title("Scenario F: Unmodeled Drag (Position Tracking)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position X (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_unmodeled_drag.png")
    print("Saved validation_unmodeled_drag.png")

def main():
    # Test A: Payload Drop
    hist_a = run_scenario("Payload Drop", duration_sec=5.0)
    plot_payload_drop(hist_a)

    # Test B: Dying Battery
    # Needs longer duration to see ramp
    hist_b = run_scenario("Dying Battery", duration_sec=10.0)
    plot_dying_battery(hist_b)

    # Test C: Blind Dive
    hist_c = run_scenario("Blind Dive", duration_sec=10.0)
    plot_blind_dive(hist_c)

    # Analysis Blind Dive
    final_px = hist_c['pos_x'][-1]
    final_py = hist_c['pos_y'][-1]
    final_pz = hist_c['alt'][-1]
    tx = hist_c['target_x'][-1]
    ty = hist_c['target_y'][-1]
    tz = 0.0 # Target Z is 0.0 in Blind Dive
    dist = np.sqrt((final_px-tx)**2 + (final_py-ty)**2 + (final_pz-tz)**2)
    min_z = np.min(hist_c['alt'])
    max_z = np.max(hist_c['alt'])
    print(f"Blind Dive Final Pos: ({final_px:.2f}, {final_py:.2f}, {final_pz:.2f})")
    print(f"Blind Dive Alt Range: [{min_z:.2f}, {max_z:.2f}]")
    print(f"Blind Dive Final Distance: {dist:.2f}m")

    # Test D: Wind Gusts
    hist_d = run_scenario("Wind Gusts", duration_sec=6.0)
    plot_wind_gusts(hist_d)

    # Test E: Heavy Configuration
    hist_e = run_scenario("Heavy Configuration", duration_sec=5.0)
    plot_heavy_conf(hist_e)

    # Test F: Unmodeled Drag
    hist_f = run_scenario("Unmodeled Drag", duration_sec=5.0)
    plot_unmodeled_drag(hist_f)

    # Analysis
    mass_err = np.mean(np.abs(np.array(hist_e['mass_est']) - np.array(hist_e['mass_true'])))
    print(f"Mean Mass Error (Heavy Conf): {mass_err:.4f}")

    # Dashboard Elements (Ghost Wars)
    # Plot probability evolution for Test A
    plt.figure(figsize=(10, 5))
    probs = np.array(hist_a['probs'])
    # Models: Nominal(0), Heavy(1), Light(2), Weak(3), Strong(4)
    # Wait, in GhostController I defined:
    # 0: Nominal, 1: Heavy, 2: Light, 3: Weak, 4: Strong
    plt.plot(hist_a['t'], probs[:, 0], label='Nominal')
    plt.plot(hist_a['t'], probs[:, 1], label='Heavy')
    plt.plot(hist_a['t'], probs[:, 2], label='Light')
    plt.title("Ghost Wars: Probability Evolution (Payload Drop)")
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_ghost_wars.png")
    print("Saved validation_ghost_wars.png")

if __name__ == "__main__":
    main()
