import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import random
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ghost_dpc.ghost_dpc import PyGhostModel, PyGhostEstimator, PyDPCSolver
from vision.projection import Projector

class GhostController:
    def __init__(self, dt=0.05):
        self.dt = dt
        # 1. Initialize Estimator with "Uncertainty Lattice"
        # 5 Models: Nominal, Heavy, Light, Strong, Weak
        # Added extra ghosts at 1kg jumps (2.0 - 6.0) to handle heavier drones
        self.models = [
            {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # Nominal
            {'mass': 1.5, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # Heavy
            {'mass': 0.5, 'drag_coeff': 0.1, 'thrust_coeff': 1.0}, # Light
            {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 0.7}, # Weak Motors
            {'mass': 1.0, 'drag_coeff': 0.1, 'thrust_coeff': 1.3}, # Strong Motors
            # Heavy ghosts with scaled thrust_coeff (Tc ~ Mass * 0.98 for 0.5 hover)
            {'mass': 2.0, 'drag_coeff': 0.1, 'thrust_coeff': 2.0},
            {'mass': 2.5, 'drag_coeff': 0.1, 'thrust_coeff': 2.5},
            {'mass': 3.0, 'drag_coeff': 0.1, 'thrust_coeff': 3.0},
            {'mass': 3.5, 'drag_coeff': 0.1, 'thrust_coeff': 3.5},
            {'mass': 4.0, 'drag_coeff': 0.1, 'thrust_coeff': 4.0},
            {'mass': 4.5, 'drag_coeff': 0.1, 'thrust_coeff': 4.5},
            {'mass': 5.0, 'drag_coeff': 0.1, 'thrust_coeff': 5.0},
            {'mass': 6.0, 'drag_coeff': 0.1, 'thrust_coeff': 6.0},
        ]
        self.estimator = PyGhostEstimator(self.models)

        # 2. Solver
        self.solver = PyDPCSolver()

        self.last_action = {
            'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0
        }

        self.history = deque(maxlen=80)

    def control(self, state, measured_accel, measured_alpha, target_pos, solver_dt=None, measured_uv=None):
        if solver_dt is None:
            solver_dt = self.dt

        # 1. Update Estimator
        self.estimator.update(state, self.last_action, measured_accel, self.dt, measured_alpha, measured_uv)

        # 2. Get Beliefs
        probs = self.estimator.get_probabilities()
        weighted_model = self.estimator.get_weighted_model()

        # 3. Create Robust Hypotheses for Solver
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

        # Populate History
        # Construct observed state for controller
        obs = {
            'time': 0.0,
            'roll': state['roll'],
            'pitch': state['pitch'],
            'yaw': state['yaw'],
            'pz': state['pz'],
            'vz': state['vz'],
            'wx': state.get('wx', 0.0),
            'wy': state.get('wy', 0.0),
            'wz': state.get('wz', 0.0),
            'thrust': self.last_action['thrust'],
            'roll_rate': self.last_action['roll_rate'],
            'pitch_rate': self.last_action['pitch_rate'],
            'yaw_rate': self.last_action['yaw_rate'],
            'u': measured_uv[0] if measured_uv else None,
            'v': measured_uv[1] if measured_uv else None
        }
        self.history.append(obs)

        # 4. Solve
        # target_pos is [LatDist, 0, 0] in Blind Dive.
        # But DPC Solver now treats target as [0,0,0] (Tracked Object)
        # However, in Blind Dive, we *are* tracking a virtual target at [LatDist, 0, 0].
        # The 'measured_uv' calculation in 'run_blind_dive_scenario' correctly projects this target.
        # So passing 'history' with this UV will make the solver try to center that UV.
        # This implicitly flies to the target.
        # The 'goal_z' is 0.0? No, we probably want to land or hover above?
        # Standard logic: fly to 2m above target.

        opt_action, _ = self.solver.solve(list(self.history), self.last_action, solver_models, solver_weights, solver_dt, goal_z=2.0)

        self.last_action = opt_action
        return opt_action, probs, weighted_model

def run_blind_dive_scenario(params, scenario_id):
    dt = 0.05
    duration_sec = 15.0
    steps = int(duration_sec / dt)

    # Init Controller
    controller = GhostController(dt=dt)

    # Init Projector for Ground Truth measurements
    # Using 120.0 FOV as requested by user
    projector = Projector(width=640, height=480, fov_deg=120.0, tilt_deg=30.0)

    # Init Simulation (Real World) with randomized parameters
    real_model = PyGhostModel(
        mass=params['mass'],
        drag=params['drag_coeff'],
        thrust_coeff=params['thrust_coeff'],
        wind_x=params['wind_x'],
        tau=params.get('tau', 0.1)
    )

    # Initial State
    start_alt = params['start_alt']
    lateral_dist = params['lateral_dist']

    state = {
        'px': 0.0, 'py': 0.0, 'pz': start_alt,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
    }

    action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

    # Data Logging
    history = {
        't': [], 'alt': [], 'pos_x': [], 'pos_y': [], 'thrust_cmd': [],
        'target_x': [], 'target_y': [], 'target_z': [],
        'tau_est': [], 'tau_true': [],
        'u_meas': [], 'v_meas': [],
        'u_pred': [], 'v_pred': [] # Not filling pred from estimator yet, maybe just log measured for tracking
    }

    target_pos = [lateral_dist, 0.0, 0.0]

    for i in range(steps):
        t = i * dt

        # --- STEP SIMULATION ---
        next_s = real_model.step(state, action, dt)

        # IMU (Kinematic Accel)
        ax = (next_s['vx'] - state['vx']) / dt
        ay = (next_s['vy'] - state['vy']) / dt
        az = (next_s['vz'] - state['vz']) / dt

        # Angular Accel (Alpha)
        # Using current w (next_s) and previous w (state)
        # next_s has wx, wy, wz if updated PyGhostModel returns them.
        alphax = (next_s.get('wx', 0.0) - state.get('wx', 0.0)) / dt
        alphay = (next_s.get('wy', 0.0) - state.get('wy', 0.0)) / dt
        alphaz = (next_s.get('wz', 0.0) - state.get('wz', 0.0)) / dt

        # Measure Screen Pos using Projector
        # world_to_normalized returns (u, v) normalized or None
        uv = projector.world_to_normalized(target_pos[0], target_pos[1], target_pos[2], state)

        u_val, v_val = None, None
        if uv is not None:
            u_val, v_val = uv

        # --- CONTROLLER ---
        # Pass u,v to controller (Estimator)
        action, probs, est_model = controller.control(state, [ax, ay, az], [alphax, alphay, alphaz], target_pos, solver_dt=dt, measured_uv=[u_val, v_val])

        # --- LOGGING ---
        history['t'].append(t)
        history['tau_est'].append(est_model.get('tau', 0.1))
        history['tau_true'].append(real_model.tau)
        history['alt'].append(state['pz'])
        history['u_meas'].append(u_val if u_val else 0.0)
        history['v_meas'].append(v_val if v_val else 0.0)
        history['pos_x'].append(state['px'])
        history['pos_y'].append(state['py'])
        history['thrust_cmd'].append(action['thrust'])
        history['target_x'].append(target_pos[0])
        history['target_y'].append(target_pos[1])
        history['target_z'].append(target_pos[2])

        state = next_s

        # Break if crashed (z < 0) or landed safely
        if state['pz'] < 0.2:
            break

    return history, state

def plot_scenario(hist, params, scenario_id):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(30, 5))

    # Top View
    ax1.plot(hist['pos_x'], hist['pos_y'], 'b-', label='Path')
    ax1.plot(hist['target_x'][0], hist['target_y'][0], 'rx', markersize=10, label='Target')
    ax1.set_title(f"Scenario {scenario_id}: Top View")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.axis('equal')
    ax1.legend()
    ax1.grid(True)

    # Side View
    ax2.plot(hist['pos_x'], hist['alt'], 'b-')
    ax2.plot(hist['target_x'][0], hist['target_z'][0], 'rx', markersize=10, label='Target')
    ax2.set_title(f"Side View (Alt={params['start_alt']:.1f}, Dist={params['lateral_dist']:.1f})")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Altitude (m)")
    ax2.grid(True)

    # Thrust & Alt Time
    ax3.plot(hist['t'], hist['alt'], 'b-', label='Alt')
    ax3_r = ax3.twinx()
    ax3_r.plot(hist['t'], hist['thrust_cmd'], 'r--', label='Thrust')
    ax3.set_title(f"Dynamics (Mass={params['mass']:.1f})")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Altitude (m)", color='b')
    ax3_r.set_ylabel("Thrust Cmd", color='r')
    ax3_r.set_ylim(0, 1)

    # Tau Estimation
    ax4.plot(hist['t'], hist['tau_true'], 'k--', label='True Tau')
    ax4.plot(hist['t'], hist['tau_est'], 'g-', label='Est Tau')
    ax4.set_title("Tau Estimation")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Tau (s)")
    ax4.legend()
    ax4.grid(True)
    ax4.set_ylim(0, 0.5)

    # Screen Pos Tracking
    ax5.plot(hist['t'], hist['u_meas'], 'b-', label='U Meas')
    ax5.plot(hist['t'], hist['v_meas'], 'r-', label='V Meas')
    ax5.set_title("Screen Position (U, V)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Screen Pos")
    ax5.legend()
    ax5.grid(True)
    ax5.set_ylim(-2, 2)

    plt.tight_layout()
    filename = f"validation_blind_dive_scenario_{scenario_id}.png"
    plt.savefig(filename)
    plt.close(fig)

def main():
    np.random.seed(42) # For reproducibility
    scenarios = []

    print(f"{'ID':<3} {'Mass':<6} {'Drag':<6} {'WindX':<6} {'ThrustK':<8} {'HoverR':<6} {'Alt':<5} {'Dist':<5} {'Result':<20}")
    print("-" * 120)

    for i in range(2):
        mass = np.random.uniform(2.0, 6.0)
        drag_coeff = np.random.uniform(0.05, 0.5)
        wind_x = np.random.uniform(-5.0, 5.0)
        start_alt = np.random.uniform(40.0, 150.0)
        lateral_dist = np.random.uniform(20.0, 150.0)

        # Hover Thrust Ratio R = 0.4-0.6
        target_hover_ratio = np.random.uniform(0.4, 0.6)
        thrust_coeff = (mass * 9.81) / (20.0 * target_hover_ratio)

        params = {
            'mass': mass,
            'drag_coeff': drag_coeff,
            'wind_x': wind_x,
            'start_alt': start_alt,
            'lateral_dist': lateral_dist,
            'thrust_coeff': thrust_coeff,
            'tau': np.random.uniform(0.05, 0.2) # Randomize True Tau
        }

        hist, final_state = run_blind_dive_scenario(params, i)
        plot_scenario(hist, params, i)

        # Metrics
        dx = final_state['px'] - lateral_dist
        dy = final_state['py'] - 0.0
        dz = final_state['pz'] - 0.0
        final_dist = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Verify Hover Ratio
        actual_hover_ratio = (mass * 9.81) / (20.0 * thrust_coeff)

        print(f"{i:<3} {mass:.2f}   {drag_coeff:.2f}   {wind_x:.2f}   {thrust_coeff:.2f}     {actual_hover_ratio:.2f}   {start_alt:.1f}   {lateral_dist:.1f}   Dist: {final_dist:.2f}m")

if __name__ == "__main__":
    main()
