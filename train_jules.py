import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from drone_env.drone import DroneEnv
from models.predictive_policy import JulesPredictiveController, Chebyshev
from visualization import Visualizer
import os

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OracleController:
    """
    Computes optimal controls (Thrust, RollRate, PitchRate, YawRate)
    using a Feedback Controller (PD + Feedforward) to track the Lissajous trajectory.
    """
    def __init__(self, num_agents, dt=0.05):
        self.num_agents = num_agents
        self.dt = dt
        self.g = 9.81

        # Planning Horizon for Min-Jerk
        self.planning_horizon = 2.0 # seconds to converge

    def solve_min_jerk(self, p0, v0, a0, pf, vf, af, T):
        """
        Solves Minimum Jerk Trajectory (Quintic Spline).
        Returns coefficients c0..c5 for P(t) = c0 + c1*t + ... + c5*t^5
        """
        # Shapes: (N, 1) or (N,).
        # We assume vectorized over N.

        # Constraints matrix inversion can be precomputed analytically:
        # P(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
        # P(0)=p0, V(0)=v0, A(0)=a0
        # P(T)=pf, V(T)=vf, A(T)=af

        c0 = p0
        c1 = v0
        c2 = 0.5 * a0

        T2 = T*T; T3 = T2*T; T4 = T3*T; T5 = T4*T

        # System for c3, c4, c5
        # [ 3T^2  4T^3  5T^4 ] [c3]   [ pf - (c0 + c1T + c2T^2) ]
        # [ 6T    12T^2 20T^3] [c4] = [ vf - (c1 + 2c2T)        ]
        # [ 10    20T   60T^2] [c5]   [ af - (2c2)              ]

        # Let DeltaP = pf - (c0 + c1*T + c2*T2)
        # Let DeltaV = vf - (c1 + 2*c2*T)
        # Let DeltaA = af - (2*c2)

        DeltaP = pf - (c0 + c1*T + c2*T2)
        DeltaV = vf - (c1 + 2*c2*T)
        DeltaA = af - (2*c2)

        # Analytical Solution
        # c3 = (10*DeltaP - 4*DeltaV*T + 0.5*DeltaA*T2) / T3  (Wait, verify)
        # Standard QuinticCoeffs:
        c3 = (10*DeltaP - 4*DeltaV*T + 0.5*DeltaA*T2) / T3
        c4 = (-15*DeltaP + 7*DeltaV*T - DeltaA*T2) / T4
        c5 = (6*DeltaP - 3*DeltaV*T + 0.5*DeltaA*T2) / T5
        # Wait, the 0.5 factor on DeltaA might be different.
        # Correct form:
        # c3 = (10(pf-p0) - 6v0*T - 4vf*T - 1.5a0*T2 + 0.5af*T2) / T^3? No.
        # Using the Delta formulation is safer.
        # Matrix inverse of [[T^3, T^4, T^5], [3T^2, 4T^3, 5T^4], [6T, 12T^2, 20T^3]] at t=T is solved as:
        # P(T) = pf => c3 T3 + c4 T4 + c5 T5 = DeltaP
        # V(T) = vf => 3c3 T2 + 4c4 T3 + 5c5 T4 = DeltaV
        # A(T) = af => 6c3 T + 12c4 T2 + 20c5 T3 = DeltaA

        # Solution:
        # c3 = (10 DeltaP - 4 DeltaV T + 0.5 DeltaA T^2) / T^3
        # c4 = (-15 DeltaP + 7 DeltaV T - 1 DeltaA T^2) / T^4
        # c5 = (6 DeltaP - 3 DeltaV T + 0.5 DeltaA T^2) / T^5

        return c0, c1, c2, c3, c4, c5

    def eval_quintic(self, coeffs, t):
        """ Evaluates P, V, A at time t (vectorized) """
        c0, c1, c2, c3, c4, c5 = coeffs
        t2 = t*t; t3 = t2*t; t4 = t3*t; t5 = t4*t

        p = c0 + c1*t + c2*t2 + c3*t3 + c4*t4 + c5*t5
        v = c1 + 2*c2*t + 3*c3*t2 + 4*c4*t3 + 5*c5*t4
        a = 2*c2 + 6*c3*t + 12*c4*t2 + 20*c5*t3
        return p, v, a

    def compute_trajectory(self, traj_params, t_start, steps, current_state=None):
        """
        Computes optimal Receding Horizon Plan.
        1. Determine Target State at t_end = t_start + planning_horizon (2.0s).
        2. Plan Min-Jerk trajectory from current_state to Target State.
        3. Sample this plan for 'steps' (0.5s).
        Returns:
            actions: (num_agents, 4, steps)
            planned_pos: (num_agents, steps, 3) # For Viz
        """
        # Output Time steps (0.5s)
        t_out = np.arange(steps) * self.dt # 0 to 0.25 (wait, steps=5 -> 0, 0.05, 0.1, 0.15, 0.2)
        # We want to output for the requested steps relative to t_start.
        # But MinJerk is parameterized by tau in [0, T_plan].
        # So we evaluate at tau = 0, dt, 2dt...

        # 1. Get Start State
        if current_state is None:
            # Cold start (assume on track at t=0)
            # We need to calc pos/vel/acc at t_start
            pass # We'll handle this by calculating target at t_start

        # Helper to get target state at specific time
        def get_target_state(time_scalar):
            # time_scalar: shape (1, 1) or scalar
            # Returns P, V, A shape (N, 1)
            t_steps = time_scalar / self.dt

            # Expand params
            params = traj_params[:, :, np.newaxis]
            Ax, Fx, Px = params[0], params[1], params[2]
            Ay, Fy, Py = params[3], params[4], params[5]
            Az, Fz, Pz, Oz = params[6], params[7], params[8], params[9]

            ph_x = Fx * t_steps + Px
            ph_y = Fy * t_steps + Py
            ph_z = Fz * t_steps + Pz

            freq_scale = 1.0 / self.dt

            # Pos
            tx = Ax * np.sin(ph_x)
            ty = Ay * np.sin(ph_y)
            tz = Oz + Az * np.sin(ph_z)

            # Vel
            tvx = Ax * Fx * freq_scale * np.cos(ph_x)
            tvy = Ay * Fy * freq_scale * np.cos(ph_y)
            tvz = Az * Fz * freq_scale * np.cos(ph_z)

            # Acc
            tax = -Ax * (Fx * freq_scale)**2 * np.sin(ph_x)
            tay = -Ay * (Fy * freq_scale)**2 * np.sin(ph_y)
            taz = -Az * (Fz * freq_scale)**2 * np.sin(ph_z)

            return (tx, ty, tz), (tvx, tvy, tvz), (tax, tay, taz)

        # Start State (P0, V0, A0)
        if current_state is not None:
            p0x = current_state['pos_x'][:, np.newaxis]
            p0y = current_state['pos_y'][:, np.newaxis]
            p0z = current_state['pos_z'][:, np.newaxis]
            v0x = current_state['vel_x'][:, np.newaxis]
            v0y = current_state['vel_y'][:, np.newaxis]
            v0z = current_state['vel_z'][:, np.newaxis]
            # Accel is not in state, assume 0 or prev? Assume 0 for start of plan
            a0x = np.zeros_like(p0x)
            a0y = np.zeros_like(p0y)
            a0z = np.zeros_like(p0z)
        else:
            (p0x, p0y, p0z), (v0x, v0y, v0z), (a0x, a0y, a0z) = get_target_state(t_start)

        # End State (Pf, Vf, Af) at t_start + Horizon
        t_end = t_start + self.planning_horizon
        (pfx, pfy, pfz), (vfx, vfy, vfz), (afx, afy, afz) = get_target_state(t_end)

        # 2. Compute Min Jerk Coeffs
        cx = self.solve_min_jerk(p0x, v0x, a0x, pfx, vfx, afx, self.planning_horizon)
        cy = self.solve_min_jerk(p0y, v0y, a0y, pfy, vfy, afy, self.planning_horizon)
        cz = self.solve_min_jerk(p0z, v0z, a0z, pfz, vfz, afz, self.planning_horizon)

        # 3. Evaluate for output steps
        # t_eval: (1, steps)
        t_eval = t_out[np.newaxis, :]

        px, vx, ax = self.eval_quintic(cx, t_eval)
        py, vy, ay = self.eval_quintic(cy, t_eval)
        pz, vz, az = self.eval_quintic(cz, t_eval)

        # 4. Differential Flatness for Controls

        # Force Vector
        # Drag comp
        drag = 0.1
        fx = ax + drag * vx
        fy = ay + drag * vy
        fz = az + drag * vz + self.g

        f_norm = np.sqrt(fx**2 + fy**2 + fz**2)

        # Thrust
        max_thrust = 20.0
        thrust_cmd = f_norm / max_thrust
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)

        # Thrust
        max_thrust = 20.0
        thrust_cmd = f_norm / max_thrust
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)

        # Desired Body Z
        zb_x = fx / f_norm
        zb_y = fy / f_norm
        zb_z = fz / f_norm

        # Desired Yaw (Look along MinJerk velocity)
        # Use planned velocity for yaw
        yaw_des = np.arctan2(vy, vx)

        # R = [xb, yb, zb] construction
        xc_des_x = np.cos(yaw_des)
        xc_des_y = np.sin(yaw_des)

        yb_x = -zb_z * xc_des_y
        yb_y = zb_z * xc_des_x
        yb_z = zb_x * xc_des_y - zb_y * xc_des_x

        yb_norm = np.sqrt(yb_x**2 + yb_y**2 + yb_z**2)
        yb_norm = np.maximum(yb_norm, 1e-6)

        yb_x /= yb_norm; yb_y /= yb_norm; yb_z /= yb_norm

        xb_x = yb_y * zb_z - yb_z * zb_y
        xb_y = yb_z * zb_x - yb_x * zb_z
        xb_z = yb_x * zb_y - yb_y * zb_x

        # Extract Roll/Pitch
        cy = np.cos(yaw_des); sy = np.sin(yaw_des)
        v1 = -sy * zb_x + cy * zb_y # -sin(roll)
        v0 = cy * zb_x + sy * zb_y # sin(pitch)cos(roll)
        v2 = zb_z # cos(pitch)cos(roll)

        roll_des = -np.arcsin(np.clip(v1, -1.0, 1.0))
        pitch_des = np.arctan2(v0, v2)

        # Rates via Finite Difference on the Planned Trajectory
        roll_rate = np.gradient(roll_des, self.dt, axis=1)
        pitch_rate = np.gradient(pitch_des, self.dt, axis=1)

        yaw_des_unwrapped = np.unwrap(yaw_des, axis=1)
        yaw_rate = np.gradient(yaw_des_unwrapped, self.dt, axis=1)

        actions = np.stack([thrust_cmd, roll_rate, pitch_rate, yaw_rate], axis=1)

        # Planned Position for Viz: (N, steps, 3)
        planned_pos = np.stack([px, py, pz], axis=2)

        return actions, planned_pos

    def compute_position_trajectory(self, traj_params, t_start, steps):
        """
        Computes the sequence of optimal POSITIONS for [t_start, t_start + steps*dt].
        traj_params: (10, num_agents)
        Returns: (num_agents, steps, 3) -> [x, y, z]
        """
        t_steps = np.arange(steps) * self.dt + t_start
        params = traj_params[:, :, np.newaxis] # (10, N, 1)
        t = t_steps[np.newaxis, :] # (1, steps)

        # Convert to steps
        t_in_steps = t / self.dt

        Ax, Fx, Px = params[0], params[1], params[2]
        Ay, Fy, Py = params[3], params[4], params[5]
        Az, Fz, Pz, Oz = params[6], params[7], params[8], params[9]

        # Position
        x = Ax * np.sin(Fx * t_in_steps + Px)
        y = Ay * np.sin(Fy * t_in_steps + Py)
        z = Oz + Az * np.sin(Fz * t_in_steps + Pz)

        # (N, 3, Steps)
        pos = np.stack([x, y, z], axis=2) # (N, Steps, 3)
        return pos


class DroneDatasetWithAux(Dataset):
    def __init__(self, history_coeffs, aux_state, future_action_coeffs):
        self.history_coeffs = history_coeffs
        self.aux_state = aux_state
        self.future_action_coeffs = future_action_coeffs

    def __len__(self):
        return len(self.history_coeffs)

    def __getitem__(self, idx):
        return self.history_coeffs[idx], self.aux_state[idx], self.future_action_coeffs[idx]

def generate_data(num_episodes=20, num_agents=50, future_steps=5):
    """
    Generates training data using the Oracle.
    """
    env = DroneEnv(num_agents=num_agents, episode_length=100)
    oracle = OracleController(num_agents)
    cheb_hist = Chebyshev(30, 3, device='cpu')
    cheb_future = Chebyshev(future_steps, 2, device='cpu')

    data_hist = []
    data_aux = []
    data_future = []

    logging.info("Generating Data...")

    for ep in range(num_episodes):
        env.reset_all_envs()

        # SLOW DOWN TARGETS
        # Scale frequencies (Fx, Fy, Fz at indices 1, 4, 7)
        env.data_dictionary['traj_params'][1, :] *= 0.3
        env.data_dictionary['traj_params'][4, :] *= 0.3
        env.data_dictionary['traj_params'][7, :] *= 0.3

        for step in range(100):
            obs = env.data_dictionary['observations']
            traj_params = env.data_dictionary['traj_params']

            # Extract history (N, 300)
            raw_hist = torch.from_numpy(obs[:, :300]).float()
            raw_hist = raw_hist.view(num_agents, 30, 10).permute(0, 2, 1) # (N, 10, 30)

            # Fit History Coeffs
            hist_coeffs = cheb_hist.fit(raw_hist) # (N, 10, 4)
            hist_coeffs = hist_coeffs.reshape(num_agents, -1) # (N, 40)

            # Extract Aux (N, 8)
            aux = torch.from_numpy(obs[:, 300:308]).float()

            # Get Oracle Future Actions
            t_current = float(step) * 0.05

            # Build current state dict for Oracle Feedback
            current_state = {
                'pos_x': env.data_dictionary['pos_x'],
                'pos_y': env.data_dictionary['pos_y'],
                'pos_z': env.data_dictionary['pos_z'],
                'vel_x': env.data_dictionary['vel_x'],
                'vel_y': env.data_dictionary['vel_y'],
                'vel_z': env.data_dictionary['vel_z']
            }

            future_actions, _ = oracle.compute_trajectory(traj_params, t_current, future_steps, current_state) # (N, 4, 5)

            # Fit Future Coeffs
            future_actions_torch = torch.from_numpy(future_actions).float()
            future_coeffs = cheb_future.fit(future_actions_torch) # (N, 4, 3)
            future_coeffs = future_coeffs.reshape(num_agents, -1) # (N, 12)

            # Store
            data_hist.append(hist_coeffs)
            data_aux.append(aux)
            data_future.append(future_coeffs)

            # Step Env with Oracle Action
            current_action = future_actions[:, :, 0] # (N, 4) at t=0

            # DEBUG
            if current_action.size != env.data_dictionary['actions'].size:
                 logging.error(f"Shape Mismatch! FutureActions: {future_actions.shape}")
                 logging.error(f"CurrentAction: {current_action.shape}, Size: {current_action.size}")
                 logging.error(f"Target Buffer: {env.data_dictionary['actions'].shape}, Size: {env.data_dictionary['actions'].size}")
                 logging.error(f"Traj Params: {traj_params.shape}")

            env.data_dictionary['actions'][:] = current_action.reshape(-1)

            # Resolve args for step_function
            step_kwargs = env.get_step_function_kwargs()
            step_args = {}
            for k, v in step_kwargs.items():
                if v in env.data_dictionary:
                    step_args[k] = env.data_dictionary[v]
                elif k == "num_agents":
                    step_args[k] = env.num_agents
                elif k == "episode_length":
                    step_args[k] = env.episode_length

            env.step_function(**step_args)

    # Concat all
    all_hist = torch.cat(data_hist, dim=0)
    all_aux = torch.cat(data_aux, dim=0)
    all_future = torch.cat(data_future, dim=0)

    logging.info(f"Dataset Size: {len(all_hist)}")
    return DroneDatasetWithAux(all_hist, all_aux, all_future)

def train(epochs=10, batch_size=256, model_path="jules_model.pth"):
    # 1. Generate Data
    dataset = generate_data(num_episodes=20, num_agents=100, future_steps=5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Setup Model
    model = JulesPredictiveController()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    logging.info("Starting Training...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_hist, batch_aux, batch_target in dataloader:
            optimizer.zero_grad()
            preds = model(batch_hist, batch_aux)
            loss = criterion(preds, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    return model

def evaluate(model_path="jules_model.pth"):
    logging.info("Starting Evaluation...")

    # Load Model
    model = JulesPredictiveController()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Setup Env
    num_agents = 10 # Just a few for eval
    env = DroneEnv(num_agents=num_agents, episode_length=100)
    oracle = OracleController(num_agents)
    viz = Visualizer()

    env.reset_all_envs()

    # SLOW DOWN TARGETS (Consistency with training)
    env.data_dictionary['traj_params'][1, :] *= 0.3
    env.data_dictionary['traj_params'][4, :] *= 0.3
    env.data_dictionary['traj_params'][7, :] *= 0.3

    # Data buffers for viz
    # We will log the first agent
    actual_traj = []
    target_traj = [] # From env
    tracker_data = []
    optimal_traj = [] # Calculated by Oracle (Position)

    # Pre-calculate optimal full trajectory for agent 0
    traj_params = env.data_dictionary['traj_params']

    # Run Episode
    for step in range(100):
        obs = env.data_dictionary['observations']
        pos_x = env.data_dictionary['pos_x']
        pos_y = env.data_dictionary['pos_y']
        pos_z = env.data_dictionary['pos_z']

        # Prepare Input
        raw_hist = torch.from_numpy(obs[:, :300]).float()
        raw_hist = raw_hist.view(num_agents, 30, 10).permute(0, 2, 1) # (N, 10, 30)
        aux = torch.from_numpy(obs[:, 300:308]).float()

        # Predict
        with torch.no_grad():
            # Fit coeffs first
            # raw_hist is (N, 10, 30), fit_history expects flattened (N, 300)?
            # No, fit_history implementation:
            # x = history.view(batch_size, 30, 10).permute(0, 2, 1)
            # So it expects (Batch, 300) flattened.
            # But here raw_hist is already reshaped to (N, 10, 30)!

            # We need to pass flattened history to fit_history
            # raw_hist_flat = obs[:, :300]
            raw_hist_flat = torch.from_numpy(obs[:, :300]).float()

            hist_coeffs = model.fit_history(raw_hist_flat)
            preds = model(hist_coeffs, aux)
            actions = model.get_action_for_execution(preds) # (N, 4)

        # Store Data for Viz (Agent 0)
        actual_traj.append([pos_x[0], pos_y[0], pos_z[0]])

        # Target (Virtual Target)
        vt_x = env.data_dictionary['vt_x'][0]
        vt_y = env.data_dictionary['vt_y'][0]
        vt_z = env.data_dictionary['vt_z'][0]
        target_traj.append([vt_x, vt_y, vt_z])

        # Tracker
        tracker_data.append(obs[0, 304:308]) # u, v, size, conf

        # Optimal Planning (Oracle)
        t_current = float(step) * 0.05

        # Current State for Planning
        current_state = {
            'pos_x': env.data_dictionary['pos_x'],
            'pos_y': env.data_dictionary['pos_y'],
            'pos_z': env.data_dictionary['pos_z'],
            'vel_x': env.data_dictionary['vel_x'],
            'vel_y': env.data_dictionary['vel_y'],
            'vel_z': env.data_dictionary['vel_z']
        }

        # Plan Min-Jerk trajectory (2.0s plan, but we visualize it)
        # We sample it for 10 steps (0.5s) or more for viz?
        # Let's visualize the 0.5s plan that matches the action window
        viz_steps = 10
        _, planned_pos = oracle.compute_trajectory(traj_params, t_current, viz_steps, current_state)
        # planned_pos: (N, 10, 3)

        optimal_traj.append(planned_pos[0]) # Store the whole plan array for this step

        # Execute Action
        # Clip actions to prevent explosion
        actions_np = actions.numpy()
        actions_np[:, 0] = np.clip(actions_np[:, 0], 0.0, 1.0) # Thrust
        actions_np[:, 1:] = np.clip(actions_np[:, 1:], -10.0, 10.0) # Rates

        env.data_dictionary['actions'][:] = actions_np.reshape(-1)

        # Resolve args for step_function
        step_kwargs = env.get_step_function_kwargs()
        step_args = {}
        for k, v in step_kwargs.items():
            if v in env.data_dictionary:
                step_args[k] = env.data_dictionary[v]
            elif k == "num_agents":
                step_args[k] = env.num_agents
            elif k == "episode_length":
                step_args[k] = env.episode_length

        env.step_function(**step_args)

    # Process Viz Data
    actual_traj = np.array(actual_traj) # (100, 3)
    target_traj = np.array(target_traj)
    tracker_data = np.array(tracker_data)
    # optimal_traj is now list of (10, 3) arrays. Keep it as list?
    # Visualization expects (N, T, 3) or similar.
    # But now we have "Plan at each step".
    # Visualizer needs update. We pass the list of plans.

    # Add dimension for batch (1, T, 3)
    actual_traj = actual_traj[np.newaxis, :, :]
    target_traj = target_traj[np.newaxis, :, :]
    tracker_data = tracker_data[np.newaxis, :, :]
    # optimal_traj is just passed as list of arrays

    # Log and Save
    viz.log_trajectory(0, actual_traj, target_traj, tracker_data, optimal_traj)
    viz.generate_trajectory_gif()

    # Also save the specific episode video for clarity
    # optimal_traj is list of (10, 3) arrays. Pass directly.
    gif_path = viz.save_episode_gif(0, actual_traj[0], target_traj[0], tracker_data[0], filename_suffix="_eval", optimal_trajectory=optimal_traj)

    # Rename to a standard name for the user
    if os.path.exists(gif_path):
        os.rename(gif_path, "jules_trajectory.gif")
        logging.info("Renamed evaluation GIF to jules_trajectory.gif")

if __name__ == "__main__":
    train(epochs=15)
    evaluate()
