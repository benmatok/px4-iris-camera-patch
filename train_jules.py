import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from drone_env.drone import DroneEnv
from models.predictive_policy import JulesPredictiveController, Chebyshev
from visualization import Visualizer
from rrt_planner import RRTController
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
        c0 = p0
        c1 = v0
        c2 = 0.5 * a0

        T2 = T*T; T3 = T2*T; T4 = T3*T; T5 = T4*T

        DeltaP = pf - (c0 + c1*T + c2*T2)
        DeltaV = vf - (c1 + 2*c2*T)
        DeltaA = af - (2*c2)

        # Analytical Solution
        c3 = (10*DeltaP - 4*DeltaV*T + 0.5*DeltaA*T2) / T3
        c4 = (-15*DeltaP + 7*DeltaV*T - DeltaA*T2) / T4
        c5 = (6*DeltaP - 3*DeltaV*T + 0.5*DeltaA*T2) / T5

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
        t_out = np.arange(steps) * self.dt

        # 1. Get Start State
        if current_state is None:
            # Cold start (assume on track at t=0)
            pass

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

def generate_data(num_episodes=20, num_agents=50, future_steps=10):
    """
    Generates training data using the RRT Planner (which uses Oracle as heuristic).
    future_steps=10 corresponds to 0.5s horizon.
    """
    env = DroneEnv(num_agents=num_agents, episode_length=100)
    oracle = OracleController(num_agents)
    planner = RRTController(env, oracle, horizon_steps=future_steps, num_samples=50)

    cheb_hist = Chebyshev(30, 3, device='cpu')

    # We need a cheb_future locally just to evaluate the first step action if needed,
    # or we can rely on planner's internal logic. But here we get coefficients.
    cheb_future = Chebyshev(future_steps, 2, device='cpu')

    data_hist = []
    data_aux = []
    data_future = []

    logging.info("Generating Data with RRT Planner...")

    for ep in range(num_episodes):
        env.reset_all_envs()

        # SLOW DOWN TARGETS
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

            # --- RRT PLANNING ---
            t_current = float(step) * 0.05

            # Build current state dict for Oracle/Planner
            current_state = {
                'pos_x': env.data_dictionary['pos_x'],
                'pos_y': env.data_dictionary['pos_y'],
                'pos_z': env.data_dictionary['pos_z'],
                'vel_x': env.data_dictionary['vel_x'],
                'vel_y': env.data_dictionary['vel_y'],
                'vel_z': env.data_dictionary['vel_z'],
                'roll': env.data_dictionary['roll'],
                'pitch': env.data_dictionary['pitch'],
                'yaw': env.data_dictionary['yaw'],
                'masses': env.data_dictionary['masses'],
                'drag_coeffs': env.data_dictionary['drag_coeffs'],
                'thrust_coeffs': env.data_dictionary['thrust_coeffs']
            }

            # Get Best Action Coefficients via RRT
            # Returns (N, 12) tensor
            future_coeffs = planner.plan(current_state, obs, traj_params, t_current)

            # Store
            data_hist.append(hist_coeffs)
            data_aux.append(aux)
            data_future.append(future_coeffs)

            # Derive Immediate Action from Coeffs to step the environment
            # Eval at x=-1 (t=0)
            # T0(-1)=1, T1(-1)=-1, T2(-1)=1
            # Coeffs: (N, 4, 3)
            fc_reshaped = future_coeffs.view(num_agents, 4, 3)
            # Action = c0 - c1 + c2
            current_action = fc_reshaped[:, :, 0] - fc_reshaped[:, :, 1] + fc_reshaped[:, :, 2]

            # Clip
            current_action[:, 0] = torch.clamp(current_action[:, 0], 0.0, 1.0)
            current_action[:, 1:] = torch.clamp(current_action[:, 1:], -10.0, 10.0)

            # Apply
            env.data_dictionary['actions'][:] = current_action.numpy().reshape(-1)

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

            # Logging progress
            if step % 20 == 0:
                # Calculate mean error of chosen action vs Oracle just for sanity?
                # or just log
                pass

    # Concat all
    all_hist = torch.cat(data_hist, dim=0)
    all_aux = torch.cat(data_aux, dim=0)
    all_future = torch.cat(data_future, dim=0)

    logging.info(f"Dataset Size: {len(all_hist)}")
    return DroneDatasetWithAux(all_hist, all_aux, all_future)

def train(epochs=10, batch_size=256, model_path="jules_model.pth"):
    # 1. Generate Data (Using RRT)
    # Increased future_steps to 10 (0.5s)
    dataset = generate_data(num_episodes=20, num_agents=100, future_steps=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Setup Model
    # Explicitly pass future_len=10
    model = JulesPredictiveController(future_len=10)
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

    # Load Model with correct future_len
    model = JulesPredictiveController(future_len=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Setup Env
    num_agents = 10
    env = DroneEnv(num_agents=num_agents, episode_length=100)
    oracle = OracleController(num_agents)
    viz = Visualizer()

    env.reset_all_envs()

    # SLOW DOWN TARGETS
    env.data_dictionary['traj_params'][1, :] *= 0.3
    env.data_dictionary['traj_params'][4, :] *= 0.3
    env.data_dictionary['traj_params'][7, :] *= 0.3

    actual_traj = []
    target_traj = []
    tracker_data = []
    optimal_traj = []

    traj_params = env.data_dictionary['traj_params']

    for step in range(100):
        obs = env.data_dictionary['observations']
        pos_x = env.data_dictionary['pos_x']
        pos_y = env.data_dictionary['pos_y']
        pos_z = env.data_dictionary['pos_z']

        # Prepare Input
        raw_hist = torch.from_numpy(obs[:, :300]).float()
        raw_hist = raw_hist.view(num_agents, 30, 10).permute(0, 2, 1)
        aux = torch.from_numpy(obs[:, 300:308]).float()

        # Predict
        with torch.no_grad():
            raw_hist_flat = torch.from_numpy(obs[:, :300]).float()
            hist_coeffs = model.fit_history(raw_hist_flat)
            preds = model(hist_coeffs, aux)
            actions = model.get_action_for_execution(preds)

        actual_traj.append([pos_x[0], pos_y[0], pos_z[0]])

        vt_x = env.data_dictionary['vt_x'][0]
        vt_y = env.data_dictionary['vt_y'][0]
        vt_z = env.data_dictionary['vt_z'][0]
        target_traj.append([vt_x, vt_y, vt_z])

        tracker_data.append(obs[0, 304:308])

        # Optimal Planning (Oracle)
        t_current = float(step) * 0.05
        current_state = {
            'pos_x': env.data_dictionary['pos_x'],
            'pos_y': env.data_dictionary['pos_y'],
            'pos_z': env.data_dictionary['pos_z'],
            'vel_x': env.data_dictionary['vel_x'],
            'vel_y': env.data_dictionary['vel_y'],
            'vel_z': env.data_dictionary['vel_z']
        }

        # Viz 10 steps
        _, planned_pos = oracle.compute_trajectory(traj_params, t_current, 10, current_state)
        optimal_traj.append(planned_pos[0])

        actions_np = actions.numpy()
        actions_np[:, 0] = np.clip(actions_np[:, 0], 0.0, 1.0)
        actions_np[:, 1:] = np.clip(actions_np[:, 1:], -10.0, 10.0)

        env.data_dictionary['actions'][:] = actions_np.reshape(-1)

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

    actual_traj = np.array(actual_traj)
    target_traj = np.array(target_traj)
    tracker_data = np.array(tracker_data)

    actual_traj = actual_traj[np.newaxis, :, :]
    target_traj = target_traj[np.newaxis, :, :]
    tracker_data = tracker_data[np.newaxis, :, :]

    viz.log_trajectory(0, actual_traj, target_traj, tracker_data, optimal_traj)
    viz.generate_trajectory_gif()

    gif_path = viz.save_episode_gif(0, actual_traj[0], target_traj[0], tracker_data[0], filename_suffix="_eval", optimal_trajectory=optimal_traj)

    if os.path.exists(gif_path):
        os.rename(gif_path, "jules_trajectory.gif")
        logging.info("Renamed evaluation GIF to jules_trajectory.gif")

if __name__ == "__main__":
    train(epochs=15)
    evaluate()
