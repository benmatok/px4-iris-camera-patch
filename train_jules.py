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

        # Gains
        self.kp_pos = 2.0
        self.kd_pos = 1.5
        self.kp_att = 10.0 # Proportional gain for attitude rates

    def compute_trajectory(self, traj_params, t_start, steps, current_state=None):
        """
        Computes the sequence of optimal actions for [t_start, t_start + steps*dt].
        traj_params: (10, num_agents)
        current_state: dict with 'pos', 'vel', 'rot' (optional).
                       If provided, the first action corrects error.
                       Subsequent actions assume perfect execution (MPC style) or we project.
        Returns: (num_agents, 4, steps) -> [Thrust, RollRate, PitchRate, YawRate]
        """
        # Time vector
        # t: (steps, 1)
        t_steps = np.arange(steps) * self.dt + t_start

        # Expand params to (10, num_agents, 1)
        params = traj_params[:, :, np.newaxis] # (10, N, 1)
        t = t_steps[np.newaxis, :] # (1, steps)

        # 1. Target Kinematics (Feedforward)
        Ax, Fx, Px = params[0], params[1], params[2]
        Ay, Fy, Py = params[3], params[4], params[5]
        Az, Fz, Pz, Oz = params[6], params[7], params[8], params[9]

        # Phases
        ph_x = Fx * t + Px
        ph_y = Fy * t + Py
        ph_z = Fz * t + Pz

        # Target Pos
        tx = Ax * np.sin(ph_x)
        ty = Ay * np.sin(ph_y)
        tz = Oz + Az * np.sin(ph_z)

        # Target Vel
        tvx = Ax * Fx * np.cos(ph_x)
        tvy = Ay * Fy * np.cos(ph_y)
        tvz = Az * Fz * np.cos(ph_z)

        # Target Accel
        tax = -Ax * Fx**2 * np.sin(ph_x)
        tay = -Ay * Fy**2 * np.sin(ph_y)
        taz = -Az * Fz**2 * np.sin(ph_z)

        # 2. Feedback Correction
        # If current_state is provided, we compute the REQUIRED accel to converge.
        # Note: This computes a SEQUENCE.
        # For t=0, we use actual error.
        # For t>0, we ideally propagate, but for simplicity/robustness in generating "Teacher" data,
        # we can assume the drone is "on track" relative to the lookahead,
        # OR we can just output the feedforward + small correction.
        # BETTER: For t=0, use feedback. For t>0, assumes we close the gap linearly?
        # Actually, let's just use the Feedforward kinematics for t>0 (open loop plan),
        # BUT for t=0 (the action we actually execute in simulation), we add the feedback.
        # This creates a "Receding Horizon" controller.

        # However, the Network is trained to predict the WHOLE sequence.
        # If the sequence at t=1, t=2 doesn't include correction, the network might learn to ignore errors?
        # A simple robust approach: Calculate the trajectory from CurrentPos to TargetPos(t+T)
        # But we want to follow the curve.

        # Let's apply the PD correction to ALL steps, assuming the error persists? No, that implies integral windup.
        # Let's apply PD correction decaying over time?
        # Or simply: The teacher demonstrates the "Ideal Recovery Path".
        # We can simulate the recovery path forward.
        # That's expensive (simulating physics).

        # Approximation:
        # We calculate the Desired Acceleration at each step.
        # For t=0, we have explicit pos/vel.
        # For t>0, we estimate pos/vel assuming we track perfectly from t=0.
        # So for t>0, error = 0.

        # Initialize arrays
        num_agents = traj_params.shape[1]

        # Current State
        if current_state is not None:
             px = current_state['pos_x'][:, np.newaxis] # (N, 1)
             py = current_state['pos_y'][:, np.newaxis]
             pz = current_state['pos_z'][:, np.newaxis]
             vx = current_state['vel_x'][:, np.newaxis]
             vy = current_state['vel_y'][:, np.newaxis]
             vz = current_state['vel_z'][:, np.newaxis]
        else:
             # Assume perfect start (on target) if no state given
             px, py, pz = tx[:, 0:1], ty[:, 0:1], tz[:, 0:1]
             vx, vy, vz = tvx[:, 0:1], tvy[:, 0:1], tvz[:, 0:1]

        # Errors (Broadcast (N, 1) vs (N, Steps)) -> (N, Steps)
        # NOTE: This assumes error is CONSTANT over the window if we don't simulate forward.
        # This is a strong assumption but better than zero feedback.
        # A better assumption is error decays.
        # Let's just use the instantaneous error for the whole window.
        # The network will learn "If error is X, plan to accelerate".

        ex = tx - px
        ey = ty - py
        ez = tz - pz
        evx = tvx - vx
        evy = tvy - vy
        evz = tvz - vz

        # Desired Accel
        ad_x = self.kp_pos * ex + self.kd_pos * evx + tax
        ad_y = self.kp_pos * ey + self.kd_pos * evy + tay
        ad_z = self.kp_pos * ez + self.kd_pos * evz + taz

        # 3. Attitude Control (Differential Flatness)

        # Force Vector
        # Drag comp
        drag = 0.1
        fx = ad_x + drag * vx
        fy = ad_y + drag * vy
        fz = ad_z + drag * vz + self.g

        f_norm = np.sqrt(fx**2 + fy**2 + fz**2)

        # Thrust
        max_thrust = 20.0
        thrust_cmd = f_norm / max_thrust
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)

        # Desired Body Z
        zb_x = fx / f_norm
        zb_y = fy / f_norm
        zb_z = fz / f_norm

        # Desired Yaw (Look along velocity)
        # If velocity is small, keep current yaw or target yaw?
        # Use target velocity for smoothness
        yaw_des = np.arctan2(tvy, tvx)

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
        # R31 = zb_x, R32 = zb_y, R33 = zb_z
        # Using the same inversion as before
        # R_tilt = R_des_yaw.T @ zb

        cy = np.cos(yaw_des); sy = np.sin(yaw_des)
        v1 = -sy * zb_x + cy * zb_y # -sin(roll)
        v0 = cy * zb_x + sy * zb_y # sin(pitch)cos(roll)
        v2 = zb_z # cos(pitch)cos(roll)

        roll_des = -np.arcsin(np.clip(v1, -1.0, 1.0))
        pitch_des = np.arctan2(v0, v2)

        # Rates via Finite Difference / Gain
        # Since we want to correct errors, we should output rates that move us to Desired Attitude.
        # But we don't have current attitude in the inputs easily accessible in this vectorized form
        # without complicating the signature heavily.
        # However, the Network inputs (History) HAS attitude.
        # The Oracle output (Target) should be the IDEAL trajectory.
        # So we should output the Derivative of the Ideal Attitude Trajectory.
        # The *Student* (Network) will see "Current Att != Ideal Att" in its history, and "Target Action = Rate to maintain Ideal".
        # This is strictly not enough for feedback on attitude.
        # BUT: The positional feedback (ad_x, ad_y) changes the Ideal Attitude (roll_des, pitch_des).
        # If we are to the left, ad_y increases -> roll_des increases (bank right).
        # So the `roll_des` computed here *already contains* the corrective term.
        # So `roll_rate = gradient(roll_des)` IS the rate of change of the corrective maneuver.
        # This should be sufficient for the "Teacher" to demonstrate recovery.

        roll_rate = np.gradient(roll_des, self.dt, axis=1)
        pitch_rate = np.gradient(pitch_des, self.dt, axis=1)

        yaw_des_unwrapped = np.unwrap(yaw_des, axis=1)
        yaw_rate = np.gradient(yaw_des_unwrapped, self.dt, axis=1)

        actions = np.stack([thrust_cmd, roll_rate, pitch_rate, yaw_rate], axis=1)
        return actions

    def compute_position_trajectory(self, traj_params, t_start, steps):
        """
        Computes the sequence of optimal POSITIONS for [t_start, t_start + steps*dt].
        traj_params: (10, num_agents)
        Returns: (num_agents, steps, 3) -> [x, y, z]
        """
        t_steps = np.arange(steps) * self.dt + t_start
        params = traj_params[:, :, np.newaxis] # (10, N, 1)
        t = t_steps[np.newaxis, :] # (1, steps)

        Ax, Fx, Px = params[0], params[1], params[2]
        Ay, Fy, Py = params[3], params[4], params[5]
        Az, Fz, Pz, Oz = params[6], params[7], params[8], params[9]

        # Position
        x = Ax * np.sin(Fx * t + Px)
        y = Ay * np.sin(Fy * t + Py)
        z = Oz + Az * np.sin(Fz * t + Pz)

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

            future_actions = oracle.compute_trajectory(traj_params, t_current, future_steps, current_state) # (N, 4, 5)

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

        # Optimal Position (Oracle)
        # We calculate it for the CURRENT time
        t_current = float(step) * 0.05
        opt_pos = oracle.compute_position_trajectory(traj_params, t_current, 1) # (N, 1, 3)
        optimal_traj.append(opt_pos[0, 0, :])

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
    optimal_traj = np.array(optimal_traj)

    # Add dimension for batch (1, T, 3)
    actual_traj = actual_traj[np.newaxis, :, :]
    target_traj = target_traj[np.newaxis, :, :]
    tracker_data = tracker_data[np.newaxis, :, :]
    optimal_traj = optimal_traj[np.newaxis, :, :]

    # Log and Save
    viz.log_trajectory(0, actual_traj, target_traj, tracker_data, optimal_traj)
    viz.generate_trajectory_gif()

    # Also save the specific episode video for clarity
    gif_path = viz.save_episode_gif(0, actual_traj[0], target_traj[0], tracker_data[0], filename_suffix="_eval", optimal_trajectory=optimal_traj[0])

    # Rename to a standard name for the user
    if os.path.exists(gif_path):
        os.rename(gif_path, "jules_trajectory.gif")
        logging.info("Renamed evaluation GIF to jules_trajectory.gif")

if __name__ == "__main__":
    train(epochs=15)
    evaluate()
