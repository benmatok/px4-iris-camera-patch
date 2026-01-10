import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import logging
import time
from drone_env.drone import DroneEnv
from models.predictive_policy import JulesPredictiveController
from models.oracle import LinearPlanner
from visualization import Visualizer

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SupervisedTrainer:
    def __init__(self, env, agent, oracle, num_agents, episode_length, batch_size=4096, lr=1e-3, debug=False):
        self.env = env
        self.agent = agent
        self.oracle = oracle
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.debug = debug
        self.device = torch.device("cpu")

        # Optimizer
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Buffers (Pre-allocate)
        # We collect full episodes
        self.obs_buffer = torch.zeros((episode_length, num_agents, 308), dtype=torch.float32)
        self.actions_buffer = torch.zeros((episode_length, num_agents, 4), dtype=torch.float32)

        # Visualization buffers
        self.target_history_buffer = np.zeros((episode_length, num_agents, 3), dtype=np.float32)
        self.tracker_history_buffer = np.zeros((episode_length, num_agents, 4), dtype=np.float32)

        # Step args pre-fetch
        d = self.env.data_dictionary
        self.step_args_list = [
            d["pos_x"], d["pos_y"], d["pos_z"],
            d["vel_x"], d["vel_y"], d["vel_z"],
            d["roll"], d["pitch"], d["yaw"],
            d["masses"], d["drag_coeffs"], d["thrust_coeffs"],
            d["wind_x"], d["wind_y"], d["wind_z"],
            d["target_vx"], d["target_vy"], d["target_vz"], d["target_yaw_rate"],
            d["vt_x"], d["vt_y"], d["vt_z"],
            d["traj_params"], d["target_trajectory"],
            d["pos_history"], d["observations"],
            d["rewards"], d["reward_components"],
            d["done_flags"], d["step_counts"], d["actions"],
            d["action_buffer"], d["delays"], d["rng_states"],
            self.num_agents, self.episode_length, d["env_ids"]
        ]

    def collect_rollout(self, use_student=False):
        """
        Runs an episode.
        If use_student=False (Training), uses Oracle to drive and records data.
        If use_student=True (Validation), uses Agent to drive.
        """
        self.env.reset_all_envs()
        data = self.env.cuda_data_manager.data_dictionary
        step_func = self.env.get_step_function()
        step_args = self.step_args_list

        obs_np = data["observations"]
        obs = torch.from_numpy(obs_np)

        d_act = data["actions"]

        # Track total reward/error for validation
        total_error = 0.0

        for t in range(self.episode_length):
            # Decide Action
            if use_student:
                # Agent Drive
                with torch.no_grad():
                    # 1. Split Obs
                    history = obs[:, :300]
                    aux = obs[:, 300:]
                    # 2. Fit History
                    hist_coeffs = self.agent.fit_history(history)
                    # 3. Predict Future Coeffs
                    pred_coeffs = self.agent(hist_coeffs, aux)
                    # 4. Extract Immediate Action
                    action_tensor = self.agent.get_action_for_execution(pred_coeffs)
                    action = action_tensor.numpy()
            else:
                # Oracle Drive
                # Construct state dict for Oracle
                current_state = {
                    'pos_x': data['pos_x'],
                    'pos_y': data['pos_y'],
                    'pos_z': data['pos_z'],
                    'vel_x': data['vel_x'],
                    'vel_y': data['vel_y'],
                    'vel_z': data['vel_z'],
                    'roll': data['roll'],
                    'pitch': data['pitch'],
                    'yaw': data['yaw'],
                    'masses': data['masses'],
                    'drag_coeffs': data['drag_coeffs'],
                    'thrust_coeffs': data['thrust_coeffs']
                }
                target_pos = np.stack([data['vt_x'], data['vt_y'], data['vt_z']], axis=1)
                action = self.oracle.compute_actions(current_state, target_pos)

            # Store for Training (even if Student drove, we might want to analyze, but usually we train on Oracle drive)
            # Clip Action
            action = np.clip(action, -1.0, 1.0)
            d_act[:] = action.flatten()

            # Execute
            step_func(*step_args)

            # Record
            self.obs_buffer[t] = obs
            self.actions_buffer[t] = torch.from_numpy(action)

            # Visualization Data
            self.target_history_buffer[t, :, 0] = data["vt_x"]
            self.target_history_buffer[t, :, 1] = data["vt_y"]
            self.target_history_buffer[t, :, 2] = data["vt_z"]
            self.tracker_history_buffer[t] = data["observations"][:, 304:308]

            # Validation Metric: Distance to target
            if use_student:
                dx = data['pos_x'] - data['vt_x']
                dy = data['pos_y'] - data['vt_y']
                dz = data['pos_z'] - data['vt_z']
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                total_error += dist.mean()

            # Next Obs
            obs = torch.from_numpy(data["observations"])

        if use_student:
            return total_error / self.episode_length
        else:
            return self.obs_buffer, self.actions_buffer

    def train_step(self):
        # 1. Collect Data (Teacher Forcing)
        obs_seq, act_seq = self.collect_rollout(use_student=False)
        # obs_seq: (T, N, 308)
        # act_seq: (T, N, 4)

        # 2. Prepare Training Batches
        # We need samples (History, FutureActions)
        # History comes from Obs. FutureActions comes from Actions.
        # Future Horizon
        future_len = self.agent.future_len
        valid_steps = self.episode_length - future_len

        if valid_steps <= 0:
            print("Warning: Episode length too short for future horizon.")
            return 0.0

        # Flatten Time and Agents?
        # We iterate through valid time steps t
        # Input: obs_seq[t]
        # Target: act_seq[t : t+future] -> (Future, N, 4) -> Permute -> (N, 4, Future)

        # To support mini-batching efficiently, let's construct big tensors
        # Indices: [0, ..., valid_steps-1]

        # We can shuffle agents and time?
        # Total samples = valid_steps * num_agents
        # Let's accumulate inputs and targets lists then stack

        # Optimization: Process all 't' at once if memory allows, or loop 't' and accumulate gradients

        loss_sum = 0.0
        batches = 0

        # We'll shuffle time indices to break correlation
        t_indices = np.arange(valid_steps)
        np.random.shuffle(t_indices)

        # For simplicity, we train on one time-slice at a time (Batch size = num_agents = 200)
        # If num_agents is small, this might be noisy, but for 200 it's okay.
        # Actually, let's accumulate a bit.

        for t in t_indices:
            # Inputs
            current_obs = obs_seq[t] # (N, 308)
            history = current_obs[:, :300]
            aux = current_obs[:, 300:]

            # Targets
            future_actions = act_seq[t : t+future_len] # (F, N, 4)
            # Permute to (N, 4, F)
            future_actions = future_actions.permute(1, 2, 0)

            # Fit Targets to Coeffs
            # Note: Chebyshev fit expects (Batch, Channels, Points)
            # Output: (N, 20)
            target_coeffs = self.agent.fit_future(future_actions)

            # Fit Inputs (History)
            hist_coeffs = self.agent.fit_history(history) # (N, 40)

            # Forward
            pred_coeffs = self.agent(hist_coeffs, aux) # (N, 20)

            # Loss
            loss = self.criterion(pred_coeffs, target_coeffs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            batches += 1

        return loss_sum / batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=20000) # Longer run
    parser.add_argument("--episode_length", type=int, default=100) # Increased for better trajectory sampling
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Environment
    env = DroneEnv(num_agents=args.num_agents, episode_length=args.episode_length, use_cuda=False)

    # Agent (Jules)
    # History 30, Future 5 (0.25s)
    agent = JulesPredictiveController(history_len=30, future_len=5, action_dim=4)

    # Oracle
    oracle = LinearPlanner(num_agents=args.num_agents)

    if args.load:
        if os.path.exists(args.load):
            print(f"Loading checkpoint from {args.load}")
            checkpoint = torch.load(args.load)
            agent.load_state_dict(checkpoint)
        else:
            print(f"Checkpoint {args.load} not found, starting fresh.")

    trainer = SupervisedTrainer(env, agent, oracle, args.num_agents, args.episode_length, debug=args.debug)
    visualizer = Visualizer()

    print(f"Starting Supervised Training: {args.num_agents} Agents, {args.iterations} Iterations")
    print(f"Teacher: LinearPlanner, Student: JulesPredictiveController")

    start_time = time.time()
    best_eval_error = float('inf')

    for itr in range(1, args.iterations + 1):
        # Train Step (Teacher Forcing)
        loss = trainer.train_step()

        # Logging
        if itr % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {itr} | Loss: {loss:.6f} | Time: {elapsed:.2f}s")

        # Validation (Student Drive)
        if itr % 50 == 0:
            eval_error = trainer.collect_rollout(use_student=True)
            print(f"   [Validation] Student Avg Dist Error: {eval_error:.4f} m")

            # Save Checkpoint
            torch.save(agent.state_dict(), "latest_jules.pth")
            if eval_error < best_eval_error:
                best_eval_error = eval_error
                torch.save(agent.state_dict(), "best_jules.pth")
                # Generate GIF for best
                pos_hist = env.cuda_data_manager.data_dictionary["pos_history"]
                targets = trainer.target_history_buffer
                tracker = trainer.tracker_history_buffer
                visualizer.save_episode_gif(itr, pos_hist[:, 0, :], targets[:, 0, :], tracker[:, 0, :], filename_suffix="_best")

        # Periodic Visualization of current behavior
        if itr % 500 == 0:
            pass

    # Save Final
    torch.save(agent.state_dict(), "final_jules.pth")
    print("Training Complete.")

if __name__ == "__main__":
    main()
