import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import logging
import time
from drone_env.drone import DroneEnv
from models.predictive_policy import JulesPredictiveController
from train_jules import LinearPlanner
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

    def sanitize_physics(self):
        """
        Overrides random physics parameters with 'safe' flyable values.
        Ensures Thrust/Weight ratio > 2.0 to guarantee climbing capability.
        Max Thrust = 20.0 * thrust_coeff
        Weight = mass * 9.81
        Sets: Mass=1.0, Tc=1.0 -> Max Thrust=20, Weight=9.8 -> Ratio=2.04
        """
        d = self.env.data_dictionary
        d['masses'][:] = 1.0
        d['thrust_coeffs'][:] = 1.0
        d['drag_coeffs'][:] = 0.1

    def collect_comparison_rollout(self):
        """
        Collects paired trajectories for visualization.
        1. Resets env with known seed.
        2. Runs Oracle (Teacher).
        3. Resets env with SAME seed.
        4. Runs Student.
        Returns: (student_pos, oracle_pos, target_pos) for Agent 0
        """
        # We need to manually handle seeding to ensure identical initial conditions
        # The Environment's reset logic uses np.random.
        # We can seed np.random before reset.
        seed = 42

        # --- 1. Oracle Run ---
        np.random.seed(seed)
        self.env.reset_all_envs()
        self.sanitize_physics()

        data = self.env.cuda_data_manager.data_dictionary
        step_func = self.env.get_step_function()
        step_args = self.step_args_list

        oracle_pos_history = np.zeros((self.episode_length, 3), dtype=np.float32)
        target_pos_history = np.zeros((self.episode_length, 3), dtype=np.float32)

        obs = torch.from_numpy(data["observations"])
        d_act = data["actions"]

        for t in range(self.episode_length):
            # Oracle Action
            current_state = {
                'pos_x': data['pos_x'], 'pos_y': data['pos_y'], 'pos_z': data['pos_z'],
                'vel_x': data['vel_x'], 'vel_y': data['vel_y'], 'vel_z': data['vel_z'],
                'roll': data['roll'], 'pitch': data['pitch'], 'yaw': data['yaw'],
                'masses': data['masses'], 'drag_coeffs': data['drag_coeffs'], 'thrust_coeffs': data['thrust_coeffs']
            }
            target_pos = np.stack([data['vt_x'], data['vt_y'], data['vt_z']], axis=1)
            action = self.oracle.compute_actions(current_state, target_pos)
            action = np.clip(action, -1.0, 1.0)
            d_act[:] = action.flatten()

            step_func(*step_args)

            # Record Agent 0
            oracle_pos_history[t, 0] = data["pos_x"][0]
            oracle_pos_history[t, 1] = data["pos_y"][0]
            oracle_pos_history[t, 2] = data["pos_z"][0]
            target_pos_history[t, 0] = data["vt_x"][0]
            target_pos_history[t, 1] = data["vt_y"][0]
            target_pos_history[t, 2] = data["vt_z"][0]

        # --- 2. Student Run ---
        np.random.seed(seed) # RESET SEED
        self.env.reset_all_envs()
        self.sanitize_physics()
        # Data dict is updated in place, so we reuse references

        student_pos_history = np.zeros((self.episode_length, 3), dtype=np.float32)
        obs = torch.from_numpy(data["observations"])

        for t in range(self.episode_length):
            # Student Action
            with torch.no_grad():
                history = obs[:, :300]
                aux = obs[:, 300:]
                hist_coeffs = self.agent.fit_history(history)
                action_tensor = self.agent(hist_coeffs, aux)
                action = action_tensor.numpy()

            action = np.clip(action, -1.0, 1.0)
            d_act[:] = action.flatten()

            step_func(*step_args)

            student_pos_history[t, 0] = data["pos_x"][0]
            student_pos_history[t, 1] = data["pos_y"][0]
            student_pos_history[t, 2] = data["pos_z"][0]

            obs = torch.from_numpy(data["observations"])

        return student_pos_history, oracle_pos_history, target_pos_history

    def collect_rollout(self, use_student=False):
        """
        Runs an episode.
        If use_student=False (Training), uses Oracle to drive and records data.
        If use_student=True (Validation), uses Agent to drive.
        """
        self.env.reset_all_envs()
        self.sanitize_physics()
        data = self.env.cuda_data_manager.data_dictionary
        step_func = self.env.get_step_function()
        step_args = self.step_args_list

        obs_np = data["observations"]
        obs = torch.from_numpy(obs_np)

        d_act = data["actions"]

        # Track total reward/error for validation
        total_dist_error = 0.0
        total_control_error = 0.0

        for t in range(self.episode_length):
            # 1. Oracle Action (Ground Truth)
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
            oracle_action = self.oracle.compute_actions(current_state, target_pos) # (N, 4)

            # 2. Student Action
            with torch.no_grad():
                history = obs[:, :300]
                aux = obs[:, 300:]
                hist_coeffs = self.agent.fit_history(history)
                student_action_tensor = self.agent(hist_coeffs, aux)
                student_action = student_action_tensor.numpy()

            # Decide who drives
            if use_student:
                action = student_action
            else:
                action = oracle_action

            # Store for Training
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

            # Validation Metrics
            if use_student:
                # 1. Distance Error
                dx = data['pos_x'] - data['vt_x']
                dy = data['pos_y'] - data['vt_y']
                dz = data['pos_z'] - data['vt_z']
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                total_dist_error += dist.mean()

                # 2. Control Deviation (MSE between Student and Oracle)
                # Compare what student DID (action) vs what Oracle WOULD have done (oracle_action)
                # Note: both based on SAME state (pre-step).
                # IMPORTANT: Clip oracle action to [-1, 1] for fair comparison, as training data is clipped.
                oracle_action_clipped = np.clip(oracle_action, -1.0, 1.0)
                diff = action - oracle_action_clipped
                control_mse = np.mean(diff**2)
                total_control_error += control_mse

            # Next Obs
            obs = torch.from_numpy(data["observations"])

        if use_student:
            avg_dist = total_dist_error / self.episode_length
            avg_control = total_control_error / self.episode_length
            return avg_dist, avg_control
        else:
            return self.obs_buffer, self.actions_buffer

    def train_step(self):
        # 1. Collect Data (Teacher Forcing)
        obs_seq, act_seq = self.collect_rollout(use_student=False)
        # obs_seq: (T, N, 308)
        # act_seq: (T, N, 4)

        loss_sum = 0.0
        batches = 0

        # Train on single step prediction: Input Obs[t] -> Target Act[t]

        # Flatten
        obs_flat = obs_seq.reshape(-1, 308)
        act_flat = act_seq.reshape(-1, 4)

        dataset_size = obs_flat.shape[0]
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        for start in range(0, dataset_size, self.batch_size):
            end = start + self.batch_size
            idx = indices[start:end]

            mb_obs = obs_flat[idx]
            mb_act = act_flat[idx] # Ground Truth Action

            history = mb_obs[:, :300]
            aux = mb_obs[:, 300:]

            hist_coeffs = self.agent.fit_history(history)

            # Forward: predicts action directly
            pred_action = self.agent(hist_coeffs, aux)

            loss = self.criterion(pred_action, mb_act)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            batches += 1

        return loss_sum / batches, 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--episode_length", type=int, default=100)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Environment
    env = DroneEnv(num_agents=args.num_agents, episode_length=args.episode_length, use_cuda=False)

    # Agent (Jules) - Future Length 5
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
    print(f"Oracle: LinearPlanner (from train_jules.py)")

    start_time = time.time()
    best_eval_dist = float('inf')

    for itr in range(1, args.iterations + 1):
        # Train Step
        loss_coeff, loss_action = trainer.train_step()

        # Logging
        if itr % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {itr:4d} | Coeff Loss: {loss_coeff:.6f} | Action MSE (Pred vs GT): {loss_action:.6f} | Time: {elapsed:.2f}s")

        # Validation (Student Drive)
        if itr % 50 == 0:
            avg_dist, avg_control_dev = trainer.collect_rollout(use_student=True)
            print(f"   [Validation] Distance to Target: {avg_dist:.4f} m | Control Deviation (Student vs Oracle): {avg_control_dev:.6f}")

            # Save Checkpoint
            if avg_dist < best_eval_dist:
                best_eval_dist = avg_dist
                torch.save(agent.state_dict(), "best_jules.pth")

        # Visualization (Comparison)
        if itr % 100 == 0:
            s_traj, o_traj, t_traj = trainer.collect_comparison_rollout()
            visualizer.save_comparison_gif(itr, s_traj, o_traj, targets=t_traj, filename_suffix="_comp")

    print("Training Complete.")

if __name__ == "__main__":
    main()
