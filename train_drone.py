import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import logging
import time
from tqdm import tqdm

from drone_env.drone import DroneEnv
from models.ae_policy import DronePolicy
from train_jules import LinearPlanner
from visualization import Visualizer

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SupervisedTrainer:
    def __init__(self, env, policy, oracle, num_agents, episode_length, learning_rate=3e-4, debug=False):
        self.env = env
        self.policy = policy
        self.oracle = oracle
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.debug = debug
        self.dt = 0.05

        # Optimizer (Adam for simplicity, training only the policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Loss Function
        self.loss_fn = nn.MSELoss()

        # Pre-resolve step arguments to avoid dict lookup in loop
        d = self.env.data_dictionary
        self.step_args_list = [
            d["pos_x"], d["pos_y"], d["pos_z"],
            d["vel_x"], d["vel_y"], d["vel_z"],
            d["roll"], d["pitch"], d["yaw"],
            d["masses"], d["drag_coeffs"], d["thrust_coeffs"],
            d["target_vx"], d["target_vy"], d["target_vz"], d["target_yaw_rate"],
            d["vt_x"], d["vt_y"], d["vt_z"],
            d["traj_params"], d["target_trajectory"],
            d["pos_history"], d["observations"],
            d["rewards"], d["reward_components"],
            d["done_flags"], d["step_counts"], d["actions"],
            self.num_agents, self.episode_length, d["env_ids"]
        ]
        self.step_func = self.env.get_step_function()

    def step_env(self):
        self.step_func(*self.step_args_list)

    def train_episode(self):
        self.env.reset_all_envs()

        total_loss = 0.0

        # Data access
        d = self.env.data_dictionary

        for t in range(self.episode_length):
            # 1. Get State for Oracle
            obs_np = d["observations"]
            if self.debug:
                 obs_np = np.nan_to_num(obs_np)

            obs_torch = torch.from_numpy(obs_np).float()

            # Construct State Dict for LinearPlanner (needs explicit keys)
            current_state = {
                'pos_x': d['pos_x'],
                'pos_y': d['pos_y'],
                'pos_z': d['pos_z'],
                'vel_x': d['vel_x'],
                'vel_y': d['vel_y'],
                'vel_z': d['vel_z'],
                'roll': d['roll'],
                'pitch': d['pitch'],
                'yaw': d['yaw'],
                'masses': d['masses'],
                'drag_coeffs': d['drag_coeffs'],
                'thrust_coeffs': d['thrust_coeffs']
            }
            target_pos = np.stack([d['vt_x'], d['vt_y'], d['vt_z']], axis=1)

            # 2. Run Oracle (Expert)
            # Returns (N, 4) actions directly
            oracle_actions_np = self.oracle.compute_actions(current_state, target_pos)
            oracle_actions = torch.from_numpy(oracle_actions_np).float()

            # 3. Run Student (Policy)
            student_logits, _ = self.policy(obs_torch)
            pred_actions = student_logits

            # 4. Compute Loss
            target_actions = oracle_actions.detach()

            loss = self.loss_fn(pred_actions, target_actions)

            # 5. Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # 6. Step Environment (Student Drives)
            action_to_step = pred_actions.detach().numpy()
            d['actions'][:] = action_to_step.flatten()

            self.step_env()

        return total_loss / self.episode_length

    def validate_episode(self, visualizer=None, iteration=0):
        """
        Runs a full episode with the Student driving, no training.
        Metrics: Final Distance to Target.
        Visualization: Store trajectories.
        """
        self.env.reset_all_envs()

        d = self.env.data_dictionary

        # Buffers for visualization
        pos_history = []
        target_history = []
        tracker_history = []

        # To avoid overhead, we only compute oracle plan if we are visualizing
        compute_oracle_viz = (visualizer is not None) and (iteration % 100 == 0)

        distances = []

        with torch.no_grad():
            for t in range(self.episode_length):
                obs_np = d["observations"]
                obs_torch = torch.from_numpy(obs_np).float()

                # Student Action
                pred_actions, _ = self.policy(obs_torch)
                action_to_step = pred_actions.numpy()

                # Step
                d['actions'][:] = action_to_step.flatten()
                self.step_env()

                # Record State
                pos = np.stack([d['pos_x'], d['pos_y'], d['pos_z']], axis=1).copy() # (N, 3)
                vt = np.stack([d['vt_x'], d['vt_y'], d['vt_z']], axis=1).copy() # (N, 3)

                # Distance
                dist = np.linalg.norm(pos - vt, axis=1)

                # Store
                pos_history.append(pos)
                target_history.append(vt)
                # Tracker: u, v, size, conf
                track = obs_np[:, 304:308].copy()
                tracker_history.append(track)

        # Final Distance (at last step)
        final_dist = dist.mean() # Mean over agents

        # Visualization
        if visualizer is not None and iteration % 100 == 0:
            # Stack: (T, N, 3) -> (N, T, 3)
            traj = np.stack(pos_history, axis=1)
            targ = np.stack(target_history, axis=1)
            track = np.stack(tracker_history, axis=1)

            # Log to visualizer (Agent 0)
            visualizer.log_trajectory(iteration, traj, targets=targ, tracker_data=track)

            # Save GIF
            visualizer.save_episode_gif(iteration, traj[0], targets=targ[0], tracker_data=track[0])

        return final_dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--episode_length", type=int, default=100) # User asked for "each step... run oracle". Longer episodes allow convergence.
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Environment
    # We use CPU backend
    env = DroneEnv(num_agents=args.num_agents, episode_length=args.episode_length, use_cuda=False)

    # Oracle
    oracle = LinearPlanner(num_agents=args.num_agents)

    # Student Policy
    policy = DronePolicy(observation_dim=308, action_dim=4, hidden_dim=256).cpu()

    if args.load:
        if os.path.exists(args.load):
            logging.info(f"Loading checkpoint from {args.load}")
            checkpoint = torch.load(args.load)
            policy.load_state_dict(checkpoint)
        else:
            logging.warning(f"Checkpoint {args.load} not found. Starting fresh.")

    # Trainer
    trainer = SupervisedTrainer(env, policy, oracle, args.num_agents, args.episode_length, debug=args.debug)

    # Visualizer
    visualizer = Visualizer()

    logging.info(f"Starting Supervised Training (LinearPlanner): {args.num_agents} Agents, {args.iterations} Iterations")

    start_time = time.time()

    for itr in range(1, args.iterations + 1):
        # Train
        loss = trainer.train_episode()

        # Validate (every 10 iters)
        if itr % 10 == 0:
            val_dist = trainer.validate_episode(visualizer, itr)

            elapsed = time.time() - start_time
            logging.info(f"Iter {itr} | Loss: {loss:.4f} | Val Dist: {val_dist:.4f} m | Time: {elapsed:.2f}s")

            visualizer.log_reward(itr, -val_dist) # Log negative distance as 'reward' for plot

        # Save
        if itr % 50 == 0:
            torch.save(policy.state_dict(), "latest_jules.pth")

    torch.save(policy.state_dict(), "final_jules.pth")
    logging.info("Training Complete.")

if __name__ == "__main__":
    main()
