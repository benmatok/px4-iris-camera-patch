import os
import argparse
import numpy as np
import torch
import logging
import time
import shutil
from tqdm import tqdm
from collections import deque

from drone_env.drone import DroneEnv
from models.ae_policy import DronePolicy
from visualization import Visualizer

# For PPO
from torch.distributions import Normal

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CPUTrainer:
    def __init__(self, env, policy, num_agents, episode_length, batch_size=4096, mini_batch_size=1024, ppo_epochs=4, clip_param=0.2):
        self.env = env
        self.policy = policy
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.clip_param = clip_param
        self.gamma = 0.99
        self.lam = 0.95
        self.value_loss_coeff = 0.5
        self.entropy_coeff = 0.01
        self.max_grad_norm = 1.0 # Gradient clipping

        # Include action_log_std in parameters to optimize
        self.optimizer_policy = torch.optim.Adam(
            list(self.policy.action_head.parameters()) + [self.policy.action_log_std],
            lr=3e-4
        )
        self.optimizer_value = torch.optim.Adam(self.policy.value_head.parameters(), lr=1e-3)
        self.optimizer_encoder = torch.optim.Adam(self.policy.feature_extractor.parameters(), lr=3e-4)

        self.device = torch.device("cpu") # Force CPU

        # Buffers (Pre-allocate)
        self.obs_buffer = torch.zeros((episode_length, num_agents, 608), dtype=torch.float32)
        self.actions_buffer = torch.zeros((episode_length, num_agents, 4), dtype=torch.float32)
        self.logprobs_buffer = torch.zeros((episode_length, num_agents), dtype=torch.float32)
        self.rewards_buffer = torch.zeros((episode_length, num_agents), dtype=torch.float32)
        self.dones_buffer = torch.zeros((episode_length, num_agents), dtype=torch.float32)
        self.values_buffer = torch.zeros((episode_length, num_agents), dtype=torch.float32)

        # New: Buffer for reward components (debug)
        self.reward_components_buffer = torch.zeros((episode_length, num_agents, 8), dtype=torch.float32)

        # Buffers for visualization
        self.target_history_buffer = np.zeros((episode_length, num_agents, 3), dtype=np.float32)
        self.tracker_history_buffer = np.zeros((episode_length, num_agents, 4), dtype=np.float32)


    def collect_rollout(self):
        # Reset Environment
        self.env.reset_all_envs()

        # Get Data Dictionary from Env (NumPy arrays)
        data = self.env.cuda_data_manager.data_dictionary

        # We need to manually manage the step loop
        # The env.step() is designed for WarpDrive CUDA, but we use step_function directly or via helper
        # Actually, WarpDrive's step() just calls the CUDA kernel.
        # We need to call our CPU/Cython step function.

        step_func = self.env.get_step_function()
        step_kwargs = self.env.get_step_function_kwargs()

        # Map kwarg names to actual data arrays
        # Note: data[name] is a numpy array.

        # Initial Observation
        obs_np = data["observations"]
        # NaN check
        if np.isnan(obs_np).any() or np.isinf(obs_np).any():
            print("Warning: Initial observations contain NaN or Inf!")
            obs_np = np.nan_to_num(obs_np)

        obs = torch.from_numpy(obs_np).float()

        total_reward = 0

        # Local refs for speed
        d_obs = data["observations"]
        d_rew = data["rewards"]
        d_done = data["done_flags"]
        d_act = data["actions"]
        d_vt_x = data["vt_x"]
        d_vt_y = data["vt_y"]
        d_vt_z = data["vt_z"]
        d_rew_comp = data["reward_components"] # New

        # Map arguments for step function
        # We construct the args dict once
        args = {}
        for k, v in step_kwargs.items():
            if v in data:
                args[k] = data[v]
            elif k == "num_agents":
                args[k] = self.num_agents
            elif k == "episode_length":
                args[k] = self.episode_length
            else:
                pass

        for t in range(self.episode_length):
            # 1. Policy Forward
            with torch.no_grad():
                # obs is (num_agents, 608)
                action_mean, value = self.policy(obs)

                # Check for NaNs in network output
                if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
                    print(f"NaN/Inf detected in action_mean at step {t}")
                    action_mean = torch.nan_to_num(action_mean)

                # Sample Action - use current std
                std = self.policy.action_log_std.exp()
                dist = Normal(action_mean, std)

                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
                log_prob = dist.log_prob(action).sum(dim=-1)

            # 2. Step Environment
            # Write action to data dict
            d_act[:] = action.numpy().flatten() # Expects flat array? Check step_cpu
            # step_cpu expects "actions" as flat or reshaped inside.
            # In drone.py: actions_reshaped = actions.reshape(num_agents, 4). So flat is fine.

            # Execute Step
            step_func(**args)

            # 3. Store Data
            self.obs_buffer[t] = obs
            self.actions_buffer[t] = action
            self.logprobs_buffer[t] = log_prob
            self.values_buffer[t] = value.squeeze()

            # Check for NaNs in rewards
            if np.isnan(d_rew).any() or np.isinf(d_rew).any():
                print(f"NaN/Inf detected in rewards at step {t}")
                print(f"Rewards Min: {d_rew.min()}, Max: {d_rew.max()}")
                d_rew = np.nan_to_num(d_rew)

            self.rewards_buffer[t] = torch.from_numpy(d_rew).float()
            self.dones_buffer[t] = torch.from_numpy(d_done).float()
            self.reward_components_buffer[t] = torch.from_numpy(d_rew_comp).float() # New

            # Next Obs
            if np.isnan(d_obs).any() or np.isinf(d_obs).any():
                print(f"NaN/Inf detected in next observations from env at step {t}")
                print(f"Obs Min: {d_obs.min()}, Max: {d_obs.max()}")
                d_obs = np.nan_to_num(d_obs)

            obs = torch.from_numpy(d_obs).float()

            # Visualization Data
            self.target_history_buffer[t, :, 0] = d_vt_x
            self.target_history_buffer[t, :, 1] = d_vt_y
            self.target_history_buffer[t, :, 2] = d_vt_z
            self.tracker_history_buffer[t] = d_obs[:, 604:608] # u, v, size, conf

        # Calculate Advantages (GAE)
        with torch.no_grad():
             _, next_value = self.policy(obs)
             next_value = next_value.squeeze()

             advantages = torch.zeros_like(self.rewards_buffer)
             lastgaelam = 0
             for t in reversed(range(self.episode_length)):
                 if t == self.episode_length - 1:
                     nextnonterminal = 1.0 - self.dones_buffer[t] # Simplified. Actually if done, next is 0.
                     nextvalues = next_value
                 else:
                     nextnonterminal = 1.0 - self.dones_buffer[t]
                     nextvalues = self.values_buffer[t+1]

                 delta = self.rewards_buffer[t] + self.gamma * nextvalues * nextnonterminal - self.values_buffer[t]
                 advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

             returns = advantages + self.values_buffer

        return self.obs_buffer, self.actions_buffer, self.logprobs_buffer, returns, advantages

    def update(self, rollouts):
        obs, actions, old_logprobs, returns, advantages = rollouts

        # Check inputs for NaNs
        if torch.isnan(obs).any(): print("NaN in obs buffer")
        if torch.isnan(actions).any(): print("NaN in actions buffer")
        if torch.isnan(old_logprobs).any(): print("NaN in old_logprobs buffer")
        if torch.isnan(returns).any(): print("NaN in returns buffer")
        if torch.isnan(advantages).any(): print("NaN in advantages buffer")

        if torch.isinf(returns).any(): print("Inf in returns buffer")
        if torch.isinf(advantages).any(): print("Inf in advantages buffer")

        # Replace NaNs/Infs
        obs = torch.nan_to_num(obs)
        returns = torch.nan_to_num(returns, nan=0.0, posinf=100.0, neginf=-100.0)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=100.0, neginf=-100.0)

        # Flatten
        # (T, N, ...) -> (T*N, ...)
        obs = obs.reshape(-1, 608)
        actions = actions.reshape(-1, 4)
        old_logprobs = old_logprobs.reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)

        # Normalize Advantages
        adv_std = advantages.std()
        if adv_std < 1e-5:
             # If std is 0 (all same reward), normalization leads to NaN or Inf.
             # Just center it?
             advantages = advantages - advantages.mean()
        else:
             advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        dataset_size = obs.size(0)
        indices = np.arange(dataset_size)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]

                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_logprobs = old_logprobs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]

                # Forward
                action_mean, value = self.policy(mb_obs)

                # Recompute std inside loop to keep graph connected
                std = self.policy.action_log_std.exp()

                # Distribution using current std
                dist = Normal(action_mean, std)
                new_logprobs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # Ratio
                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                # Surrogate Loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value = value.squeeze()
                value_loss = 0.5 * ((value - mb_returns) ** 2).mean()

                # Total Loss
                loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy

                # Update
                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                self.optimizer_encoder.zero_grad()

                loss.backward()

                # Clip Gradients
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.optimizer_policy.step()
                self.optimizer_value.step()
                self.optimizer_encoder.step()

        return policy_loss.item(), value_loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--episode_length", type=int, default=20) # Short episodes for dynamics
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    # Enable Anomaly Detection
    torch.autograd.set_detect_anomaly(True)

    # Environment
    env = DroneEnv(num_agents=args.num_agents, episode_length=args.episode_length, use_cuda=False)

    # Model
    policy = DronePolicy(observation_dim=608, action_dim=4, hidden_dim=256).cpu()

    if args.load:
        if os.path.exists(args.load):
            print(f"Loading checkpoint from {args.load}")
            checkpoint = torch.load(args.load)
            policy.load_state_dict(checkpoint)
        else:
            print(f"Checkpoint {args.load} not found, starting fresh.")

    trainer = CPUTrainer(env, policy, args.num_agents, args.episode_length)
    visualizer = Visualizer()

    print(f"Starting Training: {args.num_agents} Agents, {args.iterations} Iterations")

    start_time = time.time()

    for itr in range(1, args.iterations + 1):
        # Rollout
        obs, _, _, _, _ = trainer.collect_rollout()

        # PPO Update
        try:
            p_loss, v_loss = trainer.update((obs, trainer.actions_buffer, trainer.logprobs_buffer, _, _))
        except ValueError as e:
            print(f"Update failed at itr {itr}: {e}")
            break
        except RuntimeError as e:
             print(f"RuntimeError at itr {itr}: {e}")
             break

        # Logging
        mean_reward = trainer.rewards_buffer.sum(dim=0).mean().item() # Sum over time, mean over agents
        visualizer.log_reward(itr, mean_reward)

        if itr % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {itr} | Reward: {mean_reward:.2f} | P_Loss: {p_loss:.4f} | V_Loss: {v_loss:.4f} | Time: {elapsed:.2f}s")

            # Log Reward Components
            # (episode_length, num_agents, 8) -> sum time -> mean agents
            comp_sums = trainer.reward_components_buffer.sum(dim=0).mean(dim=0)
            # 0:pn, 1:closing, 2:gaze, 3:rate, 4:upright, 5:eff, 6:penalty, 7:bonus
            print(f"   Breakdown -> PN: {comp_sums[0]:.2f}, Close: {comp_sums[1]:.2f}, Gaze: {comp_sums[2]:.2f}")
            print(f"                Rate: {comp_sums[3]:.2f}, Upright: {comp_sums[4]:.2f}, Eff: {comp_sums[5]:.2f}")
            print(f"                Penalty: {comp_sums[6]:.2f}, Bonus: {comp_sums[7]:.2f}")

            # Log current exploration std
            curr_std = trainer.policy.action_log_std.exp().mean().item()
            print(f"                Exploration Std: {curr_std:.4f}")

        # Visualization
        if itr % 50 == 0:
            # Save checkpoint
            torch.save(policy.state_dict(), "latest_checkpoint.pth")

            # Get Trajectory from Env Data (pos_history)
            # pos_history shape: (episode_length, num_agents, 3)
            # Note: We need to get it from the environment's data dictionary,
            # because trainer.obs_buffer stores observations, not raw positions (obs has history but scrambled/local)

            pos_hist = env.cuda_data_manager.data_dictionary["pos_history"]
            # Important: Env resets pos_history at start of episode.
            # But wait, trainer.collect_rollout calls reset_all_envs() at START.
            # So pos_history contains data from the JUST COMPLETED rollout.
            # pos_history is filled up to episode_length.

            # Use data from the Trainer buffers for targets/trackers
            targets = trainer.target_history_buffer
            tracker_data = trainer.tracker_history_buffer

            # Log for graph
            visualizer.log_trajectory(itr, pos_hist, targets, tracker_data)
            visualizer.plot_rewards()
            visualizer.generate_trajectory_gif()

        if itr % 100 == 0:
            # Generate Video of specific episode (first agent)
            pos_hist = env.cuda_data_manager.data_dictionary["pos_history"]
            targets = trainer.target_history_buffer
            tracker_data = trainer.tracker_history_buffer

            # Agent 0
            traj_0 = pos_hist[:, 0, :]
            targ_0 = targets[:, 0, :]
            track_0 = tracker_data[:, 0, :]

            visualizer.save_episode_gif(itr, traj_0, targ_0, track_0, filename_suffix="_best" if mean_reward > 0 else "")


    # Save Final
    torch.save(policy.state_dict(), "final_policy.pth")
    print("Training Complete.")

if __name__ == "__main__":
    main()
