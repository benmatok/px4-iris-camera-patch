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
    def __init__(self, env, policy, num_agents, episode_length, batch_size=4096, mini_batch_size=1024, ppo_epochs=4, clip_param=0.2, debug=False):
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
        self.debug = debug

        # Include action_log_std in parameters to optimize
        self.optimizer_policy = torch.optim.Adam(
            list(self.policy.action_head.parameters()) + [self.policy.action_log_std],
            lr=3e-4
        )
        self.optimizer_value = torch.optim.Adam(self.policy.value_head.parameters(), lr=1e-3)
        self.optimizer_encoder = torch.optim.Adam(self.policy.feature_extractor.parameters(), lr=3e-4)

        self.device = torch.device("cpu") # Force CPU

        # Buffers (Pre-allocate)
        self.obs_buffer = torch.zeros((episode_length, num_agents, 308), dtype=torch.float32)
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

        # Pre-resolve step arguments to avoid dict lookup in loop
        # We need to grab the data arrays from env.data_dictionary
        # Step Signature:
        # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,
        # masses, drag_coeffs, thrust_coeffs,
        # wind_x, wind_y, wind_z,
        # target_vx, target_vy, target_vz, target_yaw_rate,
        # vt_x, vt_y, vt_z,
        # traj_params, target_trajectory,
        # pos_history, observations,
        # rewards, reward_components,
        # done_flags, step_counts, actions,
        # action_buffer, delays, rng_states,
        # num_agents, episode_length, env_ids

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


    def collect_rollout(self):
        # Reset Environment
        self.env.reset_all_envs()

        # Get Data Dictionary from Env (NumPy arrays)
        data = self.env.cuda_data_manager.data_dictionary

        # Get Step Function
        step_func = self.env.get_step_function()

        # Initial Observation
        obs_np = data["observations"]

        # NaN check (Conditional)
        if self.debug:
            if np.isnan(obs_np).any() or np.isinf(obs_np).any():
                print("Warning: Initial observations contain NaN or Inf!")
                obs_np = np.nan_to_num(obs_np)

        obs = torch.from_numpy(obs_np) # Already float32

        # Local refs for speed
        d_obs = data["observations"]
        d_rew = data["rewards"]
        d_done = data["done_flags"]
        d_act = data["actions"]
        d_vt_x = data["vt_x"]
        d_vt_y = data["vt_y"]
        d_vt_z = data["vt_z"]
        d_rew_comp = data["reward_components"]

        step_args = self.step_args_list

        for t in range(self.episode_length):
            # 1. Policy Forward
            with torch.no_grad():
                # obs is (num_agents, 308)
                action_mean, value = self.policy(obs)

                # Check for NaNs in network output
                if self.debug:
                    if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
                        print(f"NaN/Inf detected in action_mean at step {t}")
                        action_mean = torch.nan_to_num(action_mean)

                    if torch.isnan(value).any() or torch.isinf(value).any():
                        print(f"NaN/Inf detected in value at step {t}")
                        value = torch.nan_to_num(value)

                # Sample Action - use current std
                std = self.policy.action_log_std.exp()
                dist = Normal(action_mean, std)

                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
                log_prob = dist.log_prob(action).sum(dim=-1)

            # 2. Step Environment
            # Write action to data dict
            d_act[:] = action.numpy().flatten() # Expects flat array

            # Execute Step (Positional Args)
            step_func(*step_args)

            # 3. Store Data
            self.obs_buffer[t] = obs
            self.actions_buffer[t] = action
            self.logprobs_buffer[t] = log_prob
            self.values_buffer[t] = value.squeeze()

            # Check for NaNs in rewards
            if self.debug:
                if np.isnan(d_rew).any() or np.isinf(d_rew).any():
                    print(f"NaN/Inf detected in rewards at step {t}")
                    d_rew = np.nan_to_num(d_rew)

            self.rewards_buffer[t] = torch.from_numpy(d_rew)
            self.dones_buffer[t] = torch.from_numpy(d_done)
            self.reward_components_buffer[t] = torch.from_numpy(d_rew_comp)

            # Next Obs
            if self.debug:
                if np.isnan(d_obs).any() or np.isinf(d_obs).any():
                    print(f"NaN/Inf detected in next observations from env at step {t}")
                    d_obs = np.nan_to_num(d_obs)

            obs = torch.from_numpy(d_obs) # Already float32

            # Visualization Data
            self.target_history_buffer[t, :, 0] = d_vt_x
            self.target_history_buffer[t, :, 1] = d_vt_y
            self.target_history_buffer[t, :, 2] = d_vt_z
            self.tracker_history_buffer[t] = d_obs[:, 304:308] # u, v, size, conf

        # Calculate Advantages (GAE)
        with torch.no_grad():
             _, next_value = self.policy(obs)
             next_value = next_value.squeeze()

             if self.debug:
                 if torch.isnan(next_value).any() or torch.isinf(next_value).any():
                      print(f"NaN/Inf detected in next_value.")
                      next_value = torch.nan_to_num(next_value)

             advantages = torch.zeros_like(self.rewards_buffer)
             lastgaelam = 0
             for t in reversed(range(self.episode_length)):
                 if t == self.episode_length - 1:
                     nextnonterminal = 1.0 - self.dones_buffer[t]
                     nextvalues = next_value
                 else:
                     nextnonterminal = 1.0 - self.dones_buffer[t]
                     nextvalues = self.values_buffer[t+1]

                 term_next = self.gamma * nextvalues * nextnonterminal
                 # Force term_next to 0 where nextnonterminal is 0
                 term_next = torch.where(nextnonterminal == 0, torch.tensor(0.0, device=self.device), term_next)

                 delta = self.rewards_buffer[t] + term_next - self.values_buffer[t]
                 advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

             returns = advantages + self.values_buffer

        return self.obs_buffer, self.actions_buffer, self.logprobs_buffer, returns, advantages

    def update(self, rollouts):
        obs, actions, old_logprobs, returns, advantages = rollouts

        # Check inputs for NaNs
        if self.debug:
            if torch.isnan(obs).any(): print("NaN in obs buffer")
            if torch.isnan(actions).any(): print("NaN in actions buffer")
            if torch.isnan(old_logprobs).any(): print("NaN in old_logprobs buffer")
            if torch.isnan(returns).any(): print("NaN in returns buffer")
            if torch.isnan(advantages).any(): print("NaN in advantages buffer")

            # Replace NaNs/Infs
            obs = torch.nan_to_num(obs)
            returns = torch.nan_to_num(returns, nan=0.0, posinf=100.0, neginf=-100.0)
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=100.0, neginf=-100.0)

        # Flatten
        # (T, N, ...) -> (T*N, ...)
        obs = obs.reshape(-1, 308)
        actions = actions.reshape(-1, 4)
        old_logprobs = old_logprobs.reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)

        # Normalize Advantages
        adv_std = advantages.std()
        if adv_std < 1e-5:
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
    parser.add_argument("--debug", action="store_true", help="Enable anomaly detection and detailed checks")
    args = parser.parse_args()

    # Enable Anomaly Detection
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        print("Debug mode enabled: Anomaly detection on.")

    # Environment
    env = DroneEnv(num_agents=args.num_agents, episode_length=args.episode_length, use_cuda=False)

    # Model
    policy = DronePolicy(observation_dim=308, action_dim=4, hidden_dim=256).cpu()

    if args.load:
        if os.path.exists(args.load):
            print(f"Loading checkpoint from {args.load}")
            checkpoint = torch.load(args.load)
            policy.load_state_dict(checkpoint)
        else:
            print(f"Checkpoint {args.load} not found, starting fresh.")

    trainer = CPUTrainer(env, policy, args.num_agents, args.episode_length, debug=args.debug)
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
            pass

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
