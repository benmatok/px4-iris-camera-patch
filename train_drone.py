import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import logging
import time
from tqdm import tqdm
from torch.distributions import Normal

from drone_env.drone import DroneEnv
from models.ae_policy import DronePolicy, KFACOptimizer
from visualization import Visualizer

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LinearPlanner:
    """
    Plans a linear path to the target and outputs control actions (Thrust, Rates)
    to track it. Uses an Inverse Dynamics approach assuming constant velocity cruise.
    """
    def __init__(self, num_agents, dt=0.05):
        self.num_agents = num_agents
        self.dt = dt
        self.g = 9.81
        self.cruise_speed = 10.0 # m/s

    def compute_actions(self, current_state, target_pos):
        # Current State
        px = current_state['pos_x']
        py = current_state['pos_y']
        pz = current_state['pos_z']
        vx = current_state['vel_x']
        vy = current_state['vel_y']
        vz = current_state['vel_z']
        roll = current_state['roll']
        pitch = current_state['pitch']
        yaw = current_state['yaw']

        # Params
        mass = current_state['masses']
        drag = current_state['drag_coeffs']
        thrust_coeff = current_state['thrust_coeffs']
        max_thrust_force = 20.0 * thrust_coeff

        # Target Vector
        tx = target_pos[:, 0]
        ty = target_pos[:, 1]
        tz = target_pos[:, 2]

        dx = tx - px
        dy = ty - py
        dz = tz - pz
        dist_xy = np.sqrt(dx**2 + dy**2) + 1e-6

        # Elevation Angle Check
        # User said: "if we are above the object by at least 10 degrees"
        rel_h = pz - tz
        elevation_rad = np.arctan2(rel_h, dist_xy)
        threshold_rad = np.deg2rad(10.0)

        # Virtual Target Logic
        # If elevation < 10 deg, aim higher.
        target_z_eff = tz.copy()
        mask_low = elevation_rad < threshold_rad

        # For low agents, set target Z higher
        target_angle_rad = np.deg2rad(15.0)
        req_h = dist_xy * np.tan(target_angle_rad)

        target_z_eff[mask_low] = tz[mask_low] + req_h[mask_low]

        # Recalculate Delta with effective target
        dz_eff = target_z_eff - pz
        dist_eff = np.sqrt(dx**2 + dy**2 + dz_eff**2) + 1e-6

        # Desired Velocity (Linear Cruise)
        # Scale speed by distance? If close, slow down.
        speed_ref = np.minimum(self.cruise_speed, dist_eff * 1.0)

        vx_des = (dx / dist_eff) * speed_ref
        vy_des = (dy / dist_eff) * speed_ref
        vz_des = (dz_eff / dist_eff) * speed_ref

        # Velocity Error
        evx = vx_des - vx
        evy = vy_des - vy
        evz = vz_des - vz

        # PID for Acceleration Command
        Kp = 2.0
        ax_cmd = Kp * evx
        ay_cmd = Kp * evy
        az_cmd = Kp * evz

        # Inverse Dynamics to get Thrust Vector
        Fx_req = mass * ax_cmd + drag * vx
        Fy_req = mass * ay_cmd + drag * vy
        Fz_req = mass * az_cmd + drag * vz + mass * self.g

        # Compute Thrust Magnitude
        F_total = np.sqrt(Fx_req**2 + Fy_req**2 + Fz_req**2) + 1e-6
        thrust_cmd = F_total / max_thrust_force
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)

        # Compute Desired Attitude (Z-axis alignment)
        zbx = Fx_req / F_total
        zby = Fy_req / F_total
        zbz = Fz_req / F_total

        # Yaw Alignment: Point nose at target (xy plane)
        yaw_des = np.arctan2(dy, dx)

        # Better:
        # xb_temp = [cos(yaw), sin(yaw), 0]
        # yb = cross(zb, xb_temp)
        # xb = cross(yb, zb)

        xb_temp_x = np.cos(yaw_des)
        xb_temp_y = np.sin(yaw_des)
        xb_temp_z = np.zeros_like(yaw_des)

        # yb = cross(zb, xb_temp)
        yb_x = zby * xb_temp_z - zbz * xb_temp_y
        yb_y = zbz * xb_temp_x - zbx * xb_temp_z
        yb_z = zbx * xb_temp_y - zby * xb_temp_x

        norm_yb = np.sqrt(yb_x**2 + yb_y**2 + yb_z**2) + 1e-6
        yb_x /= norm_yb
        yb_y /= norm_yb
        yb_z /= norm_yb

        # xb = cross(yb, zb)
        xb_x = yb_y * zbz - yb_z * zby
        xb_y = yb_z * zbx - yb_x * zbz
        xb_z = yb_x * zby - yb_y * zbx

        # Extract Roll/Pitch from R = [xb, yb, zb]
        pitch_des = -np.arcsin(np.clip(xb_z, -1.0, 1.0))
        roll_des = np.arctan2(yb_z, zbz)

        # Rate P-Controller
        Kp_att = 5.0

        # Shortest angular distance
        roll_err = roll_des - roll
        roll_err = (roll_err + np.pi) % (2 * np.pi) - np.pi

        pitch_err = pitch_des - pitch
        pitch_err = (pitch_err + np.pi) % (2 * np.pi) - np.pi

        yaw_err = yaw_des - yaw
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi

        roll_rate_cmd = Kp_att * roll_err
        pitch_rate_cmd = Kp_att * pitch_err
        yaw_rate_cmd = Kp_att * yaw_err

        actions = np.zeros((self.num_agents, 4))
        actions[:, 0] = thrust_cmd
        actions[:, 1] = np.clip(roll_rate_cmd, -10.0, 10.0)
        actions[:, 2] = np.clip(pitch_rate_cmd, -10.0, 10.0)
        actions[:, 3] = np.clip(yaw_rate_cmd, -10.0, 10.0)

        return actions

class SupervisedTrainer:
    def __init__(self, env, policy, oracle, num_agents, episode_length, learning_rate=3e-4, debug=False, optimizer_type='adam'):
        self.env = env
        self.policy = policy
        self.oracle = oracle
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.debug = debug
        self.dt = 0.05

        # Optimizer
        if optimizer_type == 'kfac':
            self.optimizer = KFACOptimizer(self.policy, lr=learning_rate)
        else:
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

            if d['done_flags'].all() == 1.0:
                 break

        # Log duration
        duration = t + 1
        return total_loss / duration, duration

    def validate_episode(self, visualizer=None, iteration=0, visualize=False):
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
        compute_oracle_viz = (visualizer is not None) and visualize

        distances = []

        final_distances = np.full(self.num_agents, np.nan)
        already_done = np.zeros(self.num_agents, dtype=bool)

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

                # Capture distances for agents that JUST finished
                current_dists = np.linalg.norm(pos - vt, axis=1)
                done_mask = d['done_flags'].astype(bool)

                just_finished = done_mask & (~already_done)
                final_distances[just_finished] = current_dists[just_finished]
                already_done = done_mask | already_done

                # Store
                pos_history.append(pos)
                target_history.append(vt)
                # Tracker: u, v, size, conf
                # u, v are at 298, 299 (History). Size, Conf at 300, 301 (Aux).
                u_col = obs_np[:, 298:299]
                v_col = obs_np[:, 299:300]
                size_col = obs_np[:, 300:301]
                conf_col = obs_np[:, 301:302]
                track = np.concatenate([u_col, v_col, size_col, conf_col], axis=1)
                tracker_history.append(track)

                if d['done_flags'].all() == 1.0:
                     break

        # Final Distance (at termination)
        # Fill remainder with last dist
        current_dists = np.linalg.norm(pos - vt, axis=1)
        final_distances[~already_done] = current_dists[~already_done]

        final_dist = np.nanmean(final_distances) # Mean over agents

        # Visualization
        if visualizer is not None and visualize:
            # Stack: (T, N, 3) -> (N, T, 3)
            traj = np.stack(pos_history, axis=1)
            targ = np.stack(target_history, axis=1)
            track = np.stack(tracker_history, axis=1)

            # Log to visualizer (Agent 0)
            visualizer.log_trajectory(iteration, traj, targets=targ, tracker_data=track)

            # Save GIF
            visualizer.save_episode_gif(iteration, traj[0], targets=targ[0], tracker_data=track[0])

        return final_dist

class PPOTrainer:
    def __init__(self, env, policy, num_agents, episode_length, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, n_epochs=5, batch_size=2048, debug=False, optimizer_type='adam'):
        self.env = env
        self.policy = policy
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.debug = debug
        self.optimizer_type = optimizer_type

        # Optimizer
        if optimizer_type == 'kfac':
            self.optimizer = KFACOptimizer(self.policy, lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Pre-resolve step arguments
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

    def collect_rollouts(self):
        self.env.reset_all_envs()
        d = self.env.data_dictionary

        # Buffers
        obs_buf = []
        action_buf = []
        log_prob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

        total_reward = 0.0

        # PPO requires keeping track of the previous done state to mask returns correctly
        # Here we collect one full episode trajectory per agent
        # But we need to handle early terminations by masking

        for t in range(self.episode_length):
            obs_np = d["observations"]
            if self.debug:
                 obs_np = np.nan_to_num(obs_np)
            obs_torch = torch.from_numpy(obs_np).float()

            with torch.no_grad():
                mean, value = self.policy(obs_torch)
                std = self.policy.action_log_std.exp()
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(axis=-1)

            # Step Env
            # Clip action for environment safety
            action_np = action.cpu().numpy()
            clipped_action = np.clip(action_np, -1.0, 1.0) # Assume normalized action space
            d['actions'][:] = clipped_action.flatten()

            self.step_env()

            rewards = d["rewards"].copy()
            dones = d["done_flags"].copy().astype(bool)

            # Store in buffer
            obs_buf.append(obs_torch)
            action_buf.append(action)
            log_prob_buf.append(log_prob)
            value_buf.append(value.squeeze())
            reward_buf.append(torch.from_numpy(rewards).float())
            done_buf.append(torch.from_numpy(dones).float())

            total_reward += rewards.sum()

            if dones.all():
                break

        # Bootstrap value for last step (if not done)
        # We assume 0 value for terminal states, but if timeout, we might need value
        # For simplicity, just use 0 for now as episode ended.

        # Convert buffers to tensors
        # Shape: (T, N, ...)
        obs_tens = torch.stack(obs_buf)
        act_tens = torch.stack(action_buf)
        lp_tens = torch.stack(log_prob_buf)
        rew_tens = torch.stack(reward_buf)
        done_tens = torch.stack(done_buf)
        val_tens = torch.stack(value_buf)

        return obs_tens, act_tens, lp_tens, rew_tens, done_tens, val_tens, t+1

    def compute_gae(self, rewards, values, dones):
        # returns and advantages
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        # Next value is 0 for the very last step of the episode
        # For intermediate steps, next_value is values[t+1]

        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 0.0 # Assuming episode always ends at T or before
                nextvalues = 0.0
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t+1]

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, log_probs, returns, advantages):
        # Flatten (T, N, ...) -> (Batch, ...)
        # obs tensor shape is (T, N, 302)
        obs = obs.reshape(-1, obs.shape[-1])
        actions = actions.reshape(-1, self.policy.action_dim)
        log_probs = log_probs.reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)

        # Normalize Advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Dataset
        dataset_size = obs.size(0)
        indices = np.arange(dataset_size)

        total_loss = 0.0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]

                # Forward
                mean, value = self.policy(mb_obs)
                std = self.policy.action_log_std.exp()
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1).mean()

                # Ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Surrogate Loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = 0.5 * ((value.squeeze() - mb_returns) ** 2).mean()

                # Total Loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / (self.n_epochs * (dataset_size / self.batch_size))

    def train_step(self):
        obs, actions, log_probs, rewards, dones, values, duration = self.collect_rollouts()
        advantages, returns = self.compute_gae(rewards, values, dones)
        loss = self.update(obs, actions, log_probs, returns, advantages)
        return loss, duration, rewards.mean().item()

    def validate_episode(self, visualizer=None, iteration=0, visualize=False):
        # Reuse the SupervisedTrainer's validation logic or implement similar
        # Since logic is identical (run policy without noise), we can just implement it here

        self.env.reset_all_envs()
        d = self.env.data_dictionary

        pos_history = []
        target_history = []
        tracker_history = []

        final_distances = np.full(self.num_agents, np.nan)
        already_done = np.zeros(self.num_agents, dtype=bool)

        with torch.no_grad():
            for t in range(self.episode_length):
                obs_np = d["observations"]
                obs_torch = torch.from_numpy(obs_np).float()

                mean, _ = self.policy(obs_torch)
                # Deterministic action for validation (mean)
                action_to_step = mean.numpy()
                d['actions'][:] = np.clip(action_to_step, -1.0, 1.0).flatten()

                self.step_env()

                pos = np.stack([d['pos_x'], d['pos_y'], d['pos_z']], axis=1).copy()
                vt = np.stack([d['vt_x'], d['vt_y'], d['vt_z']], axis=1).copy()

                current_dists = np.linalg.norm(pos - vt, axis=1)
                done_mask = d['done_flags'].astype(bool)

                just_finished = done_mask & (~already_done)
                final_distances[just_finished] = current_dists[just_finished]
                already_done = done_mask | already_done

                pos_history.append(pos)
                target_history.append(vt)

                u_col = obs_np[:, 298:299]
                v_col = obs_np[:, 299:300]
                size_col = obs_np[:, 300:301]
                conf_col = obs_np[:, 301:302]
                track = np.concatenate([u_col, v_col, size_col, conf_col], axis=1)
                tracker_history.append(track)

                if d['done_flags'].all():
                    break

        current_dists = np.linalg.norm(pos - vt, axis=1)
        final_distances[~already_done] = current_dists[~already_done]
        final_dist = np.nanmean(final_distances)

        if visualizer is not None and visualize:
            traj = np.stack(pos_history, axis=1)
            targ = np.stack(target_history, axis=1)
            track = np.stack(tracker_history, axis=1)
            visualizer.log_trajectory(iteration, traj, targets=targ, tracker_data=track)
            visualizer.save_episode_gif(iteration, traj[0], targets=targ[0], tracker_data=track[0])

        return final_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=0, help="Total iterations (legacy, overrides split if > 0)")
    parser.add_argument("--supervised_iters", type=int, default=5000, help="Number of Supervised Learning iterations")
    parser.add_argument("--ppo_iters", type=int, default=5000, help="Number of PPO iterations")
    parser.add_argument("--episode_length", type=int, default=400)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualization generation")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "kfac"], help="Optimizer type")
    parser.add_argument("--use_resnet", action="store_true", help="Use ResNet architecture")
    parser.add_argument("--num_res_blocks", type=int, default=4, help="Number of ResNet blocks")
    args = parser.parse_args()

    # Environment
    env = DroneEnv(num_agents=args.num_agents, episode_length=args.episode_length, use_cuda=False)

    # Oracle (for Supervised)
    oracle = LinearPlanner(num_agents=args.num_agents)

    # Student Policy
    policy = DronePolicy(observation_dim=302, action_dim=4, hidden_dim=256, use_resnet=args.use_resnet, num_res_blocks=args.num_res_blocks).cpu()

    start_itr = 1
    if args.load:
        if os.path.exists(args.load):
            logging.info(f"Loading checkpoint from {args.load}")
            checkpoint = torch.load(args.load)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['model_state_dict'])
                if 'iteration' in checkpoint:
                    start_itr = checkpoint['iteration'] + 1
            else:
                policy.load_state_dict(checkpoint)
        else:
            logging.warning(f"Checkpoint {args.load} not found. Starting fresh.")

    visualizer = Visualizer()
    start_time = time.time()

    # Logic for split training
    total_supervised = args.supervised_iters
    total_ppo = args.ppo_iters

    # If legacy 'iterations' is set, use it for supervised (or split evenly? let's default to supervised)
    if args.iterations > 0:
        total_supervised = args.iterations
        total_ppo = 0

    # 1. Supervised Training Phase
    if total_supervised > 0:
        logging.info(f"Starting Supervised Training: {args.num_agents} Agents, {total_supervised} Iterations")
        trainer = SupervisedTrainer(env, policy, oracle, args.num_agents, args.episode_length, debug=args.debug, optimizer_type=args.optimizer)

        # Try to load optimizer state if resuming and we are in the right phase
        if args.load and 'optimizer_state_dict' in checkpoint and start_itr <= total_supervised:
             trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for itr in range(start_itr, total_supervised + 1):
            loss, duration = trainer.train_episode()
            visualizer.log_loss(itr, loss)

            if itr % 10 == 0:
                visualize = (itr % args.viz_freq == 0)
                val_dist = trainer.validate_episode(visualizer, itr, visualize=visualize)
                elapsed = time.time() - start_time
                logging.info(f"[Sup] Iter {itr} | Loss: {loss:.4f} | Avg Ep Len: {duration} | Val Dist: {val_dist:.4f} m | Time: {elapsed:.2f}s")
                visualizer.log_reward(itr, -val_dist)
                visualizer.plot_loss()
                if visualize:
                    try:
                        visualizer.generate_trajectory_gif()
                    except Exception as e:
                        logging.error(f"Error GIF: {e}")

            if itr % 50 == 0:
                ckpt = {'iteration': itr, 'model_state_dict': policy.state_dict(), 'optimizer_state_dict': trainer.optimizer.state_dict()}
                torch.save(ckpt, "latest_jules.pth")

        torch.save({'iteration': total_supervised, 'model_state_dict': policy.state_dict()}, "final_supervised.pth")

        # Update start_itr for PPO
        start_itr = max(start_itr, total_supervised + 1)

    # 2. PPO Training Phase
    if total_ppo > 0:
        logging.info(f"Starting PPO Training: {args.num_agents} Agents, {total_ppo} Iterations")
        ppo_trainer = PPOTrainer(env, policy, args.num_agents, args.episode_length, debug=args.debug, optimizer_type=args.optimizer)

        # Reset optimizer for PPO phase as objective changed
        # Unless we want to keep momentum? Usually better to reset or use low LR.
        # We will start fresh optimizer for PPO.

        for itr in range(start_itr, total_supervised + total_ppo + 1):
            loss, duration, avg_reward = ppo_trainer.train_step()
            visualizer.log_loss(itr, loss) # Log PPO loss

            if itr % 10 == 0:
                visualize = (itr % args.viz_freq == 0)
                val_dist = ppo_trainer.validate_episode(visualizer, itr, visualize=visualize)
                elapsed = time.time() - start_time
                logging.info(f"[PPO] Iter {itr} | Loss: {loss:.4f} | Reward: {avg_reward:.2f} | Val Dist: {val_dist:.4f} m | Time: {elapsed:.2f}s")
                visualizer.log_reward(itr, avg_reward) # Log actual reward for PPO
                visualizer.plot_loss()
                if visualize:
                    try:
                        visualizer.generate_trajectory_gif()
                    except Exception as e:
                        logging.error(f"Error GIF: {e}")

            if itr % 50 == 0:
                ckpt = {'iteration': itr, 'model_state_dict': policy.state_dict(), 'optimizer_state_dict': ppo_trainer.optimizer.state_dict()}
                torch.save(ckpt, "latest_jules.pth")

    final_checkpoint = {
        'iteration': total_supervised + total_ppo,
        'model_state_dict': policy.state_dict(),
    }
    torch.save(final_checkpoint, "final_jules.pth")

    if visualizer.rewards_history:
        visualizer.plot_rewards()
    visualizer.plot_loss()
    visualizer.generate_trajectory_gif()

    logging.info("Training Complete.")

if __name__ == "__main__":
    main()
