import argparse
import os
import yaml
import torch
import torch.nn as nn
import logging
import numpy as np
import sys

sys.path.append(os.getcwd())

# Import WarpDrive components
try:
    from warp_drive.env_wrapper import EnvWrapper
    from warp_drive.training.trainer import Trainer
    HAS_WARPDRIVE = True
except ImportError:
    HAS_WARPDRIVE = False
    print("WarpDrive not installed or not fully importable (likely missing pycuda). Using custom CPU trainer.")

from drone_env.drone import DroneEnv
from models.ae_policy import DronePolicy, KFACOptimizerPlaceholder
from visualization import Visualizer

# -----------------------------------------------------------------------------
# Custom Trainer inheriting from WarpDrive Trainer (for GPU)
# -----------------------------------------------------------------------------
if HAS_WARPDRIVE:
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Override the model with our Custom Policy
            new_model = DronePolicy(self.env_wrapper.env).cuda()
            self.models['drone_policy'] = new_model

            # Re-initialize the RL optimizer
            lr = self.config['algorithm']['lr']
            self.optimizers['drone_policy'] = torch.optim.Adam(new_model.parameters(), lr=lr)

            # Initialize KFAC for AE (Auxiliary)
            self.ae_optimizer = KFACOptimizerPlaceholder(new_model.ae.parameters(), lr=0.001)
            self.ae_criterion = nn.L1Loss()

            # Initialize Visualizer
            self.visualizer = Visualizer()

        def train(self):
            """
            Custom training loop that interleaves AE training with RL updates.
            """
            logging.info("Starting Custom Training Loop with Autoencoder...")

            num_iters = self.config['trainer']['training_iterations']
            data_manager = self.env_wrapper.cuda_data_manager

            self.env_wrapper.reset_all_envs()

            # Check if we can use a step-based approach.
            has_step = hasattr(self, 'step')
            if not has_step and hasattr(self, 'trainer_step'): has_step = True

            if has_step:
                step_fn = self.step if hasattr(self, 'step') else self.trainer_step

                for itr in range(num_iters):
                    # Run standard RL step (Rollout + Update)
                    step_fn()

                    # Interleaved AE Training
                    # 1. Fetch current observations (fresh from rollout or update)
                    obs_data = data_manager.pull_data("observations") # Shape (num_envs, 1804)
                    obs_tensor = torch.from_numpy(obs_data).cuda()

                    # 2. Update AE
                    self.ae_optimizer.zero_grad()
                    _, _, recon, history = self.models['drone_policy'](obs_tensor)

                    loss = self.ae_criterion(recon, history)
                    loss.backward()
                    self.ae_optimizer.step()

                    if itr % 10 == 0:
                        print(f"Iter {itr}: AE Loss {loss.item()}")

                    # Visualization Hooks
                    # Log Rewards
                    rewards = data_manager.pull_data("rewards")
                    mean_reward = np.mean(rewards)
                    self.visualizer.log_reward(itr, mean_reward)

                    # Capture Trajectory every 50 iterations (or 10% of total)
                    if itr % 50 == 0 or itr == num_iters - 1:
                        pos_history = data_manager.pull_data("pos_history") # (num_envs * episode_length * 3)
                        # Reshape: (num_envs, episode_length, 3)
                        episode_length = self.env_wrapper.env.episode_length
                        pos_history = pos_history.reshape(self.config['trainer']['num_envs'], episode_length, 3)
                        self.visualizer.log_trajectory(itr, pos_history)

                # Generate Plots and GIF
                self.visualizer.plot_rewards()
                gif_path = self.visualizer.generate_trajectory_gif()
                print(f"Visualization complete. GIF saved at {gif_path}")

            else:
                print("WARNING: Trainer.step() not found. Falling back to standard train loop.")
                print("Autoencoder optimization loop cannot be interleaved without modifying WarpDrive source.")
                super().train()

# -----------------------------------------------------------------------------
# CPU Fallback Trainer
# -----------------------------------------------------------------------------
class CPUTrainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.visualizer = Visualizer()

        # Initialize Model (CPU)
        # We wrap the policy to handle torch tensors
        self.policy = DronePolicy(env).to("cpu")

        # Separate optimizers:
        # 1. RL Optimizer: Policy Head + Value Head
        # 2. AE Optimizer: Autoencoder (Encoder + Decoder)
        rl_params = list(self.policy.policy_head.parameters()) + list(self.policy.value_head.parameters())
        ae_params = list(self.policy.ae.parameters())

        self.optimizer = torch.optim.Adam(rl_params, lr=config['algorithm']['lr'])
        self.ae_optimizer = torch.optim.Adam(ae_params, lr=0.001)

        self.ae_criterion = nn.L1Loss()

        # State Containers
        self.num_envs = env.num_agents # In our setup env.num_agents is total agents across parallel envs if parallelized
        # Actually DroneEnv logic: num_agents is agents PER env if using WarpDriveEnvWrapper,
        # but here we use DroneEnv directly.
        # DroneEnv.num_agents passed in config is usually 1 agent per env, but many envs.
        # But DroneEnv implementation treats num_agents as total entities if we run it directly?
        # Let's check reset_cpu logic. reset_cpu iterates `range(num_agents)`.
        # So `DroneEnv` as written represents ONE environment block with `num_agents`.
        # But `train_drone` config says `num_envs: 1024`.
        # In WarpDrive, `EnvWrapper` manages `num_envs` blocks of `DroneEnv`.
        # For CPU Trainer, we need to manage the data arrays ourselves.

        # We will initialize the data dictionary arrays manually since we don't have CUDA data manager
        self.data = {}
        data_dict = self.env.get_data_dictionary()
        for name, info in data_dict.items():
            shape = info["shape"]
            # If shape depends on num_envs, we need to adjust?
            # DroneEnv definition: "masses": (self.num_agents,)
            # It seems DroneEnv is designed to handle ALL agents in parallel (as a CUDA kernel does).
            # So if we want 1024 envs with 1 agent each, we should instantiate DroneEnv with num_agents=1024?
            # In WarpDrive config: env: num_agents: 1. trainer: num_envs: 1024.
            # WarpDrive EnvWrapper handles the multiplicity.
            # Since we don't have EnvWrapper, we should instantiate DroneEnv with num_agents = 1024 * 1.
            self.data[name] = np.zeros(shape, dtype=info["dtype"])

        self.episode_length = self.env.episode_length

    def train(self):
        num_iters = self.config['trainer']['training_iterations']
        logging.info(f"Starting CPU Training Loop for {num_iters} iterations...")

        # PPO Hyperparams
        gamma = self.config['algorithm']['gamma']
        clip_param = self.config['algorithm']['clip_param']
        vf_loss_coeff = self.config['algorithm']['vf_loss_coeff']
        entropy_coeff = self.config['algorithm']['entropy_coeff']

        for itr in range(num_iters):
            # 1. Collect Rollouts
            # Reset
            # We need to define reset indices (all)
            reset_indices = np.array([0], dtype=np.int32) # Assuming single block env_id=0 for now or manage blocks
            # Wait, DroneEnv CUDA kernel uses blockIdx.x for env_id.
            # step_cpu loops over `total_agents`.
            # reset_cpu loops over `reset_indices`.
            # If we instantiated DroneEnv with num_agents = 1024, effectively we have 1 env with 1024 agents (or 1024 envs linear).
            # The reset_cpu code: `idx = env_id * num_agents + agent_id`.
            # If we treat it as 1 giant environment with N agents, env_id=0.

            # Reset all
            # We need valid args for reset_cpu
            # We use `self.data` dict to pass arrays
            self.env.reset_function(
                pos_x=self.data["pos_x"], pos_y=self.data["pos_y"], pos_z=self.data["pos_z"],
                vel_x=self.data["vel_x"], vel_y=self.data["vel_y"], vel_z=self.data["vel_z"],
                roll=self.data["roll"], pitch=self.data["pitch"], yaw=self.data["yaw"],
                masses=self.data["masses"], drag_coeffs=self.data["drag_coeffs"], thrust_coeffs=self.data["thrust_coeffs"],
                target_vx=self.data["target_vx"], target_vy=self.data["target_vy"], target_vz=self.data["target_vz"], target_yaw_rate=self.data["target_yaw_rate"],
                pos_history=self.data["pos_history"], observations=self.data["observations"],
                rng_states=self.data["rng_states"], step_counts=self.data["step_counts"],
                num_agents=self.env.num_agents, reset_indices=np.array([0], dtype=np.int32)
            )

            # Pre-allocate Rollout Buffers (Tensors) to avoid list duplication/stack overhead
            # obs: float16, others: float32
            num_agents = self.env.num_agents
            ep_len = self.episode_length

            obs_buffer = torch.zeros((ep_len, num_agents, 1804), dtype=torch.float16)
            action_buffer = torch.zeros((ep_len, num_agents, 4), dtype=torch.float32)
            reward_buffer = torch.zeros((ep_len, num_agents), dtype=torch.float32)
            value_buffer = torch.zeros((ep_len, num_agents), dtype=torch.float32)
            log_prob_buffer = torch.zeros((ep_len, num_agents), dtype=torch.float32)

            for t in range(self.episode_length):
                # IMPORTANT: Use .float() for model input
                current_obs_np = self.data["observations"]
                current_obs = torch.from_numpy(current_obs_np).float()

                # Forward Pass
                mu, v, recon, hist = self.policy(current_obs)

                # Sample actions
                dist = torch.distributions.Normal(mu, torch.ones_like(mu)*0.5)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

                # Store in pre-allocated buffers
                obs_buffer[t] = current_obs.half() # implicit copy
                action_buffer[t] = action.detach()
                value_buffer[t] = v.squeeze().detach()
                log_prob_buffer[t] = log_prob.detach()

                # Execute Step
                actions_np = action.detach().numpy().flatten()
                env_ids_to_step = np.array([0], dtype=np.int32)

                self.env.step_function(
                    pos_x=self.data["pos_x"], pos_y=self.data["pos_y"], pos_z=self.data["pos_z"],
                    vel_x=self.data["vel_x"], vel_y=self.data["vel_y"], vel_z=self.data["vel_z"],
                    roll=self.data["roll"], pitch=self.data["pitch"], yaw=self.data["yaw"],
                    masses=self.data["masses"], drag_coeffs=self.data["drag_coeffs"], thrust_coeffs=self.data["thrust_coeffs"],
                    target_vx=self.data["target_vx"], target_vy=self.data["target_vy"], target_vz=self.data["target_vz"], target_yaw_rate=self.data["target_yaw_rate"],
                    pos_history=self.data["pos_history"],
                    observations=self.data["observations"], rewards=self.data["rewards"],
                    done_flags=self.data["done_flags"], step_counts=self.data["step_counts"],
                    actions=actions_np,
                    num_agents=self.env.num_agents, episode_length=self.episode_length,
                    env_ids=env_ids_to_step
                )

                # Store reward
                reward_buffer[t] = torch.from_numpy(self.data["rewards"]).float()

            # 2. Compute Advantages (GAE)
            # We can do this in-place or separate tensor.
            # Separate tensor for returns/advantages
            returns = torch.zeros_like(reward_buffer)
            R = torch.zeros(self.env.num_agents)
            for t in reversed(range(self.episode_length)):
                R = reward_buffer[t] + gamma * R
                returns[t] = R

            advantages = returns - value_buffer
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 3. Update Policy (PPO Step) with Mini-batches
            # Flatten batch
            b_obs_half = obs_buffer.reshape(-1, 1804)
            b_actions = action_buffer.reshape(-1, 4)
            b_log_probs = log_prob_buffer.reshape(-1)
            b_returns = returns.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_advantages = advantages.reshape(-1)

            # Mini-batch settings
            batch_size = b_obs_half.shape[0]
            minibatch_size = 4096

            # Accumulated metrics for logging
            total_loss = 0
            total_ae_loss = 0
            num_updates = 0

            indices = torch.randperm(batch_size)

            for start_idx in range(0, batch_size, minibatch_size):
                end_idx = min(start_idx + minibatch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]

                # Fetch mini-batch and cast to float
                mb_obs = b_obs_half[mb_indices].float()
                mb_actions = b_actions[mb_indices]
                mb_log_probs = b_log_probs[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_advantages = b_advantages[mb_indices]

                # Re-evaluate
                new_mu, new_v, new_recon, new_hist = self.policy(mb_obs)
                dist = torch.distributions.Normal(new_mu, torch.ones_like(new_mu)*0.5)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (mb_returns - new_v.squeeze()).pow(2).mean()

                loss = policy_loss + vf_loss_coeff * value_loss - entropy_coeff * entropy

                # AE Update
                ae_loss = self.ae_criterion(new_recon, new_hist)

                # Separate Updates
                self.optimizer.zero_grad()
                self.ae_optimizer.zero_grad()

                loss.backward(retain_graph=True)
                self.optimizer.step()

                ae_loss.backward()
                self.ae_optimizer.step()

                total_loss += loss.item()
                total_ae_loss += ae_loss.item()
                num_updates += 1

            # Logging (Average over minibatches)
            mean_reward = reward_buffer.mean().item()
            avg_loss = total_loss / num_updates
            avg_ae_loss = total_ae_loss / num_updates

            if itr % 5 == 0:
                print(f"Iter {itr}: Reward {mean_reward:.3f} Loss {avg_loss:.3f} AE {avg_ae_loss:.3f}")

            self.visualizer.log_reward(itr, mean_reward)

            if itr % 10 == 0 or itr == num_iters - 1:
                # Get pos_history from self.data
                # Shape in dict: (num_agents * episode_length * 3)
                # Reshape to (num_agents, episode_length, 3)
                ph = self.data["pos_history"].reshape(self.env.num_agents, self.episode_length, 3)
                self.visualizer.log_trajectory(itr, ph)

        # Generate Plots
        self.visualizer.plot_rewards()
        gif_path = self.visualizer.generate_trajectory_gif()
        print(f"Visualization complete. GIF saved at {gif_path}")


def setup_and_train(run_config, device_id=0):
    use_cuda = torch.cuda.is_available() and HAS_WARPDRIVE
    if use_cuda:
        torch.cuda.set_device(device_id)
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")

        env_wrapper = EnvWrapper(
            DroneEnv(**run_config["env"], use_cuda=True),
            num_envs=run_config["trainer"]["num_envs"],
            env_backend="pycuda",
            process_id=device_id
        )

        policy_map = {"drone_policy": env_wrapper.env.agent_ids}

        trainer = CustomTrainer(
            env_wrapper=env_wrapper,
            config=run_config,
            policy_tag_to_agent_id_map=policy_map,
            device_id=device_id
        )

        print("Starting GPU Training...")
        trainer.train()
        trainer.graceful_close()

    else:
        print("WARNING: CUDA or WarpDrive not available. Falling back to Custom CPU Training.")
        # Modify config for CPU feasibility (smaller batch)
        # We must reduce the load significantly to avoid OOM/Timeout on CPU

        # Override config for lightweight CPU run
        total_agents = 5000
        run_config["trainer"]["training_iterations"] = 500

        print(f"CPU Mode: Reduced agents to {total_agents} and iterations to {run_config['trainer']['training_iterations']}")

        # Instantiate DroneEnv with num_agents = total_agents
        # Note: DroneEnv original expects num_agents per env.
        # Here we just treat it as one big env.
        env = DroneEnv(num_agents=total_agents, episode_length=run_config["trainer"]["episode_length"], use_cuda=False)

        trainer = CPUTrainer(env, run_config)
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/drone.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    setup_and_train(config)
