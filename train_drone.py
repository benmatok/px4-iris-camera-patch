import argparse
import os
import yaml
import torch
import torch.nn as nn
import logging
import numpy as np

# Import WarpDrive components
try:
    from warp_drive.env_wrapper import EnvWrapper
    from warp_drive.training.trainer import Trainer
except ImportError:
    print("WarpDrive not installed. Please install it to run this script.")
    pass

import sys
sys.path.append(os.getcwd())

from drone_env.drone import DroneEnv
from models.ae_policy import DronePolicy, KFACOptimizerPlaceholder

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

    def train(self):
        """
        Custom training loop that interleaves AE training with RL updates.
        We attempt to run the standard RL steps manually or hook into the process.
        """
        logging.info("Starting Custom Training Loop with Autoencoder...")

        num_iters = self.config['trainer']['training_iterations']
        data_manager = self.env_wrapper.cuda_data_manager

        self.env_wrapper.reset_all_envs()

        # Check if we can use a step-based approach.
        # Most WarpDrive versions don't expose a clean public 'step()' that does exactly one PPO iteration.
        # However, we can construct the loop if we know the internals (fetch, rollout, loss, update).
        # Since we don't, we will rely on a "Graceful Fallback" or "Interception".

        # Attempt to detect a step method
        has_step = hasattr(self, 'step')
        if not has_step and hasattr(self, 'trainer_step'): has_step = True # Some versions

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

        else:
            print("WARNING: Trainer.step() not found. Falling back to standard train loop.")
            print("Autoencoder optimization loop cannot be interleaved without modifying WarpDrive source.")
            # We call the standard train. AE will NOT be updated.
            super().train()

def setup_and_train(run_config, device_id=0):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        print("WARNING: CUDA not available. Training will likely fail or require CPU backend (if supported).")

    env_wrapper = EnvWrapper(
        DroneEnv(**run_config["env"]),
        num_envs=run_config["trainer"]["num_envs"],
        env_backend="pycuda",
        process_id=device_id
    )

    policy_map = {"drone_policy": env_wrapper.env.agent_ids}

    # Use Custom Trainer
    trainer = CustomTrainer(
        env_wrapper=env_wrapper,
        config=run_config,
        policy_tag_to_agent_id_map=policy_map,
        device_id=device_id
    )

    print("Starting Training...")
    trainer.train()
    trainer.graceful_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/drone.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    setup_and_train(config)
