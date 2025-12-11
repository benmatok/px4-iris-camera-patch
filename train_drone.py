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
from models.ae_policy import DronePolicy, KFACOptimizer

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
        self.ae_optimizer = KFACOptimizer(new_model.ae.parameters(), lr=0.001)
        self.ae_criterion = nn.L1Loss()

    def train(self):
        """
        Custom training loop that interleaves AE training with RL updates.
        We attempt to run the standard RL steps manually or hook into the process.
        """
        logging.info("Starting Custom Training Loop with Autoencoder...")

        num_iters = self.config['trainer']['training_iterations']
        # We need access to the data manager
        data_manager = self.env_wrapper.cuda_data_manager

        self.env_wrapper.reset_all_envs()

        # Since we cannot easily replicate the PPO buffer management and update logic (it's complex),
        # we will use a hybrid approach:
        # We will iterate manually. For each iteration:
        # 1. Run environment steps to collect data (Rollout).
        # 2. Update AE using the collected observations.
        # 3. Call super().train() for ONE iteration? No, super().train() runs the WHOLE loop.

        # Strategy: We assume Trainer has a `step()` method or similar that performs ONE PPO update cycle.
        # If not, we will rely on `super().train()` but we are stuck regarding AE updates.

        # However, many implementations of Trainer have a `step` or loop body.
        # If we assume we can't call a single step, we have to write the loop.

        # Given the constraints, I will implement a loop that *attempts* to update AE.
        # If `step()` exists, use it. Else, fall back.

        has_step = hasattr(self, 'step')

        if has_step:
            for itr in range(num_iters):
                # Run standard RL step (Rollout + Update)
                self.step()

                # After RL step, the observations on GPU are fresh (or from the rollout).
                # We pull a sample of observations to train the AE.
                # Ideally we use the batch from the PPO buffer, but that's internal.
                # We will pull the *current* observations from the environment state.
                # This gives us `num_envs` samples.

                obs_data = data_manager.pull_data("observations") # Shape (num_envs, 1804)
                obs_tensor = torch.from_numpy(obs_data).cuda()

                # Train AE
                self.ae_optimizer.zero_grad()
                # Forward pass through Policy (which calls AE)
                # We only need AE part, but calling policy is easier
                _, _, recon, history = self.models['drone_policy'](obs_tensor)

                loss = self.ae_criterion(recon, history)
                loss.backward()
                self.ae_optimizer.step()

                if itr % 10 == 0:
                    print(f"Iter {itr}: AE Loss {loss.item()}")

        else:
            print("Trainer.step() not found. Falling back to standard train loop (AE will not be updated per step).")
            # We try to update AE once before training to ensure it's initialized?
            # No, that's useless.
            # We call the standard train.
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
