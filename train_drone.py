import argparse
import os
import yaml
import torch
import logging

# Import WarpDrive components
try:
    from warp_drive.env_wrapper import EnvWrapper
    from warp_drive.training.trainer import Trainer
except ImportError:
    print("WarpDrive not installed. Please install it to run this script.")
    exit(1)

# Import your Custom Env
# Ensure the current directory is in python path
import sys
sys.path.append(os.getcwd())

from drone_env.drone import DroneEnv

def setup_and_train(run_config, device_id=0):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        print("WARNING: CUDA not available. Training will likely fail or require CPU backend (if supported).")

    # 1. Initialize Wrapper with Your Custom Class
    # Note: env_backend="pycuda" requires CUDA.
    env_wrapper = EnvWrapper(
        DroneEnv(**run_config["env"]),
        num_envs=run_config["trainer"]["num_envs"],
        env_backend="pycuda",
        process_id=device_id
    )

    # 2. Map Policy
    policy_map = {"drone_policy": env_wrapper.env.agent_ids}

    # 3. Create Trainer
    trainer = Trainer(
        env_wrapper=env_wrapper,
        config=run_config,
        policy_tag_to_agent_id_map=policy_map,
        device_id=device_id
    )

    # 4. Train
    print("Starting Training...")
    trainer.train()
    trainer.graceful_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/drone.yaml")
    args = parser.parse_args()

    # Load Config locally
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    setup_and_train(config)
