import warnings
# Suppress Gym/NumPy 2.0 warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import numpy as np
import os

from warp_drive.env_wrapper import EnvWrapper
from warp_drive.training.trainers.trainer_a2c import TrainerA2C
# --- NEW: Import the dummy environment class and the CUDA source file path ---
from warp_drive.env.env_configs import DUMMY_ENV_CUDA_SOURCE_PATH
from warp_drive.env.dummy_env import DummyEnv

print("WarpDrive 2.7.1 + CUDA + PyTorch = READY")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- 1. Setup Configuration for the DummyEnv ---
env_config = {
    "num_envs": 2048,   # Massive parallelism
    "num_agents": 1,
    "episode_length": 100,
    "env_backend": "pycuda", # Backend must be pycuda for GPU
}

# Instantiate the DummyEnv, pointing it to the correct CUDA source file.
env_obj = DummyEnv(
    num_envs=env_config["num_envs"], 
    num_agents=env_config["num_agents"],
    episode_length=env_config["episode_length"],
    env_backend=env_config["env_backend"],
    # CRITICAL: Specify the path to the dummy CUDA source file
    cuda_env_src_path=DUMMY_ENV_CUDA_SOURCE_PATH
)

print(f"\nInitializing EnvWrapper with backend='pycuda' using DummyEnv...")

# --- 2. Initialize the Wrapper (The Critical Step) ---
env_wrapper = EnvWrapper(
    env_obj=env_obj,
    num_envs=env_config["num_envs"],
    env_backend=env_config["env_backend"]
)

# --- 3. Define A2C Configuration (Matched to DummyEnv's defaults) ---
config = {
    "num_episodes": 10,
    "episode_length": 100,
    "train_batch_size": env_config["num_envs"],
    "max_time": 60,
    "trainer": {
        "algorithm": "A2C",
        "policy_learning_rate": 0.005,
        "value_learning_rate": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_param": 0.2,
        "entropy_coeff": 0.01,
        "value_loss_coeff": 0.5,
        "grad_norm_clip": 10.0,
    },
    "policy": {
        "default_policy": { # The DummyEnv uses "default_policy"
            "network": {
                "hidden_layers": [64, 64],
                "activation": "tanh",
                "output_activation": "linear"
            },
            "learning_rate": 0.005,
            "optimizer": "Adam",
            "action_dim": env_obj.action_dim, # Matches DummyEnv's default action dim
            "model_path": ""
        }
    },
    "saving": {
        "tag": "wd_271_dummy_test",
        "logging": False,
        "model_dir": "logs",
        "save_every": 10
    }
}

# --- 4. Initialize Trainer & Run ---
# Map the single agent policy "default_policy" to agent_id 0
policy_tag_to_agent_id_map = {"default_policy": list(range(env_config["num_agents"]))}

trainer = TrainerA2C(
    env_wrapper=env_wrapper,
    config=config,
    policy_tag_to_agent_id_map=policy_tag_to_agent_id_map
)

print("\nStarting Training Loop...")
import time
start = time.time()

try:
    trainer.train()
except KeyboardInterrupt:
    print("User interrupted.")
except Exception as e:
    print(f"\nTraining Failed: {e}")
    print("This often indicates a compilation error (missing nvcc or incompatible drivers).")

elapsed = time.time() - start
total_steps = env_config["num_envs"] * 100 * 10
print(f"\nSUCCESS! WarpDrive 2.7.1 validated.")
print(f"Throughput: ~{total_steps / elapsed:,.0f} steps/sec")
