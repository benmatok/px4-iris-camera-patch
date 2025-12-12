
import sys
import os
import time
import numpy as np
import resource
import torch
import gc

# Add current directory to path
sys.path.append(os.getcwd())

from drone_env.drone import DroneEnv
import train_drone # This will print warnings but should import classes

def get_memory_usage_mb():
    # returns RSS in MB
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: KB. MacOS: Bytes?
    # Usually KB on Linux.
    return usage / 1024.0

def run_benchmark(num_agents):
    print(f"Benchmarking {num_agents} agents...")
    gc.collect()
    start_mem = get_memory_usage_mb()

    # Config
    episode_length = 100

    try:
        # Instantiate Env
        # Using pure CPU mode
        env = DroneEnv(num_agents=num_agents, episode_length=episode_length, use_cuda=False)

        # Instantiate Trainer (simulated)
        # We just need to allocate the buffers CPUTrainer does.
        # Let's use the actual CPUTrainer class if possible.
        # Or mock it.
        # CPUTrainer does:
        # self.data[name] = np.zeros(...)
        # self.policy = DronePolicy(...)

        # Data Allocation
        data = {}
        data_dict = env.get_data_dictionary()
        for name, info in data_dict.items():
            shape = info["shape"]
            dtype = info["dtype"]
            # Allocate
            data[name] = np.zeros(shape, dtype=dtype)
            # Touch memory to ensure allocation (RSS)
            data[name].fill(0.1)

        # Model Allocation
        # DronePolicy is in models/ae_policy.py
        # We need to import it.
        from models.ae_policy import DronePolicy
        policy = DronePolicy(env).to("cpu")

        # Optimizer Allocation
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

        # Rollout Buffer Allocation (Critical for PPO)
        # 100 steps * (Obs + Action + Reward + Value + LogProb)
        # Obs: 1804 floats
        # Action: 4 floats
        rollout_steps = 100
        # We use list of tensors in CPUTrainer, but eventually it's a big tensor.
        # Let's allocate the final big tensor to simulate peak memory.
        obs_buffer = torch.zeros((rollout_steps, num_agents, 1804))
        action_buffer = torch.zeros((rollout_steps, num_agents, 4))

        # Touch memory
        obs_buffer.fill_(0.1)
        action_buffer.fill_(0.1)

        current_mem = get_memory_usage_mb()
        diff = current_mem - start_mem
        print(f"Agents: {num_agents}, Memory: {current_mem:.2f} MB (+{diff:.2f} MB)")
        return current_mem

    except Exception as e:
        print(f"Failed with {num_agents} agents: {e}")
        return None

if __name__ == "__main__":
    # Reduced list because rollout buffer is huge (~750KB per agent)
    agents_list = [100, 1000, 5000, 10000, 20000]
    results = []

    print("Starting Memory Benchmark...")
    baseline = get_memory_usage_mb()
    print(f"Baseline Memory: {baseline:.2f} MB")

    for n in agents_list:
        mem = run_benchmark(n)
        if mem is None:
            print("Stopping benchmark due to failure.")
            break
        results.append((n, mem))

        # Simple heuristic to stop before crashing system
        # If memory > 12GB (assuming 16GB limit), stop.
        if mem > 12000:
            print("Memory limit approaching. Stopping.")
            break

    print("\nResults:")
    print("| Agents | Memory (MB) |")
    print("|---|---|")
    for n, m in results:
        print(f"| {n} | {m:.2f} |")
