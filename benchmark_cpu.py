import time
import numpy as np
from drone_env.drone import step_cpu, reset_cpu, DroneEnv
import drone_env.drone_cython as drone_cython

def run_benchmark():
    num_agents = 5000
    episode_length = 100
    num_steps = 100

    print(f"Benchmarking with {num_agents} agents for {num_steps} steps...")

    # Initialize data using DroneEnv helper to get correct shapes
    # We use use_cuda=False to get the structure, but we won't use the env's step function directly
    # to control which one we call.
    env = DroneEnv(num_agents=num_agents, episode_length=episode_length, use_cuda=False)

    # Create data dict
    data = {}
    data_dict = env.get_data_dictionary()
    for name, info in data_dict.items():
        data[name] = np.zeros(info["shape"], dtype=info["dtype"])

    # Prepare kwargs
    reset_kwargs = env.get_reset_function_kwargs()
    real_reset_kwargs = {}
    for arg, key in reset_kwargs.items():
        if key in data:
            real_reset_kwargs[arg] = data[key]
        elif key == "num_agents":
            real_reset_kwargs[arg] = num_agents
        elif key == "episode_length":
             real_reset_kwargs[arg] = episode_length
    real_reset_kwargs["reset_indices"] = np.array([0], dtype=np.int32) # Dummy env id

    step_kwargs = env.get_step_function_kwargs()
    real_step_kwargs = {}
    for arg, key in step_kwargs.items():
         if key in data:
            real_step_kwargs[arg] = data[key]
         elif key == "num_agents":
            real_step_kwargs[arg] = num_agents
         elif key == "episode_length":
            real_step_kwargs[arg] = episode_length

    # Initialize actions
    actions = np.zeros((num_agents, 4), dtype=np.float32).flatten()
    real_step_kwargs["actions"] = actions
    real_step_kwargs["env_ids"] = np.array([0], dtype=np.int32)

    # ---------------------------------------------------------
    # Benchmark Python/NumPy
    # ---------------------------------------------------------
    print("\n--- Python/NumPy ---")

    # Warmup / Reset
    start_time = time.time()
    reset_cpu(**real_reset_kwargs)
    reset_time_py = time.time() - start_time
    print(f"Reset Time: {reset_time_py:.4f} s")

    start_time = time.time()
    for _ in range(num_steps):
        step_cpu(**real_step_kwargs)
    step_time_py = time.time() - start_time
    print(f"Step Time ({num_steps} steps): {step_time_py:.4f} s")
    print(f"FPS: {num_steps * num_agents / step_time_py:.2f}")

    # ---------------------------------------------------------
    # Benchmark Cython
    # ---------------------------------------------------------
    print("\n--- Cython ---")

    # Warmup / Reset
    start_time = time.time()
    drone_cython.reset_cython(**real_reset_kwargs)
    reset_time_cy = time.time() - start_time
    print(f"Reset Time: {reset_time_cy:.4f} s")

    start_time = time.time()
    for _ in range(num_steps):
        drone_cython.step_cython(**real_step_kwargs)
    step_time_cy = time.time() - start_time
    print(f"Step Time ({num_steps} steps): {step_time_cy:.4f} s")
    print(f"FPS: {num_steps * num_agents / step_time_cy:.2f}")

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------
    print("\n--- Speedup ---")
    print(f"Reset Speedup: {reset_time_py / reset_time_cy:.2f}x")
    print(f"Step Speedup: {step_time_py / step_time_cy:.2f}x")

if __name__ == "__main__":
    run_benchmark()
