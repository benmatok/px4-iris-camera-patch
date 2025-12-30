import time
import numpy as np
from drone_env.drone import step_cpu, reset_cpu
try:
    from drone_env.drone_cython import step_cython, reset_cython
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

def benchmark():
    num_agents = 5000
    episode_length = 100
    env_ids = np.zeros(num_agents, dtype=np.int32)

    # Initialize data
    pos_x = np.zeros(num_agents, dtype=np.float32)
    pos_y = np.zeros(num_agents, dtype=np.float32)
    pos_z = np.zeros(num_agents, dtype=np.float32)
    vel_x = np.zeros(num_agents, dtype=np.float32)
    vel_y = np.zeros(num_agents, dtype=np.float32)
    vel_z = np.zeros(num_agents, dtype=np.float32)
    roll = np.zeros(num_agents, dtype=np.float32)
    pitch = np.zeros(num_agents, dtype=np.float32)
    yaw = np.zeros(num_agents, dtype=np.float32)
    masses = np.ones(num_agents, dtype=np.float32)
    drag_coeffs = np.ones(num_agents, dtype=np.float32) * 0.1
    thrust_coeffs = np.ones(num_agents, dtype=np.float32)
    target_vx = np.zeros(num_agents, dtype=np.float32)
    target_vy = np.zeros(num_agents, dtype=np.float32)
    target_vz = np.zeros(num_agents, dtype=np.float32)
    target_yaw_rate = np.zeros(num_agents, dtype=np.float32)

    # New Trajectory Params Shape: (10, num_agents)
    traj_params = np.zeros((10, num_agents), dtype=np.float32)

    # Precomputed Trajectory
    target_trajectory = np.zeros((episode_length + 1, num_agents, 3), dtype=np.float32)

    # Virtual Targets
    vt_x = np.zeros(num_agents, dtype=np.float32)
    vt_y = np.zeros(num_agents, dtype=np.float32)
    vt_z = np.zeros(num_agents, dtype=np.float32)

    # Pos History Shape: (episode_length, num_agents, 3)
    pos_history = np.zeros((episode_length, num_agents, 3), dtype=np.float32)

    observations = np.zeros((num_agents, 608), dtype=np.float32)
    rewards = np.zeros(num_agents, dtype=np.float32)
    reward_components = np.zeros((num_agents, 8), dtype=np.float32)
    done_flags = np.zeros(num_agents, dtype=np.float32)
    step_counts = np.zeros(1, dtype=np.int32)
    actions = np.zeros(num_agents * 4, dtype=np.float32)

    # Dummy actions
    actions[:] = 0.5

    # Warmup
    print("Warming up...")
    step_cpu(
        pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,
        masses, drag_coeffs, thrust_coeffs, target_vx, target_vy, target_vz, target_yaw_rate,
        vt_x, vt_y, vt_z, traj_params,
        pos_history, observations, rewards, reward_components, done_flags, step_counts, actions,
        num_agents, episode_length, env_ids
    )

    if HAS_CYTHON:
        step_cython(
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,
            masses, drag_coeffs, thrust_coeffs, target_vx, target_vy, target_vz, target_yaw_rate,
            vt_x, vt_y, vt_z, traj_params, target_trajectory,
            pos_history, observations, rewards, reward_components, done_flags, step_counts, actions,
            num_agents, episode_length, env_ids
        )

    # Benchmark CPU
    start = time.time()
    for _ in range(100):
        step_cpu(
            pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,
            masses, drag_coeffs, thrust_coeffs, target_vx, target_vy, target_vz, target_yaw_rate,
            vt_x, vt_y, vt_z, traj_params,
            pos_history, observations, rewards, reward_components, done_flags, step_counts, actions,
            num_agents, episode_length, env_ids
        )
    cpu_time = time.time() - start
    print(f"NumPy CPU Time (100 steps, {num_agents} agents): {cpu_time:.4f}s")

    if HAS_CYTHON:
        start = time.time()
        for _ in range(100):
            step_cython(
                pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,
                masses, drag_coeffs, thrust_coeffs, target_vx, target_vy, target_vz, target_yaw_rate,
                vt_x, vt_y, vt_z, traj_params, target_trajectory,
                pos_history, observations, rewards, reward_components, done_flags, step_counts, actions,
                num_agents, episode_length, env_ids
            )
        cython_time = time.time() - start
        print(f"Cython Time (100 steps, {num_agents} agents): {cython_time:.4f}s")
        print(f"Speedup: {cpu_time / cython_time:.2f}x")
    else:
        print("Cython extension not found.")

if __name__ == "__main__":
    benchmark()
