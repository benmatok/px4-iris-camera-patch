import numpy as np
import logging

try:
    from pycuda.compiler import SourceModule
    import pycuda.driver as cuda
    from warp_drive.environments.cuda_env_state import CUDAEnvironmentState
    _HAS_PYCUDA = True
except ImportError:
    _HAS_PYCUDA = False
    # Define a dummy class for inheritance if pycuda is missing
    class CUDAEnvironmentState:
        def __init__(self, **kwargs):
            pass

_DRONE_CUDA_SOURCE = """
// Helper functions
__device__ float terrain_height(float x, float y) {
    return 5.0f * sinf(0.1f * x) * cosf(0.1f * y);
}

// Pseudo-random number generator
__device__ float rand(unsigned int *seed) {
    *seed = *seed * 1664525 + 1013904223;
    return (float)(*seed) / 4294967296.0f;
}

extern "C" {
__global__ void step(
    float *pos_x, float *pos_y, float *pos_z,
    float *vel_x, float *vel_y, float *vel_z,
    float *roll, float *pitch, float *yaw,
    float *masses, float *drag_coeffs, float *thrust_coeffs,
    float *target_vx, float *target_vy, float *target_vz, float *target_yaw_rate,
    float *imu_history,
    float *pos_history,
    float *observations,
    float *rewards,
    float *done_flags,
    int *step_counts,
    const float *actions,
    const int num_agents,
    const int episode_length
)
{
    const int env_id = blockIdx.x;
    const int agent_id = threadIdx.x;

    if (agent_id >= num_agents) return;

    const int idx = env_id * num_agents + agent_id;
    const int action_idx = idx * 4; // 4 actions: Thrust, RollRate, PitchRate, YawRate

    // Physics constants
    const float dt = 0.01f; // 100 Hz simulation
    const float g = 9.81f;
    const int substeps = 10; // Run 10 sub-steps per 1 control step (0.1s total)

    // Load Dynamics Properties
    float mass = masses[idx];
    float drag_coeff = drag_coeffs[idx];
    float thrust_coeff = thrust_coeffs[idx];

    // Load State
    float px = pos_x[idx];
    float py = pos_y[idx];
    float pz = pos_z[idx];
    float vx = vel_x[idx];
    float vy = vel_y[idx];
    float vz = vel_z[idx];
    float r = roll[idx];
    float p = pitch[idx];
    float y_ang = yaw[idx];

    // Actions (Held constant during sub-steps)
    float thrust_cmd = actions[action_idx + 0];
    float roll_rate = actions[action_idx + 1];
    float pitch_rate = actions[action_idx + 2];
    float yaw_rate = actions[action_idx + 3];

    // Temporary storage for 10 samples * 6 dims
    // 60 floats. Registers/Local memory.
    float imu_buffer[60];

    for (int s = 0; s < substeps; s++) {
        // 1. Dynamics Update
        r += roll_rate * dt;
        p += pitch_rate * dt;
        y_ang += yaw_rate * dt;

        float max_thrust = 20.0f * thrust_coeff;
        float thrust_force = thrust_cmd * max_thrust;

        float sr = sinf(r); float cr = cosf(r);
        float sp = sinf(p); float cp = cosf(p);
        float sy = sinf(y_ang); float cy = cosf(y_ang);

        // Thrust Vector (Z-axis of body rotated to world)
        float ax_thrust = thrust_force * (cy * sp * cr + sy * sr) / mass;
        float ay_thrust = thrust_force * (sy * sp * cr - cy * sr) / mass;
        float az_thrust = thrust_force * (cp * cr) / mass;

        float az_gravity = -g;

        // Drag
        float ax_drag = -drag_coeff * vx;
        float ay_drag = -drag_coeff * vy;
        float az_drag = -drag_coeff * vz;

        float ax = ax_thrust + ax_drag;
        float ay = ay_thrust + ay_drag;
        float az = az_thrust + az_gravity + az_drag;

        vx += ax * dt;
        vy += ay * dt;
        vz += az * dt;

        px += vx * dt;
        py += vy * dt;
        pz += vz * dt;

        // Terrain Collision
        float terr_z = terrain_height(px, py);
        if (pz < terr_z) {
            pz = terr_z;
            vx = 0.0f; vy = 0.0f; vz = 0.0f;
        }

        // 2. Capture IMU for this sub-step
        // R^T calculation for Body Acc
        float r11 = cy * cp;
        float r12 = sy * cp;
        float r13 = -sp;
        float r21 = cy * sp * sr - sy * cr;
        float r22 = sy * sp * sr + cy * cr;
        float r23 = cp * sr;
        float r31 = cy * sp * cr + sy * sr;
        float r32 = sy * sp * cr - cy * sr;
        float r33 = cp * cr;

        float acc_w_x = ax_thrust + ax_drag;
        float acc_w_y = ay_thrust + ay_drag;
        float acc_w_z = az_thrust + az_drag;

        float acc_b_x = r11 * acc_w_x + r12 * acc_w_y + r13 * acc_w_z;
        float acc_b_y = r21 * acc_w_x + r22 * acc_w_y + r23 * acc_w_z;
        float acc_b_z = r31 * acc_w_x + r32 * acc_w_y + r33 * acc_w_z;

        int buf_idx = s * 6;
        imu_buffer[buf_idx + 0] = acc_b_x;
        imu_buffer[buf_idx + 1] = acc_b_y;
        imu_buffer[buf_idx + 2] = acc_b_z;
        imu_buffer[buf_idx + 3] = roll_rate;
        imu_buffer[buf_idx + 4] = pitch_rate;
        imu_buffer[buf_idx + 5] = yaw_rate;
    }

    // Terrain Collision (Final check or flag)
    float terr_z = terrain_height(px, py);
    bool collision = false;
    if (pz < terr_z) {
        pz = terr_z;
        collision = true;
    }

    // Store State
    pos_x[idx] = px; pos_y[idx] = py; pos_z[idx] = pz;
    vel_x[idx] = vx; vel_y[idx] = vy; vel_z[idx] = vz;
    roll[idx] = r; pitch[idx] = p; yaw[idx] = y_ang;

    // Step Count Management
    if (agent_id == 0) {
        step_counts[env_id] += 1;
    }
    int t = step_counts[env_id];

    // Store Position History
    // idx is global agent index.
    // Each agent has episode_length * 3 float storage.
    // We store at t-1 because t was just incremented?
    // step_counts starts at 0. First step makes it 1.
    // So we store at index t-1.
    if (t <= episode_length) {
        int pos_hist_idx = idx * episode_length * 3 + (t-1) * 3;
        pos_history[pos_hist_idx + 0] = px;
        pos_history[pos_hist_idx + 1] = py;
        pos_history[pos_hist_idx + 2] = pz;
    }

    // 2. Observations
    // Structure:
    // IMU History (300 steps * 6) + Target (4) = 1804 floats

    // Update History Buffer
    // Buffer size per agent: 300 * 6 = 1800
    // We shift left by 10 steps (60 floats) and append buffer.
    int hist_start = idx * 1800;

    // Shift: history[i] = history[i+60] for i < (290*6) = 1740
    for (int i = 0; i < 1740; i++) {
        imu_history[hist_start + i] = imu_history[hist_start + i + 60];
    }
    // Append
    for (int i = 0; i < 60; i++) {
        imu_history[hist_start + 1740 + i] = imu_buffer[i];
    }

    // Write to Observations
    int obs_offset = idx * 1804; // 1800 + 4
    int ptr = 0;

    // Copy History
    for (int i = 0; i < 1800; i++) {
        observations[obs_offset + ptr++] = imu_history[hist_start + i];
    }

    // Target Commands
    float tvx = target_vx[idx];
    float tvy = target_vy[idx];
    float tvz = target_vz[idx];
    float tyr = target_yaw_rate[idx];

    observations[obs_offset + ptr++] = tvx;
    observations[obs_offset + ptr++] = tvy;
    observations[obs_offset + ptr++] = tvz;
    observations[obs_offset + ptr++] = tyr;

    // 3. Rewards
    // Use final state for reward? Or average? Using final state is simpler.
    // Convert World Velocity to Body Velocity (Final)
    // We need to recompute R matrix for final R/P/Y
    float sr = sinf(r); float cr = cosf(r);
    float sp = sinf(p); float cp = cosf(p);
    float sy = sinf(y_ang); float cy = cosf(y_ang);

    float r11 = cy * cp;
    float r12 = sy * cp;
    float r13 = -sp;
    float r21 = cy * sp * sr - sy * cr;
    float r22 = sy * sp * sr + cy * cr;
    float r23 = cp * sr;
    float r31 = cy * sp * cr + sy * sr;
    float r32 = sy * sp * cr - cy * sr;
    float r33 = cp * cr;

    float vx_b = r11 * vx + r12 * vy + r13 * vz;
    float vy_b = r21 * vx + r22 * vy + r23 * vz;
    float vz_b = r31 * vx + r32 * vy + r33 * vz;

    // Target Tracking Error
    float v_err_sq = (vx_b - tvx)*(vx_b - tvx) + (vy_b - tvy)*(vy_b - tvy) + (vz_b - tvz)*(vz_b - tvz);
    float yaw_rate_err_sq = (yaw_rate - tyr)*(yaw_rate - tyr);

    float reward = 0.0f;

    // Reward for matching target velocity
    reward += 1.0f * expf(-2.0f * v_err_sq);
    reward += 0.5f * expf(-2.0f * yaw_rate_err_sq);

    // Penalties
    reward -= 0.01f * (r*r + p*p);
    if (fabsf(r) > 1.0f || fabsf(p) > 1.0f) reward -= 0.1f;

    if (collision) reward -= 10.0f;

    // Survival reward?
    reward += 0.1f;

    rewards[idx] = reward;

    // Done Flag
    if (t >= episode_length) {
        done_flags[idx] = 1.0f;
    } else {
        done_flags[idx] = 0.0f;
    }
}

__global__ void reset(
    float *pos_x, float *pos_y, float *pos_z,
    float *vel_x, float *vel_y, float *vel_z,
    float *roll, float *pitch, float *yaw,
    float *masses, float *drag_coeffs, float *thrust_coeffs,
    float *target_vx, float *target_vy, float *target_vz, float *target_yaw_rate,
    float *imu_history,
    float *pos_history,
    int *rng_states,
    int *step_counts,
    const int num_agents,
    const int *reset_indices
)
{
    int env_id = reset_indices[blockIdx.x];
    int agent_id = threadIdx.x;

    if (agent_id >= num_agents) return;
    int idx = env_id * num_agents + agent_id;

    // Random Init
    unsigned int seed = rng_states[idx] + idx + 12345 + step_counts[env_id]*6789;

    // Randomize Dynamics
    masses[idx] = 0.5f + rand(&seed) * 1.0f; // 0.5 to 1.5
    drag_coeffs[idx] = 0.05f + rand(&seed) * 0.1f; // 0.05 to 0.15
    thrust_coeffs[idx] = 0.8f + rand(&seed) * 0.4f; // 0.8 to 1.2

    // Randomize Target Command
    float rnd_cmd = rand(&seed);
    float tvx = 0.0f; float tvy = 0.0f; float tvz = 0.0f; float tyr = 0.0f;

    if (rnd_cmd < 0.2f) { // Hover
         tvx = 0.0f; tvy = 0.0f; tvz = 0.0f;
    } else if (rnd_cmd < 0.3f) { // Forward
         tvx = 1.0f;
    } else if (rnd_cmd < 0.4f) { // Backward
         tvx = -1.0f;
    } else if (rnd_cmd < 0.5f) { // Left (Slide)
         tvy = 1.0f;
    } else if (rnd_cmd < 0.6f) { // Right (Slide)
         tvy = -1.0f;
    } else if (rnd_cmd < 0.7f) { // Up
         tvz = 1.0f;
    } else if (rnd_cmd < 0.8f) { // Down
         tvz = -1.0f;
    } else if (rnd_cmd < 0.9f) { // Rot Left
         tyr = 1.0f;
    } else { // Rot Right
         tyr = -1.0f;
    }

    target_vx[idx] = tvx;
    target_vy[idx] = tvy;
    target_vz[idx] = tvz;
    target_yaw_rate[idx] = tyr;

    // Reset History (Size 1800)
    int hist_start = idx * 1800;
    for (int i = 0; i < 1800; i++) {
        imu_history[hist_start + i] = 0.0f;
    }

    // Update seed state
    rng_states[idx] = seed;

    // Reset State
    pos_x[idx] = 0.0f;
    pos_y[idx] = 0.0f;
    pos_z[idx] = 10.0f;

    vel_x[idx] = 0.0f; vel_y[idx] = 0.0f; vel_z[idx] = 0.0f;
    roll[idx] = 0.0f; pitch[idx] = 0.0f; yaw[idx] = 0.0f;

    if (agent_id == 0) {
        step_counts[env_id] = 0;
    }
}
}
"""

# -----------------------------------------------------------------------------
# Pure NumPy Implementation (CPU Fallback - Safe)
# -----------------------------------------------------------------------------

def terrain_height_cpu(x, y):
    return 5.0 * np.sin(0.1 * x) * np.cos(0.1 * y)

def step_cpu(
    pos_x, pos_y, pos_z,
    vel_x, vel_y, vel_z,
    roll, pitch, yaw,
    masses, drag_coeffs, thrust_coeffs,
    target_vx, target_vy, target_vz, target_yaw_rate,
    imu_history,
    pos_history,
    observations,
    rewards,
    done_flags,
    step_counts,
    actions,
    num_agents,
    episode_length,
    env_ids,
):
    # Vectorized NumPy implementation
    # actions: (num_agents * 4,) flattened, need to reshape

    total_agents = len(pos_x)

    # Reshape actions to (N, 4)
    actions_reshaped = actions.reshape(total_agents, 4)
    thrust_cmd = actions_reshaped[:, 0]
    roll_rate = actions_reshaped[:, 1]
    pitch_rate = actions_reshaped[:, 2]
    yaw_rate = actions_reshaped[:, 3]

    dt = 0.01
    g = 9.81
    substeps = 10

    # Pre-allocate buffer for IMU data: (substeps, total_agents, 6)
    # We will transpose later to (total_agents, substeps * 6)
    imu_buffer_list = []

    for _ in range(substeps):
        roll += roll_rate * dt
        pitch += pitch_rate * dt
        yaw += yaw_rate * dt

        max_thrust = 20.0 * thrust_coeffs
        thrust_force = thrust_cmd * max_thrust

        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sy, cy = np.sin(yaw), np.cos(yaw)

        # Thrust Vector (Z-axis of body rotated to world)
        # Vectorized ops
        ax_thrust = thrust_force * (cy * sp * cr + sy * sr) / masses
        ay_thrust = thrust_force * (sy * sp * cr - cy * sr) / masses
        az_thrust = thrust_force * (cp * cr) / masses

        az_gravity = -g

        # Drag
        ax_drag = -drag_coeffs * vel_x
        ay_drag = -drag_coeffs * vel_y
        az_drag = -drag_coeffs * vel_z

        ax = ax_thrust + ax_drag
        ay = ay_thrust + ay_drag
        az = az_thrust + az_gravity + az_drag

        vel_x += ax * dt
        vel_y += ay * dt
        vel_z += az * dt

        pos_x += vel_x * dt
        pos_y += vel_y * dt
        pos_z += vel_z * dt

        # Terrain Collision
        terr_z = terrain_height_cpu(pos_x, pos_y)
        mask = pos_z < terr_z
        pos_z[mask] = terr_z[mask]
        vel_x[mask] = 0.0
        vel_y[mask] = 0.0
        vel_z[mask] = 0.0

        # IMU Capture
        r11 = cy * cp
        r12 = sy * cp
        r13 = -sp
        r21 = cy * sp * sr - sy * cr
        r22 = sy * sp * sr + cy * cr
        r23 = cp * sr
        r31 = cy * sp * cr + sy * sr
        r32 = sy * sp * cr - cy * sr
        r33 = cp * cr

        acc_w_x = ax_thrust + ax_drag
        acc_w_y = ay_thrust + ay_drag
        acc_w_z = az_thrust + az_drag

        acc_b_x = r11 * acc_w_x + r12 * acc_w_y + r13 * acc_w_z
        acc_b_y = r21 * acc_w_x + r22 * acc_w_y + r23 * acc_w_z
        acc_b_z = r31 * acc_w_x + r32 * acc_w_y + r33 * acc_w_z

        # Stack for this substep: (N, 6)
        imu_step = np.stack([acc_b_x, acc_b_y, acc_b_z, roll_rate, pitch_rate, yaw_rate], axis=1)
        imu_buffer_list.append(imu_step)

    # Collision Check Final
    terr_z = terrain_height_cpu(pos_x, pos_y)
    collision_mask = pos_z < terr_z
    pos_z[collision_mask] = terr_z[collision_mask]

    # Step Counts
    # Assuming step_counts has shape (1,) if we treat all agents in one env,
    # OR shape (num_envs,) if parallel.
    # train_drone.py CPUTrainer: uses env.num_agents=20.
    # reset_indices=[0] means env_id=0.
    # So step_counts is probably length 1?
    # Actually DroneEnv allocs step_counts based on num_agents? No.
    # "step_counts": {"shape": (self.num_agents,), "dtype": np.int32} in `get_data_dictionary`
    # But kernel uses `step_counts[env_id]`.
    # If num_agents=20, and we treat it as 1 env with 20 agents?
    # reset_cpu iterates `range(num_agents)`. `idx = env_id * num_agents + agent_id`.
    # If `num_agents` passed to `reset_cpu` matches the total agents we allocated, then `env_id` must be 0 if we have only 1 env block.
    # Let's assume env_id=0 for all agents.

    # We only increment step_count once per environment, not per agent.
    # But here we are vectorized.
    # If we have only 1 env (id 0), we just increment step_counts[0].
    # But if we support multiple envs, we need to handle it.
    # For CPUTrainer, we use env_ids_to_step = np.array([0]).

    for env_id in env_ids:
        step_counts[env_id] += 1

    t = step_counts[0] # Assuming single env for now based on CPUTrainer usage

    # Store History
    if t <= episode_length:
        # Vectorized assignment to history
        # pos_history shape: (total_agents * episode_length * 3)
        # We need to reshape or index cleverly.
        # reshape to (total_agents, episode_length, 3)
        ph_reshaped = pos_history.reshape(total_agents, episode_length, 3)
        ph_reshaped[:, t-1, 0] = pos_x
        ph_reshaped[:, t-1, 1] = pos_y
        ph_reshaped[:, t-1, 2] = pos_z

    # Update IMU History
    # imu_buffer_list is list of substeps (10), each (N, 6).
    # We want (N, 60).
    imu_buffer = np.concatenate(imu_buffer_list, axis=1) # (N, 60)

    # imu_history shape in dict: (N * 300 * 6) -> 1800 floats per agent
    # Reshape for easier access
    ih_reshaped = imu_history.reshape(total_agents, 1800)

    # Shift left by 60
    ih_reshaped[:, :-60] = ih_reshaped[:, 60:]
    # Append new
    ih_reshaped[:, -60:] = imu_buffer

    # Observations
    # observations shape (N, 1804)
    observations[:, :1800] = ih_reshaped
    observations[:, 1800] = target_vx
    observations[:, 1801] = target_vy
    observations[:, 1802] = target_vz
    observations[:, 1803] = target_yaw_rate

    # Rewards
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    r11 = cy * cp
    r12 = sy * cp
    r13 = -sp
    r21 = cy * sp * sr - sy * cr
    r22 = sy * sp * sr + cy * cr
    r23 = cp * sr
    r31 = cy * sp * cr + sy * sr
    r32 = sy * sp * cr - cy * sr
    r33 = cp * cr

    vx_b = r11 * vel_x + r12 * vel_y + r13 * vel_z
    vy_b = r21 * vel_x + r22 * vel_y + r23 * vel_z
    vz_b = r31 * vel_x + r32 * vel_y + r33 * vel_z

    v_err_sq = (vx_b - target_vx)**2 + (vy_b - target_vy)**2 + (vz_b - target_vz)**2
    yaw_rate_err_sq = (yaw_rate - target_yaw_rate)**2

    r_val = 1.0 * np.exp(-2.0 * v_err_sq) + 0.5 * np.exp(-2.0 * yaw_rate_err_sq)
    r_val -= 0.01 * (roll**2 + pitch**2)

    mask_unstable = (np.abs(roll) > 1.0) | (np.abs(pitch) > 1.0)
    r_val[mask_unstable] -= 0.1

    r_val[collision_mask] -= 10.0
    r_val += 0.1

    rewards[:] = r_val

    if t >= episode_length:
        done_flags[:] = 1.0
    else:
        done_flags[:] = 0.0

def reset_cpu(
    pos_x, pos_y, pos_z,
    vel_x, vel_y, vel_z,
    roll, pitch, yaw,
    masses, drag_coeffs, thrust_coeffs,
    target_vx, target_vy, target_vz, target_yaw_rate,
    imu_history,
    pos_history,
    rng_states,
    step_counts,
    num_agents,
    reset_indices
):
    # Vectorized Reset
    # reset_indices: array of env_ids to reset.
    # Since we assume 1 env block (or consistent layout), we reset all agents corresponding to these envs.
    # CPUTrainer logic: env_id=0, num_agents=20.

    # We iterate over env_ids, but since we likely only have [0], we can optimize.
    # For full correctness with "reset_indices":

    for env_id in reset_indices:
        # Range of agents for this env
        start_idx = env_id * num_agents
        end_idx = start_idx + num_agents

        # Vectorized randoms
        masses[start_idx:end_idx] = 0.5 + np.random.rand(num_agents) * 1.0
        drag_coeffs[start_idx:end_idx] = 0.05 + np.random.rand(num_agents) * 0.1
        thrust_coeffs[start_idx:end_idx] = 0.8 + np.random.rand(num_agents) * 0.4

        rnd_cmd = np.random.rand(num_agents)
        tvx = np.zeros(num_agents)
        tvy = np.zeros(num_agents)
        tvz = np.zeros(num_agents)
        tyr = np.zeros(num_agents)

        # Masks
        m1 = rnd_cmd < 0.2
        m2 = (rnd_cmd >= 0.2) & (rnd_cmd < 0.3)
        m3 = (rnd_cmd >= 0.3) & (rnd_cmd < 0.4)
        m4 = (rnd_cmd >= 0.4) & (rnd_cmd < 0.5)
        m5 = (rnd_cmd >= 0.5) & (rnd_cmd < 0.6)
        m6 = (rnd_cmd >= 0.6) & (rnd_cmd < 0.7)
        m7 = (rnd_cmd >= 0.7) & (rnd_cmd < 0.8)
        m8 = (rnd_cmd >= 0.8) & (rnd_cmd < 0.9)
        m9 = (rnd_cmd >= 0.9)

        tvx[m2] = 1.0; tvx[m3] = -1.0
        tvy[m4] = 1.0; tvy[m5] = -1.0
        tvz[m6] = 1.0; tvz[m7] = -1.0
        tyr[m8] = 1.0; tyr[m9] = -1.0

        target_vx[start_idx:end_idx] = tvx
        target_vy[start_idx:end_idx] = tvy
        target_vz[start_idx:end_idx] = tvz
        target_yaw_rate[start_idx:end_idx] = tyr

        # Reset History
        # imu_history shape (N * 1800)
        # We need to zero out the slice for these agents.
        # But imu_history is flat.
        # start offset: start_idx * 1800
        # length: num_agents * 1800
        imu_history[start_idx*1800 : end_idx*1800] = 0.0

        pos_x[start_idx:end_idx] = 0.0
        pos_y[start_idx:end_idx] = 0.0
        pos_z[start_idx:end_idx] = 10.0

        vel_x[start_idx:end_idx] = 0.0
        vel_y[start_idx:end_idx] = 0.0
        vel_z[start_idx:end_idx] = 0.0

        roll[start_idx:end_idx] = 0.0
        pitch[start_idx:end_idx] = 0.0
        yaw[start_idx:end_idx] = 0.0

        step_counts[env_id] = 0

class DroneEnv(CUDAEnvironmentState):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_agents = kwargs.get("num_agents", 1)
        self.episode_length = kwargs.get("episode_length", 100)
        self.agent_ids = [f"drone_{i}" for i in range(self.num_agents)]
        self.use_cuda = _HAS_PYCUDA and kwargs.get("use_cuda", True)

        if self.use_cuda:
            logging.info("DroneEnv: Using CUDA backend.")
            # Compile CUDA Kernel
            self.cuda_module = SourceModule(_DRONE_CUDA_SOURCE, no_extern_c=True)
            self.step_function = self.cuda_module.get_function("step")
            self.reset_function = self.cuda_module.get_function("reset")
        else:
            logging.info("DroneEnv: Using CPU/NumPy backend.")
            self.step_function = step_cpu
            self.reset_function = reset_cpu

    def get_environment_info(self):
        return {
            "n_agents": self.num_agents,
            "episode_length": self.episode_length,
            "core_state_names": [
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "roll", "pitch", "yaw",
                "masses", "drag_coeffs", "thrust_coeffs",
                "target_vx", "target_vy", "target_vz", "target_yaw_rate",
                "imu_history",
                "pos_history",
                "rng_states",
                "step_counts"
            ],
        }

    def get_data_dictionary(self):
        # Explicitly define data shapes and types for allocation
        return {
            "masses": {"shape": (self.num_agents,), "dtype": np.float32},
            "drag_coeffs": {"shape": (self.num_agents,), "dtype": np.float32},
            "thrust_coeffs": {"shape": (self.num_agents,), "dtype": np.float32},
            "target_vx": {"shape": (self.num_agents,), "dtype": np.float32},
            "target_vy": {"shape": (self.num_agents,), "dtype": np.float32},
            "target_vz": {"shape": (self.num_agents,), "dtype": np.float32},
            "target_yaw_rate": {"shape": (self.num_agents,), "dtype": np.float32},
            "imu_history": {"shape": (self.num_agents * 300 * 6,), "dtype": np.float32}, # 1800 floats per agent
            "pos_history": {"shape": (self.num_agents * self.episode_length * 3,), "dtype": np.float32},
            "rng_states": {"shape": (self.num_agents,), "dtype": np.int32},
             "pos_x": {"shape": (self.num_agents,), "dtype": np.float32},
             "pos_y": {"shape": (self.num_agents,), "dtype": np.float32},
             "pos_z": {"shape": (self.num_agents,), "dtype": np.float32},
             "vel_x": {"shape": (self.num_agents,), "dtype": np.float32},
             "vel_y": {"shape": (self.num_agents,), "dtype": np.float32},
             "vel_z": {"shape": (self.num_agents,), "dtype": np.float32},
             "roll": {"shape": (self.num_agents,), "dtype": np.float32},
             "pitch": {"shape": (self.num_agents,), "dtype": np.float32},
             "yaw": {"shape": (self.num_agents,), "dtype": np.float32},
             "step_counts": {"shape": (self.num_agents,), "dtype": np.int32},
             "done_flags": {"shape": (self.num_agents,), "dtype": np.float32},
             "rewards": {"shape": (self.num_agents,), "dtype": np.float32},
             "observations": {"shape": (self.num_agents, 1804), "dtype": np.float32}, # 1800 + 4
        }

    def get_action_space(self):
        return (self.num_agents, 4)

    def get_observation_space(self):
        # 300 * 6 + 4 = 1804
        return (self.num_agents, 1804)

    def get_reward_signature(self): return (self.num_agents,)

    def get_state_names(self):
        return [
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "roll", "pitch", "yaw",
                "masses", "drag_coeffs", "thrust_coeffs",
                "target_vx", "target_vy", "target_vz", "target_yaw_rate",
                "imu_history",
                "pos_history",
                "rng_states",
                "observations", "rewards", "done_flags",
                "step_counts"
        ]

    def get_constants(self):
        return {
            "num_agents": self.num_agents,
            "episode_length": self.episode_length
        }

    def get_step_function(self): return self.step_function
    def get_reset_function(self): return self.reset_function

    def get_step_function_kwargs(self):
        return {
            "pos_x": "pos_x", "pos_y": "pos_y", "pos_z": "pos_z",
            "vel_x": "vel_x", "vel_y": "vel_y", "vel_z": "vel_z",
            "roll": "roll", "pitch": "pitch", "yaw": "yaw",
            "masses": "masses", "drag_coeffs": "drag_coeffs", "thrust_coeffs": "thrust_coeffs",
            "target_vx": "target_vx", "target_vy": "target_vy", "target_vz": "target_vz", "target_yaw_rate": "target_yaw_rate",
            "imu_history": "imu_history",
            "pos_history": "pos_history",
            "observations": "observations",
            "rewards": "rewards",
            "done_flags": "done_flags",
            "step_counts": "step_counts",
            "actions": "actions",
            "num_agents": "num_agents",
            "episode_length": "episode_length",
        }

    def get_reset_function_kwargs(self):
        return {
            "pos_x": "pos_x", "pos_y": "pos_y", "pos_z": "pos_z",
            "vel_x": "vel_x", "vel_y": "vel_y", "vel_z": "vel_z",
            "roll": "roll", "pitch": "pitch", "yaw": "yaw",
            "masses": "masses", "drag_coeffs": "drag_coeffs", "thrust_coeffs": "thrust_coeffs",
            "target_vx": "target_vx", "target_vy": "target_vy", "target_vz": "target_vz", "target_yaw_rate": "target_yaw_rate",
            "imu_history": "imu_history",
            "pos_history": "pos_history",
            "rng_states": "rng_states",
            "step_counts": "step_counts",
            "num_agents": "num_agents",
            "reset_indices": "reset_indices"
        }
