import numpy as np
from pycuda.compiler import SourceModule
from warp_drive.environments.cuda_env_state import CUDAEnvironmentState

_DRONE_CUDA_SOURCE = """
// Helper functions
__device__ float terrain_height(float x, float y) {
    return 5.0f * sinf(0.1f * x) * cosf(0.1f * y);
}

extern "C" {
__global__ void step(
    float *pos_x, float *pos_y, float *pos_z,
    float *vel_x, float *vel_y, float *vel_z,
    float *roll, float *pitch, float *yaw,
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
    const float dt = 0.1f;
    const float g = 9.81f;
    const float mass = 1.0f;
    const float drag_coeff = 0.1f;

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

    // Actions
    float thrust_cmd = actions[action_idx + 0];
    float roll_rate = actions[action_idx + 1];
    float pitch_rate = actions[action_idx + 2];
    float yaw_rate = actions[action_idx + 3];

    // 1. Dynamics Update
    r += roll_rate * dt;
    p += pitch_rate * dt;
    y_ang += yaw_rate * dt;

    float max_thrust = 20.0f;
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
    bool collision = false;
    if (pz < terr_z) {
        pz = terr_z;
        vx = 0.0f; vy = 0.0f; vz = 0.0f;
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

    // 2. Observations
    // Structure: [State(9), IMU(6), Camera(64)] -> Total 79 floats
    int obs_offset = idx * 79;
    int ptr = 0;

    // State (9)
    observations[obs_offset + ptr++] = px;
    observations[obs_offset + ptr++] = py;
    observations[obs_offset + ptr++] = pz;
    observations[obs_offset + ptr++] = vx;
    observations[obs_offset + ptr++] = vy;
    observations[obs_offset + ptr++] = vz;
    observations[obs_offset + ptr++] = r;
    observations[obs_offset + ptr++] = p;
    observations[obs_offset + ptr++] = y_ang;

    // IMU (6) - Accelerometer (World Frame approximation) and Gyro
    float acc_w_x = ax_thrust + ax_drag;
    float acc_w_y = ay_thrust + ay_drag;
    float acc_w_z = az_thrust + az_drag;

    observations[obs_offset + ptr++] = acc_w_x;
    observations[obs_offset + ptr++] = acc_w_y;
    observations[obs_offset + ptr++] = acc_w_z;
    observations[obs_offset + ptr++] = roll_rate;
    observations[obs_offset + ptr++] = pitch_rate;
    observations[obs_offset + ptr++] = yaw_rate;

    // Camera (8x8 = 64) - Depth Map
    float fov_half_size = 5.0f;
    int cam_res = 8;
    for (int cy_i = 0; cy_i < cam_res; cy_i++) {
        for (int cx_i = 0; cx_i < cam_res; cx_i++) {
            float u = (float)cx_i / (cam_res - 1) * 2.0f - 1.0f;
            float v = (float)cy_i / (cam_res - 1) * 2.0f - 1.0f;

            float sample_x = px + u * fov_half_size;
            float sample_y = py + v * fov_half_size;

            float h = terrain_height(sample_x, sample_y);
            float dist = pz - h;
            observations[obs_offset + ptr++] = dist;
        }
    }

    // 3. Rewards
    float target_z = 10.0f;
    float dist_xy = sqrtf(px*px + py*py);
    float dist_z = fabsf(pz - target_z);

    float reward = 0.0f;
    reward -= dist_xy * 0.1f;
    reward -= dist_z * 0.5f;
    reward -= (roll*roll + pitch*pitch) * 0.1f; // Stability
    reward -= 0.01f * (vx*vx + vy*vy + vz*vz); // Damping
    if (collision) reward -= 100.0f;

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
    int *step_counts,
    const int num_agents,
    const int *reset_indices
)
{
    int env_id = reset_indices[blockIdx.x];
    int agent_id = threadIdx.x;

    if (agent_id >= num_agents) return;
    int idx = env_id * num_agents + agent_id;

    // Reset Logic
    pos_x[idx] = 0.0f;
    pos_y[idx] = 0.0f;
    pos_z[idx] = 10.0f; // Reset to hover height

    vel_x[idx] = 0.0f; vel_y[idx] = 0.0f; vel_z[idx] = 0.0f;
    roll[idx] = 0.0f; pitch[idx] = 0.0f; yaw[idx] = 0.0f;

    if (agent_id == 0) {
        step_counts[env_id] = 0;
    }
}
}
"""

class DroneEnv(CUDAEnvironmentState):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_agents = kwargs.get("num_agents", 1)
        self.episode_length = kwargs.get("episode_length", 100)
        self.agent_ids = [f"drone_{i}" for i in range(self.num_agents)]

        self.cam_res = 8
        self.cam_pixels = self.cam_res * self.cam_res

        # Compile CUDA Kernel
        self.cuda_module = SourceModule(_DRONE_CUDA_SOURCE, no_extern_c=True)
        self.step_function = self.cuda_module.get_function("step")
        self.reset_function = self.cuda_module.get_function("reset")

    def get_environment_info(self):
        return {
            "n_agents": self.num_agents,
            "episode_length": self.episode_length,
            "core_state_names": [
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "roll", "pitch", "yaw",
                "step_counts"
            ],
        }

    def get_action_space(self):
        return (self.num_agents, 4)

    def get_observation_space(self):
        return (self.num_agents, 9 + 6 + self.cam_pixels)

    def get_reward_signature(self): return (self.num_agents,)

    def get_state_names(self):
        return [
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "roll", "pitch", "yaw",
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
            "step_counts": "step_counts",
            "num_agents": "num_agents",
            "reset_indices": "reset_indices" # Passed by EnvWrapper
        }
