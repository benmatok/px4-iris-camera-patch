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

# Try importing the Cython extension
try:
    from drone_env.drone_cython import step_cython, reset_cython
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False

_DRONE_CUDA_SOURCE = """
// Placeholder for CUDA source - Not updated for new observation space
// as per user instruction to focus on CPU/Cython/C++.
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
    vt_x, vt_y, vt_z,
    traj_params, # New: Trajectory Parameters
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
    # Vectorized Pure NumPy implementation
    # actions: (num_agents * 4,)

    # Reshape actions for easier access: (num_agents, 4)
    actions_reshaped = actions.reshape(num_agents, 4)
    thrust_cmd = actions_reshaped[:, 0]
    roll_rate = actions_reshaped[:, 1]
    pitch_rate = actions_reshaped[:, 2]
    yaw_rate = actions_reshaped[:, 3]

    dt = 0.01
    g = 9.81
    substeps = 10

    # Local copies of state (NumPy arrays)
    px, py, pz = pos_x, pos_y, pos_z
    vx, vy, vz = vel_x, vel_y, vel_z
    r, p, y_ang = roll, pitch, yaw

    # Update Virtual Target using Trajectory Params
    # traj_params: (num_agents, 10)
    # 0:Ax, 1:Fx, 2:Px, 3:Ay, 4:Fy, 5:Py, 6:Az, 7:Fz, 8:Pz, 9:Oz
    t = step_counts[0] + 1
    t_f = float(t)

    tp = traj_params # alias

    # x = Ax * sin(Fx * t + Px)
    vtx_val = tp[:, 0] * np.sin(tp[:, 1] * t_f + tp[:, 2])
    # y = Ay * sin(Fy * t + Py) (using sin for consistency, phase controls cos)
    vty_val = tp[:, 3] * np.sin(tp[:, 4] * t_f + tp[:, 5])
    # z = Oz + Az * sin(Fz * t + Pz)
    vtz_val = tp[:, 9] + tp[:, 6] * np.sin(tp[:, 7] * t_f + tp[:, 8])

    vt_x[:] = vtx_val
    vt_y[:] = vty_val
    vt_z[:] = vtz_val

    # History Update
    # We capture samples at substep 4 (0.05s) and 9 (0.10s)
    # 2 samples * 3 channels = 6 floats
    captured_samples = np.zeros((num_agents, 2, 3), dtype=np.float32)

    for s in range(substeps):
        # 1. Dynamics Update
        r += roll_rate * dt
        p += pitch_rate * dt
        y_ang += yaw_rate * dt

        max_thrust = 20.0 * thrust_coeffs
        thrust_force = thrust_cmd * max_thrust

        sr, cr = np.sin(r), np.cos(r)
        sp, cp = np.sin(p), np.cos(p)
        sy, cy = np.sin(y_ang), np.cos(y_ang)

        ax_thrust = thrust_force * (cy * sp * cr + sy * sr) / masses
        ay_thrust = thrust_force * (sy * sp * cr - cy * sr) / masses
        az_thrust = thrust_force * (cp * cr) / masses

        az_gravity = -g

        ax_drag = -drag_coeffs * vx
        ay_drag = -drag_coeffs * vy
        az_drag = -drag_coeffs * vz

        ax = ax_thrust + ax_drag
        ay = ay_thrust + ay_drag
        az = az_thrust + az_gravity + az_drag

        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        px += vx * dt
        py += vy * dt
        pz += vz * dt

        # Terrain Collision
        terr_z = terrain_height_cpu(px, py)

        # Vectorized collision check
        underground = pz < terr_z
        pz = np.where(underground, terr_z, pz)
        vx = np.where(underground, 0.0, vx)
        vy = np.where(underground, 0.0, vy)
        vz = np.where(underground, 0.0, vz)

        # Capture History Samples
        if s == 4 or s == 9:
            idx = 0 if s == 4 else 1
            # Add noise (Uniform +/- 0.02 rad ~ 1 deg)
            noise = (np.random.rand(num_agents, 3) - 0.5) * 0.04
            captured_samples[:, idx, 0] = r + noise[:, 0]
            captured_samples[:, idx, 1] = p + noise[:, 1]
            captured_samples[:, idx, 2] = y_ang + noise[:, 2]

    # Terrain Collision (Final check)
    terr_z = terrain_height_cpu(px, py)
    underground = pz < terr_z
    pz = np.where(underground, terr_z, pz)
    collision = underground # boolean array

    # Update State Arrays
    pos_x[:] = px
    pos_y[:] = py
    pos_z[:] = pz
    vel_x[:] = vx
    vel_y[:] = vy
    vel_z[:] = vz
    roll[:] = r
    pitch[:] = p
    yaw[:] = y_ang

    # Step Counts
    step_counts[0] += 1
    t = step_counts[0]

    # Store Position History
    if t <= episode_length:
        ph_reshaped = pos_history.reshape(num_agents, episode_length, 3)
        ph_reshaped[:, t-1, 0] = px
        ph_reshaped[:, t-1, 1] = py
        ph_reshaped[:, t-1, 2] = pz

    # Update Observations
    # observations: (num_agents, 608)
    # History: 0 to 600.
    # Shift: obs[:, 0:594] = obs[:, 6:600]
    observations[:, 0:594] = observations[:, 6:600]

    # Append new data
    new_data = captured_samples.reshape(num_agents, 6)
    observations[:, 594:600] = new_data

    # Update targets (600:604)
    observations[:, 600] = target_vx
    observations[:, 601] = target_vy
    observations[:, 602] = target_vz
    observations[:, 603] = target_yaw_rate

    # Update Tracker Features (604:608)
    dx_w = vtx_val - px
    dy_w = vty_val - py
    dz_w = vtz_val - pz

    sr, cr = np.sin(r), np.cos(r)
    sp, cp = np.sin(p), np.cos(p)
    sy, cy = np.sin(y_ang), np.cos(y_ang)

    r11 = cy * cp
    r12 = sy * cp
    r13 = -sp
    r21 = cy * sp * sr - sy * cr
    r22 = sy * sp * sr + cy * cr
    r23 = cp * sr
    r31 = cy * sp * cr + sy * sr
    r32 = sy * sp * cr - cy * sr
    r33 = cp * cr

    xb = r11 * dx_w + r12 * dy_w + r13 * dz_w
    yb = r21 * dx_w + r22 * dy_w + r23 * dz_w
    zb = r31 * dx_w + r32 * dy_w + r33 * dz_w

    s30 = 0.5
    c30 = 0.866025

    xc = yb
    yc = s30 * xb + c30 * zb
    zc = c30 * xb - s30 * zb

    zc_safe = np.maximum(zc, 0.1)
    u = xc / zc_safe
    v = yc / zc_safe
    size = 10.0 / (zc*zc + 1.0)

    w2 = roll_rate**2 + pitch_rate**2 + yaw_rate**2
    conf = np.exp(-0.1 * w2)
    # Behind camera check
    conf = np.where((c30 * xb - s30 * zb) < 0.0, 0.0, conf)

    observations[:, 604] = u
    observations[:, 605] = v
    observations[:, 606] = size
    observations[:, 607] = conf

    # -------------------------------------------------------------------------
    # Improved Reward Function
    # -------------------------------------------------------------------------
    # 1. Base Velocity/Yaw Reward (Legacy)
    vx_b = r11 * vx + r12 * vy + r13 * vz
    vy_b = r21 * vx + r22 * vy + r23 * vz
    vz_b = r31 * vx + r32 * vy + r33 * vz

    v_err_sq = (vx_b - target_vx)**2 + (vy_b - target_vy)**2 + (vz_b - target_vz)**2
    yaw_rate_err_sq = (yaw_rate - target_yaw_rate)**2

    rew = np.zeros(num_agents, dtype=np.float32)
    rew += 1.0 * np.exp(-2.0 * v_err_sq)
    rew += 0.5 * np.exp(-2.0 * yaw_rate_err_sq)

    # 2. Visual Servoing Reward (Keep target in center)
    # Penalize u^2 + v^2
    rew_vis = np.exp(-2.0 * (u**2 + v**2))
    rew += 0.5 * rew_vis

    # 3. Range Reward (Keep optimal size)
    # Target size = 1.0 (approx 3m distance)
    rew_range = np.exp(-2.0 * (size - 1.0)**2)
    rew += 0.5 * rew_range

    # 4. Smoothness Reward (Penalize high control rates)
    # w2 is sum of squared rates
    rew_smooth = np.exp(-0.1 * w2)
    rew += 0.2 * rew_smooth

    # Penalize high angles (Stability)
    rew -= 0.01 * (r*r + p*p)

    unstable = (np.abs(r) > 1.0) | (np.abs(p) > 1.0)
    rew = np.where(unstable, rew - 0.1, rew)

    rew = np.where(collision, rew - 10.0, rew)
    rew += 0.1 # Survival bonus

    rewards[:] = rew

    done_flags[:] = np.where(t >= episode_length, 1.0, 0.0)

def reset_cpu(
    pos_x, pos_y, pos_z,
    vel_x, vel_y, vel_z,
    roll, pitch, yaw,
    masses, drag_coeffs, thrust_coeffs,
    target_vx, target_vy, target_vz, target_yaw_rate,
    traj_params, # New
    pos_history,
    observations,
    rng_states,
    step_counts,
    num_agents,
    reset_indices
):
    # Vectorized Reset
    masses[:] = 0.5 + np.random.rand(num_agents) * 1.0
    drag_coeffs[:] = 0.05 + np.random.rand(num_agents) * 0.1
    thrust_coeffs[:] = 0.8 + np.random.rand(num_agents) * 0.4

    # Initialize Trajectory Parameters (Lissajous / Complex)
    # 0:Ax, 1:Fx, 2:Px, 3:Ay, 4:Fy, 5:Py, 6:Az, 7:Fz, 8:Pz, 9:Oz
    traj_params[:, 0] = 3.0 + np.random.rand(num_agents) * 4.0 # Ax: 3-7
    traj_params[:, 1] = 0.03 + np.random.rand(num_agents) * 0.07 # Fx: 0.03-0.1
    traj_params[:, 2] = np.random.rand(num_agents) * 2 * np.pi # Px

    traj_params[:, 3] = 3.0 + np.random.rand(num_agents) * 4.0 # Ay
    traj_params[:, 4] = 0.03 + np.random.rand(num_agents) * 0.07 # Fy
    traj_params[:, 5] = np.random.rand(num_agents) * 2 * np.pi # Py

    traj_params[:, 6] = 1.0 + np.random.rand(num_agents) * 2.0 # Az
    traj_params[:, 7] = 0.05 + np.random.rand(num_agents) * 0.1 # Fz
    traj_params[:, 8] = np.random.rand(num_agents) * 2 * np.pi # Pz
    traj_params[:, 9] = 8.0 + np.random.rand(num_agents) * 4.0 # Oz: 8-12

    rnd_cmd = np.random.rand(num_agents)

    # Vectorized conditions for targets
    tvx = np.zeros(num_agents, dtype=np.float32)
    tvy = np.zeros(num_agents, dtype=np.float32)
    tvz = np.zeros(num_agents, dtype=np.float32)
    tyr = np.zeros(num_agents, dtype=np.float32)

    mask = (rnd_cmd >= 0.2) & (rnd_cmd < 0.3); tvx[mask] = 1.0
    mask = (rnd_cmd >= 0.3) & (rnd_cmd < 0.4); tvx[mask] = -1.0
    mask = (rnd_cmd >= 0.4) & (rnd_cmd < 0.5); tvy[mask] = 1.0
    mask = (rnd_cmd >= 0.5) & (rnd_cmd < 0.6); tvy[mask] = -1.0
    mask = (rnd_cmd >= 0.6) & (rnd_cmd < 0.7); tvz[mask] = 1.0
    mask = (rnd_cmd >= 0.7) & (rnd_cmd < 0.8); tvz[mask] = -1.0
    mask = (rnd_cmd >= 0.8) & (rnd_cmd < 0.9); tyr[mask] = 1.0
    mask = (rnd_cmd >= 0.9); tyr[mask] = -1.0

    target_vx[:] = tvx
    target_vy[:] = tvy
    target_vz[:] = tvz
    target_yaw_rate[:] = tyr

    # Reset Observations (Size 608)
    observations[:, :600] = 0.0
    observations[:, 600] = tvx
    observations[:, 601] = tvy
    observations[:, 602] = tvz
    observations[:, 603] = tyr
    observations[:, 604:608] = 0.0

    pos_x[:] = 0.0
    pos_y[:] = 0.0
    pos_z[:] = 10.0

    vel_x[:] = 0.0
    vel_y[:] = 0.0
    vel_z[:] = 0.0
    roll[:] = 0.0
    pitch[:] = 0.0
    yaw[:] = 0.0

    if len(reset_indices) > 0:
        step_counts[reset_indices] = 0

class DroneEnv(CUDAEnvironmentState):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_agents = kwargs.get("num_agents", 1)
        self.episode_length = kwargs.get("episode_length", 100)
        self.agent_ids = [f"drone_{i}" for i in range(self.num_agents)]
        self.use_cuda = _HAS_PYCUDA and kwargs.get("use_cuda", True)

        if self.use_cuda:
             # CUDA not supported in this version
             logging.warning("CUDA backend requested but not supported in this patch. Falling back to CPU.")
             self.use_cuda = False

        if _HAS_CYTHON:
            logging.info("DroneEnv: Using CPU/Cython backend.")
            self.step_function = step_cython
            self.reset_function = reset_cython
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
                "vt_x", "vt_y", "vt_z",
                "traj_params", # New
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
            "vt_x": {"shape": (self.num_agents,), "dtype": np.float32},
            "vt_y": {"shape": (self.num_agents,), "dtype": np.float32},
            "vt_z": {"shape": (self.num_agents,), "dtype": np.float32},
            "traj_params": {"shape": (self.num_agents, 10), "dtype": np.float32}, # New
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
             "observations": {"shape": (self.num_agents, 608), "dtype": np.float32}, # 600 + 4 + 4
        }

    def get_action_space(self):
        return (self.num_agents, 4)

    def get_observation_space(self):
        # 200 * 3 + 4 + 4 = 608
        return (self.num_agents, 608)

    def get_reward_signature(self): return (self.num_agents,)

    def get_state_names(self):
        return [
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "roll", "pitch", "yaw",
                "masses", "drag_coeffs", "thrust_coeffs",
                "target_vx", "target_vy", "target_vz", "target_yaw_rate",
                "vt_x", "vt_y", "vt_z",
                "traj_params",
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
            "vt_x": "vt_x", "vt_y": "vt_y", "vt_z": "vt_z",
            "traj_params": "traj_params", # New
            "pos_history": "pos_history",
            "observations": "observations",
            "rewards": "rewards",
            "done_flags": "done_flags",
            "step_counts": "step_counts",
            "actions": "actions",
            "num_agents": "num_agents",
            "episode_length": "episode_length",
            "env_ids": "env_ids",
        }

    def get_reset_function_kwargs(self):
        return {
            "pos_x": "pos_x", "pos_y": "pos_y", "pos_z": "pos_z",
            "vel_x": "vel_x", "vel_y": "vel_y", "vel_z": "vel_z",
            "roll": "roll", "pitch": "pitch", "yaw": "yaw",
            "masses": "masses", "drag_coeffs": "drag_coeffs", "thrust_coeffs": "thrust_coeffs",
            "target_vx": "target_vx", "target_vy": "target_vy", "target_vz": "target_vz", "target_yaw_rate": "target_yaw_rate",
            "traj_params": "traj_params", # New
            "pos_history": "pos_history",
            "observations": "observations",
            "rng_states": "rng_states",
            "step_counts": "step_counts",
            "num_agents": "num_agents",
            "reset_indices": "reset_indices"
        }
