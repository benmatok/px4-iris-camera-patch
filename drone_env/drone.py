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
            self.cuda_data_manager = self # Mock

        @property
        def data_dictionary(self):
            if not hasattr(self, '_data_dictionary'):
                 self._data_dictionary = {}
            return self._data_dictionary

        def init_data_manager(self):
             # Allocate data based on get_data_dictionary
             dd = self.get_data_dictionary()
             for name, meta in dd.items():
                 shape = meta["shape"]
                 dtype = meta["dtype"]
                 self.data_dictionary[name] = np.zeros(shape, dtype=dtype)

# Try importing the Cython extension
try:
    from drone_env.drone_cython import step_cython, reset_cython, update_target_trajectory_from_params
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
    wind_x, wind_y, wind_z, # New
    target_vx, target_vy, target_vz, target_yaw_rate,
    vt_x, vt_y, vt_z,
    traj_params, # Shape: (10, num_agents)
    target_trajectory,
    pos_history, # Shape: (episode_length, num_agents, 3)
    observations,
    rewards,
    reward_components, # Shape (num_agents, 8)
    done_flags,
    step_counts,
    actions,
    action_buffer, # New: (num_agents, 11, 4)
    delays,        # New: (num_agents,)
    rng_states,    # Needed for noise
    num_agents,
    episode_length,
    env_ids,
):
    # Vectorized Pure NumPy implementation
    # actions: (num_agents * 4,)

    # Reshape actions for easier access: (num_agents, 4)
    actions_reshaped = actions.reshape(num_agents, 4)

    # -------------------------------------------------------------------------
    # Delay Logic
    # -------------------------------------------------------------------------
    # Shift Buffer: (N, 11, 4) -> (N, 10, 4) moved to indices 1..10
    # We want index 0 to be newest.
    action_buffer[:, 1:] = action_buffer[:, :-1]
    action_buffer[:, 0] = actions_reshaped

    # Select effective action
    # effective[i] = action_buffer[i, delays[i]]
    # Fancy indexing
    rows = np.arange(num_agents)
    eff_actions = action_buffer[rows, delays, :] # (num_agents, 4)

    thrust_cmd = eff_actions[:, 0]
    roll_rate = eff_actions[:, 1]
    pitch_rate = eff_actions[:, 2]
    yaw_rate = eff_actions[:, 3]

    dt = 0.05
    g = 9.81
    substeps = 2

    # Local copies of state (NumPy arrays)
    px, py, pz = pos_x, pos_y, pos_z
    vx, vy, vz = vel_x, vel_y, vel_z
    r, p, y_ang = roll, pitch, yaw

    # Wind Update (Random Walk)
    # Removed wind noise as per instruction
    # wind_x, wind_y, wind_z remain constant (likely 0.0)

    # Update Virtual Target using Trajectory Params
    t = step_counts[0] + 1
    t_f = float(t)

    # x = Ax * sin(Fx * t + Px)
    vtx_val = traj_params[0] * np.sin(traj_params[1] * t_f + traj_params[2])
    vty_val = traj_params[3] * np.sin(traj_params[4] * t_f + traj_params[5])
    vtz_val = traj_params[9] + traj_params[6] * np.sin(traj_params[7] * t_f + traj_params[8])

    vt_x[:] = vtx_val
    vt_y[:] = vty_val
    vt_z[:] = vtz_val

    # Shift History
    observations[:, 0:290] = observations[:, 10:300]

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

        # Drag relative to wind
        rel_vx = vx - wind_x
        rel_vy = vy - wind_y
        rel_vz = vz - wind_z

        ax_drag = -drag_coeffs * rel_vx
        ay_drag = -drag_coeffs * rel_vy
        az_drag = -drag_coeffs * rel_vz

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
        ph_view = pos_history.reshape(episode_length, num_agents, 3)
        ph_view[t-1, :, 0] = px
        ph_view[t-1, :, 1] = py
        ph_view[t-1, :, 2] = pz

    # Re-calculate Tracker Features for current state (for display/next step obs 304-308)
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
    yc = -s30 * xb + c30 * zb
    zc = c30 * xb + s30 * zb

    zc_safe = np.maximum(zc, 0.1)
    u = xc / zc_safe
    v = yc / zc_safe

    # Add Noise to Tracking
    u_noise = np.random.normal(0, 0.05, num_agents) # 5% noise roughly?
    v_noise = np.random.normal(0, 0.05, num_agents)
    u += u_noise
    v += v_noise

    u = np.clip(u, -1.732, 1.732)
    v = np.clip(v, -1.732, 1.732)

    size = 10.0 / (zc*zc + 1.0)
    # Noise on size
    size += np.random.normal(0, 0.01, num_agents)

    w2 = roll_rate**2 + pitch_rate**2 + yaw_rate**2
    conf = np.exp(-0.1 * w2)
    conf = np.where((c30 * xb + s30 * zb) < 0.0, 0.0, conf)
    conf = np.where(xb < 0.0, 0.0, conf)

    # Capture History Samples
    new_features = np.zeros((num_agents, 10), dtype=np.float32)
    new_features[:, 0] = r
    new_features[:, 1] = p
    new_features[:, 2] = y_ang
    new_features[:, 3] = pz
    new_features[:, 4] = thrust_cmd
    new_features[:, 5] = roll_rate
    new_features[:, 6] = pitch_rate
    new_features[:, 7] = yaw_rate
    new_features[:, 8] = u
    new_features[:, 9] = v

    observations[:, 290:300] = new_features

    # 304-308
    observations[:, 304] = u
    observations[:, 305] = v
    observations[:, 306] = size
    observations[:, 307] = conf

    # Calculate Relative Velocity
    vtvx = traj_params[0] * traj_params[1] * np.cos(traj_params[1] * t_f + traj_params[2])
    vtvy = traj_params[3] * traj_params[4] * np.cos(traj_params[4] * t_f + traj_params[5])
    vtvz = traj_params[6] * traj_params[7] * np.cos(traj_params[7] * t_f + traj_params[8])

    rvx = vtvx - vx
    rvy = vtvy - vy
    rvz = vtvz - vz

    rvx_b = r11 * rvx + r12 * rvy + r13 * rvz
    rvy_b = r21 * rvx + r22 * rvy + r23 * rvz
    rvz_b = r31 * rvx + r32 * rvy + r33 * rvz

    dist_sq = dx_w**2 + dy_w**2 + dz_w**2
    dist = np.sqrt(dist_sq)
    dist_safe = np.maximum(dist, 0.1)

    # Update Obs 300-304
    observations[:, 300] = rvx_b
    observations[:, 301] = rvy_b
    observations[:, 302] = rvz_b
    observations[:, 303] = dist

    # Rewards (Same as before)
    rvel_sq = rvx**2 + rvy**2 + rvz**2
    r_dot_v = dx_w*rvx + dy_w*rvy + dz_w*rvz
    dist_sq_safe = np.maximum(dist_sq, 0.01)
    omega_sq = (rvel_sq / dist_sq_safe) - (r_dot_v**2 / (dist_sq_safe**2))
    omega_sq = np.maximum(omega_sq, 0.0)
    rew_pn = -2.0 * omega_sq
    vd_dot_r = vx*dx_w + vy*dy_w + vz*dz_w
    closing = vd_dot_r / dist_safe
    rew_closing = 0.5 * closing
    vx_b = r11 * vx + r12 * vy + r13 * vz
    v_ideal = 0.1 * vx_b
    v_err = v - v_ideal
    gaze_err = u**2 + v_err**2
    rew_gaze = -0.01 * gaze_err
    funnel = 1.0 / (dist + 1.0)
    rew_guidance = (rew_pn + rew_gaze + rew_closing) * funnel
    rew_rate = -1.0 * w2
    upright_err = 1.0 - r33
    rew_upright = -5.0 * upright_err**2
    diff_thrust = np.maximum(0.4 - thrust_cmd, 0.0)
    rew_eff = -10.0 * diff_thrust
    rew = rew_guidance + rew_rate + rew_upright + rew_eff
    bonus = np.where(dist < 0.2, 10.0, 0.0)
    rew += bonus
    penalty = np.zeros(num_agents, dtype=np.float32)
    penalty = np.where(r33 < 0.5, penalty + 10.0, penalty)
    penalty = np.where(collision, penalty + 10.0, penalty)
    rew -= penalty
    rewards[:] = rew

    reward_components[:, 0] = rew_pn
    reward_components[:, 1] = rew_closing
    reward_components[:, 2] = rew_gaze
    reward_components[:, 3] = rew_rate
    reward_components[:, 4] = rew_upright
    reward_components[:, 5] = rew_eff
    reward_components[:, 6] = -penalty
    reward_components[:, 7] = bonus

    d_flag = np.zeros(num_agents, dtype=np.float32)
    d_flag = np.where(t >= episode_length, 1.0, d_flag)
    d_flag = np.where(dist < 0.2, 1.0, d_flag)
    d_flag = np.where(r33 < 0.5, 1.0, d_flag)
    d_flag = np.where(collision, 1.0, d_flag)
    done_flags[:] = d_flag

def reset_cpu(
    pos_x, pos_y, pos_z,
    vel_x, vel_y, vel_z,
    roll, pitch, yaw,
    masses, drag_coeffs, thrust_coeffs,
    wind_x, wind_y, wind_z, # New
    target_vx, target_vy, target_vz, target_yaw_rate,
    vt_x, vt_y, vt_z,
    traj_params,
    target_trajectory,
    pos_history,
    observations,
    rng_states,
    step_counts,
    num_agents,
    reset_indices,
    action_buffer, # New
    delays         # New
):
    # Vectorized Reset
    # Wider Randomization
    # Mass: 0.5 to 2.5 kg
    masses[:] = 0.5 + np.random.rand(num_agents) * 2.0
    # Drag: 0.05 to 0.2
    drag_coeffs[:] = 0.05 + np.random.rand(num_agents) * 0.15
    # Thrust: Fixed at 2.0 to ensure T/W > 1.0 for max mass (2.5kg)
    # Max Weight = 2.5 * 9.81 = 24.5N. Base Thrust=20N.
    # With Coeff=2.0, Max Thrust=40N.
    thrust_coeffs[:] = 2.0

    # Wind: Zeroed out
    wind_x[:] = 0.0
    wind_y[:] = 0.0
    wind_z[:] = 0.0

    # Delays: 0 to 10 steps (0-500ms)
    delays[:] = np.random.randint(0, 11, size=num_agents)

    # Clear Action Buffer
    action_buffer[:] = 0.0

    # Initialize Trajectory Parameters
    traj_params[0] = 3.0 + np.random.rand(num_agents) * 4.0
    traj_params[1] = 0.01 + np.random.rand(num_agents) * 0.03
    traj_params[2] = np.random.rand(num_agents) * np.pi
    traj_params[3] = 3.0 + np.random.rand(num_agents) * 4.0
    traj_params[4] = 0.01 + np.random.rand(num_agents) * 0.03
    traj_params[5] = np.random.rand(num_agents) * 2 * np.pi
    traj_params[6] = 0.0 + np.random.rand(num_agents) * 0.1
    traj_params[7] = 0.01 + np.random.rand(num_agents) * 0.05
    traj_params[8] = np.random.rand(num_agents) * 2 * np.pi
    traj_params[9] = 2.0

    rnd_cmd = np.random.rand(num_agents)
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

    observations[:] = 0.0

    vtx_val = traj_params[0] * np.sin(traj_params[2])
    vtvx_val = traj_params[0] * traj_params[1] * np.cos(traj_params[2])
    vty_val = traj_params[3] * np.sin(traj_params[5])
    vtvy_val = traj_params[3] * traj_params[4] * np.cos(traj_params[5])
    vtz_val = traj_params[9] + traj_params[6] * np.sin(traj_params[8])
    vtvz_val = traj_params[6] * traj_params[7] * np.cos(traj_params[8])

    vt_x[:] = vtx_val
    vt_y[:] = vty_val
    vt_z[:] = vtz_val

    init_angle = np.random.rand(num_agents) * 2 * np.pi
    dist_xy_desired = 5.0 + np.random.rand(num_agents) * 195.0

    pos_x[:] = vtx_val + dist_xy_desired * np.cos(init_angle)
    pos_y[:] = vty_val + dist_xy_desired * np.sin(init_angle)
    pos_z[:] = vtz_val

    speed = np.random.rand(num_agents) * 2.0
    dx = vtx_val - pos_x
    dy = vty_val - pos_y
    dz = vtz_val - pos_z
    dist_xy = np.sqrt(dx*dx + dy*dy)

    dir_x = dx / (dist_xy + 1e-6)
    dir_y = dy / (dist_xy + 1e-6)

    vel_x[:] = dir_x * speed
    vel_y[:] = dir_y * speed
    vel_z[:] = 0.0

    roll[:] = 0.0

    # Yaw towards target but with offset (+/- 45 deg)
    yaw[:] = np.arctan2(dy, dx) + (np.random.rand(num_agents) - 0.5) * (np.pi / 2)
    # Pitch initial offset (+/- 20 deg)
    pitch[:] = 0.1 + (np.random.rand(num_agents) - 0.5) * (40 * np.pi / 180)

    rvx = vtvx_val - vel_x
    rvy = vtvy_val - vel_y
    rvz = vtvz_val - vel_z

    observations[:, 300] = rvx
    observations[:, 301] = rvy
    observations[:, 302] = rvz

    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    observations[:, 303] = dist

    # Calc initial u/v for obs
    # For CPU reset we can just compute it.
    # Note: R is not identity anymore due to random yaw/pitch
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

    xb = r11 * dx + r12 * dy + r13 * dz
    yb = r21 * dx + r22 * dy + r23 * dz
    zb = r31 * dx + r32 * dy + r33 * dz

    s30 = 0.5
    c30 = 0.866025
    xc = yb
    yc = -s30 * xb + c30 * zb
    zc = c30 * xb + s30 * zb

    zc_safe = np.maximum(zc, 0.1)
    u = xc / zc_safe
    v = yc / zc_safe

    u = np.clip(u, -1.732, 1.732)
    v = np.clip(v, -1.732, 1.732)
    size = 10.0 / (zc*zc + 1.0)
    conf = np.ones(num_agents, dtype=np.float32)
    conf = np.where((c30 * xb + s30 * zb) < 0, 0.0, conf)

    observations[:, 304] = u
    observations[:, 305] = v
    observations[:, 306] = size
    observations[:, 307] = conf

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
             logging.warning("CUDA backend requested but not supported in this patch. Falling back to CPU.")
             self.use_cuda = False

        if not hasattr(self, "cuda_data_manager"):
             self.cuda_data_manager = self

        self.init_data_manager()

        if _HAS_CYTHON:
            logging.info("DroneEnv: Using CPU/Cython backend.")
            self.step_function = step_cython
            self.reset_function = reset_cython
        else:
            logging.info("DroneEnv: Using CPU/NumPy backend.")
            self.step_function = step_cpu
            self.reset_function = reset_cpu

    @property
    def data_dictionary(self):
        if not hasattr(self, '_data_dictionary'):
                self._data_dictionary = {}
        return self._data_dictionary

    def init_data_manager(self):
        dd = self.get_data_dictionary()
        for name, meta in dd.items():
            shape = meta["shape"]
            dtype = meta["dtype"]
            self.data_dictionary[name] = np.zeros(shape, dtype=dtype)

        if "reset_indices" not in self.data_dictionary:
            self.data_dictionary["reset_indices"] = np.zeros((self.num_agents,), dtype=np.int32)

    def reset_all_envs(self):
        all_indices = np.arange(self.num_agents, dtype=np.int32)
        self.data_dictionary["reset_indices"][:] = all_indices

        kwargs = self.get_reset_function_kwargs()
        args = {}
        for k, v in kwargs.items():
            if v in self.data_dictionary:
                args[k] = self.data_dictionary[v]
            elif k == "num_agents":
                args[k] = self.num_agents
            else:
                pass

        self.reset_function(**args)

    def update_target_trajectory(self):
        if _HAS_CYTHON:
             traj_params = self.data_dictionary["traj_params"]
             target_trajectory = self.data_dictionary["target_trajectory"]
             steps = target_trajectory.shape[0]
             update_target_trajectory_from_params(traj_params, target_trajectory, self.num_agents, steps)

    def get_environment_info(self):
        return {
            "n_agents": self.num_agents,
            "episode_length": self.episode_length,
            "core_state_names": [
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "roll", "pitch", "yaw",
                "masses", "drag_coeffs", "thrust_coeffs",
                "wind_x", "wind_y", "wind_z",
                "target_vx", "target_vy", "target_vz", "target_yaw_rate",
                "vt_x", "vt_y", "vt_z",
                "traj_params",
                "target_trajectory",
                "pos_history",
                "rng_states",
                "step_counts",
                "action_buffer", "delays"
            ],
        }

    def get_data_dictionary(self):
        return {
            "masses": {"shape": (self.num_agents,), "dtype": np.float32},
            "drag_coeffs": {"shape": (self.num_agents,), "dtype": np.float32},
            "thrust_coeffs": {"shape": (self.num_agents,), "dtype": np.float32},
            "wind_x": {"shape": (self.num_agents,), "dtype": np.float32}, # New
            "wind_y": {"shape": (self.num_agents,), "dtype": np.float32}, # New
            "wind_z": {"shape": (self.num_agents,), "dtype": np.float32}, # New
            "target_vx": {"shape": (self.num_agents,), "dtype": np.float32},
            "target_vy": {"shape": (self.num_agents,), "dtype": np.float32},
            "target_vz": {"shape": (self.num_agents,), "dtype": np.float32},
            "target_yaw_rate": {"shape": (self.num_agents,), "dtype": np.float32},
            "vt_x": {"shape": (self.num_agents,), "dtype": np.float32},
            "vt_y": {"shape": (self.num_agents,), "dtype": np.float32},
            "vt_z": {"shape": (self.num_agents,), "dtype": np.float32},
            "traj_params": {"shape": (10, self.num_agents), "dtype": np.float32},
            "target_trajectory": {"shape": (self.episode_length + 1, self.num_agents, 3), "dtype": np.float32},
            "pos_history": {"shape": (self.episode_length, self.num_agents, 3), "dtype": np.float32},
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
             "reward_components": {"shape": (self.num_agents, 8), "dtype": np.float32},
             "observations": {"shape": (self.num_agents, 308), "dtype": np.float32},
             "reset_indices": {"shape": (self.num_agents,), "dtype": np.int32},
             "actions": {"shape": (self.num_agents * 4,), "dtype": np.float32},
             "env_ids": {"shape": (self.num_agents,), "dtype": np.int32},
             "action_buffer": {"shape": (self.num_agents, 11, 4), "dtype": np.float32}, # New
             "delays": {"shape": (self.num_agents,), "dtype": np.int32}, # New
        }

    def get_action_space(self):
        return (self.num_agents, 4)

    def get_observation_space(self):
        return (self.num_agents, 308)

    def get_reward_signature(self): return (self.num_agents,)

    def get_state_names(self):
        return [
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "roll", "pitch", "yaw",
                "masses", "drag_coeffs", "thrust_coeffs",
                "wind_x", "wind_y", "wind_z",
                "target_vx", "target_vy", "target_vz", "target_yaw_rate",
                "vt_x", "vt_y", "vt_z",
                "traj_params",
                "target_trajectory",
                "pos_history",
                "rng_states",
                "observations", "rewards", "reward_components", "done_flags",
                "step_counts", "action_buffer", "delays"
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
            "wind_x": "wind_x", "wind_y": "wind_y", "wind_z": "wind_z",
            "target_vx": "target_vx", "target_vy": "target_vy", "target_vz": "target_vz", "target_yaw_rate": "target_yaw_rate",
            "vt_x": "vt_x", "vt_y": "vt_y", "vt_z": "vt_z",
            "traj_params": "traj_params",
            "target_trajectory": "target_trajectory",
            "pos_history": "pos_history",
            "observations": "observations",
            "rewards": "rewards",
            "reward_components": "reward_components",
            "done_flags": "done_flags",
            "step_counts": "step_counts",
            "actions": "actions",
            "action_buffer": "action_buffer",
            "delays": "delays",
            "rng_states": "rng_states",
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
            "wind_x": "wind_x", "wind_y": "wind_y", "wind_z": "wind_z",
            "target_vx": "target_vx", "target_vy": "target_vy", "target_vz": "target_vz", "target_yaw_rate": "target_yaw_rate",
            "vt_x": "vt_x", "vt_y": "vt_y", "vt_z": "vt_z",
            "traj_params": "traj_params",
            "target_trajectory": "target_trajectory",
            "pos_history": "pos_history",
            "observations": "observations",
            "rng_states": "rng_states",
            "step_counts": "step_counts",
            "num_agents": "num_agents",
            "reset_indices": "reset_indices",
            "action_buffer": "action_buffer",
            "delays": "delays"
        }
