import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import os
from drone_env.drone import DroneEnv
from dpc_planner import DPCPlanner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom Windy Step Function ---

def step_cpu_wind(
    pos_x, pos_y, pos_z,
    vel_x, vel_y, vel_z,
    roll, pitch, yaw,
    masses, drag_coeffs, thrust_coeffs,
    target_vx, target_vy, target_vz, target_yaw_rate,
    vt_x, vt_y, vt_z,
    traj_params,
    target_trajectory,
    pos_history,
    observations,
    rewards,
    reward_components,
    done_flags,
    step_counts,
    actions,
    num_agents,
    episode_length,
    env_ids,
    wind_vector=None # Extra arg
):
    # This is a modified copy of step_cpu from drone_env/drone.py
    # We inject wind_vector (3,) or (num_agents, 3)

    actions_reshaped = actions.reshape(num_agents, 4)
    thrust_cmd = actions_reshaped[:, 0]
    roll_rate = actions_reshaped[:, 1]
    pitch_rate = actions_reshaped[:, 2]
    yaw_rate = actions_reshaped[:, 3]

    dt = 0.05
    g = 9.81
    substeps = 2

    px, py, pz = pos_x, pos_y, pos_z
    vx, vy, vz = vel_x, vel_y, vel_z
    r, p, y_ang = roll, pitch, yaw

    if wind_vector is None:
        wind_vector = np.zeros(3)

    # Handle wind broadcasting
    wx = wind_vector[0]
    wy = wind_vector[1]
    wz = wind_vector[2]

    # Trajectory Update (Simplified for Validation - Static or Simple)
    t = step_counts[0] + 1
    t_f = float(t)

    # We update targets based on traj_params (copied from drone.py logic)
    # x = Ax * sin(Fx * t + Px)
    vtx_val = traj_params[0] * np.sin(traj_params[1] * t_f + traj_params[2])
    vty_val = traj_params[3] * np.sin(traj_params[4] * t_f + traj_params[5])
    vtz_val = traj_params[9] + traj_params[6] * np.sin(traj_params[7] * t_f + traj_params[8])
    vtz_val = np.clip(vtz_val, 0.0, 0.1)

    vt_x[:] = vtx_val
    vt_y[:] = vty_val
    vt_z[:] = vtz_val

    # Shift History
    observations[:, 0:290] = observations[:, 10:300]

    # No auto-reset during validation step (handled by runner)

    for s in range(substeps):
        r += roll_rate * dt
        p += pitch_rate * dt
        y_ang += yaw_rate * dt

        max_thrust = 20.0 * thrust_coeffs
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)
        thrust_force = thrust_cmd * max_thrust

        sr, cr = np.sin(r), np.cos(r)
        sp, cp = np.sin(p), np.cos(p)
        sy, cy = np.sin(y_ang), np.cos(y_ang)

        ax_thrust = thrust_force * (cy * sp * cr + sy * sr) / masses
        ay_thrust = thrust_force * (sy * sp * cr - cy * sr) / masses
        az_thrust = thrust_force * (cp * cr) / masses

        az_gravity = -g

        # --- WIND EFFECT ---
        # V_air = V - W
        vax = vx - wx
        vay = vy - wy
        vaz = vz - wz

        ax_drag = -drag_coeffs * vax
        ay_drag = -drag_coeffs * vay
        az_drag = -drag_coeffs * vaz

        ax = ax_thrust + ax_drag
        ay = ay_thrust + ay_drag
        az = az_thrust + az_gravity + az_drag

        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        px += vx * dt
        py += vy * dt
        pz += vz * dt

        # Ground Check
        underground = pz < 0.0
        pz = np.where(underground, 0.0, pz)
        vx = np.where(underground, 0.0, vx)
        vy = np.where(underground, 0.0, vy)
        vz = np.where(underground, 0.0, vz)

    # State Update
    pos_x[:] = px
    pos_y[:] = py
    pos_z[:] = pz
    vel_x[:] = vx
    vel_y[:] = vy
    vel_z[:] = vz
    roll[:] = r
    pitch[:] = p
    yaw[:] = y_ang

    step_counts[:] += 1

    # Store History
    if t <= episode_length:
        ph_view = pos_history.reshape(episode_length, num_agents, 3)
        ph_view[t-1, :, 0] = px
        ph_view[t-1, :, 1] = py
        ph_view[t-1, :, 2] = pz

    # Observations (Tracker)
    dx_w = vtx_val - px
    dy_w = vty_val - py
    dz_w = vtz_val - pz

    # R Recalc
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

    s30 = 0.5; c30 = 0.866025
    xc = yb
    yc = -s30 * xb + c30 * zb
    zc = c30 * xb + s30 * zb

    zc_safe = np.maximum(zc, 0.1)
    u = xc / zc_safe
    v = yc / zc_safe

    size = 10.0 / (zc*zc + 1.0)

    # Confidence
    w2 = roll_rate**2 + pitch_rate**2 + yaw_rate**2
    conf = np.exp(-0.1 * w2)
    conf = np.where((c30 * xb + s30 * zb) < 0.0, 0.0, conf)

    # Update Obs
    new_features = np.zeros((num_agents, 10), dtype=np.float32)
    new_features[:, 0] = thrust_cmd
    new_features[:, 1] = roll_rate
    new_features[:, 2] = pitch_rate
    new_features[:, 3] = yaw_rate
    new_features[:, 4] = y_ang
    new_features[:, 5] = p
    new_features[:, 6] = r
    new_features[:, 7] = pz
    new_features[:, 8] = u
    new_features[:, 9] = v

    observations[:, 290:300] = new_features
    observations[:, 300] = size
    observations[:, 301] = conf


class WindyDroneEnv(DroneEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_wind = np.zeros(3)

    def set_wind(self, wind):
        self.current_wind = np.array(wind)

    def step_env(self):
        # Override step call to inject wind
        args = self.get_step_function_kwargs()
        resolved_args = []
        for k, v in args.items():
             if v in self.data_dictionary:
                 resolved_args.append(self.data_dictionary[v])
             elif k == "num_agents":
                 resolved_args.append(self.num_agents)
             elif k == "episode_length":
                 resolved_args.append(self.episode_length)

        # Call custom step
        step_cpu_wind(*resolved_args, wind_vector=self.current_wind)


# --- Validation Runner ---

def run_validation():
    num_agents = 1
    episode_length = 200

    env = WindyDroneEnv(num_agents=num_agents, episode_length=episode_length, use_cuda=False)
    planner = DPCPlanner(num_agents=num_agents)

    scenarios = [
        ("A_WindShear", [2.0, 0.0, 0.0]), # Wind X=2 (Crosswind if facing Y, Headwind if facing X. Let's see)
        ("B_Stall", [0.0, 0.0, 0.0]),
        ("C_LostFound", [0.0, 0.0, 0.0])
    ]

    results = {}

    for name, wind in scenarios:
        logging.info(f"Running Scenario: {name} with Wind {wind}")

        env.reset_all_envs()
        env.set_wind(wind)

        d = env.data_dictionary

        # Initialize Specific Scenario Conditions

        if name == "B_Stall":
            # Start High directly above target
            d['pos_x'][:] = d['vt_x'][:]
            d['pos_y'][:] = d['vt_y'][:]
            d['pos_z'][:] = 100.0 # High
            d['vel_x'][:] = 0.0
            d['vel_y'][:] = 0.0
            d['vel_z'][:] = 0.0
            # Target at 0
            d['vt_x'][:] = 0.0
            d['vt_y'][:] = 0.0
            d['vt_z'][:] = 0.0
            # Pitch down to see it
            d['pitch'][:] = np.deg2rad(70) # Almost looking down

        elif name == "A_WindShear":
             # Start with offset
             d['pos_x'][:] = d['vt_x'][:] - 50.0
             d['pos_y'][:] = d['vt_y'][:]
             d['pos_z'][:] = 20.0
             d['yaw'][:] = 0.0 # Facing X. Wind [2,0,0] is Headwind? No, Tail parameter?
             # Wind [2,0,0] pushes in +X.
             # Drone at -50, Target at 0. Drone flies +X.
             # Wind +2 means Tailwind.
             # Let's set Wind [-5, 0, 0] for Headwind.
             if wind[0] > 0:
                 # If wind is 2.0 (Scenario), and we fly +X, it's tailwind.
                 # Let's fly Perpendicular for Crosswind.
                 d['pos_x'][:] = d['vt_x'][:]
                 d['pos_y'][:] = d['vt_y'][:] - 50.0 # Fly +Y
                 d['yaw'][:] = np.pi/2 # Face +Y
                 # Wind is X=2. Crosswind from Right.


        traj_pos = []
        traj_target = []
        traj_pitch = []
        traj_conf = []

        for t in range(episode_length):
            # 1. Observation Masking for Scenario C
            if name == "C_LostFound":
                # Mask every 20-40 steps (Flicker)
                if (t % 50) > 30:
                     d['observations'][:, 298] = 0.0 # u
                     d['observations'][:, 299] = 0.0 # v
                     d['observations'][:, 301] = 0.0 # conf

            # 2. Plan
            current_state = {
                'pos_x': d['pos_x'], 'pos_y': d['pos_y'], 'pos_z': d['pos_z'],
                'vel_x': d['vel_x'], 'vel_y': d['vel_y'], 'vel_z': d['vel_z'],
                'roll': d['roll'], 'pitch': d['pitch'], 'yaw': d['yaw'],
                'masses': d['masses'], 'drag_coeffs': d['drag_coeffs'], 'thrust_coeffs': d['thrust_coeffs']
            }
            target_pos = np.stack([d['vt_x'], d['vt_y'], d['vt_z']], axis=1)

            actions = planner.compute_actions(current_state, target_pos)

            # 3. Step
            d['actions'][:] = actions.flatten()
            env.step_env()

            # Log
            traj_pos.append(np.stack([d['pos_x'], d['pos_y'], d['pos_z']], axis=1).copy())
            traj_target.append(target_pos.copy())
            traj_pitch.append(d['pitch'].copy())
            traj_conf.append(d['observations'][:, 301].copy())

        results[name] = {
            'pos': np.stack(traj_pos), # (T, N, 3)
            'target': np.stack(traj_target),
            'pitch': np.stack(traj_pitch),
            'conf': np.stack(traj_conf)
        }

    # --- Plotting ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, (name, data) in enumerate(results.items()):
        ax = axs[i]
        pos = data['pos'][:, 0, :] # Agent 0
        tgt = data['target'][:, 0, :]

        ax.plot(pos[:, 0], pos[:, 1], label='Drone')
        ax.plot(tgt[:, 0], tgt[:, 1], 'rx', label='Target')
        ax.set_title(f"{name}\nFinal Dist: {np.linalg.norm(pos[-1]-tgt[-1]):.2f}m")
        ax.legend()
        ax.axis('equal')

    plt.savefig('dpc_validation.png')
    logging.info("Saved dpc_validation.png")

if __name__ == "__main__":
    run_validation()
