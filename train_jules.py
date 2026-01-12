import torch
import torch.nn as nn
import numpy as np
import logging
from drone_env.drone import DroneEnv
from visualization import Visualizer
from rrt_planner import AggressiveOracle
import os
import matplotlib.pyplot as plt

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LinearPlanner:
    """
    Plans a linear path to the target and outputs control actions (Thrust, Rates)
    to track it. Uses an Inverse Dynamics approach assuming constant velocity cruise.
    """
    def __init__(self, num_agents, dt=0.05):
        self.num_agents = num_agents
        self.dt = dt
        self.g = 9.81
        self.cruise_speed = 10.0 # m/s

    def compute_actions(self, current_state, target_pos):
        # Current State
        px = current_state['pos_x']
        py = current_state['pos_y']
        pz = current_state['pos_z']
        vx = current_state['vel_x']
        vy = current_state['vel_y']
        vz = current_state['vel_z']
        roll = current_state['roll']
        pitch = current_state['pitch']
        yaw = current_state['yaw']

        # Params
        mass = current_state['masses']
        drag = current_state['drag_coeffs']
        thrust_coeff = current_state['thrust_coeffs']
        max_thrust_force = 20.0 * thrust_coeff

        # Target Vector
        tx = target_pos[:, 0]
        ty = target_pos[:, 1]
        tz = target_pos[:, 2]

        dx = tx - px
        dy = ty - py
        dz = tz - pz
        dist_xy = np.sqrt(dx**2 + dy**2) + 1e-6

        # Elevation Angle Check
        # User said: "if we are above the object by at least 10 degrees"
        rel_h = pz - tz
        elevation_rad = np.arctan2(rel_h, dist_xy)
        threshold_rad = np.deg2rad(10.0)

        # Virtual Target Logic
        # If elevation < 10 deg, aim higher.
        target_z_eff = tz.copy()
        mask_low = elevation_rad < threshold_rad

        # For low agents, set target Z higher
        target_angle_rad = np.deg2rad(15.0)
        req_h = dist_xy * np.tan(target_angle_rad)

        target_z_eff[mask_low] = tz[mask_low] + req_h[mask_low]

        # Recalculate Delta with effective target
        dz_eff = target_z_eff - pz
        dist_eff = np.sqrt(dx**2 + dy**2 + dz_eff**2) + 1e-6

        # Desired Velocity (Linear Cruise)
        # Scale speed by distance? If close, slow down.
        speed_ref = np.minimum(self.cruise_speed, dist_eff * 1.0)

        vx_des = (dx / dist_eff) * speed_ref
        vy_des = (dy / dist_eff) * speed_ref
        vz_des = (dz_eff / dist_eff) * speed_ref

        # Velocity Error
        evx = vx_des - vx
        evy = vy_des - vy
        evz = vz_des - vz

        # PID for Acceleration Command
        Kp = 2.0
        ax_cmd = Kp * evx
        ay_cmd = Kp * evy
        az_cmd = Kp * evz

        # Inverse Dynamics to get Thrust Vector
        Fx_req = mass * ax_cmd + drag * vx
        Fy_req = mass * ay_cmd + drag * vy
        Fz_req = mass * az_cmd + drag * vz + mass * self.g

        # Compute Thrust Magnitude
        F_total = np.sqrt(Fx_req**2 + Fy_req**2 + Fz_req**2) + 1e-6
        thrust_cmd = F_total / max_thrust_force
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)

        # Compute Desired Attitude (Z-axis alignment)
        zbx = Fx_req / F_total
        zby = Fy_req / F_total
        zbz = Fz_req / F_total

        # Yaw Alignment: Point nose at target (xy plane)
        yaw_des = np.arctan2(dy, dx)

        # Better:
        # xb_temp = [cos(yaw), sin(yaw), 0]
        # yb = cross(zb, xb_temp)
        # xb = cross(yb, zb)

        xb_temp_x = np.cos(yaw_des)
        xb_temp_y = np.sin(yaw_des)
        xb_temp_z = np.zeros_like(yaw_des)

        # yb = cross(zb, xb_temp)
        yb_x = zby * xb_temp_z - zbz * xb_temp_y
        yb_y = zbz * xb_temp_x - zbx * xb_temp_z
        yb_z = zbx * xb_temp_y - zby * xb_temp_x

        norm_yb = np.sqrt(yb_x**2 + yb_y**2 + yb_z**2) + 1e-6
        yb_x /= norm_yb
        yb_y /= norm_yb
        yb_z /= norm_yb

        # xb = cross(yb, zb)
        xb_x = yb_y * zbz - yb_z * zby
        xb_y = yb_z * zbx - yb_x * zbz
        xb_z = yb_x * zby - yb_y * zbx

        # Extract Roll/Pitch from R = [xb, yb, zb]
        pitch_des = -np.arcsin(np.clip(xb_z, -1.0, 1.0))
        roll_des = np.arctan2(yb_z, zbz)

        # Rate P-Controller
        Kp_att = 5.0

        # Shortest angular distance
        roll_err = roll_des - roll
        roll_err = (roll_err + np.pi) % (2 * np.pi) - np.pi

        pitch_err = pitch_des - pitch
        pitch_err = (pitch_err + np.pi) % (2 * np.pi) - np.pi

        yaw_err = yaw_des - yaw
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi

        roll_rate_cmd = Kp_att * roll_err
        pitch_rate_cmd = Kp_att * pitch_err
        yaw_rate_cmd = Kp_att * yaw_err

        actions = np.zeros((self.num_agents, 4))
        actions[:, 0] = thrust_cmd
        actions[:, 1] = np.clip(roll_rate_cmd, -10.0, 10.0)
        actions[:, 2] = np.clip(pitch_rate_cmd, -10.0, 10.0)
        actions[:, 3] = np.clip(yaw_rate_cmd, -10.0, 10.0)

        return actions

def evaluate_oracle():
    logging.info("Starting Evaluation (LinearPlanner)...")

    # Scenarios: 3 Distances x 3 Heights = 9 Agents
    # Distances: 5, 40, 100
    # Heights: 0, 20, 50 (Relative to target)
    dists = [5.0, 40.0, 100.0]
    heights = [0.0, 20.0, 50.0]

    num_agents = len(dists) * len(heights) # 9

    env = DroneEnv(num_agents=num_agents, episode_length=100)
    planner = LinearPlanner(num_agents)
    viz = Visualizer()

    env.reset_all_envs()

    # --------------------------------------------------------------------------
    # FORCED INITIALIZATION FOR VALIDATION
    # --------------------------------------------------------------------------

    # 1. Uniform Target Trajectory
    # Use Agent 0's params for everyone to ensure comparable tracking task
    # SLOW DOWN TARGETS
    env.data_dictionary['traj_params'][1, :] *= 0.3 # Slow Fx
    env.data_dictionary['traj_params'][4, :] *= 0.3 # Slow Fy
    env.data_dictionary['traj_params'][7, :] *= 0.3 # Slow Fz

    ref_params = env.data_dictionary['traj_params'][:, 0].copy()
    for i in range(1, num_agents):
        env.data_dictionary['traj_params'][:, i] = ref_params

    # Re-update precomputed trajectories since we changed params
    env.update_target_trajectory()

    # Get Initial Target Positions (t=0)
    vt_x = env.data_dictionary['vt_x']
    vt_y = env.data_dictionary['vt_y']
    vt_z = env.data_dictionary['vt_z']

    # 2. Position Initialization
    scenario_labels = []
    idx = 0
    for d in dists:
        for h in heights:
            # Set Position
            angle = np.random.rand() * 2 * np.pi
            env.data_dictionary['pos_x'][idx] = vt_x[idx] + d * np.cos(angle)
            env.data_dictionary['pos_y'][idx] = vt_y[idx] + d * np.sin(angle)
            env.data_dictionary['pos_z'][idx] = vt_z[idx] + h

            # Reset Velocities
            env.data_dictionary['vel_x'][idx] = 0.0
            env.data_dictionary['vel_y'][idx] = 0.0
            env.data_dictionary['vel_z'][idx] = 0.0

            scenario_labels.append(f"d{int(d)}_h{int(h)}")
            idx += 1

    logging.info(f"Initialized {num_agents} agents with scenarios: {scenario_labels}")

    # Data collection
    actual_traj = [[] for _ in range(num_agents)]
    target_traj = [[] for _ in range(num_agents)]
    tracker_data = [[] for _ in range(num_agents)]
    optimal_traj = [[] for _ in range(num_agents)]

    distances_log = [[] for _ in range(num_agents)]

    max_steps = 400
    for step in range(max_steps):
        obs = env.data_dictionary['observations']
        pos_x = env.data_dictionary['pos_x']
        pos_y = env.data_dictionary['pos_y']
        pos_z = env.data_dictionary['pos_z']

        vt_x_all = env.data_dictionary['vt_x']
        vt_y_all = env.data_dictionary['vt_y']
        vt_z_all = env.data_dictionary['vt_z']

        current_distances = []
        for i in range(num_agents):
            actual_traj[i].append([pos_x[i], pos_y[i], pos_z[i]])
            target_traj[i].append([vt_x_all[i], vt_y_all[i], vt_z_all[i]])
            tracker_data[i].append(obs[i, 304:308].copy())

            d = np.sqrt((pos_x[i]-vt_x_all[i])**2 + (pos_y[i]-vt_y_all[i])**2 + (pos_z[i]-vt_z_all[i])**2)
            distances_log[i].append(d)
            current_distances.append(d)

        # Check Termination
        if all(d < 0.05 for d in current_distances):
            logging.info(f"All agents reached within 0.05m at step {step}. Terminating.")
            break

        current_state = {
            'pos_x': env.data_dictionary['pos_x'],
            'pos_y': env.data_dictionary['pos_y'],
            'pos_z': env.data_dictionary['pos_z'],
            'vel_x': env.data_dictionary['vel_x'],
            'vel_y': env.data_dictionary['vel_y'],
            'vel_z': env.data_dictionary['vel_z'],
            'roll': env.data_dictionary['roll'],
            'pitch': env.data_dictionary['pitch'],
            'yaw': env.data_dictionary['yaw'],
            'masses': env.data_dictionary['masses'],
            'drag_coeffs': env.data_dictionary['drag_coeffs'],
            'thrust_coeffs': env.data_dictionary['thrust_coeffs']
        }

        # Compute Reference Trajectory for Visualization
        planned_pos_linear = np.zeros((num_agents, 10, 3))
        px_sim = env.data_dictionary['pos_x'].copy()
        py_sim = env.data_dictionary['pos_y'].copy()
        pz_sim = env.data_dictionary['pos_z'].copy()

        tx = env.data_dictionary['vt_x']
        ty = env.data_dictionary['vt_y']
        tz = env.data_dictionary['vt_z']

        sim_dt = 0.05
        for step_idx in range(10):
            dx_s = tx - px_sim
            dy_s = ty - py_sim
            dist_xy_s = np.sqrt(dx_s**2 + dy_s**2) + 1e-6

            rel_h_s = pz_sim - tz
            elev_s = np.arctan2(rel_h_s, dist_xy_s)

            tz_eff_s = tz.copy()
            mask_s = elev_s < np.deg2rad(10.0)
            req_h_s = dist_xy_s * np.tan(np.deg2rad(15.0))
            tz_eff_s[mask_s] = tz[mask_s] + req_h_s[mask_s]

            dz_s = tz_eff_s - pz_sim
            dist_s = np.sqrt(dx_s**2 + dy_s**2 + dz_s**2) + 1e-6
            speed_s = 10.0

            vx_s = (dx_s / dist_s) * speed_s
            vy_s = (dy_s / dist_s) * speed_s
            vz_s = (dz_s / dist_s) * speed_s

            px_sim += vx_s * sim_dt
            py_sim += vy_s * sim_dt
            pz_sim += vz_s * sim_dt

            planned_pos_linear[:, step_idx, 0] = px_sim
            planned_pos_linear[:, step_idx, 1] = py_sim
            planned_pos_linear[:, step_idx, 2] = pz_sim

        for i in range(num_agents):
            optimal_traj[i].append(planned_pos_linear[i])

        # Compute Actions
        target_pos_current = np.stack([
            env.data_dictionary['vt_x'],
            env.data_dictionary['vt_y'],
            env.data_dictionary['vt_z']
        ], axis=1)

        current_action_np = planner.compute_actions(current_state, target_pos_current)
        env.data_dictionary['actions'][:] = current_action_np.reshape(-1)

        step_kwargs = env.get_step_function_kwargs()
        step_args = {}
        for k, v in step_kwargs.items():
            if v in env.data_dictionary:
                step_args[k] = env.data_dictionary[v]
            elif k == "num_agents":
                step_args[k] = env.num_agents
            elif k == "episode_length":
                step_args[k] = env.episode_length

        env.step_function(**step_args)

    # Visualization
    # 1. Generate GIFs
    for i in range(num_agents):
        at = np.array(actual_traj[i])[np.newaxis, :, :]
        tt = np.array(target_traj[i])[np.newaxis, :, :]
        td = np.array(tracker_data[i])[np.newaxis, :, :]

        gif_path = viz.save_episode_gif(0, at[0], tt[0], td[0], filename_suffix=f"_{scenario_labels[i]}", optimal_trajectory=optimal_traj[i])

        final_name = f"validation_{scenario_labels[i]}.gif"
        if os.path.exists(gif_path):
            if os.path.exists(final_name):
                os.remove(final_name)
            os.rename(gif_path, final_name)
            logging.info(f"Generated {final_name}")

    # 2. Generate Plots
    time_steps = np.arange(len(distances_log[0])) * 0.05

    # Plot Distance vs Time (grouped by Distance)
    # Colors for heights: 0:Blue, 20:Green, 50:Red
    colors = ['b', 'g', 'r']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    idx = 0
    for d_i, d_val in enumerate(dists):
        ax = axes[d_i]
        for h_i, h_val in enumerate(heights):
            label = f"H+{int(h_val)}m"
            ax.plot(time_steps, distances_log[idx], label=label, color=colors[h_i])
            idx += 1

        ax.set_title(f"Start Distance: {d_val}m")
        ax.set_xlabel("Time (s)")
        if d_i == 0:
            ax.set_ylabel("Distance to Target (m)")
        ax.axhline(y=0.2, color='k', linestyle='--', alpha=0.5, label='Capture')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig('validation_distances.png')
    logging.info("Saved validation_distances.png")

if __name__ == "__main__":
    evaluate_oracle()
