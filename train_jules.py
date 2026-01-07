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

class OracleController:
    """
    Computes optimal controls (Thrust, RollRate, PitchRate, YawRate)
    using a Feedback Controller (PD + Feedforward) to track the Lissajous trajectory.
    """
    def __init__(self, num_agents, dt=0.05):
        self.num_agents = num_agents
        self.dt = dt
        self.g = 9.81
        self.planning_horizon = 5.0

    def solve_min_jerk(self, p0, v0, a0, pf, vf, af, T):
        c0 = p0
        c1 = v0
        c2 = 0.5 * a0
        T2 = T*T; T3 = T2*T; T4 = T3*T; T5 = T4*T
        DeltaP = pf - (c0 + c1*T + c2*T2)
        DeltaV = vf - (c1 + 2*c2*T)
        DeltaA = af - (2*c2)
        c3 = (10*DeltaP - 4*DeltaV*T + 0.5*DeltaA*T2) / T3
        c4 = (-15*DeltaP + 7*DeltaV*T - DeltaA*T2) / T4
        c5 = (6*DeltaP - 3*DeltaV*T + 0.5*DeltaA*T2) / T5
        return c0, c1, c2, c3, c4, c5

    def eval_quintic(self, coeffs, t):
        c0, c1, c2, c3, c4, c5 = coeffs
        t2 = t*t; t3 = t2*t; t4 = t3*t; t5 = t4*t
        p = c0 + c1*t + c2*t2 + c3*t3 + c4*t4 + c5*t5
        return p

    def compute_trajectory(self, traj_params, t_start, steps, current_state=None):
        t_out = np.arange(steps) * self.dt

        def get_target_state(time_scalar):
            t_steps = time_scalar / self.dt
            params = traj_params[:, :, np.newaxis]
            Ax, Fx, Px = params[0], params[1], params[2]
            Ay, Fy, Py = params[3], params[4], params[5]
            Az, Fz, Pz, Oz = params[6], params[7], params[8], params[9]
            ph_x = Fx * t_steps + Px
            ph_y = Fy * t_steps + Py
            ph_z = Fz * t_steps + Pz
            freq_scale = 1.0 / self.dt
            tx = Ax * np.sin(ph_x)
            ty = Ay * np.sin(ph_y)
            tz = Oz + Az * np.sin(ph_z)
            tvx = Ax * Fx * freq_scale * np.cos(ph_x)
            tvy = Ay * Fy * freq_scale * np.cos(ph_y)
            tvz = Az * Fz * freq_scale * np.cos(ph_z)
            tax = -Ax * (Fx * freq_scale)**2 * np.sin(ph_x)
            tay = -Ay * (Fy * freq_scale)**2 * np.sin(ph_y)
            taz = -Az * (Fz * freq_scale)**2 * np.sin(ph_z)
            return (tx, ty, tz), (tvx, tvy, tvz), (tax, tay, taz)

        if current_state is not None:
            p0x = current_state['pos_x'][:, np.newaxis]
            p0y = current_state['pos_y'][:, np.newaxis]
            p0z = current_state['pos_z'][:, np.newaxis]
            v0x = current_state['vel_x'][:, np.newaxis]
            v0y = current_state['vel_y'][:, np.newaxis]
            v0z = current_state['vel_z'][:, np.newaxis]
            a0x = np.zeros_like(p0x)
            a0y = np.zeros_like(p0y)
            a0z = np.zeros_like(p0z)
        else:
            (p0x, p0y, p0z), (v0x, v0y, v0z), (a0x, a0y, a0z) = get_target_state(t_start)

        t_end = t_start + self.planning_horizon
        (pfx, pfy, pfz), _, (afx, afy, afz) = get_target_state(t_end)

        dx = pfx - p0x
        dy = pfy - p0y
        dz = pfz - p0z
        dist_full = np.sqrt(dx*dx + dy*dy + dz*dz)
        inv_dist = 1.0 / (dist_full + 1e-6)
        dir_x = dx * inv_dist
        dir_y = dy * inv_dist
        dir_z = dz * inv_dist

        CRUISE_DIST = 30.0
        CRUISE_SPEED = 8.0
        pfx_plan = p0x + dir_x * CRUISE_DIST
        pfy_plan = p0y + dir_y * CRUISE_DIST
        pfz_plan = p0z + dir_z * CRUISE_DIST
        vfx_plan = dir_x * CRUISE_SPEED
        vfy_plan = dir_y * CRUISE_SPEED
        vfz_plan = dir_z * CRUISE_SPEED
        afx_plan = np.zeros_like(afx)
        afy_plan = np.zeros_like(afy)
        afz_plan = np.zeros_like(afz)

        cx = self.solve_min_jerk(p0x, v0x, a0x, pfx_plan, vfx_plan, afx_plan, self.planning_horizon)
        cy = self.solve_min_jerk(p0y, v0y, a0y, pfy_plan, vfy_plan, afy_plan, self.planning_horizon)
        cz = self.solve_min_jerk(p0z, v0z, a0z, pfz_plan, vfz_plan, afz_plan, self.planning_horizon)

        t_eval = t_out[np.newaxis, :]
        px = self.eval_quintic(cx, t_eval)
        py = self.eval_quintic(cy, t_eval)
        pz = self.eval_quintic(cz, t_eval)

        planned_pos = np.stack([px, py, pz], axis=2)
        return None, planned_pos, None

def evaluate_oracle():
    logging.info("Starting Evaluation (AggressiveOracle)...")

    num_agents = 10
    env = DroneEnv(num_agents=num_agents, episode_length=100)
    oracle = OracleController(num_agents)
    planner = AggressiveOracle(env, horizon_steps=10, iterations=3)
    viz = Visualizer()

    env.reset_all_envs()

    # SLOW DOWN TARGETS
    env.data_dictionary['traj_params'][1, :] *= 0.3
    env.data_dictionary['traj_params'][4, :] *= 0.3
    env.data_dictionary['traj_params'][7, :] *= 0.3

    env.update_target_trajectory()

    actual_traj = []
    target_traj = []
    tracker_data = []
    optimal_traj = []

    distances = []
    velocities = []
    altitude_diffs = []

    traj_params = env.data_dictionary['traj_params']

    for step in range(100):
        obs = env.data_dictionary['observations']
        pos_x = env.data_dictionary['pos_x']
        pos_y = env.data_dictionary['pos_y']
        pos_z = env.data_dictionary['pos_z']
        vel_x = env.data_dictionary['vel_x']
        vel_y = env.data_dictionary['vel_y']
        vel_z = env.data_dictionary['vel_z']

        actual_traj.append([pos_x[0], pos_y[0], pos_z[0]])

        vt_x = env.data_dictionary['vt_x'][0]
        vt_y = env.data_dictionary['vt_y'][0]
        vt_z = env.data_dictionary['vt_z'][0]
        target_traj.append([vt_x, vt_y, vt_z])

        dist = np.sqrt((pos_x[0]-vt_x)**2 + (pos_y[0]-vt_y)**2 + (pos_z[0]-vt_z)**2)
        speed = np.sqrt(vel_x[0]**2 + vel_y[0]**2 + vel_z[0]**2)
        distances.append(dist)
        velocities.append(speed)

        # Altitude Diff = Drone Z - Target Z. Should be >= 0.
        altitude_diffs.append(pos_z[0] - vt_z)

        tracker_data.append(obs[0, 304:308].copy())

        t_current = float(step) * 0.05
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

        _, planned_pos_oracle, _ = oracle.compute_trajectory(traj_params, t_current, 10, current_state)
        optimal_traj.append(planned_pos_oracle[0])

        future_coeffs = planner.plan(current_state, obs, traj_params, t_current)

        fc_reshaped = future_coeffs.view(num_agents, 4, 5)
        current_action = fc_reshaped[:, :, 0] - fc_reshaped[:, :, 1] + fc_reshaped[:, :, 2] - fc_reshaped[:, :, 3] + fc_reshaped[:, :, 4]

        current_action[:, 0] = torch.clamp(current_action[:, 0], 0.0, 1.0)
        current_action[:, 1:] = torch.clamp(current_action[:, 1:], -10.0, 10.0)

        env.data_dictionary['actions'][:] = current_action.numpy().reshape(-1)

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

    actual_traj = np.array(actual_traj)
    target_traj = np.array(target_traj)
    tracker_data = np.array(tracker_data)

    actual_traj = actual_traj[np.newaxis, :, :]
    target_traj = target_traj[np.newaxis, :, :]
    tracker_data = tracker_data[np.newaxis, :, :]

    gif_path = viz.save_episode_gif(0, actual_traj[0], target_traj[0], tracker_data[0], filename_suffix="_oracle", optimal_trajectory=optimal_traj)
    if os.path.exists(gif_path):
        new_name = "aggressive_oracle.gif"
        os.rename(gif_path, new_name)
        logging.info(f"Renamed GIF to {new_name}")

    time_steps = np.arange(len(distances)) * 0.05

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, distances, label='Distance to Target (m)')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Capture Threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Agent 0: Distance vs Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('distance_vs_time.png')
    logging.info("Saved distance_vs_time.png")

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, velocities, label='Speed (m/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Agent 0: Speed vs Time')
    plt.grid(True)
    plt.savefig('speed_vs_time.png')
    logging.info("Saved speed_vs_time.png")

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, altitude_diffs, label='Altitude Diff (Drone - Target)')
    plt.axhline(y=0.0, color='r', linestyle='--', label='Same Height')
    plt.xlabel('Time (s)')
    plt.ylabel('Delta Z (m)')
    plt.title('Agent 0: Altitude Difference (Positive = Above)')
    plt.legend()
    plt.grid(True)
    plt.savefig('altitude_vs_time.png')
    logging.info("Saved altitude_vs_time.png")

if __name__ == "__main__":
    evaluate_oracle()
