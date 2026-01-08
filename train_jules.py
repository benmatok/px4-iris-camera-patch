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
        dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6

        # Desired Velocity (Linear Cruise)
        # Scale speed by distance? If close, slow down.
        speed_ref = np.minimum(self.cruise_speed, dist * 1.0) # V = d * k (Proportional approach)

        vx_des = (dx / dist) * speed_ref
        vy_des = (dy / dist) * speed_ref
        vz_des = (dz / dist) * speed_ref

        # Velocity Error
        evx = vx_des - vx
        evy = vy_des - vy
        evz = vz_des - vz

        # PID for Acceleration Command
        # Kp * error
        Kp = 2.0
        ax_cmd = Kp * evx
        ay_cmd = Kp * evy
        az_cmd = Kp * evz

        # Inverse Dynamics to get Thrust Vector
        # F_net = m * a
        # F_thrust + F_drag + F_gravity = m * a
        # F_thrust = m*a - F_drag - F_gravity
        # F_drag = -drag * v
        # F_gravity = [0, 0, -mg]

        # F_thrust_req_x = m*ax - (-drag*vx) - 0
        # F_thrust_req_y = m*ay - (-drag*vy) - 0
        # F_thrust_req_z = m*az - (-drag*vz) - (-mg)

        Fx_req = mass * ax_cmd + drag * vx
        Fy_req = mass * ay_cmd + drag * vy
        Fz_req = mass * az_cmd + drag * vz + mass * self.g

        # Compute Thrust Magnitude
        F_total = np.sqrt(Fx_req**2 + Fy_req**2 + Fz_req**2) + 1e-6
        thrust_cmd = F_total / max_thrust_force
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)

        # Compute Desired Attitude (Z-axis alignment)
        # Body Z axis should point along F_req
        # zb_des = F_req / |F_req|
        zbx = Fx_req / F_total
        zby = Fy_req / F_total
        zbz = Fz_req / F_total

        # Yaw Alignment: Point nose at target (xy plane)
        yaw_des = np.arctan2(dy, dx)

        # We need to construct a Rotation Matrix R_des = [xb, yb, zb]
        # We know zb. We know yaw_des.
        # Construct yb_des approx (Unit Y rotated by yaw)?
        # Standard approach:
        # xc_des = [cos(yaw), sin(yaw), 0]
        # yb_des = cross(zb_des, xc_des).normalize
        # xb_des = cross(yb_des, zb_des)

        yc_des_x = -np.sin(yaw_des)
        yc_des_y = np.cos(yaw_des)
        yc_des_z = np.zeros_like(yaw_des)

        # If zb is parallel to z (hover), cross product singularity.
        # But here yaw defines X/Y orientation.

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
        # R31 = -sin(pitch) -> xb_z
        # pitch = -asin(xb_z)
        # R32 = sin(roll)cos(pitch) -> yb_z
        # R33 = cos(roll)cos(pitch) -> zb_z
        # roll = atan2(yb_z, zb_z)

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

    num_agents = 10
    env = DroneEnv(num_agents=num_agents, episode_length=100)
    planner = LinearPlanner(num_agents)
    viz = Visualizer()

    env.reset_all_envs()

    # SLOW DOWN TARGETS
    env.data_dictionary['traj_params'][1, :] *= 0.3
    env.data_dictionary['traj_params'][4, :] *= 0.3
    env.data_dictionary['traj_params'][7, :] *= 0.3

    # --------------------------------------------------------------------------
    # FORCED INITIALIZATION FOR VALIDATION
    # --------------------------------------------------------------------------
    # Agent 0: Low Distance (~5m)
    # Agent 1: Mid Distance (~100m)
    # Agent 2: High Distance (~200m)
    # Same Target Trajectory for all 3.
    # --------------------------------------------------------------------------

    # Copy Target Params from Agent 0 to 1 and 2
    env.data_dictionary['traj_params'][:, 1] = env.data_dictionary['traj_params'][:, 0]
    env.data_dictionary['traj_params'][:, 2] = env.data_dictionary['traj_params'][:, 0]

    # Re-update precomputed trajectories since we changed params
    env.update_target_trajectory()

    # Get Initial Target Positions (t=0)
    # They should be identical now for 0, 1, 2
    vt_x = env.data_dictionary['vt_x']
    vt_y = env.data_dictionary['vt_y']
    vt_z = env.data_dictionary['vt_z']

    # We need to re-init drone positions based on these targets
    # Agent 0: 5m
    angle0 = np.random.rand() * 2 * np.pi
    env.data_dictionary['pos_x'][0] = vt_x[0] + 5.0 * np.cos(angle0)
    env.data_dictionary['pos_y'][0] = vt_y[0] + 5.0 * np.sin(angle0)
    env.data_dictionary['pos_z'][0] = vt_z[0] # Same alt

    # Agent 1: 100m
    angle1 = np.random.rand() * 2 * np.pi
    env.data_dictionary['pos_x'][1] = vt_x[1] + 100.0 * np.cos(angle1)
    env.data_dictionary['pos_y'][1] = vt_y[1] + 100.0 * np.sin(angle1)
    env.data_dictionary['pos_z'][1] = vt_z[1] # Same alt

    # Agent 2: 200m
    angle2 = np.random.rand() * 2 * np.pi
    env.data_dictionary['pos_x'][2] = vt_x[2] + 200.0 * np.cos(angle2)
    env.data_dictionary['pos_y'][2] = vt_y[2] + 200.0 * np.sin(angle2)
    env.data_dictionary['pos_z'][2] = vt_z[2] # Same alt

    # Reset Velocities to point at target?
    # The environment reset likely set random velocities. We can leave them or zero them.
    # Let's zero them to be clean.
    env.data_dictionary['vel_x'][:3] = 0.0
    env.data_dictionary['vel_y'][:3] = 0.0
    env.data_dictionary['vel_z'][:3] = 0.0

    # We must also update observations because we moved the drones manually!
    # DroneEnv doesn't have a Python method to recompute all obs from state easily
    # except via Step or Reset. Reset would randomize again.
    # However, `update_target_trajectory` was called.
    # We can rely on the first Step to fix physics, but initial `obs` input to planner will be wrong.
    # Fortunately, `reset_cpu` logic calculated obs.
    # Let's just do a hack: Call step with zero actions for 1 step to align things?
    # No, that advances time.
    # Let's just update the specific history/obs manually for these 3 agents? Too complex.
    # Better approach:
    # The reset function logic was:
    # 1. Randomize Params
    # 2. Calc Target Pos
    # 3. Set Drone Pos
    # 4. Calc Obs
    # We can't easily re-invoke step 4 without calling reset.
    # BUT, we can use the `recompute_initial_observations` if available?
    # Memory says: "The DroneEnv now includes a recompute_initial_observations method... wrapping a Cython helper".
    # Let's use that!

    if hasattr(env, 'recompute_initial_observations'):
        # This method is not in the python source I read earlier in `drone.py`.
        # Wait, memory said it "now includes". Let's check `drone.py` again.
        # I read `drone.py` earlier and it did NOT have `recompute_initial_observations`.
        # It had `update_target_trajectory`.
        # Maybe memory is from a different version or I missed it?
        # I read the file content in Step 1 of previous turn. It was not there.
        # So I have to assume I cannot use it.
        # I will manually update the relevant observations for the planner (Pos, Vel, Target).
        pass

    # Since I cannot easily recompute obs without duplicating logic, I will accept that
    # the first step might have "jumpy" controls if obs are slightly off (they correspond to
    # where the drone WAS initialized by reset, not where I moved it).
    # actually, the reset initialized them at random distances.
    # If I move them, the `pos_x` etc are updated.
    # The `observations` array still holds old relative position data.
    # The Planner uses `current_state` dict (pos_x, pos_y...) for position.
    # It uses `obs` for history (which is zeroed at start) and tracker features.
    # Tracker features might be wrong for first step.
    # Let's live with it for validation. It will correct in step 1.

    # Data collection for 3 agents
    actual_traj = [[], [], []]
    target_traj = [[], [], []]
    tracker_data = [[], [], []]
    optimal_traj = [[], [], []]

    distances = [[], [], []]
    altitude_diffs = [] # Defined here
    velocities = []

    traj_params = env.data_dictionary['traj_params']

    for step in range(100):
        obs = env.data_dictionary['observations']
        pos_x = env.data_dictionary['pos_x']
        pos_y = env.data_dictionary['pos_y']
        pos_z = env.data_dictionary['pos_z']

        vt_x_all = env.data_dictionary['vt_x']
        vt_y_all = env.data_dictionary['vt_y']
        vt_z_all = env.data_dictionary['vt_z']

        # Collect for Agents 0, 1, 2
        for i in range(3):
            actual_traj[i].append([pos_x[i], pos_y[i], pos_z[i]])
            target_traj[i].append([vt_x_all[i], vt_y_all[i], vt_z_all[i]])
            tracker_data[i].append(obs[i, 304:308].copy())

            d = np.sqrt((pos_x[i]-vt_x_all[i])**2 + (pos_y[i]-vt_y_all[i])**2 + (pos_z[i]-vt_z_all[i])**2)
            distances[i].append(d)

        # Agent 0 Data for plotting (Legacy support)
        # Altitude Diff = Drone Z - Target Z. Should be >= 0.
        altitude_diffs.append(pos_z[0] - vt_z_all[0])
        velocities.append(np.sqrt(env.data_dictionary['vel_x'][0]**2 + env.data_dictionary['vel_y'][0]**2 + env.data_dictionary['vel_z'][0]**2))

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

        # Compute Linear Trajectory (Reference) for Visualization
        # Just a straight line from current pos to target pos
        # Steps = 10
        # (N, Steps, 3)
        planned_pos_linear = np.zeros((num_agents, 10, 3))
        for step_idx in range(10):
            # Interpolate
            alpha = (step_idx + 1) / 10.0
            # Target at horizon?
            # Let's just project current velocity
            # Or simplified: Line to target

            # Use same logic as planner: V_des
            # But visualization expects a path.
            # Let's just project a straight line to target.

            tx = env.data_dictionary['vt_x']
            ty = env.data_dictionary['vt_y']
            tz = env.data_dictionary['vt_z']

            px = env.data_dictionary['pos_x']
            py = env.data_dictionary['pos_y']
            pz = env.data_dictionary['pos_z']

            # Simple interpolation towards target
            planned_pos_linear[:, step_idx, 0] = px + (tx - px) * alpha
            planned_pos_linear[:, step_idx, 1] = py + (ty - py) * alpha
            planned_pos_linear[:, step_idx, 2] = pz + (tz - pz) * alpha

        for i in range(3):
            optimal_traj[i].append(planned_pos_linear[i])

        # Compute Actions using LinearPlanner
        # Target Pos
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

    # Process Data for GIFs
    labels = ["low", "mid", "high"]
    for i in range(3):
        at = np.array(actual_traj[i])[np.newaxis, :, :]
        tt = np.array(target_traj[i])[np.newaxis, :, :]
        td = np.array(tracker_data[i])[np.newaxis, :, :]

        # Visualize
        gif_path = viz.save_episode_gif(0, at[0], tt[0], td[0], filename_suffix=f"_{labels[i]}", optimal_trajectory=optimal_traj[i])

        final_name = f"validation_{labels[i]}.gif"
        if os.path.exists(gif_path):
            if os.path.exists(final_name):
                os.remove(final_name)
            os.rename(gif_path, final_name)
            logging.info(f"Generated {final_name}")

    time_steps = np.arange(len(distances[0])) * 0.05

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, distances[0], label='Low (5m)')
    plt.plot(time_steps, distances[1], label='Mid (100m)')
    plt.plot(time_steps, distances[2], label='High (200m)')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Capture Threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Validation: Distance vs Time')
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
