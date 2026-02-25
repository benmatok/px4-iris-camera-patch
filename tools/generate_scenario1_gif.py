import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import io

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.validate_dive_tracking import DiveValidator
from flight_config import FlightConfig

def generate_gif():
    # Setup Scenario 1: Alt=100.0, Dist=50.0
    alt = 100.0
    dist = 50.0

    # Init Validator
    validator = DiveValidator(
        use_ground_truth=True,
        use_blind_mode=True,
        init_alt=alt,
        init_dist=dist,
        control_use_gt=True
    )

    # Store trajectory for plotting
    drone_positions = []
    frames = []

    # Duration: 20s (400 steps)
    steps = 400

    print("Generating Scenario 1 GIF...")

    # Create temp dir
    os.makedirs("visualizations/temp_frames", exist_ok=True)

    fig = plt.figure(figsize=(12, 6))

    for i in range(steps):
        # Step Sim using existing run logic (manually)
        s = validator.sim.get_state()

        # Perception
        img = validator.sim.get_image(validator.target_pos_sim_world)

        # Tracker Logic (replicated from validator.run)
        dpc_state_ned = validator.sim_to_ned(s)
        target_ned = validator.sim_pos_to_ned_pos(validator.target_pos_sim_world)
        center, target_wp_ned, radius = validator.tracker.process(img, dpc_state_ned, ground_truth_target_pos=target_ned)

        # Mission Logic
        sim_state_rel = s.copy()
        sim_state_rel['px'] = 0.0; sim_state_rel['py'] = 0.0

        target_wp_sim = None
        if target_wp_ned:
             target_wp_sim = validator.ned_rel_to_sim_rel(target_wp_ned)

        mission_state, dpc_target, extra_yaw = validator.mission.update(sim_state_rel, (center, target_wp_sim))

        # Control
        # Fake VIO for control as per validation script
        vel_est = {'vx': s['vy'], 'vy': s['vx'], 'vz': -s['vz']} # Sim (ENU) -> NED
        pos_est = {'px': s['py'], 'py': s['px'], 'pz': -s['pz']}

        state_obs = {
            'px': s['px'], 'py': s['py'], 'pz': s['pz'],
            'vx': 0.0, 'vy': 0.0, 'vz': None, # Blind Mode
            'roll': s['roll'], 'pitch': s['pitch'], 'yaw': s['yaw'],
            'wx': s['wx'], 'wy': s['wy'], 'wz': s['wz']
        }

        tracking_norm = None
        tracking_size_norm = None
        if center:
            tracking_norm = validator.projector.pixel_to_normalized(center[0], center[1])
            tracking_size_norm = radius / 480.0

        action_out, _ = validator.controller.compute_action(
            state_obs,
            dpc_target,
            tracking_uv=tracking_norm,
            tracking_size=tracking_size_norm,
            extra_yaw_rate=extra_yaw,
            velocity_est=vel_est,
            position_est=pos_est,
            velocity_reliable=True # GT for Scenario 1
        )

        # Step
        sim_action = np.array([
            action_out['thrust'],
            action_out['roll_rate'],
            action_out['pitch_rate'],
            action_out['yaw_rate']
        ])
        if mission_state == "TAKEOFF" and s['pz'] < 2.0:
             sim_action[0] = 0.8

        validator.sim.step(sim_action)

        # Record
        drone_positions.append([s['px'], s['py'], s['pz']])

        # Render Frame
        # Left: Trajectory Side View (X-Z)
        # Right: Camera View (img)

        if i % 2 == 0: # Save every 2nd frame (10fps equivalent if sim is 20hz)
            plt.clf()

            # Trajectory
            ax1 = fig.add_subplot(1, 2, 1)
            pos_arr = np.array(drone_positions)
            ax1.plot(pos_arr[:, 0], pos_arr[:, 2], 'b-')
            ax1.plot(dist, 0.0, 'rx', markersize=10, label='Target')
            ax1.set_xlim(-10, dist + 10)
            ax1.set_ylim(-5, alt + 10)
            ax1.set_xlabel("Distance X (m)")
            ax1.set_ylabel("Altitude Z (m)")
            ax1.set_title(f"Trajectory (T={i*0.05:.1f}s)")
            ax1.grid(True)

            # Camera
            ax2 = fig.add_subplot(1, 2, 2)
            # OpenCV BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw Target Marker on Image if detected or GT Project
            if center:
                 cv2.circle(img_rgb, (int(center[0]), int(center[1])), 10, (0, 255, 0), 2)

            ax2.imshow(img_rgb)
            ax2.set_title("Drone Camera View")

            # Save using Buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_frame = imageio.imread(buf)
            frames.append(img_frame)
            buf.close()

        if s['pz'] < 0.2:
             print("Ground Impact")
             break

    # Save GIF
    print(f"Saving GIF with {len(frames)} frames...")
    imageio.mimsave('scenario_1_trajectory.gif', frames, fps=10)
    print("Saved scenario_1_trajectory.gif")

if __name__ == "__main__":
    generate_gif()
