import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from sim_interface import SimDroneInterface
from vision.projection import Projector
from vision.flow_estimator import FlowVelocityEstimator
from flight_controller import DPCFlightController
from mission_manager import MissionManager
from visual_tracker import VisualTracker

def main():
    # Setup
    projector = Projector(width=640, height=480, fov_deg=120.0, tilt_deg=30.0)
    sim = SimDroneInterface(projector)

    # Target and Initial State
    target_pos_sim_world = [50.0, 0.0, 0.0]
    drone_pos = [0.0, 0.0, 100.0]

    # Calculate orientation
    dx = target_pos_sim_world[0] - drone_pos[0]
    dy = target_pos_sim_world[1] - drone_pos[1]
    dz = target_pos_sim_world[2] - drone_pos[2]
    yaw = np.arctan2(dy, dx)
    dist_xy = np.sqrt(dx*dx + dy*dy)
    pitch_vec = np.arctan2(dz, dist_xy)
    camera_tilt = np.deg2rad(30.0)
    pitch = pitch_vec - camera_tilt

    sim.reset_to_scenario("Blind Dive", pos_x=drone_pos[0], pos_y=drone_pos[1], pos_z=drone_pos[2], pitch=pitch, yaw=yaw)

    tracker = VisualTracker(projector)
    flow_estimator = FlowVelocityEstimator(projector)
    controller = DPCFlightController(dt=0.05, mode='PID')

    # Data Recording
    history = {
        'time': [],
        'true_pz': [],
        'est_pz': [],
        'ghost_z_start': [],
        'adj_ghost_z_start': [],
        'foe_u': [],
        'foe_v': []
    }

    DT = 0.05
    duration = 5.0 # Run for 5 seconds
    steps = int(duration / DT)

    print(f"Running simulation for {duration} seconds...")

    for i in range(steps):
        t = i * DT

        # 1. Perception
        s = sim.get_state()
        img = sim.get_image(target_pos_sim_world)

        # Sim to NED
        dpc_state_ned_abs = s.copy()
        dpc_state_ned_abs['px'] = s['py']
        dpc_state_ned_abs['py'] = s['px']
        dpc_state_ned_abs['pz'] = -s['pz']
        dpc_state_ned_abs['vx'] = s.get('vy', 0.0)
        dpc_state_ned_abs['vy'] = s.get('vx', 0.0)
        dpc_state_ned_abs['vz'] = -s.get('vz', 0.0)
        dpc_state_ned_abs['yaw'] = (np.pi/2) - s['yaw']

        # Tracker
        center, _, _ = tracker.process(img, dpc_state_ned_abs, ground_truth_target_pos=[target_pos_sim_world[1], target_pos_sim_world[0], -target_pos_sim_world[2]])

        # Flow
        body_rates = (s['wx'], s['wy'], s['wz'])
        foe = flow_estimator.update(dpc_state_ned_abs, body_rates, DT)

        # Controller
        state_obs = {
            'pz': s['pz'],
            'roll': s['roll'],
            'pitch': s['pitch'],
            'yaw': s['yaw'],
            'wx': s['wx'],
            'wy': s['wy'],
            'wz': s['wz']
        }

        tracking_norm = None
        if center:
            tracking_norm = projector.pixel_to_normalized(center[0], center[1])

        action, ghost_paths = controller.compute_action(
            state_obs,
            [target_pos_sim_world[0]-s['px'], target_pos_sim_world[1]-s['py'], target_pos_sim_world[2]],
            tracking_uv=tracking_norm,
            foe_uv=foe
        )

        # Sim Step
        sim_action = np.array([action['thrust'], action['roll_rate'], action['pitch_rate'], action['yaw_rate']])
        sim.step(sim_action)

        # Record
        history['time'].append(t)
        history['true_pz'].append(s['pz'])
        history['est_pz'].append(controller.est_pz)

        if ghost_paths and len(ghost_paths[0]) > 0:
            ghost_z = ghost_paths[0][0]['pz']
            history['ghost_z_start'].append(ghost_z)
            # Apply correction logic from theshow.py
            delta_z = s['pz'] - ghost_z
            history['adj_ghost_z_start'].append(ghost_z + delta_z)
        else:
            history['ghost_z_start'].append(None)
            history['adj_ghost_z_start'].append(None)

        if foe:
            history['foe_u'].append(foe[0])
            history['foe_v'].append(foe[1])
        else:
            history['foe_u'].append(None)
            history['foe_v'].append(None)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Altitude Plot
    axs[0].plot(history['time'], history['true_pz'], label='True Altitude (Sim)', linewidth=2)
    axs[0].plot(history['time'], history['est_pz'], label='Estimated Altitude (Controller)', linestyle='--')
    axs[0].plot(history['time'], history['ghost_z_start'], label='Ghost Path Start Z (Original)', linestyle=':')
    axs[0].plot(history['time'], history['adj_ghost_z_start'], label='Adjusted Ghost Z (Visualized)', color='red', linestyle='-.')
    axs[0].set_title('Altitude & Trajectory Alignment')
    axs[0].set_ylabel('Altitude (m)')
    axs[0].legend()
    axs[0].grid(True)

    # FOE U
    axs[1].plot(history['time'], history['foe_u'], label='Filtered FOE U')
    axs[1].set_title('Filtered FOE U Stability')
    axs[1].set_ylabel('U (Normalized)')
    axs[1].legend()
    axs[1].grid(True)

    # FOE V
    axs[2].plot(history['time'], history['foe_v'], label='Filtered FOE V')
    axs[2].set_title('Filtered FOE V Stability')
    axs[2].set_ylabel('V (Normalized)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('validation_plots_fixed.png')
    print("Plots saved to validation_plots_fixed.png")

if __name__ == "__main__":
    main()
