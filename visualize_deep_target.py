
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import logging
import time
from drone_env.drone import DroneEnv
from visualization import Visualizer
from train_drone import LinearPlanner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_deep_target_scenario():
    """
    Creates a specific scenario where the target is deep below the drone to demonstrate
    the 'Look Down' behavior.
    """
    logging.info("Starting Deep Target Visualization Scenario...")

    # 1 Agent, short episode
    env = DroneEnv(num_agents=1, episode_length=50, use_cuda=False)
    oracle = LinearPlanner(num_agents=1)

    # Manual Reset to specific state
    # Drone at (0, 0, 20), Level
    env.reset_all_envs()
    d = env.data_dictionary

    d['pos_x'][:] = 0.0
    d['pos_y'][:] = 0.0
    d['pos_z'][:] = 20.0
    d['vel_x'][:] = 0.0
    d['vel_y'][:] = 0.0
    d['vel_z'][:] = 0.0
    d['roll'][:] = 0.0
    d['pitch'][:] = 0.0
    d['yaw'][:] = 0.0

    # Target at (15, 10, 0) -> 3D position
    # Elevation: dist_xy = sqrt(15^2 + 10^2) = 18.0.
    # Elevation angle = atan(20 / 18.0) = 48 degrees (> 30 deg).
    # This should trigger Look Down AND maneuvering in X and Y.

    target_pos_static = np.array([15.0, 10.0, 0.0])

    d['vt_x'][:] = target_pos_static[0]
    d['vt_y'][:] = target_pos_static[1]
    d['vt_z'][:] = target_pos_static[2]

    # Run loop
    pos_hist = []
    tgt_hist = []
    pitch_hist = []
    thrust_hist = []

    for t in range(50):
        # Move target slowly to create a dynamic scenario (e.g. moving away or circle)
        # Circular motion:
        angle = t * 0.1
        radius = 5.0
        # Target circles around (15, 10)
        curr_target = target_pos_static + np.array([np.cos(angle)*radius, np.sin(angle)*radius, 0.0])

        d['vt_x'][:] = curr_target[0]
        d['vt_y'][:] = curr_target[1]
        d['vt_z'][:] = curr_target[2]

        # Planner Action
        current_state = {
            'pos_x': d['pos_x'], 'pos_y': d['pos_y'], 'pos_z': d['pos_z'],
            'vel_x': d['vel_x'], 'vel_y': d['vel_y'], 'vel_z': d['vel_z'],
            'roll': d['roll'], 'pitch': d['pitch'], 'yaw': d['yaw'],
            'masses': d['masses'], 'drag_coeffs': d['drag_coeffs'], 'thrust_coeffs': d['thrust_coeffs']
        }
        target_pos = np.stack([d['vt_x'], d['vt_y'], d['vt_z']], axis=1)

        actions = oracle.compute_actions(current_state, target_pos)

        # Log Actions
        thrust_hist.append(actions[0, 0])

        # Step
        d['actions'][:] = actions.flatten()

        # Explicit arg list required for Cython binding
        step_args = [
            d["pos_x"], d["pos_y"], d["pos_z"],
            d["vel_x"], d["vel_y"], d["vel_z"],
            d["roll"], d["pitch"], d["yaw"],
            d["masses"], d["drag_coeffs"], d["thrust_coeffs"],
            d["target_vx"], d["target_vy"], d["target_vz"], d["target_yaw_rate"],
            d["vt_x"], d["vt_y"], d["vt_z"],
            d["traj_params"], d["target_trajectory"],
            d["pos_history"], d["observations"],
            d["rewards"], d["reward_components"],
            d["done_flags"], d["step_counts"], d["actions"],
            1, 50, d["env_ids"]
        ]
        env.step_function(*step_args)

        # Record
        pos = np.array([d['pos_x'][0], d['pos_y'][0], d['pos_z'][0]])
        tgt = np.array([d['vt_x'][0], d['vt_y'][0], d['vt_z'][0]])
        pos_hist.append(pos)
        tgt_hist.append(tgt)
        pitch_hist.append(d['pitch'][0])

        # (Target update is handled at start of loop)

    # Visualization
    pos_hist = np.array(pos_hist)
    tgt_hist = np.array(tgt_hist)

    # Calculate Distances
    d_start = np.linalg.norm(pos_hist[0] - tgt_hist[0])
    d_end = np.linalg.norm(pos_hist[-1] - tgt_hist[-1])
    logging.info(f"Distance Start: {d_start:.2f} m, End: {d_end:.2f} m")

    viz = Visualizer()
    viz.log_trajectory(0, pos_hist[None, :, :], targets=tgt_hist[None, :, :], tracker_data=np.zeros((1, 50, 4)))

    # Save a specific plot or gif
    viz.save_episode_gif(999, pos_hist, targets=tgt_hist, tracker_data=np.zeros((50, 4)))

    # Pitch > 0 is Nose Down (Forward). Expected ~50 deg for deep target.
    logging.info(f"Final Pitch: {np.degrees(pitch_hist[-1]):.2f} deg")
    logging.info(f"Mean Thrust: {np.mean(thrust_hist):.2f}")
    logging.info("Generated visualization in visualizations/ (traj_999.gif)")

if __name__ == "__main__":
    visualize_deep_target_scenario()
