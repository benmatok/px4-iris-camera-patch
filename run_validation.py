import numpy as np
import torch
import argparse
import logging
import os
from drone_env.drone import DroneEnv
from models.ae_policy import DronePolicy
from visualization import Visualizer

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class ValidationScenario:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def reset(self, num_agents):
        pass

    def apply_intervention(self, env, obs):
        """
        Modify observations in-place or return modified observations.
        Args:
            env: The DroneEnv instance (to access state if needed)
            obs: The observation array (num_agents, 304)
        """
        return obs

class NoiseScenario(ValidationScenario):
    def __init__(self, noise_level=0.01):
        super().__init__("Input Noise", f"Adds {noise_level*100}% relative noise to all inputs")
        self.noise_level = noise_level
        self.file_suffix = "_noise_1pct"

    def apply_intervention(self, env, obs):
        noise = np.random.normal(0, self.noise_level, size=obs.shape)
        obs[:] = obs * (1.0 + noise)
        return obs

class TrackingScenario(ValidationScenario):
    def __init__(self, decimation=False, holding=False, tracking_noise_std=0.0):
        desc = []
        suffix_parts = []
        if decimation:
            desc.append("VGA Pixel Decimation")
            suffix_parts.append("decim")
        if holding:
            desc.append("Hold Last Known Position")
            suffix_parts.append("hold")
        if tracking_noise_std > 0:
            desc.append(f"Tracking Noise {tracking_noise_std} px")
            suffix_parts.append(f"noise{int(tracking_noise_std)}")

        super().__init__("Tracking Robustness", ", ".join(desc))
        self.decimation = decimation
        self.holding = holding
        self.tracking_noise_std = tracking_noise_std
        self.file_suffix = "_" + "_".join(suffix_parts)

        # Camera Params
        self.W = 640
        self.H = 480
        # FOV_x = 120 deg => tan(60) = 1.732
        # tan(60) = (W/2) / f => f = 320 / 1.732 = 184.75
        self.f = 184.751
        self.u_max = 1.73205
        # v_max = (H/2) / f = 240 / 184.75 = 1.299
        self.v_max = 1.299

        self.last_known_u = None
        self.last_known_v = None

    def reset(self, num_agents):
        self.last_known_u = np.zeros(num_agents, dtype=np.float32)
        self.last_known_v = np.zeros(num_agents, dtype=np.float32)

    def compute_raw_uv(self, env):
        # Extract state
        d = env.data_dictionary
        px, py, pz = d['pos_x'], d['pos_y'], d['pos_z']
        vt_x, vt_y, vt_z = d['vt_x'], d['vt_y'], d['vt_z']
        r, p, yaw = d['roll'], d['pitch'], d['yaw']

        # Relative Position (World)
        dx_w = vt_x - px
        dy_w = vt_y - py
        dz_w = vt_z - pz

        # Rotation Matrix
        sr, cr = np.sin(r), np.cos(r)
        sp, cp = np.sin(p), np.cos(p)
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

        # Body Frame
        xb = r11 * dx_w + r12 * dy_w + r13 * dz_w
        yb = r21 * dx_w + r22 * dy_w + r23 * dz_w
        zb = r31 * dx_w + r32 * dy_w + r33 * dz_w

        # Camera Frame (Up 30 deg)
        s30 = 0.5
        c30 = 0.866025

        xc = yb
        yc = -s30 * xb + c30 * zb
        zc = c30 * xb + s30 * zb

        # Project
        zc_safe = np.maximum(zc, 0.1)
        u = xc / zc_safe
        v = yc / zc_safe

        is_behind = zc <= 0.0

        return u, v, is_behind

    def apply_intervention(self, env, obs):
        num_agents = obs.shape[0]

        # 1. Get Raw U, V (Before Clipping)
        u_raw, v_raw, is_behind = self.compute_raw_uv(env)

        # 2. Logic

        # Pixel coordinates
        u_pix = u_raw * self.f
        v_pix = v_raw * self.f

        # Noise
        if self.tracking_noise_std > 0:
            noise_u = np.random.normal(0, self.tracking_noise_std, size=num_agents)
            noise_v = np.random.normal(0, self.tracking_noise_std, size=num_agents)
            u_pix += noise_u
            v_pix += noise_v

        # Decimation (Quantize to integers)
        if self.decimation:
            u_pix = np.round(u_pix)
            v_pix = np.round(v_pix)

        # Convert back to normalized
        u_final = u_pix / self.f
        v_final = v_pix / self.f

        # Holding vs Clipping
        if self.holding:
            # Check if valid (in FOV and in front)
            in_fov = (np.abs(u_final) <= self.u_max) & (np.abs(v_final) <= self.v_max) & (~is_behind)

            # If in FOV, update last known
            self.last_known_u[in_fov] = u_final[in_fov]
            self.last_known_v[in_fov] = v_final[in_fov]

            # Use last known for ALL
            u_out = self.last_known_u.copy()
            v_out = self.last_known_v.copy()

        else:
            # Standard clipping behavior
            u_out = np.clip(u_final, -self.u_max, self.u_max)
            v_out = np.clip(v_final, -self.u_max, self.u_max)

        # Update Observation
        # u, v are only in history at 298, 299.
        obs[:, 298] = u_out
        obs[:, 299] = v_out

        return obs

def run_validation(checkpoint_path, scenarios, num_agents=200, episode_length=400):
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Load Policy
    logging.info(f"Loading policy from {checkpoint_path}...")
    policy = DronePolicy(observation_dim=302, action_dim=4, hidden_dim=256).cpu()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()

    # Environment
    env = DroneEnv(num_agents=num_agents, episode_length=episode_length, use_cuda=False)

    # Visualizer
    visualizer = Visualizer(output_dir="validation_results")

    print(f"{'Scenario':<40} | {'Mean Dist':<10} | {'Std Dist':<10}")
    print("-" * 65)

    for i, scenario in enumerate(scenarios):
        env.reset_all_envs()
        scenario.reset(num_agents)

        d = env.data_dictionary

        # Buffers for visualization
        pos_history = []
        target_history = []
        tracker_history = []

        # Get initial obs
        obs_np = d["observations"]

        # Apply intervention at t=0
        obs_np = scenario.apply_intervention(env, obs_np)

        final_distances = np.full(num_agents, np.nan)
        already_done = np.zeros(num_agents, dtype=bool)

        with torch.no_grad():
            for t in range(episode_length):
                obs_torch = torch.from_numpy(obs_np).float()

                # Policy Step
                pred_actions, _ = policy(obs_torch)
                action_to_step = pred_actions.numpy()

                # Env Step
                d['actions'][:] = action_to_step.flatten()
                env.step_function(
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
                    num_agents, episode_length, d["env_ids"]
                )

                # Record State for Visualization
                pos = np.stack([d['pos_x'], d['pos_y'], d['pos_z']], axis=1).copy() # (N, 3)
                vt = np.stack([d['vt_x'], d['vt_y'], d['vt_z']], axis=1).copy() # (N, 3)

                # Capture distances for agents that JUST finished
                current_dists = np.linalg.norm(pos - vt, axis=1)
                done_mask = d['done_flags'].astype(bool)

                # Identify agents that are done now but weren't before
                just_finished = done_mask & (~already_done)
                final_distances[just_finished] = current_dists[just_finished]
                already_done = done_mask | already_done

                # Important: Capture the tracker data *as seen by the agent* (i.e. after intervention)
                # But interventions happen BEFORE step.
                # Here, we just stepped. We need to apply intervention for the NEXT step observations.

                obs_np = d["observations"]
                obs_np = scenario.apply_intervention(env, obs_np)

                # Now obs_np contains the modified tracker data.
                # u, v at 298, 299. Size, Conf at 300, 301.
                u_col = obs_np[:, 298:299]
                v_col = obs_np[:, 299:300]
                size_col = obs_np[:, 300:301]
                conf_col = obs_np[:, 301:302]
                track = np.concatenate([u_col, v_col, size_col, conf_col], axis=1)

                pos_history.append(pos)
                target_history.append(vt)
                tracker_history.append(track)

                if d['done_flags'].all() == 1.0:
                    break

        # Calculate Metric: Final Distance
        # Fill in any agents that didn't finish (timeout) with their last distance
        # Use current 'pos' and 'vt' which are from the last step
        current_dists = np.linalg.norm(pos - vt, axis=1)
        final_distances[~already_done] = current_dists[~already_done]

        mean_dist = np.mean(final_distances)
        std_dist = np.std(final_distances)

        print(f"{scenario.name:<40} | {mean_dist:.4f} m   | {std_dist:.4f} m")

        # Generate GIF
        # Stack: (T, N, 3) -> (N, T, 3)
        traj_stack = np.stack(pos_history, axis=1)
        targ_stack = np.stack(target_history, axis=1)
        track_stack = np.stack(tracker_history, axis=1)

        suffix = getattr(scenario, 'file_suffix', f"_scenario_{i}")

        # Visualize Agent 0
        try:
            visualizer.save_episode_gif(
                iteration=i,
                trajectories=traj_stack[0],
                targets=targ_stack[0],
                tracker_data=track_stack[0],
                filename_suffix=suffix
            )
        except Exception as e:
            logging.error(f"Error generating GIF for scenario {scenario.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="final_jules.pth")
    parser.add_argument("--agents", type=int, default=200)
    args = parser.parse_args()

    # Define Scenarios

    # 0. Baseline
    s0 = ValidationScenario("Baseline", "Standard environment")
    s0.file_suffix = "_baseline"

    # 1. Noise 1%
    s1 = NoiseScenario(noise_level=0.01)

    # 2. Decimation & Holding
    s2 = TrackingScenario(decimation=True, holding=True, tracking_noise_std=0.0)

    # 3. Noise 3 pixels
    s3 = TrackingScenario(decimation=True, holding=False, tracking_noise_std=3.0)

    scenarios = [s0, s1, s2, s3]

    run_validation(args.checkpoint, scenarios, num_agents=args.agents)
