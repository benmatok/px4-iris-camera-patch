import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

class Visualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.rewards_history = []
        self.trajectory_snapshots = [] # List of tuples: (iteration, trajectories)

    def log_reward(self, iteration, reward):
        self.rewards_history.append((iteration, reward))

    def log_trajectory(self, iteration, trajectories, targets=None, tracker_data=None):
        """
        trajectories: shape (num_agents, episode_length, 3)
        targets: shape (num_agents, episode_length, 3) (Optional)
        tracker_data: shape (num_agents, episode_length, 4) (Optional) [u, v, size, conf]
        """
        # We only keep one agent's trajectory for visualization clarity
        # Or a few. Let's keep the first agent's trajectory.
        # IMPORTANT: Use .copy() to ensure we store data, not a view into a buffer
        single_traj = trajectories[0].copy() # (episode_length, 3)
        single_target = targets[0].copy() if targets is not None else None
        single_tracker = tracker_data[0].copy() if tracker_data is not None else None
        self.trajectory_snapshots.append((iteration, single_traj, single_target, single_tracker))

    def plot_rewards(self):
        iterations, rewards = zip(*self.rewards_history)
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, rewards)
        plt.xlabel("Iteration")
        plt.ylabel("Mean Reward")
        plt.title("Reward vs Time")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "reward_plot.png"))
        plt.close()

    def generate_trajectory_gif(self):
        if not self.trajectory_snapshots:
            return

        images = []
        filenames = []

        # Determine bounds for consistent axes
        # item is (itr, traj, target, tracker)
        all_trajs = np.array([item[1] for item in self.trajectory_snapshots])
        # all_trajs shape: (num_snapshots, episode_length, 3)

        x_min, x_max = all_trajs[:, :, 0].min(), all_trajs[:, :, 0].max()
        y_min, y_max = all_trajs[:, :, 1].min(), all_trajs[:, :, 1].max()
        z_min, z_max = all_trajs[:, :, 2].min(), all_trajs[:, :, 2].max()

        # Add some padding
        pad = 1.0
        x_lim = (x_min - pad, x_max + pad)
        y_lim = (y_min - pad, y_max + pad)
        z_lim = (z_min - pad, z_max + pad) # Z should be > 0 ideally, but let's see.

        for i, (itr, traj, target, tracker) in enumerate(self.trajectory_snapshots):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            ax1, ax2, ax3 = axes

            # Top-down view (X vs Y)
            ax1.plot(traj[:, 0], traj[:, 1], 'b-', label="Drone")
            ax1.plot(traj[0, 0], traj[0, 1], 'go', label="Start")
            ax1.plot(traj[-1, 0], traj[-1, 1], 'rx', label="End")
            if target is not None:
                ax1.plot(target[:, 0], target[:, 1], 'r--', alpha=0.5, label="Target")
                ax1.plot(target[-1, 0], target[-1, 1], 'r*', markersize=10, label="Target End")

            ax1.set_xlim(x_lim)
            ax1.set_ylim(y_lim)
            ax1.set_title(f"Top-Down View (Iter {itr})")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.grid(True)
            ax1.legend()

            # Side view (X vs Z)
            ax2.plot(traj[:, 0], traj[:, 2], 'b-', label="Drone")
            ax2.plot(traj[0, 0], traj[0, 2], 'go', label="Start")
            ax2.plot(traj[-1, 0], traj[-1, 2], 'rx', label="End")
            if target is not None:
                ax2.plot(target[:, 0], target[:, 2], 'r--', alpha=0.5, label="Target")
                ax2.plot(target[-1, 0], target[-1, 2], 'r*', markersize=10, label="Target End")

            ax2.set_xlim(x_lim)
            ax2.set_ylim(z_lim)
            ax2.set_title(f"Side View (Iter {itr})")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Z")
            ax2.grid(True)
            ax2.legend()

            # Drone POV View
            if tracker is not None:
                # tracker: (episode_length, 4) -> u, v, size, conf
                # We plot the path of the target center in the image plane
                u, v, size, conf = tracker[:, 0], tracker[:, 1], tracker[:, 2], tracker[:, 3]

                # Plot Image Plane limits (Assuming FOV +/- 1.0 approx for u,v)
                ax3.set_xlim(-1.0, 1.0)
                ax3.set_ylim(-1.0, 1.0)
                ax3.set_title(f"Drone POV (Iter {itr})")
                ax3.set_xlabel("u (Horizontal)")
                ax3.set_ylabel("v (Vertical)")

                # Center crosshair
                ax3.axhline(0, color='k', linestyle=':', alpha=0.5)
                ax3.axvline(0, color='k', linestyle=':', alpha=0.5)

                # Scatter plot colored by confidence
                sc = ax3.scatter(u, v, c=conf, cmap='viridis', s=size*50, alpha=0.7, edgecolors='k')
                plt.colorbar(sc, ax=ax3, label="Tracker Confidence")

                # Draw the final bounding box
                from matplotlib.patches import Rectangle
                final_u, final_v, final_size = u[-1], v[-1], size[-1]
                # Box width proxy: sqrt(size) * 0.5?
                # size = 10 / (z^2+1). At z=3, size=1.0.
                # Box size in u-space depends on real size / distance.
                # u = x/z. width_u = real_width / z.
                # size ~ 10/z^2. z ~ sqrt(10/size).
                # width_u ~ real_width / sqrt(10/size) ~ k * sqrt(size).
                # Let's assume real_width = 1.0, so width_u ~ sqrt(size)/3.
                box_w = np.sqrt(final_size) * 0.4
                rect = Rectangle((final_u - box_w/2, final_v - box_w/2), box_w, box_w,
                                 linewidth=2, edgecolor='r', facecolor='none')
                ax3.add_patch(rect)
                ax3.text(final_u, final_v + box_w/2 + 0.05, "Target", color='r', ha='center')

            filename = os.path.join(self.output_dir, f"traj_{itr}.png")
            plt.savefig(filename)
            filenames.append(filename)
            plt.close()

            images.append(imageio.imread(filename))

        # Save GIF
        gif_path = os.path.join(self.output_dir, "training_evolution.gif")
        imageio.mimsave(gif_path, images, fps=5)

        # Cleanup intermediate images
        for f in filenames:
            os.remove(f)

        return gif_path

    def save_episode_gif(self, trajectory, target, tracker_data, filename):
        """
        Generates a GIF for a single episode/agent.
        trajectory: (episode_length, 3)
        target: (episode_length, 3)
        tracker_data: (episode_length, 4)
        """
        images = []
        filenames = []
        episode_length = trajectory.shape[0]

        # Bounds
        x_min, x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
        y_min, y_max = trajectory[:, 1].min(), trajectory[:, 1].max()
        z_min, z_max = trajectory[:, 2].min(), trajectory[:, 2].max()
        pad = 1.0
        x_lim = (x_min - pad, x_max + pad)
        y_lim = (y_min - pad, y_max + pad)
        z_lim = (z_min - pad, z_max + pad)

        for t in range(episode_length):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            ax1, ax2, ax3 = axes

            # Plot full trajectory up to t
            ax1.plot(trajectory[:t+1, 0], trajectory[:t+1, 1], 'b-')
            ax1.plot(trajectory[t, 0], trajectory[t, 1], 'bo') # Current pos
            if target is not None:
                ax1.plot(target[:t+1, 0], target[:t+1, 1], 'r--')
                ax1.plot(target[t, 0], target[t, 1], 'r*')

            ax1.set_xlim(x_lim); ax1.set_ylim(y_lim)
            ax1.set_title(f"Top-Down (Step {t})")
            ax1.grid(True)

            ax2.plot(trajectory[:t+1, 0], trajectory[:t+1, 2], 'b-')
            ax2.plot(trajectory[t, 0], trajectory[t, 2], 'bo')
            if target is not None:
                ax2.plot(target[:t+1, 0], target[:t+1, 2], 'r--')
                ax2.plot(target[t, 0], target[t, 2], 'r*')

            ax2.set_xlim(x_lim); ax2.set_ylim(z_lim)
            ax2.set_title(f"Side View (Step {t})")
            ax2.grid(True)

            # POV
            if tracker_data is not None:
                u, v, size, conf = tracker_data[t]
                ax3.set_xlim(-1.0, 1.0)
                ax3.set_ylim(-1.0, 1.0)
                ax3.set_title(f"Drone POV (Step {t})")

                ax3.axhline(0, color='k', linestyle=':', alpha=0.5)
                ax3.axvline(0, color='k', linestyle=':', alpha=0.5)

                sc = ax3.scatter([u], [v], c=[conf], cmap='viridis', vmin=0, vmax=1, s=size*50, edgecolors='k')

                from matplotlib.patches import Rectangle
                box_w = np.sqrt(size) * 0.4
                rect = Rectangle((u - box_w/2, v - box_w/2), box_w, box_w,
                                 linewidth=2, edgecolor='r', facecolor='none')
                ax3.add_patch(rect)

            fname = os.path.join(self.output_dir, f"ep_frame_{t}.png")
            plt.savefig(fname)
            filenames.append(fname)
            plt.close()
            images.append(imageio.imread(fname))

        imageio.mimsave(os.path.join(self.output_dir, filename), images, fps=10)
        for f in filenames:
            os.remove(f)
