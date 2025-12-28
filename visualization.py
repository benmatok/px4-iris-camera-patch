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
        tracker_data: shape (num_agents, episode_length, 4) (Optional) - u, v, size, conf
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
        # item is (itr, traj, target)
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
            fig = plt.figure(figsize=(18, 6))

            # Top-down view (X vs Y)
            ax1 = fig.add_subplot(131)
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
            ax2 = fig.add_subplot(132)
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

            # Drone POV (u vs v)
            ax3 = fig.add_subplot(133)
            if tracker is not None:
                # u, v are relative to image center.
                # Plot path of target in camera frame
                ax3.plot(tracker[:, 0], tracker[:, 1], 'm-', label="Target Trace")
                ax3.plot(tracker[0, 0], tracker[0, 1], 'go', label="Start")
                ax3.plot(tracker[-1, 0], tracker[-1, 1], 'rx', label="End")

                # Draw crosshair
                ax3.axhline(0, color='k', linestyle=':', alpha=0.5)
                ax3.axvline(0, color='k', linestyle=':', alpha=0.5)

                # Fixed limits (assuming FOV ~ 90 deg, so tan(45)=1)
                ax3.set_xlim(-2.0, 2.0)
                ax3.set_ylim(-2.0, 2.0)

                # Invert V (standard image coordinates have Y down, but our math might be different)
                # u = xc/zc (right), v = yc/zc (down/up?)
                # In `drone.py`:
                # xc = yb (right in body?)
                # yc = s30*xb + c30*zb (up in camera?)
                # So +v is up.

                ax3.set_aspect('equal')
                ax3.set_title("Drone POV (Target Gaze)")
                ax3.set_xlabel("U (Horizontal)")
                ax3.set_ylabel("V (Vertical)")
                ax3.grid(True)
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, "No Tracker Data", ha='center', va='center')

            filename = os.path.join(self.output_dir, f"traj_{itr}.png")
            plt.savefig(filename)
            filenames.append(filename)

            # Save separate POV image for the user
            extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(self.output_dir, f"drone_pov_{itr}.png"), bbox_inches=extent.expanded(1.2, 1.2))

            plt.close()

            images.append(imageio.imread(filename))

        # Save GIF
        gif_path = os.path.join(self.output_dir, "training_evolution.gif")
        imageio.mimsave(gif_path, images, fps=5)

        # Cleanup intermediate images, but keep the last one for inspection
        for f in filenames[:-1]:
            os.remove(f)

        print(f"Saved static trajectory image: {filenames[-1]}")

        return gif_path
