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

    def log_trajectory(self, iteration, trajectories):
        """
        trajectories: shape (num_agents, episode_length, 3)
        """
        # We only keep one agent's trajectory for visualization clarity
        # Or a few. Let's keep the first agent's trajectory.
        single_traj = trajectories[0] # (episode_length, 3)
        self.trajectory_snapshots.append((iteration, single_traj))

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
        all_trajs = np.array([t for _, t in self.trajectory_snapshots])
        # all_trajs shape: (num_snapshots, episode_length, 3)

        x_min, x_max = all_trajs[:, :, 0].min(), all_trajs[:, :, 0].max()
        y_min, y_max = all_trajs[:, :, 1].min(), all_trajs[:, :, 1].max()
        z_min, z_max = all_trajs[:, :, 2].min(), all_trajs[:, :, 2].max()

        # Add some padding
        pad = 1.0
        x_lim = (x_min - pad, x_max + pad)
        y_lim = (y_min - pad, y_max + pad)
        z_lim = (z_min - pad, z_max + pad) # Z should be > 0 ideally, but let's see.

        for i, (itr, traj) in enumerate(self.trajectory_snapshots):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Top-down view (X vs Y)
            ax1.plot(traj[:, 0], traj[:, 1], 'b-')
            ax1.plot(traj[0, 0], traj[0, 1], 'go', label="Start")
            ax1.plot(traj[-1, 0], traj[-1, 1], 'rx', label="End")
            ax1.set_xlim(x_lim)
            ax1.set_ylim(y_lim)
            ax1.set_title(f"Top-Down View (Iter {itr})")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.grid(True)
            ax1.legend()

            # Side view (X vs Z)
            ax2.plot(traj[:, 0], traj[:, 2], 'b-')
            ax2.plot(traj[0, 0], traj[0, 2], 'go', label="Start")
            ax2.plot(traj[-1, 0], traj[-1, 2], 'rx', label="End")
            ax2.set_xlim(x_lim)
            ax2.set_ylim(z_lim)
            ax2.set_title(f"Side View (Iter {itr})")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Z")
            ax2.grid(True)
            ax2.legend()

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
