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
        self.video_snapshots = [] # (iteration, images_list)

    def log_reward(self, iteration, reward):
        self.rewards_history.append((iteration, reward))

    def log_trajectory(self, iteration, trajectories, targets=None, tracker_data=None, optimal_trajectory=None):
        """
        trajectories: shape (num_agents, episode_length, 3)
        targets: shape (num_agents, episode_length, 3) (Optional)
        tracker_data: shape (num_agents, episode_length, 4) (Optional) - u, v, size, conf
        optimal_trajectory: shape (num_agents, episode_length, 3) (Optional) - The Oracle's path
        """
        # We only keep one agent's trajectory for visualization clarity
        single_traj = trajectories[0].copy() # (episode_length, 3)
        single_target = targets[0].copy() if targets is not None else None
        single_tracker = tracker_data[0].copy() if tracker_data is not None else None
        single_optimal = optimal_trajectory[0].copy() if optimal_trajectory is not None else None

        self.trajectory_snapshots.append((iteration, single_traj, single_target, single_tracker, single_optimal))

    def save_episode_gif(self, iteration, trajectories, targets=None, tracker_data=None, filename_suffix="", optimal_trajectory=None):
        """
        Generates a GIF video of the episode for a single agent (agent 0).
        trajectories: (episode_length, 3)
        targets: (episode_length, 3)
        tracker_data: (episode_length, 4)
        optimal_trajectory: (episode_length, 3) (Optional)
        """
        traj = trajectories
        target = targets
        tracker = tracker_data
        optimal = optimal_trajectory
        episode_length = traj.shape[0]

        # Subsample if too long (max 100 frames)
        step_size = max(1, episode_length // 100)
        indices = range(0, episode_length, step_size)

        images = []
        temp_dir = os.path.join(self.output_dir, "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)

        # Determine fixed bounds for the entire episode
        pad = 2.0

        vals_x = [traj[:,0]]
        vals_y = [traj[:,1]]
        vals_z = [traj[:,2]]

        if target is not None:
            vals_x.append(target[:,0])
            vals_y.append(target[:,1])
            vals_z.append(target[:,2])

        if optimal is not None:
            vals_x.append(optimal[:,0])
            vals_y.append(optimal[:,1])
            vals_z.append(optimal[:,2])

        x_min, x_max = min([v.min() for v in vals_x]), max([v.max() for v in vals_x])
        y_min, y_max = min([v.min() for v in vals_y]), max([v.max() for v in vals_y])
        z_min, z_max = min([v.min() for v in vals_z]), max([v.max() for v in vals_z])

        x_lim = (x_min - pad, x_max + pad)
        y_lim = (y_min - pad, y_max + pad)
        z_lim = (z_min - pad, z_max + pad)

        for t in indices:
            fig = plt.figure(figsize=(18, 6))

            # Top-down view (X vs Y)
            ax1 = fig.add_subplot(131)
            # Plot path up to t
            ax1.plot(traj[:t+1, 0], traj[:t+1, 1], 'b-', label="Agent (Actual)")
            ax1.plot(traj[t, 0], traj[t, 1], 'bo')

            if optimal is not None:
                ax1.plot(optimal[:t+1, 0], optimal[:t+1, 1], 'g--', label="Optimal (Oracle)")
                ax1.plot(optimal[t, 0], optimal[t, 1], 'g^')

            if target is not None:
                ax1.plot(target[:t+1, 0], target[:t+1, 1], 'r:', alpha=0.5, label="Target Path")
                ax1.plot(target[t, 0], target[t, 1], 'r*', markersize=10, label="Target")

            ax1.set_xlim(x_lim)
            ax1.set_ylim(y_lim)
            ax1.set_title(f"Top-Down (t={t})")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.grid(True)
            ax1.legend(loc='upper right')

            # Side view (X vs Z)
            ax2 = fig.add_subplot(132)
            ax2.plot(traj[:t+1, 0], traj[:t+1, 2], 'b-', label="Agent")
            ax2.plot(traj[t, 0], traj[t, 2], 'bo')

            if optimal is not None:
                ax2.plot(optimal[:t+1, 0], optimal[:t+1, 2], 'g--', label="Optimal")
                ax2.plot(optimal[t, 0], optimal[t, 2], 'g^')

            if target is not None:
                ax2.plot(target[:t+1, 0], target[:t+1, 2], 'r:', alpha=0.5, label="Target")
                ax2.plot(target[t, 0], target[t, 2], 'r*', markersize=10)

            ax2.set_xlim(x_lim)
            ax2.set_ylim(z_lim)
            ax2.set_title(f"Side View (t={t})")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Z")
            ax2.grid(True)

            # Drone POV (u vs v)
            ax3 = fig.add_subplot(133)
            if tracker is not None:
                # Plot trace
                ax3.plot(tracker[:t+1, 0], tracker[:t+1, 1], 'm-', alpha=0.5)
                # Current target pos
                ax3.plot(tracker[t, 0], tracker[t, 1], 'm*', markersize=12, label="Target")

                # Crosshair
                ax3.axhline(0, color='k', linestyle=':', alpha=0.5)
                ax3.axvline(0, color='k', linestyle=':', alpha=0.5)

                # Confidence indicator
                conf = tracker[t, 3]
                ax3.text(0.05, 0.95, f"Conf: {conf:.2f}", transform=ax3.transAxes)

                ax3.set_xlim(-2.0, 2.0)
                ax3.set_ylim(-2.0, 2.0)
                ax3.set_aspect('equal')
                ax3.set_title("Drone POV")
                ax3.set_xlabel("U")
                ax3.set_ylabel("V")
                ax3.grid(True)
            else:
                ax3.text(0.5, 0.5, "No Tracker Data", ha='center', va='center')

            fname = os.path.join(temp_dir, f"frame_{t:04d}.png")
            plt.savefig(fname)
            plt.close()
            images.append(imageio.imread(fname))

        # Save video
        gif_name = f"episode_video_{iteration}{filename_suffix}.gif"
        gif_path = os.path.join(self.output_dir, gif_name)
        imageio.mimsave(gif_path, images, fps=10)

        # Cleanup
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)

        print(f"Saved episode video: {gif_path}")
        return gif_path


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

        # Determine bounds
        # item is (itr, traj, target, tracker, optimal)
        all_trajs = []
        for item in self.trajectory_snapshots:
             all_trajs.append(item[1])
             if item[4] is not None:
                 all_trajs.append(item[4])

        all_trajs = np.concatenate(all_trajs, axis=0) # (num_trajs * len, 3)

        x_min, x_max = all_trajs[:, 0].min(), all_trajs[:, 0].max()
        y_min, y_max = all_trajs[:, 1].min(), all_trajs[:, 1].max()
        z_min, z_max = all_trajs[:, 2].min(), all_trajs[:, 2].max()

        pad = 1.0
        x_lim = (x_min - pad, x_max + pad)
        y_lim = (y_min - pad, y_max + pad)
        z_lim = (z_min - pad, z_max + pad)

        for i, (itr, traj, target, tracker, optimal) in enumerate(self.trajectory_snapshots):
            fig = plt.figure(figsize=(18, 6))

            # Top-down view (X vs Y)
            ax1 = fig.add_subplot(131)
            ax1.plot(traj[:, 0], traj[:, 1], 'b-', label="Agent")
            if optimal is not None:
                ax1.plot(optimal[:, 0], optimal[:, 1], 'g--', alpha=0.7, label="Optimal")

            ax1.plot(traj[0, 0], traj[0, 1], 'go', label="Start")
            ax1.plot(traj[-1, 0], traj[-1, 1], 'rx', label="End")

            if target is not None:
                ax1.plot(target[:, 0], target[:, 1], 'r:', alpha=0.5, label="Target")
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
            ax2.plot(traj[:, 0], traj[:, 2], 'b-', label="Agent")
            if optimal is not None:
                ax2.plot(optimal[:, 0], optimal[:, 2], 'g--', alpha=0.7, label="Optimal")

            ax2.plot(traj[0, 0], traj[0, 2], 'go', label="Start")
            ax2.plot(traj[-1, 0], traj[-1, 2], 'rx', label="End")
            if target is not None:
                ax2.plot(target[:, 0], target[:, 2], 'r:', alpha=0.5, label="Target")
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
                ax3.plot(tracker[:, 0], tracker[:, 1], 'm-', label="Target Trace")
                ax3.plot(tracker[0, 0], tracker[0, 1], 'go', label="Start")
                ax3.plot(tracker[-1, 0], tracker[-1, 1], 'rx', label="End")

                ax3.axhline(0, color='k', linestyle=':', alpha=0.5)
                ax3.axvline(0, color='k', linestyle=':', alpha=0.5)

                ax3.set_xlim(-2.0, 2.0)
                ax3.set_ylim(-2.0, 2.0)

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

            extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(self.output_dir, f"drone_pov_{itr}.png"), bbox_inches=extent.expanded(1.2, 1.2))

            plt.close()

            images.append(imageio.imread(filename))

        # Save GIF
        gif_path = os.path.join(self.output_dir, "training_evolution.gif")
        imageio.mimsave(gif_path, images, fps=5)

        for f in filenames[:-1]:
            os.remove(f)

        print(f"Saved static trajectory image: {filenames[-1]}")

        return gif_path
