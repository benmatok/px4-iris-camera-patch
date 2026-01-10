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
        optimal_trajectory: List of planning paths (Optional)
        """
        # We only keep one agent's trajectory for visualization clarity
        single_traj = trajectories[0].copy() # (episode_length, 3)
        single_target = targets[0].copy() if targets is not None else None
        single_tracker = tracker_data[0].copy() if tracker_data is not None else None
        # single_optimal is now a list of arrays (one per step), or None
        # We don't need to copy/index if it's already a list of the plan for agent 0
        single_optimal = optimal_trajectory

        self.trajectory_snapshots.append((iteration, single_traj, single_target, single_tracker, single_optimal))

    def save_episode_gif(self, iteration, trajectories, targets=None, tracker_data=None, filename_suffix="", optimal_trajectory=None):
        """
        Generates a GIF video of the episode for a single agent (agent 0).
        trajectories: (episode_length, 3)
        targets: (episode_length, 3)
        tracker_data: (episode_length, 4)
        optimal_trajectory: List of (steps, 3) arrays (Optional) - Plan at each timestep
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

        # We don't easily scan all optimal plans for bounds, assume target/traj covers it

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

            # Plot CURRENT Plan (Min Jerk)
            if optimal is not None and t < len(optimal):
                plan = optimal[t] # (steps, 3)
                if plan is not None:
                    ax1.plot(plan[:, 0], plan[:, 1], 'g--', alpha=0.7, label="Oracle Plan")
                    ax1.plot(plan[-1, 0], plan[-1, 1], 'g^', label="Plan End")

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

            if optimal is not None and t < len(optimal):
                plan = optimal[t]
                if plan is not None:
                    ax2.plot(plan[:, 0], plan[:, 2], 'g--', alpha=0.7)
                    ax2.plot(plan[-1, 0], plan[-1, 2], 'g^')

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

    def save_comparison_gif(self, iteration, student_traj, oracle_traj, targets=None, filename_suffix=""):
        """
        Generates a comparison GIF (Student vs Oracle).
        student_traj: (T, 3)
        oracle_traj: (T, 3)
        targets: (T, 3)
        """
        episode_length = student_traj.shape[0]
        step_size = max(1, episode_length // 100)
        indices = range(0, episode_length, step_size)

        images = []
        temp_dir = os.path.join(self.output_dir, "temp_comp_frames")
        os.makedirs(temp_dir, exist_ok=True)

        pad = 2.0
        vals_x = [student_traj[:,0], oracle_traj[:,0]]
        vals_y = [student_traj[:,1], oracle_traj[:,1]]
        vals_z = [student_traj[:,2], oracle_traj[:,2]]
        if targets is not None:
             vals_x.append(targets[:,0])
             vals_y.append(targets[:,1])
             vals_z.append(targets[:,2])

        x_min, x_max = min([v.min() for v in vals_x]), max([v.max() for v in vals_x])
        y_min, y_max = min([v.min() for v in vals_y]), max([v.max() for v in vals_y])
        z_min, z_max = min([v.min() for v in vals_z]), max([v.max() for v in vals_z])

        x_lim = (x_min - pad, x_max + pad)
        y_lim = (y_min - pad, y_max + pad)
        z_lim = (z_min - pad, z_max + pad)

        for t in indices:
            fig = plt.figure(figsize=(12, 6))

            # Top Down
            ax1 = fig.add_subplot(121)
            ax1.plot(student_traj[:t+1, 0], student_traj[:t+1, 1], 'b-', label="Student")
            ax1.plot(student_traj[t, 0], student_traj[t, 1], 'bo')

            ax1.plot(oracle_traj[:t+1, 0], oracle_traj[:t+1, 1], 'g--', label="Oracle")
            ax1.plot(oracle_traj[t, 0], oracle_traj[t, 1], 'g^')

            if targets is not None:
                ax1.plot(targets[:t+1, 0], targets[:t+1, 1], 'r:', alpha=0.5, label="Target")
                ax1.plot(targets[t, 0], targets[t, 1], 'r*')

            ax1.set_xlim(x_lim)
            ax1.set_ylim(y_lim)
            ax1.set_title(f"Comparison Top-Down (t={t})")
            ax1.legend(loc='upper right')
            ax1.grid(True)

            # Side View
            ax2 = fig.add_subplot(122)
            ax2.plot(student_traj[:t+1, 0], student_traj[:t+1, 2], 'b-', label="Student")
            ax2.plot(student_traj[t, 0], student_traj[t, 2], 'bo')

            ax2.plot(oracle_traj[:t+1, 0], oracle_traj[:t+1, 2], 'g--', label="Oracle")
            ax2.plot(oracle_traj[t, 0], oracle_traj[t, 2], 'g^')

            if targets is not None:
                ax2.plot(targets[:t+1, 0], targets[:t+1, 2], 'r:', alpha=0.5, label="Target")
                ax2.plot(targets[t, 0], targets[t, 2], 'r*')

            ax2.set_xlim(x_lim)
            ax2.set_ylim(z_lim)
            ax2.set_title(f"Comparison Side View (t={t})")
            ax2.grid(True)

            fname = os.path.join(temp_dir, f"frame_{t:04d}.png")
            plt.savefig(fname)
            plt.close()
            images.append(imageio.imread(fname))

        gif_name = f"comparison_{iteration}{filename_suffix}.gif"
        gif_path = os.path.join(self.output_dir, gif_name)
        imageio.mimsave(gif_path, images, fps=10)

        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        print(f"Saved comparison video: {gif_path}")
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
        all_points = []
        for item in self.trajectory_snapshots:
             # traj
             all_points.append(item[1])
             # target
             if item[2] is not None:
                 all_points.append(item[2])
             # optimal (list of arrays)
             if item[4] is not None:
                 if isinstance(item[4], list):
                     for plan in item[4]:
                         if plan is not None:
                             all_points.append(plan)
                 else:
                     all_points.append(item[4])

        if all_points:
            all_points_cat = np.concatenate(all_points, axis=0)
            x_min, x_max = all_points_cat[:, 0].min(), all_points_cat[:, 0].max()
            y_min, y_max = all_points_cat[:, 1].min(), all_points_cat[:, 1].max()
            z_min, z_max = all_points_cat[:, 2].min(), all_points_cat[:, 2].max()
        else:
            x_min, x_max = -10, 10
            y_min, y_max = -10, 10
            z_min, z_max = 0, 20

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
                # optimal is a list of plans. We can't plot all. Plot nothing or first/last?
                # Just skip for the summary GIF to avoid clutter/errors
                pass

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
                pass # Skip list of plans

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
