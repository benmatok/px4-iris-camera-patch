import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

from drone_env.drone import DroneEnv
from models.ae_policy import Autoencoder1D, KFACOptimizer

class AETrainer:
    def __init__(self, num_agents=5000, episode_length=100, lr=0.001, load_checkpoint=None):
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Environment (CPU)
        print(f"Initializing Environment with {num_agents} agents...")
        self.env = DroneEnv(num_agents=num_agents, episode_length=episode_length, use_cuda=False)

        # Allocate Data Arrays
        self.data = {}
        for name, info in self.env.get_data_dictionary().items():
            self.data[name] = np.zeros(info["shape"], dtype=info["dtype"])

        # Initialize Autoencoder
        # Input dim 10, seq len 30
        self.ae = Autoencoder1D(input_dim=10, seq_len=30, latent_dim=20).to(self.device)

        # Use KFAC Optimizer
        print("Using KFAC Optimizer")
        self.optimizer = KFACOptimizer(self.ae, lr=lr)

        # Use L1Loss to match train_drone.py
        self.criterion = nn.L1Loss()

        self.loss_history = []
        self.start_episode = 0

        # Checkpoint Loading Logic
        if load_checkpoint is None:
            # Auto-detect default checkpoint
            if os.path.exists("ae_model.pth"):
                print("Found existing checkpoint 'ae_model.pth', auto-resuming...")
                load_checkpoint = "ae_model.pth"

        if load_checkpoint and os.path.exists(load_checkpoint):
            print(f"Loading checkpoint from {load_checkpoint}...")
            try:
                checkpoint = torch.load(load_checkpoint, map_location=self.device)

                # Handle both old format (state_dict only) and new format (full dict)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    print("Detected full checkpoint format.")
                    self.ae.load_state_dict(checkpoint["model_state_dict"])
                    if "optimizer_state_dict" in checkpoint:
                        try:
                            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        except Exception as e:
                            print(f"Warning: Could not load optimizer state: {e}")

                    if "loss_history" in checkpoint:
                        self.loss_history = checkpoint["loss_history"]

                    if "episode" in checkpoint:
                        self.start_episode = checkpoint["episode"] + 1
                        print(f"Resuming from episode {self.start_episode}")
                else:
                    print("Detected legacy checkpoint format (weights only).")
                    self.ae.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch.")

    def simple_controller(self):
        """
        Generates actions to stabilize the drones and follow targets:
        Thrust to counter gravity + P on height/vertical velocity.
        Roll/Pitch to counter velocity error.
        """
        # Access state from self.data
        vel_x = self.data["vel_x"]
        vel_y = self.data["vel_y"]
        vel_z = self.data["vel_z"]
        roll = self.data["roll"]
        pitch = self.data["pitch"]
        yaw = self.data["yaw"]

        target_vx = self.data["target_vx"]
        target_vy = self.data["target_vy"]
        target_vz = self.data["target_vz"]
        target_yaw_rate = self.data["target_yaw_rate"]

        # Simple P controller
        kp_att = 2.0
        kp_vel = 0.5
        kp_z = 0.2
        kp_yaw = 1.0

        v_err_x = target_vx - vel_x
        v_err_y = target_vy - vel_y
        v_err_z = target_vz - vel_z

        # Desired Attitude
        target_pitch = -kp_vel * v_err_x
        target_roll = kp_vel * v_err_y

        # Rate commands
        roll_rate_cmd = kp_att * (target_roll - roll)
        pitch_rate_cmd = kp_att * (target_pitch - pitch)
        yaw_rate_cmd = target_yaw_rate

        # Thrust command
        thrust_cmd = 0.5 + kp_z * v_err_z

        # Clip
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)
        roll_rate_cmd = np.clip(roll_rate_cmd, -1.0, 1.0)
        pitch_rate_cmd = np.clip(pitch_rate_cmd, -1.0, 1.0)
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -1.0, 1.0)

        # Pack actions
        # Shape: (num_agents, 4)
        actions = np.stack([thrust_cmd, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd], axis=1)
        return actions.flatten()

    def train(self, num_episodes=50, target_loss=1e-4):
        total_episodes = self.start_episode + num_episodes
        print(f"Starting training from episode {self.start_episode+1} to {total_episodes} (max) or until loss < {target_loss}...")

        step_ctr = 0

        # Continue from start_episode
        for ep in range(self.start_episode, total_episodes):
            # Reset
            self.env.reset_function(
                pos_x=self.data["pos_x"], pos_y=self.data["pos_y"], pos_z=self.data["pos_z"],
                vel_x=self.data["vel_x"], vel_y=self.data["vel_y"], vel_z=self.data["vel_z"],
                roll=self.data["roll"], pitch=self.data["pitch"], yaw=self.data["yaw"],
                masses=self.data["masses"], drag_coeffs=self.data["drag_coeffs"], thrust_coeffs=self.data["thrust_coeffs"],
                target_vx=self.data["target_vx"], target_vy=self.data["target_vy"], target_vz=self.data["target_vz"], target_yaw_rate=self.data["target_yaw_rate"],
                traj_params=self.data["traj_params"],
                target_trajectory=self.data["target_trajectory"],
                pos_history=self.data["pos_history"], observations=self.data["observations"],
                rng_states=self.data["rng_states"], step_counts=self.data["step_counts"],
                num_agents=self.env.num_agents, reset_indices=np.array([0], dtype=np.int32)
            )

            pbar = tqdm(range(self.episode_length), desc=f"Ep {ep+1}/{total_episodes}")
            ep_loss = 0

            for t in pbar:
                # 1. Compute Actions
                actions = self.simple_controller()

                # 2. Step Env
                env_ids_to_step = np.array([0], dtype=np.int32)
                self.env.step_function(
                    pos_x=self.data["pos_x"], pos_y=self.data["pos_y"], pos_z=self.data["pos_z"],
                    vel_x=self.data["vel_x"], vel_y=self.data["vel_y"], vel_z=self.data["vel_z"],
                    roll=self.data["roll"], pitch=self.data["pitch"], yaw=self.data["yaw"],
                    masses=self.data["masses"], drag_coeffs=self.data["drag_coeffs"], thrust_coeffs=self.data["thrust_coeffs"],
                    target_vx=self.data["target_vx"], target_vy=self.data["target_vy"], target_vz=self.data["target_vz"], target_yaw_rate=self.data["target_yaw_rate"],
                    vt_x=self.data["vt_x"], vt_y=self.data["vt_y"], vt_z=self.data["vt_z"],
                    traj_params=self.data["traj_params"], # New
                    target_trajectory=self.data["target_trajectory"], # New
                    pos_history=self.data["pos_history"],
                    observations=self.data["observations"], rewards=self.data["rewards"],
                    reward_components=self.data["reward_components"], # New
                    done_flags=self.data["done_flags"], step_counts=self.data["step_counts"],
                    actions=actions,
                    num_agents=self.env.num_agents, episode_length=self.episode_length,
                    env_ids=env_ids_to_step
                )

                # 3. Train AE
                # Extract history (first 300)
                obs_np = self.data["observations"][:, :300]
                obs_tensor = torch.from_numpy(obs_np).float().to(self.device)

                # Forward
                latent, recon = self.ae(obs_tensor)

                # Loss
                loss = self.criterion(recon, obs_tensor)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_val = loss.item()
                self.loss_history.append(loss_val)
                ep_loss += loss_val

                pbar.set_postfix({"loss": f"{loss_val:.6f}"})
                step_ctr += 1

            mean_ep_loss = ep_loss / self.episode_length
            print(f"Episode {ep+1} Mean Loss: {mean_ep_loss:.6f}")
            self.plot_loss()

            # Save Model Checkpoint
            checkpoint_data = {
                "model_state_dict": self.ae.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
                "episode": ep
            }
            torch.save(checkpoint_data, "ae_model.pth")

            if mean_ep_loss < target_loss:
                print(f"Target loss {target_loss} reached! Stopping.")
                break

        print("Training finished.")

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.title("Autoencoder Training Loss (L1)")
        plt.xlabel("Step")
        plt.ylabel("L1 Loss")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig("ae_training_loss.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    trainer = AETrainer(num_agents=args.agents, load_checkpoint=args.load)
    trainer.train(num_episodes=args.episodes)
