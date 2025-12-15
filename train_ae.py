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
        self.ae = Autoencoder1D(input_dim=6, seq_len=300, latent_dim=20).to(self.device)

        if load_checkpoint and os.path.exists(load_checkpoint):
            print(f"Loading checkpoint from {load_checkpoint}...")
            self.ae.load_state_dict(torch.load(load_checkpoint, map_location=self.device))

        # Use KFAC Optimizer
        print("Using KFAC Optimizer")
        self.optimizer = KFACOptimizer(self.ae, lr=lr)

        # Use L1Loss to match train_drone.py
        self.criterion = nn.L1Loss()

        self.loss_history = []

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

        # Velocity Control (World Frame approx for small angles)
        # Pitch produces forward acceleration (positive pitch -> backward force? No.
        # Body frame: x forward. Positive pitch (nose up) -> backward force.
        # So to accelerate forward (+x), we need negative pitch (nose down).
        # target_pitch = -gain * error_x

        v_err_x = target_vx - vel_x
        v_err_y = target_vy - vel_y
        v_err_z = target_vz - vel_z

        # Desired Attitude
        target_pitch = -kp_vel * v_err_x
        target_roll = kp_vel * v_err_y # Positive roll -> right force -> +y

        # Rate commands
        roll_rate_cmd = kp_att * (target_roll - roll)
        pitch_rate_cmd = kp_att * (target_pitch - pitch)
        yaw_rate_cmd = kp_yaw * (target_yaw_rate - 0.0) # Tracking yaw rate directly usually?
        # Actually target_yaw_rate is the command for yaw rate.
        # But we control yaw rate directly?
        # Actions are: Thrust, RollRate, PitchRate, YawRate.
        # So we can just feed target_yaw_rate if we want to spin.
        yaw_rate_cmd = target_yaw_rate

        # Thrust command
        # Hover thrust ~ 0.5 (assuming T_max=20, m=1, g=9.8 -> 10/20 = 0.5)
        # Add Z velocity control
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
        print(f"Starting training for max {num_episodes} episodes or until loss < {target_loss}...")

        step_ctr = 0

        for ep in range(num_episodes):
            # Reset
            # Pass all environment IDs if needed, though pure CPU usually ignores this except for step counts.
            # Assuming env_ids matches reset_indices.
            # If we assume 1 block with num_agents, env_id=0 is correct.
            self.env.reset_function(
                pos_x=self.data["pos_x"], pos_y=self.data["pos_y"], pos_z=self.data["pos_z"],
                vel_x=self.data["vel_x"], vel_y=self.data["vel_y"], vel_z=self.data["vel_z"],
                roll=self.data["roll"], pitch=self.data["pitch"], yaw=self.data["yaw"],
                masses=self.data["masses"], drag_coeffs=self.data["drag_coeffs"], thrust_coeffs=self.data["thrust_coeffs"],
                target_vx=self.data["target_vx"], target_vy=self.data["target_vy"], target_vz=self.data["target_vz"], target_yaw_rate=self.data["target_yaw_rate"],
                pos_history=self.data["pos_history"], observations=self.data["observations"],
                rng_states=self.data["rng_states"], step_counts=self.data["step_counts"],
                num_agents=self.env.num_agents, reset_indices=np.array([0], dtype=np.int32)
            )

            pbar = tqdm(range(self.episode_length), desc=f"Ep {ep+1}/{num_episodes}")
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
                    pos_history=self.data["pos_history"],
                    observations=self.data["observations"], rewards=self.data["rewards"],
                    done_flags=self.data["done_flags"], step_counts=self.data["step_counts"],
                    actions=actions,
                    num_agents=self.env.num_agents, episode_length=self.episode_length,
                    env_ids=env_ids_to_step
                )

                # 3. Train AE
                obs_np = self.data["observations"][:, :1800]
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
            torch.save(self.ae.state_dict(), "ae_model.pth")

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
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    trainer = AETrainer(num_agents=args.agents, load_checkpoint=args.load)
    trainer.train(num_episodes=args.episodes)
