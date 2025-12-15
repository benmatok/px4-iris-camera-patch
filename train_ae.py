import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

from drone_env.drone import DroneEnv
from models.ae_policy import Autoencoder1D

class AETrainer:
    def __init__(self, num_agents=5000, episode_length=100, lr=0.001):
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
        self.optimizer = optim.Adam(self.ae.parameters(), lr=lr)
        self.criterion = nn.MSELoss() # Or L1Loss

        self.loss_history = []

    def simple_controller(self):
        """
        Generates actions to stabilize the drones:
        Thrust to counter gravity + P on height.
        Roll/Pitch to counter velocity/angle.
        """
        # Access state from self.data
        vel_x = self.data["vel_x"]
        vel_y = self.data["vel_y"]
        vel_z = self.data["vel_z"]
        roll = self.data["roll"]
        pitch = self.data["pitch"]
        yaw = self.data["yaw"]

        # Simple P controller
        # Target Thrust: counteract gravity (approx) + height correction
        # We don't have perfect gravity cancellation without known mass, but mass is ~1.0
        # Hover thrust T ~ m*g. Max thrust ~ 20.
        # Action is 0..1? No, step function uses: thrust_force = thrust_cmd * max_thrust
        # max_thrust = 20 * coeff.
        # We want thrust_force ~ 9.81 * mass.
        # So thrust_cmd ~ 0.5.

        # Stabilization
        # Roll desired = -Kp * vy
        # Pitch desired = Kp * vx
        # We act on rates.

        kp_att = 2.0
        kp_vel = 0.5

        target_roll = -kp_vel * vel_y
        target_pitch = kp_vel * vel_x

        # Rate commands
        roll_rate_cmd = kp_att * (target_roll - roll)
        pitch_rate_cmd = kp_att * (target_pitch - pitch)
        yaw_rate_cmd = -1.0 * yaw # damp yaw

        # Thrust command
        # Damp Z velocity
        thrust_cmd = 0.5 - 0.1 * vel_z

        # Clip
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)
        roll_rate_cmd = np.clip(roll_rate_cmd, -1.0, 1.0)
        pitch_rate_cmd = np.clip(pitch_rate_cmd, -1.0, 1.0)
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -1.0, 1.0)

        # Pack actions
        # Shape: (num_agents, 4)
        actions = np.stack([thrust_cmd, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd], axis=1)
        return actions.flatten()

    def train(self, num_episodes=10):
        print(f"Starting training for {num_episodes} episodes...")

        step_ctr = 0

        for ep in range(num_episodes):
            # Reset
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
                # Get observations (N, 1804)
                # History is first 1800
                obs_np = self.data["observations"][:, :1800]

                # Convert to tensor
                obs_tensor = torch.from_numpy(obs_np).float().to(self.device)

                # Forward
                latent, recon = self.ae(obs_tensor)

                # Loss (Reconstruction vs Input)
                loss = self.criterion(recon, obs_tensor)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_val = loss.item()
                self.loss_history.append(loss_val)
                ep_loss += loss_val

                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                step_ctr += 1

            print(f"Episode {ep+1} Mean Loss: {ep_loss/self.episode_length:.5f}")
            self.plot_loss()

        # Save Model
        torch.save(self.ae.state_dict(), "ae_model.pth")
        print("Model saved to ae_model.pth")

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.title("Autoencoder Training Loss")
        plt.xlabel("Step")
        plt.ylabel("MSE Loss")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig("ae_training_loss.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    trainer = AETrainer(num_agents=args.agents)
    trainer.train(num_episodes=args.episodes)
