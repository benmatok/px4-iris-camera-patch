
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
from drone_env.drone import DroneEnv
from models.ae_policy import Autoencoder1D, KFACOptimizer, ChannelScaler

# ==========================================
# 1. TCN Implementation (Locally defined)
# ==========================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNAutoencoder(nn.Module):
    def __init__(self, input_dim=10, seq_len=30, latent_dim=64, num_channels=[32, 32, 64, 64]):
        super(TCNAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.scale_att = ChannelScaler(3, init_scale=0.3, init_bias=0.0)
        self.scale_alt = ChannelScaler(1, init_scale=0.1, init_bias=-1.0)
        self.scale_ctrl = ChannelScaler(4, init_scale=[2.0, 0.2, 0.2, 0.2], init_bias=[-1.0, 0.0, 0.0, 0.0])
        self.scale_track = ChannelScaler(2, init_scale=0.1, init_bias=0.0)

        inv_scales = [3.33]*3 + [10.0] + [0.5, 5.0, 5.0, 5.0] + [10.0]*2
        inv_biases = [0.0]*3 + [10.0] + [0.5, 0.0, 0.0, 0.0] + [0.0]*2
        self.out_scaler = ChannelScaler(input_dim, init_scale=inv_scales, init_bias=inv_biases)

        layers = []
        num_levels = len(num_channels)
        kernel_size = 3
        in_channels = input_dim
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=0.05))
            in_channels = out_channels
        self.tcn_encoder = nn.Sequential(*layers)

        self.to_latent = nn.Linear(num_channels[-1], latent_dim)
        self.from_latent = nn.Linear(latent_dim, num_channels[-1])

        decoder_layers = []
        dec_channels = num_channels[::-1]
        in_channels = dec_channels[0]
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = dec_channels[i] if i < num_levels - 1 else 32
            padding = (kernel_size - 1) * dilation_size
            decoder_layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=0.05))
            in_channels = out_channels
        self.tcn_decoder = nn.Sequential(*decoder_layers)
        self.final_conv = nn.Conv1d(in_channels, input_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.seq_len, self.input_dim).permute(0, 2, 1)
        x_att = self.scale_att(x_reshaped[:, 0:3, :])
        x_alt = self.scale_alt(x_reshaped[:, 3:4, :])
        x_ctrl = self.scale_ctrl(x_reshaped[:, 4:8, :])
        x_track = self.scale_track(x_reshaped[:, 8:10, :])
        x_scaled = torch.cat([x_att, x_alt, x_ctrl, x_track], dim=1)

        enc_out = self.tcn_encoder(x_scaled)
        last_step = enc_out[:, :, -1]
        latent = torch.tanh(self.to_latent(last_step))

        dec_init = self.from_latent(latent)
        dec_in = dec_init.unsqueeze(2).expand(-1, -1, self.seq_len)
        dec_features = self.tcn_decoder(dec_in)
        recon = self.final_conv(dec_features)
        recon_scaled = self.out_scaler(recon)
        recon_flat = recon_scaled.permute(0, 2, 1).reshape(batch_size, -1)
        return latent, recon_flat

# ==========================================
# 2. RNN Autoencoder Implementation
# ==========================================
class RNNAutoencoder(nn.Module):
    def __init__(self, input_dim=10, seq_len=30, latent_dim=32, hidden_dim=64):
        super(RNNAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.scale_att = ChannelScaler(3, init_scale=0.3, init_bias=0.0)
        self.scale_alt = ChannelScaler(1, init_scale=0.1, init_bias=-1.0)
        self.scale_ctrl = ChannelScaler(4, init_scale=[2.0, 0.2, 0.2, 0.2], init_bias=[-1.0, 0.0, 0.0, 0.0])
        self.scale_track = ChannelScaler(2, init_scale=0.1, init_bias=0.0)

        # Encoder
        self.gru_enc = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.gru_dec = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)

        inv_scales = [3.33]*3 + [10.0] + [0.5, 5.0, 5.0, 5.0] + [10.0]*2
        inv_biases = [0.0]*3 + [10.0] + [0.5, 0.0, 0.0, 0.0] + [0.0]*2
        self.out_scaler = ChannelScaler(input_dim, init_scale=inv_scales, init_bias=inv_biases)

    def forward(self, x):
        batch_size = x.shape[0]
        # Reshape (N, 300) -> (N, 30, 10)
        x_reshaped = x.view(batch_size, self.seq_len, self.input_dim)

        # Scaling (Expects N, C, L)
        x_t = x_reshaped.permute(0, 2, 1)
        x_att = self.scale_att(x_t[:, 0:3, :])
        x_alt = self.scale_alt(x_t[:, 3:4, :])
        x_ctrl = self.scale_ctrl(x_t[:, 4:8, :])
        x_track = self.scale_track(x_t[:, 8:10, :])
        x_scaled = torch.cat([x_att, x_alt, x_ctrl, x_track], dim=1) # (N, 10, 30)

        # RNN Input: (N, L, C)
        x_rnn_in = x_scaled.permute(0, 2, 1)

        # Encoder
        _, h_n = self.gru_enc(x_rnn_in) # h_n: (1, N, Hidden)
        latent = torch.tanh(self.to_latent(h_n[-1])) # (N, Latent)

        # Decoder
        # Init state
        dec_h0 = self.from_latent(latent).unsqueeze(0) # (1, N, Hidden)

        # Decode sequence (Autoregressive or Forced?)
        # For AE, we typically feed the latent as input at every step OR just initial state.
        # Let's try feeding latent as input repeated.

        dec_in = dec_h0.permute(1, 0, 2).expand(-1, self.seq_len, -1) # (N, 30, Hidden)

        out, _ = self.gru_dec(dec_in, dec_h0) # (N, 30, Hidden)

        # Map to Output
        out_flat = out.reshape(-1, self.hidden_dim)
        pred_flat = self.head(out_flat) # (N*30, 10)
        pred = pred_flat.view(batch_size, self.seq_len, self.input_dim) # (N, 30, 10)

        # Inverse Scale
        pred_t = pred.permute(0, 2, 1)
        pred_scaled = self.out_scaler(pred_t)
        pred_out = pred_scaled.permute(0, 2, 1)

        recon_flat = pred_out.reshape(batch_size, -1)

        return latent, recon_flat

# ==========================================
# Benchmarking Logic
# ==========================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark():
    print("Initializing Environment...")
    num_agents = 512
    episode_length = 30
    env = DroneEnv(num_agents=num_agents, episode_length=episode_length, use_cuda=False)
    env.reset_all_envs()

    models = [
        ("Autoencoder1D (CNN)", Autoencoder1D()),
        ("RNNAutoencoder (GRU)", RNNAutoencoder(latent_dim=32)),
        ("TCNAutoencoder (TCN)", TCNAutoencoder(latent_dim=64))
    ]

    criterion = nn.L1Loss()
    results = []

    for name, model in models:
        print(f"\nBenchmarking {name}...")
        params = count_parameters(model)
        model.train()
        optimizer = KFACOptimizer(model, lr=0.001)

        start_time = time.time()
        start_loss = 0
        final_loss = 0
        loss_history = []

        for i in range(50):
            # Step env
            actions = np.zeros((num_agents * 4,), dtype=np.float32)
            env.step_function(
                env.data_dictionary["pos_x"], env.data_dictionary["pos_y"], env.data_dictionary["pos_z"],
                env.data_dictionary["vel_x"], env.data_dictionary["vel_y"], env.data_dictionary["vel_z"],
                env.data_dictionary["roll"], env.data_dictionary["pitch"], env.data_dictionary["yaw"],
                env.data_dictionary["masses"], env.data_dictionary["drag_coeffs"], env.data_dictionary["thrust_coeffs"],
                env.data_dictionary["target_vx"], env.data_dictionary["target_vy"], env.data_dictionary["target_vz"], env.data_dictionary["target_yaw_rate"],
                env.data_dictionary["vt_x"], env.data_dictionary["vt_y"], env.data_dictionary["vt_z"],
                env.data_dictionary["traj_params"], env.data_dictionary["target_trajectory"],
                env.data_dictionary["pos_history"], env.data_dictionary["observations"], env.data_dictionary["rewards"],
                env.data_dictionary["reward_components"],
                env.data_dictionary["done_flags"], env.data_dictionary["step_counts"], actions,
                num_agents, episode_length, env.data_dictionary["env_ids"]
            )

            obs_np = env.data_dictionary["observations"][:, :300]
            obs_tensor = torch.from_numpy(obs_np).float()

            optimizer.zero_grad()
            latent, out = model(obs_tensor)
            loss = criterion(out, obs_tensor)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if i == 0: start_loss = loss_val
            if i == 49: final_loss = loss_val

        end_time = time.time()
        avg_step_time = (end_time - start_time) / 50 * 1000

        results.append({
            "Model": name,
            "Params": params,
            "Time (ms)": f"{avg_step_time:.2f}",
            "Mean Loss": f"{np.mean(loss_history):.4f}",
            "Start": f"{start_loss:.2f}",
            "End": f"{final_loss:.2f}"
        })

    print("\n\n=== Benchmark Results (KFAC + Streaming) ===")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    benchmark()
