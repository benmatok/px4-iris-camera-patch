
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# --- KFAC Implementation adapted from https://github.com/alecwangcq/KFAC-Pytorch ---

def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()
    return x

def _extract_patches(x, kernel_size, stride, padding):
    if padding > 0:
        x = F.pad(x, (padding, padding))
    x = x.unfold(2, kernel_size, stride)
    x = x.transpose(1, 2)
    x = x.contiguous().view(x.size(0), x.size(1), -1)
    return x

def update_running_stat(aa, m_aa, stat_decay):
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= (1 - stat_decay)

class ComputeCovA:
    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            return cls.linear(a, layer)
        elif isinstance(layer, nn.Conv1d):
            return cls.conv1d(a, layer)
        return None

    @staticmethod
    def conv1d(a, layer):
        batch_size = a.size(0)
        padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
        kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride

        a = _extract_patches(a, kernel_size, stride, padding)
        spatial_size = a.size(1)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a / spatial_size
        return a.t() @ (a / batch_size)

    @staticmethod
    def linear(a, layer):
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)

class ComputeCovG:
    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv1d):
            return cls.conv1d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            return cls.linear(g, layer, batch_averaged)
        return None

    @staticmethod
    def conv1d(g, layer, batch_averaged):
        spatial_size = g.size(2)
        batch_size = g.shape[0]
        g = g.transpose(1, 2)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))
        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        batch_size = g.size(0)
        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)
        else:
            cov_g = g.t() @ (g / batch_size)
        return cov_g

class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True):
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged
        self.known_modules = {'Linear', 'Conv1d'}
        self.modules = []
        self.model = model
        self._prepare_model()
        self.steps = 0
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.stat_decay = stat_decay
        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            if module.__class__.__name__ in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def _update_inv(self, m):
        if m not in self.m_aa or m not in self.m_gg:
            # Skip if stats not collected (e.g. module not used in forward/backward)
            return

        eps = 1e-10
        d_a, Q_a = torch.linalg.eigh(self.m_aa[m])
        d_g, Q_g = torch.linalg.eigh(self.m_gg[m])

        self.d_a[m], self.Q_a[m] = d_a, Q_a
        self.d_g[m], self.Q_g[m] = d_g, Q_g

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())

    def _get_matrix_form_grad(self, m, classname):
        if classname == 'Conv1d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        if m.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]
        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        vg_sum = 0
        for m in self.modules:
            if m in updates:
                v = updates[m]
                vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
                if m.bias is not None:
                    vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / (vg_sum + 1e-10)))

        for m in self.modules:
            if m in updates:
                v = updates[m]
                m.weight.grad.data.copy_(v[0])
                m.weight.grad.data.mul_(nu)
                if m.bias is not None:
                    m.bias.grad.data.copy_(v[1])
                    m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                p.data.add_(d_p, alpha=-group['lr'])

    def step(self, closure=None):
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            # Skip modules without gradients or stats
            if m.weight.grad is None:
                continue
            if m not in self.m_aa or m not in self.m_gg:
                continue

            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m)

            # Check again if inv update succeeded (or existed)
            if m not in self.Q_a or m not in self.Q_g:
                continue

            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v

        if updates:
             self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1

class KFACOptimizerPlaceholder(KFACOptimizer):
    pass

# --- End KFAC Implementation ---

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out += identity
        out = self.activation(out)
        return out

class ChannelScaler(nn.Module):
    def __init__(self, num_channels, init_scale=None, init_bias=None):
        super(ChannelScaler, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))

        if init_scale is not None:
            if isinstance(init_scale, (list, tuple)):
                for i, v in enumerate(init_scale):
                    self.scale.data[0, i, 0] = v
            else:
                self.scale.data.fill_(init_scale)

        if init_bias is not None:
            if isinstance(init_bias, (list, tuple)):
                for i, v in enumerate(init_bias):
                    self.bias.data[0, i, 0] = v
            else:
                self.bias.data.fill_(init_bias)

    def forward(self, x):
        return x * self.scale + self.bias

class Autoencoder1D(nn.Module):
    def __init__(self, input_dim=10, seq_len=30, latent_dim=20):
        super(Autoencoder1D, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Structurally Aware Encoder
        # Channel Groups (Order):
        # 0-3: Control (4) [Thrust, Rates]
        # 4-6: Attitude (3) [Yaw, Pitch, Roll]
        # 7: Altitude (1) [Z]
        # 8-9: Tracker (2) [u, v]

        # Learnable Scalers (Global Parameters)

        # Branch 1: Control (4 ch)
        # Thrust: Center 0.5. (x - 0.5)*2.0 -> x*2 - 1.
        # Rates: Scale 0.2.
        self.scale_ctrl = ChannelScaler(4, init_scale=[2.0, 0.2, 0.2, 0.2], init_bias=[-1.0, 0.0, 0.0, 0.0])
        self.enc_ctrl = nn.Sequential(
            nn.Conv1d(4, 16, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 3, 2, 1), nn.LeakyReLU(0.2), # -> 15
            nn.Conv1d(32, 64, 3, 2, 1), nn.LeakyReLU(0.2), # -> 8
            nn.Flatten()
        )
        # 64 * 8 = 512

        # Branch 2: Attitude (3 ch)
        # Scale ~ 0.3 (pi -> 1). Bias 0.
        self.scale_att = ChannelScaler(3, init_scale=0.3, init_bias=0.0)
        self.enc_att = nn.Sequential(
            nn.Conv1d(3, 16, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 3, 2, 1), nn.LeakyReLU(0.2), # -> 15
            nn.Conv1d(32, 64, 3, 2, 1), nn.LeakyReLU(0.2), # -> 8
            nn.Flatten()
        )
        # 64 * 8 = 512

        # Branch 3: Altitude (1 ch)
        # 0-20. Center at 10. Scale ~ 0.1. (x-10)*0.1 -> x*0.1 - 1.0
        self.scale_alt = ChannelScaler(1, init_scale=0.1, init_bias=-1.0)
        self.enc_alt = nn.Sequential(
            nn.Conv1d(1, 8, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(8, 16, 3, 2, 1), nn.LeakyReLU(0.2), # -> 15
            nn.Conv1d(16, 32, 3, 2, 1), nn.LeakyReLU(0.2), # -> 8
            nn.Flatten()
        )
        # 32 * 8 = 256

        # Branch 4: Tracker (2 ch)
        # +/- 10. Scale 0.1.
        self.scale_track = ChannelScaler(2, init_scale=0.1, init_bias=0.0)
        self.enc_track = nn.Sequential(
            nn.Conv1d(2, 16, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 3, 2, 1), nn.LeakyReLU(0.2), # -> 15
            nn.Conv1d(32, 64, 3, 2, 1), nn.LeakyReLU(0.2), # -> 8
            nn.Flatten()
        )
        # 64 * 8 = 512

        # Total Flattened Dim = 512 + 512 + 256 + 512 = 1792
        self.flat_dim = 1792

        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        self.tanh = nn.Tanh()

        # Decoder (Shared)
        self.decoder_linear = nn.Linear(latent_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0), # 8 -> 15
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 15 -> 30
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, input_dim, kernel_size=3, stride=1, padding=1), # 30->30
        )

        # Output Scaler (Inversion)
        # Ctrl (4): Thrust (0.5, 0.5), Rates (5, 0).
        # Att (3): (3.33, 0).
        # Alt (1): (10, 10).
        # Track (2): (10, 0).
        inv_scales = [0.5, 5.0, 5.0, 5.0] + [3.33]*3 + [10.0] + [10.0]*2
        inv_biases = [0.5, 0.0, 0.0, 0.0] + [0.0]*3 + [10.0] + [0.0]*2

        self.out_scaler = ChannelScaler(input_dim, init_scale=inv_scales, init_bias=inv_biases)

    def forward(self, x):
        # x is (Batch, 300). Reshape to (Batch, 10, 30)
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.seq_len, self.input_dim).permute(0, 2, 1) # (N, 10, 30)

        # Split inputs based on structure
        # Order: Thrust, Rates(3), Yaw, Pitch, Roll, Altitude, u, v
        x_ctrl = x_reshaped[:, 0:4, :]   # Thrust, Rates
        x_att = x_reshaped[:, 4:7, :]    # Yaw, Pitch, Roll
        x_alt = x_reshaped[:, 7:8, :]    # Altitude
        x_track = x_reshaped[:, 8:10, :]  # U, V

        # Scale Inputs
        x_ctrl = self.scale_ctrl(x_ctrl)
        x_att = self.scale_att(x_att)
        x_alt = self.scale_alt(x_alt)
        x_track = self.scale_track(x_track)

        # Encode branches
        f_ctrl = self.enc_ctrl(x_ctrl)
        f_att = self.enc_att(x_att)
        f_alt = self.enc_alt(x_alt)
        f_track = self.enc_track(x_track)

        # Concatenate features
        f_all = torch.cat([f_ctrl, f_att, f_alt, f_track], dim=1)

        # Latent space
        latent = self.tanh(self.fc_enc(f_all))

        # Decode
        recon_features = self.decoder_linear(latent)
        recon_features = recon_features.view(batch_size, 128, 8)

        recon = self.decoder(recon_features) # (N, 10, 30)

        # Apply Output Scaler
        recon = self.out_scaler(recon)

        # Flatten recon to match input
        recon_flat = recon.permute(0, 2, 1).reshape(batch_size, -1)

        return latent, recon_flat

class DronePolicy(nn.Module):
    def __init__(self, observation_dim=302, action_dim=4, hidden_dim=256, env=None, hidden_dims=None, use_resnet=False, num_res_blocks=4):
        super(DronePolicy, self).__init__()
        # Observation space is 302 (300 history + 2 aux)
        # History: 30 * 10 = 300

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.history_dim = 300
        self.tracker_dim = 2 # Size, Conf (u,v are in history)
        self.latent_dim = 20

        # Compatibility with different init styles
        if hidden_dims is None:
            hidden_dims = [hidden_dim, hidden_dim]

        self.ae = Autoencoder1D(input_dim=10, seq_len=30, latent_dim=self.latent_dim)

        # RL Agent Input: Latent(20) + Tracker(2) = 22
        input_dim = self.latent_dim + self.tracker_dim

        # Separate feature extraction from heads
        if use_resnet:
            # ResNet Architecture
            # First project to hidden dim
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ]
            # Add Residual Blocks
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(hidden_dim))

            self.feature_extractor = nn.Sequential(*layers)
            in_dim = hidden_dim
        else:
            # Standard MLP Architecture
            feature_layers = []
            in_dim = input_dim
            for h_dim in hidden_dims:
                feature_layers.append(nn.Linear(in_dim, h_dim))
                feature_layers.append(nn.ReLU())
                in_dim = h_dim
            self.feature_extractor = nn.Sequential(*feature_layers)

        # Action Head
        self.action_head = nn.Sequential(
            nn.Linear(in_dim, self.action_dim),
            nn.Tanh() # Bound actions to [-1, 1]
        )

        # Value Head
        self.value_head = nn.Linear(in_dim, 1)

        # Learnable Action Variance
        # Initialize log_std to -0.69 (approx exp(-0.69) = 0.5)
        self.action_log_std = nn.Parameter(torch.ones(1, self.action_dim) * -0.69)

    def forward(self, obs):
        # Obs: (Batch, 304)

        history = obs[:, :self.history_dim]
        # tracker = 300:304
        aux_features = obs[:, self.history_dim:] # (Batch, 4)

        # Autoencode
        latent, recon = self.ae(history)

        # RL Input
        rl_input = torch.cat([latent, aux_features], dim=1) # 20 + 4 = 24

        # Feature Extraction
        features = self.feature_extractor(rl_input)

        # Heads
        action_logits = self.action_head(features)
        value = self.value_head(features)

        return action_logits, value
