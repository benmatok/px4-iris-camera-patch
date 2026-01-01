
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
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / (vg_sum + 1e-10)))

        for m in self.modules:
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
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m)
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)
        self._step(closure)
        self.steps += 1

class KFACOptimizerPlaceholder(KFACOptimizer):
    pass

# --- End KFAC Implementation ---

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

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # Conv 1
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Conv 2
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Residual Connection
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

        # Input/Output Scalers
        self.scale_att = ChannelScaler(3, init_scale=0.3, init_bias=0.0)
        self.scale_alt = ChannelScaler(1, init_scale=0.1, init_bias=-1.0)
        self.scale_ctrl = ChannelScaler(4, init_scale=[2.0, 0.2, 0.2, 0.2], init_bias=[-1.0, 0.0, 0.0, 0.0])
        self.scale_track = ChannelScaler(2, init_scale=0.1, init_bias=0.0)

        inv_scales = [3.33]*3 + [10.0] + [0.5, 5.0, 5.0, 5.0] + [10.0]*2
        inv_biases = [0.0]*3 + [10.0] + [0.5, 0.0, 0.0, 0.0] + [0.0]*2
        self.out_scaler = ChannelScaler(input_dim, init_scale=inv_scales, init_bias=inv_biases)

        # Encoder (TCN)
        layers = []
        num_levels = len(num_channels)
        kernel_size = 3
        in_channels = input_dim

        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size # Causal padding

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=0.05))
            in_channels = out_channels

        self.tcn_encoder = nn.Sequential(*layers)

        # Projection to Latent
        # We take the last timestep of the TCN output as the summary of the sequence
        self.to_latent = nn.Linear(num_channels[-1], latent_dim)

        # Decoder
        # Latent -> TCN -> Reconstructed Sequence
        # We project latent back to TCN channel dim, repeat across time, and refine with TCN
        self.from_latent = nn.Linear(latent_dim, num_channels[-1])

        decoder_layers = []
        # Same structure as encoder but can be non-causal if we want, or just causal.
        # Autoencoders often use symmetric structures.

        dec_channels = num_channels[::-1] # Reverse: [64, 64, 32, 32]
        in_channels = dec_channels[0]

        for i in range(num_levels):
            dilation_size = 2 ** i # Reuse dilation pattern or reverse? Standard is symmetric.
            # For reconstruction, causality isn't strictly required in decoder, but standard TCN blocks are handy.
            out_channels = dec_channels[i] if i < num_levels - 1 else 32 # Taper down
            padding = (kernel_size - 1) * dilation_size

            decoder_layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=0.05))
            in_channels = out_channels

        self.tcn_decoder = nn.Sequential(*decoder_layers)

        # Final projection to input_dim
        self.final_conv = nn.Conv1d(in_channels, input_dim, 1)

    def forward(self, x):
        # x: (Batch, 300) -> (Batch, 10, 30) (after reshape and permute)
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.seq_len, self.input_dim).permute(0, 2, 1) # (N, 10, 30)

        # Split and Scale
        x_att = self.scale_att(x_reshaped[:, 0:3, :])
        x_alt = self.scale_alt(x_reshaped[:, 3:4, :])
        x_ctrl = self.scale_ctrl(x_reshaped[:, 4:8, :])
        x_track = self.scale_track(x_reshaped[:, 8:10, :])

        x_scaled = torch.cat([x_att, x_alt, x_ctrl, x_track], dim=1) # (N, 10, 30)

        # Encoder
        enc_out = self.tcn_encoder(x_scaled) # (N, C_out, 30)

        # Global Pooling or Last Timestep?
        # TCNs with causal padding: Last timestep contains info from full receptive field.
        # RF check: 32 layers? No, 4 layers: 1, 2, 4, 8. RF = 1 + 2*(1+2+4+8) = 31.
        # So last timestep covers 30 inputs.

        last_step = enc_out[:, :, -1] # (N, C_out)
        latent = torch.tanh(self.to_latent(last_step)) # (N, Latent)

        # Decoder
        # Project back
        dec_init = self.from_latent(latent) # (N, C_out)

        # Repeat for sequence length
        # (N, C, 1) -> (N, C, 30)
        dec_in = dec_init.unsqueeze(2).expand(-1, -1, self.seq_len)

        # Pass through Decoder TCN
        dec_features = self.tcn_decoder(dec_in) # (N, C_final, 30)

        # Project to Output
        recon = self.final_conv(dec_features) # (N, 10, 30)

        # Inverse Scale
        recon_scaled = self.out_scaler(recon)

        # Flatten
        recon_flat = recon_scaled.permute(0, 2, 1).reshape(batch_size, -1)

        return latent, recon_flat


class DronePolicy(nn.Module):
    def __init__(self, observation_dim=308, action_dim=4, hidden_dim=256, env=None, hidden_dims=None):
        super(DronePolicy, self).__init__()
        # Observation space is 308 (300 history + 4 target + 4 tracker)

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.history_dim = 300
        self.target_dim = 4 # Target Commands
        self.tracker_dim = 4 # Tracker Features

        self.latent_dim = 64 # Increased latent space

        # Compatibility with different init styles
        if hidden_dims is None:
            hidden_dims = [hidden_dim, hidden_dim]

        self.ae = TCNAutoencoder(input_dim=10, seq_len=30, latent_dim=self.latent_dim)

        # RL Agent Input: Latent(64) + Target(4) + Tracker(4) = 72
        input_dim = self.latent_dim + self.target_dim + self.tracker_dim

        # Separate feature extraction from heads
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
        self.action_log_std = nn.Parameter(torch.ones(1, self.action_dim) * -0.69)

    def forward(self, obs):
        # Obs: (Batch, 308)

        history = obs[:, :self.history_dim]
        # targets = 300:304
        # tracker = 304:308
        aux_features = obs[:, self.history_dim:] # (Batch, 8)

        # Autoencode
        latent, recon = self.ae(history)

        # RL Input
        rl_input = torch.cat([latent, aux_features], dim=1) # 64 + 8 = 72

        # Feature Extraction
        features = self.feature_extractor(rl_input)

        # Heads
        action_logits = self.action_head(features)
        value = self.value_head(features)

        return action_logits, value
