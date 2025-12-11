
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
    # Specialized for 1D: (batch, in_c, L)
    # Conv1d expects (batch, in_c, L)
    # unfold(dimension, size, step)
    if padding > 0:
        x = F.pad(x, (padding, padding))
    x = x.unfold(2, kernel_size, stride) # (batch, in_c, n_patches, kernel_size)
    x = x.transpose(1, 2) # (batch, n_patches, in_c, kernel_size)
    x = x.contiguous().view(x.size(0), x.size(1), -1) # (batch, n_patches, in_c*kernel_size)
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
        # a: (batch, in_c, L)
        batch_size = a.size(0)
        # padding is int or tuple. nn.Conv1d padding is usually tuple or int.
        padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
        kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride

        a = _extract_patches(a, kernel_size, stride, padding)
        # a: (batch, n_patches, features)
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
        # g: (batch, out_c, L_out)
        spatial_size = g.size(2)
        batch_size = g.shape[0]
        g = g.transpose(1, 2) # (batch, L_out, out_c)
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
        # symeig is deprecated, using linalg.eigh
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
        # Q_g (1/R_g) Q_g^T @ p_grad_mat @ Q_a (1/R_a) Q_a^T
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
                        buf.mul_(momentum).add_(d_p) # Fixed Nesterov logic simplified
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

# --- End KFAC Implementation ---

class Autoencoder1D(nn.Module):
    def __init__(self, input_dim=6, seq_len=300, latent_dim=20):
        super(Autoencoder1D, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: 1D Conv for sequence length 300
        # Input: (Batch, 6, 300)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=5, stride=2, padding=2), # -> 150
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), # -> 75
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=3, padding=1), # -> 25
            nn.ReLU(),
            nn.Flatten(), # 64*25 = 1600
            nn.Linear(1600, latent_dim),
            nn.Tanh() # Latent vector
        )

        # Decoder
        self.decoder_linear = nn.Linear(latent_dim, 1600)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=3, padding=1, output_padding=0), # 25 -> 75
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1), # 75 -> 150
            nn.ReLU(),
            nn.ConvTranspose1d(16, input_dim, kernel_size=5, stride=2, padding=2, output_padding=1), # 150 -> 300
        )

    def forward(self, x):
        # x is (Batch, 1800). Reshape to (Batch, 6, 300)
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.seq_len, self.input_dim).permute(0, 2, 1) # (N, 6, 300)

        latent = self.encoder(x_reshaped)

        recon_features = self.decoder_linear(latent)
        recon_features = recon_features.view(batch_size, 64, 25)
        recon = self.decoder(recon_features) # (N, 6, 300)

        # Flatten recon to match input
        recon_flat = recon.permute(0, 2, 1).reshape(batch_size, -1)

        return latent, recon_flat

class DronePolicy(nn.Module):
    def __init__(self, env, hidden_dims=[128, 128]):
        super(DronePolicy, self).__init__()
        # Observation space is 1804 (1800 history + 4 target)

        self.history_dim = 1800
        self.target_dim = 4
        self.latent_dim = 20
        self.action_dim = 4

        self.ae = Autoencoder1D(input_dim=6, seq_len=300, latent_dim=self.latent_dim)

        # RL Agent Input: Latent(20) + Target(4) = 24
        input_dim = self.latent_dim + self.target_dim

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.action_dim))

        self.policy_head = nn.Sequential(*layers)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, obs):
        # Obs: (Batch, 1804)

        history = obs[:, :self.history_dim]
        targets = obs[:, self.history_dim:]

        # Autoencode
        latent, recon = self.ae(history)

        # RL Input
        rl_input = torch.cat([latent, targets], dim=1)

        # Policy
        x = rl_input
        for layer in self.policy_head[:-1]:
             x = layer(x)

        action_logits = self.policy_head[-1](x)
        value = self.value_head(rl_input)

        return action_logits, value, recon, history
