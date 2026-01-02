import torch
import torch.nn as nn
import numpy as np

# --- 1. MATHEMATICAL KERNEL (Spectral Encoding) ---
class ChebyshevHistoryEncoder(nn.Module):
    """
    Compresses 4 seconds of raw history (40 steps) into 4 spectral coefficients.
    Acts as a Low-Pass Filter preserving trends, velocity, and integral windup.
    """
    def __init__(self, history_steps=40, num_coeffs=4):
        super().__init__()
        # Precompute Chebyshev Basis Matrix
        # Maps time [-1, 1] to Basis Functions T_n(x)
        x = torch.linspace(-1, 1, history_steps)
        basis = [torch.ones_like(x), x]

        # Recurrence: T_{n+1} = 2xT_n - T_{n-1}
        for i in range(2, num_coeffs):
            basis.append(2 * x * basis[-1] - basis[-2])

        # Shape: (Time, Coeffs)
        basis_matrix = torch.stack(basis, dim=1)

        # Normalize for numerical stability in RL
        basis_matrix = basis_matrix / torch.norm(basis_matrix, dim=0, keepdim=True)
        self.register_buffer('basis', basis_matrix)

    def forward(self, x):
        """
        Input: (Batch, Channels, Time) -> e.g., (B, 12, 40)
        Output: (Batch, Channels * Coeffs) -> e.g., (B, 48)
        """
        # Project History onto Basis: X @ Basis
        coeffs = torch.matmul(x, self.basis)
        return coeffs.flatten(start_dim=1)

# --- 2. THE POLICY NETWORK (Predictive Controller) ---
class JulesPolicy(nn.Module):
    def __init__(self, state_dim=12, history_dim=12, history_coeffs=4, action_coeffs=4):
        super().__init__()

        # Input Size:
        #   - Current State (12 floats)
        #   - Encoded History (12 channels * 4 coeffs = 48 floats)
        input_size = state_dim + (history_dim * history_coeffs)

        # Output Size:
        #   - 4 Control Channels (Thrust, R, P, Y) * 4 Polynomial Coeffs
        output_size = 4 * action_coeffs

        self.encoder = ChebyshevHistoryEncoder(num_coeffs=history_coeffs)

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_size)
        )

        # Init last layer to near-zero (Start by doing nothing)
        nn.init.uniform_(self.net[-1].weight, -0.001, 0.001)

    def forward(self, current_state, history_buffer):
        # 1. Compress the Past
        history_features = self.encoder(history_buffer)

        # 2. Fuse with Present
        fusion = torch.cat([current_state, history_features], dim=1)

        # 3. Predict the Future (Coefficients)
        action_coeffs = self.net(fusion)
        return action_coeffs

# --- 3. RUNTIME EXECUTION UTILS ---
def expand_action_trajectory(coeffs, duration_sec=0.5, dt=0.01):
    """
    Expands predicted coefficients into a concrete control array.
    Used at inference time to get the next control command.
    """
    num_steps = int(duration_sec / dt)
    t = np.linspace(-1, 1, num_steps)

    # Reshape coeffs: (4 controls, 4 poly coeffs)
    # Assumes input coeffs is flat array of size 16
    coeffs_matrix = coeffs.reshape(4, 4)

    # Reconstruct Polynomial: C0*1 + C1*t + C2*(2t^2-1) + C3*(4t^3-3t)
    # Chebyshev Basis (Order 4)
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t

    T = np.stack([
        np.ones_like(t),
        t,
        2 * t**2 - 1,
        4 * t**3 - 3 * t
    ])

    # Result: (4_Controls, Steps) -> Transpose to (Steps, 4_Controls)
    return (coeffs_matrix @ T).T
