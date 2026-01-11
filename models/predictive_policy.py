import torch
import torch.nn as nn
import numpy as np
import math

class Chebyshev:
    """
    Helper class for Chebyshev polynomial operations.
    Handles fitting to data and evaluating polynomials.
    """
    def __init__(self, num_points, degree, device='cpu'):
        """
        Precomputes the transformation matrices for a fixed number of points and degree.
        Assumes points are linearly spaced in time.
        """
        self.num_points = num_points
        self.degree = degree
        self.device = device

        # 1. Setup domain x in [-1, 1]
        # We assume data comes in [t_start, t_end]. We map to [-1, 1].
        # For history (past -> now), points are [-3.0, ..., 0.0].
        # For future (now -> future), points are [0.0, ..., 0.5].
        # In both cases, we just linearly map the indices 0..N-1 to -1..1
        self.nodes = torch.linspace(-1, 1, steps=num_points, device=device)

        # 2. Build Design Matrix T where T[i, j] = Tj(xi)
        # Shape: (NumPoints, Degree+1)
        self.T = torch.zeros((num_points, degree + 1), device=device)
        for d in range(degree + 1):
            if d == 0:
                self.T[:, d] = 1.0
            elif d == 1:
                self.T[:, d] = self.nodes
            else:
                self.T[:, d] = 2 * self.nodes * self.T[:, d-1] - self.T[:, d-2]

        # 3. Pseudo-inverse for Least Squares fitting: (T^T T)^-1 T^T
        # Shape: (Degree+1, NumPoints)
        # Coeffs = FitMat @ Data
        self.fit_mat = torch.linalg.pinv(self.T)

    def fit(self, data):
        """
        Fits Chebyshev coefficients to the data.
        data: (Batch, Channels, NumPoints)
        Returns: (Batch, Channels, Degree+1)
        """
        # data: (B, C, N)
        # fit_mat: (D+1, N)
        # coeffs: (B, C, D+1) = data @ fit_mat.T
        return torch.matmul(data, self.fit_mat.t())

    def evaluate(self, coeffs, t_points=None):
        """
        Evaluates polynomials defined by coeffs at t_points.
        coeffs: (Batch, Channels, Degree+1)
        t_points: Optional tensor of points in [-1, 1] to evaluate at.
                  If None, evaluates at the original nodes.
        Returns: (Batch, Channels, NumPoints)
        """
        if t_points is None:
            T = self.T # (N, D+1)
        else:
            # Build T for new points
            n = t_points.shape[0]
            T = torch.zeros((n, self.degree + 1), device=self.device)
            for d in range(self.degree + 1):
                if d == 0:
                    T[:, d] = 1.0
                elif d == 1:
                    T[:, d] = t_points
                else:
                    T[:, d] = 2 * t_points * T[:, d-1] - T[:, d-2]

        # result: (B, C, N) = coeffs @ T.T
        return torch.matmul(coeffs, T.t())

class JulesPredictiveController(nn.Module):
    def __init__(self, history_len=30, history_dim=10, future_len=5, action_dim=4):
        super().__init__()

        self.history_len = history_len
        self.history_dim = history_dim
        self.future_len = future_len
        self.action_dim = action_dim

        # Configuration
        self.input_degree = 3  # Cubic fit for history -> 4 coeffs
        self.output_degree = 4 # Quartic fit for future -> 5 coeffs (Matches GradientController)

        # Input Size: 10 vars * 4 coeffs = 40
        self.encoder_input_dim = self.history_dim * (self.input_degree + 1)

        # Aux Input: Target Error (rvx, rvy, rvz, dist) + Tracker (u, v, size, conf) = 8
        self.aux_dim = 8

        # Output Size: 4 actions directly
        self.output_dim = self.action_dim

        # --- 1. THE ENCODER (The Past) ---
        self.encoder = nn.Sequential(
            nn.Linear(self.encoder_input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64)
        )

        # --- 2. THE HEAD (The Action) ---
        self.head = nn.Sequential(
            nn.Linear(64 + self.aux_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, self.output_dim),
            nn.Tanh() # Bound actions to [-1, 1]
        )

        # Helpers for Chebyshev
        # We initialize them on CPU, they will be moved to device with the model if needed (buffers)
        # Note: We can't easily register 'Chebyshev' class as a buffer, so we store the matrices
        self.cheb_hist = Chebyshev(history_len, self.input_degree)

        self.register_buffer('hist_fit_mat', self.cheb_hist.fit_mat)

    def fit_history(self, history):
        """
        Compresses raw history into coefficients.
        history: (Batch, 300) flattened -> reshape to (Batch, 10, 30)
        """
        batch_size = history.shape[0]
        # Reshape: (Batch, 30 steps, 10 features) -> (Batch, 10 features, 30 steps)
        x = history.view(batch_size, 30, 10).permute(0, 2, 1) # (B, 10, 30)

        # Fit coefficients
        # coeffs: (B, 10, 4) = x @ fit_mat.T
        coeffs = torch.matmul(x, self.hist_fit_mat.t())

        return coeffs.reshape(batch_size, -1) # Flatten to (B, 40)

    def forward(self, hist_coeffs, aux_state):
        """
        hist_coeffs: (Batch, 40) Chebyshev coefficients of history
        aux_state: (Batch, 8) current error/state (relative vel, dist, tracker)
        Returns: (Batch, 4) action
        """
        # 2. Encode
        latent = self.encoder(hist_coeffs)

        # 3. Predict Action
        fusion = torch.cat([latent, aux_state], dim=1)
        action = self.head(fusion) # (B, 4)

        return action
