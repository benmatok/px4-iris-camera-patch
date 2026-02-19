import numpy as np
import logging
import math
from flight_config import FlightConfig
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# --- Inlined SE3 Utils (NumPy) ---
def so3_exp_np(omega):
    theta_sq = np.dot(omega, omega)
    theta = math.sqrt(theta_sq)
    K = np.array([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0]
    ], dtype=np.float32)
    if theta < 1e-6:
        return np.eye(3, dtype=np.float32) + K + 0.5 * np.matmul(K, K)
    else:
        inv_theta = 1.0 / theta
        a = math.sin(theta) * inv_theta
        b = (1.0 - math.cos(theta)) * (inv_theta * inv_theta)
        return np.eye(3, dtype=np.float32) + a * K + b * np.matmul(K, K)

def rpy_to_matrix_np(roll, pitch, yaw):
    cr = math.cos(roll); sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw); sy = math.sin(yaw)
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr
    return R

def matrix_to_rpy_np(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw

class NumPyGhostModel:
    def __init__(self, mass=1.0, drag_coeff=0.1, thrust_coeff=1.0, tau=0.1, g=9.81, max_thrust_base=20.0):
        self.mass = mass
        self.drag_coeff = drag_coeff
        self.thrust_coeff = thrust_coeff
        self.tau = tau
        self.g = g
        self.max_thrust_base = max_thrust_base

    def rollout(self, initial_state_dict, action_seq, dt=0.05):
        """
        Rolls out the model trajectory given an initial state (dict) and action sequence (H, 4).
        Returns:
            trajectory: (H, 12) numpy array
        """
        px, py, pz = initial_state_dict['px'], initial_state_dict['py'], initial_state_dict['pz']
        vx, vy, vz = initial_state_dict['vx'], initial_state_dict['vy'], initial_state_dict['vz']
        roll, pitch, yaw = initial_state_dict['roll'], initial_state_dict['pitch'], initial_state_dict['yaw']
        wx, wy, wz = initial_state_dict['wx'], initial_state_dict['wy'], initial_state_dict['wz']

        H = len(action_seq)
        traj = np.zeros((H, 12), dtype=np.float32)

        for t in range(H):
            u = action_seq[t]
            thrust_cmd = u[0]
            r_rate_cmd = u[1]
            p_rate_cmd = u[2]
            y_rate_cmd = u[3]

            alpha = dt / self.tau
            denom = 1.0 + alpha

            next_wx = (wx + r_rate_cmd * alpha) / denom
            next_wy = (wy + p_rate_cmd * alpha) / denom
            next_wz = (wz + y_rate_cmd * alpha) / denom

            avg_wx = 0.5 * (wx + next_wx)
            avg_wy = 0.5 * (wy + next_wy)
            avg_wz = 0.5 * (wz + next_wz)

            # Using same PyGhostModel convention: (r, -p, -y)
            R_curr = rpy_to_matrix_np(roll, -pitch, -yaw)
            omega_vec = np.array([avg_wx, avg_wy, avg_wz], dtype=np.float32) * dt
            R_update = so3_exp_np(omega_vec)
            R_next = np.matmul(R_curr, R_update)

            next_roll, next_p_inv, next_y_inv = matrix_to_rpy_np(R_next)
            next_pitch = -next_p_inv
            next_yaw = -next_y_inv

            max_thrust = self.max_thrust_base * self.thrust_coeff
            # Ensure thrust > 0 (ReLU equivalent)
            thrust_force = max(0.0, thrust_cmd) * max_thrust

            # Thrust in Body: [0, 0, T] (ENU Z is Up, but this is weird)
            # PyGhostModel uses R_next[0,2], [1,2], [2,2]
            ax_dir = R_next[0, 2]
            ay_dir = R_next[1, 2]
            az_dir = R_next[2, 2]

            ax_thrust = thrust_force * ax_dir / self.mass
            ay_thrust = thrust_force * ay_dir / self.mass
            az_thrust = thrust_force * az_dir / self.mass

            ax_drag = -self.drag_coeff * vx
            ay_drag = -self.drag_coeff * vy
            az_drag = -self.drag_coeff * vz

            ax = ax_thrust + ax_drag
            ay = ay_thrust + ay_drag
            az = az_thrust + az_drag - self.g

            next_vx = vx + ax * dt
            next_vy = vy + ay * dt
            next_vz = vz + az * dt

            next_px = px + next_vx * dt
            next_py = py + next_vy * dt
            next_pz = pz + next_vz * dt

            # Update state
            px, py, pz = next_px, next_py, next_pz
            vx, vy, vz = next_vx, next_vy, next_vz
            roll, pitch, yaw = next_roll, next_pitch, next_yaw
            wx, wy, wz = next_wx, next_wy, next_wz

            traj[t, 0] = px
            traj[t, 1] = py
            traj[t, 2] = pz
            traj[t, 3] = vx
            traj[t, 4] = vy
            traj[t, 5] = vz
            traj[t, 6] = roll
            traj[t, 7] = pitch
            traj[t, 8] = yaw
            traj[t, 9] = wx
            traj[t, 10] = wy
            traj[t, 11] = wz

        return traj

class GDPCOptimizer:
    def __init__(self, config: FlightConfig):
        self.config = config
        self.gdpc_cfg = config.gdpc

        # Initialize Model with Physics Config
        phy = config.physics
        self.model = NumPyGhostModel(
            mass=phy.mass,
            drag_coeff=phy.drag_coeff,
            thrust_coeff=phy.thrust_coeff,
            tau=phy.tau,
            g=phy.g,
            max_thrust_base=phy.max_thrust_base
        )
        self.u_seq = None # Shape (H, 4)

    def reset(self):
        self.u_seq = None

    def compute_action(self, state_obs, target_pos_world):
        """
        Uses SciPy Optimize to find the best action sequence.
        """
        H = self.gdpc_cfg.horizon

        # Initialize u_seq if needed
        if self.u_seq is None or self.u_seq.shape[0] != H:
            self.u_seq = np.zeros((H, 4))
            self.u_seq[:, 0] = 0.55 # Hover thrust approx
        else:
            # Shift
            self.u_seq[:-1] = self.u_seq[1:]
            self.u_seq[-1] = self.u_seq[-2] # Repeat last action

        # Optimization Function
        # target_pos_world is actually RELATIVE target position
        def cost_fn(u_flat):
            u = u_flat.reshape((H, 4))
            traj = self.model.rollout(state_obs, u)

            pos_traj = traj[:, 0:3]
            vel_traj = traj[:, 3:6]
            att_traj = traj[:, 6:9] # roll, pitch, yaw

            # Target cost (Distance)
            # Expand target to match H if necessary, or just compute distance to target point for all steps
            dist_sq = np.sum((pos_traj - target_pos_world)**2, axis=1)
            # Prevent overflow by clamping
            dist_sq = np.clip(dist_sq, 0, 1e6)
            loss_pos = np.mean(dist_sq) * self.gdpc_cfg.w_pos

            # Velocity cost (Damping)
            v_sq = np.sum(vel_traj**2, axis=1)
            v_sq = np.clip(v_sq, 0, 1e6)
            loss_vel = np.mean(v_sq) * self.gdpc_cfg.w_vel

            # Attitude cost (Roll/Pitch penalty)
            loss_roll = np.mean(att_traj[:, 0]**2) * self.gdpc_cfg.w_roll
            loss_pitch = np.mean(att_traj[:, 1]**2) * self.gdpc_cfg.w_pitch

            # Control Input cost
            loss_thrust = np.mean((u[:, 0] - 0.5)**2) * self.gdpc_cfg.w_thrust

            # Smoothness
            u_delta = u[1:] - u[:-1]
            loss_smooth = np.mean(np.sum(u_delta**2, axis=1)) * self.gdpc_cfg.w_smoothness

            # Terminal Cost (Encourage getting closer at the end)
            final_pos = pos_traj[-1]
            final_dist_sq = np.sum((final_pos - target_pos_world)**2)
            final_dist_sq = np.clip(final_dist_sq, 0, 1e6)
            loss_terminal = final_dist_sq * self.gdpc_cfg.w_terminal

            # Terminal Velocity Cost (Stop at the end)
            final_vel = vel_traj[-1]
            loss_terminal_vel = np.sum(final_vel**2) * self.gdpc_cfg.w_terminal_vel
            loss_terminal_vel = np.clip(loss_terminal_vel, 0, 1e6)

            # Penalty for excessive Roll/Pitch Rate (Constraint Softening)
            rates = u[:, 1:]
            rate_penalty = np.mean(np.sum(rates**2, axis=1)) * 0.01

            # Total Loss
            total_loss = loss_pos + loss_vel + loss_roll + loss_pitch + loss_thrust + loss_smooth + loss_terminal + loss_terminal_vel + rate_penalty

            # Check for NaN/Inf
            if not np.isfinite(total_loss):
                return 1e12 # Return huge cost if exploded
            return total_loss

        # Bounds
        bounds = []
        for _ in range(H):
            bounds.append((0.0, 1.0))     # Thrust
            bounds.append((-2.5, 2.5))    # Roll Rate
            bounds.append((-2.5, 2.5))    # Pitch Rate
            bounds.append((-2.5, 2.5))    # Yaw Rate

        # Initial Guess
        x0 = self.u_seq.flatten()

        # Optimize (Using L-BFGS-B)
        # Reduce maxiter for speed
        res = minimize(cost_fn, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': self.gdpc_cfg.opt_steps, 'disp': False})

        if res.success or res.message: # Just use result even if not converged fully
             self.u_seq = res.x.reshape((H, 4))

        action_np = self.u_seq[0]

        # Generate final trajectory for visualization
        final_traj = self.model.rollout(state_obs, self.u_seq)

        return {
            'thrust': float(action_np[0]),
            'roll_rate': float(action_np[1]),
            'pitch_rate': float(action_np[2]),
            'yaw_rate': float(action_np[3])
        }, final_traj
