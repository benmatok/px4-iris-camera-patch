import numpy as np
import torch
import time
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DPCPlanner:
    """
    Differentiable Predictive Control (DPC) Planner.
    Uses a 'Fan of Beliefs' (Ghosts) to handle partial observability (Wind).
    Optimizes a single action for robustness across multiple wind hypotheses.
    """
    def __init__(self, num_agents, dt=0.05):
        self.num_agents = num_agents
        self.dt = dt
        self.g = 9.81
        self.horizon = 10 # 0.5s prediction
        self.opt_steps = 10 # Optimization steps
        self.device = 'cpu'

    def compute_actions(self, current_state, target_pos):
        t0 = time.time()

        # Unpack State
        px = current_state['pos_x']
        py = current_state['pos_y']
        pz = current_state['pos_z']
        vx = current_state['vel_x']
        vy = current_state['vel_y']
        vz = current_state['vel_z']
        roll = current_state['roll']
        pitch = current_state['pitch']
        yaw = current_state['yaw']

        mass = current_state['masses']
        drag = current_state['drag_coeffs']
        thrust_coeff = current_state['thrust_coeffs']

        # Target
        tx, ty, tz = target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]

        # --- Heuristic Initialization (from LinearPlanner) ---
        dx = tx - px
        dy = ty - py
        dz = tz - pz
        dist_xy = np.sqrt(dx**2 + dy**2) + 1e-6
        dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6

        # Simple PID for initialization
        speed_ref = np.minimum(10.0, dist)
        vx_des = (dx / dist) * speed_ref
        vy_des = (dy / dist) * speed_ref
        vz_des = (dz / dist) * speed_ref

        ax_cmd = 2.0 * (vx_des - vx)
        ay_cmd = 2.0 * (vy_des - vy)
        az_cmd = 2.0 * (vz_des - vz)

        Fx_req = mass * ax_cmd + drag * vx
        Fy_req = mass * ay_cmd + drag * vy
        Fz_req = mass * az_cmd + drag * vz + mass * self.g

        F_total = np.sqrt(Fx_req**2 + Fy_req**2 + Fz_req**2) + 1e-6
        max_thrust = 20.0 * thrust_coeff
        thrust_cmd = np.clip(F_total / max_thrust, 0.0, 1.0)

        # Attitude
        yaw_des = np.arctan2(dy, dx)
        # Simplified Pitch/Roll from F vector
        # This is just a warm start, DPC will refine it.
        # Assuming small roll for simplicity in init
        roll_cmd = 0.0
        pitch_cmd = 0.0 # Will be optimized
        yaw_rate_cmd = (yaw_des - yaw) * 2.0

        # Initial Action Guess: (N, 4)
        init_actions = np.zeros((self.num_agents, 4), dtype=np.float32)
        init_actions[:, 0] = thrust_cmd
        init_actions[:, 1] = 0.0 # Roll rate
        init_actions[:, 2] = 0.0 # Pitch rate
        init_actions[:, 3] = np.clip(yaw_rate_cmd, -5.0, 5.0)

        # --- DPC Optimization ---

        # Convert to Tensor
        act_tensor = torch.tensor(init_actions, dtype=torch.float32, requires_grad=True, device=self.device)

        # Optimizer
        opt = torch.optim.SGD([act_tensor], lr=0.1, momentum=0.5)

        # Prepare State Tensors (N, 3)
        pos_t = torch.tensor(np.stack([px, py, pz], axis=1), dtype=torch.float32, device=self.device)
        vel_t = torch.tensor(np.stack([vx, vy, vz], axis=1), dtype=torch.float32, device=self.device)
        att_t = torch.tensor(np.stack([roll, pitch, yaw], axis=1), dtype=torch.float32, device=self.device)
        target_t = torch.tensor(target_pos, dtype=torch.float32, device=self.device)
        mass_t = torch.tensor(mass, dtype=torch.float32, device=self.device).unsqueeze(1)
        drag_t = torch.tensor(drag, dtype=torch.float32, device=self.device).unsqueeze(1)
        thrust_coeff_t = torch.tensor(thrust_coeff, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Expand for Ghosts (N*3, ...)
        # Ghost 1: Wind (0,0,0)
        # Ghost 2: Headwind (Body X = -2) -> Needs rotation
        # Ghost 3: Crosswind (Body Y = 2) -> Needs rotation
        # We calculate wind in World Frame for each agent based on its CURRENT Yaw

        # Wind Vectors in Body Frame
        # Shape (3, 3) -> (Ghost, XYZ)
        wind_body = torch.tensor([
            [0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0]
        ], dtype=torch.float32, device=self.device)

        # Expand State: Repeat 3 times interleaved
        # (N, 3) -> (N, 3, 3) -> (N*3, 3)
        pos_exp = pos_t.repeat_interleave(3, dim=0)
        vel_exp = vel_t.repeat_interleave(3, dim=0)
        att_exp = att_t.repeat_interleave(3, dim=0)
        target_exp = target_t.repeat_interleave(3, dim=0)
        mass_exp = mass_t.repeat_interleave(3, dim=0)
        drag_exp = drag_t.repeat_interleave(3, dim=0)
        thrust_coeff_exp = thrust_coeff_t.repeat_interleave(3, dim=0)

        # Calculate Wind in World Frame for each Ghost/Agent
        # Yaw is in att_exp[:, 2]
        y_yaw = att_exp[:, 2]
        cy = torch.cos(y_yaw)
        sy = torch.sin(y_yaw)

        # Replicate wind_body to (N, 3, 3) -> (N*3, 3)
        # We need to map Ghost ID to row of wind_body
        # Indices: 0, 1, 2, 0, 1, 2 ...
        wind_indices = torch.arange(3, device=self.device).repeat(self.num_agents)
        wb = wind_body[wind_indices] # (N*3, 3)

        # Rotate Body Wind to World
        # Wx_w = Wx_b * cy - Wy_b * sy
        # Wy_w = Wx_b * sy + Wy_b * cy
        wx_w = wb[:, 0] * cy - wb[:, 1] * sy
        wy_w = wb[:, 0] * sy + wb[:, 1] * cy
        wz_w = wb[:, 2]

        wind_world = torch.stack([wx_w, wy_w, wz_w], dim=1) # (N*3, 3)

        # --- Optimization Loop ---
        for step in range(self.opt_steps):
            opt.zero_grad()

            # Action Expansion (Same action for all ghosts)
            # act_tensor (N, 4) -> (N*3, 4)
            act_exp = act_tensor.repeat_interleave(3, dim=0)

            # Clamp Actions
            act_clamped = torch.zeros_like(act_exp)
            act_clamped[:, 0] = torch.clamp(act_exp[:, 0], 0.0, 1.0)
            act_clamped[:, 1:] = torch.clamp(act_exp[:, 1:], -10.0, 10.0)

            # Unroll Trajectory
            curr_pos = pos_exp
            curr_vel = vel_exp
            curr_att = att_exp

            total_cost_per_ghost = torch.zeros((self.num_agents * 3, 1), device=self.device)

            # Store u predictions for Variance "Excite" term
            u_preds = []

            for h in range(self.horizon):
                next_pos, next_vel, next_att, cam_uv = self._differentiable_step(
                    curr_pos, curr_vel, curr_att, act_clamped,
                    mass_exp, drag_exp, thrust_coeff_exp, target_exp, wind_world
                )

                # Extract features
                u, v = cam_uv[:, 0], cam_uv[:, 1]
                z = next_pos[:, 2:3]
                vz = next_vel[:, 2:3]

                u_preds.append(u)

                # --- Cost Functions ---

                # 1. Stage I: Acquisition (Minimize u, v, Roll, Sink)
                # Roll is att[:, 0]
                roll_err = next_att[:, 0:1].abs()
                sink_pen = torch.relu(-vz) # Penalize falling
                cost_acq = (u**2 + v**2) + 10.0 * roll_err + 5.0 * sink_pen

                # 2. Stage II: Alignment (Line-of-Sight Rate)
                # Approx LOS Rate d_lambda ~ du/dt?
                # Actually, simple proxy: minimize velocity perpendicular to LOS?
                # Or just minimize u^2 + v^2 heavily?
                # User says: Zero out "Line-of-Sight Rate" (\dot{\lambda}).
                # \dot{\lambda} \approx (u_{t} - u_{t-1}) / dt.
                # Since we don't have t-1 in this step easily without history,
                # we can use u (since target should be centered).
                # If u is constant 0, LOS rate is 0.
                # But LOS rate is about *angular* velocity of the line of sight.
                # If we are on collision course, LOS rate is zero (Parallel Navigation).
                # Implementation: Minimize (u_next - u_curr)^2 ?
                # Let's use simple tracking error for Alignment + small rate penalty.
                cost_align = (u**2 + v**2)

                # 3. Stage III: Diving (Proportional Nav)
                # Minimize \dot{\lambda}^2 * vz^2
                # We approximate \dot{\lambda} \approx u_dot \approx (u_{new} - u_{old}) / dt
                # But here we just use u^2 weighted by vz^2 (Energy)?
                # Standard PN: Accel_cmd = N * V_closing * LOS_rate.
                # Cost: Minimize (LOS_Rate)^2 * (Closing_Speed)^2.
                # Proxy: u is proportional to LOS angle (for small angles).
                # u_dot is LOS rate.
                # Let's use (u**2 + v**2) * (vz**2 + 0.1) as a proxy for "High Gain when fast".
                cost_dive = (u**2 + v**2) * (vz.abs() + 1.0)**2

                # --- Weighting ---
                # Based on Pitch (att[:, 1]) and Confidence/Error
                # Pitch > 45 deg (0.78 rad) -> Dive
                # Error > Threshold -> Acquire

                pitch_val = next_att[:, 1:2]

                # Soft switch for Dive
                # Sigmoid centered at 45 deg (0.78)
                w_dive = torch.sigmoid((pitch_val - 0.78) * 10.0)

                # Soft switch for Acquire vs Align
                # Based on pixel error
                err_sq = u**2 + v**2
                w_acq = (1.0 - w_dive) * torch.sigmoid((err_sq - 0.5) * 5.0)

                w_align = 1.0 - w_dive - w_acq

                step_cost = w_acq * cost_acq + w_align * cost_align + w_dive * cost_dive

                # Add constraints (Crash, Altitude)
                cost_crash = torch.relu(0.5 - z) * 100.0
                cost_bounds = torch.relu(z - 100.0) * 10.0

                total_cost_per_ghost += step_cost + cost_crash + cost_bounds

                curr_pos, curr_vel, curr_att = next_pos, next_vel, next_att

            # --- Excite Term (Stage II) ---
            # "Add a small negative cost to variance"
            # Variance of u across ghosts for each agent
            # u_preds: List of (N*3)
            # Stack: (Horizon, N*3)
            u_stack = torch.stack(u_preds, dim=0) # (H, N*3, 1)
            # Reshape to (H, N, 3)
            u_reshaped = u_stack.view(self.horizon, self.num_agents, 3)
            # Variance across ghosts (dim 2)
            u_var = torch.var(u_reshaped, dim=2) # (H, N)
            # We want to Maximize Variance -> Minimize Negative Variance
            # But only in Stage II?
            # User says "In the initial phase...".
            # We can weight this by w_align (averaged over horizon?)
            # Simplified: Add globally.
            loss_excite = -0.1 * u_var.mean(dim=0) # (N,)

            # --- LogSumExp Aggregation ---
            # total_cost_per_ghost: (N*3, 1)
            # Reshape to (N, 3)
            costs_reshaped = total_cost_per_ghost.squeeze(1).view(self.num_agents, 3)

            # Robust Loss: LogSumExp(costs)
            # Standard LSE: log(sum(exp(x)))
            # Since we want to MINIMIZE, and LSE is a soft MAX, minimizing LSE(costs) minimizes the worst case.
            robust_cost = torch.logsumexp(costs_reshaped, dim=1) # (N,)

            final_loss = (robust_cost + loss_excite).mean()

            final_loss.backward()
            opt.step()

        # Output Action
        final_actions = act_tensor.detach().cpu().numpy()
        final_actions[:, 0] = np.clip(final_actions[:, 0], 0.0, 1.0)
        final_actions[:, 1:] = np.clip(final_actions[:, 1:], -10.0, 10.0)

        # Timing
        elapsed = (time.time() - t0) * 1000.0 # ms
        # logging.info(f"DPC Time: {elapsed:.2f} ms") # Optional debug

        return final_actions

    def _differentiable_step(self, pos, vel, att, action, mass, drag, thrust_coeff, target, wind):
        """
        Differentiable Physics Step with Wind.
        pos, vel, att: (Batch, 3)
        wind: (Batch, 3) - World Frame Wind Vector
        """
        dt = self.dt
        g = self.g

        r, p, y = att[:, 0:1], att[:, 1:2], att[:, 2:3]

        thrust_cmd = action[:, 0:1]
        rates = action[:, 1:]

        # Update Attitude
        next_att = att + rates * dt

        # Update Velocity
        max_thrust = 20.0 * thrust_coeff
        thrust_force = thrust_cmd * max_thrust

        sr = torch.sin(r); cr = torch.cos(r)
        sp = torch.sin(p); cp = torch.cos(p)
        sy = torch.sin(y); cy = torch.cos(y)

        # Acceleration (Thrust)
        ax = thrust_force * (cy*sp*cr + sy*sr) / mass
        ay = thrust_force * (sy*sp*cr - cy*sr) / mass
        az = thrust_force * (cp*cr) / mass - g

        # Drag (Relative to Air Mass)
        # V_air = V_ground - W_ground
        vx_air = vel[:, 0:1] - wind[:, 0:1]
        vy_air = vel[:, 1:2] - wind[:, 1:2]
        vz_air = vel[:, 2:3] - wind[:, 2:3]

        ax_drag = -drag * vx_air
        ay_drag = -drag * vy_air
        az_drag = -drag * vz_air

        acc = torch.cat([ax, ay, az], dim=1) + torch.cat([ax_drag, ay_drag, az_drag], dim=1)

        next_vel = vel + acc * dt
        next_pos = pos + next_vel * dt

        # Camera Projection
        # Target Rel
        dx_w = target[:, 0:1] - next_pos[:, 0:1]
        dy_w = target[:, 1:2] - next_pos[:, 1:2]
        dz_w = target[:, 2:3] - next_pos[:, 2:3]

        # Rotate to Body
        r11 = cy * cp
        r12 = sy * cp
        r13 = -sp
        r21 = cy * sp * sr - sy * cr
        r22 = sy * sp * sr + cy * cr
        r23 = cp * sr
        r31 = cy * sp * cr + sy * sr
        r32 = sy * sp * cr - cy * sr
        r33 = cp * cr

        xb = r11 * dx_w + r12 * dy_w + r13 * dz_w
        yb = r21 * dx_w + r22 * dy_w + r23 * dz_w
        zb = r31 * dx_w + r32 * dy_w + r33 * dz_w

        s30 = 0.5; c30 = 0.866025
        xc = yb
        yc = -s30 * xb + c30 * zb
        zc = c30 * xb + s30 * zb

        zc_safe = torch.clamp(zc, min=0.1)
        u = xc / zc_safe
        v = yc / zc_safe

        return next_pos, next_vel, next_att, torch.stack([u, v], dim=1)
