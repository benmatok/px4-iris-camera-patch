import numpy as np
import math
from .se3_math import so3_exp, rpy_to_matrix, matrix_to_rpy

class PyGhostModel:
    def __init__(self, mass, drag, thrust_coeff, wind_x=0.0, wind_y=0.0, tau=0.1):
        self.mass = float(mass)
        self.drag_coeff = float(drag)
        self.thrust_coeff = float(thrust_coeff)
        self.wind_x = float(wind_x)
        self.wind_y = float(wind_y)
        self.tau = float(tau)
        self.G = 9.81
        self.MAX_THRUST_BASE = 20.0

    def step(self, state_dict, action_dict, dt):
        # Unpack State
        px, py, pz = state_dict['px'], state_dict['py'], state_dict['pz']
        vx, vy, vz = state_dict['vx'], state_dict['vy'], state_dict['vz']
        roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']
        # Default to 0.0 if not present (initial step)
        wx = state_dict.get('wx', 0.0)
        wy = state_dict.get('wy', 0.0)
        wz = state_dict.get('wz', 0.0)

        # Unpack Action
        thrust = action_dict['thrust']
        roll_rate_cmd = action_dict['roll_rate']
        pitch_rate_cmd = action_dict['pitch_rate']
        yaw_rate_cmd = action_dict['yaw_rate']

        # 1. Update Angular Velocities (Implicit Lag Dynamics for Stability)
        # next_w = (w + cmd * alpha) / (1 + alpha) where alpha = dt/tau
        alpha = dt / self.tau
        denom = 1.0 + alpha
        next_wx = (wx + roll_rate_cmd * alpha) / denom
        next_wy = (wy + pitch_rate_cmd * alpha) / denom
        next_wz = (wz + yaw_rate_cmd * alpha) / denom

        # 2. Update Attitude (SE3 Manifold Spline Integration)
        # Compute Average Angular Velocity
        avg_wx = 0.5 * (wx + next_wx)
        avg_wy = 0.5 * (wy + next_wy)
        avg_wz = 0.5 * (wz + next_wz)

        # Convert current attitude to Rotation Matrix
        R_curr = rpy_to_matrix(roll, pitch, yaw)

        # Compute Rotation Update (Exponential Map)
        omega_vec = np.array([avg_wx, avg_wy, avg_wz], dtype=np.float32) * dt
        R_update = so3_exp(omega_vec)

        # Update Rotation: R_next = R_curr * R_update (Body frame update)
        R_next = np.matmul(R_curr, R_update)

        # Convert back to Euler Angles
        next_roll, next_pitch, next_yaw = matrix_to_rpy(R_next)

        # 3. Compute Forces based on New Attitude (R_next)
        max_thrust = self.MAX_THRUST_BASE * self.thrust_coeff
        thrust_force = thrust * max_thrust
        if thrust_force < 0:
            thrust_force = 0.0

        # Extract World Acceleration Direction from R_next
        ax_dir = R_next[0, 2]
        ay_dir = R_next[1, 2]
        az_dir = R_next[2, 2]

        # Accelerations
        ax_thrust = thrust_force * ax_dir / self.mass
        ay_thrust = thrust_force * ay_dir / self.mass
        az_thrust = thrust_force * az_dir / self.mass

        # Drag Force (Opposing velocity)
        ax_drag = -self.drag_coeff * (vx - self.wind_x)
        ay_drag = -self.drag_coeff * (vy - self.wind_y)
        az_drag = -self.drag_coeff * vz

        ax = ax_thrust + ax_drag
        ay = ay_thrust + ay_drag
        az = az_thrust + az_drag - self.G

        # 3. Update Velocity (Symplectic Euler)
        next_vx = vx + ax * dt
        next_vy = vy + ay * dt
        next_vz = vz + az * dt

        # 4. Update Position (Symplectic Euler: use next_v)
        next_px = px + next_vx * dt
        next_py = py + next_vy * dt
        next_pz = pz + next_vz * dt

        return {
            'px': next_px, 'py': next_py, 'pz': next_pz,
            'vx': next_vx, 'vy': next_vy, 'vz': next_vz,
            'roll': next_roll, 'pitch': next_pitch, 'yaw': next_yaw,
            'wx': next_wx, 'wy': next_wy, 'wz': next_wz
        }

    def get_gradients(self, state_dict, action_dict, dt):
        """
        Returns Jacobian (12x4) and grad_mass (12x1).
        J rows: px, py, pz, vx, vy, vz, r, p, y, wx, wy, wz
        J cols: thrust, roll_rate, pitch_rate, yaw_rate
        """
        roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']
        thrust = action_dict['thrust']

        r = roll
        p = pitch
        y = yaw

        cr = math.cos(r); sr = math.sin(r)
        cp = math.cos(p); sp = math.sin(p)
        cy = math.cos(y); sy = math.sin(y)

        max_thrust = self.MAX_THRUST_BASE * self.thrust_coeff
        F = thrust * max_thrust

        # Force Directions
        D_x = cy * sp * cr + sy * sr
        D_y = sy * sp * cr - cy * sr
        D_z = cp * cr

        J = np.zeros((12, 4), dtype=np.float32)

        # 1. Derivatives w.r.t THRUST (Column 0)
        da_dT_x = (max_thrust / self.mass) * D_x
        da_dT_y = (max_thrust / self.mass) * D_y
        da_dT_z = (max_thrust / self.mass) * D_z

        J[3, 0] = da_dT_x * dt
        J[4, 0] = da_dT_y * dt
        J[5, 0] = da_dT_z * dt

        J[0, 0] = J[3, 0] * dt
        J[1, 0] = J[4, 0] * dt
        J[2, 0] = J[5, 0] * dt

        # 2. Derivatives w.r.t RATES (Columns 1, 2, 3)
        factor_w = dt / (self.tau + dt)
        J[9, 1] = factor_w
        J[10, 2] = factor_w
        J[11, 3] = factor_w

        if abs(cp) < 1e-6: cp = 1e-6
        tt = sp / cp
        st = 1.0 / cp

        # Col 1 (Roll Rate Cmd -> Wx)
        J[6, 1] = 1.0 * dt * factor_w
        J[7, 1] = 0.0
        J[8, 1] = 0.0

        # Col 2 (Pitch Rate Cmd -> Wy)
        J[6, 2] = sr * tt * dt * factor_w
        J[7, 2] = cr * dt * factor_w
        J[8, 2] = sr * st * dt * factor_w

        # Col 3 (Yaw Rate Cmd -> Wz)
        J[6, 3] = cr * tt * dt * factor_w
        J[7, 3] = -sr * dt * factor_w
        J[8, 3] = cr * st * dt * factor_w

        # 3. Derivatives w.r.t MASS
        grad_mass = np.zeros(12, dtype=np.float32)
        F_m = F / self.mass
        ax_th = F_m * D_x
        ay_th = F_m * D_y
        az_th = F_m * D_z

        da_dm_x = -ax_th / self.mass
        da_dm_y = -ay_th / self.mass
        da_dm_z = -az_th / self.mass

        grad_mass[3] = da_dm_x * dt
        grad_mass[4] = da_dm_y * dt
        grad_mass[5] = da_dm_z * dt

        grad_mass[0] = grad_mass[3] * dt
        grad_mass[1] = grad_mass[4] * dt
        grad_mass[2] = grad_mass[5] * dt

        return J, grad_mass

    def get_param_sensitivities(self, state_dict, action_dict):
        # Unpack
        roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']
        thrust = action_dict['thrust']

        # Magnitude sensitivities
        sens_mass = (thrust * self.MAX_THRUST_BASE * self.thrust_coeff) / (self.mass * self.mass)
        sens_thrust_coeff = (thrust * self.MAX_THRUST_BASE) / self.mass

        vx = state_dict['vx'] - self.wind_x
        vy = state_dict['vy'] - self.wind_y
        vz = state_dict['vz']
        v_mag = math.sqrt(vx*vx + vy*vy + vz*vz)
        sens_drag = v_mag
        sens_wind = self.drag_coeff

        wx = state_dict.get('wx', 0.0)
        wy = state_dict.get('wy', 0.0)
        wz = state_dict.get('wz', 0.0)
        err_wx = action_dict['roll_rate'] - wx
        err_wy = action_dict['pitch_rate'] - wy
        err_wz = action_dict['yaw_rate'] - wz
        mag_err = math.sqrt(err_wx**2 + err_wy**2 + err_wz**2)
        sens_tau = mag_err / (self.tau * self.tau)

        return {
            'mass': sens_mass,
            'drag_coeff': sens_drag,
            'thrust_coeff': sens_thrust_coeff,
            'wind_x': sens_wind,
            'wind_y': sens_wind,
            'tau': sens_tau
        }

    def get_state_jacobian(self, state_dict, action_dict, dt):
        J_state = np.zeros((12, 12), dtype=np.float32)

        # 1. dP'/dP = I
        J_state[0, 0] = 1.0; J_state[1, 1] = 1.0; J_state[2, 2] = 1.0

        # 2. dV'/dV = I * (1 - Cd * dt)
        dv_dv = 1.0 - self.drag_coeff * dt
        J_state[3, 3] = dv_dv; J_state[4, 4] = dv_dv; J_state[5, 5] = dv_dv

        # 3. dAtt'/dAtt = I
        J_state[6, 6] = 1.0; J_state[7, 7] = 1.0; J_state[8, 8] = 1.0

        # 4. dW'/dW = I * (1 / (1 + dt/tau)) = I * (tau / (tau + dt))
        dw_dw = self.tau / (self.tau + dt)
        J_state[9, 9] = dw_dw; J_state[10, 10] = dw_dw; J_state[11, 11] = dw_dw

        # 5. dP'/dV = dt
        J_state[0, 3] = dv_dv * dt
        J_state[1, 4] = dv_dv * dt
        J_state[2, 5] = dv_dv * dt

        # 6. dV'/dAtt
        roll = state_dict['roll']
        pitch = state_dict['pitch']
        yaw = state_dict['yaw']
        thrust = action_dict['thrust']

        r, p, y = roll, pitch, yaw
        cr = math.cos(r); sr = math.sin(r)
        cp = math.cos(p); sp = math.sin(p)
        cy = math.cos(y); sy = math.sin(y)

        max_thrust = self.MAX_THRUST_BASE * self.thrust_coeff
        F = thrust * max_thrust
        F_m = F / self.mass

        dDx_dr = cy*sp*(-sr) + sy*cr
        dDx_dp = cy*cp*cr
        dDx_dy = -sy*sp*cr + cy*sr

        dDy_dr = sy*sp*(-sr) - cy*cr
        dDy_dp = sy*cp*cr
        dDy_dy = cy*sp*cr + sy*sr

        dDz_dr = cp*(-sr)
        dDz_dp = -sp*cr
        dDz_dy = 0.0

        dv_dr_x = F_m * dDx_dr * dt
        dv_dr_y = F_m * dDy_dr * dt
        dv_dr_z = F_m * dDz_dr * dt

        J_state[3, 6] = dv_dr_x
        J_state[4, 6] = dv_dr_y
        J_state[5, 6] = dv_dr_z

        dv_dp_x = F_m * dDx_dp * dt
        dv_dp_y = F_m * dDy_dp * dt
        dv_dp_z = F_m * dDz_dp * dt

        J_state[3, 7] = dv_dp_x
        J_state[4, 7] = dv_dp_y
        J_state[5, 7] = dv_dp_z

        dv_dy_x = F_m * dDx_dy * dt
        dv_dy_y = F_m * dDy_dy * dt
        dv_dy_z = F_m * dDz_dy * dt

        J_state[3, 8] = dv_dy_x
        J_state[4, 8] = dv_dy_y
        J_state[5, 8] = dv_dy_z

        # 7. dP'/dAtt = dV'/dAtt * dt
        J_state[0, 6] = dv_dr_x * dt
        J_state[1, 6] = dv_dr_y * dt
        J_state[2, 6] = dv_dr_z * dt

        J_state[0, 7] = dv_dp_x * dt
        J_state[1, 7] = dv_dp_y * dt
        J_state[2, 7] = dv_dp_z * dt

        J_state[0, 8] = dv_dy_x * dt
        J_state[1, 8] = dv_dy_y * dt
        J_state[2, 8] = dv_dy_z * dt

        # 8. dAtt'/dW (Kinematic Matrix * dt)
        if abs(cp) < 1e-6: cp = 1e-6
        tt = sp / cp
        st = 1.0 / cp

        J_state[6, 9] = 1.0 * dt
        J_state[6, 10] = sr * tt * dt
        J_state[6, 11] = cr * tt * dt

        J_state[7, 9] = 0.0
        J_state[7, 10] = cr * dt
        J_state[7, 11] = -sr * dt

        J_state[8, 9] = 0.0
        J_state[8, 10] = sr * st * dt
        J_state[8, 11] = cr * st * dt

        return J_state


class PyGhostEstimator:
    def __init__(self, models_list):
        self.models = []
        for md in models_list:
             self.models.append(PyGhostModel(
                 md['mass'], md['drag_coeff'], md['thrust_coeff'],
                 md.get('wind_x', 0.0), md.get('wind_y', 0.0),
                 md.get('tau', 0.1)
             ))
        self.probabilities = np.ones(len(self.models), dtype=np.float32) / len(self.models)
        self.lambda_param = 5.0
        self.stable_params = self._compute_weighted_params()
        self.history = {'raw_estimates': [], 'observability_scores': []}
        self.observability_scores = {'mass': 0.0, 'drag_coeff': 0.0, 'thrust_coeff': 0.0, 'wind_x': 0.0, 'wind_y': 0.0, 'tau': 0.0}

    def _compute_weighted_params(self):
        avg = {'mass': 0.0, 'drag_coeff': 0.0, 'thrust_coeff': 0.0, 'wind_x': 0.0, 'wind_y': 0.0, 'tau': 0.0}
        for i, m in enumerate(self.models):
            p = self.probabilities[i]
            avg['mass'] += m.mass * p
            avg['drag_coeff'] += m.drag_coeff * p
            avg['thrust_coeff'] += m.thrust_coeff * p
            avg['wind_x'] += m.wind_x * p
            avg['wind_y'] += m.wind_y * p
            avg['tau'] += m.tau * p
        return avg

    def update(self, state_dict, action_dict, measured_accel_list, dt, measured_alpha_list=None, measured_screen_pos_list=None):
        likelihoods = np.zeros(len(self.models), dtype=np.float32)
        meas_ax, meas_ay, meas_az = measured_accel_list

        for i, m in enumerate(self.models):
            thrust = action_dict['thrust']
            roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']

            cr = math.cos(roll); sr = math.sin(roll)
            cp = math.cos(pitch); sp = math.sin(pitch)
            cy = math.cos(yaw); sy = math.sin(yaw)

            max_thrust = m.MAX_THRUST_BASE * m.thrust_coeff
            thrust_force = max(0.0, thrust * max_thrust)

            ax_dir = cy * sp * cr + sy * sr
            ay_dir = sy * sp * cr - cy * sr
            az_dir = cp * cr

            ax_thrust = thrust_force * ax_dir / m.mass
            ay_thrust = thrust_force * ay_dir / m.mass
            az_thrust = thrust_force * az_dir / m.mass

            ax_drag = -m.drag_coeff * (state_dict['vx'] - m.wind_x)
            ay_drag = -m.drag_coeff * (state_dict['vy'] - m.wind_y)
            az_drag = -m.drag_coeff * state_dict['vz']

            pred_ax = ax_thrust + ax_drag
            pred_ay = ay_thrust + ay_drag
            pred_az = az_thrust + az_drag - m.G

            dx = meas_ax - pred_ax
            dy = meas_ay - pred_ay
            dz = meas_az - pred_az

            error_sq = dx*dx + dy*dy + dz*dz
            likelihoods[i] = math.exp(-self.lambda_param * error_sq)

        self.probabilities *= likelihoods
        self.probabilities = np.maximum(self.probabilities, 1e-6)
        self.probabilities /= np.sum(self.probabilities)

        raw_est = self._compute_weighted_params()
        temp_model = PyGhostModel(
            self.stable_params['mass'], self.stable_params['drag_coeff'], self.stable_params['thrust_coeff'],
            self.stable_params['wind_x'], self.stable_params['wind_y']
        )
        sens = temp_model.get_param_sensitivities(state_dict, action_dict)

        gains = {'mass': 0.05, 'drag_coeff': 0.1, 'thrust_coeff': 0.05, 'wind_x': 2.0, 'wind_y': 2.0, 'tau': 0.002}
        if 'tau' not in self.stable_params: self.stable_params['tau'] = 0.1

        for k in self.stable_params.keys():
            s_val = sens.get(k, 0.0)
            score = s_val * gains.get(k, 1.0)
            score = max(0.0, min(1.0, score))
            self.observability_scores[k] = score
            self.stable_params[k] += score * (raw_est.get(k, self.stable_params[k]) - self.stable_params[k])

        # Gradient Steps (simplified for brevity, matching previous implementation)
        s_m = self.stable_params['mass']
        s_d = self.stable_params['drag_coeff']
        s_t = self.stable_params['thrust_coeff']
        s_wx = self.stable_params['wind_x']
        s_wy = self.stable_params['wind_y']
        s_tau = self.stable_params.get('tau', 0.1)

        thrust = action_dict['thrust']
        max_thrust = 20.0 * s_t
        thrust_force = max(0.0, thrust * max_thrust)

        roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']
        cr = math.cos(roll); sr = math.sin(roll)
        cp = math.cos(pitch); sp = math.sin(pitch)
        cy = math.cos(yaw); sy = math.sin(yaw)

        ax_dir = cy * sp * cr + sy * sr
        ay_dir = sy * sp * cr - cy * sr
        az_dir = cp * cr

        ax_th = thrust_force * ax_dir / s_m
        ay_th = thrust_force * ay_dir / s_m
        az_th = thrust_force * az_dir / s_m

        vx_rel = state_dict['vx'] - s_wx
        vy_rel = state_dict['vy'] - s_wy
        vz = state_dict['vz']

        ax_drag = -s_d * vx_rel
        ay_drag = -s_d * vy_rel
        az_drag = -s_d * vz

        pred_ax = ax_th + ax_drag
        pred_ay = ay_th + ay_drag
        pred_az = az_th + az_drag - 9.81

        ex = meas_ax - pred_ax
        ey = meas_ay - pred_ay
        ez = meas_az - pred_az

        grad_wx = -2.0 * ex * s_d
        grad_wy = -2.0 * ey * s_d
        self.stable_params['wind_x'] -= 0.5 * grad_wx
        self.stable_params['wind_y'] -= 0.5 * grad_wy
        self.stable_params['wind_x'] = max(-20.0, min(20.0, self.stable_params['wind_x']))
        self.stable_params['wind_y'] = max(-20.0, min(20.0, self.stable_params['wind_y']))

        grad_drag = -2.0 * (ex * (-vx_rel) + ey * (-vy_rel) + ez * (-vz))
        self.stable_params['drag_coeff'] -= 0.001 * grad_drag
        self.stable_params['drag_coeff'] = max(0.01, min(2.0, self.stable_params['drag_coeff']))

        grad_mass = -2.0 * (ex * (-ax_th/s_m) + ey * (-ay_th/s_m) + ez * (-az_th/s_m))
        self.stable_params['mass'] -= 0.01 * grad_mass * self.observability_scores.get('mass', 0.0)
        self.stable_params['mass'] = max(0.1, min(8.0, self.stable_params['mass']))

        if measured_alpha_list is not None:
             meas_alphax, meas_alphay, meas_alphaz = measured_alpha_list
             wx = state_dict.get('wx', 0.0)
             wy = state_dict.get('wy', 0.0)
             wz = state_dict.get('wz', 0.0)
             cmd_r = action_dict['roll_rate']
             cmd_p = action_dict['pitch_rate']
             cmd_y = action_dict['yaw_rate']
             pred_alphax = (cmd_r - wx) / s_tau
             pred_alphay = (cmd_p - wy) / s_tau
             pred_alphaz = (cmd_y - wz) / s_tau
             e_alphax = meas_alphax - pred_alphax
             e_alphay = meas_alphay - pred_alphay
             e_alphaz = meas_alphaz - pred_alphaz
             grad_tau = -2.0 * (
                 e_alphax * (-pred_alphax / s_tau) +
                 e_alphay * (-pred_alphay / s_tau) +
                 e_alphaz * (-pred_alphaz / s_tau)
             )
             self.stable_params['tau'] -= 0.001 * grad_tau * self.observability_scores.get('tau', 0.0)
             self.stable_params['tau'] = max(0.01, min(1.0, self.stable_params['tau']))

        self.history['raw_estimates'].append(raw_est)
        self.history['observability_scores'].append(self.observability_scores.copy())

    def get_probabilities(self):
        return self.probabilities

    def get_weighted_model(self):
        return self.stable_params

    def get_observability_scores(self):
        return self.observability_scores

    def get_history(self):
        return self.history

class PyDPCSolver:
    def __init__(self, horizon=40):
        self.horizon = horizon
        self.iterations = 30
        self.learning_rate = 0.005
        self.last_estimated_vel = np.zeros(3, dtype=np.float32)

    def _estimate_relative_state(self, history, dt=0.05):
        """
        Estimates the current relative state (Drone rel to Target) from history.
        Returns:
            est: dict (px, py, pz, vx, vy, vz, ...)
            vision_active: bool (True if valid UV points were used)
        """
        est = {
            'px': 0.0, 'py': 0.0, 'pz': 2.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'wx': 0.0, 'wy': 0.0, 'wz': 0.0
        }
        vision_active = False

        if not history:
             return est, False

        last_obs = history[-1]
        est['pz'] = last_obs['pz']
        est['vz'] = last_obs.get('vz', 0.0)
        est['roll'] = last_obs['roll']
        est['pitch'] = last_obs['pitch']
        est['yaw'] = last_obs['yaw']
        est['wx'] = last_obs.get('wx', 0.0)
        est['wy'] = last_obs.get('wy', 0.0)
        est['wz'] = last_obs.get('wz', 0.0)

        # 1. Collect all valid relative positions from history
        valid_points = []
        for i, h in enumerate(history):
            if h.get('u') is not None and h.get('v') is not None:
                # Project UV to Relative Position
                u, v, alt = h['u'], h['v'], h['pz']
                s30=0.5; c30=0.866025
                ray_c = np.array([u, v, 1.0])
                xb = s30 * ray_c[1] + c30 * ray_c[2]
                yb = ray_c[0]
                zb = c30 * ray_c[1] - s30 * ray_c[2]
                ray_b = np.array([xb, yb, zb])
                R_wb = rpy_to_matrix(h['roll'], h['pitch'], h['yaw'])
                ray_w = np.matmul(R_wb, ray_b)
                if abs(ray_w[2]) > 1e-4:
                    k = -alt / ray_w[2]
                    if k > 0:
                        rel_pos = -ray_w * k
                        # Store with index to handle timing
                        valid_points.append({'idx': i, 'px': rel_pos[0], 'py': rel_pos[1]})

        # 2. Update Velocity Estimate using Regression on recent points
        # Use simple Linear Regression on last N valid points for robustness
        # Slope of P vs T is Velocity.

        if len(valid_points) >= 2:
            vision_active = True
            # Use up to last 10 points for smoothing (0.5s)
            recent = valid_points[-10:]

            # Times relative to end
            # idx is step index. time = idx * dt
            # Shift time so last point is roughly 0 for numerical stability
            end_idx = history.index(last_obs) # Should be len-1 usually
            times = np.array([(p['idx'] - end_idx) * dt for p in recent])
            pxs = np.array([p['px'] for p in recent])
            pys = np.array([p['py'] for p in recent])

            # Polyfit degree 1: P = V*t + P0
            if len(recent) >= 2:
                # Fit X
                vx_fit, px_fit = np.polyfit(times, pxs, 1)
                # Fit Y
                vy_fit, py_fit = np.polyfit(times, pys, 1)

                # Update persistent estimate with filter
                # Simple complementary filter with previous estimate?
                # For now, trust the regression result
                self.last_estimated_vel[0] = vx_fit
                self.last_estimated_vel[1] = vy_fit

                # If the last point is very recent, set Position to fit at t=0
                last_valid_idx = recent[-1]['idx']
                steps_ago = end_idx - last_valid_idx

                if steps_ago < 5: # Recent enough (0.25s)
                    est['px'] = float(px_fit) # Value at t=0 (current)
                    est['py'] = float(py_fit)
                else:
                    # Propagate from last valid measurement
                    # But the fit already extrapolates to t=0!
                    # So px_fit is the estimate at current time.
                    est['px'] = float(px_fit)
                    est['py'] = float(py_fit)
            else:
                 # Just diff
                 p_curr = recent[-1]
                 p_prev = recent[-2]
                 dt_step = (p_curr['idx'] - p_prev['idx']) * dt
                 if dt_step < 1e-4: dt_step = dt
                 vx = (p_curr['px'] - p_prev['px']) / dt_step
                 vy = (p_curr['py'] - p_prev['py']) / dt_step
                 self.last_estimated_vel[0] = vx
                 self.last_estimated_vel[1] = vy
                 est['px'] = p_curr['px'] + vx * (end_idx - p_curr['idx']) * dt
                 est['py'] = p_curr['py'] + vy * (end_idx - p_curr['idx']) * dt

        elif len(valid_points) == 1:
            vision_active = True
            # Only one point ever seen. Velocity 0?
            p = valid_points[0]
            est['px'] = float(p['px'])
            est['py'] = float(p['py'])
            # Keep existing velocity (initialized 0)
        else:
            # Blind.
            # Propagate using last known velocity
            # Assuming we are calling this sequentially, we use internal state?
            # But 'history' might be a new object or appended?
            # If blind, we don't know where we started relative to target.
            # This is "Lost" state. Mission manager handles this mostly.
            # Solver just does its best.
            pass

        est['vx'] = float(self.last_estimated_vel[0])
        est['vy'] = float(self.last_estimated_vel[1])

        # If very stale tracking, damp velocity to 0?
        if valid_points:
             steps_since_last = (history.index(last_obs) - valid_points[-1]['idx'])
             if steps_since_last > 20: # 1 second blind
                 # Damp velocity
                 self.last_estimated_vel *= 0.95
                 vision_active = False # Treat as lost if too old?

        return est, vision_active

    def solve(self, history, initial_action_dict, models_list, weights_list, dt, forced_yaw_rate=None, goal_z=2.0, intercept_mode=False):
        """
        Solves for the optimal control action using internal relative state estimation.
        Inputs:
            history: List of observation dicts (roll, pitch, yaw, alt, u, v, ...)
            goal_z: Desired relative altitude (default 2.0m)
            intercept_mode: If True, relax collision avoidance to allow impact/landing.
        Note: Absolute Position and Velocity are never known/passed. Target is implicitly the tracked object (0,0,0).
        """
        state_dict, vision_active = self._estimate_relative_state(history, dt)

        # Get latest measured UV from history if available
        measured_u, measured_v = 0.0, 0.0
        if history:
            last = history[-1]
            if last.get('u') is not None and last.get('v') is not None:
                measured_u = last['u']
                measured_v = last['v']

        models = []
        for md in models_list:
             models.append(PyGhostModel(
                 md['mass'], md['drag_coeff'], md['thrust_coeff'],
                 md.get('wind_x', 0.0), md.get('wind_y', 0.0),
                 md.get('tau', 0.1)
             ))

        current_action = {
            'thrust': initial_action_dict['thrust'],
            'roll_rate': initial_action_dict['roll_rate'],
            'pitch_rate': initial_action_dict['pitch_rate'],
            'yaw_rate': initial_action_dict['yaw_rate']
        }

        if forced_yaw_rate is not None:
             current_action['yaw_rate'] = forced_yaw_rate

        for _ in range(self.iterations):
            current_action['thrust'] = max(0.0, min(1.0, current_action['thrust']))
            if forced_yaw_rate is not None:
                 current_action['yaw_rate'] = forced_yaw_rate

            total_grad = np.zeros(4, dtype=np.float32)

            for m_idx, model in enumerate(models):
                weight = weights_list[m_idx]
                if weight < 1e-5: continue

                state = state_dict.copy()

                # Initialize previous UV with ACTUAL measurement, not calculated projection
                # This ensures we start the flow cost from reality
                u_prev = measured_u
                v_prev = measured_v
                zc_prev = 2.0 # Approximation for initial gate

                G_mat = np.zeros((12, 4), dtype=np.float32)

                for t in range(self.horizon):
                    J_act, _ = model.get_gradients(state, current_action, dt)
                    J_state = model.get_state_jacobian(state, current_action, dt)
                    G_next = np.matmul(J_state, G_mat) + J_act
                    next_state = model.step(state, current_action, dt)

                    # Altitude Error (Relative to Target Z=0 + goal_z)
                    # We assume Target is at Z=0 in our relative frame
                    dx = next_state['px']
                    dy = next_state['py']
                    dz = next_state['pz'] - goal_z
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz + 1e-6)

                    k_pos = 20.0
                    # Screen Space Tracking: Disable X/Y Position Cost
                    # Rely on Gaze Cost (u/v) to align with target
                    dL_dP = np.array([0.0, 0.0, k_pos*dz/dist], dtype=np.float32)

                    target_safe_z = goal_z
                    dz_safe = next_state['pz'] - target_safe_z

                    dL_dPz_alt = 0.0

                    vz = next_state['vz']
                    dL_dPz_ttc = 0.0
                    dL_dVz_ttc = 0.0

                    # Use TTC Barrier only if NOT in intercept mode
                    if not intercept_mode:
                        if dz_safe > 0 and vz < -0.1:
                            tau = dz_safe / -vz
                            gain = 200.0
                            denom = tau + 0.1
                            dL_dtau = -gain / (denom * denom)
                            dtau_dz = 1.0 / -vz
                            dtau_dvz = dz_safe / (vz * vz)
                            dL_dPz_ttc = dL_dtau * dtau_dz
                            dL_dVz_ttc = dL_dtau * dtau_dvz

                    # Gaze Vector: Camera (Drone) -> Target (Origin)
                    # We assume the tracked object is at (0,0,0) in the relative frame
                    dx_w = 0.0 - next_state['px']
                    dy_w = 0.0 - next_state['py']
                    dz_w = 0.0 - next_state['pz']

                    r, p, y = next_state['roll'], next_state['pitch'], next_state['yaw']
                    cr=math.cos(r); sr=math.sin(r)
                    cp=math.cos(p); sp=math.sin(p)
                    cy=math.cos(y); sy=math.sin(y)

                    r11 = cy*cp; r12 = sy*cp; r13 = -sp
                    r21 = cy*sp*sr - sy*cr; r22 = sy*sp*sr + cy*cr; r23 = cp*sr
                    r31 = cy*sp*cr + sy*sr; r32 = sy*sp*cr - cy*sr; r33 = cp*cr

                    xb = r11*dx_w + r12*dy_w + r13*dz_w
                    yb = r21*dx_w + r22*dy_w + r23*dz_w
                    zb = r31*dx_w + r32*dy_w + r33*dz_w

                    s30 = 0.5; c30 = 0.866025
                    xc = yb
                    yc = s30*xb + c30*zb
                    zc = c30*xb - s30*zb

                    if zc < 0.1: zc = 0.1
                    u_pred = xc / zc
                    v_pred = yc / zc

                    # Tune Gaze Weight higher for Intercept Mode to ensure lock
                    w_g = 5.0 if intercept_mode else 2.0
                    # Reduced Flow Weight to prioritize Centering over Stabilization
                    w_flow = 0.05
                    diff_u = u_pred - u_prev
                    diff_v = v_pred - v_prev

                    valid_curr = (zc > 1.0) and (abs(u_pred) < 3.0) and (abs(v_pred) < 3.0)
                    valid_prev = (zc_prev > 1.0) and (abs(u_prev) < 3.0) and (abs(v_prev) < 3.0)

                    dL_du = 0.0
                    dL_dv = 0.0

                    if valid_curr:
                        dL_du += 2.0 * w_g * u_pred
                        dL_dv += 2.0 * w_g * v_pred

                    if valid_curr and valid_prev:
                        error_sq = u_pred*u_pred + v_pred*v_pred
                        flow_gate = math.exp(-error_sq / 10.0)
                        dL_du += 2.0 * w_flow * flow_gate * diff_u
                        dL_dv += 2.0 * w_flow * flow_gate * diff_v

                    dL_dxb = 0.0
                    dL_dyb = 0.0
                    dL_dzb = 0.0

                    if valid_curr:
                        u_prev = u_pred
                        v_prev = v_pred
                        zc_prev = zc

                        inv_zc = 1.0 / zc
                        inv_zc2 = inv_zc * inv_zc
                        du_dxc = inv_zc
                        du_dzc = -xc * inv_zc2
                        dv_dyc = inv_zc
                        dv_dzc = -yc * inv_zc2

                        dxc_dyb = 1.0
                        dyc_dxb = s30; dyc_dzb = c30
                        dzc_dxb = c30; dzc_dzb = -s30

                        du_dxb = du_dzc * dzc_dxb
                        du_dyb = du_dxc * dxc_dyb
                        du_dzb = du_dzc * dzc_dzb

                        dv_dxb = dv_dyc * dyc_dxb + dv_dzc * dzc_dxb
                        dv_dyb = 0.0
                        dv_dzb = dv_dyc * dyc_dzb + dv_dzc * dzc_dzb

                        dL_dxb = dL_du * du_dxb + dL_dv * dv_dxb
                        dL_dyb = dL_du * du_dyb + dL_dv * dv_dyb
                        dL_dzb = dL_du * du_dzb + dL_dv * dv_dzb
                    else:
                        # Out of FOV: Maximize Alignment with Camera Axis
                        # Camera Axis in Body Frame: (c30, 0, s30) (Upward Tilt)
                        # Vector: (xb, yb, zb)
                        norm_sq = xb*xb + yb*yb + zb*zb + 1e-6
                        norm = math.sqrt(norm_sq)
                        inv_norm = 1.0 / norm

                        # Dot Product
                        dot = xb * c30 + zb * s30

                        # Cost = -dot / norm (Maximize Cosine Similarity)
                        # Gradient of (-dot/norm) w.r.t x:
                        # - ( c30*norm - dot*(x/norm) ) / norm^2
                        # = - ( c30/norm - dot*x/norm^3 )
                        # = dot*x*inv_norm**3 - c30*inv_norm

                        inv_norm3 = inv_norm * inv_norm * inv_norm

                        dL_dxb = dot * xb * inv_norm3 - c30 * inv_norm
                        dL_dyb = dot * yb * inv_norm3
                        dL_dzb = dot * zb * inv_norm3 - s30 * inv_norm

                        # Scale the guidance gradient
                        w_align = 5.0
                        dL_dxb *= w_align
                        dL_dyb *= w_align
                        dL_dzb *= w_align

                    dL_dP_g = np.zeros(3, dtype=np.float32)
                    dL_dP_g[0] = -(r11*dL_dxb + r21*dL_dyb + r31*dL_dzb)
                    dL_dP_g[1] = -(r12*dL_dxb + r22*dL_dyb + r32*dL_dzb)
                    dL_dP_g[2] = -(r13*dL_dxb + r23*dL_dyb + r33*dL_dzb)

                    dL_dYaw_g = dL_dxb * yb + dL_dyb * (-xb)
                    dL_dPitch_g = dL_dxb * (-zb) + dL_dzb * xb

                    # --- Dive Angle Cost ---
                    dL_dVx_gamma = 0.0
                    dL_dVy_gamma = 0.0
                    dL_dVz_gamma = 0.0
                    dL_dPz_gamma = 0.0

                    # Only apply Dive Cost if we are above the goal altitude
                    if next_state['pz'] > goal_z + 0.5:
                        # gamma_ref based on RELATIVE altitude to goal
                        # 0 deg at goal_z, 70 deg at goal_z + 100m
                        rel_alt = next_state['pz'] - goal_z
                        gamma_ratio = max(0.0, min(rel_alt / 100.0, 1.0))

                        # User requested 20 deg convergence. But we must level off (0 deg) to hover.
                        # We use 20 deg minimum for the "Approach" phase, but blend to 0 close to target?
                        # Let's stick to 0 at goal to ensure safety.
                        gamma_ref_deg = 70.0 * gamma_ratio
                        gamma_ref = gamma_ref_deg * math.pi / 180.0

                        vx = next_state['vx']
                        vy = next_state['vy']
                        vz = next_state['vz']
                        h_speed = math.sqrt(vx*vx + vy*vy + 1e-6)
                        gamma = math.atan2(-vz, h_speed)

                        w_gamma = 10.0
                        gamma_diff = gamma - gamma_ref

                        dL_dGamma = 2.0 * w_gamma * gamma_diff

                        speed_sq = vx*vx + vy*vy + vz*vz + 1e-6

                        # dGamma/dVz = -h_speed / speed_sq
                        dL_dVz_gamma = dL_dGamma * (-h_speed / speed_sq)

                        # dGamma/dVx = (vz / speed_sq) * (vx / h_speed)
                        # Fix Singularity at Hover: If h_speed is small, gradients vanish.
                        # Add Forward Progress Kick if diving and slow.
                        if h_speed < 0.5:
                             # Assume forward direction is towards target (screen space u)
                             # Or just Body X.
                             # Kick: Maximize H-Speed
                             kick = 10.0 * (0.5 - h_speed)
                             dL_d_hspeed_kick = -kick
                             if h_speed < 1e-3:
                                  # Force X
                                  dL_dVx_gamma += dL_d_hspeed_kick
                             else:
                                  dL_dVx_gamma += dL_d_hspeed_kick * (vx / h_speed)
                                  dL_dVy_gamma += dL_d_hspeed_kick * (vy / h_speed)

                        term_h = vz / (speed_sq * h_speed)
                        dL_dVx_gamma += dL_dGamma * term_h * vx
                        dL_dVy_gamma += dL_dGamma * term_h * vy

                        # dL/dPz (Gradient of Ref w.r.t Pz)
                        if 0.0 < next_state['pz'] < 100.0:
                             dRef_dPz = 0.5 * math.pi / 180.0 # 50/100 * deg2rad
                             dL_dPz_gamma = dL_dGamma * (-1.0) * dRef_dPz

                    dL_dS = np.zeros(12, dtype=np.float32)
                    dL_dS[0] += dL_dP[0] + dL_dP_g[0]
                    dL_dS[1] += dL_dP[1] + dL_dP_g[1]
                    dL_dS[2] += dL_dP[2] + dL_dPz_alt + dL_dPz_ttc + dL_dP_g[2] + dL_dPz_gamma

                    dL_dS[7] += dL_dPitch_g
                    dL_dS[8] += dL_dYaw_g

                    k_damp = 2.0
                    dL_dS[3] += k_damp * next_state['vx'] + dL_dVx_gamma
                    dL_dS[4] += k_damp * next_state['vy'] + dL_dVy_gamma
                    dL_dS[5] += k_damp * next_state['vz'] + dL_dVz_ttc + dL_dVz_gamma

                    dL_dS[9] += 0.1 * next_state.get('wx', 0.0)
                    dL_dS[10] += 0.1 * next_state.get('wy', 0.0)
                    dL_dS[11] += 0.1 * next_state.get('wz', 0.0)

                    # Descent Velocity Constraint (Disable in Intercept Mode)
                    if not intercept_mode:
                        safe_limit = 15.0
                        w_vel_limit = 5000.0
                        violation = (-next_state['vz']) - safe_limit
                        if violation > 0:
                             dL_dS[5] += -2.0 * w_vel_limit * violation

                    dL_dU_rate = np.zeros(4, dtype=np.float32)
                    dL_dU_rate[1] = 0.2 * current_action['roll_rate']
                    dL_dU_rate[2] = 0.2 * current_action['pitch_rate']
                    dL_dU_rate[3] = 0.2 * current_action['yaw_rate']

                    # Smoothness Cost: Penalize large changes from initial action
                    w_smooth = 1.0
                    dL_dU_rate[0] += w_smooth * (current_action['thrust'] - initial_action_dict['thrust'])
                    dL_dU_rate[1] += w_smooth * (current_action['roll_rate'] - initial_action_dict['roll_rate'])
                    dL_dU_rate[2] += w_smooth * (current_action['pitch_rate'] - initial_action_dict['pitch_rate'])
                    if forced_yaw_rate is None:
                        dL_dU_rate[3] += w_smooth * (current_action['yaw_rate'] - initial_action_dict['yaw_rate'])

                    term = np.matmul(dL_dS, G_next)
                    total_grad += weight * (term + dL_dU_rate)

                    state = next_state
                    G_mat = G_next

            total_grad = np.clip(total_grad, -10.0, 10.0)

            current_action['thrust'] -= self.learning_rate * total_grad[0]
            current_action['roll_rate'] -= self.learning_rate * total_grad[1]
            current_action['pitch_rate'] -= self.learning_rate * total_grad[2]
            if forced_yaw_rate is None:
                current_action['yaw_rate'] -= self.learning_rate * total_grad[3]
            else:
                current_action['yaw_rate'] = forced_yaw_rate
            current_action['thrust'] = max(0.0, min(1.0, current_action['thrust']))

        ghost_paths = []
        if len(models) > 0:
             m = models[0]
             path = []
             curr = state_dict.copy()
             for _ in range(self.horizon):
                 ns = m.step(curr, current_action, dt)
                 path.append(ns)
                 curr = ns
             ghost_paths.append(path)

        return current_action, ghost_paths
