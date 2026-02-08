import numpy as np
import math

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

        # 1. Update Angular Velocities (Lag Dynamics)
        # next_w = w + (cmd - w) * (dt/tau)
        next_wx = wx + (roll_rate_cmd - wx) * (dt / self.tau)
        next_wy = wy + (pitch_rate_cmd - wy) * (dt / self.tau)
        next_wz = wz + (yaw_rate_cmd - wz) * (dt / self.tau)

        # 2. Update Attitude (Kinematics)
        sp = math.sin(roll) # wait, roll is phi. Convention: r, p, y.
        # r->phi, p->theta, y->psi
        sr = math.sin(roll); cr = math.cos(roll)
        sp = math.sin(pitch); cp = math.cos(pitch)

        # Avoid singularity
        if abs(cp) < 1e-6: cp = 1e-6
        tt = sp / cp
        st = 1.0 / cp

        r_dot = next_wx + next_wy * sr * tt + next_wz * cr * tt
        p_dot = next_wy * cr - next_wz * sr
        y_dot = (next_wy * sr + next_wz * cr) * st

        next_roll = roll + r_dot * dt
        next_pitch = pitch + p_dot * dt
        next_yaw = yaw + y_dot * dt

        # 3. Compute Forces based on New Attitude
        max_thrust = self.MAX_THRUST_BASE * self.thrust_coeff
        thrust_force = thrust * max_thrust
        if thrust_force < 0:
            thrust_force = 0.0

        cr = math.cos(next_roll)
        sr = math.sin(next_roll)
        cp = math.cos(next_pitch)
        sp = math.sin(next_pitch)
        cy = math.cos(next_yaw)
        sy = math.sin(next_yaw)

        # Rotation Matrix Elements (World Acceleration components)
        # R31
        ax_dir = cy * sp * cr + sy * sr
        # R32
        ay_dir = sy * sp * cr - cy * sr
        # R33
        az_dir = cp * cr

        # Accelerations
        ax_thrust = thrust_force * ax_dir / self.mass
        ay_thrust = thrust_force * ay_dir / self.mass
        az_thrust = thrust_force * az_dir / self.mass

        ax_drag = -self.drag_coeff * (vx - self.wind_x)
        ay_drag = -self.drag_coeff * (vy - self.wind_y)
        az_drag = -self.drag_coeff * vz

        ax = ax_thrust + ax_drag
        ay = ay_thrust + ay_drag
        az = az_thrust + az_drag - self.G

        # 3. Update Velocity
        next_vx = vx + ax * dt
        next_vy = vy + ay * dt
        next_vz = vz + az * dt

        # 4. Update Position
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
        # Intermediate values
        roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']
        thrust = action_dict['thrust']
        # wx, wy, wz from state are not used for computing next attitude BASELINE in gradient
        # but needed for linearization point.
        # Actually, J is dState_next / dAction.

        # Current State
        r = roll
        p = pitch
        y = yaw

        # Next Attitude is affected by Action through Next Angular Velocity
        # next_w = w + (cmd - w)*dt/tau
        # next_att = att + att_dot(next_w) * dt

        # Force Direction depends on Next Attitude
        # For simplification in gradient, we can assume linearization at r,p,y
        # but technically we should use next_r, next_p

        # Let's approximate Gradient at current attitude for force direction
        # to avoid complex chain rule of att_next wrt cmd for force direction.
        # The dominant term for force direction is Attitude.

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
        # d(Att)/d(Thrust) = 0
        # d(W)/d(Thrust) = 0

        # d(a)/d(Thrust) = (F_max/m) * Direction
        da_dT_x = (max_thrust / self.mass) * D_x
        da_dT_y = (max_thrust / self.mass) * D_y
        da_dT_z = (max_thrust / self.mass) * D_z

        # d(v)/d(Thrust) = da/dT * dt
        J[3, 0] = da_dT_x * dt
        J[4, 0] = da_dT_y * dt
        J[5, 0] = da_dT_z * dt

        # d(p)/d(Thrust) = dv/dT * dt
        J[0, 0] = J[3, 0] * dt
        J[1, 0] = J[4, 0] * dt
        J[2, 0] = J[5, 0] * dt

        # 2. Derivatives w.r.t RATES (Columns 1, 2, 3)
        # d(W_next)/d(Rate_cmd) = dt / tau
        factor_w = dt / self.tau
        J[9, 1] = factor_w
        J[10, 2] = factor_w
        J[11, 3] = factor_w

        # d(Att_next)/d(Rate_cmd) = d(Att_next)/d(W_next) * d(W_next)/d(Rate_cmd)
        # d(Att_next)/d(W_next) = Kinematic_Matrix * dt
        # Kinematic Matrix K:
        # [ 1, sr*tt, cr*tt ]
        # [ 0, cr,    -sr   ]
        # [ 0, sr*st, cr*st ]

        # Avoid singularity
        if abs(cp) < 1e-6: cp = 1e-6
        tt = sp / cp
        st = 1.0 / cp

        # d(Roll)/d(Wx)=dt, d(Roll)/d(Wy)=sr*tt*dt, d(Roll)/d(Wz)=cr*tt*dt
        # Multiplied by factor_w (dt/tau)

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

        # Effect of Rate Cmd on Position/Velocity via Attitude?
        # d(Pos)/d(Rate) is roughly 0 for one step, or small (dAtt is small).
        # Ignoring d(Force)/d(Att) * d(Att)/d(Rate) for single step gradient
        # as it is higher order (dt^3 or dt^4 effects).

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
        """
        Returns scalar sensitivity of acceleration magnitude w.r.t parameters.
        Returns: {'mass': val, 'drag': val, 'thrust_coeff': val, 'wind': val, 'tau': val}
        """
        # Unpack
        roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']
        thrust = action_dict['thrust']

        cr = math.cos(roll); sr = math.sin(roll)
        cp = math.cos(pitch); sp = math.sin(pitch)
        cy = math.cos(yaw); sy = math.sin(yaw)

        # Force Directions
        ax_dir = cy * sp * cr + sy * sr
        ay_dir = sy * sp * cr - cy * sr
        az_dir = cp * cr

        max_thrust = self.MAX_THRUST_BASE * self.thrust_coeff
        F = thrust * max_thrust

        # 1. Mass Sensitivity: da/dm = -F/m^2
        # accel_thrust = F/m. d(F/m)/dm = -F/m^2.
        # Magnitude is F/m^2.
        sens_mass = (F / (self.mass * self.mass))

        # 2. Thrust Coeff Sensitivity: da/dk = (T_cmd * BaseMax) / m
        sens_thrust_coeff = (thrust * self.MAX_THRUST_BASE) / self.mass

        # 3. Drag Sensitivity: da/dCd = -(v - wind)
        # Magnitude is |v - wind|
        vx = state_dict['vx'] - self.wind_x
        vy = state_dict['vy'] - self.wind_y
        vz = state_dict['vz']
        v_mag = math.sqrt(vx*vx + vy*vy + vz*vz)
        sens_drag = v_mag

        # 4. Wind Sensitivity: da/dW = Cd
        # Accel_drag = -Cd * (v - W) = -Cd*v + Cd*W
        # da/dW = Cd.
        sens_wind = self.drag_coeff

        # 5. Tau Sensitivity: d(alpha)/d(tau) = -alpha / tau
        # alpha_x = (cmd_x - wx)/tau. d(alpha)/d(tau) = -(cmd-wx)/tau^2 = -alpha/tau
        # Magnitude: sqrt(sum(alpha^2)) / tau
        # Or simply magnitude of alpha predicted.
        # We need (cmd - w)
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
        """
        Returns State Jacobian (12x12)
        Rows: P(0-2), V(3-5), Att(6-8), W(9-11)
        Cols: P(0-2), V(3-5), Att(6-8), W(9-11)
        """
        J_state = np.zeros((12, 12), dtype=np.float32)

        # 1. dP'/dP = I
        J_state[0, 0] = 1.0; J_state[1, 1] = 1.0; J_state[2, 2] = 1.0

        # 2. dV'/dV = I * (1 - Cd * dt)
        dv_dv = 1.0 - self.drag_coeff * dt
        J_state[3, 3] = dv_dv; J_state[4, 4] = dv_dv; J_state[5, 5] = dv_dv

        # 3. dAtt'/dAtt = I
        J_state[6, 6] = 1.0; J_state[7, 7] = 1.0; J_state[8, 8] = 1.0

        # 4. dW'/dW = I * (1 - dt/tau)
        dw_dw = 1.0 - dt / self.tau
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

        # Derivatives of D_x, D_y, D_z w.r.t r, p, y
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
                 md.get('wind_x', 0.0), md.get('wind_y', 0.0)
             ))
        self.probabilities = np.ones(len(self.models), dtype=np.float32) / len(self.models)
        self.lambda_param = 5.0

        # Initialize Stable Parameters (Weighted Avg of Priors)
        self.stable_params = self._compute_weighted_params()

        # History for Analysis
        self.history = {
            'raw_estimates': [],
            'observability_scores': []
        }

        # Current Observability Scores
        self.observability_scores = {
            'mass': 0.0, 'drag_coeff': 0.0, 'thrust_coeff': 0.0, 'wind_x': 0.0, 'wind_y': 0.0, 'tau': 0.0
        }

    def _compute_weighted_params(self):
        avg_mass = 0.0
        avg_drag = 0.0
        avg_thrust = 0.0
        avg_wind_x = 0.0
        avg_wind_y = 0.0
        avg_tau = 0.0

        for i, m in enumerate(self.models):
            p = self.probabilities[i]
            avg_mass += m.mass * p
            avg_drag += m.drag_coeff * p
            avg_thrust += m.thrust_coeff * p
            avg_wind_x += m.wind_x * p
            avg_wind_y += m.wind_y * p
            avg_tau += m.tau * p

        return {
            'mass': avg_mass,
            'drag_coeff': avg_drag,
            'thrust_coeff': avg_thrust,
            'wind_x': avg_wind_x,
            'wind_y': avg_wind_y,
            'tau': avg_tau
        }

    def update(self, state_dict, action_dict, measured_accel_list, dt, measured_alpha_list=None, measured_screen_pos_list=None):
        likelihoods = np.zeros(len(self.models), dtype=np.float32)

        # measured_accel_list is [ax, ay, az]
        meas_ax, meas_ay, meas_az = measured_accel_list

        for i, m in enumerate(self.models):
            # Compute Predicted Acceleration
            # From GhostPhysics logic (step 2)
            thrust = action_dict['thrust']

            # Use State Attitude (Instantaneous)
            roll = state_dict['roll']
            pitch = state_dict['pitch']
            yaw = state_dict['yaw']

            cr = math.cos(roll)
            sr = math.sin(roll)
            cp = math.cos(pitch)
            sp = math.sin(pitch)
            cy = math.cos(yaw)
            sy = math.sin(yaw)

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

        # Update Probabilities
        self.probabilities *= likelihoods
        self.probabilities = np.maximum(self.probabilities, 1e-6)
        self.probabilities /= np.sum(self.probabilities)

        # --- Observability Gating ---

        # 1. Get Raw Estimate (MMAE Output)
        raw_est = self._compute_weighted_params()

        # 2. Compute Sensitivities using current stable parameters (as linearization point)
        # Create temp model
        temp_model = PyGhostModel(
            self.stable_params['mass'], self.stable_params['drag_coeff'], self.stable_params['thrust_coeff'],
            self.stable_params['wind_x'], self.stable_params['wind_y']
        )
        sens = temp_model.get_param_sensitivities(state_dict, action_dict)

        # 3. Compute Observability Scores & Update
        # Gains need to be tuned.
        # Mass: accel ~ 5-20 m/s^2. Sens ~ F/m^2 ~ 20. Gain 0.05 -> score 1.0.
        # Drag: v ~ 0-10 m/s. Sens ~ v. Gain 0.1 -> score 1.0 at 10m/s.
        # Tau: alpha ~ 10-50 rad/s^2. Sens ~ alpha/tau ~ 500. Gain small ~ 0.002.

        gains = {
            'mass': 0.05,
            'drag_coeff': 0.1,
            'thrust_coeff': 0.05,
            'wind_x': 2.0,
            'wind_y': 2.0,
            'tau': 0.002
        }

        # Ensure 'tau' is initialized in stable_params if missing (e.g. if loaded from old dict)
        if 'tau' not in self.stable_params:
             self.stable_params['tau'] = 0.1

        for k in self.stable_params.keys():
            s_val = sens.get(k, 0.0)
            score = s_val * gains.get(k, 1.0)
            score = max(0.0, min(1.0, score)) # Clamp [0, 1]

            self.observability_scores[k] = score

            # Gated Update: new = old + score * (raw - old)
            self.stable_params[k] += score * (raw_est.get(k, self.stable_params[k]) - self.stable_params[k])

        # 4. Adaptive Gradient Step (Direct Error Minimization)
        # Re-compute predicted acceleration with current stable_params
        s_m = self.stable_params['mass']
        s_d = self.stable_params['drag_coeff']
        s_t = self.stable_params['thrust_coeff']
        s_wx = self.stable_params['wind_x']
        s_wy = self.stable_params['wind_y']
        s_tau = self.stable_params.get('tau', 0.1)

        # Forces
        thrust = action_dict['thrust']
        max_thrust = 20.0 * s_t # MAX_THRUST_BASE hardcoded
        thrust_force = max(0.0, thrust * max_thrust)

        # Attitude
        roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']
        cr = math.cos(roll); sr = math.sin(roll)
        cp = math.cos(pitch); sp = math.sin(pitch)
        cy = math.cos(yaw); sy = math.sin(yaw)

        # Directions
        ax_dir = cy * sp * cr + sy * sr
        ay_dir = sy * sp * cr - cy * sr
        az_dir = cp * cr

        # Thrust Accel
        ax_th = thrust_force * ax_dir / s_m
        ay_th = thrust_force * ay_dir / s_m
        az_th = thrust_force * az_dir / s_m

        # Drag Accel
        vx_rel = state_dict['vx'] - s_wx
        vy_rel = state_dict['vy'] - s_wy
        vz = state_dict['vz']

        ax_drag = -s_d * vx_rel
        ay_drag = -s_d * vy_rel
        az_drag = -s_d * vz

        # Total Pred
        pred_ax = ax_th + ax_drag
        pred_ay = ay_th + ay_drag
        pred_az = az_th + az_drag - 9.81

        # Error
        ex = meas_ax - pred_ax
        ey = meas_ay - pred_ay
        ez = meas_az - pred_az

        # Gradients & Updates
        # Cost J = e^2. dJ/dParam = -2 * e * da/dParam

        # 1. Wind (da/dw = Cd)
        # da_x/dw_x = Cd. da_y/dw_y = Cd.
        grad_wx = -2.0 * ex * s_d
        grad_wy = -2.0 * ey * s_d

        lr_wind = 0.5
        self.stable_params['wind_x'] -= lr_wind * grad_wx
        self.stable_params['wind_y'] -= lr_wind * grad_wy
        self.stable_params['wind_x'] = max(-20.0, min(20.0, self.stable_params['wind_x']))
        self.stable_params['wind_y'] = max(-20.0, min(20.0, self.stable_params['wind_y']))

        # 2. Drag Coeff (da/dCd = -(v-w))
        grad_drag = -2.0 * (ex * (-vx_rel) + ey * (-vy_rel) + ez * (-vz))

        lr_drag = 0.001 # Sensitive
        self.stable_params['drag_coeff'] -= lr_drag * grad_drag
        self.stable_params['drag_coeff'] = max(0.01, min(2.0, self.stable_params['drag_coeff']))

        # 3. Mass (da/dm = -a_thrust / m)
        # da/dm = - (F/m) / m = -a_th / m
        grad_mass = -2.0 * (ex * (-ax_th/s_m) + ey * (-ay_th/s_m) + ez * (-az_th/s_m))

        # Tune learning rate and gate by observability
        lr_mass = 0.01
        obs_mass = self.observability_scores.get('mass', 0.0)

        self.stable_params['mass'] -= lr_mass * grad_mass * obs_mass
        self.stable_params['mass'] = max(0.1, min(8.0, self.stable_params['mass']))

        # 4. Tau Update (Angular Velocity Error)
        if measured_alpha_list is not None:
             # Unpack Measured Angular Accel
             meas_alphax, meas_alphay, meas_alphaz = measured_alpha_list

             # Predicted Alpha: (cmd - w) / tau
             # Using current w from state_dict (assuming it's W_t, we predict Alpha_t)
             # Wait, alpha is change rate. Alpha = (W_{t+1} - W_t)/dt.
             # Model: W_dot = (Cmd - W)/tau.

             # Get current W
             wx = state_dict.get('wx', 0.0)
             wy = state_dict.get('wy', 0.0)
             wz = state_dict.get('wz', 0.0)

             # Cmd
             cmd_r = action_dict['roll_rate']
             cmd_p = action_dict['pitch_rate']
             cmd_y = action_dict['yaw_rate']

             pred_alphax = (cmd_r - wx) / s_tau
             pred_alphay = (cmd_p - wy) / s_tau
             pred_alphaz = (cmd_y - wz) / s_tau

             # Error
             e_alphax = meas_alphax - pred_alphax
             e_alphay = meas_alphay - pred_alphay
             e_alphaz = meas_alphaz - pred_alphaz

             # Gradient dJ/dTau = -2 * e * dAlpha/dTau
             # dAlpha/dTau = -(Cmd-W)/Tau^2 = -Alpha/Tau

             grad_tau = -2.0 * (
                 e_alphax * (-pred_alphax / s_tau) +
                 e_alphay * (-pred_alphay / s_tau) +
                 e_alphaz * (-pred_alphaz / s_tau)
             )

             # Update
             lr_tau = 0.001
             obs_tau = self.observability_scores.get('tau', 0.0)

             self.stable_params['tau'] -= lr_tau * grad_tau * obs_tau
             self.stable_params['tau'] = max(0.01, min(1.0, self.stable_params['tau']))

        # 5. Screen Position Tracking (for validation/debugging)
        # Store predicted screen pos error if measurement provided
        if measured_screen_pos_list is not None:
             # Just store simple error of weighted model for now
             # This requires re-predicting state with updated stable_params?
             # Or just storing what we have.
             # Ideally we want to track: u_meas, v_meas, u_pred, v_pred
             # We can store this in a separate history key
             pass

        # 6. History
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
        self.learning_rate = 0.01

    def solve(self, state_dict, target_pos, initial_action_dict, models_list, weights_list, dt):
        # Convert models to PyGhostModel objects
        models = []
        for md in models_list:
             models.append(PyGhostModel(
                 md['mass'], md['drag_coeff'], md['thrust_coeff'],
                 md.get('wind_x', 0.0), md.get('wind_y', 0.0)
             ))

        # Initial Action
        current_action = {
            'thrust': initial_action_dict['thrust'],
            'roll_rate': initial_action_dict['roll_rate'],
            'pitch_rate': initial_action_dict['pitch_rate'],
            'yaw_rate': initial_action_dict['yaw_rate']
        }

        # Compute Initial Screen Pos (u_prev, v_prev)
        # Assuming we start from state_dict
        def compute_uv(s, t_pos):
             dx_w = s['px'] - t_pos[0]
             dy_w = s['py'] - t_pos[1]
             dz_w = s['pz'] - t_pos[2]

             r, p, y = s['roll'], s['pitch'], s['yaw']
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
             yc = -s30*xb + c30*zb
             zc = c30*xb + s30*zb

             if zc < 0.1: zc = 0.1
             return xc/zc, yc/zc, zc

        # Gradient Descent
        for _ in range(self.iterations):
            # Clamp Thrust
            current_action['thrust'] = max(0.0, min(1.0, current_action['thrust']))

            total_grad = np.zeros(4, dtype=np.float32) # thrust, roll_rate, pitch_rate, yaw_rate

            for m_idx, model in enumerate(models):
                weight = weights_list[m_idx]
                if weight < 1e-5: continue

                state = state_dict.copy()
                u_prev, v_prev, zc_prev = compute_uv(state, target_pos)

                # G is sensitivity dS/dU. 12x4.
                G_mat = np.zeros((12, 4), dtype=np.float32)

                for t in range(self.horizon):
                    # 1. Get Gradients (J_act: 12x4)
                    J_act, _ = model.get_gradients(state, current_action, dt)
                    J_state = model.get_state_jacobian(state, current_action, dt)

                    # 2. Propagate Sensitivity: G_next = J_state * G + J_act
                    G_next = np.matmul(J_state, G_mat) + J_act

                    # 3. Step Physics
                    next_state = model.step(state, current_action, dt)

                    # 4. Compute Cost Gradient
                    # A. Distance
                    dx = next_state['px'] - target_pos[0]
                    dy = next_state['py'] - target_pos[1]
                    dz = next_state['pz'] - target_pos[2]
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz + 1e-6)

                    # Scale Position Cost by 10.0 to overcome damping
                    k_pos = 100.0
                    dL_dP = np.array([k_pos*dx/dist, k_pos*dy/dist, k_pos*dz/dist], dtype=np.float32)

                    # B. Altitude & TTC Barrier
                    target_safe_z = target_pos[2] + 2.0
                    dz_safe = next_state['pz'] - target_safe_z

                    # Linear Cost (Clipped to prevent overriding safety)
                    # Clip dz_safe effect to +/- 50m equivalent (2500.0)
                    clipped_dz = max(-50.0, min(50.0, dz_safe))
                    # dL_dPz_alt = 50.0 * clipped_dz
                    dL_dPz_alt = 0.0

                    # TTC Barrier (Scale Less)
                    # tau = dz / -vz. Cost = 1/tau.
                    vz = next_state['vz']
                    dL_dPz_ttc = 0.0
                    dL_dVz_ttc = 0.0

                    # Tuned Barrier: Reduce gain to allow closer approach
                    if dz_safe > 0 and vz < -0.1:
                        tau = dz_safe / -vz
                        gain = 200.0 # Reduced from 1000.0
                        denom = tau + 0.1
                        dL_dtau = -gain / (denom * denom)

                        dtau_dz = 1.0 / -vz
                        dtau_dvz = dz_safe / (vz * vz)

                        dL_dPz_ttc = dL_dtau * dtau_dz
                        dL_dVz_ttc = dL_dtau * dtau_dvz

                    # New Term: Regulate Height Proportional to TTC
                    # Target: dz_safe = k_vel * tau
                    # Cost J = w * (dz_safe - k_vel * tau)^2
                    # Use k_vel = 2.0 (Target Approach Speed)
                    # Use w = 1.0
                    # k_vel = 2.0
                    # w_tau_track = 1.0

                    # Ensure vz is negative (closing) for tau to be valid
                    # If moving up or stationary, tau is undefined or negative (collision behind).
                    # We only apply this if closing.
                    # if dz_safe > 0 and vz < -0.1:
                    #     tau_val = dz_safe / -vz
                    #     err = dz_safe - k_vel * tau_val

                    #     # dJ/dZ = 2 * w * err * (1 - k_vel/(-v))
                    #     # dJ/dV = 2 * w * err * (-k_vel * (z/v^2))

                    #     dJ_dZ = 2.0 * w_tau_track * err * (1.0 - k_vel / -vz)

                    #     # dTau/dV = z / v^2.  -k * dTau/dV = -k * z / v^2
                    #     dJ_dV = 2.0 * w_tau_track * err * (-k_vel * (dz_safe / (vz*vz)))

                    #     dL_dPz_ttc += dJ_dZ
                    #     dL_dVz_ttc += dJ_dV

                    # --- Gaze Cost ---
                    # Compute u,v for current prediction step
                    # To minimize u,v error (centering target)
                    # u = yb / zc_safe
                    # v = (-0.5*xb + 0.866*zb) / zc_safe (if cam up)
                    # We need u,v calculation here.

                    dx_w = next_state['px'] - target_pos[0]
                    dy_w = next_state['py'] - target_pos[1]
                    dz_w = next_state['pz'] - target_pos[2]

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
                    yc = -s30*xb + c30*zb
                    zc = c30*xb + s30*zb

                    if zc < 0.1: zc = 0.1
                    u_pred = xc / zc
                    v_pred = yc / zc

                    # Cost = w_g * (u^2 + v^2) + w_flow * ((u - u_prev)^2 + (v - v_prev)^2)
                    w_g = 1.0
                    w_flow = 0.5

                    diff_u = u_pred - u_prev
                    diff_v = v_pred - v_prev

                    # Gate Visual Costs if target is behind camera plane
                    # We only add Flow Cost if we were visible AND are now visible
                    # Added tighter bounds to ignore "edge of FOV" singularities
                    valid_curr = (zc > 1.0) and (abs(u_pred) < 3.0) and (abs(v_pred) < 3.0)
                    valid_prev = (zc_prev > 1.0) and (abs(u_prev) < 3.0) and (abs(v_prev) < 3.0)

                    dL_du = 0.0
                    dL_dv = 0.0

                    if valid_curr:
                        dL_du += 2.0 * w_g * u_pred
                        dL_dv += 2.0 * w_g * v_pred

                    if valid_curr and valid_prev:
                        # Gated Flow Cost (only damp when tracking, not when acquiring)
                        error_sq = u_pred*u_pred + v_pred*v_pred
                        flow_gate = math.exp(-error_sq / 10.0)

                        dL_du += 2.0 * w_flow * flow_gate * diff_u
                        dL_dv += 2.0 * w_flow * flow_gate * diff_v

                    # Update Prev for next step
                    u_prev = u_pred
                    v_prev = v_pred
                    zc_prev = zc

                    # Gradients du/dS, dv/dS
                    # u = xc/zc => du = (dxc*zc - xc*dzc)/zc^2
                    # v = yc/zc => dv = (dyc*zc - yc*dzc)/zc^2
                    inv_zc = 1.0 / zc
                    inv_zc2 = inv_zc * inv_zc

                    du_dxc = inv_zc
                    du_dzc = -xc * inv_zc2
                    dv_dyc = inv_zc
                    dv_dzc = -yc * inv_zc2

                    # Transform back to xb, yb, zb
                    # xc = yb
                    # yc = -s30*xb + c30*zb
                    # zc = c30*xb + s30*zb

                    dxc_dyb = 1.0

                    dyc_dxb = -s30; dyc_dzb = c30
                    dzc_dxb = c30; dzc_dzb = s30

                    du_dxb = du_dzc * dzc_dxb # (term from zc)
                    du_dyb = du_dxc * dxc_dyb # (term from xc)
                    du_dzb = du_dzc * dzc_dzb # (term from zc)

                    dv_dxb = dv_dyc * dyc_dxb + dv_dzc * dzc_dxb
                    dv_dyb = 0.0
                    dv_dzb = dv_dyc * dyc_dzb + dv_dzc * dzc_dzb

                    dL_dxb = dL_du * du_dxb + dL_dv * dv_dxb
                    dL_dyb = dL_du * du_dyb + dL_dv * dv_dyb
                    dL_dzb = dL_du * du_dzb + dL_dv * dv_dzb

                    # Now dL/dP and dL/dAtt
                    # xb = R * dP_w
                    # dxb/dP = -R (since dP_w = T - P, dP_w/dP = -1)
                    # dL/dP = -R.T * dL/db

                    dL_dP_g = np.zeros(3, dtype=np.float32)
                    dL_dP_g[0] = -(r11*dL_dxb + r21*dL_dyb + r31*dL_dzb)
                    dL_dP_g[1] = -(r12*dL_dxb + r22*dL_dyb + r32*dL_dzb)
                    dL_dP_g[2] = -(r13*dL_dxb + r23*dL_dyb + r33*dL_dzb)

                    # dL/dAtt
                    # Simplified: Assume Gaze cost mainly affects Yaw (for x) and Pitch (for y)
                    # Implementing full dR/dAtt logic is verbose here but we can approximate or use existing
                    # J_state handles dP/dAtt propagation but dL_direct/dAtt is also needed
                    # Let's add direct terms for Yaw and Pitch to align
                    # Yaw affects xb (approx). Pitch affects zb/yb.

                    # dL/dYaw: Rotates (xb, yb)
                    # dxb/dy = yb, dyb/dy = -xb (roughly)
                    dL_dYaw_g = dL_dxb * yb + dL_dyb * (-xb)

                    # dL/dPitch: Rotates (xb, zb)
                    dL_dPitch_g = dL_dxb * (-zb) + dL_dzb * xb # Approx

                    # Combined dL/dS (12,)
                    dL_dS = np.zeros(12, dtype=np.float32)
                    dL_dS[0] += dL_dP[0] + dL_dP_g[0]
                    dL_dS[1] += dL_dP[1] + dL_dP_g[1]
                    dL_dS[2] += dL_dP[2] + dL_dPz_alt + dL_dPz_ttc + dL_dP_g[2]

                    dL_dS[7] += dL_dPitch_g
                    dL_dS[8] += dL_dYaw_g

                    # Velocity Damping (dL/dV = 0.1 * V)
                    # Increased to reduce overshoot
                    k_damp = 0.5 # Increased from 0.2
                    dL_dS[3] += k_damp * next_state['vx']
                    dL_dS[4] += k_damp * next_state['vy']
                    dL_dS[5] += k_damp * next_state['vz'] + dL_dVz_ttc

                    # Angular Rate Damping (Optional but good for smoothing)
                    # Penalize high angular rates in state
                    # dL_dW = 0.1 * W
                    dL_dS[9] += 0.1 * next_state.get('wx', 0.0)
                    dL_dS[10] += 0.1 * next_state.get('wy', 0.0)
                    dL_dS[11] += 0.1 * next_state.get('wz', 0.0)

                    # --- Descent Velocity Constraint ---
                    # Penalize if vz < -safe_descent_rate (ENU: Up is +Z, Falling is -Vz)
                    # Cost = w * ReLU(-vz - limit)^2
                    # dCost/dVz = 2 * w * ReLU(-vz - limit) * (-1)
                    safe_limit = 15.0 # m/s
                    w_vel_limit = 5000.0 # Dominant penalty

                    violation = (-next_state['vz']) - safe_limit
                    if violation > 0:
                         # dL/dVz = -2 * w * violation
                         dL_dS[5] += -2.0 * w_vel_limit * violation

                    # Rate Penalty
                    dL_dU_rate = np.zeros(4, dtype=np.float32)
                    dL_dU_rate[1] = 0.2 * current_action['roll_rate']
                    dL_dU_rate[2] = 0.2 * current_action['pitch_rate']
                    dL_dU_rate[3] = 0.2 * current_action['yaw_rate']

                    # Term dL/dU = dL/dS * G_next + dL/dU_direct
                    # (1, 9) * (9, 4) -> (1, 4)
                    term = np.matmul(dL_dS, G_next)

                    total_grad += weight * (term + dL_dU_rate)

                    # Update
                    state = next_state
                    G_mat = G_next

            # Gradient Clipping to prevent oscillation
            total_grad = np.clip(total_grad, -10.0, 10.0)

            # Update Action
            current_action['thrust'] -= self.learning_rate * total_grad[0]
            current_action['roll_rate'] -= self.learning_rate * total_grad[1]
            current_action['pitch_rate'] -= self.learning_rate * total_grad[2]
            current_action['yaw_rate'] -= self.learning_rate * total_grad[3]

            # Clamp Inside Loop to prevent explosion
            current_action['thrust'] = max(0.0, min(1.0, current_action['thrust']))

        return current_action
