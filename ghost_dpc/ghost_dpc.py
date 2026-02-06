import numpy as np
import math

class PyGhostModel:
    def __init__(self, mass, drag, thrust_coeff, wind_x=0.0, wind_y=0.0):
        self.mass = float(mass)
        self.drag_coeff = float(drag)
        self.thrust_coeff = float(thrust_coeff)
        self.wind_x = float(wind_x)
        self.wind_y = float(wind_y)
        self.G = 9.81
        self.MAX_THRUST_BASE = 20.0

    def step(self, state_dict, action_dict, dt):
        # Unpack State
        px, py, pz = state_dict['px'], state_dict['py'], state_dict['pz']
        vx, vy, vz = state_dict['vx'], state_dict['vy'], state_dict['vz']
        roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']

        # Unpack Action
        thrust = action_dict['thrust']
        roll_rate = action_dict['roll_rate']
        pitch_rate = action_dict['pitch_rate']
        yaw_rate = action_dict['yaw_rate']

        # 1. Update Attitude (Euler Integration)
        next_roll = roll + roll_rate * dt
        next_pitch = pitch + pitch_rate * dt
        next_yaw = yaw + yaw_rate * dt

        # 2. Compute Forces based on New Attitude
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
            'roll': next_roll, 'pitch': next_pitch, 'yaw': next_yaw
        }

    def get_gradients(self, state_dict, action_dict, dt):
        """
        Returns Jacobian (9x4) and grad_mass (9x1).
        J rows: px, py, pz, vx, vy, vz, r, p, y
        J cols: thrust, roll_rate, pitch_rate, yaw_rate
        """
        # Intermediate values
        roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']
        thrust = action_dict['thrust']
        roll_rate = action_dict['roll_rate']
        pitch_rate = action_dict['pitch_rate']
        yaw_rate = action_dict['yaw_rate']

        r = roll + roll_rate * dt
        p = pitch + pitch_rate * dt
        y = yaw + yaw_rate * dt

        cr = math.cos(r); sr = math.sin(r)
        cp = math.cos(p); sp = math.sin(p)
        cy = math.cos(y); sy = math.sin(y)

        max_thrust = self.MAX_THRUST_BASE * self.thrust_coeff
        F = thrust * max_thrust

        # Force Directions
        D_x = cy * sp * cr + sy * sr
        D_y = sy * sp * cr - cy * sr
        D_z = cp * cr

        J = np.zeros((9, 4), dtype=np.float32)

        # 1. Derivatives w.r.t THRUST (Column 0)
        # d(Att)/d(Thrust) = 0

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
        # d(Att_next)/d(Rate) = dt
        J[6, 1] = dt
        J[7, 2] = dt
        J[8, 3] = dt

        F_m = F / self.mass

        # Partial derivatives of Directions D_x, D_y, D_z w.r.t r, p, y
        dDx_dr = cy*sp*(-sr) + sy*cr
        dDx_dp = cy*cp*cr
        dDx_dy = -sy*sp*cr + cy*sr

        dDy_dr = sy*sp*(-sr) - cy*cr
        dDy_dp = sy*cp*cr
        dDy_dy = cy*sp*cr + sy*sr

        dDz_dr = cp*(-sr)
        dDz_dp = -sp*cr
        dDz_dy = 0.0

        # Column 1: Roll Rate
        da_dRr_x = F_m * dDx_dr
        da_dRr_y = F_m * dDy_dr
        da_dRr_z = F_m * dDz_dr

        J[3, 1] = da_dRr_x * dt * dt
        J[4, 1] = da_dRr_y * dt * dt
        J[5, 1] = da_dRr_z * dt * dt

        J[0, 1] = J[3, 1] * dt
        J[1, 1] = J[4, 1] * dt
        J[2, 1] = J[5, 1] * dt

        # Column 2: Pitch Rate
        da_dPr_x = F_m * dDx_dp
        da_dPr_y = F_m * dDy_dp
        da_dPr_z = F_m * dDz_dp

        J[3, 2] = da_dPr_x * dt * dt
        J[4, 2] = da_dPr_y * dt * dt
        J[5, 2] = da_dPr_z * dt * dt

        J[0, 2] = J[3, 2] * dt
        J[1, 2] = J[4, 2] * dt
        J[2, 2] = J[5, 2] * dt

        # Column 3: Yaw Rate
        da_dYr_x = F_m * dDx_dy
        da_dYr_y = F_m * dDy_dy
        da_dYr_z = F_m * dDz_dy

        J[3, 3] = da_dYr_x * dt * dt
        J[4, 3] = da_dYr_y * dt * dt
        J[5, 3] = da_dYr_z * dt * dt

        J[0, 3] = J[3, 3] * dt
        J[1, 3] = J[4, 3] * dt
        J[2, 3] = J[5, 3] * dt

        # 3. Derivatives w.r.t MASS
        grad_mass = np.zeros(9, dtype=np.float32)
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
        Returns: {'mass': val, 'drag': val, 'thrust_coeff': val, 'wind': val}
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

        return {
            'mass': sens_mass,
            'drag_coeff': sens_drag,
            'thrust_coeff': sens_thrust_coeff,
            'wind_x': sens_wind,
            'wind_y': sens_wind
        }

    def get_state_jacobian(self, state_dict, action_dict, dt):
        """
        Returns State Jacobian (9x9)
        Rows: P(0-2), V(3-5), Att(6-8)
        Cols: P(0-2), V(3-5), Att(6-8)
        """
        J_state = np.zeros((9, 9), dtype=np.float32)

        # dAtt'/dAtt = I
        J_state[6, 6] = 1.0
        J_state[7, 7] = 1.0
        J_state[8, 8] = 1.0

        # dV'/dV = I * (1 - Cd * dt)
        dv_dv = 1.0 - self.drag_coeff * dt
        J_state[3, 3] = dv_dv
        J_state[4, 4] = dv_dv
        J_state[5, 5] = dv_dv

        # dP'/dP = I
        J_state[0, 0] = 1.0
        J_state[1, 1] = 1.0
        J_state[2, 2] = 1.0

        # dP'/dV = dV'/dV * dt
        J_state[0, 3] = dv_dv * dt
        J_state[1, 4] = dv_dv * dt
        J_state[2, 5] = dv_dv * dt

        # dV'/dAtt
        roll = state_dict['roll']
        pitch = state_dict['pitch']
        yaw = state_dict['yaw']
        thrust = action_dict['thrust']
        roll_rate = action_dict['roll_rate']
        pitch_rate = action_dict['pitch_rate']
        yaw_rate = action_dict['yaw_rate']

        r = roll + roll_rate * dt
        p = pitch + pitch_rate * dt
        y = yaw + yaw_rate * dt

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

        # dP'/dAtt = dV'/dAtt * dt
        J_state[0, 6] = dv_dr_x * dt
        J_state[1, 6] = dv_dr_y * dt
        J_state[2, 6] = dv_dr_z * dt

        J_state[0, 7] = dv_dp_x * dt
        J_state[1, 7] = dv_dp_y * dt
        J_state[2, 7] = dv_dp_z * dt

        J_state[0, 8] = dv_dy_x * dt
        J_state[1, 8] = dv_dy_y * dt
        J_state[2, 8] = dv_dy_z * dt

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
            'mass': 0.0, 'drag_coeff': 0.0, 'thrust_coeff': 0.0, 'wind_x': 0.0, 'wind_y': 0.0
        }

    def _compute_weighted_params(self):
        avg_mass = 0.0
        avg_drag = 0.0
        avg_thrust = 0.0
        avg_wind_x = 0.0
        avg_wind_y = 0.0

        for i, m in enumerate(self.models):
            p = self.probabilities[i]
            avg_mass += m.mass * p
            avg_drag += m.drag_coeff * p
            avg_thrust += m.thrust_coeff * p
            avg_wind_x += m.wind_x * p
            avg_wind_y += m.wind_y * p

        return {
            'mass': avg_mass,
            'drag_coeff': avg_drag,
            'thrust_coeff': avg_thrust,
            'wind_x': avg_wind_x,
            'wind_y': avg_wind_y
        }

    def update(self, state_dict, action_dict, measured_accel_list, dt):
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

        gains = {
            'mass': 0.05,
            'drag_coeff': 0.1,
            'thrust_coeff': 0.05,
            'wind_x': 2.0, # wind sens is small (0.1), so high gain needed?
            'wind_y': 2.0
        }

        for k in self.stable_params.keys():
            s_val = sens.get(k, 0.0)
            score = s_val * gains.get(k, 1.0)
            score = max(0.0, min(1.0, score)) # Clamp [0, 1]

            self.observability_scores[k] = score

            # Gated Update: new = old + score * (raw - old)
            self.stable_params[k] += score * (raw_est[k] - self.stable_params[k])

        # 4. Adaptive Gradient Step (Direct Error Minimization)
        # Re-compute predicted acceleration with current stable_params
        s_m = self.stable_params['mass']
        s_d = self.stable_params['drag_coeff']
        s_t = self.stable_params['thrust_coeff']
        s_wx = self.stable_params['wind_x']
        s_wy = self.stable_params['wind_y']

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

        lr_wind = 0.1
        self.stable_params['wind_x'] -= lr_wind * grad_wx
        self.stable_params['wind_y'] -= lr_wind * grad_wy
        self.stable_params['wind_x'] = max(-20.0, min(20.0, self.stable_params['wind_x']))
        self.stable_params['wind_y'] = max(-20.0, min(20.0, self.stable_params['wind_y']))

        # 2. Drag Coeff (da/dCd = -(v-w))
        grad_drag = -2.0 * (ex * (-vx_rel) + ey * (-vy_rel) + ez * (-vz))

        lr_drag = 0.0001 # Sensitive
        self.stable_params['drag_coeff'] -= lr_drag * grad_drag
        self.stable_params['drag_coeff'] = max(0.01, min(2.0, self.stable_params['drag_coeff']))

        # 3. Mass (da/dm = -a_thrust / m)
        # da/dm = - (F/m) / m = -a_th / m
        grad_mass = -2.0 * (ex * (-ax_th/s_m) + ey * (-ay_th/s_m) + ez * (-az_th/s_m))

        # Tune learning rate and gate by observability
        lr_mass = 0.003
        obs_mass = self.observability_scores.get('mass', 0.0)

        self.stable_params['mass'] -= lr_mass * grad_mass * obs_mass
        self.stable_params['mass'] = max(0.1, min(8.0, self.stable_params['mass']))

        # 4. History
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

        # Gradient Descent
        for _ in range(self.iterations):
            # Clamp Thrust
            current_action['thrust'] = max(0.0, min(1.0, current_action['thrust']))

            total_grad = np.zeros(4, dtype=np.float32) # thrust, roll_rate, pitch_rate, yaw_rate

            for m_idx, model in enumerate(models):
                weight = weights_list[m_idx]
                if weight < 1e-5: continue

                state = state_dict.copy()

                # G is sensitivity dS/dU. 9x4.
                # In C++ it was flat 36 array. Here numpy 9x4.
                G_mat = np.zeros((9, 4), dtype=np.float32)

                for t in range(self.horizon):
                    # 1. Get Gradients (J_act: 9x4)
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

                    dL_dP = np.array([dx/dist, dy/dist, dz/dist], dtype=np.float32)

                    # B. Altitude & TTC Barrier
                    target_safe_z = target_pos[2] + 2.0
                    dz_safe = next_state['pz'] - target_safe_z

                    # Linear Cost (Clipped to prevent overriding safety)
                    # Clip dz_safe effect to +/- 10m equivalent (200.0)
                    clipped_dz = max(-10.0, min(10.0, dz_safe))
                    dL_dPz_alt = 20.0 * clipped_dz

                    # TTC Barrier (Scale Less)
                    # tau = dz / -vz. Cost = 1/tau.
                    vz = next_state['vz']
                    dL_dPz_ttc = 0.0
                    dL_dVz_ttc = 0.0

                    if dz_safe > 0 and vz < -0.1:
                        tau = dz_safe / -vz
                        # Barrier: J = gain / (tau + 0.1)
                        # Scale gain by dz to make it relevant at distance?
                        # User wants "scale less". 1/tau is frequency.
                        # J = 10.0 / (tau + 0.1)

                        gain = 2.0
                        denom = tau + 0.1
                        dL_dtau = -gain / (denom * denom)

                        dtau_dz = 1.0 / -vz
                        dtau_dvz = dz_safe / (vz * vz)

                        dL_dPz_ttc = dL_dtau * dtau_dz
                        dL_dVz_ttc = dL_dtau * dtau_dvz

                    # Combined dL/dS (9,)
                    dL_dS = np.zeros(9, dtype=np.float32)
                    dL_dS[0] += dL_dP[0]
                    dL_dS[1] += dL_dP[1]
                    dL_dS[2] += dL_dP[2] + dL_dPz_alt + dL_dPz_ttc

                    # Velocity Damping (dL/dV = 2.0 * V)
                    # Helps prevent overshoot in long dives
                    dL_dS[3] += 2.0 * next_state['vx']
                    dL_dS[4] += 2.0 * next_state['vy']
                    dL_dS[5] += 2.0 * next_state['vz'] + dL_dVz_ttc

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
