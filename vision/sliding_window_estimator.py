import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import logging

logger = logging.getLogger(__name__)

class SlidingWindowEstimator:
    def __init__(self, window_size=5):
        self.window_size = window_size

        # Window State
        # List of frames. Each frame: {
        #   't': timestamp,
        #   'p': [x,y,z], 'q': [x,y,z,w], 'v': [vx,vy,vz],
        #   'bg': [x,y,z], 'ba': [x,y,z],
        #   'imu_preint': { 'dp':.., 'dv':.., 'dq':.., 'dt':.., 'cov':.. } (constraint to prev frame)
        # }
        self.frames = []

        # Map Points
        # id -> {'p': [x,y,z], 'obs': [(frame_idx, u, v), ...]}
        # We optimize point positions directly for simplicity (or inverse depth anchored in first obs)
        self.points = {}

        # Gravity
        self.g = np.array([0, 0, 9.81])

        # Config
        self.w_vis = 1.0
        self.w_imu = 10.0
        self.w_baro = 20.0
        self.w_prior = 1.0

    def add_frame(self, t, p_prior, q_prior, v_prior, imu_data, image_obs):
        """
        Add a new frame to the window.
        imu_data: list of (dt, acc, gyro) since last frame.
        image_obs: dict {pt_id: (u, v)} normalized.
        """
        # Pre-integrate IMU
        imu_constraint = self._preintegrate_imu(imu_data) if self.frames else None

        # Init State (use priors or propagate from last)
        if self.frames:
            last = self.frames[-1]
            # Propagate for init guess
            # Simple Euler for guess
            dt = sum(x[0] for x in imu_data)
            p_guess = last['p'] + last['v'] * dt
            v_guess = last['v'] # Gravity?
            q_guess = last['q'] # Rotation?
            # Better: use p_prior if available from VIO prop, else use IMU prop
            frame = {
                't': t,
                'p': p_prior if p_prior is not None else p_guess,
                'q': q_prior if q_prior is not None else q_guess,
                'v': v_prior if v_prior is not None else v_guess,
                'bg': last['bg'].copy(),
                'ba': last['ba'].copy(),
                'imu_preint': imu_constraint,
                'baro': -p_prior[2] if p_prior is not None else 0.0 # Store measured altitude (Z-down Pz is -Alt)
            }
        else:
            # First frame
            frame = {
                't': t,
                'p': p_prior if p_prior is not None else np.zeros(3),
                'q': q_prior if q_prior is not None else np.array([0,0,0,1]),
                'v': v_prior if v_prior is not None else np.zeros(3),
                'bg': np.zeros(3),
                'ba': np.zeros(3),
                'imu_preint': None,
                'baro': 50.0 # Default start
            }

        self.frames.append(frame)

        # Add Observations
        curr_frame_idx = len(self.frames) - 1
        for pid, uv in image_obs.items():
            if pid not in self.points:
                # Triangulate later. For now init at random depth along ray?
                # Or wait until 2 obs?
                self.points[pid] = {'p': None, 'obs': []}
            self.points[pid]['obs'].append((curr_frame_idx, uv[0], uv[1]))

        # Marginalize/Slide if full
        if len(self.frames) > self.window_size:
            # Simple strategy: Fix first frame (Prior) or drop it?
            # ICE-BA marginalizes. Here we just drop and fix the new oldest frame (Keyframe-based).
            # To keep it simple: Just pop(0) and delete points observed only in 0.

            # Note: A proper marginalization prior is hard to implement quickly.
            # We will assume "Fixed Lag Smoother" where we just optimize the window and discard history.
            removed_frame = self.frames.pop(0)

            # Shift observation indices
            for pid in list(self.points.keys()):
                new_obs = []
                for (fidx, u, v) in self.points[pid]['obs']:
                    if fidx > 0:
                        new_obs.append((fidx - 1, u, v))
                self.points[pid]['obs'] = new_obs
                if not new_obs:
                    del self.points[pid]

    def _preintegrate_imu(self, imu_data):
        # Simplified Pre-integration
        # Returns {dp, dv, dq, dt} in Body frame of Frame i
        # Using simple integration for guess
        dp = np.zeros(3)
        dv = np.zeros(3)
        dq = R.identity()
        total_dt = 0

        for dt, acc, gyro in imu_data:
            # R_k is current rotation in integration
            # acc is in body. R_k * acc is in "i" frame (since R_i is identity for preint)

            acc_i = dq.apply(acc)

            dp += dv * dt + 0.5 * acc_i * dt**2
            dv += acc_i * dt

            # Update rotation
            # Exp map for gyro
            d_rot = R.from_rotvec(gyro * dt)
            dq = dq * d_rot

            total_dt += dt

        return {
            'dp': dp, 'dv': dv, 'dq': dq, 'dt': total_dt
        }

    def solve(self):
        """
        Optimize the window.
        """
        if len(self.frames) < 2:
            return

        # 1. Triangulate new points (linear)
        # For points with >= 2 obs and no 'p'
        for pid, pt in self.points.items():
            if pt['p'] is None and len(pt['obs']) >= 2:
                # Triangulate
                pt['p'] = self._triangulate_point(pt['obs'])

        # 2. Build Optimization Problem
        # Variables: Frames * 15 (p, q, v, bg, ba) + Points * 3
        # We store them in a flat vector 'x'
        # To avoid manifold issues with Quaternion, we optimize Lie algebra (rotation vector) delta?
        # Or just use parameterization support in least_squares?
        # For simplicity: Use local parameterization update in the residual function wrapper.
        # But scipy.least_squares works on vector.
        # Let's parameterize R as rotvec (3 params).
        # State per frame: 3(p) + 3(r) + 3(v) + 3(bg) + 3(ba) = 15 params.

        # Initial Guess vector
        x0 = []
        for f in self.frames:
            # Rotvec from Quat
            r_vec = R.from_quat(f['q']).as_rotvec()
            x0.extend(f['p'])
            x0.extend(r_vec)
            x0.extend(f['v'])
            x0.extend(f['bg'])
            x0.extend(f['ba'])

        # Point map (index in x to point id)
        point_ids = []
        for pid, pt in self.points.items():
            if pt['p'] is not None:
                x0.extend(pt['p'])
                point_ids.append(pid)

        x0 = np.array(x0)
        n_frames = len(self.frames)
        frame_block_size = 15

        def residual_fun(x):
            res = []

            # Unpack Frames
            curr_frames = []
            for i in range(n_frames):
                base = i * frame_block_size
                p = x[base:base+3]
                r_vec = x[base+3:base+6]
                v = x[base+6:base+9]
                bg = x[base+9:base+12]
                ba = x[base+12:base+15]

                curr_frames.append({
                    'p': p, 'r': r_vec, 'v': v, 'bg': bg, 'ba': ba,
                    'R': R.from_rotvec(r_vec)
                })

            # Unpack Points
            curr_points = {}
            pt_base = n_frames * frame_block_size
            for i, pid in enumerate(point_ids):
                p_pt = x[pt_base + i*3 : pt_base + i*3 + 3]
                curr_points[pid] = p_pt

            # --- Residuals ---

            # 1. IMU Residuals (Frame i -> i+1)
            for i in range(n_frames - 1):
                f1 = curr_frames[i]
                f2 = curr_frames[i+1]
                constraint = self.frames[i+1]['imu_preint'] # Constraint is from i to i+1

                if constraint is None: continue

                dt = constraint['dt']

                # Predict State j from i using IMU model
                # p_j = p_i + v_i*dt + 0.5*g*dt^2 + R_i * (dp_preint)
                # Note: biases affect preint. Simplified: ignore bias effect on preint Jacobian for now.

                # Gravity in World is [0,0,9.81] (Down).
                # P is NED.
                # a_world = R * a_body + g.
                # p_new = p + v*dt + 0.5*g*dt^2 + double_int(R*a_body)

                # Correct eq:
                # p_j = p_i + v_i * dt + 0.5 * self.g * dt**2 + f1['R'].apply(constraint['dp'])
                pred_p = f1['p'] + f1['v'] * dt + 0.5 * self.g * dt**2 + f1['R'].apply(constraint['dp'])
                res.extend((f2['p'] - pred_p) * self.w_imu)

                # v_j = v_i + g*dt + R_i * dv_preint
                pred_v = f1['v'] + self.g * dt + f1['R'].apply(constraint['dv'])
                res.extend((f2['v'] - pred_v) * self.w_imu)

                # R_j = R_i * R_preint
                # Residual in log space
                # R_err = (R_pred.T * R_j) -> 0
                pred_R = f1['R'] * constraint['dq'] # rotation composition
                # mismatch
                R_err = pred_R.inv() * f2['R']
                res.extend(R_err.as_rotvec() * self.w_imu)

                # Bias random walk
                res.extend((f2['bg'] - f1['bg']) * 100.0) # Tight constraint
                res.extend((f2['ba'] - f1['ba']) * 100.0)

            # 2. Visual Residuals
            for pid, pt_pos in curr_points.items():
                obs_list = self.points[pid]['obs']
                for (fidx, u_obs, v_obs) in obs_list:
                    if fidx >= n_frames: continue
                    frame = curr_frames[fidx]

                    # Project World Point to Camera
                    # P_c = R_c2b.T * R_b2w.T * (P_w - P_wb)
                    # We assume R_c2b is Identity for now (Sim is perfect, or handled in wrapper).
                    # Actually Wrapper passes normalized coords which are already in Cam frame?
                    # No, wrapper detects in image.
                    # Let's assume standard R_c2b exists.
                    # Pass it in?
                    # For now assume Body=Camera (or handled by user).
                    # Project: P_body = frame['R'].inv().apply(pt_pos - frame['p'])

                    P_body = frame['R'].inv().apply(pt_pos - frame['p'])

                    # Simple Projection: x/z, y/z
                    if P_body[2] > 0.1:
                        u_pred = P_body[0] / P_body[2]
                        v_pred = P_body[1] / P_body[2]

                        res.append((u_pred - u_obs) * self.w_vis)
                        res.append((v_pred - v_obs) * self.w_vis)
                    else:
                        res.append(5.0) # Penalty for being behind
                        res.append(5.0)

            # 3. Barometer/Prior Residuals
            for i in range(n_frames):
                # Baro constraint on Z
                # Pz should match measurement (with some noise)
                # Baro is -Alt. Pz is NED (Down). So Pz ~ Baro.
                # If measurement says Alt=50, Pz should be -50?
                # User says: "inject scale if delta pz > 0.5".
                # Here we just put a soft constraint on absolute Z matching baro.
                # Barometer in frame struct is "measured altitude" (positive).
                # Pz is NED (negative up).
                # So Pz should be -baro.

                # BUT, baro drifts.
                # We should constrain RELATIVE vertical motion?
                # Or just absolute if we trust baro scale?
                # Let's constrain absolute for scale observability.

                # We used frame['baro'] as a value.
                # Assuming frame['baro'] is -Pz.

                baro_val = self.frames[i].get('baro', 0.0)
                # If we stored actual baro reading (Pos Up):
                # Pz_est (Down) should be -baro_val

                err_z = curr_frames[i]['p'][2] - (-baro_val)
                res.append(err_z * self.w_baro)

            return np.array(res)

        # Optimize
        res = least_squares(residual_fun, x0, verbose=0, max_nfev=10, ftol=1e-3)

        # Update State
        x_opt = res.x
        for i in range(n_frames):
            base = i * frame_block_size
            f = self.frames[i]
            f['p'] = x_opt[base:base+3]
            r_vec = x_opt[base+3:base+6]
            f['q'] = R.from_rotvec(r_vec).as_quat()
            f['v'] = x_opt[base+6:base+9]
            f['bg'] = x_opt[base+9:base+12]
            f['ba'] = x_opt[base+12:base+15]

        pt_base = n_frames * frame_block_size
        for i, pid in enumerate(point_ids):
            self.points[pid]['p'] = x_opt[pt_base + i*3 : pt_base + i*3 + 3]

    def _triangulate_point(self, obs):
        # Linear triangulation from N views
        # A * P = 0
        A = []
        for (fidx, u, v) in obs:
            if fidx >= len(self.frames): continue
            frame = self.frames[fidx]
            R_wc = R.from_quat(frame['q']).as_matrix() # Body to World
            p_wc = frame['p']

            # P_c = R_wc.T * (P_w - p_wc)
            # u = x/z, v = y/z
            # u * z - x = 0

            # Row 3 * u - Row 1
            RT = R_wc.T
            t = -RT @ p_wc

            r1 = RT[0]; r2 = RT[1]; r3 = RT[2]

            A.append(u * r3 - r1)
            A.append(v * r3 - r2)

        if len(A) < 4: return None # Need 2 views

        A = np.array(A)
        # SVD
        try:
            u, s, vh = np.linalg.svd(A)
            P_hom = vh[-1]
            P_w = P_hom[:3] / P_hom[3]
            return P_w
        except:
            return None

    def get_latest_state(self):
        if not self.frames:
            return None
        f = self.frames[-1]
        return {
            'px': f['p'][0], 'py': f['p'][1], 'pz': f['p'][2],
            'vx': f['v'][0], 'vy': f['v'][1], 'vz': f['v'][2],
            'q': f['q']
        }
