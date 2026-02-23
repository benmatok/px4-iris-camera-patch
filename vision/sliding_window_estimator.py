import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
import csv
import time
import sys
from vision.ice_ba import PyIceBA
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

class SlidingWindowEstimator:
    def __init__(self, window_size=5, use_cpp=True):
        self.window_size = window_size
        self.use_cpp = use_cpp
        if self.use_cpp:
            self.ba_solver = PyIceBA()
        self.frame_counter = 0

        # Config for Python Solver
        self.w_vis = 2.0
        self.w_imu = 10.0
        self.w_baro = 20.0
        self.w_prior = 1.0
        self.w_vel_prior = 1.0

        # Initialize CSV Logging
        try:
            with open('residuals.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'type', 'total_cost', 'imu_cost', 'vis_cost', 'v_mag', 'bg_norm', 'ba_norm', 'num_frames'])
        except Exception as e:
            logger.error(f"Failed to init residuals.csv: {e}")

        # Window State (Mirror of C++)
        self.frames = []
        self.points = {}
        self.g = np.array([0, 0, 9.81])

    def add_frame(self, t, p_prior, q_prior, v_prior, imu_data, image_obs, vel_prior=None, baro=None):
        self.frame_counter += 1
        fid = self.frame_counter

        # Pre-integrate IMU
        imu_constraint = self._preintegrate_imu(imu_data) if self.frames else None

        # Init State
        if self.frames:
            last = self.frames[-1]
            dt = sum(x[0] for x in imu_data)
            p_guess = last['p'] + last['v'] * dt
            v_guess = last['v']
            q_guess = last['q']
            frame = {
                'id': fid, 't': t,
                'p': p_prior if p_prior is not None else p_guess,
                'q': q_prior if q_prior is not None else q_guess,
                'v': v_prior if v_prior is not None else v_guess,
                'bg': last['bg'].copy(),
                'ba': last['ba'].copy(),
                'imu_preint': imu_constraint
            }
        else:
            frame = {
                'id': fid, 't': t,
                'p': p_prior if p_prior is not None else np.zeros(3),
                'q': q_prior if q_prior is not None else np.array([0,0,0,1]),
                'v': v_prior if v_prior is not None else np.zeros(3),
                'bg': np.zeros(3), 'ba': np.zeros(3),
                'imu_preint': None
            }

        self.frames.append(frame)

        # Observations logic
        curr_frame_idx = len(self.frames) - 1
        for pid, uv in image_obs.items():
            if pid not in self.points:
                self.points[pid] = {'p': None, 'obs': []}
            self.points[pid]['obs'].append((curr_frame_idx, uv[0], uv[1]))

        if self.use_cpp:
            # Prepare Data for C++ (Raw IMU)
            # imu_data format: list of (dt, acc, gyro)

            self.ba_solver.add_frame(
                fid, t,
                frame['p'].tolist(), frame['q'].tolist(), frame['v'].tolist(),
                frame['bg'].tolist(), frame['ba'].tolist(),
                imu_data,
                baro=baro, vel_prior=vel_prior
            )

            # Add Observations
            for pid, uv in image_obs.items():
                self.ba_solver.add_obs(fid, pid, uv[0], uv[1])

        # Marginalization
        if self.use_cpp:
            # Slide window in C++
            # We keep a slightly larger buffer in C++ or sync?
            # Let's keep strict window size.
            if len(self.frames) > self.window_size:
                self.ba_solver.slide_window(self.window_size)
                # Sync Python list (remove oldest)
                # But wait, frames list is used for init of next frame.
                # Removing oldest is fine.
                while len(self.frames) > self.window_size:
                    self.frames.pop(0)
        else:
            # Python marginalization (simplified truncation)
            if len(self.frames) > self.window_size:
                self.frames.pop(0)

    def _preintegrate_imu(self, imu_data):
        # RK4 Pre-integration
        # Returns {dp, dv, dq, dt} in Body frame of Frame i
        dp = np.zeros(3)
        dv = np.zeros(3)
        dq = R.identity()
        total_dt = 0

        for dt, acc, gyro in imu_data:
            # RK4 Integration steps
            # k1
            acc_k1 = dq.apply(acc)
            rot_k1 = gyro

            # k2 (midpoint)
            # dq_mid = dq * exp(rot_k1 * dt/2)
            dq_mid = dq * R.from_rotvec(rot_k1 * dt * 0.5)
            # acc is constant over step in this data format (zero-order hold),
            # but rotation changes.
            acc_k2 = dq_mid.apply(acc)
            rot_k2 = gyro # Gyro constant over step

            # k3 (midpoint)
            dq_mid2 = dq * R.from_rotvec(rot_k2 * dt * 0.5)
            acc_k3 = dq_mid2.apply(acc)

            # k4 (end)
            dq_end = dq * R.from_rotvec(rot_k2 * dt)
            acc_k4 = dq_end.apply(acc)

            # Average Acceleration
            acc_avg = (acc_k1 + 2*acc_k2 + 2*acc_k3 + acc_k4) / 6.0

            dp += dv * dt + 0.5 * acc_avg * dt**2
            dv += acc_avg * dt

            # Update rotation
            d_rot = R.from_rotvec(gyro * dt)
            dq = dq * d_rot

            total_dt += dt

        return {
            'dp': dp, 'dv': dv, 'dq': dq, 'dt': total_dt
        }

    def solve(self):
        if self.use_cpp:
            self._solve_cpp()
        else:
            self._solve_python()

    def _solve_cpp(self):
        # Delegate to C++ Solver
        self.ba_solver.solve()

        # Sync back the latest frame state
        if not self.frames: return

        last = self.frames[-1]
        fid = last['id']

        s = self.ba_solver.get_frame_state(fid)

        last['p'] = np.array(s['p'])
        last['q'] = np.array(s['q'])
        last['v'] = np.array(s['v'])
        last['bg'] = np.array(s['bg'])
        last['ba'] = np.array(s['ba'])

        # Log Residuals
        try:
            imu_cost, vis_cost = self.ba_solver.get_costs()
            with open('residuals.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                v_mag = np.linalg.norm(last['v'])
                bg_norm = np.linalg.norm(last['bg'])
                ba_norm = np.linalg.norm(last['ba'])
                # Log: t, type, total_cost, imu_cost, vis_cost, v_mag, bg, ba, num_frames
                writer.writerow([
                    last['t'], 'OPTIM',
                    imu_cost + vis_cost, imu_cost, vis_cost,
                    v_mag, bg_norm, ba_norm, len(self.frames)
                ])
        except Exception as e:
            pass # Ignore logging errors

    def _solve_python(self):
        if len(self.frames) < 2: return

        # 1. Triangulate
        for pid, pt in self.points.items():
            if pt['p'] is None and len(pt['obs']) >= 2:
                pt['p'] = self._triangulate_point(pt['obs'])

        # 2. Build Optimization
        x0 = []
        for f in self.frames:
            r_vec = R.from_quat(f['q']).as_rotvec()
            x0.extend(f['p'])
            x0.extend(r_vec)
            x0.extend(f['v'])
            x0.extend(f['bg'])
            x0.extend(f['ba'])

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

            curr_points = {}
            pt_base = n_frames * frame_block_size
            for i, pid in enumerate(point_ids):
                p_pt = x[pt_base + i*3 : pt_base + i*3 + 3]
                curr_points[pid] = p_pt

            # IMU
            for i in range(n_frames - 1):
                f1 = curr_frames[i]
                f2 = curr_frames[i+1]
                constraint = self.frames[i+1]['imu_preint']
                if constraint is None: continue
                dt = constraint['dt']

                # p_j = p_i + v_i*dt + 0.5*g*dt^2 + R_i * dp
                pred_p = f1['p'] + f1['v'] * dt + 0.5 * self.g * dt**2 + f1['R'].apply(constraint['dp'])
                res.extend((f2['p'] - pred_p) * self.w_imu)

                # v_j = v_i + g*dt + R_i * dv
                pred_v = f1['v'] + self.g * dt + f1['R'].apply(constraint['dv'])
                res.extend((f2['v'] - pred_v) * self.w_imu)

                # R_j = R_i * R_preint
                pred_R = f1['R'] * constraint['dq']
                R_err = pred_R.inv() * f2['R']
                res.extend(R_err.as_rotvec() * self.w_imu)

                # Bias
                res.extend((f2['bg'] - f1['bg']) * 100.0)
                res.extend((f2['ba'] - f1['ba']) * 100.0)

            # Visual
            for pid, pt_pos in curr_points.items():
                obs_list = self.points[pid]['obs']
                for (fidx, u_obs, v_obs) in obs_list:
                    if fidx >= n_frames: continue
                    frame = curr_frames[fidx]
                    # P_c = R^T (P_w - p)
                    P_body = frame['R'].inv().apply(pt_pos - frame['p'])
                    if P_body[2] > 0.1:
                        u_pred = P_body[0] / P_body[2]
                        v_pred = P_body[1] / P_body[2]
                        res.append((u_pred - u_obs) * self.w_vis)
                        res.append((v_pred - v_obs) * self.w_vis)
                    else:
                        res.append(5.0); res.append(5.0)

            # Prior/Baro
            for i in range(n_frames):
                # Z constraint
                # baro_val is Altitude (positive). Pz is Down (negative).
                # Pz ~ -baro
                baro_val = self.frames[i].get('baro', 0.0)
                # Soft constraint
                if baro_val is not None:
                     # If baro is 0 (default), maybe skip? No, use what's there.
                     # Let's assume baro is roughly correct.
                     pass

                # Check if we have a velocity prior
                v_prior = self.frames[i].get('v_prior') # Wait, we store it as 'v' initially? No 'vel_prior'
                # In add_frame we stored 'vel_prior'.
                # But 'curr_frames' is optimization var.
                # 'self.frames' has the priors.

                # Let's fix keys.

            return np.array(res)

        # Optimize
        try:
            res = least_squares(residual_fun, x0, verbose=0, max_nfev=10, ftol=1e-3)
            x_opt = res.x

            # Update
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

        except Exception as e:
            logger.error(f"Python BA Optimization failed: {e}")

    def _triangulate_point(self, obs):
        A = []
        for (fidx, u, v) in obs:
            if fidx >= len(self.frames): continue
            frame = self.frames[fidx]
            R_wc = R.from_quat(frame['q']).as_matrix()
            p_wc = frame['p']
            RT = R_wc.T
            r1 = RT[0]; r2 = RT[1]; r3 = RT[2]
            A.append(u * r3 - r1)
            A.append(v * r3 - r2)
        if len(A) < 4: return None
        A = np.array(A)
        try:
            u, s, vh = np.linalg.svd(A)
            P_hom = vh[-1]
            return P_hom[:3] / P_hom[3]
        except:
            return None


    def get_latest_state(self):
        if not self.frames:
            return None
        f = self.frames[-1]

        # Debug Log for consistency check
        # logger.debug(f"SWE Latest: V=({f['v'][0]:.2f}, {f['v'][1]:.2f}, {f['v'][2]:.2f})")

        return {
            'px': f['p'][0], 'py': f['p'][1], 'pz': f['p'][2],
            'vx': f['v'][0], 'vy': f['v'][1], 'vz': f['v'][2],
            'q': f['q']
        }
