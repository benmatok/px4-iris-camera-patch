import numpy as np
import math
import logging
import csv
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

class MSCKF:
    """
    Velocity-Only EKF (Simplified MSCKF).
    State Vector (3): [vx, vy, vz] in NED Frame
    """
    def __init__(self, projector):
        self.projector = projector
        self.g = np.array([0, 0, 9.81]) # NED Gravity (Down)

        # State
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)

        # Clones
        self.max_clones = 5
        self.cam_clones = []
        self.next_clone_id = 0
        self.current_time = 0.0

        # Covariance (3x3 - Velocity Only)
        self.P = np.eye(3) * 0.1
        self.Qc = np.diag([0.01, 0.01, 0.01])

        self.R_c2b = self.projector.R_c2b
        self.p_c2b = np.zeros(3)

        self.initialized = False
        self.features_processed = False
        self.last_res_norm = 0.0

        self.log_file = open("vio_state.csv", "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["time", "vx", "vy", "vz", "foe_u", "foe_v", "P_det", "res_norm"])

    def is_reliable(self):
        return self.initialized and self.features_processed

    def initialize(self, q, p, v):
        self.q = q
        self.p = p
        self.v = v
        self.P = np.eye(3) * 0.1
        self.initialized = True
        logger.info("MSCKF (Velocity Only) Initialized")

    def propagate(self, gyro, accel, dt):
        if not self.initialized: return

        self.current_time += dt

        # Unbias
        w = gyro - self.bg
        a = accel - self.ba

        # Rotation (Deterministic)
        angle = np.linalg.norm(w) * dt
        axis = w / np.linalg.norm(w) if np.linalg.norm(w) > 1e-6 else np.array([1,0,0])
        dq = R.from_rotvec(axis * angle).as_quat()
        r_old = R.from_quat(self.q)
        self.q = (r_old * R.from_quat(dq)).as_quat()
        R_mat = r_old.as_matrix()

        # Velocity & Position
        acc_world = R_mat @ a + self.g
        self.p = self.p + self.v * dt + 0.5 * acc_world * dt**2
        self.v = self.v + acc_world * dt

        # Covariance (Velocity Random Walk)
        self.P = self.P + self.Qc * dt

        # Log
        foe = self.predict_foe()
        foe_u, foe_v = foe if foe else (0, 0)
        self.csv_writer.writerow([
            f"{self.current_time:.3f}",
            f"{self.v[0]:.3f}", f"{self.v[1]:.3f}", f"{self.v[2]:.3f}",
            f"{foe_u:.1f}", f"{foe_v:.1f}",
            f"{np.linalg.det(self.P):.6e}",
            f"{self.last_res_norm:.4f}"
        ])
        self.log_file.flush()

    def augment_state(self):
        if not self.initialized: return
        r_wb = R.from_quat(self.q).as_matrix()
        p_cam = self.p + r_wb @ self.p_c2b
        r_wc = r_wb @ self.R_c2b
        q_cam = R.from_matrix(r_wc).as_quat()

        self.cam_clones.append({
            'p': p_cam, 'q': q_cam, 'id': self.next_clone_id, 'ts': self.current_time
        })
        self.next_clone_id += 1
        if len(self.cam_clones) > self.max_clones:
            self.cam_clones.pop(0)

    def predict_foe(self):
        if not self.initialized: return None
        r_wb = R.from_quat(self.q).as_matrix()
        v_b = r_wb.T @ self.v
        v_c = self.R_c2b.T @ v_b

        if v_c[2] < 0.1: return None
        # Return normalized coordinates (x/z, y/z)
        u = v_c[0] / v_c[2]
        v = v_c[1] / v_c[2]
        return (u, v)

    def update_height(self, height_meas):
        # Reset Position Z to avoid drift affecting geometry
        # height_meas is NED pz (negative altitude)
        self.p[2] = height_meas

    def update_features(self, tracks):
        H_list = []
        r_list = []

        for track in tracks:
            obs = track['obs']
            if len(obs) < 2: continue

            poses = []
            measurements = []
            valid_obs = []

            # For Jacobian calculation
            clone_timestamps = []

            for o in obs:
                # Find clone
                clone = next((c for c in self.cam_clones if c['id'] == o['clone_idx']), None)
                if not clone: continue

                R_wc = R.from_quat(clone['q']).as_matrix()
                p_wc = clone['p']

                poses.append((R_wc, p_wc))
                measurements.append((o['u'], o['v']))
                valid_obs.append(o)
                clone_timestamps.append(clone['ts'])

            if len(poses) < 2: continue

            p_feat_w = self._triangulate(poses, measurements)
            if p_feat_w is None: continue

            # Residuals & Jacobians
            H_v_j = np.zeros((2 * len(valid_obs), 3)) # w.r.t Velocity (3)
            H_f_j = np.zeros((2 * len(valid_obs), 3)) # w.r.t Feature (3)
            r_j = np.zeros(2 * len(valid_obs))

            for i, o in enumerate(valid_obs):
                R_wc, p_wc = poses[i]
                u_meas, v_meas = measurements[i]
                ts = clone_timestamps[i]
                dt_i = self.current_time - ts

                P_c = R_wc.T @ (p_feat_w - p_wc)
                if P_c[2] < 0.1: continue

                u_pred = P_c[0] / P_c[2]
                v_pred = P_c[1] / P_c[2]

                r_j[2*i] = u_meas - u_pred
                r_j[2*i+1] = v_meas - v_pred

                # Jacobians
                # d(uv)/dP_c
                J_uv_Pc = (1.0/P_c[2]) * np.array([
                    [1, 0, -u_pred],
                    [0, 1, -v_pred]
                ])

                # H_f: d(uv)/dP_w = J_uv_Pc * R_wc.T
                H_f_j[2*i:2*i+2, :] = J_uv_Pc @ R_wc.T

                # H_p_clone: d(uv)/dp_clone = J_uv_Pc * (-R_wc.T)
                H_p_clone = J_uv_Pc @ (-R_wc.T)

                # Chain Rule for Velocity: d(uv)/dv = d(uv)/dp_clone * dp_clone/dv
                # dp_clone/dv = -dt_i (assuming velocity error was constant bias)
                H_v_block = H_p_clone * (-dt_i)

                H_v_j[2*i:2*i+2, :] = H_v_block

            # Null Space Projection to remove Feature dependence
            try:
                Q_mat, _ = np.linalg.qr(H_f_j, mode='complete')
                Q2 = Q_mat[:, 3:] # Left Null Space

                if Q2.shape[1] > 0:
                    r_proj = Q2.T @ r_j
                    H_proj = Q2.T @ H_v_j

                    H_list.append(H_proj)
                    r_list.append(r_proj)
            except:
                pass

        if not H_list: return

        self.features_processed = True
        H_stack = np.vstack(H_list)
        r_stack = np.hstack(r_list)

        noise_var = (1.0 / 480.0)**2
        R_noise = np.eye(len(r_stack)) * noise_var

        # EKF Update
        S = H_stack @ self.P @ H_stack.T + R_noise
        S += np.eye(S.shape[0]) * 1e-6 # Regularization

        try:
            # Solve K = P H.T S^-1
            # Or solve S X = H P => X = S^-1 H P => K = X.T
            # dx = K r

            # Using lstsq for stability
            # S is symmetric positive definite
            K = (np.linalg.lstsq(S, H_stack @ self.P, rcond=None)[0]).T
            dx = K @ r_stack

            self.last_res_norm = np.linalg.norm(r_stack) / np.sqrt(len(r_stack))

            # Apply Correction
            self.v += dx

            # Update P
            I = np.eye(3)
            self.P = (I - K @ H_stack) @ self.P
        except Exception as e:
            logger.error(f"VIO Update Error: {e}")

    def _triangulate(self, poses, measurements):
        A = []
        for (R_wc, p_wc), (u, v) in zip(poses, measurements):
            RT = R_wc.T
            t = -RT @ p_wc
            r1, r2, r3 = RT[0, :], RT[1, :], RT[2, :]
            A.append(u * r3 - r1)
            A.append(v * r3 - r2)

        A_mat = np.array(A)

        B = []
        for (R_wc, p_wc), (u, v) in zip(poses, measurements):
            RT = R_wc.T
            t = -RT @ p_wc
            B.append(t[0] - u * t[2])
            B.append(t[1] - v * t[2])
        B_vec = np.array(B)

        try:
            res, _, _, _ = np.linalg.lstsq(A_mat, B_vec, rcond=None)
            return res
        except:
            return None

    def get_velocity(self):
        return self.v
