
import numpy as np
import math
import logging
from scipy.spatial.transform import Rotation as R
from scipy.stats import chi2

logger = logging.getLogger(__name__)

class MSCKF:
    """
    Multi-State Constraint Kalman Filter (Simplified for Mono VIO + Height).

    State Vector (21 + 6*N):
    - [0:4]   Quaternion (q_wb) [x, y, z, w]
    - [4:7]   Position (p_w)
    - [7:10]  Velocity (v_w)
    - [10:13] Gyro Bias (bg)
    - [13:16] Accel Bias (ba)
    - [16:22] Camera Pose 1 (p, q_err) ... (N clones)

    Error State (15 + 6*N):
    - [0:3]   Theta (orientation error)
    - [3:6]   Position Error
    - [6:9]   Velocity Error
    - [9:12]  Gyro Bias Error
    - [12:15] Accel Bias Error
    - ... Clones
    """
    def __init__(self, projector):
        self.projector = projector

        # Constants
        # World Frame is NED (Z-Down). Gravity points Down.
        self.g = np.array([0, 0, 9.81])

        # State
        self.q = np.array([0.0, 0.0, 0.0, 1.0]) # Identity Quaternion
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)

        # Clones (Sliding Window)
        self.max_clones = 5
        self.cam_clones = [] # List of dicts: {'p': ..., 'q': ..., 'id': ...}
        self.next_clone_id = 0

        # Covariance
        self.P = np.eye(15) * 0.01
        self.P[9:12, 9:12] = 0.0001 # Low initial bias error
        self.P[12:15, 12:15] = 0.0001

        # Process Noise (Moderate for Stability)
        self.Qc = np.diag([
            1e-5, 1e-5, 1e-5, # Gyro Noise
            1e-4, 1e-4, 1e-4, # Accel Noise
            1e-7, 1e-7, 1e-7, # Gyro Bias Walk
            1e-7, 1e-7, 1e-7  # Accel Bias Walk
        ])

        # Extrinsics (Camera to Body)
        # R_c2b is provided by projector. p_c2b assumed 0.
        self.R_c2b = self.projector.R_c2b
        self.p_c2b = np.zeros(3)

        self.initialized = False
        self.features_processed = False

        # Scale Injection Gating
        self.last_scale_injection_pz = None

    def is_reliable(self):
        return self.initialized and self.features_processed

    def initialize(self, q, p, v):
        self.q = q
        self.p = p
        self.v = v
        self.initialized = True
        logger.info("MSCKF Initialized")

    def propagate(self, gyro, accel, dt):
        """
        Standard IMU Propagation with Sub-stepping.
        """
        if not self.initialized:
            return

        # Sub-stepping configuration
        # 0.05s is too coarse for IMU integration (20Hz)
        # Use 5 sub-steps -> 100Hz equivalent
        n_steps = 5
        sub_dt = dt / n_steps

        # Unbias (Constant over the step)
        w = gyro - self.bg
        a = accel - self.ba

        for _ in range(n_steps):
            # 1. Nominal State Propagation
            angle = np.linalg.norm(w) * sub_dt
            axis = w / np.linalg.norm(w) if np.linalg.norm(w) > 1e-6 else np.array([1,0,0])
            dq = R.from_rotvec(axis * angle).as_quat()

            r_old = R.from_quat(self.q)
            r_new = r_old * R.from_quat(dq)
            self.q = r_new.as_quat()

            R_mat = r_old.as_matrix() # R_wb at start of sub-step

            acc_world = R_mat @ a + self.g
            self.p = self.p + self.v * sub_dt + 0.5 * acc_world * sub_dt**2
            self.v = self.v + acc_world * sub_dt

            # Clamp Velocity
            v_norm_p = np.linalg.norm(self.v)
            if v_norm_p > 60.0: # Slightly higher than update clamp to allow transient
                 self.v = (self.v / v_norm_p) * 60.0

            # 2. Error State Covariance Propagation
            # F_x matrix (15x15)
            F = np.eye(15)

            # Theta block
            F[0:3, 0:3] = np.eye(3) - self.skew(w) * sub_dt
            F[0:3, 9:12] = -np.eye(3) * sub_dt

            # Pos block
            F[3:6, 6:9] = np.eye(3) * sub_dt

            # Vel block
            F[6:9, 0:3] = -R_mat @ self.skew(a) * sub_dt
            F[6:9, 12:15] = -R_mat * sub_dt

            # Noise Jacobian G (15 x 12)
            G = np.zeros((15, 12))
            G[0:3, 0:3] = -np.eye(3)
            G[6:9, 3:6] = -R_mat
            G[9:12, 6:9] = np.eye(3)
            G[12:15, 9:12] = np.eye(3)

            Q = G @ self.Qc @ G.T * sub_dt

            P_ii = self.P[0:15, 0:15]
            P_ic = self.P[0:15, 15:]

            self.P[0:15, 0:15] = F @ P_ii @ F.T + Q
            self.P[0:15, 15:] = F @ P_ic
            self.P[15:, 0:15] = self.P[0:15, 15:].T

        # P_cc (clones-clones) remains static

    def augment_state(self):
        """
        Adds current pose to state vector (Cloning).
        """
        if not self.initialized:
            return

        # Get Current Pose
        p_c = self.p.copy() # R_c2b is static rotation, usually small offset ignored for simple augment logic or transform properly
        # Technically P_cam_w = P_body_w + R_body_w * P_cam_b
        # Let's handle extrinsic properly
        r_wb = R.from_quat(self.q).as_matrix()
        p_cam = self.p + r_wb @ self.p_c2b

        # q_cam_w = q_body_w * q_cam_body
        r_cb = self.R_c2b # Cam to Body
        r_wc = r_wb @ r_cb
        q_cam = R.from_matrix(r_wc).as_quat()

        # Add to State
        self.cam_clones.append({'p': p_cam, 'q': q_cam, 'id': self.next_clone_id})
        self.next_clone_id += 1

        # Augment Covariance
        # J = [ dPc/dx ] (6 x (15+6N))
        # P_cam = P_body + R_wb * P_cb (const) => dPc/dp = I, dPc/dtheta = -R_wb [p_cb]x
        # q_cam = q_body * q_cb => dq_cam/dtheta_body = R_cb.T (if right perturbation) or similar.
        # Simplification: if p_cb = 0, P_cam = P_body.

        # Jacobian of New Clone w.r.t Current Error State
        J = np.zeros((6, 15 + 6 * len(self.cam_clones) - 6)) # -6 because we haven't added rows yet
        # Correct dimensions: P is current size.
        current_dim = self.P.shape[0]
        J = np.zeros((6, current_dim))

        # Orientation part
        # theta_cam ~ theta_body
        J[0:3, 0:3] = np.eye(3) # Simplification: R_c2b is constant

        # Position part
        # p_cam = p_body + R_wb * p_c2b
        # dp_cam = dp_body + d(R_wb) * p_c2b = dp_body - R_wb [p_c2b]x dtheta
        J[3:6, 3:6] = np.eye(3)
        # J[3:6, 0:3] = -r_wb @ self.skew(self.p_c2b) # If p_c2b is 0, this is 0

        # Augment P
        # P_new = [ P   P J^T ]
        #         [ J P J P J^T ]

        P11 = self.P
        P12 = P11 @ J.T
        P21 = P12.T
        P22 = J @ P11 @ J.T # + Noise? No, cloning is deterministic function of state

        self.P = np.block([[P11, P12], [P21, P22]])

        # Prune if too many
        if len(self.cam_clones) > self.max_clones:
            self._marginalize_clone(0)

    def _marginalize_clone(self, idx):
        # Remove clone at index idx (0 usually)
        # 15 static + 6 * idx
        start = 15 + 6 * idx
        end = start + 6

        # Remove rows/cols from P
        keep = np.delete(np.arange(self.P.shape[0]), np.arange(start, end))
        self.P = self.P[np.ix_(keep, keep)]

        self.cam_clones.pop(idx)

    def update_height(self, height_meas, noise_std=0.1):
        """
        Updates state with height measurement (pz).
        Constraints global Z.
        """
        if not self.initialized:
            return

        # Residual
        r = height_meas - self.p[2]

        # Safety Check: If residual is huge, reset height directly
        # This prevents massive linearized corrections from destabilizing orientation/velocity
        if abs(r) > 5.0:
            logger.warning(f"Large Height Residual: {r:.2f}m. Resetting Height State and Vertical Velocity.")
            self.p[2] = height_meas
            # Also reset Vertical Velocity to avoid immediate re-divergence
            self.v[2] = 0.0

            # Increase uncertainty
            self.P[5, 5] = 100.0 # Pos Z
            self.P[8, 8] = 10.0  # Vel Z
            return

        # H matrix (1 x N)
        # z = p_z
        # dz/dx = [0 0 0 | 0 0 1 | 0 ... ]
        H = np.zeros((1, self.P.shape[0]))
        H[0, 5] = 1.0 # Index 5 is Z component of Position Error

        # EKF Update
        S = H @ self.P @ H.T + noise_std**2

        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
            dx = K @ np.array([r])

            self._inject_error(dx)

            # Update P
            I = np.eye(self.P.shape[0])
            self.P = (I - K @ H) @ self.P
        except Exception as e:
            logger.error(f"Height Update Failed: {e}")

    def update_measurements(self, height_meas, vz_meas, tracks):
        """
        Updates state with Height, Vz, and Feature tracks.
        Only injects scale (Height/Vz) if significant vertical motion is detected.
        """
        if not self.initialized:
            return

        # Scale Injection Logic
        # "only inject scale if delta pz is above 0.5 meter"
        # We compare current height measurement with the last time we successfully injected scale.

        inject_scale = False
        if height_meas is not None:
            if self.last_scale_injection_pz is None:
                inject_scale = True
            elif abs(height_meas - self.last_scale_injection_pz) > 0.5:
                inject_scale = True

        # 1. Height Update
        if inject_scale and height_meas is not None:
            self._update_height_internal(height_meas)
            self.last_scale_injection_pz = height_meas # Update reference

        # 2. Vz Update (Gate with same logic? Usually good to update Vz if height is updating)
        if inject_scale and vz_meas is not None:
            self._update_vz_internal(vz_meas)

        # 3. Features Update (Always update features)
        if tracks:
            self.update_features(tracks)

    def _log_state(self, tag):
        v_mag = np.linalg.norm(self.v)
        logger.debug(f"[{tag}] V={self.v} (|V|={v_mag:.2f}) P={self.p}")

    def _update_height_internal(self, height_meas, noise_std=0.1):
        # Residual
        r = height_meas - self.p[2]

        # Safety Check
        if abs(r) > 5.0:
            logger.warning(f"Large Height Residual: {r:.2f}m. Resetting Height State and Vertical Velocity.")
            self.p[2] = height_meas
            self.v[2] = 0.0
            self.P[5, 5] = 100.0
            self.P[8, 8] = 10.0
            return

        H = np.zeros((1, self.P.shape[0]))
        H[0, 5] = 1.0

        S = H @ self.P @ H.T + noise_std**2
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
            dx = K @ np.array([r])

            # Log dx for velocity
            dv = dx[6:9]
            if np.linalg.norm(dv) > 1.0:
                 logger.warning(f"Height Update induced Large Velocity Jump: {dv}")

            self._inject_error(dx)
            I = np.eye(self.P.shape[0])
            self.P = (I - K @ H) @ self.P
            self._log_state("Post-Height")
        except Exception as e:
            logger.error(f"Height Update Failed: {e}")

    def _update_vz_internal(self, vz_meas, noise_std=0.1):
        # Residual
        r = vz_meas - self.v[2]

        # Safety Check
        if abs(r) > 10.0:
            logger.warning(f"Large Vz Residual: {r:.2f}m/s. Resetting Vertical Velocity.")
            self.v[2] = vz_meas
            self.P[8, 8] = 10.0
            return

        H = np.zeros((1, self.P.shape[0]))
        H[0, 8] = 1.0

        S = H @ self.P @ H.T + noise_std**2
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
            dx = K @ np.array([r])

            # Log dx for velocity
            dv = dx[6:9]
            if np.linalg.norm(dv) > 1.0:
                 logger.warning(f"Vz Update induced Large Velocity Jump: {dv}")

            self._inject_error(dx)
            I = np.eye(self.P.shape[0])
            self.P = (I - K @ H) @ self.P
            self._log_state("Post-Vz")
        except Exception as e:
            logger.error(f"Vz Update Failed: {e}")

    def update_height(self, height_meas, noise_std=0.1):
        # Legacy wrapper
        self._update_height_internal(height_meas, noise_std)

    def update_vertical_velocity(self, vz_meas, noise_std=0.1):
        # Legacy wrapper
        self._update_vz_internal(vz_meas, noise_std)

    def update_velocity_vector(self, v_ned_meas, noise_std=0.1):
        """
        Updates state with full 3D velocity measurement (Validation Only).
        """
        if not self.initialized:
            return

        r = v_ned_meas - self.v

        # H is Identity at index 6 (Velocity)
        # 3xN
        H = np.zeros((3, self.P.shape[0]))
        H[0:3, 6:9] = np.eye(3)

        R_noise = np.eye(3) * noise_std**2

        try:
            S = H @ self.P @ H.T + R_noise
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
            dx = K @ r

            self._inject_error(dx)
            I = np.eye(self.P.shape[0])
            self.P = (I - K @ H) @ self.P
        except Exception as e:
            logger.error(f"Vel Vector Update Failed: {e}")

    def update_features(self, tracks):
        """
        MSCKF Update.
        tracks: list of tuples (feature_id, observations)
        observations: list of (clone_index, u_norm, v_norm)
        """
        # For each track, triangulate and compute residuals
        # We process tracks that have lost tracking or are ready

        # Stack residuals and H matrices
        H_list = []
        r_list = []

        R_ic = self.R_c2b.T # Camera to IMU (Body) if needed, but we have clones in World frame?
        # Clones are Camera Poses in World Frame.

        for track in tracks:
            obs = track['obs'] # list of (clone_idx, u, v)
            if len(obs) < 2: continue

            # 1. Triangulate
            # We need world poses of cameras
            poses = []
            measurements = []

            valid_obs = []

            for o in obs:
                c_idx = o['clone_idx']
                if c_idx >= len(self.cam_clones): continue # Should not happen

                clone = self.cam_clones[c_idx]
                R_wc = R.from_quat(clone['q']).as_matrix()
                p_wc = clone['p']

                poses.append((R_wc, p_wc))
                measurements.append((o['u'], o['v']))
                valid_obs.append(o)

            if len(poses) < 2: continue

            p_feat_w = self._triangulate(poses, measurements)
            if p_feat_w is None: continue

            # 2. Compute Residuals & Jacobians
            # Linearize re-projection error
            # r = z - h(x)
            # r ~= H_x * dx + H_f * dp_f

            H_x_j = np.zeros((2 * len(valid_obs), self.P.shape[0]))
            H_f_j = np.zeros((2 * len(valid_obs), 3))
            r_j = np.zeros(2 * len(valid_obs))

            for i, o in enumerate(valid_obs):
                c_idx = o['clone_idx']
                R_wc, p_wc = poses[i]
                u_meas, v_meas = measurements[i]

                # Transform feature to camera frame
                # P_c = R_wc.T * (P_w - p_wc)
                P_c = R_wc.T @ (p_feat_w - p_wc)

                if P_c[2] < 0.1: continue # Behind camera

                # Predict uv
                u_pred = P_c[0] / P_c[2]
                v_pred = P_c[1] / P_c[2]

                r_j[2*i] = u_meas - u_pred
                r_j[2*i+1] = v_meas - v_pred

                # Jacobian w.r.t Feature Position P_w
                # d(uv)/dP_w = d(uv)/dP_c * dPc/dP_w
                # d(uv)/dP_c = [1/z 0 -x/z2; 0 1/z -y/z2]
                J_uv_Pc = (1.0/P_c[2]) * np.array([
                    [1, 0, -u_pred],
                    [0, 1, -v_pred]
                ])
                J_Pc_Pw = R_wc.T

                H_f_j[2*i:2*i+2, :] = J_uv_Pc @ J_Pc_Pw

                # Jacobian w.r.t State (Pose of Clone i)
                # dPc/dtheta_c = [P_c]x
                # dPc/dp_c = -R_wc.T

                J_Pc_theta = self.skew(P_c)
                J_Pc_p = -R_wc.T

                # Map to State Vector Indices
                # Clone State: [theta_err, p_err] at 15 + 6*c_idx
                col_idx = 15 + 6 * c_idx

                H_x_j[2*i:2*i+2, col_idx:col_idx+3] = J_uv_Pc @ J_Pc_theta
                H_x_j[2*i:2*i+2, col_idx+3:col_idx+6] = J_uv_Pc @ J_Pc_p

            # 3. Null Space Projection (Remove H_f)
            # Project residuals onto null space of feature jacobian
            # U, S, V = svd(H_f_j)
            # Left Null Space is last rows of U.T?
            # Or QR decomposition. H_f = [Q1 Q2] [R; 0]. Q2 is null space.
            # r_proj = Q2.T * r
            # H_proj = Q2.T * H_x

            # Using QR
            try:
                Q_mat, R_mat = np.linalg.qr(H_f_j, mode='complete')
                # Rank of H_f_j is 3 (point in 3D). Null space dim is 2N - 3.
                # Q1 is 2N x 3. Q2 is 2N x (2N-3).
                Q2 = Q_mat[:, 3:]

                if Q2.shape[1] > 0:
                    r_proj = Q2.T @ r_j
                    H_proj = Q2.T @ H_x_j

                    # Chi-Square Gating
                    try:
                        # Noise for this feature's residuals
                        # r_proj is (2N-3)
                        dof = r_proj.shape[0]
                        noise_var = (2.0 / 480.0)**2
                        R_noise_i = np.eye(dof) * noise_var

                        # Innovation Covariance S_i = H P H^T + R
                        S_i = H_proj @ self.P @ H_proj.T + R_noise_i

                        # Chi2 Statistic
                        # gamma = r^T * S^-1 * r
                        # Use solve for stability
                        gamma = r_proj.T @ np.linalg.solve(S_i, r_proj)

                        # Threshold (95% confidence)
                        threshold = chi2.ppf(0.95, df=dof)

                        if gamma < threshold:
                            H_list.append(H_proj)
                            r_list.append(r_proj)
                        else:
                            logger.warning(f"Feature Rejected: Chi2 {gamma:.2f} > {threshold:.2f} (DoF {dof})")

                    except np.linalg.LinAlgError:
                        logger.warning("Chi2 Gating Failed (Singular Matrix), skipping feature")
                        pass

            except Exception as e:
                logger.error(f"Projection/Nullspace Failed: {e}")
                pass

        if not H_list:
            return

        # Mark as reliable since we are processing features
        self.features_processed = True

        # Stack all projected residuals
        H_stack = np.vstack(H_list)
        r_stack = np.hstack(r_list)

        # EKF Update
        noise_var = (2.0 / 480.0)**2
        R_noise = np.eye(len(r_stack)) * noise_var

        S = H_stack @ self.P @ H_stack.T + R_noise

        # Add small regularization for numerical stability
        S += np.eye(S.shape[0]) * 1e-6

        try:
            # Use pseudo-inverse for robustness against singular S
            # S is symmetric, so pinvh is good or just pinv
            S_inv = np.linalg.pinv(S)
            K = self.P @ H_stack.T @ S_inv
            dx = K @ r_stack

            # Log dx for velocity
            dv = dx[6:9]
            if np.linalg.norm(dv) > 1.0:
                 logger.warning(f"Feature Update induced Large Velocity Jump: {dv} (Resid Norm: {np.linalg.norm(r_stack):.2f})")

            self._inject_error(dx)

            I = np.eye(self.P.shape[0])
            self.P = (I - K @ H_stack) @ self.P
            self._log_state("Post-Feature")
        except Exception as e:
            logger.error(f"VIO Update Failed: {e}")

    def _inject_error(self, dx):
        # Apply error state to nominal state

        # 1. Orientation
        try:
            # Check for NaN or Inf
            if not np.all(np.isfinite(dx[0:3])):
                 logger.error(f"Invalid Orientation Update: {dx[0:3]}")
                 return

            dq = R.from_rotvec(dx[0:3]).as_quat()
            r_old = R.from_quat(self.q)
            self.q = (r_old * R.from_quat(dq)).as_quat()
        except ValueError as e:
            logger.error(f"Orientation Update Failed: {e}. dx={dx[0:3]}")
            return

        # 2. Pos/Vel/Biases
        self.p += dx[3:6]
        self.v += dx[6:9]

        # Clamp Velocity to avoid explosion
        v_norm = np.linalg.norm(self.v)
        if v_norm > 50.0:
             self.v = (self.v / v_norm) * 50.0

        self.bg += dx[9:12]
        self.ba += dx[12:15]

        # 3. Clones
        for i, clone in enumerate(self.cam_clones):
            base = 15 + 6 * i
            dq_c = R.from_rotvec(dx[base:base+3]).as_quat()
            r_c = R.from_quat(clone['q'])
            clone['q'] = (r_c * R.from_quat(dq_c)).as_quat()
            clone['p'] += dx[base+3:base+6]

    def _triangulate(self, poses, measurements):
        # Simple Linear Triangulation
        # P_c = z * [u, v, 1]
        # P_w = R_wc * P_c + p_wc = z * R_wc * [u,v,1] + p_wc
        # We want P_w that satisfies all rays.

        # For each view: [u v 1] x (R_wc.T * (P_w - p_wc)) = 0
        # A * P_w = 0

        A = []
        for (R_wc, p_wc), (u, v) in zip(poses, measurements):
            # P_c = R_wc.T @ P_w - R_wc.T @ p_wc
            # u * P_cz - P_cx = 0
            # v * P_cz - P_cy = 0

            # Row 1: u * r3 - r1
            # Row 2: v * r3 - r2
            # where r1, r2, r3 are rows of R_wc.T

            RT = R_wc.T
            t = -RT @ p_wc

            r1 = RT[0, :]
            r2 = RT[1, :]
            r3 = RT[2, :]

            A.append(u * r3 - r1)
            A.append(v * r3 - r2)

            # Should account for t in inhomogenous system A P_w = -b
            # u * (r3.P + t3) - (r1.P + t1) = 0 => (u r3 - r1) P = t1 - u t3

        A_mat = np.array(A)

        # B vector
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

    def skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def get_velocity(self):
        return self.v

    def get_state_dict(self):
        # Return state in standard format
        r = R.from_quat(self.q)
        euler = r.as_euler('xyz', degrees=False) # Check frame convention
        # We used standard aerospace propagation, likely ENU if initialized ENU

        return {
            'px': self.p[0], 'py': self.p[1], 'pz': self.p[2],
            'vx': self.v[0], 'vy': self.v[1], 'vz': self.v[2],
            'roll': euler[0], 'pitch': euler[1], 'yaw': euler[2]
        }
