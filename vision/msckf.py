import numpy as np
import logging
import csv
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

logger = logging.getLogger(__name__)

class MSCKF:
    """
    Batch Window Velocity Estimator.
    Accumulates IMU and Features for 1s, then resolves state.
    """
    def __init__(self, projector):
        self.projector = projector
        self.g = np.array([0, 0, 9.81]) # NED Gravity

        # State at start of window
        self.start_state = {
            'q': np.array([0., 0., 0., 1.]),
            'p': np.zeros(3),
            'v': np.zeros(3),
            'bg': np.zeros(3),
            'ba': np.zeros(3)
        }

        # Live State (Integrated)
        self.live_state = self.start_state.copy()

        # Buffers
        self.imu_buffer = [] # (gyro, accel, dt)
        self.obs_buffer = [] # list of {'id':, 'u':, 'v':, 't_rel':}
        self.baro_buffer = [] # list of (pz, t_rel)

        self.window_duration = 1.0
        self.time_in_window = 0.0
        self.total_time = 0.0

        self.R_c2b = self.projector.R_c2b
        self.p_c2b = np.zeros(3)

        self.initialized = False

        self.log_file = open("vio_batch_log.csv", "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["time", "vx_live", "vy_live", "vz_live", "vx_res", "vy_res", "vz_res"])

    def initialize(self, q, p, v):
        self.start_state['q'] = q
        self.start_state['p'] = p
        self.start_state['v'] = v
        self.live_state = self.start_state.copy()
        self.initialized = True
        logger.info("Batch VIO Initialized")

    def is_reliable(self):
        return self.initialized

    def propagate(self, gyro, accel, dt):
        if not self.initialized: return

        self.imu_buffer.append((gyro, accel, dt))

        # Propagate Live State
        self._propagate_state(self.live_state, gyro, accel, dt)

        self.time_in_window += dt
        self.total_time += dt

        if self.time_in_window >= self.window_duration:
            self._resolve_window()

    def add_observation(self, feature_id, u, v):
        # Called for each feature point visible in current frame
        if not self.initialized: return
        self.obs_buffer.append({
            'id': feature_id,
            'u': u,
            'v': v,
            't_rel': self.time_in_window
        })

    def update_height(self, pz):
        if not self.initialized: return
        self.baro_buffer.append((pz, self.time_in_window))
        # Simple Live Update
        # self.live_state['p'][2] = pz # Optional: Clamp live Z

    def predict_foe(self):
        if not self.initialized: return None
        r_wb = R.from_quat(self.live_state['q']).as_matrix()
        v_b = r_wb.T @ self.live_state['v']
        v_c = self.R_c2b.T @ v_b

        if v_c[2] < 0.1: return None
        u = v_c[0] / v_c[2]
        v = v_c[1] / v_c[2]
        return (u, v)

    def get_velocity(self):
        return self.live_state['v']

    def _propagate_state(self, state, gyro, accel, dt):
        # Unbias
        w = gyro - state['bg']
        a = accel - state['ba']

        # Rotation
        angle = np.linalg.norm(w) * dt
        axis = w / np.linalg.norm(w) if np.linalg.norm(w) > 1e-6 else np.array([1,0,0])
        dq = R.from_rotvec(axis * angle).as_quat()
        r_old = R.from_quat(state['q'])
        state['q'] = (r_old * R.from_quat(dq)).as_quat()

        # Velocity & Position
        R_mat = r_old.as_matrix()
        acc_world = R_mat @ a + self.g

        state['p'] = state['p'] + state['v'] * dt + 0.5 * acc_world * dt**2
        state['v'] = state['v'] + acc_world * dt

    def _resolve_window(self):
        # 1. Re-integrate to generate Poses for all observations
        # We need pose at each t_rel in obs_buffer
        # Sort obs by time
        sorted_obs = sorted(self.obs_buffer, key=lambda x: x['t_rel'])

        temp_state = {k: v.copy() for k, v in self.start_state.items()}

        obs_idx = 0
        imu_idx = 0
        current_t = 0.0

        poses = [] # list of (t_rel, R_wb, p_w)

        # Store pose at t=0
        poses.append((0.0, R.from_quat(temp_state['q']).as_matrix(), temp_state['p'].copy()))

        # Integrate and capture poses
        for (gyro, accel, dt) in self.imu_buffer:
            next_t = current_t + dt
            self._propagate_state(temp_state, gyro, accel, dt)
            poses.append((next_t, R.from_quat(temp_state['q']).as_matrix(), temp_state['p'].copy()))
            current_t = next_t

        final_state_prior = temp_state

        # 2. Group Obs by ID
        features = defaultdict(list)
        for o in sorted_obs:
            features[o['id']].append(o)

        # 3. Batch Solve for Correction
        # We want to correct the Velocity at the START of the window (v0)
        # or Velocity at END?
        # Let's estimate Velocity Error (constant bias) over the window.
        # H matrix: d(residual)/d(v_error).

        # For each feature:
        # Triangulate using nominal poses.
        # Compute residuals.
        # Compute H_v for each residual.

        H_list = []
        r_list = []

        for fid, obs_list in features.items():
            if len(obs_list) < 3: continue

            # Get poses for these obs (Interpolate if needed, but NN is fine for high rate IMU)
            # Find closest pose in `poses`
            feat_poses = []
            feat_meas = []

            for o in obs_list:
                t = o['t_rel']
                # Find pose closest to t
                # poses is sorted by time.
                # Simple search
                best_pose = min(poses, key=lambda x: abs(x[0] - t))

                # Transform to Camera Frame
                R_wb = best_pose[1]
                p_w = best_pose[2]

                R_wc = R_wb @ self.R_c2b
                p_wc = p_w + R_wb @ self.p_c2b

                feat_poses.append((R_wc, p_wc))
                feat_meas.append((o['u'], o['v']))

            # Triangulate
            p_feat_w = self._triangulate(feat_poses, feat_meas)
            if p_feat_w is None: continue

            # Compute Residuals & Jacobian
            for i, (R_wc, p_wc) in enumerate(feat_poses):
                u_m, v_m = feat_meas[i]

                P_c = R_wc.T @ (p_feat_w - p_wc)
                if P_c[2] < 0.1: continue

                u_p = P_c[0] / P_c[2]
                v_p = P_c[1] / P_c[2]

                res = np.array([u_m - u_p, v_m - v_p])

                # Jacobian w.r.t P_c
                J_uv_Pc = (1.0/P_c[2]) * np.array([
                    [1, 0, -u_p],
                    [0, 1, -v_p]
                ])

                # We want Jacobian w.r.t Velocity Error (dv)
                # Position Error dp(t) ~ t * dv
                # P_c = R.T * (P_w - p_w)
                # dPc/dp_w = -R.T
                # dPc/dv = dPc/dp_w * dp_w/dv = -R.T * t

                t_obs = obs_list[i]['t_rel']
                J_Pc_v = -R_wc.T * t_obs

                H_block = J_uv_Pc @ J_Pc_v # 2x3

                H_list.append(H_block)
                r_list.append(res)

        # Solve (Direction Only)
        if len(H_list) > 0:
            H_stack = np.vstack(H_list)
            r_stack = np.hstack(r_list)

            # Current estimated velocity at end of window
            v_curr = final_state_prior['v']
            speed = np.linalg.norm(v_curr)
            if speed < 0.1:
                # Too slow to determine direction reliably, skip update
                self.start_state = final_state_prior
                self.live_state = final_state_prior.copy()
                # Reset Buffers
                self.imu_buffer = []
                self.obs_buffer = []
                self.baro_buffer = []
                self.time_in_window = 0.0
                return

            # Compute Basis for velocity direction
            # v = speed * direction
            # We want to correct direction using angles alpha (azimuth) and beta (elevation)
            # Let's use a local tangent plane to the velocity vector.
            # v_dir = v / speed
            # b1, b2 are orthogonal to v_dir

            v_dir = v_curr / speed

            # Arbitrary vector not parallel to v_dir
            if abs(v_dir[2]) < 0.9:
                arb = np.array([0, 0, 1])
            else:
                arb = np.array([0, 1, 0])

            b1 = np.cross(v_dir, arb)
            b1 /= np.linalg.norm(b1)
            b2 = np.cross(v_dir, b1)

            # Jacobian of v w.r.t local angle perturbations d_theta (2x1)
            # v_new ~= v_curr + speed * (b1 * d_theta1 + b2 * d_theta2)
            # dv/d_theta = speed * [b1, b2] (3x2)

            J_v_theta = speed * np.column_stack((b1, b2))

            # Total Jacobian H_theta = H_v * J_v_theta
            H_theta = H_stack @ J_v_theta # (N x 3) @ (3 x 2) = (N x 2)

            # Solve for d_theta
            HTH = H_theta.T @ H_theta
            HTr = H_theta.T @ r_stack

            # Small prior to avoid singularity
            prior_inf = np.eye(2) * 1.0

            d_theta = np.linalg.solve(HTH + prior_inf, HTr)

            # Update Velocity Direction
            # v_corr = speed * (b1 * d_theta[0] + b2 * d_theta[1])
            # v_new = v_curr + v_corr
            # Renormalize to keep speed constant

            dv = speed * (b1 * d_theta[0] + b2 * d_theta[1])
            v_new = v_curr + dv
            v_new = v_new / np.linalg.norm(v_new) * speed

            final_state_prior['v'] = v_new

            logger.info(f"Resolved Window: t={self.total_time:.2f}, dv={dv}")

            # Log correction
            self.csv_writer.writerow([
                f"{self.total_time:.3f}",
                f"{self.live_state['v'][0]:.3f}", f"{self.live_state['v'][1]:.3f}", f"{self.live_state['v'][2]:.3f}",
                f"{final_state_prior['v'][0]:.3f}", f"{final_state_prior['v'][1]:.3f}", f"{final_state_prior['v'][2]:.3f}"
            ])
            self.log_file.flush()

            # Update State
            self.start_state = final_state_prior
            self.live_state = final_state_prior.copy()

        else:
            # No features, just accept IMU propagation
            self.start_state = final_state_prior
            self.live_state = final_state_prior.copy()

        # Reset Buffers
        self.imu_buffer = []
        self.obs_buffer = []
        self.baro_buffer = []
        self.time_in_window = 0.0

    def _triangulate(self, poses, measurements):
        A = []
        B = []
        for (R_wc, p_wc), (u, v) in zip(poses, measurements):
            RT = R_wc.T
            t = -RT @ p_wc
            r1, r2, r3 = RT[0, :], RT[1, :], RT[2, :]
            A.append(u * r3 - r1)
            A.append(v * r3 - r2)
            B.append(t[0] - u * t[2])
            B.append(t[1] - v * t[2])

        try:
            res, _, _, _ = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)
            return res
        except:
            return None
