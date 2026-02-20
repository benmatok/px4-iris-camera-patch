import numpy as np
from flight_config import FlightConfig
import logging
import cv2

logger = logging.getLogger(__name__)

class FeatureTracker:
    def __init__(self, projector, config: FlightConfig = None):
        self.projector = projector
        self.config = config or FlightConfig()
        vis = self.config.vision

        self.num_points = vis.num_flow_points
        self.world_points = {}  # id -> np.array([x, y, z]) (Ground Truth World Position)
        self.next_id = 0

        # Track history for VIO
        # id -> list of {'obs': (u, v), 'clone_idx': clone_idx}
        self.active_tracks = {}

        self.prev_projections = {}  # id -> (u_norm, v_norm) (For FOE calc)

        # FOE Smoothing
        self.filtered_foe_u = None
        self.filtered_foe_v = None

    def update(self, drone_state, body_rates, dt, clone_idx):
        """
        Updates tracker, managing feature lifecycle.
        Returns:
            foe: (u, v) or None
            finished_tracks: list of dicts suitable for MSCKF update
        """
        # 1. Replenish Points
        if len(self.world_points) < self.num_points:
             self._generate_points(drone_state, self.num_points - len(self.world_points))

        # 2. Project Points to Current Frame
        curr_projections = {}
        valid_ids = set()

        # To store finished tracks this frame
        finished_tracks = []

        for pid, pt in self.world_points.items():
            res = self.projector.world_to_normalized(pt[0], pt[1], pt[2], drone_state)
            visible = False

            if res:
                u_norm, v_norm = res
                u_px = u_norm * self.projector.fx + self.projector.cx
                v_px = v_norm * self.projector.fy + self.projector.cy

                if 0 <= u_px <= self.projector.width and 0 <= v_px <= self.projector.height:
                    visible = True
                    curr_projections[pid] = res
                    valid_ids.add(pid)

                    # Update Track History
                    if pid not in self.active_tracks:
                        self.active_tracks[pid] = {'obs': []}

                    self.active_tracks[pid]['obs'].append({
                        'clone_idx': clone_idx,
                        'u': u_norm,
                        'v': v_norm
                    })

            if not visible and pid in self.active_tracks:
                # Feature lost -> Finished Track
                track = self.active_tracks.pop(pid)
                if len(track['obs']) >= 2:
                    finished_tracks.append(track)

        # Pruning (Lost or Invisible)
        keys = list(self.world_points.keys())
        for k in keys:
            if k not in valid_ids:
                del self.world_points[k]
                if k in self.prev_projections:
                    del self.prev_projections[k]
                # Note: We already handled 'finished_tracks' logic above based on visibility check logic

        # 3. Calculate FOE (Simple Flow for Flight Controller logic - kept for legacy/backup)
        foe = self._calculate_foe(curr_projections, body_rates, dt)

        # Store correspondences for Homography estimation
        self.last_correspondences = []
        for pid, (u_curr, v_curr) in curr_projections.items():
            if pid in self.prev_projections:
                u_prev, v_prev = self.prev_projections[pid]
                self.last_correspondences.append(((u_prev, v_prev), (u_curr, v_curr)))

        self.prev_projections = curr_projections

        return foe, finished_tracks

    def estimate_homography_velocity(self, dt, height, camera_matrix):
        """
        Estimates velocity from homography of the ground plane.
        Returns:
            v_world: (vx, vy, vz) in World Frame (Sim/ENU)
            or None if failed
        """
        if len(self.last_correspondences) < 8:
            return None

        # Pixel Coordinates
        # u_norm is xc/zc. u_px = u_norm * fx + cx
        # We need pixel coords for findHomography? No, normalized is fine, but K needs to be identity.
        # Or convert normalized back to pixel if using K.
        # Normalized coords are essentially "calibrated pixels".

        pts_prev = []
        pts_curr = []

        fx = self.projector.fx
        fy = self.projector.fy
        cx = self.projector.cx
        cy = self.projector.cy

        for (u_prev_n, v_prev_n), (u_curr_n, v_curr_n) in self.last_correspondences:
            u_prev_px = u_prev_n * fx + cx
            v_prev_px = v_prev_n * fy + cy
            u_curr_px = u_curr_n * fx + cx
            v_curr_px = v_curr_n * fy + cy

            pts_prev.append([u_prev_px, v_prev_px])
            pts_curr.append([u_curr_px, v_curr_px])

        pts_prev = np.array(pts_prev)
        pts_curr = np.array(pts_curr)

        # Estimate Homography H: x2 = H x1
        H, mask = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, 5.0)

        if H is None:
            return None

        # Decompose Homography
        # x2 = (R - t n^T / d) x1
        # Need Camera Matrix K
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

        # Select best solution
        # Ground plane normal is roughly [0, 0, 1] in World.
        # In Camera frame?
        # Cam: X Right, Y Down, Z Forward (30 deg tilt up)
        # World Z is Up. So Ground Normal is +Z (0,0,1).
        # Body Z is Down. Ground Normal in Body is -Z? Or +Z if flying normal?
        # Let's assume standard flight: Ground is "down" relative to drone body.
        # Drone Body Z points Down. So Ground Normal (pointing up out of ground) is -Z_body.
        # R_c2b converts Cam to Body.
        # n_cam = R_c2b.T @ n_body = R_c2b.T @ [0, 0, -1]

        n_ground_world = np.array([0, 0, 1]) # Up

        # We don't know attitude perfectly here? We do have drone_state in update() but not passed here.
        # But decomposeHomography gives us n relative to Camera.
        # We can check which n is closest to "expected ground normal" if we know attitude.
        # Assuming we are roughly level flight for now.
        # If we don't assume attitude, we just take the one with valid height?

        best_idx = -1
        min_error = 1e9

        # Expected normal in Camera Frame (Approx)
        # Tilt 30 deg up. Z_cam points forward-up. Y_cam points down.
        # Ground is below Y_cam. Normal points -Y_cam (approx).
        # Let's refine:
        # Cam Y is Down. Ground is "below". Normal points UP (towards camera).
        # So Normal is -Y_cam roughly.

        expected_n_cam = np.array([0, -1, 0])

        for i in range(num):
            n = Ns[i].flatten()
            t = Ts[i].flatten()

            # Check normal direction
            # If n is opposite to expected, we might need to flip?
            # Ideally dot product should be close to 1.

            score = np.dot(n, expected_n_cam)

            # Also check t?
            # v = t / dt.

            if score > 0.5: # Approx alignment
                best_idx = i
                break # Take first good one?

        if best_idx == -1:
            return None

        # Get Translation t (scaled by d)
        # T_decomp = t_real / d
        # t_real = T_decomp * d
        # d is distance from camera center to plane along normal.
        # d = altitude / (n . z_world_in_cam)?
        # Height is vertical distance (Z world).
        # d = h / cos(theta) where theta is angle between optical axis and vertical?
        # Simpler: d = height (approx for now if looking down).
        # Let's use d = height.

        t_estimated = Ts[best_idx].flatten() * height

        # Velocity in Camera Frame
        v_cam = t_estimated / dt

        # Convert to Body Frame
        v_body = self.projector.R_c2b @ v_cam

        # Convert to World Frame (NED) - wait, we need R_b2w (Attitude)
        # We don't have attitude in this method signature.
        # BUT the user said "aggregate to a velocity measurement".
        # Usually VIO wants Body Velocity or World Velocity.
        # MSCKF tracks World Velocity.
        # If we return v_body, we can rotate it using MSCKF's current q estimate.

        return v_body # Return Body Velocity for now

    def _calculate_foe(self, curr_projections, body_rates, dt):
        flow_vectors = []
        w_b = np.array(body_rates)
        w_c = self.projector.R_c2b.T @ w_b
        wx, wy, wz = w_c

        for pid, (u_curr, v_curr) in curr_projections.items():
            if pid in self.prev_projections:
                u_prev, v_prev = self.prev_projections[pid]
                du = u_curr - u_prev
                dv = v_curr - v_prev

                x, y = u_prev, v_prev
                du_rot = (x * y * wx - (1 + x**2) * wy + y * wz) * dt
                dv_rot = ((1 + y**2) * wx - x * y * wy - x * wz) * dt

                du_trans = du - du_rot
                dv_trans = dv - dv_rot
                flow_vectors.append((x, y, du_trans, dv_trans))

        if len(flow_vectors) < 5: return None

        # Simple LS for FOE
        A = []; b = []
        for x, y, du, dv in flow_vectors:
            if abs(du) < 1e-6 and abs(dv) < 1e-6: continue
            A.append([dv, -du])
            b.append(dv * x - du * y)

        if len(A) < 5: return None

        try:
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            u_foe, v_foe = res[0], res[1]

            # Filter
            alpha = 0.2
            if self.filtered_foe_u is None:
                 self.filtered_foe_u = u_foe
                 self.filtered_foe_v = v_foe
            else:
                 self.filtered_foe_u = alpha * u_foe + (1 - alpha) * self.filtered_foe_u
                 self.filtered_foe_v = alpha * v_foe + (1 - alpha) * self.filtered_foe_v

            return (self.filtered_foe_u, self.filtered_foe_v)
        except:
            return None

    def _get_terrain_height(self, x, y):
        # Same synthetic terrain
        z = 5.0 * np.sin(x * 0.05) * np.cos(y * 0.05)
        z += 2.0 * np.sin(x * 0.15 + 1.0) * np.sin(y * 0.1)
        z += 10.0 * np.sin(x * 0.01)
        return z

    def _generate_points(self, drone_state, count):
        # Same generation logic
        pos = np.array([drone_state['px'], drone_state['py'], drone_state['pz']])
        roll = drone_state['roll']; pitch = drone_state['pitch']; yaw = drone_state['yaw']
        cphi, sphi = np.cos(roll), np.sin(roll)
        ctheta, stheta = np.cos(pitch), np.sin(pitch)
        cpsi, spsi = np.cos(yaw), np.sin(yaw)

        r11 = ctheta * cpsi
        r12 = cpsi * sphi * stheta - cphi * spsi
        r13 = sphi * spsi + cphi * cpsi * stheta
        r21 = ctheta * spsi
        r22 = cphi * cpsi + sphi * spsi * stheta
        r23 = cphi * spsi * stheta - cpsi * sphi
        r31 = -stheta
        r32 = ctheta * sphi
        r33 = cphi * ctheta

        R_b2w = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        R_c2b = self.projector.R_c2b

        u_samples = np.random.uniform(0, self.projector.width, size=count)
        v_samples = np.random.uniform(0, self.projector.height, size=count)

        x_c = (u_samples - self.projector.cx) / self.projector.fx
        y_c = (v_samples - self.projector.cy) / self.projector.fy
        z_c = np.ones(count)
        vecs_c = np.vstack((x_c, y_c, z_c))
        vecs_b = R_c2b @ vecs_c
        vecs_w = R_b2w @ vecs_b

        norms = np.linalg.norm(vecs_w, axis=0)
        vecs_w = vecs_w / norms

        valid_indices = vecs_w[2, :] > 0.1
        vecs_w_valid = vecs_w[:, valid_indices]
        if vecs_w_valid.shape[1] == 0: return

        O = pos.reshape(3, 1)
        t = -O[2] / vecs_w_valid[2, :]

        for _ in range(3):
            P = O + vecs_w_valid * t
            target_z = self._get_terrain_height(P[0, :], P[1, :])
            t = (target_z - O[2]) / vecs_w_valid[2, :]

        P_final = O + vecs_w_valid * t
        for i in range(P_final.shape[1]):
            self.world_points[self.next_id] = P_final[:, i]
            self.next_id += 1
