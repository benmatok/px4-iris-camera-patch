import numpy as np
from flight_config import FlightConfig
import logging

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
        current_obs = []

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
                    current_obs.append({'id': pid, 'u': u_norm, 'v': v_norm})

            # Force finish track if too long to ensure MSCKF update (reduce drift)
            # Threshold must be <= max_clones (5) to ensure clones exist!
            if pid in self.active_tracks and len(self.active_tracks[pid]['obs']) >= 4:
                 track = self.active_tracks[pid]
                 if len(track['obs']) >= 2:
                     # We need to send a COPY or just the list?
                     # The list is in the dict.
                     # We want to reset the dict in self.active_tracks.
                     finished_tracks.append(track)

                     # Start new track with current observation as anchor
                     last_obs = track['obs'][-1]
                     self.active_tracks[pid] = {'obs': [last_obs]}

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
        # Just use average flow direction? Or same logic as before?
        # Let's keep the simple LS FOE estimator for immediate control feedback if needed,
        # or rely on VIO velocity direction.
        # Flight Controller uses FOE for some heuristics?
        # Actually FC uses `foe_uv` mostly for debugging or simple servoing?
        # Looking at FC: `yaw_rate_cmd = -ctrl.k_yaw * u` uses tracking_uv (Target).
        # `foe_uv` is not used in main control logic! It is just passed for ... ?
        # Checked `flight_controller.py`: `foe_uv` is NOT used in `compute_action`.
        # So we can just return None for FOE or implement it if we want visualization.

        foe = self._calculate_foe(curr_projections, body_rates, dt)

        self.prev_projections = curr_projections

        return foe, finished_tracks, current_obs

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
