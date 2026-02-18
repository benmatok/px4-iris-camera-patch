import numpy as np
from scipy.optimize import least_squares
from flight_config import FlightConfig

class FlowVelocityEstimator:
    def __init__(self, projector, config: FlightConfig = None):
        self.projector = projector
        self.config = config or FlightConfig()
        vis = self.config.vision

        self.num_points = vis.num_flow_points
        self.filtered_foe_u = None
        self.filtered_foe_v = None
        self.world_points = {}  # id -> np.array([x, y, z])
        self.next_id = 0
        self.prev_projections = {}  # id -> (u_norm, v_norm)

        # Generation parameters
        self.gen_dist_min = vis.flow_gen_dist_min
        self.gen_dist_max = vis.flow_gen_dist_max

        # Calculate Body Forward direction in Normalized Camera Coordinates
        # Body Forward = [1, 0, 0] in Body Frame
        # P_c = R_c2b.T @ P_b
        # R_c2b maps Camera to Body. Transpose maps Body to Camera.
        P_b = np.array([1.0, 0.0, 0.0])
        P_c = self.projector.R_c2b.T @ P_b

        # Project to Normalized Image Plane (Z is forward)
        if P_c[2] > 0.001:
            self.prior_u = P_c[0] / P_c[2]
            self.prior_v = P_c[1] / P_c[2]
        else:
            # Fallback if camera looks backward relative to body (unlikely)
            self.prior_u = 0.0
            self.prior_v = 0.0

    def update(self, drone_state, body_rates, dt):
        """
        Updates the estimator with new drone state and calculates FOE.

        Args:
            drone_state: dict with keys 'px', 'py', 'pz', 'roll', 'pitch', 'yaw' (NED)
            body_rates: tuple/list (wx, wy, wz) in Body Frame (rad/s)
            dt: time step in seconds

        Returns:
            (u, v): Normalized coordinates of Focus of Expansion (FOE), or None if unstable.
        """
        # 1. Replenish Points if needed
        if len(self.world_points) < self.num_points:
             self._generate_points(drone_state, self.num_points - len(self.world_points))

        # 2. Project Points to Current Frame
        curr_projections = {}
        valid_ids = set()

        points_to_remove = []

        for pid, pt in self.world_points.items():
            res = self.projector.world_to_normalized(pt[0], pt[1], pt[2], drone_state)
            if res:
                # Check if within image bounds
                u_norm, v_norm = res
                u_px = u_norm * self.projector.fx + self.projector.cx
                v_px = v_norm * self.projector.fy + self.projector.cy

                if 0 <= u_px <= self.projector.width and 0 <= v_px <= self.projector.height:
                    curr_projections[pid] = res
                    valid_ids.add(pid)

        # Pruning Strategy: Remove points that are not visible in the current frame
        keys = list(self.world_points.keys())
        for k in keys:
            if k not in valid_ids:
                del self.world_points[k]
                if k in self.prev_projections:
                    del self.prev_projections[k]

        # 3. Calculate Flow
        flow_vectors = []

        # Transform body rates to camera frame
        # w_c = R_c2b.T @ w_b
        w_b = np.array(body_rates)
        w_c = self.projector.R_c2b.T @ w_b
        wx, wy, wz = w_c

        for pid, (u_curr, v_curr) in curr_projections.items():
            if pid in self.prev_projections:
                u_prev, v_prev = self.prev_projections[pid]

                du = u_curr - u_prev
                dv = v_curr - v_prev

                # Expected Rotational Flow (using prev coords as approximation)
                x, y = u_prev, v_prev

                # Formula:
                # u_rot = x*y*wx - (1+x^2)*wy + y*wz
                # v_rot = (1+y^2)*wx - x*y*wy - x*wz

                du_rot = (x * y * wx - (1 + x**2) * wy + y * wz) * dt
                dv_rot = ((1 + y**2) * wx - x * y * wy - x * wz) * dt

                du_trans = du - du_rot
                dv_trans = dv - dv_rot

                flow_vectors.append((x, y, du_trans, dv_trans))

        self.prev_projections = curr_projections

        # 4a. Estimate Velocity (Full 3D)
        # We assume flat ground at current altitude to derive depth for each point.
        vel_est, vel_reliable = self._estimate_velocity(flow_vectors, drone_state, dt)

        # 4. Estimate FOE (Initial Guess via Algebraic Least Squares)
        if len(flow_vectors) < 10:
            return None, vel_est, vel_reliable

        A = []
        b = []

        for x, y, du, dv in flow_vectors:
            # Filter small flow to avoid noise?
            if abs(du) < 1e-5 and abs(dv) < 1e-5:
                continue

            # Equation: v_trans * (x - xf) - u_trans * (y - yf) = 0
            # v * xf - u * yf = v * x - u * y

            A.append([dv, -du])
            b.append(dv * x - du * y)

        if len(A) < 5:
            return None, vel_est, vel_reliable

        A = np.array(A)
        b = np.array(b)

        try:
            # Initial Guess (Algebraic)
            res, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            xf_init, yf_init = res[0], res[1]

            # 5. Refine with Levenberg-Marquardt (Geometric Error)
            # Minimize sum of squared perpendicular distances from points to lines passing through FOE.
            # Or simpler: Minimize angular error between flow vector and radial line from FOE.
            # Vector from FOE to point: P - F = (x - xf, y - yf)
            # Flow Vector: V = (du, dv)
            # Cross product (2D) should be zero if collinear: (x-xf)*dv - (y-yf)*du = 0
            # This is exactly the algebraic error!
            # BUT, the algebraic error is weighted by magnitude of flow vector (du, dv).
            # Geometric error (perpendicular distance to line):
            # dist = |(x-xf)*dv - (y-yf)*du| / sqrt(du^2 + dv^2)
            # Minimizing this normalizes the weight of each point.

            # Prepare data for optimization
            flow_data = np.array(flow_vectors) # N x 4: x, y, du, dv

            def algebraic_residuals(params):
                xf, yf = params
                x = flow_data[:, 0]
                y = flow_data[:, 1]
                du = flow_data[:, 2]
                dv = flow_data[:, 3]

                # Algebraic error: (x-xf)*dv - (y-yf)*du
                # This error corresponds to the geometric distance from the epipolar line,
                # weighted by the flow magnitude sqrt(du^2 + dv^2).
                # Since flow magnitude is proportional to 1/depth (and thus SNR),
                # minimizing this algebraic error is statistically superior for
                # homoscedastic pixel noise compared to normalized geometric error.
                base_residuals = (x - xf) * dv - (y - yf) * du

                # Add Regularization Prior towards Body Forward (Yaw/Sideslip = 0)
                # We assume the drone flies primarily forward relative to its body.
                # This corresponds to regularizing the horizontal component (x_f) towards the
                # projected Body Forward direction (self.prior_u).
                # We do NOT regularize the vertical component (y_f) as Angle of Attack varies significantly.

                prior_weight_yaw = 0.1
                prior_weight_pitch = 0.0 # Free pitch

                # Residuals for prior
                prior_x = prior_weight_yaw * (xf - self.prior_u)
                prior_y = prior_weight_pitch * (yf - self.prior_v)

                priors = np.array([prior_x, prior_y])

                return np.concatenate([base_residuals, priors])

            # Optimize using Least Squares with robust loss to handle outliers.
            # We use 'trf' or 'dogbox' because standard 'lm' implementation in scipy
            # does not support robust loss functions.
            # 'soft_l1' loss reduces the influence of outliers (large residuals).
            # f_scale=0.1 corresponds to a robust threshold of ~0.1 normalized units.

            opt_res = least_squares(
                algebraic_residuals,
                x0=[xf_init, yf_init],
                method='trf',
                loss='soft_l1', f_scale=0.1,
                ftol=1e-4, xtol=1e-4, gtol=1e-4,
                max_nfev=30
            )

            u_foe, v_foe = opt_res.x

            # Sanity Check / Clamping
            limit = 100.0
            norm = np.sqrt(u_foe**2 + v_foe**2)
            if norm > limit:
                scale = limit / norm
                u_foe *= scale
                v_foe *= scale

            # Filtering
            alpha = 0.2
            if self.filtered_foe_u is None:
                 self.filtered_foe_u = u_foe
                 self.filtered_foe_v = v_foe
            else:
                 self.filtered_foe_u = alpha * u_foe + (1 - alpha) * self.filtered_foe_u
                 self.filtered_foe_v = alpha * v_foe + (1 - alpha) * self.filtered_foe_v

            return (self.filtered_foe_u, self.filtered_foe_v), vel_est, vel_reliable

        except np.linalg.LinAlgError:
            return None, vel_est, vel_reliable
        except Exception as e:
            # Fallback or log error
            return None, vel_est, vel_reliable

    def _estimate_velocity(self, flow_vectors, drone_state, dt):
        """
        Estimates velocity (Vx, Vy, Vz) in NED Frame using flow vectors and altitude.
        Assumes flat ground at Z=0 (Altitude = -pz).

        Args:
            flow_vectors: List of (x, y, du, dv) tuples (displacement over dt).
            drone_state: dict with 'px', 'py', 'pz', 'roll', 'pitch', 'yaw'.
            dt: Time step in seconds.

        Returns:
            velocity_ned: dict {'vx': ..., 'vy': ..., 'vz': ...} or None if unreliable.
            reliable: bool
        """
        altitude = -drone_state['pz']
        if len(flow_vectors) < 5 or altitude < 0.5 or dt < 1e-4:
             return None, False

        # Calculate Rotation Matrix Body to World (NED)
        roll = drone_state['roll']
        pitch = drone_state['pitch']
        yaw = drone_state['yaw']

        cphi, sphi = np.cos(roll), np.sin(roll)
        ctheta, stheta = np.cos(pitch), np.sin(pitch)
        cpsi, spsi = np.cos(yaw), np.sin(yaw)

        # R_b2w construction
        r11 = ctheta * cpsi
        r12 = cpsi * sphi * stheta - cphi * spsi
        r13 = sphi * spsi + cphi * cpsi * stheta
        r21 = ctheta * spsi
        r22 = cphi * cpsi + sphi * spsi * stheta
        r23 = cphi * spsi * stheta - cpsi * sphi
        r31 = -stheta
        r32 = ctheta * sphi
        r33 = cphi * ctheta

        R_b2w = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])

        # Total Rotation Camera to World (NED)
        # R_c2w = R_b2w @ R_c2b
        R_total = R_b2w @ self.projector.R_c2b

        A = []
        b = []

        for x, y, udot, vdot in flow_vectors:
            # Normalized ray direction in Camera Frame: D_c = [x, y, 1.0]
            # Ray direction in World Frame: D_w = R_total @ D_c

            D_c = np.array([x, y, 1.0])
            D_w = R_total @ D_c

            denom = D_w[2] # Z component (Down)

            # If ray points up or horizontal, ignore (sky)
            if denom <= 0.01:
                continue

            z_c = altitude / denom

            # LS Equation for V_cam:
            # [1, 0, -x] * V = -z_c * udot
            # [0, 1, -y] * V = -z_c * vdot
            # where udot = du / dt

            udot = udot / dt
            vdot = vdot / dt

            A.append([1.0, 0.0, -x])
            b.append(-z_c * udot)

            A.append([0.0, 1.0, -y])
            b.append(-z_c * vdot)

        if len(A) < 10:
            return None, False

        A_mat = np.array(A)
        b_vec = np.array(b)

        try:
            res, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            V_cam = res # [Vx, Vy, Vz] in Camera Frame

            # Convert V_cam to V_ned
            # V_ned = R_total @ V_cam
            V_ned = R_total @ V_cam

            vel_dict = {
                'vx': V_ned[0],
                'vy': V_ned[1],
                'vz': V_ned[2]
            }
            return vel_dict, True

        except np.linalg.LinAlgError:
            return None, False

    def _get_terrain_height(self, x, y):
        """
        Returns the height of the terrain at (x, y) using a synthetic DTM function.
        Simulates smooth rolling terrain.
        """
        # Multi-frequency sine waves
        # Z is usually Down in NED, so height is negative Z.
        # But here let's assume 'height' implies altitude above sea level,
        # and ground is at some Z.
        # If NED: Z=0 is reference. Z < 0 is Above. Z > 0 is Below.
        # Let's say flat ground is Z=0. Hills go up (Z < 0) or valleys go down (Z > 0).

        # Function h(x,y) returns Z coordinate.

        # Scale ~ 50-100m wavelength, amplitude ~5-10m
        z = 5.0 * np.sin(x * 0.05) * np.cos(y * 0.05)
        z += 2.0 * np.sin(x * 0.15 + 1.0) * np.sin(y * 0.1)
        z += 10.0 * np.sin(x * 0.01) # Long slope

        return z

    def _generate_points(self, drone_state, count):
        """
        Generates 3D points on the synthetic terrain surface visible from the camera.
        Uses iterative raycasting.
        """
        pos = np.array([drone_state['px'], drone_state['py'], drone_state['pz']])

        roll = drone_state['roll']
        pitch = drone_state['pitch']
        yaw = drone_state['yaw']

        cphi, sphi = np.cos(roll), np.sin(roll)
        ctheta, stheta = np.cos(pitch), np.sin(pitch)
        cpsi, spsi = np.cos(yaw), np.sin(yaw)

        # R_b2w construction
        r11 = ctheta * cpsi
        r12 = cpsi * sphi * stheta - cphi * spsi
        r13 = sphi * spsi + cphi * cpsi * stheta

        r21 = ctheta * spsi
        r22 = cphi * cpsi + sphi * spsi * stheta
        r23 = cphi * spsi * stheta - cpsi * sphi

        r31 = -stheta
        r32 = ctheta * sphi
        r33 = cphi * ctheta

        R_b2w = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])

        # Camera to Body
        R_c2b = self.projector.R_c2b

        # Sample random pixels in the image
        u_samples = np.random.uniform(0, self.projector.width, size=count)
        v_samples = np.random.uniform(0, self.projector.height, size=count)

        # Unproject to Unit Vectors in World Frame
        # 1. Pixel to Camera Ray
        x_c = (u_samples - self.projector.cx) / self.projector.fx
        y_c = (v_samples - self.projector.cy) / self.projector.fy
        z_c = np.ones(count)

        vecs_c = np.vstack((x_c, y_c, z_c)) # 3 x N

        # 2. Camera to Body
        vecs_b = R_c2b @ vecs_c

        # 3. Body to World
        vecs_w = R_b2w @ vecs_b

        # Normalize vectors? Not strictly needed for raycasting if we use t parameter correctly,
        # but helpful for conditioning.
        norms = np.linalg.norm(vecs_w, axis=0)
        vecs_w = vecs_w / norms

        # Raycasting: P = O + t * D
        # Find t such that P_z = terrain_z(P_x, P_y)
        # O_z + t * D_z = h(O_x + t * D_x, O_y + t * D_y)

        # Initial guess: Intersect with Flat Earth (Z=0)
        # t0 = -O_z / D_z
        # If D_z is 0 or positive (pointing up/horizon), we might not hit ground.
        # Filter rays pointing up (D_z < 0 in NED means pointing UP).
        # Wait, Z is Down in NED. So D_z < 0 is Up. D_z > 0 is Down.
        # Ground is usually at Z >= 0 (or some positive value).
        # Drone is at -100 (if Z-Up sim) -> wait.
        # Let's check coordinates.
        # Sim uses Z-Up? 'theshow.py' says:
        # "Sim: X=East, Y=North, Z=Up"
        # "NED: X=North, Y=East, Z=Down"
        # The drone state passed here is NED (from 'theshow.py' conversion).
        # So Z is Down. Altitude 100m => Z = -100.
        # Ground is at Z = 0 (approximately).
        # So we are looking Down (Positive Z direction in NED).
        # So valid rays should have D_z > 0.

        # Filter valid rays
        valid_indices = vecs_w[2, :] > 0.1 # Must point somewhat down

        vecs_w_valid = vecs_w[:, valid_indices]
        if vecs_w_valid.shape[1] == 0:
            return

        # O = pos (3x1)
        O = pos.reshape(3, 1)

        # t0: Intersect with Z=0
        # O_z + t0 * D_z = 0 => t0 = -O_z / D_z
        # Since O_z is negative (high up), and D_z is positive (down), t0 is positive.
        t = -O[2] / vecs_w_valid[2, :]

        # Iteration
        for _ in range(3):
            P = O + vecs_w_valid * t # 3 x N_valid
            target_z = self._get_terrain_height(P[0, :], P[1, :]) # Vectorized

            # Error = current_z - target_z
            # We want current_z = target_z
            # O_z + t * D_z = target_z
            # t_new = (target_z - O_z) / D_z

            t = (target_z - O[2]) / vecs_w_valid[2, :]

        # Final Points
        P_final = O + vecs_w_valid * t

        # Store
        for i in range(P_final.shape[1]):
            self.world_points[self.next_id] = P_final[:, i]
            self.next_id += 1
