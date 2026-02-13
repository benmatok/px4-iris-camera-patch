import numpy as np

class FlowVelocityEstimator:
    def __init__(self, projector, num_points=200):
        self.projector = projector
        self.num_points = num_points
        self.world_points = {}  # id -> np.array([x, y, z])
        self.next_id = 0
        self.prev_projections = {}  # id -> (u_norm, v_norm)

        # Generation parameters
        self.gen_dist_min = 5.0
        self.gen_dist_max = 150.0 # Extended for DTM raycasting

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

        # Optimize: Iterate and filter
        # But world_points might grow large if we keep adding without deleting.
        # Strategy: Delete points that are far behind or too far away?

        points_to_remove = []

        for pid, pt in self.world_points.items():
            # Simple distance check to cull very far points
            # dist_sq = np.sum((pt - np.array([drone_state['px'], drone_state['py'], drone_state['pz']]))**2)
            # if dist_sq > (self.gen_dist_max * 2)**2:
            #    points_to_remove.append(pid)
            #    continue

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

        # 4. Estimate FOE
        if len(flow_vectors) < 10:
            return None

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
            return None

        A = np.array(A)
        b = np.array(b)

        try:
            # Least Squares
            res, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return (res[0], res[1])
        except np.linalg.LinAlgError:
            return None

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
