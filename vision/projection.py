import numpy as np

class Projector:
    def __init__(self, width=640, height=480, fov_deg=60.0, tilt_deg=0.0):
        self.width = width
        self.height = height
        self.cx = width / 2.0
        self.cy = height / 2.0

        # fx = fy assumption
        # tan(fov/2) = (w/2) / fx
        self.fx = (width / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
        self.fy = self.fx

        # Camera to Body Rotation
        # Base (Forward, No Tilt):
        # Cam: X(Right), Y(Down), Z(Forward)
        # Body: X(Forward), Y(Right), Z(Down)
        # Xb = Zc, Yb = Xc, Zb = Yc
        R_c2b_base = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)

        # Add Tilt (Pitch relative to Body)
        # Tilt around Body Y axis.
        # Positive Tilt = Nose Up (Standard Aerospace).
        # R_tilt = [ c 0 s; 0 1 0; -s 0 c ]
        theta = np.deg2rad(tilt_deg)
        ct, st = np.cos(theta), np.sin(theta)
        R_tilt = np.array([
            [ct, 0, st],
            [0, 1, 0],
            [-st, 0, ct]
        ])

        # R_c2b = R_tilt @ R_c2b_base
        # (Rotate the camera frame by tilt)
        self.R_c2b = R_tilt @ R_c2b_base

    def pixel_to_world(self, u, v, drone_state):
        """
        Projects a pixel to the ground plane (z=0).
        Args:
            u, v: Pixel coordinates
            drone_state: dict with keys 'px', 'py', 'pz', 'roll', 'pitch', 'yaw'
                         Positions in NED (m). Angles in radians (Standard Aerospace).
        Returns:
            (x, y, z): World coordinates of intersection with z=0.
            Returns None if no intersection (ray points up or parallel).
        """
        # 1. Ray in Camera Frame
        xc = (u - self.cx) / self.fx
        yc = (v - self.cy) / self.fy
        zc = 1.0
        vec_c = np.array([xc, yc, zc])

        # 2. Ray in Body Frame
        vec_b = self.R_c2b @ vec_c

        # 3. Ray in World Frame
        # Rotation Matrix Body to World (NED)
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        roll = drone_state['roll']
        pitch = drone_state['pitch']
        yaw = drone_state['yaw']

        cphi, sphi = np.cos(roll), np.sin(roll)
        ctheta, stheta = np.cos(pitch), np.sin(pitch)
        cpsi, spsi = np.cos(yaw), np.sin(yaw)

        # R_b2w construction
        # Row 1
        r11 = ctheta * cpsi
        r12 = cpsi * sphi * stheta - cphi * spsi
        r13 = sphi * spsi + cphi * cpsi * stheta

        # Row 2
        r21 = ctheta * spsi
        r22 = cphi * cpsi + sphi * spsi * stheta
        r23 = cphi * spsi * stheta - cpsi * sphi

        # Row 3
        r31 = -stheta
        r32 = ctheta * sphi
        r33 = cphi * ctheta

        R_b2w = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])

        vec_w = R_b2w @ vec_b

        # 4. Intersection with Z=0
        # P = P0 + t * V
        # Pz = P0z + t * Vz = 0 => t = -P0z / Vz

        p0 = np.array([drone_state['px'], drone_state['py'], drone_state['pz']])
        vz = vec_w[2]

        if abs(vz) < 1e-6:
            return None # Parallel

        t = -p0[2] / vz

        if t < 0:
            return None # Intersection behind camera

        intersection = p0 + t * vec_w
        return tuple(intersection)

    def world_to_pixel(self, x, y, z, drone_state):
        """
        Projects a world point to camera pixel coordinates.
        Args:
            x, y, z: World coordinates
            drone_state: dict with keys 'px', 'py', 'pz', 'roll', 'pitch', 'yaw'
        Returns:
            (u, v): Pixel coordinates
            Returns None if point is behind camera.
        """
        # 1. World to Body
        # P_b = R_b2w.T @ (P_w - P_drone)

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

        p_w = np.array([x, y, z])
        p_drone = np.array([drone_state['px'], drone_state['py'], drone_state['pz']])

        vec_w = p_w - p_drone
        vec_b = R_b2w.T @ vec_w

        # 2. Body to Camera
        # P_c = R_c2b.T @ P_b
        # self.R_c2b is C to B. So we need transpose.
        vec_c = self.R_c2b.T @ vec_b

        # 3. Project to Pixel
        xc, yc, zc = vec_c

        if zc <= 0:
            return None # Behind camera or on plane

        u = (xc / zc) * self.fx + self.cx
        v = (yc / zc) * self.fy + self.cy

        return (u, v)

    def project_point_with_size(self, x, y, z, drone_state, object_radius=0.5):
        """
        Projects a world point to camera pixel coordinates and estimates projected radius.
        Args:
            x, y, z: World coordinates
            drone_state: dict with keys 'px', 'py', 'pz', 'roll', 'pitch', 'yaw'
            object_radius: Radius of the object in world units (meters)
        Returns:
            (u, v, projected_radius_pixels): Pixel coordinates and radius
            Returns None if point is behind camera.
        """
        # 1. World to Body
        # P_b = R_b2w.T @ (P_w - P_drone)

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

        p_w = np.array([x, y, z])
        p_drone = np.array([drone_state['px'], drone_state['py'], drone_state['pz']])

        vec_w = p_w - p_drone
        vec_b = R_b2w.T @ vec_w

        # 2. Body to Camera
        # P_c = R_c2b.T @ P_b
        # self.R_c2b is C to B. So we need transpose.
        vec_c = self.R_c2b.T @ vec_b

        # 3. Project to Pixel
        xc, yc, zc = vec_c

        if zc <= 0:
            return None # Behind camera or on plane

        u = (xc / zc) * self.fx + self.cx
        v = (yc / zc) * self.fy + self.cy

        # Projected Radius = (Real Radius / Depth) * fx
        proj_radius = (object_radius / zc) * self.fx

        return (u, v, proj_radius)
