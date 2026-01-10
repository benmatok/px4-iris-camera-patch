import numpy as np

class LinearPlanner:
    """
    Plans a linear path to the target and outputs control actions (Thrust, Rates)
    to track it. Uses an Inverse Dynamics approach assuming constant velocity cruise.
    """
    def __init__(self, num_agents, dt=0.05):
        self.num_agents = num_agents
        self.dt = dt
        self.g = 9.81
        self.cruise_speed = 10.0 # m/s

    def compute_actions(self, current_state, target_pos):
        # Current State
        px = current_state['pos_x']
        py = current_state['pos_y']
        pz = current_state['pos_z']
        vx = current_state['vel_x']
        vy = current_state['vel_y']
        vz = current_state['vel_z']
        roll = current_state['roll']
        pitch = current_state['pitch']
        yaw = current_state['yaw']

        # Params
        mass = current_state['masses']
        drag = current_state['drag_coeffs']
        thrust_coeff = current_state['thrust_coeffs']
        max_thrust_force = 20.0 * thrust_coeff

        # Target Vector
        tx = target_pos[:, 0]
        ty = target_pos[:, 1]
        tz = target_pos[:, 2]

        dx = tx - px
        dy = ty - py
        dz = tz - pz
        dist_xy = np.sqrt(dx**2 + dy**2) + 1e-6

        # Elevation Angle Check
        rel_h = pz - tz
        elevation_rad = np.arctan2(rel_h, dist_xy)
        threshold_rad = np.deg2rad(10.0)

        # Virtual Target Logic
        target_z_eff = tz.copy()
        mask_low = elevation_rad < threshold_rad

        # For low agents, set target Z higher
        target_angle_rad = np.deg2rad(15.0)
        req_h = dist_xy * np.tan(target_angle_rad)

        target_z_eff[mask_low] = tz[mask_low] + req_h[mask_low]

        # Recalculate Delta with effective target
        dz_eff = target_z_eff - pz
        dist_eff = np.sqrt(dx**2 + dy**2 + dz_eff**2) + 1e-6

        # Desired Velocity (Linear Cruise)
        speed_ref = np.minimum(self.cruise_speed, dist_eff * 1.0)

        vx_des = (dx / dist_eff) * speed_ref
        vy_des = (dy / dist_eff) * speed_ref
        vz_des = (dz_eff / dist_eff) * speed_ref

        # Velocity Error
        evx = vx_des - vx
        evy = vy_des - vy
        evz = vz_des - vz

        # PID for Acceleration Command
        Kp = 2.0
        ax_cmd = Kp * evx
        ay_cmd = Kp * evy
        az_cmd = Kp * evz

        # Inverse Dynamics to get Thrust Vector
        Fx_req = mass * ax_cmd + drag * vx
        Fy_req = mass * ay_cmd + drag * vy
        Fz_req = mass * az_cmd + drag * vz + mass * self.g

        # Compute Thrust Magnitude
        F_total = np.sqrt(Fx_req**2 + Fy_req**2 + Fz_req**2) + 1e-6
        thrust_cmd = F_total / max_thrust_force
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)

        # Compute Desired Attitude (Z-axis alignment)
        zbx = Fx_req / F_total
        zby = Fy_req / F_total
        zbz = Fz_req / F_total

        # Yaw Alignment: Point nose at target (xy plane)
        yaw_des = np.arctan2(dy, dx)

        xb_temp_x = np.cos(yaw_des)
        xb_temp_y = np.sin(yaw_des)
        xb_temp_z = np.zeros_like(yaw_des)

        # yb = cross(zb, xb_temp)
        yb_x = zby * xb_temp_z - zbz * xb_temp_y
        yb_y = zbz * xb_temp_x - zbx * xb_temp_z
        yb_z = zbx * xb_temp_y - zby * xb_temp_x

        norm_yb = np.sqrt(yb_x**2 + yb_y**2 + yb_z**2) + 1e-6
        yb_x /= norm_yb
        yb_y /= norm_yb
        yb_z /= norm_yb

        # xb = cross(yb, zb)
        xb_x = yb_y * zbz - yb_z * zby
        xb_y = yb_z * zbx - yb_x * zbz
        xb_z = yb_x * zby - yb_y * zbx

        # Extract Roll/Pitch
        pitch_des = -np.arcsin(np.clip(xb_z, -1.0, 1.0))
        roll_des = np.arctan2(yb_z, zbz)

        # Rate P-Controller
        Kp_att = 5.0

        roll_err = roll_des - roll
        roll_err = (roll_err + np.pi) % (2 * np.pi) - np.pi

        pitch_err = pitch_des - pitch
        pitch_err = (pitch_err + np.pi) % (2 * np.pi) - np.pi

        yaw_err = yaw_des - yaw
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi

        roll_rate_cmd = Kp_att * roll_err
        pitch_rate_cmd = Kp_att * pitch_err
        yaw_rate_cmd = Kp_att * yaw_err

        actions = np.zeros((self.num_agents, 4))
        actions[:, 0] = thrust_cmd
        actions[:, 1] = np.clip(roll_rate_cmd, -10.0, 10.0)
        actions[:, 2] = np.clip(pitch_rate_cmd, -10.0, 10.0)
        actions[:, 3] = np.clip(yaw_rate_cmd, -10.0, 10.0)

        return actions
