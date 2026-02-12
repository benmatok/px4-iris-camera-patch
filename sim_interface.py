import numpy as np
import logging
import cv2
import math

logger = logging.getLogger(__name__)

# --- INLINED SE3 MATH ---
def so3_exp(omega):
    theta_sq = np.dot(omega, omega)
    theta = math.sqrt(theta_sq)
    K = np.array([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0]
    ], dtype=np.float32)
    if theta < 1e-6:
        return np.eye(3, dtype=np.float32) + K + 0.5 * np.matmul(K, K)
    else:
        inv_theta = 1.0 / theta
        a = math.sin(theta) * inv_theta
        b = (1.0 - math.cos(theta)) * (inv_theta * inv_theta)
        return np.eye(3, dtype=np.float32) + a * K + b * np.matmul(K, K)

def rpy_to_matrix(roll, pitch, yaw):
    cr = math.cos(roll); sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw); sy = math.sin(yaw)
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr
    return R

def matrix_to_rpy(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw

# --- INLINED PYGHOSTMODEL ---
class PyGhostModel:
    def __init__(self, mass, drag, thrust_coeff, wind_x=0.0, wind_y=0.0, tau=0.1):
        self.mass = float(mass)
        self.drag_coeff = float(drag)
        self.thrust_coeff = float(thrust_coeff)
        self.wind_x = float(wind_x)
        self.wind_y = float(wind_y)
        self.tau = float(tau)
        self.G = 9.81
        self.MAX_THRUST_BASE = 20.0

    def step(self, state_dict, action_dict, dt):
        px, py, pz = state_dict['px'], state_dict['py'], state_dict['pz']
        vx, vy, vz = state_dict['vx'], state_dict['vy'], state_dict['vz']
        roll, pitch, yaw = state_dict['roll'], state_dict['pitch'], state_dict['yaw']
        wx = state_dict.get('wx', 0.0)
        wy = state_dict.get('wy', 0.0)
        wz = state_dict.get('wz', 0.0)

        thrust = action_dict['thrust']
        roll_rate_cmd = action_dict['roll_rate']
        pitch_rate_cmd = action_dict['pitch_rate']
        yaw_rate_cmd = action_dict['yaw_rate']

        alpha = dt / self.tau
        denom = 1.0 + alpha
        next_wx = (wx + roll_rate_cmd * alpha) / denom
        next_wy = (wy + pitch_rate_cmd * alpha) / denom
        next_wz = (wz + yaw_rate_cmd * alpha) / denom

        avg_wx = 0.5 * (wx + next_wx)
        avg_wy = 0.5 * (wy + next_wy)
        avg_wz = 0.5 * (wz + next_wz)

        R_curr = rpy_to_matrix(roll, -pitch, -yaw)
        omega_vec = np.array([avg_wx, avg_wy, avg_wz], dtype=np.float32) * dt
        R_update = so3_exp(omega_vec)
        R_next = np.matmul(R_curr, R_update)

        r, p, y = matrix_to_rpy(R_next)
        next_roll = r
        next_pitch = -p
        next_yaw = -y

        max_thrust = self.MAX_THRUST_BASE * self.thrust_coeff
        thrust_force = max(0.0, thrust * max_thrust)

        ax_dir = R_next[0, 2]
        ay_dir = R_next[1, 2]
        az_dir = R_next[2, 2]

        ax_thrust = thrust_force * ax_dir / self.mass
        ay_thrust = thrust_force * ay_dir / self.mass
        az_thrust = thrust_force * az_dir / self.mass

        ax_drag = -self.drag_coeff * (vx - self.wind_x)
        ay_drag = -self.drag_coeff * (vy - self.wind_y)
        az_drag = -self.drag_coeff * vz

        ax = ax_thrust + ax_drag
        ay = ay_thrust + ay_drag
        az = az_thrust + az_drag - self.G

        next_vx = vx + ax * dt
        next_vy = vy + ay * dt
        next_vz = vz + az * dt

        next_px = px + next_vx * dt
        next_py = py + next_vy * dt
        next_pz = pz + next_vz * dt

        return {
            'px': next_px, 'py': next_py, 'pz': next_pz,
            'vx': next_vx, 'vy': next_vy, 'vz': next_vz,
            'roll': next_roll, 'pitch': next_pitch, 'yaw': next_yaw,
            'wx': next_wx, 'wy': next_wy, 'wz': next_wz
        }

class SimDroneInterface:
    def __init__(self, projector):
        self.projector = projector
        self.mass = 1.0
        self.drag_coeff = 0.1
        self.thrust_coeff = 1.0
        self.tau = 0.1

        self.model = PyGhostModel(
            mass=self.mass,
            drag=self.drag_coeff,
            thrust_coeff=self.thrust_coeff,
            tau=self.tau,
            wind_x=0.0,
            wind_y=0.0
        )

        self.state = {
            'px': 0.0, 'py': 0.0, 'pz': 1.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'wx': 0.0, 'wy': 0.0, 'wz': 0.0
        }

        self.masses = [self.mass]
        self.thrust_coeffs = [self.thrust_coeff]
        self.dd = {'drag_coeffs': [self.drag_coeff]}

        logger.info("SimDroneInterface initialized with PyGhostModel (Inlined).")

    def reset_to_scenario(self, name, **kwargs):
        if name == "Blind Dive":
            self.state['px'] = kwargs.get('pos_x', 0.0)
            self.state['py'] = kwargs.get('pos_y', 0.0)
            self.state['pz'] = kwargs.get('pos_z', 100.0)
            self.state['vx'] = 0.0; self.state['vy'] = 0.0; self.state['vz'] = 0.0
            self.state['roll'] = 0.0
            self.state['pitch'] = kwargs.get('pitch', 0.0)
            self.state['yaw'] = kwargs.get('yaw', 0.0)
            self.state['wx'] = 0.0; self.state['wy'] = 0.0; self.state['wz'] = 0.0
            logger.info(f"Reset to Scenario: {name}")
        else:
            logger.warning(f"Unknown Scenario: {name}")

    def step(self, action):
        action_dict = {
            'thrust': float(action[0]),
            'roll_rate': float(action[1]),
            'pitch_rate': float(action[2]),
            'yaw_rate': float(action[3])
        }
        dt = 0.05
        self.state = self.model.step(self.state, action_dict, dt)

    def get_state(self):
        return self.state.copy()

    def get_image(self, target_pos_world):
        width = 640
        height = 480
        img = np.zeros((height, width, 3), dtype=np.uint8)
        s = self.state
        yaw_ned = (math.pi / 2.0) - s['yaw']
        yaw_ned = (yaw_ned + math.pi) % (2 * math.pi) - math.pi
        drone_state_ned = {
            'px': s['py'], 'py': s['px'], 'pz': -s['pz'],
            'roll': s['roll'], 'pitch': s['pitch'], 'yaw': yaw_ned
        }
        tx_sim, ty_sim, tz_sim = target_pos_world
        tx_ned = ty_sim; ty_ned = tx_sim; tz_ned = -tz_sim
        res = self.projector.project_point_with_size(tx_ned, ty_ned, tz_ned, drone_state_ned, object_radius=0.5)
        if res:
            u, v, r = res
            if 0 <= u < width and 0 <= v < height:
                draw_radius = max(2, int(r))
                cv2.circle(img, (int(u), int(v)), draw_radius, (0, 0, 255), -1)
        return img
