import numpy as np
import asyncio
import time
import os
from datetime import datetime

# Optional Dependencies for Control
try:
    from mavsdk import System
    from mavsdk.offboard import OffboardError, VelocityBodyYawspeed, Attitude, AttitudeRate
    HAS_MAVSDK = True
except ImportError:
    HAS_MAVSDK = False
    # Dummy classes for MAVSDK types
    class VelocityBodyYawspeed:
        def __init__(self, *args): pass
    class Attitude:
        def __init__(self, *args): pass
    class AttitudeRate:
        def __init__(self, *args): pass
    class OffboardError(Exception): pass

# --- Helper Functions ---

def get_rotation_body_to_world(roll_rad, pitch_rad, yaw_rad):
    cr = np.cos(roll_rad)
    sr = np.sin(roll_rad)
    cp = np.cos(pitch_rad)
    sp = np.sin(pitch_rad)
    cy = np.cos(yaw_rad)
    sy = np.sin(yaw_rad)
    R_body_to_world = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R_body_to_world

def estimate_body_velocities(thrust_normalized, attitude_rad, mass_kg, stable_thrust):
    g = 9.81
    roll, pitch, yaw = attitude_rad
    effective_g = (thrust_normalized / stable_thrust) * g
    v_x = effective_g * np.tan(-pitch)
    v_y = effective_g * np.tan(roll)
    v_z = effective_g * np.cos(pitch) * np.cos(roll)
    return v_x, v_y, v_z

def velocity_to_attitude(vx_des_body, vy_des_body, vz_des_body, yaw_rate_des_body, current_yaw_deg_world, dt, drone_state):
    g = 9.81
    stable_thrust = drone_state.stable_thrust
    mass_kg = drone_state.drone_mass
    tau = 5.0 # Time constant
    attitude_rad = (drone_state.current_roll_rad, drone_state.current_pitch_rad, np.deg2rad(current_yaw_deg_world))
    thrust_normalized = drone_state.current_thrust
    v_est_x_body, v_est_y_body, v_est_z_body = estimate_body_velocities(thrust_normalized, attitude_rad, mass_kg, stable_thrust)

    a_body = np.array([(vx_des_body - v_est_x_body) / tau, (vy_des_body - v_est_y_body) / tau, (vz_des_body - v_est_z_body)/tau ])
    g_world = np.array([0, 0, g])
    R_body_to_world = get_rotation_body_to_world(drone_state.current_roll_rad,drone_state.current_pitch_rad,np.deg2rad(current_yaw_deg_world))
    g_body = R_body_to_world.T @ g_world
    t_des_body = mass_kg * (a_body + g_body)
    T_mag = np.linalg.norm(t_des_body)

    if T_mag == 0:
        return 0.0, 0.0, current_yaw_deg_world, stable_thrust

    z_des_body = t_des_body / T_mag
    pitch_target_world_deg = np.degrees(np.arctan2(-z_des_body[0], z_des_body[2]))*0.8
    roll_target_world_deg = np.degrees(np.arcsin(z_des_body[1]))

    roll_target_world_deg = np.clip(roll_target_world_deg,-5,5)*0.0
    pitch_target_world_deg = np.clip(pitch_target_world_deg,-15,15)

    max_force = mass_kg * g / stable_thrust
    thrust = T_mag / max_force
    thrust = np.clip(thrust, 0.1, 0.9)
    yaw_target_world_deg = current_yaw_deg_world + yaw_rate_des_body * dt

    return roll_target_world_deg, pitch_target_world_deg, yaw_target_world_deg, thrust

def calc_pursuit_velocities(pursuit_state, drone_state, bbox_center, frame_width, frame_height):
    if bbox_center is None:
        pursuit_state.stable_count = 0
        return 0.0, 0.0, 0.0, 0.0
    ref_x = frame_width // 2
    ref_y = int(frame_height * (1 - pursuit_state.target_ratio))
    bbox_x, bbox_y = bbox_center
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    hfov_deg = 110.0
    vfov_deg = 90.0

    hfov_rad = np.deg2rad(hfov_deg)
    vfov_rad = np.deg2rad(vfov_deg)

    fx = (frame_width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = (frame_height / 2.0) / np.tan(vfov_rad / 2.0)

    u = (bbox_x - cx) / fx
    v = -(bbox_y - cy) / fy
    u_des = (ref_x - cx) / fx
    v_des = -(ref_y - cy) / fy
    angle_x = np.arctan(u)
    angle_y = np.arctan(v)
    angle_x_des = np.arctan(u_des)
    angle_y_des = np.arctan(v_des)
    error_angle_x = angle_x - angle_x_des
    error_angle_y = angle_y - angle_y_des
    error_angle_x_deg = np.rad2deg(error_angle_x)
    error_angle_y_deg = np.rad2deg(error_angle_y)
    yaw_rate = error_angle_x_deg
    current_attitude = (drone_state.current_roll_rad,drone_state.current_pitch_rad,np.deg2rad(drone_state.current_yaw_deg))
    est_vx,est_vy,est_vz = estimate_body_velocities(drone_state.current_thrust,current_attitude,drone_state.drone_mass,drone_state.stable_thrust)
    throttle = 0.8
    target_speed = pursuit_state.forward_velocity * throttle
    target_speed = max(target_speed, 3.0)
    if np.abs(error_angle_x_deg) < pursuit_state.alignment_threshold_deg:
        pursuit_state.stable_count += 1
    else:
        pursuit_state.stable_count = 0
    camera_mount_pitch_deg = 30.0
    object_vertical_angle_camera_deg = np.rad2deg(angle_y)
    object_vertical_angle_body_deg = object_vertical_angle_camera_deg + camera_mount_pitch_deg
    drone_pitch_deg = np.rad2deg(drone_state.current_pitch_rad)
    object_attack_angle_world_deg = object_vertical_angle_body_deg + drone_pitch_deg
    target_attack_angle_world_deg = -20

    if pursuit_state.stable_count == 1:
        pursuit_state.initial_attack_angle = object_attack_angle_world_deg
    target_attack_deg = max(pursuit_state.initial_attack_angle, -20.0) if pursuit_state.stable_count > 0 else -20.0
    if pursuit_state.stable_count > pursuit_state.stable_threshold:
        vx = pursuit_state.pursuit_pid_forward.update(pursuit_state.forward_velocity - est_vx)
        vz = pursuit_state.pursuit_pid_alt.update(pursuit_state.initial_attack_angle - object_attack_angle_world_deg)
        vz = np.clip(vz, -pursuit_state.pursuit_pid_alt.output_limit, pursuit_state.pursuit_pid_alt.output_limit)
    else:
        vz = 0.01
        vx = 0.01

    vy = 0.0
    if abs(vz) > pursuit_state.max_linear_speed:
        vz = pursuit_state.max_linear_speed*vz/abs(vz)

    return vx, vy, vz, yaw_rate

# --- Mock Classes (Needed for DroneState) ---

class MockDrone:
    class MockOffboard:
        async def set_velocity_body(self, *args): pass
        async def set_attitude(self, *args): pass
        async def set_attitude_rate(self, *args): pass
        async def start(self): pass
        async def stop(self): pass

    class MockAction:
        async def arm(self): pass
        async def land(self): pass
        async def disarm(self): pass
        async def hold(self): pass

    class MockTelemetry:
        async def attitude_euler(self):
            class Euler:
                roll_deg=0.0
                pitch_deg=0.0
                yaw_deg=0.0
            while True:
                yield Euler()
                await asyncio.sleep(0.1)

        async def velocity_ned(self):
            class Velocity:
                north_m_s=0.0
                east_m_s=0.0
                down_m_s=0.0
            while True:
                yield Velocity()
                await asyncio.sleep(0.1)

        async def attitude_angular_velocity_body(self):
            class Angular:
                yaw_rad_s=0.0
            while True:
                yield Angular()
                await asyncio.sleep(0.1)

        async def actuator_output_status(self, group):
            class Status:
                actuator=[0.0]*4
            while True:
                yield Status()
                await asyncio.sleep(0.1)

    class MockCore:
        async def connection_state(self):
            class State:
                is_connected = True
            yield State()

    def __init__(self):
        self.offboard = self.MockOffboard()
        self.action = self.MockAction()
        self.telemetry = self.MockTelemetry()
        self.core = self.MockCore()

    async def connect(self, system_address=None):
        pass

# --- Classes for State and Logic ---

class PIDController:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, output_limit=0.0, smoothing_factor=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.smoothing_factor = smoothing_factor
        self.integral = 0.0
        self.previous_error = 0.0
        self.smoothed_output = 0.0
    def update(self, error):
        self.integral += error
        derivative = error - self.previous_error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        if self.output_limit > 0:
            if output > self.output_limit:
                output = self.output_limit
                self.integral -= error
            elif output < -self.output_limit:
                output = -self.output_limit
                self.integral -= error
        if self.smoothing_factor > 0:
            output = (self.smoothing_factor * output) + ((1 - self.smoothing_factor) * self.smoothed_output)
            self.smoothed_output = output
        self.previous_error = error
        return output
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.smoothed_output = 0.0

class DualSenseConfig:
    def __init__(self):
        self.MAX_VELOCITY = 30.0
        self.MAX_CLIMB = 5.0
        self.MAX_DESCENT = 25.0
        self.MAX_YAW_RATE = 90.0
        self.GAMMA = 3.0
        self.SHOW_VIDEO = True
        self.PUBLISH_VIDEO = False
        self.RECORD_VIDEO = False

class DroneState:
    def __init__(self):
        self.drone = System() if HAS_MAVSDK else MockDrone()
        self.is_armed = False
        self.is_offboard = False
        self.stable_thrust = 0.25
        self.drone_mass = 2.8
        self.current_pitch_rad = 0.0
        self.current_roll_rad = 0.0
        self.current_yaw_deg = 0.0
        self.current_body_velocities = {'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'vyaw': 0.0}
        self.current_thrust = 0.0

class PursuitState:
    def __init__(self):
        self.autonomous_pursuit_active = False
        self.pursuit_pid_yaw = PIDController(kp=0.5, ki=0.005, kd=0.02, output_limit=25.0)
        vx_p = 0.02
        vx_i = 0.00015
        vx_d = 0.00125
        self.pursuit_pid_alt = PIDController(kp=vx_p*40, ki=vx_i*40, kd=vx_d*40, output_limit=25.0)
        self.pursuit_pid_forward = PIDController(kp=vx_p, ki=vx_i, kd=vx_d, output_limit=25.0, smoothing_factor=0.1)
        self.target_ratio = 0.2
        self.forward_velocity = 25.0
        self.pursuit_debug_info = {}
        self.stable_count = 0
        self.stable_threshold = 30
        self.alignment_threshold_deg = 5.0
        self.down_movement_threshold_deg = 5.0
        self.max_linear_speed = 2.0
        self.initial_attack_angle = -20.0

class FrameState:
    def __init__(self):
        self.current_frame = None
        self.current_frame_width = 1280
        self.current_frame_height = 800
        self.current_bbox_center = None
        self.current_bbox_is_large = False
        self.bbox = None
        self.bbox_mode = False
        self.tracking_mode = False
        self.tracker_instance = None
        self.bbox_size_mode = False
        self.show_stabilized = True
        self.init_tracking = False
        self.last_success_frame = None
        self.last_success_bbox = None

class ButtonState:
    def __init__(self):
        self.prev_circle_state = False
        self.prev_square_state = False
        self.prev_cross_state = False
        self.prev_triangle_state = False
        self.prev_o_state = False
        self.prev_r1_state = False
        self.prev_r2_state = False

class VideoState:
    def __init__(self):
        self.video_writer = None
        self.video_filename = None

class CacheState:
    def __init__(self):
        self.cache_dir = os.path.expanduser("~/cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_frame_path = os.path.join(self.cache_dir, "roi_frame.png")
        self.cache_bbox_path = os.path.join(self.cache_dir, "roi_bbox.pkl")

class InputState:
    def __init__(self, ds_state=None):
        if ds_state:
            self.LX = ds_state.LX
            self.LY = ds_state.LY
            self.RX = ds_state.RX
            self.RY = ds_state.RY
            self.L1 = ds_state.L1
            self.L2Btn = ds_state.L2Btn
            self.R1 = ds_state.R1
            self.R2Btn = ds_state.R2Btn
            self.square = ds_state.square
            self.cross = ds_state.cross
            self.circle = ds_state.circle
            self.triangle = ds_state.triangle
            self.DpadUp = ds_state.DpadUp
            self.DpadDown = ds_state.DpadDown
            self.DpadLeft = ds_state.DpadLeft
            self.DpadRight = ds_state.DpadRight
        else:
            self.LX = 0; self.LY = 0; self.RX = 0; self.RY = 0
            self.L1 = False; self.L2Btn = False; self.R1 = False; self.R2Btn = False
            self.square = False; self.cross = False; self.circle = False; self.triangle = False
            self.DpadUp = False; self.DpadDown = False; self.DpadLeft = False; self.DpadRight = False

class ControlMode:
    def get_setpoint(self, input_state, config, pursuit_state, frame_state, drone_state):
        raise NotImplementedError("Subclasses must implement get_setpoint")

class ManualMode(ControlMode):
    def get_setpoint(self, input_state, config, pursuit_state, frame_state, drone_state):
        if pursuit_state.autonomous_pursuit_active:
            if frame_state.current_bbox_center:
                vx, vy, vz, yaw_rate = calc_pursuit_velocities(pursuit_state, drone_state, frame_state.current_bbox_center, frame_state.current_frame_width, frame_state.current_frame_height)
                return 'velocity', (vx, vy, vz, yaw_rate)
            return 'velocity', (0.0, 0.0, 0.0, 0.0)
        lx = input_state.LX / 128.0
        ly = input_state.LY / 128.0
        rx = input_state.RX / 128.0
        ry = input_state.RY / 128.0

        lx = np.sign(lx) * (abs(lx) ** config.GAMMA)
        ly = np.sign(ly) * (abs(ly) ** config.GAMMA)
        rx = np.sign(rx) * (abs(rx) ** config.GAMMA)
        ry = np.sign(ry) * (abs(ry) ** config.GAMMA)
        vx = -ly * config.MAX_VELOCITY
        vy = lx * config.MAX_VELOCITY
        vz = ry * (config.MAX_CLIMB if ry < 0 else config.MAX_DESCENT) if abs(ry) > 0 else 0.0
        yaw_rate = rx * config.MAX_YAW_RATE
        return 'velocity', (vx, vy, vz, yaw_rate)

class AngleMode(ControlMode):
    def get_setpoint(self, input_state, config, pursuit_state, frame_state, drone_state):
        if pursuit_state.autonomous_pursuit_active:
            if frame_state.current_bbox_center:
                vx, vy, vz, yaw_rate = calc_pursuit_velocities(pursuit_state, drone_state, frame_state.current_bbox_center, frame_state.current_frame_width, frame_state.current_frame_height)
            else:
                vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0
        else:
            lx = input_state.LX / 128.0
            ly = input_state.LY / 128.0
            rx = input_state.RX / 128.0
            ry = input_state.RY / 128.0
            lx = np.sign(lx) * (abs(lx) ** config.GAMMA)
            ly = np.sign(ly) * (abs(ly) ** config.GAMMA)
            rx = np.sign(rx) * (abs(rx) ** config.GAMMA)
            ry = np.sign(ry) * (abs(ry) ** config.GAMMA)
            vx = -ly * config.MAX_VELOCITY
            vy = lx * config.MAX_VELOCITY
            vz = ry * (config.MAX_CLIMB if ry < 0 else config.MAX_DESCENT) if abs(ry) > 0 else 0.0
            yaw_rate = rx * config.MAX_YAW_RATE

        dt = 1.0
        roll_target, pitch_target, yaw_target, thrust = velocity_to_attitude(
            vx, vy, -vz, yaw_rate, drone_state.current_yaw_deg, dt, drone_state
        )
        return 'attitude', (roll_target, pitch_target, yaw_target, thrust)

# --- Logic Functions ---

def handle_input_logic(input_state, button_state, frame_state, pursuit_state, config, control_mode):
    # Toggle Stabilization (Circle)
    if input_state.circle and not button_state.prev_o_state:
        frame_state.show_stabilized = not frame_state.show_stabilized
        if frame_state.bbox_mode and not pursuit_state.autonomous_pursuit_active:
             frame_state.bbox_size_mode = not frame_state.bbox_size_mode
    button_state.prev_o_state = input_state.circle

    # Control Mode Toggle (R1/R2)
    if input_state.R1 and not button_state.prev_r1_state:
        control_mode.__class__ = AngleMode
    if input_state.R2Btn and not button_state.prev_r2_state:
        control_mode.__class__ = ManualMode
    button_state.prev_r1_state = input_state.R1
    button_state.prev_r2_state = input_state.R2Btn

    # Toggle BBox Mode (Square)
    if input_state.square and not button_state.prev_square_state:
        if not pursuit_state.autonomous_pursuit_active:
            frame_state.bbox_mode = not frame_state.bbox_mode
            frame_state.tracking_mode = False
            frame_state.bbox_size_mode = False
    button_state.prev_square_state = input_state.square

    # Toggle Tracking (Cross)
    if input_state.cross and not button_state.prev_cross_state:
        frame_state.tracking_mode = not frame_state.tracking_mode
        if frame_state.tracking_mode:
            frame_state.init_tracking = True
            frame_state.bbox_mode = False
        else:
            frame_state.tracker_instance = None
    button_state.prev_cross_state = input_state.cross

    # Toggle Pursuit (Triangle)
    if input_state.triangle and not button_state.prev_triangle_state:
        pursuit_state.autonomous_pursuit_active = not pursuit_state.autonomous_pursuit_active
        if pursuit_state.autonomous_pursuit_active:
            pursuit_state.pursuit_pid_yaw.reset()
            pursuit_state.pursuit_pid_alt.reset()
            pursuit_state.pursuit_pid_forward.reset()
    button_state.prev_triangle_state = input_state.triangle

def handle_bbox_controls(frame_state, input_state, pursuit_state, width, height):
    if not frame_state.bbox_mode or pursuit_state.autonomous_pursuit_active:
        return
    move_step = 5
    size_step = 10
    if frame_state.bbox_size_mode:
        x, y, w, h = frame_state.bbox
        center_x = x + w // 2
        center_y = y + h // 2
        if input_state.DpadUp:
            h = max(20, h - size_step)
            y = center_y - h // 2
        if input_state.DpadDown:
            h = min(height - 20, h + size_step)
            y = center_y - h // 2
        if input_state.DpadLeft:
            w = max(20, w - size_step)
            x = center_x - w // 2
        if input_state.DpadRight:
            w = min(width - 20, w + size_step)
            x = center_x - w // 2
        frame_state.bbox = [x, y, w, h]
        frame_state.bbox[0] = max(0, min(frame_state.bbox[0], width - frame_state.bbox[2]))
        frame_state.bbox[1] = max(0, min(frame_state.bbox[1], height - frame_state.bbox[3]))
    else:
        x, y, w, h = frame_state.bbox
        if input_state.DpadUp:
            y = max(0, y - move_step)
        if input_state.DpadDown:
            y = min(height - h, y + move_step)
        if input_state.DpadLeft:
            x = max(0, x - move_step)
        if input_state.DpadRight:
            x = min(width - w, x + move_step)
        frame_state.bbox = [x, y, w, h]
