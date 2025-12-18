import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import threading
import queue
import time
import copy
import asyncio
import os
from datetime import datetime
from cv_bridge import CvBridge, CvBridgeError
from pydualsense import *
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed, Attitude, AttitudeRate

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

def stabilize_frame(frame, pitch_rad, roll_rad, pitch0_rad=0.0, hfov_deg=110.0):
    height, width = frame.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), np.rad2deg(roll_rad), 1.0)
    stabilized = cv2.warpAffine(frame, rotation_matrix, (width, height))
    shift_y = int(height * (pitch_rad - pitch0_rad) / np.deg2rad(hfov_deg / 2))
    M = np.float32([[1, 0, 0], [0, 1, shift_y]])
    stabilized = cv2.warpAffine(stabilized, M, (width, height))
    return stabilized

def draw_alignment_info(frame, pursuit_state, self_obj, target_ratio=0.4):
    if self_obj.current_bbox_center is None:
        return frame
    height, width = frame.shape[:2]
    ref_x = self_obj.frame_center_x
    ref_y = int(height * (1 - pursuit_state.target_ratio))
    cv2.line(frame, (ref_x, 0), (ref_x, height), (255, 0, 0), 1)
    cv2.line(frame, (0, ref_y), (width, ref_y), (255, 0, 0), 1)
    cv2.circle(frame, self_obj.current_bbox_center, 5, (0, 255, 0), -1)
    cv2.putText(frame, "Aligned", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def draw_control_mode(frame, control_mode):
    mode_name = "Manual" if isinstance(control_mode, ManualMode) else "Angle" if isinstance(control_mode, AngleMode) else "Acro"
    cv2.putText(frame, f"Mode: {mode_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def compute_homography(last_frame, current_frame):
    orb = cv2.ORB_create()
    gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray_last, None)
    kp2, des2 = orb.detectAndCompute(gray_current, None)
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return None, False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        return None, False
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, False
    return H, True

def warp_bbox(H, last_bbox):
    x, y, w, h = last_bbox
    pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    warped_pts = cv2.perspectiveTransform(pts, H)
    x_new = int(min(warped_pts[:, 0, 0]))
    y_new = int(min(warped_pts[:, 0, 1]))
    w_new = int(max(warped_pts[:, 0, 0]) - x_new)
    h_new = int(max(warped_pts[:, 0, 1]) - y_new)
    if w_new <= 0 or h_new <= 0:
        return None, False
    new_bbox = [int(x_new), int(y_new), int(w_new), int(h_new)]
    return new_bbox, True

def megatrack():
    return cv2.TrackerCSRT_create()

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
        self.drone = System()
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

# Data structure to hold DualSense state in the queue/processing
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

# --- Processing Functions ---

def process_visuals(config, frame_state, button_state, video_state, cache_state, input_state, drone_state, pursuit_state, control_mode, logger):
    frame = frame_state.current_frame.copy()
    height, width = frame.shape[:2]
    frame_state.current_frame_height, frame_state.current_frame_width = height, width

    display_frame = get_stabilized_frame(frame, frame_state, drone_state)
    if len(display_frame.shape) == 2:
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

    handle_video_recording(config, video_state, display_frame, width, height, cache_state)
    handle_modes(frame_state, drone_state, button_state, pursuit_state, input_state, display_frame, width, height)
    handle_pursuit_alignment(pursuit_state, drone_state, frame_state, display_frame)
    display_frame = draw_control_mode(display_frame, control_mode)
    show_video(config, display_frame, logger)

def handle_modes(frame_state, drone_state, button_state, pursuit_state, input_state, display_frame, width, height):
    bbox_size = 50
    if frame_state.bbox is None:
        frame_state.bbox = [width//2 - bbox_size//2, height//2 - bbox_size//2, bbox_size, bbox_size]

    handle_bbox_controls(frame_state, input_state, pursuit_state, width, height)

    if pursuit_state.autonomous_pursuit_active:
        process_pursuit_mode(frame_state, drone_state, pursuit_state, display_frame)
    elif frame_state.tracking_mode:
        process_tracking_mode(frame_state, display_frame)
    elif frame_state.bbox_mode:
        process_bbox_mode(frame_state, display_frame)

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

def process_bbox_mode(frame_state, display_frame):
    x, y, w, h = frame_state.bbox
    large_threshold = 0.1 * display_frame.shape[0]
    frame_state.current_bbox_is_large = w > large_threshold or h > large_threshold
    bbox_center_x = x + w // 2
    if frame_state.current_bbox_is_large:
        bbox_center_y = int(y * 0.25 + (y + h) * 0.75)
    else:
        bbox_center_y = y + h // 2
    bbox_center_local = (bbox_center_x, bbox_center_y)
    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    frame_state.current_bbox_center = bbox_center_local

def process_tracking_mode(frame_state, display_frame):
    color_frame = cv2.cvtColor(frame_state.current_frame, cv2.COLOR_GRAY2BGR)
    if frame_state.init_tracking:
        if frame_state.tracker_instance is None:
            frame_state.tracker_instance = megatrack()
        success = frame_state.tracker_instance.init(color_frame, tuple(frame_state.bbox))
        frame_state.init_tracking = False
        if success:
            frame_state.last_success_frame = color_frame.copy()
            frame_state.last_success_bbox = frame_state.bbox[:]
        else:
            frame_state.tracking_mode = False
            frame_state.tracker_instance = None
            return
    if frame_state.tracker_instance:
        success, bbox = frame_state.tracker_instance.update(color_frame)
        if success:
            frame_state.bbox = [int(v) for v in bbox]
            frame_state.last_success_frame = color_frame.copy()
            frame_state.last_success_bbox = frame_state.bbox[:]
        else:
            whole_process_success = False
            if frame_state.last_success_frame is not None:
                H, hom_success = compute_homography(frame_state.last_success_frame, color_frame)
                if hom_success:
                    new_bbox, warp_success = warp_bbox(H, frame_state.last_success_bbox)
                    if warp_success:
                        frame_state.tracker_instance = megatrack()
                        whole_process_success = frame_state.tracker_instance.init(color_frame, tuple(new_bbox))
                        if whole_process_success:
                            frame_state.bbox = new_bbox
                            frame_state.last_success_frame = color_frame.copy()
                            frame_state.last_success_bbox = new_bbox[:]
            if not whole_process_success:
                frame_state.tracking_mode = False
                frame_state.tracker_instance = None
    if frame_state.bbox:
        x, y, w, h = frame_state.bbox
        large_threshold = 0.1 * display_frame.shape[0]
        frame_state.current_bbox_is_large = w > large_threshold or h > large_threshold
        bbox_center_x = x + w // 2
        if frame_state.current_bbox_is_large:
            bbox_center_y = int(y * 0.25 + (y + h) * 0.75)
        else:
            bbox_center_y = y + h // 2
        bbox_center_local = (bbox_center_x, bbox_center_y)
        bbox_color = (128, 0, 255) if frame_state.current_bbox_is_large else (0, 0, 255)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), bbox_color, 2)
        frame_state.current_bbox_center = bbox_center_local

def process_pursuit_mode(frame_state, drone_state, pursuit_state, display_frame):
    color_frame = cv2.cvtColor(frame_state.current_frame, cv2.COLOR_GRAY2BGR)
    if frame_state.tracker_instance is None:
        frame_state.tracker_instance = megatrack()
        success = frame_state.tracker_instance.init(color_frame, tuple(frame_state.bbox))
        if success:
            frame_state.last_success_frame = color_frame.copy()
            frame_state.last_success_bbox = frame_state.bbox[:]
        else:
            pursuit_state.autonomous_pursuit_active = False
            return
    else:
        success, bbox = frame_state.tracker_instance.update(color_frame)
        if success:
            frame_state.bbox = [int(v) for v in bbox]
            frame_state.last_success_frame = color_frame.copy()
            frame_state.last_success_bbox = frame_state.bbox[:]
        else:
            whole_process_success = False
            if frame_state.last_success_frame is not None:
                H, hom_success = compute_homography(frame_state.last_success_frame, color_frame)
                if hom_success:
                    new_bbox, warp_success = warp_bbox(H, frame_state.last_success_bbox)
                    if warp_success:
                        frame_state.tracker_instance = megatrack()
                        whole_process_success = frame_state.tracker_instance.init(color_frame, tuple(new_bbox))
                        if whole_process_success:
                            frame_state.bbox = new_bbox
                            frame_state.last_success_frame = color_frame.copy()
                            frame_state.last_success_bbox = new_bbox[:]
            if not whole_process_success:
                pursuit_state.autonomous_pursuit_active = False
                frame_state.tracker_instance = None
                return
    if frame_state.bbox:
        x, y, w, h = frame_state.bbox
        large_threshold = 0.1 * display_frame.shape[0]
        frame_state.current_bbox_is_large = w > large_threshold or h > large_threshold
        bbox_center_x = x + w // 2
        if frame_state.current_bbox_is_large:
            bbox_center_y = int(y * 0.25 + (y + h) * 0.75)
        else:
            bbox_center_y = y + h // 2
        bbox_center_local = (bbox_center_x, bbox_center_y)
        bbox_color = (128, 0, 255) if frame_state.current_bbox_is_large else (255, 0, 0)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), bbox_color, 3)
        cv2.circle(display_frame, bbox_center_local, 3, (255, 255, 255), -1)
        frame_state.current_bbox_center = bbox_center_local

def handle_pursuit_alignment(pursuit_state, drone_state, frame_state, display_frame):
    if not pursuit_state.autonomous_pursuit_active or None == frame_state.current_bbox_center:
        return
    class MockSelf:
        frame_width = display_frame.shape[1]
        frame_height = display_frame.shape[0]
        frame_center_x = display_frame.shape[1] // 2
        frame_center_y = display_frame.shape[0] // 2
        current_bbox_center = frame_state.current_bbox_center
        current_velocities = (0, 0, 0, 0)
        center_proximity_threshold = 0.5
        landing_proximity_threshold = 0.2
        current_pitch_rad = drone_state.current_pitch_rad
        current_roll_rad = drone_state.current_roll_rad
    display_frame = draw_alignment_info(display_frame, pursuit_state, MockSelf())

def show_video(config, display_frame, logger):
    if not config.SHOW_VIDEO:
        return
    cv2.imshow('Forward Camera Feed', display_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        logger.info('Shutting down')
        rclpy.shutdown()

def handle_video_recording(config, video_state, display_frame, width, height, cache_state):
    if config.RECORD_VIDEO and video_state.video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_state.video_filename = os.path.join(cache_state.cache_dir, f"stabilized_video_{timestamp}.mp4")
        video_state.video_writer = cv2.VideoWriter(video_state.video_filename, fourcc, 30, (width, height))
    if config.RECORD_VIDEO and video_state.video_writer:
        video_state.video_writer.write(display_frame)

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

def get_stabilized_frame(frame, frame_state, drone_state):
    if frame_state.show_stabilized:
        return stabilize_frame(frame, drone_state.current_pitch_rad, drone_state.current_roll_rad, pitch0_rad=np.deg2rad(0.0), hfov_deg=110)
    return frame.copy()

# --- Threads and Workers ---

class ImageViewer(Node):
    def __init__(self, data_queue):
        super().__init__('image_viewer')
        self.data_queue = data_queue
        self.subscription = self.create_subscription(
            Image,
            '/forward_camera/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info('Subscribed to /forward_camera/image_raw')

    def image_callback(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.data_queue.put((ts, 'IMAGE', msg))

def dualsense_worker(data_queue):
    dualsense = pydualsense()
    dualsense.init()
    try:
        while True:
            # Polling is implicit in pydualsense usually, or we assume it updates state
            # If explicit update needed, call it here.
            # dualsense.update() # Not standard in pydualsense lib, assuming bg thread or direct read

            ts = time.time()
            input_state = InputState(dualsense.state)
            data_queue.put((ts, 'INPUT', input_state))
            time.sleep(0.01) # 100 Hz
    finally:
        dualsense.close()

async def attitude_tracker(drone_state, data_queue):
    try:
        async for euler in drone_state.drone.telemetry.attitude_euler():
            data_queue.put((time.time(), 'ATTITUDE', euler))
    except:
        pass

async def body_velocity_tracker(drone_state, data_queue):
    try:
        async for velocity in drone_state.drone.telemetry.velocity_ned():
            data_queue.put((time.time(), 'VELOCITY_NED', velocity))
        async for angular in drone_state.drone.telemetry.attitude_angular_velocity_body():
            data_queue.put((time.time(), 'VELOCITY_ANGULAR', angular))
    except:
        pass

async def actuator_tracker(drone_state, data_queue):
    try:
        async for status in drone_state.drone.telemetry.actuator_output_status(0):
            data_queue.put((time.time(), 'ACTUATOR', status))
    except:
        pass

async def mavsdk_controller(frame_state, drone_state, pursuit_state, shared_input_state, config, logger, control_mode, data_queue):
    await drone_state.drone.connect(system_address="udp://:14540")
    async for state in drone_state.drone.core.connection_state():
        if state.is_connected:
            break

    # Start telemetry tasks
    asyncio.create_task(attitude_tracker(drone_state, data_queue))
    asyncio.create_task(body_velocity_tracker(drone_state, data_queue))
    asyncio.create_task(actuator_tracker(drone_state, data_queue))
   
    # Control loop
    while rclpy.ok():
        # Use shared_input_state which is updated by processing thread
        setpoint_type, setpoint = control_mode.get_setpoint(shared_input_state, config, pursuit_state, frame_state, drone_state)

        if drone_state.is_armed and drone_state.is_offboard:
            if setpoint_type == 'velocity':
                await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(*setpoint))
            elif setpoint_type == 'attitude':
                await drone_state.drone.offboard.set_attitude(Attitude(*setpoint))
            elif setpoint_type == 'attitude_rate':
                await drone_state.drone.offboard.set_attitude_rate(AttitudeRate(*setpoint))

        if shared_input_state.L1 and not drone_state.is_armed:
            try:
                await arm_and_takeoff(drone_state, logger)
                drone_state.is_armed = True
            except Exception as e:
                print(f"Arm/takeoff failed: {e} - Retry after disarm complete.")

        if shared_input_state.L2Btn and drone_state.is_armed:
            pursuit_state.autonomous_pursuit_active = False
            await land_and_disarm(drone_state, logger)
            drone_state.is_armed = False

        await asyncio.sleep(0.01)

async def arm_and_takeoff(drone_state, logger):
    print("Arming...")
    await drone_state.drone.action.arm()
    await asyncio.sleep(1)
    print("Setting initial setpoint before offboard...")
    await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    print("Starting offboard...")
    try:
        await drone_state.drone.offboard.start()
        drone_state.is_offboard = True
    except OffboardError as e:
        print(f"Offboard start failed: {e}")
        return
    print("Taking off via offboard (ascending)...")
    await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, -3.0, 0.0))
    await asyncio.sleep(5)
    await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

async def land_and_disarm(drone_state, logger):
    print("Landing...")
    await drone_state.drone.action.land()
    await asyncio.sleep(5)
    if drone_state.is_offboard:
        await drone_state.drone.offboard.stop()
        drone_state.is_offboard = False
    print("Disarming...")
    await drone_state.drone.action.disarm()
    await asyncio.sleep(2)
    await drone_state.drone.action.hold()

def processing_loop(data_queue, frame_state, drone_state, pursuit_state, button_state, video_state, cache_state, config, control_mode, logger, bridge, shared_input_state):
    while rclpy.ok():
        try:
            timestamp, data_type, data = data_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if data_type == 'IMAGE':
            try:
                frame_state.current_frame = bridge.imgmsg_to_cv2(data, 'mono8')
                process_visuals(config, frame_state, button_state, video_state, cache_state, shared_input_state, drone_state, pursuit_state, control_mode, logger)
            except CvBridgeError as e:
                logger.error(str(e))

        elif data_type == 'INPUT':
            # Update shared state
            shared_input_state.LX = data.LX
            shared_input_state.LY = data.LY
            shared_input_state.RX = data.RX
            shared_input_state.RY = data.RY
            shared_input_state.L1 = data.L1
            shared_input_state.L2Btn = data.L2Btn
            shared_input_state.R1 = data.R1
            shared_input_state.R2Btn = data.R2Btn
            shared_input_state.square = data.square
            shared_input_state.cross = data.cross
            shared_input_state.circle = data.circle
            shared_input_state.triangle = data.triangle
            shared_input_state.DpadUp = data.DpadUp
            shared_input_state.DpadDown = data.DpadDown
            shared_input_state.DpadLeft = data.DpadLeft
            shared_input_state.DpadRight = data.DpadRight

            # Handle logic
            handle_input_logic(data, button_state, frame_state, pursuit_state, config, control_mode)

        elif data_type == 'ATTITUDE':
            drone_state.current_pitch_rad = np.deg2rad(data.pitch_deg)
            drone_state.current_roll_rad = np.deg2rad(data.roll_deg)
            drone_state.current_yaw_deg = data.yaw_deg

        elif data_type == 'VELOCITY_NED':
            drone_state.current_body_velocities['vx'] = data.north_m_s
            drone_state.current_body_velocities['vy'] = data.east_m_s
            drone_state.current_body_velocities['vz'] = data.down_m_s

        elif data_type == 'VELOCITY_ANGULAR':
            drone_state.current_body_velocities['vyaw'] = np.rad2deg(data.yaw_rad_s)

        elif data_type == 'ACTUATOR':
            thrusts = data.actuator[:4]
            drone_state.current_thrust = np.mean(thrusts) if thrusts else 0.0

        data_queue.task_done()

def run_mavsdk_thread(frame_state, drone_state, pursuit_state, shared_input_state, config, logger, control_mode, data_queue):
    asyncio.run(mavsdk_controller(frame_state, drone_state, pursuit_state, shared_input_state, config, logger, control_mode, data_queue))

def main(args=None):
    rclpy.init(args=args)

    # Init shared objects
    data_queue = queue.PriorityQueue()
    config = DualSenseConfig()
    frame_state = FrameState()
    drone_state = DroneState()
    pursuit_state = PursuitState()
    button_state = ButtonState()
    video_state = VideoState()
    cache_state = CacheState()
    shared_input_state = InputState()
    control_mode = ManualMode()
    bridge = CvBridge()

    # 1. ROS Node & Thread
    viewer = ImageViewer(data_queue)
    ros_thread = threading.Thread(target=rclpy.spin, args=(viewer,), daemon=True)
    ros_thread.start()

    # 2. DualSense Thread
    ds_thread = threading.Thread(target=dualsense_worker, args=(data_queue,), daemon=True)
    ds_thread.start()

    # 3. MAVSDK Thread
    mav_thread = threading.Thread(target=run_mavsdk_thread, args=(frame_state, drone_state, pursuit_state, shared_input_state, config, viewer.get_logger(), control_mode, data_queue), daemon=True)
    mav_thread.start()

    # 4. Processing Loop (Main Thread)
    try:
        processing_loop(data_queue, frame_state, drone_state, pursuit_state, button_state, video_state, cache_state, config, control_mode, viewer.get_logger(), bridge, shared_input_state)
    except KeyboardInterrupt:
        pass
    finally:
        if video_state.video_writer:
            video_state.video_writer.release()
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
