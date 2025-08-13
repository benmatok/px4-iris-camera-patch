import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
from pydualsense import *
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed, Attitude, AttitudeRate
import numpy as np
import threading
import os
import pickle
from datetime import datetime

def velocity_to_attitude(vx_des, vy_des, vz_des, yaw_rate_des, current_yaw_deg, dt, config):
    g = 9.81 # gravity m/s^2
    hover_thrust = 0.3 # normalized hover thrust
    # Feedforward: map desired velocity to required acceleration (assume steady-state accel = 0, but scale tilt by v_des)
    # Simplified: tilt = arctan(v_des^2 / (r * g)) but for straight line, use proportional or arctan(v_des / ref)
    # Here, use arctan(v_des / g) treating v_des as "effective accel" for tilt (common approximation for open-loop)
    # Pitch for forward vx (negative for forward tilt)
    pitch_target = -np.rad2deg(np.arctan(vx_des / g))
    # Roll for sideways vy (positive for right tilt)
    roll_target = np.rad2deg(np.arctan(vy_des / g))
    # Compensate thrust for pitch/roll tilt to maintain hover vertically
    pitch_rad = np.deg2rad(pitch_target)
    roll_rad = np.deg2rad(roll_target)
    tilt_correction = 1.0 / (np.cos(pitch_rad) * np.cos(roll_rad)) if np.cos(pitch_rad) * np.cos(roll_rad) > 0 else 1.0
    base_thrust = hover_thrust * tilt_correction
    # Thrust correction for vz_des
    # vz_des negative for climb; scale to thrust correction
    thrust_correction = vz_des / (config.MAX_CLIMB) # normalize (sign: negative vz_des increases thrust for climb)
    thrust = base_thrust - thrust_correction # Adjust sign based on vz convention (PX4: positive down)
    thrust = np.clip(thrust, 0.1, 0.9)
    # Yaw: incremental absolute target (as before)
    yaw_target = current_yaw_deg + yaw_rate_des * dt
    return roll_target, pitch_target, yaw_target, thrust
   
def map_to_betaflight_rc(roll_target, pitch_target, yaw_rate_des, thrust, max_angle=45.0, max_yaw_rate=200.0):
    # Scaled to -500 to 500 (Betaflight rate/angle units)
    rc_roll = (roll_target / max_angle) * 500
    rc_pitch = (pitch_target / max_angle) * 500
    rc_yaw = (yaw_rate_des / max_yaw_rate) * 500
    rc_throttle = thrust * 1000 + 1000 # 1000-2000 PWM
    return rc_roll, rc_pitch, rc_yaw, rc_throttle
   
# Simple PID Controller definition (adapted from custom tracking_utils)
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
                self.integral -= error # Anti-windup: stop integral growth
            elif output < -self.output_limit:
                output = -self.output_limit
                self.integral -= error # Anti-windup: stop integral growth
        if self.smoothing_factor > 0:
            output = (self.smoothing_factor * output) + ((1 - self.smoothing_factor) * self.smoothed_output)
            self.smoothed_output = output
        self.previous_error = error
        return output
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.smoothed_output = 0.0
# Simple stabilize_frame function (adapted; assumes basic warp based on pitch/roll)
def stabilize_frame(frame, pitch_rad, roll_rad, pitch0_rad=0.0, hfov_deg=110.0):
    # Basic affine transformation for stabilization
    height, width = frame.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), np.rad2deg(roll_rad), 1.0)
    stabilized = cv2.warpAffine(frame, rotation_matrix, (width, height))
    # Simple shift for pitch (not accurate, for demo)
    shift_y = int(height * (pitch_rad - pitch0_rad) / np.deg2rad(hfov_deg / 2))
    M = np.float32([[1, 0, 0], [0, 1, shift_y]])
    stabilized = cv2.warpAffine(stabilized, M, (width, height))
    return stabilized
# Simple draw_alignment_info function (adapted; draws basic crosshair and text)
def draw_alignment_info(frame, self_obj, target_ratio=0.4):
    if self_obj.current_bbox_center is None:
        return frame
    height, width = frame.shape[:2]
    ref_x = self_obj.frame_center_x
    ref_y = int(frame_height * (1 - pursuit_state.target_ratio))
    cv2.line(frame, (ref_x, 0), (ref_x, height), (255, 0, 0), 1) # Vertical line
    cv2.line(frame, (0, ref_y), (width, ref_y), (255, 0, 0), 1) # Horizontal line
    cv2.circle(frame, self_obj.current_bbox_center, 5, (0, 255, 0), -1)
    cv2.putText(frame, "Aligned", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
# Draw control mode on frame
def draw_control_mode(frame, control_mode):
    mode_name = "Manual" if isinstance(control_mode, ManualMode) else "Angle" if isinstance(control_mode, AngleMode) else "Acro"
    cv2.putText(frame, f"Mode: {mode_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
# Configuration
class DualSenseConfig:
    def __init__(self):
        self.MAX_VELOCITY = 30.0
        self.MAX_CLIMB = 5.0
        self.MAX_DESCENT = 25.0
        self.MAX_YAW_RATE = 90.0
        self.GAMMA = 3.0
        self.SHOW_VIDEO = True
        self.PUBLISH_VIDEO = False # Set to True if you add VideoPublisher
        self.RECORD_VIDEO = False
class DroneState:
    def __init__(self):
        self.drone = System()
        self.is_armed = False
        self.is_offboard = False
        self.current_pitch_rad = 0.0
        self.current_roll_rad = 0.0
        self.current_yaw_deg = 0.0
        self.current_body_velocities = {'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'vyaw': 0.0}
class PursuitState:
    def __init__(self):
        self.autonomous_pursuit_active = False
        self.pursuit_pid_yaw = PIDController(kp=0.1, ki=0.1, kd=0.03, output_limit=25.0)
        self.pursuit_pid_alt = PIDController(kp=0.1, ki=0.03, kd=0.1, output_limit=5.0) # Lower gains, tighter limit
        self.pursuit_pid_forward = PIDController(kp=0.3, ki=0.05, kd=0.01, output_limit=5.0, smoothing_factor=0.1)
        self.target_ratio = 0.4
        self.forward_velocity = 10.0
        self.pursuit_debug_info = {}
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
# Strategy for control modes
class ControlMode:
    def get_setpoint(self, dualsense, config, pursuit_state, frame_state, drone_state):
        raise NotImplementedError("Subclasses must implement get_setpoint")
class ManualMode(ControlMode):
    def get_setpoint(self, dualsense, config, pursuit_state, frame_state, drone_state):
        if pursuit_state.autonomous_pursuit_active:
            if frame_state.current_bbox_center:
                vx, vy, vz, yaw_rate = calc_pursuit_velocities(pursuit_state, drone_state, frame_state.current_bbox_center, frame_state.current_frame_width, frame_state.current_frame_height)
                return 'velocity', (vx, vy, vz, yaw_rate)
            return 'velocity', (0.0, 0.0, 0.0, 0.0)
        lx = dualsense.state.LX / 128.0
        ly = dualsense.state.LY / 128.0
        rx = dualsense.state.RX / 128.0
        ry = dualsense.state.RY / 128.0
        # Gamma map (simple)
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
    def get_setpoint(self, dualsense, config, pursuit_state, frame_state, drone_state):
        # Compute desired body velocities and yaw rate (same as manual)
        if pursuit_state.autonomous_pursuit_active:
            if frame_state.current_bbox_center:
                vx, vy, vz, yaw_rate = calc_pursuit_velocities(pursuit_state, drone_state, frame_state.current_bbox_center, frame_state.current_frame_width, frame_state.current_frame_height)
            else:
                vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0
        else:
            lx = dualsense.state.LX / 128.0
            ly = dualsense.state.LY / 128.0
            rx = dualsense.state.RX / 128.0
            ry = dualsense.state.RY / 128.0
            # Gamma map (simple)
            lx = np.sign(lx) * (abs(lx) ** config.GAMMA)
            ly = np.sign(ly) * (abs(ly) ** config.GAMMA)
            rx = np.sign(rx) * (abs(rx) ** config.GAMMA)
            ry = np.sign(ry) * (abs(ry) ** config.GAMMA)
            vx = -ly * config.MAX_VELOCITY
            vy = lx * config.MAX_VELOCITY
            vz = ry * (config.MAX_CLIMB if ry < 0 else config.MAX_DESCENT) if abs(ry) > 0 else 0.0
            yaw_rate = rx * config.MAX_YAW_RATE
        # Translate to attitude setpoints
        dt = 1.0  # Loop timestep
        roll_target, pitch_target, yaw_target, thrust = velocity_to_attitude(
            vx, vy, vz, yaw_rate, drone_state.current_yaw_deg, dt, config
        )
        # Optional: Add coordinated roll for yaw (to prevent sideslip)
        coord_roll = np.rad2deg(np.arctan(vx * np.deg2rad(yaw_rate) / 9.81)) # Sign may need adjustment
        roll_target += coord_roll
        # Limit angles (for stabilized feel)
        max_tilt = 30.0
        roll_target = np.clip(roll_target, -max_tilt, max_tilt)
        pitch_target = np.clip(pitch_target, -max_tilt, max_tilt)
        return 'attitude', (roll_target, pitch_target, yaw_target, thrust)

# Helper functions for quaternion math
def euler_to_quat(roll_deg, pitch_deg, yaw_deg):
    # Convert Euler angles (deg) to quaternion (w, x, y, z)
    roll = np.deg2rad(roll_deg / 2)
    pitch = np.deg2rad(pitch_deg / 2)
    yaw = np.deg2rad(yaw_deg / 2)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_to_body_rates(error_quat, kp_roll=4.0, kp_pitch=4.0, kp_yaw=2.0):
    # Compute body angular rates from error quaternion
    # Normalize error_quat if needed
    error_quat /= np.linalg.norm(error_quat)
    # Ensure shortest rotation
    if error_quat[0] < 0:
        error_quat = -error_quat
    # For small angles, omega ≈ 2 * vec(error_quat)
    # Apply separate gains per axis (assuming x=roll, y=pitch, z=yaw)
    vec = error_quat[1:]
    rates = 2 * np.array([kp_roll * vec[0], kp_pitch * vec[1], kp_yaw * vec[2]])
    return rates  # [roll_rate, pitch_rate, yaw_rate]

class ImageViewer(Node):
    def __init__(self, config):
        super().__init__('image_viewer')
        self.config = config
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/forward_camera/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info('Subscribed to /forward_camera/image_raw')
        self.dualsense = pydualsense()
        self.dualsense.init()
        self.drone_state = DroneState()
        self.pursuit_state = PursuitState()
        self.frame_state = FrameState()
        self.button_state = ButtonState()
        self.video_state = VideoState()
        self.cache_state = CacheState()
        self.prev_bbox = None
        self.control_mode = ManualMode() # Default; switch with R1 (angle), R2 (manual)
        # Start MAVSDK connection in thread
        threading.Thread(target=self.run_mavsdk_controller, daemon=True).start()
        # Timer for state machine (run every 0.01s)
        self.timer = self.create_timer(0.01, self.state_machine_callback_wrapper) # 100 fps
    def run_mavsdk_controller(self):
        asyncio.run(mavsdk_controller(self.frame_state,self.drone_state, self.pursuit_state, self.dualsense, self.config, self.get_logger(), self.control_mode))
    def image_callback(self, msg):
        try:
            self.frame_state.current_frame = self.bridge.imgmsg_to_cv2(msg, 'mono8')  # Monochrome camera
        except CvBridgeError as e:
            self.get_logger().error(str(e))
    def state_machine_callback_wrapper(self):
        state_machine_callback(self.config, self.frame_state, self.button_state, self.video_state, self.cache_state, self.dualsense, self.drone_state, self.pursuit_state, self.prev_bbox, self.get_logger(), self.control_mode)
def state_machine_callback(config, frame_state, button_state, video_state, cache_state, dualsense, drone_state, pursuit_state, prev_bbox, logger, control_mode):
    if frame_state.current_frame is None:
        return
    frame = frame_state.current_frame.copy()
    height, width = frame.shape[:2]
    frame_state.current_frame_height, frame_state.current_frame_width = height, width
    # Toggle stabilization
    handle_stabilization_toggle(frame_state,button_state, dualsense)
    # Toggle control mode
    handle_control_mode_toggle(button_state, dualsense, control_mode)
    display_frame = get_stabilized_frame(frame, frame_state, drone_state)
    # For drawing, convert grayscale to BGR
    if len(display_frame.shape) == 2:
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
    # Record if enabled
    handle_video_recording(config, video_state, display_frame, width, height, cache_state)
    # BBox and tracking logic
    handle_modes(frame_state, drone_state, button_state, pursuit_state, dualsense, display_frame, width, height)
    # Draw alignment if pursuit
    handle_pursuit_alignment(pursuit_state, drone_state, frame_state, display_frame)
    # Draw control mode
    display_frame = draw_control_mode(display_frame, control_mode)
    show_video(config, display_frame, logger)
def handle_stabilization_toggle(frame_state,button_state, dualsense):
    o_state = dualsense.state.circle
    if o_state and not button_state.prev_o_state:
        frame_state.show_stabilized = not frame_state.show_stabilized
    button_state.prev_o_state = o_state
def handle_control_mode_toggle(button_state, dualsense, control_mode):
    r1_state = dualsense.state.R1
    r2_state = dualsense.state.R2Btn
    if r1_state and not button_state.prev_r1_state:
        control_mode.__class__ = AngleMode # Switch to Angle mode
    if r2_state and not button_state.prev_r2_state:
        control_mode.__class__ = ManualMode # Switch to Manual mode
    button_state.prev_r1_state = r1_state
    button_state.prev_r2_state = r2_state
def get_stabilized_frame(frame, frame_state, drone_state):
    if frame_state.show_stabilized:
        return stabilize_frame(frame, drone_state.current_pitch_rad, drone_state.current_roll_rad, pitch0_rad=np.deg2rad(0.0), hfov_deg=110)
    return frame.copy()
def handle_video_recording(config, video_state, display_frame, width, height, cache_state):
    if config.RECORD_VIDEO and video_state.video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_state.video_filename = os.path.join(cache_state.cache_dir, f"stabilized_video_{timestamp}.mp4")
        video_state.video_writer = cv2.VideoWriter(video_state.video_filename, fourcc, 30, (width, height))
    if config.RECORD_VIDEO and video_state.video_writer:
        video_state.video_writer.write(display_frame)
def handle_modes(frame_state, drone_state, button_state, pursuit_state, dualsense, display_frame, width, height):
    bbox_size = 50
    if frame_state.bbox is None:
        frame_state.bbox = [width//2 - bbox_size//2, height//2 - bbox_size//2, bbox_size, bbox_size]
    # Mode toggles with buttons
    handle_button_toggles(frame_state, button_state, pursuit_state, dualsense)
    # BBox controls
    handle_bbox_controls(frame_state, dualsense, pursuit_state, width, height)
    if pursuit_state.autonomous_pursuit_active:
        process_pursuit_mode(frame_state, drone_state, pursuit_state, display_frame)
    elif frame_state.tracking_mode:
        process_tracking_mode(frame_state, display_frame)
    elif frame_state.bbox_mode:
        process_bbox_mode(frame_state, display_frame)
        
def handle_button_toggles(frame_state, button_state, pursuit_state, dualsense):
    if dualsense.state.square and not button_state.prev_square_state:
        if not pursuit_state.autonomous_pursuit_active:
            frame_state.bbox_mode = not frame_state.bbox_mode
            frame_state.tracking_mode = False
            frame_state.bbox_size_mode = False
    button_state.prev_square_state = dualsense.state.square
    if frame_state.bbox_mode and dualsense.state.circle and not button_state.prev_circle_state:
        if not pursuit_state.autonomous_pursuit_active:
            frame_state.bbox_size_mode = not frame_state.bbox_size_mode
    button_state.prev_circle_state = dualsense.state.circle
    if dualsense.state.cross and not button_state.prev_cross_state:
        frame_state.tracking_mode = not frame_state.tracking_mode
        if frame_state.tracking_mode:
            frame_state.init_tracking = True
            frame_state.bbox_mode = False  # Exit bbox mode when entering tracking
        else:
            frame_state.tracker_instance = None  # Reset tracker when exiting
    button_state.prev_cross_state = dualsense.state.cross
    if dualsense.state.triangle and not button_state.prev_triangle_state:
        pursuit_state.autonomous_pursuit_active = not pursuit_state.autonomous_pursuit_active
        if pursuit_state.autonomous_pursuit_active:
            pursuit_state.pursuit_pid_yaw.reset()
            pursuit_state.pursuit_pid_alt.reset()
            pursuit_state.pursuit_pid_forward.reset()
    button_state.prev_triangle_state = dualsense.state.triangle
    
def handle_bbox_controls(frame_state, dualsense, pursuit_state, width, height):
    if not frame_state.bbox_mode or pursuit_state.autonomous_pursuit_active:
        return
    move_step = 5
    size_step = 10
    if frame_state.bbox_size_mode:
        # Size adjustments
        x, y, w, h = frame_state.bbox
        center_x = x + w // 2
        center_y = y + h // 2
        if dualsense.state.DpadUp:
            h = max(20, h - size_step)
            y = center_y - h // 2
        if dualsense.state.DpadDown:
            h = min(height - 20, h + size_step)
            y = center_y - h // 2
        if dualsense.state.DpadLeft:
            w = max(20, w - size_step)
            x = center_x - w // 2
        if dualsense.state.DpadRight:
            w = min(width - 20, w + size_step)
            x = center_x - w // 2
        frame_state.bbox = [x, y, w, h]
        frame_state.bbox[0] = max(0, min(frame_state.bbox[0], width - frame_state.bbox[2]))
        frame_state.bbox[1] = max(0, min(frame_state.bbox[1], height - frame_state.bbox[3]))
    else:
        # Move
        x, y, w, h = frame_state.bbox
        if dualsense.state.DpadUp:
            y = max(0, y - move_step)
        if dualsense.state.DpadDown:
            y = min(height - h, y + move_step)
        if dualsense.state.DpadLeft:
            x = max(0, x - move_step)
        if dualsense.state.DpadRight:
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
    # Initialize or update KCF tracker (using grayscale current_frame for consistency)
    if frame_state.init_tracking:
        if frame_state.tracker_instance is None:
            frame_state.tracker_instance = cv2.TrackerKCF_create()
        success = frame_state.tracker_instance.init(frame_state.current_frame, tuple(frame_state.bbox))
        frame_state.init_tracking = False
        if not success:
            frame_state.tracking_mode = False
            frame_state.tracker_instance = None
            return  # Early exit if init fails

    if frame_state.tracker_instance:
        success, bbox = frame_state.tracker_instance.update(frame_state.current_frame)
        if success:
            frame_state.bbox = [int(v) for v in bbox]
        else:
            # Tracking lost; exit mode
            frame_state.tracking_mode = False
            frame_state.tracker_instance = None

    # Proceed with bbox processing (even if tracking lost, use last known)
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
    # Assume tracker logic
    # For example:
    # if frame_state.tracker_instance:
    # out = frame_state.tracker_instance.track(display_frame)
    # frame_state.bbox = [int(s) for s in out["target_bbox"]]
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
    calc_pursuit_velocities(pursuit_state, drone_state, bbox_center_local, frame_state.current_frame_width, frame_state.current_frame_height)
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
        current_velocities = (0, 0, 0, 0) # Placeholder
        center_proximity_threshold = 0.5
        landing_proximity_threshold = 0.2
        current_pitch_rad = drone_state.current_pitch_rad
        current_roll_rad = drone_state.current_roll_rad
    display_frame = draw_alignment_info(display_frame, MockSelf())
    
def show_video(config, display_frame, logger):
    if not config.SHOW_VIDEO:
        return
    cv2.imshow('Forward Camera Feed', display_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        logger.info('Shutting down')
        rclpy.shutdown()
        
def get_control_velocities(pursuit_state, drone_state, dualsense, config):
    if pursuit_state.autonomous_pursuit_active:
        if frame_state.current_bbox_center:
            return calc_pursuit_velocities(pursuit_state, drone_state, frame_state.current_bbox_center, frame_state.current_frame_width, frame_state.current_frame_height)
        return 0.0, 0.0, 0.0, 0.0
    lx = dualsense.state.LX / 128.0
    ly = dualsense.state.LY / 128.0
    rx = dualsense.state.RX / 128.0
    ry = dualsense.state.RY / 128.0
    # Gamma map (simple)
    lx = np.sign(lx) * (abs(lx) ** config.GAMMA)
    ly = np.sign(ly) * (abs(ly) ** config.GAMMA)
    rx = np.sign(rx) * (abs(rx) ** config.GAMMA)
    ry = np.sign(ry) * (abs(ry) ** config.GAMMA)
    vx = -ly * config.MAX_VELOCITY
    vy = lx * config.MAX_VELOCITY
    vz = ry * (config.MAX_CLIMB if ry < 0 else config.MAX_DESCENT) if abs(ry) > 0 else 0.0
    yaw_rate = rx * config.MAX_YAW_RATE
    return vx, vy, vz, yaw_rate
    
def calc_pursuit_velocities(pursuit_state, drone_state, bbox_center, frame_width, frame_height):
    if bbox_center is None:
        return 0.0, 0.0, 0.0, 0.0
    ref_x = frame_width // 2
    ref_y = int(frame_height * (1 - pursuit_state.target_ratio))
    bbox_x, bbox_y = bbox_center
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    hfov_deg = 110.0  # From camera parameters
    hfov_rad = np.deg2rad(hfov_deg)
    fx = (frame_width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx  # Assuming square pixels and VFOV derived from HFOV and aspect ratio
    u = (bbox_x - cx) / fx
    v = (bbox_y - cy) / fy
    u_des = (ref_x - cx) / fx
    v_des = (ref_y - cy) / fy
    angle_x = np.arctan(u)
    angle_y = np.arctan(v)
    angle_x_des = np.arctan(u_des)
    angle_y_des = np.arctan(v_des)
    error_angle_x = angle_x - angle_x_des
    error_angle_y = angle_y - angle_y_des
    error_angle_x_deg = np.rad2deg(error_angle_x)
    error_angle_y_deg = np.rad2deg(error_angle_y)
    error_norm_angle = np.sqrt(error_angle_x**2 + error_angle_y**2)
    error_norm_angle_deg = np.rad2deg(error_norm_angle)
    yaw_rate = pursuit_state.pursuit_pid_yaw.update(error_angle_x_deg)
    # Updated to allow full range (climb and descend) without max( ,0)
    vz = pursuit_state.pursuit_pid_alt.update(error_angle_y_deg)
    # Geometric throttle based on angular misalignment
    misalignment_start_deg = 12.0  # Start reducing speed (tuned close to original effective ~12.75 deg)
    misalignment_range_deg = 18.0  # Range to reach zero throttle (tuned close to original effective ~17.85 deg)
    throttle = np.clip(1 - (error_norm_angle_deg - misalignment_start_deg) / misalignment_range_deg, 0.0, 1.0)
    target_speed = pursuit_state.forward_velocity * throttle
    target_speed = max(target_speed, 0.4)
    vx = pursuit_state.pursuit_pid_forward.update(target_speed - drone_state.current_body_velocities['vx'])
    pursuit_state.pursuit_debug_info = {
        'error_angle_x_deg': error_angle_x_deg,
        'error_angle_y_deg': error_angle_y_deg,
        'yaw_rate': yaw_rate,
        'vx': vx,
        'vz': vz,
        'throttle': throttle,
        'current_speed': drone_state.current_body_velocities['vx'],
        'target_speed': target_speed,
        'error_norm_angle_deg': error_norm_angle_deg,
        'target_ratio_used': pursuit_state.target_ratio
    }
    return vx, 0.0, vz, yaw_rate
    
    
    
def is_bbox_area_black(frame, bbox, threshold=30):
    if frame is None or bbox is None:
        return False
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, min(x, frame.shape[1] - 1))
    y = max(0, min(y, frame.shape[0] - 1))
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    if w <= 0 or h <= 0:
        return True
    roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(roi)
    return mean_brightness < threshold
async def mavsdk_controller(frame_state, drone_state, pursuit_state, dualsense, config, logger, control_mode):
    await drone_state.drone.connect(system_address="udp://:14540")
    async for state in drone_state.drone.core.connection_state():
        if state.is_connected:
            break
    attitude_task = asyncio.create_task(attitude_tracker(drone_state))
    velocity_task = asyncio.create_task(body_velocity_tracker(drone_state))
    while rclpy.ok():
        setpoint_type, setpoint = control_mode.get_setpoint(dualsense, config, pursuit_state, frame_state, drone_state)
        if drone_state.is_armed and drone_state.is_offboard:
            if setpoint_type == 'velocity':
                await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(*setpoint))
            elif setpoint_type == 'attitude':
                await drone_state.drone.offboard.set_attitude(Attitude(*setpoint))
            elif setpoint_type == 'attitude_rate':
                await drone_state.drone.offboard.set_attitude_rate(AttitudeRate(*setpoint))
        if dualsense.state.L1 and not drone_state.is_armed:
            try:
                await arm_and_takeoff(drone_state, logger)
                drone_state.is_armed = True
            except Exception as e:
                print(f"Arm/takeoff failed: {e} - Retry after disarm complete.")
        if dualsense.state.L2Btn and drone_state.is_armed:
            pursuit_state.autonomous_pursuit_active = False
            await land_and_disarm(drone_state, logger)
            drone_state.is_armed = False
        await asyncio.sleep(0.01)
async def attitude_tracker(drone_state):
    try:
        async for euler in drone_state.drone.telemetry.attitude_euler():
            drone_state.current_pitch_rad = np.deg2rad(euler.pitch_deg)
            drone_state.current_roll_rad = np.deg2rad(euler.roll_deg)
            drone_state.current_yaw_deg = euler.yaw_deg
    except:
        pass
async def body_velocity_tracker(drone_state):
    try:
        async for velocity in drone_state.drone.telemetry.velocity_ned():
            # Simple assignment (adapt if needed)
            drone_state.current_body_velocities['vx'] = velocity.north_m_s
            drone_state.current_body_velocities['vy'] = velocity.east_m_s
            drone_state.current_body_velocities['vz'] = velocity.down_m_s
        async for angular in drone_state.drone.telemetry.attitude_angular_velocity_body():
            drone_state.current_body_velocities['vyaw'] = np.rad2deg(angular.yaw_rad_s) # To deg/s
    except:
        pass
async def arm_and_takeoff(drone_state, logger):
    print("Arming...")
    await drone_state.drone.action.arm()
    await asyncio.sleep(1) # Wait for arm confirmation
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
    # Ramp up velocity for takeoff (e.g., -3 m/s up for 5s)
    await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, -3.0, 0.0))
    await asyncio.sleep(5) # Adjust time for desired height
    await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)) # Hover
async def land_and_disarm(drone_state, logger):
    print("Landing...")
    await drone_state.drone.action.land()
    await asyncio.sleep(5)
    if drone_state.is_offboard:
        await drone_state.drone.offboard.stop()
        drone_state.is_offboard = False
    print("Disarming...")
    await drone_state.drone.action.disarm()
    await asyncio.sleep(2) # Wait for full disarm
    # Optional: Set to HOLD mode for clean state
    await drone_state.drone.action.hold()
def main(args=None):
    rclpy.init(args=args)
    config = DualSenseConfig()
    viewer = ImageViewer(config)
    rclpy.spin(viewer)
    viewer.dualsense.close()
    if viewer.video_state.video_writer:
        viewer.video_state.video_writer.release()
    cv2.destroyAllWindows()
    viewer.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
