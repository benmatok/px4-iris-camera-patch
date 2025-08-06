import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
from pydualsense import *
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
import numpy as np
import threading
import os
import pickle
from datetime import datetime

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
            output = np.clip(output, -self.output_limit, self.output_limit)
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
def stabilize_frame(frame, pitch_rad, roll_rad, pitch0_rad=0.0, hfov_deg=80.0):
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
    ref_y = int(height * target_ratio)
    cv2.line(frame, (ref_x, 0), (ref_x, height), (255, 0, 0), 1)  # Vertical line
    cv2.line(frame, (0, ref_y), (width, ref_y), (255, 0, 0), 1)  # Horizontal line
    cv2.circle(frame, self_obj.current_bbox_center, 5, (0, 255, 0), -1)
    cv2.putText(frame, "Aligned", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# Configuration
class DualSenseConfig:
    def __init__(self):
        self.MAX_VELOCITY = 30.0
        self.MAX_CLIMB = 5.0
        self.MAX_DESCENT = 25.0
        self.MAX_YAW_RATE = 45.0
        self.GAMMA = 3.0
        self.SHOW_VIDEO = True
        self.PUBLISH_VIDEO = False  # Set to True if you add VideoPublisher
        self.RECORD_VIDEO = False

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
        self.current_frame = None
        self.dualsense = pydualsense()
        self.dualsense.init()
        self.drone = System()
        self.is_armed = False
        self.is_offboard = False
        self.current_pitch_rad = 0.0
        self.current_roll_rad = 0.0
        self.autonomous_pursuit_active = False
        self.pursuit_pid_yaw = PIDController(kp=0.1, ki=0.1, kd=0.03, output_limit=25.0)
        self.pursuit_pid_alt = PIDController(kp=0.03, ki=0.005, kd=0.001, output_limit=5.0)
        self.pursuit_pid_forward = PIDController(kp=0.3, ki=0.05, kd=0.01, output_limit=5.0, smoothing_factor=0.1)
        self.target_ratio = 0.4
        self.forward_velocity = 10.0
        self.current_body_velocities = {'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'vyaw': 0.0}
        self.current_bbox_center = None
        self.current_frame_width = 640
        self.current_frame_height = 480
        self.pursuit_debug_info = {}
        self.current_bbox_is_large = False
        self.video_writer = None
        self.video_filename = None
        self.bbox = None
        self.bbox_mode = False
        self.tracking_mode = False
        self.tracker_instance = None  # Assume litetrack is installed; otherwise comment out tracking parts
        self.bbox_size_mode = False
        self.show_stabilized = True
        self.prev_circle_state = False
        self.prev_square_state = False
        self.prev_cross_state = False
        self.prev_triangle_state = False
        self.prev_o_state = False
        self.prev_bbox = None

        # Cache setup
        self.cache_dir = os.path.expanduser("~/cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_frame_path = os.path.join(self.cache_dir, "roi_frame.png")
        self.cache_bbox_path = os.path.join(self.cache_dir, "roi_bbox.pkl")

        # Start MAVSDK connection in thread
        threading.Thread(target=self.run_mavsdk_controller, daemon=True).start()

        # Timer for state machine (run every 0.05s)
        self.timer = self.create_timer(0.05, self.state_machine_callback)

    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))

    def state_machine_callback(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        height, width = frame.shape[:2]
        self.current_frame_height, self.current_frame_width = height, width

        # Toggle stabilization
        self.handle_stabilization_toggle()

        display_frame = self.get_stabilized_frame(frame)

        # Record if enabled
        self.handle_video_recording(display_frame, width, height)

        # BBox and tracking logic
        self.handle_modes(display_frame, width, height)

        # Draw alignment if pursuit
        self.handle_pursuit_alignment(display_frame)

        self.show_video(display_frame)

    def handle_stabilization_toggle(self):
        o_state = self.dualsense.state.circle
        if o_state and not self.prev_o_state:
            self.show_stabilized = not self.show_stabilized
        self.prev_o_state = o_state

    def get_stabilized_frame(self, frame):
        if self.show_stabilized:
            return stabilize_frame(frame, self.current_pitch_rad, self.current_roll_rad, pitch0_rad=np.deg2rad(-15.0), hfov_deg=100)
        return frame.copy()

    def handle_video_recording(self, display_frame, width, height):
        if self.config.RECORD_VIDEO and self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = os.path.join(self.cache_dir, f"stabilized_video_{timestamp}.mp4")
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, 30, (width, height))
        if self.config.RECORD_VIDEO and self.video_writer:
            self.video_writer.write(display_frame)

    def handle_modes(self, display_frame, width, height):
        bbox_size = 50
        if self.bbox is None:
            self.bbox = [width//2 - bbox_size//2, height//2 - bbox_size//2, bbox_size, bbox_size]

        # Mode toggles with buttons
        self.handle_button_toggles()

        # BBox controls
        self.handle_bbox_controls(width, height)

        if self.autonomous_pursuit_active:
            self.process_pursuit_mode(display_frame)
        elif self.tracking_mode:
            self.process_tracking_mode(display_frame)
        elif self.bbox_mode:
            self.process_bbox_mode(display_frame)

    def handle_button_toggles(self):
        if self.dualsense.state.square and not self.prev_square_state:
            if not self.autonomous_pursuit_active:
                self.bbox_mode = not self.bbox_mode
                self.tracking_mode = False
                self.bbox_size_mode = False
        self.prev_square_state = self.dualsense.state.square

        if self.bbox_mode and self.dualsense.state.circle and not self.prev_circle_state:
            if not self.autonomous_pursuit_active:
                self.bbox_size_mode = not self.bbox_size_mode
        self.prev_circle_state = self.dualsense.state.circle

        if self.dualsense.state.cross and not self.prev_cross_state:
            # Enter tracking (if tracker available)
            pass  # Add tracker init
        self.prev_cross_state = self.dualsense.state.cross

        if self.dualsense.state.triangle and not self.prev_triangle_state:
            self.autonomous_pursuit_active = not self.autonomous_pursuit_active
            if self.autonomous_pursuit_active:
                self.pursuit_pid_yaw.reset()
                self.pursuit_pid_alt.reset()
                self.pursuit_pid_forward.reset()
        self.prev_triangle_state = self.dualsense.state.triangle

    def handle_bbox_controls(self, width, height):
        if not self.bbox_mode or self.autonomous_pursuit_active:
            return

        move_step = 5
        size_step = 10
        if self.bbox_size_mode:
            # Size adjustments
            x, y, w, h = self.bbox
            center_x = x + w // 2
            center_y = y + h // 2
            if self.dualsense.state.DpadUp:
                h = max(20, h - size_step)
                y = center_y - h // 2
            if self.dualsense.state.DpadDown:
                h = min(height - 20, h + size_step)
                y = center_y - h // 2
            if self.dualsense.state.DpadLeft:
                w = max(20, w - size_step)
                x = center_x - w // 2
            if self.dualsense.state.DpadRight:
                w = min(width - 20, w + size_step)
                x = center_x - w // 2
            self.bbox = [x, y, w, h]
            self.bbox[0] = max(0, min(self.bbox[0], width - self.bbox[2]))
            self.bbox[1] = max(0, min(self.bbox[1], height - self.bbox[3]))
        else:
            # Move
            x, y, w, h = self.bbox
            if self.dualsense.state.DpadUp:
                y = max(0, y - move_step)
            if self.dualsense.state.DpadDown:
                y = min(height - h, y + move_step)
            if self.dualsense.state.DpadLeft:
                x = max(0, x - move_step)
            if self.dualsense.state.DpadRight:
                x = min(width - w, x + move_step)
            self.bbox = [x, y, w, h]

    def process_bbox_mode(self, display_frame):
        x, y, w, h = self.bbox
        large_threshold = 0.1 * display_frame.shape[0]
        self.current_bbox_is_large = w > large_threshold or h > large_threshold
        bbox_center_x = x + w // 2
        if self.current_bbox_is_large:
            bbox_center_y = int(y * 0.25 + (y + h) * 0.75)
        else:
            bbox_center_y = y + h // 2
        bbox_center_local = (bbox_center_x, bbox_center_y)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.current_bbox_center = bbox_center_local

    def process_tracking_mode(self, display_frame):
        # Assume tracker logic here
        # For example:
        # if self.tracker_instance:
        #     out = self.tracker_instance.track(display_frame)
        #     self.bbox = [int(s) for s in out["target_bbox"]]
        x, y, w, h = self.bbox
        large_threshold = 0.1 * display_frame.shape[0]
        self.current_bbox_is_large = w > large_threshold or h > large_threshold
        bbox_center_x = x + w // 2
        if self.current_bbox_is_large:
            bbox_center_y = int(y * 0.25 + (y + h) * 0.75)
        else:
            bbox_center_y = y + h // 2
        bbox_center_local = (bbox_center_x, bbox_center_y)
        bbox_color = (128, 0, 255) if self.current_bbox_is_large else (0, 0, 255)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), bbox_color, 2)
        self.current_bbox_center = bbox_center_local

    def process_pursuit_mode(self, display_frame):
        # Assume tracker logic
        # For example:
        # if self.tracker_instance:
        #     out = self.tracker_instance.track(display_frame)
        #     self.bbox = [int(s) for s in out["target_bbox"]]
        x, y, w, h = self.bbox
        large_threshold = 0.1 * display_frame.shape[0]
        self.current_bbox_is_large = w > large_threshold or h > large_threshold
        bbox_center_x = x + w // 2
        if self.current_bbox_is_large:
            bbox_center_y = int(y * 0.25 + (y + h) * 0.75)
        else:
            bbox_center_y = y + h // 2
        bbox_center_local = (bbox_center_x, bbox_center_y)
        bbox_color = (128, 0, 255) if self.current_bbox_is_large else (255, 0, 0)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), bbox_color, 3)
        cv2.circle(display_frame, bbox_center_local, 3, (255, 255, 255), -1)
        self.calc_pursuit_velocities(bbox_center_local, display_frame.shape[1], display_frame.shape[0])
        self.current_bbox_center = bbox_center_local

    def handle_pursuit_alignment(self, display_frame):
        if not self.autonomous_pursuit_active:
            return
        class MockSelf:
            frame_width = display_frame.shape[1]
            frame_height = display_frame.shape[0]
            frame_center_x = display_frame.shape[1] // 2
            frame_center_y = display_frame.shape[0] // 2
            current_bbox_center = self.current_bbox_center
            current_velocities = (0, 0, 0, 0)  # Placeholder
            center_proximity_threshold = 0.5
            landing_proximity_threshold = 0.2
            current_pitch_rad = self.current_pitch_rad
            current_roll_rad = self.current_roll_rad

        display_frame = draw_alignment_info(display_frame, MockSelf())

    def show_video(self, display_frame):
        if not self.config.SHOW_VIDEO:
            return
        cv2.imshow('Forward Camera Feed', display_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.get_logger().info('Shutting down')
            rclpy.shutdown()

    async def attitude_tracker(self):
        try:
            async for euler in self.drone.telemetry.attitude_euler():
                self.current_pitch_rad = np.deg2rad(euler.pitch_deg)
                self.current_roll_rad = np.deg2rad(euler.roll_deg)
        except:
            pass

    async def body_velocity_tracker(self):
        try:
            async for velocity in self.drone.telemetry.velocity_ned():
                # Simple assignment (adapt if needed)
                self.current_body_velocities['vx'] = velocity.north_m_s
                self.current_body_velocities['vy'] = velocity.east_m_s
                self.current_body_velocities['vz'] = velocity.down_m_s
            async for angular in self.drone.telemetry.attitude_angular_velocity_body():
                self.current_body_velocities['vyaw'] = np.rad2deg(angular.yaw_rad_s)  # To deg/s
        except:
            pass

    def run_mavsdk_controller(self):
        asyncio.run(self.mavsdk_controller())

    async def mavsdk_controller(self):
        await self.drone.connect(system_address="udp://:14540")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                break

        attitude_task = asyncio.create_task(self.attitude_tracker())
        velocity_task = asyncio.create_task(self.body_velocity_tracker())

        async def arm_and_takeoff(self):
            print("Arming...")
            await self.drone.action.arm()
            await asyncio.sleep(1)  # Wait for arm confirmation
            print("Setting initial setpoint before offboard...")
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            print("Starting offboard...")
            try:
                await self.drone.offboard.start()
                self.is_offboard = True
            except OffboardError as e:
                print(f"Offboard start failed: {e}")
                return
            print("Taking off via offboard (ascending)...")
            # Ramp up velocity for takeoff (e.g., -3 m/s up for 5s)
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, -3.0, 0.0))
            await asyncio.sleep(5)  # Adjust time for desired height
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))  # Hover

        async def land_and_disarm(self):
            print("Landing...")
            await self.drone.action.land()
            await asyncio.sleep(5)
            if self.is_offboard:
                await self.drone.offboard.stop()
                self.is_offboard = False
            print("Disarming...")
            await self.drone.action.disarm()
            await asyncio.sleep(2)  # Wait for full disarm
            # Optional: Set to HOLD mode for clean state
            await self.drone.action.set_current_flight_mode(5)  # 5 = HOLD

        while rclpy.ok():
            vx, vy, vz, yaw_rate = self.get_control_velocities()

            if self.is_armed and self.is_offboard:
                await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, vz, yaw_rate))

            if self.dualsense.state.L1 and not self.is_armed:
                try:
                    await arm_and_takeoff(self)
                    self.is_armed = True
                except Exception as e:
                    print(f"Arm/takeoff failed: {e} - Retry after disarm complete.")
            if self.dualsense.state.L2Btn and self.is_armed:
                self.autonomous_pursuit_active = False
                await land_and_disarm()
                self.is_armed = False

            await asyncio.sleep(0.05)

    def get_control_velocities(self):
        if self.autonomous_pursuit_active:
            if self.current_bbox_center:
                return self.calc_pursuit_velocities(self.current_bbox_center, self.current_frame_width, self.current_frame_height)
            return 0.0, 0.0, 0.0, 0.0
        lx = self.dualsense.state.LX / 128.0
        ly = self.dualsense.state.LY / 128.0
        rx = self.dualsense.state.RX / 128.0
        ry = self.dualsense.state.RY / 128.0
        # Gamma map (simple)
        lx = np.sign(lx) * (abs(lx) ** self.config.GAMMA)
        ly = np.sign(ly) * (abs(ly) ** self.config.GAMMA)
        rx = np.sign(rx) * (abs(rx) ** self.config.GAMMA)
        ry = np.sign(ry) * (abs(ry) ** self.config.GAMMA)
        vx = -ly * self.config.MAX_VELOCITY
        vy = lx * self.config.MAX_VELOCITY
        vz = ry * (self.config.MAX_CLIMB if ry < 0 else self.config.MAX_DESCENT) if abs(ry) > 0 else 0.0
        yaw_rate = rx * self.config.MAX_YAW_RATE
        return vx, vy, vz, yaw_rate

    def calc_pursuit_velocities(self, bbox_center, frame_width, frame_height):
        if bbox_center is None:
            return 0.0, 0.0, 0.0, 0.0
        ref_x = frame_width // 2
        ref_y = int(frame_height * (1 - self.target_ratio))
        bbox_x, bbox_y = bbox_center
        error_x = bbox_x - ref_x
        error_y = bbox_y - ref_y
        error_norm = np.sqrt(error_x**2 + error_y**2) / (frame_height * 0.5)
        yaw_rate = self.pursuit_pid_yaw.update(error_x)
        throttle = np.clip(1 - (error_norm - 0.25) / 0.35, 0.0, 1.0)  # Simplified
        target_speed = self.forward_velocity * throttle
        target_speed = max(target_speed, 0.4)
        vx = self.pursuit_pid_forward.update(target_speed - self.current_body_velocities['vx'])
        vz = self.pursuit_pid_alt.update(max(error_y, 0.0))
        self.pursuit_debug_info = {
            'error_x': error_x,
            'error_y': error_y,
            'yaw_rate': yaw_rate,
            'vx': vx,
            'vz': vz,
            'throttle': throttle,
            'current_speed': self.current_body_velocities['vx'],
            'target_speed': target_speed,
            'error_norm': error_norm,
            'target_ratio_used': self.target_ratio,
            'bbox_is_large': self.current_bbox_is_large
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

def main(args=None):
    rclpy.init(args=args)
    config = DualSenseConfig()
    viewer = ImageViewer(config)
    rclpy.spin(viewer)
    viewer.dualsense.close()
    if viewer.video_writer:
        viewer.video_writer.release()
    cv2.destroyAllWindows()
    viewer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
