import cv2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import time
try:
    from pydualsense import *
except ImportError:
    pass
import asyncio
try:
    from mavsdk import System
    from mavsdk.offboard import OffboardError, VelocityBodyYawspeed, Attitude, AttitudeRate
except ImportError:
    from control import MockDrone as System # MockDrone doesn't match System sig perfectly but logic uses instance
    # Wait, control.py has MockDrone but old_main uses System()
    # control.py MockDrone __init__ takes no args. System() takes no args.
    # old_main.DroneState does self.drone = System()
    # So we can alias it if we import it.
    from control import MockDrone, OffboardError, VelocityBodyYawspeed, Attitude, AttitudeRate
    class System(MockDrone): pass

import numpy as np
import threading
import os
import pickle
from datetime import datetime

# Import shared control logic
import control
from control import (
    DualSenseConfig, FrameState, DroneState, PursuitState, ButtonState,
    VideoState, CacheState, InputState, ManualMode, AngleMode,
    handle_input_logic, HAS_MAVSDK
)

# ROS is required for old_main.py generally, but for "test both" we might need mocking too.
# However, the user said "original unchanged main logic". The original had:
# import rclpy
# from rclpy.node import Node
# ...
# If I change it to use mocks, it's not "unchanged main logic".
# But if I don't, it won't run.
# The prompt says "old_main should be the original unchanged main logic, with control function delegated to control.py."
# This implies structure/threading model should be "unchanged" (i.e. single threaded monolithic loop driven by ROS timer),
# but implementation details of control should use the new module.
# So I will assume I should add `try-except` for ROS/etc to allow testing, otherwise "test both" is impossible.

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    class Node: pass
    class Image: pass

# Mocking for cv_bridge if ROS is not present but cv_bridge is installed as a pip package
# The pip package `cv_bridge` requires `sensor_msgs` which might be missing.
try:
    from cv_bridge import CvBridge, CvBridgeError
except ImportError:
    class CvBridge:
        def imgmsg_to_cv2(self, msg, encoding='passthrough'):
            # In mock mode, msg is already a numpy array (cv2 image)
            return msg
    class CvBridgeError(Exception): pass

# --- Helper Functions (CV) ---
# Keeping original CV functions or reusing? Original had them inline.
# Reusing `stabilize_frame` and `draw_alignment_info` from main/control might be cleaner,
# but original logic had specific implementations. I will copy the helpers from `main.py` (which were refactored)
# or keep them here if they differ.
# Actually, the refactored `control.py` has physics/math helpers.
# `old_main.py` needs CV helpers. I'll include them here to keep it self-contained logic-wise.

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
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    class MockTracker:
        def init(self, image, bbox): return True
        def update(self, image): return True, (100, 100, 50, 50)
    return MockTracker()

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

        # Dependency handling for DualSense
        try:
            self.dualsense = pydualsense()
            self.dualsense.init()
        except (ImportError, OSError):
            print("DualSense not found, using Mock")
            # We need a mock that mimics pydualsense structure
            class MockDS:
                class State:
                    LX=0; LY=0; RX=0; RY=0
                    L1=False; L2Btn=False; R1=False; R2Btn=False
                    square=False; cross=False; circle=False; triangle=False
                    DpadUp=False; DpadDown=False; DpadLeft=False; DpadRight=False
                def __init__(self): self.state = self.State()
                def init(self): pass
                def close(self): pass
            self.dualsense = MockDS()

        # Use State classes from control.py
        self.drone_state = DroneState()
        self.pursuit_state = PursuitState()
        self.frame_state = FrameState()
        self.button_state = ButtonState()
        self.video_state = VideoState()
        self.cache_state = CacheState()
        self.prev_bbox = None
        self.control_mode = ManualMode()

        # Start MAVSDK connection in thread
        threading.Thread(target=self.run_mavsdk_controller, daemon=True).start()
        # Timer for state machine (run every 0.01s)
        self.timer = self.create_timer(0.01, self.state_machine_callback_wrapper)

    def run_mavsdk_controller(self):
        asyncio.run(mavsdk_controller(self.frame_state, self.drone_state, self.pursuit_state, self.dualsense, self.config, self.get_logger(), self.control_mode))

    def image_callback(self, msg):
        try:
            self.frame_state.current_frame = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))

    def state_machine_callback_wrapper(self):
        # We need to wrap the dualsense state into InputState to pass to control functions
        # But `state_machine_callback` in old_main passed `dualsense` directly.
        # We should adapt the callback to use the `control` module functions which expect `InputState` or logic.
        # BUT, `old_main` logic for button toggles was inside `main.py`.
        # `control.py` now has `handle_input_logic`. We should delegate.

        # Create InputState snapshot
        input_state = InputState(self.dualsense.state)

        # Delegate input logic
        handle_input_logic(input_state, self.button_state, self.frame_state, self.pursuit_state, self.config, self.control_mode)

        state_machine_callback(self.config, self.frame_state, self.button_state, self.video_state, self.cache_state, input_state, self.drone_state, self.pursuit_state, self.prev_bbox, self.get_logger(), self.control_mode)

def state_machine_callback(config, frame_state, button_state, video_state, cache_state, input_state, drone_state, pursuit_state, prev_bbox, logger, control_mode):
    if frame_state.current_frame is None:
        return
    frame = frame_state.current_frame.copy()
    height, width = frame.shape[:2]
    frame_state.current_frame_height, frame_state.current_frame_width = height, width

    # Toggle Logic (handled by handle_input_logic now called in wrapper, but visualization might depend on state)

    display_frame = get_stabilized_frame(frame, frame_state, drone_state)

    if len(display_frame.shape) == 2:
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

    handle_video_recording(config, video_state, display_frame, width, height, cache_state)

    # BBox and tracking logic
    handle_modes(frame_state, drone_state, button_state, pursuit_state, input_state, display_frame, width, height)

    # Draw alignment if pursuit
    handle_pursuit_alignment(pursuit_state, drone_state, frame_state, display_frame)

    # Draw control mode
    display_frame = draw_control_mode(display_frame, control_mode)
    show_video(config, display_frame, logger)

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

def handle_modes(frame_state, drone_state, button_state, pursuit_state, input_state, display_frame, width, height):
    bbox_size = 50
    if frame_state.bbox is None:
        frame_state.bbox = [width//2 - bbox_size//2, height//2 - bbox_size//2, bbox_size, bbox_size]

    # Delegate to control module
    control.handle_bbox_controls(frame_state, input_state, pursuit_state, width, height)

    if pursuit_state.autonomous_pursuit_active:
        process_pursuit_mode(frame_state, drone_state, pursuit_state, display_frame)
    elif frame_state.tracking_mode:
        process_tracking_mode(frame_state, display_frame)
    elif frame_state.bbox_mode:
        process_bbox_mode(frame_state, display_frame)

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
        if frame_state.tracker_instance:
            success = frame_state.tracker_instance.init(color_frame, tuple(frame_state.bbox))
        else:
            success = False
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
                        if frame_state.tracker_instance:
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
        if frame_state.tracker_instance:
            success = frame_state.tracker_instance.init(color_frame, tuple(frame_state.bbox))
        else: success = False
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
                        if frame_state.tracker_instance:
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
    try:
        cv2.imshow('Forward Camera Feed', display_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            logger.info('Shutting down')
            rclpy.shutdown()
    except cv2.error: pass

async def mavsdk_controller(frame_state, drone_state, pursuit_state, dualsense, config, logger, control_mode):
    await drone_state.drone.connect(system_address="udp://:14540")
    async for state in drone_state.drone.core.connection_state():
        if state.is_connected:
            break

    # We pass data_queue=None or simple queue if needed, but old main did explicit loops.
    # Control.py logic needs to be called.
    # Since we don't have separate threads updating telemetry in old_main (it was async tasks in main thread event loop? No, it was threaded).
    # Original old_main had:
    # threading.Thread(target=self.run_mavsdk_controller).start()
    # async def mavsdk_controller(...):
    #    asyncio.create_task(attitude_tracker(drone_state)) ...
    # This implies the MAVSDK thread runs an asyncio loop which updates the state.

    # We must replicate the trackers here or use the ones from control?
    # Control.py doesn't have trackers (they were in main.py).
    # I should redefine them here as they were in original main.

    asyncio.create_task(attitude_tracker(drone_state))
    asyncio.create_task(body_velocity_tracker(drone_state))
    asyncio.create_task(actuator_tracker(drone_state))

    while HAS_ROS and rclpy.ok(): # Check ROS ok
        # Convert dualsense to InputState
        input_state = InputState(dualsense.state)
        setpoint_type, setpoint = control_mode.get_setpoint(input_state, config, pursuit_state, frame_state, drone_state)

        if drone_state.is_armed and drone_state.is_offboard:
            if setpoint_type == 'velocity':
                await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(*setpoint))
            elif setpoint_type == 'attitude':
                await drone_state.drone.offboard.set_attitude(Attitude(*setpoint))
            elif setpoint_type == 'attitude_rate':
                await drone_state.drone.offboard.set_attitude_rate(AttitudeRate(*setpoint))

        if input_state.L1 and not drone_state.is_armed:
            try:
                await arm_and_takeoff(drone_state, logger)
                drone_state.is_armed = True
            except Exception as e:
                print(f"Arm/takeoff failed: {e}")

        if input_state.L2Btn and drone_state.is_armed:
            pursuit_state.autonomous_pursuit_active = False
            await land_and_disarm(drone_state, logger)
            drone_state.is_armed = False
        await asyncio.sleep(0.01)

# Original trackers
async def actuator_tracker(drone_state):
    try:
        async for status in drone_state.drone.telemetry.actuator_output_status(0):
            thrusts = status.actuator[:4]
            drone_state.current_thrust = np.mean(thrusts) if thrusts else 0.0
    except: pass
async def attitude_tracker(drone_state):
    try:
        async for euler in drone_state.drone.telemetry.attitude_euler():
            drone_state.current_pitch_rad = np.deg2rad(euler.pitch_deg)
            drone_state.current_roll_rad = np.deg2rad(euler.roll_deg)
            drone_state.current_yaw_deg = euler.yaw_deg
    except: pass
async def body_velocity_tracker(drone_state):
    try:
        async for velocity in drone_state.drone.telemetry.velocity_ned():
            drone_state.current_body_velocities['vx'] = velocity.north_m_s
            drone_state.current_body_velocities['vy'] = velocity.east_m_s
            drone_state.current_body_velocities['vz'] = velocity.down_m_s
        async for angular in drone_state.drone.telemetry.attitude_angular_velocity_body():
            drone_state.current_body_velocities['vyaw'] = np.rad2deg(angular.yaw_rad_s)
    except: pass

async def arm_and_takeoff(drone_state, logger):
    print("Arming...")
    await drone_state.drone.action.arm()
    await asyncio.sleep(1)
    await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    try:
        await drone_state.drone.offboard.start()
        drone_state.is_offboard = True
    except OffboardError as e:
        print(f"Offboard start failed: {e}")
        return
    await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, -3.0, 0.0))
    await asyncio.sleep(5)
    await drone_state.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

async def land_and_disarm(drone_state, logger):
    await drone_state.drone.action.land()
    await asyncio.sleep(5)
    if drone_state.is_offboard:
        await drone_state.drone.offboard.stop()
        drone_state.is_offboard = False
    await drone_state.drone.action.disarm()
    await asyncio.sleep(2)
    await drone_state.drone.action.hold()

def main(args=None):
    if HAS_ROS:
        rclpy.init(args=args)

    if HAS_ROS:
        config = DualSenseConfig()
        viewer = ImageViewer(config)
        try:
            rclpy.spin(viewer)
        except KeyboardInterrupt:
            pass
        finally:
            if hasattr(viewer, 'dualsense'): viewer.dualsense.close()
            if viewer.video_state.video_writer:
                viewer.video_state.video_writer.release()
            cv2.destroyAllWindows()
            viewer.destroy_node()
            rclpy.shutdown()
    else:
        # Mock Run
        print("Running in Mock Mode (No ROS)")
        # We need a MockImageViewer logic here but driven by main thread loop because rclpy.spin isn't available.
        # But old_main architecture was `rclpy.spin`.
        # I'll simulate a spin loop.
        class MockViewer(ImageViewer):
            def __init__(self, config):
                # Init without ROS Node
                self.config = config
                self.bridge = CvBridge()
                # Mock subscription
                self.get_logger = lambda: type('obj', (object,), {'info': print, 'error': print})

                # Mock DualSense and State (copy paste from above or rely on imports)
                try:
                    self.dualsense = pydualsense()
                    self.dualsense.init()
                except:
                    class MockDS:
                        class State:
                            LX=0; LY=0; RX=0; RY=0
                            L1=False; L2Btn=False; R1=False; R2Btn=False
                            square=False; cross=False; circle=False; triangle=False
                            DpadUp=False; DpadDown=False; DpadLeft=False; DpadRight=False
                        def __init__(self): self.state = self.State()
                        def init(self): pass
                        def close(self): pass
                    self.dualsense = MockDS()

                self.drone_state = DroneState()
                self.pursuit_state = PursuitState()
                self.frame_state = FrameState()
                self.button_state = ButtonState()
                self.video_state = VideoState()
                self.cache_state = CacheState()
                self.prev_bbox = None
                self.control_mode = ManualMode()

                threading.Thread(target=self.run_mavsdk_controller, daemon=True).start()

            def destroy_node(self): pass

            def spin_once(self):
                # Simulate image callback
                img = np.random.randint(0, 255, (800, 1280), dtype=np.uint8)
                self.frame_state.current_frame = img
                # Call state machine
                self.state_machine_callback_wrapper()
                time.sleep(0.01)

        viewer = MockViewer(DualSenseConfig())
        try:
            while True:
                viewer.spin_once()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
