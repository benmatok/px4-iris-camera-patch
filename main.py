import cv2
import numpy as np
import threading
import queue
import time
import copy
import asyncio
import os
import argparse
from datetime import datetime

# Import Control Logic
import control
from control import (
    DualSenseConfig, FrameState, DroneState, PursuitState, ButtonState,
    VideoState, CacheState, InputState, ManualMode, AngleMode,
    handle_input_logic, HAS_MAVSDK, VelocityBodyYawspeed, Attitude, AttitudeRate, OffboardError
)

# Optional Dependencies
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("ROS 2 (rclpy) not found. Will use MockImageViewer if needed.")
    # Define dummy classes to prevent NameError in class definitions
    class Node: pass
    class Image: pass

try:
    from pydualsense import pydualsense
    HAS_DUALSENSE = True
except ImportError:
    HAS_DUALSENSE = False
    print("pydualsense not found. Will use MockDualSense.")


# --- Helper Functions (CV & Visualization) ---

def stabilize_frame(frame, pitch_rad, roll_rad, pitch0_rad=0.0, hfov_deg=110.0):
    height, width = frame.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), np.rad2deg(roll_rad), 1.0)
    stabilized = cv2.warpAffine(frame, rotation_matrix, (width, height))
    shift_y = int(height * (pitch_rad - pitch0_rad) / np.deg2rad(hfov_deg / 2))
    M = np.float32([[1, 0, 0], [0, 1, shift_y]])
    stabilized = cv2.warpAffine(stabilized, M, (width, height))
    return stabilized

def get_stabilized_frame(frame, frame_state, drone_state):
    if frame_state.show_stabilized:
        return stabilize_frame(frame, drone_state.current_pitch_rad, drone_state.current_roll_rad, pitch0_rad=np.deg2rad(0.0), hfov_deg=110)
    return frame.copy()

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
    else:
        # Fallback or Mock tracker if contrib not available
        class MockTracker:
            def init(self, image, bbox): return True
            def update(self, image): return True, (100, 100, 50, 50)
        return MockTracker()

# --- Mock Classes (Main specific) ---

class MockImageViewer:
    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.running = True
        self.thread = threading.Thread(target=self.generate_images, daemon=True)
        self.thread.start()

    def generate_images(self):
        print("Starting Mock Image Generation")
        while self.running:
            # Create a noisy image
            img = np.random.randint(0, 255, (800, 1280), dtype=np.uint8)
            # Add a white square to simulate a target for tracking
            cv2.rectangle(img, (600, 350), (680, 450), 255, -1)

            # Create a mock ROS message structure
            class MockHeader:
                def __init__(self):
                    now = time.time()
                    self.sec = int(now)
                    self.nanosec = int((now - self.sec) * 1e9)
                    self.stamp = self

            class MockImageMsg:
                def __init__(self, cv_img):
                    self.header = MockHeader()
                    self.height = cv_img.shape[0]
                    self.width = cv_img.shape[1]
                    self.encoding = 'mono8'
                    self.data = cv_img
                    self.cv_img_payload = cv_img

            msg = MockImageMsg(img)
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.data_queue.put((ts, 'IMAGE', msg))

            time.sleep(1/30.0) # 30 FPS

    def destroy_node(self):
        self.running = False

    def get_logger(self):
        class MockLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
        return MockLogger()

class MockDualSense:
    class MockState:
        def __init__(self):
            self.LX = 0; self.LY = 0; self.RX = 0; self.RY = 0
            self.L1 = False; self.L2Btn = False; self.R1 = False; self.R2Btn = False
            self.square = False; self.cross = False; self.circle = False; self.triangle = False
            self.DpadUp = False; self.DpadDown = False; self.DpadLeft = False; self.DpadRight = False

    def __init__(self):
        self.state = self.MockState()

    def init(self):
        pass
    def close(self):
        pass

# --- Global Class for MockSelf to ensure scope visibility ---

class MockSelf:
    def __init__(self, frame_state, drone_state, display_frame):
        self.frame_width = display_frame.shape[1]
        self.frame_height = display_frame.shape[0]
        self.frame_center_x = display_frame.shape[1] // 2
        self.frame_center_y = display_frame.shape[0] // 2
        self.current_bbox_center = frame_state.current_bbox_center
        self.current_velocities = (0, 0, 0, 0)
        self.center_proximity_threshold = 0.5
        self.landing_proximity_threshold = 0.2
        self.current_pitch_rad = drone_state.current_pitch_rad
        self.current_roll_rad = drone_state.current_roll_rad

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
    return show_video(config, display_frame, logger)

def handle_modes(frame_state, drone_state, button_state, pursuit_state, input_state, display_frame, width, height):
    bbox_size = 50
    if frame_state.bbox is None:
        frame_state.bbox = [width//2 - bbox_size//2, height//2 - bbox_size//2, bbox_size, bbox_size]

    # This logic calls the control module logic to update bbox
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
        if frame_state.tracker_instance is not None:
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
                        if frame_state.tracker_instance is not None:
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
        if frame_state.tracker_instance is not None:
            success = frame_state.tracker_instance.init(color_frame, tuple(frame_state.bbox))
        else:
            success = False

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
                        if frame_state.tracker_instance is not None:
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

    display_frame = draw_alignment_info(display_frame, pursuit_state, MockSelf(frame_state, drone_state, display_frame))

def show_video(config, display_frame, logger):
    if not config.SHOW_VIDEO:
        return
    # In headless env, imshow might fail or do nothing useful.
    # We'll try, but wrap it.
    try:
        cv2.imshow('Forward Camera Feed', display_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            logger.info('Shutting down')
            # Signal shutdown
            return False
    except cv2.error:
        pass # Ignore in headless
    return True

def handle_video_recording(config, video_state, display_frame, width, height, cache_state):
    if config.RECORD_VIDEO and video_state.video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_state.video_filename = os.path.join(cache_state.cache_dir, f"stabilized_video_{timestamp}.mp4")
        video_state.video_writer = cv2.VideoWriter(video_state.video_filename, fourcc, 30, (width, height))
    if config.RECORD_VIDEO and video_state.video_writer:
        video_state.video_writer.write(display_frame)

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
            ts = time.time()
            input_state = InputState(dualsense.state)
            data_queue.put((ts, 'INPUT', input_state))
            time.sleep(0.01) # 100 Hz
    finally:
        dualsense.close()

def mock_dualsense_worker(data_queue):
    mock_ds = MockDualSense()
    while True:
        ts = time.time()
        input_state = InputState(mock_ds.state)
        data_queue.put((ts, 'INPUT', input_state))
        time.sleep(0.01)

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
    while True: # Loop forever until interrupted
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
    # Determine if we should shutdown based on rclpy state or general interrupt
    check_ok = rclpy.ok if HAS_ROS else lambda: True

    while check_ok():
        try:
            timestamp, data_type, data = data_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if data_type == 'IMAGE':
            try:
                if HAS_ROS and isinstance(data, Image):
                    frame_state.current_frame = bridge.imgmsg_to_cv2(data, 'mono8')
                else:
                    # Mock data payload
                    if hasattr(data, 'cv_img_payload'):
                         frame_state.current_frame = data.cv_img_payload
                    else:
                         frame_state.current_frame = data

                should_continue = process_visuals(config, frame_state, button_state, video_state, cache_state, shared_input_state, drone_state, pursuit_state, control_mode, logger)
                if should_continue is False:
                    break
            except Exception as e:
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
    if HAS_ROS:
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
    bridge = CvBridge() if HAS_ROS else None

    # 1. ROS Node or Mock Viewer
    if HAS_ROS:
        viewer = ImageViewer(data_queue)
        ros_thread = threading.Thread(target=rclpy.spin, args=(viewer,), daemon=True)
        ros_thread.start()
        logger = viewer.get_logger()
    else:
        viewer = MockImageViewer(data_queue)
        # Mock viewer runs its own thread in init
        logger = viewer.get_logger()

    # 2. DualSense Thread
    if HAS_DUALSENSE:
        ds_thread = threading.Thread(target=dualsense_worker, args=(data_queue,), daemon=True)
        ds_thread.start()
    else:
        ds_thread = threading.Thread(target=mock_dualsense_worker, args=(data_queue,), daemon=True)
        ds_thread.start()


    # 3. MAVSDK Thread
    mav_thread = threading.Thread(target=run_mavsdk_thread, args=(frame_state, drone_state, pursuit_state, shared_input_state, config, logger, control_mode, data_queue), daemon=True)
    mav_thread.start()

    # 4. Processing Loop (Main Thread)
    try:
        processing_loop(data_queue, frame_state, drone_state, pursuit_state, button_state, video_state, cache_state, config, control_mode, logger, bridge, shared_input_state)
    except KeyboardInterrupt:
        pass
    finally:
        if video_state.video_writer:
            video_state.video_writer.release()
        cv2.destroyAllWindows()
        if HAS_ROS:
            viewer.destroy_node()
            rclpy.shutdown()
        else:
            viewer.destroy_node()

if __name__ == '__main__':
    main()
