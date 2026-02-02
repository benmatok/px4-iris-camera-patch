import sys
import os
import asyncio
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mavsdk import System
from mavsdk.offboard import AttitudeRate, OffboardError

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from vision.detector import RedObjectDetector
    from vision.projection import Projector
    from ghost_dpc.ghost_dpc import PyDPCSolver
except ImportError as e:
    print(f"Error: Could not import project modules: {e}")
    sys.exit(1)

"""
================================================================================
INSTRUCTIONS: ADDING A RED OBJECT TO BAYLANDS WORLD
================================================================================
To run this scenario, you need a red object in the simulation environment.
1. Ensure Gazebo is running with the Baylands world.
2. In the Gazebo UI, go to the 'Insert' tab (left panel).
3. Find a simple shape like 'Box' or 'Sphere'.
4. Drag and drop it into the scene, ideally 20-50m away from the drone's spawn point.
5. Right-click the object -> 'Edit Model' (or use the Link Inspector).
6. Go to the 'Visual' tab, find 'Material', and set it to 'Gazebo/Red'.
   - Alternatively, use 'Gazebo/RedBright'.
7. Save/Apply if necessary.

Alternatively, you can use ROS2 to spawn a red box (if gazebo_ros is active):
ros2 run gazebo_ros spawn_entity.py -entity red_target -database unit_box -x 30 -y 30 -z 0.5 -R 0 -P 0 -Y 0
(Note: You might need to manually set the color to red in Gazebo after spawning
if the default model is white/grey).
================================================================================
"""

# Constants
TARGET_ALT = 50.0        # Meters (Relative) - Target Altitude
SCAN_YAW_RATE = 15.0     # deg/s (Override for scanning)
DT = 0.05                # 20Hz

class TheShow(Node):
    def __init__(self):
        super().__init__('the_show')
        self.bridge = CvBridge()
        self.latest_image = None
        self.detector = RedObjectDetector()

        # Projector Config (Matches video_viewer.py)
        # 1280x800, 110 deg FOV, 30 deg Tilt
        self.projector = Projector(width=1280, height=800, fov_deg=110.0, tilt_deg=30.0)

        # DPC Solver
        self.solver = PyDPCSolver()
        # Nominal Model (Mass ~3.33kg to match video_viewer logic, or adaptive?)
        # User requested GDPC because mass/thrust is unknown.
        # We use a bank of models or a robust nominal.
        # For now, sticking to the single model from video_viewer.py as a baseline.
        self.models = [{'mass': 3.33, 'drag_coeff': 0.3, 'thrust_coeff': 54.5}]
        self.weights = [1.0]
        self.last_action = {'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0}

        # Subscribe to camera
        self.create_subscription(Image, '/forward_camera/image_raw', self.img_cb, 10)

        self.drone = System()

        # Telemetry State (MAVSDK)
        self.pos_ned = None
        self.vel_ned = None
        self.att_euler = None
        self.connected = False

        # Logic State
        self.state = "INIT"
        self.loops = 0
        self.dpc_target = [0.0, 0.0, -TARGET_ALT] # Default target

    def img_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    async def connect_drone(self):
        print("Connecting to drone on udp://:14540...")
        await self.drone.connect(system_address="udp://:14540")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Drone Connected!")
                self.connected = True
                break

        # Start Telemetry Tasks
        asyncio.create_task(self.telemetry_loop())

    async def telemetry_loop(self):
        async def att_loop():
            async for att in self.drone.telemetry.attitude_euler():
                self.att_euler = att

        async def pos_vel_loop():
            async for pv in self.drone.telemetry.position_velocity_ned():
                self.pos_ned = pv.position
                self.vel_ned = pv.velocity

        await asyncio.gather(att_loop(), pos_vel_loop())

    def get_dpc_state(self):
        if self.pos_ned is None or self.vel_ned is None or self.att_euler is None:
            return None

        # Convert MAVSDK state to DPC state dict
        # MAVSDK Euler: deg. DPC: rad.
        # MAVSDK NED: m, m/s. Matches DPC physics.
        return {
            'px': self.pos_ned.north_m,
            'py': self.pos_ned.east_m,
            'pz': self.pos_ned.down_m, # NED Z (Positive Down)
            'vx': self.vel_ned.north_m_s,
            'vy': self.vel_ned.east_m_s,
            'vz': self.vel_ned.down_m_s,
            'roll': math.radians(self.att_euler.roll_deg),
            'pitch': math.radians(self.att_euler.pitch_deg),
            'yaw': math.radians(self.att_euler.yaw_deg)
        }

    async def control_loop(self):
        # Arm and Start Offboard
        print("Arming...")
        try:
            await self.drone.action.arm()
        except:
            print("Arming failed (might be already armed).")

        print("Wait for Telemetry...")
        while self.pos_ned is None:
            await asyncio.sleep(0.1)

        print("Starting Offboard...")
        # Send initial setpoint
        await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, 0.0))

        try:
            await self.drone.offboard.start()
        except OffboardError as e:
            print(f"Offboard failed: {e}")
            return

        self.state = "TAKEOFF"

        # Set initial target directly above current position
        start_x = self.pos_ned.north_m
        start_y = self.pos_ned.east_m
        self.dpc_target = [start_x, start_y, -TARGET_ALT]

        print(f"State: {self.state} - Target: {self.dpc_target}")

        cv2.namedWindow("The Show", cv2.WINDOW_NORMAL)

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.001)

                # 1. Get State
                dpc_state = self.get_dpc_state()
                if dpc_state is None:
                    await asyncio.sleep(0.01)
                    continue

                # 2. Logic Update (Determines Target)
                self.update_logic(dpc_state)

                # 3. Solve Control (GDPC)
                # action_out: {'thrust', 'roll_rate', 'pitch_rate', 'yaw_rate'}
                action_out = self.solver.solve(
                    dpc_state,
                    self.dpc_target,
                    self.last_action,
                    self.models,
                    self.weights,
                    DT
                )
                self.last_action = action_out

                # 4. Apply Overrides (Scanning)
                roll_rate = math.degrees(action_out['roll_rate'])
                pitch_rate = math.degrees(action_out['pitch_rate'])
                yaw_rate = math.degrees(action_out['yaw_rate'])
                thrust = action_out['thrust']

                if self.state == "SCAN":
                    yaw_rate = SCAN_YAW_RATE

                # Clamp Thrust
                thrust = max(0.0, min(1.0, thrust))

                # 5. Send to Drone
                await self.drone.offboard.set_attitude_rate(
                    AttitudeRate(roll_rate, pitch_rate, yaw_rate, thrust)
                )

                # 6. Visualization
                if self.latest_image is not None:
                    disp = self.latest_image.copy()
                    cv2.putText(disp, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(disp, f"Alt: {-dpc_state['pz']:.1f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(disp, f"Tgt: [{self.dpc_target[0]:.1f}, {self.dpc_target[1]:.1f}, {self.dpc_target[2]:.1f}]", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                    if self.state == "HOMING":
                        center, area, _ = self.detector.detect(self.latest_image)
                        if center:
                            cv2.circle(disp, (int(center[0]), int(center[1])), 10, (0, 0, 255), 2)

                    cv2.imshow("The Show", disp)
                    cv2.waitKey(1)

                await asyncio.sleep(DT)
                self.loops += 1

        except KeyboardInterrupt:
            pass
        finally:
            print("Stopping...")
            cv2.destroyAllWindows()
            try:
                await self.drone.offboard.stop()
                await self.drone.action.land()
            except:
                pass

    def update_logic(self, state):
        current_alt = -state['pz'] # Up is negative Z in NED, so Alt is -pz

        if self.state == "TAKEOFF":
            # Check if reached altitude (within 2m)
            if current_alt >= TARGET_ALT - 2.0:
                print("Altitude Reached. Switching to SCAN.")
                self.state = "SCAN"
                # Keep same target (Hover)

        elif self.state == "SCAN":
            # Check Vision
            if self.latest_image is not None:
                center, area, bbox = self.detector.detect(self.latest_image)
                if center:
                    print("Target Detected! Switching to HOMING.")
                    self.state = "HOMING"

        elif self.state == "HOMING":
            if self.latest_image is None:
                return

            center, area, bbox = self.detector.detect(self.latest_image)

            if not center:
                # Lost target logic
                # For now, hold position (or previous target)
                return

            # Project to World
            # Center is (u, v)
            world_pt = self.projector.pixel_to_world(center[0], center[1], state)

            if world_pt:
                # Update DPC Target
                # Target: [ObjX, ObjY, -2.0 (Hover 2m above)]
                self.dpc_target = [world_pt[0], world_pt[1], -2.0]

                # Check for completion
                # If we are close to the target (XY) and altitude is low
                dist_xy = math.sqrt((state['px'] - world_pt[0])**2 + (state['py'] - world_pt[1])**2)
                dist_z = abs(state['pz'] - (-2.0))

                if dist_xy < 1.0 and dist_z < 1.0:
                     print("Target Reached! Stopping.")
                     self.state = "DONE"

        elif self.state == "DONE":
             # Just Land
             raise KeyboardInterrupt

def main():
    rclpy.init()
    node = TheShow()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(node.connect_drone())
    loop.run_until_complete(node.control_loop())

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
