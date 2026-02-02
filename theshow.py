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

# Add current directory to path to find vision module
sys.path.append(os.getcwd())
try:
    from vision.detector import RedObjectDetector
except ImportError:
    print("Error: Could not import vision.detector. Make sure you are in the repo root.")
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
TARGET_ALT = 50.0        # Meters (Relative)
HOVER_THRUST = 0.58      # Approximate hover thrust (0-1)
SCAN_YAW_RATE = 15.0     # deg/s
HOMING_PITCH = -12.0     # deg (Nose down to fly forward)
HOMING_THRUST = 0.52     # Slightly less than hover to descend while pitching down
Kp_YAW = 40.0            # Yaw P-gain
Kp_PITCH = 2.0           # Pitch P-gain (for stabilization)
Kp_ROLL = 2.0            # Roll P-gain
Kp_ALT = 0.05            # Altitude P-gain (Thrust)

class TheShow(Node):
    def __init__(self):
        super().__init__('the_show')
        self.bridge = CvBridge()
        self.latest_image = None
        self.detector = RedObjectDetector()

        # Subscribe to camera
        self.create_subscription(Image, '/forward_camera/image_raw', self.img_cb, 10)

        self.drone = System()

        # Telemetry State
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.altitude = 0.0
        self.connected = False

        # Logic State
        self.state = "INIT"
        self.target_yaw_rate = 0.0
        self.target_pitch = 0.0
        self.target_roll = 0.0
        self.target_thrust = HOVER_THRUST
        self.loops = 0

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
                self.roll = att.roll_deg
                self.pitch = att.pitch_deg
                self.yaw = att.yaw_deg

        async def alt_loop():
            async for pos in self.drone.telemetry.position():
                self.altitude = pos.relative_altitude_m

        await asyncio.gather(att_loop(), alt_loop())

    async def control_loop(self):
        # Arm and Start Offboard
        print("Arming...")
        try:
            await self.drone.action.arm()
        except:
            print("Arming failed (might be already armed).")

        print("Starting Offboard...")
        # Send initial setpoint
        await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, 0.0))

        try:
            await self.drone.offboard.start()
        except OffboardError as e:
            print(f"Offboard failed: {e}")
            return

        self.state = "TAKEOFF"
        print(f"State: {self.state} - Target Alt: {TARGET_ALT}m")

        cv2.namedWindow("The Show", cv2.WINDOW_NORMAL)

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.001)

                # Logic Update
                await self.update_logic()

                # Send Controls
                # Convert Desired Angles to Rates (P-Controller)
                # Roll/Pitch Rates to stabilize or reach target angle

                roll_rate = Kp_ROLL * (self.target_roll - self.roll)

                # If in Homing, we might drive Pitch Rate directly or use Angle Target
                # In this implementation, I set target_pitch in update_logic.
                pitch_rate = Kp_PITCH * (self.target_pitch - self.pitch)

                # Yaw Rate is usually set directly
                yaw_rate = self.target_yaw_rate

                thrust = self.target_thrust

                # Clamp
                thrust = max(0.0, min(1.0, thrust))

                await self.drone.offboard.set_attitude_rate(
                    AttitudeRate(roll_rate, pitch_rate, yaw_rate, thrust)
                )

                # Visualization
                if self.latest_image is not None:
                    disp = self.latest_image.copy()
                    cv2.putText(disp, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(disp, f"Alt: {self.altitude:.1f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if self.state == "HOMING":
                        center, area, _ = self.detector.detect(self.latest_image)
                        if center:
                            cv2.circle(disp, (int(center[0]), int(center[1])), 10, (0, 0, 255), 2)
                            cv2.putText(disp, f"Area: {area:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    cv2.imshow("The Show", disp)
                    cv2.waitKey(1)

                await asyncio.sleep(0.05) # 20Hz
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

    async def update_logic(self):
        if self.state == "TAKEOFF":
            self.target_roll = 0.0
            self.target_pitch = 0.0
            self.target_yaw_rate = 0.0

            # Altitude Control
            err_alt = TARGET_ALT - self.altitude
            self.target_thrust = HOVER_THRUST + Kp_ALT * err_alt

            # Transition
            if self.altitude >= TARGET_ALT - 2.0:
                print("Altitude Reached. Switching to SCAN.")
                self.state = "SCAN"

        elif self.state == "SCAN":
            # Hold Altitude
            err_alt = TARGET_ALT - self.altitude
            self.target_thrust = HOVER_THRUST + Kp_ALT * err_alt

            self.target_roll = 0.0
            self.target_pitch = 0.0
            self.target_yaw_rate = SCAN_YAW_RATE

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
            h, w, _ = self.latest_image.shape

            if not center:
                # Lost target, go back to scan? Or Scan briefly?
                print("Lost target...")
                self.target_yaw_rate = 0.0
                # Could switch back to SCAN or Hold
                # Let's switch back to SCAN after a moment?
                # For now, just hover/drift
                self.target_pitch = 0.0
                self.target_thrust = HOVER_THRUST
                return

            cx, cy = center

            # Yaw Control (Horizontal Center)
            err_x = (cx - w/2) / (w/2)
            self.target_yaw_rate = Kp_YAW * err_x

            # Pitch Control (Vertical Center / Forward Motion)
            # We want to maintain a forward dive.
            self.target_pitch = HOMING_PITCH

            # If target is too low (y large), pitch down more?
            # If target is too high (y small), pitch up (less neg)?
            # err_y = (cy - h/2) / (h/2)
            # self.target_pitch += 5.0 * err_y

            self.target_roll = 0.0

            # Thrust Control (Descent/Speed)
            # If area is large, we are close.
            if area > (w * h * 0.15):
                print("Target Reached! Stopping.")
                self.state = "DONE"

            self.target_thrust = HOMING_THRUST

        elif self.state == "DONE":
            self.target_pitch = 15.0 # Flare?
            self.target_thrust = 0.0 # Cut motors? Or Land?
            # Let's just Land
            self.target_yaw_rate = 0.0
            # MAVSDK Land will take over if we break loop, but here we just wait
            raise KeyboardInterrupt # Trigger cleanup/land

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
