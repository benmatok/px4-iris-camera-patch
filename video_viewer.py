import cv2
import rclpy
import asyncio
import argparse
import numpy as np
import sys
import os
import math
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mavsdk import System
from mavsdk.offboard import (VelocityBodyYawspeed, AttitudeRate, OffboardError)

# Import Project Modules
sys.path.append(os.getcwd())
from vision.detector import RedObjectDetector
from vision.projection import Projector
from ghost_dpc.ghost_dpc import PyDPCSolver, PyGhostModel

class VideoViewer(Node):
    def __init__(self):
        super().__init__("video_viewer")
        self.bridge = CvBridge()
        self.latest_frame = None

        self.create_subscription(
            Image,
            "/forward_camera/image_raw",
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")

class DroneController:
    def __init__(self, mode):
        self.mode = mode
        self.drone = System()
        self.is_offboard = False

        # State
        self.pos_ned = None
        self.vel_ned = None
        self.att_euler = None

        # DPC Components
        self.solver = PyDPCSolver()
        # Single Nominal Model for Chimera CX10 (Mass ~3.33kg, Hover ~0.6)
        # thrust_coeff = Weight / HoverThrust = (3.33 * 9.81) / 0.6 = 54.5
        self.models = [{'mass': 3.33, 'drag_coeff': 0.3, 'thrust_coeff': 54.5}]
        self.weights = [1.0]

        # Vision Components
        self.detector = RedObjectDetector()
        # Camera Params from apply_patch.sh: 1280x800, 110 deg HFOV, 30 deg Tilt Up
        self.projector = Projector(width=1280, height=800, fov_deg=110.0, tilt_deg=30.0)

        # Control Targets
        self.target_pos = None # [x, y, z] (NED)
        self.last_action = {
            'thrust': 0.5, 'roll_rate': 0.0, 'pitch_rate': 0.0, 'yaw_rate': 0.0
        }

    async def connect(self):
        print("Connecting to drone...")
        await self.drone.connect(system_address="udp://:14540")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("PX4 connected!")
                break

        # Start Telemetry Tasks
        asyncio.create_task(self.telemetry_loop())

    async def telemetry_loop(self):
        # We need Position (NED), Velocity (NED), Attitude (Euler)
        # MAVSDK separates these streams.

        async def pos_vel_loop():
            async for pv in self.drone.telemetry.position_velocity_ned():
                self.pos_ned = pv.position
                self.vel_ned = pv.velocity

        async def att_loop():
            async for att in self.drone.telemetry.attitude_euler():
                self.att_euler = att

        await asyncio.gather(pos_vel_loop(), att_loop())

    async def arm_and_start_offboard(self):
        print("Arming...")
        await self.drone.action.arm()
        await asyncio.sleep(1)

        print("Setting initial setpoint...")
        # Send 0 rates, hover thrust (0.6)
        await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, 0.6))

        try:
            print("Starting Offboard...")
            await self.drone.offboard.start()
            self.is_offboard = True
        except OffboardError as e:
            print(f"Offboard failed: {e}")
            return False
        return True

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

    async def control_step(self, image):
        if not self.is_offboard:
            return

        state = self.get_dpc_state()
        if state is None:
            return

        # Initialize Target if None (for Hover)
        if self.target_pos is None:
            # Hold current XY, Target Z=5.0 (5m Altitude -> -5.0m NED)
            # Wait. "Target = [CurX, CurY, 5.0]".
            # If Z=5.0 means Altitude 5m (Up).
            # In NED, Altitude 5m is Z=-5.0.
            # But the plan says "Target = [CurX, CurY, 5.0]".
            # And "Drone climbs to 5.0m".
            # If I interpret 5.0 as Altitude, then Z=-5.0.
            # If I interpret as coordinate 5.0.
            # DPC Physics: pz is Z.
            # Usually we use NED.
            # If the user says "Climbs to 5.0m", they mean Altitude.
            # So I should target Z = -5.0.

            # Let's verify coordinate system of Physics.
            # drone.py: gravity pulls Z down.
            # So Z increases downwards.
            # If I want to fly UP to 5m altitude.
            # I should target Z = -5.0 (assuming ground is 0).
            # Or if ground is 0 and Z is Down. Ground is 0. Sky is Negative.

            self.target_pos = [state['px'], state['py'], -5.0]
            print(f"Initialized Hover Target: {self.target_pos}")

        # Update Target based on Mode
        if self.mode == "track" and image is not None:
            center, area, bbox = self.detector.detect(image)
            if center:
                # Project
                world_pt = self.projector.pixel_to_world(center[0], center[1], state)
                if world_pt:
                    # Update Target to be above the object?
                    # "Drone moves from (0,0) to approx (10,10)."
                    # "Visual Lock: The red object remains in the center".
                    # If we just output the World Point of the object as Target.
                    # The DPC Solver will try to go TO the target (Collision?).
                    # We want to hover ABOVE it or look at it.
                    # "The Solver ... minimizes Cost(GhostState, Target)".
                    # Cost function usually penalizes distance to target.
                    # If Target is the object at Z=0.
                    # The drone will try to hit the ground.
                    # We need to target a point ABOVE the object.
                    # e.g. Object (x, y, 0). Target (x, y, -5.0).

                    self.target_pos = [world_pt[0], world_pt[1], -5.0]

                    # Draw Debug
                    cv2.circle(image, (int(center[0]), int(center[1])), 5, (0, 255, 0), 2)
                    cv2.putText(image, f"Tgt: {self.target_pos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Solve Control
        dt = 0.05 # 20Hz

        # Current Action (Warm Start)
        action_in = self.last_action

        # Solve
        # Target format for solver: [x, y, z]
        action_out = self.solver.solve(state, self.target_pos, action_in, self.models, self.weights, dt)

        self.last_action = action_out

        # Send to MAVSDK
        # Convert rad/s to deg/s
        roll_rate_deg = math.degrees(action_out['roll_rate'])
        pitch_rate_deg = math.degrees(action_out['pitch_rate'])
        yaw_rate_deg = math.degrees(action_out['yaw_rate'])
        thrust = action_out['thrust']

        # MAVSDK expects thrust 0..1? Yes.
        # But wait, PX4 Actuator Control usually maps 0..1 to PWM.
        # Does 0.5 hover? Depends on thrust/weight ratio.
        # My model assumes mass=1.5, thrust_coeff=1.0 -> Max Thrust = 20N.
        # Hover Thrust = mg = 1.5 * 9.81 = 14.7N.
        # 14.7 / 20 = 0.735.
        # If I send 0.5, it will drop.
        # I should probably update mass or thrust_coeff to match reality or simulation.
        # "apply_patch.sh": base link mass 0.826kg.
        # If mass ~ 0.83 (plus battery?). Say 1.0kg.
        # Iris Max Thrust ~ 20-30N?
        # If mass=1.0. Hover = 9.8N.
        # If Max=20N. Hover = 0.5.
        # I used mass=1.5 in 'models'.
        # I should probably use mass=1.5 and let the solver figure it out.
        # If the solver thinks mass=1.5, it will ask for more thrust.

        await self.drone.offboard.set_attitude_rate(
            AttitudeRate(roll_rate_deg, pitch_rate_deg, yaw_rate_deg, thrust)
        )

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="hover", choices=["hover", "track"])
    args = parser.parse_args()

    rclpy.init()
    viewer = VideoViewer()

    controller = DroneController(args.mode)
    await controller.connect()

    # Wait for some telemetry
    print("Waiting for telemetry...")
    while controller.pos_ned is None:
        await asyncio.sleep(0.1)

    success = await controller.arm_and_start_offboard()
    if not success:
        return

    cv2.namedWindow("Gazebo Camera", cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(viewer, timeout_sec=0.01)

            if viewer.latest_frame is not None:
                img_display = viewer.latest_frame.copy()

                # Run Control
                await controller.control_step(img_display)

                cv2.imshow("Gazebo Camera", img_display)
                cv2.waitKey(1)
            else:
                # Run Control even if no image (for hover stability)
                await controller.control_step(None)

            await asyncio.sleep(0.05) # 20Hz

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.shutdown()
        # Clean up drone connection?
        try:
            await controller.drone.offboard.stop()
            await controller.drone.action.land()
        except:
            pass

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
