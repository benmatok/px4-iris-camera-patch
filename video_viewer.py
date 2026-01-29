import cv2
import rclpy
import asyncio
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError


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

        self.get_logger().info("Subscribed to /forward_camera/image_raw")

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")


class DroneController:
    def __init__(self):
        self.drone = System()
        self.is_offboard = False

    async def connect(self):
        await self.drone.connect(system_address="udp://:14540")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("PX4 connected")
                break

    async def arm_and_start_offboard(self):
        print("Arming")
        await self.drone.action.arm()
        await asyncio.sleep(1)

        print("Sending initial setpoint")
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )

        try:
            print("Starting offboard")
            await self.drone.offboard.start()
            self.is_offboard = True
        except OffboardError as e:
            print(f"Offboard failed: {e}")

    async def move_forward(self):
        if not self.is_offboard:
            return
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(1.0, 0.0, 0.0, 0.0)
        )


async def main_async():
    rclpy.init()
    viewer = VideoViewer()

    controller = DroneController()
    await controller.connect()
    await controller.arm_and_start_offboard()

    cv2.namedWindow("Gazebo Camera", cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(viewer, timeout_sec=0.01)

            if viewer.latest_frame is not None:
                cv2.imshow("Gazebo Camera", viewer.latest_frame)
                cv2.waitKey(1)

                # Example: always move forward slowly
                await controller.move_forward()

            await asyncio.sleep(0.05)

    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.shutdown()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()