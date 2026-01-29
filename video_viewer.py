import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VideoViewer(Node):
    def __init__(self):
        super().__init__("video_viewer")

        # Bridge converts ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Subscribe to Gazebo forward camera topic
        self.subscription = self.create_subscription(
            Image,
            "/forward_camera/image_raw",
            self.image_callback,
            10
        )

        self.latest_frame = None
        self.get_logger().info("Subscribed to /forward_camera/image_raw")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_frame = frame



def main():
    # Initialize ROS client
    rclpy.init()

    viewer = VideoViewer()
    cv2.namedWindow("Gazebo Camera (OpenCV)", cv2.WINDOW_NORMAL)


    try:
        while rclpy.ok():
            # Process incoming ROS messages
            rclpy.spin_once(viewer, timeout_sec=0.01)

            # Display frame if available
            if viewer.latest_frame is not None:
                frame = viewer.latest_frame.copy()
                cv2.imshow("Gazebo Camera (OpenCV)", frame)
                cv2.waitKey(1)
                
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
