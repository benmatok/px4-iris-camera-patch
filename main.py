import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/forward_camera/image_raw',
            self.callback,
            10
        )
        self.get_logger().info('Subscribed to /forward_camera/image_raw')

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imshow('Forward Camera Feed', cv_image)
            cv2.waitKey(1)  # Refresh the window
        except CvBridgeError as e:
            self.get_logger().error(str(e))

def main(args=None):
    rclpy.init(args=args)
    viewer = ImageViewer()
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        viewer.get_logger().info('Shutting down')
    finally:
        viewer.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
