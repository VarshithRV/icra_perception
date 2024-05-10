#!/usr/bin/env python3
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np
import image_geometry

class PixelTo3DNode(Node):
    def __init__(self):
        super().__init__('pixel_to_3d_node')
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/chest_camera/aligned_depth_to_color/camera_info',
            self.camera_info_callback,
            10
        )
        self.depth_image_sub = self.create_subscription(
            Image,
            '/chest_camera/aligned_depth_to_color/image_raw',
            self.depth_image_callback,
            10
        )
        self.cv_bridge = CvBridge()
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_info = None
        self.depth_image = None

    def camera_info_callback(self, msg):
        self.camera_model.fromCameraInfo(msg)
        self.camera_info = msg

    def depth_image_callback(self, msg):
        if self.camera_info is None:
            return  # Wait until camera info is received
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # print(self.depth_image)
        self.project_pixel_to_3d(320, 240)  # Example pixel (center of the image)

    def project_pixel_to_3d(self, x, y):
        if self.depth_image is None:
            return  # Wait until depth image is received
        depth = self.depth_image[y, x]  # Convert to meters
        if np.isnan(depth) or depth == 0:
            self.get_logger().warn("Invalid depth at pixel ({}, {})".format(x, y))
            return
        # Project the 2D pixel to 3D point in the camera frame
        point_3d = self.camera_model.projectPixelTo3dRay((x, y))
        point_3d = np.array(point_3d) * depth  # Scale the ray by the depth
        self.get_logger().info(f"3D point at pixel ({x}, {y}): {point_3d}")

def main(args=None):
    rclpy.init(args=args)

    node = PixelTo3DNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()