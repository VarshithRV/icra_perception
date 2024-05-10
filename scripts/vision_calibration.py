#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PointStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import struct
import pyrealsense2 as rs
import image_geometry
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener, TransformException

######### Crop values of the image #########
X1 = 125 # left
Y1 = 50  # top 
X2 = 60  # right
Y2 = 200 # bottom
############################################
MASK_THRESHOLD = 2 # mask will become slimmer if the value is decreased, smaller => more precise range(0, 255)
############################################

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')
        
        self.get_logger().info("Image Processing Started")
        
        # Initialize a CvBridge to convert between ROS Image and OpenCV
        self.cv_bridge = CvBridge()
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_info = None
        self.depth_image = None
        self.counter1 = 0

        # Create a subscriber to receive Image messages
        self.depth_to_color_subscriber = self.create_subscription(
            Image, '/chest_camera/aligned_depth_to_color/image_raw', self.image_callback, 10
        )


    # Callback function for the Image message, finding the centroid of the DONs and the 3d point
    def image_callback(self, msg):
        try:
            if self.counter1==0:
                self.get_logger().info("Depth Image Received")
            self.counter1+=1
            
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            normalized_img_parent = cv2.normalize(
                cv_image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            cv2.imwrite('normalized_parent.png', normalized_img_parent) # use this to check the cv_bridge conversion               
            # cropping the image here
            cv_image = cv_image[Y1:-Y2,X1:-X2]
            # normalize this once more
            normalized_img = cv2.normalize(
                cv_image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            cv2.imwrite('normalized_cropped.png', normalized_img) # use this to validate the CROP values
            
            # create a mask for the boundary of the DON
            mask1 = np.zeros(normalized_img.shape, dtype=np.uint8)
            mask2 = np.zeros(normalized_img.shape, dtype=np.uint8)
            
            # divide the image into two parts and create a mask for each
            for x in range(0, normalized_img.shape[1]):
                for y in range(0, normalized_img.shape[0]):
                    if normalized_img[y][x] < MASK_THRESHOLD:
                        if y < (normalized_img.shape[0])/2:
                            mask1[y][x] = 255
                        else:
                            mask2[y][x] = 255
            cv2.imwrite('mask1.png', mask1) # use this to validate the MASK_THRESHOLD value
            cv2.imwrite('mask2.png', mask2) # use this to validate the MASK_THRESHOLD value
            
            # get the centroid
            M = cv2.moments(mask1)
            cx1 = int(M["m10"] / M["m00"]) + X1
            cy1 = int(M["m01"] / M["m00"]) + Y1
            M = cv2.moments(mask2)
            cx2 = int(M["m10"] / M["m00"]) + X1
            cy2 = int(M["m01"] / M["m00"]) + Y1
            
            cv2.circle(normalized_img_parent, (cx1, cy1), 5, (255, 0, 0), -1)
            cv2.circle(normalized_img_parent, (cx2, cy2), 5, (255, 0, 0), -1)
            cv2.imwrite('estimation.png', normalized_img_parent) # use this to validate the centroid estimation

        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = ImageProcessorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()