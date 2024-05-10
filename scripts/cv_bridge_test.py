#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

######### Crop values of the image #########
X1 = 125 # left
Y1 = 65  # top 
X2 = 50  # right
Y2 = 200 # bottom
MASK_THRESHOLD = 100 # mask will become slimmer if the value is decreased, smaller => more precise
############################################

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')
        # Initialize a CvBridge to convert between ROS Image and OpenCV
        self.cv_bridge = CvBridge()
        # Create a subscriber to receive Image messages
        self.subscriber = self.create_subscription(
            Image, '/chest_camera/aligned_depth_to_color/image_raw', self.image_callback, 10
        )
        self.counter = 0



    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            normalized_img = cv2.normalize(
                cv_image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            
            if self.counter == 1:                
                cv2.imwrite('don.png', normalized_img)
                cv_image = cv_image[Y1:-Y2,X1:-X2]
                normalized_img = normalized_img[Y1:-Y2,X1:-X2]
                cv2.imwrite('don1.png', normalized_img)
                # Display the maximum and minimum of the nnormalized image
                self.get_logger().info(f"Max: {normalized_img.max()}, Min: {normalized_img.min()}")
                # normalize this once more
                normalized_img = cv2.normalize(
                    cv_image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                cv2.imwrite('don2.png', normalized_img)
                # create a mask for the boundary of the DON
                mask = np.zeros(normalized_img.shape, dtype=np.uint8)
                mask[normalized_img < MASK_THRESHOLD] = 255
                cv2.imwrite('mask.png', mask)
                # get the centroid
                M = cv2.moments(mask)
                cx = int(M["m10"] / M["m00"]) + X1
                cy = int(M["m01"] / M["m00"]) + Y1
                self.get_logger().info(f"Centroid: ({cx}, {cy})")
                # draw a circle at the centroid in the noarmalized image
                # make the normalized_img to BGR format
                normalized_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
                cv2.circle(normalized_img, (cx, cy), 5, (255, 0, 0), -1)
                cv2.imwrite('don3.png', normalized_img)


        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

        self.counter +=1


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