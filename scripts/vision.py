#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PointStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import threading as td
import pyrealsense2 as rs
import image_geometry
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener, TransformException
import time

######### Crop values of the image #########
X1 = 125 # left
Y1 = 50  # top 
X2 = 60  # right
Y2 = 200 # bottom
############################################
MASK_THRESHOLD = 2 # mask will become slimmer if the value is decreased, smaller => more precise range(0, 255)
############################################
FEEDBACK_THRESHOLD = 10 # threshold for the image change detection

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')
        
        self.get_logger().info("Image Processing Started")
        
        # Initialize a CvBridge to convert between ROS Image and OpenCV
        self.cv_bridge = CvBridge()
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_info = None
        self.depth_image = None
        self.color_image = None
        self.counter0 = 0
        self.counter1 = 0
        self.counter2 = 0

        # Create a TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create a subscriber to receive Image messages
        self.color_subscriber = self.create_subscription(
            Image, '/chest_camera/color/image_raw', self.color_callback, 10
        )

        # Create a subscriber to receive Image messages
        self.depth_to_color_subscriber = self.create_subscription(
            Image, '/chest_camera/aligned_depth_to_color/image_raw', self.image_callback, 10
        )

        # Creating a subscriber for camera info
        self.camera_info_sub = self.create_subscription(
            CameraInfo,'/chest_camera/aligned_depth_to_color/camera_info',self.camera_info_callback,10
        )

        # Create a publisher to send PointStamped messages
        self.point_publisher0 = self.create_publisher(PointStamped, '/don0_centroid_3d', 10)
        self.point_publisher1 = self.create_publisher(PointStamped, '/don1_centroid_3d', 10)


    # function that checks if the image has changed given two sets of average rgb values
    def has_image_changed(self, avg_rgb1, avg_rgb2):
        if avg_rgb1 is None or avg_rgb2 is None:
            return False
        # Calculate the difference between the average RGB values
        diff = np.linalg.norm(avg_rgb1 - avg_rgb2)
        return diff > FEEDBACK_THRESHOLD

    # function to get the average RGB values of the full image
    def get_avg_rgb(self):
        if self.color_image is None:
            return
        # get the average RGB values of the full image
        avg_rgb = np.mean(self.color_image, axis=(0, 1))
        return avg_rgb
        
    
    # Callback function for the Image message
    def color_callback(self, msg):
        if self.counter0 == 0:
            self.get_logger().info("Color Image Received")
        self.counter0 += 1
        self.color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.color_image = self.color_image[Y1:-Y2,X1:-X2]
        # write the image to check the cropping
        cv2.imwrite('color_image.png', self.color_image)


    # Callback function for the CameraInfo message
    def camera_info_callback(self, msg):
        self.camera_model.fromCameraInfo(msg)
        self.camera_info = msg
        if self.counter0 == 0:
            self.get_logger().info("Camera Info Received")
        self.counter0 += 1
    
    
    # projecting the pixel to 3d
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
        return point_3d

    
    # Transformer
    def transform_point(self, point:PointStamped, target_frame):
        try:
            # Get the latest transform from point.header.frame_id to target_frame
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                point.header.frame_id,
                rclpy.time.Time()
            )

            # Transform the point
            transformed_point = tf2_geometry_msgs.do_transform_point(point, transform)
            return transformed_point

        except TransformException as ex:
            self.get_logger().error(f"Transform failed: {ex}")
            return None
        


    # Callback function for the Image message, finding the centroid of the DONs and the 3d point
    def image_callback(self, msg):
        timer_now = time.time()
        try:
            if self.counter1==0:
                self.get_logger().info("Depth Image Received")
            self.counter1+=1
            if self.camera_info is None:
                return
            
            # make a pass through depth image
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            normalized_img_parent = cv2.normalize(
                cv_image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            # cv2.imwrite('normalized_parent.png', normalized_img_parent) # use this to check the cv_bridge conversion               
            # cropping the image here
            cv_image = cv_image[Y1:-Y2,X1:-X2]
            # normalize this once more
            normalized_img = cv2.normalize(
                cv_image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            # cv2.imwrite('normalized_cropped.png', normalized_img) # use this to validate the CROP values
            
            # create a mask for the boundary of the DON
            mask1 = np.zeros(normalized_img.shape, dtype=np.uint8)
            mask2 = np.zeros(normalized_img.shape, dtype=np.uint8)
            # if y is less than 240, then mask1 is 1
            for x in range(0, normalized_img.shape[1]):
                for y in range(0, normalized_img.shape[0]):
                    if normalized_img[y][x] < MASK_THRESHOLD:
                        if y < (normalized_img.shape[0])/2:
                            mask1[y][x] = 255
                        else:
                            mask2[y][x] = 255
            # cv2.imwrite('mask.png', mask) # use this to validate the MASK_THRESHOLD value
            
            # get the centroid
            M = cv2.moments(mask1)
            cx1 = int(M["m10"] / M["m00"]) + X1
            cy1 = int(M["m01"] / M["m00"]) + Y1
            M = cv2.moments(mask2)
            cx2 = int(M["m10"] / M["m00"]) + X1
            cy2 = int(M["m01"] / M["m00"]) + Y1
            
            # project the centroid to 3d
            point_3d1 = self.project_pixel_to_3d(cx1, cy1)
            point_3d2 = self.project_pixel_to_3d(cx2, cy2)
            
            point0= PointStamped()
            point0.header = msg.header
            point0.point.x = point_3d2[0]
            point0.point.y = point_3d2[1]
            point0.point.z = point_3d2[2]
            point0 = self.transform_point(point0, 'base_link')
            
            point1 = PointStamped()
            point1.header = msg.header
            point1.point.x = point_3d1[0]
            point1.point.y = point_3d1[1]
            point1.point.z = point_3d1[2]
            point1 = self.transform_point(point1, 'base_link')

            if self.counter2 == 0:
                self.get_logger().info(f"Publishing to /don0_centroid_3d and /don1_centroid_3d topics, Type : PointStamped")
            
            self.counter2 += 1
            self.point_publisher0.publish(point0)
            self.point_publisher1.publish(point1)

        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def spin_node(node):
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    node = ImageProcessorNode()
    # Create a thread to spin the node
    thread = td.Thread(target=spin_node, args=(node,), daemon=True)
    thread.start()
    
    # get the average RGB values
    avg_rgb1 = node.get_avg_rgb()

    time.sleep(2)

    avg_rgb2 = node.get_avg_rgb()

    # Check if the image has changed
    if node.has_image_changed(avg_rgb1, avg_rgb2):
        print("The image has changed")
    else:
        print("The image has not changed")


    thread.join()


if __name__ == '__main__':
    main()