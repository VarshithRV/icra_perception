#!/usr/bin/env python3
from  rclpy.node import Node
import rclpy
from sensor_msgs.msg import Image, PointCloud2
import cv2 
import time
import numpy as np
import struct

######### Crop values of the image #########
X1 = 125 # left
Y1 = 40  # top 
X2 = 50  # right
Y2 = 200 # bottom
############################################

class DepthSubscriberNode(Node):
    def __init__(self):
        
        super().__init__("depth_image_subscriber")
        self.subscription = self.create_subscription(PointCloud2, "/chest_camera/depth/color/points", self.listener_callback, 10)
        self.received_message = PointCloud2()
    
    def listener_callback(self, msg:PointCloud2):
        # Get the width and height from the message
        width = msg.width
        height = msg.height
        point_step = msg.point_step
        data = msg.data
        # Find the offset of the 'z' field
        z_offset = None
        for field in msg.fields:
            if field.name == 'z':
                z_offset = field.offset
                break
        if z_offset is None:
            self.get_logger().error("No 'z' field found in PointCloud2 message")
            return
        # Initialize an array to store depth data
        depth_map = np.zeros((height, width), dtype=np.float32)
        # Extract the 'z' value for each point and place it in the 2D array
        for i in range(height):
            for j in range(width):
                point_index = i * width + j
                byte_offset = point_index * point_step + z_offset
                # Unpack the 'z' value from the byte data (as a 32-bit float)
                z_value = struct.unpack_from('f', data, byte_offset)[0]
                depth_map[i, j] = z_value
        
        # Example of processing or visualizing the depth map
        self.get_logger().info(f"Received depth map of shape: {depth_map.shape}")

        # Normalize the depth map to the range [0, 255]
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)
        
        # crop the image to the desired size
        depth_map = depth_map[Y1:-Y2,X1:-X2]


def main(args=None):
    rclpy.init(args=args)

    node = DepthSubscriberNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

main()