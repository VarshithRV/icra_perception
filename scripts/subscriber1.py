#!/bin/python3

import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__("minimal_subscriber") # node name
        self.get_logger().info("Hello from minimal subscriber")
        self.subscription = self.create_subscription(String,'topic',self.listener_callback,10)
        self.subscription
        self.received_message = String()
        for i in range(20):
            print(self.received_message)
            time.sleep(0.05)
    
    def listener_callback(self,msg:String):
        # self.get_logger().info("I heard %s" %msg.data)
        print("Callback activated")
        self.received_message= msg


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    for i in range(20):
            print(minimal_subscriber.received_message)
            time.sleep(0.05)
    rclpy.spin(minimal_subscriber) # While this is active, all the callbacks will be active
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()