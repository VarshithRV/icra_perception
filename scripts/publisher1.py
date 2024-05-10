#!/bin/python3

import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String,'topic',10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.subscriber = self.create_subscription(String,"topic",callback=self.msgcb,qos_profile=10)

        print("waiting for 1.5 seconds here ....")
        time.sleep(1.5)

        # # publishing without using any timer callback
        # for i in range(10):
        #     msg = String()
        #     msg.data = "Hello world" + str(i)
        #     self.publisher_.publish(msg)
        #     self.get_logger().info("Publishing in the constructor : %s" %msg.data)
        #     time.sleep(0.5)
    
    def msgcb(self,msg:String):
        self.get_logger().info("Received message : %s" %msg.data)

    def timer_callback(self):
        msg=String()
        msg.data = "Hello world" + str(self.i)
        self.publisher_.publish(msg)
        self.get_logger().info("Publishing in the timer: %s" %msg.data)
        self.i +=1

def main(args=None ):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    
    while rclpy.ok():
        rclpy.spin_once(minimal_publisher, timeout_sec=0.1)

    minimal_publisher.destroy_node()
    
    print("Node finished execution, now  sleeping")
    rclpy.shutdown()

if __name__ == "__main__":
    main()