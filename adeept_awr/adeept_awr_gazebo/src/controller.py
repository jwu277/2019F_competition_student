#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class AdeeptAWRController:

    def __init__(self, src_camera_topic, dst_vel_topic):

        self.__src_camera_topic = src_camera_topic
        self.__dst_vel_topic = dst_vel_topic

        if not self.__src_camera_topic:
          raise ValueError("source topic is an empty string")

        if not self.__dst_vel_topic:
          raise ValueError("dest topic is an empty string")

        self.camera_sub = rospy.Subscriber(self.__src_camera_topic, Image, self.callback)
        self.vel_pub = rospy.Publisher(self.__dst_vel_topic, Twist, queue_size = 1)

    def callback(self, msg):

        lin = 0
        ang = 0.5

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = lin
        cmd_vel_msg.angular.z = ang

        self.vel_pub.publish(cmd_vel_msg)

if __name__ == '__main__':
    try:
        rospy.init_node('adeept_awr_controller', anonymous=True)
        ad = AdeeptAWRController(rospy.get_param('~src_topic'),rospy.get_param('~dst_topic'))
        rate = rospy.Rate(rospy.get_param('~publish_rate', 10))
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
