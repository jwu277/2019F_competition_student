#!/usr/bin/env python

import rospy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2

import numpy as np

#from PIL import Image as Im

class AdeeptAWRController:

    # tunable = please tune
    # can be tuned = not crucial to be tuned but can be tuned anyways

    def __init__(self, src_camera_topic, dst_vel_topic):

        self.__src_camera_topic = src_camera_topic
        self.__dst_vel_topic = dst_vel_topic

        if not self.__src_camera_topic:
          raise ValueError("source topic is an empty string")

        if not self.__dst_vel_topic:
          raise ValueError("dest topic is an empty string")

        self.camera_sub = rospy.Subscriber(self.__src_camera_topic, Image, self.callback, queue_size=1)
        self.vel_pub = rospy.Publisher(self.__dst_vel_topic, Twist, queue_size = 1)

        # TODO Temporary: state_counter = -1 is initial wait
        self.__state_counter = -1

        self.init_constants()

        self.reinit_state()

        self.bridge = CvBridge()

        self.temp = False

    
    def init_constants(self):

        # Camera Feed
        # y is inverted
        self.__IMG_WIDTH = 1280
        self.__IMG_HEIGHT = 720

        # turn
        # duty cycle: turn x out of y cycles (drive forward for the other y - x cycles)
        self.__TURN_DUTY_PERIOD = 3 # y
        self.__TURN_DUTY_VAL = 2 # x

        # wait
        self.__WAIT_TIME = 0.6 # can be tuned
        
        # white_border
        self.__WHITE_PIXEL_THRESH = 3 * 0xF0 # threshold for sum of BGR values to be white
        self.__WHITE_BORDER_CUTOFF = 450 # tunable, image cutoff
        self.__WHITE_BORDER_THRESH = 1000 # tunable, number of pixels


    def pub_vel_msg(self, lin, ang):

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = lin
        cmd_vel_msg.angular.z = ang

        self.vel_pub.publish(cmd_vel_msg)


    def reinit_state(self):

        # State timer
        self.__timer = rospy.get_time()

        # turn
        self.turn_duty_counter = 0

        # wait
        self.waiting = True
        
        # Add more state variables

    
    def debug_img(self, img):
        cv2.circle(img, (200, self.__WHITE_BORDER_CUTOFF), 10, (0x33, 0x99, 0xff), thickness=-1)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def callback(self, msg):

        #print(rospy.get_time() - msg.header.stamp.secs)
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Waiting state
        if self.waiting:
            self.wait()
            return

        # TODO: remove initial wait (kept rn for testing)
        if self.__state_counter == -1:

            if rospy.get_time() - self.__timer >= 4.0:
                self.__state_counter += 1
                self.reinit_state()
                return

        elif self.__state_counter == 0:

            if self.white_border(img):
                self.__state_counter += 1
                self.reinit_state()
                return
            
            self.drive()

        elif self.__state_counter == 1:

            if self.turn_complete():
                self.__state_counter += 1
                self.reinit_state()
                return
            
            self.turn()
        
        elif self.__state_counter == 2:

            # TODO: add termination condition (at corner)

            self.drive()

        # TODO: complete


        # def next_state(lin, ang):

        #     self.__timer = rospy.get_time()
        #     self.__state_counter += 1

        # if rospy.get_time() - self.__timer >= self.__state_times[self.__state_counter]:
        #     next_state(*self.__state_speeds[self.__state_counter])


    #############
    ## ACTIONS ##
    #############


    # TODO: "pid" control
    def drive(self):
        self.pub_vel_msg(1, 0)


    def turn(self):

        if self.turn_duty_counter < self.__TURN_DUTY_VAL:
            self.pub_vel_msg(0, 1)
        else:
            self.pub_vel_msg(1, 0)
        
        self.turn_duty_counter += 1

        if self.turn_duty_counter == self.__TURN_DUTY_PERIOD:
            self.turn_duty_counter = 0
    

    def wait(self):

        self.pub_vel_msg(0, 0)

        if rospy.get_time() - self.__timer >= self.__WAIT_TIME:
            self.waiting = False
        

    ############
    ## EVENTS ##
    ############

    def white_border(self, img):

        #return False

        # strategy: check if there is a white line close
        #    enough to bot (threshold = self.__WHITE_BORDER_CUTOFF)

        clipped_img = img[self.__WHITE_BORDER_CUTOFF:]

        # how many verticles slices in clipped_img have white pixels?
        # TODO: can update code to only check contiguous white regions
        #   this would be more complicated but arguably better
        tr = np.transpose(clipped_img, (1, 0, 2))
        
        px_sum = np.sum(tr, axis=2)

        maxed = np.max(px_sum, axis=1)

        count = np.sum(maxed >= self.__WHITE_PIXEL_THRESH)

        return count >= self.__WHITE_BORDER_THRESH
    
    def corner(self):
        return False
    
    def pedestrian(self):
        return False
    
    def pickup(self):
        return False
    
    def junction(self):
        return False
    
    def turn_complete(self):
        return False

        
if __name__ == '__main__':
    try:
        rospy.init_node('adeept_awr_controller', anonymous=True)
        ad = AdeeptAWRController(rospy.get_param('~src_topic'),rospy.get_param('~dst_topic'))
        rate = rospy.Rate(rospy.get_param('~publish_rate', 10))
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
