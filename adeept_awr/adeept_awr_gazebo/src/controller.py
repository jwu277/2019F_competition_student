#!/usr/bin/env python

import rospy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2

import numpy as np

from scipy import ndimage

#from PIL import Image as Im

class AdeeptAWRController:

    ## Anything that looks like it can be tuned probably can be tuned...

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

    
    def init_constants(self):

        # Camera Feed
        # y is inverted
        self.__IMG_WIDTH = 1280
        self.__IMG_HEIGHT = 720

        # drive
        # current setup: use bottom row of image
        self.__ROAD_CENTRE = (self.__IMG_WIDTH - 1) / 2.0
        self.__DRIVE_MARGIN = 70
        self.__GRAY_LOWER = 0x50
        self.__GRAY_UPPER = 0x5A
        self.__CROSSWALK_CUTOFF = 700
        self.__CROSSWALK_THRESH = 800 # total number of px in cutoff image
        self.__RED_THRESH = 0xF0
        self.__NOT_RED_THRESH = 0x10
        self.__BLACK_PIXEL_SUM_THRESH = 0xF0 # sum
        self.__CROSSWALK_DIFFERENCE_THRESH = 12 # number of pixels
        self.__CROSSWALK_PASSING_TIME = 1.5

        # turn
        # duty cycle: turn x out of y cycles (drive forward for the other y - x cycles)
        self.__TURN_DUTY_PERIOD = 3 # y
        self.__TURN_DUTY_VAL = 2 # x

        # wait
        self.__WAIT_TIME = 0.6 # can be tuned
        
        # white_border
        self.__WHITE_PIXEL_THRESH = 3 * 0xF0 # threshold for sum of BGR values to be white
        self.__WHITE_BORDER_CUTOFF = 450 # image cutoff
        self.__WHITE_BORDER_THRESH = 1000 # number of pixels needed

        # turn complete
        self.__TURN_TIME = 3.0


    def pub_vel_msg(self, lin, ang):

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = lin
        cmd_vel_msg.angular.z = ang

        self.vel_pub.publish(cmd_vel_msg)


    def reinit_state(self):

        # State timer
        self.__timer = rospy.get_time()

        # drive
        self.crosswalk_state = "free"
        self.last_crosswalk_image = None

        # turn
        self.turn_duty_counter = 0
        self.turn_timer = rospy.get_time()

        # wait
        self.waiting = True
        
        # Add more state variables

    
    def debug_img(self, img):
        cv2.circle(img, (200, self.__WHITE_BORDER_CUTOFF), 15, (0x00, 0x00, 0xff), thickness=-1)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def callback(self, msg):

        #print(rospy.get_time() - msg.header.stamp.secs)
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

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
        # drive out of starting position into outer loop
        elif self.__state_counter == 0:

            if self.white_border(img):
                self.__state_counter += 1
                self.reinit_state()
                return
            
            self.drive(img)
        # turn into outer loop
        elif self.__state_counter == 1:

            if self.turn_complete():
                self.__state_counter += 1
                self.reinit_state()
                return
            
            self.turn()
        # drive around outer loop
        elif self.__state_counter == 2:

            # TODO: add termination condition

            self.drive(img)

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
    def drive(self, img):

        # currently only detects one crosswalk per state. TODO: use debounce instead

        # TODO: debounce crosswalks
        if self.crosswalk_state == "waiting":
            if np.sum(np.sum(cv2.subtract(img, self.last_crosswalk_image), axis=2) > self.__BLACK_PIXEL_SUM_THRESH) <= self.__CROSSWALK_DIFFERENCE_THRESH:
                self.crosswalk_state = "passing"
                self.crosswalk_passing_timer = rospy.get_time()
            else:
                self.last_crosswalk_image = img
            return

        if self.crosswalk_state == "passing":
            self.pub_vel_msg(1, 0)
            if rospy.get_time() - self.crosswalk_passing_timer >= self.__CROSSWALK_PASSING_TIME:
                self.crosswalk_state = "free"
            return

        def at_crosswalk(v):
            return np.sum(np.logical_and(v[:,:,2] >= self.__RED_THRESH,
                np.logical_and(v[:,:,1] <= self.__NOT_RED_THRESH, v[:,:,0] <= self.__NOT_RED_THRESH))) >= self.__CROSSWALK_THRESH
        
        def is_gray(v):
            return np.logical_and((v >= self.__GRAY_LOWER), (v <= self.__GRAY_UPPER))

        if at_crosswalk(img[self.__CROSSWALK_CUTOFF:]):
            self.pub_vel_msg(0, 0)
            self.crosswalk_state = "waiting"
            self.last_crosswalk_image = img
            return
        
        img_line = img[-1]

        gray_bool = np.all(is_gray(img_line), axis=1)

        centroid =  ndimage.measurements.center_of_mass(gray_bool)[0] if np.any(gray_bool) else self.__ROAD_CENTRE

        if centroid - self.__ROAD_CENTRE < -self.__DRIVE_MARGIN:
            # steer left (ccw)
            self.pub_vel_msg(0, 1)
        elif centroid - self.__ROAD_CENTRE > self.__DRIVE_MARGIN:
            # steer right (cw)
            self.pub_vel_msg(0, -1)
        else:
            # drive forward
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
        return rospy.get_time() - self.turn_timer >= self.__TURN_TIME

        
if __name__ == '__main__':
    try:
        rospy.init_node('adeept_awr_controller', anonymous=True)
        ad = AdeeptAWRController(rospy.get_param('~src_topic'),rospy.get_param('~dst_topic'))
        rate = rospy.Rate(rospy.get_param('~publish_rate', 10))
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
