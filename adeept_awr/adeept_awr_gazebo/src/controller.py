#!/usr/bin/env python

import rospy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2

import numpy as np
from scipy import ndimage

import collections

from license_processing import LicenseProcessor

class AdeeptAWRController:

    ## Anything that looks like it can be tuned probably can be tuned...

    def __init__(self, src_camera_topic, dst_vel_topic, dst_plate_topic):

        self.__src_camera_topic = src_camera_topic
        self.__dst_vel_topic = dst_vel_topic
        self.__dst_plate_topic = dst_plate_topic

        if not self.__src_camera_topic:
          raise ValueError("source topic is an empty string")

        if not self.__dst_vel_topic:
          raise ValueError("dest topic is an empty string")

        self.camera_sub = rospy.Subscriber(self.__src_camera_topic, Image, self.callback, queue_size=1)
        self.vel_pub = rospy.Publisher(self.__dst_vel_topic, Twist, queue_size=1)
        self.plate_pub = rospy.Publisher(self.__dst_plate_topic, String, queue_size=1)

        # TODO Temporary: state_counter = -1 is initial wait
        self.__state_counter = -1

        self.init_constants()

        self.reinit_state()

        self.bridge = CvBridge()
        self.license_processor = LicenseProcessor()

        self.license_spotted = [False] * 8

    
    def init_constants(self):

        # Camera Feed
        # y is inverted
        self.__IMG_WIDTH = 1280
        self.__IMG_HEIGHT = 720

        # drive
        # current setup: use bottom row of image
        self.__WIDTH_CENTRE = (self.__IMG_WIDTH - 1) / 2.0
        self.__DRIVE_MARGIN = 70
        self.__GRAY_LOWER = 0x50
        self.__GRAY_UPPER = 0x5A
        self.__CROSSWALK_CUTOFF = 660
        self.__CROSSWALK_THRESH = 800 # total number of px in cutoff image
        self.__RED_THRESH = 0xF0
        self.__NOT_RED_THRESH = 0x10
        self.__BLACK_PIXEL_SUM_THRESH = 0xF0 # sum
        self.__CROSSWALK_DIFFERENCE_THRESH = 12 # number of pixels
        self.__CROSSWALK_PASSING_TIME = 2.0
        self.__CROSSWALK_INIT_DEBOUNCE_TIME = 0.6
        self.__MOTION_DEQUE_LENGTH = 2
        # waiting for motion, only detect motion between LEFT_CUTOFF and RIGHT_CUTOFF (inclusive)
        self.__WAITING_TRIGGER_LEFT_CUTOFF = int(self.__IMG_WIDTH * 0.4)
        self.__WAITING_TRIGGER_RIGHT_CUTOFF = self.__IMG_WIDTH - self.__WAITING_TRIGGER_LEFT_CUTOFF
        self.__CROSSWALK_MOVEMENT_THRESH = 300 # for getting out of waiting_init state
        # crosswalk align
        self.__CROSSWALK_ALIGN_CUTOFF = 550
        self.__CROSSWALK_ALIGN_THRESH = 200 # xy-moment threshold
        self.__CROSSWALK_ALIGN_DUTY_PERIOD = 8
        self.__CROSSWALK_ALIGN_DUTY_VAL = 1

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


    def reinit_crosswalk_state(self):
        # drive
        self.last_crosswalk_image = None
        # safe to proceed when every element in deque is True
        self.crosswalk_motion_deque = collections.deque([False] * self.__MOTION_DEQUE_LENGTH, self.__MOTION_DEQUE_LENGTH)
        self.align_counter = 0


    def reinit_state(self):

        # State timer
        self.__timer = rospy.get_time()

        self.crosswalk_state = "free"
        self.reinit_crosswalk_state()

        # turn
        self.turn_duty_counter = 0
        self.turn_timer = rospy.get_time()

        # wait
        self.waiting = True
        
        # Add more state variables

    
    def debug_img(self, img):
        cv2.circle(img, (200, 200), 15, (0x00, 0x00, 0xff), thickness=-1)
        cv2.circle(img, (200, 400), 15, (0x00, 0xff, 0x00), thickness=-1)
        cv2.circle(img, (200, 600), 15, (0xff, 0x00, 0x00), thickness=-1)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def callback(self, msg):

        # print(rospy.get_time() - msg.header.stamp.secs)
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

        def red_filter(v):
            return np.logical_and(v[:,:,2] >= self.__RED_THRESH, np.logical_and(v[:,:,1] <=
                self.__NOT_RED_THRESH, v[:,:,0] <= self.__NOT_RED_THRESH))

        if self.crosswalk_state == "aligning":
            
            # use xy-moment of red pixels in cutoff image

            red_img = red_filter(img[self.__CROSSWALK_ALIGN_CUTOFF:,:,:])
            red_weight = np.sum(red_img)

            # if not any(red_img), xy_moment would be zero
            y_centroid, x_centroid = ndimage.measurements.center_of_mass(red_img) if red_weight > 0 else (0, 0)

            x = np.broadcast_to(np.arange(red_img.shape[1]), red_img.shape)
            y = np.broadcast_to(np.arange(red_img.shape[0])[:, None], red_img.shape)
            xy_moment = np.sum(np.multiply(red_img, np.multiply(x - x_centroid, y - y_centroid)))
            
            # ignore this procedure of alignment if red_weight = 0
            normalized_xy = xy_moment / red_weight if red_weight > 0 else 0
            # print(normalized_xy)

            if (normalized_xy > self.__CROSSWALK_ALIGN_THRESH):
                # turn left (cw)
                self.pub_vel_msg(0, -1 * (self.align_counter < self.__CROSSWALK_ALIGN_DUTY_VAL))
                self.align_counter += 1
            elif (normalized_xy < -self.__CROSSWALK_ALIGN_THRESH):
                # turn right (ccw)
                self.pub_vel_msg(0, 1 * (self.align_counter < self.__CROSSWALK_ALIGN_DUTY_VAL))
                self.align_counter += 1
            else:
                self.pub_vel_msg(0, 0)
                self.crosswalk_state = "init_debounce"
                self.crosswalk_debounce_timer = rospy.get_time()
            
            if self.align_counter == self.__CROSSWALK_ALIGN_DUTY_PERIOD:
                self.align_counter = 0

            return


        if self.crosswalk_state == "init_debounce":
            if rospy.get_time() - self.crosswalk_debounce_timer >= self.__CROSSWALK_INIT_DEBOUNCE_TIME:
                self.last_crosswalk_image = img
                self.crosswalk_state = "waiting_init"
            return

        # wait for ped to cross i.e. make motion
        if self.crosswalk_state == "waiting_init":
            # TODO idea: confine search area to middle of camera view, since we want translational
            #   ped movement, not rotational
            if np.sum(np.sum(cv2.subtract(img, self.last_crosswalk_image)
                [:,self.__WAITING_TRIGGER_LEFT_CUTOFF:self.__WAITING_TRIGGER_RIGHT_CUTOFF + 1,:], axis=2) >
                self.__BLACK_PIXEL_SUM_THRESH) > self.__CROSSWALK_MOVEMENT_THRESH:
                self.crosswalk_state = "waiting"
            return
        
        if self.crosswalk_state == "waiting":
            if np.sum(np.sum(cv2.subtract(img, self.last_crosswalk_image), axis=2) >
                self.__BLACK_PIXEL_SUM_THRESH) <= self.__CROSSWALK_DIFFERENCE_THRESH:
                self.crosswalk_motion_deque.append(True)
                self.last_crosswalk_image = img
                if all(self.crosswalk_motion_deque):
                    self.crosswalk_state = "passing"
                    self.crosswalk_passing_timer = rospy.get_time()
            else:
                self.last_crosswalk_image = img
                self.crosswalk_motion_deque.append(False)
            return

        if self.crosswalk_state == "passing":
            self.pub_vel_msg(1, 0)
            if rospy.get_time() - self.crosswalk_passing_timer >= self.__CROSSWALK_PASSING_TIME:
                self.crosswalk_state = "free"
            return

        def at_crosswalk(v):
            return np.sum(red_filter(v)) >= self.__CROSSWALK_THRESH
        
        # def is_red(v):
        #     return np.logical_and(v[:,2] >= self.__RED_THRESH,
        #         np.logical_and(v[:,1] <= self.__NOT_RED_THRESH, v[:,0] <= self.__NOT_RED_THRESH))

        def is_gray(v):
            return np.all(np.logical_and((v >= self.__GRAY_LOWER), (v <= self.__GRAY_UPPER)), axis=1)

        if at_crosswalk(img[self.__CROSSWALK_CUTOFF:]):
            self.pub_vel_msg(0, 0)
            self.crosswalk_state = "aligning"
            self.reinit_crosswalk_state()
            return
        
        img_line = img[-1]

        gray_bool = is_gray(img_line)#np.logical_or(is_gray(img_line), is_red(img_line))

        centroid =  ndimage.measurements.center_of_mass(gray_bool)[0] if np.any(gray_bool) else self.__WIDTH_CENTRE

        if centroid - self.__WIDTH_CENTRE < -self.__DRIVE_MARGIN:
            # steer left (ccw)
            self.pub_vel_msg(0, 1)
        elif centroid - self.__WIDTH_CENTRE > self.__DRIVE_MARGIN:
            # steer right (cw)
            self.pub_vel_msg(0, -1)
        else:
            # drive forward
            self.pub_vel_msg(1, 0)
        
        ## LICENSE PLATE DETECTION
        # TODO
        # ttt = rospy.get_time()
        if self.license_processor.license_finder(img):

            lp_chars = np.array(self.license_processor.parse_plate(self.license_processor.mem()))
            prediction = self.license_processor.predict_plate(lp_chars)
            # print(prediction)

            # Plates are 0 indexed
            
            def valid_prediction(pred):
                # Parking spot 49 to 56
                return ord(pred[4]) >= 49 and ord(pred[4]) <= 56 and ord(pred[3]) >= 48 and ord(pred[3]) <= 57 \
                    and ord(pred[2]) >= 48 and ord(pred[2]) <= 57 and ord(pred[1]) >= 65 and ord(pred[1]) <= 90 \
                    and ord(pred[0]) >= 65 and ord(pred[0]) <= 90

            if True:# if valid_prediction(prediction): #and not self.license_spotted[int(prediction[4]) - 1]:
                self.plate_pub.publish(String("LM&JW,teampw,{0},{1}{2}{3}{4}".format(
                    prediction[4], prediction[0], prediction[1], prediction[2], prediction[3])))
                # self.license_spotted[int(prediction[4]) - 1] = True

        # print(rospy.get_time() - ttt)


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
        ad = AdeeptAWRController(rospy.get_param('~src_topic'), rospy.get_param('~dst_topic'), rospy.get_param('~dst_topic2'))
        rate = rospy.Rate(rospy.get_param('~publish_rate', 10))
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
