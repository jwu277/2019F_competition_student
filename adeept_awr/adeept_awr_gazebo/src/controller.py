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

        self.pid_bias = 0 # bias setpoint by certain amount

    
    def init_constants(self):

        # Camera Feed
        # y is inverted
        self.__IMG_WIDTH = 1280
        self.__IMG_HEIGHT = 720

        self.__HEIGHT_CENTRE = (self.__IMG_HEIGHT - 1) / 2.0

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
        self.__TURN_TIME = 2.7

        # don't drive when cnn is rendering
        # self.__CNN_PAUSE_TIME = 1.0
        # self.__CNN_DEBOUNCE_TIME = 5.0

        self.__PARKING_BLUE_PX_COUNT_THRESH = 60000 # number of parking blue pixels in frame
        self.__INNER_TURN_DELAY = 2.5
        # self.__INNER_TURN_GRAY_THRESH = 180000 # currently confined to bottom-left corner

        self.__TRUCK_GRAY_LOWER_THRESH = 0x70
        self.__TRUCK_GRAY_UPPER_THRESH = 0xD0


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

        # turn into inner ring timer
        self.inner_turn_timer = rospy.get_time()
        self.last_lp = 0

        # self.cnn_waiting = False
        # self.cnn_wait_timer = rospy.get_time()
        # self.cnn_debounce_timer = rospy.get_time()
        
        # Add more state variables

    
    def debug_img(self, img):
        cv2.circle(img, (200, 200), 15, (0x00, 0x00, 0xff), thickness=-1)
        cv2.circle(img, (200, 400), 15, (0x00, 0xff, 0x00), thickness=-1)
        cv2.circle(img, (200, 600), 15, (0xff, 0x00, 0x00), thickness=-1)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def gray_filter(self, img):
    #     s = np.sum(img, axis=2)
    #     return np.logical_and(s >= 3 * self.__GRAY_LOWER, s <= 3 * self.__GRAY_UPPER)

    def parking_blue_filter(self, img):
        return np.logical_and(np.logical_and(img[:,:,1] <= 0x18, img[:,:,2] <= 0x18), img[:,:,0] >= 0x60)

    def callback(self, msg):

        # looptime = rospy.get_time()

        # print(rospy.get_time() - msg.header.stamp.secs)

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Waiting state
        if self.waiting:
            self.wait()
            return

        # TODO: remove initial wait (kept rn for testing)
        if self.__state_counter == -1:

            if rospy.get_time() - self.__timer >= 0.1:
                self.__state_counter += 1
                self.reinit_state()
                self.plate_pub.publish(String("LM&JW,teampw,0,AB01")) # Publish initial msg
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

            # Turn into inner ring when ready
            # 1 is the last license plate before turning into inner ring
            if self.last_lp == 1:# and sum(self.license_spotted) >= 6:
                if np.sum(self.parking_blue_filter(img)) >= self.__PARKING_BLUE_PX_COUNT_THRESH:
                    self.inner_turn_timer = rospy.get_time()
                elif rospy.get_time() - self.inner_turn_timer > self.__INNER_TURN_DELAY:
                # elif np.sum(self.gray_filter(img[int(self.__HEIGHT_CENTRE):,:int(self.__WIDTH_CENTRE)])) >= self.__INNER_TURN_GRAY_THRESH:
                    # check botom-right corner
                    self.__state_counter += 1
                    self.reinit_state()
                    print("MAMA")
                    return

            self.drive(img)
        
        # elif self.__state_counter == 3:
        #     # drive up a bit
        #     if rospy.get_time() - self.__timer >= 0.1:
        #         self.__state_counter += 1
        #         self.reinit_state()
        #         return

        #     self.drive(img)

        elif self.__state_counter == 3:

            self.pid_bias = 71

            if rospy.get_time() - self.__timer > 4.0:
                print("FOO")
                self.__state_counter += 1
                self.reinit_state()
                self.pid_bias = 0
                return

            self.drive(img)
        
        elif self.__state_counter == 4:

            if self.truck_safe(img):
                self.__state_counter += 1
                self.reinit_state()
                return

            self.pub_vel_msg(0, 0)
        
        elif self.__state_counter == 5:

            if self.white_border(img):
                self.__state_counter += 1
                self.reinit_state()
                self.__TURN_TIME = 1.0
                return
            
            self.drive(img)
        
        elif self.__state_counter == 6:

            if self.turn_complete():
                self.__state_counter += 1
                self.reinit_state()
                return

            self.turn()
        
        elif self.__state_counter == 7:

            # inner ring driving

            self.drive(img)

        # TODO: complete


        # def next_state(lin, ang):

        #     self.__timer = rospy.get_time()
        #     self.__state_counter += 1

        # if rospy.get_time() - self.__timer >= self.__state_times[self.__state_counter]:
        #     next_state(*self.__state_speeds[self.__state_counter])

        # print(rospy.get_time() - msg.header.stamp.secs)
        # if rospy.get_time() - looptime > 0.1:
        #     print(rospy.get_time() - looptime)


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

        if centroid - (self.__WIDTH_CENTRE + self.pid_bias) < -self.__DRIVE_MARGIN:
            # steer left (ccw)
            self.pub_vel_msg(0, 1)
        elif centroid - (self.__WIDTH_CENTRE + self.pid_bias) > self.__DRIVE_MARGIN:
            # steer right (cw)
            self.pub_vel_msg(0, -1)
        else:
            # drive forward
            self.pub_vel_msg(1, 0)
        
        # tt = rospy.get_time()
        # madeit = False

        ## LICENSE PLATE DETECTION
        # TODO
        # ttt = rospy.get_time()
        if self.license_processor.license_finder(img):

            # ttt = rospy.get_time()
            # print("=======")

            # madeit = True

            # if self.cnn_waiting:
            #     if rospy.get_time() - self.cnn_wait_timer > self.__CNN_PAUSE_TIME:
            #         self.cnn_waiting = False
            #         self.cnn_debounce_timer = rospy.get_time()
            #     else:
            #         self.pub_vel_msg(0, 0)
            #     self.license_processor.savemem("_clear")
            # elif rospycnn.get_time() - self.cnn_debounce_timer > self.__CNN_DEBOUNCE_TIME:
            #     self.cnn_wait_timer = rospy.get_time()
            #     self.cnn_waiting = True
            
            # if not self.cnn_waiting:
            #     self.license_processor.savemem()    

            # Indexing:
            # Stallnum: 4, LP: 0123 (XY67)

            self.pub_vel_msg(0, 0)

            lp_chars = np.array(self.license_processor.parse_plate(self.license_processor.mem()))
            prediction = np.hstack((self.license_processor.predict_plate(lp_chars[0:2], True), self.license_processor.predict_plate(lp_chars[2:], False)))

            # print(prediction)

            def valid_prediction(pred):
                # Parking spot 49 to 56
                return ord(pred[4]) >= 49 and ord(pred[4]) <= 56 and ord(pred[3]) >= 48 and ord(pred[3]) <= 57 \
                    and ord(pred[2]) >= 48 and ord(pred[2]) <= 57 and ord(pred[1]) >= 65 and ord(pred[1]) <= 90 \
                    and ord(pred[0]) >= 65 and ord(pred[0]) <= 90

            if valid_prediction(prediction): #and not self.license_spotted[int(prediction[4]) - 1]:
                self.plate_pub.publish(String("LM&JW,teampw,{0},{1}{2}{3}{4}".format(
                    prediction[4], prediction[0], prediction[1], prediction[2], prediction[3])))
                self.license_spotted[int(prediction[4])] = True
                self.last_lp = int(prediction[4])
            
            self.inner_turn_timer = rospy.get_time()
            
            # if self.cnn_waiting:
            #     return

        # if rospy.get_time() - tt > 0.05:
        #     print(str(rospy.get_time() - tt) + ("M" if madeit else "k"))


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
    
    def turn_complete(self):
        return rospy.get_time() - self.turn_timer >= self.__TURN_TIME
    
    def truck_safe(self, img):

        def truck_gray_filter(im):

            def gray_filter(i):
                return np.logical_and(i[:,:,0] == i[:,:,1], i[:,:,1] == i[:,:,2])

            s = np.sum(im, axis=2)
            return np.logical_and(np.logical_and(s >= 3 * self.__TRUCK_GRAY_LOWER_THRESH, s <= 3 * self.__TRUCK_GRAY_UPPER_THRESH), gray_filter(im))

        gray_filtered = truck_gray_filter(img[int(self.__HEIGHT_CENTRE):]) # check bottom half of frame
        count = np.sum(gray_filtered)
        return count <= 8000

        
if __name__ == '__main__':
    try:
        rospy.init_node('adeept_awr_controller', anonymous=True)
        ad = AdeeptAWRController(rospy.get_param('~src_topic'), rospy.get_param('~dst_topic'), rospy.get_param('~dst_topic2'))
        rate = rospy.Rate(rospy.get_param('~publish_rate', 10))
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
