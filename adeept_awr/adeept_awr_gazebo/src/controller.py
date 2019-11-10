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

        # TODO Temporary: state_counter = -1 is initial wait
        self.__state_counter = -1

        self.__WAIT_TIME = 0.6

        self.reinit_state()

        # self.__timer = rospy.get_time()

        # # Number of seconds in n+1 to time spent in state n
        # # First two values = rest time before starting
        # self.__state_times = [0.0, 7.0, 1.27, 0.6, 1.05, 0.6, 4.2, 0.6, 0.55, 0.6, 0.5, 0.6, 0.55, 0.6, 3.0, 9999999.0]
        # # (lin, ang)
        # self.__state_speeds = [(0, 0), (1, 0), (0, 0), (0, 1), (0, 0), (1, 0), (0, 0), (0, 1), (0, 0),
        #     (1, 0), (0, 0), (0, 1), (0, 0), (1, 0), (0, 0)]


    def pub_vel_msg(self, lin, ang):

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = lin
        cmd_vel_msg.angular.z = ang

        self.vel_pub.publish(cmd_vel_msg)


    def reinit_state(self):

        self.waiting = True
        self.__timer = rospy.get_time()

        # Add more state variables


    def callback(self, msg):

        # Waiting state
        if self.waiting:
            self.wait()
            return

        # TODO: remove initial wait
        if self.__state_counter == -1:

            if rospy.get_time() - self.__timer >= 5.0:
                self.__state_counter += 1
                self.reinit_state()
                return

        elif self.__state_counter == 0:

            if self.white_border():
                self.__state_counter += 1
                self.reinit_state()
                return
            
            self.drive()
        
        elif self.__state_counter == 1:

            print("Hi")

            if self.turn_complete():
                self.__state_counter +=1
                self.reinit_state()
                return
        
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
        pass
    
    def wait(self):

        self.pub_vel_msg(0, 0)

        if rospy.get_time() - self.__timer >= self.__WAIT_TIME:
            self.waiting = False


    ############
    ## EVENTS ##
    ############

    def white_border(self):
        return False
    
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
