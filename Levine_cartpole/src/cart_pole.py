#!/usr/bin/env python
#
# Benjamin Levine
# SES 598: Autonomous Exploration Systems, SP22
# Dr. Jnaneshwar Das

import numpy as np
import scipy.linalg
import math
import rospy
import time

from std_msgs.msg import (UInt16, Float64)
from sensor_msgs.msg import JointState
from control_msgs.msg import JointControllerState
from std_srvs.srv import Empty
from gazebo_msgs.msg import LinkState
from geometry_msgs.msg import Point

# Define constants per Gazebo world
M = 20.
m = 2.
g = 9.8
l = 1.

# A and B matrices defined by course material
A = np.matrix([
     [0, 1, 0,                                          0],
     [0, 0, (-12 * m * g) / ((13 * M) + m),             0],
     [0, 0, 0,                                          1],
     [0, 0, (12 * g * (M + m)) / (1 + (13 * M + m)),    0]])
B = np.matrix([[0],
     [13 / (13 * M + m)],
     [0],
     [-12 / (1 + (13 * M + m))]])
Q = np.diag([1, 1, 10, 100])*50.
R = np.diag([0.1])

# Control toolbox is not available in Python2.
# LQR function provided by:
# https://www.mwm.im/lqr-controllers-with-python/
def lqr(A, B, Q, R):
    S = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    K = np.matrix(scipy.linalg.inv(R) * (B.T * S))
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    return K, S, eigVals

# Perform LQR calculation to determine gain
K, S, E = lqr(A, B, Q, R)

# Class for containing methods and procedures for LQR stabilization
class Cart_pole:
    def __init__(self):
        # Initialize cartpole Node, Publisher, and Subscribers
        rospy.init_node('Levine_cartpole', anonymous=True)

        self.pub_cart_vel = rospy.Publisher('/invpend/joint1_velocity_controller/command', Float64, queue_size=10)

        self.sub_invpend_states = rospy.Subscriber('/invpend/joint_states', JointState, self.update_joint_state)
        self.sub_pole = rospy.Subscriber('/invpend/joint2_position_controller/state', JointControllerState, self.update_pole)

        # Arrays to be populated with data. Pole spawns upwards so desired theta is 0rad
        self.current = np.array([0., 0., 0., 0.])
        self.desired = np.array([0., 0., 0., 0.])
        self.new_data = Float64()   # to publish to controller

    def update_joint_state(self, cart_pose):
        # Cart position and velocity
        self.current[0] = cart_pose.position[1]
        self.current[1] = cart_pose.velocity[1]

    def update_pole(self, pole_pose):
        # Pole theta and theta_dot
        self.current[2] = pole_pose.process_value
        self.current[3] = pole_pose.process_value_dot

    def stabilize(self):
        # Multiply error by gain to determine Publisher data
        self.new_data = np.matmul(K, (self.desired - self.current))
        self.pub_cart_vel.publish(self.new_data)


def main():
    system = Cart_pole()
    while not rospy.is_shutdown():
        system.stabilize()

if __name__ == '__main__':
    main()