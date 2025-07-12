#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray
import sys
from getkey import getkey  # Using the getkey library as mentioned in README

def keyboard():
    rospy.init_node("keyboard", anonymous=True)
    # publisher
    pub = rospy.Publisher('/real_robot_arm_joint', Float64MultiArray, queue_size = 10)

    #initialize
    joint_pos = [0.0, 0.0, 0.0, 0.0]

    print ("W/S: 1st joint rotates +/- 0.1 rad")
    print ("E/D: 2nd joint rotates +/- 0.1 rad")
    print ("R/F: 3rd joint rotates +/- 0.1 rad")
    print ("T/G: 4nd joint rotates +/- 0.1 rad")
    print ("Q: quit")

    rate = rospy.Rate(10) #10hz
    
    while not rospy.is_shutdown():
        key = getkey()  # Using getkey library
        
        # Process key input
        if key == 'w':
            joint_pos[0] += 0.1
            print("1st joint rotates +0.1 rad")
        elif key == 's':
            joint_pos[0] -= 0.1
            print("1st joint rotates -0.1 rad")
        elif key == 'e':
            joint_pos[1] += 0.1
            print("2nd joint rotates +0.1 rad")
        elif key == 'd':
            joint_pos[1] -= 0.1
            print("2nd joint rotates -0.1 rad")
        elif key == 'r':
            joint_pos[2] += 0.1
            print("3rd joint rotates +0.1 rad")
        elif key == 'f':
            joint_pos[2] -= 0.1
            print("3rd joint rotates -0.1 rad")
        elif key == 't':
            joint_pos[3] += 0.1
            print("4th joint rotates +0.1 rad")
        elif key == 'g':
            joint_pos[3] -= 0.1
            print("4th joint rotates -0.1 rad")
        elif key == 'q':
            print("Exiting...")
            break
        
        # Create and publish the message
        msg = Float64MultiArray()
        msg.data = joint_pos
        pub.publish(msg)
        
        rate.sleep()

if __name__== "__main__":
    try:
        keyboard()
    except rospy.ROSInterruptException:
        pass
