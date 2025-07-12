#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import atan2, acos, asin, sqrt, sin, cos, pi
from moveit_commander.conversions import pose_to_list
import numpy as np
from numpy import sin, cos
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool  # Added for magnet control

def all_close(goal, actual, tolerance):

  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True


class MoveGroupPythonIntefaceTutorial(object):
  
  def __init__(self):
    super(MoveGroupPythonIntefaceTutorial, self).__init__()
    
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    group_name = "ldsc_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    display_trajectory_publisher = rospy.Publisher('/real_robot_arm_joint',
                                                   Float64MultiArray,
                                                   queue_size=10)
    
    # Add magnet control publisher
    magnet_publisher = rospy.Publisher('/SetEndEffector', Bool, queue_size=10)

    planning_frame = move_group.get_planning_frame()
    group_names = robot.get_group_names()

    self.robot = robot
    self.scene = scene
    self.move_group = move_group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.magnet_publisher = magnet_publisher  # FIXED: Properly assign to self
    self.planning_frame = planning_frame
    self.group_names = group_names

    joint_angles = move_group.get_current_joint_values()
    self.joint_angles = joint_angles
    joint_msg = Float64MultiArray()
    joint_msg.data = joint_angles
    display_trajectory_publisher.publish(joint_msg)

  def set_magnet(self, state):
    """
    Control the electromagnet
    Args:
        state (bool): True to turn on magnet, False to turn off
    """
    magnet_msg = Bool()
    magnet_msg.data = state
    self.magnet_publisher.publish(magnet_msg)
    rospy.sleep(0.1)  # Small delay to ensure message is sent
    print(f"Magnet {'ON' if state else 'OFF'}")
    
  def go_to_joint_state(self, auto_magnet=True):
    
    move_group = self.move_group
    joint_angles = self.joint_angles

    joint_goal = move_group.get_current_joint_values()

    # FIXED: Handle both numpy array and list cases
    if hasattr(joint_angles[0], 'item'):
        joint_goal[0] = joint_angles[0].item()
        joint_goal[1] = joint_angles[1].item()
        joint_goal[2] = joint_angles[2].item()
        joint_goal[3] = joint_angles[3].item()
    else:
        joint_goal[0] = joint_angles[0]
        joint_goal[1] = joint_angles[1]
        joint_goal[2] = joint_angles[2]
        joint_goal[3] = joint_angles[3]

    move_group.go(joint_goal, wait=True)
    move_group.stop()

    # Check if we reached the goal
    current_joints = move_group.get_current_joint_values()
    goal_reached = all_close(joint_goal, current_joints, 0.01)
    
    if goal_reached and auto_magnet:
        print("Goal reached! Turning on magnet...")
        rospy.sleep(0.5)  # Wait a bit for arm to stabilize
        self.set_magnet(True)
    
    # Display current pose information
    current_pose = self.move_group.get_current_pose('link5').pose
    print ("current pose:")
    print ("x: %.5f" %current_pose.position.x)
    print ("y: %.5f" %current_pose.position.y)
    print ("z: %.5f" %current_pose.position.z)

    current_rpy = self.move_group.get_current_rpy('link5')
    print ("rol: %.5f" %current_rpy[0])
    print ("pit: %.5f" %current_rpy[1])
    print ("yaw: %.5f" %current_rpy[2])
    print ("")
    
    return goal_reached


def Your_IK(x, y, z, p = pi/2): 
    Xd = np.array([x, y, z], ndmin=2).T
    
    # Increase step size for faster convergence
    K = 0.15  # Increased from 0.05
    
    # Slightly relax error margin for faster convergence
    error_margin = 0.0015  # Increased slightly from 0.001
    
    # Better initial guess for faster convergence
    # Use previous solution as initial guess if available
    if hasattr(Your_IK, "last_solution") and Your_IK.last_solution is not None:
        joint_angle = Your_IK.last_solution.copy()
    else:
        joint_angle = np.array([0.5, 1.14, 0.93, 0.4], ndmin=2).T
    
    # Add maximum iterations to prevent infinite loops
    max_iterations = 50
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        j1, j2, j3, j4 = joint_angle.T[0]
        # kinematics
        x_ = (
             (33 * cos(j1) * sin(j2)) / 250
             + (6 * cos(p - j2 - j3) * (cos(j1) * cos(j2) * sin(j3) + cos(j1) * cos(j3) * sin(j2))) / 125
             + (cos(p - j2 - j3) * (cos(j1) * cos(j2) * cos(j3) - cos(j1) * sin(j2) * sin(j3))) / 250
             - (sin(p - j2 - j3) * (cos(j1) * cos(j2) * sin(j3) + cos(j1) * cos(j3) * sin(j2))) / 250
             - (6 * sin(j2 + j3 - p) * (cos(j1) * cos(j2) * cos(j3) - cos(j1) * sin(j2) * sin(j3))) / 125
             + (104 * cos(j1) * cos(j2) * sin(j3)) / 625
             + (104 * cos(j1) * cos(j3) * sin(j2)) / 625
             )
         
        y_ = (
              (33 * sin(j1) * sin(j2)) / 250
              + (6 * cos(p - j2 - j3) * (cos(j2) * sin(j1) * sin(j3) + cos(j3) * sin(j1) * sin(j2))) / 125
              + (cos(p - j2 - j3) * (cos(j2) * cos(j3) * sin(j1) - sin(j1) * sin(j2) * sin(j3))) / 250
              - (sin(p - j2 - j3) * (cos(j2) * sin(j1) * sin(j3) + cos(j3) * sin(j1) * sin(j2))) / 250
              - (6 * sin(j2 + j3 - p) * (cos(j2) * cos(j3) * sin(j1) - sin(j1) * sin(j2) * sin(j3))) / 125
              + (104 * cos(j2) * sin(j1) * sin(j3)) / 625
              + (104 * cos(j3) * sin(j1) * sin(j2)) / 625
              )
        z_ = (
              (33 * cos(j2)) / 250
              - (cos(p - j2 - j3) * (cos(j2) * sin(j3) + cos(j3) * sin(j2))) / 250
              - (6 * cos(p - j2 - j3) * (sin(j2) * sin(j3) - cos(j2) * cos(j3))) / 125
              - (104 * sin(j2) * sin(j3)) / 625
              - (6 * sin(p - j2 - j3) * (cos(j2) * sin(j3) + cos(j3) * sin(j2))) / 125
              + (sin(p - j2 - j3) * (sin(j2) * sin(j3) - cos(j2) * cos(j3))) / 250
              + (104 * cos(j2) * cos(j3)) / 625
              + 0.142
              )

        Xe = np.array([x_, y_, z_], ndmin=2).T

        # Reduce verbosity - only print every 10th iteration
        if iteration % 10 == 0:
            print("Target:", Xd.T[0])
            print("Current Joint Angles:", joint_angle.T[0])
            print("Current End Effector Position:", Xe.T[0])
            print("Iteration:", iteration)

        dist = np.linalg.norm(Xd-Xe)
        
        # Only print distance occasionally
        if iteration % 10 == 0:
            print("Distance to target:", dist)
            
        if dist < error_margin:
            break
            
        Ja = [
             [
             (6 * sin(j2 + j3 - p) * (cos(j2) * cos(j3) * sin(j1) - sin(j1) * sin(j2) * sin(j3))) / 125
             - (6 * cos(j2 + j3 - p) * (cos(j2) * sin(j1) * sin(j3) + cos(j3) * sin(j1) * sin(j2))) / 125
             - (cos(j2 + j3 - p) * (cos(j2) * cos(j3) * sin(j1) - sin(j1) * sin(j2) * sin(j3))) / 250
             - (sin(j2 + j3 - p) * (cos(j2) * sin(j1) * sin(j3) + cos(j3) * sin(j1) * sin(j2))) / 250
             - (33 * sin(j1) * sin(j2)) / 250
             - (104 * cos(j2) * sin(j1) * sin(j3)) / 625
             - (104 * cos(j3) * sin(j1) * sin(j2)) / 625,
        
               (33 * cos(j1) * cos(j2)) / 250
             + (104 * cos(j1) * cos(j2) * cos(j3)) / 625
             - (104 * cos(j1) * sin(j2) * sin(j3)) / 625,
        
               (104 * cos(j1) * cos(j2) * cos(j3)) / 625
             - (104 * cos(j1) * sin(j2) * sin(j3)) / 625,
        
               0
               ],
               [
               (33 * cos(j1) * sin(j2)) / 250
               + (6 * cos(j2 + j3 - p) * (cos(j1) * cos(j2) * sin(j3) + cos(j1) * cos(j3) * sin(j2))) / 125
               + (cos(j2 + j3 - p) * (cos(j1) * cos(j2) * cos(j3) - cos(j1) * sin(j2) * sin(j3))) / 250
               + (sin(j2 + j3 - p) * (cos(j1) * cos(j2) * sin(j3) + cos(j1) * cos(j3) * sin(j2))) / 250
               - (6 * sin(j2 + j3 - p) * (cos(j1) * cos(j2) * cos(j3) - cos(j1) * sin(j2) * sin(j3))) / 125
               + (104 * cos(j1) * cos(j2) * sin(j3)) / 625
               + (104 * cos(j1) * cos(j3) * sin(j2)) / 625,
        
                 (33 * cos(j2) * sin(j1)) / 250
               + (104 * cos(j2) * cos(j3) * sin(j1)) / 625
               - (104 * sin(j1) * sin(j2) * sin(j3)) / 625,
        
                 (104 * cos(j2) * cos(j3) * sin(j1)) / 625
               - (104 * sin(j1) * sin(j2) * sin(j3)) / 625,
        
                  0 
                ],
                [
                0,
                -(33 * sin(j2)) / 250
                - (104 * cos(j2) * sin(j3)) / 625
                - (104 * cos(j3) * sin(j2)) / 625,
        
                -(104 * cos(j2) * sin(j3)) / 625
                - (104 * cos(j3) * sin(j2)) / 625,
        
                  0
                ]
                ]

        Ja = np.array(Ja)
        
        # Use a more numerically stable pseudoinverse method
        # Add small regularization term for stability
        lambda_factor = 1e-6
        J_hash = np.matmul(Ja.T, np.linalg.inv(np.matmul(Ja, Ja.T) + lambda_factor * np.eye(3)))
        
        # Compute the update step
        update = K * np.matmul(J_hash, (Xd-Xe))
        
        # Add momentum term to speed up convergence
        if iteration > 1 and hasattr(Your_IK, "last_update"):
            momentum = 0.3  # Momentum coefficient
            update = update + momentum * Your_IK.last_update
        
        Your_IK.last_update = update.copy()
        
        # Update joint angles
        joint_angle = joint_angle + update
        joint_angle[3] = p - (joint_angle[1]+joint_angle[2])
    
    # Store solution for next call
    Your_IK.last_solution = joint_angle.copy()
    
    # FIX: Convert numpy arrays to plain Python floats
    result = joint_angle.flatten().tolist()
    print("Final joint angles:", result)
    return result

def main():
    try:
        path_object = MoveGroupPythonIntefaceTutorial()
        print("ctrl + z to close")
        print("Commands: 'on' = magnet on, 'off' = magnet off, or enter x,y,z,q coordinates")
        
        while not rospy.is_shutdown():
            try:
                user_input = input("Enter command: ").strip().lower()
                
                if user_input == "on":
                    path_object.set_magnet(True)
                elif user_input == "off":
                    path_object.set_magnet(False)
                else:
                    # Parse as position input or ask for coordinates
                    try:
                        # Try to parse as "x y z q" format
                        values = user_input.split()
                        if len(values) == 4:
                            x_input = float(values[0])
                            y_input = float(values[1])
                            z_input = float(values[2])
                            q_input = float(values[3])
                        else:
                            raise ValueError("Need individual inputs")
                    except:
                        # Ask for individual inputs
                        x_input = float(input("x: "))
                        y_input = float(input("y: "))
                        z_input = float(input("z: "))
                        q_input = float(input("q: "))
                    
                    path_object.joint_angles = Your_IK(x_input, y_input, z_input, q_input)
                    # This will automatically turn on magnet after reaching goal
                    path_object.go_to_joint_state()
                    
            except Exception as e:
                print(f"Error: {e}")
                '''go back to home if weird input'''
                path_object.joint_angles = [0, -pi/2, pi/2, 0]
                path_object.go_to_joint_state(auto_magnet=False)  # Don't turn on magnet when going home
                path_object.set_magnet(False)  # Turn off magnet

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == '__main__':
  main()