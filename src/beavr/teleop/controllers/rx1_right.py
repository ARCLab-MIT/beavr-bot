#!/usr/bin/env python

import rospy
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from .dex_arm_control import DexArmControl
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

import logging

logger = logging.getLogger(__name__)


class RX1RosLink(DexArmControl):
    def __init__(self, record_type=None, robot_type='right'):
        """Initialize RX1 ROS Link"""
        if not rospy.get_node_uri():
            rospy.init_node('rx1_right_controller', anonymous=True)
        super().__init__(record_type, robot_type)
        
        # Define joint names (matching the controller's claimed resources)
        self.right_arm_joints = [
            'right_shoul_base2shoul_joint',
            'right_shoul2shoul_rot_joint',
            'right_arm2armrot_joint',
            'right_armrot2elbow_joint',
            'right_forearm2forearmrot_joint',
            'right_forearmrot2forearm_pitch_joint',
            'right_forearm_pitch2forearm_roll_joint'
        ]

        # Initialize action client for trajectory execution
        self.trajectory_client = actionlib.SimpleActionClient(
            '/right_arm_position_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        
        rospy.loginfo("Waiting for trajectory action server...")
        if not self.trajectory_client.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.logerr("Trajectory action server not available!")
            return
        rospy.loginfo("Connected to trajectory action server")

        # Subscribe to joint states
        self.joint_states = None
        rospy.Subscriber('/joint_states', JointState, self._joint_state_callback)
        
        # Wait for first joint state message
        rospy.loginfo("Waiting for joint states...")
        while self.joint_states is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Received joint states")

    def _joint_state_callback(self, msg):
        """Callback for joint states"""
        self.joint_states = msg

    def get_robot_position(self):
        """Get current joint positions"""
        if self.joint_states is None:
            return None
            
        positions = []
        for joint_name in self.right_arm_joints:
            idx = self.joint_states.name.index(joint_name)
            positions.append(self.joint_states.position[idx])
        return positions

    def get_robot_velocity(self):
        """Get current joint velocities"""
        if self.joint_states is None:
            return None
        velocities = []
        for joint_name in self.right_arm_joints:
            idx = self.joint_states.name.index(joint_name)
            velocities.append(self.joint_states.velocity[idx])
        return velocities

    def get_robot_torque(self):
        """Get current joint torques"""
        if self.joint_states is None:
            return None
        torques = []
        for joint_name in self.right_arm_joints:
            idx = self.joint_states.name.index(joint_name)
            torques.append(self.joint_states.effort[idx])
        return torques

    def move_robot(self, joint_angles):
        """Move robot to specified joint angles"""
        if len(joint_angles) != len(self.right_arm_joints):
            rospy.logerr(f"Expected {len(self.right_arm_joints)} joint angles, got {len(joint_angles)}")
            return False

        # Create a trajectory goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.right_arm_joints
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.velocities = [0.0] * len(joint_angles)
        point.accelerations = [0.0] * len(joint_angles)
        point.time_from_start = rospy.Duration(2.0)  # 2 second movement
        
        goal.trajectory.points.append(point)
        
        # Send goal and wait for result
        rospy.loginfo(f"Sending trajectory goal: {joint_angles}")
        self.trajectory_client.send_goal(goal)
        self.trajectory_client.wait_for_result(rospy.Duration(5.0))
        
        if self.trajectory_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Trajectory execution succeeded")
            return True
        else:
            rospy.logerr("Trajectory execution failed")
            return False

    def home_robot(self):
        """Move robot to home position"""
        return self.move_robot([0.0] * len(self.right_arm_joints))

    def reset_robot(self):
        """Reset robot to home position"""
        return self.home_robot()

    def arm_control(self, pose):
        """
        Control arm using Cartesian pose
        Args:
            pose: [x, y, z, qx, qy, qz, qw]
        """
        # Print detailed information about incoming pose
        rospy.loginfo("="*50)
        rospy.loginfo("Received Hand Keypoint Data:")
        rospy.loginfo("-"*50)
        rospy.loginfo("Position:")
        rospy.loginfo(f"  x: {pose[0]:.6f}")
        rospy.loginfo(f"  y: {pose[1]:.6f}")
        rospy.loginfo(f"  z: {pose[2]:.6f}")
        rospy.loginfo("Orientation (Quaternion):")
        rospy.loginfo(f"  qx: {pose[3]:.6f}")
        rospy.loginfo(f"  qy: {pose[4]:.6f}")
        rospy.loginfo(f"  qz: {pose[5]:.6f}")
        rospy.loginfo(f"  qw: {pose[6]:.6f}")
        rospy.loginfo("-"*50)
        
        # Create and fill pose message
        pose_msg = Pose()
        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]
        pose_msg.position.z = pose[2]
        pose_msg.orientation.x = pose[3]
        pose_msg.orientation.y = pose[4]
        pose_msg.orientation.z = pose[5]
        pose_msg.orientation.w = pose[6]
        
        # Log current robot state for comparison
        current_pos = self.get_robot_position()
        if current_pos:
            rospy.loginfo("Current Robot Joint States:")
            for joint, pos in zip(self.right_arm_joints, current_pos):
                rospy.loginfo(f"  {joint}: {pos:.6f} rad")
        
        rospy.loginfo("="*50)
