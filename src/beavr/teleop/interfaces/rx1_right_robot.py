#!/usr/bin/env python3
import time
import zmq
import numpy as np
from beavr.teleop.utils.network import EnhancedZMQKeypointSubscriber as ZMQKeypointSubscriber, EnhancedZMQKeypointPublisher as ZMQKeypointPublisher
from scipy.spatial.transform import Rotation as R

import logging

logger = logging.getLogger(__name__)


class RX1RightRobot:
    """
    RX1RightRobot is a Python class that:
      Acts as a bridge between the operator and the robot
      Streams the robot's end-effector pose to the operator and ROS
      Subscribes to the operator's end-effector pose and reset signal
      Subscribes to the robot's joint states from ROS
    """

    def __init__(self,
                 host,
                 ee_pose_op_sub,
                 ee_pose_op_pub,
                 reset_op_sub,
                 ee_pose_ros_sub,
                 ee_pose_ros_pub,
                 joint_state_ros_sub,
                 data_frequency=60,
                 debug=False):

        self.debug = debug
        self._current_ee_pose = None
        self._data_frequency = data_frequency
        
        # Note a pose is a cartesian homogeneous matrix

        # Subscriber for the end-effector pose from the operator
        self._ee_pose_op_sub = ZMQKeypointSubscriber(
            host = host, 
            port = ee_pose_op_sub,
            topic = 'endeff_coords'
        )

        # Publisher for the end-effector pose to the operator
        self._ee_pose_op_pub = ZMQKeypointPublisher(
            host = host, 
            port = ee_pose_op_pub
        )

        # Subscriber for the reset signal from the operator
        self._reset_op_sub = ZMQKeypointSubscriber(
            host = host,
            port = reset_op_sub,
            topic = 'reset'
        )

        # Publisher for the end-effector pose to ROS
        self._ee_pose_ros_pub = ZMQKeypointPublisher(
            host = host,
            port = ee_pose_ros_pub
        )

        # Subscriber for the end-effector pose from ROS
        self._ee_pose_ros_sub = ZMQKeypointSubscriber(
            host = host,
            port = ee_pose_ros_sub,
            topic = 'ee_pose'
        )

        # Subscriber for the joint states from ROS
        self._joint_state_ros_sub = ZMQKeypointSubscriber(
            host = host,
            port = joint_state_ros_sub,
            topic = 'joint_states'
        )

    def check_reset(self):
        try:
            reset_bool = self._reset_op_sub.recv_keypoints(flags=zmq.NOBLOCK)
            if reset_bool is not None:
                logger.info(f"Received data from reset subscriber: {reset_bool}")
                return True
        except zmq.Again:
            pass
        return False

    def stream(self):
        frame_count = 0
        start_time = time.time()
        last_fps_print = start_time
        
        while True:
            try:
                recv_coords = self._ee_pose_op_sub.recv_keypoints(flags=zmq.NOBLOCK)
                # Convert recv_coords from quaternion to axis-angle
                if recv_coords is not None:
                    self._current_ee_pose = recv_coords
                    frame_count += 1
                    current_time = time.time()

                    self._ee_pose_ros_pub.pub_keypoints(self._current_ee_pose, 'pose')
                    
                    if current_time - last_fps_print >= 5.0:
                        fps = frame_count / (current_time - last_fps_print)
                        if self.debug:
                            logger.info(f"Average FPS over last 5 seconds: {fps:.2f}")
                        frame_count = 0
                        last_fps_print = current_time
                    
                    if self.debug:
                        logger.info(recv_coords[3:])
                
                joint_states = self._joint_state_ros_sub.recv_keypoints(flags=zmq.NOBLOCK)
                if joint_states is not None:
                    self._current_joint_states = joint_states

                # Check for reset signal
                if self.check_reset():
                    self.send_robot_pose()
                
                # Sleep to maintain desired frequency
                time.sleep(1/self._data_frequency)
                
            except Exception as e:
                logger.error(f"[RX1Right] Error in stream loop: {e}")
                time.sleep(1)  # Wait a bit before retrying on error


    def robot_pose_aa_to_affine(self, pose_aa: np.ndarray) -> np.ndarray:
        """Converts a robot pose in axis-angle format to an affine matrix.
        Args:
            pose_aa (list): [x, y, z, ax, ay, az] where (x, y, z) is the position and (ax, ay, az) is the axis-angle rotation.
            x, y, z are in mm and ax, ay, az are in radians.
        Returns:
            np.ndarray: 4x4 affine matrix [[R, t],[0, 1]]
        """

        rotation = R.from_rotvec(pose_aa[3:]).as_matrix()
        # translation = np.array(pose_aa[:3]) / 1000
        translation = np.array(pose_aa[:3])

        return np.block([[rotation, translation[:, np.newaxis]],
                        [0, 0, 0, 1]])
    
    def send_robot_pose(self):
        """
        Called from external code in response to a 'reset' event or similar.
        This publishes the current end-effector pose under 'endeff_homo'.
        """
        logger.info("Sending robot pose")
        robot_ee_pose = self._ee_pose_ros_sub.recv_keypoints(flags=zmq.NOBLOCK)
        if robot_ee_pose is None:
            logger.error("[RX1Right] No end-effector pose available to send.")
            return

        try:
            # Convert the pose array directly to homogeneous matrix
            pose_homo = self.robot_pose_aa_to_affine(robot_ee_pose)

            if self.debug:
                logger.info("\n[RX1Right] Publishing Robot Pose:")
                logger.info(f"Position: {robot_ee_pose[:3]}")
                logger.info(f"Orientation: {robot_ee_pose[3:]}")

            self._ee_pose_op_pub.pub_keypoints(pose_homo, 'endeff_homo')
            logger.info("[RX1Right] Sent current EE pose as 'endeff_homo'.")
            
        except Exception as e:
            logger.error(f"[RX1Right] Error in send_robot_pose: {e}")

    def get_current_ee_pose(self):
        """Return last known end-effector pose dict."""
        return self._current_ee_pose

    def get_current_joint_states(self):
        """Return last known joint states dict."""
        return self._current_joint_states

    def cleanup(self):
        logger.info("[RX1Right] Cleaning up ZMQ connections...")
        if self._ee_pose_ros_sub:
            self._ee_pose_ros_sub.close()
        if self._joint_state_ros_sub:
            self._joint_state_ros_sub.close()
        if self._reset_op_sub:  # Add cleanup for reset subscriber
            self._reset_op_sub.close()

    @property
    def recorder_functions(self):
        """Define available recording functions for the robot"""
        return {
            'joint_states': self.get_current_joint_states,
            'cartesian_states': self.get_current_ee_pose,
            'position': lambda: self.get_current_joint_states()['position'] if self.get_current_joint_states() else None,
            'velocity': lambda: self.get_current_joint_states()['velocity'] if self.get_current_joint_states() else None,
            'effort': lambda: self.get_current_joint_states()['effort'] if self.get_current_joint_states() else None
        }

    @property
    def name(self):
        return 'rx1_right'

    @property
    def data_frequency(self):
        return self._data_frequency