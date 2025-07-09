from beavr.teleop.components.environment.pybullet_base_env import PyBulletBaseEnv
from beavr.teleop.utils.network import ZMQCompressedImageTransmitter
from beavr.teleop.utils.images import rescale_image
from beavr.teleop.constants import VIZ_PORT_OFFSET

import pybullet as p
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation
import logging

from beavr.teleop.utils.logger import RobotLogger

logger = logging.getLogger(__name__)


class XArmEnv(PyBulletBaseEnv):
    def __init__(self, 
                 host,
                 camport,
                 timestamppublisherport,
                 endeff_publish_port,
                 endeffpossubscribeport,
                 robotposepublishport,
                 stream_oculus=False,
                 sim_frequency=30,
                 control_mode='relative',
                 logging_config=None):
        """Initialize P environment with all required parameters."""

        # Initialize variables before super().__init__
        self.name = "XArm_Sim"
        self.ref_link = 0
        self.ref_state = None
        self.end_effector_index = 7
        self.base_pose = None
        
        # Initialize logging config
        self.logging_config = logging_config or {"enabled": False}
        self.print_target_analysis = self.logging_config.get("print_target_analysis", False)
        self.print_workspace_warnings = self.logging_config.get("print_workspace_warnings", True)
        self.print_joint_changes = self.logging_config.get("print_joint_changes", False)

        self.robot_logger = None

        # Cache joint-related data
        self._joint_info = None
        self._lower_limits = None
        self._upper_limits = None
        
        # Add camera matrix cache
        self._view_matrix = None
        self._proj_matrix = None
        self._camera_width = 640  # Match Libero's resolution
        self._camera_height = 480
        
        # Call parent's __init__ which will call our load_assets()
        super().__init__(
            host=host,
            camport=camport,
            timestamppublisherport=timestamppublisherport,
            endeff_publish_port=endeff_publish_port,
            endeffpossubscribeport=endeffpossubscribeport,
            robotposepublishport=robotposepublishport,
            stream_oculus=stream_oculus,
            sim_frequency=sim_frequency
        )
        
        # Post-initialization setup
        self.control_mode = control_mode
        self.update_reference_frame()
        
        # Publish initial position
        logger.info("Publishing initial position...")
        initial_position = self.get_endeff_position()
        for _ in range(5):
            self.endeff_publisher.pub_keypoints(initial_position, 'endeff_coords')
            time.sleep(0.1)

        # Set default camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )

        # Get movable joint indices for the arm only (first 7 joints)
        self.movable_joint_indices = []
        for i in range(self.num_joints):
            if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED:
                self.movable_joint_indices.append(i)
        # Only keep the first 7 joints (the arm joints)
        self.movable_joint_indices = self.movable_joint_indices[:7]
        logger.info(f"Arm joint indices: {self.movable_joint_indices}")

        # Add visualization publisher for Oculus
        if stream_oculus:
            self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                host=host,
                port=camport + VIZ_PORT_OFFSET
            )
        self._stream_oculus = stream_oculus

        if self.robot_logger:
            self.robot_logger = RobotLogger()

    def load_assets(self):
        """Load robot and table assets."""
        logger.info("Loading XArm assets...")  # Debug print
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Print existing objects before loading
        logger.info(f"Objects before loading: {[p.getBodyInfo(i)[0].decode('utf-8') for i in range(p.getNumBodies())]}")
        
        # Load ground plane at z=0
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load table at origin first
        table_urdf_path = os.path.join(
            current_dir,
            "..",
            "..",
            "..",
            "..",
            "assets",
            "urdf",
            "table",
            "square_table.urdf"
        )

        self.table_height = 0.3  # From URDF: <box size="0.4 0.4 0.3"/>
        self.table_size = [0.4, 0.4, self.table_height]  # Length, Width, Height
        
        # Place table with its bottom at z=0 (on the plane)
        table_pos_z = self.table_height/2  # Offset up by half height since center is origin
        self.table_id = p.loadURDF(
            table_urdf_path,
            [0.0, 0.0, table_pos_z],  # Center of table offset up from plane
            [0, 0, 0, 1],
            useFixedBase=True
        )

        # Load robot on top of table
        xarm_urdf_path = os.path.join(
            current_dir,
            "..",
            "..",
            "..",
            "..",
            "assets",
            "urdf",
            "leap_xarm7",
            "leap_xarm7.urdf"
        )
        # Place robot at table surface: table_pos_z + half_table_height = table_height
        self.robot_id = p.loadURDF(
            xarm_urdf_path,
            [0.0, 0.0, self.table_height],  # Place robot at table surface height
            [0, 0, 0, 1],
            useFixedBase=True
        )

        # Store number of joints for later use
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Initialize robot state
        self.reset()
        
        # Print objects after loading
        logger.info(f"Objects after loading: {[p.getBodyInfo(i)[0].decode('utf-8') for i in range(p.getNumBodies())]}")

    def reset(self):
        """Reset robot state."""
        logger.info("Resetting XArm...")  # Debug print

        # Cache joint info and limits only once
        if self._joint_info is None:
            self._joint_info = self.get_joint_info(self.robot_id)
            self._lower_limits = [joint['lowerLimit'] for joint in self._joint_info]
            self._upper_limits = [joint['upperLimit'] for joint in self._joint_info]
        
        # Verify end effector position
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        pos = state[0]
        orn = state[1]

        # expected_pos = (0.20600000000000002, 3.4032245283784415e-06, 0.12050000000622334)
        # expected_orn = (0.9999999999932537, 0.0, 0.0, -3.6732051035825997e-06)
        
        # # Print warning if position is significantly different
        # if np.any(np.abs(np.array(pos) - np.array(expected_pos)) > 0.01) or np.any(np.abs(np.array(orn) - np.array(expected_orn)) > 0.01):
        #     print(f"Warning: End effector position {pos} and orientation {orn} differs from expected {expected_pos} and {expected_orn}")

        # Create initial pose array [x,y,z, qx,qy,qz,qw]
        initial_pose = np.concatenate([pos, orn])
        
        # Publish initial pose
        self.endeff_publisher.pub_keypoints(initial_pose, "endeff_coords")
        
        return initial_pose

    def update_reference_frame(self):
        """Update reference frame state from base link."""
        # Get base link state with full computation
        self.ref_state = p.getLinkState(self.robot_id, 
                                      self.ref_link,
                                      computeLinkVelocity=1,
                                      computeForwardKinematics=1)
        """
        ref_state contains:
        (linkWorldPosition, linkWorldOrientation,
        localInertialFramePosition, localInertialFrameOrientation,
        worldLinkFramePosition, worldLinkFrameOrientation, ...)
        """

    def get_endeff_position(self):
        """Get end effector position relative to reference frame (base link)."""
        # Get end effector state in world frame
        state = p.getLinkState(self.robot_id, 
                              self.end_effector_index,  # End effector link
                              computeLinkVelocity=1,
                              computeForwardKinematics=1)
        
        # Get positions and orientations
        pos_world = np.array(state[0])  # World position
        orn_world = state[1]  # World orientation (quaternion)
        ref_pos = np.array(self.ref_state[0])  # Reference frame position
        ref_orn = self.ref_state[1]  # Reference frame orientation
        
        # Convert world position to reference frame
        ref_rot = Rotation.from_quat(ref_orn)
        pos_rel = ref_rot.inv().apply(pos_world - ref_pos)
        
        # Convert world orientation to reference frame
        orn_rel = Rotation.from_quat(orn_world)
        rot_rel = ref_rot.inv() * orn_rel
        orn_rel = rot_rel.as_quat()
        
        return np.concatenate([pos_rel, orn_rel])

    def take_action(self):
        """Execute received action using IK."""
        action = self.endeff_pos_subscriber.recv_keypoints()

        target_pos = action[:3]
        target_quat = action[3:7]

        if self._joint_info is None:
            self._joint_info = self.get_joint_info(self.robot_id)
        
        # Get current joint positions as rest poses
        rest_poses = []
        for joint in self._joint_info:
            state = p.getJointState(self.robot_id, joint['id'])
            rest_poses.append(state[0])

        # Calculate IK with proper constraints
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_pos,
            target_quat,
            lowerLimits=self._lower_limits,
            upperLimits=self._upper_limits,
            jointRanges=[u-l for u,l in zip(self._upper_limits, self._lower_limits)],
            restPoses=rest_poses,
            maxNumIterations=100,
            residualThreshold=0.001
        )

        # # Check if IK solution is within joint limits
        # limits_violated = False
        # for i, angle in enumerate(joint_poses):
        #     if angle < lower_limits[i] or angle > upper_limits[i]:
        #         if self.print_workspace_warnings:
        #             print(f"Warning: Joint {i} exceeds limits: {angle:.2f} (limits: {lower_limits[i]:.2f}, {upper_limits[i]:.2f})")
        #         limits_violated = True
        #         break

        # # Only apply IK solution if within limits
        # if not limits_violated:
        for i, joint_idx in enumerate(self.movable_joint_indices):
                if i < len(joint_poses):
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_poses[i],
                        maxVelocity=1.0
                    )
        
        # Log only the arm joint data
        joint_data = list(zip(rest_poses, joint_poses[:7]))  # Ensure we only take arm joint poses
        if self.robot_logger:
            self.robot_logger.log_frame(self.get_endeff_position(), action, joint_data)
        
        # Step simulation
        self.step_simulation()
    
    def get_joint_info(self, robot_id):
        """Get information about robot joints"""
        joint_info = []
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                joint_info.append({
                    'id': info[0],
                    'name': info[1].decode('utf-8'),
                    'type': info[2],
                    'lowerLimit': info[8],
                    'upperLimit': info[9],
                    'maxForce': info[10],
                    'maxVelocity': info[11],
                })
        return joint_info


    def get_rgb_depth_images(self):
        """Get RGB and depth images from PyBullet camera with cached matrices."""
        # Initialize view and projection matrices if not cached
        if self._view_matrix is None:
            self._view_matrix = p.computeViewMatrix(
                cameraEyePosition=[1.2, 0, 1.0],    # Further back (x), same height (y), higher up (z)
                cameraTargetPosition=[0.0, 0, 0.3],  # Looking at lower point of workspace
                cameraUpVector=[0, 0, 1]            # Standard up vector
            )
        
        if self._proj_matrix is None:
            self._proj_matrix = p.computeProjectionMatrixFOV(
                fov=60.0,
                aspect=float(self._camera_width)/self._camera_height,
                nearVal=0.1,
                farVal=100.0
            )
        
        # Get camera images using cached matrices
        (_, _, rgb, depth, _) = p.getCameraImage(
            width=self._camera_width,
            height=self._camera_height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Process images
        rgb = np.array(rgb).reshape(self._camera_height, self._camera_width, 4)[:, :, :3]  # Remove alpha
        depth = np.array(depth).reshape(self._camera_height, self._camera_width)
        
        return rgb, depth, time.time()

    def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up network resources
            for publisher in self.publishers:
                publisher.stop()  # Use stop() instead of close()
            for subscriber in self.subscribers:
                subscriber.stop()
            
            # Close the robot logger
            if self.robot_logger:
                self.robot_logger.close()
            
            # Disconnect from PyBullet
            p.disconnect()
            
            logger.info("Successfully cleaned up XArmEnv resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def stream(self):
        """Stream environment state."""
        self.notify_component_start('{} environment'.format(self.name))
        logger.info("Start controlling the Simulation Arm using the Oculus Headset.\n")
        
        try:
            while True:
                self.timer.start_loop()
                
                # Get and publish images in one batch
                color_image, depth_image, timestamp = self.get_rgb_depth_images()
                
                # Batch all publishing operations
                publish_tasks = [
                    (self.rgb_publisher.pub_rgb_image, (color_image, timestamp)),
                    (self.timestamp_publisher.pub_keypoints, (timestamp, 'timestamps')),
                ]
                
                if self._stream_oculus:
                    scaled_image = rescale_image(color_image, 2)
                    publish_tasks.append((self.rgb_viz_publisher.send_image, (scaled_image,)))
                
                # Get and publish position
                position = self.get_endeff_position()
                publish_tasks.append(
                    (self.endeff_publisher.pub_keypoints, (position, 'endeff_coords'))
                )
                
                # Execute all publish tasks
                for task_func, task_args in publish_tasks:
                    task_func(*task_args)
                
                # Take action
                self.take_action()
                
                self.timer.end_loop()
                
        except KeyboardInterrupt:
            logger.info('Stopping the environment!')
            self.cleanup()