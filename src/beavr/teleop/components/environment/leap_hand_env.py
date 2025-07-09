from beavr.teleop.components.environment.pybullet_base_env_hand import PyBulletBaseEnvHand
import pybullet as p
import pybullet_data
import numpy as np
import os
import time

import logging

logger = logging.getLogger(__name__)


'''
This is an adaptation of the AVP LEAP Hand teleop to work with Meta Quest hand tracking.
It inherits from PyBulletBaseEnvHand and integrates Meta Quest hand tracking with LEAP Hand.
'''

# Meta Quest hand tracking joint indices
QUEST_JOINTS = {
    'metacarpals': [2, 6, 9, 12, 15],
    'knuckles': [6, 9, 12, 16],
    'thumb': [2, 3, 4, 5, 19],
    'index': [6, 7, 8, 20],
    'middle': [9, 10, 11, 21],
    'ring': [12, 13, 14, 22],
    'pinky': [15, 16, 17, 18, 23]
}

LEAP_HAND_JOINTS = [
    "base_joint",  # PyBullet base Joint
    "a", "b", "c", "d", "index_tip",  # index finger joints
    "e", "f", "g", "h", "middle_tip",  # middle finger joints
    "i", "j", "k", "l", "ring_tip",  # ring finger joints
    "n", "m", "o", "p", "thumb_tip",  # thumb finger joints
]

class LeapHandEnv(PyBulletBaseEnvHand):
    def __init__(self,
                 host,
                 camport,
                 transformed_keypoints_port,
                 jointanglepublishport,
                 jointanglesubscribeport,
                 timestamppublisherport,
                 endeff_publish_port,
                 endeffpossubscribeport,
                 actualanglepublishport,
                 stream_oculus=False,
                 sim_frequency=30):
        """
        Initialize LeapHand environment
        
        Args:
            host (str): Network host address
            camport (int): Port for camera streaming
            transformed_keypoints_port (int): Port for transformed keypoints
            jointanglepublishport (int): Port for joint angle publishing
            jointanglesubscribeport (int): Port for joint angle subscription
            timestamppublisherport (int): Port for timestamp publishing
            endeff_publish_port (int): Port for end effector position publishing
            endeffpossubscribeport (int): Port for end effector position subscription
            actualanglepublishport (int): Port for actual angle publishing
            stream_oculus (bool): Whether to stream to Oculus
            sim_frequency (int): Simulation frequency
        """
        
        # Call parent's __init__ 
        super().__init__(
            host=host,
            camport=camport,
            transformed_keypoints_port=transformed_keypoints_port,
            jointanglepublishport=jointanglepublishport,
            jointanglesubscribeport=jointanglesubscribeport,
            timestamppublisherport=timestamppublisherport,
            endeff_publish_port=endeff_publish_port,
            endeffpossubscribeport=endeffpossubscribeport,
            actualanglepublishport=actualanglepublishport,
            stream_oculus=stream_oculus,
            sim_frequency=sim_frequency
        )
        self.name = "LeapHand_Sim"
        
        # Set scaling factor for Meta Quest data
        self.glove_to_leap_mapping_scale = 1.1
        
        # Add PyBullet's data path to find built-in URDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load assets and initialize robot
        self.load_assets()

        # Set default camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )

        # Get movable joint indices
        # self.tip_joint_indices = [0]  # 0th joint is the base joint in PyBullet which does not move
        self.tip_joint_indices = []
        # for i in range(self.num_joints):
        #     joint_name = p.getJointInfo(self.hand_id, i)[1].decode("utf-8")  # Decode bytes to string
        #     if joint_name.endswith("_tip"):  # Exclude joints with "_tip" in the name
        #         self.tip_joint_indices.append(i)
        
        # Create array for movable joint indices
        self.movable_joint_indices = []
        for i in range(self.num_joints):
            if p.getJointInfo(self.hand_id, i)[2] != p.JOINT_FIXED and i not in self.tip_joint_indices:
                self.movable_joint_indices.append(i)
                
        # Define the end effector indices for IK
        self.leapEndEffectorIndex = [
            LEAP_HAND_JOINTS.index("c"),       # index middle joint
            LEAP_HAND_JOINTS.index("d"),   # index tip
            LEAP_HAND_JOINTS.index("g"),       # middle middle joint
            LEAP_HAND_JOINTS.index("h"),   # middle tip
            LEAP_HAND_JOINTS.index("k"),       # ring middle joint
            LEAP_HAND_JOINTS.index("l"),     # ring tip
            LEAP_HAND_JOINTS.index("o"),       # thumb middle joint
            LEAP_HAND_JOINTS.index("p")     # thumb tip
        ]
        
        self.create_target_vis()

    def load_assets(self):
        """Load LeapHand specific assets."""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Load ground plane at z=0
        self.plane_id = p.loadURDF("plane.urdf")

        # Load table at origin first
        table_urdf_path = os.path.join(
            current_dir,
            "assets",
            "urdf",
            "table",
            "square_table.urdf"
        )

        self.table_height = 0.3  # From URDF: <box size="0.4 0.4 0.3"/>
        self.table_size = [0.4, 0.4, self.table_height]  # Length, Width, Height

        # Place table with its bottom at z=0 (on the plane)
        table_pos_z = self.table_height / 2  # Offset up by half height since center is origin
        # self.table_id = p.loadURDF(
        #     table_urdf_path,
        #     [0.0, 0.0, table_pos_z],  # Center of table offset up from plane
        #     [0, 0, 0, 1],
        #     useFixedBase=True
        # )

        # Load LeapHand on top of table
        leap_hand_urdf_path = os.path.join(
            current_dir,
            "assets",
            "urdf",
            "leap_hand",
            "leap_hand_right.urdf"
        )
        
        
        self.base_position = [0.0, 0.0, self.table_height + 0.05]  # Place at table surface height
        self.hand_id = p.loadURDF(
            leap_hand_urdf_path,
            self.base_position,
            [0, 0, 0, 1],
            useFixedBase=True
        )

        # Store number of joints for later use
        self.num_joints = p.getNumJoints(self.hand_id)
      
        # Initialize LeapHand state
        self.reset()

    def create_target_vis(self):
        # load balls
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0, 0, self.table_height + 0.05]  # Place at table surface height
        
        self.ballMbt = []
        for i in range(0,4):
            self.ballMbt.append(p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)) # for base and finger tip joints    
            no_collision_group = 0
            no_collision_mask = 0
            p.setCollisionFilterGroupMask(self.ballMbt[i], -1, no_collision_group, no_collision_mask)
        p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1]) 
        p.changeVisualShape(self.ballMbt[1], -1, rgbaColor=[0, 1, 0, 1]) 
        p.changeVisualShape(self.ballMbt[2], -1, rgbaColor=[0, 0, 1, 1])  
        p.changeVisualShape(self.ballMbt[3], -1, rgbaColor=[1, 1, 1, 1])
    
    def update_target_vis(self, hand_pos):
        _, current_orientation = p.getBasePositionAndOrientation( self.ballMbt[0])
        p.resetBasePositionAndOrientation(self.ballMbt[0], hand_pos[1], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[1])
        p.resetBasePositionAndOrientation(self.ballMbt[1], hand_pos[3], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[2])
        p.resetBasePositionAndOrientation(self.ballMbt[2], hand_pos[5], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[3])
        p.resetBasePositionAndOrientation(self.ballMbt[3], hand_pos[7], current_orientation)

    def reset(self):
        """Reset the environment to initial state."""
        # Reset joints to home position
        for joint in range(self.num_joints):
            if p.getJointInfo(self.hand_id, joint)[2] != p.JOINT_FIXED:
                p.resetJointState(self.hand_id, joint, 0)

        # Verify end effector position
        state = p.getLinkState(self.hand_id, 0)  # Use the correct link index here
        pos = state[0]
        orn = state[1]
        expected_pos = (0.0, 0.0, self.table_height + 0.05)
        expected_orn = (0.0, 0.0, 0.0, 1.0)

        # Log warning if position is significantly different
        if np.any(np.abs(np.array(pos) - np.array(expected_pos)) > 0.01) or np.any(np.abs(np.array(orn) - np.array(expected_orn)) > 0.01):
            logger.warning(f"Warning: End effector position {pos} and orientation {orn} differs from expected {expected_pos} and {expected_orn}")

        # Create initial pose array [x,y,z, qx,qy,qz,qw]
        initial_pose = np.concatenate([pos, orn])

        # Publish initial pose
        self.endeff_publisher.pub_keypoints(initial_pose, "endeff_coords")

        return initial_pose

    def get_endeff_position(self):
        """Get the end-effector position."""
        state = p.getLinkState(self.hand_id, 0)
        pos = state[0]
        orn = state[1]
        return np.concatenate([pos, orn])
    
    def compute_IK(self, hand_pos):
        """Compute inverse kinematics for LeapHand using hand position data."""
        p.stepSimulation()
        
        # Verify we have the expected number of positions
        if len(hand_pos) != 8:
            logger.error(f"Error: Expected 8 positions but got {len(hand_pos)}")
            return None
        
        # Calculate inverse kinematics
        jointPoses = p.calculateInverseKinematics2(
            self.hand_id,
            self.leapEndEffectorIndex,
            hand_pos,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=0.0001,
        )
        
        # Create a 16-dimensional joint array for the LEAP Hand
        combined_jointPoses = [0] * 16
        
        # Map the IK results to the 16 joints expected by LEAP Hand
        combined_jointPoses[0:4] = jointPoses[0:4]       # Index finger
        combined_jointPoses[4:8] = jointPoses[4:8]       # Middle finger
        combined_jointPoses[8:12] = jointPoses[8:12]     # Ring finger
        combined_jointPoses[12:16] = jointPoses[12:16]   # Thumb
        
        # Reverse specific joint pairs like in avp_leap.py
        # combined_jointPoses[0:2] = combined_jointPoses[0:2][::-1]
        # combined_jointPoses[4:6] = combined_jointPoses[4:6][::-1]
        # combined_jointPoses[8:10] = combined_jointPoses[8:10][::-1]

        

        combined_jointPoses = ([0.0,] + combined_jointPoses[0:4] + 
        [0.0,] + combined_jointPoses[4:8] + 
        [0.0,] + combined_jointPoses[8:12] + 
        [0.0,] + combined_jointPoses[12:16])

        logger.info("Combined joint poses:", combined_jointPoses)
        combined_jointPoses = list(combined_jointPoses)

        return combined_jointPoses

    def take_action(self):
        """Execute received action using joint angles."""
        # Get the joint angles from IK based on Meta Quest data
        hand_pos = self.get_meta_quest_data()
        
        if hand_pos is not None:
            
            # Compute IK for the target positions
            action = self.compute_IK(hand_pos)

            
            if action is not None:
                for i in range(20):
                    p.setJointMotorControl2(
                        bodyUniqueId=self.hand_id,
                        jointIndex=i,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=action[i],
                        force=500,
                        positionGain=0.3,
                        velocityGain=1.0
                    )
                    logger.info("Moving", LEAP_HAND_JOINTS[i], "to", action[i])
                # Step the simulation
                p.stepSimulation()

                # After applying action, get and publish current state
                current_pose = self.get_endeff_position()
                self.endeff_publisher.pub_keypoints(current_pose, 'endeff_coords')
                
                # Publish joint angles
              
                
                self.joint_angle_publisher.pub_keypoints(np.array(action), 'joint_angles')

    def get_meta_quest_data(self):
        """
        Get the finger positions from Meta Quest hand tracking data and transform them into the simulation's coordinate frame.
        Returns an array of 8 positions: [index_middle, index_tip, middle_middle, middle_tip, 
                                        ring_middle, ring_tip, thumb_middle, thumb_tip]
        """
        # Receive transformed keypoints from subscriber
        raw_keypoints = self.transformed_keypoint_subscriber.recv_keypoints()
        
        if raw_keypoints is None or len(raw_keypoints) < max(QUEST_JOINTS['pinky'])+1:
            logger.warning("Warning: Invalid or missing keypoint data from Meta Quest")
            return None
        
        # Extract the 8 key positions we need (tips and middle joints of 4 fingers)
        hand_pos = [
            np.array(raw_keypoints[QUEST_JOINTS['index'][1]]),    # Index middle joint
            np.array(raw_keypoints[QUEST_JOINTS['index'][3]]),    # Index tip
            np.array(raw_keypoints[QUEST_JOINTS['middle'][1]]),   # Middle finger middle joint
            np.array(raw_keypoints[QUEST_JOINTS['middle'][3]]),   # Middle finger tip
            np.array(raw_keypoints[QUEST_JOINTS['ring'][1]]),     # Ring finger middle joint
            np.array(raw_keypoints[QUEST_JOINTS['ring'][3]]),     # Ring finger tip
            np.array(raw_keypoints[QUEST_JOINTS['thumb'][1]]),    # Thumb middle joint
            np.array(raw_keypoints[QUEST_JOINTS['thumb'][4]]),    # Thumb tip
        ]
        
        # Transform each position to match simulation coordinate frame
        for i in range(len(hand_pos)):
            # Meta Quest coordinates (X: right, Y: up, Z: forward) 
            # Convert to simulation coordinates (X: forward, Y: left, Z: up)
            x, y, z = hand_pos[i]
            
            # Rotate axes
            sim_x = -z   # Quest's forward (Z) becomes simulation's X (forward)
            sim_y = x  # Quest's right (X) becomes simulation's Y (left)
            sim_z = y   # Quest's up (Y) becomes simulation's Z (up)
            
            # Apply scaling
            sim_x *= self.glove_to_leap_mapping_scale
            sim_y *= self.glove_to_leap_mapping_scale
            sim_z *= self.glove_to_leap_mapping_scale
            
            # Translate to base position (wrist position in simulation)
            sim_x += self.base_position[0]
            sim_y += self.base_position[1]
            sim_z += self.base_position[2]
            
            hand_pos[i] = np.array([sim_x, sim_y, sim_z])
        
        self.update_target_vis(hand_pos)
        return hand_pos

    def get_rgb_depth_images(self):
        """Get RGB and depth images from PyBullet camera."""
        width = 640   # Match standard resolution
        height = 480

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.2, 0, 1.0],           # Camera position
            cameraTargetPosition=[0.0, 0, 0.3],        # Looking at hand workspace
            cameraUpVector=[0, 0, 1]                   # Standard up vector
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(width) / height,
            nearVal=0.1,
            farVal=100.0
        )

        # Get camera images
        (_, _, rgb, depth, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Process images
        rgb = np.array(rgb).reshape(height, width, 4)[:, :, :3]  # Remove alpha
        depth = np.array(depth).reshape(height, width)

        return rgb, depth, time.time()
    
    def get_dof_position(self):
        """
        Get the scalar DOF positions for each joint in the hand.
        """
        # List to store the DOF positions
        dof_positions = []
        
        # Iterate over each joint and retrieve its position
        for i in range(self.num_joints):
            if i not in self.tip_joint_indices: 
                joint_position = p.getJointState(self.hand_id, i)[0]  # The position of the joint
                dof_positions.append(joint_position)
        
        return np.array(dof_positions)
    
    def apply_offset(self, joints):
        ret_joints = np.array(joints)
        ret_joints[1] -= 0.2
        ret_joints[5] -= 0.2
        ret_joints[9] -= 0.2
        return ret_joints