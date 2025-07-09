#!/usr/bin/env python3
"""
LEAP Hand IK Solver using PyBullet.
This class provides inverse kinematics solutions for the LEAP hand
without any simulation overhead.
"""

import pybullet as p
import numpy as np
import os

from beavr.teleop.configs.constants import robots

import logging

logger = logging.getLogger(__name__)


class LeapHandIKSolver:
    """
    Minimal Inverse Kinematics solver for the LEAP hand using PyBullet.
    Eliminates all simulation aspects and focuses solely on IK calculation.
    """
    def __init__(self, urdf_path=None, use_gui=False, smoothing_factor=0.1):
        """
        Initialize the IK solver.
        
        Args:
            urdf_path: Path to the LEAP hand URDF file
            use_gui: Whether to use PyBullet GUI for visualization (for debugging)
            smoothing_factor: Smoothing factor for joint angle transitions (0.0 = no smoothing, 1.0 = ignore new solution)
        """
        # Initialize PyBullet in GUI or DIRECT mode
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        
        # Disable all physics simulation
        p.setGravity(0, 0, 0)
        
        # Find URDF path if not provided
        if urdf_path is None:
            urdf_path = os.path.join(os.getcwd(), "assets/urdf/leap_hand/leap_hand_right.urdf")
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found at: {urdf_path}")
        
        # Load the LEAP hand URDF with fixed base
        self.hand_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        
        # Apply joint limits for specified indices - expanded for better movement
        joint_limits = {
            1: (-0.1, 0.1),   # joint 'b' (index finger) - expanded range
            7: (-0.2, 0.2),   # joint 'f' (middle finger)
            12: (-0.2, 0.2),  # joint 'j' (ring finger)
            16: (-0.2, 0.2)   # joint 'n' (thumb)
        }
        
        for joint_index, (lower_limit, upper_limit) in joint_limits.items():
            p.changeDynamics(
                self.hand_id, 
                joint_index, 
                jointLowerLimit=lower_limit, 
                jointUpperLimit=upper_limit
            )
            if use_gui:
                logger.info(f"Modified joint {joint_index} limits to [{lower_limit}, {upper_limit}]")
        
        # Define end effector indices to match the visualization script exactly
        # Fingertip joints (d, h, l, p)
        self.fingertip_indices = [4, 9, 14, 19]  # Updated to match visualization script
        # Tip_head links
        self.tip_head_indices = [5, 10, 15, 20]  # Updated to match visualization script
        
        # Combine both sets of indices
        self.all_indices = self.fingertip_indices + self.tip_head_indices
        
        # Scale factor for transformations (moved to constants)
        self.scale_factor = robots.LEAP_FINGER_SCALE_FACTOR
        
        # Store last calculated joint angles to use as seed for next calculation
        self.last_joint_angles = None
        
        # Smoothing factor for joint angle transitions (0.0 = no smoothing, 1.0 = ignore new solution)
        self.smoothing_factor = smoothing_factor
        
        # Set up camera view if in GUI mode
        if use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw=0,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0]
            )
            
            # Create a coordinate frame at the origin
            self._create_coordinate_frame()
    
    def transform_position(self, pos, is_ring_finger=False, is_index_finger=False, is_thumb=False):
        """
        Apply standard transformation to a position.
        
        Args:
            pos: Original position [x, y, z]
            is_ring_finger: Whether to apply ring finger displacement
            is_index_finger: Whether to apply index finger displacement
            is_thumb: Whether this is a thumb position
            
        Returns:
            Transformed position [x', y', z']
        """
        # Reflect and convert from y-up to z-up
        pos_reflected = [
            -pos[0],  # Negate x to reflect
            -pos[2],  # Negate z (which becomes y in PyBullet)
            pos[1]    # y in log becomes z in PyBullet
        ]

        # Use different scale for thumb if needed
        scale_factor = robots.LEAP_THUMB_SCALE_FACTOR if is_thumb else self.scale_factor
        
        # Rotate 90° around z-axis and scale
        pos_transformed = [
            pos_reflected[1] * scale_factor,           # y becomes x (scaled)
            -pos_reflected[0] * scale_factor,          # -x becomes y (scaled)
            pos_reflected[2] * scale_factor            # z stays z (scaled)
        ]
        
        # Apply special displacement for ring finger if needed
        if is_ring_finger:
            ring_displacement = -0.03
            pos_transformed[1] += ring_displacement
        
        # Apply special displacement for index finger if needed
        if is_index_finger:
            index_displacement = 0.02  # Opposite direction of ring displacement
            pos_transformed[1] += index_displacement
            
        # Apply special handling for thumb if needed
        if is_thumb:
            # Adjust the thumb position to account for its different orientation
            thumb_x_offset = 0.02
            thumb_y_offset = 0.01
            
            pos_transformed[0] += thumb_x_offset
            pos_transformed[1] += thumb_y_offset
            
            # First, apply rotation in x-y plane
            angle_xy = 0.2  # radians
            x = pos_transformed[0]
            y = pos_transformed[1]
            pos_transformed[0] = x * np.cos(angle_xy) - y * np.sin(angle_xy)
            pos_transformed[1] = x * np.sin(angle_xy) + y * np.cos(angle_xy)
            
            # Then, apply anti-clockwise rotation around y-axis (in x-z plane)
            angle_y = 0.6  # radians - adjust as needed
            x = pos_transformed[0]
            z = pos_transformed[2]
            pos_transformed[0] = x * np.cos(angle_y) + z * np.sin(angle_y)
            pos_transformed[2] = -x * np.sin(angle_y) + z * np.cos(angle_y)
        
        return pos_transformed
    
    def compute_hand_pos(self, finger_positions):
        """
        Build a list of 8 key positions for IK.
        
        Args:
            finger_positions: Dictionary with keys 'thumb', 'index', 'middle', 'ring'
                             Each value is a list of positions for that finger
        
        Returns:
            List of 8 positions in the order:
            [thumb_middle, thumb_tip, index_middle, index_tip, 
             middle_middle, middle_tip, ring_middle, ring_tip]
        """
        hand_pos = []
        for finger in ['thumb', 'index', 'middle', 'ring']:
            positions = finger_positions.get(finger, [])
            if len(positions) >= 2:
                middle_pos = positions[-2]
                tip_pos = positions[-1]
            else:
                raise ValueError(f"Not enough positions for finger {finger}")
            
            # Transform the positions
            is_ring = finger == 'ring'
            is_index = finger == 'index'
            is_thumb = finger == 'thumb'
            middle_pos_transformed = self.transform_position(middle_pos, is_ring, is_index, is_thumb)
            tip_pos_transformed = self.transform_position(tip_pos, is_ring, is_index, is_thumb)
            
            hand_pos.append(np.array(middle_pos_transformed, dtype=float))
            hand_pos.append(np.array(tip_pos_transformed, dtype=float))
        return hand_pos
    
    def solve_ik(self, finger_positions, current_positions=None):
        """
        Solve inverse kinematics for the LEAP hand.
        
        Args:
            finger_positions: Dictionary with keys 'thumb', 'index', 'middle', 'ring'
                             Each value is a list of positions for that finger
            current_positions: Optional list of current joint positions to use as seed
        
        Returns:
            List of 16 joint angles for the LEAP hand
        """
        # Create a full dictionary with empty lists for missing fingers
        full_finger_positions = {
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': []
        }
        
        # Update with the provided finger positions
        for finger, positions in finger_positions.items():
            full_finger_positions[finger] = positions
        
        # Check which fingers are active (have positions)
        active_fingers = {finger: positions for finger, positions in full_finger_positions.items() if len(positions) > 0}
        
        # Compute target positions only for active fingers
        target_positions_dict = {}
        for finger in ['thumb', 'index', 'middle', 'ring']:
            if finger in active_fingers:
                positions = active_fingers[finger]
                if len(positions) >= 2:
                    middle_pos = positions[-2]
                    tip_pos = positions[-1]
                    
                    # Transform the positions - convert to Python lists to avoid NumPy array comparison issues
                    is_ring = finger == 'ring'
                    is_index = finger == 'index'
                    middle_pos_transformed = list(self.transform_position(middle_pos, is_ring, is_index))
                    tip_pos_transformed = list(self.transform_position(tip_pos, is_ring, is_index))
                    
                    
                    target_positions_dict[finger] = {
                        'middle': middle_pos_transformed,
                        'tip': tip_pos_transformed
                    }
        
        # Create the target_positions list in the order expected by compute_ik
        # [thumb_middle, thumb_tip, index_middle, index_tip, middle_middle, middle_tip, ring_middle, ring_tip]
        target_positions = []
        
        # Add positions for each finger, using defaults if not active
        finger_defaults = {
            'thumb': [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
            'index': [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
            'middle': [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])],
            'ring': [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
        }
        
        for finger in ['thumb', 'index', 'middle', 'ring']:
            if finger in target_positions_dict:
                target_positions.append(target_positions_dict[finger]['middle'])
                target_positions.append(target_positions_dict[finger]['tip'])
            else:
                target_positions.append(finger_defaults[finger][0])
                target_positions.append(finger_defaults[finger][1])
        
        # Extract positions for IK calculation in the order expected by PyBullet
        thumb_middle_pos = target_positions[0]
        thumb_tip_pos = target_positions[1]
        index_middle_pos = target_positions[2]
        index_tip_pos = target_positions[3]
        middle_middle_pos = target_positions[4]
        middle_tip_pos = target_positions[5]
        ring_middle_pos = target_positions[6]
        ring_tip_pos = target_positions[7]
        
        # Map target positions to both sets of end effectors
        leap_end_effector_pos = [
            # For fingertip joints
            index_middle_pos,   # Index fingertip
            middle_middle_pos,  # Middle fingertip
            ring_middle_pos,    # Ring fingertip
            thumb_middle_pos,   # Thumb fingertip
            
            # For tip_head links
            index_tip_pos,      # Index tip_head
            middle_tip_pos,     # Middle tip_head
            ring_tip_pos,       # Ring tip_head
            thumb_tip_pos       # Thumb tip_head
        ]
        
        # Visualize target positions if in GUI mode
        if p.getConnectionInfo(self.physics_client)['connectionMethod'] == p.GUI:
            # Clear previous markers
            if hasattr(self, '_markers'):
                for marker_id in self._markers:
                    p.removeBody(marker_id)
            self._markers = []
            
            # Create markers for each target position
            finger_names = ['thumb', 'thumb', 'index', 'index', 'middle', 'middle', 'ring', 'ring']
            joint_types = ['middle', 'tip', 'middle', 'tip', 'middle', 'tip', 'middle', 'tip']
            colors = {
                'thumb': [1, 0, 0, 0.7],    # Red
                'index': [0, 1, 0, 0.7],    # Green
                'middle': [0, 0, 1, 0.7],   # Blue
                'ring': [1, 1, 0, 0.7]      # Yellow
            }
            
            # Add markers for all target positions
            for i, pos in enumerate(target_positions):
                # Create sphere marker
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.005,
                    rgbaColor=colors[finger_names[i]]
                )
                marker_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=pos
                )
                self._markers.append(marker_id)
                
                # Add text label
                text_id = p.addUserDebugText(
                    f"{finger_names[i]}_{joint_types[i]}",
                    pos,
                    textColorRGB=colors[finger_names[i]][:3],
                    textSize=1.0
                )
                self._markers.append(text_id)
        
        # Set fixed random seed for IK solver to ensure deterministic results
        # np.random.seed(42)
        
        # Get the number of joints in the robot
        num_joints = p.getNumJoints(self.hand_id)
        
        # Use last calculated joint angles as seed if available, otherwise use provided positions or get current state
        if self.last_joint_angles is not None and current_positions is None:
            current_positions = self.last_joint_angles
        elif current_positions is None:
            # Only collect positions for movable joints (DOFs)
            movable_joint_indices = []
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.hand_id, i)
                if joint_info[2] != p.JOINT_FIXED:  # Skip fixed joints
                    movable_joint_indices.append(i)
            
            current_positions = []
            for i in movable_joint_indices:
                joint_state = p.getJointState(self.hand_id, i)
                current_positions.append(joint_state[0])
        
        # Calculate IK using all end effectors with fixed parameters
        try:
            joint_poses = p.calculateInverseKinematics2(
                self.hand_id,
                self.all_indices,
                leap_end_effector_pos,
                currentPositions=current_positions,
                solver=p.IK_DLS,
                maxNumIterations=100,  # Increased from 50 to give solver more chances
                residualThreshold=0.001,  # Increased slightly to make the solver less strict
                jointDamping=[0.1] * len(current_positions)  # Match the number of DOFs
            )
        except Exception as e:
            logger.error(f"IK calculation failed: {e}")
            # Return zeros as a fallback
            return [0.0] * 16
        
        # Create the output array with 16 joints
        real_robot_hand_q = [0.0] * 16
        
        # Map the joint angles to the output array
        if len(joint_poses) >= 16:
            # Copy all joint angles
            real_robot_hand_q[0:4] = joint_poses[0:4]
            real_robot_hand_q[4:8] = joint_poses[4:8]
            real_robot_hand_q[8:12] = joint_poses[8:12]
            real_robot_hand_q[12:16] = joint_poses[12:16]
            
            # Reverse some joint pairs as in the visualization script
            real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
            real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
            real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]

            real_robot_hand_q[12] -= 0.5
            real_robot_hand_q[13] += 0.5
            
            # Apply smoothing if we have a previous solution
            if hasattr(self, 'last_solution') and self.last_solution is not None:
                for i in range(len(real_robot_hand_q)):
                    real_robot_hand_q[i] = (self.smoothing_factor * self.last_solution[i] + 
                                           (1 - self.smoothing_factor) * real_robot_hand_q[i])
            
            # Store this solution for next time
            self.last_solution = real_robot_hand_q.copy()
        
        # Apply the calculated joint angles to the hand model for visualization
        # but don't step the simulation to avoid affecting the IK solution
        if p.getConnectionInfo(self.physics_client)['connectionMethod'] == p.GUI:
            for i, angle in enumerate(real_robot_hand_q):
                p.resetJointState(self.hand_id, i, angle)
            # Don't call p.stepSimulation() here to avoid affecting the solution
        
        # Don't force joint 'b' to be 0.0 - allow it to move
        # real_robot_hand_q[0] = 0.0  # Comment out this line to allow movement
        
        # Store the calculated joint angles for next iteration
        self.last_joint_angles = joint_poses
        
        return [float(i) for i in real_robot_hand_q]
    
    def _create_coordinate_frame(self, position=[0, 0, 0], size=0.1):
        """Create a coordinate frame at the specified position."""
        # X-axis (red)
        p.addUserDebugLine(position, [position[0] + size, position[1], position[2]], [1, 0, 0], 2)
        # Y-axis (green)
        p.addUserDebugLine(position, [position[0], position[1] + size, position[2]], [0, 1, 0], 2)
        # Z-axis (blue)
        p.addUserDebugLine(position, [position[0], position[1], position[2] + size], [0, 0, 1], 2)
    
    def close(self):
        """Clean up PyBullet resources."""
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)