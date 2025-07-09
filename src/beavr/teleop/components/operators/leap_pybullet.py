from beavr.teleop.utils.network import (
    ZMQKeypointSubscriber,
    ZMQPublisherManager,
    cleanup_zmq_resources,
    get_global_context,
)
from .operator import Operator
from beavr.teleop.utils.timer import FrequencyTimer
from beavr.teleop.configs.constants import robots
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from beavr.teleop.utils.logger import HandLogger
from beavr.teleop.components.operators.solvers.leap_solver import LeapHandIKSolver
import queue

import logging

logger = logging.getLogger(__name__)


class LeapHandOperator(Operator):
    def __init__(self, host, transformed_keypoints_port, joint_angle_subscribe_port, joint_angle_publish_port, reset_publish_port, finger_configs, logging_config=None):
        self.notify_component_start('leap hand operator with PyBullet IK')
        self._host, self._port = host, transformed_keypoints_port
        
        # Shared ZMQ context
        self._context = get_global_context()

        # Subscriber for the transformed hand keypoints
        self._transformed_hand_keypoint_subscriber = ZMQKeypointSubscriber(
            host=self._host,
            port=self._port,
            topic='transformed_hand_coords',
            context=self._context,
        )
        # Subscriber for the transformed arm frame
        self._transformed_arm_keypoint_subscriber = ZMQKeypointSubscriber(
            host=self._host,
            port=self._port,
            topic='transformed_hand_frame',
            context=self._context,
        )
        # Centralized publisher manager (new pub/sub structure)
        self._publisher_manager = ZMQPublisherManager.get_instance()
        self._publisher_host = host
        self._joint_angle_publish_port = joint_angle_publish_port
        self._reset_publish_port = reset_publish_port  # Currently unused

        self._joint_angle_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=joint_angle_subscribe_port,
            topic='joint_angles',
            context=self._context,
        )   

        # Initializing the finger configs
        self.finger_configs = finger_configs

        # Initialize the PyBullet IK solver
        self.ik_solver = LeapHandIKSolver(use_gui=False)

        # Initialzing the moving average queues (kept for compatibility)
        self.moving_average_queues = {
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': []
        }

        self._robot = None
        self._timer = FrequencyTimer(robots.VR_FREQ)

        # Create thread pool for parallel processing if needed
        self.thread_pool = ThreadPoolExecutor(max_workers=1)  # Single thread as PyBullet handles all fingers at once
        
        # Initialize logging with background thread
        self.logging_config = logging_config or {"enabled": False}
        self.logging_enabled = self.logging_config.get("enabled", False)
        
        if self.logging_enabled:
            logger.info("Initializing hand logger with config:", self.logging_config)
            self.hand_logger = HandLogger(prefix="leap_hand_pybullet")
            
            # Create a queue for logging data
            self.log_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
            
            # Start background logging thread
            self._logging_thread_active = True
            self._logging_thread = threading.Thread(target=self._background_logging, daemon=True)
            self._logging_thread.start()
        else:
            self.hand_logger = None
            self.log_queue = None
            self._logging_thread = None

    @property
    def timer(self):
        return self._timer

    @property
    def transformed_arm_keypoint_subscriber(self):
        return self._transformed_arm_keypoint_subscriber
    
    @property
    def robot(self):
        return self._robot
    
    @property
    def transformed_hand_keypoint_subscriber(self):
        return self._transformed_hand_keypoint_subscriber
    
    # This function differentiates between the real robot and simulation
    def return_real(self):
        return False
    
    # Get the transformed finger coordinates
    def _get_finger_coords(self):
        raw_keypoints = self.transformed_hand_keypoint_subscriber.recv_keypoints()
        if raw_keypoints is None:
            logger.warning("Warning: No keypoints received from subscriber")
            return None
            
        try:
            return dict(
                index = np.vstack([raw_keypoints[0], raw_keypoints[robots.OCULUS_JOINTS['index']]]),
                middle = np.vstack([raw_keypoints[0], raw_keypoints[robots.OCULUS_JOINTS['middle']]]),
                ring = np.vstack([raw_keypoints[0], raw_keypoints[robots.OCULUS_JOINTS['ring']]]),
                thumb = np.vstack([raw_keypoints[0], raw_keypoints[robots.OCULUS_JOINTS['thumb']]])
            )
        except (IndexError, TypeError) as e:
            logger.error(f"Error processing keypoints: {e}")
            logger.error(f"Received keypoints shape/content: {raw_keypoints}")
            return None
    
    # Generate frozen angles for the fingers
    def _generate_frozen_angles(self, joint_angles, finger_type):
        for idx in range(robots.LEAP_JOINTS_PER_FINGER):
            if idx > 0:
                joint_angles[idx + robots.LEAP_JOINT_OFFSETS[finger_type]] = 0.05
            else:
                joint_angles[idx + robots.LEAP_JOINT_OFFSETS[finger_type]] = 0

        return joint_angles

    def _background_logging(self):
        """Background thread for processing log queue"""
        while self._logging_thread_active:
            try:
                # Get data from queue with timeout to allow checking thread_active flag
                log_data = self.log_queue.get(timeout=0.5)
                
                # Log the data
                self.hand_logger.log_frame(
                    finger_input_positions=log_data['finger_input_positions'],
                    finger_computed_angles=log_data['finger_computed_angles'],
                    finger_states=log_data['finger_states']
                )
                
                # Mark task as done
                self.log_queue.task_done()
            except queue.Empty:
                # Queue was empty, just continue
                pass
            except Exception as e:
                logger.error(f"Error in background logging thread: {e}")

    # Apply the retargeted angles to the robot using PyBullet IK
    def _apply_retargeted_angles(self):
        """Apply retargeted angles from hand motion to robot using PyBullet IK"""
        # Get finger coordinates
        hand_keypoints = self._get_finger_coords()
        if hand_keypoints is None:
            # If we can't get keypoints, just wait for next iteration
            time.sleep(0.01)  # Small sleep to prevent busy waiting
            return
            
        # Get current joint positions from robot
        desired_joint_angles = self._joint_angle_subscriber.recv_keypoints()
        if desired_joint_angles is None:
            logger.warning("Warning: No joint angles received from subscriber")
            time.sleep(0.01)
            return
            

        # Process all fingers using PyBullet IK
        try:
            # Check if any fingers are frozen or disabled
            active_fingers = {}
            for finger_type in ['thumb', 'index', 'middle', 'ring']:
                if not self.finger_configs[f'freeze_{finger_type}'] and not self.finger_configs[f'no_{finger_type}']:
                    active_fingers[finger_type] = hand_keypoints[finger_type]
            
            # If we have active fingers, solve IK
            if active_fingers:
                # Solve IK for all active fingers at once
                calculated_joint_angles = self.ik_solver.solve_ik(active_fingers)
                
                # Update the desired joint angles with the calculated ones
                # Only update active fingers
                if 'thumb' in active_fingers:
                    desired_joint_angles[12:16] = calculated_joint_angles[12:16]
                if 'index' in active_fingers:
                    desired_joint_angles[0:4] = calculated_joint_angles[0:4]
                if 'middle' in active_fingers:
                    desired_joint_angles[4:8] = calculated_joint_angles[4:8]
                if 'ring' in active_fingers:
                    desired_joint_angles[8:12] = calculated_joint_angles[8:12]
            
            # Handle frozen fingers
            for finger_type in ['thumb', 'index', 'middle', 'ring']:
                if self.finger_configs[f'freeze_{finger_type}'] or self.finger_configs[f'no_{finger_type}']:
                    self._generate_frozen_angles(desired_joint_angles, finger_type)
            
            # Prepare finger data for logging
            finger_data = {}
            for finger_type in ['thumb', 'index', 'middle', 'ring']:
                offset = robots.LEAP_JOINT_OFFSETS[finger_type]
                finger_data[finger_type] = {
                    'input_positions': hand_keypoints[finger_type],
                    'computed_angles': desired_joint_angles[offset:offset + robots.LEAP_JOINTS_PER_FINGER]
                }
                
        except Exception as e:
            logger.error(f"Error in PyBullet IK processing: {e}")
            finger_data = {}
            return
        
        
        # After processing fingers and before publishing
        if self.logging_enabled and self.hand_logger and finger_data:
            try:
                # Instead of logging directly, add to queue for background processing
                if not self.log_queue.full():  # Skip logging if queue is full rather than blocking
                    log_data = {
                        'finger_input_positions': {k: v['input_positions'].copy() if isinstance(v['input_positions'], np.ndarray) else v['input_positions'] for k, v in finger_data.items()},
                        'finger_computed_angles': {k: v['computed_angles'].copy() if isinstance(v['computed_angles'], np.ndarray) else v['computed_angles'] for k, v in finger_data.items()},
                        'finger_states': desired_joint_angles.copy() if isinstance(desired_joint_angles, np.ndarray) else desired_joint_angles
                    }
                    self.log_queue.put_nowait(log_data)
            except Exception as e:
                logger.error(f"Error queuing log data: {e}")

        # Publish joint angles using the new publisher manager
        self._publisher_manager.publish(
            self._publisher_host,
            self._joint_angle_publish_port,
            'joint_angles',
            desired_joint_angles,
        )

    def __del__(self):
        # Clean shutdown of logging thread
        if hasattr(self, '_logging_thread_active') and self._logging_thread_active:
            self._logging_thread_active = False
            if self._logging_thread and self._logging_thread.is_alive():
                self._logging_thread.join(timeout=1.0)  # Wait up to 1 second for thread to finish
                
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown()
        if hasattr(self, 'ik_solver'):
            self.ik_solver.close()
        if hasattr(self, 'hand_logger') and self.hand_logger:
            # Try to process any remaining items in the queue
            if hasattr(self, 'log_queue') and self.log_queue:
                try:
                    while not self.log_queue.empty():
                        self.log_queue.get_nowait()
                        self.log_queue.task_done()
                except Exception as e:
                    logger.error(f"Error closing hand logger: {e}")
                    pass
            self.hand_logger.close()

        # Ensure all ZMQ resources are cleaned up
        cleanup_zmq_resources()

def allegro_to_LEAPhand(joints, teleop = True, zeros = True):
    ret_joints = np.array(joints)
    ret_joints[1] -= 0.2
    ret_joints[5] -= 0.2
    ret_joints[9] -= 0.2
    return ret_joints