from copy import deepcopy as copy
from beavr.teleop.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from .operator import Operator
from shapely.geometry import Point, Polygon 
from shapely.ops import nearest_points

from beavr.teleop.utils.files import get_path_in_package, get_yaml_data
from beavr.teleop.utils.vectorops import coord_in_bound
from beavr.teleop.utils.timer import FrequencyTimer
from beavr.teleop.constants import VR_FREQ, OCULUS_JOINTS, LEAP_JOINTS_PER_FINGER, LEAP_JOINT_OFFSETS
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from beavr.teleop.utils.logger import HandLogger
import logging

logger = logging.getLogger(__name__)

class LeapHandOperator(Operator):
    def __init__(self, host, transformed_keypoints_port, joint_angle_subscribe_port, joint_angle_publish_port, reset_publish_port, finger_configs, logging_config=None):
        self.notify_component_start('leap hand operator')
        self._host, self._port = host, transformed_keypoints_port
        
        # Subscriber for the transformed hand keypoints
        self._transformed_hand_keypoint_subscriber = ZMQKeypointSubscriber(
            host = self._host,
            port = self._port,
            topic = 'transformed_hand_coords'
        )
        # Subscriber for the transformed arm frame
        self._transformed_arm_keypoint_subscriber = ZMQKeypointSubscriber(
            host = self._host,
            port = self._port,
            topic = 'transformed_hand_frame'
        )
        # New publishers for joint control
        self._joint_angle_publisher = ZMQKeypointPublisher(
            host = host,
            port = joint_angle_publish_port
        )
        
        self._reset_publisher = ZMQKeypointPublisher(
            host = host,
            port = reset_publish_port
        )

        self._joint_angle_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = joint_angle_subscribe_port,
            topic = 'joint_angles'
        )   

        # Initializing the  finger configs
        self.finger_configs = finger_configs

        # NOTE: This is a bypass when doing our own code this should be removed
        self.return_real = lambda: False

        # Initialzing the moving average queues
        self.moving_average_queues = {
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': []
        }

        # Calibrating to get the thumb bounds
        self._calibrate_bounds()

        # Getting the bounds for the allegro hand, 
        # will involve trial and erorr to get exactly right for the leaphand
        # These work just well. 
        allegro_bounds_path = get_path_in_package('components/operators/configs/allegro.yaml')
        self.allegro_bounds = get_yaml_data(allegro_bounds_path)

        self._robot = None

        self._timer = FrequencyTimer(VR_FREQ)

        # Using 3 dimensional thumb motion or two dimensional thumb motion
        if self.finger_configs['three_dim']:
            self.thumb_angle_calculator = self._get_3d_thumb_angles
        else:
            self.thumb_angle_calculator = self._get_2d_thumb_angles

        # Initialize FPS counter and start time
        self._frame_count = 0
        self._start_time = time.time()
        self._last_angles = None

        # Create thread pool for parallel finger processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)  # One thread per finger
        self._finger_locks = {
            'thumb': threading.Lock(),
            'index': threading.Lock(),
            'middle': threading.Lock(),
            'ring': threading.Lock()
        }

        # Initialize logging
        self.logging_config = logging_config or {"enabled": False}
        self.logging_enabled = self.logging_config.get("enabled", False)
        
        if self.logging_enabled:
            logger.info("Initializing hand logger with config:", self.logging_config)
            self.hand_logger = HandLogger(prefix="leap_hand")
        else:
            self.hand_logger = None

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
        return True

    # Calibrate the thumb bounds
    def _calibrate_bounds(self):
        self.notify_component_start('calibration')
        logger.info(f'THUMB BOUNDS IN THE OPERATOR: {self.hand_thumb_bounds}')

    # Get the transformed finger coordinates
    def _get_finger_coords(self):
        raw_keypoints = self.transformed_hand_keypoint_subscriber.recv_keypoints()
        return dict(
            index = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['index']]]),
            middle = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['middle']]]),
            ring = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['ring']]]),
            thumb =  np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['thumb']]])
        )

    # Get robot thumb angles when moving only in 2D motion
    def _get_2d_thumb_angles(self, thumb_keypoints, curr_angles):
        for idx, thumb_bounds in enumerate(self.hand_thumb_bounds):
            if coord_in_bound(thumb_bounds[:4], thumb_keypoints[:2]) > -1:
                return self.fingertip_solver.thumb_motion_2D(
                    hand_coordinates = thumb_keypoints, 
                    xy_hand_bounds = thumb_bounds[:4],
                    yz_robot_bounds = self.allegro_bounds['thumb_bounds'][idx]['projective_bounds'],
                    robot_x_val = self.allegro_bounds['x_coord'],
                    moving_avg_arr = self.moving_average_queues['thumb'], 
                    curr_angles = curr_angles
                )
        
        return curr_angles

    # Get robot thumb angles when moving in 3D motion
    def _get_3d_thumb_angles(self, thumb_keypoints, curr_angles):
        """Get robot thumb angles when moving in 3D motion"""
        # We will be using polygon implementations of shapely library to test this
        planar_point = Point(thumb_keypoints)
        planar_thumb_bounds = Polygon(self.hand_thumb_bounds[:4])

        # Get the closest point from the thumb to the point within the bounds
        closest_point = nearest_points(planar_thumb_bounds, planar_point)[0]        
        closest_point_coords = [closest_point.x, closest_point.y, thumb_keypoints[2]]
        
        result = self.fingertip_solver.thumb_motion_3D(
            hand_coordinates = closest_point_coords,
            xy_hand_bounds = self.hand_thumb_bounds[:4],
            yz_robot_bounds = self.allegro_bounds['thumb_bounds'][0]['projective_bounds'], # NOTE: We assume there is only one bound now
            z_hand_bound = self.hand_thumb_bounds[4],
            x_robot_bound = self.allegro_bounds['thumb_bounds'][0]['x_bounds'],
            moving_avg_arr = self.moving_average_queues['thumb'], 
            curr_angles = curr_angles
        )
        
        return result
        
    # Generate frozen angles for the fingers
    def _generate_frozen_angles(self, joint_angles, finger_type):
        for idx in range(LEAP_JOINTS_PER_FINGER):
            if idx > 0:
                joint_angles[idx + LEAP_JOINT_OFFSETS[finger_type]] = 0.05
            else:
                joint_angles[idx + LEAP_JOINT_OFFSETS[finger_type]] = 0

        return joint_angles
    
    def _process_finger(self, finger_type, hand_keypoints, current_angles):
        """Process a single finger's angles in parallel"""
        with self._finger_locks[finger_type]:
            if finger_type == 'thumb':
                if not self.finger_configs[f'freeze_{finger_type}'] and not self.finger_configs[f'no_{finger_type}']:
                    return self.thumb_angle_calculator(hand_keypoints[finger_type][-1], current_angles)
            else:
                if not self.finger_configs[f'freeze_{finger_type}'] and not self.finger_configs[f'no_{finger_type}']:
                    return self.finger_joint_solver.calculate_finger_angles(
                        finger_type=finger_type,
                        finger_joint_coords=hand_keypoints[finger_type],
                        curr_angles=current_angles,
                        moving_avg_arr=self.moving_average_queues[finger_type]
                    )
            
            # If frozen or disabled
            self._generate_frozen_angles(current_angles, finger_type)
            return current_angles

    # Apply the retargeted angles to the robot
    def _apply_retargeted_angles(self):
        """Apply retargeted angles from hand motion to robot"""
        loop_start = time.time()        
        
        # Get finger coordinates
        t1 = time.time()
        hand_keypoints = self._get_finger_coords()
        t2 = time.time()
        
        # Get current joint positions from robot
        desired_joint_angles = self._joint_angle_subscriber.recv_keypoints()
        t3 = time.time()

        # Process all fingers in parallel
        finger_futures = {
            finger_type: self.thread_pool.submit(
                self._process_finger,
                finger_type,
                hand_keypoints,
                copy(desired_joint_angles)
            )
            for finger_type in ['thumb', 'index', 'middle', 'ring']
        }

        # Collect results and prepare logging data
        finger_data = {}
        for finger_type, future in finger_futures.items():
            try:
                result = future.result()
                # Store input positions and computed angles for logging
                finger_data[finger_type] = {
                    'input_positions': hand_keypoints[finger_type],
                    'computed_angles': result[LEAP_JOINT_OFFSETS[finger_type]:
                                           LEAP_JOINT_OFFSETS[finger_type] + LEAP_JOINTS_PER_FINGER]
                }
                
                # Update joint angles as before
                if finger_type == 'thumb':
                    desired_joint_angles[12:16] = result[12:16]
                elif finger_type == 'index':
                    desired_joint_angles[0:4] = result[0:4]
                elif finger_type == 'middle':
                    desired_joint_angles[4:8] = result[4:8]
                elif finger_type == 'ring':
                    desired_joint_angles[8:12] = result[8:12]
            except Exception as e:
                logger.error(f"Error processing {finger_type}: {e}")
        
        
        # After processing fingers and before publishing
        if self.logging_enabled and self.hand_logger:
            try:
                self.hand_logger.log_frame(
                    finger_input_positions={k: v['input_positions'] for k, v in finger_data.items()},
                    finger_computed_angles={k: v['computed_angles'] for k, v in finger_data.items()},
                    finger_states=desired_joint_angles
                )
            except Exception as e:
                logger.error(f"Error logging frame: {e}")

        # Publish joint angles
        self._joint_angle_publisher.pub_keypoints(
            desired_joint_angles,
            'joint_angles'
        )

    def __del__(self):
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown()
        if hasattr(self, 'hand_logger') and self.hand_logger:
            self.hand_logger.close()

def allegro_to_LEAPhand(joints, teleop = True, zeros = True):
    ret_joints = np.array(joints)
    ret_joints[1] -= 0.2
    ret_joints[5] -= 0.2
    ret_joints[9] -= 0.2
    return ret_joints