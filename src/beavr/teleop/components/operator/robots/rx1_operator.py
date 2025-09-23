import logging
import time
from copy import deepcopy as copy

import numpy as np
import zmq
from scipy.spatial.transform import Rotation, Slerp

from beavr.teleop.common.logging.logger import PoseLogger
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components.operator.operator_base import Operator
from beavr.teleop.configs.constants.robots import (
    ARM_HIGH_RESOLUTION,
    ARM_LOW_RESOLUTION,
    ARM_TELEOP_CONT,
    ARM_TELEOP_STOP,
    VR_FREQ,
)

logger = logging.getLogger(__name__)


# Simple complementary filter with SLERP for orientation (same as Allegro)
class CompStateFilter:
    def __init__(self, init_state, comp_ratio=0.6):
        self.pos_state = init_state[:3]
        self.ori_state = init_state[3:7]
        self.comp_ratio = comp_ratio

    def __call__(self, next_state):
        self.pos_state = self.pos_state * self.comp_ratio + next_state[:3] * (1 - self.comp_ratio)
        ori_interp = Slerp([0, 1], Rotation.from_quat([self.ori_state, next_state[3:7]]))
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_quat()
        return np.concatenate([self.pos_state, self.ori_state])


class RX1RightOperator(Operator):
    """
    Operator for the RX1 right arm.
    This operator is responsible for receiving hand coordinates and transforming them to a cartesian space.
    It also handles teleoperation, resolution scaling, and logging.
    """

    def __init__(
        self,
        host,
        transformed_keypoints_port,
        stream_configs,
        stream_oculus,
        endeff_publish_port,
        endeff_subscribe_port,
        moving_average_limit,
        use_filter=False,
        arm_resolution_port=None,
        teleoperation_reset_port=None,
        reset_publish_port=None,
        logging_config=None,
    ):
        # Basic initialization
        self.notify_component_start("rx1 right_operator")
        self._host, self._port = host, transformed_keypoints_port

        # Add finger coordinate subscriber (like Allegro)
        self._hand_transformed_keypoint_subscriber = ZMQSubscriber(
            host=self._host, port=self._port, topic="transformed_hand_coords"
        )

        self._arm_transformed_keypoint_subscriber = ZMQSubscriber(
            host=host, port=transformed_keypoints_port, topic="transformed_hand_frame"
        )

        # Optional subscribers (same as Allegro)
        self._arm_resolution_subscriber = ZMQSubscriber(host=host, port=arm_resolution_port, topic="button")

        self._arm_teleop_state_subscriber = ZMQSubscriber(
            host=host, port=teleoperation_reset_port, topic="pause"
        )

        self.endeff_homo_subscriber = ZMQSubscriber(
            host=host, port=endeff_subscribe_port, topic="endeff_homo"
        )

        self.end_eff_position_publisher = ZMQPublisherManager(host=host, port=endeff_publish_port)

        self.reset_publisher = ZMQPublisherManager(host=host, port=reset_publish_port)

        self._stream_oculus = stream_oculus
        self.stream_configs = stream_configs

        # State initialization
        self.arm_teleop_state = ARM_TELEOP_STOP
        self.resolution_scale = 1.0
        self.is_first_frame = True
        self._timer = FrequencyTimer(VR_FREQ)
        self._robot = None
        self.real = False

        # Transformation matrices
        self.robot_init_H = None
        self.robot_moving_H = None
        self.hand_init_H = None
        self.hand_moving_H = None
        self.hand_init_t = None

        # Filter setup
        self.use_filter = use_filter
        self.comp_filter = None

        # Moving average
        # self.moving_Average_queue = []
        # self.moving_average_limit = moving_average_limit
        # self.hand_frames = []

        self._stream_oculus = stream_oculus
        self.stream_configs = stream_configs

        # Initialize pose logger based on config
        self.logging_config = logging_config or {"enabled": False}
        self.logging_enabled = self.logging_config.get("enabled", False)

        if self.logging_enabled:
            logger.info("Initializing pose logger with config:", self.logging_config)
            self.pose_logger = PoseLogger()
        else:
            self.pose_logger = None

        # Precalculate constant matrices
        self.H_R_V_inv = np.linalg.pinv(np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        self.H_T_V = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    @property
    def timer(self):
        return self._timer

    @property
    def robot(self):
        return self._robot

    @property
    def transformed_hand_keypoint_subscriber(self):
        """Required by Operator base class."""
        return self._hand_transformed_keypoint_subscriber

    @property
    def transformed_arm_keypoint_subscriber(self):
        return self._arm_transformed_keypoint_subscriber

    def return_real(self):
        return self.real

    # ------------------------------
    # Frame / Matrix utilities
    # ------------------------------
    def _get_hand_frame(self):
        for _ in range(10):
            data = self._arm_transformed_keypoint_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
            if data is not None:
                break
        if data is None:
            return None
        return np.asanyarray(data).reshape(4, 3)  # shape (4,3)

    def _turn_frame_to_homo_mat(self, frame):
        """Convert 4x3 frame to a 4x4 homogeneous transform."""
        t = frame[0]
        r = frame[1:]

        homo_mat = np.eye(4)
        homo_mat[:3, :3] = r.T
        homo_mat[:3, 3] = t
        homo_mat[3, 3] = 1

        return homo_mat

    def _homo2cart(self, homo_mat):
        # Here we will use the resolution scale to set the translation resolution
        t = homo_mat[:3, 3]
        r = Rotation.from_matrix(homo_mat[:3, :3]).as_quat()

        cart = np.concatenate([t, r], axis=0)

        return cart

    def cart2homo(self, cart):
        """Inverse of _homo2cart."""
        homo = np.eye(4)
        t = cart[:3]
        r = Rotation.from_quat(cart[3:]).as_matrix()
        homo[:3, 3] = t
        homo[:3, :3] = r
        return homo

    def project_to_rotation_matrix(self, r):
        """Adjust a near-rotation matrix to be a valid rotation matrix using SVD."""
        u, _, vt = np.linalg.svd(r)  # Perform SVD
        r_fixed = u @ vt  # Reconstruct the rotation matrix

        # Ensure determinant is +1 (no reflection)
        if np.linalg.det(r_fixed) < 0:
            u[:, -1] *= -1  # Flip last column of U
            r_fixed = u @ vt  # Recalculate R
        return r_fixed

    def quat_to_euler_rad(self, qx: float, qy: float, qz: float, qw: float) -> list[float, float, float]:
        """
        Convert quaternion to Euler angles in degrees

        Args:
            qx, qy, qz, qw: Quaternion components in qx,qy,qz,qw order

        Returns:
            List of (roll, pitch, yaw) in degrees
        """
        rot = Rotation.from_quat([qx, qy, qz, qw])
        euler_rad = rot.as_euler("xyz")  # Get Euler angles in radians
        return euler_rad

    def _get_resolution_scale_mode(self):
        if not self._arm_resolution_subscriber:
            return 1.0  # default
        data = self._arm_resolution_subscriber.recv_keypoints()
        if data is None:
            return 1.0
        scale = np.asanyarray(data).reshape(1)[0]
        return scale

    def _get_arm_teleop_state(self):
        if not self._arm_teleop_state_subscriber:
            return ARM_TELEOP_CONT  # or STOP
        data = self._arm_teleop_state_subscriber.recv_keypoints()
        if data is None:
            return self.arm_teleop_state
        state = np.asanyarray(data).reshape(1)[0]
        return state

    # ------------------------------
    # Teleop reset logic
    # ------------------------------
    def _reset_teleop(self):
        logger.info("****** RESETTING TELEOP ******")
        self.reset_publisher.publish(1, "reset")
        self.robot_frame = self.endeff_homo_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
        while self.robot_frame is None:
            self.reset_publisher.publish(1, "reset")
            self.robot_frame = self.endeff_homo_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
            # print("Didn't receive robot homo pose")
            time.sleep(0.01)
        if self.robot_frame is None:
            logger.error("ERROR: No robot frame received.")
            return None
        self.robot_init_H = self.robot_frame
        self.robot_moving_H = copy(self.robot_init_H)

        logger.info("Robot init H:", self.robot_init_H)

        # Get initial hand frame
        first_hand_frame = None
        while first_hand_frame is None:
            first_hand_frame = self._get_hand_frame()
            time.sleep(0.01)

        self.hand_init_H = self._turn_frame_to_homo_mat(first_hand_frame)
        self.hand_init_t = copy(self.hand_init_H[:3, 3])

        logger.info("Hand init H:", self.hand_init_H)

        self.is_first_frame = False
        return first_hand_frame

    # ------------------------------
    # Main teleop: transforms
    # ------------------------------
    def _get_finger_coords(self):
        """Get transformed finger coordinates."""
        finger_coords = self._arm_transformed_keypoint_subscriber.recv_keypoints()
        if finger_coords is not None:
            return np.array(finger_coords).reshape(-1, 3)
        return None

    def _apply_retargeted_angles(self):
        """Apply retargeted angles from hand motion to robot."""
        # Get arm teleop state first
        new_arm_teleop_state = self._get_arm_teleop_state()

        # Get resolution scale (like Allegro)
        arm_teleoperation_scale_mode = self._get_resolution_scale_mode()
        if arm_teleoperation_scale_mode == ARM_HIGH_RESOLUTION:
            self.resolution_scale = 1
        elif arm_teleoperation_scale_mode == ARM_LOW_RESOLUTION:
            self.resolution_scale = 0.6

        # Check for reset or get hand frame
        is_reset = self.is_first_frame or (
            self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT
        )

        if is_reset:
            # Reposition initial values for the robot and hand, then get moving hand frame
            moving_hand_frame = self._reset_teleop()
            self.is_first_frame = False
        else:
            moving_hand_frame = self._get_hand_frame()

        # Update state before None check
        self.arm_teleop_state = new_arm_teleop_state

        if moving_hand_frame is None:
            return

        # Convert frame to homogeneous matrix
        self.hand_moving_H = self._turn_frame_to_homo_mat(moving_hand_frame)

        # Transformation code (same order as Allegro)
        h_hi_hh = copy(self.hand_init_H)
        h_ht_hh = copy(self.hand_moving_H)
        h_ri_rh = copy(self.robot_init_H)

        # Note: This is the transformation matrix for the hand frame to the robot frame
        # This transformation indicates that the hand frame is +x right, +y forward (out of the page), +z up
        h_r_v = np.array(
            [
                [0, 1, 0, 0],  # x axis becomes the y axis
                [1, 0, 0, 0],  # y axis becomes the x axis
                [0, 0, -1, 0],  # z axis remains the same
                [0, 0, 0, 1],
            ]
        )

        h_t_v = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # H_R_V = np.array([[-1, 0, 0, 0],
        #                 [0 , -1, 0, 0],
        #                 [0, 0, -1, 0],
        #                 [0, 0 ,0 , 1]])

        # H_R_V = np.array([[ 0, -1,  0,  0],
        #                   [ 0,  0, -1,  0],
        #                   [ 1,  0,  0,  0],
        #                   [ 0,  0,  0,  1]])

        # H_T_V = np.array([[-1, 0, 0, 0],
        #                 [0 , -1, 0, 0],
        #                 [0, 0, -1, 0],
        #                 [0, 0, 0, 1]])

        # H_T_V = np.array([[ 0, -1,  0,  0],
        #                   [ 0,  0, -1,  0],
        #                   [ 1,  0,  0,  0],
        #                   [ 0,  0,  0,  1]])

        # Calculate transformations (same order as Allegro)
        h_ht_hi = np.linalg.solve(h_hi_hh, h_ht_hh)  # More efficient than pinv for square matrices
        h_ht_hi_r = np.linalg.solve(h_r_v, h_ht_hi @ h_r_v)[:3, :3]
        h_ht_hi_t = np.linalg.solve(h_t_v, h_ht_hi @ h_t_v)[:3, 3]
        relative_affine = np.block([[h_ht_hi_r, h_ht_hi_t.reshape(3, 1)], [0, 0, 0, 1]])

        # Update robot state (like Allegro)
        h_rt_rh = h_ri_rh @ relative_affine
        # H_RT_RH = self.project_to_rotation_matrix(H_RT_RH)
        self.robot_moving_H = copy(h_rt_rh)

        # Get cart pose (this returns quaternion)
        # cart = self._homo2cart(h_rt_rh)
        # position = cart[0:3]
        # orientation = cart[3:7]  # Already quaternion [x,y,z,w]

        # After all transformations are done, publish the final robot pose
        final_robot_pose = {
            "position": self.robot_moving_H[:3, 3].tolist(),  # Extract translation
            "orientation": Rotation.from_matrix(self.robot_moving_H[:3, :3])
            .as_quat()
            .tolist(),  # Extract rotation as quaternion
            "timestamp": time.time(),
            "frame": "right_forearm_roll_link",
        }

        # Publish the pose
        self.end_eff_position_publisher.publish(final_robot_pose, "endeff_coords")

        if self.logging_enabled and self.pose_logger:
            try:
                self.pose_logger.log_frame(self.hand_init_H, self.robot_init_H, self.hand_moving_H, h_rt_rh)
            except Exception as e:
                logger.error(f"Error logging frame: {e}")

    def moving_average(self, action, queue, limit):
        """Apply moving average filter to action."""
        queue.append(action)
        if len(queue) > limit:
            queue.pop(0)
        return np.mean(queue, axis=0)

    def __del__(self):
        self._cleanup()

    def _cleanup(self):
        """Ensure logger saves all data before exit"""
        if self.pose_logger:
            self.pose_logger.close()

    def move_coords(self, cartesian_coords, duration=3):
        """
        Move robot to cartesian pose
        Args:
            cartesian_coords: dict with 'position' and 'orientation' keys
        """
        # Format for ROS controller
        pose = np.concatenate([cartesian_coords["position"], cartesian_coords["orientation"]])
        logger.info(pose)
        self._controller.arm_control(pose)
