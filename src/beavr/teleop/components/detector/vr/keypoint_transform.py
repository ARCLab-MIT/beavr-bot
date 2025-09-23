import logging
import time
from copy import deepcopy as copy
from enum import IntEnum

import numpy as np

from beavr.teleop.common.math.vectorops import moving_average, normalize_vector
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import InputFrame
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class HandMode(IntEnum):
    ABSOLUTE = 1
    RELATIVE = 0


class TransformHandPositionCoords(Component):
    def __init__(
        self,
        host: str,
        keypoint_sub_port: int,
        keypoint_transform_pub_port: int,
        hand_side: str = robots.RIGHT,
        moving_average_limit: int = 5,
    ):
        """
        Initialize the unified keypoint transform component for both left and right hands.

        Args:
            host: Network host address
            keypoint_sub_port: Port to subscribe to keypoints from
            keypoint_transform_pub_port: Port to publish transformed keypoints to
            hand_side: 'right' or 'left' to specify which hand to process
            moving_average_limit: Number of frames for moving average smoothing
        """
        # Validate hand_side parameter
        if hand_side not in [robots.LEFT, robots.RIGHT]:
            raise ValueError(f"hand_side must be {robots.LEFT} or {robots.RIGHT}")

        self.hand_side = hand_side

        # Notify component start with appropriate name
        component_name = f"{hand_side}_hand_keypoint_transform"
        self.notify_component_start(component_name)

        # Store connection info
        self.host = host
        self.keypoint_sub_port = keypoint_sub_port
        self.keypoint_transform_pub_port = keypoint_transform_pub_port

        # Initialize subscriber based on hand side
        if hand_side == robots.RIGHT:
            self.keypoint_subscriber = ZMQSubscriber(self.host, self.keypoint_sub_port, robots.RIGHT)
        else:  # left hand
            self.keypoint_subscriber = ZMQSubscriber(self.host, self.keypoint_sub_port, robots.LEFT)

        # Use publisher manager for both hands consistently
        self.publisher_manager = ZMQPublisherManager.get_instance()

        # Define topic names based on hand side
        if hand_side == robots.RIGHT:
            self.coords_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_COORDS}"
            self.frame_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}"
            self.absolute_mode = robots.ABSOLUTE
            self.relative_mode = robots.RELATIVE
        else:
            self.coords_topic = f"{robots.LEFT}_{robots.TRANSFORMED_HAND_COORDS}"
            self.frame_topic = f"{robots.LEFT}_{robots.TRANSFORMED_HAND_FRAME}"
            self.absolute_mode = robots.ABSOLUTE
            self.relative_mode = robots.RELATIVE

        # Timer
        self.timer = FrequencyTimer(robots.VR_FREQ)

        # Define key landmark indices for stable frame calculation
        self.wrist_idx = 0  # Wrist is typically the first point
        self.index_knuckle_idx = robots.OCULUS_JOINTS["knuckles"][0]  # Index finger knuckle
        self.middle_knuckle_idx = robots.OCULUS_JOINTS["knuckles"][1]  # Middle finger knuckle
        self.pinky_knuckle_idx = robots.OCULUS_JOINTS["knuckles"][-1]  # Pinky finger knuckle

        # Moving average queue
        self.moving_average_limit = moving_average_limit
        # Create a queue for moving average
        self.coord_moving_average_queue, self.frame_moving_average_queue = [], []

    def _get_hand_coords(self):
        """Get hand coordinates from the subscriber.

        Returns:
            Tuple of (data_type, coordinates) or (None, None) if no data received
        """
        input_frame = self.keypoint_subscriber.recv_keypoints()
        if input_frame is None:
            return None, None

        # Extract keypoints from InputFrame object
        keypoints = np.asanyarray(input_frame.keypoints)

        # Determine data type from is_relative field
        data_type = self.relative_mode if input_frame.is_relative else self.absolute_mode

        return data_type, keypoints.reshape(robots.OCULUS_NUM_KEYPOINTS, 3)

    def _orthogonalize_frame(self, x_vec, y_vec, z_vec):
        """Ensure three vectors form an orthogonal frame using Gram-Schmidt process"""
        # Ensure x is normalized
        x_vec = normalize_vector(x_vec)

        # Make y orthogonal to x
        y_vec = y_vec - np.dot(y_vec, x_vec) * x_vec
        y_vec = normalize_vector(y_vec)

        # Make z orthogonal to both x and y
        z_vec = np.cross(x_vec, y_vec)
        z_vec = normalize_vector(z_vec)

        return x_vec, y_vec, z_vec

    def _get_stable_coord_frame(self, hand_coords):
        """Create a more stable coordinate frame using multiple hand landmarks"""
        wrist = hand_coords[self.wrist_idx]
        v1 = hand_coords[self.index_knuckle_idx] - wrist
        v2 = hand_coords[self.pinky_knuckle_idx] - wrist
        v3 = hand_coords[self.middle_knuckle_idx] - wrist

        # Calculate frame vectors using multiple references
        palm_normal = normalize_vector(np.cross(v1, v3))  # Z direction
        palm_direction = normalize_vector((v1 + v2 + v3) / 3)  # Y direction
        cross_product = normalize_vector(np.cross(palm_direction, palm_normal))  # X direction

        # Orthogonalize explicitly to ensure a valid rotation matrix
        x_vec, y_vec, z_vec = self._orthogonalize_frame(cross_product, palm_direction, palm_normal)

        # Return as a list of vectors for compatibility with existing code
        return [x_vec, y_vec, z_vec]

    def _get_stable_hand_dir_frame(self, hand_coords):
        """Create a more stable frame for hand direction using multiple landmarks"""
        wrist = hand_coords[self.wrist_idx]
        v1 = hand_coords[self.index_knuckle_idx] - wrist
        v2 = hand_coords[self.pinky_knuckle_idx] - wrist
        v3 = hand_coords[self.middle_knuckle_idx] - wrist

        # Calculate frame vectors using multiple references
        if self.hand_side == robots.RIGHT:
            # Right hand coordinate frame calculation
            palm_normal = normalize_vector(np.cross(v1, v3))  # Unity space - Y
            palm_direction = normalize_vector((v1 + v2 + v3) / 3)  # Unity space - Z
            cross_product = normalize_vector(np.cross(palm_direction, palm_normal))  # Unity space - X
        else:
            # Left hand coordinate frame calculation (slightly different)
            palm_normal = normalize_vector(np.cross(v1, v3))  # Unity space - Y
            palm_direction = normalize_vector((v1 + v2 + v3) / 3)  # Unity space - Z
            cross_product = normalize_vector(np.cross(palm_direction, palm_normal))  # Unity space - X

        # Orthogonalize explicitly
        x_vec, y_vec, z_vec = self._orthogonalize_frame(cross_product, palm_normal, palm_direction)

        return [wrist, x_vec, y_vec, z_vec]

    def transform_keypoints(self, hand_coords):
        """Transform hand keypoints to create a coordinate frame."""
        translated_coords = copy(hand_coords) - hand_coords[0]

        # Use the new, more stable coordinate frame method
        original_coord_frame = self._get_stable_coord_frame(translated_coords)

        # Finding the rotation matrix and rotating the coordinates
        rotation_matrix = np.linalg.solve(original_coord_frame, np.eye(3)).T
        transformed_hand_coords = (rotation_matrix @ translated_coords.T).T

        # Use the new, more stable hand direction frame method
        hand_dir_frame = self._get_stable_hand_dir_frame(hand_coords)

        return transformed_hand_coords, hand_dir_frame

    def stream(self):
        """Main streaming loop for processing hand keypoints."""
        while True:
            self.timer.start_loop()
            data_type, hand_coords = self._get_hand_coords()

            # If no data was available just continue the loop
            if hand_coords is None or data_type is None:
                self.timer.end_loop()
                continue

            # Shift the points to required axes
            (
                transformed_hand_coords,
                translated_hand_coord_frame,
            ) = self.transform_keypoints(hand_coords)

            # Passing the transformed coords into a moving average
            self.averaged_hand_coords = moving_average(
                transformed_hand_coords,
                self.coord_moving_average_queue,
                self.moving_average_limit,
            )

            # Apply moving average to frame vectors
            self.averaged_hand_frame = moving_average(
                translated_hand_coord_frame,
                self.frame_moving_average_queue,
                self.moving_average_limit,
            )

            # Ensure frame vectors remain orthogonal regardless of data type
            # Keep origin point as is
            origin = self.averaged_hand_frame[0]
            # Extract the rotation vectors
            x_vec = normalize_vector(self.averaged_hand_frame[1])
            y_vec = normalize_vector(self.averaged_hand_frame[2])
            z_vec = normalize_vector(self.averaged_hand_frame[3])

            # Re-orthogonalize the frame
            x_vec, y_vec, z_vec = self._orthogonalize_frame(x_vec, y_vec, z_vec)

            # Reconstruct orthogonal frame
            self.averaged_hand_frame = [origin, x_vec, y_vec, z_vec]

            data = InputFrame(
                timestamp_s=time.time(),
                hand_side=self.hand_side,
                keypoints=self.averaged_hand_coords,
                is_relative=data_type == self.relative_mode,
                frame_vectors=self.averaged_hand_frame,
            )

            # Publish both transformed coords and frame for consumers
            self.publisher_manager.publish(
                host=self.host,
                port=self.keypoint_transform_pub_port,
                topic=self.coords_topic,
                data=data,
            )
            self.publisher_manager.publish(
                host=self.host,
                port=self.keypoint_transform_pub_port,
                topic=self.frame_topic,
                data=data,
            )

            self.timer.end_loop()

    # Cleanup
    def cleanup(self):
        self.keypoint_subscriber.stop()
        cleanup_zmq_resources()

    def __del__(self):
        self.cleanup()
