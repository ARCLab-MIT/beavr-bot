import json
import logging
import time
from copy import deepcopy as copy
from datetime import datetime
from enum import IntEnum
from pathlib import Path

import numpy as np

from beavr.teleop.common.math.vectorops import moving_average, normalize_vector
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import InputFrame
from beavr.teleop.components.detector.vr.log_keypoints import KeypointLogger
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


def contains_nan(arr: np.ndarray) -> bool:
    """Check if numpy array contains any NaN or Inf values."""
    if arr is None:
        return True
    if not isinstance(arr, np.ndarray):
        return False
    bool_result = bool(np.any(np.isnan(arr)))
    if bool_result:
        pass
    return bool_result or bool(np.any(np.isinf(arr)))


def check_array_validity(arr: np.ndarray, name: str) -> bool:
    """Check if array is valid and log detailed information about issues."""
    if arr is None:
        logger.warning(f"{name}: Array is None")
        return False

    if not isinstance(arr, np.ndarray):
        logger.warning(f"{name}: Expected numpy array, got {type(arr)}")
        return False

    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)

    has_nan = np.any(nan_mask)
    has_inf = np.any(inf_mask)

    if has_nan or has_inf:
        nan_count = np.sum(nan_mask)
        inf_count = np.sum(inf_mask)
        logger.warning(
            f"{name}: Found {nan_count} NaN values and {inf_count} Inf values. "
            f"Shape: {arr.shape}, dtype: {arr.dtype}, min/max: {np.nanmin(arr)}/{np.nanmax(arr)}"
        )
        if arr.size < 20:
            logger.warning(f"{name} full array:\n{arr}")
        return False

    return True


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
        enable_logging: bool = False,
        log_dir: str = "data/keypoint_logs",
        auto_save_interval: int = 100,
    ):
        """
        Initialize the unified keypoint transform component for both left and right hands.

        Args:
            host: Network host address
            keypoint_sub_port: Port to subscribe to keypoints from
            keypoint_transform_pub_port: Port to publish transformed keypoints to
            hand_side: 'right' or 'left' to specify which hand to process
            moving_average_limit: Number of frames for moving average smoothing
            enable_logging: Flag to enable/disable frame logging (default: False)
            log_dir: Directory to save log files (default: "data/keypoint_logs")
            auto_save_interval: Number of frames between auto-saves (default: 100)
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

        # Initialize keypoint logger if enabled
        self.keypoint_logger = None
        if enable_logging:
            self.keypoint_logger = KeypointLogger(
                hand_side=hand_side,
                log_dir=log_dir,
                auto_save_interval=auto_save_interval,
                moving_average_limit=moving_average_limit,
            )

        # Raw keypoint recording for debugging
        self.raw_keypoint_records = []
        self.raw_keypoint_log_file = None
        if enable_logging:
            raw_log_dir = Path(log_dir)
            raw_log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.raw_keypoint_log_file = raw_log_dir / f"raw_keypoints_{hand_side}_{timestamp}.json"

    def _log_raw_keypoints(self, keypoints: np.ndarray, is_relative: bool):
        """Record raw keypoints from VR device for debugging purposes."""
        if self.raw_keypoint_log_file is None:
            return

        record = {
            "timestamp": time.time(),
            "shape": list(keypoints.shape),
            "size": int(keypoints.size),
            "is_relative": is_relative,
            "keypoints": keypoints.tolist() if keypoints.size < 1000 else f"<large array: {keypoints.size} elements>",
        }
        self.raw_keypoint_records.append(record)

        # Auto-save every 100 records
        if len(self.raw_keypoint_records) % 100 == 0:
            self._save_raw_keypoints()

    def _save_raw_keypoints(self):
        """Save raw keypoint records to JSON file."""
        if self.raw_keypoint_log_file is None or len(self.raw_keypoint_records) == 0:
            return

        try:
            data = {
                "hand_side": self.hand_side,
                "total_records": len(self.raw_keypoint_records),
                "records": self.raw_keypoint_records.copy(),
            }
            with open(self.raw_keypoint_log_file, "w") as f:
                json.dump(data, f, indent=2)
            # logger.debug(f"Saved {len(self.raw_keypoint_records)} raw keypoint records")
            self.raw_keypoint_records.clear()
        except Exception as e:
            logger.error(f"Error saving raw keypoints: {e}")

    def _get_hand_coords(self):
        """Get hand coordinates from the subscriber.

        Returns:
            Tuple of (data_type, coordinates) or (None, None) if no data received
        """
        try:
            input_frame = self.keypoint_subscriber.recv_keypoints()
            if input_frame is None:
                return None, None

            # Extract keypoints from InputFrame object
            keypoints = np.asanyarray(input_frame.keypoints)

            # Check for NaN/Inf in raw keypoints
            if not check_array_validity(keypoints, "raw_keypoints"):
                logger.error("Received keypoints contain invalid values (NaN/Inf)")
                self._log_raw_keypoints(keypoints, input_frame.is_relative)
                return None, None

            # Record raw keypoints from VR device for debugging
            # logger.debug(f"Raw keypoints shape: {keypoints.shape}, data: {keypoints}")
            self._log_raw_keypoints(keypoints, input_frame.is_relative)

            # Determine data type from is_relative field
            data_type = self.relative_mode if input_frame.is_relative else self.absolute_mode

            # return data_type, keypoints.reshape(robots.OCULUS_NUM_KEYPOINTS, 3)
            expected_size = robots.OCULUS_NUM_KEYPOINTS * 3
            if keypoints.size != expected_size:
                logger.error(
                    f"Invalid keypoints size: got {keypoints.size}, expected {expected_size}. Raw data: {keypoints}"
                )
                return None, None

            reshaped_keypoints = keypoints.reshape(robots.OCULUS_NUM_KEYPOINTS, 3)

            # Check reshaped keypoints for validity
            if not check_array_validity(reshaped_keypoints, "reshaped_keypoints"):
                logger.error("Reshaped keypoints contain invalid values")
                return None, None

            return data_type, reshaped_keypoints

        except Exception as e:
            logger.error(f"Error in _get_hand_coords: {e}", exc_info=True)
            return None, None

    def _orthogonalize_frame(self, x_vec, y_vec, z_vec):
        """Ensure three vectors form an orthogonal frame using Gram-Schmidt process"""
        # Check validity of input vectors
        if (
            not check_array_validity(x_vec, "x_vec_orthogonalize")
            or not check_array_validity(y_vec, "y_vec_orthogonalize")
            or not check_array_validity(z_vec, "z_vec_orthogonalize")
        ):
            logger.error("Input vectors for orthogonalization contain invalid values")
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])

        x_vec = x_vec.astype(np.float64)
        y_vec = y_vec.astype(np.float64)
        z_vec = z_vec.astype(np.float64)

        # Ensure x is normalized
        x_vec = normalize_vector(x_vec)
        check_array_validity(x_vec, "normalized_x_vec")

        # Make y orthogonal to x
        y_vec = y_vec - np.dot(y_vec, x_vec) * x_vec
        y_vec = normalize_vector(y_vec)
        check_array_validity(y_vec, "normalized_y_vec")

        # Make z orthogonal to both x and y
        z_vec = np.cross(x_vec, y_vec)
        z_vec = normalize_vector(z_vec)
        check_array_validity(z_vec, "normalized_z_vec")

        return x_vec, y_vec, z_vec

    def _get_stable_coord_frame(self, hand_coords):
        """Create a more stable coordinate frame using multiple hand landmarks"""
        try:
            # Check input validity
            if not check_array_validity(hand_coords, "hand_coords_stable_frame"):
                logger.error("Invalid hand coordinates for stable frame calculation")
                return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            
            # Check individual keypoints before calculations
            wrist = hand_coords[self.wrist_idx]
            index_pos = hand_coords[self.index_knuckle_idx]
            pinky_pos = hand_coords[self.pinky_knuckle_idx]
            middle_pos = hand_coords[self.middle_knuckle_idx]
            
            if contains_nan(wrist):
                logger.error(
                    f"WRIST position contains NaN: {wrist}. "
                    f"Keypoint indices: wrist={self.wrist_idx}, index={self.index_knuckle_idx}, "
                    f"pinky={self.pinky_knuckle_idx}, middle={self.middle_knuckle_idx}"
                )
                logger.error(f"Full hand_coords shape: {hand_coords.shape}, sample coords:\n{hand_coords[:5]}")
                return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            
            if contains_nan(index_pos):
                logger.error(f"INDEX knuckle position contains NaN: {index_pos}")
                return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            
            if contains_nan(pinky_pos):
                logger.error(f"PINKY knuckle position contains NaN: {pinky_pos}")
                return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            
            if contains_nan(middle_pos):
                logger.error(f"MIDDLE knuckle position contains NaN: {middle_pos}")
                return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            
            v1 = hand_coords[self.index_knuckle_idx] - wrist
            v2 = hand_coords[self.pinky_knuckle_idx] - wrist
            v3 = hand_coords[self.middle_knuckle_idx] - wrist

            # Check vector calculations
            if not check_array_validity(v1, "v1_index_knuckle"):
                logger.error("Invalid v1 vector (index knuckle)")
            if not check_array_validity(v2, "v2_pinky_knuckle"):
                logger.error("Invalid v2 vector (pinky knuckle)")
            if not check_array_validity(v3, "v3_middle_knuckle"):
                logger.error("Invalid v3 vector (middle knuckle)")

            # Calculate frame vectors using multiple references
            cross_v1_v3 = np.cross(v1, v3)
            palm_normal = normalize_vector(cross_v1_v3)  # Z direction

            avg_direction = (v1 + v2 + v3) / 3
            palm_direction = normalize_vector(avg_direction)  # Y direction

            cross_palm = np.cross(palm_direction, palm_normal)
            cross_product = normalize_vector(cross_palm)  # X direction

            # Log if any normalization step produced invalid results
            if not check_array_validity(palm_normal, "palm_normal"):
                logger.warning("Palm normal normalization produced invalid values, using fallback")
                palm_normal = np.array([0.0, 0.0, 1.0])

            if not check_array_validity(palm_direction, "palm_direction"):
                logger.warning("Palm direction normalization produced invalid values, using fallback")
                palm_direction = np.array([0.0, 1.0, 0.0])

            if not check_array_validity(cross_product, "cross_product"):
                logger.warning("Cross product normalization produced invalid values, using fallback")
                cross_product = np.array([1.0, 0.0, 0.0])

            # Orthogonalize explicitly to ensure a valid rotation matrix
            x_vec, y_vec, z_vec = self._orthogonalize_frame(cross_product, palm_direction, palm_normal)

            # Return as a list of vectors for compatibility with existing code
            return [x_vec, y_vec, z_vec]

        except Exception as e:
            logger.error(f"Error in _get_stable_coord_frame: {e}", exc_info=True)
            return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

    def _get_stable_hand_dir_frame(self, hand_coords):
        """Create a more stable frame for hand direction using multiple landmarks"""
        try:
            # Check input validity
            if not check_array_validity(hand_coords, "hand_coords_dir_frame"):
                logger.error("Invalid hand coordinates for direction frame calculation")
                return [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                ]
            
            # Check individual keypoints before calculations
            wrist = hand_coords[self.wrist_idx]
            index_pos = hand_coords[self.index_knuckle_idx]
            pinky_pos = hand_coords[self.pinky_knuckle_idx]
            middle_pos = hand_coords[self.middle_knuckle_idx]
            
            if contains_nan(wrist):
                logger.error(
                    f"DIR FRAME - WRIST position contains NaN: {wrist}. "
                    f"Keypoint indices: wrist={self.wrist_idx}, index={self.index_knuckle_idx}, "
                    f"pinky={self.pinky_knuckle_idx}, middle={self.middle_knuckle_idx}"
                )
                logger.error(f"DIR FRAME - Full hand_coords shape: {hand_coords.shape}, sample coords:\n{hand_coords[:5]}")
                return [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 
                        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            
            if contains_nan(index_pos):
                logger.error(f"DIR FRAME - INDEX knuckle position contains NaN: {index_pos}")
                return [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 
                        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            
            if contains_nan(pinky_pos):
                logger.error(f"DIR FRAME - PINKY knuckle position contains NaN: {pinky_pos}")
                return [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 
                        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            
            if contains_nan(middle_pos):
                logger.error(f"DIR FRAME - MIDDLE knuckle position contains NaN: {middle_pos}")
                return [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 
                        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

            wrist = wrist.copy()
            v1 = hand_coords[self.index_knuckle_idx] - wrist
            v2 = hand_coords[self.pinky_knuckle_idx] - wrist
            v3 = hand_coords[self.middle_knuckle_idx] - wrist

            # Check vector calculations
            if not check_array_validity(v1, "v1_direction"):
                logger.error("Invalid v1 vector in direction frame")
            if not check_array_validity(v2, "v2_direction"):
                logger.error("Invalid v2 vector in direction frame")
            if not check_array_validity(v3, "v3_direction"):
                logger.error("Invalid v3 vector in direction frame")
            if not check_array_validity(wrist, "wrist_position"):
                logger.error("Invalid wrist position in direction frame")

            # Calculate frame vectors using multiple references
            cross_v1_v3 = np.cross(v1, v3)
            palm_normal = normalize_vector(cross_v1_v3)

            avg_direction = (v1 + v2 + v3) / 3
            palm_direction = normalize_vector(avg_direction)

            cross_palm = np.cross(palm_direction, palm_normal)
            cross_product = normalize_vector(cross_palm)

            # Validate normalized vectors
            if not check_array_validity(palm_normal, "palm_normal_dir"):
                logger.warning("Palm normal normalization failed in direction frame, using fallback")
                palm_normal = np.array([0.0, -1.0, 0.0])

            if not check_array_validity(palm_direction, "palm_direction_dir"):
                logger.warning("Palm direction normalization failed in direction frame, using fallback")
                palm_direction = np.array([0.0, 0.0, 1.0])

            if not check_array_validity(cross_product, "cross_product_dir"):
                logger.warning("Cross product normalization failed in direction frame, using fallback")
                cross_product = np.array([1.0, 0.0, 0.0])

            # Orthogonalize explicitly
            x_vec, y_vec, z_vec = self._orthogonalize_frame(cross_product, palm_normal, palm_direction)

            # Validate final frame vectors
            if not check_array_validity(x_vec, "x_vec_final"):
                logger.warning("Final x_vector invalid, resetting to basis")
                x_vec = np.array([1.0, 0.0, 0.0])
            if not check_array_validity(y_vec, "y_vec_final"):
                logger.warning("Final y_vector invalid, resetting to basis")
                y_vec = np.array([0.0, 1.0, 0.0])
            if not check_array_validity(z_vec, "z_vec_final"):
                logger.warning("Final z_vector invalid, resetting to basis")
                z_vec = np.array([0.0, 0.0, 1.0])

            return [wrist, x_vec, y_vec, z_vec]

        except Exception as e:
            logger.error(f"Error in _get_stable_hand_dir_frame: {e}", exc_info=True)
            return [
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ]

    def transform_keypoints(self, hand_coords):
        """
        Input: Hand coordinates from VR detector (N,3)
        Returns:
        - Transformed hand coordinates (N,3) - Hand keypoints translated to "VR frame"
        - Hand direction frame (4,3) - Coordinate frame vectors for "VR frame"
        """
        try:
            # Check input validity
            if not check_array_validity(hand_coords, "hand_coords_transform"):
                logger.error("Invalid hand coordinates in transform_keypoints")
                return np.zeros_like(hand_coords), [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                ]

            translated_coords = copy(hand_coords) - hand_coords[0]

            if not check_array_validity(translated_coords, "translated_coords"):
                logger.error("Translated coordinates contain invalid values")
                return np.zeros_like(hand_coords), [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                ]

            # Use the new, more stable coordinate frame method
            original_coord_frame = self._get_stable_coord_frame(translated_coords)

            # Validate coordinate frame
            coord_frame_array = np.array(original_coord_frame)
            if not check_array_validity(coord_frame_array, "original_coord_frame"):
                logger.warning("Original coordinate frame contains invalid values, using identity")
                original_coord_frame = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
                coord_frame_array = np.array(original_coord_frame)

            # Finding the rotation matrix and rotating the coordinates
            coord_frame_stack = np.stack(original_coord_frame, axis=1)

            try:
                rotation_matrix = np.linalg.solve(coord_frame_stack, np.eye(3)).T

                if not check_array_validity(rotation_matrix, "rotation_matrix"):
                    logger.error("Rotation matrix contains NaN/Inf values")
                    # Use identity matrix as fallback
                    rotation_matrix = np.eye(3)

                # Log matrix condition number to detect ill-conditioned matrices
                cond_number = np.linalg.cond(rotation_matrix)
                if cond_number > 1e10:
                    logger.warning(f"Rotation matrix is ill-conditioned. Condition number: {cond_number:.2e}")

            except np.linalg.LinAlgError as e:
                logger.error(f"Singular matrix in transform_keypoints: {e}")
                rotation_matrix = np.eye(3)
            except Exception as e:
                logger.error(f"Unexpected error computing rotation matrix: {e}", exc_info=True)
                rotation_matrix = np.eye(3)

            transformed_keypoints = (rotation_matrix @ translated_coords.T).T

            if not check_array_validity(transformed_keypoints, "transformed_keypoints"):
                logger.error("Transformed keypoints contain invalid values, using translated coords as fallback")
                transformed_keypoints = translated_coords

            # Use the new, more stable hand direction frame method
            coordinate_frame = self._get_stable_hand_dir_frame(hand_coords)

            # Validate coordinate frame result
            frame_valid = True
            for i, vec in enumerate(coordinate_frame):
                if not check_array_validity(vec, f"coordinate_frame_vec_{i}"):
                    frame_valid = False
                    logger.warning(f"Coordinate frame vector {i} contains invalid values")

            if not frame_valid:
                logger.warning("Reconstructing coordinate frame with fallback values")
                coordinate_frame = [
                    hand_coords[0].copy(),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                ]

            return transformed_keypoints, coordinate_frame

        except Exception as e:
            logger.error(f"Error in transform_keypoints: {e}", exc_info=True)
            return np.zeros_like(hand_coords), [
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ]

    def _log_frame(self, keypoints, coordinate_frame):
        """Log frame data if logging is enabled."""
        if self.keypoint_logger is not None:
            self.keypoint_logger.log_frame(keypoints, coordinate_frame)

    def stream(self):
        """Main streaming loop for processing hand keypoints."""
        frame_count = 0
        error_count = 0
        max_errors = 100

        logger.info(f"Starting keypoint transform stream for {self.hand_side} hand")
        logger.info(f"Moving average limit: {self.moving_average_limit}")
        logger.info(f"Logging enabled: {self.keypoint_logger is not None}")

        while True:
            try:
                self.timer.start_loop()
                frame_count += 1

                data_type, hand_coords = self._get_hand_coords()

                # If no data was available just continue the loop
                if hand_coords is None or data_type is None:
                    # Todo: is this neccessary
                    if frame_count % 100 == 0:
                        logger.debug(f"Frame {frame_count}: No data available")
                    self.timer.end_loop()
                    continue

                # Shift the points to required axes
                (
                    transformed_keypoints,
                    coordinate_frame,
                ) = self.transform_keypoints(hand_coords)

                # Validate transformation results
                if not check_array_validity(transformed_keypoints, "transformed_keypoints_after_transform"):
                    logger.error(f"Frame {frame_count}: Transformed keypoints are invalid")
                    error_count += 1
                    self.timer.end_loop()
                    if error_count >= max_errors:
                        logger.error(f"Too many errors ({error_count}), stopping stream")
                        break
                    continue

                # Passing the transformed coords into a moving average
                self.averaged_keypoints = moving_average(
                    transformed_keypoints,
                    self.coord_moving_average_queue,
                    self.moving_average_limit,
                )

                # Apply moving average to frame vectors
                self.averaged_coordinate_frame = moving_average(
                    coordinate_frame,
                    self.frame_moving_average_queue,
                    self.moving_average_limit,
                )

                # Ensure frame vectors remain orthogonal regardless of data type
                # Keep origin point as is
                origin = self.averaged_coordinate_frame[0]

                # Check origin validity
                if contains_nan(origin):
                    logger.warning(f"Frame {frame_count}: Origin contains NaN, using hand_coords[0] as fallback")
                    origin = hand_coords[0].copy()

                # Extract the rotation vectors
                x_vec = normalize_vector(self.averaged_coordinate_frame[1])
                y_vec = normalize_vector(self.averaged_coordinate_frame[2])
                z_vec = normalize_vector(self.averaged_coordinate_frame[3])

                # Check if normalization produced valid results
                if contains_nan(x_vec) or contains_nan(y_vec) or contains_nan(z_vec):
                    logger.warning(
                        f"Frame {frame_count}: Frame vectors contain NaN after normalization, using fallback"
                    )
                    x_vec = np.array([1.0, 0.0, 0.0])
                    y_vec = np.array([0.0, 1.0, 0.0])
                    z_vec = np.array([0.0, 0.0, 1.0])

                # Re-orthogonalize the frame
                x_vec, y_vec, z_vec = self._orthogonalize_frame(x_vec, y_vec, z_vec)

                # Validate re-orthogonalization results
                if contains_nan(x_vec) or contains_nan(y_vec) or contains_nan(z_vec):
                    logger.error(f"Frame {frame_count}: Failed to produce valid orthogonal frame")
                    x_vec = np.array([1.0, 0.0, 0.0])
                    y_vec = np.array([0.0, 1.0, 0.0])
                    z_vec = np.array([0.0, 0.0, 1.0])

                # Reconstruct orthogonal frame
                self.averaged_coordinate_frame = [origin, x_vec, y_vec, z_vec]

                # Validate averaged keypoints
                if not check_array_validity(self.averaged_keypoints, "averaged_keypoints"):
                    logger.error(f"Frame {frame_count}: Averaged keypoints are invalid")
                    error_count += 1
                    self.timer.end_loop()
                    if error_count >= max_errors:
                        logger.error(f"Too many errors ({error_count}), stopping stream")
                        break
                    continue

                data = InputFrame(
                    timestamp_s=time.time(),
                    hand_side=self.hand_side,
                    keypoints=self.averaged_keypoints,
                    is_relative=data_type == self.relative_mode,
                    frame_vectors=self.averaged_coordinate_frame,
                )

                # Validate data before publishing
                if not check_array_validity(data.keypoints, "published_keypoints"):
                    logger.error(f"Frame {frame_count}: Keypoints to publish are invalid")
                    error_count += 1
                    self.timer.end_loop()
                    if error_count >= max_errors:
                        logger.error(f"Too many errors ({error_count}), stopping stream")
                        break
                    continue

                # Publish both transformed coords and frame for consumers
                self.publisher_manager.publish(
                    host=self.host,
                    port=self.keypoint_transform_pub_port,
                    topic=self.coords_topic,
                    data=data,
                )

                # Log and save in a JSON file the hand keypoints and the coordinate frame
                if self.keypoint_logger is not None:
                    self._log_frame(self.averaged_keypoints, self.averaged_coordinate_frame)
                # This is redundant
                # TODO: Remove this and modify LEAPoperator to use coords topic
                self.publisher_manager.publish(
                    host=self.host,
                    port=self.keypoint_transform_pub_port,
                    topic=self.frame_topic,
                    data=data,
                )

                # Log status periodically
                if frame_count % 1000 == 0:
                    logger.info(f"Processed {frame_count} frames successfully. Error count: {error_count}")

                self.timer.end_loop()

            except Exception as e:
                logger.error(f"Error in stream loop at frame {frame_count}: {e}", exc_info=True)
                error_count += 1
                self.timer.end_loop()  # Ensure timer loop is ended even on error

                if error_count >= max_errors:
                    logger.error(f"Too many errors ({error_count}), stopping stream")
                    break

                # Small delay to prevent tight error loops
                time.sleep(0.01)

    # Cleanup
    def cleanup(self):
        """Clean up resources and save any remaining logged data."""
        # Save logged data before cleanup
        if self.keypoint_logger is not None:
            logger.info("Cleanup called. Saving final logged data...")
            self.keypoint_logger.save_data()

        # Save raw keypoint records
        if len(self.raw_keypoint_records) > 0:
            self._save_raw_keypoints()

        self.keypoint_subscriber.stop()
        cleanup_zmq_resources()

    def __del__(self):
        self.cleanup()
