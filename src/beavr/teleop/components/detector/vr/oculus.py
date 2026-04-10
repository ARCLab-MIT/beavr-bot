import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import zmq

from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.utils import create_pull_socket
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import (
    ButtonEvent,
    InputFrame,
    SessionCommand,
)
from beavr.teleop.configs.constants import network, robots

logger = logging.getLogger(__name__)


class OculusVRHandDetector(Component):
    """
    Unified OculusVRHandDetector that can handle left, right, or bimanual hand detection.

    This class dynamically configures itself based on the provided hand configuration,
    eliminating the need for separate single-hand and bimanual detector classes.
    """

    def __init__(
        self,
        host: str,
        oculus_pub_port: int,
        button_port: int,
        teleop_reset_port: int,
        hand_config: Union[str, str] = robots.RIGHT,
        right_hand_port: Optional[int] = None,
        left_hand_port: Optional[int] = None,
    ):
        """
        Initialize the unified OculusVRHandDetector component.

        Args:
            host: The host address of the Oculus VR headset.
            oculus_pub_port: The port number for publishing keypoint data.
            button_port: The port number for button events.
            teleop_reset_port: The port number for teleop reset commands.
            hand_config: Configuration mode - 'left', 'right', or 'bimanual'
            right_hand_port: Port for right hand data (required for right/bimanual)
            left_hand_port: Port for left hand data (required for left/bimanual)
        """
        self.notify_component_start(robots.VR_DETECTOR)

        self.host = host
        self.oculus_pub_port = oculus_pub_port
        self.button_port = button_port
        self.teleop_reset_port = teleop_reset_port
        self.hand_config = hand_config

        # Validate and set hand ports based on configuration
        self._configure_hand_ports(right_hand_port, left_hand_port)

        # Initialize sockets based on configuration
        self._initialize_sockets()

        # Initialize publisher and timing
        self.publisher_manager = ZMQPublisherManager.get_instance()
        self.timer = FrequencyTimer(robots.VR_FREQ)
        self.last_received = dict.fromkeys(self.sockets, 0)

        # Raw keypoint recording for debugging
        self.raw_keypoint_records = {hand_side: [] for hand_side in self.hand_ports}
        self.raw_keypoint_log_files = {}
        raw_log_dir = Path("data/keypoint_logs")
        raw_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for hand_side in self.hand_ports:
            self.raw_keypoint_log_files[hand_side] = raw_log_dir / f"raw_vr_data_{hand_side}_{timestamp}.json"

    def _log_raw_vr_data(
        self,
        hand_side: str,
        raw_data: bytes,
        processed_keypoints: list,
        rotated_keypoints: list | None = None,
    ):
        """Record raw VR data for debugging purposes."""
        if hand_side not in self.raw_keypoint_log_files:
            return

        record = {
            "timestamp": time.time(),
            "raw_bytes": raw_data.decode().strip() if raw_data else None,
            "processed_keypoints": processed_keypoints,
            "keypoints_shape": len(processed_keypoints),
            "rotated_keypoints": rotated_keypoints if rotated_keypoints is not None else [],
        }
        self.raw_keypoint_records[hand_side].append(record)

        # Auto-save every 500 records
        if len(self.raw_keypoint_records[hand_side]) % 500 == 0:
            self._save_raw_vr_data(hand_side)

    def _save_raw_vr_data(self, hand_side: str):
        """Save raw VR data records to JSON file."""
        if hand_side not in self.raw_keypoint_log_files or len(self.raw_keypoint_records[hand_side]) == 0:
            return

        try:
            data = {
                "hand_side": hand_side,
                "total_records": len(self.raw_keypoint_records[hand_side]),
                "records": self.raw_keypoint_records[hand_side].copy(),
            }
            with open(self.raw_keypoint_log_files[hand_side], "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.raw_keypoint_records[hand_side])} raw VR records for {hand_side}")
            self.raw_keypoint_records[hand_side].clear()
        except Exception as e:
            logger.error(f"Error saving raw VR data for {hand_side}: {e}")

    def _configure_hand_ports(self, right_hand_port: Optional[int], left_hand_port: Optional[int]):
        """Configure hand ports based on the hand configuration."""
        self.hand_ports = {}

        if self.hand_config in [robots.RIGHT, robots.BIMANUAL]:
            if right_hand_port is None:
                right_hand_port = network.RIGHT_HAND_PORT
            self.hand_ports[robots.RIGHT] = right_hand_port

        if self.hand_config in [robots.LEFT, robots.BIMANUAL]:
            if left_hand_port is None:
                left_hand_port = network.LEFT_HAND_PORT
            self.hand_ports[robots.LEFT] = left_hand_port

    def _initialize_sockets(self):
        """Initialize sockets based on hand configuration."""
        self.sockets = {}

        # Create hand-specific keypoint sockets
        for hand_side, port in self.hand_ports.items():
            socket_key = f"{robots.KEYPOINTS}_{hand_side}"
            self.sockets[socket_key] = create_pull_socket(self.host, port)

        # Shared sockets for button and pause (only one instance needed)
        self.sockets[robots.BUTTON] = create_pull_socket(self.host, self.button_port)
        self.sockets[robots.PAUSE] = create_pull_socket(self.host, self.teleop_reset_port)

    def _process_keypoints(self, data):
        """Process raw keypoint data into a list of coordinate values."""
        data_str = data.decode().strip()
        values = []

        # Parse coordinates (format: <hand>:x,y,z|x,y,z|x,y,z)
        coords = data_str.split(":")[1].strip().split("|")
        for coord in coords:
            values.extend(float(val) for val in coord.split(",")[:3])

        return values

    def _rotate_90_around_x(self, keypoints: list) -> list:
        """Rotate keypoints 90 degrees around the X axis.

        Rotation matrix for 90° around X:
        [1  0  0]
        [0  0 -1]
        [0  1  0]

        Transforms: x' = x, y' = -z, z' = y
        """
        if len(keypoints) == 0:
            return keypoints

        keypoints_array = np.array(keypoints).reshape(-1, 3)
        rotated = np.zeros_like(keypoints_array)
        rotated[:, 0] = keypoints_array[:, 0]  # x stays the same
        rotated[:, 1] = -keypoints_array[:, 2]  # y' = -z
        rotated[:, 2] = keypoints_array[:, 1]  # z' = y

        return rotated.flatten().tolist()

    def _receive_data(self, socket_name):
        """Receive data from a socket."""
        try:
            data = self.sockets[socket_name].recv(zmq.NOBLOCK)
            self.last_received[socket_name] = time.time()
            return data
        except zmq.Again:
            return None

    def stream(self):
        """Main streaming loop for unified VR hand detection."""
        logger.info(f"Starting VR hand detection with configuration: {self.hand_config}")
        logger.info(f"Hand ports: {self.hand_ports}")
        logger.info(f"Sockets: {list(self.sockets.keys())}")

        data_received_count = {hand_side: 0 for hand_side in self.hand_ports}

        while True:
            self.timer.start_loop()

            # Process keypoint data for all configured hands
            for hand_side in self.hand_ports:
                socket_key = f"{robots.KEYPOINTS}_{hand_side}"
                keypoint_data = self._receive_data(socket_key)

                if keypoint_data is not None:
                    data_received_count[hand_side] += 1
                    if data_received_count[hand_side] % 100 == 0:
                        logger.info(f"Received {data_received_count[hand_side]} frames for {hand_side}")

                    # Process and publish keypoints for this hand
                    keypoints = self._process_keypoints(keypoint_data)
                    is_relative = not keypoint_data.decode().strip().startswith(robots.ABSOLUTE)

                    # Rotate keypoints 90 degrees around X axis for OpenArm
                    rotated_keypoints = self._rotate_90_around_x(keypoints)

                    # Log raw VR data for debugging (both original and rotated)
                    self._log_raw_vr_data(hand_side, keypoint_data, keypoints, rotated_keypoints)

                    # TODO: We really only need to publish ONCE!
                    # We can store all information in a single schema table

                    logging.getLogger("movePerf").log(logging.DEBUG, f"Oculus Keypoints received ({len(keypoints)}) on topic {hand_side}")

                    self.publisher_manager.publish(
                        host=self.host,
                        port=self.oculus_pub_port,
                        topic=hand_side,
                        data=InputFrame(
                            timestamp_s=time.time(),
                            hand_side=hand_side,
                            keypoints=rotated_keypoints,
                            is_relative=is_relative,
                            frame_vectors=None,
                        ),
                    )

            # Process and publish button state (shared across hands)
            if button_data := self._receive_data(robots.BUTTON):
                # For button events, use the first configured hand side as the source
                # or 'right' as default for bimanual setups
                hand_side = (
                    robots.RIGHT if robots.RIGHT in self.hand_ports else list(self.hand_ports.keys())[0]
                )

                self.publisher_manager.publish(
                    host=self.host,
                    port=self.oculus_pub_port,
                    topic=robots.BUTTON,
                    data=ButtonEvent(
                        timestamp_s=time.time(),
                        hand_side=hand_side,
                        name=robots.BUTTON,
                        value=robots.ARM_LOW_RESOLUTION
                        if button_data == b"Low"
                        else robots.ARM_HIGH_RESOLUTION,
                    ),
                )

            # Process and publish pause state (shared across hands)
            if pause_data := self._receive_data(robots.PAUSE):
                self.publisher_manager.publish(
                    host=self.host,
                    port=self.oculus_pub_port,
                    topic=robots.PAUSE,
                    data=SessionCommand(
                        timestamp_s=time.time(),
                        command="resume" if pause_data == b"Low" else "pause",
                    ),
                )

            self.timer.end_loop()

        # TODO: We need better cleanup than this
        # Cleanup sockets on exit
        for hand_side in self.hand_ports:
            if len(self.raw_keypoint_records[hand_side]) > 0:
                self._save_raw_vr_data(hand_side)
        for name, socket in self.sockets.items():
            socket.close()
            logger.info(f"Closed {name} socket")
        logger.info("Stopped VR hand detection process.")
