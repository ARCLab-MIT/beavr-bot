import logging
import time
from typing import Optional, Union

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

        while True:
            self.timer.start_loop()

            # Process keypoint data for all configured hands
            for hand_side in self.hand_ports:
                socket_key = f"{robots.KEYPOINTS}_{hand_side}"
                keypoint_data = self._receive_data(socket_key)

                if keypoint_data is not None:
                    # Process and publish keypoints for this hand
                    keypoints = self._process_keypoints(keypoint_data)
                    is_relative = not keypoint_data.decode().strip().startswith(robots.ABSOLUTE)

                    self.publisher_manager.publish(
                        host=self.host,
                        port=self.oculus_pub_port,
                        topic=hand_side,
                        data=InputFrame(
                            timestamp_s=time.time(),
                            hand_side=hand_side,
                            keypoints=keypoints,
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

        # Cleanup sockets on exit
        for name, socket in self.sockets.items():
            socket.close()
            logger.info(f"Closed {name} socket")
        logger.info("Stopped VR hand detection process.")
