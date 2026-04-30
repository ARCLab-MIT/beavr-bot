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


class VRDataRecorder:
    """Records VR data to NumPy format for later replay."""

    def __init__(self, output_path: str, auto_save_interval: int = 10):
        """Initialize recorder with output path."""
        self.output_path = output_path
        self.auto_save_interval = auto_save_interval
        self.frame_count = 0
        self.data = {
            "left_timestamps": [],
            "left_keypoints": [],
            "left_is_relative": [],
            "right_timestamps": [],
            "right_keypoints": [],
            "right_is_relative": [],
            "button_timestamps": [],
            "button_values": [],
            "pause_timestamps": [],
            "pause_commands": [],
        }
        self.recording_start_timestamp = None

    def record_keypoint(self, hand_side: str, input_frame: InputFrame):
        """Record a keypoint frame."""
        if hand_side == "left":
            self.data["left_timestamps"].append(input_frame.timestamp_s)
            self.data["left_keypoints"].append(input_frame.keypoints)
            self.data["left_is_relative"].append(input_frame.is_relative)
        elif hand_side == "right":
            self.data["right_timestamps"].append(input_frame.timestamp_s)
            self.data["right_keypoints"].append(input_frame.keypoints)
            self.data["right_is_relative"].append(input_frame.is_relative)

        if self.recording_start_timestamp is None:
            self.recording_start_timestamp = input_frame.timestamp_s

    def record_button_event(self, button_event: ButtonEvent):
        """Record a button event."""
        self.data["button_timestamps"].append(button_event.timestamp_s)
        self.data["button_values"].append(1 if button_event.value == robots.ARM_HIGH_RESOLUTION else 0)

    def record_session_command(self, session_command: SessionCommand):
        """Record a session command."""
        self.data["pause_timestamps"].append(session_command.timestamp_s)
        self.data["pause_commands"].append(1 if session_command.command == "pause" else 0)

    def save(self):
        """Save recorded data to .npz file."""
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {}

        # Convert lists to arrays for each hand that has data
        if len(self.data["left_timestamps"]) > 0:
            save_data["left_timestamps"] = np.array(self.data["left_timestamps"], dtype=np.float64)
            keypoints_array = np.array(self.data["left_keypoints"], dtype=np.float64)
            if keypoints_array.ndim == 1:
                keypoints_array = keypoints_array.reshape(1, -1)
            save_data["left_keypoints"] = keypoints_array
            save_data["left_is_relative"] = np.array(self.data["left_is_relative"], dtype=bool)

        if len(self.data["right_timestamps"]) > 0:
            save_data["right_timestamps"] = np.array(self.data["right_timestamps"], dtype=np.float64)
            keypoints_array = np.array(self.data["right_keypoints"], dtype=np.float64)
            if keypoints_array.ndim == 1:
                keypoints_array = keypoints_array.reshape(1, -1)
            save_data["right_keypoints"] = keypoints_array
            save_data["right_is_relative"] = np.array(self.data["right_is_relative"], dtype=bool)

        if len(self.data["button_timestamps"]) > 0:
            save_data["button_timestamps"] = np.array(self.data["button_timestamps"], dtype=np.float64)
            save_data["button_values"] = np.array(self.data["button_values"], dtype=np.int32)

        if len(self.data["pause_timestamps"]) > 0:
            save_data["pause_timestamps"] = np.array(self.data["pause_timestamps"], dtype=np.float64)
            save_data["pause_commands"] = np.array(self.data["pause_commands"], dtype=np.int32)

        # Add metadata
        recording_start = 0.0 if self.recording_start_timestamp is None else self.recording_start_timestamp
        metadata = {
            "version": "1.0",
            "recording_start": float(recording_start),
            "recording_end": float(time.time()),
            "hand_sides": list(self.hand_side_set()),
            "num_left_frames": len(self.data["left_timestamps"]),
            "num_right_frames": len(self.data["right_timestamps"]),
            "num_button_events": len(self.data["button_timestamps"]),
            "num_pause_events": len(self.data["pause_timestamps"]),
        }
        save_data["metadata"] = metadata

        np.savez_compressed(self.output_path, **save_data)
        logger.info(f"💾 Saved recording to {self.output_path}")

    def hand_side_set(self):
        """Return set of hand sides that were recorded."""
        hand_sides = set()
        if len(self.data["left_timestamps"]) > 0:
            hand_sides.add("left")
        if len(self.data["right_timestamps"]) > 0:
            hand_sides.add("right")
        return hand_sides

    def increment_frame_and_save_if_needed(self):
        """Increment frame count and auto-save at regular intervals."""
        self.frame_count += 1
        if self.frame_count % self.auto_save_interval == 0:
            self.save()
            logger.debug(f"💾 Auto-saved recording at frame {self.frame_count}")


class VRDataReplayer:
    """Replays recorded VR data from NumPy format."""

    def __init__(self, mock_data_path: str, host: str, oculus_pub_port: int):
        """Initialize replayer with mock data path."""
        self.data = np.load(mock_data_path, allow_pickle=True)
        self.host = host
        self.oculus_pub_port = oculus_pub_port
        self.publisher_manager = ZMQPublisherManager.get_instance()

        # Load metadata
        self.metadata = dict(self.data["metadata"].item())

        # Initialize indices for replay
        self.left_index = 0
        self.right_index = 0
        self.button_index = 0
        self.pause_index = 0

        # Determine which hands are being replayed
        self.hand_sides = set(self.metadata.get("hand_sides", []))
        if "right" in self.metadata and len(self.data.get("right_timestamps", [])) > 0:
            self.hand_sides.add("right")
        if "left" in self.metadata and len(self.data.get("left_timestamps", [])) > 0:
            self.hand_sides.add("left")

        logger.info(f"📼 Loaded recording with metadata: {self.metadata}")
        logger.info(f"🖐️  Replaying hands: {self.hand_sides}")

    def get_next_keypoint_frame(self, hand_side: str):
        """Get next keypoint frame for specified hand side, or cycle last frame if exhausted."""
        prefix = f"{hand_side}_"
        timestamps_key = f"{prefix}timestamps"
        keypoints_key = f"{prefix}keypoints"
        is_relative_key = f"{prefix}is_relative"

        if timestamps_key not in self.data:
            return None

        index_attr = f"{hand_side}_index"
        current_index = getattr(self, index_attr)
        max_index = len(self.data[timestamps_key]) - 1

        if current_index >= max_index:
            # Stay at last index for cycling
            pass
        else:
            current_index += 1
            setattr(self, index_attr, current_index)

        timestamp = self.data[timestamps_key][current_index]
        keypoints = self.data[keypoints_key][current_index].tolist()
        is_relative = self.data[is_relative_key][current_index]

        return InputFrame(
            timestamp_s=timestamp,
            hand_side=hand_side,
            keypoints=keypoints,
            is_relative=is_relative,
            frame_vectors=None,
        )

    def publish_next_frame(self):
        """Publish next frame for all configured hands."""
        for hand_side in self.hand_sides:
            frame = self.get_next_keypoint_frame(hand_side)
            if frame:
                self.publisher_manager.publish(
                    host=self.host,
                    port=self.oculus_pub_port,
                    topic=hand_side,
                    data=frame,
                )


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
        use_mock_mode: bool = False,
        mock_data_path: Optional[str] = None,
        record_mode: bool = False,
        record_output_path: Optional[str] = None,
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
            use_mock_mode: If True, publish pre-recorded data instead of receiving from hardware
            mock_data_path: Path to .npz file containing recorded VR data for mock mode
            record_mode: If True, record all published data to file
            record_output_path: Path where recording should be saved
        """

        self.notify_component_start(robots.VR_DETECTOR)

        self.host = host
        self.oculus_pub_port = oculus_pub_port
        self.button_port = button_port
        self.teleop_reset_port = teleop_reset_port
        self.hand_config = hand_config
        self.use_mock_mode = use_mock_mode
        self.mock_data_path = mock_data_path
        self.record_mode = record_mode
        self.record_output_path = record_output_path

        # Initialize mock or record mode based on configuration
        self.replayer = None
        self.recorder = None

        if use_mock_mode:
            if not mock_data_path:
                raise ValueError("mock_data_path must be provided when use_mock_mode is True")
            self.replayer = VRDataReplayer(mock_data_path, host, oculus_pub_port)
            logger.info(f"📼 Running in MOCK mode - replaying from {mock_data_path}")

        if record_mode:
            if not record_output_path:
                raise ValueError("record_output_path must be provided when record_mode is True")
            self.recorder = VRDataRecorder(record_output_path)
            logger.info(f"🔴 Running in RECORD mode - saving to {record_output_path}")

        # Skip socket initialization in mock mode
        if not use_mock_mode:
            # Validate and set hand ports based on configuration
            self._configure_hand_ports(right_hand_port, left_hand_port)

            # Initialize sockets based on configuration
            self._initialize_sockets()

            # Initialize publisher and timing
            self.publisher_manager = ZMQPublisherManager.get_instance()
            self.timer = FrequencyTimer(robots.VR_FREQ)
            self.last_received = {k: 0.0 for k in self.sockets} if self.sockets else {}

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

        # Mock mode: replay recorded data
        if self.use_mock_mode:
            logger.info(f"📼 Starting MOCK mode - replaying from {self.mock_data_path}")
            logger.info(f"🖐️  Replaying hands: {self.replayer.hand_sides}")

            try:
                while True:
                    self.replayer.publish_next_frame()
                    time.sleep(1.0 / robots.VR_FREQ)
            except Exception as e:
                logger.error(f"Error in mock mode: {e}")
            finally:
                logger.info("Mock mode stopped")
            return

        # Record mode or normal mode
        logger.info(f"Starting VR hand detection with configuration: {self.hand_config}")
        logger.info(f"Hand ports: {self.hand_ports}")
        logger.info(f"Sockets: {list(self.sockets.keys())}")
        if self.record_mode:
            logger.info(f"🔴 RECORD mode enabled - saving to {self.record_output_path}")

        data_received_count = {hand_side: 0 for hand_side in self.hand_ports}

        try:
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

                        # Create input frame
                        input_frame = InputFrame(
                            timestamp_s=time.time(),
                            hand_side=hand_side,
                            keypoints=rotated_keypoints,
                            is_relative=is_relative,
                            frame_vectors=None,
                        )

                        # Record if in record mode
                        if self.record_mode and self.recorder:
                            self.recorder.record_keypoint(hand_side, input_frame)
                            self.recorder.increment_frame_and_save_if_needed()

                        self.publisher_manager.publish(
                            host=self.host,
                            port=self.oculus_pub_port,
                            topic=hand_side,
                            data=input_frame,
                        )

                # Process and publish button state (shared across hands)
                if button_data := self._receive_data(robots.BUTTON):
                    # For button events, use the first configured hand side as the source
                    # or 'right' as default for bimanual setups
                    hand_side = robots.RIGHT if robots.RIGHT in self.hand_ports else list(self.hand_ports.keys())[0]

                    button_event = ButtonEvent(
                        timestamp_s=time.time(),
                        hand_side=hand_side,
                        name=robots.BUTTON,
                        value=robots.ARM_LOW_RESOLUTION if button_data == b"Low" else robots.ARM_HIGH_RESOLUTION,
                    )

                    # Record if in record mode
                    if self.record_mode and self.recorder:
                        self.recorder.record_button_event(button_event)

                    self.publisher_manager.publish(
                        host=self.host,
                        port=self.oculus_pub_port,
                        topic=robots.BUTTON,
                        data=button_event,
                    )

                # Process and publish pause state (shared across hands)
                if pause_data := self._receive_data(robots.PAUSE):
                    session_command = SessionCommand(
                        timestamp_s=time.time(),
                        command="resume" if pause_data == b"Low" else "pause",
                    )

                    # Record if in record mode
                    if self.record_mode and self.recorder:
                        self.recorder.record_session_command(session_command)

                    self.publisher_manager.publish(
                        host=self.host,
                        port=self.oculus_pub_port,
                        topic=robots.PAUSE,
                        data=session_command,
                    )

                self.timer.end_loop()

        finally:
            # Cleanup
            if self.record_mode and self.recorder:
                self.recorder.save()

            # TODO: We need better cleanup than this
            # Cleanup sockets on exit
            for hand_side in self.hand_ports:
                if len(self.raw_keypoint_records.get(hand_side, [])) > 0:
                    self._save_raw_vr_data(hand_side)
            for name, socket in self.sockets.items():
                socket.close()
                logger.info(f"Closed {name} socket")
            logger.info("Stopped VR hand detection process.")
