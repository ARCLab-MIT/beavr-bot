from unittest.mock import patch

from beavr.components.detector.oculus import OculusVRHandDetector
from beavr.constants import ARM_LOW_RESOLUTION, ARM_TELEOP_CONT


class DummySocket:
    """A minimal dummy socket that returns predefined messages and then raises StopIteration."""

    def __init__(self, messages):
        # ensure we always have at least one message list
        self._messages = list(messages)
        self.closed = False

    def recv(self):
        if self._messages:
            return self._messages.pop(0)
        # Raise an error after messages are exhausted to break the detector loop
        raise StopIteration("No more messages")

    def close(self):
        self.closed = True


class DummyPublisher:
    """A dummy publisher that records published data for inspection in tests."""

    def __init__(self, *args, **kwargs):
        self.published = []

    def pub_keypoints(self, keypoint_array, topic_name):
        # Record the topic and data for later assertions
        self.published.append((topic_name, keypoint_array))

    def stop(self):
        pass


class DummyTimer:
    """A dummy frequency timer that does nothing to avoid busy waiting during tests."""

    def __init__(self, *args, **kwargs):
        pass

    def start_loop(self):
        pass

    def end_loop(self):
        pass


@patch("beavr.components.detector.oculus.FrequencyTimer", DummyTimer)
@patch("beavr.components.detector.oculus.ZMQKeypointPublisher", DummyPublisher)
@patch("beavr.components.detector.oculus.create_pull_socket")
def test_oculus_stream_single_iteration(mock_create_pull_socket, *_) :
    """Ensure OculusVRHandDetector receives from all sockets and publishes correct messages."""

    # Define the messages each socket should produce on the first iteration
    raw_keypoints_msg = b"left:1,2,3|4,5,6|7,8,9"
    button_msg = b"Low"  # Should map to ARM_LOW_RESOLUTION
    teleop_msg = b"Low"  # Should map to ARM_TELEOP_CONT

    # Map ports to their dummy sockets with the expected message list
    def pull_socket_side_effect(host, port):
        if port == 8110:  # raw keypoints port for left hand
            return DummySocket([raw_keypoints_msg])
        elif port == 9001:  # button feedback port
            return DummySocket([button_msg])
        elif port == 9002:  # teleop reset port
            return DummySocket([teleop_msg])
        else:
            # Any unexpected port gets an empty socket that immediately stops the loop
            return DummySocket([])

    mock_create_pull_socket.side_effect = pull_socket_side_effect

    # Instantiate the detector (will use patched sockets/publisher/timer)
    detector = OculusVRHandDetector(
        host="localhost",
        oculus_port=8110,
        unified_pub_port=9000,
        button_port=9001,
        teleop_reset_port=9002,
    )

    # The DummyPublisher instance is attached as detector.unified_publisher
    publisher = detector.unified_publisher

    # Run the stream method – it should exit after the first iteration because sockets raise StopIteration.
    detector.stream()

    # Verify three publishes occurred (keypoints, button, pause)
    assert len(publisher.published) == 3, "Expected three publish calls (keypoints, button, pause)"

    # Unpack the published topics and data
    topics = [t for t, _ in publisher.published]
    data = {t: d for t, d in publisher.published}

    # Expected topics
    assert "left" in topics  # Hand side inferred from oculus_port
    assert "button" in topics
    assert "pause" in topics

    # Validate button feedback mapping
    assert data["button"] == ARM_LOW_RESOLUTION

    # Validate pause status mapping
    assert data["pause"] == ARM_TELEOP_CONT

    # Validate keypoint data structure – should start with a 1 followed by 9 floats
    left_keypoints = data["left"]
    assert left_keypoints[0] == 1  # relative coordinates flag
    assert len(left_keypoints) == 10  # 1 flag + 3*3 coordinates

    # Ensure sockets were closed after streaming finished
    assert detector.raw_keypoint_socket.closed
