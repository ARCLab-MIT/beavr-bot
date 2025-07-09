import unittest
from unittest.mock import patch, MagicMock
from beavr.components.detector.oculus import OculusVRHandDetector
from beavr.constants import ARM_LOW_RESOLUTION, ARM_TELEOP_CONT


class TestOculusVRHandDetector(unittest.TestCase):

    @patch('beavr.components.detector.oculus.create_pull_socket')
    @patch('beavr.components.detector.oculus.ZMQKeypointPublisher')
    def test_stream(self, mock_publisher, mock_create_socket):
        # Mock the sockets and publishers
        mock_socket = MagicMock()
        mock_create_socket.return_value = mock_socket
        mock_publisher_instance = MagicMock()
        mock_publisher.return_value = mock_publisher_instance

        # Mock the data returned by the sockets
        mock_socket.recv.side_effect = [
            b'absolute:0.1,0.2,0.3|0.4,0.5,0.6|0.7,0.8,0.9',  # raw_keypoints
            b'Low',  # button_feedback
            b'High'  # pause_status
        ]

        # Initialize the OculusVRHandDetector
        detector = OculusVRHandDetector(
            host="localhost",
            oculus_port=9000,
            keypoint_pub_port=9001,
            button_port=9002,
            button_publish_port=9003,
            teleop_reset_port=9004,
            teleop_reset_publish_port=9005
        )

        # Run the stream method once
        with patch.object(detector.timer, 'start_loop', return_value=None), \
             patch.object(detector.timer, 'end_loop', return_value=None):
            detector.stream()

        # Check that the publishers were called with the expected data
        mock_publisher_instance.pub_keypoints.assert_any_call(
            keypoint_array=[0] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            topic_name='right'
        )
        mock_publisher_instance.pub_keypoints.assert_any_call(
            keypoint_array=ARM_LOW_RESOLUTION,
            topic_name='button'
        )
        mock_publisher_instance.pub_keypoints.assert_any_call(
            keypoint_array=ARM_TELEOP_CONT,
            topic_name='pause'
        )

        # Ensure sockets are closed
        mock_socket.close.assert_called()

if __name__ == '__main__':
    unittest.main()