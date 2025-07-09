import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from beavr.components.operators.rx1_right_operator import RX1RightOperator
from beavr.constants import ARM_TELEOP_STOP


class TestRX1RightOperator(unittest.TestCase):
    
    def setUp(self):
        # Mock all network-related dependencies
        self.mock_keypoint_subscriber_patcher = patch('beavr.components.operators.rx1_right_operator.ZMQKeypointSubscriber')
        self.mock_keypoint_publisher_patcher = patch('beavr.components.operators.rx1_right_operator.ZMQKeypointPublisher')
        self.mock_frequency_timer_patcher = patch('beavr.components.operators.rx1_right_operator.FrequencyTimer')
        
        self.mock_keypoint_subscriber = self.mock_keypoint_subscriber_patcher.start()
        self.mock_keypoint_publisher = self.mock_keypoint_publisher_patcher.start()
        self.mock_frequency_timer = self.mock_frequency_timer_patcher.start()
        
        # Create mock instances
        self.mock_subscriber_instance = MagicMock()
        self.mock_publisher_instance = MagicMock()
        self.mock_timer_instance = MagicMock()
        
        self.mock_keypoint_subscriber.return_value = self.mock_subscriber_instance
        self.mock_keypoint_publisher.return_value = self.mock_publisher_instance
        self.mock_frequency_timer.return_value = self.mock_timer_instance
        
        # Initialize test data
        self.test_host = "localhost"
        self.test_port = 5555
        self.test_configs = {"some_config": "value"}
        self.test_stream_oculus = True
        
        # Create the operator instance
        self.operator = RX1RightOperator(
            host=self.test_host,
            transformed_keypoints_port=self.test_port,
            stream_configs=self.test_configs,
            stream_oculus=self.test_stream_oculus,
            endeff_publish_port=5556,
            endeff_subscribe_port=5557,
            moving_average_limit=5,
            use_filter=False,
            arm_resolution_port=5558,
            teleoperation_reset_port=5559,
            reset_publish_port=5560,
            logging_config={"enabled": False}
        )
    
    def tearDown(self):
        self.mock_keypoint_subscriber_patcher.stop()
        self.mock_keypoint_publisher_patcher.stop()
        self.mock_frequency_timer_patcher.stop()
    
    def test_initialization(self):
        """Test that the operator initializes correctly."""
        # Check that subscribers and publishers were created with correct parameters
        self.mock_keypoint_subscriber.assert_any_call(
            host=self.test_host,
            port=self.test_port,
            topic='transformed_hand_coords'
        )
        
        self.mock_keypoint_subscriber.assert_any_call(
            host=self.test_host,
            port=self.test_port,
            topic='transformed_hand_frame'
        )
        
        self.mock_keypoint_publisher.assert_any_call(
            host=self.test_host,
            port=5556
        )
        
        # Check initial state
        self.assertEqual(self.operator.arm_teleop_state, ARM_TELEOP_STOP)
        self.assertEqual(self.operator.resolution_scale, 1.0)
        self.assertTrue(self.operator.is_first_frame)
    
    def test_get_hand_frame(self):
        """Test the _get_hand_frame method."""
        # Mock the subscriber to return test data
        test_frame = np.array([
            [0.1, 0.2, 0.3],  # Translation
            [1.0, 0.0, 0.0],  # X axis
            [0.0, 1.0, 0.0],  # Y axis
            [0.0, 0.0, 1.0]   # Z axis
        ]).flatten()
        
        self.mock_subscriber_instance.recv_keypoints.return_value = test_frame
        
        # Call the method
        result = self.operator._get_hand_frame()
        
        # Check the result
        expected = np.array([
            [0.1, 0.2, 0.3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_turn_frame_to_homo_mat(self):
        """Test the _turn_frame_to_homo_mat method."""
        # Test data
        test_frame = np.array([
            [0.1, 0.2, 0.3],  # Translation
            [1.0, 0.0, 0.0],  # X axis
            [0.0, 1.0, 0.0],  # Y axis
            [0.0, 0.0, 1.0]   # Z axis
        ])
        
        # Call the method
        result = self.operator._turn_frame_to_homo_mat(test_frame)
        
        # Expected result
        expected = np.array([
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Check the result
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_homo2cart_and_cart2homo(self):
        """Test the _homo2cart and cart2homo methods for round-trip conversion."""
        # Test homogeneous matrix
        test_homo = np.array([
            [0.0, 0.0, 1.0, 0.1],
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Convert to cartesian
        cart = self.operator._homo2cart(test_homo)
        
        # Convert back to homogeneous
        homo_result = self.operator.cart2homo(cart)
        
        # Check round-trip conversion
        np.testing.assert_array_almost_equal(test_homo, homo_result, decimal=5)
    
    def test_reset_teleop(self):
        """Test the _reset_teleop method."""
        # Mock the subscriber to return test data
        robot_frame = np.eye(4).flatten()
        hand_frame = np.array([
            [0.1, 0.2, 0.3],  # Translation
            [1.0, 0.0, 0.0],  # X axis
            [0.0, 1.0, 0.0],  # Y axis
            [0.0, 0.0, 1.0]   # Z axis
        ])
        
        self.mock_subscriber_instance.recv_keypoints.side_effect = [
            robot_frame,  # First call for robot frame
            None,         # Second call for hand frame (first attempt)
            hand_frame.flatten()  # Third call for hand frame (second attempt)
        ]
        
        # Patch the _get_hand_frame method to return test data
        with patch.object(self.operator, '_get_hand_frame', side_effect=[None, hand_frame]):
            # Call the method
            result = self.operator._reset_teleop()
        
        # Check the result
        np.testing.assert_array_almost_equal(result, hand_frame)
        
        # Check that the publisher was called
        self.mock_publisher_instance.pub_keypoints.assert_called_with(1, 'reset')
        
        # Check that the internal state was updated
        self.assertFalse(self.operator.is_first_frame)
        np.testing.assert_array_almost_equal(self.operator.robot_init_H.flatten(), robot_frame)
        np.testing.assert_array_almost_equal(self.operator.hand_init_H, 
                                            self.operator._turn_frame_to_homo_mat(hand_frame))
    
    def test_moving_average(self):
        """Test the moving_average method."""
        # Test data
        action1 = np.array([1.0, 2.0, 3.0])
        action2 = np.array([2.0, 3.0, 4.0])
        action3 = np.array([3.0, 4.0, 5.0])
        
        queue = []
        limit = 2
        
        # First action
        result1 = self.operator.moving_average(action1, queue, limit)
        np.testing.assert_array_almost_equal(result1, action1)
        
        # Second action
        result2 = self.operator.moving_average(action2, queue, limit)
        np.testing.assert_array_almost_equal(result2, np.mean([action1, action2], axis=0))
        
        # Third action (should pop the first)
        result3 = self.operator.moving_average(action3, queue, limit)
        np.testing.assert_array_almost_equal(result3, np.mean([action2, action3], axis=0))
        
        # Check queue length
        self.assertEqual(len(queue), limit)

if __name__ == '__main__':
    unittest.main()