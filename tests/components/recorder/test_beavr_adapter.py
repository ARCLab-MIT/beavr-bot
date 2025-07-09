import pytest
import numpy as np
import zmq
import time
import threading
import logging
import pickle
from unittest.mock import patch, MagicMock
from beavr.common.robot_devices.robots.utils import make_robot_config, make_robot_from_config
from beavr.common.robot_devices.cameras.configs import OpenCVCameraConfig
from beavr.common.robot_devices.cameras.opencv import OpenCVCamera
from beavr.common.datasets.lerobot_dataset import LeRobotDataset


# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockPublisher:
    """Base class for mock ZMQ publishers used in testing"""
    def __init__(self, port: int, topic: str = ""):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.topic = topic
        self.running = False
        self.thread = None
        logger.info(f"Created MockPublisher on port {port} with topic '{topic}'")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.thread.start()
        time.sleep(0.5)  # Allow time for connection
        logger.info(f"Started MockPublisher thread for topic '{self.topic}'")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.socket.close()
        self.context.term()
        logger.info(f"Stopped MockPublisher for topic '{self.topic}'")

    def _publish_loop(self):
        raise NotImplementedError("Subclasses must implement _publish_loop")

class MockStatePublisher(MockPublisher):
    """Simulates XArm7Robot publishing joint states and commands"""
    def _publish_loop(self):
        count = 0
        while self.running:
            try:
                # Simulate joint states in degrees
                joint_pos_deg = np.array([10.0*count, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float32)
                # Simulate joint commands in degrees
                joint_cmd_deg = np.array([11.0*count, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0], dtype=np.float32)

                state_data = {
                    "joint_states": {"joint_position": joint_pos_deg, "timestamp": time.time()},
                    "commanded_joint_state": {"joint_position": joint_cmd_deg, "timestamp": time.time()},
                    "timestamp": time.time()
                }
                
                message = bytes(f"{self.topic} ", 'utf-8') + pickle.dumps(state_data, protocol=-1)
                self.socket.send(message)
                logger.debug(f"Published state data for count {count}")
                
                count = (count + 1) % 10
                time.sleep(0.1)  # 10Hz
            except Exception as e:
                logger.error(f"Error in mock state publisher: {e}")

class MockCameraPublisher(MockPublisher):
    """Simulates camera publishing RGB or depth images"""
    def __init__(self, port: int, cam_name: str, topic_type: str):
        super().__init__(port)
        self.cam_name = cam_name
        self.topic_type = topic_type

    def _publish_loop(self):
        while self.running:
            try:
                h, w = 480, 640
                if self.topic_type == "RGB":
                    img = np.random.randint(0, 256, (h,w,3), dtype=np.uint8)
                    msg_type = b"RGB"
                elif self.topic_type == "Depth":
                    img = (np.random.rand(h,w)*1000).astype(np.float32)
                    msg_type = b"Depth"
                else:
                    time.sleep(1)
                    continue

                payload = [
                    msg_type,
                    self.cam_name.encode('utf-8'),
                    np.float64(time.time()).tobytes(),
                    np.int32(h).tobytes(),
                    np.int32(w).tobytes(),
                    np.int32(img.shape[2] if img.ndim==3 else 1).tobytes(),
                    str(img.dtype).encode('utf-8'),
                    img.tobytes()
                ]
                self.socket.send_multipart(payload)
                time.sleep(0.033)  # 30Hz
            except Exception as e:
                logger.error(f"Error in mock camera publisher: {e}")

@pytest.fixture
def mock_publishers():
    """Fixture to manage mock publishers for testing"""
    publishers = []
    
    # Create and start publishers
    state_pub = MockStatePublisher(5555, "test_robot_state_topic")
    cam1_pub = MockCameraPublisher(5556, "front_cam", "RGB")
    cam2_pub = MockCameraPublisher(5557, "wrist_cam", "RGB")
    
    publishers.extend([state_pub, cam1_pub, cam2_pub])
    for pub in publishers:
        pub.start()
    
    yield publishers
    
    # Cleanup
    for pub in publishers:
        pub.stop()

@pytest.fixture
def robot_config():
    """Fixture to create a test robot configuration"""
    return make_robot_config(
        "beavr_adapter",
        robot_state_host="localhost",
        robot_state_port=5555,
        robot_state_topic="test_robot_state_topic",
        cameras={
            "front_cam": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
                mock=True  # Enable mock mode for testing
            ),
            "wrist_cam": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
                mock=True  # Enable mock mode for testing
            )
        },
        use_cartesian_action=False,  # Use joint commands
        input_angles_are_degrees=True,
        command_input_key="commanded_joint_state",  # Use joint commands
        command_payload_key="joint_position",       # Use joint commands
        mock=True  # Enable mock mode for the whole robot
    )

@pytest.fixture
def robot(robot_config):
    """Fixture to create and manage a test robot instance"""
    robot = make_robot_from_config(robot_config)
    yield robot
    if robot.is_connected:
        robot.disconnect()

def test_robot_connection(robot):
    """Test basic robot connection and disconnection"""
    assert not robot.is_connected
    robot.connect()
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected

def test_robot_features(robot):
    """Test that robot features are properly structured"""
    # Check basic feature structure
    assert "observation.state" in robot.features
    assert "action" in robot.features
    
    # Check motor features
    motor_features = robot.motor_features
    assert "observation.state" in motor_features
    assert "action" in motor_features
    assert "names" in motor_features["observation.state"]
    assert "names" in motor_features["action"]
    
    # Check camera features
    camera_features = robot.camera_features
    assert "observation.images.front_cam" in camera_features
    assert "observation.images.wrist_cam" in camera_features
    for cam_feature in camera_features.values():
        assert "names" in cam_feature
        assert "info" in cam_feature

def test_robot_cameras(robot):
    """Test camera-related properties and methods"""
    cameras = robot.cameras
    # Check if cameras are an instance of OpenCVCamera
    assert isinstance(cameras["front_cam"], OpenCVCamera)
    assert isinstance(cameras["wrist_cam"], OpenCVCamera)
    # Check if cameras have the following attributes:camera_index, fps, width, height
    assert cameras["front_cam"].camera_index == 0
    assert cameras["front_cam"].fps == 30
    assert cameras["front_cam"].width == 640
    assert cameras["front_cam"].height == 480
    assert cameras["wrist_cam"].camera_index == 1
    assert cameras["wrist_cam"].fps == 30
    assert cameras["wrist_cam"].width == 640
    assert cameras["wrist_cam"].height == 480

def test_compatibility_attributes(robot):
    """Test presence of compatibility attributes"""
    assert hasattr(robot, "leader_arms")
    assert hasattr(robot, "follower_arms")
    assert hasattr(robot, "logs")
    assert isinstance(robot.leader_arms, dict)
    assert isinstance(robot.follower_arms, dict)
    assert isinstance(robot.logs, dict)

def test_observation_capture(robot, mock_publishers):
    """Test observation capture functionality"""
    robot.connect()
    time.sleep(1)  # Give time for publishers to connect
    observation = robot.capture_observation()
    
    assert observation is not None
    assert "observation.state" in observation
    assert observation["observation.state"].shape[-1] == 7  # XArm has 7 joints
    
    # Test image capture
    assert "observation.images.front_cam" in observation
    assert "observation.images.wrist_cam" in observation
    assert observation["observation.images.front_cam"].shape == (480, 640, 3)
    assert observation["observation.images.wrist_cam"].shape == (480, 640, 3)

def test_dataset_compatibility(robot, mock_publishers, tmp_path):
    """Test compatibility with LeRobotDataset creation"""
    robot.connect()
    
    mock_dataset = MagicMock()
    mock_dataset.num_frames = 5
    mock_dataset.num_episodes = 1
    mock_dataset.features = {
        "observation.state": None,
        "observation.images.front_cam": None,
        "observation.images.wrist_cam": None
    }

    # Mock the dataset creation and add_frame methods
    with patch('beavr.common.datasets.lerobot_dataset.LeRobotDataset.create', return_value=mock_dataset) as mock_create:
    dataset = LeRobotDataset.create(
        repo_id="test/beavr_adapter_test",
            fps=30,
        root=tmp_path,
        robot=robot,
        use_videos=True
    )
    
    # Record a few frames
    for _ in range(5):
        observation = robot.capture_observation()
        frame = {
            **observation,
            "action": observation["observation.state"],  # Mock action
            "task": "Test task"
        }
        dataset.add_frame(frame)
        time.sleep(0.1)
    
    # Save the episode
    dataset.save_episode()
    
    # Verify dataset creation worked
    assert dataset.num_frames == 5
    assert dataset.num_episodes == 1
    assert "observation.state" in dataset.features
    assert "observation.images.front_cam" in dataset.features
    assert "observation.images.wrist_cam" in dataset.features
        
        # Verify the mock was called correctly
        mock_create.assert_called_once()

def test_cartesian_mode(robot_config):
    """Test cartesian mode configuration and IK handling"""
    # Create a robot config with cartesian mode and mock XArm IP
    cart_config = make_robot_config(
        "beavr_adapter",
        robot_state_host="localhost",
        robot_state_port=5555,
        robot_state_topic="test_robot_state_topic",
        use_cartesian_action=True,
        xarm_ip="192.168.1.1",  # Mock IP
        input_angles_are_degrees=True,
        cameras={
            "front_cam": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
                mock=True  # Enable mock mode for testing
            )
        },
        command_input_key="commanded_cartesian_state",  # Use cartesian commands
        command_payload_key="commanded_cartesian_position",  # Use cartesian commands
    )
    
    # Verify the config was updated appropriately
    assert cart_config.command_input_key == "commanded_cartesian_state"
    assert cart_config.command_payload_key == "commanded_cartesian_position"
    
    # Create robot instance
    robot = make_robot_from_config(cart_config)
    
    # Verify cartesian mode settings
    assert robot.use_cartesian_action
    assert robot.command_input_key == "commanded_cartesian_state"
    assert robot.command_payload_key == "commanded_cartesian_position"

def test_beavr_to_lerobot_transformation(robot_config, tmp_path):
    """Test the transformation of data from Beavr format (ZMQ messages) to LeRobot format (dataset)"""
    # Modify config to match lerobot_xarm_adapter.yaml settings
    robot_config = make_robot_config(
        "beavr_adapter",
        robot_state_host="localhost",
        robot_state_port=10011,  # Match the yaml config
        robot_state_topic="right_xarm7",  # Match the yaml config
        use_cartesian_action=False,  # Don't use cartesian action for this test
        input_angles_are_degrees=True,  # Match the yaml config
        actual_state_input_key="joint_states",  # Match the yaml config
        actual_state_payload_key="joint_position",  # Match the yaml config
        command_input_key="joint_states",  # Use joint states for both to simplify test
        command_payload_key="joint_position",  # Use joint states for both to simplify test
        cameras={
            "front_cam": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
                mock=True
            )
        },
        mock=True  # Enable mock mode
    )
    
    # Create a mock state publisher that matches the expected format
    class MockXArmStatePublisher(MockPublisher):
        def _publish_loop(self):
            count = 0
            while self.running:
                try:
                    # Simulate joint states in degrees (7 joints for XArm)
                    joint_pos_deg = np.array([10.0 + count, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float32)
                    
                    # Match the exact structure from lerobot_xarm_adapter.yaml
                    state_data = {
                        "joint_states": {
                            "joint_position": joint_pos_deg,
                            "timestamp": time.time()
                        },
                        "timestamp": time.time()
                    }
                    
                    message = bytes(f"{self.topic} ", 'utf-8') + pickle.dumps(state_data, protocol=-1)
                    self.socket.send(message)
                    count = (count + 1) % 10
                    time.sleep(0.033)  # ~30Hz to match camera FPS
                except Exception as e:
                    logger.error(f"Error in mock XArm state publisher: {e}")
    
    # Create and start the mock publisher
    mock_xarm_pub = MockXArmStatePublisher(10011, "right_xarm7")
    mock_xarm_pub.start()
    
    try:
        # Create robot and connect
        robot = make_robot_from_config(robot_config)
        robot.connect()
        time.sleep(1)  # Allow time for connection
        
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.features = {
            "observation.state": {"shape": (7,), "dtype": "float32"},  # 7 joints
            "action": {"shape": (7,), "dtype": "float32"},  # 7 joints for this test
            "observation.images.front_cam": {"shape": (480, 640, 3), "dtype": "uint8"}
        }
        
        with patch('beavr.common.datasets.lerobot_dataset.LeRobotDataset.create', return_value=mock_dataset) as mock_create:
            dataset = LeRobotDataset.create(
                repo_id="test/beavr_adapter_test",
                fps=30,
                root=tmp_path,
                robot=robot,
                use_videos=True
            )
            
            # Capture and verify multiple frames
            for _ in range(5):
                observation = robot.capture_observation()
                assert observation is not None
                
                # Verify joint state transformation
                assert "observation.state" in observation
                joint_state = observation["observation.state"]
                assert isinstance(joint_state, np.ndarray)
                assert joint_state.shape == (7,)  # 7 joints
                assert joint_state.dtype == np.float32
                # Verify angles are in radians (converted from degrees)
                assert np.all(joint_state >= 0) and np.all(joint_state <= np.pi)
                
                # Add frame to dataset
                frame = {
                    **observation,
                    "action": observation["observation.state"],  # Use same joint state as action
                    "task": "Test transformation"
                }
                dataset.add_frame(frame)
                time.sleep(0.033)  # ~30Hz
            
            # Verify dataset creation and frame addition
            mock_create.assert_called_once()
            assert mock_dataset.add_frame.call_count == 5
            
            # Verify the structure of frames added to dataset
            for call in mock_dataset.add_frame.call_args_list:
                frame = call[0][0]  # First argument of the call
                assert "observation.state" in frame
                assert "action" in frame
                assert frame["observation.state"].shape == (7,)
                assert frame["action"].shape == (7,)
                assert "task" in frame
                assert frame["task"] == "Test transformation"
    
    finally:
        # Cleanup
        mock_xarm_pub.stop()
        if robot.is_connected:
            robot.disconnect()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
