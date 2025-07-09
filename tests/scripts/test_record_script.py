import numpy as np
import time
import pickle
import threading
import zmq
from unittest.mock import patch, MagicMock

from beavr.scripts.control_robot import record
from beavr.common.robot_devices.robots.configs import BeavrRobotAdapterConfig
from beavr.common.robot_devices.cameras.configs import OpenCVCameraConfig
from beavr.common.robot_devices.control_configs import RecordControlConfig
from beavr.common.robot_devices.robots.utils import make_robot_from_config
import logging


# Setup logging for tests
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

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.thread.start()
        time.sleep(0.5)  # Allow time for connection

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.socket.close()
        self.context.term()

    def _publish_loop(self):
        raise NotImplementedError("Subclasses must implement _publish_loop")

class MockXArmStatePublisher(MockPublisher):
    """Simulates a robot publisher sending joint states and cartesian commands."""
    def _publish_loop(self):
        while self.running:
            try:
                # Simulate joint states in degrees (7 joints for XArm)
                joint_pos_deg = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float32)
                # Simulate cartesian command (x,y,z,rx,ry,rz) in mm and rad
                cartesian_cmd = np.array([200, 250, 300, 0.1, 0.2, 0.3], dtype=np.float32)

                state_data = {
                    "joint_states": {
                        "joint_position": joint_pos_deg,
                        "timestamp": time.time()
                    },
                    "commanded_cartesian_state": {
                        "commanded_cartesian_position": cartesian_cmd,
                        "timestamp": time.time()
                    },
                    "timestamp": time.time()
                }

                message = bytes(f"{self.topic} ", 'utf-8') + pickle.dumps(state_data, protocol=-1)
                self.socket.send(message)
                time.sleep(1 / 30)  # ~30Hz
            except Exception as e:
                logger.error(f"Error in mock XArm state publisher: {e}")


@patch('beavr.common.robot_devices.robots.beavr_robot_adapter.DexArmControl')
def test_record_with_beavr_adapter(mock_dex_arm_control, tmp_path):
    """
    Tests the main `record` function from `control_robot.py` with the BeavrRobotAdapter.
    This is an integration test that verifies the pipeline from mock robot data to dataset generation,
    including cartesian to joint action conversion via a mocked IK solver.
    """
    # 1. Mock the IK solver (DexArmControl)
    mock_ik_solver = MagicMock()
    # Simulate successful IK: returns code 0 and a list of 7 joint angles in radians
    mock_ik_solver.get_inverse_kinematics.return_value = (0, np.random.rand(7).tolist())
    mock_dex_arm_control.return_value = mock_ik_solver

    # 2. Start the mock ZMQ publisher
    publisher = MockXArmStatePublisher(port=10011, topic="right_xarm7")
    publisher.start()

    # 3. Configure the BeavrRobotAdapter based on lerobot_xarm_adapter.yaml
    adapter_config = BeavrRobotAdapterConfig(
        robot_state_host="localhost",
        robot_state_port=10011,
        robot_state_topic="right_xarm7",
        actual_state_input_key="joint_states",
        actual_state_payload_key="joint_position",
        command_input_key="commanded_cartesian_state",
        command_payload_key="commanded_cartesian_position",
        use_cartesian_action=True,
        input_angles_are_degrees=True,
        xarm_ip="192.168.1.208",  # Dummy IP to trigger IK initialization
        cameras={
            "front_cam": OpenCVCameraConfig(camera_index=0, mock=True, fps=30, width=640, height=480)
        }
    )

    # 4. Configure the recording process for a short, non-interactive run
    record_config = RecordControlConfig(
        repo_id="test/test_dataset",
        root=str(tmp_path),
        num_episodes=1,
        episode_time_s=0.5,  # short episode
        warmup_time_s=0.2,  # short warmup
        reset_time_s=0,
        fps=30,
        display_data=False,
        play_sounds=False,
        push_to_hub=False,
    )

    # 5. Create the robot instance from the configuration
    robot = make_robot_from_config(adapter_config)

    # 6. Mock dataset creation and keyboard listener to avoid side effects
    with patch('beavr.scripts.control_robot.LeRobotDataset') as mock_lerobot_dataset_class, \
         patch('beavr.scripts.control_robot.init_keyboard_listener') as mock_init_keyboard:

        # Configure the mock dataset instance
        mock_dataset_instance = MagicMock()
        mock_lerobot_dataset_class.create.return_value = mock_dataset_instance
        type(mock_dataset_instance).num_episodes = 0 # So the recording loop starts

        # Configure the mock keyboard listener to not interfere
        mock_init_keyboard.return_value = (MagicMock(), {"stop_recording": False, "rerecord_episode": False, "exit_early": False})

        # 7. Run the `record` function under test
        try:
            record(robot, record_config)
        finally:
            # 8. Cleanup resources
            publisher.stop()
            if robot.is_connected:
                robot.disconnect()

    # 9. Assert that all components were called as expected
    mock_dex_arm_control.assert_called_with(ip="192.168.1.208", is_radian=True)
    assert mock_ik_solver.get_inverse_kinematics.call_count > 0

    mock_lerobot_dataset_class.create.assert_called_once()
    assert mock_dataset_instance.add_frame.call_count > 0
    mock_dataset_instance.save_episode.assert_called_once()
