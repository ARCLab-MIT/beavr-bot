import logging
import threading
import time
from typing import Optional

import rclpy
from control_msgs.action import GripperCommand as GripperCommandAction
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.components.interface.interface_base import RobotWrapper
from beavr.teleop.components.operator.operator_types import GripperCommand
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class OpenArmGripperController:
    """ROS2 gripper controller that receives commands via ZMQ and sends to ROS2 gripper action server."""

    def __init__(
        self,
        gripper_action_name: str = "/left_gripper_controller/gripper_cmd",
        max_width: float = None,
        min_width: float = None,
        default_speed: float = None,
    ):
        self.max_width = max_width or robots.OPENARM_GRIPPER_MAX_WIDTH_M
        self.min_width = min_width or 0.0
        self.default_speed = default_speed or robots.OPENARM_GRIPPER_DEFAULT_SPEED_MPS

        self._initialize_ros2()

        self._action_client = ActionClient(
            self._node,
            GripperCommandAction,
            gripper_action_name,
            callback_group=ReentrantCallbackGroup(),
        )

        logger.info(f"Waiting for gripper action server: {gripper_action_name}")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            logger.error("Gripper action server not available!")
        else:
            logger.info("Connected to gripper action server")

        self._latest_command: Optional[GripperCommand] = None
        self._latest_command_lock = threading.Lock()

    def _initialize_ros2(self):
        logger.info("Starting ROS2 initialization for gripper controller...")
        if not rclpy.ok():
            logger.info("rclpy not initialized, calling rclpy.init()")
            try:
                rclpy.init()
                logger.info("rclpy.init() successful")
            except Exception as e:
                logger.error(f"Failed to initialize rclpy: {e}")
                raise

        logger.info("Creating ROS2 node: openarm_gripper_controller_node")
        try:
            self._node = Node("openarm_gripper_controller_node")
        except Exception as e:
            logger.error(f"Failed to create ROS2 node: {e}")
            raise

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)

        logger.info("Starting ROS2 executor thread for gripper controller")
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        logger.info("ROS2 node initialized: openarm_gripper_controller_node")

    def send_gripper_command(self, command: GripperCommand) -> bool:
        """Send gripper command to ROS2 gripper action server."""
        if not self._action_client.server_is_ready():
            logger.debug("Gripper action server is not ready")
            return False

        # Clamp width to gripper limits
        clamped_width = max(self.min_width, min(command.width_m, self.max_width))

        goal_msg = GripperCommandAction.Goal()
        goal_msg.command.position = clamped_width
        goal_msg.command.max_effort = 50.0  # Safe default grasping force

        duration = command.speed_mps or self.default_speed
        goal_msg.command.max_effort = 50.0 / duration if duration > 0 else 50.0

        logger.info(f"Sending gripper command: width={clamped_width:.3f}m")

        send_goal_future = self._action_client.send_goal_async(goal_msg)

        timeout_sec = duration + 1.0
        start_time = time.time()
        while not send_goal_future.done():
            if time.time() - start_time > timeout_sec:
                logger.error("Gripper goal send timed out")
                return False
            time.sleep(0.01)

        try:
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                logger.error("Gripper goal rejected")
                return False
            logger.debug("Gripper goal accepted")
            return True
        except Exception as e:
            logger.error(f"Exception during gripper command: {e}")
            return False

    def get_latest_command(self) -> Optional[GripperCommand]:
        with self._latest_command_lock:
            return self._latest_command

    def set_latest_command(self, command: Optional[GripperCommand]):
        with self._latest_command_lock:
            self._latest_command = command

    def cleanup(self):
        logger.info("Cleaning up gripper controller...")
        if hasattr(self, "_action_client"):
            self._action_client.destroy()
        if hasattr(self, "_executor"):
            self._executor.shutdown()
        if hasattr(self, "_node"):
            self._node.destroy_node()
        cleanup_zmq_resources()


class OpenArmGripperRobot(RobotWrapper):
    """OpenArm gripper control wrapper for integration with existing teleop framework."""

    def __init__(
        self,
        host: str,
        gripper_subscribe_port: int,
        **kwargs,
    ):
        logger.info(
            f"Initializing OpenArmGripperRobot with host={host}, gripper_subscribe_port={gripper_subscribe_port}"
        )

        self._controller = OpenArmGripperController()

        self._gripper_command_subscriber = ZMQSubscriber(
            host=host,
            port=gripper_subscribe_port,
            topic="gripper_cmd",
            message_type=GripperCommand,
        )

        self._subscribers = {
            "gripper_command": self._gripper_command_subscriber,
        }

        self._latest_command = None

    @property
    def name(self):
        return robots.ROBOT_IDENTIFIER_OPENARM_GRIPPER

    @property
    def recorder_functions(self):
        return {
            "gripper_command": self.get_gripper_command,
        }

    @property
    def data_frequency(self):
        return robots.VR_FREQ

    def get_gripper_command(self):
        cmd = self._gripper_command_subscriber.recv_keypoints()
        if cmd is not None:
            self._latest_command = cmd
        if self._latest_command is None:
            return None
        return {
            "width_m": self._latest_command.width_m,
            "hand_side": self._latest_command.hand_side,
            "timestamp": self._latest_command.timestamp_s,
        }

    # RobotWrapper interface methods (placeholders for arm-specific methods)
    def get_joint_state(self):
        return None

    def get_joint_position(self):
        return None

    def get_cartesian_position(self):
        return None

    def get_joint_velocity(self):
        return None

    def get_joint_torque(self):
        return None

    def home(self):
        logger.info("[openarm_gripper] Home (no-op for gripper)")

    def move(self, input_angles):
        logger.warning("[openarm_gripper] move() not applicable for gripper")

    def move_coords(self, input_coords):
        logger.warning("[openarm_gripper] move_coords() not applicable for gripper")

    def stream(self):
        """Main loop: receive gripper commands and execute them."""
        logger.info("[openarm_gripper] Starting gripper control loop")
        while True:
            cmd = self._gripper_command_subscriber.recv_keypoints()
            if cmd is not None:
                logger.debug(
                    f"[openarm_gripper] Received command: width={cmd.width_m:.3f}m, "
                    f"hand_side={cmd.hand_side}, speed={cmd.speed_mps:.2f}m/s"
                )
                self._controller.send_gripper_command(cmd)
            time.sleep(0.001)

    def __del__(self):
        self._controller.cleanup()
        cleanup_zmq_resources()
