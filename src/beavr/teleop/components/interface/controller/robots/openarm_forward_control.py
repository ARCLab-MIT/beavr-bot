import logging
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class OpenArmForwardController:
    def __init__(
        self,
        joint_names: list,
        command_topic_name: str = "",
        max_delta: float = 0.2,
    ):
        self.joint_names = joint_names
        self.num_joints = len(self.joint_names)
        self.command_topic_name = command_topic_name
        self.max_delta = max_delta

        self._joint_states: Optional[JointState] = None
        self._joint_states_lock = threading.Lock()
        self._pedal_pressed = False
        self._pedal_pressed_lock = threading.Lock()
        self._current_joint_positions: Optional[np.ndarray] = None
        self._current_joint_velocities: Optional[np.ndarray] = None
        self._current_joint_efforts: Optional[np.ndarray] = None

        self._initialize_ros2()

        self._joint_command_publisher: Optional[Publisher] = self._node.create_publisher(
            Float64MultiArray, self.command_topic_name, 10
        )

        self._joint_state_subscriber = self._node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        # The pedal press can be simulated on the keyboard with Ctrl + Alt + 1 (pressed)
        # and Ctrl + Alt + 2 (released).
        self.pedal_pressed_subscriber = self._node.create_subscription(
            Bool,
            "/pedal_pressed",
            self._pedal_pressed_callback,
            10
        )

        self._wait_for_joint_states()
        logger.debug("Waiting for first joint state message (non-blocking)...")
        logger.info(f"OpenArmForwardController initialized, publishing to {self.command_topic_name}")

    def _initialize_ros2(self):
        logger.info("Starting ROS2 initialization...")
        if not rclpy.ok():
            logger.info("rclpy not initialized, calling rclpy.init()")
            try:
                rclpy.init()
                logger.info("rclpy.init() successful")
            except Exception as e:
                logger.error(f"Failed to initialize rclpy: {e}")
                raise

        logger.info("Creating ROS2 node: openarm_forward_controller_node")
        try:
            self._node = Node("openarm_forward_controller_node")
        except Exception as e:
            logger.error(f"Failed to create ROS2 node: {e}")
            raise

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)

        logger.info("Starting ROS2 executor thread")
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        logger.info("ROS2 node initialized: openarm_forward_controller_node")

    def _joint_state_callback(self, msg: JointState):
        with self._joint_states_lock:
            self._joint_states = msg
            positions = []
            velocities = []
            efforts = []
            for joint_name in self.joint_names:
                try:
                    idx = msg.name.index(joint_name)
                    positions.append(msg.position[idx])
                    if msg.velocity:
                        velocities.append(msg.velocity[idx])
                    if msg.effort:
                        efforts.append(msg.effort[idx])
                except ValueError:
                    pass

            if len(positions) == self.num_joints:
                self._current_joint_positions = np.array(positions, dtype=np.float32)
            if len(velocities) == self.num_joints:
                self._current_joint_velocities = np.array(velocities, dtype=np.float32)
            if len(efforts) == self.num_joints:
                self._current_joint_efforts = np.array(efforts, dtype=np.float32)

    def _pedal_pressed_callback(self, msg: Bool):
        with self._pedal_pressed_lock:
            self._pedal_pressed = msg.data

    def _wait_for_joint_states(self, timeout: float = 10.0):
        start_time = time.time()
        while self._current_joint_positions is None:
            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for joint states")
                return False
            time.sleep(0.1)
        return True

    def get_arm_position(self) -> Optional[np.ndarray]:
        return self._current_joint_positions

    def get_arm_velocity(self) -> Optional[np.ndarray]:
        return self._current_joint_velocities

    def get_arm_torque(self) -> Optional[np.ndarray]:
        return self._current_joint_efforts

    def get_arm_states(self) -> dict:
        with self._joint_states_lock:
            result = {
                "joint_position": self._current_joint_positions,
                "joint_velocity": self._current_joint_velocities,
                "joint_torque": self._current_joint_efforts,
                "timestamp": time.time(),
            }
        return result

    def move_arm_joints(self, joint_angles: np.ndarray, duration: Optional[float] = None) -> bool:
        """Publish joint position commands to topic with max_delta limit per step"""

        if len(joint_angles) != self.num_joints:
            logger.error(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")
            return False

        if self._current_joint_positions is None:
            logger.warning("No current joint positions available, sending goal directly")
            scaled_joint_angles = joint_angles
        else:
            delta = joint_angles - self._current_joint_positions
            max_delta_abs = np.max(np.abs(delta))

            if max_delta_abs > self.max_delta:
                scale_factor = self.max_delta / max_delta_abs
                scaled_delta = delta * scale_factor
                scaled_joint_angles = self._current_joint_positions + scaled_delta

            else:
                scaled_joint_angles = joint_angles

        command = Float64MultiArray()
        command.data = [float(x) for x in scaled_joint_angles]


        with self._pedal_pressed_lock:
            if self._pedal_pressed:
                self._joint_command_publisher.publish(command)
        return True

    def home_arm(self) -> bool:
        logger.info("Homing arm to zero position")
        return self.move_arm_joints(np.array(robots.OPENARM_HOME_JS))

    def reset_arm(self) -> bool:
        return self.home_arm()

    def cleanup(self):
        logger.info("Cleaning up OpenArm forward controller...")
        if hasattr(self, "_joint_state_subscriber"):
            self._node.destroy_subscription(self._joint_state_subscriber)
        if hasattr(self, "_pedal_pressed_subscriber"):
            self._node.destroy_subscription(self._pedal_pressed_subscriber)
        if hasattr(self, "_joint_command_publisher"):
            self._node.destroy_publisher(self._joint_command_publisher)
        if hasattr(self, "_executor"):
            self._executor.shutdown()
        if hasattr(self, "_node"):
            self._node.destroy_node()


class DexArmControl:
    def __init__(self, **kwargs):
        self._controller = OpenArmForwardController(**kwargs)

    def get_arm_position(self):
        return self._controller.get_arm_position()

    def get_arm_velocity(self):
        return self._controller.get_arm_velocity()

    def get_arm_torque(self):
        return self._controller.get_arm_torque()

    def get_arm_states(self):
        return self._controller.get_arm_states()

    def move_arm_joints(self, joint_angles: np.ndarray, duration: Optional[float] = None) -> bool:
        return self._controller.move_arm_joints(joint_angles, duration)

    def home_arm(self) -> bool:
        return self._controller.home_arm()

    def reset_arm(self):
        return self._controller.reset_arm()

    def cleanup(self):
        return self._controller.cleanup()
