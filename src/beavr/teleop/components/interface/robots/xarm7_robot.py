import logging
import time

import numpy as np

from beavr.teleop.common.network.handshake import HandshakeCoordinator
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.ops import Ops
from beavr.teleop.components.detector.detector_types import SessionCommand
from beavr.teleop.components.interface.controller.robots.xarm7_control import (
    DexArmControl,
)
from beavr.teleop.components.interface.interface_base import RobotWrapper
from beavr.teleop.components.interface.interface_types import (
    CartesianState,
    CommandedCartesianState,
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class XArm7Robot(RobotWrapper):
    """XArm7 teleop interface and state publisher.

    Data flow overview:
    - Subscribers receive operator inputs on topics: 'endeff_coords', 'joint', 'reset', 'home', 'pause'.
    - Internal getters return strongly typed boundary objects (e.g., CartesianState) or native dicts.
    - publish_current_state() serializes those boundary objects back to the legacy wire format expected by
      downstream consumers (see BeavrBot in beavrbot.py and config paths in configs.py).

    Wire format (unchanged):
    - 'joint_states' -> { 'joint_position': List[float], 'timestamp': float }
    - 'operator_cartesian_states' -> { 'cartesian_position': [x, y, z], 'timestamp': float }
    - 'xarm_cartesian_states' -> { 'cartesian_position': [x, y, z], 'timestamp': float }
    - 'commanded_cartesian_state' -> { 'commanded_cartesian_position': [x,y,z,qx,qy,qz,qw], 'timestamp': float }
    - plus 'joint_angles_rad' for convenience (not consumed by BeavrBot)
    """

    def __init__(
        self,
        host,
        endeff_subscribe_port,
        joint_subscribe_port,
        home_subscribe_port,
        reset_subscribe_port,
        teleoperation_state_port,
        robot_ip,
        is_right_arm=True,
        endeff_publish_port: int = 10009,
        state_publish_port: int = 10010,
        **kwargs,
    ):
        """
        Args:
            host: Network host address
            endeff_subscribe_port: Port for end effector command subscription
            joint_subscribe_port: Port for joint command subscription (if used)
            reset_subscribe_port: Port for reset subscription
            robot_ip: IP address of the XArm robot
            is_right_arm: Whether this is the right arm (True) or left arm (False)
            endeff_publish_port: Port for publishing general end-effector data.
            state_publish_port: Port for publishing the full state dictionary for recording.
        """
        # Ensure required ports are present
        if not endeff_publish_port:
            raise ValueError("XArm7Robot requires an 'endeff_publish_port'")
        if not state_publish_port:
            raise ValueError("XArm7Robot requires a 'state_publish_port'")

        self._controller = DexArmControl(ip=robot_ip)
        self._is_right_arm = is_right_arm

        self._data_frequency = robots.VR_FREQ

        # Subscribers
        self._cartesian_coords_subscriber = ZMQSubscriber(
            host=host,
            port=endeff_subscribe_port,
            topic="endeff_coords",
            message_type=CartesianTarget,
        )

        # Dedicated RESET subscriber -------------------------------------------------
        self._reset_subscriber = ZMQSubscriber(
            host=host,
            port=reset_subscribe_port,
            topic="reset",
            message_type=SessionCommand,
        )

        # Dedicated HOME subscriber --------------------------------------------------
        self._home_subscriber = ZMQSubscriber(
            host=host,
            port=home_subscribe_port,
            topic="home",
            message_type=SessionCommand,
        )

        # Ops state subscriber --------------------------------------------------------
        # Checks if operation is stopped or continued.
        self._arm_teleop_state_subscriber = Ops(
            arm_teleop_state_subscriber=ZMQSubscriber(
                host=host,
                port=teleoperation_state_port,
                topic="pause",
                message_type=SessionCommand,
            )
        )

        self._subscribers = {
            "cartesian_coords": self._cartesian_coords_subscriber,
            "reset": self._reset_subscriber,
            "home": self._home_subscriber,
            "teleop_state": self._arm_teleop_state_subscriber.get_arm_teleop_state,
        }

        # Centralized publisher manager and publishing configuration
        self._publisher_manager = ZMQPublisherManager.get_instance()
        self._publisher_host = host
        self._endeff_publish_port = endeff_publish_port
        self._state_publish_port = state_publish_port

        # Add caches for received messages
        self._latest_cartesian_coords = None
        self._latest_joint_state = None
        self._latest_cartesian_state_timestamp = 0
        self._latest_joint_state_timestamp = 0

        # Recording control
        self._is_recording_enabled = False

        # Add cache for the last valid commanded cartesian position received
        self._latest_commanded_cartesian_position = None
        self._latest_commanded_cartesian_timestamp = 0.0

        # Add handshake coordination between operator and robot ----------------------
        self._handshake_coordinator = HandshakeCoordinator.get_instance()
        self._handshake_server_id = f"{self.name}_handshake"

        # Start handshake server for this robot
        self._handshake_coordinator.start_server(
            subscriber_id=self._handshake_server_id,
            bind_host="*",
            port=robots.TELEOP_HANDSHAKE_PORT + (1 if self._is_right_arm else 2),  # Unique ports
        )
        logger.info(f"Handshake server started for {self.name}")

        self._is_homed = False

    # recorder_functions removed in favor of inlined state gathering in publish_current_state()

    @property
    def name(self):
        return f"{robots.ROBOT_NAME_XARM7}_{'right' if self._is_right_arm else 'left'}"

    @property
    def recorder_functions(self):
        """Required by abstract base class. Returns mapping of state keys to getter functions."""
        return {
            "joint_states": self.get_joint_state,
            "operator_cartesian_states": self.get_cartesian_state_from_operator,
            "xarm_cartesian_states": self.get_robot_actual_cartesian_position,
            "commanded_cartesian_state": self.get_cartesian_commanded_position,
            "joint_angles_rad": self.get_joint_position,
        }

    @property
    def data_frequency(self):
        return self._data_frequency

    # State information functions
    def get_joint_state(self):
        arm_states = self._controller.get_arm_states()
        if arm_states is None or arm_states.get("joint_position") is None:
            return None
        # Publish only the wire-format keys expected downstream
        return {
            "joint_position": list(np.array(arm_states["joint_position"], dtype=np.float32)),
            "timestamp": arm_states.get("timestamp", time.time()),
        }

    def get_joint_velocity(self):
        return self._controller.get_arm_velocity()

    def get_joint_torque(self):
        return self._controller.get_arm_torque()

    def get_cartesian_state(self):
        cartesian_state = self._controller.get_cartesian_state()
        return cartesian_state

    def get_joint_position(self):
        arm_position = self._controller.get_arm_position()
        if arm_position is None:
            return None
        return list(np.array(arm_position, dtype=np.float32))

    def get_cartesian_position(self):
        return self._controller.get_arm_cartesian_coords()

    def reset(self):
        return self._controller._init_xarm_control()

    def get_teleop_state(self):
        """
        Checks if operation is stopped or continued.
        """
        return self._arm_teleop_state_subscriber.get_arm_teleop_state()

    def get_pose(self):
        return self._controller.get_arm_pose()

    # Movement functions
    def home(self):
        return self._controller.home_arm()

    def move(self, input_angles):
        self._controller.move_arm_joint(input_angles)

    def move_coords(self, cartesian_coords, duration=3):
        self._controller.move_arm_cartesian(cartesian_coords, duration=duration)

    def arm_control(self, cartesian_coords):
        self._controller.arm_control(cartesian_coords)

    def move_velocity(self, input_velocity_values, duration):
        pass

    def get_cartesian_state_from_operator(self):
        if self._latest_cartesian_coords is None:
            return None
        # Keep typed boundary internally; publish as dict via to_dict() later
        position = tuple(np.asarray(self._latest_cartesian_coords, dtype=np.float32).tolist())
        return CartesianState(position_m=position, timestamp_s=self._latest_cartesian_state_timestamp)

    def get_joint_state_from_operator(self):
        if self._latest_joint_state is None:
            return None
        return {
            "joint_position": list(np.array(self._latest_joint_state, dtype=np.float32)),
            "timestamp": self._latest_joint_state_timestamp,
        }

    def get_cartesian_commanded_position(self):
        if self._latest_commanded_cartesian_position is None:
            return None
        return CommandedCartesianState(
            commanded_cartesian_position=self._latest_commanded_cartesian_position.tolist()
            if isinstance(self._latest_commanded_cartesian_position, np.ndarray)
            else list(self._latest_commanded_cartesian_position),
            timestamp_s=self._latest_commanded_cartesian_timestamp,
        )

    def get_robot_actual_cartesian_position(self):
        cartesian_state = self.get_cartesian_position()
        position = tuple(np.asarray(cartesian_state, dtype=np.float32).tolist())
        return CartesianState(position_m=position, timestamp_s=time.time())

    def get_robot_actual_joint_position(self):
        # Reuse the joint state getter for observed joint data
        return self.get_joint_state()

    def send_robot_pose(self):
        pose_homo = self._controller.get_arm_pose()
        try:
            h_matrix = tuple(tuple(float(x) for x in row) for row in pose_homo)

            # TODO: Remove the literal in the topic arg use a constant.
            self._publisher_manager.publish(
                host=self._publisher_host,
                port=self._endeff_publish_port,
                topic="endeff_homo",
                data=CartesianState(
                    timestamp_s=time.time(),
                    h_matrix=h_matrix,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to publish robot pose for {self.name}: {e}")

    def check_reset(self):
        reset_bool = self._reset_subscriber.recv_keypoints()
        return reset_bool is not None

    def check_home(self):
        home_bool = self._home_subscriber.recv_keypoints()

        if home_bool == robots.ARM_TELEOP_STOP:
            return True
        elif home_bool == robots.ARM_TELEOP_CONT:
            return False

        return False

    # Modified stream method with automatic recording start after reset
    def stream(self):
        self.home()
        assert self._controller.robot.set_mode_and_state(1, 0), "Failed to enter SERVO-READY"

        target_interval = 1.0 / self._data_frequency
        next_frame_time = time.time()

        while True:
            current_time = time.time()

            # Only process at the target frequency
            if current_time >= next_frame_time:
                # Calculate next frame time
                next_frame_time = current_time + target_interval

                if self.check_home() and not self._is_homed:
                    # Execute the homing motion.
                    self.home()
                    self._is_homed = True
                    self.send_robot_pose()
                elif not self.check_home() and self._is_homed:
                    self._is_homed = False

                if self.check_reset():
                    self.send_robot_pose()

                # Check operation state --------------------------------------------------
                if self.get_teleop_state() == robots.ARM_TELEOP_STOP:
                    # Stop the robot from moving and wait until we operate again.
                    continue

                msg = self._cartesian_coords_subscriber.recv_keypoints()
                cmd = msg
                if cmd is not None:
                    self._latest_commanded_cartesian_position = np.concatenate(
                        [
                            np.asarray(cmd.position_m, dtype=np.float32),
                            np.asarray(cmd.orientation_xyzw, dtype=np.float32),
                        ]
                    )
                    self._latest_commanded_cartesian_timestamp = cmd.timestamp_s

                # Only move if we have a valid cartesian position
                if self._latest_commanded_cartesian_position is not None:
                    self.move_coords(self._latest_commanded_cartesian_position)

                # Publish the current state every cycle so that external
                # adapters (e.g. MultiRobotAdapter) receive up-to-date joint
                # information for observation building.
                self.publish_current_state()

                # Calculate sleep time to maintain consistent frequency
                sleep_time = max(0, next_frame_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def publish_current_state(self):
        # TODO: This needs to be updated along with how external adapters receive data.
        """
        Gathers interface state and publishes a single dictionary via ZMQ using
        self.name as topic on the state_publish_port. Uses existing types for
        cartesian data (CartesianState, CommandedCartesianState) and simple
        dict/list for joints for clarity.
        """
        publish_time = time.time()

        # 1) Joint state (dict: joint_position, timestamp)
        joint_states = self.get_joint_state()

        # 2) Operator cartesian state (CartesianState)
        operator_cart = self.get_cartesian_state_from_operator()

        # 3) Robot actual cartesian state (CartesianState)
        robot_cart = self.get_robot_actual_cartesian_position()

        # 4) Commanded cartesian state (CommandedCartesianState)
        commanded_cart = self.get_cartesian_commanded_position()

        # 5) Convenience raw joint angles list
        joint_angles_rad = self.get_joint_position()

        current_state_dict = {}
        if joint_states is not None:
            current_state_dict["joint_states"] = joint_states
        if operator_cart is not None:
            current_state_dict["operator_cartesian_states"] = operator_cart.to_dict()
        if robot_cart is not None:
            current_state_dict["xarm_cartesian_states"] = robot_cart.to_dict()
        if commanded_cart is not None:
            current_state_dict["commanded_cartesian_state"] = commanded_cart.to_dict()
        if joint_angles_rad is not None:
            current_state_dict["joint_angles_rad"] = joint_angles_rad

        current_state_dict["timestamp"] = publish_time

        # Publish the state dictionary using the dedicated state publisher and self.name topic
        self._publisher_manager.publish(
            host=self._publisher_host,
            port=self._state_publish_port,
            topic=self.name,
            data=current_state_dict,
        )

    def __del__(self):
        # Stop handshake server for this robot
        if hasattr(self, "_handshake_coordinator") and hasattr(self, "_handshake_server_id"):
            self._handshake_coordinator.stop_server(self._handshake_server_id)
        cleanup_zmq_resources()
