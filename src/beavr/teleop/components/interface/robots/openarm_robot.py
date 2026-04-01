import logging
import signal
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

_local_index = 0
_local_index_lock = threading.Lock()


def _get_next_index():
    global _local_index
    with _local_index_lock:
        _local_index += 1
        return _local_index


from beavr.teleop.common.network.handshake import HandshakeCoordinator
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.ops import Ops
from beavr.teleop.components.detector.detector_types import SessionCommand
from beavr.teleop.components.interface.controller.robots.openarm_forward_control import DexArmControl

from beavr.teleop.components.interface.controller.robots.openarm_kinematics import (
    OpenArmKinematics,
)
from beavr.teleop.components.interface.interface_base import RobotWrapper
from beavr.teleop.components.interface.interface_types import (
    CartesianState,
    CommandedCartesianState,
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OpenArmRobot(RobotWrapper):
    def __init__(
        self,
        host: str,
        endeff_subscribe_port: int,
        reset_subscribe_port: int,
        home_subscribe_port: int,
        teleoperation_state_port: int,
        endeff_publish_port: int,
        state_publish_port: int,
        **kwargs,
    ):
        logger.info(
            f"Initializing OpenArmRobot with host={host}, endeff_publish_port={endeff_publish_port}, state_publish_port={state_publish_port}"
        )
        if not endeff_publish_port:
            raise ValueError("OpenArmRobot requires an 'endeff_publish_port'")
        if not state_publish_port:
            raise ValueError("OpenArmRobot requires a 'state_publish_port'")

        logger.info("Creating OpenArmKinematics...")
        self._kinematics = OpenArmKinematics()
        logger.info("OpenArmKinematics created successfully")

        logger.info("Creating OpenArmController...")
        self._controller = DexArmControl()
        logger.info("OpenArmController created successfully")

        self._data_frequency = robots.VR_FREQ
        self._num_joints = len(robots.OPENARM_LEFT_JOINT_NAMES)

        self._cartesian_coords_subscriber = ZMQSubscriber(
            host=host,
            port=endeff_subscribe_port,
            topic="endeff_coords",
            message_type=CartesianTarget,
        )

        self._reset_subscriber = ZMQSubscriber(
            host=host,
            port=reset_subscribe_port,
            topic="reset",
            message_type=SessionCommand,
        )

        self._home_subscriber = ZMQSubscriber(
            host=host,
            port=home_subscribe_port,
            topic="home",
            message_type=SessionCommand,
        )

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

        self._publisher_manager = ZMQPublisherManager.get_instance()
        self._publisher_host = host
        self._endeff_publish_port = endeff_publish_port
        self._state_publish_port = state_publish_port

        self._latest_cartesian_coords = None
        self._latest_joint_state = None
        self._latest_cartesian_state_timestamp = 0
        self._latest_joint_state_timestamp = 0

        self._latest_commanded_cartesian_position = None
        self._latest_commanded_cartesian_timestamp = 0.0

        self._latest_joint_angles = None
        self._latest_cartesian_pose = None
        self._cartesian_tolerance = 0.001

        self._joint_angles_lock = threading.Lock()
        self._cartesian_pose_lock = threading.Lock()

        self._frame_rate_history = []
        self._frame_timestamps = []
        self._start_time = None
        self._last_frame_time = None
        self._ik_call_timestamps = []
        self._ik_complete_timestamps = []
        self._ik_called_history = np.zeros(1000, dtype=bool)
        self._ik_completed_history = np.zeros(1000, dtype=bool)
        self._history_index = 0

        self._pending_fk_joints = None

        self._handshake_coordinator = HandshakeCoordinator.get_instance()
        self._handshake_server_id = f"{self.name}_handshake"

        self._handshake_coordinator.start_server(
            subscriber_id=self._handshake_server_id,
            bind_host="*",
            port=robots.TELEOP_HANDSHAKE_PORT + 10,
        )
        logger.info(f"Handshake server started for {self.name}")

        self._is_homed = False

    def _cartesian_positions_close(self, pos1, pos2):
        """Check if two cartesian positions are close within tolerance"""
        if pos1 is None or pos2 is None:
            return False
        return np.linalg.norm(np.array(pos1) - np.array(pos2)) < self._cartesian_tolerance

    def _send_joint_trajectory(self, joint_angles, duration=None):
        """Delegate trajectory publishing to OpenArmController"""
        op_id = _get_next_index()
        start = time.perf_counter()
        success = self._controller.move_arm_joint(joint_angles, duration)
        elapsed = time.perf_counter() - start
        logger.debug(f"[Timing] _send_joint_trajectory op_id={op_id} elapsed={elapsed * 1000:.2f}ms success={success}")
        if not success:
            logger.error("Failed to send joint trajectory")
        return success

    @property
    def name(self):
        return robots.ROBOT_IDENTIFIER_LEFT_OPENARM

    @property
    def recorder_functions(self):
        return {
            "joint_states": self.get_joint_state,
            "operator_cartesian_states": self.get_cartesian_state_from_operator,
            "openarm_cartesian_states": self.get_robot_actual_cartesian_position,
            "commanded_cartesian_state": self.get_cartesian_commanded_position,
            "joint_angles_rad": self.get_joint_position,
        }

    @property
    def data_frequency(self):
        return self._data_frequency

    def get_joint_state(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        joint_states = self._controller.get_arm_states()
        elapsed = time.perf_counter() - start
        logger.debug(f"[Timing] get_joint_state op_id={op_id} elapsed={elapsed * 1000:.2f}ms")
        if joint_states is None or joint_states["joint_position"] is None:
            return None
        return {
            "joint_position": list(np.array(joint_states["joint_position"], dtype=np.float32)),
            "timestamp": joint_states["timestamp"],
        }

    def get_joint_velocity(self):
        return self._controller.get_arm_velocity()

    def get_joint_torque(self):
        return self._controller.get_arm_torque()

    def get_cartesian_state(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        joint_positions = self._controller.get_arm_position()
        elapsed1 = time.perf_counter() - start
        logger.debug(f"[Timing] get_cartesian_state op_id={op_id} get_pos={elapsed1 * 1000:.2f}ms")
        if joint_positions is None:
            return None

        check_start = time.perf_counter()
        joints_tuple = tuple(joint_positions) if hasattr(joint_positions, "__iter__") else (joint_positions,)

        with self._cartesian_pose_lock:
            pose = self._latest_cartesian_pose
            if self._pending_fk_joints is not None and self._pending_fk_joints == joints_tuple:
                logger.debug(f"[Timing] get_cartesian_state op_id={op_id} pending_fk=True (duplicate)")
                return {"cartesian_position": pose, "timestamp": time.time()} if pose is not None else None
            self._pending_fk_joints = joints_tuple

        if pose is not None:
            elapsed_total = time.perf_counter() - start
            logger.debug(f"[Timing] get_cartesian_state op_id={op_id} total={elapsed_total * 1000:.2f}ms (cached)")
            return {
                "cartesian_position": pose,
                "timestamp": time.time(),
            }

        def fk_callback(future, joint_angles=joint_positions.copy(), callback_id="get_cartesian_state"):
            start_time = time.perf_counter()
            try:
                callback_start = time.perf_counter()
                response = future.result()
                get_result_elapsed = time.perf_counter() - callback_start
                if response.error_code.val == 1:
                    if len(response.pose_stamped) > 0:
                        pose_stamped = response.pose_stamped[0]
                        position = pose_stamped.pose.position
                        orientation = pose_stamped.pose.orientation

                        r = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
                        rotation_matrix = r.as_matrix()

                        h_matrix = (
                            (
                                float(rotation_matrix[0, 0]),
                                float(rotation_matrix[0, 1]),
                                float(rotation_matrix[0, 2]),
                                float(position.x),
                            ),
                            (
                                float(rotation_matrix[1, 0]),
                                float(rotation_matrix[1, 1]),
                                float(rotation_matrix[1, 2]),
                                float(position.y),
                            ),
                            (
                                float(rotation_matrix[2, 0]),
                                float(rotation_matrix[2, 1]),
                                float(rotation_matrix[2, 2]),
                                float(position.z),
                            ),
                            (0.0, 0.0, 0.0, 1.0),
                        )
                        before_lock = time.perf_counter() - callback_start
                        with self._cartesian_pose_lock:
                            self._latest_cartesian_pose = h_matrix
                            self._pending_fk_joints = None
                        after_lock = time.perf_counter() - callback_start
                        total_callback_elapsed = time.perf_counter() - start_time
                        logger.debug(
                            f"[Timing] FK callback ({callback_id}) future.result={get_result_elapsed * 1000:.2f}ms "
                            f"pre_lock={before_lock * 1000:.2f}ms post_lock={after_lock * 1000:.2f}ms "
                            f"total={total_callback_elapsed * 1000:.2f}ms"
                        )
                        logger.debug(
                            f"[ROBOT] FK callback completed in get_cartesian_state: position=[{position.x:.4f}, {position.y:.4f}, {position.z:.4f}]"
                        )
                    else:
                        logger.error("FK returned no poses in get_cartesian_state")
                else:
                    logger.error(f"FK failed with error code: {response.error_code.val}")
            except Exception as e:
                logger.error(f"FK callback exception in get_cartesian_state: {e}")

        self._kinematics.compute_fk_callback(joint_positions, fk_callback)
        return None

    def get_joint_position(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        joint_positions = self._controller.get_arm_position()
        elapsed = time.perf_counter() - start
        logger.debug(f"[Timing] get_joint_position op_id={op_id} elapsed={elapsed * 1000:.2f}ms")
        if joint_positions is None:
            return None
        return list(np.array(joint_positions, dtype=np.float32))

    def get_cartesian_position(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        joint_positions = self._controller.get_arm_position()
        elapsed1 = time.perf_counter() - start
        log_prefix = f"[Timing] get_cartesian_position op_id={op_id} get_pos={elapsed1 * 1000:.2f}ms"
        if joint_positions is None:
            logger.debug(f"{log_prefix} result=None (no_positions)")
            return None

        check_start = time.perf_counter()
        joints_tuple = tuple(joint_positions) if hasattr(joint_positions, "__iter__") else (joint_positions,)
        check_pending = time.perf_counter() - check_start

        with self._cartesian_pose_lock:
            result = self._latest_cartesian_pose
            pending_check = time.perf_counter() - check_start
            if self._pending_fk_joints is not None and self._pending_fk_joints == joints_tuple:
                logger.debug(f"{log_prefix} pending_fk=True (duplicate), skipping")
                return result
            self._pending_fk_joints = joints_tuple

        if result is not None:
            elapsed_total = time.perf_counter() - start
            logger.debug(
                f"{log_prefix} fk=CACHED total={elapsed_total * 1000:.2f}ms pending_check={pending_check * 1000:.2f}ms"
            )

        def fk_callback(future, joint_angles=joint_positions.copy(), callback_id="get_cartesian_position"):
            start_time = time.perf_counter()
            try:
                callback_start = time.perf_counter()
                response = future.result()
                get_result_elapsed = time.perf_counter() - callback_start
                if response.error_code.val == 1:
                    if len(response.pose_stamped) > 0:
                        pose_stamped = response.pose_stamped[0]
                        position = pose_stamped.pose.position
                        orientation = pose_stamped.pose.orientation
                        logger.info(f"[robot] pose {pose_stamped.pose.orientation}")
                        r = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
                        rotation_matrix = r.as_matrix()

                        h_matrix = (
                            (
                                float(rotation_matrix[0, 0]),
                                float(rotation_matrix[0, 1]),
                                float(rotation_matrix[0, 2]),
                                float(position.x),
                            ),
                            (
                                float(rotation_matrix[1, 0]),
                                float(rotation_matrix[1, 1]),
                                float(rotation_matrix[1, 2]),
                                float(position.y),
                            ),
                            (
                                float(rotation_matrix[2, 0]),
                                float(rotation_matrix[2, 1]),
                                float(rotation_matrix[2, 2]),
                                float(position.z),
                            ),
                            (0.0, 0.0, 0.0, 1.0),
                        )
                        before_lock = time.perf_counter() - callback_start
                        with self._cartesian_pose_lock:
                            self._latest_cartesian_pose = h_matrix
                            self._pending_fk_joints = None
                        after_lock = time.perf_counter() - callback_start
                        total_callback_elapsed = time.perf_counter() - start_time
                        logger.debug(
                            f"[Timing] FK callback ({callback_id}) future.result={get_result_elapsed * 1000:.2f}ms "
                            f"pre_lock={before_lock * 1000:.2f}ms post_lock={after_lock * 1000:.2f}ms "
                            f"total={total_callback_elapsed * 1000:.2f}ms"
                        )
                        logger.debug(
                            f"[ROBOT] FK callback completed: position=[{position.x:.4f}, {position.y:.4f}, {position.z:.4f}]"
                        )
                    else:
                        logger.error("FK returned no poses")
                else:
                    logger.error(f"FK failed with error code: {response.error_code.val}")
            except Exception as e:
                logger.error(f"FK callback exception: {e}")

        self._kinematics.compute_fk_callback(joint_positions, fk_callback)
        return result

    def reset(self):
        return self._send_joint_trajectory(np.array(robots.OPENARM_HOME_JS))

    def get_teleop_state(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        result = self._arm_teleop_state_subscriber.get_arm_teleop_state()
        elapsed = time.perf_counter() - start
        logger.debug(f"[Timing] get_teleop_state op_id={op_id} elapsed={elapsed * 1000:.2f}ms")
        return result

    def home(self):
        return self._send_joint_trajectory(np.array(robots.OPENARM_HOME_JS))

    def move(self, input_angles):
        self._send_joint_trajectory(input_angles)

    def move_coords(self, input_coords, duration=None):
        """Comute IK and send joint trajectory"""
        position = input_coords[:3]
        orientation = input_coords[3:7]
        joint_angles = self._kinematics.compute_ik(position, orientation)
        if joint_angles is None:
            logger.error("Failed to compute IK solution")
            return
        self._send_joint_trajectory(joint_angles, duration)

    def arm_control(self, cartesian_coords):
        """Compute IK and send joint trajectory"""
        position = cartesian_coords[:3]
        orientation = cartesian_coords[3:7]
        joint_angles = self._kinematics.compute_ik(position, orientation)
        if joint_angles is None:
            logger.error("Failed to compute IK solution")
            return
        self._send_joint_trajectory(joint_angles)

    def get_pose(self):
        return self.get_cartesian_position()

    def get_cartesian_state_from_operator(self):
        if self._latest_cartesian_coords is None:
            return None
        position = tuple(np.asarray(self._latest_cartesian_coords, dtype=np.float32).tolist())
        return CartesianState(position_m=position, timestamp_s=self._latest_cartesian_state_timestamp)

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
        if cartesian_state is None:
            return CartesianState(position_m=(0.0, 0.0, 0.0), timestamp_s=time.time())
        position = tuple(np.asarray(cartesian_state, dtype=np.float32).tolist())
        return CartesianState(position_m=position, timestamp_s=time.time())

    def send_robot_pose(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        joint_positions = self._controller.get_arm_position()
        elapsed1 = time.perf_counter() - start
        logger.debug(f"[Timing] send_robot_pose op_id={op_id} get_pos={elapsed1 * 1000:.2f}ms")
        if joint_positions is None:
            logger.warning("Could not get joint positions for robot pose")
            return

        with self._cartesian_pose_lock:
            pose_homo = self._latest_cartesian_pose

        if pose_homo is None or not isinstance(pose_homo, (list, tuple)) or len(pose_homo) != 4:
            logger.debug(f"[Timing] send_robot_pose op_id={op_id} no_cached_pose, triggering async FK")

        def fk_callback(future, joint_angles=joint_positions.copy(), callback_id="get_cartesian_state"):
            start_time = time.perf_counter()
            try:
                callback_start = time.perf_counter()
                response = future.result()
                get_result_elapsed = time.perf_counter() - callback_start
                if response.error_code.val == 1:
                    if len(response.pose_stamped) > 0:
                        pose_stamped = response.pose_stamped[0]
                        position = pose_stamped.pose.position
                        orientation = pose_stamped.pose.orientation

                        r = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
                        rotation_matrix = r.as_matrix()

                        h_matrix = (
                            (
                                float(rotation_matrix[0, 0]),
                                float(rotation_matrix[0, 1]),
                                float(rotation_matrix[0, 2]),
                                float(position.x),
                            ),
                            (
                                float(rotation_matrix[1, 0]),
                                float(rotation_matrix[1, 1]),
                                float(rotation_matrix[1, 2]),
                                float(position.y),
                            ),
                            (
                                float(rotation_matrix[2, 0]),
                                float(rotation_matrix[2, 1]),
                                float(rotation_matrix[2, 2]),
                                float(position.z),
                            ),
                            (0.0, 0.0, 0.0, 1.0),
                        )
                        before_lock = time.perf_counter() - callback_start
                        with self._cartesian_pose_lock:
                            self._latest_cartesian_pose = h_matrix
                            self._pending_fk_joints = None
                        after_lock = time.perf_counter() - callback_start
                        total_callback_elapsed = time.perf_counter() - start_time
                        logger.debug(
                            f"[Timing] FK callback ({callback_id}) future.result={get_result_elapsed * 1000:.2f}ms "
                            f"pre_lock={before_lock * 1000:.2f}ms post_lock={after_lock * 1000:.2f}ms "
                            f"total={total_callback_elapsed * 1000:.2f}ms"
                        )
                        logger.debug(
                            f"[ROBOT] FK callback completed in get_cartesian_state: position=[{position.x:.4f}, {position.y:.4f}, {position.z:.4f}]"
                        )
                    else:
                        logger.error("FK returned no poses in get_cartesian_state")
                else:
                    logger.error(f"FK failed with error code: {response.error_code.val}")
            except Exception as e:
                logger.error(f"FK callback exception in get_cartesian_state: {e}")
            try:
                callback_start = time.perf_counter()
                response = future.result()
                get_result_elapsed = time.perf_counter() - callback_start
                if response.error_code.val == 1:
                    if len(response.pose_stamped) > 0:
                        pose_stamped = response.pose_stamped[0]
                        position = pose_stamped.pose.position
                        orientation = pose_stamped.pose.orientation

                        r = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
                        rotation_matrix = r.as_matrix()

                        h_matrix = (
                            (
                                float(rotation_matrix[0, 0]),
                                float(rotation_matrix[0, 1]),
                                float(rotation_matrix[0, 2]),
                                float(position.x),
                            ),
                            (
                                float(rotation_matrix[1, 0]),
                                float(rotation_matrix[1, 1]),
                                float(rotation_matrix[1, 2]),
                                float(position.y),
                            ),
                            (
                                float(rotation_matrix[2, 0]),
                                float(rotation_matrix[2, 1]),
                                float(rotation_matrix[2, 2]),
                                float(position.z),
                            ),
                            (0.0, 0.0, 0.0, 1.0),
                        )
                        before_lock = time.perf_counter() - callback_start
                        with self._cartesian_pose_lock:
                            self._latest_cartesian_pose = h_matrix
                            self._pending_fk_joints = None
                        after_lock = time.perf_counter() - callback_start
                        total_callback_elapsed = time.perf_counter() - start_time
                        logger.debug(
                            f"[Timing] FK callback ({callback_id}) future.result={get_result_elapsed * 1000:.2f}ms "
                            f"pre_lock={before_lock * 1000:.2f}ms post_lock={after_lock * 1000:.2f}ms "
                            f"total={total_callback_elapsed * 1000:.2f}ms"
                        )
                        logger.debug(
                            f"[ROBOT] FK callback completed in get_cartesian_state: position=[{position.x:.4f}, {position.y:.4f}, {position.z:.4f}]"
                        )
                    else:
                        logger.error("FK returned no poses in get_cartesian_state")
                else:
                    logger.error(f"FK failed with error code: {response.error_code.val}")
            except Exception as e:
                logger.error(f"FK callback exception in get_cartesian_state: {e}")
                try:
                    callback_start = time.perf_counter()
                    response = future.result()
                    get_result_elapsed = time.perf_counter() - callback_start
                    if response.error_code.val == 1:
                        if len(response.pose_stamped) > 0:
                            pose_stamped = response.pose_stamped[0]
                            position = pose_stamped.pose.position
                            orientation = pose_stamped.pose.orientation

                            r = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
                            rotation_matrix = r.as_matrix()

                            h_matrix = (
                                (
                                    float(rotation_matrix[0, 0]),
                                    float(rotation_matrix[0, 1]),
                                    float(rotation_matrix[0, 2]),
                                    float(position.x),
                                ),
                                (
                                    float(rotation_matrix[1, 0]),
                                    float(rotation_matrix[1, 1]),
                                    float(rotation_matrix[1, 2]),
                                    float(position.y),
                                ),
                                (
                                    float(rotation_matrix[2, 0]),
                                    float(rotation_matrix[2, 1]),
                                    float(rotation_matrix[2, 2]),
                                    float(position.z),
                                ),
                                (0.0, 0.0, 0.0, 1.0),
                            )
                            before_lock = time.perf_counter() - callback_start
                            with self._cartesian_pose_lock:
                                self._latest_cartesian_pose = h_matrix
                            after_lock = time.perf_counter() - callback_start

                            logger.info(
                                f"[ROBOT] Publishing robot pose to 'endeff_homo' on port {self._endeff_publish_port}: "
                                f"position={h_matrix[0][3]:.3f}, {h_matrix[1][3]:.3f}, {h_matrix[2][3]:.3f}"
                            )
                            h_matrix_typed = tuple(tuple(float(x) for x in row) for row in h_matrix)
                            self._publisher_manager.publish(
                                host=self._publisher_host,
                                port=self._endeff_publish_port,
                                topic="endeff_homo",
                                data=CartesianState(
                                    timestamp_s=time.time(),
                                    h_matrix=h_matrix_typed,
                                ),
                            )
                            publish_elapsed = time.perf_counter() - callback_start
                            total_callback_elapsed = time.perf_counter() - start_time
                            logger.debug(
                                f"[Timing] FK callback (send_robot_pose) future.result={get_result_elapsed * 1000:.2f}ms "
                                f"pre_lock={before_lock * 1000:.2f}ms post_lock={after_lock * 1000:.2f}ms "
                                f"publish={publish_elapsed * 1000:.2f}ms total={total_callback_elapsed * 1000:.2f}ms"
                            )
                            logger.info(f"[ROBOT] Successfully published robot pose to 'endeff_homo'")
                        else:
                            logger.error("FK returned no poses in send_robot_pose")
                    else:
                        logger.error(f"FK failed with error code: {response.error_code.val}")
                except Exception as e:
                    logger.error(f"FK callback exception in send_robot_pose: {e}")

            self._kinematics.compute_fk_callback(joint_positions, fk_callback)
            return

        elapsed_total = time.perf_counter() - start
        logger.debug(f"[Timing] send_robot_pose op_id={op_id} using cached_pose total={elapsed_total * 1000:.2f}ms")

        try:
            h_matrix = tuple(tuple(float(x) for x in row) for row in pose_homo)

            logger.info(
                f"[ROBOT] Publishing robot pose to 'endeff_homo' on port {self._endeff_publish_port}: "
                f"position={h_matrix[0][3]:.3f}, {h_matrix[1][3]:.3f}, {h_matrix[2][3]:.3f}"
            )
            self._publisher_manager.publish(
                host=self._publisher_host,
                port=self._endeff_publish_port,
                topic="endeff_homo",
                data=CartesianState(
                    timestamp_s=time.time(),
                    h_matrix=h_matrix,
                ),
            )
            logger.info(f"[ROBOT] Successfully published robot pose to 'endeff_homo'")
        except Exception as e:
            logger.error(f"Failed to publish robot pose for {self.name}: {e}")

    def check_reset(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        reset_bool = self._reset_subscriber.recv_keypoints()
        elapsed = time.perf_counter() - start
        logger.debug(f"[Timing] check_reset op_id={op_id} elapsed={elapsed * 1000:.2f}ms")
        return reset_bool is not None

    def check_home(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        home_bool = self._home_subscriber.recv_keypoints()
        elapsed = time.perf_counter() - start
        logger.debug(f"[Timing] check_home op_id={op_id} elapsed={elapsed * 1000:.2f}ms")
        if home_bool == robots.ARM_TELEOP_STOP:
            return True
        elif home_bool == robots.ARM_TELEOP_CONT:
            return False
        return False

    def stream(self):
        logger.info("*** STARTED OPENARM ROBOT")
        self.home()

        target_interval = 1.0 / self._data_frequency
        next_frame_time = time.time()
        frame_count = 0
        self._start_time = time.time()

        while True:
            logger.info("[robot] while true")
            current_time = time.time()

            self._history_index = frame_count % 1000

            if self._last_frame_time is not None:
                frame_time = current_time - self._last_frame_time
                frame_rate = 1.0 / frame_time if frame_time > 0 else 0.0
                self._frame_rate_history.append(frame_rate)
                self._frame_timestamps.append(current_time - self._start_time)
                if len(self._frame_rate_history) > 1000:
                    self._frame_rate_history.pop(0)
                    self._frame_timestamps.pop(0)
                avg_frame_rate = sum(self._frame_rate_history) / len(self._frame_rate_history)
                logger.info(
                    f"[ROBOT] Frame time: {frame_time * 1000:.2f}ms, Frame rate: {frame_rate:.2f}Hz, Avg rate: {avg_frame_rate:.2f}Hz"
                )
            self._last_frame_time = current_time

            if current_time >= next_frame_time:
                frame_start = time.perf_counter()
                next_frame_time = current_time + target_interval

                t1 = time.perf_counter()
                home_signaled = self.check_home()
                t2 = time.perf_counter()
                logger.debug(f"[Timing] frame {frame_count} check_home={(t2 - t1) * 1000:.2f}ms")

                if home_signaled and not self._is_homed:
                    t3 = time.perf_counter()
                    self.home()
                    joint_angles = np.array(robots.OPENARM_HOME_JS)
                    with self._joint_angles_lock:
                        self._latest_joint_angles = joint_angles
                    self._is_homed = True
                    self.send_robot_pose()
                    t4 = time.perf_counter()
                    logger.debug(f"[Timing] frame {frame_count} home_sequence={(t4 - t3) * 1000:.2f}ms")
                elif not home_signaled and self._is_homed:
                    self._is_homed = False

                t5 = time.perf_counter()
                reset_signaled = self.check_reset()
                t6 = time.perf_counter()
                logger.debug(f"[Timing] frame {frame_count} check_reset={(t6 - t5) * 1000:.2f}ms")
                if reset_signaled:
                    self.send_robot_pose()

                t7 = time.perf_counter()
                teleop_state = self.get_teleop_state()
                t8 = time.perf_counter()
                logger.debug(f"[Timing] frame {frame_count} get_teleop_state={(t8 - t7) * 1000:.2f}ms")
                if teleop_state == robots.ARM_TELEOP_STOP:
                    logger.debug(f"Teleop state is STOP, skipping movement")
                    continue

                recv_start = time.perf_counter()
                msg = self._cartesian_coords_subscriber.recv_keypoints()
                recv_elapsed = time.perf_counter() - recv_start
                logger.debug(f"[Timing] frame {frame_count} recv_keypoints={recv_elapsed * 1000:.2f}ms")
                cmd = msg
                if cmd is not None:
                    logger.debug(
                        f"[ROBOT] Received cartesian command: pos={cmd.position_m}, orient={cmd.orientation_xyzw}"
                    )
                    new_cartesian_position = np.concatenate(
                        [
                            np.asarray(cmd.position_m, dtype=np.float32),
                            np.asarray(cmd.orientation_xyzw, dtype=np.float32),
                        ]
                    )
                    new_cartesian_timestamp = cmd.timestamp_s

                    t9 = time.perf_counter()
                    position_changed = self._cartesian_positions_close(
                        new_cartesian_position, self._latest_commanded_cartesian_position
                    )
                    t10 = time.perf_counter()
                    logger.debug(f"[Timing] frame {frame_count} position_check={(t10 - t9) * 1000:.2f}ms")

                    if self._latest_commanded_cartesian_position is None or not position_changed:
                        logger.debug("[ROBOT] Cartesian position changed, submitting IK request")
                        position = new_cartesian_position[:3]
                        orientation = new_cartesian_position[3:7]

                        t11 = time.perf_counter()
                        target_pos = new_cartesian_position.copy()
                        target_time = new_cartesian_timestamp

                        def ik_callback(
                            future,
                            target_pos=target_pos,
                            target_time=target_time,
                            history_index=self._history_index,
                            start_time=time.perf_counter(),
                        ):
                            try:
                                callback_start = time.perf_counter()
                                response = future.result()
                                get_result_elapsed = time.perf_counter() - callback_start
                                if response.error_code.val == 1:
                                    self._ik_completed_history[history_index] = True
                                    completion_time = time.time()
                                    self._ik_complete_timestamps.append(
                                        completion_time - self._start_time if self._start_time else completion_time
                                    )
                                    joint_angles = [
                                        response.solution.joint_state.position[i] for i in range(self._num_joints)
                                    ]
                                    before_lock = time.perf_counter() - callback_start
                                    with self._joint_angles_lock:
                                        self._latest_joint_angles = joint_angles
                                        self._latest_commanded_cartesian_position = target_pos
                                        self._latest_commanded_cartesian_timestamp = target_time
                                    after_lock = time.perf_counter() - callback_start
                                    total_callback_elapsed = time.perf_counter() - start_time
                                    logger.debug(
                                        f"[Timing] IK callback future.result={get_result_elapsed * 1000:.2f}ms "
                                        f"pre_lock={before_lock * 1000:.2f}ms post_lock={after_lock * 1000:.2f}ms "
                                        f"total={total_callback_elapsed * 1000:.2f}ms"
                                    )
                                    logger.debug(f"[ROBOT] IK callback completed: {joint_angles}")
                                else:
                                    logger.error(f"IK failed with error code: {response.error_code.val}")
                            except Exception as e:
                                logger.error(f"IK callback exception: {e}")

                        self._ik_called_history[self._history_index] = True
                        call_time = time.time()
                        self._ik_call_timestamps.append(call_time - self._start_time if self._start_time else call_time)

                        self._kinematics.compute_ik_callback(position, orientation, ik_callback)
                        t12 = time.perf_counter()
                        logger.debug(f"[Timing] frame {frame_count} IK_submit={(t12 - t11) * 1000:.2f}ms")
                    else:
                        logger.debug("[ROBOT] Cartesian position unchanged, using cached joint angles")
                else:
                    logger.debug("[ROBOT] No cartesian command received")

                t13 = time.perf_counter()
                if self._latest_joint_angles is not None:
                    logger.debug(f"[ROBOT] Sending joint angles to controller: {self._latest_joint_angles}")
                    self._send_joint_trajectory(self._latest_joint_angles)
                else:
                    logger.debug("[ROBOT] No joint angles available to send")
                t14 = time.perf_counter()
                logger.debug(f"[Timing] frame {frame_count} _send_joint_trajectory={(t14 - t13) * 1000:.2f}ms")

                t15 = time.perf_counter()
                self.publish_current_state()
                t16 = time.perf_counter()
                logger.debug(f"[Timing] frame {frame_count} publish_current_state={(t16 - t15) * 1000:.2f}ms")

                sleep_time = max(0, next_frame_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)

                frame_count += 1
                # if frame_count % 200 == 0:
                #    self.save_frame_rate_history(f"frame_rate_history_{frame_count}.npy")
                frame_elapsed = time.perf_counter() - frame_start
                logger.debug(f"[Timing] frame {frame_count} TOTAL={frame_elapsed * 1000:.2f}ms")

    def publish_current_state(self):
        op_id = _get_next_index()
        start = time.perf_counter()

        joint_states = self.get_joint_state()
        operator_cart = self.get_cartesian_state_from_operator()
        robot_cart = self.get_robot_actual_cartesian_position()
        commanded_cart = self.get_cartesian_commanded_position()
        joint_angles_rad = self.get_joint_position()

        elapsed_getters = time.perf_counter() - start

        current_state_dict = {}
        if joint_states is not None:
            current_state_dict["joint_states"] = joint_states
        if operator_cart is not None:
            current_state_dict["operator_cartesian_states"] = operator_cart.to_dict()
        if robot_cart is not None:
            current_state_dict["openarm_cartesian_states"] = robot_cart.to_dict()
        if commanded_cart is not None:
            current_state_dict["commanded_cartesian_state"] = commanded_cart.to_dict()
        if joint_angles_rad is not None:
            current_state_dict["joint_angles_rad"] = joint_angles_rad

        current_state_dict["timestamp"] = time.perf_counter()

        publish_start = time.perf_counter()
        self._publisher_manager.publish(
            host=self._publisher_host,
            port=self._state_publish_port,
            topic=self.name,
            data=current_state_dict,
        )
        publish_elapsed = time.perf_counter() - publish_start
        total_elapsed = time.perf_counter() - start

        logger.debug(
            f"[Timing] publish_current_state op_id={op_id} "
            f"getters={elapsed_getters * 1000:.2f}ms "
            f"publish={publish_elapsed * 1000:.2f}ms "
            f"total={total_elapsed * 1000:.2f}ms"
        )

    def save_frame_rate_history(self, filename="frame_rate_history.npy"):
        """Save frame rate history to file"""
        if self._frame_rate_history:
            data = {
                "frame_rates": np.array(self._frame_rate_history),
                "timestamps": np.array(self._frame_timestamps),
                "ik_call_timestamps": np.array(self._ik_call_timestamps),
                "ik_complete_timestamps": np.array(self._ik_complete_timestamps),
                "ik_called": self._ik_called_history.copy(),
                "ik_completed": self._ik_completed_history.copy(),
                "start_time": self._start_time,
            }
            np.save(filename, data)
            logger.info(f"Saved {len(self._frame_rate_history)} frame rate samples to {filename}")
        else:
            logger.warning("No frame rate history to save")

    def shutdown(self):
        """Graceful shutdown - save frame rate history"""
        logger.info("Shutting down OpenArmRobot...")
        self.save_frame_rate_history()
        if hasattr(self, "_handshake_coordinator") and hasattr(self, "_handshake_server_id"):
            self._handshake_coordinator.stop_server(self._handshake_server_id)
        if hasattr(self, "_kinematics"):
            self._kinematics.cleanup()
        if hasattr(self, "_controller"):
            self._controller.cleanup()
        cleanup_zmq_resources()

    def __del__(self):
        self.shutdown()
