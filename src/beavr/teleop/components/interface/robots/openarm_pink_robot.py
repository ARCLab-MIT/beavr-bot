import logging
import threading
import time
from typing import Optional, Tuple, List

import numpy as np

from beavr.teleop.configs.constants import robots
from beavr.teleop.components.interface.controller.robots.openarm_forward_control import DexArmControl
from beavr.teleop.components.interface.interface_base import RobotWrapper
from beavr.teleop.components.interface.interface_types import (
    CartesianState,
    CommandedCartesianState,
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.common.network.handshake import HandshakeCoordinator
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.ops import Ops
from beavr.teleop.components.detector.detector_types import SessionCommand

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_local_index = 0
_local_index_lock = threading.Lock()


def _get_next_index():
    global _local_index
    with _local_index_lock:
        _local_index += 1
        return _local_index


# ============================================================================
# Pink Configuration Constants
# ============================================================================
# Task costs for FrameTask (end-effector positioning)
PINK_POSITION_COST = 100.0  # [cost] / [m] - increased to prioritize positioning
PINK_ORIENTATION_COST = 1.0  # [cost] / [rad] - increased to prioritize orientation
PINK_LM_DAMPING = 0.1  # Levenberg-Marquardt damping

# Posture task for joint regularization
PINK_POSTURE_COST = 1e-6  # [cost] / [rad] - reduced to minimize interference with frame task

# IK velocity integration time step
PINK_IK_DT = 1.1  # seconds - increased for more movement per IK step

# Iterative IK parameters
PINK_MAX_ITERATIONS = 5  # max IK iterations per call
PINK_POS_TOLERANCE = 0.01  # position tolerance in meters

# Best-effort joint limits (radians)
PINK_JOINT_LIMIT_RANGE = np.pi  # +/- π for clamping


class PinkKinematics:
    """
    Pink-based kinematics solver for OpenArm robot.
    Replaces MoveIt services with local pinocchio/pink kinematics.
    """

    def __init__(
        self,
        joint_names=None,
        ik_link_name="openarm_left_hand_tcp",
        urdf_path=None,
    ):
        import pink
        from pink.tasks import FrameTask, PostureTask
        import qpsolvers
        from scipy.spatial.transform import Rotation
        from robot_descriptions.loaders.pinocchio import load_robot_description

        # Joint configuration
        self.joint_names = joint_names or robots.OPENARM_LEFT_JOINT_NAMES
        self.ik_link_name = ik_link_name
        self.num_joints = len(self.joint_names)
        # openarm_description is a CMake-based ROS2 package, not compatible with robot_descriptions
        # Use direct URDF loading instead
        self._load_robot_from_urdf(urdf_path=urdf_path)

    def _load_robot_from_urdf(self, urdf_path=None):
        import xacro
        import pinocchio as pin
        from pathlib import Path
        import os

        try:
            logger.info(f"[PinkKinematics] Loading URDF from: {urdf_path}")
            urdf_file = urdf_path

            # Resolve package path
            urdf_path_obj = Path(urdf_file)
            openarm_path = urdf_path_obj.parent.parent.parent.parent
            openarm_path = openarm_path.parent  # Go up one more level to get the openarm_description root

            if not openarm_path.exists() or not openarm_path.is_dir():
                openarm_path = Path("/home/ubuntu/workshop-robotics/src/external_dependencies/openarm_description")

            logger.info(f"[PinkKinematics] Resolved package path: {openarm_path}")

            self._robot_wrapper = pin.RobotWrapper.BuildFromURDF(
                filename=urdf_file,
                package_dirs=[str(urdf_path_obj.parent)],
                root_joint=None,
            )
            logger.info("[PinkKinematics] build robot wrapper from urdf")
            self._robot_model = self._robot_wrapper.model
            self._robot_data = self._robot_wrapper.data

            # Disable joint limit checking by setting limits to very large values
            if hasattr(self._robot_model, "lowerPositionLimit"):
                logger.info(f"[PinkKinematics] Original lower limits: {self._robot_model.lowerPositionLimit}")
                self._robot_model.lowerPositionLimit[:] = -np.pi
            if hasattr(self._robot_model, "upperPositionLimit"):
                logger.info(f"[PinkKinematics] Original upper limits: {self._robot_model.upperPositionLimit}")
                self._robot_model.upperPositionLimit[:] = np.pi

            logger.info(
                f"[PinkKinematics] Robot model loaded: nq={self._robot_model.nq}, nv={self._robot_model.nv}, "
                f"frames={[f.name for f in self._robot_model.frames]}...\n joints: {[joint for joint in self._robot_model.joints]}"
            )

            import pink
            from pink.tasks import FrameTask, PostureTask
            import qpsolvers

            logger.info(
                f"[PinkKinematics] Robot model joint limits: lower={self._robot_model.lowerPositionLimit.shape}, upper={self._robot_model.upperPositionLimit.shape}"
            )
            logger.info(f"[PinkKinematics] Robot model lower limits: {self._robot_model.lowerPositionLimit}")
            logger.info(f"[PinkKinematics] Robot model upper limits: {self._robot_model.upperPositionLimit}")
            logger.info(
                f"[PinkKinematics] Robot model nq={self._robot_model.nq}, nv={self._robot_model.nv}, q0 shape={len(self._robot_wrapper.q0)}"
            )

            # Create Configuration with the full robot model
            # pink will solve IK for all joints, but we'll extract only left arm joints later
            self._configuration = pink.Configuration(self._robot_model, self._robot_data, self._robot_wrapper.q0)

            logger.info(
                f"[PinkKinematics] Configuration q shape: {self._configuration.q.shape}, value: {self._configuration.q}"
            )

            # Log complete Pink model joint structure for debugging
            logger.info("[PinkKinematics] Complete Pink model joint structure:")
            for i in range(self._robot_model.njoints):
                joint = self._robot_model.joints[i]
                joint_name = self._robot_model.names[i] if i < len(self._robot_model.names) else f"joint_{i}_unnamed"
                logger.info(
                    f"[PinkKinematics]   Pink index {i}: name='{joint_name}', "
                    f"type={type(joint).__name__}, nq={joint.nq}, nv={joint.nv}"
                )

            # Build correct joint mapping: controller joint -> Pink model DOF index (idx_q)
            # Map each joint from OPENARM_LEFT_JOINT_NAMES to its Pinocchio configuration index (idx_q)
            left_joint_dof_indices = []
            logger.info("[PinkKinematics] Building joint mapping from controller to Pink model DOF indices:")
            for i, joint_name in enumerate(self.joint_names):
                try:
                    # Get joint ID first, then look up joint object to find idx_q
                    joint_id = self._robot_model.getJointId(joint_name)
                    joint = self._robot_model.joints[joint_id]
                    idx_q = joint.idx_q
                    logger.info(
                        f"[PinkKinematics]   Controller[{i}] '{joint_name}' -> joint_id={joint_id}, "
                        f"type={type(joint).__name__}, nq={joint.nq}, nv={joint.nv}, idx_q={idx_q}, idx_v={joint.idx_v}"
                    )
                    if idx_q < 0 or idx_q >= self._robot_model.nq:
                        logger.error(
                            f"[PinkKinematics]   ERROR: idx_q={idx_q} out of range (nq={self._robot_model.nq})"
                        )
                        raise ValueError(f"Joint DOF index {idx_q} invalid for {joint_name}")
                    left_joint_dof_indices.append(idx_q)
                except Exception as e:
                    logger.error(f"[PinkKinematics]   ERROR: Failed to find Pink DOF index for '{joint_name}': {e}")
                    raise

            self._left_joint_dof_indices = left_joint_dof_indices

            if len(left_joint_dof_indices) != len(self.joint_names):
                logger.error(
                    f"[PinkKinematics] Mismatch: Expected {len(self.joint_names)} left arm joints, "
                    f"but found {len(left_joint_dof_indices)} in model"
                )
                raise ValueError("Failed to map all left arm joints to Pink model")

            logger.info(f"[PinkKinematics] Final joint mapping (controller -> Pinocchio DOF indices):")
            for i, dof_idx in enumerate(left_joint_dof_indices):
                logger.info(f"[PinkKinematics]   {i}: '{self.joint_names[i]}' -> idx_q={dof_idx}")

            # Try position-only task to avoid orientation conflicts
            self._end_effector_task = FrameTask(
                self.ik_link_name,
                position_cost=PINK_POSITION_COST,
                orientation_cost=PINK_ORIENTATION_COST,
                lm_damping=PINK_LM_DAMPING,
            )

            # Disable posture task for now to debug tiny velocity issue
            self._posture_task = None
            self._tasks = [self._end_effector_task]

            for task in self._tasks:
                task.set_target_from_configuration(self._configuration)
            logger.info(
                f"[PinkKinematics] Tasks initialized: end_effector_task={self._end_effector_task}, posture_task={self._posture_task}"
            )

            self._solver = qpsolvers.available_solvers[0]
            if "daqp" in qpsolvers.available_solvers:
                self._solver = "daqp"

            logger.info(f"Using QP solver: {self._solver}")
            logger.info(
                f"[PinkKinematics] Configuration initialized: q0 has {len(self._robot_wrapper.q0)} elements, nq={self._robot_model.nq}"
            )

        except Exception as e:
            logger.error(f"[PinkKinematics] Failed to load robot from URDF: {e}", exc_info=True)
            raise

    def compute_ik(self, position, orientation_quat, seed_state=None) -> Optional[List[float]]:
        """
        Compute IK solution for given pose.

        Args:
            position: 3D position [x, y, z]
            orientation_quat: Quaternion [x, y, z, w]
            seed_state: Optional seed joint configuration

        Returns:
            List of 7 joint angles, or best-effort solution on failure
        """
        from scipy.spatial.transform import Rotation
        import pink

        logger.debug(f"[Pink IK] Computing IK for position={position}, orientation={orientation_quat}")
        start_time = time.perf_counter()

        # Update configuration from seed if provided
        if seed_state is not None:
            logger.debug(f"[Pink IK] Setting seed state (7 DOF): {seed_state}")
            # Map seed_state (7 DOF) to full configuration (18 DOF)
            full_q = self._configuration.q.copy()
            if hasattr(self, "_left_joint_dof_indices") and len(self._left_joint_dof_indices) >= len(seed_state):
                logger.debug(f"[Pink IK] Mapping to DOF indices: {self._left_joint_dof_indices}")
                for i, idx in enumerate(self._left_joint_dof_indices):
                    if i < len(seed_state):
                        full_q[idx] = seed_state[i]
                logger.debug(f"[Pink IK] After mapping, full config (18 DOF): {full_q}")
                # Update configuration using Pink's update() method
                self._configuration.update(full_q)
            else:
                logger.warning("[Pink IK] Could not map seed state to full configuration, using as-is")
                if len(full_q) >= len(seed_state):
                    full_q[: len(seed_state)] = np.array(seed_state)
                    self._configuration.update(full_q)
        else:
            logger.debug(f"[Pink IK] No seed state provided, using current config")

        # Create target transform from position + quaternion
        # Normalize quaternion to ensure unit quaternion
        orientation_quat = np.array(orientation_quat)
        norm = np.linalg.norm(orientation_quat)
        if norm > 1e-6:
            orientation_quat = orientation_quat / norm
        else:
            logger.warning(f"[Pink IK] Quaternion norm ({norm:.2e}) too small, using identity")
            orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])
        logger.debug(f"[Pink IK] Normalized quaternion: {orientation_quat}")

        r = Rotation.from_quat(orientation_quat)
        rotation_matrix = r.as_matrix()
        logger.debug(f"[Pink IK] Target rotation matrix:\n{rotation_matrix}")

        # Update FrameTask target
        target_transform = self._end_effector_task.transform_target_to_world
        target_transform.translation[:] = position
        target_transform.rotation[:] = rotation_matrix
        logger.debug(f"[Pink IK] Target transform updated")

        # Solve IK with small dt (velocity-based integration)
        dt = PINK_IK_DT
        logger.debug(f"[Pink IK] Solving with dt={dt}, solver={self._solver}")

        try:
            logger.debug(f"[Pink IK] Starting iterative IK with max iterations={PINK_MAX_ITERATIONS}")

            # Iterative IK loop
            for iteration in range(PINK_MAX_ITERATIONS):
                # Get current pose and check convergence
                current_pose = self._configuration.get_transform_frame_to_world(self.ik_link_name)
                position_error = np.array(target_transform.translation) - np.array(current_pose.translation)
                position_error_norm = np.linalg.norm(position_error)

                logger.info(
                    f"[Pink IK] Iteration {iteration + 1}/{PINK_MAX_ITERATIONS}: "
                    f"current={current_pose.translation}, target={target_transform.translation}, "
                    f"error={position_error_norm:.4f}m"
                )

                if position_error_norm < PINK_POS_TOLERANCE:
                    logger.info(f"[Pink IK] Converged at iteration {iteration + 1}, error={position_error_norm:.4f}m")
                    break

                # Compute velocity
                velocity = pink.solve_ik(self._configuration, self._tasks, dt, solver=self._solver, safety_break=False)
                max_velocity = np.max(np.abs(velocity))
                logger.debug(f"[Pink IK] Velocity solution: {velocity}")
                logger.debug(f"[Pink IK] Max velocity: {max_velocity:.6f} rad/s")

                if max_velocity < 1e-8 and position_error_norm > PINK_POS_TOLERANCE:
                    logger.warning(
                        f"[Pink IK] Velocity too small ({max_velocity:.2e} rad/s) but error is {position_error_norm:.4f}m"
                    )
                    logger.warning(
                        f"[Pink IK] Possible issues: wrong DOF indices, task costs, or collision constraints"
                    )

                # Integrate velocity
                self._configuration.integrate_inplace(velocity, dt)
                logger.debug(f"[Pink IK] Configuration after integration: {self._configuration.q}")

            # Final position check
            current_pose = self._configuration.get_transform_frame_to_world(self.ik_link_name)
            final_error = np.array(target_transform.translation) - np.array(current_pose.translation)
            final_error_norm = np.linalg.norm(final_error)
            logger.info(f"[Pink IK] Final position error: {final_error_norm:.4f}m")

            # Get joint angles and apply best-effort clamping
            full_joint_angles = self._configuration.q.copy()

            # Extract only left arm joint angles
            if hasattr(self, "_left_joint_dof_indices") and len(self._left_joint_dof_indices) > 0:
                joint_angles = np.array([full_joint_angles[i] for i in self._left_joint_dof_indices])
            else:
                joint_angles = full_joint_angles

            # Ensure joint angles are within reasonable bounds
            # (best-effort: clamp to [-π, π])
            joint_angles = np.clip(joint_angles, -PINK_JOINT_LIMIT_RANGE, PINK_JOINT_LIMIT_RANGE)

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[Pink IK] SUCCESS: computed {len(joint_angles)} joint angles in {elapsed * 1000:.2f}ms: {joint_angles}"
            )
            logger.debug(f"[Pink IK] Full configuration after IK: {full_joint_angles}")

            return joint_angles.tolist()

        except Exception as e:
            logger.error(f"[Pink IK] FAILED: {e}", exc_info=True)

            # Return best-effort current joint positions
            best_effort = self._configuration.q.copy()
            logger.warning(f"[Pink IK] Returning best-effort solution: {best_effort}")
            return best_effort.tolist()

    def compute_fk(self, joint_angles) -> Optional[Tuple[Tuple]]:
        """
        Compute forward kinematics for given joint angles.

        Args:
            joint_angles: List or array of joint positions (7 DOF)

        Returns:
            4x4 homogeneous matrix as tuple of tuples, or None on failure
        """
        start_time = time.perf_counter()

        try:
            # Map 7-DOF joint angles to full configuration (18 DOF)
            full_q = self._configuration.q.copy()
            if hasattr(self, "_left_joint_dof_indices") and len(self._left_joint_dof_indices) >= len(joint_angles):
                logger.debug(f"[Pink FK] Mapping {len(joint_angles)} DOF to indices {self._left_joint_dof_indices}")
                for i, idx in enumerate(self._left_joint_dof_indices):
                    if i < len(joint_angles):
                        full_q[idx] = joint_angles[i]
                logger.debug(f"[Pink FK] Full config after mapping: {full_q}")
                # Update configuration using Pink's update() method
                self._configuration.update(full_q)
            else:
                logger.warning("[Pink FK] Could not map joint angles to full configuration")
                if len(full_q) == len(joint_angles):
                    self._configuration.update(np.array(joint_angles))
                else:
                    raise ValueError(f"Configuration length mismatch: {len(full_q)} != {len(joint_angles)}")

            # Get transform to end-effector frame
            transform = self._configuration.get_transform_frame_to_world(self.ik_link_name)

            # Convert to 4x4 homogeneous matrix
            h_matrix = (
                (
                    float(transform.rotation[0, 0]),
                    float(transform.rotation[0, 1]),
                    float(transform.rotation[0, 2]),
                    float(transform.translation[0]),
                ),
                (
                    float(transform.rotation[1, 0]),
                    float(transform.rotation[1, 1]),
                    float(transform.rotation[1, 2]),
                    float(transform.translation[1]),
                ),
                (
                    float(transform.rotation[2, 0]),
                    float(transform.rotation[2, 1]),
                    float(transform.rotation[2, 2]),
                    float(transform.translation[2]),
                ),
                (0.0, 0.0, 0.0, 1.0),
            )

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[Pink FK] SUCCESS: computed in {elapsed * 1000:.2f}ms, position=({h_matrix[0][3]:.4f}, {h_matrix[1][3]:.4f}, {h_matrix[2][3]:.4f})"
            )

            return h_matrix

        except Exception as e:
            logger.error(f"FK computation failed: {e}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up PinkKinematics...")


class OpenArmPinkRobot(RobotWrapper):
    """
    Pink-based robot interface for OpenArm.
    Replaces OpenArmRobot with pink kinematics instead of MoveIt.
    """

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
            f"Initializing OpenArmPinkRobot with host={host}, endeff_publish_port={endeff_publish_port}, state_publish_port={state_publish_port}"
        )
        if not endeff_publish_port:
            raise ValueError("OpenArmPinkRobot requires an 'endeff_publish_port'")
        if not state_publish_port:
            raise ValueError("OpenArmPinkRobot requires a 'state_publish_port'")

        logger.info("Creating PinkKinematics...")
        urdf_path = "/home/ubuntu/workshop-robotics/src/external_dependencies/openarm_description/urdf/robot/v10.urdf"
        self._kinematics = PinkKinematics(ik_link_name="openarm_left_hand_tcp", urdf_path=urdf_path)
        logger.info("PinkKinematics created successfully")

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
            host=host, port=reset_subscribe_port, topic="reset", message_type=SessionCommand
        )

        self._home_subscriber = ZMQSubscriber(
            host=host, port=home_subscribe_port, topic="home", message_type=SessionCommand
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
        self._cartesian_tolerance = 0.001

        self._joint_angles_lock = threading.Lock()

        self._frame_rate_history = []
        self._frame_timestamps = []
        self._start_time = None
        self._last_frame_time = None
        self._ik_call_timestamps = []
        self._ik_complete_timestamps = []
        self._ik_called_history = np.zeros(1000, dtype=bool)
        self._ik_completed_history = np.zeros(1000, dtype=bool)
        self._history_index = 0

        self._handshake_coordinator = HandshakeCoordinator.get_instance()
        self._handshake_server_id = f"{self.name}_handshake"

        try:
            self._handshake_coordinator.start_server(
                subscriber_id=self._handshake_server_id,
                bind_host="*",
                port=robots.TELEOP_HANDSHAKE_PORT + 10,
            )
            logger.info(f"Handshake server started for {self.name}")
        except Exception as e:
            logger.error(f"Failed to start handshake server on port {robots.TELEOP_HANDSHAKE_PORT + 10}: {e}")
            logger.info("Attempting to continue without handshake server...")
            # Set a flag to indicate handshake is not available
            self._handshake_available = False

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
        # logger.debug(f"[Timing] _send_joint_trajectory op_id={op_id} elapsed={elapsed * 1000:.2f}ms success={success}")
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
        # logger.debug(f"[Timing] get_joint_state op_id={op_id} elapsed={elapsed * 1000:.2f}ms")
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

        # Compute FK directly (synchronous) - no caching
        h_matrix = self._kinematics.compute_fk(joint_positions)
        if h_matrix is not None:
            elapsed_total = time.perf_counter() - start
            logger.debug(f"[Timing] get_cartesian_state op_id={op_id} total={elapsed_total * 1000:.2f}ms")
            return {"cartesian_position": h_matrix, "timestamp": time.time()}
        else:
            logger.warning(f"[Timing] get_cartesian_state op_id={op_id} FK returned None")
            return None

    def get_joint_position(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        joint_positions = self._controller.get_arm_position()
        elapsed = time.perf_counter() - start
        # logger.debug(f"[Timing] get_joint_position op_id={op_id} elapsed={elapsed * 1000:.2f}ms")
        if joint_positions is None:
            return None
        return list(np.array(joint_positions, dtype=np.float32))

    def get_cartesian_position(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        joint_positions = self._controller.get_arm_position()
        elapsed1 = time.perf_counter() - start
        logger.debug(f"[Timing] get_cartesian_position op_id={op_id} get_pos={elapsed1 * 1000:.2f}ms")
        if joint_positions is None:
            logger.warning(f"[Timing] get_cartesian_position op_id={op_id} result=None (no_positions)")
            return None

        # Compute FK directly (synchronous) - no caching
        result = self._kinematics.compute_fk(joint_positions)
        elapsed_total = time.perf_counter() - start
        logger.debug(f"[Timing] get_cartesian_position op_id={op_id} total={elapsed_total * 1000:.2f}ms")
        return result

    def reset(self):
        return self._send_joint_trajectory(np.array(robots.OPENARM_HOME_JS))

    def get_teleop_state(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        result = self._arm_teleop_state_subscriber.get_arm_teleop_state()
        elapsed = time.perf_counter() - start
        # logger.debug(f"[Timing] get_teleop_state op_id={op_id} elapsed={elapsed * 1000:.2f}ms")
        return result

    def home(self):
        return self._send_joint_trajectory(np.array(robots.OPENARM_HOME_JS))

    def move(self, input_angles):
        self._send_joint_trajectory(input_angles)

    def move_coords(self, input_coords, duration=None):
        """Compute IK and send joint trajectory (synchronous)"""
        position = input_coords[:3]
        orientation = input_coords[3:7]

        # Synchronous IK call
        joint_angles = self._kinematics.compute_ik(position, orientation)

        if joint_angles is None:
            logger.warning("IK returned None, using last valid or home")
            return

        self._send_joint_trajectory(joint_angles, duration)

    def arm_control(self, cartesian_coords):
        """Compute IK and send joint trajectory (synchronous)"""
        position = cartesian_coords[:3]
        orientation = cartesian_coords[3:7]

        # Synchronous IK call
        joint_angles = self._kinematics.compute_ik(position, orientation)

        if joint_angles is None:
            logger.warning("IK returned None, using last valid angles or home")
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

        # Compute FK directly - no caching
        pose_homo = self._kinematics.compute_fk(joint_positions)

        # Publish the pose
        if pose_homo is not None:
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

        elapsed_total = time.perf_counter() - start
        logger.debug(f"[Timing] send_robot_pose op_id={op_id} total={elapsed_total * 1000:.2f}ms")

    def check_reset(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        reset_bool = self._reset_subscriber.recv_keypoints()
        elapsed = time.perf_counter() - start
        # logger.debug(f"[Timing] check_reset op_id={op_id} elapsed={elapsed * 1000:.2f}ms")
        return reset_bool is not None

    def check_home(self):
        op_id = _get_next_index()
        start = time.perf_counter()
        home_bool = self._home_subscriber.recv_keypoints()
        elapsed = time.perf_counter() - start
        # logger.debug(f"[Timing] check_home op_id={op_id} elapsed={elapsed * 1000:.2f}ms")
        if home_bool == robots.ARM_TELEOP_STOP:
            return True
        elif home_bool == robots.ARM_TELEOP_CONT:
            return False
        return False

    def stream(self):
        logger.info("*** STARTED PINK OPENARM ROBOT ***")
        self.home()

        target_interval = 1.0 / self._data_frequency
        next_frame_time = time.time()
        frame_count = 0
        self._start_time = time.time()

        while True:
            logger.info("[robot] while true")
            current_time = time.time()
            logger.debug(f"[ROBOT] Controller joint positions: {self._controller.get_arm_position()}")

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
                # logger.debug(f"[Timing] frame {frame_count} check_home={(t2 - t1) * 1000:.2f}ms")

                if home_signaled and not self._is_homed:
                    t3 = time.perf_counter()
                    self.home()
                    joint_angles = np.array(robots.OPENARM_HOME_JS)
                    with self._joint_angles_lock:
                        self._latest_joint_angles = joint_angles
                    self._is_homed = True
                    self.send_robot_pose()
                    t4 = time.perf_counter()
                    # logger.debug(f"[Timing] frame {frame_count} home_sequence={(t4 - t3) * 1000:.2f}ms")
                elif not home_signaled and self._is_homed:
                    self._is_homed = False

                t5 = time.perf_counter()
                reset_signaled = self.check_reset()
                t6 = time.perf_counter()
                # logger.debug(f"[Timing] frame {frame_count} check_reset={(t6 - t5) * 1000:.2f}ms")
                if reset_signaled:
                    self.send_robot_pose()

                t7 = time.perf_counter()
                teleop_state = self.get_teleop_state()
                t8 = time.perf_counter()
                # logger.debug(f"[Timing] frame {frame_count} get_teleop_state={(t8 - t7) * 1000:.2f}ms")
                if teleop_state == robots.ARM_TELEOP_STOP:
                    # logger.debug(f"Teleop state is STOP, skipping movement")
                    continue

                recv_start = time.perf_counter()
                msg = self._cartesian_coords_subscriber.recv_keypoints()
                recv_elapsed = time.perf_counter() - recv_start
                # logger.debug(f"[Timing] frame {frame_count} recv_keypoints={recv_elapsed * 1000:.2f}ms")
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
                    # logger.debug(f"[Timing] frame {frame_count} position_check={(t10 - t9) * 1000:.2f}ms")
                    position_changed = False
                    if self._latest_commanded_cartesian_position is None or not position_changed:
                        logger.debug("[ROBOT] Cartesian position changed, computing IK")
                        position = new_cartesian_position[:3]
                        orientation = new_cartesian_position[3:7]

                        t11 = time.perf_counter()
                        target_pos = new_cartesian_position.copy()
                        target_time = new_cartesian_timestamp

                        # Get current joint positions as seed for IK
                        seed_joints = self._controller.get_arm_position()
                        if seed_joints is not None:
                            logger.debug(f"[ROBOT] Using seed joints for IK: {seed_joints}")
                        else:
                            logger.warning(
                                "[ROBOT] Could not get seed joints from controller, IK will use current config"
                            )

                        # Synchronous IK call with seed state (if available)
                        ik_start = time.perf_counter()
                        joint_angles = self._kinematics.compute_ik(position, orientation, seed_state=seed_joints)
                        ik_elapsed = time.perf_counter() - ik_start

                        if joint_angles is not None:
                            self._ik_completed_history[self._history_index] = True
                            completion_time = time.time()
                            self._ik_complete_timestamps.append(
                                completion_time - self._start_time if self._start_time else completion_time
                            )

                            # Update joint angles
                            with self._joint_angles_lock:
                                self._latest_joint_angles = joint_angles
                                self._latest_commanded_cartesian_position = target_pos
                                self._latest_commanded_cartesian_timestamp = target_time

                            t12 = time.perf_counter() - t11
                            logger.debug(
                                f"[Timing] frame {frame_count} IK_sync={(ik_elapsed * 1000):.2f}ms update={(t12 * 1000):.2f}ms"
                            )
                            logger.debug(f"[ROBOT] IK completed: {joint_angles}")
                        else:
                            logger.warning(f"[ROBOT] IK returned None, keeping previous joint angles")
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
                # logger.debug(f"[Timing] frame {frame_count} _send_joint_trajectory={(t14 - t13) * 1000:.2f}ms")

                self.publish_current_state()
                sleep_time = max(0, next_frame_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)

                frame_count += 1

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

        logger.debug(f"[PINK] publish state {current_state_dict}")
        publish_elapsed = time.perf_counter() - publish_start
        total_elapsed = time.perf_counter() - start

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down OpenArmPinkRobot...")
        if hasattr(self, "_handshake_coordinator") and hasattr(self, "_handshake_server_id"):
            try:
                self._handshake_coordinator.stop_server(self._handshake_server_id)
                logger.info("Handshake server stopped successfully")
            except Exception as e:
                logger.warning(f"Failed to stop handshake server: {e}")
        if hasattr(self, "_kinematics"):
            self._kinematics.cleanup()
        if hasattr(self, "_controller"):
            self._controller.cleanup()
        cleanup_zmq_resources()
        logger.info("OpenArmPinkRobot shutdown complete")

    def __del__(self):
        self.shutdown()
