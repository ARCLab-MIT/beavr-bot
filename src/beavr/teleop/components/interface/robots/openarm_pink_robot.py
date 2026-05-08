import logging
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import pink
import pinocchio as pin
import qpsolvers
import rclpy
from pink.tasks import DampingTask, FrameTask, PostureTask
from rclpy.qos import QoSProfile, DurabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import String

from beavr.teleop.common.configs.loader import Laterality
from beavr.teleop.common.network.handshake import HandshakeCoordinator
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.ops import Ops
from beavr.teleop.components.detector.detector_types import SessionCommand
from beavr.teleop.components.interface.controller.robots.openarm_forward_control import DexArmControl
from beavr.teleop.components.interface.interface_base import RobotWrapper
from beavr.teleop.components.interface.interface_types import (
    CartesianState,
    CommandedCartesianState,
)
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def get_urdf_from_ros_topic(timeout_sec: float = 5.0) -> str:
    """Fetch URDF content from /robot_description ROS2 topic.

    Uses TRANSIENT_LOCAL durability to receive latched messages from
    robot_state_publisher.
    """
    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node("_urdf_fetcher_node")
    try:
        msg = None
        start_time = time.time()

        def callback(msg_in):
            nonlocal msg
            msg = msg_in

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sub = node.create_subscription(String, "/robot_description", callback, qos)

        while msg is None and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.1)

        if msg is None:
            raise TimeoutError(f"Timeout waiting for /robot_description topic after {timeout_sec}s")

        return msg.data
    finally:
        node.destroy_node()


# ============================================================================
# Pink Configuration Constants
# ============================================================================
# Task costs for FrameTask (end-effector positioning)
PINK_POSITION_COST = 1.0  # [cost] / [m] - aggressive positioning priority
PINK_ORIENTATION_COST = 0.5  # [cost] / [rad] - low cost to enable orientation tracking
PINK_LM_DAMPING = 0.1  # Levenberg-Marquardt damping - very low for faster convergence

# Posture task for joint regularization. Two cost levels: a near-zero default
# leaves most joints free (so the FrameTask fully owns them), while a stronger
# cost on the redundant DoF (joint3) biases the IK nullspace toward an
# "elbow-out" reference posture. Both costs stay well below PINK_POSITION_COST
# so end-effector tracking remains the dominant objective.
PINK_POSTURE_COST_DEFAULT = 1e-3  # [cost] / [rad] - effectively frees non-biased joints
PINK_POSTURE_COST_ELBOW = 0.4  # [cost] / [rad] - nullspace bias strength on joint3

# Target swivel angle for joint3 (shoulder/upper-arm roll) that pushes the
# elbow laterally outward. ~0.6 rad (~34 deg), well within the URDF limit
# (joint3 is bounded to +/- pi/2 in joint_limits.yaml). The sign depends on
# the left/right axis convention encoded in the URDF (`reflect` parameter in
# openarm_arm.xacro). If an arm's elbow ends up pointing *inward* in practice,
# flip the corresponding sign below.
PINK_ELBOW_OUT_ANGLE_RAD = 0.6
PINK_ELBOW_OUT_SIGN_LEFT = +1.0
PINK_ELBOW_OUT_SIGN_RIGHT = -1.0

# IK velocity integration time step
PINK_IK_DT = 0.01  # seconds - smaller steps for stability

# Iterative IK parameters
PINK_MAX_ITERATIONS = 3  # max IK iterations per call
PINK_POS_TOLERANCE = 0.01  # position tolerance in meters
PINK_ORIENTATION_TOLERANCE = 0.0174533  # orientation tolerance (1 degree)

# Best-effort joint limits (radians) - only used as a final clamp on the
# returned arm joint vector. Pink itself enforces the actual URDF limits via
# its ConfigurationLimit constraint inside the QP.
PINK_JOINT_LIMIT_RANGE = np.pi  # +/- π for clamping


class PinkKinematics:
    """
    Pink-based kinematics solver for OpenArm robot.
    Replaces MoveIt services with local pinocchio/pink kinematics.
    """

    def __init__(self, joint_names, ik_link_name=""):
        # Joint configuration
        self.joint_names = joint_names
        self.ik_link_name = ik_link_name
        self.num_joints = len(self.joint_names)
        # openarm_description is a CMake-based ROS2 package, not compatible with robot_descriptions
        # Use direct URDF loading instead
        self._load_robot_from_urdf()

    def _load_robot_from_urdf(self):
        try:
            t0 = time.perf_counter()

            logger.info("[Startup] Fetching URDF from /robot_description topic...")
            urdf_content = get_urdf_from_ros_topic()
            logger.info(f"[Startup] URDF content length: {len(urdf_content)} chars")
            logger.debug(f"[Startup] URDF first 500 chars: {urdf_content[:500]}")

            model = pin.buildModelFromXML(urdf_content)
            logger.info(f"[Startup] URDF kinematic model loaded in {time.perf_counter() - t0:.2f}s")
            logger.info(f"[Startup] Model: nq={model.nq}, nv={model.nv}, njoints={model.njoints}")
            logger.info(f"[Startup] Joint names: {model.names}")
            self._robot_model = model
            self._robot_data = model.createData()

            # Trust the joint position/velocity limits coming from the URDF
            # (rendered from openarm_description's joint_limits.yaml). Pink's
            # ConfigurationLimit / VelocityLimit auto-detect bounded joints by
            # checking model.{upper,lower}PositionLimit and model.velocityLimit
            # against +/-1e20, so we must not overwrite them with +/-inf if we
            # want the QP to actually enforce them.
            self._log_urdf_joint_limits()

            # Create Configuration with the full robot model
            # pink will solve IK for all joints, but we'll extract only left arm joints later
            q0 = pin.neutral(model)
            self._configuration = pink.Configuration(self._robot_model, self._robot_data, q0)

            # Build correct joint mapping for this side's arm joints:
            #   * idx_q: index in the configuration vector q (used to read/write angles)
            #   * idx_v: index in the tangent / velocity vector (used for per-joint
            #            posture-cost weighting because PostureTask jacobian is in nv).
            joint_dof_indices = []
            joint_v_indices = []
            for joint_name in self.joint_names:
                try:
                    joint_id = self._robot_model.getJointId(joint_name)
                    joint = self._robot_model.joints[joint_id]
                    joint_dof_indices.append(joint.idx_q)
                    joint_v_indices.append(joint.idx_v)
                except Exception as e:
                    logger.error(f"[PinkKinematics]   ERROR: Failed to find Pink DOF index for '{joint_name}': {e}")
                    raise

            self._joint_dof_indices = joint_dof_indices
            self._joint_v_indices = joint_v_indices

            if len(joint_dof_indices) != len(self.joint_names):
                logger.error(
                    f"[PinkKinematics] Mismatch: Expected {len(self.joint_names)} arm joints, "
                    f"but found {len(joint_dof_indices)} in model"
                )
                raise ValueError("Failed to map all arm joints to Pink model")

            # Enable both position and orientation tasks with proper quaternion handling
            # Check if the ik_link_name exists in the model
            frame_names = [f.name for f in self._robot_model.frames]
            joint_names_model = list(self._robot_model.names)
            logger.info(f"[Startup] Frame names in model: {frame_names}")
            logger.info(f"[Startup] Joint names in model: {joint_names_model}")

            missing_joints = [j for j in self.joint_names if j not in joint_names_model]
            if missing_joints:
                logger.error(
                    f"[Startup] ERROR: The following configured joint names were not found in the URDF: {missing_joints}. "
                    f"Available joints: {joint_names_model}"
                )
                raise ValueError(f"Joint names not found in URDF: {missing_joints}")

            if self.ik_link_name not in frame_names:
                logger.error(
                    f"[Startup] ERROR: ik_link_name '{self.ik_link_name}' not found in model frames! "
                    f"Available frames: {frame_names}"
                )
                raise ValueError(f"Frame '{self.ik_link_name}' not found in model")

            self._end_effector_task = FrameTask(
                self.ik_link_name,
                position_cost=PINK_POSITION_COST,
                orientation_cost=PINK_ORIENTATION_COST,
                lm_damping=PINK_LM_DAMPING,
            )

            # Posture task with per-joint cost weighting: drives only joint3
            # (the redundant DoF / elbow swivel) toward an "elbow-out" target,
            # while leaving every other joint nearly free so the FrameTask
            # remains dominant. This realises a soft nullspace bias.
            self._elbow_out_sign = self._detect_elbow_out_sign()
            posture_cost, q_ref = self._build_elbow_out_posture(self._elbow_out_sign)
            self._posture_task = PostureTask(cost=posture_cost)
            self._posture_task.set_target(q_ref)

            self._damping_task = DampingTask(
                cost=1e-3,  # [cost] / [rad/s]
            )
            self._tasks = [self._end_effector_task, self._posture_task, self._damping_task]

            # Anchor the FrameTask at the current pose so we don't generate a
            # transient at startup. The PostureTask target was already set
            # above to the elbow-out reference and must NOT be overwritten.
            self._end_effector_task.set_target_from_configuration(self._configuration)

            self._solver = qpsolvers.available_solvers[0]
            if "daqp" in qpsolvers.available_solvers:
                self._solver = "daqp"

            logger.info(
                f"[Startup] PinkKinematics initialized in {time.perf_counter() - t0:.2f}s (solver={self._solver})"
            )

        except Exception as e:
            logger.error(f"[PinkKinematics] Failed to load robot from URDF: {e}", exc_info=True)
            raise

    def _rotation_error_angle(self, R1, R2):
        R_err = R1.T @ R2
        value = (np.trace(R_err) - 1) / 2
        value = np.clip(value, -1.0, 1.0)  # numerical safety
        return np.arccos(value)

    def _log_urdf_joint_limits(self) -> None:
        """Log the position/velocity limits Pinocchio extracted from the URDF.

        Useful as a smoke test: if these come back as +/-inf, the URDF was
        rendered without `<limit>` tags (e.g. xacro args missing) and Pink's
        ConfigurationLimit will silently skip those joints.
        """
        model = self._robot_model
        for joint_name in self.joint_names:
            try:
                joint = model.joints[model.getJointId(joint_name)]
                idx_q = joint.idx_q
                idx_v = joint.idx_v
                lo = float(model.lowerPositionLimit[idx_q])
                hi = float(model.upperPositionLimit[idx_q])
                vmax = float(model.velocityLimit[idx_v])
                bounded = np.isfinite(lo) and np.isfinite(hi) and (hi - lo) > 1e-6
                logger.info(
                    f"[Startup] URDF limit {joint_name}: q in [{lo:+.4f}, {hi:+.4f}] rad, "
                    f"|qdot|<= {vmax:.3f} rad/s, bounded={bounded}"
                )
                if not bounded:
                    logger.warning(
                        f"[Startup] Joint '{joint_name}' has no usable position limits in URDF; "
                        f"Pink's ConfigurationLimit will not constrain it."
                    )
            except Exception as e:
                logger.warning(f"[Startup] Could not read URDF limits for '{joint_name}': {e}")

    def _detect_elbow_out_sign(self) -> Optional[float]:
        """Return +1/-1 depending on which arm side this kinematics serves.

        The sign defines the joint3 (shoulder/upper-arm roll) direction that
        swings the elbow laterally outward. The mapping is empirical and
        depends on the URDF axis convention for each side; if elbows go
        inward instead, flip ``PINK_ELBOW_OUT_SIGN_LEFT`` / ``..._RIGHT``.
        """
        blob = " ".join(self.joint_names).lower()
        if "left" in blob:
            return PINK_ELBOW_OUT_SIGN_LEFT
        if "right" in blob:
            return PINK_ELBOW_OUT_SIGN_RIGHT
        logger.info(
            "[Startup] Could not infer arm side from joint names; elbow-out "
            "nullspace bias disabled (joint3 will sit at neutral)."
        )
        return None

    def _build_elbow_out_posture(self, elbow_out_sign: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Build the (cost_vector, target_q) pair for the elbow-out PostureTask.

        Strategy:
          * cost_vector has length nv; default value is PINK_POSTURE_COST_DEFAULT
            (near zero, so most joints stay free for the FrameTask).
          * Only this side's joint3 receives PINK_POSTURE_COST_ELBOW.
          * target_q starts at the model's neutral configuration; only this
            side's joint3 is set to PINK_ELBOW_OUT_ANGLE_RAD * sign. All other
            entries effectively don't matter because their cost is near zero.
        """
        nv = self._robot_model.nv
        cost_vector = np.full(nv, PINK_POSTURE_COST_DEFAULT, dtype=float)
        q_ref = pin.neutral(self._robot_model).copy()

        if elbow_out_sign is None or len(self._joint_dof_indices) < 3:
            return cost_vector, q_ref

        joint3_idx_q = self._joint_dof_indices[2]
        joint3_idx_v = self._joint_v_indices[2]
        target_angle = float(elbow_out_sign) * PINK_ELBOW_OUT_ANGLE_RAD

        target_angle = self._clip_to_urdf_limits(joint3_idx_q, target_angle)

        q_ref[joint3_idx_q] = target_angle
        cost_vector[joint3_idx_v] = PINK_POSTURE_COST_ELBOW

        logger.info(
            f"[Startup] Elbow-out posture bias active: "
            f"joint3='{self.joint_names[2]}' target={target_angle:+.4f} rad, "
            f"cost={PINK_POSTURE_COST_ELBOW} (default cost on other DoFs={PINK_POSTURE_COST_DEFAULT})"
        )
        return cost_vector, q_ref

    def _clip_to_urdf_limits(self, idx_q: int, value: float) -> float:
        """Clamp a posture-target angle into the URDF's [lower, upper] range.

        Prevents the case where the chosen elbow-out target would itself lie
        outside the joint limits and cause the QP to fight its own
        ConfigurationLimit constraint.
        """
        try:
            lo = float(self._robot_model.lowerPositionLimit[idx_q])
            hi = float(self._robot_model.upperPositionLimit[idx_q])
        except Exception:
            return value
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return value
        margin = 0.05  # rad - keep a small distance from the limit
        clipped = float(np.clip(value, lo + margin, hi - margin))
        if abs(clipped - value) > 1e-6:
            logger.warning(
                f"[Startup] Elbow-out target {value:+.3f} rad clipped to "
                f"{clipped:+.3f} rad to respect URDF limits [{lo:+.3f}, {hi:+.3f}]"
            )
        return clipped

    def compute_ik(self, position: np.ndarray, orientation_quat: np.ndarray, seed_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute IK solution for given pose.

        Args:
            position: 3D position [x, y, z]
            orientation_quat: Quaternion [x, y, z, w]
            seed_state: Optional numpy array of 7 joint angles

        Returns:
            numpy array of 7 joint angles, or best-effort solution (current joint angles) on failure
        """

        start_time = time.perf_counter()
        logger.info(f"[Pink IK] Input: position={position}, quat={orientation_quat}, seed={seed_state}")

        # Update configuration from seed if provided
        if seed_state is not None:
            # Map seed_state (7 DOF) to full configuration (18 DOF)
            full_q = self._configuration.q.copy()
            if hasattr(self, "_joint_dof_indices") and len(self._joint_dof_indices) >= len(seed_state):
                for i, idx in enumerate(self._joint_dof_indices):
                    if i < len(seed_state):
                        full_q[idx] = seed_state[i]
                # Update configuration using Pink's update() method
                self._configuration.update(full_q)
            else:
                logger.warning("[Pink IK] Could not map seed state to full configuration, using as-is")
                if len(full_q) >= len(seed_state):
                    full_q[: len(seed_state)] = np.array(seed_state)
                    self._configuration.update(full_q)
        else:
            logger.debug(f"[Pink IK] No seed state provided, using current config")

        # Usually, scalar-last order (x, y, z, w) - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        r = Rotation.from_quat(orientation_quat)
        rotation_matrix = r.as_matrix()

        # Validate rotation matrix
        det = np.linalg.det(rotation_matrix)
        if abs(det - 1.0) > 0.1:
            logger.warning(
                f"[Pink IK] Invalid rotation matrix (det={det:.3f}), last element used as w={orientation_quat[3]:.3f}"
            )
            rotation_matrix = np.eye(3)
        logger.debug(f"[Pink IK] Target rotation matrix (det={det:.6f}):\n{rotation_matrix}")

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

            position_error_norm_old = None
            orientation_error_old = None
            configuration_q_old = None

            # Iterative IK loop
            for iteration in range(PINK_MAX_ITERATIONS):
                start_time_iter = time.perf_counter()

                # Get current pose and check convergence
                current_pose = self._configuration.get_transform_frame_to_world(self.ik_link_name)
                position_error = np.array(target_transform.translation) - np.array(current_pose.translation)
                position_error_norm = np.linalg.norm(position_error)
                orientation_error = self._rotation_error_angle(target_transform.rotation, current_pose.rotation)

                if iteration % 5 == 0:  # Log every 5 iterations
                    logger.debug(
                        f"[Pink IK] Iteration {iteration + 1}/{PINK_MAX_ITERATIONS}: error={position_error_norm:.4f}m"
                    )

                # Starting with the second iteration, check for convergence based on error change
                if position_error_norm_old is not None and configuration_q_old is not None:
                    # Break if the error does not change by more than 1mm (0.001m)
                    if (
                        abs(position_error_norm - position_error_norm_old) < 0.001
                        and abs(orientation_error - orientation_error_old) < 0.001
                    ):
                        # logging.getLogger("movePerf").log(logging.DEBUG, f"Converged because of no significant error change ({abs(position_error_norm - position_error_norm_old)})")
                        break

                    # If error increased, use previous configuration and break
                    if (position_error_norm_old < position_error_norm) and (orientation_error_old < orientation_error):
                        # logging.getLogger("movePerf").log(logging.DEBUG, f"Converged because of increasing error ({position_error_norm:.4f}). Using previous configuration")
                        self._configuration.update(configuration_q_old)
                        break

                if position_error_norm < PINK_POS_TOLERANCE and orientation_error < PINK_ORIENTATION_TOLERANCE:
                    logger.info(f"[Pink IK] Converged at iteration {iteration + 1}, error={position_error_norm:.4f}m")
                    break

                # Compute velocity
                velocity = pink.solve_ik(self._configuration, self._tasks, dt, solver=self._solver, safety_break=False)
                max_velocity = np.max(np.abs(velocity))

                # Log velocity details for the first iteration or if velocity is significant
                if iteration == 0 or max_velocity > 1e-4:
                    logger.info(f"[Pink IK] Velocity vector (all {len(velocity)} DOF): {velocity}")
                    logger.info(f"[Pink IK] Max velocity: {max_velocity:.6f} rad/s")

                    # Log what joints are being moved at the given arm DOF indices
                    if hasattr(self, "_joint_dof_indices"):
                        logger.info(f"[Pink IK] Arm DOF indices: {self._joint_dof_indices}")
                        for i, dof_idx in enumerate(self._joint_dof_indices):
                            if dof_idx < len(velocity):
                                vel = velocity[dof_idx]
                                joint_name = self.joint_names[i]
                                logger.info(
                                    f"[Pink IK]   Joint {i} '{joint_name}' (DOF {dof_idx}): velocity={vel:.8f} rad/s"
                                )

                if max_velocity < 1e-8 and position_error_norm > PINK_POS_TOLERANCE:
                    logger.warning(
                        f"[Pink IK] Velocity too small ({max_velocity:.2e} rad/s) but error is {position_error_norm:.4f}m"
                    )
                    logger.warning(
                        f"[Pink IK] This suggests wrong DOF indices or task conflicts. Full velocity: {velocity}"
                    )

                configuration_q_old = (
                    self._configuration.q.copy()
                )  # todo here and use the old config? maybe copy it before the intergrate_inplace??
                position_error_norm_old = position_error_norm
                orientation_error_old = orientation_error

                # Integrate velocity
                self._configuration.integrate_inplace(velocity, dt)

            # Final position check
            current_pose = self._configuration.get_transform_frame_to_world(self.ik_link_name)
            final_error = np.array(target_transform.translation) - np.array(current_pose.translation)
            final_error_norm = np.linalg.norm(final_error)
            logger.debug(f"[Pink IK] Final position error: {final_error_norm:.4f}m")

            # Get joint angles and apply best-effort clamping
            full_joint_angles = self._configuration.q.copy()

            # Extract only one side arm joint angles
            if hasattr(self, "_joint_dof_indices") and len(self._joint_dof_indices) > 0:
                joint_angles = np.array([full_joint_angles[i] for i in self._joint_dof_indices])
            else:
                joint_angles = full_joint_angles

            # Ensure joint angles are within reasonable bounds
            # (best-effort: clamp to [-π, π])
            joint_angles = np.clip(joint_angles, -PINK_JOINT_LIMIT_RANGE, PINK_JOINT_LIMIT_RANGE)

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[Pink IK] SUCCESS: computed {len(joint_angles)} joint angles in {elapsed * 1000:.2f}ms: {joint_angles}"
            )

            return joint_angles

        except Exception as e:
            logger.error(f"[Pink IK] FAILED: {e}", exc_info=True)

            # Return best-effort current joint positions
            best_effort = np.array(self._configuration.q.copy())
            logger.warning(f"[Pink IK] Returning best-effort solution: {best_effort}")
            return best_effort

    def compute_fk(self, joint_angles) -> Optional[np.ndarray]:
        """
        Compute forward kinematics for given joint angles.

        Args:
            joint_angles: List or array of joint positions (7 DOF)

        Returns:
            4x4 homogeneous transformation matrix as numpy array, or None on failure
        """
        try:
            full_q = self._configuration.q.copy()
            if hasattr(self, "_joint_dof_indices") and len(self._joint_dof_indices) >= len(joint_angles):
                for i, idx in enumerate(self._joint_dof_indices):
                    if i < len(joint_angles):
                        full_q[idx] = joint_angles[i]
                self._configuration.update(full_q)
            else:
                if len(full_q) == len(joint_angles):
                    self._configuration.update(np.array(joint_angles))
                else:
                    raise ValueError(f"Configuration length mismatch: {len(full_q)} != {len(joint_angles)}")

            transform = self._configuration.get_transform_frame_to_world(self.ik_link_name)

            h_matrix = np.eye(4)
            h_matrix[:3, :3] = transform.rotation
            h_matrix[:3, 3] = transform.translation

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
        laterality: Laterality,
        endeff_subscribe_port: int,
        reset_subscribe_port: int,
        home_subscribe_port: int,
        teleoperation_state_port: int,
        endeff_publish_port: int,
        state_publish_port: int,
        **kwargs,
    ):
        if not endeff_publish_port:
            raise ValueError("OpenArmPinkRobot requires an 'endeff_publish_port'")
        if not state_publish_port:
            raise ValueError("OpenArmPinkRobot requires a 'state_publish_port'")

        _init_t0 = time.perf_counter()

        self._laterality = laterality
        if laterality == Laterality.LEFT:
            ik_link_name = "openarm_left_hand_tcp"
            command_topic_name = "/openarm_left_arm_forward_position_controller/commands"
            joint_names = robots.OPENARM_LEFT_JOINT_NAMES
        else:  # if laterality == Laterality.RIGHT:
            ik_link_name = "openarm_right_hand_tcp"
            command_topic_name = "/openarm_right_arm_forward_position_controller/commands"
            joint_names = robots.OPENARM_RIGHT_JOINT_NAMES

        self._kinematics = PinkKinematics(joint_names=joint_names, ik_link_name=ik_link_name)
        logger.info("PinkKinematics created successfully")

        self._controller = DexArmControl(command_topic_name=command_topic_name, joint_names=joint_names)

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

        self._latest_joint_angles: Optional[np.ndarray] = None
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
                port=robots.TELEOP_HANDSHAKE_PORT + (10 if self._laterality == Laterality.LEFT else 9),  # Unique ports
            )
            logger.info(f"Handshake server started for {self.name}")
        except Exception as e:
            logger.error(
                f"Failed to start handshake server on port {robots.TELEOP_HANDSHAKE_PORT + (10 if self._laterality == Laterality.LEFT else 9)}: {e}"
            )
            logger.info("Attempting to continue without handshake server...")
            # Set a flag to indicate handshake is not available
            self._handshake_available = False

        self._is_homed = False

    def _cartesian_positions_close(self, pos1, pos2):
        """Check if two cartesian positions are close within tolerance"""
        if pos1 is None or pos2 is None:
            return False
        return np.linalg.norm(np.array(pos1) - np.array(pos2)) < self._cartesian_tolerance

    def _send_joint_positions(self, joint_angles: np.ndarray, duration: Optional[float] = None) -> bool:
        """Delegate joint positions publishing to OpenArmController"""
        success = self._controller.move_arm_joints(joint_angles, duration)
        if not success:
            logger.error("Failed to send joint positions")
        return success

    @property
    def name(self):
        if self._laterality == Laterality.LEFT:
            return robots.ROBOT_IDENTIFIER_LEFT_OPENARM
        else:  # if self._laterality == Laterality.RIGHT:
            return robots.ROBOT_IDENTIFIER_RIGHT_OPENARM

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
        joint_states = self._controller.get_arm_states()
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
        joint_positions = self._controller.get_arm_position()
        if joint_positions is None:
            return None

        h_matrix = self._kinematics.compute_fk(joint_positions)
        if h_matrix is not None:
            return {"cartesian_position": h_matrix, "timestamp": time.time()}
        return None

    def get_joint_position(self):
        joint_positions = self._controller.get_arm_position()
        if joint_positions is None:
            return None
        return list(np.array(joint_positions, dtype=np.float32))

    def get_cartesian_position(self):
        joint_positions = self._controller.get_arm_position()
        if joint_positions is None:
            return None
        return self._kinematics.compute_fk(joint_positions)

    def reset(self):
        return self._send_joint_positions(np.array(robots.OPENARM_HOME_JS))

    def get_teleop_state(self):
        return self._arm_teleop_state_subscriber.get_arm_teleop_state()

    def home(self) -> bool:
        return self._controller.home_arm()

    def move(self, input_angles):
        self._send_joint_positions(input_angles)

    def move_coords(self, input_coords, duration=None):
        """Compute IK and send joint trajectory (synchronous)"""
        position = input_coords[:3]
        orientation = input_coords[3:7]

        # Synchronous IK call
        joint_angles = self._kinematics.compute_ik(position, orientation)

        if joint_angles is None:
            logger.warning("IK returned None, using last valid or home")
            return

        self._send_joint_positions(joint_angles, duration)

    def arm_control(self, cartesian_coords):
        """Compute IK and send joint trajectory (synchronous)"""
        position = cartesian_coords[:3]
        orientation = cartesian_coords[3:7]

        # Synchronous IK call
        joint_angles = self._kinematics.compute_ik(position, orientation)

        if joint_angles is None:
            logger.warning("IK returned None, using last valid angles or home")
            return

        self._send_joint_positions(joint_angles)

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
        position = tuple(np.asarray(cartesian_state[:3, 3], dtype=np.float32).tolist())
        return CartesianState(position_m=position, timestamp_s=time.time())

    def send_robot_pose(self):
        joint_positions = self._controller.get_arm_position()
        if joint_positions is None:
            logger.warning("Could not get joint positions for robot pose")
            return

        # Compute FK directly - no caching
        pose_homo = self._kinematics.compute_fk(joint_positions)

        # Publish the pose
        if pose_homo is not None:
            try:
                h_matrix = tuple(tuple(float(x) for x in row) for row in pose_homo)

                logger.debug(
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

    def stream(self):
        logger.info("[Startup] OpenArmPinkRobot stream() starting...")
        _stream_t0 = time.perf_counter()
        self.home()
        logger.info(f"[Startup] Home command sent ({time.perf_counter() - _stream_t0:.2f}s)")

        target_interval = 1.0 / self._data_frequency
        next_frame_time = time.time()
        frame_count = 0
        self._start_time = time.time()
        iter_times = deque(maxlen=1000)

        while True:
            current_time = time.time()
            start_time_iter = time.perf_counter()

            self._history_index = frame_count % 1000

            if self._last_frame_time is not None:
                frame_time = current_time - self._last_frame_time
                frame_rate = 1.0 / frame_time if frame_time > 0 else 0.0
                self._frame_rate_history.append(frame_rate)
                self._frame_timestamps.append(current_time - self._start_time)
                if len(self._frame_rate_history) > 1000:
                    self._frame_rate_history.pop(0)
                    self._frame_timestamps.pop(0)

            self._last_frame_time = current_time

            next_frame_time = current_time + target_interval

            home_signaled = self.check_home()

            if home_signaled and not self._is_homed:
                self.home()
                joint_angles = np.array(robots.OPENARM_HOME_JS)
                with self._joint_angles_lock:
                    self._latest_joint_angles = joint_angles
                self._is_homed = True
                self.send_robot_pose()

            elif not home_signaled and self._is_homed:
                self._is_homed = False

            reset_signaled = self.check_reset()

            if reset_signaled:
                self.send_robot_pose()

            teleop_state = self.get_teleop_state()

            if teleop_state == robots.ARM_TELEOP_STOP:
                continue

            msg = self._cartesian_coords_subscriber.recv_keypoints()

            cmd = msg
            if cmd is not None:
                logger.debug(f"[ROBOT] Received cartesian command: pos={cmd.position_m}, orient={cmd.orientation_xyzw}")
                new_cartesian_position = np.concatenate(
                    [
                        np.asarray(cmd.position_m, dtype=np.float32),
                        np.asarray(cmd.orientation_xyzw, dtype=np.float32),
                    ]
                )
                new_cartesian_timestamp = cmd.timestamp_s

                logger.debug("[ROBOT] Cartesian position changed, computing IK")
                position = new_cartesian_position[:3]
                orientation = new_cartesian_position[3:7]

                target_pos = new_cartesian_position.copy()
                target_time = new_cartesian_timestamp

                # Get current joint positions as seed for IK
                seed_joints = self._controller.get_arm_position()

                # Synchronous IK call with seed state (if available)
                joint_angles = self._kinematics.compute_ik(position, orientation, seed_state=seed_joints)

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

                    self._latest_cartesian_coords = target_pos
                    self._latest_cartesian_state_timestamp = target_time

                    logger.debug(f"[ROBOT] IK completed: {joint_angles}")
                else:
                    logger.warning("[ROBOT] IK returned None, keeping previous joint angles")
            else:
                logger.debug("[ROBOT] No cartesian command received")

            if self._latest_joint_angles is not None:
                logger.debug(f"[ROBOT] Sending joint angles to controller: {self._latest_joint_angles}")
                self._send_joint_positions(self._latest_joint_angles)
            else:
                logger.debug("[ROBOT] No joint angles available to send")

            frame_count += 1

            self.publish_current_state()

            behind = np.sum(iter_times) - len(iter_times) * target_interval
            sleep_time = min(target_interval, target_interval - behind)  # Sleep at most 33ms or less if behind

            if sleep_time > 0:
                time.sleep(sleep_time)

            elapsed_iter = time.perf_counter() - start_time_iter
            iter_times.append(elapsed_iter)

    def publish_current_state(self):

        joint_states = self.get_joint_state()
        operator_cart = self.get_cartesian_state_from_operator()
        robot_cart = self.get_robot_actual_cartesian_position()
        commanded_cart = self.get_cartesian_commanded_position()
        joint_angles_rad = self.get_joint_position()

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

        self._publisher_manager.publish(
            host=self._publisher_host,
            port=self._state_publish_port,
            topic=self.name,
            data=current_state_dict,
        )

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
