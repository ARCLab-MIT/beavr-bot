import logging
import time
import threading

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation

from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class OpenArmKinematics:
    def __init__(
        self,
        ik_service_name: str = "/compute_ik",
        fk_service_name: str = "/compute_fk",
        joint_names = None,
        ik_group_name: str = robots.OPENARM_IK_GROUP_NAME,
        ik_frame_id: str = robots.OPENARM_IK_FRAME_ID,
        ik_link_name: str = robots.OPENARM_IK_LINK_NAME,
    ):
        self.joint_names = joint_names or robots.OPENARM_LEFT_JOINT_NAMES
        self.ik_group_name = ik_group_name
        self.ik_frame_id = ik_frame_id
        self.ik_link_name = ik_link_name
        self.num_joints = len(self.joint_names)

        self._initialize_ros2()

        self._ik_client = self._node.create_client(
            GetPositionIK,
            ik_service_name,
        )

        self._fk_client = self._node.create_client(
            GetPositionFK,
            fk_service_name,
            callback_group=ReentrantCallbackGroup(),
        )

        logger.info(f"Waiting for IK service: {ik_service_name}")
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            logger.error("IK service not available!")
        else:
            logger.info("Connected to IK service")

    def _initialize_ros2(self):
        logger.info("Starting ROS2 initialization for kinematics...")
        if not rclpy.ok():
            logger.info("rclpy not initialized, calling rclpy.init()")
            try:
                rclpy.init()
                logger.info("rclpy.init() successful")
            except Exception as e:
                logger.error(f"Failed to initialize rclpy: {e}")
                raise

        logger.info("Creating ROS2 node: openarm_kinematics_node")
        try:
            self._node = Node("openarm_kinematics_node")
        except Exception as e:
            logger.error(f"Failed to create ROS2 node: {e}")
            raise

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)

        logger.info("Starting ROS2 executor thread")
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        logger.info("ROS2 node initialized: openarm_kinematics_node")

    def compute_ik_callback(self, position, orientation_quat, callback, seed_state=None):
        if not self._ik_client.service_is_ready():
            logger.error("IK service is not ready")
            return None
        logger.info(f"position {position}")
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.ik_group_name
        request.ik_request.pose_stamped.header.frame_id = self.ik_frame_id
        request.ik_request.pose_stamped.header.stamp = self._node.get_clock().now().to_msg()
        request.ik_request.ik_link_name = self.ik_link_name

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(orientation_quat[0])
        pose.orientation.y = float(orientation_quat[1])
        pose.orientation.z = float(orientation_quat[2])
        pose.orientation.w = float(orientation_quat[3])
        request.ik_request.pose_stamped.pose = pose

        request.ik_request.timeout.sec = 1

        if seed_state is not None:
            robot_state = RobotState()
            robot_state.joint_state.name = self.joint_names
            robot_state.joint_state.position = [float(x) for x in seed_state]
            robot_state.joint_state.velocity = [0.0] * self.num_joints
            request.ik_request.robot_state = robot_state
        else:
            logger.warning("No seed state provided, IK will use default seed")

        future = self._ik_client.call_async(request)
        future.add_done_callback(callback)
        logger.debug("Submitted IK request with callback")
        return future

    def compute_ik(self, position, orientation_quat, seed_state=None):
        if not self._ik_client.service_is_ready():
            logger.error("IK service is not ready")
            return None
        logger.info(f"position {position}")
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.ik_group_name
        request.ik_request.pose_stamped.header.frame_id = self.ik_frame_id
        request.ik_request.pose_stamped.header.stamp = self._node.get_clock().now().to_msg()
        request.ik_request.ik_link_name = self.ik_link_name

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(orientation_quat[0])
        pose.orientation.y = float(orientation_quat[1])
        pose.orientation.z = float(orientation_quat[2])
        pose.orientation.w = float(orientation_quat[3])
        request.ik_request.pose_stamped.pose = pose

        request.ik_request.timeout.sec = 1

        if seed_state is not None:
            robot_state = RobotState()
            robot_state.joint_state.name = self.joint_names
            robot_state.joint_state.position = [float(x) for x in seed_state]
            robot_state.joint_state.velocity = [0.0] * self.num_joints
            request.ik_request.robot_state = robot_state
        else:
            logger.warning("No seed state provided, IK will use default seed")

        future = self._ik_client.call_async(request)

        timeout_sec = 2.0
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > timeout_sec:
                logger.error("IK service call timed out")
                return None
            time.sleep(0.01)

        try:
            response = future.result()
            if response.error_code.val == 1:
                joint_positions = [
                    response.solution.joint_state.position[i] for i in range(self.num_joints)
                ]
                logger.debug(f"IK solution found: {joint_positions}")
                return joint_positions
            else:
                logger.error(f"IK failed with error code: {response.error_code.val}")
                return None
        except Exception as e:
            logger.error(f"Exception during IK call: {e}")
            return None

    def compute_fk_callback(self, joint_angles, callback):
        """Compute forward kinematics asynchronously with callback"""
        if not self._fk_client.service_is_ready():
            logger.error("FK service is not ready")
            return None

        request = GetPositionFK.Request()
        request.fk_link_names = [self.ik_link_name]
        request.header.frame_id = self.ik_frame_id
        request.header.stamp = self._node.get_clock().now().to_msg()

        request.robot_state = RobotState()
        request.robot_state.joint_state.name = self.joint_names
        request.robot_state.joint_state.position = [float(x) for x in joint_angles]
        request.robot_state.joint_state.velocity = [0.0] * self.num_joints

        future = self._fk_client.call_async(request)
        future.add_done_callback(callback)
        logger.debug("Submitted FK request with callback")
        return future

    def compute_fk(self, joint_angles):
        """Compute the forward kinematics (4x4 homogeneous matrix) for given joint angles (blocking)"""
        start = time.perf_counter()

        if not self._fk_client.service_is_ready():
            logger.error("FK service is not ready")
            return None

        request = GetPositionFK.Request()
        request.fk_link_names = [self.ik_link_name]
        request.header.frame_id = self.ik_frame_id
        request.header.stamp = self._node.get_clock().now().to_msg()

        request.robot_state = RobotState()
        request.robot_state.joint_state.name = self.joint_names
        request.robot_state.joint_state.position = [float(x) for x in joint_angles]
        request.robot_state.joint_state.velocity = [0.0] * self.num_joints

        future = self._fk_client.call_async(request)

        timeout_sec = 2.0
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > timeout_sec:
                logger.error("FK service call timed out")
                return None
            time.sleep(0.01)

        elapsed_call = time.perf_counter() - start

        try:
            response = future.result()
            if response.error_code.val == 1:
                if len(response.pose_stamped) > 0:
                    pose_stamped = response.pose_stamped[0]
                    position = pose_stamped.pose.position
                    orientation = pose_stamped.pose.orientation

                    r = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
                    rotation_matrix = r.as_matrix()

                    h_matrix = (
                        (float(rotation_matrix[0,0]), float(rotation_matrix[0,1]), float(rotation_matrix[0,2]), float(position.x)),
                        (float(rotation_matrix[1,0]), float(rotation_matrix[1,1]), float(rotation_matrix[1,2]), float(position.y)),
                        (float(rotation_matrix[2,0]), float(rotation_matrix[2,1]), float(rotation_matrix[2,2]), float(position.z)),
                        (0.0, 0.0, 0.0, 1.0),
                    )

                    elapsed_total = time.perf_counter() - start
                    logger.debug(f"[Timing] compute_fk_call_blocking call={elapsed_call*1000:.2f}ms format={(elapsed_total-elapsed_call)*1000:.2f}ms total={elapsed_total*1000:.2f}ms")
                    logger.info(f"compute_fk result - position=[{position.x:.4f}, {position.y:.4f}, {position.z:.4f}]")
                    return h_matrix
                else:
                    logger.error("FK returned no poses")
                    return None
            else:
                logger.error(f"FK failed with error code: {response.error_code.val}")
                return None
        except Exception as e:
            logger.error(f"Exception during FK call: {e}")
            return None

    def cleanup(self):
        logger.info("Cleaning up OpenArm kinematics...")
        if hasattr(self, "_ik_client"):
            self._ik_client.destroy()
        if hasattr(self, "_fk_client"):
            self._fk_client.destroy()
        if hasattr(self, "_executor"):
            self._executor.shutdown()
        if hasattr(self, "_node"):
            self._node.destroy_node()
