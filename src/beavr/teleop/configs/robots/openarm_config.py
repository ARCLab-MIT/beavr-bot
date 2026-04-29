"""Configuration for OpenArm robot (ROS2-based 7-DOF left arm).

This config is for the left arm only, controlled via ROS2:
- Action: /left_joint_trajectory_controller/follow_joint_trajectory
- Service: /compute_ik for inverse kinematics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from beavr.teleop.common.configs.loader import Laterality, log_laterality_configuration
from beavr.teleop.components.interface.controller.robots.openarm_gripper_control import OpenArmGripperRobot
from beavr.teleop.components.interface.robots.openarm_robot import OpenArmRobot
from beavr.teleop.components.interface.robots.openarm_pink_robot import OpenArmPinkRobot
from beavr.teleop.configs.constants import network, ports, robots
from beavr.teleop.configs.robots import TeleopRobotConfig
from beavr.teleop.configs.robots.shared_components import SharedComponentRegistry

logger = logging.getLogger(__name__)


@dataclass
class OpenArmRobotCfg:
    host: str = network.HOST_ADDRESS
    laterality: Laterality = Laterality.LEFT
    endeff_publish_port: int = ports.OPENARM_ENDEFF_PUBLISH_PORT
    endeff_subscribe_port: int = ports.OPENARM_ENDEFF_SUBSCRIBE_PORT
    reset_subscribe_port: int = ports.OPENARM_RESET_SUBSCRIBE_PORT
    home_subscribe_port: int = ports.OPENARM_HOME_SUBSCRIBE_PORT
    state_publish_port: int = ports.OPENARM_STATE_PUBLISH_PORT
    teleoperation_state_port: int = ports.KEYPOINT_STREAM_PORT
    recorder_config: dict[str, Any] = field(
        default_factory=lambda: {}
    )

    def __post_init__(self):
        all_ports = [
            self.endeff_publish_port,
            self.endeff_subscribe_port,
            self.reset_subscribe_port,
            self.home_subscribe_port,
            self.state_publish_port,
            self.teleoperation_state_port,
        ]
        for port in all_ports:
            if not (1 <= port <= 65535):
                raise ValueError(f"Port out of valid range (1-65535): {port}")

    def build(self):
        return OpenArmPinkRobot(
            host=self.host,
            laterality=self.laterality,
            endeff_publish_port=self.endeff_publish_port,
            endeff_subscribe_port=self.endeff_subscribe_port,
            reset_subscribe_port=self.reset_subscribe_port,
            home_subscribe_port=self.home_subscribe_port,
            state_publish_port=self.state_publish_port,
            teleoperation_state_port=self.teleoperation_state_port,
        )


@dataclass
class OpenArmOperatorCfg:
    host: str = network.HOST_ADDRESS
    laterality: Laterality = Laterality.LEFT
    transformed_keypoints_port: int = ports.LEFT_KEYPOINT_TRANSFORM_PORT
    stream_configs: dict[str, Any] = field(
        default_factory=lambda: {
            "host": network.HOST_ADDRESS,
            "port": ports.CONTROL_STREAM_PORT,
        }
    )
    stream_oculus: bool = True
    endeff_publish_port: int = ports.OPENARM_ENDEFF_SUBSCRIBE_PORT
    endeff_subscribe_port: int = ports.OPENARM_ENDEFF_PUBLISH_PORT
    moving_average_limit: int = 3
    arm_resolution_port: int = ports.KEYPOINT_STREAM_PORT
    use_filter: bool = False
    teleoperation_state_port: int = ports.KEYPOINT_STREAM_PORT
    logging_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "log_dir": "logs",
            "log_poses": True,
            "log_prefix": "openarm_left",
        }
    )

    def __post_init__(self):
        all_ports = [
            self.transformed_keypoints_port,
            self.endeff_publish_port,
            self.endeff_subscribe_port,
            self.arm_resolution_port,
            self.teleoperation_state_port,
        ]
        for port in all_ports:
            if not (1 <= port <= 65535):
                raise ValueError(f"Port out of valid range (1-65535): {port}")

        if self.moving_average_limit < 1:
            raise ValueError(f"moving_average_limit must be >= 1: {self.moving_average_limit}")

    def build(self):
        if self.laterality == Laterality.LEFT:
            from beavr.teleop.components.operator.robots.openarm_left_operator import (
                OpenArmLeftOperator,
            )

            return OpenArmLeftOperator(
                host=self.host,
                transformed_keypoints_port=self.transformed_keypoints_port,
                stream_configs=self.stream_configs,
                stream_oculus=self.stream_oculus,
                endeff_publish_port=self.endeff_publish_port,
                endeff_subscribe_port=self.endeff_subscribe_port,
                moving_average_limit=self.moving_average_limit,
                arm_resolution_port=self.arm_resolution_port,
                use_filter=self.use_filter,
                teleoperation_state_port=self.teleoperation_state_port,
                logging_config=self.logging_config,
            )
        else: # if self.laterality == Laterality.RIGHT
            from beavr.teleop.components.operator.robots.openarm_right_operator import (
                OpenArmRightOperator,
            )

            return OpenArmRightOperator(
                host=self.host,
                transformed_keypoints_port=self.transformed_keypoints_port,
                stream_configs=self.stream_configs,
                stream_oculus=self.stream_oculus,
                endeff_publish_port=self.endeff_publish_port,
                endeff_subscribe_port=self.endeff_subscribe_port,
                moving_average_limit=self.moving_average_limit,
                arm_resolution_port=self.arm_resolution_port,
                use_filter=self.use_filter,
                teleoperation_state_port=self.teleoperation_state_port,
                logging_config=self.logging_config,
            )



@dataclass
class OpenArmGripperRobotCfg:
    host: str = network.HOST_ADDRESS
    laterality: Laterality = Laterality.LEFT
    gripper_subscribe_port: int = robots.OPENARM_GRIPPER_SUBSCRIBE_PORT
    recorder_config: dict[str, Any] = field(
        default_factory=lambda: {
            "robot_identifier": robots.ROBOT_IDENTIFIER_OPENARM_GRIPPER,
            "recorded_data": [],
        }
    )

    def __post_init__(self):
        if not (1 <= self.gripper_subscribe_port <= 65535):
            raise ValueError(f"Port out of valid range (1-65535): {self.gripper_subscribe_port}")

    def build(self):
        if self.laterality == Laterality.LEFT:
            gripper_action_name = "/openarm_left_gripper_controller/gripper_cmd"
        else: #if laterality == Laterality.RIGHT:
            gripper_action_name = "/openarm_right_gripper_controller/gripper_cmd"

        return OpenArmGripperRobot(
            host=self.host,
            gripper_action_name=gripper_action_name,
            gripper_subscribe_port=self.gripper_subscribe_port,
        )


@dataclass
@TeleopRobotConfig.register_subclass(robots.ROBOT_NAME_OPENARM)
class OpenArmConfig:
    robot_name: str = robots.ROBOT_NAME_OPENARM
    laterality: Laterality = Laterality.LEFT

    detector: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    visualizers: list = field(default_factory=list)
    robots: list = field(default_factory=list)
    operators: list = field(default_factory=list)

    def __post_init__(self):
        log_laterality_configuration(self.laterality, robots.ROBOT_NAME_OPENARM)
        self._configure_for_laterality()

    def _configure_for_laterality(self):
        self.detector = []
        self.detector.append(
            SharedComponentRegistry.get_detector_config(
                hand_side=self.laterality.value,
                host=network.HOST_ADDRESS,
            )
        )

        self.transforms = []
        self.transforms.append(
            SharedComponentRegistry.get_transform_config(
                hand_side=self.laterality.value,
                host=network.HOST_ADDRESS,
                keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                moving_average_limit=3,
            )
        )

        self.visualizers = []

        if self.laterality == Laterality.LEFT:
            identifier = robots.ROBOT_IDENTIFIER_LEFT_OPENARM
            log_prefix = "openarm_left"
            transform_port = ports.LEFT_KEYPOINT_TRANSFORM_PORT
        else: # if self.laterality == Laterality.RIGHT:
            identifier = robots.ROBOT_IDENTIFIER_RIGHT_OPENARM
            log_prefix = "openarm_right"
            transform_port = ports.KEYPOINT_TRANSFORM_PORT

        self.robots = []
        self.robots.append(
            OpenArmRobotCfg(
                host=network.HOST_ADDRESS,
                laterality=self.laterality,
                endeff_publish_port=ports.OPENARM_ENDEFF_PUBLISH_PORT,
                endeff_subscribe_port=ports.OPENARM_ENDEFF_SUBSCRIBE_PORT,
                reset_subscribe_port=ports.OPENARM_RESET_SUBSCRIBE_PORT,
                home_subscribe_port=ports.OPENARM_HOME_SUBSCRIBE_PORT,
                state_publish_port=ports.OPENARM_STATE_PUBLISH_PORT,
                teleoperation_state_port=ports.OPENARM_TELEOPERATION_STATE_PORT,
                recorder_config={
                    "robot_identifier": identifier,
                    "recorded_data": [
                        robots.RECORDED_DATA_JOINT_STATES,
                        robots.RECORDED_DATA_CARTESIAN_STATES,
                        robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE,
                        robots.RECORDED_DATA_JOINT_ANGLES_RAD,
                    ],
                },
            )
        )
        self.robots.append(
            OpenArmGripperRobotCfg(
                host=network.HOST_ADDRESS,
                laterality=self.laterality,
                gripper_subscribe_port=robots.OPENARM_GRIPPER_SUBSCRIBE_PORT,
                recorder_config={
                    "robot_identifier": robots.ROBOT_IDENTIFIER_OPENARM_GRIPPER,
                    "recorded_data": [],
                },
            )
        )

        self.operators = []
        self.operators.append(
            OpenArmOperatorCfg(
                host=network.HOST_ADDRESS,
                transformed_keypoints_port=transform_port,
                stream_configs={
                    "host": network.HOST_ADDRESS,
                    "port": ports.CONTROL_STREAM_PORT,
                },
                laterality=self.laterality,
                stream_oculus=True,
                endeff_publish_port=ports.OPENARM_ENDEFF_SUBSCRIBE_PORT,
                endeff_subscribe_port=ports.OPENARM_ENDEFF_PUBLISH_PORT,
                moving_average_limit=3,
                arm_resolution_port=ports.KEYPOINT_STREAM_PORT,
                use_filter=False,
                teleoperation_state_port=ports.OPENARM_TELEOPERATION_STATE_PORT,
                logging_config={
                    "enabled": False,
                    "log_dir": "logs",
                    "log_poses": True,
                    "log_prefix": log_prefix,
                },
            )
        )

    def build(self):
        return {
            "robot_name": self.robot_name,
            "detector": [detector.build() for detector in self.detector],
            "transforms": [item.build() for item in self.transforms],
            "visualizers": [item.build() for item in self.visualizers],
            "robots": [item.build() for item in self.robots],
            "operators": [item.build() for item in self.operators],
        }
