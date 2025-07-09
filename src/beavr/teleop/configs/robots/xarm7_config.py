"""Flexible config for XArm7 robot supporting right, left, or bimanual laterality.

This config can be used for single-arm (right/left) or dual-arm (bimanual) XArm7 setups.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# Import shared component configurations
from beavr.teleop.configs.robots.shared_components import SharedComponentRegistry

from beavr.teleop.interfaces.xarm7_robot import XArm7Robot
from beavr.teleop.configs.robots import TeleopRobotConfig

# Import shared constants and utilities
from beavr.teleop.configs.constants import network, ports, robots
from beavr.teleop.utils.configs import (
    Laterality,
    log_laterality_configuration
)

import logging

logger = logging.getLogger(__name__)


@dataclass
class XArm7RobotCfg:
    host: str = network.HOST_ADDRESS
    robot_ip: str = network.RIGHT_XARM_IP
    is_right_arm: bool = True  # Will be overridden based on laterality
    endeff_publish_port: int = ports.XARM_ENDEFF_PUBLISH_PORT
    endeff_subscribe_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT
    joint_subscribe_port: int = ports.XARM_JOINT_SUBSCRIBE_PORT
    reset_subscribe_port: int = ports.XARM_RESET_SUBSCRIBE_PORT
    state_publish_port: int = ports.XARM_STATE_PUBLISH_PORT
    home_subscribe_port: int = ports.XARM_HOME_SUBSCRIBE_PORT
    teleoperation_state_port: int = ports.XARM_TELEOPERATION_STATE_PORT
    hand_side: str = robots.RIGHT  # Will be overridden based on laterality
    recorder_config: dict[str, Any] = field(
        default_factory=lambda: {
            "robot_identifier": robots.ROBOT_IDENTIFIER_RIGHT_XARM7,
            "recorded_data": [
                robots.RECORDED_DATA_JOINT_STATES,
                robots.RECORDED_DATA_XARM_CARTESIAN_STATES,
                robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE,
                robots.RECORDED_DATA_JOINT_ANGLES_RAD
            ]
        }
    )

    def __post_init__(self):
        """Validate configuration."""
        # Validate all ports are in valid range
        all_ports = [
            self.endeff_publish_port,
            self.endeff_subscribe_port, 
            self.joint_subscribe_port,
            self.reset_subscribe_port,
            self.state_publish_port,
            self.home_subscribe_port,
            self.teleoperation_state_port
        ]
        for port in all_ports:
            if not (1 <= port <= 65535):
                raise ValueError(f"Port out of valid range (1-65535): {port}")
        
        # Validate IP address format
        ip_parts = self.robot_ip.split(".")
        if len(ip_parts) != 4:
            raise ValueError(f"Invalid IP address format: {self.robot_ip}")

    def build(self):
        return XArm7Robot(
            host=self.host,
            robot_ip=self.robot_ip,
            is_right_arm=self.is_right_arm,
            endeff_publish_port=self.endeff_publish_port,
            endeff_subscribe_port=self.endeff_subscribe_port,
            joint_subscribe_port=self.joint_subscribe_port,
            reset_subscribe_port=self.reset_subscribe_port,
            state_publish_port=self.state_publish_port,
            home_subscribe_port=self.home_subscribe_port,
            recorder_config=self.recorder_config,
            teleoperation_state_port=self.teleoperation_state_port
        )


@dataclass
class XArm7OperatorCfg:
    host: str = network.HOST_ADDRESS
    transformed_keypoints_port: int = ports.KEYPOINT_TRANSFORM_PORT
    stream_configs: dict[str, Any] = field(
        default_factory=lambda: {
            "host": network.HOST_ADDRESS, 
            "port": ports.CONTROL_STREAM_PORT
        }
    )
    stream_oculus: bool = True
    endeff_publish_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT
    endeff_subscribe_port: int = ports.XARM_ENDEFF_PUBLISH_PORT
    moving_average_limit: int = 1
    arm_resolution_port: int = ports.KEYPOINT_STREAM_PORT
    use_filter: bool = False
    teleoperation_state_port: int = ports.XARM_TELEOPERATION_STATE_PORT
    logging_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False, 
            "log_dir": "logs", 
            "log_poses": True, 
            "log_prefix": "xarm"
        }
    )
    hand_side: str = robots.RIGHT  # Will be overridden based on laterality

    def __post_init__(self):
        """Validate configuration."""
        # Validate ports
        all_ports = [
            self.transformed_keypoints_port,
            self.endeff_publish_port,
            self.endeff_subscribe_port,
            self.arm_resolution_port,
            self.teleoperation_state_port
        ]
        for port in all_ports:
            if not (1 <= port <= 65535):
                raise ValueError(f"Port out of valid range (1-65535): {port}")
        
        if self.moving_average_limit < 1:
            raise ValueError(f"moving_average_limit must be >= 1: {self.moving_average_limit}")

    def build(self):
        # Import here to avoid circular imports
        from beavr.teleop.components.operators.xarm7_right import XArm7RightOperator
        
        return XArm7RightOperator(
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
            logging_config=self.logging_config
        )


@dataclass
@TeleopRobotConfig.register_subclass(robots.ROBOT_NAME_XARM7)
class XArm7Config:
    robot_name: str = robots.ROBOT_NAME_XARM7
    laterality: Laterality = Laterality.RIGHT
    
    # Configuration components - will be populated based on laterality
    detector: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    visualizers: list = field(default_factory=list)
    robots: list = field(default_factory=list)
    operators: list = field(default_factory=list)

    def __post_init__(self):
        """Configure components based on laterality setting."""
        log_laterality_configuration(self.laterality, robots.ROBOT_NAME_XARM7)
        self._configure_for_laterality()
    
    def _configure_for_laterality(self):
        """Configure all components based on the laterality setting - explicit and simple."""
        
        # Create detector configurations
        self.detector = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.detector.append(SharedComponentRegistry.get_detector_config(
                hand_side=robots.RIGHT,
                host=network.HOST_ADDRESS,
            ))
        
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.detector.append(SharedComponentRegistry.get_detector_config(
                hand_side=robots.LEFT,
                host=network.HOST_ADDRESS,
            ))
        
        # Create transform configurations
        self.transforms = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.transforms.append(SharedComponentRegistry.get_transform_config(
                hand_side=robots.RIGHT,
                host=network.HOST_ADDRESS,
                keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                moving_average_limit=1,
            ))
        
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.transforms.append(SharedComponentRegistry.get_transform_config(
                hand_side=robots.LEFT,
                host=network.HOST_ADDRESS,
                keypoint_sub_port=ports.KEYPOINT_STREAM_PORT,
                moving_average_limit=1,
            ))
        
        # Create visualizer configurations
        self.visualizers = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.visualizers.append(SharedComponentRegistry.get_visualizer_config(
                hand_side=robots.RIGHT,
                host=network.HOST_ADDRESS,
                oculus_feedback_port=ports.OCULUS_GRAPH_PORT,
                display_plot=False,
            ))
        
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.visualizers.append(SharedComponentRegistry.get_visualizer_config(
                hand_side=robots.LEFT,
                host=network.HOST_ADDRESS,
                oculus_feedback_port=ports.OCULUS_GRAPH_PORT,
                display_plot=False,
            ))
        
        # Create robot configurations
        self.robots = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.robots.append(XArm7RobotCfg(
                host=network.HOST_ADDRESS,
                robot_ip=network.RIGHT_XARM_IP,
                is_right_arm=True,
                endeff_publish_port=ports.XARM_ENDEFF_PUBLISH_PORT,
                endeff_subscribe_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT,
                joint_subscribe_port=ports.XARM_JOINT_SUBSCRIBE_PORT,
                reset_subscribe_port=ports.XARM_RESET_SUBSCRIBE_PORT,
                state_publish_port=ports.XARM_STATE_PUBLISH_PORT,
                home_subscribe_port=ports.XARM_HOME_SUBSCRIBE_PORT,
                teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                hand_side=robots.RIGHT,
                recorder_config={
                    "robot_identifier": robots.ROBOT_IDENTIFIER_RIGHT_XARM7,
                    "recorded_data": [
                        robots.RECORDED_DATA_JOINT_STATES,
                        robots.RECORDED_DATA_XARM_CARTESIAN_STATES,
                        robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE,
                        robots.RECORDED_DATA_JOINT_ANGLES_RAD
                    ]
                }
            ))
        
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.robots.append(XArm7RobotCfg(
                host=network.HOST_ADDRESS,
                robot_ip=network.LEFT_XARM_IP,
                is_right_arm=False,
                endeff_publish_port=ports.XARM_ENDEFF_PUBLISH_PORT + 2,  # Different ports for left
                endeff_subscribe_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + 2,
                joint_subscribe_port=ports.XARM_JOINT_SUBSCRIBE_PORT + 1,
                reset_subscribe_port=ports.XARM_RESET_SUBSCRIBE_PORT + 2,
                state_publish_port=ports.XARM_STATE_PUBLISH_PORT + 1,
                home_subscribe_port=ports.XARM_HOME_SUBSCRIBE_PORT,
                teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                hand_side=robots.LEFT,
                recorder_config={
                    "robot_identifier": robots.ROBOT_IDENTIFIER_LEFT_XARM7,
                    "recorded_data": [
                        robots.RECORDED_DATA_JOINT_STATES,
                        robots.RECORDED_DATA_XARM_CARTESIAN_STATES,
                        robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE,
                        robots.RECORDED_DATA_JOINT_ANGLES_RAD
                    ]
                }
            ))
        
        # Create operator configurations
        self.operators = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.operators.append(XArm7OperatorCfg(
                host=network.HOST_ADDRESS,
                transformed_keypoints_port=ports.KEYPOINT_TRANSFORM_PORT,
                stream_configs={
                    "host": network.HOST_ADDRESS, 
                    "port": ports.CONTROL_STREAM_PORT
                },
                stream_oculus=True,
                endeff_publish_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT,
                endeff_subscribe_port=ports.XARM_ENDEFF_PUBLISH_PORT,
                moving_average_limit=1,
                arm_resolution_port=ports.KEYPOINT_STREAM_PORT,
                use_filter=False,
                teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                hand_side=robots.RIGHT,
                logging_config={
                    "enabled": False, 
                    "log_dir": "logs", 
                    "log_poses": True, 
                    "log_prefix": "xarm_right"
                }
            ))
        
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.operators.append(XArm7OperatorCfg(
                host=network.HOST_ADDRESS,
                transformed_keypoints_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                stream_configs={
                    "host": network.HOST_ADDRESS, 
                    "port": ports.CONTROL_STREAM_PORT
                },
                stream_oculus=True,
                endeff_publish_port=ports.XARM_ENDEFF_SUBSCRIBE_PORT + 2,
                endeff_subscribe_port=ports.XARM_ENDEFF_PUBLISH_PORT + 2,
                moving_average_limit=1,
                arm_resolution_port=ports.KEYPOINT_STREAM_PORT,
                use_filter=False,
                teleoperation_state_port=ports.XARM_TELEOPERATION_STATE_PORT,
                hand_side=robots.LEFT,
                logging_config={
                    "enabled": False, 
                    "log_dir": "logs", 
                    "log_poses": True, 
                    "log_prefix": "xarm_left"
                }
            ))

    def build(self):
        """Build the robot configuration components."""
        return {
            'robot_name': self.robot_name,
            'detector': [detector.build() for detector in self.detector],
            'transforms': [item.build() for item in self.transforms],
            'visualizers': [item.build() for item in self.visualizers],
            'robots': [item.build() for item in self.robots],
            'operators': [item.build() for item in self.operators],
        }