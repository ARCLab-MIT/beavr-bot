"""Auto-generated strongly-typed config for robot `leap_pybullet`."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# Shared component configs driven by `configs.constants`
from beavr.teleop.configs.robots.shared_components import SharedComponentRegistry

# Constants (host address, port numbers, …)
from beavr.teleop.components.operators.leap_pybullet import LeapHandOperator
from beavr.teleop.interfaces.leap_robot import LeapHandRobot
from beavr.teleop.configs.robots import TeleopRobotConfig
from beavr.teleop.configs.constants import ports, network, robots
from beavr.teleop.utils.configs import (
    Laterality,
    log_laterality_configuration
)

import logging

logger = logging.getLogger(__name__)


@dataclass
class LeapHandOperatorCfg:
    host: str = network.HOST_ADDRESS
    transformed_keypoints_port: int = ports.KEYPOINT_TRANSFORM_PORT
    joint_angle_subscribe_port: int = ports.JOINT_PUBLISHER_PORT
    joint_angle_publish_port: int = ports.CARTESIAN_COMMAND_PUBLISHER_PORT
    reset_publish_port: int = ports.TELEOP_RESET_PUBLISH_PORT
    hand_side: str = robots.RIGHT  # Will be overridden based on laterality
    finger_configs: dict[str, Any] = field(
        default_factory=lambda: {
            "freeze_index": False,
            "freeze_middle": False,
            "freeze_ring": False,
            "freeze_thumb": False,
            "no_index": False,
            "no_middle": False,
            "no_ring": False,
            "no_thumb": False,
            "three_dim": True,
        }
    )
    logging_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "log_dir": "logs",
            "log_poses": True,
            "log_prefix": "leap",
        }
    )

    def __post_init__(self):
        """Validate configuration."""
        # Validate ports
        all_ports = [
            self.transformed_keypoints_port,
            self.joint_angle_subscribe_port,
            self.joint_angle_publish_port,
            self.reset_publish_port
        ]
        for port in all_ports:
            if not (1 <= port <= 65535):
                raise ValueError(f"Port out of valid range (1-65535): {port}")

    def build(self):
        return LeapHandOperator(
            host=self.host,
            transformed_keypoints_port=self.transformed_keypoints_port,
            joint_angle_subscribe_port=self.joint_angle_subscribe_port,
            joint_angle_publish_port=self.joint_angle_publish_port,
            reset_publish_port=self.reset_publish_port,
            finger_configs=self.finger_configs,
            logging_config=self.logging_config,
        )

@dataclass
class LeapHandRobotCfg:
    host: str = network.HOST_ADDRESS
    joint_angle_subscribe_port: int = ports.CARTESIAN_COMMAND_PUBLISHER_PORT
    joint_angle_publish_port: int = ports.JOINT_PUBLISHER_PORT
    reset_subscribe_port: int = ports.TELEOP_RESET_PUBLISH_PORT
    simulation_mode: bool = False
    hand_side: str = robots.RIGHT  # Will be overridden based on laterality
    state_publish_port: int = ports.LEAP_STATE_PUBLISH_PORT  # Use correct Leap state port
    recorder_config: dict[str, Any] = field(
        default_factory=lambda: {
            "robot_identifier": robots.ROBOT_IDENTIFIER_RIGHT_LEAP_HAND,  # Will be overridden
            "recorded_data": [
                robots.RECORDED_DATA_JOINT_STATES,
                robots.RECORDED_DATA_COMMANDED_JOINT_STATES,
            ]
        }
    )

    def __post_init__(self):
        """Validate configuration."""
        # Validate ports
        all_ports = [
            self.joint_angle_subscribe_port,
            self.joint_angle_publish_port,
            self.reset_subscribe_port,
            self.state_publish_port
        ]
        for port in all_ports:
            if not (1 <= port <= 65535):
                raise ValueError(f"Port out of valid range (1-65535): {port}")

    def build(self):
        return LeapHandRobot(
            host=self.host,
            joint_angle_subscribe_port=self.joint_angle_subscribe_port,
            joint_angle_publish_port=self.joint_angle_publish_port,
            reset_subscribe_port=self.reset_subscribe_port,
            state_publish_port=self.state_publish_port,
            simulation_mode=self.simulation_mode,
        )

@dataclass
@TeleopRobotConfig.register_subclass(robots.ROBOT_NAME_LEAP)
class LeapHandConfig:
    robot_name: str = robots.ROBOT_NAME_LEAP
    laterality: Laterality = Laterality.RIGHT
    
    # Configuration components - will be populated based on laterality
    detector: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    visualizers: list = field(default_factory=list)
    operators: list = field(default_factory=list)
    robots: list = field(default_factory=list)
    recorded_data: list = field(
        default_factory=lambda: [
            [
                robots.RECORDED_DATA_JOINT_STATES,
                robots.RECORDED_DATA_COMMANDED_JOINT_STATES,
            ]
        ]
    )

    def __post_init__(self):
        """Configure components based on laterality setting."""
        log_laterality_configuration(self.laterality, robots.ROBOT_NAME_LEAP)
        self._configure_for_laterality()
    
    def _configure_for_laterality(self):
        """Configure all components based on the laterality setting - explicit and simple."""
        
        # Create detector configurations
        self.detector = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.detector.append(SharedComponentRegistry.get_detector_config(
                host=network.HOST_ADDRESS,
                hand_side=robots.RIGHT
            ))
        
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.detector.append(SharedComponentRegistry.get_detector_config(
                host=network.HOST_ADDRESS,
                hand_side=robots.LEFT
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
            self.robots.append(LeapHandRobotCfg(
                host=network.HOST_ADDRESS,
                joint_angle_subscribe_port=ports.CARTESIAN_COMMAND_PUBLISHER_PORT,
                joint_angle_publish_port=ports.JOINT_PUBLISHER_PORT,
                reset_subscribe_port=ports.TELEOP_RESET_PUBLISH_PORT,
                state_publish_port=ports.LEAP_STATE_PUBLISH_PORT,
                simulation_mode=False,
                hand_side=robots.RIGHT,
                recorder_config={
                    "robot_identifier": getattr(robots, 'ROBOT_IDENTIFIER_RIGHT_LEAP_HAND', 'right_leap'),
                    "recorded_data": [
                        robots.RECORDED_DATA_JOINT_STATES,
                        robots.RECORDED_DATA_COMMANDED_JOINT_STATES,
                    ]
                }
            ))
        
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.robots.append(LeapHandRobotCfg(
                host=network.HOST_ADDRESS,
                joint_angle_subscribe_port=ports.CARTESIAN_COMMAND_PUBLISHER_PORT_LEFT,
                joint_angle_publish_port=ports.JOINT_PUBLISHER_PORT_LEFT,
                reset_subscribe_port=ports.TELEOP_RESET_PUBLISH_PORT,
                state_publish_port=ports.LEAP_STATE_PUBLISH_PORT,
                simulation_mode=False,
                hand_side=robots.LEFT,
                recorder_config={
                    "robot_identifier": getattr(robots, 'ROBOT_IDENTIFIER_LEFT_LEAP_HAND', 'left_leap'),
                    "recorded_data": [
                        robots.RECORDED_DATA_JOINT_STATES,
                        robots.RECORDED_DATA_COMMANDED_JOINT_STATES,
                    ]
                }
            ))
        
        # Create operator configurations
        self.operators = []
        if self.laterality in [Laterality.RIGHT, Laterality.BIMANUAL]:
            self.operators.append(LeapHandOperatorCfg(
                host=network.HOST_ADDRESS,
                transformed_keypoints_port=ports.KEYPOINT_TRANSFORM_PORT,
                joint_angle_subscribe_port=ports.JOINT_PUBLISHER_PORT,
                joint_angle_publish_port=ports.CARTESIAN_COMMAND_PUBLISHER_PORT,
                reset_publish_port=ports.TELEOP_RESET_PUBLISH_PORT,
                hand_side=robots.RIGHT,
                finger_configs={
                    "freeze_index": False,
                    "freeze_middle": False,
                    "freeze_ring": False,
                    "freeze_thumb": False,
                    "no_index": False,
                    "no_middle": False,
                    "no_ring": False,
                    "no_thumb": False,
                    "three_dim": True,
                },
                logging_config={
                    "enabled": False,
                    "log_dir": "logs",
                    "log_poses": True,
                    "log_prefix": "leap",
                }
            ))
        
        if self.laterality in [Laterality.LEFT, Laterality.BIMANUAL]:
            self.operators.append(LeapHandOperatorCfg(
                host=network.HOST_ADDRESS,
                transformed_keypoints_port=ports.LEFT_KEYPOINT_TRANSFORM_PORT,
                joint_angle_subscribe_port=ports.JOINT_PUBLISHER_PORT_LEFT,
                joint_angle_publish_port=ports.CARTESIAN_COMMAND_PUBLISHER_PORT_LEFT,
                reset_publish_port=ports.TELEOP_RESET_PUBLISH_PORT,
                hand_side=robots.LEFT,
                finger_configs={
                    "freeze_index": False,
                    "freeze_middle": False,
                    "freeze_ring": False,
                    "freeze_thumb": False,
                    "no_index": False,
                    "no_middle": False,
                    "no_ring": False,
                    "no_thumb": False,
                    "three_dim": True,
                },
                logging_config={
                    "enabled": False,
                    "log_dir": "logs",
                    "log_poses": True,
                    "log_prefix": "leap",
                }
            ))

    def build(self):
        """Build the robot configuration components."""
        return {
            'robot_name': self.robot_name,
            'detector': [detector.build() for detector in self.detector],
            'transforms': [item.build() for item in self.transforms],
            'visualizers': [item.build() for item in self.visualizers],
            'operators': [item.build() for item in self.operators],
            'robots': [item.build() for item in self.robots],
            'recorded_data': self.recorded_data,
        }