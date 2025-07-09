"""Auto-generated strongly-typed config for robot `xarm_pybullet_sim`."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from beavr.teleop.components.detector.keypoint_transform import TransformHandPositionCoords
from beavr.teleop.components.detector.oculus import OculusVRHandDetector
from beavr.teleop.components.environment.xarm_env import XArmEnv
from beavr.teleop.components.operators.xarm_pybullet_sim import XArmPyBulletOperator
from beavr.teleop.components.visualizers.visualizer_2d import Hand2DVisualizer
from beavr.teleop.configs_teleop import TeleopRobotConfig


@dataclass
class OculusVRHandDetectorCfg:
    host: str = '10.31.152.148'
    oculus_port: str = '${oculus_reciever_port}'
    keypoint_pub_port: str = '${keypoint_port}'
    button_port: str = '8095'
    button_publish_port: str = '8093'
    teleop_reset_port: str = '8100'
    teleop_reset_publish_port: str = '8102'

    def build(self):
        return OculusVRHandDetector(host=self.host, oculus_port=self.oculus_port, keypoint_pub_port=self.keypoint_pub_port, button_port=self.button_port, button_publish_port=self.button_publish_port, teleop_reset_port=self.teleop_reset_port, teleop_reset_publish_port=self.teleop_reset_publish_port)

@dataclass
class TransformHandPositionCoordsCfg:
    host: str = '10.31.152.148'
    keypoint_sub_port: str = '${keypoint_port}'
    keypoint_transform_pub_port: str = '${transformed_position_keypoint_port}'
    moving_average_limit: int = 1

    def build(self):
        return TransformHandPositionCoords(host=self.host, keypoint_sub_port=self.keypoint_sub_port, keypoint_transform_pub_port=self.keypoint_transform_pub_port, moving_average_limit=self.moving_average_limit)

@dataclass
class Hand2DVisualizerCfg:
    host: str = '10.31.152.148'
    transformed_keypoint_port: str = '${transformed_position_keypoint_port}'
    oculus_feedback_port: str = '15001'
    display_plot: str = '${visualize_right_2d}'

    def build(self):
        return Hand2DVisualizer(host=self.host, transformed_keypoint_port=self.transformed_keypoint_port, oculus_feedback_port=self.oculus_feedback_port, display_plot=self.display_plot)

@dataclass
class XArmPyBulletOperatorCfg:
    host: str = '10.31.152.148'
    transformed_keypoints_port: str = '${transformed_position_keypoint_port}'
    finger_configs: dict[str, Any] = field(default_factory=lambda: {"freeze_index": False, "freeze_middle": False, "freeze_ring": False, "freeze_thumb": False, "no_index": False, "no_middle": False, "no_ring": False, "no_thumb": False, "three_dim": True})
    stream_configs: dict[str, Any] = field(default_factory=lambda: {"host": "10.31.152.148", "port": "10005"})
    stream_oculus: bool = True
    jointanglepublishport: int = 10013
    jointanglesubscribeport: int = 10012
    endeff_publish_port: int = 10010
    endeffpossubscribeport: int = 10009
    moving_average_limit: int = 1
    arm_resolution_port: str = '8093'
    use_filter: bool = False
    teleoperation_reset_port: str = '8102'
    logging_config: dict[str, Any] = field(default_factory=lambda: {"enabled": True, "log_dir": "logs", "log_poses": True, "log_prefix": "xarm"})

    def build(self):
        return XArmPyBulletOperator(host=self.host, transformed_keypoints_port=self.transformed_keypoints_port, finger_configs=self.finger_configs, stream_configs=self.stream_configs, stream_oculus=self.stream_oculus, jointanglepublishport=self.jointanglepublishport, jointanglesubscribeport=self.jointanglesubscribeport, endeff_publish_port=self.endeff_publish_port, endeffpossubscribeport=self.endeffpossubscribeport, moving_average_limit=self.moving_average_limit, arm_resolution_port=self.arm_resolution_port, use_filter=self.use_filter, teleoperation_reset_port=self.teleoperation_reset_port, logging_config=self.logging_config)

@dataclass
class XArmEnvCfg:
    host: str = '10.31.152.148'
    camport: str = '10005'
    timestamppublisherport: int = 10008
    endeff_publish_port: int = 10009
    endeffpossubscribeport: int = 10010
    robotposepublishport: int = 11111
    stream_oculus: bool = True
    sim_frequency: int = 30
    control_mode: str = 'absolute'
    logging_config: dict[str, Any] = field(default_factory=lambda: {"enabled": True, "print_joint_changes": False, "print_target_analysis": False, "print_workspace_warnings": True})

    def build(self):
        return XArmEnv(host=self.host, camport=self.camport, timestamppublisherport=self.timestamppublisherport, endeff_publish_port=self.endeff_publish_port, endeffpossubscribeport=self.endeffpossubscribeport, robotposepublishport=self.robotposepublishport, stream_oculus=self.stream_oculus, sim_frequency=self.sim_frequency, control_mode=self.control_mode, logging_config=self.logging_config)

@dataclass
@TeleopRobotConfig.register_subclass("xarm_pybullet_sim")
class XarmPybulletSimConfig:
    robot_name: str = 'xarm_pybullet'
    detector: OculusVRHandDetectorCfg = OculusVRHandDetectorCfg(host='10.31.152.148', oculus_port='${oculus_reciever_port}', keypoint_pub_port='${keypoint_port}', button_port='8095', button_publish_port='8093', teleop_reset_port='8100', teleop_reset_publish_port='8102')
    transforms: list = field(default_factory=lambda: [TransformHandPositionCoordsCfg(host='10.31.152.148', keypoint_sub_port='${keypoint_port}', keypoint_transform_pub_port='${transformed_position_keypoint_port}', moving_average_limit=1)])
    visualizers: list = field(default_factory=lambda: [Hand2DVisualizerCfg(host='10.31.152.148', transformed_keypoint_port='${transformed_position_keypoint_port}', oculus_feedback_port='15001', display_plot='${visualize_right_2d}')])
    operators: list = field(default_factory=lambda: [XArmPyBulletOperatorCfg(host='10.31.152.148', transformed_keypoints_port='${transformed_position_keypoint_port}', finger_configs={"freeze_index": False, "freeze_middle": False, "freeze_ring": False, "freeze_thumb": False, "no_index": False, "no_middle": False, "no_ring": False, "no_thumb": False, "three_dim": True}, stream_configs={"host": "10.31.152.148", "port": "10005"}, stream_oculus=True, jointanglepublishport=10013, jointanglesubscribeport=10012, endeff_publish_port=10010, endeffpossubscribeport=10009, moving_average_limit=1, arm_resolution_port='8093', use_filter=False, teleoperation_reset_port='8102', logging_config={"enabled": True, "log_dir": "logs", "log_poses": True, "log_prefix": "xarm"})])
    environment: list = field(default_factory=lambda: [XArmEnvCfg(host='10.31.152.148', camport='10005', timestamppublisherport=10008, endeff_publish_port=10009, endeffpossubscribeport=10010, robotposepublishport=11111, stream_oculus=True, sim_frequency=30, control_mode='absolute', logging_config={"enabled": True, "print_joint_changes": False, "print_target_analysis": False, "print_workspace_warnings": True})])
    recorded_data: list = field(default_factory=lambda: [["cartesian_states", "commanded_cartesian_states"]])
    port_configs: list = field(default_factory=lambda: [{"robot": "xarm_pybullet", "host": "10.31.152.148", "port": "10005", "jointanglepublishport": 10012, "jointanglesubscribeport": 10013, "timestampssubscribeport": 10008, "actualjointanglesubscribeport": 10011, "endeffpossubscribeport": 10010, "endeff_publish_port": 10009}])

    def build(self):
        return {
            'robot_name': self.robot_name,
            'detector': self.detector.build(),
            'transforms': [item.build() for item in self.transforms],
            'visualizers': [item.build() for item in self.visualizers],
            'operators': [item.build() for item in self.operators],
            'environment': [item.build() for item in self.environment],
            'recorded_data': self.recorded_data,
            'port_configs': self.port_configs,
        }