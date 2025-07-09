"""Auto-generated strongly-typed config for robot `template_robot`."""
from __future__ import annotations
from dataclasses import dataclass, field
from beavr.teleop.components.detector.keypoint_transform import TransformHandPositionCoords
from beavr.teleop.components.detector.oculus import OculusVRHandDetector
from beavr.teleop.components.operators.template import TemplateArmOperator
from beavr.teleop.components.visualizers.visualizer_2d import Hand2DVisualizer
from beavr.teleop.interfaces.robot import RobotWrapper
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
class TemplateArmOperatorCfg:
    host: str = '10.31.152.148'
    transformed_keypoints_port: str = '${transformed_position_keypoint_port}'
    arm_resolution_port: str = '8093'
    gripper_port: str = '8108'
    use_filter: bool = True
    cartesian_publisher_port: str = '8118'
    joint_publisher_port: str = '8119'
    cartesian_command_publisher_port: str = '8120'

    def build(self):
        return TemplateArmOperator(host=self.host, transformed_keypoints_port=self.transformed_keypoints_port, arm_resolution_port=self.arm_resolution_port, gripper_port=self.gripper_port, use_filter=self.use_filter, cartesian_publisher_port=self.cartesian_publisher_port, joint_publisher_port=self.joint_publisher_port, cartesian_command_publisher_port=self.cartesian_command_publisher_port)

@dataclass
class RobotWrapperCfg:
    record: bool = False

    def build(self):
        return RobotWrapper(record=self.record)

@dataclass
@TeleopRobotConfig.register_subclass("template_robot")
class TemplateRobotConfig:
    robot_name: str = 'robot_arm'
    detector: OculusVRHandDetectorCfg = OculusVRHandDetectorCfg(host='10.31.152.148', oculus_port='${oculus_reciever_port}', keypoint_pub_port='${keypoint_port}', button_port='8095', button_publish_port='8093', teleop_reset_port='8100', teleop_reset_publish_port='8102')
    transforms: list = field(default_factory=lambda: [TransformHandPositionCoordsCfg(host='10.31.152.148', keypoint_sub_port='${keypoint_port}', keypoint_transform_pub_port='${transformed_position_keypoint_port}', moving_average_limit=1)])
    visualizers: list = field(default_factory=lambda: [Hand2DVisualizerCfg(host='10.31.152.148', transformed_keypoint_port='${transformed_position_keypoint_port}', oculus_feedback_port='15001', display_plot='${visualize_right_2d}')])
    operators: list = field(default_factory=lambda: [TemplateArmOperatorCfg(host='10.31.152.148', transformed_keypoints_port='${transformed_position_keypoint_port}', arm_resolution_port='8093', gripper_port='8108', use_filter=True, cartesian_publisher_port='8118', joint_publisher_port='8119', cartesian_command_publisher_port='8120')])
    controllers: list = field(default_factory=lambda: [RobotWrapperCfg(record=False)])
    recorded_data: list = field(default_factory=lambda: [["joint_states", "cartesian_states"]])

    def build(self):
        return {
            'robot_name': self.robot_name,
            'detector': self.detector.build(),
            'transforms': [item.build() for item in self.transforms],
            'visualizers': [item.build() for item in self.visualizers],
            'operators': [item.build() for item in self.operators],
            'controllers': [item.build() for item in self.controllers],
            'recorded_data': self.recorded_data,
        }