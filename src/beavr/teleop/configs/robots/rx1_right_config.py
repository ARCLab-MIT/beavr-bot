"""Auto-generated strongly-typed config for robot `rx1_right`."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from beavr.teleop.components.detector.keypoint_transform import TransformHandPositionCoords
from beavr.teleop.components.detector.oculus import OculusVRHandDetector
from beavr.teleop.components.operators.rx1_right_operator import RX1RightOperator
from beavr.teleop.components.visualizers.visualizer_2d import Hand2DVisualizer
from beavr.teleop.interfaces.rx1.rx1_right_robot import RX1Right
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
class RX1RightCfg:
    host: str = '10.31.152.148'
    ee_pose_op_pub: int = 10010
    ee_pose_op_sub: int = 10009
    ee_pose_ros_pub: int = 9118
    reset_op_sub: int = 9069
    ee_pose_ros_sub: int = 5555
    joint_state_ros_sub: int = 5556

    def build(self):
        return RX1Right(host=self.host, ee_pose_op_pub=self.ee_pose_op_pub, ee_pose_op_sub=self.ee_pose_op_sub, ee_pose_ros_pub=self.ee_pose_ros_pub, reset_op_sub=self.reset_op_sub, ee_pose_ros_sub=self.ee_pose_ros_sub, joint_state_ros_sub=self.joint_state_ros_sub)

@dataclass
class RX1RightOperatorCfg:
    host: str = '10.31.152.148'
    transformed_keypoints_port: str = '${transformed_position_keypoint_port}'
    stream_configs: dict[str, Any] = field(default_factory=lambda: {"host": "10.31.152.148", "port": "10005"})
    stream_oculus: bool = True
    endeff_publish_port: int = 10009
    endeff_subscribe_port: int = 10010
    moving_average_limit: int = 1
    arm_resolution_port: str = '8093'
    use_filter: bool = False
    teleoperation_reset_port: str = '8102'
    reset_publish_port: int = 9069
    logging_config: dict[str, Any] = field(default_factory=lambda: {"enabled": False, "log_dir": "logs", "log_poses": True, "log_prefix": "rx1"})

    def build(self):
        return RX1RightOperator(host=self.host, transformed_keypoints_port=self.transformed_keypoints_port, stream_configs=self.stream_configs, stream_oculus=self.stream_oculus, endeff_publish_port=self.endeff_publish_port, endeff_subscribe_port=self.endeff_subscribe_port, moving_average_limit=self.moving_average_limit, arm_resolution_port=self.arm_resolution_port, use_filter=self.use_filter, teleoperation_reset_port=self.teleoperation_reset_port, reset_publish_port=self.reset_publish_port, logging_config=self.logging_config)

@dataclass
@TeleopRobotConfig.register_subclass("rx1_right")
class Rx1RightConfig:
    robot_name: str = 'rx1_right'
    detector: OculusVRHandDetectorCfg = OculusVRHandDetectorCfg(host='10.31.152.148', oculus_port='${oculus_reciever_port}', keypoint_pub_port='${keypoint_port}', button_port='8095', button_publish_port='8093', teleop_reset_port='8100', teleop_reset_publish_port='8102')
    transforms: list = field(default_factory=lambda: [TransformHandPositionCoordsCfg(host='10.31.152.148', keypoint_sub_port='${keypoint_port}', keypoint_transform_pub_port='${transformed_position_keypoint_port}', moving_average_limit=1)])
    visualizers: list = field(default_factory=lambda: [Hand2DVisualizerCfg(host='10.31.152.148', transformed_keypoint_port='${transformed_position_keypoint_port}', oculus_feedback_port='15001', display_plot='${visualize_right_2d}')])
    robots: list = field(default_factory=lambda: [RX1RightCfg(host='10.31.152.148', ee_pose_op_pub=10010, ee_pose_op_sub=10009, ee_pose_ros_pub=9118, reset_op_sub=9069, ee_pose_ros_sub=5555, joint_state_ros_sub=5556)])
    operators: list = field(default_factory=lambda: [RX1RightOperatorCfg(host='10.31.152.148', transformed_keypoints_port='${transformed_position_keypoint_port}', stream_configs={"host": "10.31.152.148", "port": "10005"}, stream_oculus=True, endeff_publish_port=10009, endeff_subscribe_port=10010, moving_average_limit=1, arm_resolution_port='8093', use_filter=False, teleoperation_reset_port='8102', reset_publish_port=9069, logging_config={"enabled": False, "log_dir": "logs", "log_poses": True, "log_prefix": "rx1"})])
    recorded_data: list = field(default_factory=lambda: [["joint_states", "cartesian_states", "position", "velocity", "effort"]])

    def build(self):
        return {
            'robot_name': self.robot_name,
            'detector': self.detector.build(),
            'transforms': [item.build() for item in self.transforms],
            'visualizers': [item.build() for item in self.visualizers],
            'robots': [item.build() for item in self.robots],
            'operators': [item.build() for item in self.operators],
            'recorded_data': self.recorded_data,
        }