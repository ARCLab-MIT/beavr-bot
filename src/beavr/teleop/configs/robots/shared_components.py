"""Reusable component configuration dataclasses shared across robot configs.

Each class exposes sensible *defaults* that come from ``configs.constants`` so
individual robot config files no longer need to duplicate IP addresses or port
numbers.  Override any field as usual when you instantiate the dataclass.
"""

from dataclasses import dataclass
from typing import Dict, Any

from beavr.teleop.configs.constants import ports, network, robots

from beavr.teleop.components.detector.oculus import OculusVRHandDetector
from beavr.teleop.components.detector.keypoint_transform import TransformHandPositionCoords
from beavr.teleop.components.visualizers.visualizer_2d import Hand2DVisualizer

import logging

logger = logging.getLogger(__name__)


class SharedComponentRegistry:
    """
    Registry for shared VR components that should be singleton per hand side.
    
    This eliminates the need for complex deduplication logic by ensuring
    only one detector/transform/visualizer exists per hand side across all robots.
    """
    
    _instances: Dict[str, Dict[str, Any]] = {
        'detector': {},      # hand_side -> config instance
        'transform': {},     # hand_side -> config instance  
        'visualizer': {}     # hand_side -> config instance
    }

    @classmethod
    def get_detector_config(
        cls,
        hand_side: str,
        host: str = network.HOST_ADDRESS,
        oculus_pub_port: int = ports.KEYPOINT_STREAM_PORT,
        button_port: int = ports.RESOLUTION_BUTTON_PORT,
        teleop_reset_port: int = ports.TELEOP_RESET_PORT,
    ) -> 'OculusVRHandDetectorCfg':
        
        """Get or create detector config for specified hand side."""

        if hand_side not in cls._instances['detector']:
            # Set hand-side specific ports
            if hand_side == robots.LEFT:
                oculus_hand_port = ports.LEFT_HAND_OCULUS_RECEIVER_PORT
            else:  # RIGHT or default
                oculus_hand_port = ports.RIGHT_HAND_OCULUS_RECEIVER_PORT
            
            cls._instances['detector'][hand_side] = OculusVRHandDetectorCfg(
                host=host,
                oculus_hand_port=oculus_hand_port,
                oculus_pub_port=oculus_pub_port,
                button_port=button_port,
                teleop_reset_port=teleop_reset_port,
                hand_side=hand_side
            )
            logger.debug(f"📡 Created shared detector config for {hand_side} hand")
        
        return cls._instances['detector'][hand_side]
    
    @classmethod  
    def get_transform_config(
        cls, hand_side: str,
        host: str = network.HOST_ADDRESS,
        keypoint_sub_port: int = ports.KEYPOINT_STREAM_PORT,
        moving_average_limit: int = 1,
    ) -> 'TransformHandPositionCoordsCfg':
        
        """Get or create transform config for specified hand side."""

        if hand_side not in cls._instances['transform']:
            # Set hand-side specific ports
            if hand_side == robots.LEFT:
                keypoint_transform_pub_port = ports.LEFT_KEYPOINT_TRANSFORM_PORT
            else:  # RIGHT or default
                keypoint_transform_pub_port = ports.KEYPOINT_TRANSFORM_PORT
            
            cls._instances['transform'][hand_side] = TransformHandPositionCoordsCfg(
                host=host,
                keypoint_sub_port=keypoint_sub_port,
                keypoint_transform_pub_port=keypoint_transform_pub_port,
                moving_average_limit=moving_average_limit,
                hand_side=hand_side
            )
            logger.debug(f"🔄 Created shared transform config for {hand_side} hand")
        
        return cls._instances['transform'][hand_side]
    
    @classmethod
    def get_visualizer_config(
        cls, hand_side: str,
        host: str = network.HOST_ADDRESS,
        oculus_feedback_port: int = ports.OCULUS_GRAPH_PORT,
        display_plot: bool = False,
    ) -> 'Hand2DVisualizerCfg':
        
        """Get or create visualizer config for specified hand side."""

        if hand_side not in cls._instances['visualizer']:
            # Set hand-side specific ports
            if hand_side == robots.LEFT:
                transformed_keypoint_port = ports.LEFT_KEYPOINT_TRANSFORM_PORT
            else:  # RIGHT or default
                transformed_keypoint_port = ports.KEYPOINT_TRANSFORM_PORT
            
            cls._instances['visualizer'][hand_side] = Hand2DVisualizerCfg(
                host=host,
                transformed_keypoint_port=transformed_keypoint_port,
                oculus_feedback_port=oculus_feedback_port,
                display_plot=display_plot,
                hand_side=hand_side
            )
            logger.debug(f"👁️  Created shared visualizer config for {hand_side} hand")
        
        return cls._instances['visualizer'][hand_side]
    
    @classmethod
    def clear(cls):
        """Clear all cached instances. Useful for testing."""
        for component_type in cls._instances:
            cls._instances[component_type].clear()
        logger.debug("🧹 Cleared all shared component instances")
    
    @classmethod
    def get_registered_hands(cls) -> Dict[str, list]:
        """Get list of registered hand sides by component type."""
        return {
            component_type: list(instances.keys()) 
            for component_type, instances in cls._instances.items()
        }


@dataclass
class OculusVRHandDetectorCfg:
    """Configuration for :class:`beavr.components.detector.oculus.OculusVRHandDetector`."""

    host: str = network.HOST_ADDRESS
    oculus_hand_port: int = ports.RIGHT_HAND_OCULUS_RECEIVER_PORT
    oculus_pub_port: int = ports.KEYPOINT_STREAM_PORT  # alias: unified data
    button_port: int = ports.RESOLUTION_BUTTON_PORT
    teleop_reset_port: int = ports.TELEOP_RESET_PORT
    hand_side: str = robots.RIGHT  # Will be overridden based on laterality

    def __post_init__(self):
        """Validate port configuration."""
        all_ports = [self.oculus_hand_port, self.oculus_pub_port, self.button_port, self.teleop_reset_port]
        if len(set(all_ports)) != len(all_ports):
            logger.error("Duplicate ports found in OculusVRHandDetector configuration!")
            raise ValueError("Duplicate ports in configuration")
        
        # Validate port ranges
        for port_name, port_value in [
            ("oculus_hand_port", self.oculus_hand_port),
            ("oculus_pub_port", self.oculus_pub_port),
            ("button_port", self.button_port),
            ("teleop_reset_port", self.teleop_reset_port)
        ]:
            if not (1 <= port_value <= 65535):
                raise ValueError(f"{port_name} out of valid range (1-65535): {port_value}")

    def build(self):
        return OculusVRHandDetector(
            host=self.host,
            oculus_hand_port=self.oculus_hand_port,
            oculus_pub_port=self.oculus_pub_port,
            button_port=self.button_port,
            teleop_reset_port=self.teleop_reset_port,
        )

@dataclass
class TransformHandPositionCoordsCfg:
    """Right-hand keypoint transform (VR frame → robot frame)."""

    host: str = network.HOST_ADDRESS
    keypoint_sub_port: int = ports.KEYPOINT_STREAM_PORT
    keypoint_transform_pub_port: int = ports.KEYPOINT_TRANSFORM_PORT
    moving_average_limit: int = 1
    hand_side: str = robots.RIGHT

    def __post_init__(self):
        """Validate configuration."""
        if not (1 <= self.keypoint_sub_port <= 65535):
            raise ValueError(f"keypoint_sub_port out of range: {self.keypoint_sub_port}")
        if not (1 <= self.keypoint_transform_pub_port <= 65535):
            raise ValueError(f"keypoint_transform_pub_port out of range: {self.keypoint_transform_pub_port}")
        if self.moving_average_limit < 1:
            raise ValueError(f"moving_average_limit must be >= 1: {self.moving_average_limit}")

    def build(self):
        return TransformHandPositionCoords(
            host=self.host,
            keypoint_sub_port=self.keypoint_sub_port,
            keypoint_transform_pub_port=self.keypoint_transform_pub_port,
            moving_average_limit=self.moving_average_limit,
            hand_side=self.hand_side,
        )

@dataclass
class Hand2DVisualizerCfg:
    """2-D hand visualizer (Matplotlib / OpenCV)."""

    host: str = network.HOST_ADDRESS
    transformed_keypoint_port: int = ports.KEYPOINT_TRANSFORM_PORT
    oculus_feedback_port: int = ports.OCULUS_GRAPH_PORT
    display_plot: bool = False
    hand_side: str = robots.RIGHT  # Will be overridden based on laterality

    def __post_init__(self):
        """Validate port configuration."""
        for port_name, port_value in [
            ("transformed_keypoint_port", self.transformed_keypoint_port),
            ("oculus_feedback_port", self.oculus_feedback_port)
        ]:
            if not (1 <= port_value <= 65535):
                raise ValueError(f"{port_name} out of valid range: {port_value}")

    def build(self):
        return Hand2DVisualizer(
            host=self.host,
            transformed_keypoint_port=self.transformed_keypoint_port,
            oculus_feedback_port=self.oculus_feedback_port,
            display_plot=self.display_plot,
        )