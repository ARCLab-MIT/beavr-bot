import logging

from beavr.teleop.components.interface.controller.robots.openarm_gripper_control import (
    OpenArmGripperRobot,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


__all__ = ["OpenArmGripperRobot"]
