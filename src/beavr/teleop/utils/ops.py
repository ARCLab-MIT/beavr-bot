import numpy as np
from beavr.teleop.configs.constants import robots
import logging
from beavr.teleop.utils.network import ZMQKeypointSubscriber

logger = logging.getLogger(__name__)

class Ops:
    """
    Class to handle operations for the robot. Checks if operation is stopped or continued.
    """
    def __init__(self, arm_teleop_state_subscriber: ZMQKeypointSubscriber, arm_teleop_state: int = robots.ARM_TELEOP_CONT):
        self._arm_teleop_state_subscriber = arm_teleop_state_subscriber
        self.arm_teleop_state = arm_teleop_state

    def get_arm_teleop_state(self) -> int:
        """
        Gets the arm operation state (STOP/CONT) from the subscriber.
        Returns:
            int: 1 if operation is stopped, 0 if operation is continued.
        """
        if not self._arm_teleop_state_subscriber:
            # Default to CONT if no subscriber, assuming continuous operation unless stopped externally
            return robots.ARM_TELEOP_CONT

        # Use NOBLOCK to avoid waiting
        data = self._arm_teleop_state_subscriber.recv_keypoints()
        if data is None:
            return self.arm_teleop_state # Return current state if no new message
        try:
            state = int(np.asanyarray(data).reshape(1)[0])
            if state in [robots.ARM_TELEOP_STOP, robots.ARM_TELEOP_CONT]:
                return state
            else:
                return self.arm_teleop_state # Return current state if unknown value
        except Exception as e:
            logger.error(f"Error processing arm teleop state data: {e}")
            return self.arm_teleop_state # Return current state on error