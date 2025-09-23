import logging
from enum import IntEnum
from typing import Generic, Optional, TypeVar

from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class TeleopState(IntEnum):
    """Arm teleoperation state using strong typing.

    Matches wire constants in robots: 0 = STOP, 1 = CONT.
    """

    STOP = robots.ARM_TELEOP_STOP
    CONT = robots.ARM_TELEOP_CONT


T = TypeVar("T")


class Ops(Generic[T]):
    """Typed teleop state reader.

    - Accepts a typed ZMQSubscriber[T]. The payload T can be a simple int
      (legacy) or a typed command dataclass (e.g., SessionCommand).
    - Ensures non-blocking access and robust fallbacks.
    """

    def __init__(
        self,
        arm_teleop_state_subscriber: ZMQSubscriber[T],
        arm_teleop_state: TeleopState = TeleopState.CONT,
    ):
        self._arm_teleop_state_subscriber = arm_teleop_state_subscriber
        self.arm_teleop_state: TeleopState = arm_teleop_state

    def get_arm_teleop_state(self) -> int:
        """Return current teleop state as int (for compatibility).

        - Tries to parse typed command messages first (objects with `command`).
        - Falls back to legacy numeric payloads.
        - Returns previous state on unknown value or errors.
        """
        if not self._arm_teleop_state_subscriber:
            return int(TeleopState.CONT)

        data: Optional[T] = self._arm_teleop_state_subscriber.recv_keypoints()
        if data is None:
            return int(self.arm_teleop_state)

        try:
            # If message has a `command` attribute (e.g., SessionCommand)
            command = getattr(data, "command", None)
            if command is not None:
                if command == robots.PAUSE:
                    return int(TeleopState.STOP)
                if command == robots.RESUME:
                    return int(TeleopState.CONT)
                return int(self.arm_teleop_state)

        except Exception as e:
            logger.error(f"Error processing arm teleop state data: {e}")
            return int(self.arm_teleop_state)
