import abc
import draccus


class TeleopRobotConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base class for teleoperation *combo* robot configurations.

    Any dataclass inheriting from this class can be registered with the
    ``@TeleopRobotConfig.register_subclass("my_name")`` decorator and later
    instantiated from that short name via
    ``TeleopRobotConfig.get_choice_class("my_name")``.
    """

    pass