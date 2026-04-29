import logging
import os
from abc import ABC, abstractmethod
from multiprocessing import Process

from beavr.teleop.common.factory.instantiator import instantiate_from_target

logger = logging.getLogger(__name__)


class ProcessInstantiator(ABC):
    """
    Base class for instantiating processes.
    """

    def __init__(self, configs):
        self.configs = configs
        self.processes = []

    @abstractmethod
    def _start_component(self, configs):
        """Abstract method that must be implemented by subclasses."""
        pass

    def get_processes(self):
        return self.processes


class TeleOperator(ProcessInstantiator):
    """
    Returns all the teleoperation processes. Start the list of processes
    to run the teleop.

    Now uses only the structured MainConfig format.
    """

    def __init__(self, main_config):
        """
        Initialize TeleOperator with structured MainConfig.

        Args:
            main_config: MainConfig instance with teleop and robot sections
        """
        self.main_config = main_config
        self.teleop_config = main_config.teleop
        self.robot_config = main_config.robot
        self.processes = []

        logger.info("🔧 Initializing TeleOperator with structured configuration")

        # For Simulation environment start the environment as well
        if self.teleop_config.flags.sim_env:
            self._init_sim_environment()
        # Start the Hand Detector
        self._init_detector()
        # Start the keypoint transform
        self._init_keypoint_transform()
        self._init_visualizers()

        if self.teleop_config.flags.robot_interface:
            self._init_robot_interface()

        if self.teleop_config.flags.operate:
            self._init_operator()

    # Function to start the components
    def _start_component(self, configs):
        import traceback

        logger.info(f"Starting component: {type(configs).__name__}")
        try:
            component = instantiate_from_target(configs)
            logger.info(f"Component instantiated successfully: {type(component).__name__}")
            # Handle both single component and list of components
            if isinstance(component, list):
                for comp in component:
                    comp.stream()
            else:
                component.stream()
        except Exception as e:
            logger.error(f"Error starting component: {e}\n{traceback.format_exc()}")
            raise

    # Function to start the detector component
    def _init_detector(self):
        self.processes.append(Process(target=self._start_component, args=(self.robot_config.detector,)))

    # Function to start the sim environment
    def _init_sim_environment(self):
        for env_config in self.robot_config.environment:
            self.processes.append(Process(target=self._start_component, args=(env_config,)))

    # Function to start the keypoint transform
    def _init_keypoint_transform(self):
        for transform_config in self.robot_config.transforms:
            self.processes.append(Process(target=self._start_component, args=(transform_config,)))

    # Function to start the visualizers
    def _init_visualizers(self):
        for visualizer_config in self.robot_config.visualizers:
            self.processes.append(Process(target=self._start_component, args=(visualizer_config,)))
        # XELA visualizer
        if self.teleop_config.flags.run_xela:
            xela_visualizers = getattr(self.robot_config, "xela_visualizers", [])
            for visualizer_config in xela_visualizers:
                self.processes.append(Process(target=self._start_component, args=(visualizer_config,)))

    # Function to start the operator
    def _init_operator(self):
        for operator_config in self.robot_config.operators:
            self.processes.append(Process(target=self._start_component, args=(operator_config,)))

    def _init_robot_interface(self):
        for robot_config in self.robot_config.robots:
            # Derive a human-readable robot name from the dataclass type.
            # Instantiate the robot config in a separate process.
            # This is where the ``build()`` method is called.
            # Create the process first
            process = Process(
                target=self._start_component,
                args=(robot_config,),
                name=f"robot_interface_{type(robot_config).__name__}",
            )
            self.processes.append(process)


# Data Collector Class
class Collector(ProcessInstantiator):
    """
    Returns all the recorder processes. Start the list of processes
    to run the record data.
    """

    def __init__(self, main_config, demo_num):
        """
        Initialize Collector with structured MainConfig.

        Args:
            main_config: MainConfig instance with teleop and robot sections
            demo_num: Demonstration number for storage path
        """
        self.main_config = main_config
        self.teleop_config = main_config.teleop
        self.robot_config = main_config.robot
        self.processes = []
        self.demo_num = demo_num

        # Get storage path from config - may need to be added to config structure
        storage_path = getattr(main_config, "storage_path", "data/recordings")
        self._storage_path = os.path.join(storage_path, "demonstration_{}".format(self.demo_num))

        self._create_storage_dir()
        self._init_camera_recorders()
        # Initializing the recorders
        if self.teleop_config.flags.sim_env is True:
            self._init_sim_recorders()
        else:
            logger.info("Initialising robot recorders")
            self._init_robot_recorders()

        # Check for XELA flag - may need to be added to flags if not present
        is_xela = getattr(self.teleop_config.flags, "run_xela", False)
        if is_xela is True:
            self._init_sensor_recorders()

    def _create_storage_dir(self):
        if os.path.exists(self._storage_path):
            return
        else:
            os.makedirs(self._storage_path)

    # Function to start the components
    def _start_component(self, component):
        # Handle both single component and list of components
        if isinstance(component, list):
            for comp in component:
                comp.stream()
        else:
            component.stream()
