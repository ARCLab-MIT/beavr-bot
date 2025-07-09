import os
from abc import ABC
from .recorders.image import RGBImageRecorder, DepthImageRecorder, FishEyeImageRecorder
from .recorders.robot_state import RobotInformationRecord
from .recorders.sim_state import SimInformationRecord
from .recorders.sensors import XelaSensorRecorder
from .sensors import RealsenseCamera, FishEyeCamera
from multiprocessing import Process
from beavr.teleop.configs.constants import cameras
from omegaconf import ListConfig # Import ListConfig for type checking
from beavr.teleop.utils.instantiator import instantiate_from_target

import logging

logger = logging.getLogger(__name__)


class ProcessInstantiator(ABC):
    def __init__(self, configs):
        self.configs = configs
        self.processes = []

    def _start_component(self,configs):
        raise NotImplementedError(f"Function not implemented! {configs}")

    def get_processes(self):
        return self.processes


class RealsenseCameras(ProcessInstantiator):
    """
    Returns all the camera processes. Start the list of processes to start
    the camera stream.
    """
    def __init__(self, configs):
        super().__init__(configs)
        # Creating all the camera processes
        self._init_camera_processes()

    def _start_component(self, cam_idx):
        component = RealsenseCamera(
            stream_configs = dict(
                host = self.configs.teleop.network.host_address,
                port = self.configs.teleop.ports.cam_port_offset + cam_idx
            ),
            cam_serial_num = self.configs.teleop.camera.robot_cam_serial_numbers[cam_idx],
            cam_id = cam_idx + 1,
            cam_configs = {
                'width': self.configs.teleop.camera.cam_configs_width,
                'height': self.configs.teleop.camera.cam_configs_height,
                'fps': self.configs.teleop.camera.cam_configs_fps,
                'processing_preset': self.configs.teleop.camera.cam_configs_processing_preset,
                'rotation_angle': self.configs.teleop.camera.cam_configs_rotation_angle,
            },
            stream_oculus = True if self.configs.teleop.camera.oculus_cam == cam_idx else False
        )
        component.stream()

    def _init_camera_processes(self):
        for cam_idx in range(len(self.configs.teleop.camera.robot_cam_serial_numbers)):
            self.processes.append(Process(
                target = self._start_component,
                args = (cam_idx, )
            ))

class FishEyeCameras(ProcessInstantiator):
    """
    Returns all the fish eye camera processes. Start the list of processes to start
    the camera stream.
    """
    def __init__(self, configs):
        super().__init__(configs)
        # Creating all the camera processes
        self._init_camera_processes()

    def _start_component(self, cam_idx):
        # Get fisheye camera numbers from config, default to empty list if not configured
        fisheye_cam_numbers = getattr(self.configs.teleop.camera, 'fisheye_cam_numbers', [])
        
        logger.info('cam_idx: {}, stream_oculus: {}'.format(cam_idx, True if self.configs.teleop.camera.oculus_cam == cam_idx else False))
        component = FishEyeCamera(
            cam_index=fisheye_cam_numbers[cam_idx],
            stream_configs = dict(
                host = self.configs.teleop.network.host_address,
                port = self.configs.teleop.ports.fish_eye_cam_port_offset + cam_idx,
                set_port_offset = self.configs.teleop.ports.fish_eye_cam_port_offset 
            ),
            
            stream_oculus = True if self.configs.teleop.flags.stream_oculus and self.configs.teleop.camera.oculus_cam == cam_idx else False,
            
        )
        component.stream()

    def _init_camera_processes(self):
        fisheye_cam_numbers = getattr(self.configs.teleop.camera, 'fisheye_cam_numbers', [])
        for cam_idx in range(len(fisheye_cam_numbers)):
            self.processes.append(Process(
                target = self._start_component,
                args = (cam_idx, )
            ))


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
        try:
            component = instantiate_from_target(configs)
            # Handle both single component and list of components
            if isinstance(component, list):
                for comp in component:
                    comp.stream()
            else:
                component.stream()
        except Exception as e:
            logger.error(f"Error starting component: {e}")
            raise

    # Function to start the detector component
    def _init_detector(self):
        self.processes.append(Process(
            target = self._start_component,
            args = (self.robot_config.detector, )
        ))

    # Function to start the sim environment
    def _init_sim_environment(self):
         for env_config in self.robot_config.environment:
            self.processes.append(Process(
                target = self._start_component,
                args = (env_config, )
            ))

    # Function to start the keypoint transform
    def _init_keypoint_transform(self):
        for transform_config in self.robot_config.transforms:
            self.processes.append(Process(
                target = self._start_component,
                args = (transform_config, )
            ))

    # Function to start the visualizers
    def _init_visualizers(self):
        for visualizer_config in self.robot_config.visualizers:
            self.processes.append(Process(
                target = self._start_component,
                args = (visualizer_config, )
            ))
        # XELA visualizer
        if self.teleop_config.flags.run_xela:
            xela_visualizers = getattr(self.robot_config, 'xela_visualizers', [])
            for visualizer_config in xela_visualizers:
                self.processes.append(Process(
                    target = self._start_component,
                    args = (visualizer_config, )
                ))

    # Function to start the operator
    def _init_operator(self):
        for operator_config in self.robot_config.operators:
            self.processes.append(Process(
                target = self._start_component,
                args = (operator_config, )
            ))
    
    def _init_robot_interface(self):
        for robot_config in self.robot_config.robots:
            # Derive a human-readable robot name from the dataclass type.
            # Instantiate the robot config in a separate process.
            # This is where the ``build()`` method is called.
            # Create the process first
            process = Process(
                target = self._start_component,
                args = (robot_config, )
            )
            self.processes.append(process)


# Helper function to be the target of the recorder process
# This makes error handling within the specific recorder process easier
def _start_robot_recorder_component(
    host: str,
    port: int,
    topic: str,
    record_key: str,
    storage_path: str,
    robot_name: str):
    """Instantiates and runs RobotInformationRecord in a separate process."""
    try:
        logger.info(f"Starting RobotInformationRecord process for {robot_name} - Key: '{record_key}', Topic: '{topic}', Port: {port}")
        component = RobotInformationRecord(
            host=host,
            port=port,
            topic=topic,
            record_key=record_key,
            storage_path=storage_path,
            robot_name=robot_name # Used for filename/logging within recorder
        )
        component.stream()
    except Exception as e:
        logger.error(f"FATAL ERROR in RobotInformationRecord process for {robot_name}-{record_key} (Topic: {topic}): {e}")
        # Consider more robust logging here
        raise # Re-raise the exception so the process terminates clearly


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
        storage_path = getattr(main_config, 'storage_path', 'data/recordings')
        self._storage_path = os.path.join(
            storage_path, 
            'demonstration_{}'.format(self.demo_num)
        )
       
        self._create_storage_dir()
        self._init_camera_recorders()
        # Initializing the recorders
        if self.teleop_config.flags.sim_env is True:
            self._init_sim_recorders()
        else:
            logger.info("Initialising robot recorders")
            self._init_robot_recorders()
        
        # Check for XELA flag - may need to be added to flags if not present
        is_xela = getattr(self.teleop_config.flags, 'run_xela', False)
        if is_xela is True:
            self._init_sensor_recorders()

    def _create_storage_dir(self):
        if os.path.exists(self._storage_path):
            return 
        else:
            os.makedirs(self._storage_path)

    #Function to start the components
    def _start_component(self, component):
        # Handle both single component and list of components
        if isinstance(component, list):
            for comp in component:
                comp.stream()
        else:
            component.stream()

    # Record the rgb components
    def _start_rgb_component(self, cam_idx=0):
        # This part has been isolated and made different for the sim and real robot
        # If using simulation and real robot on the same network, only one of them will stream into the VR. Close the real robot realsense camera stream before launching simulation.
        if self.teleop_config.flags.sim_env is False:
            logger.info("RGB function")
            component = RGBImageRecorder(
                host = self.teleop_config.network.host_address,
                image_stream_port = self.teleop_config.ports.cam_port_offset + cam_idx,
                storage_path = self._storage_path,
                filename = 'cam_{}_rgb_video'.format(cam_idx)
            )
        else:
            logger.info("Reaching correct function")
            component = RGBImageRecorder(
                host = self.teleop_config.network.host_address,
                image_stream_port = self.teleop_config.ports.sim_image_port + cam_idx,
                storage_path = self._storage_path,
                filename = 'cam_{}_rgb_video'.format(cam_idx),
                sim = True
            )
        component.stream()

    # Record the depth components
    def _start_depth_component(self, cam_idx):
        if self.teleop_config.flags.sim_env is not True:
            component = DepthImageRecorder(
                host = self.teleop_config.network.host_address,
                image_stream_port = self.teleop_config.ports.cam_port_offset + cam_idx + cameras.DEPTH_PORT_OFFSET,
                storage_path = self._storage_path,
                filename = 'cam_{}_depth'.format(cam_idx)
            )
        else:
            component = DepthImageRecorder(
                host = self.teleop_config.network.host_address,
                image_stream_port = self.teleop_config.ports.sim_image_port + cam_idx + cameras.DEPTH_PORT_OFFSET,
                storage_path = self._storage_path,
                filename = 'cam_{}_depth'.format(cam_idx)
            )
        component.stream()

    #Function to start the camera recorders
    def _init_camera_recorders(self):
        if self.teleop_config.flags.sim_env is not True:
            logger.info("Camera recorder starting")
            for cam_idx in range(len(self.teleop_config.camera.robot_cam_serial_numbers)):
                self.processes.append(Process(
                    target = self._start_rgb_component,
                    args = (cam_idx, )
                ))

                self.processes.append(Process(
                    target = self._start_depth_component,
                    args = (cam_idx, )
                ))
        else:
            for cam_idx in range(self.teleop_config.camera.num_cams):
                self.processes.append(Process(
                    target = self._start_rgb_component,
                    args = (cam_idx, )
                ))

                self.processes.append(Process(
                    target = self._start_depth_component,
                    args = (cam_idx, )
                ))

    #Function to start the sim recorders
    def _init_sim_recorders(self):
        port_configs = self.robot_config.port_configs
        for key in self.robot_config.recorded_data[0]:
            self.processes.append(Process(
                        target = self._start_sim_component,
                        args = (port_configs[0],key)))

    #Function to start the xela sensor recorders
    def _start_xela_component(self,
        controller_config
    ):
        component = XelaSensorRecorder(
            controller_configs=controller_config,
            storage_path=self._storage_path
        )
        component.stream()

    #Function to start the sensor recorders
    def _init_sensor_recorders(self):
        """
        For the XELA sensors or any other sensors
        """
        xela_controllers = getattr(self.robot_config, 'xela_controllers', [])
        for controller_config in xela_controllers:
            self.processes.append(Process(
                target = self._start_xela_component,
                args = (controller_config, )
            ))

    #Function to start the fish eye recorders
    def _start_fish_eye_component(self, cam_idx):
        component = FishEyeImageRecorder(
            host = self.teleop_config.network.host_address,
            image_stream_port = self.teleop_config.ports.fish_eye_cam_port_offset + cam_idx,
            storage_path = self._storage_path,
            filename = 'cam_{}_fish_eye_video'.format(cam_idx)
        )
        component.stream()

    #Function to start the robot recorders
    def _init_robot_recorders(self):
        """
        Initializes RobotInformationRecord processes based on configuration
        found within each robot's definition in `robot_config.robots`.

        Expects each robot config block to have:
        - state_publish_port: The port the robot publishes its state dict on.
        - recorder_config:
            - robot_identifier: A name for recorder filenames/logs.
            - recorded_data: A list of keys to record from the state dict.
        - Information to derive the ZMQ topic (e.g., '_target_', 'is_right_arm').
        """
        # Safety checks for config structure
        if not hasattr(self.robot_config, 'robots'):
            logger.warning("'robot_config.robots' not found in configuration. Skipping robot recorder initialization.")
            return
        if not isinstance(self.robot_config.robots, (list, ListConfig)):
             logger.warning("'robot_config.robots' is not a list. Skipping robot recorder initialization.")
             return
        if not hasattr(self, '_storage_path') or not self._storage_path:
            logger.error("FATAL ERROR: Collector._storage_path is not set! Cannot initialize robot recorders.")
            return

        host_address = self.teleop_config.network.host_address

        logger.info("Initializing Robot Recorders...")
        # Iterate through each robot defined in the configuration
        for robot_index, robot_config in enumerate(self.robot_config.robots):
            try:
                # --- 1. Extract Configuration ---
                if not hasattr(robot_config, 'state_publish_port'):
                    logger.error(f"  ERROR: Robot config at index {robot_index} missing 'state_publish_port'. Skipping.")
                    continue
                state_publish_port = robot_config.state_publish_port

                if not hasattr(robot_config, 'recorder_config'):
                    logger.error(f"  ERROR: Robot config for port {state_publish_port} missing 'recorder_config'. Skipping.")
                    continue
                recorder_cfg = robot_config.recorder_config

                if not hasattr(recorder_cfg, 'robot_identifier'):
                    logger.error(f"  ERROR: Robot config for port {state_publish_port} missing 'recorder_config.robot_identifier'. Skipping.")
                    continue
                robot_identifier = recorder_cfg.robot_identifier # For filenames

                if not hasattr(recorder_cfg, 'recorded_data'):
                    logger.error(f"  ERROR: Robot config for '{robot_identifier}' missing 'recorder_config.recorded_data'. Skipping.")
                    continue
                recorded_data_keys = recorder_cfg.recorded_data
                if not isinstance(recorded_data_keys, (list, ListConfig)):
                    logger.error(f"  ERROR: 'recorded_data' for robot '{robot_identifier}' is not a list. Skipping.")
                    continue

                # --- 2. Determine Expected ZMQ Topic ---
                # This logic needs to match how the robot interface determines its topic.
                expected_zmq_topic = None
                # Determine robot type from the dataclass class name.
                target_class = robot_config.__class__.__name__

                if target_class.endswith("XArm7Robot"):
                    is_right = getattr(robot_config, 'is_right_arm', None)
                    if is_right is True:
                        expected_zmq_topic = "right_xarm7"
                    elif is_right is False:
                        expected_zmq_topic = "left_xarm7"
                    else:
                        logger.error(f"  ERROR: XArm7Robot config for '{robot_identifier}' missing or invalid 'is_right_arm'. Cannot determine topic.")
                        continue # Skip this robot
                # Add elif blocks here for other robot types and their topic logic
                # elif target_class.endswith("LeapHandRobot"):
                #     expected_zmq_topic = getattr(recorder_cfg, 'expected_zmq_topic', None) # Example: Read explicit topic
                else:
                    logger.warning(f"  Warning: Unknown robot type '{target_class}' for '{robot_identifier}'. Cannot determine ZMQ topic automatically.")
                    # Optionally try reading an explicit 'expected_zmq_topic' from recorder_cfg
                    expected_zmq_topic = getattr(recorder_cfg, 'expected_zmq_topic', None)
                    if not expected_zmq_topic:
                        logger.error(f"  ERROR: Could not determine ZMQ topic for '{robot_identifier}'. Skipping recorders.")
                        continue # Skip this robot

                # --- 3. Create Recorder Processes ---
                logger.info(f"  Configuring recorders for robot: '{robot_identifier}' (Topic: '{expected_zmq_topic}', Port: {state_publish_port})")
                for key in recorded_data_keys:
                    if not isinstance(key, str) or not key:
                        logger.warning(f"    Warning: Invalid record key '{key}' found for '{robot_identifier}'. Skipping.")
                        continue

                    logger.info(f"    - Creating recorder process for key: '{key}'")
                    process = Process(
                        target=_start_robot_recorder_component, # Use the helper function
                        args=(
                            host_address,          # host
                            state_publish_port,    # port
                            expected_zmq_topic,    # topic
                            key,                   # record_key
                            self._storage_path,    # storage_path
                            robot_identifier       # robot_name (for recorder's internal use)
                        )
                    )
                    self.processes.append(process)

            except Exception as e:
                logger.error(f"  ERROR: Unexpected error processing robot config for '{robot_identifier}' (Index {robot_index}): {e}. Skipping.")
                continue

    #Function to start the sim recorders
    def _start_sim_component(self,port_configs, recorder_function_key):
        component = SimInformationRecord(
                   port_configs = port_configs,
                   recorder_function_key= recorder_function_key,
                   storage_path=self._storage_path
        )
        component.stream()