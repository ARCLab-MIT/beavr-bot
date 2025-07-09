from beavr.teleop.controllers.leap_control import DexArmControl
from beavr.teleop.interfaces.robot import RobotWrapper
import numpy as np
import threading
import time
from queue import Queue
from beavr.teleop.utils.ops import Ops

from beavr.teleop.utils.network import (
    ZMQKeypointSubscriber,
    ZMQPublisherManager,
    cleanup_zmq_resources,
)
from beavr.teleop.utils.registry import GlobalRegistry
from beavr.teleop.configs.constants import robots
import logging

logger = logging.getLogger(__name__)


class LeapHandRobot(RobotWrapper):
    def __init__(
        self,
        host,
        joint_angle_subscribe_port,
        joint_angle_publish_port,
        reset_subscribe_port,
        state_publish_port=10008,
        home_subscribe_port=10007,
        simulation_mode=False,
        is_right_arm=True,
        teleoperation_state_port=8089,
        **kwargs,
    ):
        """Initialize the robot adapter for data acquisition.
        
        Args:
            host: Network host address
            joint_angle_subscribe_port: Port for joint angle command subscription
            joint_angle_publish_port: Port for publishing joint angles
            reset_subscribe_port: Port for reset subscription
            state_publish_port: Port for publishing the full state dictionary for recording
            simulation_mode: Whether to run in simulation mode
            **kwargs: Additional configuration parameters from Hydra
        """
        # Ensure required ports are present
        if not state_publish_port:
            raise ValueError("LeapHandRobot requires a 'state_publish_port'")

        # Set arm side first (needed for name property)
        self._is_right_arm = is_right_arm
        self._is_homed = False
        
        # Store recorder config if provided
        self.recorder_config = kwargs.get('recorder_config', {})
        self.robot_identifier = self.recorder_config.get('robot_identifier', self.name)
        
        if simulation_mode:
            self._controller = None  # Skip hardware initialization in sim mode
        else:
            self._controller = DexArmControl()  # Only init hardware in real mode
        
        self._data_frequency = robots.VR_FREQ
        self._command_queue = Queue(maxsize=1)
        self._last_command = None
        self._movement_thread = threading.Thread(target=self._execute_movement_loop, daemon=True)
        self._movement_thread.start()

        logger.info(f"LeapHandRobot '{self.name}' initialized with command frequency: {self._data_frequency}Hz")
        logger.info(f"  Publishing state dictionary on tcp://{host}:{state_publish_port} with topic '{self.name}'")

        # Subscribers
        self._joint_angle_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = joint_angle_subscribe_port,
            topic = 'joint_angles'
        )
        self._reset_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = reset_subscribe_port,
            topic = 'reset'
        )

        self._home_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = home_subscribe_port,
            topic = 'home'
        )

        self._subscribers = {
            'joint_angles': self._joint_angle_subscriber,
            'reset': self._reset_subscriber,
            'home': self._home_subscriber
        }

        # Ops state subscriber --------------------------------------------------------
        # Checks if operation is stopped or continued.
        self._arm_teleop_state_subscriber = Ops(
            arm_teleop_state_subscriber=ZMQKeypointSubscriber(
                host=host,
                port=teleoperation_state_port,
                topic='pause',
            )
        )

        # Centralized publisher manager (new pub/sub structure)
        self._publisher_manager = ZMQPublisherManager.get_instance()
        self._publisher_host = host
        self._joint_angle_publish_port = joint_angle_publish_port
        self._state_publish_port = state_publish_port

        # Pre-bind sockets so that subscribers can connect early
        self._publisher_manager.get_publisher(self._publisher_host, self._joint_angle_publish_port)
        self._publisher_manager.get_publisher(self._publisher_host, self._state_publish_port)

        # Add recording control
        self._is_recording_enabled = False
        
        # Add caches for data collection
        self._latest_joint_angles = None
        self._latest_joint_angles_timestamp = 0
        self._latest_commanded_angles = None
        self._latest_commanded_angles_timestamp = 0
        
        # Reference to the operator will be set when it's available
        self._operator = None

    @property
    def name(self):
        return f"{robots.ROBOT_NAME_LEAP}_{'right' if self._is_right_arm else 'left'}"

    @property
    def recorder_functions(self):
        """Define available recording functions for the robot"""
        return {
            'joint_states': self.get_joint_state, 
            'commanded_joint_states': self.get_commanded_joint_position,
            'transformed_keypoints': self.get_transformed_keypoints,
            'joint_angles_rad': self.get_joint_position,
        }

    @property
    def data_frequency(self):
        return self._data_frequency

    # Recording control methods
    def start_recording(self):
        """Enable data recording"""
        self._is_recording_enabled = True
        logger.info(f"Recording started for {self.name}")

    def stop_recording(self):
        """Disable data recording"""
        self._is_recording_enabled = False
        logger.info(f"Recording stopped for {self.name}")

    def is_recording(self):
        """Check if recording is enabled"""
        return self._is_recording_enabled

    # State information functions
    def get_joint_state(self):
        """Get current joint state"""
        return {
            'position': np.array(self._state, dtype=np.float32),
            'timestamp': time.time()
        }

    def get_commanded_joint_state(self):
        return self._state

    def get_joint_position(self):
        return self._controller.get_hand_position()

    def get_joint_velocity(self):
        return self._controller.get_hand_velocity()

    def get_joint_torque(self):
        return self._controller.get_hand_torque()

    def get_commanded_joint_position(self):
        """Get current commanded action"""
        if self._action is None:
            return self.get_joint_state()
        return {
            'position': np.array(self._action, dtype=np.float32),
            'timestamp': time.time()
        }

    # Movement functions
    def home(self):
        self._controller.home_hand()

    def move_coords(self, coords):
        """Required by RobotWrapper but not used for Leap hand"""
        # Leap hand doesn't use coordinate-based control
        pass

    def move(self, desired_angles):
        """Non-blocking move command"""
        try:
            if isinstance(desired_angles, dict):
                desired_angles = desired_angles['position']
            
            # Update command queue (clear old commands)
            while not self._command_queue.empty():
                self._command_queue.get_nowait()
            self._command_queue.put(desired_angles)
            
            # Update cache for recording if enabled
            if self._is_recording_enabled:
                self._latest_commanded_angles = desired_angles
                self._latest_commanded_angles_timestamp = time.time()
                
            self._last_command = desired_angles
        except Exception as e:
            logger.error(f"Error queueing command: {e}")

    def _execute_movement_loop(self):
        """Background thread that executes movements"""
        logger.info("Movement thread started")
        
        while True:
            try:
                if not self._command_queue.empty():
                    target = self._command_queue.get()
                    if not self._controller.move_hand(target):
                        # If move failed, put command back in queue
                        self._command_queue.put(target)
                # Sleep should be relative to the frequency described
                time.sleep(1 / self._data_frequency)
            except Exception as e:
                logger.error(f"Movement error: {e}")
                time.sleep(1 / self._data_frequency)
    
    def check_reset(self):
        """Check if a reset message was received"""
        reset_bool = self._reset_subscriber.recv_keypoints()
        if reset_bool is not None:
            return True
        return False

    def check_home(self):
        """Check if a home message was received"""
        home_bool = self._home_subscriber.recv_keypoints()

        if home_bool == robots.ARM_TELEOP_STOP:
            return True
        elif home_bool == robots.ARM_TELEOP_CONT:
            return False

        return False
    
    def get_teleop_state(self):
        """Get the teleop state"""
        return self._arm_teleop_state_subscriber.get_arm_teleop_state()
    
    def stream(self):
        """Stream loop for LeapHand"""
        self.home()
        
        target_interval = 1.0 / self._data_frequency
        next_frame_time = time.time()

        avg_start_time  = time.time() # window start for average frequency
        msg_counter     = 0           # how many commands arrived in this window
        
        while True:
            current_time = time.time()

            # Only process at the target frequency
            if current_time >= next_frame_time:
                # Calculate next frame time
                next_frame_time = current_time + target_interval

            if self.check_home() and not self._is_homed:
                # Execute the homing motion.
                self.home()
                self._is_homed = True
            elif not self.check_home() and self._is_homed:
                self._is_homed = False

            if self.get_teleop_state() == robots.ARM_TELEOP_STOP:
                # Stop the robot from moving and wait until we operate again.
                continue

            # Get joint angles with non-blocking receive
            msg = self._joint_angle_subscriber.recv_keypoints()
            self._action = msg
                
            if msg is not None:
                # Process the command
                self.move(msg)
                
            # Update the joint state cache if recording is enabled
            if self._is_recording_enabled:
                current_joint_position = self._controller.get_hand_position()
                self._latest_joint_angles = current_joint_position
                self._latest_joint_angles_timestamp = time.time()
                
            # Publish current state regardless of recording state
            self._publisher_manager.publish(
                self._publisher_host,
                self._joint_angle_publish_port,
                'joint_angles',
                self.get_joint_position(),
            )
            self._state = self.get_joint_position()

            now = time.time()

            msg_counter += 1
            if now - avg_start_time >= 1.0:
                avg_freq = msg_counter / (now - avg_start_time)
                # logger.info(f"Average cmd freq over last second: {avg_freq:6.2f} Hz")
                avg_start_time = now
                msg_counter = 0
                
            # Publish comprehensive state data for recording
            self.publish_current_state()
                
            # Check for reset signal
            if self.check_reset():
                # Automatically start recording after reset
                self.start_recording()
                
            time.sleep(1/self._data_frequency)

    def _get_operator(self):
        """Get or retrieve the operator reference"""
        if self._operator is None:
            self._operator = GlobalRegistry.get('leap_hand_operator')
        return self._operator
        
    def get_transformed_keypoints(self):
        """Get the transformed keypoints from the operator"""
        if not self._is_recording_enabled:
            return None
            
        operator = self._get_operator()
        if operator is None:
            return None
            
        return operator.get_latest_transformed_keypoints()

    def publish_current_state(self):
        """
        Gathers all state data defined in recorder_functions
        and publishes it as a single dictionary via ZMQ using self.name as topic
        """
        current_state_dict = {}
        publish_time = time.time()

        for key, getter_function in self.recorder_functions.items():
            try:
                state_data = getter_function()
                if state_data is not None:
                    current_state_dict[key] = state_data
            except Exception as e:
                logger.error(f"Failed to get state for '{key}' on robot '{self.name}': {e}")
                current_state_dict[key] = None

        current_state_dict['timestamp'] = publish_time

        # Publish the state dictionary using self.name as topic
        try:
            self._publisher_manager.publish(
                self._publisher_host,
                self._state_publish_port,
                self.name,
                current_state_dict,
            )
        except Exception as e:
            logger.error(f"Error publishing state dictionary for robot '{self.name}': {e}")

    def __del__(self):
        for subscriber in self._subscribers.values():
            subscriber.stop()
        cleanup_zmq_resources()
