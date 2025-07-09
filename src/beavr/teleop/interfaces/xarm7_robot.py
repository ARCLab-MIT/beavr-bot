from beavr.teleop.controllers.xarm7_control import DexArmControl
from .robot import RobotWrapper
from beavr.teleop.utils.network import (
    ZMQKeypointSubscriber,
    ZMQPublisherManager,
    HandshakeCoordinator,
    cleanup_zmq_resources,
)
import numpy as np
import time
from beavr.teleop.configs.constants import robots
import logging
from beavr.teleop.utils.ops import Ops

logger = logging.getLogger(__name__)

class XArm7Robot(RobotWrapper):
    def __init__(
            self, host, endeff_subscribe_port, joint_subscribe_port, home_subscribe_port,
            reset_subscribe_port, teleoperation_state_port, robot_ip,
            is_right_arm=True,
            endeff_publish_port: int = 10009,
            state_publish_port: int = 10010,
            **kwargs,
        ):
        """
        Args:
            host: Network host address
            endeff_subscribe_port: Port for end effector command subscription
            joint_subscribe_port: Port for joint command subscription (if used)
            reset_subscribe_port: Port for reset subscription
            robot_ip: IP address of the XArm robot
            is_right_arm: Whether this is the right arm (True) or left arm (False)
            endeff_publish_port: Port for publishing general end-effector data.
            state_publish_port: Port for publishing the full state dictionary for recording.
        """
        # Ensure required ports are present
        if not endeff_publish_port:
             raise ValueError("XArm7Robot requires an 'endeff_publish_port'")
        if not state_publish_port:
             raise ValueError("XArm7Robot requires a 'state_publish_port'")

        self._controller = DexArmControl(ip=robot_ip)
        self._is_right_arm = is_right_arm
        
        self._data_frequency = robots.VR_FREQ
        
        # Subscribers
        self._cartesian_coords_subscriber = ZMQKeypointSubscriber(
            host = host, 
            port = endeff_subscribe_port,
            topic = 'endeff_coords'
        )
        
        self._joint_state_subscriber = ZMQKeypointSubscriber(
            host = host, 
            port = joint_subscribe_port,
            topic = 'joint'
        )

        # Dedicated RESET subscriber -------------------------------------------------
        self._reset_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = reset_subscribe_port,
            topic = 'reset'
        )

        # Dedicated HOME subscriber --------------------------------------------------
        self._home_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = home_subscribe_port,
            topic = 'home'
        )

        # Ops state subscriber --------------------------------------------------------
        # Checks if operation is stopped or continued.
        self._arm_teleop_state_subscriber = Ops(
            arm_teleop_state_subscriber=ZMQKeypointSubscriber(
                host=host,
                port=teleoperation_state_port,
                topic='pause',
            )
        )

        self._subscribers = {
            'cartesian_coords': self._cartesian_coords_subscriber,
            'joint_state': self._joint_state_subscriber,
            'reset': self._reset_subscriber,
            'home': self._home_subscriber,
            'teleop_state': self._arm_teleop_state_subscriber.get_arm_teleop_state
        }

        # Centralized publisher manager and publishing configuration
        self._publisher_manager    = ZMQPublisherManager.get_instance()
        self._publisher_host       = host
        self._endeff_publish_port  = endeff_publish_port
        self._state_publish_port   = state_publish_port

        # Add caches for received messages
        self._latest_cartesian_coords = None
        self._latest_joint_state = None
        self._latest_cartesian_state_timestamp = 0
        self._latest_joint_state_timestamp = 0
        
        # Recording control
        self._is_recording_enabled = False

        # Add cache for the last valid commanded cartesian position received
        self._latest_commanded_cartesian_position = None
        self._latest_commanded_cartesian_timestamp = 0.0

        # Add handshake coordination between operator and robot ----------------------
        self._handshake_coordinator = HandshakeCoordinator.get_instance()
        self._handshake_server_id = f"{self.name}_handshake"
        
        # Start handshake server for this robot
        try:
            self._handshake_coordinator.start_server(
                subscriber_id=self._handshake_server_id,
                bind_host="*", 
                port=robots.TELEOP_HANDSHAKE_PORT + (1 if self._is_right_arm else 2)  # Unique ports
            )
            logger.info(f"Handshake server started for {self.name}")
        except Exception as e:
            logging.warning(f"Failed to start handshake server for {self.name}: {e}")

        self._is_homed = False


    @property
    def recorder_functions(self):
        return {
            'joint_states': self.get_joint_state,
            'operator_cartesian_states': self.get_cartesian_state_from_operator,
            'xarm_cartesian_states': self.get_robot_actual_cartesian_position,
            'commanded_cartesian_state': self.get_cartesian_commanded_position,
            'joint_angles_rad': self.get_joint_position, # These are the robot's joint angles in radians
        }

    @property
    def name(self):
        return f"{robots.ROBOT_NAME_XARM7}_{'right' if self._is_right_arm else 'left'}"

    @property
    def data_frequency(self):
        return self._data_frequency

    # State information functions
    def get_joint_state(self):
        return self._controller.get_arm_states()
    
    def get_arm_velocity(self):
        return self._controller.get_arm_velocity()

    def get_arm_torque(self):
        return self._controller.get_arm_torque()

    def get_cartesian_state(self):
        cartesian_state=self._controller.get_cartesian_state() 
        return cartesian_state
    
    def get_joint_position(self):
        return self._controller.get_arm_position()
    
    def get_cartesian_position(self):
        return self._controller.get_arm_cartesian_coords()

    def reset(self):
        return self._controller._init_xarm_control()
    
    def get_teleop_state(self):
        """
        Checks if operation is stopped or continued.
        """
        return self._arm_teleop_state_subscriber.get_arm_teleop_state()
    
    def get_pose(self):
        return self._controller.get_arm_pose()

    # Movement functions
    def home(self):
        return self._controller.home_arm()

    def move(self, input_angles):
        self._controller.move_arm_joint(input_angles)

    def move_coords(self, cartesian_coords, duration=3):
        self._controller.move_arm_cartesian(cartesian_coords, duration=duration)

    def arm_control(self, cartesian_coords):
        self._controller.arm_control(cartesian_coords)

    def move_velocity(self, input_velocity_values, duration):
        pass

    def get_cartesian_state_from_operator(self):
        if self._latest_cartesian_coords is None:
            return None

        cartesian_state_dict = dict(
            cartesian_position = np.array(self._latest_cartesian_coords, dtype=np.float32),
            timestamp = self._latest_cartesian_state_timestamp
        )
        return cartesian_state_dict

    def get_joint_state_from_operator(self):
        if self._latest_joint_state is None:
            return None
        
        joint_state_dict = dict(
            joint_position = np.array(self._latest_joint_state, dtype=np.float32),
            timestamp = self._latest_joint_state_timestamp
        )
        return joint_state_dict

    def get_cartesian_commanded_position(self):
        if self._latest_commanded_cartesian_position is None:
            return None
        return {
            "commanded_cartesian_position": self._latest_commanded_cartesian_position,
            "timestamp": self._latest_commanded_cartesian_timestamp,
        }

    def get_robot_actual_cartesian_position(self):
        cartesian_state=self.get_cartesian_position()
        cartesian_dict = dict(
            cartesian_position = np.array(cartesian_state, dtype=np.float32),
            timestamp = time.time()
        )
        return cartesian_dict
    
    def get_robot_actual_joint_position(self):
        joint_state_dict = self._controller.get_arm_joint_state()
        return joint_state_dict
    
    def send_robot_pose(self):
        cartesian_state = self._controller.get_arm_pose()
        try:
            self._publisher_manager.publish(
                self._publisher_host,
                self._endeff_publish_port,
                "endeff_homo",
                cartesian_state,
            )
        except Exception as e:
            logger.error(f"Failed to publish robot pose for {self.name}: {e}")

    def check_reset(self):
        reset_bool = self._reset_subscriber.recv_keypoints()
        if reset_bool is not None:
            return True
        else:
            return False

    def check_home(self):
        home_bool = self._home_subscriber.recv_keypoints()

        if home_bool == robots.ARM_TELEOP_STOP:
            return True
        elif home_bool == robots.ARM_TELEOP_CONT:
            return False

        return False

    # Modified stream method with automatic recording start after reset
    def stream(self):
        self.home()
        assert self._controller.robot.set_mode_and_state(1, 0), "Failed to enter SERVO-READY"
        
        target_interval = 1.0 / self._data_frequency
        next_frame_time = time.time()
        
        # At the top of the stream() method, **before the main loop**:
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
                    self.send_robot_pose()
                elif not self.check_home() and self._is_homed:
                    self._is_homed = False

                if self.check_reset():
                    self.send_robot_pose()

                # Check operation state --------------------------------------------------
                if self.get_teleop_state() == robots.ARM_TELEOP_STOP:
                    # Stop the robot from moving and wait until we operate again.
                    continue

                msg = self._cartesian_coords_subscriber.recv_keypoints()
                if msg is not None:
                    self._latest_commanded_cartesian_position = np.concatenate(
                        [np.asarray(msg["position"], dtype=np.float32),
                         np.asarray(msg["orientation"], dtype=np.float32)]
                    )
                    self._latest_commanded_cartesian_timestamp = msg["timestamp"]
                    self.move_coords(self._latest_commanded_cartesian_position)

                now = time.time()

                msg_counter += 1
                if now - avg_start_time >= 1.0:
                    avg_freq = msg_counter / (now - avg_start_time)
                    # logger.info(f"Average cmd freq over last second: {avg_freq:6.2f} Hz")
                    avg_start_time = now
                    msg_counter = 0

                # Publish the current state every cycle so that external
                # adapters (e.g. MultiRobotAdapter) receive up-to-date joint
                # information for observation building.
                try:
                    self.publish_current_state()
                except Exception as e:
                    logging.error(f"Failed to publish current state for {self.name}: {e}")

                # Calculate sleep time to maintain consistent frequency
                sleep_time = max(0, next_frame_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def publish_current_state(self):
        """
        Gathers all state data defined in recorder_functions
        and publishes it as a single dictionary via ZMQ using self.name as topic
        on the state_publish_port.
        """
        current_state_dict = {}
        publish_time = time.time()

        for key, getter_function in self.recorder_functions.items():
            try:
                state_data = getter_function()
                if state_data is not None:
                    current_state_dict[key] = state_data
            except Exception as e:
                logging.error(f"Failed to get state for '{key}' on robot '{self.name}': {e}")
                current_state_dict[key] = None

        current_state_dict['timestamp'] = publish_time

        # Publish the state dictionary using the dedicated state publisher and self.name topic
        try:
            self._publisher_manager.publish(
                self._publisher_host,
                self._state_publish_port,
                self.name,
                current_state_dict,
            )
        except Exception as e:
            logging.error(f"Error publishing state dictionary for robot '{self.name}': {e}")
    
    def __del__(self):
        # Stop handshake server for this robot
        if hasattr(self, '_handshake_coordinator') and hasattr(self, '_handshake_server_id'):
            try:
                self._handshake_coordinator.stop_server(self._handshake_server_id)
            except Exception as e:
                logging.warning(f"Error stopping handshake server for {self.name}: {e}")
        
        for subscriber in self._subscribers.values():
            subscriber.stop()
        cleanup_zmq_resources()