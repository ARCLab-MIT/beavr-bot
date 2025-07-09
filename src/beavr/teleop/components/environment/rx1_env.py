from beavr.teleop.utils.network import ZMQCompressedImageTransmitter
from beavr.teleop.utils.images import rescale_image
from beavr.teleop.constants import VIZ_PORT_OFFSET
from beavr.teleop.utils.logger import RobotLogger

import numpy as np
import time
import roslibpy
import logging

logger = logging.getLogger(__name__)


class RX1GazeboEnv:
    def __init__(self, 
                 host,
                 camport,
                 timestamppublisherport,
                 endeff_publish_port,
                 endeffpossubscribeport,
                 robotposepublishport,
                 stream_oculus=False,
                 sim_frequency=30,
                 control_mode='relative',
                 logging_config=None):
        """Initialize Gazebo environment with ROS bridge connection."""
        
        # Initialize variables
        self.name = "RX1_Sim"
        self.ref_link = "base_link"  # Base link name in URDF
        self.end_effector_link = "right_forearm_pitch2forearm_roll_joint"  # End effector link name
        
        # Initialize logging config
        self.logging_config = logging_config or {"enabled": False}
        self.print_target_analysis = self.logging_config.get("print_target_analysis", False)
        self.print_workspace_warnings = self.logging_config.get("print_workspace_warnings", True)
        self.print_joint_changes = self.logging_config.get("print_joint_changes", False)
        
        # Initialize ROS bridge connection
        self.client = roslibpy.Ros(host='localhost', port=9090)
        self.client.run()
        
        # Initialize publishers and subscribers
        self.joint_state_sub = roslibpy.Topic(self.client, '/joint_states', 'sensor_msgs/JointState')
        self.joint_command_pub = roslibpy.Topic(self.client, '/right_arm_position_controller/command', 'trajectory_msgs/JointTrajectory')
        self.camera_sub = roslibpy.Topic(self.client, '/camera/rgb/image_raw', 'sensor_msgs/Image')
        self.depth_sub = roslibpy.Topic(self.client, '/camera/depth/image_raw', 'sensor_msgs/Image')
        
        # Cache for latest states
        self.current_joint_states = None
        self.current_rgb_image = None
        self.current_depth_image = None
        
        # Setup subscribers
        self.joint_state_sub.subscribe(self._joint_state_callback)
        self.camera_sub.subscribe(self._rgb_callback)
        self.depth_sub.subscribe(self._depth_callback)
        
        # Initialize ZMQ publishers for VR visualization
        if stream_oculus:
            self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                host=host,
                port=camport + VIZ_PORT_OFFSET
            )
        self._stream_oculus = stream_oculus
        
        # Initialize robot logger if enabled
        self.robot_logger = RobotLogger() if self.logging_config["enabled"] else None
        
        # Define joint names (matching URDF)
        self.joint_names = [
            'right_shoul_base2shoul_joint',
            'right_shoul2shoul_rot_joint',
            'right_arm2armrot_joint',
            'right_armrot2elbow_joint',
            'right_forearm2forearmrot_joint',
            'right_forearmrot2forearm_pitch_joint',
            'right_forearm_pitch2forearm_roll_joint'
        ]
        
        logger.info("RX1 Gazebo Environment initialized and connected to ROS")

    def _joint_state_callback(self, message):
        """Callback for joint state updates."""
        self.current_joint_states = message
        
    def _rgb_callback(self, message):
        """Callback for RGB camera updates."""
        self.current_rgb_image = np.frombuffer(message['data'], dtype=np.uint8).reshape(
            message['height'], message['width'], 3)
        
    def _depth_callback(self, message):
        """Callback for depth camera updates."""
        self.current_depth_image = np.frombuffer(message['data'], dtype=np.float32).reshape(
            message['height'], message['width'])

    def get_endeff_position(self):
        """Get end effector position using TF information."""
        tf_client = roslibpy.Topic(self.client, '/tf', 'tf2_msgs/TFMessage')
        
        # Request transform between base_link and end effector
        transform = tf_client.call_service('/tf2_ros/transform', {
            'target_frame': self.end_effector_link,
            'source_frame': self.ref_link,
            'time': roslibpy.Time.now()
        })
        
        # Extract position and orientation
        pos = transform['transform']['translation']
        quat = transform['transform']['rotation']
        
        return np.array([pos['x'], pos['y'], pos['z'], 
                        quat['x'], quat['y'], quat['z'], quat['w']])

    def take_action(self):
        """Execute received action by publishing to gripper pose topic"""
        action = self.endeff_pos_subscriber.recv_keypoints()

        if action is None:
            return
    
        # Create pose message
        pose_msg = {
            'position': {
                'x': action[0],
                'y': action[1],
                'z': action[2]
            },
            'orientation': {
                'x': action[3],
                'y': action[4],
                'z': action[5],
                'w': action[6]
            }
        }
        
        # Publish to the right gripper pose topic that rx1_ik_node expects
        gripper_pose_pub = roslibpy.Topic(self.client, 
                                         '/right_gripper_pose', 
                                         'geometry_msgs/Pose')
        gripper_pose_pub.publish(roslibpy.Message(pose_msg))
        
        # Subscribe to joint states output from rx1_ik_node
        joint_states_sub = roslibpy.Topic(self.client,
                                         '/right_arm_joint_states',
                                         'sensor_msgs/JointState')

    def get_rgb_depth_images(self):
        """Get RGB and depth images from Gazebo camera."""
        if self.current_rgb_image is None or self.current_depth_image is None:
            return None, None, time.time()
            
        return self.current_rgb_image, self.current_depth_image, time.time()

    def stream(self):
        """Stream environment state."""
        logger.info("Start controlling the RX1 in Gazebo using the Oculus Headset.\n")
        
        try:
            while self.client.is_connected:
                # Get and publish images
                color_image, depth_image, timestamp = self.get_rgb_depth_images()
                
                if color_image is not None:
                    # Handle Oculus streaming if enabled
                    if self._stream_oculus:
                        scaled_image = rescale_image(color_image, 2)
                        self.rgb_viz_publisher.send_image(scaled_image)
                
                # Get and publish position
                position = self.get_endeff_position()
                self.endeff_publisher.pub_keypoints(position, 'endeff_coords')
                
                # Take action
                self.take_action()
                
                # Sleep to maintain frequency
                time.sleep(1.0/30.0)  # 30Hz update rate
                
        except KeyboardInterrupt:
            logger.info('Stopping the environment!')
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        try:
            # Unsubscribe from topics
            self.joint_state_sub.unsubscribe()
            self.camera_sub.unsubscribe()
            self.depth_sub.unsubscribe()
            
            # Terminate ROS bridge connection
            self.client.terminate()
            
            # Close robot logger
            if self.robot_logger:
                self.robot_logger.close()
            
            logger.info("Successfully cleaned up RX1GazeboEnv resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()