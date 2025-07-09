from abc import abstractmethod
from beavr.teleop.components.environment.hand_env import Hand_Env
import pybullet as p
import pybullet_data
from beavr.teleop.utils.timer import FrequencyTimer
from beavr.teleop.constants import CAM_FPS_SIM, DEPTH_PORT_OFFSET, VIZ_PORT_OFFSET
import time

from beavr.teleop.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter,ZMQKeypointPublisher,ZMQKeypointSubscriber

import logging

logger = logging.getLogger(__name__)


class PyBulletBaseEnvHand(Hand_Env):
    def __init__(self,
                host,
                camport,
                transformed_keypoints_port, 
                jointanglepublishport,
                jointanglesubscribeport,
                timestamppublisherport,
                endeff_publish_port,
                endeffpossubscribeport,
                actualanglepublishport,
                stream_oculus=False,
                sim_frequency=CAM_FPS_SIM
                ):
        """
        Initialize PyBullet base environment
        
         Args:
            host (str): Network host address to connect to the publisher and subscriber components (e.g., "127.0.0.1").
            camport (int): Port number for streaming RGB camera data via ZeroMQ.
            jointanglepublishport (int): Port for publishing joint angles of the robotic hand to the network.
            jointanglesubscribeport (int): Port for subscribing to desired joint angles from the network (e.g., for a teleoperator).
            timestamppublisherport (int): Port for publishing timestamps from the simulation for synchronization.
            endeff_publish_port (int): Port for publishing end-effector position data to the network.
            endeffpossubscribeport (int): Port for subscribing to end-effector position data from the network (e.g., for feedback).
            actualanglepublishport (int): Port for publishing the actual current joint angles of the robotic hand to the network.
            stream_oculus (bool, optional): Flag to indicate whether to stream the camera images to Oculus. Default is False.
            sim_frequency (int, optional): Simulation frequency in frames per second (FPS). Default is set by `CAM_FPS_SIM` constant.
        """
        # Store oculus streaming flag first
        self._stream_oculus = stream_oculus
        
        # Initialize parent class (Hand_Env) with just self
        super().__init__()
    
        # Define timer, network IP and ports
        self._timer=FrequencyTimer(sim_frequency)
        self.host=host
        self.camport=camport
        self.jointanglepublishport=jointanglepublishport
        self.jointanglesubscribeport=jointanglesubscribeport
        self._stream_oculus = stream_oculus

        # PyBullet initialization
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Camera parameters (can be overridden in subclasses)
        self.camera_params = {
            "width": 640,
            "height": 480,
            "fov": 60,
            "near": 0.1,
            "far": 10,
            "view_matrix": p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=1.5,
                yaw=90,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
            ),
            "proj_matrix": p.computeProjectionMatrixFOV(
                fov=60, aspect=640 / 480, nearVal=0.1, farVal=10
            ),
        }

        #Define ZMQ pub/sub
        #Port for publishing rgb images.
        #Define ZMQ pub/sub
        #Port for publishing rgb images.

    
        self.transformed_keypoint_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = transformed_keypoints_port,
            topic = 'transformed_hand_coords'
        )

        self.rgb_publisher = ZMQCameraPublisher(
                host = host,
                port = camport
        )
        
        #Publishing the stream into the oculus.
        if self._stream_oculus:
                self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                        host = host,
                        port = camport + VIZ_PORT_OFFSET
                )

        #Publisher for Depth data
        self.depth_publisher = ZMQCameraPublisher(
                host = host,
                port = camport + DEPTH_PORT_OFFSET 
        )


        #Publisher for Joint Angle 
        self.joint_angle_publisher = ZMQKeypointPublisher(
                host = host,
                port = jointanglepublishport
        )

        #Publisher for Actual Current Joint Angles
        self.actualanglepublisher = ZMQKeypointPublisher(
                host = host,
                port = actualanglepublishport
        )

        #Publisher for calculated angles from teleoperator.
        self.joint_angle_subscriber = ZMQKeypointSubscriber(
                host=host,
                port= jointanglesubscribeport,
                topic='desired_angles'
        )

        #Publisher for endeffector Positions 
        self.endeff_publisher = ZMQKeypointPublisher(
                host = host,
                port = endeff_publish_port
        )

        #Publisher for endeffector Velocities
        self.endeff_pos_subscriber = ZMQKeypointSubscriber(
                host = host,
                port = endeffpossubscribeport,
                topic='endeff_coords'
        )

        #Publisher for timestamps
        self.timestamp_publisher = ZMQKeypointPublisher(
                host=host,
                port=timestamppublisherport
        )

        self.subscribers = [self.endeff_pos_subscriber, self.joint_angle_subscriber, self.transformed_keypoint_subscriber]

        self.publishers = [
            self.rgb_publisher,
            self.depth_publisher,
            self.joint_angle_publisher,
            self.actualanglepublisher,
            self.endeff_publisher,
            self.timestamp_publisher,
        ]

        if self._stream_oculus:
            self.publishers.append(self.rgb_viz_publisher)
        
    @abstractmethod
    def load_assets(self):
        """Abstract method to load assets (e.g., robot arm, hand, etc.)."""
        pass

    @abstractmethod
    def take_action(self):
        """Abstract method for performing actions in the simulation."""
        pass

    @abstractmethod
    def get_endeff_position(self):
        """Abstract method to get the end-effector position."""
        pass

    def get_rgb_depth_images(self, camera_name=None):
        """Get RGB and depth images from simulation with timestamp."""
        images = p.getCameraImage(
            width=self.camera_params["width"],
            height=self.camera_params["height"],
            viewMatrix=self.camera_params["view_matrix"],
            projectionMatrix=self.camera_params["proj_matrix"],
        )
        rgb_image = images[2].reshape((self.camera_params["height"], self.camera_params["width"], 4))[:, :, :3]
        depth_image = images[3].reshape((self.camera_params["height"], self.camera_params["width"]))
        timestamp = time.time()
        return rgb_image, depth_image, timestamp

    def step_simulation(self):
        """Common method to step the PyBullet simulation."""
        p.stepSimulation()

    @property
    def timer(self):
        return self._timer

    def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up network resources
            for publisher in self.publishers:
                publisher.stop()  # Use stop() instead of close()
            for subscriber in self.subscribers:
                subscriber.stop()
            
            # Disconnect from PyBullet
            p.disconnect()
            
            logger.info("Successfully cleaned up XArmEnv resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()