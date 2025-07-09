from abc import abstractmethod
from beavr.teleop.components.environment.arm_env import Arm_Env
import pybullet as p
import pybullet_data
from beavr.teleop.utils.timer import FrequencyTimer
from beavr.teleop.constants import CAM_FPS_SIM
import time

from beavr.teleop.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter,ZMQKeypointPublisher,ZMQKeypointSubscriber

import logging

logger = logging.getLogger(__name__)


class PyBulletBaseEnv(Arm_Env):
    def __init__(self, 
                 host,
                camport,
                timestamppublisherport,
                endeff_publish_port,
                endeffpossubscribeport,
                robotposepublishport,
                stream_oculus=False,
                sim_frequency=CAM_FPS_SIM):
        """
        Initialize PyBullet base environment
        
        Args:
            host (str): Network host address
            camport (int): Port for camera streaming
            timestamppublisherport (int): Port for timestamp publishing
            endeff_publish_port (int): Port for end effector position publishing
            endeffpossubscribeport (int): Port for end effector position subscription
            robotposepublishport (int): Port for robot pose publishing
            rgb_port (int): Port for RGB camera streaming
            depth_port (int): Port for depth camera streaming
            stream_oculus (bool): Whether to stream to Oculus
            sim_frequency (int): Simulation frequency
        """
        # Store oculus streaming flag first
        self._stream_oculus = stream_oculus
        
        # Initialize parent class (Arm_Env) with just self
        super().__init__()

        # Store parameters
        self.host = host
        self.camport = camport
        self.timestamppublisherport = timestamppublisherport
        self.endeff_publish_port = endeff_publish_port
        self.endeffpossubscribeport = endeffpossubscribeport
        self.robotposepublishport = robotposepublishport

        # PyBullet initialization
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Simulation timer
        self._timer = FrequencyTimer(sim_frequency)

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

        # Robot pose publisher
        self.robot_pose_publisher = ZMQKeypointPublisher(
            host = host,
            port = robotposepublishport
        )

        self.timestamp_publisher = ZMQKeypointPublisher(
            host=host,
            port=timestamppublisherport
        )

        self.endeff_publisher = ZMQKeypointPublisher(
            host=host,
            port=endeff_publish_port
        )

        # Initialize subscriber
        self.endeff_pos_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=endeffpossubscribeport,
            topic='endeff_coords'
        )

        #Define ZMQ pub/sub
        #Port for publishing rgb images.
        self.rgb_publisher = ZMQCameraPublisher(
                host = host,
                port = camport
        )

        
        # #Publisher for Depth data
        # self.depth_publisher = ZMQCameraPublisher(
        #         host = host,
        #         port = camport + DEPTH_PORT_OFFSET 
        # )

        # For Oculus streaming
        if self._stream_oculus:
            self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                host=host,
                port=camport + 2
            )

        # Store publishers/subscribers for cleanup
        self.publishers = [
            self.timestamp_publisher,
            self.endeff_publisher,
            self.robot_pose_publisher
        ]
        self.subscribers = [self.endeff_pos_subscriber]

        # Load robot and environment assets
        self.load_assets()

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
        """Disconnect PyBullet on cleanup."""
        p.disconnect()
