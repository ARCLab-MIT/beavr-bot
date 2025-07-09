import numpy as np
import pyrealsense2 as rs
from beavr.teleop.components import Component
from beavr.teleop.utils.images import rotate_image, rescale_image
from beavr.teleop.utils.timer import FrequencyTimer
from beavr.teleop.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter
from beavr.teleop.configs.constants import cameras

import time
import logging

logger = logging.getLogger(__name__)


class RealsenseCamera(Component):
    def __init__(self, stream_configs, cam_serial_num, cam_id, cam_configs, stream_oculus = False):
        # Disabling scientific notations
        np.set_printoptions(suppress=True)
        self.cam_id = cam_id
        self.cam_configs = cam_configs
        self._cam_serial_num = cam_serial_num
        self._stream_configs = stream_configs
        self._stream_oculus = stream_oculus

        # Different publishers to avoid overload
        self.rgb_publisher = ZMQCameraPublisher(
            host = stream_configs['host'],
            port = stream_configs['port']
        )
        
        if self._stream_oculus:
            self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                host = stream_configs['host'],
                port = stream_configs['port'] + cameras.VIZ_PORT_OFFSET
            )

        self.depth_publisher = ZMQCameraPublisher(
            host = stream_configs['host'],
            port = stream_configs['port'] + cameras.DEPTH_PORT_OFFSET
        )

        self.timer = FrequencyTimer(cameras.CAM_FPS)

        # Starting the realsense pipeline
        self._start_realsense(self._cam_serial_num)

    def _start_realsense(self, cam_serial_num):
        config = rs.config()
        self.pipeline = rs.pipeline()
        config.enable_device(cam_serial_num)

        # Enabling camera streams
        config.enable_stream(
            rs.stream.color, 
            self.cam_configs.width, 
            self.cam_configs.height, 
            rs.format.bgr8, 
            self.cam_configs.fps
        )
        config.enable_stream(
            rs.stream.depth, 
            self.cam_configs.width, 
            self.cam_configs.height, 
            rs.format.z16, 
            self.cam_configs.fps
        )

        # Starting the pipeline
        cfg = self.pipeline.start(config)
        device = cfg.get_device()

        # Add a small delay before setting the preset
        time.sleep(0.5)  # Give the camera time to initialize fully

        try:
            # Confirm we're using the right device
            logger.info(f"Setting visual_preset {self.cam_configs.processing_preset} for camera {self._cam_serial_num}")
            depth_sensor = device.first_depth_sensor()
            depth_sensor.set_option(rs.option.visual_preset, self.cam_configs.processing_preset)
            current = depth_sensor.get_option(rs.option.visual_preset)
            logger.info(f"Verified: Current preset is {current}")
        except Exception as e:
            logger.warning(f"Warning: Could not set visual_preset: {e}")
            logger.warning("Continuing with default preset...")
        self.realsense = self.pipeline

        # Obtaining the color intrinsics matrix for aligning the color and depth images
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy], 
            [0, 0, 1]
        ])

        # Align function - aligns other frames with the color frame
        self.align = rs.align(rs.stream.color)

    def get_rgb_depth_images(self):
        frames = None

        while frames is None:
            # Obtaining and aligning the frames
            frames = self.realsense.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Getting the images from the frames
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image, frames.get_timestamp()

    def stream(self):
        # Starting the realsense stream
        self.notify_component_start('realsense')
        logger.info(f"Started the Realsense pipeline for camera: {self._cam_serial_num}!")
        logger.info("Starting stream on {}:{}...\n".format(self._stream_configs['host'], self._stream_configs['port']))
        
        if self._stream_oculus:
            logger.info('Starting oculus stream on port: {}\n'.format(self._stream_configs['port'] + cameras.VIZ_PORT_OFFSET))

        while True:
            try:
                self.timer.start_loop()
                color_image, depth_image, timestamp = self.get_rgb_depth_images()

                color_image = rotate_image(color_image, self.cam_configs.rotation_angle)
                depth_image = rotate_image(depth_image, self.cam_configs.rotation_angle)

                # Use existing pub_rgb_image method which already has JPEG compression
                self.rgb_publisher.pub_rgb_image(color_image, timestamp)

                # TODO - move the oculus publisher to a separate process - this cycle works at 40 FPS
                if self._stream_oculus:
                    # Already using compression in viz_publisher
                    self.rgb_viz_publisher.send_image(rescale_image(color_image, 2))

                # Use existing pub_depth_image method which already has blosc compression
                self.depth_publisher.pub_depth_image(depth_image, timestamp)
                
                self.depth_publisher.pub_intrinsics(self.intrinsics_matrix)

                self.timer.end_loop()
            except KeyboardInterrupt:
                break
        
        logger.info('Shutting down realsense pipeline for camera {}.'.format(self.cam_id))
        self.rgb_publisher.stop()
        if self._stream_oculus:
            self.rgb_viz_publisher.stop()
        self.depth_publisher.stop()
        self.pipeline.stop()