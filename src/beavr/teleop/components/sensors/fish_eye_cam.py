import numpy as np
from beavr.teleop.components import Component
from beavr.teleop.utils.images import rescale_image
from beavr.teleop.utils.timer import FrequencyTimer
from beavr.teleop.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter
from beavr.teleop.configs.constants import cameras
import cv2
import time
import logging

logger = logging.getLogger(__name__)


class FishEyeCamera(Component):
    def __init__(self,cam_index,stream_configs, stream_oculus = False):
        # Disabling scientific notations
        np.set_printoptions(suppress=True)
        self.cam_id = cam_index
        #self.output_file = output_file
        self._stream_configs = stream_configs
        self._stream_oculus = stream_oculus
       

        # Different publishers to avoid overload
        self.rgb_publisher = ZMQCameraPublisher(
            host = stream_configs['host'],
            port = stream_configs['port']#(0 if self.cam_id == 24 else self.cam_id)
        )

        
        if self._stream_oculus:
            self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                host = stream_configs['host'],
                port = stream_configs['set_port_offset'] + cameras.VIZ_PORT_OFFSET # Oculus only reads from a set port - this shouldn't change with the camera ID
                # port= 10005 + VIZ_PORT_OFFSET
            )
            logger.info('STREAMING HERE IN FISH EYE CAM: {}'.format(cam_index))

        self.timer = FrequencyTimer(cameras.CAM_FPS) # 30 fps

        # Starting the Fisheye pipeline
        self._start_fisheye()

    def _start_fisheye(self):
        
        logger.info("Cam Id is ", self.cam_id)
        self.cap = cv2.VideoCapture(self.cam_id)
       
       
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
       
        logger.info("Cap is ", self.cap.isOpened())
        # Check if the camera is opened successfully, wait until it is
        while not self.cap.isOpened():
            cap=self.cap.isOpened()


    def get_rgb_depth_images(self):
        frame = None
        while frame is None:
            ret, frame = self.cap.read()
        timestamp = time.time()
        return frame, timestamp
    def stream(self):
        # Starting the fisheye stream
        self.notify_component_start('FishEye')
        logger.info(f"Started the pipeline for FishEye camera: {self.cam_id}!")
        logger.info("Starting stream on {}:{}...\n".format(self._stream_configs['host'], self._stream_configs['port']))
        
        if self._stream_oculus:
            logger.info('Starting oculus stream on port: {}\n'.format(self._stream_configs['port'] + cameras.VIZ_PORT_OFFSET))

        while True:
            try:
                self.timer.start_loop()
                color_image,timestamp = self.get_rgb_depth_images()

                # Publishing the rgb images
                self.rgb_publisher.pub_rgb_image(color_image, timestamp)
                if self._stream_oculus:
                    self.rgb_viz_publisher.send_image(rescale_image(color_image, 2)) # 640 * 360

                self.timer.end_loop()
                if cv2.waitKey(1) == ord('q'):
                    break
            except KeyboardInterrupt:
                break
        self.cap.release()
        logger.info('Shutting down pipeline for camera {}.'.format(self.cam_id))
        self.rgb_publisher.stop()
        if self._stream_oculus:
            self.rgb_viz_publisher.stop()
        