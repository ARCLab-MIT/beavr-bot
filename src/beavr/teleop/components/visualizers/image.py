import cv2
from beavr.teleop.components import Component
from beavr.teleop.constants import CAM_FPS, VISUAL_RESCALE_FACTOR
from beavr.teleop.utils.images import rescale_image
from beavr.teleop.utils.network import ZMQCameraSubscriber
from beavr.teleop.utils.timer import FrequencyTimer

import logging

logger = logging.getLogger(__name__)


class RobotImageVisualizer(Component):
    def __init__(self, host, cam_port_offset, cam_id):        
        self.camera_number = cam_id

        self.notify_component_start('camera {} rgb visualizer'.format(cam_id))
        self.subscriber = ZMQCameraSubscriber(host = host, port = cam_port_offset + cam_id - 1, topic_type = 'RGB')
        
        # Setting frequency
        self.timer = FrequencyTimer(CAM_FPS) 

    def stream(self):
        while True:
            try:
                self.timer.start_loop()

                image, _ = self.subscriber.recv_rgb_image()
                rescaled_image = rescale_image(image, VISUAL_RESCALE_FACTOR)
                cv2.imshow('Robot camera {} - RGB stream'.format(self.camera_number), rescaled_image)
                cv2.waitKey(1)

                self.timer.end_loop()
            except KeyboardInterrupt:
                break
                
        logger.info('Exiting visualizer.')