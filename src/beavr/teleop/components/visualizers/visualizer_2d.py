import numpy as np
from .plotters.plotter_2d import PlotHand2D
from beavr.teleop.components import Component
from beavr.teleop.configs.constants import robots
from beavr.teleop.utils.network import ZMQKeypointSubscriber

import logging

logger = logging.getLogger(__name__)


class Hand2DVisualizer(Component):
    def __init__(self, host, transformed_keypoint_port, oculus_feedback_port, display_plot):
        self.notify_component_start('hand 2D plotter')
        self.subscriber = ZMQKeypointSubscriber(
            host = host,
            port = transformed_keypoint_port,
            topic = 'transformed_hand_coords'
        )

        self.plotter2D = PlotHand2D(host, oculus_feedback_port, display_plot)

    def _get_keypoints(self):
        raw_keypoints = self.subscriber.recv_keypoints()
        return np.array(raw_keypoints).reshape(robots.OCULUS_NUM_KEYPOINTS, 3)

    def stream(self):
        while True:
            try:
                keypoints = self._get_keypoints()
                self.plotter2D.draw(keypoints[:, 0], keypoints[:, 1])
            except Exception as e:
                logger.error(f"Error in hand 2D visualizer: {e}")
                break

        self.subscriber.stop()
        logger.info('Stopping the hand 2D visualizer process')