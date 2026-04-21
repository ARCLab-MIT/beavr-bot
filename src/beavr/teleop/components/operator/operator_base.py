import time
import logging
from collections import deque
from abc import ABC, abstractmethod

import numpy as np
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.components import Component

logger = logging.getLogger(__name__)


class Operator(Component, ABC):
    @property
    @abstractmethod
    def timer(self):
        return self._timer

    # This function is used to create the robot
    @property
    @abstractmethod
    def robot(self):
        return self._robot

    # This function is the subscriber for the hand keypoints
    @property
    @abstractmethod
    def transformed_hand_keypoint_subscriber(self):
        return self._transformed_hand_keypoint_subscriber

    # This function is the subscriber for the arm keypoints
    @property
    @abstractmethod
    def transformed_arm_keypoint_subscriber(self):
        return self._transformed_arm_keypoint_subscriber

    # This function has the majority of retargeting code happening
    @abstractmethod
    def _apply_retargeted_angles(self):
        pass

    def cleanup(self):
        """Clean up resources before shutdown."""
        logger.info(f"Cleaning up {self.__class__.__name__}...")
        try:
            # Stop subscribers in a safe way
            if (
                hasattr(self, "transformed_arm_keypoint_subscriber")
                and self.transformed_arm_keypoint_subscriber
            ):
                try:
                    self.transformed_arm_keypoint_subscriber.stop()
                except Exception as e:
                    logger.error(f"Error stopping arm subscriber: {e}")

            if (
                hasattr(self, "transformed_hand_keypoint_subscriber")
                and self.transformed_hand_keypoint_subscriber
            ):
                try:
                    self.transformed_hand_keypoint_subscriber.stop()
                except Exception as e:
                    logger.error(f"Error stopping hand subscriber: {e}")

            # Clean up any ZMQ resources
            cleanup_zmq_resources()

            logger.info(f"{self.__class__.__name__} cleanup complete")
        except Exception as e:
            logger.error(f"Error during {self.__class__.__name__} cleanup: {e}")

    # This function applies the retargeted angles to the robot
    def stream(self):
        """Main operator loop with proper cleanup."""
        try:
            self.notify_component_start("{} control".format(self.robot))
            logger.info("Start controlling the robot hand using the Oculus Headset.\n")

            frame_count = 0
            target_interval = 1.0 / 30.0 
            self._start_time = time.time()
            iter_times = deque(maxlen=1000)

            while True:
                start_time_iter = time.perf_counter()
                try:
                    if self.return_real() is True:
                        if self.robot.get_joint_position() is not None:
                            self._apply_retargeted_angles()
                    else:
                        self._apply_retargeted_angles()

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in operator loop: {e}")
                    break

                frame_count += 1

                behind = np.sum(iter_times) - len(iter_times)*target_interval
                sleep_time = min(target_interval, target_interval - behind) # Sleep at most 33ms or less if behind

                if sleep_time > 0:
                    time.sleep(sleep_time)

                elapsed_iter = time.perf_counter() - start_time_iter

                # At the first iteration, the loop has to wait at get_hand_frame, so we use the target interval as default here
                if frame_count == 1:
                    iter_times.append(target_interval)
                else:
                    iter_times.append(elapsed_iter)

        finally:
            self.cleanup()
