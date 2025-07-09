from abc import ABC, abstractmethod
from beavr.teleop.components import Component
from beavr.teleop.utils.network import cleanup_zmq_resources
import logging

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
    
    #This function is the subscriber for the arm keypoints
    @property
    @abstractmethod
    def transformed_arm_keypoint_subscriber(self):
        return self._transformed_arm_keypoint_subscriber

    #This function has the majority of retargeting code happening
    @abstractmethod
    def _apply_retargeted_angles(self):
        pass

    def cleanup(self):
        """Clean up resources before shutdown."""
        logger.info(f'Cleaning up {self.__class__.__name__}...')
        try:
            # Stop subscribers in a safe way
            if hasattr(self, 'transformed_arm_keypoint_subscriber') and self.transformed_arm_keypoint_subscriber:
                try:
                    self.transformed_arm_keypoint_subscriber.stop()
                except Exception as e:
                    logger.error(f"Error stopping arm subscriber: {e}")

            if hasattr(self, 'transformed_hand_keypoint_subscriber') and self.transformed_hand_keypoint_subscriber:
                try:
                    self.transformed_hand_keypoint_subscriber.stop()
                except Exception as e:
                    logger.error(f"Error stopping hand subscriber: {e}")

            # Clean up any ZMQ resources
            cleanup_zmq_resources()
            
            logger.info(f'{self.__class__.__name__} cleanup complete')
        except Exception as e:
            logger.error(f"Error during {self.__class__.__name__} cleanup: {e}")

    #This function applies the retargeted angles to the robot
    def stream(self):
        """Main operator loop with proper cleanup."""
        try:
            self.notify_component_start('{} control'.format(self.robot))
            logger.info("Start controlling the robot hand using the Oculus Headset.\n")

            while True:
                try:
                    if self.return_real() is True:
                        if self.robot.get_joint_position() is not None:
                            self.timer.start_loop()
                            self._apply_retargeted_angles()
                            self.timer.end_loop()
                    else:
                        self.timer.start_loop()
                        self._apply_retargeted_angles()
                        self.timer.end_loop()

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in operator loop: {e}")
                    break
        finally:
            self.cleanup()