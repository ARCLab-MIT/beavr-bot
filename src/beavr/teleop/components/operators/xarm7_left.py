import numpy as np
from typing import Dict, Any, Optional

# Import the base class
from .xarm_base import XArmOperator

# Define the transformation matrices specific to the LEFT arm
# !!! IMPORTANT: Replace these placeholder matrices with the correct ones for your left arm setup !!!
# H_R_V: Transformation from Robot base frame to VR base frame
H_R_V_LEFT = np.array([
    [ 0,  0,  1,  0],
    [ 0, -1,  0,  0],
    [-1,  0,  0,  0],
    [ 0,  0,  0,  1]
])

# H_T_V: Transformation from Hand Tracking base frame to VR base frame
H_T_V_LEFT = np.array([
    [ 0,  1,  0,  0],
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
    [ 0,  0,  0,  1]
])


class XArm7LeftOperator(XArmOperator):
    """
    Operator for controlling the LEFT XArm7 robot arm via teleoperation.
    Inherits common logic from XArmOperator and provides left-arm specific
    transformation matrices.
    """
    def __init__(
        self,
        host: str,
        transformed_keypoints_port: int,
        stream_configs: Dict[str, Any],
        stream_oculus: bool,
        endeff_publish_port: int,
        endeff_subscribe_port: int,
        moving_average_limit: int,
        use_filter: bool = True,
        arm_resolution_port: Optional[int] = None,
        teleoperation_state_port: Optional[int] = None,
        logging_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the XArm7LeftOperator.

        Args:
            host: Network host address for ZMQ communication.
            transformed_keypoints_port: Port for receiving transformed hand keypoints.
            stream_configs: Configuration for streaming data.
            stream_oculus: Flag indicating if Oculus streaming is used.
            endeff_publish_port: Port for publishing end-effector commands.
            endeff_subscribe_port: Port for subscribing to end-effector state.
            moving_average_limit: Number of samples for moving average filter.
            use_filter: Whether to enable the complementary state filter.
            arm_resolution_port: Optional port for arm resolution control messages.
            teleoperation_state_port: Optional port for teleoperation reset/pause messages.
            logging_config: Optional configuration dictionary for pose logging.
        """
        # Call the base class constructor with left-arm specific parameters
        super().__init__(
            operator_name='xarm7_left_operator', # Specific name for this instance
            host=host,
            transformed_keypoints_port=transformed_keypoints_port,
            stream_configs=stream_configs,
            stream_oculus=stream_oculus,
            endeff_publish_port=endeff_publish_port,
            endeff_subscribe_port=endeff_subscribe_port,
            moving_average_limit=moving_average_limit,
            h_r_v=H_R_V_LEFT, # Pass the left arm's H_R_V
            h_t_v=H_T_V_LEFT, # Pass the left arm's H_T_V
            use_filter=use_filter,
            arm_resolution_port=arm_resolution_port,
            teleoperation_state_port=teleoperation_state_port,
            logging_config=logging_config,
        )

    # No need to redefine methods like _apply_retargeted_angles, run, etc.
    # They are inherited from XArmOperator.
    # Add any left-arm specific overrides or methods here if needed in the future.
