# -----------------------------------------------------------------------------
# Port Configuration Constants
# -----------------------------------------------------------------------------

# Generic ports
KEYPOINT_STREAM_PORT = 8088        # All keypoint data from VR/Oculus
CONTROL_STREAM_PORT = 8086         # All control signals (buttons, resets)
ROBOT_STATE_PORT = 8090            # Robot state information
ROBOT_COMMAND_PORT = 8091          # Robot commands

# Oculus/VR ports
RIGHT_HAND_OCULUS_RECEIVER_PORT = 8087        # Raw data from Oculus
LEFT_HAND_OCULUS_RECEIVER_PORT = 8110         # Raw data from left hand sensor
OCULUS_RECEIVER_PORT = 8087                   # Alias for right hand
LEFT_HAND_RECEIVER_PORT = 8110                # Alias for left hand

# Button and control ports
RESOLUTION_BUTTON_PORT = 8095      # Button input (subscribe)
RESOLUTION_BUTTON_PUBLISH_PORT = 8094  # ✅ FIX: Moved to avoid conflict with left keypoint transform
TELEOP_RESET_PORT = 8100
TELEOP_RESET_PUBLISH_PORT = 8102

# Transformed keypoints ports
KEYPOINT_TRANSFORM_PORT = 8092
LEFT_KEYPOINT_TRANSFORM_PORT = 8093

# Camera / image streaming ports
CAM_PORT_OFFSET = 10005
SIM_IMAGE_PORT = 10005
FISH_EYE_CAM_PORT_OFFSET = 10010
OCULUS_GRAPH_PORT = 15001

# Deployment and data ports
DEPLOYMENT_PORT = 10000
UNIFIED_DATA_PORT = 8088  # All data from oculus.py

# Robot-specific ports
PRE_ACTION_THUMB_EE_POSITION_PORT = 8091
POST_ACTION_THUMB_EE_POSITION_PORT = 8092

# Gripper ports
GRIPPER_PUBLISH_PORT_RIGHT = 8108
GRIPPER_PUBLISH_PORT_LEFT = 8115

# Cartesian and joint control ports
CARTESIAN_PUBLISHER_PORT = 8118
JOINT_PUBLISHER_PORT = 8119
CARTESIAN_COMMAND_PUBLISHER_PORT = 8120
CARTESIAN_PUBLISHER_PORT_LEFT = 8116
JOINT_PUBLISHER_PORT_LEFT = 8117
CARTESIAN_COMMAND_PUBLISHER_PORT_LEFT = 8121

# XArm specific ports
XARM_ENDEFF_PUBLISH_PORT = 10010
XARM_ENDEFF_SUBSCRIBE_PORT = 10009
XARM_JOINT_SUBSCRIBE_PORT = 10029
XARM_RESET_SUBSCRIBE_PORT = 10009
XARM_HOME_SUBSCRIBE_PORT = 10007
XARM_STATE_PUBLISH_PORT = 10016
XARM_TELEOPERATION_STATE_PORT = 8089

# Leap hand specific ports
LEAP_JOINT_ANGLE_SUBSCRIBE_PORT = 8120
LEAP_JOINT_ANGLE_PUBLISH_PORT = 8119
LEAP_RESET_SUBSCRIBE_PORT = 8102
LEAP_STATE_PUBLISH_PORT_RIGHT = 10014
LEAP_HOME_SUBSCRIBE_PORT = 10007
LEAP_STATE_PUBLISH_PORT_LEFT = 10015

# Camera port offsets
VIZ_PORT_OFFSET = 500
DEPTH_PORT_OFFSET = 1000

# LeKiwi ports
LEKIWI_PORT = 5555
LEKIWI_VIDEO_PORT = 5556