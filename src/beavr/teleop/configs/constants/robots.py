# -----------------------------------------------------------------------------
# Robot Configuration Constants
# -----------------------------------------------------------------------------

import os

import numpy as np

ROBOT_NAME_LEAP = "leap"
ROBOT_NAME_XARM7 = "xarm7"
ROBOT_NAME_OPENARM = "openarm"

# -----------------------------------------------------------------------------
# VR detector constants
# -----------------------------------------------------------------------------
# Arm movement
WRIST_HOME_STATE = {
    "translation": [0, 0, 0],
    "rotation_matrix": [1, 0, 0, 0, 1, 0, 0, 0, -1],
}

# -----------------------------------------------------------------------------
# Joint Information
# -----------------------------------------------------------------------------
OCULUS_NUM_KEYPOINTS = 26
VR_THUMB_BOUND_VERTICES = 4
GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1

OCULUS_JOINTS = {
    # Indices follow XRHandJointID order:
    # 0 Wrist, 1 Palm,
    # Thumb: 2 Metacarpal, 3 Proximal, 4 Distal, 5 Tip
    # Index: 6 Metacarpal, 7 Proximal, 8 Intermediate, 9 Distal, 10 Tip
    # Middle: 11 Metacarpal, 12 Proximal, 13 Intermediate, 14 Distal, 15 Tip
    # Ring: 16 Metacarpal, 17 Proximal, 18 Intermediate, 19 Distal, 20 Tip
    # Little: 21 Metacarpal, 22 Proximal, 23 Intermediate, 24 Distal, 25 Tip
    "metacarpals": [2, 6, 11, 16, 21],
    # Knuckles = first finger joints at the hand (proximal joints for non-thumb)
    "knuckles": [7, 12, 17, 22],  # index, middle, ring, little
    # Per-finger chains (4 joints) used with wrist stacked in code -> 5 points total
    "thumb": [2, 3, 4, 5],
    "index": [7, 8, 9, 10],
    "middle": [12, 13, 14, 15],
    "ring": [17, 18, 19, 20],
    "pinky": [22, 23, 24, 25],  # little finger
    # Fingertip indices for convenience
    "thumb_tip": 5,
    "index_tip": 10,
}

OCULUS_VIEW_LIMITS = {
    "x_limits": [-0.04, 0.04],
    "y_limits": [-0.02, 0.25],
    "z_limits": [-0.04, 0.04],
}

# -----------------------------------------------------------------------------
# XELA Sensor parameters
# -----------------------------------------------------------------------------
XELA_FPS = 100
XELA_NUM_SENSORS = 18  # 3 in thumb 4 in other 3 fingers
XELA_PALM_NUM_SENSORS = 3
XELA_FINGERTIP_NUM_SENSORS = 4
XELA_FINGER_NUM_SENSORS = 11
XELA_PALM_NUM_TAXELS = 24
XELA_FINGERTIP_NUM_TAXELS = 30
XELA_FINGER_NUM_TAXELS = 16
XELA_NUM_TAXELS = 16

# -----------------------------------------------------------------------------
# Robot parameters
# -----------------------------------------------------------------------------
# Allegro and Leap
ALLEGRO_JOINTS_PER_FINGER = 4
LEAP_JOINTS_PER_FINGER = 4
ALLEGRO_JOINT_OFFSETS = {"index": 0, "middle": 4, "ring": 8, "thumb": 12}
LEAP_JOINT_OFFSETS = {"index": 0, "middle": 4, "ring": 8, "thumb": 12}

# -----------------------------------------------------------------------------
# XArm constants
# -----------------------------------------------------------------------------
XARM_SCALE_FACTOR = 1000

# XArm home poses
BIMANUAL_LEFT_HOME = [206, 0, 475, 3.142, 0, 0]
BIMANUAL_RIGHT_HOME = [206, 0, 475, 3.142, 0, 0]
ROBOT_HOME_POSE_AA = [206.0, -0.0, 475, 3.142, 0.0, 0.0]
ROBOT_HOME_JS = [
    0.0,
    -0.4363323129985824,
    -0.017453292519943295,
    0.4537856055185257,
    0.0,
    0.8726646259971648,
    0.0,
]

# -----------------------------------------------------------------------------
# OpenArm constants (ROS2-based 7-DOF arm)
# -----------------------------------------------------------------------------
OPENARM_LEFT_JOINT_NAMES = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]
OPENARM_HOME_JS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
OPENARM_TRAJECTORY_DURATION_SEC = 0.1
OPENARM_NUM_JOINTS = 7

# MoveIt IK solver configuration
OPENARM_IK_GROUP_NAME = "left_arm"
OPENARM_IK_FRAME_ID = "world"
OPENARM_IK_LINK_NAME = "openarm_left_hand_tcp"

# -----------------------------------------------------------------------------
# LEAP hand solver scaling factors (used by LeapHandIKSolver)
# -----------------------------------------------------------------------------
LEAP_FINGER_SCALE_FACTOR = 1.8  # default scaling for non-thumb fingertips
LEAP_THUMB_SCALE_FACTOR = 1.7  # separate scaling for thumb positions
LEAP_HOME_JS = np.zeros(16)

# -----------------------------------------------------------------------------
# Oculus constants
# -----------------------------------------------------------------------------
VR_DETECTOR = "vr detector"
KEYPOINTS = "keypoints"
BUTTON = "button"
PAUSE = "pause"
RESUME = "resume"
RIGHT = "right"
LEFT = "left"
BIMANUAL = "bimanual"

# -----------------------------------------------------------------------------
# Keypoint transform constants
# -----------------------------------------------------------------------------
KEYPOINT_POSITION_TRANSFORM = "keypoint position transform"
ABSOLUTE = "absolute"
RELATIVE = "relative"
TRANSFORMED_HAND_COORDS = "transformed_hand_coords"
TRANSFORMED_HAND_FRAME = "transformed_hand_frame"

# -----------------------------------------------------------------------------
# Data recording parameters
# -----------------------------------------------------------------------------
ALLEGRO_SAMPLE_OFFSET = 10  # For sampling states
SAMPLE_WRITER_FPS = 5

# -----------------------------------------------------------------------------
# Robot configuration flags
# -----------------------------------------------------------------------------
# To run the robot
OPERATE = True
# To run the robot interface
ROBOT_INTERFACE = True
# To run the simulation environment
SIM_ENV = False
# To run the Xela controller
RUN_XELA = False

# Visualize 3D and 2D plots
VISUALIZE_XELA = False
VISUALIZE_RIGHT_2D = False
VISUALIZE_RIGHT_3D = False
VISUALIZE_RIGHT_DIR = False

# -----------------------------------------------------------------------------
# Robot identifiers for recording
# -----------------------------------------------------------------------------
ROBOT_IDENTIFIER_RIGHT_XARM7 = "right_xarm7"
ROBOT_IDENTIFIER_LEFT_XARM7 = "left_xarm7"
ROBOT_IDENTIFIER_RIGHT_LEAP_HAND = "right_leap"
ROBOT_IDENTIFIER_LEFT_LEAP_HAND = "left_leap"
ROBOT_IDENTIFIER_LEAP = "leap"
ROBOT_IDENTIFIER_LEFT_OPENARM = "left_openarm"

# -----------------------------------------------------------------------------
# Recorded data types
# -----------------------------------------------------------------------------
RECORDED_DATA_JOINT_STATES = "joint_states"
RECORDED_DATA_CARTESIAN_STATES = "cartesian_states"
RECORDED_DATA_XARM_CARTESIAN_STATES = "xarm_cartesian_states"
RECORDED_DATA_COMMANDED_CARTESIAN_STATE = "commanded_cartesian_state"
RECORDED_DATA_COMMANDED_JOINT_STATES = "commanded_joint_states"
RECORDED_DATA_JOINT_ANGLES_RAD = "joint_angles_rad"

# -----------------------------------------------------------------------------
# VR frequency constants
# -----------------------------------------------------------------------------
VR_FREQ = 30
RECORDER_FREQ = 30
ARM_TELEOP_CONT = 1
ARM_TELEOP_STOP = 0
ARM_HIGH_RESOLUTION = 1
ARM_LOW_RESOLUTION = 0
TELEOP_HANDSHAKE_PORT = 8150

# -----------------------------------------------------------------------------
# OpenArm Gripper constants
# -----------------------------------------------------------------------------
GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1
OPENARM_GRIPPER_OPEN = 0
OPENARM_GRIPPER_CLOSE = 1
OPENARM_GRIPPER_THRESHOLD_M = 0.09  # 9cm max distance for continuous mapping
OPENARM_GRIPPER_MAX_WIDTH_M = 0.045  # 4.5cm maximum gripper opening
OPENARM_GRIPPER_MIN_WIDTH_M = 0.0  # 0cm minimum (fully closed)
OPENARM_GRIPPER_DEFAULT_SPEED_MPS = 0.15  # 15cm/s default speed
ROBOT_IDENTIFIER_OPENARM_GRIPPER = "openarm_gripper"

# -----------------------------------------------------------------------------
# Port mappings for OpenArm components
# -----------------------------------------------------------------------------
# Operator publishes gripper commands on this port
OPENARM_GRIPPER_CMD_PORT = 10046
# OpenArmGripperRobot subscribes to gripper commands on this port
OPENARM_GRIPPER_SUBSCRIBE_PORT = 10046

# -----------------------------------------------------------------------------
# VR display constants
# -----------------------------------------------------------------------------
# Calibration file paths
CALIBRATION_FILES_PATH = "calibration_files"
VR_THUMB_BOUNDS_PATH = os.path.join(CALIBRATION_FILES_PATH, "vr_thumb_bounds.npy")
VR_DISPLAY_THUMB_BOUNDS_PATH = os.path.join(CALIBRATION_FILES_PATH, "vr_thumb_plot_bounds.npy")
VR_2D_PLOT_SAVE_PATH = os.path.join(CALIBRATION_FILES_PATH, "oculus_hand_2d_plot.jpg")
XELA_PLOT_SAVE_PATH = os.path.join(CALIBRATION_FILES_PATH, "xela_plot.png")
