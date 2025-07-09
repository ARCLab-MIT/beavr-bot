# -----------------------------------------------------------------------------
# Camera Configuration Constants
# -----------------------------------------------------------------------------

import os.path as path

# -----------------------------------------------------------------------------
# Realsense Camera parameters
# -----------------------------------------------------------------------------
NUM_CAMS = 4
CAM_FPS = 30
CAM_FPS_SIM = 60
WIDTH = 1280
HEIGHT = 720
PROCESSING_PRESET = 1  # High accuracy post-processing mode
VISUAL_RESCALE_FACTOR = 2

# Camera configuration
CAM_CONFIGS_WIDTH = 1280
CAM_CONFIGS_HEIGHT = 720
CAM_CONFIGS_FPS = 30
CAM_CONFIGS_PROCESSING_PRESET = 1  # High accuracy mode
CAM_CONFIGS_ROTATION_ANGLE = 0

# Camera selection
OCULUS_CAM = 0  # First camera
NUM_CAMS_CONFIG = 1

VIZ_PORT_OFFSET = 10000
DEPTH_PORT_OFFSET = 20000

# -----------------------------------------------------------------------------
# Robot camera serial numbers
# -----------------------------------------------------------------------------
ROBOT_CAM_SERIAL_NUMBERS = ['211122063527']  # D415

# -----------------------------------------------------------------------------
# Data recording parameters - Images are recorded at CAM_FPS rate
# -----------------------------------------------------------------------------
IMAGE_RECORD_RESOLUTION = (1280, 720) 
IMAGE_RECORD_RESOLUTION_SIM = (480, 480)
DEPTH_RECORD_FPS = 30

# -----------------------------------------------------------------------------
# Calibration file paths
# -----------------------------------------------------------------------------
CALIBRATION_FILES_PATH = 'calibration_files'
VR_THUMB_BOUNDS_PATH = path.join(CALIBRATION_FILES_PATH, 'vr_thumb_bounds.npy')
VR_DISPLAY_THUMB_BOUNDS_PATH = path.join(CALIBRATION_FILES_PATH, 'vr_thumb_plot_bounds.npy')
VR_2D_PLOT_SAVE_PATH = path.join(CALIBRATION_FILES_PATH, 'oculus_hand_2d_plot.jpg')
XELA_PLOT_SAVE_PATH = path.join(CALIBRATION_FILES_PATH, 'xela_plot.png')

# -----------------------------------------------------------------------------
# Camera port configuration
# -----------------------------------------------------------------------------
CAMERA_PORT = 10005  # This port is correct

# -----------------------------------------------------------------------------
# Stream configuration defaults
# -----------------------------------------------------------------------------
STREAM_CONFIGS_HOST = "10.31.152.148"
STREAM_CONFIGS_PORT = "10005"

# -----------------------------------------------------------------------------
# Camera device paths
# -----------------------------------------------------------------------------
CAMERA_INDEX_FRONT = "/dev/video0"
CAMERA_INDEX_WRIST = "/dev/video2"

# -----------------------------------------------------------------------------
# Camera rotation settings
# -----------------------------------------------------------------------------
CAMERA_ROTATION_FRONT = 90
CAMERA_ROTATION_WRIST = 180

# -----------------------------------------------------------------------------
# Camera resolution settings
# -----------------------------------------------------------------------------
CAMERA_WIDTH_DEFAULT = 640
CAMERA_HEIGHT_DEFAULT = 480
CAMERA_FPS_DEFAULT = 30

# -----------------------------------------------------------------------------
# LeKiwi camera configuration
# -----------------------------------------------------------------------------
LEKIWI_CAMERA_FRONT_INDEX = "/dev/video0"
LEKIWI_CAMERA_WRIST_INDEX = "/dev/video2"
LEKIWI_CAMERA_FRONT_ROTATION = 90
LEKIWI_CAMERA_WRIST_ROTATION = 180
LEKIWI_CAMERA_WIDTH = 640
LEKIWI_CAMERA_HEIGHT = 480
LEKIWI_CAMERA_FPS = 30