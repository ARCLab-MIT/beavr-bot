
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Import all constants from the consolidated config files using terse imports
from beavr.teleop.configs.constants import network, ports, robots, cameras

# -----------------------------------------------------------------------------
# Configuration Models
# -----------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """Network configuration for teleop system."""
    host_address: str = network.HOST_ADDRESS
    left_xarm_ip: str = network.LEFT_XARM_IP
    right_xarm_ip: str = network.RIGHT_XARM_IP
    left_arm_ip: str = network.LEFT_ARM_IP
    right_arm_ip: str = network.RIGHT_ARM_IP
    xarm_left_ip: str = network.XARM_LEFT_IP
    xarm_right_ip: str = network.XARM_RIGHT_IP
    teleop_handshake_port: int = network.TELEOP_HANDSHAKE_PORT
    
    def __post_init__(self):
        """Lightweight validation for network configuration."""
        # Validate IP addresses are in correct format
        for ip_name, ip_addr in [
            ("host_address", self.host_address),
            ("left_xarm_ip", self.left_xarm_ip),
            ("right_xarm_ip", self.right_xarm_ip),
        ]:
            ip_parts = ip_addr.split(".")
            assert len(ip_parts) == 4, f"{ip_name} must be IPv4 format, got: {ip_addr}"
            
        # Validate handshake port range
        assert 1 <= self.teleop_handshake_port <= 65535, f"handshake port out of range: {self.teleop_handshake_port}"


@dataclass
class PortsConfig:
    """Port configuration for all network communication."""
    # Generic ports
    keypoint_stream_port: int = ports.KEYPOINT_STREAM_PORT
    control_stream_port: int = ports.CONTROL_STREAM_PORT
    robot_state_port: int = ports.ROBOT_STATE_PORT
    robot_command_port: int = ports.ROBOT_COMMAND_PORT
    
    # Oculus/VR ports
    right_hand_oculus_receiver_port: int = ports.RIGHT_HAND_OCULUS_RECEIVER_PORT
    left_hand_oculus_receiver_port: int = ports.LEFT_HAND_OCULUS_RECEIVER_PORT
    
    # Button and control ports
    resolution_button_port: int = ports.RESOLUTION_BUTTON_PORT
    resolution_button_publish_port: int = ports.RESOLUTION_BUTTON_PUBLISH_PORT
    teleop_reset_port: int = ports.TELEOP_RESET_PORT
    teleop_reset_publish_port: int = ports.TELEOP_RESET_PUBLISH_PORT
    
    # Transformed keypoints ports
    keypoint_transform_port: int = ports.KEYPOINT_TRANSFORM_PORT
    left_keypoint_transform_port: int = ports.LEFT_KEYPOINT_TRANSFORM_PORT
    
    # Camera streaming ports
    cam_port_offset: int = ports.CAM_PORT_OFFSET
    sim_image_port: int = ports.SIM_IMAGE_PORT
    fish_eye_cam_port_offset: int = ports.FISH_EYE_CAM_PORT_OFFSET
    oculus_graph_port: int = ports.OCULUS_GRAPH_PORT
    
    # Deployment and data ports
    deployment_port: int = ports.DEPLOYMENT_PORT
    unified_data_port: int = ports.UNIFIED_DATA_PORT
    
    # Robot-specific ports
    pre_action_thumb_ee_position_port: int = ports.PRE_ACTION_THUMB_EE_POSITION_PORT
    post_action_thumb_ee_position_port: int = ports.POST_ACTION_THUMB_EE_POSITION_PORT
    
    # Gripper ports
    gripper_publish_port_right: int = ports.GRIPPER_PUBLISH_PORT_RIGHT
    gripper_publish_port_left: int = ports.GRIPPER_PUBLISH_PORT_LEFT
    
    # Cartesian and joint control ports
    cartesian_publisher_port: int = ports.CARTESIAN_PUBLISHER_PORT
    joint_publisher_port: int = ports.JOINT_PUBLISHER_PORT
    cartesian_command_publisher_port: int = ports.CARTESIAN_COMMAND_PUBLISHER_PORT
    cartesian_publisher_port_left: int = ports.CARTESIAN_PUBLISHER_PORT_LEFT
    joint_publisher_port_left: int = ports.JOINT_PUBLISHER_PORT_LEFT
    cartesian_command_publisher_port_left: int = ports.CARTESIAN_COMMAND_PUBLISHER_PORT_LEFT
    
    # XArm specific ports
    xarm_endeff_publish_port: int = ports.XARM_ENDEFF_PUBLISH_PORT
    xarm_endeff_subscribe_port: int = ports.XARM_ENDEFF_SUBSCRIBE_PORT
    xarm_joint_subscribe_port: int = ports.XARM_JOINT_SUBSCRIBE_PORT
    xarm_reset_subscribe_port: int = ports.XARM_RESET_SUBSCRIBE_PORT
    xarm_home_subscribe_port: int = ports.XARM_HOME_SUBSCRIBE_PORT
    xarm_state_publish_port: int = ports.XARM_STATE_PUBLISH_PORT
    xarm_teleoperation_state_port: int = ports.XARM_TELEOPERATION_STATE_PORT
    
    # Leap hand specific ports
    leap_joint_angle_subscribe_port: int = ports.LEAP_JOINT_ANGLE_SUBSCRIBE_PORT
    leap_joint_angle_publish_port: int = ports.LEAP_JOINT_ANGLE_PUBLISH_PORT
    leap_reset_subscribe_port: int = ports.LEAP_RESET_SUBSCRIBE_PORT
    leap_state_publish_port: int = ports.LEAP_STATE_PUBLISH_PORT
    leap_home_subscribe_port: int = ports.LEAP_HOME_SUBSCRIBE_PORT
    
    # Camera port offsets
    viz_port_offset: int = ports.VIZ_PORT_OFFSET
    depth_port_offset: int = ports.DEPTH_PORT_OFFSET
    
    # LeKiwi ports
    lekiwi_port: int = ports.LEKIWI_PORT
    lekiwi_video_port: int = ports.LEKIWI_VIDEO_PORT
    
    def __post_init__(self):
        """Lightweight validation for port configuration."""
        # Check key ports are in valid range
        key_ports = [
            ("keypoint_stream_port", self.keypoint_stream_port),
            ("control_stream_port", self.control_stream_port),
            ("robot_state_port", self.robot_state_port),
        ]
        for port_name, port_value in key_ports:
            assert 1 <= port_value <= 65535, f"{port_name} out of range: {port_value}"


@dataclass
class ResolutionConfig:
    """Resolution and scaling configuration."""
    high: int = network.ARM_HIGH_RESOLUTION
    low: int = network.ARM_LOW_RESOLUTION
    
    def __post_init__(self):
        """Validate resolution values."""
        assert self.high != self.low, "high and low resolution values must be different"


@dataclass
class TeleopFlags:
    """Boolean flags for teleop operation modes."""
    operate: bool = robots.OPERATE
    robot_interface: bool = robots.ROBOT_INTERFACE
    sim_env: bool = robots.SIM_ENV
    run_xela: bool = robots.RUN_XELA
    visualize_xela: bool = robots.VISUALIZE_XELA
    visualize_right_2d: bool = robots.VISUALIZE_RIGHT_2D
    visualize_right_3d: bool = robots.VISUALIZE_RIGHT_3D
    visualize_right_dir: bool = robots.VISUALIZE_RIGHT_DIR
    stream_oculus: bool = True  # Flag to enable streaming to Oculus VR


@dataclass
class TeleopControlConfig:
    """Teleop control constants and thresholds."""
    arm_teleop_cont: int = network.ARM_TELEOP_CONT
    arm_teleop_stop: int = network.ARM_TELEOP_STOP
    deploy_reach_threshold: float = network.DEPLOY_REACH_THRESHOLD
    deploy_freq: int = network.DEPLOY_FREQ
    vr_freq: int = robots.VR_FREQ
    recorder_freq: int = robots.RECORDER_FREQ
    
    def __post_init__(self):
        """Validate control configuration."""
        assert self.deploy_reach_threshold > 0, "deploy_reach_threshold must be positive"
        assert self.deploy_freq > 0, "deploy_freq must be positive"
        assert self.vr_freq > 0, "vr_freq must be positive"


@dataclass
class RobotConfig:
    """Robot configuration for teleop system."""
    # VR detector constants
    wrist_home_state: Dict[str, Any] = field(default_factory=lambda: robots.WRIST_HOME_STATE)
    
    # Joint Information
    oculus_num_keypoints: int = robots.OCULUS_NUM_KEYPOINTS
    vr_thumb_bound_vertices: int = robots.VR_THUMB_BOUND_VERTICES
    gripper_open: int = robots.GRIPPER_OPEN
    gripper_close: int = robots.GRIPPER_CLOSE
    oculus_joints: Dict[str, List[int]] = field(default_factory=lambda: robots.OCULUS_JOINTS)
    oculus_view_limits: Dict[str, List[float]] = field(default_factory=lambda: robots.OCULUS_VIEW_LIMITS)
    
    # XELA Sensor parameters
    xela_fps: int = robots.XELA_FPS
    xela_num_sensors: int = robots.XELA_NUM_SENSORS
    xela_palm_num_sensors: int = robots.XELA_PALM_NUM_SENSORS
    xela_fingertip_num_sensors: int = robots.XELA_FINGERTIP_NUM_SENSORS
    xela_finger_num_sensors: int = robots.XELA_FINGER_NUM_SENSORS
    xela_palm_num_taxels: int = robots.XELA_PALM_NUM_TAXELS
    xela_fingertip_num_taxels: int = robots.XELA_FINGERTIP_NUM_TAXELS
    xela_finger_num_taxels: int = robots.XELA_FINGER_NUM_TAXELS
    xela_num_taxels: int = robots.XELA_NUM_TAXELS
    
    # Robot parameters
    allegro_joints_per_finger: int = robots.ALLEGRO_JOINTS_PER_FINGER
    leap_joints_per_finger: int = robots.LEAP_JOINTS_PER_FINGER
    allegro_joint_offsets: Dict[str, int] = field(default_factory=lambda: robots.ALLEGRO_JOINT_OFFSETS)
    leap_joint_offsets: Dict[str, int] = field(default_factory=lambda: robots.LEAP_JOINT_OFFSETS)
    
    # XArm constants
    xarm_scale_factor: int = robots.XARM_SCALE_FACTOR
    bimanual_left_home: List[float] = field(default_factory=lambda: robots.BIMANUAL_LEFT_HOME)
    bimanual_right_home: List[float] = field(default_factory=lambda: robots.BIMANUAL_RIGHT_HOME)
    robot_home_pose_aa: List[float] = field(default_factory=lambda: robots.ROBOT_HOME_POSE_AA)
    robot_home_js: List[float] = field(default_factory=lambda: robots.ROBOT_HOME_JS)
    
    # LEAP hand solver scaling factors
    leap_finger_scale_factor: float = robots.LEAP_FINGER_SCALE_FACTOR
    leap_thumb_scale_factor: float = robots.LEAP_THUMB_SCALE_FACTOR
    leap_home_js: Any = field(default_factory=lambda: robots.LEAP_HOME_JS)
    
    # Oculus constants
    vr_detector: str = robots.VR_DETECTOR
    keypoints: str = robots.KEYPOINTS
    button: str = robots.BUTTON
    pause: str = robots.PAUSE
    right: str = robots.RIGHT
    left: str = robots.LEFT
    
    # Keypoint transform constants
    keypoint_position_transform: str = robots.KEYPOINT_POSITION_TRANSFORM
    absolute: str = robots.ABSOLUTE
    relative: str = robots.RELATIVE
    transformed_hand_coords: str = robots.TRANSFORMED_HAND_COORDS
    transformed_hand_frame: str = robots.TRANSFORMED_HAND_FRAME
    
    # Data recording parameters
    allegro_sample_offset: int = robots.ALLEGRO_SAMPLE_OFFSET
    sample_writer_fps: int = robots.SAMPLE_WRITER_FPS
    
    # Robot identifiers for recording
    robot_identifier_right_xarm7: str = robots.ROBOT_IDENTIFIER_RIGHT_XARM7
    robot_identifier_left_xarm7: str = robots.ROBOT_IDENTIFIER_LEFT_XARM7
    robot_identifier_leap: str = robots.ROBOT_IDENTIFIER_LEAP
    
    # Recorded data types
    recorded_data_joint_states: str = robots.RECORDED_DATA_JOINT_STATES
    recorded_data_cartesian_states: str = robots.RECORDED_DATA_CARTESIAN_STATES
    recorded_data_xarm_cartesian_states: str = robots.RECORDED_DATA_XARM_CARTESIAN_STATES
    recorded_data_commanded_cartesian_state: str = robots.RECORDED_DATA_COMMANDED_CARTESIAN_STATE
    recorded_data_commanded_joint_states: str = robots.RECORDED_DATA_COMMANDED_JOINT_STATES
    recorded_data_joint_angles_rad: str = robots.RECORDED_DATA_JOINT_ANGLES_RAD
    
    def __post_init__(self):
        """Validate robot configuration."""
        assert self.xarm_scale_factor > 0, "xarm_scale_factor must be positive"
        assert self.oculus_num_keypoints > 0, "oculus_num_keypoints must be positive"
        assert 0 < self.leap_finger_scale_factor < 10, "leap_finger_scale_factor should be reasonable (0-10)"
        assert 0 < self.leap_thumb_scale_factor < 10, "leap_thumb_scale_factor should be reasonable (0-10)"


@dataclass
class CameraConfig:
    """Camera configuration for teleop system."""
    # Realsense Camera parameters
    num_cams: int = cameras.NUM_CAMS
    cam_fps: int = cameras.CAM_FPS
    cam_fps_sim: int = cameras.CAM_FPS_SIM
    width: int = cameras.WIDTH
    height: int = cameras.HEIGHT
    processing_preset: int = cameras.PROCESSING_PRESET
    visual_rescale_factor: int = cameras.VISUAL_RESCALE_FACTOR
    
    # Camera configuration
    cam_configs_width: int = cameras.CAM_CONFIGS_WIDTH
    cam_configs_height: int = cameras.CAM_CONFIGS_HEIGHT
    cam_configs_fps: int = cameras.CAM_CONFIGS_FPS
    cam_configs_processing_preset: int = cameras.CAM_CONFIGS_PROCESSING_PRESET
    cam_configs_rotation_angle: int = cameras.CAM_CONFIGS_ROTATION_ANGLE
    
    # Camera selection
    oculus_cam: int = cameras.OCULUS_CAM
    num_cams_config: int = cameras.NUM_CAMS_CONFIG
    
    # Robot camera serial numbers
    robot_cam_serial_numbers: List[str] = field(default_factory=lambda: cameras.ROBOT_CAM_SERIAL_NUMBERS)
    
    # Fish eye camera configuration
    fisheye_cam_numbers: List[str] = field(default_factory=list)  # Fish eye camera device indices
    
    # Data recording parameters
    image_record_resolution: tuple = field(default_factory=lambda: cameras.IMAGE_RECORD_RESOLUTION)
    image_record_resolution_sim: tuple = field(default_factory=lambda: cameras.IMAGE_RECORD_RESOLUTION_SIM)
    depth_record_fps: int = cameras.DEPTH_RECORD_FPS
    
    # Calibration file paths
    calibration_files_path: str = cameras.CALIBRATION_FILES_PATH
    vr_thumb_bounds_path: str = field(default_factory=lambda: cameras.VR_THUMB_BOUNDS_PATH)
    vr_display_thumb_bounds_path: str = field(default_factory=lambda: cameras.VR_DISPLAY_THUMB_BOUNDS_PATH)
    vr_2d_plot_save_path: str = field(default_factory=lambda: cameras.VR_2D_PLOT_SAVE_PATH)
    xela_plot_save_path: str = field(default_factory=lambda: cameras.XELA_PLOT_SAVE_PATH)
    
    # Camera port configuration
    camera_port: int = cameras.CAMERA_PORT
    
    # Stream configuration defaults
    stream_configs_host: str = cameras.STREAM_CONFIGS_HOST
    stream_configs_port: str = cameras.STREAM_CONFIGS_PORT
    
    # Camera device paths
    camera_index_front: str = cameras.CAMERA_INDEX_FRONT
    camera_index_wrist: str = cameras.CAMERA_INDEX_WRIST
    
    # Camera rotation settings
    camera_rotation_front: int = cameras.CAMERA_ROTATION_FRONT
    camera_rotation_wrist: int = cameras.CAMERA_ROTATION_WRIST
    
    # Camera resolution settings
    camera_width_default: int = cameras.CAMERA_WIDTH_DEFAULT
    camera_height_default: int = cameras.CAMERA_HEIGHT_DEFAULT
    camera_fps_default: int = cameras.CAMERA_FPS_DEFAULT
    
    # LeKiwi camera configuration
    lekiwi_camera_front_index: str = cameras.LEKIWI_CAMERA_FRONT_INDEX
    lekiwi_camera_wrist_index: str = cameras.LEKIWI_CAMERA_WRIST_INDEX
    lekiwi_camera_front_rotation: int = cameras.LEKIWI_CAMERA_FRONT_ROTATION
    lekiwi_camera_wrist_rotation: int = cameras.LEKIWI_CAMERA_WRIST_ROTATION
    lekiwi_camera_width: int = cameras.LEKIWI_CAMERA_WIDTH
    lekiwi_camera_height: int = cameras.LEKIWI_CAMERA_HEIGHT
    lekiwi_camera_fps: int = cameras.LEKIWI_CAMERA_FPS
    
    def __post_init__(self):
        """Validate camera configuration."""
        assert self.num_cams >= 0, "num_cams must be non-negative"
        assert self.cam_fps > 0, "cam_fps must be positive"
        assert self.width > 0 and self.height > 0, "camera resolution must be positive"
        assert len(self.image_record_resolution) == 2, "image_record_resolution must be (width, height)"


@dataclass
class TeleopConfig:
    """Main teleop configuration for the system."""
    # Nested configuration sections
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ports: PortsConfig = field(default_factory=PortsConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    flags: TeleopFlags = field(default_factory=TeleopFlags)
    control: TeleopControlConfig = field(default_factory=TeleopControlConfig)
    operate: bool = robots.OPERATE
    robot_name: str = robots.ROBOT_NAME_XARM7