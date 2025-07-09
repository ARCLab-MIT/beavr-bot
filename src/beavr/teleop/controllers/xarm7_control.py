#import rospy
import numpy as np
import time
from xarm import XArmAPI
from enum import Enum
import math

from beavr.teleop.configs.constants import robots
from scipy.spatial.transform import Rotation as R
from beavr.teleop.utils.orientation import quat_positive, quat_to_axis_angle

class RobotControlMode(Enum):
    CARTESIAN_CONTROL = 0
    SERVO_CONTROL = 1

# Wrapper for XArm
class Robot(XArmAPI):
    def __init__(self, ip="192.168.1.197", is_radian=True, simulation_mode=False):
        super(Robot, self).__init__(
            port=ip, is_radian=is_radian, is_tool_coord=False, enable_report=True, report_type="rich")
        if simulation_mode and not self.is_simulation_robot:
            self.set_simulation_robot(on_off=True)
        self.ip = ip
        
        # Add these attributes to the Robot class
        self._max_step_distance = 50.0  # Maximum allowed movement in mm
        self._last_command_time = 0     # Fix for the attribute error
        self._command_interval = 1.0 / robots.VR_FREQ  # Use VR_FREQ constant (50Hz = 0.02s)

        # Add performance metrics
        self._metrics = {
            "command_times": [],       # Timestamps of commands
            "command_latencies": [],   # Time between command generation and execution
            "position_deltas": [],     # Change in position between commands
            "interval_times": []       # Time between successive commands
        }
        self._last_position = None
        self._metrics_print_interval = 5.0  # Print metrics every 5 seconds
        self._last_metrics_print = time.time()

    def clear(self):
        self.clean_error()
        self.clean_warn()
        # self.motion_enable(enable=False)
        self.motion_enable(enable=True)

    def set_mode_and_state(self, mode, state=0):
        """Set mode correctly and verify state change to READY"""
        # First set the mode
        self.set_mode(mode)
        time.sleep(0.2)
        
        # Then set state to STANDBY (0), which should transition to READY (2)
        self.set_state(state)
        time.sleep(0.3)  # Wait for transition to READY
        
        # Check if we're in the expected mode and READY state
        if self.mode == mode and self.state == 2:
            # print(f"Robot ready: Mode={mode}, State={self.state} (READY)")
            return True
        else:
            # print(f"Warning: Robot not ready. Mode={self.mode} (expected {mode}), State={self.state} (expected 2)")
            return False

    def reset(self):
        """Reset arm to home position with proper mode setting"""
        # Clean errors
        self.clean_error()
        self.clean_warn()
        self.motion_enable(enable=True)
        
        print("Resetting arm to home...")
        # Set to position control mode
        self.set_mode_and_state(0, 0)  # Cartesian control mode
        time.sleep(0.5)  # Longer wait between mode setting and movement
        
        # Move to home position
        status = self.set_servo_angle(angle=robots.ROBOT_HOME_JS, wait=True, 
                                   is_radian=True, speed=math.radians(30))
        if status != 0:
            print(f"Warning: Failed to home robot via joint angles, status={status}")
        
        time.sleep(0.5)  # Wait before changing modes again
        
        # Set back to servo mode for teleoperation
        self.set_mode_and_state(1, 0)  # Servo control mode
        
        return status

    def get_arm_pose(self):
        status, home_pose = self.get_position_aa()
        home_affine = self.robot_pose_aa_to_affine(home_pose)
        return home_affine

    def get_arm_position(self):
        code, joint_state = self.get_servo_angle(is_radian=True, is_real=True)
        if code != 0:
            print('\033[93m' + f"Warning: Failed to get joint states, error code: {code}" + '\033[0m')
            return None

        # The return value 'joint_state' should be the array of 7 joint angles
        if isinstance(joint_state, (list, tuple, np.ndarray)) and len(joint_state) == 7:
            return np.array(joint_state, dtype=np.float32)
        else:
            print('\033[93m' + f"Warning: Unexpected joint state format from get_servo_angle(): {joint_state}" + '\033[0m')
            return None

    def get_arm_velocity(self):
        """
        Returns the velocity of the joints
        
        Returns:
            numpy.ndarray: Array of joint velocities [joint1_velocity, ..., joint7_velocity] in radians/second
            or None if failed to get joint states
        """
        status, states = self.get_joint_states()
        if status != 0:
            print('\033[93m' + f"Warning: Failed to get joint states, error code: {status}" + '\033[0m')
            return None
        
        velocities = np.array(states[1], dtype=np.float32)
        return velocities

    def get_arm_torque(self):
        """ 
        Returns the torque of the joints
        
        Returns:
            numpy.ndarray: Array of joint torques [joint1_torque, ..., joint7_torque] in Nm
            or None if failed to get joint torques
        """
        status, torques = self.get_joints_torque()
        if status != 0:
            print('\033[93m' + f"Warning: Failed to get joint torques, error code: {status}" + '\033[0m')
            return None
        
        torques = np.array(torques, dtype=np.float32)
        return torques
    
    def get_arm_cartesian_coords(self):
        """
            Returns the cartesian coordinates of the arm
            Output Format:
            [x, y, z] (mm), [roll, pitch, yaw] (radians/degrees)
        """
        status, home_pose = self.get_position_aa()
        return home_pose

    def move_arm_joint(self, joint_angles):
        """
        Move robot to specified joint angles
        
        Args:
            joint_angles (list/ndarray): Target joint angles in radians
            
        Returns:
            int: Status code (0 for success)
        """
        status = self.set_servo_angle(
            angle=joint_angles, 
            wait=True, 
            is_radian=True, 
            mvacc=80, 
            speed=10
        )
        if status != 0:
            print('\033[93m' + f"Warning: Failed to move robot, error code: {status}" + '\033[0m')
        return status

    def _ensure_correct_mode(self, desired_mode, desired_state=0):
        """Minimal mode checking - only changes if needed"""
        if self.mode != desired_mode or self.state != desired_state:
            print(f"Mode correction: current={self.mode},{self.state}, desired={desired_mode},{desired_state}")
            self.set_mode(desired_mode)
            self.set_state(desired_state)
            time.sleep(0.1)
        return True

    def _exceeds_movement_limit(self, target_pose):
        """
        Check if proposed movement exceeds the maximum allowed distance in a single step.
        If it does, split it into smaller movements.
        
        Args:
            target_pose: Target cartesian pose [x, y, z, roll, pitch, yaw]
            
        Returns:
            tuple: (exceeds_limit, waypoints)
                exceeds_limit (bool): True if movement exceeds limit
                waypoints (list): List of intermediate waypoints if exceeds_limit is True, 
                                 otherwise empty list
        """
        # Always refresh current position to avoid drift issues
        status, current_pose = self.get_position_aa()
        if status != 0:
            # If we can't get current position, play it safe and allow the movement
            print("Warning: Couldn't get current position, allowing movement")
            return False, []
            
        # Use actual robot position as reference, not last commanded position
        current_position = np.array(current_pose[0:3])
        
        # Calculate distance (only for position xyz, not orientation)
        target_position = np.array(target_pose[0:3])
        distance = np.linalg.norm(target_position - current_position)
        
        # Check if distance exceeds limit
        if distance > self._max_step_distance:
            # Generate intermediate waypoints
            waypoints = self._split_large_movement(current_pose, target_pose)
            print(f"\033[93mMovement distance {distance:.2f}mm exceeds {self._max_step_distance}mm limit. "
                  f"Split into {len(waypoints)} segments.\033[0m")
            return True, waypoints
            
        return False, []
        
    def _split_large_movement(self, current_pose, target_pose):
        """
        Split large movements into smaller segments that stay within movement limits
        
        Args:
            current_pose: Current cartesian pose [x, y, z, roll, pitch, yaw]
            target_pose: Target cartesian pose [x, y, z, roll, pitch, yaw]
            
        Returns:
            list: List of intermediate waypoints
        """
        distance = np.linalg.norm(np.array(target_pose[0:3]) - np.array(current_pose[0:3]))
        segments = max(2, int(np.ceil(distance / (self._max_step_distance * 0.8))))
        waypoints = []
        
        current_pos = np.array(current_pose[0:3])
        target_pos = np.array(target_pose[0:3])
        
        current_ori = np.array(current_pose[3:6])
        target_ori = np.array(target_pose[3:6])
        
        for i in range(1, segments + 1):
            t = i / segments
            # Linear interpolation for position
            pos = current_pos * (1-t) + target_pos * t
            # Linear interpolation for orientation (simple approach)
            ori = current_ori * (1-t) + target_ori * t
            
            waypoints.append(np.concatenate([pos, ori]).tolist())
        
        return waypoints

    def arm_control(self, cartesian_pose):
        # Check if movement exceeds safety limit
        exceeds_limit, waypoints = self._exceeds_movement_limit(cartesian_pose)
        
        if exceeds_limit:
            # Execute each waypoint in sequence
            for waypoint in waypoints:
                # Use servo mode (1) for real-time teleoperation
                self._ensure_correct_mode(1, 0)
                self.set_servo_cartesian_aa(
                    waypoint, wait=True, relative=False, mvacc=5, speed=1)
        else:
            # Execute single movement as before
            self._ensure_correct_mode(1, 0)
            self.set_servo_cartesian_aa(
                cartesian_pose, wait=False, relative=False, mvacc=5, speed=1)
        
    def move_arm_cartesian(self, cartesian_pos, duration=3):
        """Move the arm with performance metrics"""
        try:
            current_time = time.time()
            
            # Collect metrics for timing analysis
            if hasattr(cartesian_pos, 'get') and cartesian_pos.get('timestamp'):
                # Calculate command latency if timestamp is available
                command_timestamp = cartesian_pos.get('timestamp')
                latency = (current_time - command_timestamp) * 1000  # ms
                self._metrics["command_latencies"].append(latency)
                
            # Check time since last command
            if self._last_command_time > 0:
                interval = current_time - self._last_command_time
                self._metrics["interval_times"].append(interval * 1000)  # ms
                
            # Track position changes for jerk analysis
            if self._last_position is not None:
                position_delta = np.linalg.norm(np.array(cartesian_pos[0:3]) - np.array(self._last_position[0:3]))
                self._metrics["position_deltas"].append(position_delta)
            self._last_position = cartesian_pos.copy()
            
            # Rate limiting
            if current_time - self._last_command_time < self._command_interval:
                return 0  # Skip command if too frequent
            self._last_command_time = current_time
            self._metrics["command_times"].append(current_time)
            
            # ------------------------------------------------------------------
            # Convert xyz + quaternion -> xyz + axis–angle expected by xArm SDK
            # ------------------------------------------------------------------
            if len(cartesian_pos) != 7:
                raise ValueError("Expected 7-D pose (x,y,z,qx,qy,qz,qw) with quaternion orientation")

            pos_m = np.asarray(cartesian_pos[0:3], dtype=np.float32)
            quat  = np.asarray(cartesian_pos[3:7], dtype=np.float32)

            # Normalise and force positive hemisphere using shared utils
            quat = quat_positive(quat)

            # Quaternion → rotation-vector (axis–angle, length-3)
            rotvec = quat_to_axis_angle(quat)

            # Map to xArm convention (rx, -rz, ry)  – keeps behaviour identical to
            # the previous operator-side conversion.
            aa = np.array([rotvec[0], -rotvec[2], rotvec[1]], dtype=np.float32)

            # Assemble 6-D pose in *millimetres* for the SDK call
            pose_mm = np.zeros(6, dtype=np.float32)
            pose_mm[0:3] = pos_m * robots.XARM_SCALE_FACTOR
            pose_mm[3:6] = aa

            # ------------------------------------------------------------------
            # Ensure correct servo mode / state before sending the command
            # ------------------------------------------------------------------
            if self.mode != 1 or (self.state != 1 and self.state != 2):
                print(f"Robot not in correct mode/state. Current: Mode={self.mode}, State={self.state}")
                self.set_mode_and_state(1, 0)
                time.sleep(0.2)

            # Execute move in servo mode using axis–angle format
            status = self.set_servo_cartesian_aa(
                pose_mm, wait=False, relative=False, mvacc=50, speed=10, is_radian=True)
            
            if status != 0:
                print(f"Servo cartesian command failed with status {status}")
            
            # Print metrics periodically
            if current_time - self._last_metrics_print > self._metrics_print_interval:
                self._print_performance_metrics(print_metrics=False)
                self._last_metrics_print = current_time
            
            return status
        except Exception as e:
            print(f"Movement failed: {e}")
            return -1
    
    def home_arm(self):
        """
        Move the arm to home position using joint angles
        """
        try:
            # Convert ROBOT_HOME_JS to numpy array
            home_joints = np.array(robots.ROBOT_HOME_JS, dtype=np.float32)
            
            # Use position control mode (0) for homing
            self.set_mode_and_state(0, 0)
            
            # Move to home position
            status = self.set_servo_angle(angle=home_joints, wait=True, 
                                              is_radian=True, mvacc=5, speed=1)
            return status
        except Exception as e:
            print(f"Error in home_arm: {e}")
            return -1

    def home_robot(self):
        """
        Home the robot to the home position defined as a constant in constants.py
        """
        try:
            status = self.home_arm()
            if status != 0:
                print(f"Warning: home_arm returned status {status}")
            return status
        except Exception as e:
            print(f"Error in home_robot: {e}")
            return -1

    def reset_arm(self):
        self.home_arm()

    # Full robot commands
    def move_robot(self, arm_angles):
        """
        Move robot to specified joint angles. This is an alias for move_arm_joint()
        for backward compatibility.
        
        Args:
            arm_angles (list/ndarray): Target joint angles in radians
            
        Returns:
            int: Status code (0 for success)
        """
        # For consistency, use the existing move_arm_joint method
        return self.move_arm_joint(arm_angles)

    def robot_pose_aa_to_affine(self,pose_aa: np.ndarray) -> np.ndarray:
        """Converts a robot pose in axis-angle format to an affine matrix.
        Args:
            pose_aa (list): [x, y, z, ax, ay, az] where (x, y, z) is the position and (ax, ay, az) is the axis-angle rotation.
            x, y, z are in mm and ax, ay, az are in radians.
        Returns:
            np.ndarray: 4x4 affine matrix [[R, t],[0, 1]]
        """

        rotation = R.from_rotvec(pose_aa[3:]).as_matrix()
        translation = np.array(pose_aa[:3]) / robots.XARM_SCALE_FACTOR

        return np.block([[rotation, translation[:, np.newaxis]],
                        [0, 0, 0, 1]])

    def _print_performance_metrics(self, print_metrics=False):
        """Print detailed performance metrics"""
        metrics = self._metrics
        
        # Keep only recent data (last 100 points)
        for key in metrics:
            if len(metrics[key]) > 100:
                metrics[key] = metrics[key][-100:]
        
        # Initialize variables with default values
        avg_latency = max_latency = 0
        avg_interval = interval_jitter = 0
        avg_delta = max_delta = 0
        
        if print_metrics:
            # Calculate statistics only if we have data
            if len(metrics["command_latencies"]) > 0:
                avg_latency = sum(metrics["command_latencies"]) / len(metrics["command_latencies"])
                max_latency = max(metrics["command_latencies"])
            
            if len(metrics["interval_times"]) > 1:
                avg_interval = sum(metrics["interval_times"]) / len(metrics["interval_times"])
                interval_jitter = np.std(metrics["interval_times"])
            
            if len(metrics["position_deltas"]) > 1:
                avg_delta = sum(metrics["position_deltas"]) / len(metrics["position_deltas"])
                max_delta = max(metrics["position_deltas"])
            
            print("\nPerformance Metrics:")
            print(f"  Command Latency: {avg_latency:.2f}ms avg, {max_latency:.2f}ms max")
            print(f"  Command Interval: {avg_interval:.2f}ms avg, {interval_jitter:.2f}ms jitter")
            print(f"  Position Change: {avg_delta:.4f} avg, {max_delta:.4f} max")
            
            # Diagnostic conclusion
            if interval_jitter > 10:
                print("  Warning: High timing jitter detected - commands arriving inconsistently")
            if max_latency > 50:
                print("  Warning: High maximum latency detected - some commands delayed significantly")
            if max_delta > 0.05:
                print("  Warning: Large position jumps detected - may cause jerky motion")

class DexArmControl:
    """Controller class for XArm robot using the Robot implementation"""
    
    def __init__(self, ip="192.168.1.197"):
        """Initialize the XArm controller with velocity limits"""
        self.robot = Robot(ip, is_radian=True, simulation_mode=False) 
        
        # Set global velocity and acceleration limits
        self.robot.set_tcp_maxacc(50)    # Lower max acceleration (mm/s²)
        self.robot.set_joint_maxacc(10)  # Lower joint acceleration (rad/s²)
        
        # For orientation specifically
        self.robot.set_tcp_jerk(100)     # Lower jerk (smooths changes in acceleration)
        
        self.robot.reset()
        
        # Configuration parameters
        self._command_interval = 1.0 / robots.VR_FREQ
        self._last_command_time = 0
    
    def _init_xarm_control(self):
        """Initialize the XArm control by resetting it"""
        return self.robot.reset()
    
    def get_arm_states(self):
        """Get the current joint state of the arm"""
        position = self.robot.get_arm_position()
        velocity = self.robot.get_arm_velocity()
        torque = self.robot.get_arm_torque()
        
        joint_state_dict = dict(
            joint_position = position,
            joint_velocity = velocity,
            joint_torque = torque,
            timestamp = time.time()
        )
        return joint_state_dict
    
    def get_arm_position(self):
        """Get the current joint positions"""
        return self.robot.get_arm_position()
    
    def get_arm_velocity(self):
        """Get the current joint velocities"""
        return self.robot.get_arm_velocity()
    
    def get_arm_torque(self):
        """Get the current joint torques"""
        return self.robot.get_arm_torque()
    
    def get_arm_cartesian_coords(self):
        """Get the current cartesian coordinates"""
        return self.robot.get_arm_cartesian_coords()
    
    def get_cartesian_state(self):
        """Get the current cartesian state"""
        cartesian_state = self.robot.get_arm_cartesian_coords()
        cartesian_dict = dict(
            cartesian_position = np.array(cartesian_state, dtype=np.float32),
            timestamp = time.time()
        )
        return cartesian_dict
    
    def get_arm_pose(self):
        """Get the current arm pose as affine matrix"""
        return self.robot.get_arm_pose()
    
    def move_arm_joint(self, joint_angles):
        """Move the arm to specified joint angles"""
        return self.robot.move_arm_joint(joint_angles)
    
    def move_arm_cartesian(self, cartesian_pos, duration=3):
        """Move the arm to specified cartesian position with rate limiting"""
        # Apply rate limiting
        current_time = time.time()
        if current_time - self._last_command_time < self._command_interval:
            return 0  # Skip command if too frequent
        self._last_command_time = current_time
        
        # Delegate to the robot implementation
        return self.robot.move_arm_cartesian(cartesian_pos, duration)
    
    def arm_control(self, cartesian_pos):
        """Direct arm control in cartesian space"""
        return self.robot.arm_control(cartesian_pos)
    
    def home_arm(self):
        """Move the arm to home position"""
        return self.robot.home_arm()