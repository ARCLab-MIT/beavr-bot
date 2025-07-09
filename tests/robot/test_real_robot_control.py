import tkinter as tk
from tkinter import ttk
import numpy as np
from beavr.src.ros_links.xarm7_right import DexArmControl
from scipy.spatial.transform import Rotation

import json
from datetime import datetime
import time

import traceback
import traceback


class Log2Xarm:
    def __init__(self):
        self.log_file = "logs/operator_log_20250206_160527.json"
        self.cart_poses = []  # Initialize before calling log_to_sim()
        self.timestamps = []  # Store timestamps
        self.log_to_xarm()
    

    def log_to_xarm(self):
        with open(self.log_file, "r") as f:
            data = json.load(f)
        poses, delays = self._transform_log(data)
        
        # Create output filename based on input filename
        output_file = self.log_file.replace('.json', '_transformed_to_real_robot.json')
        
        # Save poses and their timing information to JSON file
        with open(output_file, 'w') as f:
            json.dump({
                'transformed_poses': [pose.tolist() for pose in poses],  # Convert numpy arrays to lists
                'delays': delays,  # Add delays between poses
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }, f, indent=2)
        print(f"Transformed poses saved to: {output_file}")

    def _transform_log(self, data):
        """
        Gets the hand movements from the log and transforms them into a simulation
        feasible format. Homogeneous transformation to the robot initial pose knowing
        the robot initial pose.
        """
        
        frames_data = [data["frames"][str(i)] for i in range(len(data["frames"]))]
        delays = []
        
        # Validate frames
        if not frames_data:
            raise ValueError("No frames found in log data")
            
        # Get timestamps and calculate delays
        timestamps = [frame["timestamp"] for frame in frames_data]
        for i in range(len(timestamps) - 1):
            delay = timestamps[i + 1] - timestamps[i]
            delays.append(delay)
        delays.append(0)  # Add 0 delay for last pose

        print(f"Processing {len(frames_data)} frames...")
        
        for i, frame in enumerate(frames_data):
            # Validate frame data
            required_keys = ["init_hand_pose", "current_hand_pose"]
            if not all(key in frame for key in required_keys):
                print(f"Warning: Frame {i} missing required data")
                continue
                
            # Convert lists to numpy arrays
            hand_init = np.array(frame["init_hand_pose"])
            hand_moving = np.array(frame["current_hand_pose"])

            robot_init = np.array([[0.999848,     -0.000061 ,        0.017452 , 0.3166          ],
                                   [ -0.       , -0.999994     ,   -0.003491      , -0.0003     ],
                                   [  0.017452  , 0.00349      ,   -0.999842 , 0.3026           ],
                                   [  0.       , 0.       ,  0.         , 1.                    ]])

            H_HI_HH = hand_init
            H_HT_HH = hand_moving
            H_RI_RH = robot_init

            H_R_V = np.array([[ 0, -1,  0,  0],
                            [ 0,  0, -1,  0],
                            [ 1,  0,  0,  0],
                            [ 0,  0,  0,  1]])

            H_T_V = np.array([[ 0, -1,  0,  0],
                            [ 0,  0, -1,  0],
                            [ 1,  0,  0,  0],
                            [ 0,  0,  0,  1]])
            
            # Calculate transformations (same order as Allegro)
            H_HT_HI = np.linalg.solve(H_HI_HH, H_HT_HH)
            H_HT_HI_r = np.linalg.solve(H_R_V, H_HT_HI @ H_R_V)[:3,:3]
            H_HT_HI_t = np.linalg.solve(H_T_V, H_HT_HI @ H_T_V)[:3,3]
            relative_affine = np.block([[H_HT_HI_r, H_HT_HI_t.reshape(3,1)], [0, 0, 0, 1]])
            
            # Update robot state (like Allegro)
            H_RT_RH = H_RI_RH @ relative_affine

            # Get cart pose and publish (like Allegro)
            cart = self._homo2cart(H_RT_RH)
            self.cart_poses.append(cart)

            # Print some stats occasionally
            if i % 50 == 0:
                print(f"Processing frame {i}/{len(frames_data)}")
                print(f"Position delta: {np.linalg.norm(H_HT_HH[:3,3] - H_HI_HH[:3,3]):.3f} mm")
                print(f"Current position: {H_RT_RH[:3,3]}")
                print(f"Current orientation (matrix):\n{H_RT_RH[:3,:3]}")

        return self.cart_poses, delays
            
    def _homo2cart(self, homo_mat):
        # Here we will use the resolution scale to set the translation resolution
        t = homo_mat[:3, 3]
        R = Rotation.from_matrix(
            homo_mat[:3, :3]).as_quat()

        cart = np.concatenate(
            [t, R], axis=0
        )

        return cart
        
    
class RobotController:
    """
    Handles all robot control logic and communication with the robot hardware.
    Separates control logic from GUI elements.
    """
    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
        self.initialize_robot()
        self.load_poses()
        self.current_pose_index = 0

    def initialize_robot(self):
        """Initialize robot connection and settings"""
        try:
            self.robot_control = DexArmControl(ip="192.168.1.197")
            if self.simulation_mode:
                self.robot_control.robot.set_simulation_robot(on_off=True)
                print("Robot connected in SIMULATION mode!")
            else:
                print("WARNING: Running in REAL robot mode!")
            
            self.robot_control.robot.clean_error()
            self.robot_control.robot.clean_warn()
            self.robot_control.robot.motion_enable(enable=True)

            # Set global speed and acceleration limits
            # self.robot_control.robot.set_tcp_maxacc(500)    # 500 mm/s²
            # self.robot_control.robot.set_tcp_jerk(1000)     # 1000 mm/s³
            
            mode_result = self.robot_control.robot.set_mode(0)
            state_result = self.robot_control.robot.set_state(state=0)
            
            if mode_result == 0 and state_result == 0:
                print("Robot mode and state set successfully")
            else:
                print(f"Warning: Mode set result: {mode_result}, State set result: {state_result}")
            
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            raise

    def load_poses(self):
        """Load poses and delays from transformed JSON file"""
        self.poses = []
        self.delays = []  # Add delays list
        try:
            with open("logs/operator_log_20250206_160527_transformed_to_real_robot.json", 'r') as f:
                data = json.load(f)
                self.poses = data['transformed_poses']
                self.delays = data['delays']  # Load delays from JSON
            print(f"Loaded {len(self.poses)} poses for replay")
        except Exception as e:
            print(f"Failed to load poses: {e}")

    def reset_robot(self):
        """Reset robot to initial state"""
        self.robot_control.robot.reset()
        return "Robot reset completed"

    def home_robot(self):
        """Home the robot"""
        current_mode = self.robot_control.robot.mode
        if current_mode != 0:
            self.robot_control.robot.set_mode(0)
            self.robot_control.robot.set_state(0)
        self.robot_control.home_robot()
        return "Robot homed successfully"

    def move_forward(self):
        """Move robot forward by 5mm"""
        cart_state = self.robot_control.get_cartesian_state()
        current_position = cart_state['position']
        current_orientation = cart_state['orientation']
        
        pose_aa = np.concatenate([current_position, current_orientation])
        pose_aa[0] += 5.0
        
        return self.robot_control.move_arm_cartesian(pose_aa)

    def get_status(self):
        """Get current robot status"""
        joint_state = self.robot_control.get_arm_joint_state()
        cart_state = self.robot_control.get_cartesian_state()
        return {
            'joint_positions': joint_state['position'],
            'cartesian_position': cart_state['position'],
            'orientation': cart_state['orientation']
        }

    def execute_pose(self, index):
        """Execute a specific pose from loaded poses"""
        print(f"Executing pose {index} of {len(self.poses)}")
        if 0 <= index < len(self.poses):
            pose = self.poses[index]
            # Split pose into position and quaternion
            position = np.array(pose[:3]) * 1000.0  # Convert position from m to mm
            quaternion = pose[3:]  # qx, qy, qz, qw
            
            # Convert quaternion to euler angles    
            euler_angles = Rotation.from_quat(quaternion).as_euler('xyz', degrees=False)
            
            print(f"Position (mm) (x,y,z): {position}")
            print(f"Quaternion (x,y,z,w): {quaternion}")
            print(f"Euler angles (rad) (x,y,z): {euler_angles}")

            # Create movement pose with position in mm and angles in degrees
            mvpose = np.concatenate([position, euler_angles])
            
            result = self.robot_control.move_arm_cartesian(mvpose)
            while self.robot_control.robot.get_is_moving():
                time.sleep(0.01)
                
            # Wait for the delay time from the recorded motion
            if index < len(self.delays):
                delay = self.delays[index]
                print(f"Waiting for {delay:.3f} seconds")
                time.sleep(delay)
                
            return result, position, quaternion
        return None, None, None

class RobotControlGUI:
    """
    Handles all GUI-related functionality and user interaction.
    Uses RobotController for actual robot control.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("XArm Robot Control Interface")
        
        # Initialize robot controller
        try:
            self.controller = RobotController(simulation_mode=False)
        except Exception as e:
            print(f"Failed to initialize robot controller: {e}")
            return

        # Initialize GUI-specific attributes
        self.current_pose_index = self.controller.current_pose_index
        self.poses = self.controller.poses

        self.setup_window()
        self.create_gui()

    def setup_window(self):
        """Configure window size and position"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width // 1.4)
        window_height = int(screen_height // 1.4)
        position_x = int((screen_width - window_width) // 1.4)
        position_y = int((screen_height - window_height) // 1.4)
        self.root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

    def create_gui(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid to expand with window
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Basic Controls - with larger font
        title_font = ('Arial', 14, 'bold')
        button_font = ('Arial', 12)
        
        ttk.Label(main_frame, text="Robot Controls", font=title_font).grid(row=0, column=0, columnspan=2, pady=20)
        
        ttk.Button(main_frame, text="Reset Robot", 
                  command=self.reset_robot,
                  style='Large.TButton').grid(row=1, column=0, sticky='ew', padx=10)
        
        ttk.Button(main_frame, text="Home Robot", 
                  command=self.home_robot,
                  style='Large.TButton').grid(row=1, column=1, sticky='ew', padx=10)

        ttk.Button(main_frame, text="Move forward",
                   command=self.move_forward,
                   style='Large.TButton').grid(row=2, column=0, sticky='ew', padx=10)
        # Status Display
        ttk.Label(main_frame, text="Robot Status", font=title_font).grid(row=2, column=0, columnspan=2, pady=20)
        
        self.status_text = tk.Text(main_frame, height=15, width=50, font=('Courier', 11))
        self.status_text.grid(row=3, column=0, columnspan=2, sticky='nsew', padx=10)

        # Update Status Button
        ttk.Button(main_frame, text="Update Status", 
                  command=self.update_status,
                  style='Large.TButton').grid(row=4, column=0, columnspan=2, sticky='ew', padx=10, pady=20)

        # Add Replay Controls Section
        ttk.Label(main_frame, text="Log Replay Controls", font=title_font).grid(row=5, column=0, columnspan=2, pady=20)
        
        # Frame for replay controls
        replay_frame = ttk.Frame(main_frame)
        replay_frame.grid(row=6, column=0, columnspan=2, sticky='ew', pady=10)
        
        # Add step size input
        ttk.Label(replay_frame, text="Steps per click:").grid(row=0, column=0, padx=5)
        self.step_size = ttk.Entry(replay_frame, width=5)
        self.step_size.insert(0, "1")  # Default step size
        self.step_size.grid(row=0, column=1, padx=5)
        
        ttk.Button(replay_frame, text="Previous Step", 
                  command=self.previous_pose,
                  style='Large.TButton').grid(row=0, column=2, padx=5)
                  
        ttk.Button(replay_frame, text="Next Step", 
                  command=self.next_pose,
                  style='Large.TButton').grid(row=0, column=3, padx=5)
        
        # Add pose counter display
        self.pose_counter = ttk.Label(replay_frame, 
                                    text=f"Pose: {self.current_pose_index + 1}/{len(self.controller.poses)}")
        self.pose_counter.grid(row=0, column=4, padx=10)

        # Configure button style
        style = ttk.Style()
        style.configure('Large.TButton', font=button_font, padding=10)

        # Add spacing between elements
        for child in main_frame.winfo_children():
            child.grid_configure(pady=5)

    def reset_robot(self):
        try:
            result = self.controller.reset_robot()
            self.status_text.insert(tk.END, f"{result}\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Reset failed: {str(e)}\n")
        self.status_text.see(tk.END)

    def home_robot(self):
        try:
            result = self.controller.home_robot()
            self.status_text.insert(tk.END, f"{result}\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Homing failed: {str(e)}\n")
            self.status_text.insert(tk.END, f"Traceback:\n{traceback.format_exc()}\n")
        self.status_text.see(tk.END)

    def move_forward(self):
        try:
            result = self.controller.move_forward()
            self.status_text.insert(tk.END, f"Move result: {result}\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Move forward failed with error: {str(e)}\n")
            self.status_text.insert(tk.END, f"Traceback:\n{traceback.format_exc()}\n")
        self.status_text.see(tk.END)

    def update_status(self):
        try:
            # Clear previous status
            self.status_text.delete(1.0, tk.END)
            
            # Get joint positions
            joint_state = self.controller.get_status()['joint_positions']
            self.status_text.insert(tk.END, f"Joint Positions: \n{joint_state}\n\n")
            
            # Get cartesian position
            cart_state = self.controller.get_status()['cartesian_position']
            self.status_text.insert(tk.END, f"Cartesian Position: \n{cart_state}\n")
            self.status_text.insert(tk.END, f"Orientation: \n{self.controller.get_status()['orientation']}\n")
            
        except Exception as e:
            self.status_text.insert(tk.END, f"Status update failed: {str(e)}\n")
        self.status_text.see(tk.END)

    def next_pose(self):
        try:
            step = int(self.step_size.get())
        except ValueError:
            self.status_text.insert(tk.END, "Invalid step size. Using defaults.\n")
            step = 1
            
        target_index = min(self.current_pose_index + step, len(self.controller.poses) - 1)
        
        if self.current_pose_index >= target_index:
            self.status_text.insert(tk.END, "Reached end of poses\n")
            return
            
        # Execute all poses between current and target
        for idx in range(self.current_pose_index, target_index + 1):
            self.execute_pose(idx)
            self.current_pose_index = idx
            self.pose_counter.config(text=f"Pose: {self.current_pose_index + 1}/{len(self.controller.poses)}")
            self.root.update()
        
    def previous_pose(self):
        try:
            step = int(self.step_size.get())
        except ValueError:
            self.status_text.insert(tk.END, "Invalid step size. Using defaults.\n")
            step = 1
            
        target_index = max(self.current_pose_index - step, 0)
        
        if self.current_pose_index <= target_index:
            self.status_text.insert(tk.END, "At beginning of poses\n")
            return
            
        # Execute all poses between current and target (in reverse)
        for idx in range(self.current_pose_index - 1, target_index - 1, -1):
            self.execute_pose(idx)
            self.current_pose_index = idx
            self.pose_counter.config(text=f"Pose: {self.current_pose_index + 1}/{len(self.controller.poses)}")
            self.root.update()
        
    def execute_pose(self, index):
        try:
            while self.controller.robot_control.robot.get_is_moving():
                time.sleep(0.01)
            result, position, quaternion = self.controller.execute_pose(index)
            if result is not None and position is not None:
                self.status_text.insert(tk.END, f"\nExecuting pose {index + 1}:\n")
                self.status_text.insert(tk.END, f"Position (mm) (x,y,z): {position}\n")
                self.status_text.insert(tk.END, f"Quaternion (x,y,z,w): {quaternion}\n")
                self.status_text.insert(tk.END, f"Move result: {result}\n")
            else:
                self.status_text.insert(tk.END, "Failed to execute pose: Invalid index\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Failed to execute pose: {str(e)}\n")
        self.status_text.see(tk.END)

    def close_GUI(self):
        self.root.destroy()

def main_GUI():
    root = tk.Tk()
    app = RobotControlGUI(root)
    root.mainloop()

def main_Parsing():
    log_to_sim = Log2Xarm()

if __name__ == "__main__":
    if input("Select 1 for parsing, 2 for GUI: ") == "1":
        main_Parsing() 
    else:
        main_GUI()