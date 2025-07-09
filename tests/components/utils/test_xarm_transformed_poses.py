import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import json
from scipy.spatial.transform import Rotation


def load_pose_log(log_file):
    """Load pose log file."""
    with open(log_file, 'r') as f:
        return json.load(f)
    

def test_xarm_transformed_poses():
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Enable GUI controls
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
    
    # Set up simulation environment
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Load assets
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    table_urdf_path = os.path.join(current_dir, "beavr/src/components/environment/assets/urdf/table/square_table.urdf")
    xarm_urdf_path = os.path.join(current_dir, "beavr/src/components/environment/assets/urdf/leap_xarm7/leap_xarm7.urdf")
    
    # Load table
    table_height = 0.3
    table = p.loadURDF(table_urdf_path, basePosition=[0, 0, table_height/2], useFixedBase=True)
    
    # Load xArm7
    robot_id = p.loadURDF(xarm_urdf_path, basePosition=[0, 0, table_height], useFixedBase=True)
    
    # Set camera view
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
    
    # Print information about specific links
    print("\nLink Information:")
    print("-" * 50)
    for i in [6, 7, 8]:  # Only print links 6, 7, 8
        link_state = p.getLinkState(robot_id, i)
        link_info = p.getJointInfo(robot_id, i)  # To get link name
        print(f"Link {i}: {link_info[12].decode('utf-8')}")  # Link name
        print(f"  Position: {link_state[0]}")
        print(f"  Orientation (quaternion): {link_state[1]}")
        print(f"  Local inertial pos: {link_state[2]}")
        print(f"  Local inertial orn: {link_state[3]}")
        print(f"  World Link pos: {link_state[4]}")
        print(f"  World Link orn: {link_state[5]}")
        print("-" * 50)
    
    # Get end effector index
    end_effector_index = 8  # link_eef
    ee_state = p.getLinkState(robot_id, end_effector_index)
    print(f"\nEnd Effector (Link {end_effector_index}):")
    print(f"World Position: {ee_state[4]}")  # World link frame position
    print(f"World Orientation: {ee_state[5]}")  # World link frame orientation
    
    # Load pose log
    # log_file = os.path.join(current_dir, "tests/calc_pose_log_20250123_123125.json")
    log_file = os.path.join(current_dir, "logs/operator_log_20250204_144117.json")
    pose_log = load_pose_log(log_file)
    
    # Track base pose
    pose_log = pose_log['frames']
    
    # Get initial robot pose from log
    initial_pose = np.array(pose_log[str(0)]['init_robot_pose'])
    print("\nInitial Robot Pose from Log:")
    print(f"Position: {initial_pose[:3, 3]}")
    print(f"Rotation:\n{initial_pose[:3, :3]}")
    
    # Get movable joints
    movable_joint_indices = []
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED:
            movable_joint_indices.append(i)
    
    # Add debug parameters
    scale_slider = p.addUserDebugParameter("Position Scale", 0.1, 2.0, 1.0)
    speed_slider = p.addUserDebugParameter("Playback Speed", 0.1, 2.0, 1.0)
    pause_button = p.addUserDebugParameter("Pause/Resume", 1, 0, 1)
    reset_button = p.addUserDebugParameter("Reset", 1, 0, 1)
    
    # Add visualization for target position
    target_visual_id = None
    last_button_state = 1
    paused = False
    frame_idx = 0
    last_time = time.time()
    target_fps = 25.0  # Original log FPS
    
    # Add parameters for error checking
    MAX_POSITION_CHANGE = 0.1  # meters
    last_target_pos = None
    skipped_frames = 0
    
    def reset_simulation():
        """Reset robot and tracking variables."""
        nonlocal frame_idx, last_target_pos, skipped_frames, last_time
        
        # Reset frame counter and tracking
        frame_idx = 0
        last_target_pos = None
        skipped_frames = 0
        last_time = time.time()
        
        # Reset robot joints to home position
        for joint in range(p.getNumJoints(robot_id)):
            if p.getJointInfo(robot_id, joint)[2] != p.JOINT_FIXED:
                p.resetJointState(robot_id, joint, 0)
        
        # Reset camera view
        p.resetDebugVisualizerCamera(1.5, 0, -45, [0, 0, 0])
        
        # Remove target visualization
        nonlocal target_visual_id
        if target_visual_id is not None:
            p.removeUserDebugItem(target_visual_id)
            target_visual_id = None
    
    hasPrevPose = 0
    while True:
        try:
            current_time = time.time()
            
            # Handle pause button
            button_state = p.readUserDebugParameter(pause_button)
            if button_state != last_button_state:
                paused = not paused
                last_button_state = button_state
            
            # Handle reset button with proper reset function
            if p.readUserDebugParameter(reset_button) != 1:
                reset_simulation()
                continue  # Skip this frame to ensure clean reset
            
            if not paused and frame_idx < len(pose_log):
                playback_speed = p.readUserDebugParameter(speed_slider)
                desired_dt = (1.0 / target_fps) / playback_speed
                
                if current_time - last_time >= desired_dt:
                    frame = pose_log[str(frame_idx)]
                    transformed_pose = np.array(frame['transformed_pose'])
                    
                    """
                    # Calculate relative motion from initial pose
                    motion = np.linalg.inv(base_pose) @ transformed_pose
                    """
                    # Use transformed pose directly
                    motion = transformed_pose
                    
                    # Debug print transformations
                    if frame_idx < 5 or frame_idx > 60:  # Print first 5 and after frame 60
                        print(f"\nFrame {frame_idx} Transformation:")
                        print("Current Position:", motion[:3, 3])
                        if last_target_pos is not None:
                            delta = motion[:3, 3] - last_target_pos
                            print("Delta:", delta)
                            print("Delta magnitude:", np.linalg.norm(delta))
                    
                    # Extract position and orientation
                    position_scale = p.readUserDebugParameter(scale_slider)
                    target_pos = motion[:3, 3] * position_scale
                                        
                    # Check for large position changes
                    if last_target_pos is not None:
                        position_change = np.linalg.norm(target_pos - last_target_pos)
                        if position_change > MAX_POSITION_CHANGE:
                            print(f"\nSkipping frame {frame_idx}")
                            print(f"Current pos: {target_pos}")
                            print(f"Last pos: {last_target_pos}")
                            print(f"Change: {position_change:.3f}m")
                            skipped_frames += 1
                            frame_idx += 1
                            continue
                    
                    # Convert rotation matrix to quaternion
                    target_orn = Rotation.from_matrix(
                        motion[:3, :3]
                    ).as_quat()
                    
                    # Update visualization and calculate IK
                    if target_visual_id is not None:
                        p.removeUserDebugItem(target_visual_id)
                    target_visual_id = p.addUserDebugLine(
                        target_pos,
                        [target_pos[0], target_pos[1], target_pos[2] + 0.1],
                        [1, 0, 0], 2
                    )
                    
                    # Calculate IK
                    joint_poses = p.calculateInverseKinematics(
                        robot_id,
                        end_effector_index,
                        target_pos,
                        target_orn,
                        maxNumIterations=100,
                        residualThreshold=.01
                    )
                    
                    # Get current state and print debug info
                    current_state = p.getLinkState(robot_id, end_effector_index)
                    current_pos = current_state[0]
                    current_orn = current_state[1]
                    
                    print(f"\rFrame {frame_idx}/{len(pose_log)} | "
                          f"Error: pos={np.linalg.norm(np.array(current_pos) - np.array(target_pos)):.3f}m | "
                          f"Skipped: {skipped_frames}", 
                          end="")
                    
                    # Apply IK with velocity limits
                    for i, joint_idx in enumerate(movable_joint_indices):
                        if i < len(joint_poses):
                            p.setJointMotorControl2(
                                bodyIndex=robot_id,
                                jointIndex=joint_idx,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_poses[i],
                                maxVelocity=2.0
                            )
                    
                    last_target_pos = target_pos
                    frame_idx += 1
                    last_time = current_time
            
                    trailDuration = 1000
                    if (hasPrevPose):
                        p.addUserDebugLine(prevPose, current_pos, [0, 0, 0.3], 1, trailDuration)
                        p.addUserDebugLine(prevPose1, current_state[4], [1, 0, 0], 1, trailDuration)
                    prevPose = current_pos
                    prevPose1 = current_state[4]
                    hasPrevPose = 1

            p.stepSimulation()
            time.sleep(1./240.)
            
        except KeyboardInterrupt:
            print(f"\nTotal frames skipped: {skipped_frames}")
            break
        
    input("Press Enter to continue...")
    p.disconnect()

if __name__ == "__main__":
    test_xarm_transformed_poses()