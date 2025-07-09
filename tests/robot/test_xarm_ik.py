import pybullet as p
import pybullet_data
import time
import numpy as np
import os


def test_xarm_ik():
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Enable GUI controls and rendering
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
    
    # Set up simulation environment
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Load table and robot like in your test
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    table_urdf_path = os.path.join(current_dir, "beavr/src/components/environment/assets/urdf/table/square_table.urdf")
    xarm_urdf_path = os.path.join(current_dir, "beavr/src/components/environment/assets/urdf/leap_xarm7/leap_xarm7.urdf")
    
    # Load table
    table_height = 0.3
    table = p.loadURDF(table_urdf_path, basePosition=[0, 0, table_height/2], useFixedBase=True)
    
    # Load xArm7
    p.setAdditionalSearchPath(os.path.dirname(xarm_urdf_path))
    robot_id = p.loadURDF(xarm_urdf_path, basePosition=[0, 0, table_height], useFixedBase=True)
    
    # Set camera view
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
    
    # Print joint information to understand the structure
    print("\nJoint Information:")
    print("-" * 50)
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}:")
        print(f"  Name: {joint_info[1].decode('utf-8')}")
        print(f"  Type: {joint_info[2]}")  # JOINT_FIXED=4, JOINT_REVOLUTE=0
        print(f"  Lower limit: {joint_info[8]}")
        print(f"  Upper limit: {joint_info[9]}")
        print("-" * 50)

    # Get number of joints
    end_effector_index = 6  # The arm end-effector link should be at index 6
    
    # Get number of joints and identify movable joints
    movable_joint_indices = []
    for i in range(num_joints):
        if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED:
            movable_joint_indices.append(i)
    
    # Create sliders for target position
    target_x = p.addUserDebugParameter("Target X", -1, 1, 0.3)
    target_y = p.addUserDebugParameter("Target Y", -1, 1, 0)
    target_z = p.addUserDebugParameter("Target Z", 0, 1.5, 0.5)
    
    # Create sliders for target orientation (Euler angles)
    target_roll = p.addUserDebugParameter("Target Roll", -3.14, 3.14, 0)
    target_pitch = p.addUserDebugParameter("Target Pitch", -3.14, 3.14, 0)
    target_yaw = p.addUserDebugParameter("Target Yaw", -3.14, 3.14, 0)
    
    # Add debug line for visualizing target
    target_visual_id = None
    
    while True:
        try:
            # Get target position from sliders
            target_pos = [
                p.readUserDebugParameter(target_x),
                p.readUserDebugParameter(target_y),
                p.readUserDebugParameter(target_z)
            ]
            
            # Get target orientation from sliders (Euler angles)
            euler = [
                p.readUserDebugParameter(target_roll),
                p.readUserDebugParameter(target_pitch),
                p.readUserDebugParameter(target_yaw)
            ]
            target_orn = p.getQuaternionFromEuler(euler)
            
            # Update target visualization
            if target_visual_id is not None:
                p.removeUserDebugItem(target_visual_id)
            target_visual_id = p.addUserDebugLine(target_pos, 
                                                [target_pos[0], target_pos[1], target_pos[2] + 0.1],
                                                [1, 0, 0], 2)
            
            # Calculate IK
            joint_poses = p.calculateInverseKinematics(
                robot_id,
                end_effector_index,
                target_pos,
                target_orn,
                maxNumIterations=100,
                residualThreshold=.01
            )
            
            # Get current end effector state
            current_state = p.getLinkState(robot_id, end_effector_index)
            current_pos = current_state[0]
            current_orn = current_state[1]
            
            # Print debug info
            print("\rCurrent pos: {:.3f}, {:.3f}, {:.3f} | ".format(*current_pos) + 
                  "Target pos: {:.3f}, {:.3f}, {:.3f} | ".format(*target_pos) + 
                  "Error: {:.3f}".format(np.linalg.norm(np.array(current_pos) - np.array(target_pos))), 
                  end="")
            
            # Apply IK solution only to movable joints
            for i, joint_idx in enumerate(movable_joint_indices):
                if i < len(joint_poses):  # Make sure we don't exceed joint_poses length
                    p.setJointMotorControl2(
                        bodyIndex=robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_poses[i],
                        maxVelocity=1.0
                    )
            
            p.stepSimulation()
            time.sleep(1./240.)
            
        except KeyboardInterrupt:
            break
    
    p.disconnect()

if __name__ == "__main__":
    test_xarm_ik()