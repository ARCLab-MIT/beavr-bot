import pybullet as p
import pybullet_data
import numpy as np
import os
import time


def test_xarm_joints_setup():
    """Test XArm joints with exact same setup as XArmEnv."""
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load ground plane at z=0
    plane_id = p.loadURDF("plane.urdf")
    
    # Get paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load table
    table_urdf_path = os.path.join(
        current_dir,
        "beavr",
        "src",
        "components",
        "environment",
        "assets",
        "urdf",
        "table",
        "square_table.urdf"
    )
    
    # Table dimensions (same as XArmEnv)
    table_height = 0.3
    
    # Place table with bottom at z=0
    table_pos_z = table_height/2
    table_id = p.loadURDF(
        table_urdf_path,
        [0.0, 0.0, table_pos_z],
        [0, 0, 0, 1],
        useFixedBase=True
    )
    
    # Load robot on top of table
    xarm_urdf_path = os.path.join(
        current_dir,
        "beavr",
        "src",
        "components",
        "environment",
        "assets",
        "urdf",
        "leap_xarm7",
        "leap_xarm7.urdf"
    )
    
    robot_id = p.loadURDF(
        xarm_urdf_path,
        [0.0, 0.0, table_height],
        [0, 0, 0, 1],
        useFixedBase=True
    )
    
    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=0,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # Constants matching XArmEnv
    end_effector_index = 8
    ref_link = 0
    
    # Get movable joint indices (first 7 joints only)
    movable_joint_indices = []
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED:
            movable_joint_indices.append(i)
    movable_joint_indices = movable_joint_indices[:7]
    
    print("\nJoint Configuration:")
    print("-" * 50)
    print(f"End effector index: {end_effector_index}")
    print(f"Movable joint indices: {movable_joint_indices}")
    
    # Print initial state
    print("\nInitial Joint States:")
    print("-" * 50)
    for idx in movable_joint_indices:
        joint_info = p.getJointInfo(robot_id, idx)
        joint_state = p.getJointState(robot_id, idx)
        print(f"Joint {idx} ({joint_info[1].decode('utf-8')}):")
        print(f"  Position: {joint_state[0]}")
        print(f"  Velocity: {joint_state[1]}")
        print(f"  Limits: [{joint_info[8]}, {joint_info[9]}]")
    
    # Test movement sequence
    print("\nTesting joint movement sequence...")
    print("-" * 50)
    
    try:
        while True:
            # Test 1: Move to home position
            print("\nMoving to home position...")
            for joint_idx in movable_joint_indices:
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0,
                    maxVelocity=1.0
                )
            
            # Step simulation for movement to complete
            for _ in range(100):  # Simulate for 100 steps
                p.stepSimulation()
                time.sleep(1./240.)
            
            # Test 2: Move each joint individually
            for joint_idx in movable_joint_indices:
                joint_name = p.getJointInfo(robot_id, joint_idx)[1].decode('utf-8')
                print(f"\nTesting joint {joint_idx} ({joint_name})")
                
                # Move to positive position
                print("Moving to positive position...")
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0.5,
                    maxVelocity=1.0
                )
                
                # Step simulation for movement to complete
                for _ in range(100):
                    p.stepSimulation()
                    time.sleep(1./240.)
                
                # Get end effector position after moving
                state = p.getLinkState(robot_id, end_effector_index)
                print(f"End effector position at +0.5: {state[0]}")
                
                # Move to negative position
                print("Moving to negative position...")
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=-0.5,
                    maxVelocity=1.0
                )
                
                # Step simulation for movement to complete
                for _ in range(100):
                    p.stepSimulation()
                    time.sleep(1./240.)
                
                state = p.getLinkState(robot_id, end_effector_index)
                print(f"End effector position at -0.5: {state[0]}")
                
                # Reset joint
                print("Resetting joint...")
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0,
                    maxVelocity=1.0
                )
                
                # Step simulation for movement to complete
                for _ in range(100):
                    p.stepSimulation()
                    time.sleep(1./240.)
                
                state = p.getLinkState(robot_id, end_effector_index)
                print(f"End effector position after reset: {state[0]}")
            
            # Test 3: Test IK to a target position
            print("\nTesting IK movement...")
            target_pos = [0.3, 0.0, 0.5]
            target_orn = p.getQuaternionFromEuler([0, 0, 0])
            
            # Get current joint positions
            current_joints = [p.getJointState(robot_id, i)[0] for i in movable_joint_indices]
            
            # Calculate IK
            joint_poses = p.calculateInverseKinematics(
                robot_id,
                end_effector_index,
                target_pos,
                target_orn,
                maxNumIterations=100,
                residualThreshold=.01
            )
            
            print("Applying IK solution...")
            # Apply IK solution
            for i, joint_idx in enumerate(movable_joint_indices):
                if i < len(joint_poses):
                    print(f"Setting joint {joint_idx} to {joint_poses[i]}")
                    p.setJointMotorControl2(
                        bodyIndex=robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_poses[i],
                        maxVelocity=1.0
                    )
            
            # Step simulation for IK movement to complete
            for _ in range(200):  # More steps for IK movement
                p.stepSimulation()
                time.sleep(1./240.)
            
            # Verify end effector position
            state = p.getLinkState(robot_id, end_effector_index)
            print(f"\nTarget position: {target_pos}")
            print(f"Achieved position: {state[0]}")
            print(f"Position error: {np.array(state[0]) - np.array(target_pos)}")
            
            # Print final joint positions
            print("\nFinal joint positions:")
            for idx in movable_joint_indices:
                joint_state = p.getJointState(robot_id, idx)
                print(f"Joint {idx}: {joint_state[0]}")
            
            time.sleep(1)  # Pause before next cycle
            
            # Compare end effector indices 7 and 8
            print("\nComparing End Effector Indices:")
            print("-" * 50)
            
            # Get info for both potential end effectors
            link_7_info = p.getJointInfo(robot_id, 7)
            link_8_info = p.getJointInfo(robot_id, 8)
            state_7 = p.getLinkState(robot_id, 7)
            state_8 = p.getLinkState(robot_id, 8)
            
            print("\nLink 7:")
            print(f"  Name: {link_7_info[12].decode('utf-8')}")
            print(f"  Position: {state_7[0]}")
            print(f"  Orientation: {state_7[1]}")
            
            print("\nLink 8:")
            print(f"  Name: {link_8_info[12].decode('utf-8')}")
            print(f"  Position: {state_8[0]}")
            print(f"  Orientation: {state_8[1]}")
            
            print("\nPosition difference (8 - 7):")
            pos_diff = np.array(state_8[0]) - np.array(state_7[0])
            print(f"  {pos_diff}")
            
            # Test IK with both end effectors
            target_pos = [0.3, 0.0, 0.5]
            target_orn = p.getQuaternionFromEuler([0, 0, 0])
            
            print("\nTesting IK for both end effectors:")
            print("-" * 50)
            
            # Test with end_effector_index = 7
            joint_poses_7 = p.calculateInverseKinematics(
                robot_id,
                7,
                target_pos,
                target_orn,
                maxNumIterations=100,
                residualThreshold=.01
            )
            
            # Test with end_effector_index = 8
            joint_poses_8 = p.calculateInverseKinematics(
                robot_id,
                8,
                target_pos,
                target_orn,
                maxNumIterations=100,
                residualThreshold=.01
            )
            
            print("\nIK Solutions:")
            print(f"End effector 7: {joint_poses_7[:7]}")
            print(f"End effector 8: {joint_poses_8[:7]}")
            print("\nDifference in solutions (8 - 7):")
            solution_diff = np.array(joint_poses_8[:7]) - np.array(joint_poses_7[:7])
            print(f"{solution_diff}")
            
            # Apply both solutions and check results
            print("\nTesting both solutions:")
            
            # Test solution for end effector 7
            print("\nApplying solution for end effector 7:")
            for i, joint_idx in enumerate(movable_joint_indices):
                if i < len(joint_poses_7):
                    p.setJointMotorControl2(
                        bodyIndex=robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_poses_7[i],
                        maxVelocity=1.0
                    )
            
            # Step simulation
            for _ in range(100):
                p.stepSimulation()
                time.sleep(1./240.)
            
            # Check positions
            state_7 = p.getLinkState(robot_id, 7)
            state_8 = p.getLinkState(robot_id, 8)
            print(f"Target position: {target_pos}")
            print(f"Achieved position (link 7): {state_7[0]}")
            print(f"Achieved position (link 8): {state_8[0]}")
            print(f"Error (link 7): {np.array(state_7[0]) - np.array(target_pos)}")
            print(f"Error (link 8): {np.array(state_8[0]) - np.array(target_pos)}")
            
            time.sleep(2)  # Pause to observe
            
            # Test solution for end effector 8
            print("\nApplying solution for end effector 8:")
            for i, joint_idx in enumerate(movable_joint_indices):
                if i < len(joint_poses_8):
                    p.setJointMotorControl2(
                        bodyIndex=robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_poses_8[i],
                        maxVelocity=1.0
                    )
            
            # Step simulation
            for _ in range(100):
                p.stepSimulation()
                time.sleep(1./240.)
            
            # Check positions again
            state_7 = p.getLinkState(robot_id, 7)
            state_8 = p.getLinkState(robot_id, 8)
            print(f"Target position: {target_pos}")
            print(f"Achieved position (link 7): {state_7[0]}")
            print(f"Achieved position (link 8): {state_8[0]}")
            print(f"Error (link 7): {np.array(state_7[0]) - np.array(target_pos)}")
            print(f"Error (link 8): {np.array(state_8[0]) - np.array(target_pos)}")
            
            input("\nPress Enter to exit...")
            p.disconnect()
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    p.disconnect()

if __name__ == "__main__":
    test_xarm_joints_setup()