import pybullet as p
import pybullet_data
import time
import os
import argparse
import re

import tkinter as tk
from tkinter import simpledialog
import traceback


def test_leap_hand_joint_limits():
    """
    Test the joint limits of the Leap Hand in PyBullet.
    This helps identify if joint limits are causing problems in IK calculations.
    """
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set up simulation environment
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Get path to Leap Hand URDF
    hand_urdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "beavr",
        "src",
        "components",
        "environment",
        "assets",
        "urdf",
        "leap_hand",
        "leap_hand_right.urdf"
    )
    
    # Add the URDF directory to the search path so PyBullet can find the meshes
    p.setAdditionalSearchPath(os.path.dirname(hand_urdf_path))
    
    # Load Leap Hand
    hand_id = p.loadURDF(hand_urdf_path, 
                       basePosition=[0, 0, 0],
                       useFixedBase=True)
    
    # Disable collisions between links
    for link1 in range(p.getNumJoints(hand_id)):
        for link2 in range(link1+1, p.getNumJoints(hand_id)):
            p.setCollisionFilterPair(hand_id, hand_id, link1, link2, 0)
    
    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # Get joint information
    num_joints = p.getNumJoints(hand_id)
    print(f"\nTotal number of joints: {num_joints}")
    
    # Collect movable joints and their limits
    movable_joints = []
    joint_limits = {}
    finger_joints = {
        "thumb": [],
        "index": [],
        "middle": [],
        "ring": [],
        "pinky": []
    }
    
    # First, print all joint names to see what we're working with
    print("\nAll Joint Names:")
    for i in range(num_joints):
        joint_info = p.getJointInfo(hand_id, i)
        joint_name = joint_info[1].decode('utf-8')
        print(f"Joint {i}: {joint_name}")

    print("\nJoint Information:")
    print("-" * 50)
    for i in range(num_joints):
        joint_info = p.getJointInfo(hand_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        
        print(f"Joint {i}: {joint_name}")
        print(f"  Type: {joint_type}")
        print(f"  Lower limit: {lower_limit}")
        print(f"  Upper limit: {upper_limit}")
        print(f"  Max force: {joint_info[10]}")
        print(f"  Max velocity: {joint_info[11]}")
        print("-" * 50)
        
        # Only consider movable joints
        if joint_type != p.JOINT_FIXED:
            movable_joints.append(i)
            joint_limits[i] = (lower_limit, upper_limit)
            
            # Manual mapping based on the joint indices
            if i in [16, 17, 18, 19]:  # Joints n, m, o, p
                finger_joints["thumb"].append(i)
            elif i in [1, 2, 3, 4]:    # Joints a, b, c, d
                finger_joints["index"].append(i)
            elif i in [6, 7, 8, 9]:    # Joints e, f, g, h
                finger_joints["middle"].append(i)
            elif i in [11, 12, 13, 14]: # Joints i, j, k, l
                finger_joints["ring"].append(i)
    
    print(f"\nMovable joints: {len(movable_joints)}")
    for finger, joints in finger_joints.items():
        print(f"{finger.capitalize()} finger joints: {joints}")
    
    # Create sliders for each movable joint
    joint_sliders = {}
    for joint_idx in movable_joints:
        joint_info = p.getJointInfo(hand_id, joint_idx)
        joint_name = joint_info[1].decode('utf-8')
        lower, upper = joint_limits[joint_idx]
        
        # If limits are invalid (0,0), use reasonable defaults
        if lower == upper:
            lower, upper = -3.14, 3.14
            
        # Create slider with joint limits
        joint_sliders[joint_idx] = p.addUserDebugParameter(
            f"Joint {joint_idx}: {joint_name}",
            lower,
            upper,
            0.0  # Initial value
        )
    
    # Add buttons for testing preset poses
    open_hand_button = p.addUserDebugParameter("Open Hand", 1, 0, 1)
    close_hand_button = p.addUserDebugParameter("Close Hand", 1, 0, 1)
    last_open_state = 1
    last_close_state = 1
    
    # Add test mode selector
    test_mode = p.addUserDebugParameter("Test Mode (0=Manual, 1=Auto)", 0, 1, 0)
    auto_test_speed = p.addUserDebugParameter("Auto Test Speed", 0.1, 2.0, 0.5)
    
    # Add this after your other buttons
    apply_array_button = p.addUserDebugParameter("Apply Joint Array", 1, 0, 1)
    last_array_button_state = 1
    
    # Auto test variables
    current_joint_idx = 0
    test_direction = 1  # 1 = increasing, -1 = decreasing
    auto_test_timer = time.time()
    
    # Add these variables at the beginning of the function
    applied_joint_values = {}  # Store the values applied from the array
    use_applied_values = False  # Flag to use applied values instead of sliders
    
    # Add a reset button at the beginning of the script
    reset_button = p.addUserDebugParameter("Reset Hand", 1, 0, 1)
    last_reset_state = 1
    
    try:
        while True:
            # Check for preset buttons
            open_state = p.readUserDebugParameter(open_hand_button)
            if open_state != last_open_state:
                # Open hand - set all joints to minimum values
                for joint_idx in movable_joints:
                    p.setJointMotorControl2(
                        bodyIndex=hand_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=0.0
                    )
                last_open_state = open_state
            
            close_state = p.readUserDebugParameter(close_hand_button)
            if close_state != last_close_state:
                # Close hand - set all joints to maximum values
                for joint_idx in movable_joints:
                    lower, upper = joint_limits[joint_idx]
                    # Use upper limit if valid, otherwise use a default value
                    target_pos = upper if upper != 0 else 1.5
                    p.setJointMotorControl2(
                        bodyIndex=hand_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_pos
                    )
                last_close_state = close_state
            
            # Get current test mode
            mode = int(p.readUserDebugParameter(test_mode))
            
            # Then check it in the main loop
            reset_state = p.readUserDebugParameter(reset_button)
            if reset_state != last_reset_state:
                # Reset all joints to zero
                for joint_idx in movable_joints:
                    p.resetJointState(
                        bodyUniqueId=hand_id,
                        jointIndex=joint_idx,
                        targetValue=0.0
                    )
                    p.setJointMotorControl2(
                        bodyIndex=hand_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=0.0,
                        force=100  # Use moderate force
                    )
                
                # Turn off array mode
                use_applied_values = False
                print("Hand reset to neutral position")
                
                last_reset_state = reset_state
            
            if mode == 0:  # Manual mode
                # Check if we should use applied values or sliders
                if use_applied_values:
                    # Apply the stored values with moderate force
                    for joint_idx, value in applied_joint_values.items():
                        p.setJointMotorControl2(
                            bodyIndex=hand_id,
                            jointIndex=joint_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=value,
                            force=100,  # Moderate force
                            positionGain=0.5,  # Lower gain
                            velocityGain=0.5   # Lower gain
                        )
                else:
                    # Use slider values as normal
                    for joint_idx, slider_id in joint_sliders.items():
                        target_pos = p.readUserDebugParameter(slider_id)
                        p.setJointMotorControl2(
                            bodyIndex=hand_id,
                            jointIndex=joint_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_pos,
                            force=100  # Moderate force
                        )
            else:  # Auto test mode
                speed = p.readUserDebugParameter(auto_test_speed)
                current_time = time.time()
                
                # Update every 0.05 seconds adjusted by speed
                if current_time - auto_test_timer > (0.05 / speed):
                    # Get current joint being tested
                    joint_idx = movable_joints[current_joint_idx]
                    joint_info = p.getJointInfo(hand_id, joint_idx)
                    joint_name = joint_info[1].decode('utf-8')
                    lower, upper = joint_limits[joint_idx]
                    
                    # If limits are invalid, use reasonable defaults
                    if lower == upper:
                        lower, upper = -3.14, 3.14
                    
                    # Get current position
                    current_pos = p.getJointState(hand_id, joint_idx)[0]
                    
                    # Calculate new position
                    new_pos = current_pos + (0.05 * test_direction)
                    
                    # Check if we've reached a limit
                    if new_pos >= upper:
                        new_pos = upper
                        test_direction = -1
                    elif new_pos <= lower:
                        new_pos = lower
                        test_direction = 1
                        # Move to next joint when we complete a cycle
                        current_joint_idx = (current_joint_idx + 1) % len(movable_joints)
                    
                    # Apply position
                    p.setJointMotorControl2(
                        bodyIndex=hand_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=new_pos
                    )
                    
                    # Print current joint info
                    print(f"\rTesting Joint {joint_idx}: {joint_name} | "
                          f"Position: {new_pos:.2f} | "
                          f"Limits: [{lower:.2f}, {upper:.2f}]", end="")
                    
                    auto_test_timer = current_time
            
            # Check for array input button
            array_button_state = p.readUserDebugParameter(apply_array_button)
            if array_button_state != last_array_button_state:
                try:
                    
                    # Create a root window but keep it hidden
                    root = tk.Tk()
                    root.withdraw()
                    
                    # Use a simple string dialog instead of a custom dialog
                    array_str = simpledialog.askstring(
                        "Joint Array Input", 
                        "Paste your JSON array below:",
                        parent=root
                    )
                    
                    if array_str:  # Only proceed if the user didn't cancel
                        # Extract all numbers from the input string using regex
                        numbers = re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', array_str)
                        joint_values = [float(num) for num in numbers]
                        
                        # Check if we have enough values
                        if len(joint_values) < len(movable_joints):
                            print(f"Warning: Only {len(joint_values)} values provided for {len(movable_joints)} joints.")
                            print("Applying values to the first joints only.")
                        
                        # Apply values to joints with much higher force
                        for i, joint_idx in enumerate(movable_joints):
                            if i < len(joint_values):
                                joint_info = p.getJointInfo(hand_id, joint_idx)
                                joint_name = joint_info[1].decode('utf-8')
                                
                                # First reset the joint to ensure it moves
                                p.resetJointState(
                                    bodyUniqueId=hand_id,
                                    jointIndex=joint_idx,
                                    targetValue=joint_values[i]
                                )
                                
                                # Then apply motor control with moderate force
                                p.setJointMotorControl2(
                                    bodyIndex=hand_id,
                                    jointIndex=joint_idx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_values[i],
                                    force=100,  # Use moderate force
                                    positionGain=0.5,  # Lower position gain
                                    velocityGain=0.5   # Lower velocity gain
                                )
                                
                                print(f"Applied value {joint_values[i]} to joint {joint_idx}: {joint_name}")
                        
                        # Store the applied values and set the flag
                        applied_joint_values = {movable_joints[i]: joint_values[i] for i in range(min(len(movable_joints), len(joint_values)))}
                        use_applied_values = True
                        
                        # Add a small delay to allow the simulation to update
                        for _ in range(10):
                            p.stepSimulation()
                            time.sleep(0.01)
                    
                    # Destroy the root window
                    root.destroy()
                    
                except Exception as e:
                    print(f"Error applying array: {e}")
                    traceback.print_exc()  # Print the full error traceback
                    print("Please paste any text containing the joint values")
                
                last_array_button_state = array_button_state
            
            # Step simulation
            p.stepSimulation()
            time.sleep(1./240.)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Leap Hand Joint Limits Test')
    args = parser.parse_args()
    test_leap_hand_joint_limits()