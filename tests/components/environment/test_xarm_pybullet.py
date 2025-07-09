import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import argparse


def test_xarm():
    """Base test showing link/joint structure and movement."""
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load xArm7
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xarm_urdf_path = os.path.join(current_dir, "beavr/src/components/environment/assets/urdf/leap_xarm7/leap_xarm7.urdf")
    robot_id = p.loadURDF(xarm_urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    
    # Print information about joints and links
    num_joints = p.getNumJoints(robot_id)
    print(f"\nTotal number of joints: {num_joints}")
    print("\nJoint/Link Information:")
    print("-" * 50)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        link_name = joint_info[12].decode('utf-8')  # Link name
        joint_name = joint_info[1].decode('utf-8')  # Joint name
        joint_type = joint_info[2]  # Joint type
        
        print(f"\nJoint/Link {i}:")
        print(f"  Link Name: {link_name}")
        print(f"  Joint Name: {joint_name}")
        print(f"  Joint Type: {joint_type}")
        
        # Get link state
        state = p.getLinkState(robot_id, i)
        print(f"  Position: {state[0]}")
        print(f"  Orientation: {state[1]}")
    
    input("\nPress Enter to exit...")
    p.disconnect()

def test_xarm_link(link_index=8):
    """Test to examine a specific link."""
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xarm_urdf_path = os.path.join(current_dir, "beavr/src/components/environment/assets/urdf/leap_xarm7/leap_xarm7.urdf")
    robot_id = p.loadURDF(xarm_urdf_path, [0, 0, 0], useFixedBase=True)
    
    # Get detailed information about the specified link
    joint_info = p.getJointInfo(robot_id, link_index)
    link_state = p.getLinkState(robot_id, link_index)
    
    print(f"\nDetailed Information for Link {link_index}:")
    print("-" * 50)
    print(f"Link Name: {joint_info[12].decode('utf-8')}")
    print(f"Joint Name: {joint_info[1].decode('utf-8')}")
    print(f"Joint Type: {joint_info[2]}")
    print(f"Position: {link_state[0]}")
    print(f"Orientation (quaternion): {link_state[1]}")
    print(f"Orientation (euler): {p.getEulerFromQuaternion(link_state[1])}")
    
    input("\nPress Enter to exit...")
    p.disconnect()

def test_xarm_table():
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
    
    # Load table
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    
    # Table dimensions
    table_height = 0.3
    table = p.loadURDF(table_urdf_path, 
                      basePosition=[0, 0, table_height/2],
                      useFixedBase=True)
    
    # Get path to xArm URDF
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
    
    # Add the URDF directory to the search path for meshes
    p.setAdditionalSearchPath(os.path.dirname(xarm_urdf_path))
    
    # Load xArm7 with hand
    robot_id = p.loadURDF(xarm_urdf_path, 
                         basePosition=[0, 0, table_height],
                         useFixedBase=True)
    
    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=0,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # Print information about joints
    num_joints = p.getNumJoints(robot_id)
    print(f"\nTotal number of joints: {num_joints}")
    print("\nJoint Information:")
    print("-" * 50)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}:")
        print(f"  Name: {joint_info[1].decode('utf-8')}")
        print(f"  Type: {joint_info[2]}")
        print(f"  Lower limit: {joint_info[8]}")
        print(f"  Upper limit: {joint_info[9]}")
        print("-" * 50)
    
    # Test moving joints
    while True:
        try:
            # Move joints in a simple pattern
            for joint in range(num_joints):
                # Only move actual joints (not fixed joints)
                if p.getJointInfo(robot_id, joint)[2] != p.JOINT_FIXED:
                    p.setJointMotorControl2(
                        bodyIndex=robot_id,
                        jointIndex=joint,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=np.sin(time.time()) * 0.5
                    )
            
            # Get and print end-effector position
            state = p.getLinkState(robot_id, 6)  # arm end-effector
            print(f"End effector position: {state[0]}, orientation: {state[1]}")
            
            p.stepSimulation()
            time.sleep(1./240.)  # 240 Hz simulation
            
        except KeyboardInterrupt:
            break
    
    p.disconnect()

def test_xarm_initial_pose():
    """Test to get the initial pose of xArm when mounted on table."""
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Enable GUI controls
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
    
    # Load ground plane
    p.loadURDF("plane.urdf")
    
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
    table = p.loadURDF(table_urdf_path, 
                      basePosition=[0, 0, table_height/2],
                      useFixedBase=True)
    
    # Load xArm
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
    
    # Load robot on table (same as XArmEnv)
    robot_id = p.loadURDF(xarm_urdf_path, 
                         basePosition=[0, 0, table_height],
                         useFixedBase=True)
    
    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=0,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0]
    )
    
    print("\nInitial Robot State:")
    print("-" * 50)
    
    # Get end effector state
    state = p.getLinkState(robot_id, 6)  # arm end-effector
    pos = state[0]
    orn = state[1]
    
    print(f"Position: {pos}")
    print(f"Orientation (quaternion): {orn}")
    print("Orientation (euler):", p.getEulerFromQuaternion(orn))
    
    # Also print joint states
    print("\nJoint States:")
    print("-" * 50)
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED:
            state = p.getJointState(robot_id, i)
            print(f"Joint {i}: {state[0]}")  # Position
    
    input("\nPress Enter to exit...")
    p.disconnect()

def test_xarm_links():
    """Test to verify link structure matches URDF specifications."""
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Get paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xarm_urdf_path = os.path.join(current_dir, "beavr/src/components/environment/assets/urdf/leap_xarm7/leap_xarm7.urdf")
    
    # Expected link/joint names from URDF
    expected_links = {
        0: "world",  # Base link
        1: "link1",
        2: "link2",
        3: "link3",
        4: "link4",
        5: "link5",
        6: "link6",  # End effector link
        # Add any additional links from URDF
    }
    
    # Load robot
    robot_id = p.loadURDF(xarm_urdf_path, [0, 0, 0], useFixedBase=True)
    
    # Verify link structure
    num_joints = p.getNumJoints(robot_id)
    print("\nVerifying Link Structure:")
    print("-" * 50)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        link_name = joint_info[12].decode('utf-8')  # Link name
        joint_name = joint_info[1].decode('utf-8')  # Joint name
        joint_type = joint_info[2]  # Joint type
        
        print(f"\nLink {i}:")
        print(f"  Link Name: {link_name}")
        print(f"  Joint Name: {joint_name}")
        print(f"  Joint Type: {joint_type}")
        
        # Verify against expected names
        if i in expected_links:
            if link_name == expected_links[i]:
                print("  ✓ Link name matches URDF")
            else:
                print(f"  ✗ Link name mismatch! Expected: {expected_links[i]}, Got: {link_name}")
    
    # Verify end effector (link6)
    print("\nVerifying End Effector (Link 6):")
    print("-" * 50)
    link_6_info = p.getJointInfo(robot_id, 6)
    link_6_name = link_6_info[12].decode('utf-8')
    print(f"End Effector Link Name: {link_6_name}")
    if link_6_name == "link6":
        print("✓ Confirmed: Link 6 is the end effector")
    else:
        print("✗ Warning: Link 6 name doesn't match expected end effector name")
    
    input("\nPress Enter to exit...")
    p.disconnect()

def test_xarm_movement():
    """Test basic joint movement of xArm."""
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # ... simplified movement test code ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XArm PyBullet Tests')
    parser.add_argument('--test', type=str, default='base',
                      choices=['base', 'link', 'initial_pose', 'movement'],
                      help='Test to run (default: base)')
    parser.add_argument('--link', type=int, default=6,
                      help='Link index to examine (default: 6)')
    
    args = parser.parse_args()
    
    if args.test == 'base':
        test_xarm()
    elif args.test == 'link':
        test_xarm_link(args.link)
    elif args.test == 'initial_pose':
        test_xarm_initial_pose()
    elif args.test == 'movement':
        test_xarm_movement()