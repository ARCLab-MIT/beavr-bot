import pybullet as p
import pybullet_data
import numpy as np
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

def test_leap_hand(headless=True):
    # Connect to PyBullet in headless mode for testing
    physics_client = p.connect(p.DIRECT if headless else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set up simulation environment
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Get path to Leap Hand URDF
    hand_urdf_path = os.path.join(
        current_dir,
        "..",
        "..",
        "..",
        "assets",
        "urdf",
        "leap_hand",
        "leap_hand_right.urdf"
    )
    
    # Add the URDF directory to the search path so PyBullet can find the meshes
    p.setAdditionalSearchPath(os.path.dirname(hand_urdf_path))
    
    # Load Leap Hand
    robot_id = p.loadURDF(hand_urdf_path, 
                         basePosition=[0, 0, 0],
                         useFixedBase=True)
    
    # Print information about joints
    num_joints = p.getNumJoints(robot_id)
    # print(f"\nTotal number of joints: {num_joints}")
    # print("\nJoint Information:")
    # print("-" * 50)
    # for i in range(num_joints):
    #     joint_info = p.getJointInfo(robot_id, i)
    #     print(f"Joint {i}:")
    #     print(f"  Name: {joint_info[1].decode('utf-8')}")
    #     print(f"  Type: {joint_info[2]}")
    #     print(f"  Lower limit: {joint_info[8]}")
    #     print(f"  Upper limit: {joint_info[9]}")
    #     print(f"  Max force: {joint_info[10]}")
    #     print(f"  Max velocity: {joint_info[11]}")
    #     print("-" * 50)
    
    # For testing, just run a few simulation steps with deterministic joint positions
    for step in range(100):  # Run 100 simulation steps
        # Use a deterministic value for joint positions (based on step number)
        angle = np.sin(step * 0.1) * 0.5
        
        # Move joints in a simple pattern
        for joint in range(num_joints):
            # Only move actual joints (not fixed joints)
            if p.getJointInfo(robot_id, joint)[2] != p.JOINT_FIXED:
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=angle
                )
        
        p.stepSimulation()
    
    # Get final state for verification
    final_state = p.getLinkState(robot_id, num_joints-1)
    final_position = final_state[0]
    
    print(f"Final thumb tip position: {final_position}")
    
    # Run the test once to get the expected position, then hardcode it here
    expected_position = (0.046643734870256734, 0.16743929520186446, 0.04758022679205392)
    assert np.allclose(final_position, expected_position, atol=1e-2), f"Final position {final_position} is not as expected."
    
    p.disconnect()

if __name__ == "__main__":
    # When run directly, you can choose to show the GUI
    test_leap_hand(headless=False)