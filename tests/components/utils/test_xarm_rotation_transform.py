import pybullet as p
import pybullet_data
import time
import numpy as np
import os
from scipy.spatial.transform import Rotation


def rotate_x_90_degrees(quat_pos_array, sign=1):
    """Rotate a pose array [x,y,z,qx,qy,qz,qw] by 90 degrees around the X-axis."""
    pos = quat_pos_array[:3]
    quat = quat_pos_array[3:]
    
    angle = np.pi/2 * sign  # sign=-1 for z-up to y-up
    rot = Rotation.from_euler('x', angle)
    rot_quat = rot.as_quat()
    
    rotated_pos = rot.apply(pos)
    orig_rot = Rotation.from_quat(quat)
    new_rot = Rotation.from_quat(rot_quat) * orig_rot
    new_quat = new_rot.as_quat()
    
    return np.concatenate([rotated_pos, new_quat])

def test_rotation_transform():
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set up simulation environment
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Load table and robot
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
    
    # Get end effector state
    end_effector_index = 7  # The end effector link
    state = p.getLinkState(robot_id, end_effector_index)
    pos = state[0]  # world position
    orn = state[1]  # world orientation (quaternion)
    
    # Create initial pose array [x,y,z, qx,qy,qz,qw]
    initial_pose = np.concatenate([pos, orn])
    
    # Print original position and orientation (z-up)
    orig_euler = Rotation.from_quat(initial_pose[3:]).as_euler('xyz', degrees=True)
    print("\nOriginal pose (z-up):")
    print(f"Position (x,y,z): [{initial_pose[0]:.3f}, {initial_pose[1]:.3f}, {initial_pose[2]:.3f}]")
    print(f"Orientation (roll,pitch,yaw): [{orig_euler[0]:.1f}°, {orig_euler[1]:.1f}°, {orig_euler[2]:.1f}°]")
    
    # Apply rotation for z-up to y-up transformation
    rotated_pose = rotate_x_90_degrees(initial_pose, sign=1)  # Negative for z-up to y-up
    rot_euler = Rotation.from_quat(rotated_pose[3:]).as_euler('xyz', degrees=True)
    print("\nAfter transformation to y-up:")
    print(f"Position (x,y,z): [{rotated_pose[0]:.3f}, {rotated_pose[1]:.3f}, {rotated_pose[2]:.3f}]")
    print(f"Orientation (roll,pitch,yaw): [{rot_euler[0]:.1f}°, {rot_euler[1]:.1f}°, {rot_euler[2]:.1f}°]")
    
    # Visualize the transformation
    # Red line shows original position to rotated position
    p.addUserDebugLine(initial_pose[:3], rotated_pose[:3], [1, 0, 0], 2)
    # Green line shows up direction in rotated frame
    p.addUserDebugLine(rotated_pose[:3], [rotated_pose[0], rotated_pose[1] + 0.1, rotated_pose[2]], [0, 1, 0], 2)
    
    # Keep the simulation running for a few seconds
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1./240.)
    
    p.disconnect()

if __name__ == "__main__":
    test_rotation_transform()
