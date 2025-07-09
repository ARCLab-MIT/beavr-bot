import numpy as np
from beavr.interfaces.xarm7_robot import XArm7Robot
import zmq
import time
from threading import Thread


def setup_dummy_publisher(port):
    """Setup a dummy ZMQ publisher to simulate data"""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    
    def publish_dummy_data():
        while True:
            # Simulate some dummy data
            dummy_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).astype(np.float32)
            socket.send_multipart([b"endeff_coords", dummy_data.tobytes()])
            socket.send_multipart([b"joint", dummy_data.tobytes()])
            time.sleep(0.1)
    
    thread = Thread(target=publish_dummy_data, daemon=True)
    thread.start()
    return socket

def main(): 
    # Setup dummy publishers for testing
    endeff_pub = setup_dummy_publisher(10009)  # endeff_subscribe_port
    joint_pub = setup_dummy_publisher(8119)    # joint_subscribe_port
    
    # Create robot instance
    robot = XArm7Robot(
        host="10.31.153.63",
        endeff_subscribe_port=10009,
        endeff_publish_port=10010,
        joint_subscribe_port=8119,
        robot_ip="192.168.1.197"  # Using dummy IP for simulation
    )

    # Print all states
    print("\n=== Robot States ===")
    print(f"Joint State: {robot.get_joint_state()}")
    print(f"Cartesian State: {robot.get_cartesian_state()}")
    print(f"Joint Velocity: {robot.get_joint_velocity()}")
    print(f"Joint Torque: {robot.get_joint_torque()}")
    print(f"Cartesian Position: {robot.get_cartesian_position()}")
    print(f"Robot Pose: {robot.get_pose()}")

if __name__ == "__main__":
    main()
