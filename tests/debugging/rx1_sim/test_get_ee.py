#!/usr/bin/env python3

import zmq
import time


class EndEffectorMonitor:
    def __init__(self):
        # Define connection parameters
        self.host = "10.29.190.204"  # IP of the Docker container
        self.joint_port = "5555"     # Match the port from the bridge (5555)
        
        # ZMQ setup
        self.context = zmq.Context()
        self.state_subscriber = self.context.socket(zmq.SUB)
        
        # Connect subscriber
        state_address = f"tcp://{self.host}:{self.joint_port}"
        self.state_subscriber.connect(state_address)
        self.state_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        
        print(f"ZMQ subscriber connected to: {state_address}")
        time.sleep(1.0)  # Allow connection to establish

    def receive_single_position(self):
        """Receive and print a single position update for the end effector."""
        try:
            print("Waiting for end effector position...")
            data = self.state_subscriber.recv_pyobj()  # This will unpickle the data automatically
            
            # Extract position and orientation from the received dictionary
            position = data['position']
            orientation = data['orientation']
            
            # Print in the same format as ROS
            print(f"Position: x: {position['x']}")
            print(f"y: {position['y']}")
            print(f"z: {position['z']}, ", end="")
            print(f"Orientation: x: {orientation['x']}")
            print(f"y: {orientation['y']}")
            print(f"z: {orientation['z']}")
            print(f"w: {orientation['w']}")
            return True
            
        except Exception as e:
            print(f"Error receiving data: {e}")
            return False
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up ZMQ connections."""
        print("\nCleaning up...")
        self.state_subscriber.close()
        self.context.term()
        print("Cleanup complete")

if __name__ == '__main__':
    monitor = EndEffectorMonitor()
    monitor.receive_single_position()