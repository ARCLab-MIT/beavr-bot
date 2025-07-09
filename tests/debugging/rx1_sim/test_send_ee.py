#!/usr/bin/env python3

import zmq
import time
import numpy as np


class EndEffectorMonitor:
    def __init__(self):
        self.host = "10.29.190.204"
        self.subscribe_port = "5555"
        self.publish_port = "9118"
        
        # Setup subscriber
        self.context = zmq.Context()
        self.state_subscriber = self.context.socket(zmq.SUB)
        state_address = f"tcp://{self.host}:{self.subscribe_port}"
        self.state_subscriber.connect(state_address)
        self.state_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Setup publisher
        self.state_publisher = self.context.socket(zmq.PUB)
        publish_address = f"tcp://{self.host}:{self.publish_port}"
        self.state_publisher.bind(publish_address)
        
        print(f"ZMQ subscriber connected to: {state_address}")
        print(f"ZMQ publisher bound to: {publish_address}")
        time.sleep(1.0)

    def generate_random_position(self, current_position, max_delta=0.2):
        """Generate a random position within max_delta meters of current position."""
        new_position = {
            'x': current_position['x'] + np.random.uniform(-max_delta, max_delta),
            'y': current_position['y'] + np.random.uniform(-max_delta, max_delta),
            'z': current_position['z'] + np.random.uniform(-max_delta, max_delta)
        }
        return new_position

    def receive_and_transform(self):
        """Receive position, apply random position change, and publish back."""
        try:
            print("Waiting for end effector position...")
            data = self.state_subscriber.recv_pyobj()
            
            # Get original pose
            position = data['position']
            orientation = data['orientation']
            
            print("\nOriginal pose:")
            print(f"Position: x: {position['x']}")
            print(f"y: {position['y']}")
            print(f"z: {position['z']}")
            print(f"Orientation: x: {orientation['x']}")
            print(f"y: {orientation['y']}")
            print(f"z: {orientation['z']}")
            print(f"w: {orientation['w']}")
            
            # Generate random position
            new_position = self.generate_random_position(position)
            
            # Create message format matching the bridge's expectations
            transformed_pose = {
                'position': new_position,
                'orientation': orientation,
                'timestamp': time.time(),
                'frame': 'base_link'
            }
            
            print("\nTransformed pose (random position):")
            print(f"Position: x: {new_position['x']}")
            print(f"y: {new_position['y']}")
            print(f"z: {new_position['z']}")
            print(f"Orientation: x: {orientation['x']}")
            print(f"y: {orientation['y']}")
            print(f"z: {orientation['z']}")
            print(f"w: {orientation['w']}")
            
            # Publish the transformed pose
            self.state_publisher.send_pyobj(transformed_pose)
            print("\nPublished transformed pose")
            
            return True
            
        except Exception as e:
            print(f"Error receiving/transforming data: {e}")
            return False
        finally:
            self.cleanup()

    def run_continuous(self):
        """Run continuously, publishing new poses every second."""
        try:
            while True:
                success = self.receive_and_transform()
                if not success:
                    break
                time.sleep(1.0)  # Wait a second before next transform
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up ZMQ connections."""
        print("\nCleaning up...")
        self.state_subscriber.close()
        self.state_publisher.close()
        self.context.term()
        print("Cleanup complete")

if __name__ == '__main__':
    monitor = EndEffectorMonitor()
    monitor.run_continuous()  # Run continuously instead of just once