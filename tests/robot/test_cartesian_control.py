#!/usr/bin/env python3

import zmq
import time
import pickle


def test_cartesian_control():
    # ZMQ setup
    context = zmq.Context()
    command_pub = context.socket(zmq.PUB)
    command_pub.connect("tcp://10.29.187.186:8120")  # Use your robot's IP
    
    # Subscribe to robot state
    cartesian_sub = context.socket(zmq.SUB)
    cartesian_sub.connect("tcp://10.29.187.186:8118")
    cartesian_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    
    # Wait for connection
    time.sleep(1)
    print("Connected to robot bridge")
    
    try:
        # First, get current pose
        print("Waiting for current robot pose...")
        data = cartesian_sub.recv()
        current_pose = pickle.loads(data)
        print(f"Current pose: \nPosition: {current_pose[:3]}\nOrientation: {current_pose[3:]}")
        
        # Test positions (relative to current position)
        test_positions = [
            # Move 10cm up
            {'position': [current_pose[0], current_pose[1], current_pose[2] + 0.1],
             'orientation': current_pose[3:]},
            
            # Move 10cm forward
            {'position': [current_pose[0] + 0.1, current_pose[1], current_pose[2] + 0.1],
             'orientation': current_pose[3:]},
            
            # Move right
            {'position': [current_pose[0] + 0.1, current_pose[1] - 0.1, current_pose[2] + 0.1],
             'orientation': current_pose[3:]},
            
            # Return to original height
            {'position': [current_pose[0] + 0.1, current_pose[1] - 0.1, current_pose[2]],
             'orientation': current_pose[3:]},
            
            # Back to start
            {'position': current_pose[:3],
             'orientation': current_pose[3:]},
        ]
        
        for i, pose in enumerate(test_positions):
            print(f"\nSending position {i+1}/{len(test_positions)}")
            print(f"Target position: {pose['position']}")
            
            # Create command
            command = {
                'type': 'cartesian',
                'position': pose['position'],
                'orientation': pose['orientation']
            }
            
            # Send command
            command_pub.send(pickle.dumps(command))
            print("Command sent")
            
            # Wait and monitor position
            time.sleep(2.0)
            try:
                data = cartesian_sub.recv(flags=zmq.NOBLOCK)
                current_pose = pickle.loads(data)
                print(f"Current position: {current_pose[:3]}")
            except zmq.Again:
                print("No position feedback received")
    
    except KeyboardInterrupt:
        print("\nStopping test...")
    
    finally:
        # Cleanup
        command_pub.close()
        cartesian_sub.close()
        context.term()

if __name__ == "__main__":
    test_cartesian_control()