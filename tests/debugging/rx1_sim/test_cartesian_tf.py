#!/usr/bin/env python3

import zmq
import time


class CartesianController:
    def __init__(self):
        # ZMQ setup
        self.context = zmq.Context()
        self.command_publisher = self.context.socket(zmq.PUB)
        self.pose_subscriber = self.context.socket(zmq.SUB)
        
        # Connection parameters
        self.host = "10.0.0.55"
        self.command_port = "5555"
        self.cartesian_port = "9118"
        
        # Connect sockets
        self.command_publisher.bind(f"tcp://*:{self.command_port}")
        self.pose_subscriber.connect(f"tcp://{self.host}:{self.cartesian_port}")
        self.pose_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        
        print("Waiting for connection...")
        time.sleep(1.0)

    def test_connection(self):
        """Send test command"""
        self.command_publisher.send_pyobj({
            'type': 'test',
            'message': 'Testing cartesian control'
        })
        time.sleep(0.1)

    def move_ee(self, translation):
        """Move end effector by translation"""
        try:
            # Wait for current pose
            while True:
                msg = self.pose_subscriber.recv_pyobj()
                if msg.get('type') == 'ee_pose':
                    current_pose = msg
                    break

            # Apply translation
            new_position = [
                current_pose['position'][0] + translation[0],
                current_pose['position'][1] + translation[1],
                current_pose['position'][2] + translation[2]
            ]

            # Send cartesian goal
            goal = {
                'type': 'cartesian_goal',
                'position': new_position,
                'orientation': current_pose['orientation']  # Keep current orientation
            }
            
            self.command_publisher.send_pyobj(goal)
            print(f"Sent cartesian goal: {goal}")

        except Exception as e:
            print(f"Error moving end effector: {e}")

if __name__ == '__main__':
    controller = CartesianController()
    
    # Test connection
    controller.test_connection()
    
    # Move end effector
    translations = [
        [0.1, 0, 0],   # Forward 10cm
        [0, 0.1, 0],   # Right 10cm
        [0, 0, 0.1],   # Up 10cm
    ]
    
    for trans in translations:
        input(f"Press Enter to move {trans}...")
        controller.move_ee(trans)
        time.sleep(2.0)  # Wait for motion to complete