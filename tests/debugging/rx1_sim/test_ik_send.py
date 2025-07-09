#!/usr/bin/env python3

import time
import numpy as np
import zmq
from beavr.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from scipy.spatial.transform import Rotation as R


class EndEffectorMonitor:
    def __init__(self, debug=True):
        self.host = "10.29.190.204"
        self.debug = debug
        
        # Ports configuration matching the bridge
        self.ee_pose_port = "5555"
        self.transformed_pose_port = "9118"
        
        # Setup ZMQ subscriber for current EE pose
        self.ee_pose_subscriber = ZMQKeypointSubscriber(
            host=self.host,
            port=self.ee_pose_port,
            topic="ee_pose"
        )
        
        # Setup ZMQ publisher for target poses
        self.target_pose_publisher = ZMQKeypointPublisher(
            host=self.host,
            port=self.transformed_pose_port
        )
        
        if self.debug:
            print(f"ZMQ subscriber connected to: tcp://{self.host}:{self.ee_pose_port}")
            print(f"ZMQ publisher bound to: tcp://{self.host}:{self.transformed_pose_port}")
        
        # Define translations to test [dx, dy, dz]
        self.translations = [
            np.array([0.1, 0.0, 0.0]),  # Move 10cm in X
            np.array([0.0, 0.1, 0.0]),  # Move 10cm in Y
            np.array([0.0, 0.0, 0.1])   # Move 10cm in Z
        ]
        self.current_translation = 0
        time.sleep(1.0)

    def receive_and_publish(self):
        """Receive current pose, translate it, and publish in position + quaternion format."""
        try:
            if self.debug:
                print("Waiting for current end effector pose...")
            
            # Try to receive current pose (non-blocking)
            current_pose = self.ee_pose_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
            if current_pose is None:
                print("No current pose received")
                return False

            # Extract current position and axis-angle orientation
            current_position = current_pose[:3]
            current_rotation_aa = current_pose[3:]
            
            # Convert axis-angle to quaternion
            r = R.from_rotvec(current_rotation_aa)
            current_quat = r.as_quat()  # [x, y, z, w]

            if self.debug:
                print("\nReceived current pose:")
                print(f"Position [x, y, z]: {current_position}")
                print(f"Orientation (axis-angle) [rx, ry, rz]: {current_rotation_aa}")
                print(f"Orientation (quaternion) [x, y, z, w]: {current_quat}")

            # Get current translation
            translation = self.translations[self.current_translation]
            
            # Create new position by applying translation
            new_position = current_position + translation
            
            # Create target pose array [x, y, z, qx, qy, qz, qw]
            target_pose = np.concatenate([new_position, current_quat])
            
            if self.debug:
                print(f"\nApplying translation #{self.current_translation}:")
                print(f"Translation [dx, dy, dz]: {translation}")
                print(f"New position [x, y, z]: {new_position}")
                print(f"Keeping orientation (quaternion) [x, y, z, w]: {current_quat}")

            # Publish target pose
            self.target_pose_publisher.pub_keypoints(
                target_pose,
                'pose'
            )
            
            if self.debug:
                print("\nPublished target pose [x, y, z, qx, qy, qz, qw]")
                print(f"Published array: {target_pose}")

            # Increment the translation counter
            self.current_translation = (self.current_translation + 1) % len(self.translations)

            # Sleep to allow visualization of each point
            time.sleep(2.0)
            return True
            
        except Exception as e:
            print(f"Error in receive_and_publish: {e}")
            return False

    def cleanup(self):
        """Clean up ZMQ connections."""
        if self.debug:
            print("\nCleaning up...")
        self.ee_pose_subscriber.stop()
        self.target_pose_publisher.stop()
        if self.debug:
            print("Cleanup complete")

if __name__ == '__main__':
    monitor = EndEffectorMonitor(debug=True)
    try:
        # Test all three translations in sequence
        for _ in range(3):
            monitor.receive_and_publish()
    finally:
        monitor.cleanup()
