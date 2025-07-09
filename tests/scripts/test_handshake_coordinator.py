#!/usr/bin/env python3
"""
Test script for the HandshakeCoordinator functionality.

This script demonstrates how to use the HandshakeCoordinator for guaranteed
delivery of teleop state changes.

Usage:
    # Terminal 1 - Start subscriber (robot/operator side)
    python test_handshake_coordinator.py subscriber

    # Terminal 2 - Test publisher (adapter side)  
    python test_handshake_coordinator.py publisher
"""

import sys
import time
import logging
from beavr.utils.network import HandshakeCoordinator, publish_with_guaranteed_delivery
from beavr.constants import ARM_TELEOP_STOP, ARM_TELEOP_CONT


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_subscriber():
    """Test the subscriber side (robot/operator)."""
    print("Starting handshake subscriber test...")
    
    coordinator = HandshakeCoordinator.get_instance()
    
    # Start handshake server for this subscriber
    subscriber_id = "test_robot"
    port = 8151
    
    try:
        coordinator.start_server(subscriber_id, "*", port)
        print(f"Handshake server started for '{subscriber_id}' on port {port}")
        
        # Keep running to handle handshake requests
        print("Waiting for handshake requests... (Press Ctrl+C to stop)")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down subscriber...")
    except Exception as e:
        logger.error(f"Error in subscriber: {e}")
    finally:
        coordinator.stop_server(subscriber_id)
        coordinator.shutdown()

def test_publisher():
    """Test the publisher side (adapter)."""
    print("Starting handshake publisher test...")
    
    coordinator = HandshakeCoordinator.get_instance()
    
    # Register the test subscriber
    subscriber_id = "test_robot"
    host = "127.0.0.1"
    port = 8151
    
    coordinator.register_subscriber(subscriber_id, host, port)
    print(f"Registered subscriber '{subscriber_id}' at {host}:{port}")
    
    # Wait a moment for subscriber to be ready
    time.sleep(2)
    
    try:
        # Test publishing teleop stop with guaranteed delivery
        print("Testing TELEOP_STOP with guaranteed delivery...")
        success = publish_with_guaranteed_delivery(
            host="127.0.0.1",
            port=8089,
            topic="pause",
            data=ARM_TELEOP_STOP,
            subscriber_ids=[subscriber_id],
            handshake_timeout=5.0
        )
        
        if success:
            print("✅ TELEOP_STOP delivery confirmed!")
        else:
            print("❌ TELEOP_STOP delivery failed!")
        
        time.sleep(1)
        
        # Test publishing teleop resume with guaranteed delivery
        print("Testing TELEOP_CONT with guaranteed delivery...")
        success = publish_with_guaranteed_delivery(
            host="127.0.0.1",
            port=8089,
            topic="pause",
            data=ARM_TELEOP_CONT,
            subscriber_ids=[subscriber_id],
            handshake_timeout=5.0
        )
        
        if success:
            print("✅ TELEOP_CONT delivery confirmed!")
        else:
            print("❌ TELEOP_CONT delivery failed!")
            
        # Test without handshake (should succeed immediately)
        print("Testing normal publishing without handshake...")
        success = publish_with_guaranteed_delivery(
            host="127.0.0.1",
            port=8089,
            topic="pause",
            data=ARM_TELEOP_STOP,
            subscriber_ids=None,  # No handshake required
            handshake_timeout=5.0
        )
        
        if success:
            print("✅ Normal publishing without handshake succeeded!")
        else:
            print("❌ Normal publishing failed!")
            
    except Exception as e:
        logger.error(f"Error in publisher: {e}")
    finally:
        coordinator.unregister_subscriber(subscriber_id)
        coordinator.shutdown()

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "subscriber":
        test_subscriber()
    elif mode == "publisher":
        test_publisher()
    else:
        print("Invalid mode. Use 'subscriber' or 'publisher'")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()