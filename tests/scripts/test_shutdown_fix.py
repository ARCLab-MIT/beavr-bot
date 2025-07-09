#!/usr/bin/env python3
"""
Test script to verify the shutdown fixes work correctly.
This script simulates the main control_robot.py structure with proper cleanup.
"""

import time
import signal
import sys
import threading
from beavr.utils.network import cleanup_zmq_resources, request_shutdown, is_shutdown_requested
from beavr.utils.timer import FrequencyTimer


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal, starting graceful shutdown...")
    request_shutdown()
    sys.exit(0)

def test_timer():
    """Test the improved FrequencyTimer"""
    print("Testing FrequencyTimer with cooperative sleep...")
    timer = FrequencyTimer(10)  # 10 Hz
    
    for i in range(5):
        timer.start_loop()
        print(f"Timer iteration {i+1}")
        time.sleep(0.05)  # Simulate some work
        timer.end_loop()
    
    print("Timer test completed successfully")

def test_shutdown_flag():
    """Test the shutdown flag mechanism"""
    print("Testing shutdown flag mechanism...")
    
    def worker():
        count = 0
        while not is_shutdown_requested():
            count += 1
            time.sleep(0.1)
            if count > 50:  # 5 seconds max
                break
        print(f"Worker stopped after {count} iterations")
    
    # Start worker thread
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    
    # Let it run for a bit
    time.sleep(2)
    
    # Request shutdown
    print("Requesting shutdown...")
    request_shutdown()
    
    # Wait for worker to finish
    worker_thread.join(timeout=1)
    print("Shutdown flag test completed")

def main():
    """Main test function"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Testing shutdown fixes...")
    print("Press Ctrl+C to test graceful shutdown")
    
    try:
        # Test timer
        test_timer()
        
        # Test shutdown flag
        test_shutdown_flag()
        
        # Simulate long-running operation
        print("Simulating long-running operation...")
        timer = FrequencyTimer(30)  # 30 Hz
        
        for i in range(100):  # Run for ~3 seconds
            timer.start_loop()
            if i % 30 == 0:
                print(f"Running iteration {i}")
            timer.end_loop()
            
            if is_shutdown_requested():
                print("Shutdown requested, breaking loop")
                break
        
        print("Test completed successfully!")
        
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main")
    finally:
        print("Cleaning up ZMQ resources...")
        cleanup_zmq_resources()
        print("Cleanup complete")

if __name__ == "__main__":
    main()