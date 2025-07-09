import time
import argparse
from beavr.utils.network import ZMQKeypointSubscriber


def monitor_oculus_data(host='localhost', port=5556, timeout=None):
    """
    Monitor data published by OculusVRHandDetector.
    
    Args:
        host (str): Host address to connect to (default: localhost)
        port (int): Port number to connect to (default: 5556)
        timeout (float): How long to monitor in seconds (None for indefinite)
    """
    print(f"\nStarting monitoring on {host}:{port}")
    print("Creating subscribers for each topic...")
    
    # Create subscribers for each topic
    subscribers = {
        'left': ZMQKeypointSubscriber(host=host, port=port, topic='left'),
        'right': ZMQKeypointSubscriber(host=host, port=port, topic='right'),
        'button': ZMQKeypointSubscriber(host=host, port=port, topic='button'),
        'pause': ZMQKeypointSubscriber(host=host, port=port, topic='pause')
    }
    
    print("\nWaiting for data...")
    print("Press Ctrl+C to stop...\n")
    
    start_time = time.time()
    data_received = {topic: False for topic in subscribers.keys()}
    
    try:
        while True:
            if timeout and (time.time() - start_time) > timeout:
                print(f"\nTimeout after {timeout} seconds")
                break
                
            for topic, sub in subscribers.items():
                data = sub.recv_keypoints()
                if data is not None:
                    if not data_received[topic]:
                        print(f"✓ First data received on topic '{topic}'")
                        data_received[topic] = True
                    print(f"{topic}: {data}")
            
            time.sleep(0.01)  # Small sleep to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\nStopping monitors...")
    finally:
        # Print summary
        print("\nSummary:")
        for topic, received in data_received.items():
            status = "✓" if received else "✗"
            print(f"{status} Topic '{topic}': {'Data received' if received else 'No data received'}")
            
        # Cleanup subscribers
        for sub in subscribers.values():
            sub.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor Oculus VR data streams')
    parser.add_argument('--host', default='10.31.152.148', help='Host address (default: 10.31.152.148)')
    parser.add_argument('--port', type=int, default=8088, help='Port number (default: 8088)')
    parser.add_argument('--timeout', type=float, help='Timeout in seconds (default: None)')
    
    args = parser.parse_args()
    monitor_oculus_data(host=args.host, port=args.port, timeout=args.timeout)