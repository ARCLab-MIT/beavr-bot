import roslibpy
import time

import traceback


def debug_ros_connections():
    client = roslibpy.Ros(host='localhost', port=9090)
    
    try:
        client.run()
        print("\nROS connection established")
        
        # List all topics
        service = roslibpy.Service(client, '/rosapi/topics', 'rosapi/Topics')
        topics_result = service.call({''})
        print("\nActive ROS topics:")
        for topic in topics_result['topics']:
            print(f"- {topic}")
        
        # Check controller state
        controller_state = roslibpy.Topic(client, '/controller_manager/controller_state', 'controller_manager_msgs/ControllerState')
        def state_callback(message):
            print(f"\nController state: {message}")
        controller_state.subscribe(state_callback)
        
        # Monitor command topic
        cmd_topic = '/right_arm_position_controller/command'
        cmd_sub = roslibpy.Topic(client, cmd_topic, 'trajectory_msgs/JointTrajectory')
        def command_callback(message):
            print(f"\nCommand received on {cmd_topic}:")
            print(message)
        cmd_sub.subscribe(command_callback)
        
        print("\nMonitoring for 30 seconds...")
        time.sleep(30)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if client.is_connected:
            client.terminate()

if __name__ == "__main__":
    debug_ros_connections()