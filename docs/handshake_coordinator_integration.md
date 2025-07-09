# HandshakeCoordinator Integration Guide

The `HandshakeCoordinator` provides guaranteed delivery confirmation for critical teleop state changes. This ensures that stop/resume signals are properly acknowledged by all relevant subscribers before proceeding.

## Architecture Overview

```
Publisher (beavr_robot_adapter)
    ↓ publish teleop state
    ↓ request acknowledgments  
    ↓
HandshakeCoordinator
    ↓ coordinate multiple subscribers
    ↓
Subscribers (robots, operators)
    ↓ acknowledge state changes
    ↓
Confirmation back to Publisher
```

## Integration Steps

### 1. Publisher Side (beavr_robot_adapter.py)

```python
from beavr.utils.network import HandshakeCoordinator, publish_with_guaranteed_delivery

class MultiRobotAdapter:
    def __init__(self, ...):
        # Initialize handshake coordinator
        self.handshake_coordinator = HandshakeCoordinator.get_instance()
    
    def register_handshake_subscriber(self, subscriber_id: str, host: str, port: int):
        """Register a subscriber for handshake coordination."""
        self.handshake_coordinator.register_subscriber(subscriber_id, host, port)
    
    def teleop_stop(self):
        """Send stop signal with guaranteed delivery."""
        registered_subscribers = self.handshake_coordinator.get_registered_subscribers()
        
        success = publish_with_guaranteed_delivery(
            host=self.op_state_publish_info["host"],
            port=self.op_state_publish_info["port"], 
            topic=self.op_state_publish_info["topic"],
            data=ARM_TELEOP_STOP,
            subscriber_ids=registered_subscribers,
            handshake_timeout=3.0,
            require_all_acks=False  # Allow partial success
        )
        
        if not success:
            logging.warning("Not all subscribers acknowledged teleop stop")
```

### 2. Subscriber Side (xarm7_robot.py, operators)

```python
from beavr.utils.network import HandshakeCoordinator

class XArm7Robot:
    def __init__(self, ...):
        # Initialize handshake coordination
        self._handshake_coordinator = HandshakeCoordinator.get_instance()
        self._handshake_server_id = f"{self.name}_handshake"
        
        # Start handshake server with unique port
        self._handshake_coordinator.start_server(
            subscriber_id=self._handshake_server_id,
            bind_host="*",
            port=TELEOP_HANDSHAKE_PORT + unique_offset  # Avoid port conflicts
        )
    
    def __del__(self):
        # Clean up handshake server
        if hasattr(self, '_handshake_coordinator'):
            self._handshake_coordinator.stop_server(self._handshake_server_id)
```

### 3. Configuration Integration

Update your robot configurations to include handshake port information:

```python
# In leap_xarm_right_config.py
@dataclass
class XArm7RobotCfg:
    # ... existing fields ...
    handshake_port: int = 8151  # Unique port for handshake
    handshake_subscriber_id: str = "right_xarm7"  # Unique ID
```

### 4. Registration During System Startup

Register all handshake subscribers when initializing the system:

```python
# In your main control script
adapter = MultiRobotAdapter(...)

# Register handshake subscribers for coordinated stop/resume
adapter.register_handshake_subscriber("right_xarm7", "10.31.152.148", 8151)
adapter.register_handshake_subscriber("leap_operator", "10.31.152.148", 8152)
adapter.register_handshake_subscriber("xarm_operator", "10.31.152.148", 8153)
```

## Port Assignment Strategy

To avoid port conflicts, use a systematic port assignment:

```python
BASE_HANDSHAKE_PORT = 8150

# Port assignments
TELEOP_HANDSHAKE_PORT = 8150  # Legacy, don't use
RIGHT_XARM_HANDSHAKE_PORT = 8151
LEFT_XARM_HANDSHAKE_PORT = 8152
LEAP_OPERATOR_HANDSHAKE_PORT = 8153
XARM_OPERATOR_HANDSHAKE_PORT = 8154
```

## Testing

Use the provided test script to verify handshake coordination:

```bash
# Terminal 1 - Start subscriber
python tests/scripts/test_handshake_coordinator.py subscriber

# Terminal 2 - Test publisher
python tests/scripts/test_handshake_coordinator.py publisher
```

## Benefits

1. **Guaranteed Delivery**: Critical teleop state changes are confirmed
2. **Race Condition Prevention**: Synchronized state transitions
3. **Fault Tolerance**: Partial acknowledgment support for robustness
4. **Centralized Management**: Single coordinator for multiple subscribers
5. **Thread Safety**: Safe for concurrent use across components

## Migration from Legacy Handshake

### Before (Multiple HandshakeServer instances):
```python
# In multiple files, causing port conflicts
self._handshake_server = HandshakeServer(host="*", port=TELEOP_HANDSHAKE_PORT)
```

### After (Centralized HandshakeCoordinator):
```python
# Single coordinator with unique ports per subscriber
coordinator = HandshakeCoordinator.get_instance()
coordinator.start_server("unique_id", "*", unique_port)
```

## Error Handling

The coordinator includes comprehensive error handling:

- **Port conflicts**: Automatic error reporting
- **Network timeouts**: Configurable timeout values
- **Partial failures**: Optional partial success mode
- **Resource cleanup**: Automatic cleanup on shutdown

## Logging

Enable debug logging to monitor handshake activity:

```python
import logging
logging.getLogger('beavr.utils.network').setLevel(logging.DEBUG)
```

This will show detailed handshake coordination messages for debugging. 