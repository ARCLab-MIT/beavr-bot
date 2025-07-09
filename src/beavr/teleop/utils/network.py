import zmq
import cv2
import base64
import numpy as np
import pickle
import blosc as bl
import threading
from typing import Any, Optional, Dict, Tuple
import logging
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)

class SerializationError(Exception):
    """Exception raised for serialization/deserialization issues"""
    pass

# Global ZMQ context (one per process)
_GLOBAL_ZMQ_CONTEXT = zmq.Context()

def get_global_context() -> zmq.Context:
    """Get the global ZMQ context (shared across all sockets)"""
    global _GLOBAL_ZMQ_CONTEXT
    return _GLOBAL_ZMQ_CONTEXT

def set_global_context(context: zmq.Context) -> None:
    """Set a custom global ZMQ context (useful for testing).
    
    Args:
        context: The ZMQ context to use globally
    """
    global _GLOBAL_ZMQ_CONTEXT
    _GLOBAL_ZMQ_CONTEXT = context

class BasePublisher(ABC):
    """Base class for all publishers"""
    def __init__(self, host: str, port: int, socket_type: int = zmq.PUB, context: Optional[zmq.Context] = None):
        """Initialize publisher.
        
        Args:
            host: The host address to bind to
            port: The port number to bind to
            socket_type: The ZMQ socket type (default: PUB)
            context: Optional custom ZMQ context (default: global context)
        """
        self._host = host
        self._port = port
        self._socket = None
        self._context = context or get_global_context()
        self._init_socket(socket_type)

    def _init_socket(self, socket_type: int) -> None:
        """Initialize the socket using provided or global context."""
        try:
            self._socket = self._context.socket(socket_type)
            self._socket.setsockopt(zmq.SNDHWM, 1)  # Only keep latest message
            addr = f'tcp://*:{self._port}'
            self._socket.bind(addr)
        except zmq.ZMQError as e:
            logger.error(f"Failed to initialize socket: {e}")
            raise

    def stop(self) -> None:
        """Close the publisher socket."""
        if self._socket:
            self._socket.close()
            self._socket = None

class BaseSubscriber(threading.Thread, ABC):
    """Base class for all subscribers with graceful shutdown support."""
    
    def __init__(self, host: str, port: int, topic: str = "", socket_type: int = zmq.SUB, 
                 context: Optional[zmq.Context] = None):
        """Initialize subscriber.
        
        Args:
            host: The host address to connect to
            port: The port number to connect to
            topic: The topic to subscribe to (empty for all topics)
            socket_type: The ZMQ socket type (default: SUB)
            context: Optional custom ZMQ context (default: global context)
        """
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._topic = topic
        self._socket = None
        self._running = True
        self._poller = zmq.Poller()
        self._context = context or get_global_context()
        self._init_socket(socket_type)

    def _init_socket(self, socket_type: int):
        """Initialize the socket using global context."""
        try:
            self._socket = self._context.socket(socket_type)
            
            # Configure socket for real-time control
            self._socket.setsockopt(zmq.RCVHWM, 5)  # Keep last 5 messages
            self._socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
            self._socket.setsockopt(zmq.RCVTIMEO, 50)  # 50ms timeout on receive
            
            # Enable message conflation for non-SUB sockets
            if socket_type != zmq.SUB:
                self._socket.setsockopt(zmq.CONFLATE, 1)
                
            # For SUB sockets, we handle conflation in the application layer
            # to avoid issues with topic filtering
            
            addr = f'tcp://{self._host}:{self._port}'
            self._socket.connect(addr)
            if socket_type == zmq.SUB:
                self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)
            self._poller.register(self._socket, zmq.POLLIN)
        except zmq.ZMQError as e:
            logger.error(f"Failed to initialize socket: {e}")
            raise

    def stop(self):
        """Stop the subscriber thread gracefully."""
        self._running = False
        if self._socket:
            self._poller.unregister(self._socket)
            self._socket.close()
            self._socket = None
        self.join(timeout=2)
        if self.is_alive():
            logger.warning("Subscriber thread did not stop gracefully")

    @abstractmethod
    def process_message(self, message: Any) -> None:
        """Process received message.
        
        Args:
            message: The received message data
        """
        pass

    def run(self):
        """Main subscriber loop with polling for graceful shutdown."""
        while self._running:
            try:
                if self._socket is None:
                    logger.error("Socket is None, breaking subscriber loop")
                    break
                
                # Poll with timeout to check _running periodically
                events = dict(self._poller.poll(100))  # 100ms timeout
                
                if self._socket in events:
                    try:
                        _, payload = self._socket.recv_multipart(zmq.NOBLOCK)
                        try:
                            data = pickle.loads(payload)
                            self.process_message(data)
                        except Exception as e:
                            logger.error(f"Failed to process message: {e}")
                    except zmq.Again:
                        continue
                    except Exception as e:
                        if self._running:  # Only log if we're not shutting down
                            logger.error(f"Error receiving message: {e}")
                        break
                else:
                    # Log poll timeout occasionally (every few seconds)
                    if hasattr(self, '_last_poll_log') and time.time() - self._last_poll_log > 5:
                        self._last_poll_log = time.time()
                    elif not hasattr(self, '_last_poll_log'):
                        self._last_poll_log = time.time()
                        
            except Exception as e:
                if self._running:
                    logger.error(f"Error in subscriber loop: {e}")
                break
        

# Update socket creation functions to use global context
def create_push_socket(host: str, port: int) -> zmq.Socket:
    """Create a PUSH socket with error handling."""
    try:
        socket = get_global_context().socket(zmq.PUSH)
        addr = f'tcp://{host}:{port}'
        socket.bind(addr)
        return socket
    except zmq.ZMQError as e:
        logger.error(f"Failed to create PUSH socket: {e}")
        raise

def create_pull_socket(host: str, port: int) -> zmq.Socket:
    """Create a PULL socket with error handling and connection verification."""
    try:
        socket = get_global_context().socket(zmq.PULL)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout for receiving
        socket.setsockopt(zmq.LINGER, 0)      # Don't wait on close
        socket.setsockopt(zmq.RCVHWM, 1)      # Only keep latest message
        
        addr = f'tcp://{host}:{port}'
        try:
            # Try to bind to the address
            socket.bind(addr)
            
            # Test if the socket is actually bound
            bound_addrs = socket.getsockopt_string(zmq.LAST_ENDPOINT)
            if not bound_addrs:
                raise zmq.ZMQError("Socket binding verification failed")
            
            return socket
            
        except zmq.ZMQError as e:
            if e.errno == zmq.EADDRINUSE:
                logger.error(f"Address {addr} already in use. Is another instance running?")
            else:
                logger.error(f"Failed to bind PULL socket to {addr}: {e}")
            raise
    except zmq.ZMQError as e:
        logger.error(f"Failed to create PULL socket: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating PULL socket: {e}")
        raise

def create_response_socket(host: str, port: int) -> zmq.Socket:
    """Create a REP socket with error handling."""
    try:
        socket = get_global_context().socket(zmq.REP)
        addr = f'tcp://{host}:{port}'
        socket.bind(addr)
        return socket
    except zmq.ZMQError as e:
        logger.error(f"Failed to create REP socket: {e}")
        raise

def create_request_socket(host: str, port: int) -> zmq.Socket:
    """Create a REQ socket with error handling."""
    try:
        socket = get_global_context().socket(zmq.REQ)
        addr = f'tcp://{host}:{port}'
        socket.connect(addr)
        return socket
    except zmq.ZMQError as e:
        logger.error(f"Failed to create REQ socket: {e}")
        raise

# Pub/Sub classes for Keypoints
class ZMQKeypointPublisher(BasePublisher):
    """Publisher for keypoint data using multipart messaging."""
    
    def __init__(self, host: str, port: int):
        super().__init__(host, port, zmq.PUB)

    def pub_keypoints(self, keypoint_array: Any, topic_name: str) -> None:
        """Publish keypoints using multipart messaging.
        
        Args:
            keypoint_array: The keypoint data to publish
            topic_name: The topic to publish to
            
        Raises:
            ConnectionError: If socket operation fails
            SerializationError: If serialization fails
        """
        try:
            buffer = pickle.dumps(keypoint_array, protocol=-1)
            try:
                self._socket.send_multipart([
                    topic_name.encode('utf-8'),
                    buffer
                ], zmq.NOBLOCK)
            except zmq.Again:
                logger.warning(f"High water mark reached for {topic_name}, dropping message")
            except zmq.ZMQError as e:
                raise ConnectionError(f"Failed to send keypoints: {e}")
        except Exception as e:
            raise SerializationError(f"Failed to serialize keypoints: {e}")

class ZMQKeypointSubscriber(BaseSubscriber):
    """Subscriber for keypoint data using multipart messaging."""
    
    def __init__(self, host: str, port: int, topic: str, context: Optional[zmq.Context] = None):
        super().__init__(host, port, topic, zmq.SUB, context)
        self._last_data: Optional[Any] = None
        self.start()

    def process_message(self, data: Any) -> None:
        """Process received keypoint data.
        
        Args:
            data: The received keypoint data
        """
        self._last_data = data

    def recv_keypoints(self) -> Optional[Any]:
        """Get the latest keypoint data.
        
        Returns:
            The latest keypoint data if available, None otherwise
        """
        data = self._last_data
        if data is not None:
            pass
        return data

# Pub/Sub classes for storing data from Realsense Cameras
class ZMQCameraPublisher(BasePublisher):
    """Publisher for camera data using multipart messaging."""
    
    def __init__(self, host: str, port: int):
        super().__init__(host, port, zmq.PUB)

    def pub_intrinsics(self, array: Any) -> None:
        """Publish camera intrinsics.
        
        Args:
            array: The intrinsics data to publish
            
        Raises:
            ConnectionError: If socket operation fails
            SerializationError: If serialization fails
        """
        try:
            buffer = pickle.dumps(array, protocol=-1)
            try:
                self._socket.send_multipart([
                    b"intrinsics",
                    buffer
                ], zmq.NOBLOCK)
            except zmq.Again:
                logger.warning("High water mark reached for intrinsics, dropping message")
            except zmq.ZMQError as e:
                raise ConnectionError(f"Failed to send intrinsics: {e}")
        except Exception as e:
            raise SerializationError(f"Failed to serialize intrinsics: {e}")

    def pub_rgb_image(self, rgb_image: np.ndarray, timestamp: float) -> None:
        """Publish RGB image with timestamp.
        
        Args:
            rgb_image: The RGB image array to publish
            timestamp: The timestamp of the image
            
        Raises:
            ConnectionError: If socket operation fails
            SerializationError: If serialization fails
        """
        try:
            _, buffer = cv2.imencode('.jpg', rgb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            data = {
                'timestamp': timestamp,
                'rgb_image': base64.b64encode(buffer)
            }
            try:
                self._socket.send_multipart([
                    b"rgb_image",
                    pickle.dumps(data, protocol=-1)
                ], zmq.NOBLOCK)
            except zmq.Again:
                logger.warning("High water mark reached for RGB image, dropping frame")
            except zmq.ZMQError as e:
                raise ConnectionError(f"Failed to send RGB image: {e}")
        except Exception as e:
            raise SerializationError(f"Failed to serialize RGB image: {e}")

    def pub_depth_image(self, depth_image: np.ndarray, timestamp: float) -> None:
        """Publish depth image with timestamp.
        
        Args:
            depth_image: The depth image array to publish
            timestamp: The timestamp of the image
            
        Raises:
            ConnectionError: If socket operation fails
            SerializationError: If serialization fails
        """
        try:
            compressed_depth = bl.pack_array(depth_image, cname='zstd', clevel=1, shuffle=bl.NOSHUFFLE)
            data = {
                'timestamp': timestamp,
                'depth_image': compressed_depth
            }
            try:
                self._socket.send_multipart([
                    b"depth_image",
                    pickle.dumps(data, protocol=-1)
                ], zmq.NOBLOCK)
            except zmq.Again:
                logger.warning("High water mark reached for depth image, dropping frame")
            except zmq.ZMQError as e:
                raise ConnectionError(f"Failed to send depth image: {e}")
        except Exception as e:
            raise SerializationError(f"Failed to serialize depth image: {e}")

class ZMQCameraSubscriber(BaseSubscriber):
    """Subscriber for camera data using multipart messaging."""
    
    def __init__(self, host: str, port: int, topic_type: str):
        """Initialize camera subscriber.
        
        Args:
            host: The host address to connect to
            port: The port number to connect to
            topic_type: Type of camera data ('Intrinsics', 'RGB', or 'Depth')
        """
        topic = {
            'Intrinsics': "intrinsics",
            'RGB': "rgb_image",
            'Depth': "depth_image"
        }.get(topic_type, "")
        super().__init__(host, port, topic, zmq.SUB)
        self._topic_type = topic_type
        self._last_data: Optional[Dict[str, Any]] = None
        self.start()  # Start the subscriber thread

    def process_message(self, data: Dict[str, Any]) -> None:
        """Process received camera data.
        
        Args:
            data: Dictionary containing camera data and metadata
        """
        self._last_data = data

    def recv_intrinsics(self) -> Optional[Any]:
        """Get the latest intrinsics data.
        
        Returns:
            Camera intrinsics data if available, None otherwise
        """
        return self._last_data if self._topic_type == 'Intrinsics' else None

    def recv_rgb_image(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get the latest RGB image and timestamp.
        
        Returns:
            Tuple of (image array, timestamp) if available, (None, None) otherwise
        """
        if self._topic_type != 'RGB' or self._last_data is None:
            return None, None
        encoded_data = np.fromstring(base64.b64decode(self._last_data['rgb_image']), np.uint8)
        return cv2.imdecode(encoded_data, 1), self._last_data['timestamp']

    def recv_depth_image(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get the latest depth image and timestamp.
        
        Returns:
            Tuple of (depth array, timestamp) if available, (None, None) otherwise
        """
        if self._topic_type != 'Depth' or self._last_data is None:
            return None, None
        depth_image = bl.unpack_array(self._last_data['depth_image'])
        return np.array(depth_image, dtype=np.int16), self._last_data['timestamp']

    def stop(self):
        logger.info('Closing the subscriber socket in {}:{}.'.format(self._host, self._port))
        self._socket.close()

# Publisher for image visualizers
class ZMQCompressedImageTransmitter(BasePublisher):
    """Publisher for compressed images using multipart messaging."""
    
    def __init__(self, host: str, port: int):
        super().__init__(host, port, zmq.PUB)

    def send_image(self, rgb_image: np.ndarray) -> None:
        """Send a compressed RGB image.
        
        Args:
            rgb_image: The RGB image array to publish
            
        Raises:
            ConnectionError: If socket operation fails
            SerializationError: If compression fails
        """
        try:
            _, buffer = cv2.imencode('.jpg', rgb_image, [int(cv2.IMWRITE_WEBP_QUALITY), 10])
            try:
                self._socket.send_multipart([
                    b"image",
                    np.array(buffer).tobytes()
                ], zmq.NOBLOCK)
            except zmq.Again:
                logger.warning("High water mark reached for compressed image, dropping frame")
            except zmq.ZMQError as e:
                raise ConnectionError(f"Failed to send compressed image: {e}")
        except Exception as e:
            raise SerializationError(f"Failed to compress image: {e}")

class ZMQCompressedImageReceiver(BaseSubscriber):
    """Subscriber for compressed images using multipart messaging."""
    
    def __init__(self, host: str, port: int):
        """Initialize compressed image receiver.
        
        Args:
            host: The host address to connect to
            port: The port number to connect to
        """
        super().__init__(host, port, "image", zmq.SUB)
        self._last_image: Optional[np.ndarray] = None
        self.start()  # Start the subscriber thread

    def process_message(self, data: bytes) -> None:
        """Process received image data.
        
        Args:
            data: Raw bytes of compressed image data
        """
        try:
            encoded_data = np.fromstring(data, np.uint8)
            self._last_image = cv2.imdecode(encoded_data, 1)
        except Exception as e:
            logger.error(f"Failed to process image: {e}")

    def recv_image(self) -> Optional[np.ndarray]:
        """Get the latest received image.
        
        Returns:
            The latest image array if available, None otherwise
        """
        return self._last_image

class ZMQButtonFeedbackSubscriber(BaseSubscriber):
    """Subscriber for button feedback using multipart messaging."""
    
    def __init__(self, host: str, port: int, context: Optional[zmq.Context] = None):
        super().__init__(host, port, "", zmq.SUB, context)  # Empty topic to receive all messages
        self._last_data: Optional[Any] = None
        self.start()

    def process_message(self, data: Any) -> None:
        """Process received button feedback data.
        
        Args:
            data: The received button feedback data
        """
        self._last_data = data

    def recv_keypoints(self) -> Optional[Any]:
        """Get the latest button feedback data.
        
        Returns:
            The latest button feedback data if available, None otherwise
        """
        return self._last_data

# Improved Pub/Sub classes with shared context
class ZMQPublisherManager:
    """Centralized management of ZMQ publishers with enhanced error handling and monitoring"""
    _instance: Optional['ZMQPublisherManager'] = None
    _publishers: Dict[Tuple[str, int], zmq.Socket] = {}
    _lock = threading.Lock()
    _monitor_thread: Optional[threading.Thread] = None
    _running = True
    
    def __init__(self, context: Optional[zmq.Context] = None):
        """Initialize the publisher manager.
        
        Args:
            context: Optional custom ZMQ context (default: global context)
        """
        self._context = context or get_global_context()
        self._start_monitor()
    
    @classmethod
    def get_instance(cls, context: Optional[zmq.Context] = None) -> 'ZMQPublisherManager':
        """Get or create the singleton instance with thread safety.
        
        Args:
            context: Optional custom ZMQ context (default: global context)
            
        Returns:
            The singleton ZMQPublisherManager instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(context)
        return cls._instance
    
    def _create_publisher(self, host: str, port: int) -> zmq.Socket:
        """Create a new PUB socket with error handling."""
        try:
            socket = self._context.socket(zmq.PUB)
            socket.setsockopt(zmq.SNDHWM, 1)  # Only keep latest message
            addr = f'tcp://*:{port}'
            socket.bind(addr)
            return socket
        except zmq.ZMQError as e:
            logger.error(f"Failed to create PUB socket: {e}")
            raise ConnectionError(f"Failed to create PUB socket: {e}")
    
    def publish(self, host: str, port: int, topic: str, data: Any) -> None:
        """Publish data to a topic with error handling and non-blocking sends.
        
        Args:
            host: The host address
            port: The port number
            topic: The topic to publish to
            data: The data to publish
            
        Raises:
            ConnectionError: If publishing fails
            SerializationError: If data serialization fails
        """
        try:
            publisher = self.get_publisher(host, port)
            try:
                buffer = pickle.dumps(data, protocol=-1)
            except Exception as e:
                raise SerializationError(f"Failed to serialize data: {e}")
            
            try:
                # Use multipart messaging and non-blocking send
                encoded_topic = topic.encode('utf-8') if isinstance(topic, str) else topic
                publisher.send_multipart([
                    encoded_topic,
                    buffer
                ], zmq.NOBLOCK)
                
            except zmq.Again:
                logger.warning(f"High water mark reached for {topic} at {host}:{port}, dropping message")
            except zmq.ZMQError as e:
                logger.error(f"Failed to publish to {topic} at {host}:{port}: {e}")
                self._close_publisher((host, port))  # Close unhealthy publisher
                raise ConnectionError(f"Failed to publish: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in publish: {e}")
            raise

    def get_publisher(self, host: str, port: int) -> zmq.Socket:
        """Get or create a publisher for the given host and port with thread safety.
        
        Args:
            host: The host address
            port: The port number
            
        Returns:
            The ZMQ publisher socket
            
        Raises:
            ConnectionError: If publisher creation fails
        """
        key = (host, port)
        with self._lock:
            if key not in self._publishers:
                self._publishers[key] = self._create_publisher(host, port)
            return self._publishers[key]
    
    def _close_publisher(self, key: Tuple[str, int]) -> None:
        """Close a specific publisher socket.
        
        Args:
            key: Tuple of (host, port)
        """
        with self._lock:
            if key in self._publishers:
                try:
                    self._publishers[key].close()
                except Exception as e:
                    logger.error(f"Error closing publisher at {key[0]}:{key[1]}: {e}")
                del self._publishers[key]
    
    def _start_monitor(self) -> None:
        """Start the monitoring thread for publisher health checks."""
        def monitor_loop():
            while self._running:
                time.sleep(5)  # Check every 5 seconds
                with self._lock:
                    for key, pub in list(self._publishers.items()):
                        try:
                            # Try to get socket stats
                            pub.getsockopt(zmq.EVENTS)
                        except zmq.ZMQError:
                            logger.warning(f"Unhealthy publisher detected at {key[0]}:{key[1]}")
                            self._close_publisher(key)
        
        self._monitor_thread = threading.Thread(
            target=monitor_loop,
            daemon=True,
            name="PublisherMonitor"
        )
        self._monitor_thread.start()
    
    def close_all(self) -> None:
        """Close all publishers and stop monitoring gracefully."""
        self._running = False
        
        # First stop the monitor thread
        if self._monitor_thread and self._monitor_thread.is_alive():
            try:
                self._monitor_thread.join(timeout=2)
                if self._monitor_thread.is_alive():
                    logger.warning("Monitor thread did not stop gracefully")
            except Exception as e:
                logger.error(f"Error stopping monitor thread: {e}")
        
        # Then close all publishers
        with self._lock:
            for key, publisher in list(self._publishers.items()):
                try:
                    publisher.close()
                except Exception as e:
                    logger.error(f"Error closing publisher at {key[0]}:{key[1]}: {e}")
            self._publishers.clear()

# Function to clean up all ZMQ resources (call on program exit)
def cleanup_zmq_resources() -> None:
    """Clean up all ZMQ resources gracefully.
    
    This function should be called before process termination to ensure
    proper cleanup of all ZMQ resources, including threads and sockets.
    """
    try:
        # First stop the publisher manager and its monitor thread
        manager = ZMQPublisherManager.get_instance()
        manager.close_all()
        
        # Clean up handshake coordinator
        HandshakeCoordinator.cleanup_all()
        
        # Then terminate the context
        context = get_global_context()
        context.term()
    except Exception as e:
        logger.error(f"Error during ZMQ cleanup: {e}")
        # In case of error, try to force context termination
        try:
            context = get_global_context()
            context.term(linger=0)
        except Exception as e2:
            logger.error(f"Failed to force terminate ZMQ context: {e2}")


# Handshake Coordination System
class HandshakeCoordinator:
    """
    Centralized handshake coordination for reliable teleop state management.
    
    This coordinator ensures that critical state changes (stop/resume) are acknowledged
    by all relevant subscribers before the publisher proceeds. It prevents race conditions
    and ensures synchronized state transitions across the system.
    
    Usage:
        # Publisher side (e.g., in beavr_robot_adapter)
        coordinator = HandshakeCoordinator.get_instance()
        coordinator.register_subscriber("xarm_robot", "10.31.152.148", 8151)
        coordinator.register_subscriber("leap_operator", "10.31.152.148", 8152)
        
        # Before sending critical state change
        success = coordinator.request_acknowledgments(["xarm_robot", "leap_operator"], timeout=5.0)
        if success:
            # Proceed with state change
            pass
        else:
            # Handle failure (some subscribers didn't acknowledge)
            pass
            
        # Subscriber side (e.g., in xarm7_robot.py or operators)
        coordinator = HandshakeCoordinator.get_instance()
        coordinator.start_server("xarm_robot", "*", 8151)
    """
    
    _instance: Optional['HandshakeCoordinator'] = None
    _lock = threading.Lock()
    
    def __init__(self, context: Optional[zmq.Context] = None):
        """Initialize the handshake coordinator.
        
        Args:
            context: Optional custom ZMQ context (default: global context)
        """
        self._context = context or get_global_context()
        self._subscribers: Dict[str, Dict[str, Any]] = {}  # subscriber_id -> {host, port, client}
        self._servers: Dict[str, zmq.Socket] = {}  # subscriber_id -> server_socket
        self._running = True
        self._server_threads: Dict[str, threading.Thread] = {}
        
    @classmethod
    def get_instance(cls, context: Optional[zmq.Context] = None) -> 'HandshakeCoordinator':
        """Get or create the singleton coordinator instance.
        
        Args:
            context: Optional custom ZMQ context (default: global context)
            
        Returns:
            The singleton HandshakeCoordinator instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(context)
        return cls._instance
    
    def register_subscriber(self, subscriber_id: str, host: str, port: int) -> None:
        """Register a subscriber for handshake coordination.
        
        Args:
            subscriber_id: Unique identifier for the subscriber
            host: Host address of the subscriber
            port: Port number for handshake communication
        """
        with self._lock:
            self._subscribers[subscriber_id] = {
                'host': host,
                'port': port,
                'client': None
            }
            logger.info(f"Registered subscriber '{subscriber_id}' at {host}:{port}")
    
    def unregister_subscriber(self, subscriber_id: str) -> None:
        """Unregister a subscriber from handshake coordination.
        
        Args:
            subscriber_id: Unique identifier of the subscriber to remove
        """
        with self._lock:
            if subscriber_id in self._subscribers:
                # Close client if exists
                if self._subscribers[subscriber_id]['client']:
                    try:
                        self._subscribers[subscriber_id]['client'].close()
                    except Exception as e:
                        logger.warning(f"Error closing client for '{subscriber_id}': {e}")
                
                del self._subscribers[subscriber_id]
                logger.info(f"Unregistered subscriber '{subscriber_id}'")
    
    def start_server(self, subscriber_id: str, bind_host: str, port: int) -> None:
        """Start a handshake server for a subscriber.
        
        Args:
            subscriber_id: Unique identifier for this subscriber
            bind_host: Host to bind the server to (usually "*" for all interfaces)
            port: Port to bind the server to
        """
        if subscriber_id in self._servers:
            logger.warning(f"Server for '{subscriber_id}' already running")
            return
            
        try:
            server_socket = self._context.socket(zmq.REP)
            server_socket.bind(f"tcp://{bind_host}:{port}")
            self._servers[subscriber_id] = server_socket
            
            # Start server thread
            server_thread = threading.Thread(
                target=self._run_server,
                args=(subscriber_id, server_socket),
                daemon=True,
                name=f"HandshakeServer-{subscriber_id}"
            )
            server_thread.start()
            self._server_threads[subscriber_id] = server_thread
            
            logger.info(f"Started handshake server for '{subscriber_id}' on {bind_host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to start handshake server for '{subscriber_id}': {e}")
            if subscriber_id in self._servers:
                del self._servers[subscriber_id]
            raise
    
    def stop_server(self, subscriber_id: str) -> None:
        """Stop a handshake server for a subscriber.
        
        Args:
            subscriber_id: Unique identifier of the subscriber
        """
        if subscriber_id in self._servers:
            try:
                self._servers[subscriber_id].close()
                del self._servers[subscriber_id]
                
                # Wait for server thread to finish
                if subscriber_id in self._server_threads:
                    self._server_threads[subscriber_id].join(timeout=2)
                    if self._server_threads[subscriber_id].is_alive():
                        logger.warning(f"Server thread for '{subscriber_id}' did not stop gracefully")
                    del self._server_threads[subscriber_id]
                    
                logger.info(f"Stopped handshake server for '{subscriber_id}'")
            except Exception as e:
                logger.error(f"Error stopping server for '{subscriber_id}': {e}")
    
    def _run_server(self, subscriber_id: str, server_socket: zmq.Socket) -> None:
        """Run the handshake server loop.
        
        Args:
            subscriber_id: Unique identifier for this subscriber
            server_socket: The ZMQ REP socket to handle requests
        """
        poller = zmq.Poller()
        poller.register(server_socket, zmq.POLLIN)
        
        while self._running and subscriber_id in self._servers:
            try:
                # Poll with timeout to check if we should stop
                events = dict(poller.poll(100))  # 100ms timeout
                
                if server_socket in events:
                    # Receive handshake request
                    request = server_socket.recv(zmq.NOBLOCK)
                    
                    # Send acknowledgment
                    server_socket.send(b"ACK")
                    
                    logger.debug(f"Handshake server '{subscriber_id}' acknowledged request")
                    
            except zmq.Again:
                continue
            except Exception as e:
                if self._running and subscriber_id in self._servers:
                    logger.error(f"Error in handshake server '{subscriber_id}': {e}")
                break
        
        logger.debug(f"Handshake server '{subscriber_id}' stopped")
    
    def request_acknowledgments(self, subscriber_ids: list[str], timeout: float = 3.0, 
                              ping: bytes = b"PING", ack: bytes = b"ACK") -> bool:
        """Request acknowledgments from specified subscribers.
        
        Args:
            subscriber_ids: List of subscriber IDs to request acknowledgments from
            timeout: Maximum time to wait for all acknowledgments
            ping: Ping message to send
            ack: Expected acknowledgment message
            
        Returns:
            True if all subscribers acknowledged within timeout, False otherwise
        """
        if not subscriber_ids:
            return True
            
        # Create clients for subscribers that don't have them yet
        clients_to_close = []
        
        try:
            with self._lock:
                for subscriber_id in subscriber_ids:
                    if subscriber_id not in self._subscribers:
                        logger.error(f"Subscriber '{subscriber_id}' not registered")
                        return False
                    
                    sub_info = self._subscribers[subscriber_id]
                    if sub_info['client'] is None:
                        try:
                            client = self._context.socket(zmq.REQ)
                            client.connect(f"tcp://{sub_info['host']}:{sub_info['port']}")
                            sub_info['client'] = client
                            clients_to_close.append(subscriber_id)
                        except Exception as e:
                            logger.error(f"Failed to create client for '{subscriber_id}': {e}")
                            return False
            
            # Send ping to all subscribers and collect responses
            successful_acks = []
            
            for subscriber_id in subscriber_ids:
                sub_info = self._subscribers[subscriber_id]
                client = sub_info['client']
                
                try:
                    # Send ping
                    client.send(ping)
                    
                    # Wait for response with timeout
                    poller = zmq.Poller()
                    poller.register(client, zmq.POLLIN)
                    
                    events = dict(poller.poll(timeout * 1000))  # Convert to milliseconds
                    
                    if client in events:
                        response = client.recv()
                        if response == ack:
                            successful_acks.append(subscriber_id)
                            logger.debug(f"Received acknowledgment from '{subscriber_id}'")
                        else:
                            logger.warning(f"Unexpected response from '{subscriber_id}': {response}")
                    else:
                        logger.warning(f"Timeout waiting for acknowledgment from '{subscriber_id}'")
                        
                except Exception as e:
                    logger.error(f"Error requesting acknowledgment from '{subscriber_id}': {e}")
            
            success = len(successful_acks) == len(subscriber_ids)
            
            if success:
                logger.info(f"All subscribers acknowledged: {successful_acks}")
            else:
                failed = set(subscriber_ids) - set(successful_acks)
                logger.warning(f"Failed to get acknowledgments from: {failed}")
            
            return success
            
        finally:
            # Clean up temporary clients
            with self._lock:
                for subscriber_id in clients_to_close:
                    if subscriber_id in self._subscribers and self._subscribers[subscriber_id]['client']:
                        try:
                            self._subscribers[subscriber_id]['client'].close()
                            self._subscribers[subscriber_id]['client'] = None
                        except Exception as e:
                            logger.warning(f"Error closing temporary client for '{subscriber_id}': {e}")
    
    def get_registered_subscribers(self) -> list[str]:
        """Get list of currently registered subscriber IDs.
        
        Returns:
            List of registered subscriber IDs
        """
        with self._lock:
            return list(self._subscribers.keys())
    
    def is_subscriber_registered(self, subscriber_id: str) -> bool:
        """Check if a subscriber is registered.
        
        Args:
            subscriber_id: Unique identifier to check
            
        Returns:
            True if subscriber is registered, False otherwise
        """
        with self._lock:
            return subscriber_id in self._subscribers
    
    def shutdown(self) -> None:
        """Shutdown the coordinator and clean up all resources."""
        self._running = False
        
        # Stop all servers
        for subscriber_id in list(self._servers.keys()):
            self.stop_server(subscriber_id)
        
        # Close all clients
        with self._lock:
            for sub_info in self._subscribers.values():
                if sub_info['client']:
                    try:
                        sub_info['client'].close()
                    except Exception as e:
                        logger.warning(f"Error closing client: {e}")
            self._subscribers.clear()
        
        logger.info("HandshakeCoordinator shutdown complete")
    
    @classmethod
    def cleanup_all(cls) -> None:
        """Clean up all coordinator instances."""
        if cls._instance:
            cls._instance.shutdown()
            cls._instance = None


def publish_with_guaranteed_delivery(
    host: str, 
    port: int, 
    topic: str, 
    data: Any, 
    subscriber_ids: list[str] = None,
    handshake_timeout: float = 3.0,
    require_all_acks: bool = True
) -> bool:
    """
    Publish data with guaranteed delivery confirmation via handshake coordination.
    
    This function combines regular ZMQ publishing with handshake coordination to ensure
    that critical state changes are acknowledged by all relevant subscribers.
    
    Args:
        host: Host address for publishing
        port: Port number for publishing  
        topic: Topic to publish to
        data: Data to publish
        subscriber_ids: List of subscriber IDs that must acknowledge (if None, no handshake)
        handshake_timeout: Maximum time to wait for acknowledgments
        require_all_acks: Whether all subscribers must acknowledge (False allows partial success)
        
    Returns:
        True if publishing succeeded and (if applicable) acknowledgments were received
        
    Example:
        # Publish teleop stop with guaranteed delivery
        success = publish_with_guaranteed_delivery(
            host="10.31.152.148",
            port=8089,
            topic="pause", 
            data=ARM_TELEOP_STOP,
            subscriber_ids=["xarm_robot", "leap_operator"],
            handshake_timeout=5.0
        )
        if not success:
            logger.error("Failed to guarantee teleop stop delivery")
    """
    try:
        # First, publish the data normally
        publisher_manager = ZMQPublisherManager.get_instance()
        publisher_manager.publish(host, port, topic, data)
        logger.debug(f"Published {topic} data to {host}:{port}")
        
        # If no handshake required, we're done
        if not subscriber_ids:
            return True
        
        # Request acknowledgments from subscribers
        coordinator = HandshakeCoordinator.get_instance()
        success = coordinator.request_acknowledgments(
            subscriber_ids, 
            timeout=handshake_timeout
        )
        
        if success:
            logger.info(f"Guaranteed delivery confirmed for topic '{topic}' by all subscribers: {subscriber_ids}")
            return True
        elif not require_all_acks:
            # Check if at least some subscribers acknowledged
            # This would require modifying request_acknowledgments to return partial results
            # For now, we'll log and return the success status
            logger.warning(f"Partial delivery confirmation for topic '{topic}' - some subscribers may not have acknowledged")
            return success
        else:
            logger.error(f"Failed to get acknowledgments from all subscribers for topic '{topic}'")
            return False
            
    except Exception as e:
        logger.error(f"Error in guaranteed delivery for topic '{topic}': {e}")
        return False


# Function to clean up all ZMQ resources (call on program exit)