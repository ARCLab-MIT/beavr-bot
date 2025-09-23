import contextlib
import inspect
import logging
import pickle
import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import zmq

from .utils import get_global_context

logger = logging.getLogger(__name__)


class BasePublisher:
    """Base class for all publishers"""

    def __init__(
        self,
        host: str,
        port: int,
        socket_type: int = zmq.PUB,
        context: Optional[zmq.Context] = None,
    ):
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
            addr = f"tcp://*:{self._port}"
            self._socket.bind(addr)
        except zmq.ZMQError as e:
            logger.error(f"Failed to initialize socket: {e}")
            raise

    def stop(self) -> None:
        """Close the publisher socket."""
        if self._socket:
            self._socket.close()
            self._socket = None


class PublisherThread(threading.Thread):
    """Thread that owns a PUB socket and handles publishing via a queue."""

    def __init__(self, host: str, port: int, context: Optional[zmq.Context] = None):
        """Initialize publisher thread.

        Args:
            host: The host address to bind to
            port: The port number to bind to
            context: Optional custom ZMQ context (default: global context)
        """
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._context = context or get_global_context()
        self._socket = None
        self._running = True
        self._queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self._started = threading.Event()

    def send(self, topic: str, data: Any) -> None:
        """Send data to the publisher queue (thread-safe).

        Args:
            topic: The topic to publish to
            data: The data to publish
        """
        try:
            # Serialize data here to avoid blocking the main thread
            buffer = pickle.dumps(data, protocol=-1)
            self._queue.put_nowait((topic, buffer))
        except queue.Full:
            logger.warning(f"Publisher queue full for {self._host}:{self._port}, dropping message")
        except Exception as e:
            logger.error(f"Error serializing data for publisher: {e}")

    def stop(self) -> None:
        """Stop the publisher thread gracefully."""
        self._running = False
        # Put a sentinel value to unblock the queue
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait((None, None))
        self.join(timeout=2)
        if self.is_alive():
            logger.warning("Publisher thread did not stop gracefully")

    def run(self) -> None:
        """Main publisher loop with socket creation in worker thread."""
        try:
            # Create socket in the worker thread
            self._socket = self._context.socket(zmq.PUB)
            self._socket.setsockopt(zmq.SNDHWM, 1)  # Only keep latest message
            addr = f"tcp://*:{self._port}"
            self._socket.bind(addr)

            # Signal that socket is ready
            self._started.set()

            while self._running:
                try:
                    # Get data from queue with timeout to allow checking _running
                    topic, buffer = self._queue.get(timeout=0.1)

                    # Check for sentinel value
                    if topic is None:
                        break

                    try:
                        # Use multipart messaging and non-blocking send
                        encoded_topic = topic.encode("utf-8") if isinstance(topic, str) else topic
                        self._socket.send_multipart([encoded_topic, buffer], zmq.NOBLOCK)

                    except zmq.Again:
                        logger.warning(
                            f"High water mark reached for {topic} at {self._host}:{self._port}, dropping message"
                        )
                    except zmq.ZMQError as e:
                        logger.error(f"Failed to publish to {topic} at {self._host}:{self._port}: {e}")
                        break

                except queue.Empty:
                    # Queue was empty, just continue
                    continue
                except Exception as e:
                    if self._running:
                        logger.error(f"Error in publisher loop: {e}")
                    break

        except Exception as e:
            logger.error(f"Failed to initialize publisher socket: {e}")
        finally:
            # Ensure socket is closed when thread exits
            if self._socket:
                try:
                    self._socket.close()
                except Exception as e:
                    logger.warning(f"Error closing socket in cleanup: {e}")
                self._socket = None

    def wait_for_start(self, timeout: float = 5.0) -> bool:
        """Wait for the publisher thread to start and socket to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if started successfully, False if timeout
        """
        return self._started.wait(timeout=timeout)


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
        _, buffer = cv2.imencode(".jpg", rgb_image, [int(cv2.IMWRITE_WEBP_QUALITY), 10])
        self._socket.send_multipart([b"image", np.array(buffer).tobytes()], zmq.NOBLOCK)


# Improved Pub/Sub classes with shared context
class ZMQPublisherManager:
    """Centralized management of ZMQ publishers with thread-safe socket ownership"""

    _instance: Optional["ZMQPublisherManager"] = None
    _publishers: Dict[Tuple[str, int], PublisherThread] = {}
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
    def get_instance(cls, context: Optional[zmq.Context] = None) -> "ZMQPublisherManager":
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

    def _create_publisher_thread(self, host: str, port: int) -> PublisherThread:
        """Create a new publisher thread with error handling."""
        try:
            publisher_thread = PublisherThread(host, port, self._context)
            publisher_thread.start()

            # Wait for the thread to start and socket to be ready
            if not publisher_thread.wait_for_start(timeout=5.0):
                raise ConnectionError(f"Publisher thread failed to start for {host}:{port}")

            return publisher_thread
        except Exception as e:
            caller = "unknown"
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                module = inspect.getmodule(caller_frame)
                mod_name = module.__name__ if module else "unknown"
                caller = f"{mod_name}.{caller_frame.f_code.co_name}"

            logger.error(f"Failed to create publisher thread in {caller}: {e}")
            raise ConnectionError(f"Failed to create publisher for {host}:{port}: {e}") from e

    def publish(self, host: str, port: int, topic: str, data: Any) -> None:
        """Publish data to a topic with thread-safe queue-based communication.

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
            publisher_thread = self.get_publisher_thread(host, port)
            publisher_thread.send(topic, data)
        except Exception as e:
            logger.error(f"Unexpected error in publish: {e}")
            raise

    def get_publisher_thread(self, host: str, port: int) -> PublisherThread:
        """Get or create a publisher thread for the given host and port with thread safety.

        Args:
            host: The host address
            port: The port number

        Returns:
            The PublisherThread instance

        Raises:
            ConnectionError: If publisher creation fails
        """
        key = (host, port)
        with self._lock:
            if key not in self._publishers:
                self._publishers[key] = self._create_publisher_thread(host, port)
            return self._publishers[key]

    def _close_publisher(self, key: Tuple[str, int]) -> None:
        """Close a specific publisher thread.

        Args:
            key: Tuple of (host, port)
        """
        with self._lock:
            if key in self._publishers:
                try:
                    self._publishers[key].stop()
                except Exception as e:
                    logger.error(f"Error stopping publisher at {key[0]}:{key[1]}: {e}")
                del self._publishers[key]

    def get_bound_ports(self) -> Dict[Tuple[str, int], str]:
        """Get the endpoint for each currently bound publisher.

        Returns:
            Mapping of ``(host, port)`` tuples to their bound endpoint strings.
        """
        with self._lock:
            return {
                key: f"tcp://*:{key[1]}"  # key is (host, port), so key[1] is port
                for key in self._publishers
            }

    def _start_monitor(self) -> None:
        """Start the monitoring thread for publisher health checks."""

        def monitor_loop():
            while self._running:
                time.sleep(5)  # Check every 5 seconds
                with self._lock:
                    for key, publisher_thread in list(self._publishers.items()):
                        try:
                            # Check if thread is still alive
                            if not publisher_thread.is_alive():
                                logger.warning(f"Dead publisher thread detected at {key[0]}:{key[1]}")
                                self._close_publisher(key)
                        except Exception as e:
                            logger.error(f"Error monitoring publisher at {key[0]}:{key[1]}: {e}")

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name="PublisherMonitor")
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
            for key, publisher_thread in list(self._publishers.items()):
                try:
                    publisher_thread.stop()
                except Exception as e:
                    logger.error(f"Error stopping publisher at {key[0]}:{key[1]}: {e}")
            self._publishers.clear()
