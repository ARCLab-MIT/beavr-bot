import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar

import cv2
import numpy as np
import zmq
from typing_extensions import Generic

from .serialization import PickleSerializer, Serializer
from .utils import get_global_context

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseSubscriber(threading.Thread, ABC, Generic[T]):
    """Base class for all subscribers with graceful shutdown support.

    Optionally supply ``message_type`` for static and runtime type safety.
    When provided, received objects are checked with ``isinstance`` and cast to ``T``.
    """

    def __init__(
        self,
        host: str,
        port: int,
        topic: str = "",
        socket_type: int = zmq.SUB,
        context: Optional[zmq.Context] = None,
        message_type: Optional[Type[T]] = None,
        serializer: Optional[Serializer[T]] = None,
    ):
        """Initialize subscriber.

        Args:
            host: The host address to connect to
            port: The port number to connect to
            topic: The topic to subscribe to (empty for all topics)
            socket_type: The ZMQ socket type (default: SUB)
            context: Optional custom ZMQ context (default: global context)
            message_type: Optional expected Python type for incoming messages
            serializer: Optional serializer for payloads; defaults to PickleSerializer[T]
        """
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._topic = topic
        self._socket = None
        self._running = True
        self._poller = None
        self._context = context or get_global_context()
        # Store socket configuration for creation in worker thread
        self._socket_type = socket_type
        # Optional runtime type check for received payloads
        self._message_type: Optional[Type[T]] = message_type
        # Decoder/encoder for payloads; default to Pickle with expected_type
        self._serializer: Serializer[T] = serializer or PickleSerializer(message_type)

    def _init_socket(self, socket_type: int):
        """Initialize the socket in the worker thread."""
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

            addr = f"tcp://{self._host}:{self._port}"
            self._socket.connect(addr)
            if socket_type == zmq.SUB:
                self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)

            # Create poller and register socket
            self._poller = zmq.Poller()
            self._poller.register(self._socket, zmq.POLLIN)

        except zmq.ZMQError as e:
            logger.error(f"Failed to initialize socket: {e}")
            raise

    def stop(self):
        """Stop the subscriber thread gracefully."""
        self._running = False
        if self._socket and self._poller:
            try:
                self._poller.unregister(self._socket)
                self._socket.close()
                self._socket = None
                self._poller = None
            except Exception as e:
                logger.warning(f"Error closing socket: {e}")
        self.join(timeout=2)
        if self.is_alive():
            logger.warning("Subscriber thread did not stop gracefully")

    @abstractmethod
    def process_message(self, message: T) -> None:
        """Process received message.

        Args:
            message: The received message data
        """
        pass

    def run(self):
        """Main subscriber loop with socket creation in worker thread."""
        try:
            # Create socket in the worker thread
            self._init_socket(self._socket_type)

            while self._running:
                try:
                    if self._socket is None or self._poller is None:
                        logger.error("Socket or poller is None, breaking subscriber loop")
                        break

                    # Poll with timeout to check _running periodically
                    events = dict(self._poller.poll(100))  # 100ms timeout

                    if self._socket in events:
                        try:
                            _, payload = self._socket.recv_multipart(zmq.NOBLOCK)
                            try:
                                data_typed = self._serializer.decode(payload)
                                self.process_message(data_typed)
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
                        if (
                            hasattr(self, "_last_poll_log")
                            and time.time() - self._last_poll_log > 5
                            or not hasattr(self, "_last_poll_log")
                        ):
                            self._last_poll_log = time.time()

                except Exception as e:
                    if self._running:
                        logger.error(f"Error in subscriber loop: {e}")
                    break
        finally:
            # Ensure socket is closed when thread exits
            if self._socket:
                try:
                    self._socket.close()
                except Exception as e:
                    logger.warning(f"Error closing socket in cleanup: {e}")
                self._socket = None


class ZMQCompressedImageReceiver(BaseSubscriber):
    """Subscriber for compressed images using multipart messaging."""

    def __init__(self, host: str, port: int):
        """Initialize compressed image receiver.

        Args:
            host: The host address to connect to
            port: The port number to connect to
        """
        from .serialization import RawBytesSerializer

        super().__init__(host, port, "image", zmq.SUB, serializer=RawBytesSerializer())
        self._last_image: Optional[np.ndarray] = None
        self.start()  # Start the subscriber thread

    def process_message(self, data: bytes) -> None:
        """Process received image data.

        Args:
            data: Raw bytes of compressed image data
        """
        encoded_data = np.fromstring(data, np.uint8)
        self._last_image = cv2.imdecode(encoded_data, 1)

    def recv_image(self) -> Optional[np.ndarray]:
        """Get the latest received image.

        Returns:
            The latest image array if available, None otherwise
        """
        return self._last_image


class ZMQSubscriber(BaseSubscriber[T], Generic[T]):
    """Subscriber for data using multipart messaging.

    Provide ``message_type`` to enable static and runtime typing of payloads.
    """

    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        context: Optional[zmq.Context] = None,
        message_type: Optional[Type[T]] = None,
        serializer: Optional[Serializer[T]] = None,
    ):
        super().__init__(
            host,
            port,
            topic,
            zmq.SUB,
            context,
            message_type=message_type,
            serializer=serializer,
        )
        self._last_data: Optional[T] = None
        self.start()

    def process_message(self, data: T) -> None:
        """Process received keypoint data.

        Args:
            data: The received keypoint data
        """
        self._last_data = data

    def recv_keypoints(self) -> Optional[T]:
        """Get the latest keypoint data.

        Returns:
            The latest keypoint data if available, None otherwise
        """
        data = self._last_data
        if data is not None:
            pass
        return data


class ZMQButtonFeedbackSubscriber(BaseSubscriber[T], Generic[T]):
    """Subscriber for button/feedback using multipart messaging.

    Optionally supply ``message_type`` for typed payloads.
    """

    def __init__(
        self,
        host: str,
        port: int,
        context: Optional[zmq.Context] = None,
        message_type: Optional[Type[T]] = None,
        serializer: Optional[Serializer[T]] = None,
    ):
        super().__init__(
            host,
            port,
            "",
            zmq.SUB,
            context,
            message_type=message_type,
            serializer=serializer,
        )  # Empty topic to receive all messages
        self._last_data: Optional[T] = None
        self.start()

    def process_message(self, data: T) -> None:
        """Process received button feedback data.

        Args:
            data: The received button feedback data
        """
        self._last_data = data

    def recv_keypoints(self) -> Optional[T]:
        """Get the latest button feedback data.

        Returns:
            The latest button feedback data if available, None otherwise
        """
        return self._last_data
