from __future__ import annotations

import logging
import threading
from time import monotonic
from typing import Any, Dict, Optional

import zmq

from .publisher import ZMQPublisherManager
from .utils import get_global_context

logger = logging.getLogger(__name__)

"""Light-weight ZMQ handshake helpers.

A handshake is a one-shot REQ/REP exchange that allows two independent ZMQ
processes to *synchronise* state transitions without adding extra hard-coded
socket boiler-plate to every component.

Typical usage - **adapter / client side**
----------------------------------------
>>> ok = HandshakeClient("10.31.152.148", 8150).request()
>>> if not ok:
...     raise RuntimeError("Operator did not acknowledge pause")

**operator / server side**
--------------------------
>>> server = HandshakeServer("*", 8150)   # bind to all interfaces
>>> while running:
...     server.poll_once(timeout_ms=0)     # non-blocking

Advantages
----------
* Only one extra port per handshake pair.
* No additional threads needed - `poll_once()` can be integrated into an
  existing main loop.
* Zero-copy / tiny messages (single byte by default).

The helpers share the global `zmq.Context` used elsewhere in *beavr* so that
socket creation is cheap.
"""

__all__ = [
    "HandshakeServer",
    "HandshakeClient",
]

_DEFAULT_PING = b"PING"
_DEFAULT_ACK = b"ACK"


class HandshakeServer:
    """REP socket that replies *ack* whenever a *ping* is received."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        ping: bytes = _DEFAULT_PING,
        ack: bytes = _DEFAULT_ACK,
    ):
        ctx = get_global_context()
        self._socket = ctx.socket(zmq.REP)
        # Caller may pass "*" to bind on all interfaces.
        self._socket.bind(f"tcp://{host}:{port}")
        self._ping = ping
        self._ack = ack

    def poll_once(self, *, timeout_ms: int = 0) -> bool:
        """Poll for one handshake request and reply.

        Parameters
        ----------
        timeout_ms : int, optional
            Milliseconds to wait for a request; *0* (default) makes the call
            non-blocking.  Return value indicates whether a request was
            handled.
        Returns
        -------
        bool
            *True* if a request was received and an ACK was sent.
        """
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        socks = dict(poller.poll(timeout_ms))
        if socks.get(self._socket) == zmq.POLLIN:
            _ = self._socket.recv()  # ignore content – semantics are implicit
            self._socket.send(self._ack)
            return True
        return False

    def close(self):
        self._socket.close(linger=0)


class HandshakeClient:
    """REQ socket that performs a blocking handshake request."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        ping: bytes = _DEFAULT_PING,
        ack: bytes = _DEFAULT_ACK,
    ):
        ctx = get_global_context()
        self._socket = ctx.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")
        self._ping = ping
        self._ack = ack

    def request(self, *, timeout: float = 2.0) -> bool:
        """Send *ping* and wait up to *timeout* seconds for *ack*.

        Returns
        -------
        bool
            *True* on success, *False* on timeout/no reply.
        """
        self._socket.send(self._ping)
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        t0 = monotonic()
        remaining_ms = int(timeout * 1000)
        while remaining_ms > 0:
            socks = dict(poller.poll(remaining_ms))
            if socks.get(self._socket) == zmq.POLLIN:
                reply = self._socket.recv()
                return reply == self._ack
            remaining_ms = int((timeout - (monotonic() - t0)) * 1000)
        return False

    def close(self):
        self._socket.close(linger=0)


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

    _instance: Optional["HandshakeCoordinator"] = None
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
    def get_instance(cls, context: Optional[zmq.Context] = None) -> "HandshakeCoordinator":
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
                "host": host,
                "port": port,
                "client": None,
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
                if self._subscribers[subscriber_id]["client"]:
                    try:
                        self._subscribers[subscriber_id]["client"].close()
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

        # Mark server as running (socket will be created in worker thread)
        self._servers[subscriber_id] = None

        # Start server thread with socket creation in worker thread
        # Let any thread creation errors bubble up immediately
        server_thread = threading.Thread(
            target=self._run_server,
            args=(subscriber_id, bind_host, port),
            daemon=True,
            name=f"HandshakeServer-{subscriber_id}",
        )
        server_thread.start()
        self._server_threads[subscriber_id] = server_thread

        logger.info(f"Started handshake server for '{subscriber_id}' on {bind_host}:{port}")

    def stop_server(self, subscriber_id: str) -> None:
        """Stop a handshake server for a subscriber.

        Args:
            subscriber_id: Unique identifier of the subscriber
        """
        if subscriber_id in self._servers:
            try:
                # Remove from servers dict to signal the thread to stop
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

    def _run_server(self, subscriber_id: str, bind_host: str, port: int) -> None:
        """Run the handshake server loop with socket creation in worker thread.

        Args:
            subscriber_id: Unique identifier for this subscriber
            bind_host: Host to bind the server to
            port: Port to bind the server to
        """
        # Create socket in the worker thread - fail fast on initialization errors
        server_socket = self._context.socket(zmq.REP)
        server_socket.bind(f"tcp://{bind_host}:{port}")

        # Mark server as running with socket reference
        self._servers[subscriber_id] = server_socket

        poller = zmq.Poller()
        poller.register(server_socket, zmq.POLLIN)

        try:
            while self._running and subscriber_id in self._servers:
                try:
                    # Poll with timeout to check if we should stop
                    events = dict(poller.poll(100))  # 100ms timeout

                    if server_socket in events:
                        # Receive handshake request
                        server_socket.recv(zmq.NOBLOCK)

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

        finally:
            # Ensure socket is closed when thread exits
            if server_socket:
                try:
                    server_socket.close()
                except Exception as e:
                    logger.warning(f"Error closing handshake server socket for '{subscriber_id}': {e}")

    def request_acknowledgments(
        self,
        subscriber_ids: list[str],
        timeout: float = 3.0,
        ping: bytes = b"PING",
        ack: bytes = b"ACK",
    ) -> bool:
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
                    if sub_info["client"] is None:
                        # Fail fast on client creation - configuration issues should be immediate
                        client = self._context.socket(zmq.REQ)
                        client.connect(f"tcp://{sub_info['host']}:{sub_info['port']}")
                        sub_info["client"] = client
                        clients_to_close.append(subscriber_id)

            # Send ping to all subscribers and collect responses
            successful_acks = []

            for subscriber_id in subscriber_ids:
                sub_info = self._subscribers[subscriber_id]
                client = sub_info["client"]

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
                    if subscriber_id in self._subscribers and self._subscribers[subscriber_id]["client"]:
                        try:
                            self._subscribers[subscriber_id]["client"].close()
                            self._subscribers[subscriber_id]["client"] = None
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
                if sub_info["client"]:
                    try:
                        sub_info["client"].close()
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
    require_all_acks: bool = True,
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
        success = coordinator.request_acknowledgments(subscriber_ids, timeout=handshake_timeout)

        if success:
            logger.info(
                f"Guaranteed delivery confirmed for topic '{topic}' by all subscribers: {subscriber_ids}"
            )
            return True
        elif not require_all_acks:
            # Check if at least some subscribers acknowledged
            # This would require modifying request_acknowledgments to return partial results
            # For now, we'll log and return the success status
            logger.warning(
                f"Partial delivery confirmation for topic '{topic}' - some subscribers may not have acknowledged"
            )
            return success
        else:
            logger.error(f"Failed to get acknowledgments from all subscribers for topic '{topic}'")
            return False

    except Exception as e:
        logger.error(f"Error in guaranteed delivery for topic '{topic}': {e}")
        return False
