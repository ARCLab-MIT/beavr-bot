from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pytest


@dataclass
class BusMessage:
    topic: str
    data: Any


class InMemoryBus:
    """A very small in-memory pub/sub for tests keyed by (port, topic).

    - publish(host, port, topic, data) stores the latest message
    - subscribers call recv() to fetch the latest message
    """

    def __init__(self) -> None:
        self._latest_by_key: Dict[Tuple[int, str], BusMessage] = {}

    def publish(self, host: str, port: int, topic: str, data: Any) -> None:
        self._latest_by_key[(port, topic)] = BusMessage(topic=topic, data=data)

    def recv_latest(self, port: int, topic: str) -> Optional[Any]:
        msg = self._latest_by_key.get((port, topic))
        return None if msg is None else msg.data


class FakeZMQPublisherThread:
    """Placeholder used by FakeZMQPublisherManager.get_publisher_thread()."""

    def __init__(self, bus: InMemoryBus, host: str, port: int) -> None:
        self._bus = bus
        self._host = host
        self._port = port

    def send(self, topic: str, data: Any) -> None:
        self._bus.publish(self._host, self._port, topic, data)


class FakeZMQPublisherManager:
    """Drop-in replacement for ZMQPublisherManager for tests."""

    def __init__(self, bus: InMemoryBus) -> None:
        self._bus = bus
        self._threads: Dict[Tuple[str, int], FakeZMQPublisherThread] = {}

    def get_publisher_thread(self, host: str, port: int) -> FakeZMQPublisherThread:
        key = (host, port)
        if key not in self._threads:
            self._threads[key] = FakeZMQPublisherThread(self._bus, host, port)
        return self._threads[key]

    # keep signature parity with real manager
    def publish(self, host: str, port: int, topic: str, data: Any) -> None:
        self.get_publisher_thread(host, port).send(topic, data)

    def close_all(self) -> None:
        """Clean up all publisher threads."""
        self._threads.clear()


class FakeHandshakeCoordinator:
    """Drop-in replacement for HandshakeCoordinator for tests."""

    def __init__(self) -> None:
        self._subscribers = {}
        self._servers = {}

    def start_server(self, subscriber_id: str, bind_host: str, port: int) -> None:
        """Mock starting a handshake server."""
        self._servers[subscriber_id] = {"host": bind_host, "port": port}

    def stop_server(self, subscriber_id: str) -> None:
        """Mock stopping a handshake server."""
        if subscriber_id in self._servers:
            del self._servers[subscriber_id]

    @classmethod
    def get_instance(cls, context: Any = None) -> "FakeHandshakeCoordinator":
        """Mock singleton getter."""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    @classmethod
    def cleanup_all(cls) -> None:
        """Mock cleanup."""
        if hasattr(cls, "_instance"):
            delattr(cls, "_instance")


class FakeZMQSubscriber:
    """Synchronous test subscriber with recv_keypoints() API.

    Matches the constructor signature used across the codebase.
    """

    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        context: Any = None,
        message_type: Any = None,
        serializer: Any = None,
    ) -> None:
        self._bus = _TEST_BUS
        self._port = port
        self._topic = topic

    def recv_keypoints(self) -> Optional[Any]:
        return self._bus.recv_latest(self._port, self._topic)

    # API compatibility for cleanup used in components
    def stop(self) -> None:
        return None


# Global bus used by fakes. Initialized in fixture setup.
_TEST_BUS: InMemoryBus = InMemoryBus()


@pytest.fixture(autouse=True)
def _patch_messaging_layer(monkeypatch):
    """Auto-apply fakes for publisher and subscriber in all tests.

    - ZMQPublisherManager.get_instance(...) returns our FakeZMQPublisherManager
    - ZMQSubscriber in operator/robot/transform modules points to FakeZMQSubscriber
    """
    # Fresh bus per test
    global _TEST_BUS
    _TEST_BUS = InMemoryBus()

    fake_manager = FakeZMQPublisherManager(_TEST_BUS)

    # Patch manager factory
    import beavr.teleop.common.network.handshake as handshake_mod
    import beavr.teleop.common.network.publisher as pub_mod

    def _fake_get_instance(context: Any = None):
        return fake_manager

    monkeypatch.setattr(
        pub_mod.ZMQPublisherManager,
        "get_instance",
        classmethod(lambda cls, context=None: fake_manager),
    )

    # Patch HandshakeCoordinator
    fake_handshake = FakeHandshakeCoordinator()
    monkeypatch.setattr(
        handshake_mod.HandshakeCoordinator,
        "get_instance",
        classmethod(lambda cls, context=None: fake_handshake),
    )

    # Patch ZMQSubscriber symbol in modules that import it directly
    monkeypatch.setattr(
        "beavr.teleop.components.operator.robots.xarm7_operator.ZMQSubscriber",
        FakeZMQSubscriber,
        raising=False,
    )
    monkeypatch.setattr(
        "beavr.teleop.components.interface.robots.xarm7_robot.ZMQSubscriber",
        FakeZMQSubscriber,
        raising=False,
    )
    monkeypatch.setattr(
        "beavr.teleop.components.detector.vr.keypoint_transform.ZMQSubscriber",
        FakeZMQSubscriber,
        raising=False,
    )

    yield


@pytest.fixture
def bus() -> InMemoryBus:
    return _TEST_BUS
