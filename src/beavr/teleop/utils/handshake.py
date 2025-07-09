from __future__ import annotations

from time import monotonic
import zmq
from .network import get_global_context

"""Light-weight ZMQ handshake helpers.

A handshake is a one-shot REQ/REP exchange that allows two independent ZMQ
processes to *synchronise* state transitions without adding extra hard-coded
socket boiler-plate to every component.

Typical usage – **adapter / client side**
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
~~~~~~~~~~
* Only one extra port per handshake pair.
* No additional threads needed – `poll_once()` can be integrated into an
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

    def __init__(self, host: str, port: int, *, ping: bytes = _DEFAULT_PING, ack: bytes = _DEFAULT_ACK):
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

    def __init__(self, host: str, port: int, *, ping: bytes = _DEFAULT_PING, ack: bytes = _DEFAULT_ACK):
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