"""Minimal Source-RCON client. Factorio implements the same protocol.

Shared by server.py (model inference round trip) and parity.py (engine
parity harness); lives in its own module so parity.py can talk to
Factorio without importing the torch/model stack.
"""

from __future__ import annotations

import socket
import struct
from typing import Optional


class RconError(RuntimeError):
    pass


class RconClient:
    """Minimal Source-RCON client. Factorio implements the same protocol."""

    AUTH = 3
    EXEC = 2
    RESP = 0

    def __init__(self, host: str, port: int, password: str, timeout: float = 5.0):
        self.host, self.port, self.password, self.timeout = host, port, password, timeout
        self._sock: Optional[socket.socket] = None
        self._counter = 1

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *a):
        self.close()

    def connect(self):
        self._sock = socket.create_connection((self.host, self.port), self.timeout)
        rid = self._send(self.AUTH, self.password)
        # Factorio sends an empty RESP first, then an AUTH response.
        resp_id, _, _ = self._recv()
        if resp_id == -1:
            raise RconError("RCON auth failed (bad password)")
        if resp_id != rid:
            # Drain the empty packet and read again.
            resp_id, _, _ = self._recv()
            if resp_id == -1:
                raise RconError("RCON auth failed (bad password)")

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def exec(self, command: str) -> str:
        if self._sock is None:
            self.connect()
        self._send(self.EXEC, command)
        _, _, body = self._recv()
        return body

    def _send(self, ptype: int, body: str) -> int:
        assert self._sock is not None
        rid = self._counter
        self._counter += 1
        payload = body.encode("utf-8")
        # length = id(4) + type(4) + body(N) + 2 null bytes
        packet = struct.pack("<ii", rid, ptype) + payload + b"\x00\x00"
        self._sock.sendall(struct.pack("<i", len(packet)) + packet)
        return rid

    def _recv(self):
        assert self._sock is not None
        raw_len = self._read_exact(4)
        (length,) = struct.unpack("<i", raw_len)
        data = self._read_exact(length)
        rid, ptype = struct.unpack("<ii", data[:8])
        body = data[8:-2].decode("utf-8", errors="replace")
        return rid, ptype, body

    def _read_exact(self, n: int) -> bytes:
        assert self._sock is not None
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise RconError("RCON socket closed")
            buf += chunk
        return buf
