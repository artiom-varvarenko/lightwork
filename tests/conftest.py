"""Shared fixtures.

``no_network``: any connection to a non-loopback address raises. Replay, verify, export and audit tests must use it
(they must work with no key and no network). Loopback stays allowed for in-process servers.
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class NetworkBlocked(RuntimeError):
    pass


def _is_loopback(address) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return True
    host = address[0]
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture
def no_network(monkeypatch):
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guard(address):
        if not _is_loopback(address):
            raise NetworkBlocked(f"outbound connection to {address!r} blocked by no_network")

    def connect(self, address):
        guard(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        guard(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if host not in (None, "localhost") and not _is_loopback((host,)):
            raise NetworkBlocked(f"DNS lookup of {host!r} blocked by no_network")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    yield


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def synthetic_kit_run() -> Path:
    return FIXTURES / "runs" / "synthetic-kit"
