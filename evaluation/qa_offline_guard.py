"""Launcher: run serve.py with outbound networking blocked (Windows stand-in for `docker run --network none`).

Usage: python evaluation/qa_offline_guard.py serve.py --port 8020
Any connect()/DNS lookup to a non-loopback host raises OSError and is appended (one JSON line) to $QA_OFFLINE_LOG.
Binding/listening on 0.0.0.0 or 127.0.0.1 is untouched, so the server itself still works.
"""
from __future__ import annotations

import json
import os
import runpy
import socket
import sys
import traceback

LOG = os.environ.get("QA_OFFLINE_LOG")
LOCAL = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::", "", None}


def _record(kind, host):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "host": str(host), "stack": traceback.format_stack(limit=6)[:-1]}) + "\n")


def _host(addr):
    return addr[0] if isinstance(addr, tuple) and addr else addr


_connect, _connect_ex, _gai = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo


def connect(self, addr):
    h = _host(addr)
    if self.family in (socket.AF_INET, socket.AF_INET6) and h not in LOCAL:
        _record("connect", h); raise OSError(f"QA offline guard: blocked connect to {h}")
    return _connect(self, addr)


def connect_ex(self, addr):
    h = _host(addr)
    if self.family in (socket.AF_INET, socket.AF_INET6) and h not in LOCAL:
        _record("connect_ex", h); return 113
    return _connect_ex(self, addr)


def getaddrinfo(host, *a, **k):
    if host not in LOCAL:
        try:
            import ipaddress
            ipaddress.ip_address(host)
        except Exception:
            _record("dns", host); raise socket.gaierror(f"QA offline guard: blocked DNS lookup of {host}")
    return _gai(host, *a, **k)


socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
