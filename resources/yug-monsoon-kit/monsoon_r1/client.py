"""HTTP client side shared by the grader and the team-facing schema validator: URL safety, one /simulate call with a hard
deadline, /health. Contains nothing about the hidden windows, so it is safe to ship to teams."""
from __future__ import annotations
import ipaddress, json, socket, threading, time
from dataclasses import dataclass
from urllib.parse import urlparse
import numpy as np

from . import contract as C

LIMIT_S = 5.0
MAX_BYTES = 80 * 1024 * 1024
S_STEPS = C.SCORE_STEPS


@dataclass
class CaseResult:
    ok: bool
    out: np.ndarray | None = None
    wall_s: float = 0.0
    server_ms: float | None = None
    error: str = ""


def check_url(url: str, allow_private: bool = False) -> str:
    """Return '' if the URL is acceptable, else the reason it is refused."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return "only absolute http(s) URLs are allowed"
    if allow_private:
        return ""
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except OSError as e:
        return f"host does not resolve: {e}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return f"refusing non-public address {ip}"
    return ""


def _quiet(fn):
    try: fn()
    except Exception: pass


def call_simulate(url: str, payload: dict, session=None, limit_s: float = LIMIT_S, allow_private: bool = False) -> CaseResult:
    import requests
    why = check_url(url, allow_private)
    if why:
        return CaseResult(False, error=why)
    sess = session or requests.Session(); sess.trust_env = False
    t0 = time.perf_counter()
    try:
        r = sess.post(url.rstrip("/") + "/simulate", json=payload, timeout=(3.0, limit_s), allow_redirects=False, stream=True)
        box: dict = {}

        def reader():                                   # read in a worker so the deadline is real even if the server drips bytes
            try:
                buf = bytearray()
                for chunk in r.iter_content(1 << 16):
                    buf += chunk
                    if len(buf) > MAX_BYTES:
                        box["big"] = True; return
                box["body"] = bytes(buf)
            except Exception as e:
                box["err"] = type(e).__name__
        th = threading.Thread(target=reader, daemon=True); th.start()
        th.join(max(0.0, limit_s - (time.perf_counter() - t0)))
        if th.is_alive():
            threading.Thread(target=lambda: _quiet(r.close), daemon=True).start()   # close() can block on the reader's lock: never wait for it
            return CaseResult(False, wall_s=time.perf_counter() - t0, error=f"timeout (> {limit_s}s)")
        wall = time.perf_counter() - t0
        if box.get("big"):
            return CaseResult(False, wall_s=wall, error="response larger than 80 MB")
        if "err" in box:
            return CaseResult(False, wall_s=wall, error=f"request failed: {box['err']}")
        body = box.get("body", b"")
    except requests.exceptions.Timeout:
        return CaseResult(False, wall_s=time.perf_counter() - t0, error=f"timeout (> {limit_s}s)")
    except requests.RequestException as e:
        return CaseResult(False, wall_s=time.perf_counter() - t0, error=f"request failed: {type(e).__name__}")
    if wall > limit_s:
        return CaseResult(False, wall_s=wall, error=f"too slow: {wall:.2f}s > {limit_s}s")
    if r.status_code != 200:
        return CaseResult(False, wall_s=wall, error=f"HTTP {r.status_code}")
    try:
        js = json.loads(body)
        out = C.decode_array(js["rollout"], (S_STEPS, C.NC, C.NLAT, C.NLON))
        ms = js.get("inference_ms"); ms = float(ms) if isinstance(ms, (int, float)) and not isinstance(ms, bool) else None
    except Exception as e:
        return CaseResult(False, wall_s=wall, error=f"bad response: {type(e).__name__}: {str(e)[:80]}")
    return CaseResult(True, out=out, wall_s=wall, server_ms=ms)


def health(url: str, allow_private: bool = False) -> tuple[bool, str]:
    import requests
    why = check_url(url, allow_private)
    if why:
        return False, why
    try:
        r = requests.post(url.rstrip("/") + "/health", timeout=10, allow_redirects=False)
        ok = r.status_code == 200 and r.json().get("status") == "ready"
        return ok, "" if ok else f"HTTP {r.status_code} / not ready"
    except Exception as e:
        return False, f"{type(e).__name__}"


def wait_ready(url: str, timeout_s: float = 120.0, allow_private: bool = False, poll_s: float = 2.0, abort=None) -> tuple[bool, str]:
    """Warm-up: poll /health until it answers ready or the deadline passes. Cold starts happen here, not inside a scored call.
    `abort` (optional callable) returns True when waiting is pointless, e.g. the offline container already exited."""
    t0 = time.time(); ok, why = health(url, allow_private)
    while not ok and time.time() - t0 < timeout_s and "non-public" not in why and "only absolute" not in why:
        if abort is not None and abort():
            return False, "server process exited before becoming ready"
        time.sleep(poll_s); ok, why = health(url, allow_private)
    return ok, why
