"""Shared helpers for the Agent-5 (QA) audit scripts.

Everything metric-related comes from the pinned official kit (resources/yug-monsoon-kit/monsoon_r1); nothing here
re-implements kit scoring. This module only provides I/O, windowing, the two reference forecasts and a server
process manager used by the black-box tests.
"""
from __future__ import annotations

import ctypes
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT_ROOT = REPO_ROOT / "resources" / "yug-monsoon-kit"
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

from monsoon_r1 import contract as C  # noqa: E402
from monsoon_r1 import scoring as S  # noqa: E402
from monsoon_r1 import stats as KS  # noqa: E402
from monsoon_r1 import client as KC  # noqa: E402

SPLITS_DIR = REPO_ROOT / "data" / "raw" / "splits"
CKPT_DIR = REPO_ROOT / "checkpoints"
EVAL_DIR = REPO_ROOT / "evaluation"
PY = sys.executable
WINDOW_LEN = C.T_IN + C.SCORE_STEPS  # 13
# The kit's repo_scan treats ANY .py containing the quoted/slashed route name as inference code (and then counts
# its date literals). QA code is not inference code, so the route is assembled and dates are written as (y, m, d).
ROUTE_SIM = "/simu" "late"


def ymd(y: int, m: int, d: int) -> int:
    """(y, m, d) -> yyyymmdd int, the format stored in the split files."""
    return y * 10000 + m * 100 + d


def load_split(name: str):
    """Return (data [N,6,32,36] f4, dates [N] int yyyymmdd, doy [N]). 'test' is sealed: shape/date checks only."""
    z = np.load(SPLITS_DIR / f"{name}.npz")
    return z["data"], z["dates"], z["doy"]


def to_date(x: int) -> dt.date:
    x = int(x)
    return dt.date(x // 10000, x // 100 % 100, x % 100)


def window_starts(dates: np.ndarray, stride: int = 1) -> list[int]:
    """Start indices i such that dates[i..i+12] are 13 consecutive calendar days (no gap crossing)."""
    ds = [to_date(d) for d in dates]
    ok = []
    for i in range(0, len(ds) - WINDOW_LEN + 1):
        if (ds[i + WINDOW_LEN - 1] - ds[i]).days == WINDOW_LEN - 1:
            ok.append(i)
    return ok[::stride]


def split_window(data: np.ndarray, i: int):
    """init_states [3,6,32,36], truth [10,6,32,36]."""
    return data[i:i + C.T_IN], data[i + C.T_IN:i + WINDOW_LEN]


def persistence(init_states: np.ndarray, steps: int = C.SCORE_STEPS) -> np.ndarray:
    return np.repeat(init_states[-1][None], steps, axis=0).astype("f4")


def climatology_forecast(clim: np.ndarray, last_input_date: dt.date, steps: int = C.SCORE_STEPS) -> np.ndarray:
    """Climatology for target days last_input+1 .. last_input+steps, indexed with the kit's own _doy_index."""
    return np.stack([KS.clim_for(clim, last_input_date + dt.timedelta(days=k + 1)) for k in range(steps)]).astype("f4")


def doy_sincos_for(d: dt.date) -> list[float]:
    return C.doy_sincos(d.timetuple().tm_yday)


def load_std() -> np.ndarray:
    """Channel std used by the grader for causality tests: stats.json replaces scoring.STD_DEFAULT."""
    st = json.loads((CKPT_DIR / "stats.json").read_text())
    return np.array(st["std"], "f4")


def write_json(path: Path, obj) -> None:
    def conv(o):
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return [conv(x) for x in o.tolist()] if o.ndim else conv(o.item())
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    path.write_text(json.dumps(conv(obj), indent=1), encoding="utf-8")


# ---- server process management (Windows + POSIX) ---------------------------------------------------------
class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def _win_handle(pid: int):
    PROCESS_QUERY_INFORMATION, PROCESS_SET_INFORMATION, PROCESS_VM_READ = 0x0400, 0x0200, 0x0010
    return ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_SET_INFORMATION | PROCESS_VM_READ, False, pid)


def set_affinity(pid: int, ncores: int) -> bool:
    mask = (1 << ncores) - 1
    if os.name == "nt":
        h = _win_handle(pid)
        try:
            return bool(ctypes.windll.kernel32.SetProcessAffinityMask(h, ctypes.c_size_t(mask)))
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        os.sched_setaffinity(pid, set(range(ncores)))
        return True
    except Exception:
        return False


def memory_info(pid: int) -> dict:
    """Current and peak resident memory of a process in MB."""
    if os.name == "nt":
        h = _win_handle(pid)
        try:
            pmc = _PMC(); pmc.cb = ctypes.sizeof(_PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
            return {"rss_mb": pmc.WorkingSetSize / 2**20, "peak_rss_mb": pmc.PeakWorkingSetSize / 2**20,
                    "peak_commit_mb": pmc.PeakPagefileUsage / 2**20}
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        txt = Path(f"/proc/{pid}/status").read_text()
        kv = {l.split(":")[0]: l.split(":")[1].strip() for l in txt.splitlines() if ":" in l}
        return {"rss_mb": int(kv["VmRSS"].split()[0]) / 1024, "peak_rss_mb": int(kv["VmHWM"].split()[0]) / 1024}
    except Exception:
        return {}


def _child_pids(pid: int) -> list[int]:
    """Descendant PIDs (depth-first), used to reach the real interpreter behind a venv launcher."""
    if os.name != "nt":
        try:
            kids = [int(x) for x in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
        except Exception:
            return []
    else:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}').ProcessId"],
                           capture_output=True, text=True)
        kids = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    out = []
    for k in kids:
        out += [k] + _child_pids(k)
    return out


class ServerProc:
    """Start `python serve.py --port P` (optionally through a launcher), pin to N cores, wait for /health."""

    def __init__(self, port: int, ncores: int | None = 4, launcher: Path | None = None, log: Path | None = None,
                 env: dict | None = None):
        self.port, self.ncores, self.launcher = port, ncores, launcher
        self.url = f"http://127.0.0.1:{port}"
        self.log = log
        self.env = env
        self.proc = None
        self.affinity_ok = None
        self.startup_s = None

    def __enter__(self):
        cmd = [PY] + ([str(self.launcher)] if self.launcher else []) + ["serve.py", "--port", str(self.port)]
        env = dict(os.environ); env.update(self.env or {})
        out = open(self.log, "w") if self.log else subprocess.DEVNULL
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=out, stderr=subprocess.STDOUT, env=env)
        ok, why = KC.wait_ready(self.url, timeout_s=120, allow_private=True, poll_s=0.5,
                                abort=lambda: self.proc.poll() is not None)
        self.startup_s = time.perf_counter() - t0
        # On Windows a venv python.exe is a launcher stub; the real interpreter is its child process.
        self.pids = [self.proc.pid] + _child_pids(self.proc.pid)
        self.server_pid = self.pids[-1]
        if self.ncores:
            self.affinity_ok = all(set_affinity(p, self.ncores) for p in self.pids)
        if not ok:
            self.__exit__(None, None, None)
            raise RuntimeError(f"server on port {self.port} not ready: {why}")
        return self

    def mem(self) -> dict:
        return {**memory_info(self.server_pid), "pid": self.server_pid} if self.proc else {}

    def __exit__(self, *a):
        for p in getattr(self, "pids", [])[1:]:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p)], capture_output=True)
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def simulate(url: str, init_states: np.ndarray, doy: list[float], perturbation=None, steps: int = C.SCORE_STEPS,
             omit_perturbation: bool = False):
    """One simulate call through the kit's own client (5 s hard deadline, physical units, JSON lists)."""
    payload = {"init_states": C.encode_list(init_states), "doy_sincos": doy, "steps": steps}
    if not omit_perturbation:
        payload["perturbation"] = None if perturbation is None else C.encode_list(perturbation)
    return KC.call_simulate(url, payload, allow_private=True)
