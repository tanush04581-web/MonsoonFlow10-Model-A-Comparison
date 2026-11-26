"""Independent Agent-5 QA head-to-head evaluator: Model A (local promoted) vs Model B (origin/main@80648e83).

Strict rules enforced:
- Read-only on models/, inference/, checkpoints/, data/, serve.py, dashboard/, and existing QA files.
- Never touches data/raw/splits/test.npz (2021 sealed holdout); evaluates strictly on 2019-2020 val.npz (719 windows).
- Strict weights_only=True checkpoint loading (never falls back to weights_only=False).
- Strict module isolation via separate Python subprocesses (CUDA_VISIBLE_DEVICES="", 4 CPU threads/cores)
  so local models/ and extracted Model B models/ never collide in sys.modules.
- Official kit metrics (monsoon_r1.scoring) separated clearly from QA diagnostic metrics and repo/brief points.

Run:
  .\\.venv\\Scripts\\python.exe evaluation\\qa_head_to_head.py
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "evaluation"
DEFAULT_REF_DIR = REPO_ROOT.parent / "team_model_ref_80648e"
PROBE_YMD_TUPLES = (
    (2019, 1, 15),
    (2019, 4, 20),
    (2019, 7, 15),
    (2020, 8, 10),
    (2020, 10, 15),
    (2019, 6, 17),
)
QA_SHOCK_SUBSET_COUNT = 5  # First 5 match qa_shock_tests.py WINDOW_DATES


def _ymd(y: int, m: int, d: int) -> int:
    return y * 10000 + m * 100 + d


def _to_date(x: int) -> dt.date:
    x = int(x)
    return dt.date(x // 10000, (x // 100) % 100, x % 100)


def _doy_sincos(d: dt.date) -> list[float]:
    yday = d.timetuple().tm_yday
    ang = 2.0 * math.pi * yday / 365.0
    return [float(math.sin(ang)), float(math.cos(ang))]


def _sha256_and_size(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": int(len(raw)),
        "mib": float(len(raw) / (1024 * 1024)),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _set_affinity_self(ncores: int = 4) -> bool:
    pid = os.getpid()
    mask = (1 << ncores) - 1
    if os.name == "nt":
        h = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0200 | 0x0010, False, pid)
        try:
            return bool(ctypes.windll.kernel32.SetProcessAffinityMask(h, ctypes.c_size_t(mask)))
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        os.sched_setaffinity(pid, set(range(ncores)))
        return True
    except Exception:
        return False


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _memory_info_self() -> dict[str, float]:
    pid = os.getpid()
    if os.name == "nt":
        h = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0200 | 0x0010, False, pid)
        try:
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
            return {
                "rss_mb": float(pmc.WorkingSetSize / (1024 * 1024)),
                "peak_rss_mb": float(pmc.PeakWorkingSetSize / (1024 * 1024)),
                "peak_commit_mb": float(pmc.PeakPagefileUsage / (1024 * 1024)),
            }
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        txt = Path(f"/proc/{pid}/status").read_text()
        kv = {line.split(":")[0]: line.split(":")[1].strip() for line in txt.splitlines() if ":" in line}
        return {
            "rss_mb": float(int(kv["VmRSS"].split()[0]) / 1024.0),
            "peak_rss_mb": float(int(kv["VmHWM"].split()[0]) / 1024.0),
        }
    except Exception:
        return {}


def _configure_torch_cpu(ncores: int = 4) -> tuple[int, bool]:
    import torch

    torch.set_num_threads(ncores)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    affinity_ok = _set_affinity_self(ncores)
    return int(torch.get_num_threads()), affinity_ok


# --------------------------------------------------------------------------- #
# Model A & Model B isolated adapters (for worker subprocesses)
# --------------------------------------------------------------------------- #
def _load_model_a() -> tuple[Any, dict[str, Any]]:
    import torch

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import models
    from models.backbones import load_model_from_checkpoint

    assert Path(models.__file__).resolve().is_relative_to(REPO_ROOT.resolve())
    ckpt_path = REPO_ROOT / "checkpoints" / "model.pt"
    # Verify strict weights_only=True directly as well as via loader
    _ = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model, ckpt = load_model_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    static_disk = np.load(REPO_ROOT / "checkpoints" / "static.npy")
    stats_disk = json.loads((REPO_ROOT / "checkpoints" / "stats.json").read_text(encoding="utf-8"))
    info = {
        "weights_only_ok": True,
        "strict_load_ok": True,
        "module_file": str(Path(models.__file__).resolve()),
        "architecture": str(ckpt.get("architecture") or ckpt.get("model_config", {}).get("arch")),
        "model_config": ckpt.get("model_config", {}),
        "checkpoint_keys": sorted(ckpt.keys()),
        "epoch": ckpt.get("epoch"),
        "tp_transform": ckpt.get("tp_transform", "linear"),
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "embedded_stats_mean": [float(x) for x in model.stats_mean.detach().cpu().tolist()],
        "embedded_stats_std": [float(x) for x in model.stats_std.detach().cpu().tolist()],
        "static_matches_disk": bool(np.array_equal(model.static_raw.detach().cpu().numpy(), static_disk)),
        "mean_max_diff_vs_stats_json": float(
            np.max(np.abs(model.stats_mean.detach().cpu().numpy() - np.asarray(stats_disk["mean"], dtype="f4")))
        ),
        "std_max_diff_vs_stats_json": float(
            np.max(np.abs(model.stats_std.detach().cpu().numpy() - np.asarray(stats_disk["std"], dtype="f4")))
        ),
        "seasonal_anchor_alpha": (
            [float(x) for x in model.seasonal_anchor.alpha().squeeze().detach().cpu().tolist()]
            if getattr(model, "seasonal_anchor", None) is not None
            else None
        ),
    }
    return model, info


def _rotate_doy_model_b(sin_val: float, cos_val: float, days_delta: int) -> tuple[float, float]:
    """Exact replica of 80648e83:inference/predictor.py::rotate_doy_sincos."""
    if days_delta == 0:
        return sin_val, cos_val
    rad_per_day = 2.0 * math.pi / 365.0
    angle = days_delta * rad_per_day
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    new_sin = sin_val * cos_a + cos_val * sin_a
    new_cos = cos_val * cos_a - sin_val * sin_a
    norm = math.hypot(new_sin, new_cos)
    if norm > 1e-9:
        return new_sin / norm, new_cos / norm
    return new_sin, new_cos


class _ModelBAdapter:
    """Strict replica of 80648e83:inference/{loader,predictor}.py using weights_only=True."""

    def __init__(self, ref_dir: Path) -> None:
        import types
        import torch

        # Enforce zero collision with local repo models/ or data/
        sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT.resolve()]
        if str(ref_dir.resolve()) not in sys.path:
            sys.path.insert(0, str(ref_dir.resolve()))
        for mod_name in list(sys.modules):
            if mod_name == "models" or mod_name.startswith("models.") or mod_name == "data" or mod_name.startswith("data."):
                del sys.modules[mod_name]

        # Note: 80648e83:models/__init__.py eagerly imports .physics_losses -> `from data.preprocessing import NLAT, NLON`.
        # Provide minimal in-memory constants stub so extracted models/ runs standalone without REPO_ROOT/data.
        data_pkg = types.ModuleType("data")
        data_pkg.__path__ = []  # type: ignore[attr-defined]
        data_prep = types.ModuleType("data.preprocessing")
        data_prep.NLAT = 32  # type: ignore[attr-defined]
        data_prep.NLON = 36  # type: ignore[attr-defined]
        sys.modules["data"] = data_pkg
        sys.modules["data.preprocessing"] = data_prep

        import models
        from models.mf10 import MonsoonFlow10, climatology_to_yday_table

        assert Path(models.__file__).resolve().is_relative_to(ref_dir.resolve()), (
            f"Collision: imported {models.__file__} instead of {ref_dir}"
        )

        ckpt_dir = ref_dir / "checkpoints"
        self.stats = json.loads((ckpt_dir / "stats.json").read_text(encoding="utf-8"))
        self.mean_np = np.asarray(self.stats["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
        self.std_np = np.asarray(self.stats["std"], dtype=np.float32).reshape(1, 6, 1, 1)
        self.std_safe = np.maximum(self.std_np, 1e-6)

        raw_static = np.load(ckpt_dir / "static.npy").astype(np.float32)
        self.static_norm = np.concatenate([raw_static[0:1], raw_static[1:2] / 8000.0], axis=0)  # (2, 32, 36)

        clim_disk = np.load(ckpt_dir / "climatology.npy")
        ckpt_path = ckpt_dir / "model.pt"

        # CRITICAL: strict weights_only=True ONLY, never fall back to weights_only=False
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected dict state_dict from weights_only=True, got {type(loaded)}")
        state_dict = loaded.get("state_dict", loaded)

        model = MonsoonFlow10(
            in_channels=10,
            dynamic_channels=6,
            hidden_dim=64,
            num_lstm_layers=2,
            stats=self.stats,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        expected_clim = climatology_to_yday_table(clim_disk, self.stats)
        s_coeff, k_coeff = model.transport.coefficients()
        self.info = {
            "weights_only_ok": True,
            "strict_load_ok": True,
            "module_file": str(Path(models.__file__).resolve()),
            "architecture": "MonsoonFlow10 (AnomalyTransport + 2L ConvLSTM)",
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "state_dict_keys": sorted(state_dict.keys()),
            "state_dict_key_count": int(len(state_dict)),
            "embedded_mean": [float(x) for x in model.mean.detach().cpu().reshape(-1).tolist()],
            "embedded_std": [float(x) for x in model.std.detach().cpu().reshape(-1).tolist()],
            "mean_matches_stats_json": bool(
                np.array_equal(model.mean.detach().cpu().numpy().reshape(-1), np.asarray(self.stats["mean"], dtype="f4"))
            ),
            "std_matches_stats_json": bool(
                np.array_equal(model.std.detach().cpu().numpy().reshape(-1), np.asarray(self.stats["std"], dtype="f4"))
            ),
            "clim_norm_shape": list(model.clim_norm.shape),
            "clim_norm_matches_disk_table": bool(np.array_equal(model.clim_norm.detach().cpu().numpy(), expected_clim)),
            "damping_r": [float(x) for x in model.damping().detach().cpu().reshape(-1).tolist()],
            "transport_speed": [float(x) for x in s_coeff.detach().cpu().reshape(-1).tolist()],
            "transport_kappa": [float(x) for x in k_coeff.detach().cpu().reshape(-1).tolist()],
            "repo_loader_note": "Extracted 80648e83 inference/loader.py calls weights_only=False in source code, "
            "but checkpoints/model.pt itself is a pure tensor state_dict that passes weights_only=True cleanly.",
            "transitive_data_import_note": "Extracted 80648e83 models/__init__.py eagerly imports .physics_losses -> "
            "from data.preprocessing import NLAT, NLON, coupling inference imports to data/.",
        }

    def predict_batch(
        self,
        hist_phys: np.ndarray,
        doy_last: np.ndarray,
        steps: int = 10,
        perturbation_phys: np.ndarray | None = None,
    ) -> np.ndarray:
        """Exact replica of 80648e83:inference/predictor.py physical->norm->model->physical pipeline."""
        import torch

        single = hist_phys.ndim == 4
        h = np.asarray(hist_phys, dtype=np.float32).copy()
        d = np.asarray(doy_last, dtype=np.float32)
        if single:
            h = h[None, ...]
            d = d[None, ...]
        B = h.shape[0]

        if perturbation_phys is not None:
            pert = np.asarray(perturbation_phys, dtype=np.float32)
            if pert.ndim == 3:
                pert = pert[None, ...]
            # Exact threshold & clamping order from 80648e83:inference/predictor.py
            if np.any(np.abs(pert) > 1e-12):
                h[:, -1] = h[:, -1] + pert
                h[:, -1, 1] = np.maximum(h[:, -1, 1], 0.0)
                h[:, -1, 0] = np.clip(h[:, -1, 0], 233.15, 333.15)
                h[:, -1, 5] = np.maximum(h[:, -1, 5], 0.0)

        norm_hist = (h - self.mean_np[None, ...]) / self.std_safe[None, ...]
        doy_planes = np.zeros((B, 3, 2, 32, 36), dtype=np.float32)
        for b in range(B):
            s0, c0 = float(d[b, 0]), float(d[b, 1])
            for idx, offset in enumerate((-2, -1, 0)):
                si, ci = _rotate_doy_model_b(s0, c0, offset)
                doy_planes[b, idx, 0] = si
                doy_planes[b, idx, 1] = ci

        static_tiled = np.broadcast_to(self.static_norm[None, None, ...], (B, 3, 2, 32, 36))
        model_in = np.concatenate([norm_hist, static_tiled, doy_planes], axis=2)  # (B, 3, 10, 32, 36)
        in_tensor = torch.from_numpy(np.ascontiguousarray(model_in))

        with torch.inference_mode():
            pred_norm = self.model(in_tensor, steps=steps, perturbation=None)
            pred_norm_np = pred_norm.cpu().numpy()

        out = (pred_norm_np * self.std_np[None, ...] + self.mean_np[None, ...]).astype(np.float32)
        out[:, :, 1] = np.maximum(out[:, :, 1], 0.0)
        out[:, :, 0] = np.clip(out[:, :, 0], 233.15, 333.15)
        out[:, :, 5] = np.maximum(out[:, :, 5], 0.0)
        return out[0] if single else out


def _make_shock_np(name: str, last_state: np.ndarray) -> np.ndarray:
    """Replicates monsoon_r1.scoring.make_shock without importing kit inside isolated Model B worker."""
    lat = np.arange(6.5, 37.5 + 1e-9, 1.0)
    lon = np.arange(65.5, 100.5 + 1e-9, 1.0)
    d = np.zeros((6, 32, 36), dtype=np.float32)
    if name == "warm_ocean":
        la0, la1, lo0, lo1 = (8.0, 20.0, 82.0, 94.0)
        m = ((lat >= la0) & (lat <= la1))[:, None] & ((lon >= lo0) & (lon <= lo1))[None, :]
        d[0][m] = 3.0
        d[5][m] = 0.15 * last_state[5][m]
    elif name == "monsoon_stall":
        la0, la1, lo0, lo1 = (8.0, 20.0, 65.0, 75.0)
        m = ((lat >= la0) & (lat <= la1))[:, None] & ((lon >= lo0) & (lon <= lo1))[None, :]
        d[2][m] = -0.4 * last_state[2][m]
        d[3][m] = -0.4 * last_state[3][m]
    elif name != "null":
        raise ValueError(name)
    return d


def _run_eval_worker(which: str, ref_dir: Path, out_dir: Path) -> int:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    threads, affinity_ok = _configure_torch_cpu(4)

    val_npz = REPO_ROOT / "data" / "raw" / "splits" / "val.npz"
    with np.load(val_npz) as z:
        data = np.asarray(z["data"], dtype=np.float32)
        dates = np.asarray(z["dates"], dtype=np.int32)

    ds = [_to_date(d) for d in dates]
    starts = [i for i in range(len(ds) - 12) if (ds[i + 12] - ds[i]).days == 12]
    n_win = len(starts)
    hists = np.stack([data[i : i + 3] for i in starts], axis=0)
    doys = np.asarray([_doy_sincos(ds[i + 2]) for i in starts], dtype=np.float32)

    if which == "a":
        model, info = _load_model_a()
        info["threads"] = threads
        info["affinity_ok"] = affinity_ok
        t0 = time.perf_counter()
        preds = np.empty((n_win, 10, 6, 32, 36), dtype=np.float32)
        with torch.inference_mode():
            for b0 in range(0, n_win, 32):
                bx = torch.from_numpy(hists[b0 : b0 + 32])
                bd = torch.from_numpy(doys[b0 : b0 + 32])
                preds[b0 : b0 + 32] = model(bx, bd, steps=10, perturbation=None).cpu().numpy()
        eval_rollout_s = time.perf_counter() - t0

        probe_dates = [_ymd(*t) for t in PROBE_YMD_TUPLES]
        n_probes = len(probe_dates)
        shock_base = np.empty((n_probes, 10, 6, 32, 36), dtype=np.float32)
        shock_zero = np.empty_like(shock_base)
        shock_warm = np.empty_like(shock_base)
        shock_stall = np.empty_like(shock_base)
        with torch.inference_mode():
            for idx, wd in enumerate(probe_dates):
                i_last = int(np.where(dates == wd)[0][0])
                init = data[i_last - 2 : i_last + 1].copy()
                x = torch.from_numpy(init)
                dd = torch.tensor(_doy_sincos(_to_date(wd)), dtype=torch.float32)
                shock_base[idx] = model(x, dd, steps=10, perturbation=None).cpu().numpy()
                shock_zero[idx] = model(x, dd, steps=10, perturbation=torch.zeros((6, 32, 36), dtype=torch.float32)).cpu().numpy()
                p_w = torch.from_numpy(_make_shock_np("warm_ocean", init[-1]))
                p_s = torch.from_numpy(_make_shock_np("monsoon_stall", init[-1]))
                shock_warm[idx] = model(x, dd, steps=10, perturbation=p_w).cpu().numpy()
                shock_stall[idx] = model(x, dd, steps=10, perturbation=p_s).cpu().numpy()
    else:
        try:
            adapter = _ModelBAdapter(ref_dir)
        except Exception as exc:
            err_info = {
                "weights_only_ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "threads": threads,
                "affinity_ok": affinity_ok,
            }
            (out_dir / "meta.json").write_text(json.dumps(err_info, indent=2), encoding="utf-8")
            print(f"[Model B worker] STRICT weights_only=True LOAD FAILED: {type(exc).__name__}: {exc}")
            return 2
        info = adapter.info
        info["threads"] = threads
        info["affinity_ok"] = affinity_ok

        t0 = time.perf_counter()
        preds = np.empty((n_win, 10, 6, 32, 36), dtype=np.float32)
        for b0 in range(0, n_win, 32):
            preds[b0 : b0 + 32] = adapter.predict_batch(hists[b0 : b0 + 32], doys[b0 : b0 + 32], steps=10)
        eval_rollout_s = time.perf_counter() - t0

        probe_dates = [_ymd(*t) for t in PROBE_YMD_TUPLES]
        n_probes = len(probe_dates)
        shock_base = np.empty((n_probes, 10, 6, 32, 36), dtype=np.float32)
        shock_zero = np.empty_like(shock_base)
        shock_warm = np.empty_like(shock_base)
        shock_stall = np.empty_like(shock_base)
        for idx, wd in enumerate(probe_dates):
            i_last = int(np.where(dates == wd)[0][0])
            init = data[i_last - 2 : i_last + 1].copy()
            dd = np.asarray(_doy_sincos(_to_date(wd)), dtype=np.float32)
            shock_base[idx] = adapter.predict_batch(init, dd, steps=10, perturbation_phys=None)
            shock_zero[idx] = adapter.predict_batch(init, dd, steps=10, perturbation_phys=np.zeros((6, 32, 36), dtype=np.float32))
            shock_warm[idx] = adapter.predict_batch(init, dd, steps=10, perturbation_phys=_make_shock_np("warm_ocean", init[-1]))
            shock_stall[idx] = adapter.predict_batch(init, dd, steps=10, perturbation_phys=_make_shock_np("monsoon_stall", init[-1]))

    info["val_batch_eval_s"] = float(eval_rollout_s)
    info["n_windows"] = int(n_win)
    np.save(out_dir / "preds.npy", preds)
    np.savez(
        out_dir / "shocks.npz",
        probe_dates=np.asarray(probe_dates, dtype=np.int32),
        base=shock_base,
        zero=shock_zero,
        warm=shock_warm,
        stall=shock_stall,
    )
    (out_dir / "meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return 0


def _run_bench_worker(which: str, ref_dir: Path, out_json: Path) -> int:
    import torch

    threads, affinity_ok = _configure_torch_cpu(4)
    val_npz = REPO_ROOT / "data" / "raw" / "splits" / "val.npz"
    with np.load(val_npz) as z:
        data = np.asarray(z["data"], dtype=np.float32)
        dates = np.asarray(z["dates"], dtype=np.int32)

    wd = _ymd(2019, 7, 15)
    i_last = int(np.where(dates == wd)[0][0])
    x_np = np.ascontiguousarray(data[i_last - 2 : i_last + 1].copy(), dtype=np.float32)
    doy_np = np.asarray(_doy_sincos(_to_date(wd)), dtype=np.float32)

    t_load0 = time.perf_counter()
    if which == "a":
        model, info = _load_model_a()
        ckpt_path = REPO_ROOT / "checkpoints" / "model.pt"

        def run_one():
            x_t = torch.from_numpy(x_np)
            d_t = torch.from_numpy(doy_np)
            with torch.inference_mode():
                out_t = model(x_t, d_t, steps=10, perturbation=None)
            return np.ascontiguousarray(out_t.cpu().numpy(), dtype=np.float32)

    else:
        adapter = _ModelBAdapter(ref_dir)
        info = adapter.info
        ckpt_path = ref_dir / "checkpoints" / "model.pt"

        def run_one():
            return adapter.predict_batch(x_np, doy_np, steps=10, perturbation_phys=None)

    cold_load_ms = (time.perf_counter() - t_load0) * 1000.0
    mem_after_load = _memory_info_self()

    # 5 warm-up calls
    t_w0 = time.perf_counter()
    for _ in range(5):
        out = run_one()
    warmup_5_ms = (time.perf_counter() - t_w0) * 1000.0

    # 30 timed single-request 10-step rollouts
    latencies_ms = []
    for _ in range(30):
        ts = time.perf_counter()
        out = run_one()
        latencies_ms.append((time.perf_counter() - ts) * 1000.0)

    mem_steady = _memory_info_self()
    arr = np.asarray(latencies_ms, dtype=np.float64)
    ckpt_bytes = ckpt_path.stat().st_size

    bench = {
        "model": f"model_{which}",
        "threads": threads,
        "affinity_ok": affinity_ok,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "param_count": info["param_count"],
        "checkpoint_bytes": int(ckpt_bytes),
        "checkpoint_mib": float(ckpt_bytes / (1024 * 1024)),
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "output_bytes": int(out.nbytes),
        "output_mib": float(out.nbytes / (1024 * 1024)),
        "cold_load_ms": float(cold_load_ms),
        "warmup_5_calls_ms": float(warmup_5_ms),
        "n_warmup": 5,
        "n_timed": 30,
        "latency_ms": {
            "min": float(np.min(arr)),
            "p50": float(np.percentile(arr, 50)),
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        },
        "rss_after_load": mem_after_load,
        "rss_steady": mem_steady,
        "under_5s_limit": bool(np.max(arr) <= 5000.0),
        "under_6gb_ram_limit": bool(mem_steady.get("peak_rss_mb", 0.0) <= 6144.0),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(bench, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# Scoring & QA diagnostic computation (Phase 3 & Phase 4)
# --------------------------------------------------------------------------- #
def _run_subprocess_worker(args: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    cmd = [sys.executable, str(Path(__file__).resolve())] + args
    return subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)


def _compute_metrics_and_shocks(tmp_dir: Path) -> dict[str, Any]:
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    from qa_common import (
        C,
        KS,
        S,
        CKPT_DIR,
        climatology_forecast,
        load_split,
        load_std,
        persistence,
        to_date,
        window_starts,
    )

    w_lat = C.lat_weights()[:, None]
    std = load_std()
    data, dates, _ = load_split("val")
    clim = np.load(CKPT_DIR / "climatology.npy")
    starts = np.asarray(window_starts(dates), dtype=np.int32)
    n_win = len(starts)

    truth = np.stack([data[i + C.T_IN : i + C.T_IN + C.SCORE_STEPS] for i in starts], axis=0)
    pers = np.stack([persistence(data[i : i + C.T_IN], C.SCORE_STEPS) for i in starts], axis=0)
    cl = np.stack([climatology_forecast(clim, to_date(dates[i + C.T_IN - 1]), C.SCORE_STEPS) for i in starts], axis=0)

    def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w = np.broadcast_to(w_lat, a.shape[-2:])
        am = a - (a * w).sum((-2, -1), keepdims=True) / w.sum()
        bm = b - (b * w).sum((-2, -1), keepdims=True) / w.sum()
        num = (am * bm * w).sum((-2, -1))
        den = np.sqrt((am ** 2 * w).sum((-2, -1)) * (bm ** 2 * w).sum((-2, -1)))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def _rain_contingency(pred_tp: np.ndarray, true_tp: np.ndarray, thr: float = 10.0) -> dict[str, Any]:
        p = pred_tp > thr
        t = true_tp > thr
        h = int((p & t).sum())
        m = int((~p & t).sum())
        f = int((p & ~t).sum())
        cn = int((~p & ~t).sum())
        pod = float(h / (h + m)) if (h + m) > 0 else float("nan")
        far = float(f / (h + f)) if (h + f) > 0 else float("nan")
        csi_val = float(h / (h + m + f)) if (h + m + f) > 0 else float("nan")
        fbias = float((h + f) / (h + m)) if (h + m) > 0 else float("nan")
        return {
            "hits": h,
            "misses": m,
            "false_alarms": f,
            "correct_negatives": cn,
            "csi_10mm": csi_val,
            "pod_10mm_qa": pod,
            "far_10mm_qa": far,
            "frequency_bias_10mm_qa": fbias,
        }

    def evaluate_candidate(name: str, preds: np.ndarray) -> dict[str, Any]:
        by_lead = {}
        norm_rmse_curve = []
        max_anom_curve = []

        for k in range(C.SCORE_STEPS):
            pk = preds[:, k]
            tk = truth[:, k]
            rk = pers[:, k]
            ck = cl[:, k]

            sk = S.skill_vs_best_ref(pk, tk, rk, ck)
            mae = (np.abs(pk - tk) * w_lat).mean(axis=(-2, -1)).reshape(-1, C.NC).mean(axis=0)
            bias = ((pk - tk) * w_lat).mean(axis=(-2, -1)).reshape(-1, C.NC).mean(axis=0)
            pc = _corr(pk, tk)
            acc = _corr(pk - ck, tk - ck)
            vr = np.stack([S.variance_ratio(pk[w], tk[w]) for w in range(n_win)])
            z = np.abs((pk - ck) / std[None, :, None, None])

            csi_m = S.csi(pk[:, 1], tk[:, 1], S.RAIN_THRESH)
            csi_p = S.csi(rk[:, 1], tk[:, 1], S.RAIN_THRESH)
            csi_c = S.csi(ck[:, 1], tk[:, 1], S.RAIN_THRESH)
            csi_ss = S.csi_skill(csi_m, csi_p, csi_c)
            rain_qa = _rain_contingency(pk[:, 1], tk[:, 1], S.RAIN_THRESH)

            mean_norm_rmse = float(np.mean(sk["rmse"] / std))
            max_z = float(np.nanmax(z)) if np.isfinite(z).any() else float("nan")
            norm_rmse_curve.append(mean_norm_rmse)
            max_anom_curve.append(max_z)

            by_lead[f"day_{k + 1}"] = {
                "rmse_lat": dict(zip(C.CHANNELS, [float(v) for v in sk["rmse"]])),
                "ss_vs_best_ref": dict(zip(C.CHANNELS, [float(v) for v in sk["ss"]])),
                "ss_vs_best_ref_mean": float(np.nanmean(sk["ss"])),
                "ss_vs_persistence": dict(zip(C.CHANNELS, [float(v) for v in sk["ss_persist"]])),
                "ss_vs_persistence_mean": float(np.nanmean(sk["ss_persist"])),
                "ss_vs_climatology": dict(zip(C.CHANNELS, [float(v) for v in sk["ss_clim"]])),
                "ss_vs_climatology_mean": float(np.nanmean(sk["ss_clim"])),
                "csi_10mm": float(csi_m),
                "csi_skill": float(csi_ss),
                "rain_10mm_qa": rain_qa,
                "variance_ratio_mean": dict(zip(C.CHANNELS, [float(v) for v in np.nanmean(vr, axis=0)])),
                "mae_lat_qa": dict(zip(C.CHANNELS, [float(v) for v in mae])),
                "bias_lat_qa": dict(zip(C.CHANNELS, [float(v) for v in bias])),
                "pattern_corr_qa": dict(zip(C.CHANNELS, [float(v) for v in np.nanmean(pc, axis=0)])),
                "acc_vs_clim_qa": dict(
                    zip(C.CHANNELS, [float(np.nanmean(col)) if np.isfinite(col).any() else float("nan") for col in acc.T])
                ),
                "mean_norm_rmse_std_qa": mean_norm_rmse,
                "max_abs_anomaly_std": max_z,
                "nonfinite_frac": float(1.0 - np.isfinite(pk).mean()),
            }

        # Physical validity audit across all 719 windows
        viol_counts: dict[str, int] = {}
        n_bad = 0
        for w in range(n_win):
            bad = S.validity(preds[w], C.SCORE_STEPS)
            if bad:
                n_bad += 1
                for item in bad:
                    viol_counts[item] = viol_counts.get(item, 0) + 1

        extra_bounds_qa = {
            "nonfinite_count": int((~np.isfinite(preds)).sum()),
            "tp_lt_0_count": int((preds[:, :, 1] < 0.0).sum()),
            "tp_lt_neg1e6_count": int((preds[:, :, 1] < -1e-6).sum()),
            "tcwv_lt_0_count": int((preds[:, :, 5] < 0.0).sum()),
            "t2m_lt_233_15_count": int((preds[:, :, 0] < 233.15).sum()),
            "t2m_gt_333_15_count": int((preds[:, :, 0] > 333.15).sum()),
            "t2m_min": float(np.nanmin(preds[:, :, 0])),
            "t2m_max": float(np.nanmax(preds[:, :, 0])),
            "tp_min": float(np.nanmin(preds[:, :, 1])),
            "tp_max": float(np.nanmax(preds[:, :, 1])),
            "tcwv_min": float(np.nanmin(preds[:, :, 5])),
            "tcwv_max": float(np.nanmax(preds[:, :, 5])),
        }

        # Official Day-8 anti-blur
        k8 = 7
        ratios_d8 = np.stack([S.variance_ratio(preds[w, k8], truth[w, k8]) for w in range(n_win)])
        anti_blur_pts = S.anti_blur_points(ratios_d8, max_pts=10.0)
        d8_vr_mean = dict(zip(C.CHANNELS, [float(v) for v in np.nanmean(ratios_d8, axis=0)]))

        idx_148 = [d - 1 for d in C.LEAD_DAYS]

        def _compute_modes(win_idx: list[int] | np.ndarray, collect_dist: bool = False) -> dict[str, Any]:
            win_list = [int(x) for x in win_idx]
            # Mode P1: Per-Window x Per-Lead (qa_model_eval.py style)
            ss_w_p1, csi_w_p1 = [], []
            p1_ch_rows = []
            p1_worst = None  # (ss_val, t0, lead, ch, r_m, r_p, r_c)

            # Mode P2: Per-Window Stacked Leads [3, 6, 32, 36] (rmse_lat multi-lead spec)
            ss_w_p2, csi_w_p2 = [], []
            p2_ch_rows = []

            for w in win_list:
                bad_w = S.validity(preds[w], C.SCORE_STEPS)
                if bad_w:
                    ss_w_p1.append(0.0)
                    csi_w_p1.append(0.0)
                    ss_w_p2.append(0.0)
                    csi_w_p2.append(0.0)
                    continue

                # P1
                ss_l = []
                for d in C.LEAD_DAYS:
                    k = d - 1
                    sk_w = S.skill_vs_best_ref(preds[w, k], truth[w, k], pers[w, k], cl[w, k])
                    ss_l.append(float(np.nanmean(sk_w["ss"])))
                    if collect_dist:
                        p1_ch_rows.append(sk_w["ss"])
                        rp = S.rmse_lat(pers[w, k], truth[w, k])
                        rc = S.rmse_lat(cl[w, k], truth[w, k])
                        for ci, ch in enumerate(C.CHANNELS):
                            val = float(sk_w["ss"][ci])
                            if np.isfinite(val) and (p1_worst is None or val < p1_worst["ss"]):
                                p1_worst = {
                                    "ss": val,
                                    "t0_date": int(dates[starts[w] + C.T_IN - 1]),
                                    "lead_day": int(d),
                                    "channel": ch,
                                    "model_rmse": float(sk_w["rmse"][ci]),
                                    "persist_rmse": float(rp[ci]),
                                    "clim_rmse": float(rc[ci]),
                                }
                ss_w_p1.append(float(np.nanmean(ss_l)))
                csi_win = S.csi_skill(
                    S.csi(preds[w, idx_148, 1], truth[w, idx_148, 1]),
                    S.csi(pers[w, idx_148, 1], truth[w, idx_148, 1]),
                    S.csi(cl[w, idx_148, 1], truth[w, idx_148, 1]),
                )
                csi_w_p1.append(csi_win)

                # P2
                sk_stack = S.skill_vs_best_ref(preds[w, idx_148], truth[w, idx_148], pers[w, idx_148], cl[w, idx_148])
                ss_w_p2.append(float(np.nanmean(sk_stack["ss"])))
                csi_w_p2.append(csi_win)
                if collect_dist:
                    p2_ch_rows.append(sk_stack["ss"])

            fp_p1 = S.forecast_points(ss_w_p1, csi_w_p1)
            fp_p2 = S.forecast_points(ss_w_p2, csi_w_p2)

            # Mode P3: Pooled-Per-Lead (DIAGNOSTIC - training/evaluate_val.py style)
            ss_l_p3, csi_l_p3 = [], []
            for d in C.LEAD_DAYS:
                k = d - 1
                sk_p = S.skill_vs_best_ref(preds[win_list, k], truth[win_list, k], pers[win_list, k], cl[win_list, k])
                ss_l_p3.append(float(np.nanmean(sk_p["ss"])))
                csi_l_p3.append(
                    S.csi_skill(
                        S.csi(preds[win_list, k, 1], truth[win_list, k, 1]),
                        S.csi(pers[win_list, k, 1], truth[win_list, k, 1]),
                        S.csi(cl[win_list, k, 1], truth[win_list, k, 1]),
                    )
                )
            fp_p3 = S.forecast_points(ss_l_p3, csi_l_p3)

            # Mode P4: Pooled-All [W, 3, 6, 32, 36] (DIAGNOSTIC)
            p_sub = preds[win_list][:, idx_148]
            t_sub = truth[win_list][:, idx_148]
            r_sub = pers[win_list][:, idx_148]
            c_sub = cl[win_list][:, idx_148]
            sk_all = S.skill_vs_best_ref(p_sub, t_sub, r_sub, c_sub)
            fp_p4 = S.forecast_points(
                [float(np.nanmean(sk_all["ss"]))],
                [
                    S.csi_skill(
                        S.csi(p_sub[:, :, 1], t_sub[:, :, 1]),
                        S.csi(r_sub[:, :, 1], t_sub[:, :, 1]),
                        S.csi(c_sub[:, :, 1], t_sub[:, :, 1]),
                    )
                ],
            )

            out_modes: dict[str, Any] = {
                "P1_per_window_per_lead": {**fp_p1, "n_inputs_ss": len(ss_w_p1), "official_per_window_compliant": True},
                "P2_per_window_stacked_leads": {**fp_p2, "n_inputs_ss": len(ss_w_p2), "official_per_window_compliant": True},
                "P3_pooled_per_lead_diagnostic": {**fp_p3, "n_inputs_ss": len(ss_l_p3), "official_per_window_compliant": False},
                "P4_pooled_all_diagnostic": {**fp_p4, "n_inputs_ss": 1, "official_per_window_compliant": False},
            }

            if collect_dist and p1_ch_rows and p2_ch_rows:
                arr_p1 = np.asarray(p1_ch_rows, dtype=np.float64)  # (W*3, 6)
                arr_p2 = np.asarray(p2_ch_rows, dtype=np.float64)  # (W, 6)

                def _dist_stats(arr2d: np.ndarray) -> dict[str, Any]:
                    st_d = {}
                    for ci, ch in enumerate(C.CHANNELS):
                        col = arr2d[:, ci]
                        fin = col[np.isfinite(col)]
                        st_d[ch] = {
                            "n": int(len(fin)),
                            "min": float(np.min(fin)) if len(fin) else None,
                            "p05": float(np.percentile(fin, 5)) if len(fin) else None,
                            "p25": float(np.percentile(fin, 25)) if len(fin) else None,
                            "median": float(np.median(fin)) if len(fin) else None,
                            "mean": float(np.mean(fin)) if len(fin) else None,
                            "max": float(np.max(fin)) if len(fin) else None,
                            "count_lt_neg1": int((fin < -1.0).sum()),
                            "count_lt_neg5": int((fin < -5.0).sum()),
                        }
                    return st_d

                out_modes["spike_diagnostics_qa"] = {
                    "P1_per_window_lead_ss_dist": _dist_stats(arr_p1),
                    "P2_per_window_stacked_ss_dist": _dist_stats(arr_p2),
                    "P1_worst_spike": p1_worst,
                }
            return out_modes

        all_modes = _compute_modes(range(n_win), collect_dist=True)

        # 5-window stratified subset (winter Jan, pre-monsoon Apr, onset Jun, active Jul, retreat Oct)
        strat5_t0_dates = [_ymd(*t) for t in ((2019, 1, 15), (2019, 4, 20), (2019, 6, 17), (2019, 7, 15), (2020, 10, 15))]
        t0_all = dates[starts + C.T_IN - 1]
        strat5_win_idx = [int(np.where(t0_all == td)[0][0]) for td in strat5_t0_dates]
        strat5_modes = _compute_modes(strat5_win_idx, collect_dist=False)
        strat5_ratios_d8 = np.stack([S.variance_ratio(preds[w, k8], truth[w, k8]) for w in strat5_win_idx])
        strat5_modes["day8_anti_blur_points_of_10"] = float(S.anti_blur_points(strat5_ratios_d8, max_pts=10.0))
        strat5_modes["t0_dates"] = strat5_t0_dates

        fp_per_window = {
            k: all_modes["P1_per_window_per_lead"][k] for k in ("mean_ss", "mean_csi_skill", "rmse_pts", "csi_pts", "points")
        }
        fp_pooled = {
            k: all_modes["P3_pooled_per_lead_diagnostic"][k] for k in ("mean_ss", "mean_csi_skill", "rmse_pts", "csi_pts", "points")
        }

        d4 = by_lead["day_4"]
        d4_pred_mean_rmse = float(np.mean(list(d4["rmse_lat"].values())))
        d4_pers_mean_rmse = float(np.mean(S.rmse_lat(pers[:, 3], truth[:, 3])))
        d4_pred_norm_rmse = float(d4["mean_norm_rmse_std_qa"])
        d4_pers_norm_rmse = float(np.mean(S.rmse_lat(pers[:, 3], truth[:, 3]) / std))

        return {
            "candidate": name,
            "n_windows": int(n_win),
            "by_lead": by_lead,
            "validity": {
                "all_windows_valid": n_bad == 0,
                "windows_with_violation": int(n_bad),
                "violation_counts": viol_counts,
                "bounds_audit_qa": extra_bounds_qa,
            },
            "day8_variance_ratio_mean": d8_vr_mean,
            "day8_anti_blur_points_of_10": float(anti_blur_pts),
            "beats_persistence_day4": {
                "mean_ss_vs_persistence": float(d4["ss_vs_persistence_mean"]),
                "channels_better": [c for c, v in d4["ss_vs_persistence"].items() if v > 0],
                "all_channels_better": all(v > 0 for v in d4["ss_vs_persistence"].values()),
                "beats_raw_mean_rmse": bool(d4_pred_mean_rmse < d4_pers_mean_rmse),
                "beats_norm_mean_rmse": bool(d4_pred_norm_rmse < d4_pers_norm_rmse),
                "day4_pred_mean_rmse": d4_pred_mean_rmse,
                "day4_persist_mean_rmse": d4_pers_mean_rmse,
                "day4_pred_norm_rmse": d4_pred_norm_rmse,
                "day4_persist_norm_rmse": d4_pers_norm_rmse,
            },
            "aggregation_modes_719_windows": all_modes,
            "aggregation_modes_5_window_stratified": strat5_modes,
            "forecast_points_per_window_qa_style": fp_per_window,
            "forecast_points_pooled_lead_style": fp_pooled,
            "instability": {
                "mean_norm_rmse_by_lead": norm_rmse_curve,
                "max_abs_anomaly_std_by_lead": max_anom_curve,
                "nonfinite_any": any(by_lead[f"day_{k + 1}"]["nonfinite_frac"] > 0 for k in range(C.SCORE_STEPS)),
            },
        }

    def evaluate_shocks(shock_npz_path: Path) -> dict[str, Any]:
        z = np.load(shock_npz_path)
        p_dates = [int(x) for x in z["probe_dates"]]
        bases = z["base"]
        zeros = z["zero"]
        warms = z["warm"]
        stalls = z["stall"]

        probes = []
        for idx, wd in enumerate(p_dates):
            base = bases[idx]
            zero = zeros[idx]
            warm = warms[idx]
            stall = stalls[idx]

            null_ok = S.null_consistent(base, zero, std)
            null_max_abs = float(np.abs(base - zero).max())
            null_max_norm = float((np.abs(base - zero) / std[None, :, None, None]).max())

            def _shock_detail(roll: np.ndarray, box: tuple[float, float, float, float]) -> dict[str, Any]:
                cc = S.causality_checks(base, roll, box, std)
                d = roll - base
                e = ((d / std[None, :, None, None]) ** 2).sum(axis=1)
                near = S.near_mask(box)
                out_e = np.array([e[k][~near].sum() for k in range(C.SCORE_STEPS)], dtype=np.float64)
                tot_max = float(max(e.sum(axis=(1, 2)).max(), 1e-12))
                late = float(out_e[3:8].mean())
                day1_out = float(out_e[0])
                prop_ratio = float(late / day1_out) if day1_out > 0 else (float("inf") if late > 0 else 0.0)
                z_max = float(np.abs(d / std[None, :, None, None]).max()) if np.isfinite(d).all() else float("nan")
                val_issues = S.validity(roll, C.SCORE_STEPS)
                return {
                    **cc,
                    "outside_energy_day1": day1_out,
                    "outside_energy_late_d4_d8_mean": late,
                    "propagation_ratio_late_over_day1": prop_ratio,
                    "late_energy_frac_of_peak_total": float(late / tot_max),
                    "max_abs_z_diff": z_max,
                    "validity_violations": val_issues,
                    "valid": len(val_issues) == 0,
                }

            w_det = _shock_detail(warm, C.BOX_BAY_OF_BENGAL)
            s_det = _shock_detail(stall, C.BOX_ARABIAN_SEA)
            pts = S.causality_points([w_det, s_det], null_ok, max_pts=15.0)
            probes.append(
                {
                    "date_t0": wd,
                    "base_validity_violations": S.validity(base, C.SCORE_STEPS),
                    "null_consistent": bool(null_ok),
                    "null_max_abs_diff": null_max_abs,
                    "null_max_norm_diff_std": null_max_norm,
                    "null_within_1e_6": bool(null_max_abs <= 1e-6),
                    "warm_ocean": w_det,
                    "monsoon_stall": s_det,
                    "causality_points_of_15": float(pts),
                }
            )

        def _agg_subset(sub: list[dict[str, Any]]) -> dict[str, Any]:
            def _pass(rec):
                return all(bool(rec[k]) for k in ("finite", "nonzero", "localized", "propagates", "stable", "valid"))

            return {
                "n_probes": len(sub),
                "null_all_consistent": all(p["null_consistent"] for p in sub),
                "null_all_within_1e_6": all(p["null_within_1e_6"] for p in sub),
                "null_worst_max_abs_diff": float(max(p["null_max_abs_diff"] for p in sub)),
                "warm_ocean_pass_count": int(sum(_pass(p["warm_ocean"]) for p in sub)),
                "monsoon_stall_pass_count": int(sum(_pass(p["monsoon_stall"]) for p in sub)),
                "mean_causality_points_of_15": float(np.mean([p["causality_points_of_15"] for p in sub])),
                "warm_near_frac_day1_mean": float(np.mean([p["warm_ocean"]["near_frac_day1"] for p in sub])),
                "stall_near_frac_day1_mean": float(np.mean([p["monsoon_stall"]["near_frac_day1"] for p in sub])),
                "warm_far_frac_max": float(max(p["warm_ocean"]["far_frac_max"] for p in sub)),
                "stall_far_frac_max": float(max(p["monsoon_stall"]["far_frac_max"] for p in sub)),
            }

        return {
            "probe_dates": p_dates,
            "probes": probes,
            "summary_all_6_probes": _agg_subset(probes),
            "summary_5_qa_shock_probes": _agg_subset(probes[:QA_SHOCK_SUBSET_COUNT]),
        }

    preds_a = np.load(tmp_dir / "model_a" / "preds.npy")
    preds_b = np.load(tmp_dir / "model_b" / "preds.npy")

    res_persist = evaluate_candidate("persistence", pers)
    res_clim = evaluate_candidate("climatology", cl)
    res_a = evaluate_candidate("model_a", preds_a)
    res_b = evaluate_candidate("model_b", preds_b)

    shocks_a = evaluate_shocks(tmp_dir / "model_a" / "shocks.npz")
    shocks_b = evaluate_shocks(tmp_dir / "model_b" / "shocks.npz")

    def _score70(ev: dict[str, Any], sh: dict[str, Any]) -> dict[str, Any]:
        c_pts = sh["summary_all_6_probes"]["mean_causality_points_of_15"]
        c_pts_5 = sh["summary_5_qa_shock_probes"]["mean_causality_points_of_15"]
        ab = ev["day8_anti_blur_points_of_10"]
        modes = ev["aggregation_modes_719_windows"]
        p1 = modes["P1_per_window_per_lead"]
        p2 = modes["P2_per_window_stacked_leads"]
        p3 = modes["P3_pooled_per_lead_diagnostic"]
        p4 = modes["P4_pooled_all_diagnostic"]

        def _pack(fp_dict: dict[str, Any], label: str) -> dict[str, Any]:
            return {
                "mode": label,
                "official_per_window_compliant": bool(fp_dict["official_per_window_compliant"]),
                "forecast_pts_of_45": float(fp_dict["points"]),
                "rmse_pts_of_30": float(fp_dict["rmse_pts"]),
                "csi_pts_of_15": float(fp_dict["csi_pts"]),
                "mean_ss": float(fp_dict["mean_ss"]),
                "mean_csi_skill": float(fp_dict["mean_csi_skill"]),
                "anti_blur_pts_of_10": float(ab),
                "causality_pts_of_15_all6": float(c_pts),
                "causality_pts_of_15_qa5": float(c_pts_5),
                "model_score_of_70_all6": float(fp_dict["points"] + ab + c_pts),
                "model_score_of_70_qa5": float(fp_dict["points"] + ab + c_pts_5),
            }

        s5 = ev["aggregation_modes_5_window_stratified"]
        ab5 = s5["day8_anti_blur_points_of_10"]
        return {
            "P1_per_window_per_lead": _pack(p1, "Official per-window x per-lead (qa_model_eval.py)"),
            "P2_per_window_stacked_leads": _pack(p2, "Official per-window stacked [3,6,32,36] (rmse_lat spec)"),
            "P3_pooled_per_lead_diagnostic": _pack(p3, "DIAGNOSTIC pooled-per-lead (training/evaluate_val.py; NOT official contract)"),
            "P4_pooled_all_diagnostic": _pack(p4, "DIAGNOSTIC pooled-all [W,3,6,32,36] (NOT official contract)"),
            "stratified_5_windows_70": {
                "P1": float(s5["P1_per_window_per_lead"]["points"] + ab5 + c_pts),
                "P2": float(s5["P2_per_window_stacked_leads"]["points"] + ab5 + c_pts),
                "P3_diagnostic": float(s5["P3_pooled_per_lead_diagnostic"]["points"] + ab5 + c_pts),
                "P4_diagnostic": float(s5["P4_pooled_all_diagnostic"]["points"] + ab5 + c_pts),
            },
            # Legacy keys kept for backward readability
            "per_window_qa_emulation": _pack(p1, "P1_per_window_per_lead"),
            "pooled_lead_emulation": _pack(p3, "P3_pooled_per_lead_diagnostic"),
            "note": "Official contract scoring.forecast_points(ss_by_window, csi_ss_by_window) requires per-window aggregation "
            "(P1 or P2) so failed/invalid windows score 0. P3/P4 pass pooled statistics and are DIAGNOSTIC ONLY. "
            "Repo integrity (/15) and Brief (/15) are scored separately from the /70 automated model score.",
        }

    # Explicit invalid-window=0 verification on 5-window stratified subset (Model A)
    strat5_t0 = [_ymd(*t) for t in ((2019, 1, 15), (2019, 4, 20), (2019, 6, 17), (2019, 7, 15), (2020, 10, 15))]
    t0_all = dates[starts + C.T_IN - 1]
    s5_idx = [int(np.where(t0_all == td)[0][0]) for td in strat5_t0]
    preds_corrupt = preds_a[s5_idx].copy()
    preds_corrupt[0, 3, 1, 0, 0] = -5.0  # negative rain invalidates window 0 only
    inv_ss_p1, inv_csi_p1, inv_ss_p2, inv_csi_p2 = [], [], [], []
    idx_148 = [d - 1 for d in C.LEAD_DAYS]
    for loc_i, w in enumerate(s5_idx):
        if S.validity(preds_corrupt[loc_i], C.SCORE_STEPS):
            inv_ss_p1.append(0.0); inv_csi_p1.append(0.0)
            inv_ss_p2.append(0.0); inv_csi_p2.append(0.0)
        else:
            inv_ss_p1.append(float(np.nanmean([np.nanmean(S.skill_vs_best_ref(preds_corrupt[loc_i, d - 1], truth[w, d - 1], pers[w, d - 1], cl[w, d - 1])["ss"]) for d in C.LEAD_DAYS])))
            csi_val = S.csi_skill(S.csi(preds_corrupt[loc_i, idx_148, 1], truth[w, idx_148, 1]), S.csi(pers[w, idx_148, 1], truth[w, idx_148, 1]), S.csi(cl[w, idx_148, 1], truth[w, idx_148, 1]))
            inv_csi_p1.append(csi_val)
            inv_ss_p2.append(float(np.nanmean(S.skill_vs_best_ref(preds_corrupt[loc_i, idx_148], truth[w, idx_148], pers[w, idx_148], cl[w, idx_148])["ss"])))
            inv_csi_p2.append(csi_val)

    invalid_window_guard_test = {
        "n_windows": 5,
        "corrupted_window_index": 0,
        "clean_P1_points": float(res_a["aggregation_modes_5_window_stratified"]["P1_per_window_per_lead"]["points"]),
        "corrupted_P1_points_with_zero_penalty": float(S.forecast_points(inv_ss_p1, inv_csi_p1)["points"]),
        "clean_P2_points": float(res_a["aggregation_modes_5_window_stratified"]["P2_per_window_stacked_leads"]["points"]),
        "corrupted_P2_points_with_zero_penalty": float(S.forecast_points(inv_ss_p2, inv_csi_p2)["points"]),
        "corrupted_window_scored_zero": inv_ss_p1[0] == 0.0 and inv_csi_p1[0] == 0.0 and inv_ss_p2[0] == 0.0 and inv_csi_p2[0] == 0.0,
    }

    # Jensen's inequality diagnostic on 719 windows
    jensen_diag = {}
    for d in C.LEAD_DAYS:
        k = d - 1
        rp_w = np.stack([S.rmse_lat(pers[w, k], truth[w, k]) for w in range(n_win)])  # (719, 6)
        rc_w = np.stack([S.rmse_lat(cl[w, k], truth[w, k]) for w in range(n_win)])    # (719, 6)
        e_min = np.mean(np.minimum(rp_w, rc_w), axis=0)
        min_e = np.minimum(S.rmse_lat(pers[:, k], truth[:, k]), S.rmse_lat(cl[:, k], truth[:, k]))
        jensen_diag[f"day_{d}"] = {
            "mean_per_window_min_ref_rmse": dict(zip(C.CHANNELS, [float(v) for v in e_min])),
            "pooled_min_ref_rmse": dict(zip(C.CHANNELS, [float(v) for v in min_e])),
            "ratio_per_window_over_pooled": dict(zip(C.CHANNELS, [float(a / b) for a, b in zip(e_min, min_e)])),
        }

    def _gates(ev: dict[str, Any], sh: dict[str, Any]) -> dict[str, Any]:
        s6 = sh["summary_all_6_probes"]
        d4g = ev["beats_persistence_day4"]
        return {
            "gate1_validity_all_719": bool(ev["validity"]["all_windows_valid"]),
            "gate2_beats_day4_persistence_raw_mean_rmse": bool(d4g["beats_raw_mean_rmse"]),
            "gate2_beats_day4_persistence_norm_mean_rmse": bool(d4g["beats_norm_mean_rmse"]),
            "gate2_beats_day4_persistence_6_of_6_channels": bool(d4g["all_channels_better"]),
            "gate3_nonzero_localized_shock_warm": bool(s6["warm_ocean_pass_count"] >= 1),
            "gate3_nonzero_localized_shock_stall": bool(s6["monsoon_stall_pass_count"] >= 1),
            "all_3_hard_admission_gates_pass": bool(
                ev["validity"]["all_windows_valid"]
                and d4g["beats_raw_mean_rmse"]
                and d4g["beats_norm_mean_rmse"]
                and d4g["all_channels_better"]
                and (s6["warm_ocean_pass_count"] >= 1 or s6["monsoon_stall_pass_count"] >= 1)
            ),
        }

    return {
        "baselines": {"persistence": res_persist, "climatology": res_clim},
        "models": {
            "model_a": {
                "forecast_eval": res_a,
                "shocks": shocks_a,
                "scorecard_70": _score70(res_a, shocks_a),
                "hard_admission_gates": _gates(res_a, shocks_a),
            },
            "model_b": {
                "forecast_eval": res_b,
                "shocks": shocks_b,
                "scorecard_70": _score70(res_b, shocks_b),
                "hard_admission_gates": _gates(res_b, shocks_b),
            },
        },
        "jensen_inequality_diagnostic": jensen_diag,
        "invalid_window_guard_test": invalid_window_guard_test,
        "std_used": [float(x) for x in std],
        "window_t0_range": [int(dates[starts[0] + 2]), int(dates[starts[-1] + 2])],
    }


def _git_info() -> dict[str, Any]:
    def _run(cmd: list[str]) -> str:
        r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
        return r.stdout.strip()

    return {
        "branch": _run(["git", "branch", "--show-current"]),
        "head_commit": _run(["git", "rev-parse", "HEAD"]),
        "origin_main_commit": _run(["git", "rev-parse", "origin/main"]),
        "remote": _run(["git", "remote", "-v"]).splitlines(),
        "status_short_lines": _run(["git", "status", "--short"]).splitlines(),
    }


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_json(x) for x in obj.tolist()] if obj.ndim else _sanitize_json(obj.item())
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["eval_a", "eval_b", "bench_a", "bench_b"], default=None)
    ap.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    if args.worker == "eval_a":
        return _run_eval_worker("a", args.ref_dir, args.out_dir)
    if args.worker == "eval_b":
        return _run_eval_worker("b", args.ref_dir, args.out_dir)
    if args.worker == "bench_a":
        return _run_bench_worker("a", args.ref_dir, args.out_json)
    if args.worker == "bench_b":
        return _run_bench_worker("b", args.ref_dir, args.out_json)

    # Orchestrator mode
    ref_dir = args.ref_dir.resolve()
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Extracted teammate reference directory missing: {ref_dir}")
    tmp_dir = ref_dir / "_qa_tmp_eval"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Provenance & Checksums
    local_ckpt = REPO_ROOT / "checkpoints"
    ref_ckpt = ref_dir / "checkpoints"
    checksums = {
        "model_a": {f: _sha256_and_size(local_ckpt / f) for f in ("model.pt", "stats.json", "climatology.npy", "static.npy")},
        "model_b": {f: _sha256_and_size(ref_ckpt / f) for f in ("model.pt", "stats.json", "climatology.npy", "static.npy")},
    }
    companion_identical = {
        f: checksums["model_a"][f]["sha256"] == checksums["model_b"][f]["sha256"]
        for f in ("stats.json", "climatology.npy", "static.npy")
    }

    # Phase 2–4: Isolated evaluation subprocesses
    print("[QA] Running Model A isolated evaluation worker (719 val windows + 6 shocks)...")
    p_a = _run_subprocess_worker(["--worker", "eval_a", "--ref-dir", str(ref_dir), "--out-dir", str(tmp_dir / "model_a")])
    if p_a.returncode != 0:
        raise RuntimeError(f"Model A eval worker failed ({p_a.returncode}):\n{p_a.stdout}\n{p_a.stderr}")

    print("[QA] Running Model B isolated evaluation worker (strict weights_only=True, 719 val windows + 6 shocks)...")
    p_b = _run_subprocess_worker(["--worker", "eval_b", "--ref-dir", str(ref_dir), "--out-dir", str(tmp_dir / "model_b")])
    meta_b_path = tmp_dir / "model_b" / "meta.json"
    meta_b = json.loads(meta_b_path.read_text(encoding="utf-8")) if meta_b_path.exists() else {}
    if p_b.returncode != 0 or not meta_b.get("weights_only_ok", False):
        # Strictly stop Model B execution without weights_only=False fallback
        report_err = {
            "status": "MODEL_B_WEIGHTS_ONLY_BLOCKED",
            "git": _git_info(),
            "checksums": checksums,
            "companion_identical": companion_identical,
            "model_b_meta": meta_b,
            "stderr": p_b.stderr,
        }
        (EVAL_DIR / "head_to_head_report.json").write_text(json.dumps(report_err, indent=2), encoding="utf-8")
        print("[QA] BLOCKED: Model B failed strict weights_only=True load! See head_to_head_report.json.")
        return 2

    meta_a = json.loads((tmp_dir / "model_a" / "meta.json").read_text(encoding="utf-8"))

    # Phase 5: Isolated 4-core CPU benchmarks
    print("[QA] Running Model A 4-core CPU benchmark (5 warmup + 30 timed calls)...")
    pb_a = _run_subprocess_worker(["--worker", "bench_a", "--ref-dir", str(ref_dir), "--out-json", str(tmp_dir / "bench_a.json")])
    if pb_a.returncode != 0:
        raise RuntimeError(f"Model A bench worker failed:\n{pb_a.stdout}\n{pb_a.stderr}")
    bench_a = json.loads((tmp_dir / "bench_a.json").read_text(encoding="utf-8"))

    print("[QA] Running Model B 4-core CPU benchmark (5 warmup + 30 timed calls)...")
    pb_b = _run_subprocess_worker(["--worker", "bench_b", "--ref-dir", str(ref_dir), "--out-json", str(tmp_dir / "bench_b.json")])
    if pb_b.returncode != 0:
        raise RuntimeError(f"Model B bench worker failed:\n{pb_b.stdout}\n{pb_b.stderr}")
    bench_b = json.loads((tmp_dir / "bench_b.json").read_text(encoding="utf-8"))

    print("[QA] Computing official monsoon_r1.scoring + QA diagnostics + causality comparisons...")
    scored = _compute_metrics_and_shocks(tmp_dir)

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split_evaluated": "val.npz (2019-01-01..2020-12-31, 719 windows)",
        "sealed_test_split_accessed": False,
        "git": _git_info(),
        "artifacts": {
            "checksums": checksums,
            "companion_files_bit_identical": companion_identical,
            "model_a_metadata": meta_a,
            "model_b_metadata": meta_b,
        },
        "baselines": scored["baselines"],
        "models": {
            "model_a": {**scored["models"]["model_a"], "cpu_benchmark": bench_a},
            "model_b": {**scored["models"]["model_b"], "cpu_benchmark": bench_b},
        },
        "jensen_inequality_diagnostic": scored["jensen_inequality_diagnostic"],
        "invalid_window_guard_test": scored["invalid_window_guard_test"],
        "std_used": scored["std_used"],
        "window_t0_range": scored["window_t0_range"],
    }

    out_path = EVAL_DIR / "head_to_head_report.json"
    out_path.write_text(json.dumps(_sanitize_json(report), indent=2), encoding="utf-8")
    if not args.keep_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # Print crisp console summary
    for key, label in (("model_a", "Model A (local exp4)"), ("model_b", "Model B (80648e83)")):
        sc = report["models"][key]["scorecard_70"]
        cpu = report["models"][key]["cpu_benchmark"]
        sh6 = report["models"][key]["shocks"]["summary_all_6_probes"]
        print(
            f"[{label}] "
            f"P1={sc['P1_per_window_per_lead']['model_score_of_70_all6']:.2f}/70 "
            f"P2={sc['P2_per_window_stacked_leads']['model_score_of_70_all6']:.2f}/70 "
            f"P3(diag)={sc['P3_pooled_per_lead_diagnostic']['model_score_of_70_all6']:.2f}/70 "
            f"P4(diag)={sc['P4_pooled_all_diagnostic']['model_score_of_70_all6']:.2f}/70 | "
            f"shocks={sh6['warm_ocean_pass_count']}/6 warm, {sh6['monsoon_stall_pass_count']}/6 stall | "
            f"CPU p50={cpu['latency_ms']['p50']:.1f}ms p95={cpu['latency_ms']['p95']:.1f}ms peakRSS={cpu['rss_steady']['peak_rss_mb']:.1f}MB"
        )
    print(f"[QA] Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
