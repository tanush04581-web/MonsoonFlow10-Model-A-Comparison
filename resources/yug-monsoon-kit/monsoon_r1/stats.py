"""Climatology and normalisation statistics from the training archive (shipped to teams as metadata JSON)."""
from __future__ import annotations
import datetime as dt, json
from pathlib import Path
import numpy as np

from .contract import CHANNELS


def _doy_index(d: dt.date) -> int:
    """0..365; Feb 29 gets its own slot so every calendar day has a climatology."""
    ref = dt.date(2000, d.month, d.day)                                  # 2000 is a leap year
    return (ref - dt.date(2000, 1, 1)).days


def climatology(data: np.ndarray, dates: np.ndarray, smooth_days: int = 15) -> np.ndarray:
    """[N,6,32,36] daily fields + yyyymmdd ints -> [366,6,32,36]: mean per calendar day, circular running mean."""
    idx = np.array([_doy_index(dt.date(int(x) // 10000, int(x) // 100 % 100, int(x) % 100)) for x in dates])
    clim = np.zeros((366,) + data.shape[1:], "f8"); cnt = np.zeros(366)
    for i, k in enumerate(idx):
        clim[k] += data[i]; cnt[k] += 1
    clim /= np.maximum(cnt, 1)[:, None, None, None]
    h = smooth_days // 2; out = np.zeros_like(clim)
    for k in range(366):
        out[k] = np.mean([clim[(k + o) % 366] for o in range(-h, h + 1)], axis=0)
    return out.astype("f4")


def clim_for(clim: np.ndarray, day: dt.date) -> np.ndarray:
    return clim[_doy_index(day)]


def channel_stats(data: np.ndarray) -> dict:
    return {"channels": list(CHANNELS), "mean": [float(x) for x in data.mean(axis=(0, 2, 3))],
            "std": [float(x) for x in data.std(axis=(0, 2, 3))]}


def build_artifacts(train_npz: Path, out_dir: Path) -> dict:
    z = np.load(train_npz); data, dates = z["data"], z["dates"]
    out_dir.mkdir(parents=True, exist_ok=True)
    clim = climatology(data, dates); np.save(out_dir / "climatology.npy", clim)
    st = channel_stats(data); (out_dir / "stats.json").write_text(json.dumps(st, indent=1))
    return st
