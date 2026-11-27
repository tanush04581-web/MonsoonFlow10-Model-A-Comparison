"""Training-only statistics and circular climatology computation.
Strictly computed from 2005-2018 training split.
Outputs:
- checkpoints/stats.json
- checkpoints/climatology.npy (366, 6, 32, 36)
"""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")


def _doy_index(d: dt.date) -> int:
    """Return index 0..365 for day of year, where Feb 29 has its own slot (using leap year 2000)."""
    ref = dt.date(2000, d.month, d.day)
    return (ref - dt.date(2000, 1, 1)).days


def compute_training_stats(
    train_data: np.ndarray,
    out_path: str | Path = "checkpoints/stats.json"
) -> Dict[str, Any]:
    """Compute spatial-temporal mean and std per channel strictly on training split (N, 6, 32, 36)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    means = [float(train_data[:, c].mean()) for c in range(len(CHANNELS))]
    stds = [float(train_data[:, c].std()) for c in range(len(CHANNELS))]
    
    stats = {
        "channels": list(CHANNELS),
        "mean": means,
        "std": stds,
    }
    
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"[Agent-ML] Saved training-only stats to {out_path}:")
    for c, ch in enumerate(CHANNELS):
        print(f"    {ch:6s}: mean = {means[c]:12.4f}, std = {stds[c]:12.4f}")
    return stats


def compute_climatology(
    train_data: np.ndarray,
    train_dates: np.ndarray,
    smooth_days: int = 15,
    out_path: str | Path = "checkpoints/climatology.npy"
) -> np.ndarray:
    """Compute 15-day circularly smoothed daily climatology for 366 calendar days."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    idx = np.array([
        _doy_index(dt.date(int(x) // 10000, int(x) // 100 % 100, int(x) % 100))
        for x in train_dates
    ])
    
    clim = np.zeros((366,) + train_data.shape[1:], dtype=np.float64)
    cnt = np.zeros(366, dtype=np.float64)
    
    for i, k in enumerate(idx):
        clim[k] += train_data[i]
        cnt[k] += 1
        
    clim /= np.maximum(cnt, 1.0)[:, None, None, None]
    
    # Circular moving average smoothing across calendar days
    h = smooth_days // 2
    out = np.zeros_like(clim)
    for k in range(366):
        neighbors = [clim[(k + offset) % 366] for offset in range(-h, h + 1)]
        out[k] = np.mean(neighbors, axis=0)
        
    out = out.astype(np.float32)
    np.save(out_path, out)
    print(f"[Agent-ML] Saved 366-day climatology to {out_path} with shape {out.shape}")
    return out
