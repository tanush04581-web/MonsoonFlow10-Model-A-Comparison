"""Chronological dataset splitting and preprocessing for MonsoonFlow-10.
Enforces strict chronological partitioning:
- Train: 2005-01-01 to 2018-12-31
- Val:   2019-01-01 to 2020-12-31
- Test:  2021-01-01 to 2021-12-31 (Sealed holdout)
Zero random shuffling to prevent temporal auto-correlation data leakage.
"""
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import xarray as xr

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
NLAT, NLON = 32, 36


def load_raw_dataset(nc_path: str | Path = "data/india_climate_train.nc") -> Tuple[np.ndarray, np.ndarray]:
    """Load daily fields and dates from india_climate_train.nc.
    
    Returns:
        data: np.ndarray of shape (N, 6, 32, 36), float32, physical units
        dates: np.ndarray of int dates YYYYMMDD, shape (N,)
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(f"Climate dataset not found at {nc_path}")

    ds = xr.open_dataset(nc_path)
    
    # Extract time array and convert to YYYYMMDD integers
    time_coords = ds["time"].values
    dates_list = []
    for t in time_coords:
        ts = np.datetime64(t, 'D')
        d = dt.date.fromisoformat(str(ts))
        dates_list.append(d.year * 10000 + d.month * 100 + d.day)
    dates = np.array(dates_list, dtype=np.int32)
    
    # Check if packaged as single 4D 'data' variable or individual channel variables
    if "data" in ds.variables:
        data = ds["data"].values.astype(np.float32)
        # Verify channel coordinate ordering if present
        if "channel" in ds.coords:
            nc_channels = tuple(str(c) for c in ds["channel"].values)
            if nc_channels != CHANNELS:
                # Re-order channels to match canonical order
                indices = [nc_channels.index(ch) for ch in CHANNELS]
                data = data[:, indices]
    else:
        channel_arrays = []
        for ch in CHANNELS:
            if ch not in ds.variables:
                raise KeyError(f"Missing required channel '{ch}' in {nc_path}")
            arr = ds[ch].values.astype(np.float32)
            channel_arrays.append(arr)
        data = np.stack(channel_arrays, axis=1) # (N, 6, 32, 36)
        
    if data.shape[1:] != (6, NLAT, NLON):
        raise ValueError(f"Expected data shape (N, 6, 32, 36), got {data.shape}")
        
    # Enforce physical validity checks
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Raw climate dataset contains NaNs or Infs")
    
    # Ensure non-negative precipitation
    tp_idx = CHANNELS.index("tp")
    data[:, tp_idx] = np.maximum(data[:, tp_idx], 0.0)
    
    # Ensure non-negative tcwv
    tcwv_idx = CHANNELS.index("tcwv")
    data[:, tcwv_idx] = np.maximum(data[:, tcwv_idx], 0.0)

    print(f"[Agent-ML] Loaded raw dataset: shape={data.shape} from date {dates[0]} to {dates[-1]}")
    return data, dates


def chronological_split(
    data: np.ndarray,
    dates: np.ndarray
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Strict chronological splitting into train, val, and test partitions."""
    train_mask = (dates >= 20050101) & (dates <= 20181231)
    val_mask = (dates >= 20190101) & (dates <= 20201231)
    test_mask = (dates >= 20210101) & (dates <= 20211231)

    splits = {
        "train": (data[train_mask], dates[train_mask]),
        "val": (data[val_mask], dates[val_mask]),
        "test": (data[test_mask], dates[test_mask]),
    }

    print(f"[Agent-ML] Chronological split complete:")
    print(f"  Train (2005-2018): {splits['train'][0].shape[0]} days ({splits['train'][1][0]} to {splits['train'][1][-1]})")
    print(f"  Val   (2019-2020): {splits['val'][0].shape[0]} days ({splits['val'][1][0]} to {splits['val'][1][-1]})")
    print(f"  Test  (2021):      {splits['test'][0].shape[0]} days ({splits['test'][1][0]} to {splits['test'][1][-1]})")

    return splits
