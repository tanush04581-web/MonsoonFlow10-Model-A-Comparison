"""Static environmental field extractor for MonsoonFlow-10.
Extracts 2 static channels: land_sea_mask (0=ocean, 1=land) and orography_m (meters).
Output shape strictly (2, 32, 36) saved to checkpoints/static.npy.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

CHANNELS_STATIC = ("land_sea_mask", "orography_m")


def extract_static_fields(nc_path: str | Path = "data/india_climate_train.nc", out_path: str | Path = "checkpoints/static.npy") -> np.ndarray:
    """Extract the (2, 32, 36) static terrain tensor from the NetCDF file."""
    nc_path = Path(nc_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    
    if "static" in ds.variables:
        static_arr = ds["static"].values.astype(np.float32)
        if "static_channel" in ds.coords:
            sc_names = tuple(str(c) for c in ds["static_channel"].values)
            if sc_names != CHANNELS_STATIC:
                indices = [sc_names.index(ch) for ch in CHANNELS_STATIC]
                static_arr = static_arr[indices]
    elif "land_sea_mask" in ds.variables and "orography_m" in ds.variables:
        lsm = ds["land_sea_mask"].values.astype(np.float32)
        oro = ds["orography_m"].values.astype(np.float32)
        static_arr = np.stack([lsm, oro], axis=0)
    else:
        raise KeyError(f"Neither 'static' nor ('land_sea_mask', 'orography_m') found in {list(ds.variables.keys())}")

    if static_arr.shape != (2, 32, 36):
        raise ValueError(f"Expected static shape (2, 32, 36), got {static_arr.shape}")

    if np.any(np.isnan(static_arr)) or np.any(np.isinf(static_arr)):
        raise ValueError("Static terrain tensor contains NaNs or Infs")

    np.save(out_path, static_arr)
    print(f"[Agent-ML] Successfully saved static terrain tensor to {out_path} with shape {static_arr.shape}")
    print(f"  Channel 0 (land_sea_mask): min={static_arr[0].min():.4f}, max={static_arr[0].max():.4f}")
    print(f"  Channel 1 (orography_m):   min={static_arr[1].min():.1f}m, max={static_arr[1].max():.1f}m")
    return static_arr
