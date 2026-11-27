"""Chronological dataset splitter and training-only artifact builder for MonsoonFlow-10.

Validates the raw NetCDF climate archive against the Monsoon-R1 physical and grid
contract, partitions daily fields strictly chronologically without shuffling or
cross-boundary windowing, and writes the 7 split artifacts into ``--out-dir``:

  1. ``train.npz``             (2005-01-01..2018-12-31; keys: data, dates, doy)
  2. ``val.npz``               (2019-01-01..2020-12-31; keys: data, dates, doy)
  3. ``test.npz``              (2021-01-01..2021-12-31; keys: data, dates, doy)
  4. ``static.npy``            (float32, shape (2, 32, 36); fractional LSM + raw orography)
  5. ``train_stats.json``      (channels/mean/std computed strictly on 2005-2018 train split)
  6. ``train_climatology.npy`` (float32, shape (366, 6, 32, 36) computed strictly on train split)
  7. ``splits.json``           (metadata, provenance, shapes, dtypes, DOY & windowing notes)

Important conventions:
- Rolling 10-day forecast windows (``window_len = t_in + t_rollout = 13``) are NEVER
  built across split boundaries; they are constructed independently per split
  downstream by ``data.dataset.MonsoonDataset`` (expected counts: 5101 train,
  719 val, 353 test for full 2005-2021 data).
- ``doy`` saved in ``*.npz`` uses ordinary calendar day-of-year ``1..365/366`` via
  ``d.timetuple().tm_yday`` (matching ``data/dataset.py`` and
  ``monsoon_r1.contract.doy_sincos``).
- ``train_climatology.npy`` axis 0 uses ``monsoon_r1.stats._doy_index``
  (``0..365`` leap-year-2000 calendar indexing where Feb 29 has slot 59 and
  Mar 1 always has slot 60).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Sequence

import numpy as np
import xarray as xr

try:
    from monsoon_r1 import contract as C
    from monsoon_r1 import stats as S
except ImportError:  # pragma: no cover - fallback when monsoon_r1 is not editable-installed
    _KIT_DIR = Path(__file__).resolve().parent.parent / "resources" / "yug-monsoon-kit"
    if _KIT_DIR.exists() and str(_KIT_DIR) not in sys.path:
        sys.path.insert(0, str(_KIT_DIR))
    from monsoon_r1 import contract as C
    from monsoon_r1 import stats as S

STATIC_CHANNELS: tuple[str, str] = ("land_sea_mask", "orography_m")
TRAIN_START_DATE = dt.date(2005, 1, 1)
TRAIN_END_DATE = dt.date(2018, 12, 31)
VAL_START_DATE = dt.date(2019, 1, 1)
VAL_END_DATE = dt.date(2020, 12, 31)
TEST_START_DATE = dt.date(2021, 1, 1)
TEST_END_DATE = dt.date(2021, 12, 31)
ALLOWED_YEARS = frozenset(range(2005, 2022))

EXPECTED_SPLIT_COUNTS: dict[str, int] = {
    "train": (TRAIN_END_DATE - TRAIN_START_DATE).days + 1,  # 5113
    "val": (VAL_END_DATE - VAL_START_DATE).days + 1,        # 731
    "test": (TEST_END_DATE - TEST_START_DATE).days + 1,     # 365
}

T2M_MIN_K = 233.15
T2M_MAX_K = 333.15
SUPPORTED_CALENDARS = frozenset({"standard", "gregorian", "proleptic_gregorian"})
OUTPUT_FILENAMES: tuple[str, ...] = (
    "train.npz",
    "val.npz",
    "test.npz",
    "static.npy",
    "train_stats.json",
    "train_climatology.npy",
    "splits.json",
)


@dataclass(frozen=True)
class SplitPartition:
    """In-memory daily arrays and calendar metadata for a single chronological split."""

    name: str
    role: str
    configured_start: dt.date
    configured_end: dt.date
    data: np.ndarray
    dates_int: np.ndarray
    doy_int: np.ndarray
    date_objs: list[dt.date]


def _decode_coord_str(val: object) -> str:
    """Decode byte or numpy string coordinate labels without altering ordering or casing."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8")
    return str(val)


def _validate_coord_1d(ds: xr.Dataset, name: str, expected_len: int) -> np.ndarray:
    """Validate that a required 1D coordinate exists with matching dimension name and length."""
    if name not in ds.coords:
        raise ValueError(f"Missing required coordinate '{name}' in dataset.")
    coord = ds.coords[name]
    if coord.dims != (name,):
        raise ValueError(f"Coordinate '{name}' must have dims ('{name}',), got {coord.dims}.")
    vals = np.asarray(coord.values)
    if vals.ndim != 1 or len(vals) != expected_len:
        raise ValueError(
            f"Coordinate '{name}' must have shape ({expected_len},), got {vals.shape}."
        )
    return vals


def _validate_dataset_structure(ds: xr.Dataset) -> None:
    """Strictly validate variables, dimension orders, channel orders, and lat/lon grids."""
    if "data" not in ds.data_vars:
        raise ValueError("Missing required variable 'data' in NetCDF dataset.")
    if "static" not in ds.data_vars:
        raise ValueError("Missing required variable 'static' in NetCDF dataset.")

    data_var = ds["data"]
    expected_data_dims = ("time", "channel", "lat", "lon")
    if data_var.dims != expected_data_dims:
        raise ValueError(
            f"Variable 'data' must have exact dims {expected_data_dims}, got {data_var.dims}."
        )
    if data_var.ndim != 4 or data_var.shape[1:] != (C.NC, C.NLAT, C.NLON) or data_var.shape[0] <= 0:
        raise ValueError(
            f"Variable 'data' must have shape (N>0, {C.NC}, {C.NLAT}, {C.NLON}), got {data_var.shape}."
        )

    static_var = ds["static"]
    expected_static_dims = ("static_channel", "lat", "lon")
    if static_var.dims != expected_static_dims:
        raise ValueError(
            f"Variable 'static' must have exact dims {expected_static_dims}, got {static_var.dims}."
        )
    expected_static_shape = (len(STATIC_CHANNELS), C.NLAT, C.NLON)
    if static_var.shape != expected_static_shape:
        raise ValueError(
            f"Variable 'static' must have shape {expected_static_shape}, got {static_var.shape}."
        )

    if "time" not in ds.coords or ds.coords["time"].dims != ("time",):
        raise ValueError("Missing or malformed 'time' coordinate with dims ('time',).")

    raw_channels = _validate_coord_1d(ds, "channel", C.NC)
    channels = tuple(_decode_coord_str(v) for v in raw_channels)
    if channels != tuple(C.CHANNELS):
        raise ValueError(
            f"Channel coordinate mismatch: expected {tuple(C.CHANNELS)}, got {channels}."
        )

    raw_static_channels = _validate_coord_1d(ds, "static_channel", len(STATIC_CHANNELS))
    static_channels = tuple(_decode_coord_str(v) for v in raw_static_channels)
    if static_channels != STATIC_CHANNELS:
        raise ValueError(
            f"Static channel coordinate mismatch: expected {STATIC_CHANNELS}, got {static_channels}."
        )

    lat_vals = _validate_coord_1d(ds, "lat", C.NLAT).astype(np.float64)
    if not np.isfinite(lat_vals).all() or not np.all(np.diff(lat_vals) > 0):
        raise ValueError("Coordinate 'lat' must be finite and strictly ascending.")
    if not np.allclose(lat_vals, C.LAT, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Coordinate 'lat' mismatch: expected {C.LAT[0]}..{C.LAT[-1]} (1 deg), "
            f"got {lat_vals[0]}..{lat_vals[-1]}."
        )

    lon_vals = _validate_coord_1d(ds, "lon", C.NLON).astype(np.float64)
    if not np.isfinite(lon_vals).all() or not np.all(np.diff(lon_vals) > 0):
        raise ValueError("Coordinate 'lon' must be finite and strictly ascending.")
    if not np.allclose(lon_vals, C.LON, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Coordinate 'lon' mismatch: expected {C.LON[0]}..{C.LON[-1]} (1 deg), "
            f"got {lon_vals[0]}..{lon_vals[-1]}."
        )


def _parse_time_coords(time_values: np.ndarray) -> tuple[list[dt.date], np.ndarray, np.ndarray]:
    """Convert NetCDF time coordinates into Python dates, YYYYMMDD int32, and tm_yday int32."""
    if time_values.ndim != 1 or len(time_values) == 0:
        raise ValueError(f"'time' coordinate must be non-empty 1D, got shape {time_values.shape}.")

    date_objs: list[dt.date] = []
    if np.issubdtype(time_values.dtype, np.datetime64):
        if np.isnat(time_values).any():
            raise ValueError("Time coordinate contains NaT values.")
        day_values = time_values.astype("datetime64[D]")
        for item in day_values:
            date_objs.append(dt.date.fromisoformat(str(item)))
    else:
        for item in time_values:
            if item is None:
                raise ValueError("Time coordinate contains null/None entries.")
            cal = getattr(item, "calendar", None)
            if cal is not None and str(cal).lower() not in SUPPORTED_CALENDARS:
                raise ValueError(f"Unsupported calendar '{cal}'; expected Gregorian/standard.")
            if hasattr(item, "year") and hasattr(item, "month") and hasattr(item, "day"):
                date_objs.append(dt.date(int(item.year), int(item.month), int(item.day)))
            else:
                raise ValueError(f"Unsupported time coordinate entry type: {type(item).__name__}.")

    dates_int = np.array(
        [d.year * 10000 + d.month * 100 + d.day for d in date_objs],
        dtype=np.int32,
    )
    # Ordinary calendar day-of-year (1..365/366) matching data/dataset.py and contract.doy_sincos.
    doy_int = np.array([d.timetuple().tm_yday for d in date_objs], dtype=np.int32)
    return date_objs, dates_int, doy_int


def _validate_time_sequence(date_objs: list[dt.date]) -> None:
    """Reject out-of-range years, duplicate dates, non-monotonic order, and missing days."""
    if not date_objs:
        raise ValueError("Dataset contains zero timestamps.")

    seen: set[dt.date] = set()
    for idx, current in enumerate(date_objs):
        if current.year not in ALLOWED_YEARS:
            raise ValueError(
                f"Unexpected year {current.year} at time index {idx} ({current.isoformat()}); "
                f"allowed years are 2005..2021."
            )
        if current in seen:
            raise ValueError(f"Duplicate date detected: {current.isoformat()} (index {idx}).")
        seen.add(current)

    for idx in range(len(date_objs) - 1):
        prev_d = date_objs[idx]
        next_d = date_objs[idx + 1]
        if next_d < prev_d:
            raise ValueError(
                f"Non-monotonic time axis at index {idx}: "
                f"{prev_d.isoformat()} followed by {next_d.isoformat()}."
            )

    for idx in range(len(date_objs) - 1):
        prev_d = date_objs[idx]
        next_d = date_objs[idx + 1]
        delta_days = (next_d - prev_d).days
        if delta_days != 1:
            raise ValueError(
                f"Missing dates / calendar gap between {prev_d.isoformat()} and "
                f"{next_d.isoformat()} (step={delta_days} days; expected 1)."
            )


def _validate_and_cast_arrays(
    raw_data: np.ndarray,
    raw_static: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate finiteness and physical bounds without mutating raw dynamic or static values."""
    if not np.issubdtype(raw_data.dtype, np.floating):
        raise ValueError(f"'data' must have floating-point dtype, got {raw_data.dtype}.")
    if not np.issubdtype(raw_static.dtype, np.floating):
        raise ValueError(f"'static' must have floating-point dtype, got {raw_static.dtype}.")

    if not np.isfinite(raw_data).all():
        bad_cnt = int(np.size(raw_data) - np.isfinite(raw_data).sum())
        raise ValueError(f"Dynamic 'data' contains {bad_cnt} NaN or Inf values.")
    if not np.isfinite(raw_static).all():
        bad_cnt = int(np.size(raw_static) - np.isfinite(raw_static).sum())
        raise ValueError(f"Static terrain field contains {bad_cnt} NaN or Inf values.")

    # Use raw dtype bounds for exact boundary representability before float32 cast.
    raw_t2m = raw_data[:, 0]
    t2m_lo = raw_data.dtype.type(T2M_MIN_K)
    t2m_hi = raw_data.dtype.type(T2M_MAX_K)
    if bool((raw_t2m < t2m_lo).any() or (raw_t2m > t2m_hi).any()):
        n_bad = int(((raw_t2m < t2m_lo) | (raw_t2m > t2m_hi)).sum())
        raise ValueError(
            f"Physical constraint violated for 't2m': {n_bad} cells outside "
            f"[{T2M_MIN_K}, {T2M_MAX_K}] K."
        )

    raw_tp = raw_data[:, 1]
    if bool((raw_tp < 0.0).any()):
        n_bad = int((raw_tp < 0.0).sum())
        raise ValueError(f"Physical constraint violated for 'tp': {n_bad} cells < 0.0 mm/day.")

    raw_tcwv = raw_data[:, 5]
    if bool((raw_tcwv < 0.0).any()):
        n_bad = int((raw_tcwv < 0.0).sum())
        raise ValueError(f"Physical constraint violated for 'tcwv': {n_bad} cells < 0.0 kg/m2.")

    raw_lsm = raw_static[0]
    if bool((raw_lsm < 0.0).any() or (raw_lsm > 1.0).any()):
        n_bad = int(((raw_lsm < 0.0) | (raw_lsm > 1.0)).sum())
        raise ValueError(
            f"Physical constraint violated for 'land_sea_mask': {n_bad} cells outside [0.0, 1.0]."
        )

    # Preserve exact fractional land_sea_mask, raw orography_m, and dynamic values as float32.
    data = np.asarray(raw_data, dtype=np.float32)
    static = np.asarray(raw_static, dtype=np.float32)

    if not np.isfinite(data).all():
        raise ValueError("Dynamic 'data' overflowed to non-finite float32 values.")
    if not np.isfinite(static).all():
        raise ValueError("Static terrain array overflowed to non-finite float32 values.")

    return data, static


def _assert_contiguous_slice(mask: np.ndarray, name: str) -> slice:
    """Verify a boolean mask selects a non-empty contiguous index range and return its slice."""
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"Chronological partition '{name}' is empty.")
    start = int(indices[0])
    stop = int(indices[-1]) + 1
    if stop - start != len(indices):
        raise ValueError(f"Chronological partition '{name}' is non-contiguous.")
    return slice(start, stop)


def _partition_chronological(
    data: np.ndarray,
    dates_int: np.ndarray,
    doy_int: np.ndarray,
    date_objs: list[dt.date],
    strict_full_counts: bool = False,
) -> dict[str, SplitPartition]:
    """Partition data strictly chronologically into train (2005-2018), val (2019-2020), test (2021)."""
    specs = (
        ("train", "training", TRAIN_START_DATE, TRAIN_END_DATE),
        ("val", "validation", VAL_START_DATE, VAL_END_DATE),
        ("test", "sealed_holdout", TEST_START_DATE, TEST_END_DATE),
    )

    masks: dict[str, np.ndarray] = {}
    for name, _, start_d, end_d in specs:
        masks[name] = np.array([(start_d <= d <= end_d) for d in date_objs], dtype=bool)

    assignment_counts = sum(m.astype(np.int32) for m in masks.values())
    if not np.all(assignment_counts == 1):
        raise ValueError(
            "Chronological split masks are not disjoint/complete over input timestamps."
        )

    slices = {name: _assert_contiguous_slice(masks[name], name) for name, _, _, _ in specs}

    if slices["train"].start != 0:
        raise ValueError("Train partition must begin at index 0.")
    if slices["train"].stop != slices["val"].start:
        raise ValueError("Train and val partitions must meet with zero gap or overlap.")
    if slices["val"].stop != slices["test"].start:
        raise ValueError("Val and test partitions must meet with zero gap or overlap.")
    if slices["test"].stop != len(date_objs):
        raise ValueError("Test partition must end at final timestamp.")

    partitions: dict[str, SplitPartition] = {}
    for name, role, start_d, end_d in specs:
        sl = slices[name]
        partitions[name] = SplitPartition(
            name=name,
            role=role,
            configured_start=start_d,
            configured_end=end_d,
            data=data[sl],
            dates_int=dates_int[sl],
            doy_int=doy_int[sl],
            date_objs=date_objs[sl],
        )

    train_p = partitions["train"]
    val_p = partitions["val"]
    test_p = partitions["test"]

    if (val_p.date_objs[0] - train_p.date_objs[-1]).days != 1:
        raise ValueError("Cross-boundary gap or overlap between train and val partitions.")
    if (test_p.date_objs[0] - val_p.date_objs[-1]).days != 1:
        raise ValueError("Cross-boundary gap or overlap between val and test partitions.")
    if not (train_p.date_objs[-1] < val_p.date_objs[0] and val_p.date_objs[-1] < test_p.date_objs[0]):
        raise ValueError("Partitions violate strict chronological ordering.")

    reconstructed_dates = np.concatenate([train_p.dates_int, val_p.dates_int, test_p.dates_int])
    if not np.array_equal(reconstructed_dates, dates_int):
        raise ValueError("Partitioned date arrays do not exactly reconstruct input timeline.")

    if strict_full_counts:
        for name, expected_days in EXPECTED_SPLIT_COUNTS.items():
            actual_days = len(partitions[name].date_objs)
            if actual_days != expected_days:
                raise ValueError(
                    f"Split '{name}' day count {actual_days} != expected full-span {expected_days}."
                )

    return partitions


def _compute_train_artifacts(
    train_data: np.ndarray,
    train_dates: np.ndarray,
    smooth_days: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute channel stats and smoothed 366-day climatology strictly from training data."""
    if smooth_days < 1:
        raise ValueError(f"--smooth-days must be >= 1, got {smooth_days}.")

    stats = S.channel_stats(train_data)
    if set(stats.keys()) != {"channels", "mean", "std"}:
        raise ValueError(f"Unexpected stats schema keys: {sorted(stats.keys())}.")
    if tuple(stats["channels"]) != tuple(C.CHANNELS):
        raise ValueError(f"Stats channels mismatch: {stats['channels']}.")

    means = np.asarray(stats["mean"], dtype=np.float64)
    stds = np.asarray(stats["std"], dtype=np.float64)
    if not (np.isfinite(means).all() and np.isfinite(stds).all()):
        raise ValueError("Training statistics contain non-finite mean or std values.")
    if not np.all(stds > 0.0):
        raise ValueError(f"Training channel standard deviations must all be > 0, got {stats['std']}.")

    clim = S.climatology(train_data, train_dates, smooth_days=smooth_days)
    expected_clim_shape = (366, C.NC, C.NLAT, C.NLON)
    if clim.shape != expected_clim_shape or clim.dtype != np.float32:
        raise ValueError(
            f"Unexpected climatology output shape/dtype: {clim.shape} {clim.dtype}, "
            f"expected {expected_clim_shape} float32."
        )
    if not np.isfinite(clim).all():
        raise ValueError("Training climatology contains NaN or Inf values.")

    return stats, clim


def _build_splits_metadata(
    nc_path: Path,
    partitions: dict[str, SplitPartition],
    smooth_days: int,
    t_in: int,
    t_rollout: int,
) -> dict[str, Any]:
    """Assemble splits.json metadata without exposing val/test data statistics."""
    window_len = t_in + t_rollout
    split_entries: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        part = partitions[name]
        actual_days = len(part.date_objs)
        expected_days = EXPECTED_SPLIT_COUNTS[name]
        downstream_windows = max(0, actual_days - window_len + 1)
        expected_windows = max(0, expected_days - window_len + 1)
        split_entries[name] = {
            "file": f"{name}.npz",
            "role": part.role,
            "configured_range": [
                part.configured_start.isoformat(),
                part.configured_end.isoformat(),
            ],
            "actual_range": [
                part.date_objs[0].isoformat(),
                part.date_objs[-1].isoformat(),
            ],
            "actual_dates_int": [
                int(part.dates_int[0]),
                int(part.dates_int[-1]),
            ],
            "expected_days": expected_days,
            "actual_days": actual_days,
            "matches_expected_days": actual_days == expected_days,
            "expected_windows_10day": expected_windows,
            "downstream_windows": downstream_windows,
            "shapes": {
                "data": [int(d) for d in part.data.shape],
                "dates": [int(part.dates_int.shape[0])],
                "doy": [int(part.doy_int.shape[0])],
            },
            "dtypes": {
                "data": str(part.data.dtype),
                "dates": str(part.dates_int.dtype),
                "doy": str(part.doy_int.dtype),
            },
        }

    train_part = partitions["train"]
    return {
        "source_nc": nc_path.as_posix(),
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contract": {
            "channels": list(C.CHANNELS),
            "static_channels": list(STATIC_CHANNELS),
            "grid_shape": [C.NLAT, C.NLON],
            "lat_bounds": [float(C.LAT[0]), float(C.LAT[-1])],
            "lon_bounds": [float(C.LON[0]), float(C.LON[-1])],
            "lat_count": C.NLAT,
            "lon_count": C.NLON,
            "t_in": t_in,
            "t_rollout": t_rollout,
            "window_len": window_len,
            "step_stride": 1,
        },
        "windowing_note": (
            "Daily arrays (data, dates, doy) are saved without cross-boundary sliding windows. "
            f"10-day rollout windows (window_len = t_in + t_rollout = {window_len}) are constructed "
            "independently per split downstream by data.dataset.MonsoonDataset, yielding expected "
            "window counts 5101 (train), 719 (val), 353 (test) on the full 2005-2021 archive."
        ),
        "stats_provenance": {
            "train_only": "2005-2018",
            "computed_on_split": "train",
            "train_actual_range": [
                train_part.date_objs[0].isoformat(),
                train_part.date_objs[-1].isoformat(),
            ],
            "val_test_leakage": False,
            "stats_file": "train_stats.json",
            "stats_function": "monsoon_r1.stats.channel_stats",
            "climatology_file": "train_climatology.npy",
            "climatology_function": "monsoon_r1.stats.climatology",
            "climatology_shape": [366, C.NC, C.NLAT, C.NLON],
            "climatology_dtype": "float32",
            "climatology_smooth_days": smooth_days,
            "climatology_indexing_convention": (
                "monsoon_r1.stats._doy_index: 0..365 leap-year-2000 calendar indexing "
                "((date(2000, month, day) - date(2000, 1, 1)).days), where Feb 29 occupies "
                "slot 59 and Mar 1 always occupies slot 60. Distinct from the ordinary calendar "
                "d.timetuple().tm_yday (1..365/366) stored in *.npz['doy'] for "
                "data.dataset.MonsoonDataset and monsoon_r1.contract.doy_sincos."
            ),
        },
        "doy_convention": {
            "npz_doy_key": (
                "int32 ordinary calendar day-of-year via d.timetuple().tm_yday (1..365 or 1..366), "
                "matching data/dataset.py and monsoon_r1.contract.doy_sincos."
            ),
            "climatology_axis0": (
                "monsoon_r1.stats._doy_index (0..365 reference leap year 2000; Feb 29 -> 59, Mar 1 -> 60)."
            ),
        },
        "static": {
            "file": "static.npy",
            "shape": [len(STATIC_CHANNELS), C.NLAT, C.NLON],
            "dtype": "float32",
            "channels": list(STATIC_CHANNELS),
            "fractional_lsm_preserved": True,
            "raw_orography_preserved": True,
        },
        "splits": split_entries,
    }


def _atomic_write_file(target_path: Path, writer: Callable[[BinaryIO], None]) -> None:
    """Write a file atomically via a same-directory temporary file and os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _check_overwrite_safety(out_dir: Path, allow_overwrite: bool) -> None:
    """Abort before writing if any target artifact already exists unless --allow-overwrite is set."""
    if allow_overwrite or not out_dir.exists():
        return
    existing = [name for name in OUTPUT_FILENAMES if (out_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing split artifacts in '{out_dir}': "
            f"{', '.join(existing)}. Pass --allow-overwrite to replace them."
        )


def run_split(
    nc_path: Path,
    out_dir: Path,
    smooth_days: int = 15,
    t_in: int = C.T_IN,
    t_rollout: int = C.SCORE_STEPS,
    allow_overwrite: bool = False,
    strict_full_counts: bool = False,
) -> dict[str, Any]:
    """Validate NetCDF input, create chronological splits, and save all 7 artifacts."""
    if t_in < 1:
        raise ValueError(f"--t-in must be >= 1, got {t_in}.")
    if t_rollout < 1:
        raise ValueError(f"--t-rollout must be >= 1, got {t_rollout}.")
    if smooth_days < 1:
        raise ValueError(f"--smooth-days must be >= 1, got {smooth_days}.")

    nc_path = Path(nc_path)
    out_dir = Path(out_dir)
    if not nc_path.is_file():
        raise FileNotFoundError(f"Input NetCDF dataset not found: {nc_path}")

    _check_overwrite_safety(out_dir, allow_overwrite=allow_overwrite)

    with xr.open_dataset(nc_path) as ds:
        _validate_dataset_structure(ds)
        date_objs, dates_int, doy_int = _parse_time_coords(np.asarray(ds["time"].values))
        _validate_time_sequence(date_objs)
        raw_data = np.asarray(ds["data"].values)
        raw_static = np.asarray(ds["static"].values)

    data, static = _validate_and_cast_arrays(raw_data, raw_static)
    partitions = _partition_chronological(
        data=data,
        dates_int=dates_int,
        doy_int=doy_int,
        date_objs=date_objs,
        strict_full_counts=strict_full_counts,
    )

    # Compute normalization stats and climatology STRICTLY on 2005-2018 training data only.
    train_part = partitions["train"]
    train_stats, train_clim = _compute_train_artifacts(
        train_data=train_part.data,
        train_dates=train_part.dates_int,
        smooth_days=smooth_days,
    )

    metadata = _build_splits_metadata(
        nc_path=nc_path,
        partitions=partitions,
        smooth_days=smooth_days,
        t_in=t_in,
        t_rollout=t_rollout,
    )

    _check_overwrite_safety(out_dir, allow_overwrite=allow_overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        part = partitions[split_name]
        _atomic_write_file(
            out_dir / f"{split_name}.npz",
            lambda f, p=part: np.savez(f, data=p.data, dates=p.dates_int, doy=p.doy_int),
        )

    _atomic_write_file(
        out_dir / "static.npy",
        lambda f: np.save(f, static),
    )
    _atomic_write_file(
        out_dir / "train_stats.json",
        lambda f: f.write((json.dumps(train_stats, indent=2) + "\n").encode("utf-8")),
    )
    _atomic_write_file(
        out_dir / "train_climatology.npy",
        lambda f: np.save(f, train_clim),
    )
    _atomic_write_file(
        out_dir / "splits.json",
        lambda f: f.write((json.dumps(metadata, indent=2) + "\n").encode("utf-8")),
    )

    return metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for dataset splitting."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate india_climate_train.nc against the Monsoon-R1 contract and "
            "write chronological train/val/test splits plus train-only stats and climatology."
        )
    )
    parser.add_argument(
        "--nc-path",
        type=Path,
        default=Path("data/raw/india_climate_train.nc"),
        help="Path to input NetCDF dataset (default: data/raw/india_climate_train.nc).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/splits"),
        help="Directory for split artifacts (default: data/raw/splits).",
    )
    parser.add_argument(
        "--smooth-days",
        type=int,
        default=15,
        help="Circular smoothing window in days for train_climatology.npy (default: 15).",
    )
    parser.add_argument(
        "--t-in",
        type=int,
        default=C.T_IN,
        help=f"Input context days for downstream window metadata (default: {C.T_IN}).",
    )
    parser.add_argument(
        "--t-rollout",
        type=int,
        default=C.SCORE_STEPS,
        help=f"Rollout forecast days for downstream window metadata (default: {C.SCORE_STEPS}).",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting existing split artifacts in --out-dir.",
    )
    parser.add_argument(
        "--strict-full-counts",
        action="store_true",
        help="Require exact full-archive day counts (5113 train, 731 val, 365 test).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args(argv)
    try:
        metadata = run_split(
            nc_path=args.nc_path,
            out_dir=args.out_dir,
            smooth_days=args.smooth_days,
            t_in=args.t_in,
            t_rollout=args.t_rollout,
            allow_overwrite=args.allow_overwrite,
            strict_full_counts=args.strict_full_counts,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"[split_data] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[split_data] Wrote 7 split artifacts to {args.out_dir}")
    for split_name in ("train", "val", "test"):
        info = metadata["splits"][split_name]
        print(
            f"  {split_name:5s}: {info['actual_days']:4d} days "
            f"({info['actual_range'][0]} .. {info['actual_range'][1]}), "
            f"downstream {metadata['contract']['t_rollout']}-day windows = {info['downstream_windows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
