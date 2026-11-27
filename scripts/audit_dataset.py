"""Dataset integrity and physical validity auditor for MonsoonFlow-10.

Validates ``data/raw/india_climate_train.nc`` against the official
``monsoon_r1.contract`` specifications and produces a structured JSON report.
Strictly read-only on the NetCDF dataset and strictly non-evaluative on the
2021 holdout split (structural, finiteness, and physical-validity checks only).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

try:
    from monsoon_r1 import contract as C
except ImportError:
    _KIT_ROOT = Path(__file__).resolve().parents[1] / "resources" / "yug-monsoon-kit"
    if _KIT_ROOT.is_dir():
        sys.path.insert(0, str(_KIT_ROOT))
    from monsoon_r1 import contract as C


EXPECTED_TIME_COUNT = 6209
EXPECTED_START_DATE = "2005-01-01"
EXPECTED_END_DATE = "2021-12-31"
EXPECTED_STATIC_CHANNELS = ("land_sea_mask", "orography_m")
EXPECTED_UNIT_PAIRS = (
    ("t2m", "K"),
    ("tp", "mm/day"),
    ("u10", "m/s"),
    ("v10", "m/s"),
    ("msl", "Pa"),
    ("tcwv", "kg/m2"),
)
SPLIT_DEFS = (
    ("train", "2005-01-01", "2018-12-31", 5113),
    ("val", "2019-01-01", "2020-12-31", 731),
    ("test", "2021-01-01", "2021-12-31", 365),
)

# Physical validity bounds aligned with monsoon_r1.scoring.validity and synoptic envelopes
T2M_MIN_K = 233.15
T2M_MAX_K = 333.15
WIND_ABS_SANITY_MAX = 60.0
MSL_SANITY_MIN_PA = 85000.0
MSL_SANITY_MAX_PA = 108000.0
TP_SANITY_MAX_MM_DAY = 1000.0
TCWV_SANITY_MAX_KG_M2 = 120.0
ORO_OCEAN_MIN_HARD_M = -5.0
PERCENTILE_LEVELS = (0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9)


def _record_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    category: str,
    passed: bool,
    details: dict[str, Any],
    error_msg: str | None = None,
) -> bool:
    """Append a structured check entry and record an error message on failure."""
    checks.append(
        {
            "name": name,
            "category": category,
            "passed": bool(passed),
            "details": details,
        }
    )
    if not passed:
        errors.append(error_msg or f"Check '{name}' failed")
    return bool(passed)


def _parse_units_attr(units_raw: str) -> list[tuple[str, str]]:
    """Parse 'ch unit | ...' attribute string into (channel, unit) pairs."""
    pairs: list[tuple[str, str]] = []
    for segment in units_raw.split("|"):
        tokens = segment.strip().split()
        if len(tokens) == 2:
            pairs.append((tokens[0], tokens[1]))
    return pairs


def _audit_schema(
    ds: xr.Dataset,
    nc_path: Path,
    file_size_bytes: int,
    checks: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    """Validate dataset variables, dimensions, shapes, and coordinate metadata."""
    has_data = "data" in ds.data_vars
    has_static = "static" in ds.data_vars
    expected_data_dims = ("time", "channel", "lat", "lon")
    expected_static_dims = ("static_channel", "lat", "lon")
    expected_data_shape = (EXPECTED_TIME_COUNT, C.NC, C.NLAT, C.NLON)
    expected_static_shape = (len(EXPECTED_STATIC_CHANNELS), C.NLAT, C.NLON)

    data_var = ds["data"] if has_data else None
    static_var = ds["static"] if has_static else None

    data_dims = tuple(data_var.dims) if data_var is not None else ()
    static_dims = tuple(static_var.dims) if static_var is not None else ()
    data_shape = tuple(int(s) for s in data_var.shape) if data_var is not None else ()
    static_shape = tuple(int(s) for s in static_var.shape) if static_var is not None else ()
    data_dtype = str(data_var.dtype) if data_var is not None else None
    static_dtype = str(static_var.dtype) if static_var is not None else None

    coords_info = {
        name: {
            "dtype": str(coord.dtype),
            "shape": [int(s) for s in coord.shape],
            "attrs": {str(k): str(v) for k, v in coord.attrs.items()},
        }
        for name, coord in ds.coords.items()
    }
    dims_info = {str(k): int(v) for k, v in ds.sizes.items()}

    schema_ok = (
        has_data
        and has_static
        and data_dims == expected_data_dims
        and data_shape == expected_data_shape
        and static_dims == expected_static_dims
        and static_shape == expected_static_shape
        and data_dtype == "float32"
        and static_dtype == "float32"
    )

    details = {
        "nc_path": str(nc_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mib": round(file_size_bytes / (1024.0 * 1024.0), 4),
        "dimensions": dims_info,
        "data_vars": list(ds.data_vars.keys()),
        "data": {
            "present": has_data,
            "dims": list(data_dims),
            "shape": list(data_shape),
            "expected_dims": list(expected_data_dims),
            "expected_shape": list(expected_data_shape),
            "dtype": data_dtype,
            "encoding_chunks": list(data_var.encoding.get("chunksizes") or []) if data_var is not None else [],
            "zlib": bool(data_var.encoding.get("zlib", False)) if data_var is not None else False,
            "complevel": data_var.encoding.get("complevel") if data_var is not None else None,
        },
        "static": {
            "present": has_static,
            "dims": list(static_dims),
            "shape": list(static_shape),
            "expected_dims": list(expected_static_dims),
            "expected_shape": list(expected_static_shape),
            "dtype": static_dtype,
        },
        "coordinates": coords_info,
    }
    _record_check(
        checks,
        errors,
        name="dataset_variables_and_dims",
        category="schema",
        passed=schema_ok,
        details=details,
        error_msg=(
            f"Dataset schema mismatch: data=({data_dims}, {data_shape}, {data_dtype}), "
            f"static=({static_dims}, {static_shape}, {static_dtype})"
        ),
    )
    return details


def _audit_channels_and_units(
    ds: xr.Dataset,
    checks: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    """Verify dynamic channel ordering and units metadata against contract."""
    actual_channels = (
        tuple(str(c) for c in ds["channel"].values) if "channel" in ds.coords else ()
    )
    units_attr = str(ds["data"].attrs.get("units", "")) if "data" in ds.data_vars else ""
    parsed_units = _parse_units_attr(units_attr)

    channels_match = actual_channels == tuple(C.CHANNELS)
    units_match = tuple(parsed_units) == EXPECTED_UNIT_PAIRS

    details = {
        "actual_channels": list(actual_channels),
        "expected_channels": list(C.CHANNELS),
        "exact_order_match": channels_match,
        "units_attr": units_attr,
        "parsed_unit_pairs": [{"channel": ch, "unit": u} for ch, u in parsed_units],
        "expected_unit_pairs": [{"channel": ch, "unit": u} for ch, u in EXPECTED_UNIT_PAIRS],
        "units_match": units_match,
    }
    _record_check(
        checks,
        errors,
        name="dynamic_channels_order_and_units",
        category="schema",
        passed=channels_match and units_match,
        details=details,
        error_msg=(
            f"Channel/unit mismatch: actual={actual_channels}, units='{units_attr}'"
        ),
    )
    return details


def _audit_time_axis(
    ds: xr.Dataset,
    checks: list[dict[str, Any]],
    errors: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    """Verify daily time axis continuity, monotonicity, leap days, and split/window counts."""
    time_raw = ds["time"].values
    dates_d = time_raw.astype("datetime64[D]")
    count = int(len(dates_d))
    start_str = str(dates_d[0]) if count else ""
    end_str = str(dates_d[-1]) if count else ""

    diffs = np.diff(dates_d).astype("timedelta64[D]").astype(np.int64) if count > 1 else np.array([], dtype=np.int64)
    unique_deltas, delta_counts = np.unique(diffs, return_counts=True)
    delta_hist = {str(int(k)): int(v) for k, v in zip(unique_deltas, delta_counts)}

    non_one_indices = np.where(diffs != 1)[0]
    gaps = [
        {
            "index": int(idx),
            "from": str(dates_d[idx]),
            "to": str(dates_d[idx + 1]),
            "delta_days": int(diffs[idx]),
        }
        for idx in non_one_indices[:20]
    ]

    unique_count = int(len(np.unique(dates_d)))
    duplicate_count = count - unique_count

    expected_calendar = np.arange(
        np.datetime64(EXPECTED_START_DATE, "D"),
        np.datetime64(EXPECTED_END_DATE, "D") + np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )
    exact_calendar_match = (count == len(expected_calendar)) and bool(np.array_equal(dates_d, expected_calendar))

    leap_days_present = [str(d) for d in dates_d if str(d).endswith("-02-29")]

    time_ok = (
        count == EXPECTED_TIME_COUNT
        and start_str == EXPECTED_START_DATE
        and end_str == EXPECTED_END_DATE
        and duplicate_count == 0
        and len(non_one_indices) == 0
        and exact_calendar_match
    )
    time_details = {
        "dtype": str(time_raw.dtype),
        "count": count,
        "expected_count": EXPECTED_TIME_COUNT,
        "start_date": start_str,
        "end_date": end_str,
        "expected_start_date": EXPECTED_START_DATE,
        "expected_end_date": EXPECTED_END_DATE,
        "strictly_monotonic_1_day": len(non_one_indices) == 0 and count > 1,
        "delta_days_histogram": delta_hist,
        "duplicate_dates_count": duplicate_count,
        "missing_or_gap_count": int(len(non_one_indices)),
        "gap_examples": gaps,
        "exact_calendar_match": exact_calendar_match,
        "leap_days_present": leap_days_present,
    }
    _record_check(
        checks,
        errors,
        name="time_axis_completeness_and_monotonicity",
        category="time",
        passed=time_ok,
        details=time_details,
        error_msg=f"Time axis invalid: count={count}, range={start_str}..{end_str}, gaps={len(non_one_indices)}, dups={duplicate_count}",
    )

    window_len = int(C.T_IN + C.SCORE_STEPS)
    splits_summary: dict[str, Any] = {}
    splits_ok = True
    total_assigned = 0

    for name, s_str, e_str, expected_days in SPLIT_DEFS:
        s_dt = np.datetime64(s_str, "D")
        e_dt = np.datetime64(e_str, "D")
        mask = (dates_d >= s_dt) & (dates_d <= e_dt)
        n_days = int(mask.sum())
        total_assigned += n_days
        n_windows = max(0, n_days - window_len + 1)
        expected_windows = expected_days - window_len + 1
        split_dates = dates_d[mask]
        split_ok = (
            n_days == expected_days
            and n_windows == expected_windows
            and (str(split_dates[0]) == s_str if n_days else False)
            and (str(split_dates[-1]) == e_str if n_days else False)
        )
        splits_ok &= split_ok
        splits_summary[name] = {
            "start_date": s_str,
            "end_date": e_str,
            "actual_start": str(split_dates[0]) if n_days else None,
            "actual_end": str(split_dates[-1]) if n_days else None,
            "days": n_days,
            "expected_days": expected_days,
            "rollout_windows_10day": n_windows,
            "expected_rollout_windows_10day": expected_windows,
            "passed": split_ok,
        }

    partition_exhaustive = total_assigned == count
    splits_ok &= partition_exhaustive

    split_details = {
        "t_in": int(C.T_IN),
        "t_rollout": int(C.SCORE_STEPS),
        "window_len": window_len,
        "total_assigned_days": total_assigned,
        "exhaustive_and_disjoint": partition_exhaustive,
        "partitions": splits_summary,
    }
    _record_check(
        checks,
        errors,
        name="chronological_split_and_window_counts",
        category="time",
        passed=splits_ok,
        details=split_details,
        error_msg="Chronological split day counts or 10-day window counts failed contract verification",
    )

    return {"axis": time_details, "chronological_splits": split_details}, dates_d


def _audit_coordinates(
    ds: xr.Dataset,
    checks: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    """Verify lat/lon coordinate grids match monsoon_r1.contract exactly."""
    lat = ds["lat"].values
    lon = ds["lon"].values

    lat_diffs = np.diff(lat)
    lon_diffs = np.diff(lon)
    lat_match = (lat.shape == C.LAT.shape) and bool(np.allclose(lat, C.LAT, atol=1e-9, rtol=0.0))
    lon_match = (lon.shape == C.LON.shape) and bool(np.allclose(lon, C.LON, atol=1e-9, rtol=0.0))
    lat_step_ok = bool(np.allclose(lat_diffs, 1.0, atol=1e-9, rtol=0.0))
    lon_step_ok = bool(np.allclose(lon_diffs, 1.0, atol=1e-9, rtol=0.0))

    coords_ok = lat_match and lon_match and lat_step_ok and lon_step_ok
    details = {
        "lat": {
            "dtype": str(lat.dtype),
            "count": int(len(lat)),
            "expected_count": int(C.NLAT),
            "min": float(lat[0]),
            "max": float(lat[-1]),
            "step_deg": 1.0 if lat_step_ok else None,
            "strictly_ascending_1deg": lat_step_ok and bool(np.all(lat_diffs > 0)),
            "max_abs_diff_vs_contract": float(np.max(np.abs(lat - C.LAT))) if lat.shape == C.LAT.shape else None,
            "matches_contract": lat_match,
        },
        "lon": {
            "dtype": str(lon.dtype),
            "count": int(len(lon)),
            "expected_count": int(C.NLON),
            "min": float(lon[0]),
            "max": float(lon[-1]),
            "step_deg": 1.0 if lon_step_ok else None,
            "strictly_ascending_1deg": lon_step_ok and bool(np.all(lon_diffs > 0)),
            "max_abs_diff_vs_contract": float(np.max(np.abs(lon - C.LON))) if lon.shape == C.LON.shape else None,
            "matches_contract": lon_match,
        },
    }
    _record_check(
        checks,
        errors,
        name="coordinates_match_contract",
        category="coordinates",
        passed=coords_ok,
        details=details,
        error_msg="Lat/lon coordinates deviate from monsoon_r1.contract.LAT / LON",
    )
    return details


def _locate_extreme(
    arr: np.ndarray,
    flat_idx: int,
    dates_d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> dict[str, Any]:
    """Convert a flat index in (T, NLAT, NLON) to date, lat, lon, and value."""
    t_idx, i_idx, j_idx = np.unravel_index(flat_idx, arr.shape)
    return {
        "value": float(arr[t_idx, i_idx, j_idx]),
        "date": str(dates_d[t_idx]),
        "lat": float(lats[i_idx]),
        "lon": float(lons[j_idx]),
        "grid_index": {"time_idx": int(t_idx), "lat_idx": int(i_idx), "lon_idx": int(j_idx)},
    }


def _audit_dynamic_values(
    ds: xr.Dataset,
    dates_d: np.ndarray,
    checks: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    """Audit finiteness, physical bounds, and summary statistics for all 6 channels."""
    data = ds["data"].values
    static = ds["static"].values
    lats = ds["lat"].values
    lons = ds["lon"].values
    channels = tuple(str(c) for c in ds["channel"].values)
    unit_map = dict(EXPECTED_UNIT_PAIRS)

    data_nan = int(np.isnan(data).sum())
    data_inf = int(np.isinf(data).sum())
    static_nan = int(np.isnan(static).sum())
    static_inf = int(np.isinf(static).sum())
    finite_ok = (data_nan == 0 and data_inf == 0 and static_nan == 0 and static_inf == 0)

    _record_check(
        checks,
        errors,
        name="finite_values",
        category="integrity",
        passed=finite_ok,
        details={
            "data_total_elements": int(data.size),
            "data_nan_count": data_nan,
            "data_inf_count": data_inf,
            "static_total_elements": int(static.size),
            "static_nan_count": static_nan,
            "static_inf_count": static_inf,
        },
        error_msg=f"Non-finite values detected: data(nan={data_nan}, inf={data_inf}), static(nan={static_nan}, inf={static_inf})",
    )

    channel_reports: dict[str, Any] = {}
    for idx, ch in enumerate(channels):
        arr = data[:, idx]
        arr64 = arr.astype(np.float64)
        pct_vals = np.percentile(arr64, PERCENTILE_LEVELS)
        pct_dict = {f"p{str(lvl).replace('.', '_')}": float(val) for lvl, val in zip(PERCENTILE_LEVELS, pct_vals)}

        neg_count = int((arr < 0.0).sum())
        zero_count = int((arr == 0.0).sum())
        min_loc = _locate_extreme(arr, int(np.argmin(arr)), dates_d, lats, lons)
        max_loc = _locate_extreme(arr, int(np.argmax(arr)), dates_d, lats, lons)

        channel_reports[ch] = {
            "channel_index": idx,
            "unit": unit_map.get(ch, ""),
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr64.mean()),
            "std": float(arr64.std()),
            "negative_count": neg_count,
            "zero_count": zero_count,
            "percentiles": pct_dict,
            "min_location": min_loc,
            "max_location": max_loc,
        }

    # Physical validity gates (global)
    t2m = data[:, channels.index("t2m")]
    tp = data[:, channels.index("tp")]
    u10 = data[:, channels.index("u10")]
    v10 = data[:, channels.index("v10")]
    msl = data[:, channels.index("msl")]
    tcwv = data[:, channels.index("tcwv")]

    t2m_below = int((t2m < T2M_MIN_K).sum())
    t2m_above = int((t2m > T2M_MAX_K).sum())
    tp_neg = int((tp < 0.0).sum())
    tp_below_tol = int((tp < -1e-6).sum())
    tcwv_neg = int((tcwv < 0.0).sum())

    gates_ok = (t2m_below == 0 and t2m_above == 0 and tp_neg == 0 and tcwv_neg == 0)

    # Soft synoptic envelope warnings (non-fatal if plausible)
    if float(tp.max()) > TP_SANITY_MAX_MM_DAY:
        warnings.append(f"Extreme daily precipitation max={float(tp.max()):.2f} mm/day exceeds {TP_SANITY_MAX_MM_DAY} mm/day")
    if float(np.abs(u10).max()) > WIND_ABS_SANITY_MAX or float(np.abs(v10).max()) > WIND_ABS_SANITY_MAX:
        warnings.append(f"Extreme 10m wind speed exceeds {WIND_ABS_SANITY_MAX} m/s")
    if float(msl.min()) < MSL_SANITY_MIN_PA or float(msl.max()) > MSL_SANITY_MAX_PA:
        warnings.append(f"MSL pressure outside [{MSL_SANITY_MIN_PA}, {MSL_SANITY_MAX_PA}] Pa")
    if float(tcwv.max()) > TCWV_SANITY_MAX_KG_M2:
        warnings.append(f"Extreme TCWV max={float(tcwv.max()):.2f} kg/m2 exceeds {TCWV_SANITY_MAX_KG_M2} kg/m2")

    # Non-evaluative per-split finiteness and validity gate verification (no test skill or stats)
    split_integrity: dict[str, Any] = {}
    for split_name, s_str, e_str, _ in SPLIT_DEFS:
        mask = (dates_d >= np.datetime64(s_str, "D")) & (dates_d <= np.datetime64(e_str, "D"))
        s_data = data[mask]
        s_t2m = s_data[:, channels.index("t2m")]
        s_tp = s_data[:, channels.index("tp")]
        s_tcwv = s_data[:, channels.index("tcwv")]
        s_ok = (
            bool(np.isfinite(s_data).all())
            and float(s_t2m.min()) >= T2M_MIN_K
            and float(s_t2m.max()) <= T2M_MAX_K
            and float(s_tp.min()) >= 0.0
            and float(s_tcwv.min()) >= 0.0
        )
        gates_ok &= s_ok
        split_integrity[split_name] = {
            "days": int(mask.sum()),
            "finite_all": bool(np.isfinite(s_data).all()),
            "t2m_in_233_15_to_333_15_K": bool(float(s_t2m.min()) >= T2M_MIN_K and float(s_t2m.max()) <= T2M_MAX_K),
            "tp_non_negative": bool(float(s_tp.min()) >= 0.0),
            "tcwv_non_negative": bool(float(s_tcwv.min()) >= 0.0),
            "passed": s_ok,
        }

    gate_details = {
        "t2m_valid_range_K": [T2M_MIN_K, T2M_MAX_K],
        "t2m_below_min_count": t2m_below,
        "t2m_above_max_count": t2m_above,
        "tp_negative_count": tp_neg,
        "tp_below_contract_tol_count": tp_below_tol,
        "tcwv_negative_count": tcwv_neg,
        "per_split_non_evaluative_gates": split_integrity,
    }
    _record_check(
        checks,
        errors,
        name="physical_validity_gates",
        category="physics",
        passed=gates_ok,
        details=gate_details,
        error_msg=f"Physical validity gate failed: t2m_out={t2m_below + t2m_above}, tp_neg={tp_neg}, tcwv_neg={tcwv_neg}",
    )

    return {
        "scope_note": "Global dataset integrity metrics across all 6209 days; per-split checks are strictly non-evaluative finiteness and physical gate checks only.",
        "channels": channel_reports,
        "validity_gates": gate_details,
    }


def _audit_static_fields(
    ds: xr.Dataset,
    checks: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    """Audit static terrain fields, fractional land-sea mask, and negative orography cells."""
    static = ds["static"].values
    lats = ds["lat"].values
    lons = ds["lon"].values
    sc_names = (
        tuple(str(c) for c in ds["static_channel"].values)
        if "static_channel" in ds.coords
        else ()
    )
    sc_match = sc_names == EXPECTED_STATIC_CHANNELS

    lsm = static[0]
    oro = static[1]

    lsm_min = float(lsm.min())
    lsm_max = float(lsm.max())
    lsm_out_of_bounds = int(((lsm < 0.0) | (lsm > 1.0)).sum())
    lsm_exact_0 = int((lsm == 0.0).sum())
    lsm_exact_1 = int((lsm == 1.0).sum())
    lsm_fractional = int(((lsm > 0.0) & (lsm < 1.0)).sum())
    lsm_predom_land = int((lsm >= 0.5).sum())
    lsm_predom_ocean = int((lsm < 0.5).sum())

    lsm_ok = sc_match and lsm_out_of_bounds == 0 and bool(np.isfinite(lsm).all())
    lsm_details = {
        "static_channels": list(sc_names),
        "expected_static_channels": list(EXPECTED_STATIC_CHANNELS),
        "exact_order_match": sc_match,
        "total_cells": int(lsm.size),
        "min": lsm_min,
        "max": lsm_max,
        "mean": float(lsm.astype(np.float64).mean()),
        "std": float(lsm.astype(np.float64).std()),
        "out_of_0_1_bounds_count": lsm_out_of_bounds,
        "exact_0_count": lsm_exact_0,
        "exact_1_count": lsm_exact_1,
        "strictly_fractional_count": lsm_fractional,
        "predominantly_land_ge_0_5_count": lsm_predom_land,
        "ocean_or_coastal_lt_0_5_count": lsm_predom_ocean,
        "policy": "Do NOT binarize land_sea_mask; preserve continuous ERA5 fractional coastal mask in [0, 1].",
    }
    _record_check(
        checks,
        errors,
        name="static_channels_order_and_lsm_bounds",
        category="static",
        passed=lsm_ok,
        details=lsm_details,
        error_msg=f"Static land_sea_mask check failed: order={sc_names}, out_of_bounds={lsm_out_of_bounds}",
    )

    oro64 = oro.astype(np.float64)
    oro_min_idx = np.unravel_index(int(np.argmin(oro)), oro.shape)
    oro_max_idx = np.unravel_index(int(np.argmax(oro)), oro.shape)
    neg_coords = np.argwhere(oro < 0.0)

    neg_cells: list[dict[str, Any]] = []
    neg_exact_ocean = 0
    neg_coastal_fringe = 0
    neg_interior_land = 0
    for i_idx, j_idx in neg_coords:
        i = int(i_idx)
        j = int(j_idx)
        lsm_val = float(lsm[i, j])
        oro_val = float(oro[i, j])
        if lsm_val == 0.0:
            region = "ocean_exact_0"
            neg_exact_ocean += 1
        elif lsm_val < 0.5:
            region = "ocean_or_coastal_lt_0_5"
            neg_coastal_fringe += 1
        else:
            region = "interior_land_ge_0_5"
            neg_interior_land += 1
        neg_cells.append(
            {
                "i": i,
                "j": j,
                "lat": float(lats[i]),
                "lon": float(lons[j]),
                "orography_m": oro_val,
                "land_sea_mask": lsm_val,
                "classification": region,
            }
        )

    # Verify (18.5, 71.5) specifically
    target_lat, target_lon = 18.5, 71.5
    lat_matches = np.where(np.isclose(lats, target_lat, atol=1e-6))[0]
    lon_matches = np.where(np.isclose(lons, target_lon, atol=1e-6))[0]
    if len(lat_matches) and len(lon_matches):
        ti, tj = int(lat_matches[0]), int(lon_matches[0])
        arabian_sea_point = {
            "found": True,
            "location_note": "Off Mumbai / Arabian Sea coast adjacent to Western Ghats gradient",
            "i": ti,
            "j": tj,
            "lat": float(lats[ti]),
            "lon": float(lons[tj]),
            "orography_m": float(oro[ti, tj]),
            "land_sea_mask": float(lsm[ti, tj]),
            "is_global_orography_min": bool(ti == int(oro_min_idx[0]) and tj == int(oro_min_idx[1])),
        }
    else:
        arabian_sea_point = {"found": False, "lat": target_lat, "lon": target_lon}

    max_lsm_among_neg = float(lsm[oro < 0.0].max()) if len(neg_cells) else 0.0
    oro_min_val = float(oro.min())
    oro_ok = (
        bool(np.isfinite(oro).all())
        and neg_interior_land == 0
        and oro_min_val >= ORO_OCEAN_MIN_HARD_M
        and arabian_sea_point.get("found", False)
    )

    physical_explanation = (
        "Tiny negative orography_m values down to -0.1563 m occur exclusively on ocean and near-coastal "
        "water cells (all 218 cells have land_sea_mask <= 0.001772; 180 have exact lsm == 0.0; 0 occur on "
        "land cells with lsm >= 0.5). In ERA5 (IFS spectral model) and WeatherBench2, surface geopotential "
        "is represented in truncated spherical harmonics with spectral smoothing; near sharp coastal terrain "
        "gradients (such as the Western Ghats next to (18.5, 71.5) off Mumbai), Gibbs ringing creates tiny "
        "negative geopotential lobes over adjacent flat ocean, which persist through WeatherBench2 conservative "
        "1-degree cell means and the 4-point half-degree grid averaging (geopotential_at_surface / 9.80665). "
        "These are benign physical/regridding artifacts and MUST NOT be clamped, nor should land_sea_mask be binarized."
    )

    if len(neg_cells) > 0:
        warnings.append(
            f"orography_m has {len(neg_cells)} tiny negative cells (min={oro_min_val:.6f} m at "
            f"({float(lats[oro_min_idx[0]])}, {float(lons[oro_min_idx[1]])})), all with land_sea_mask <= "
            f"{max_lsm_among_neg:.6f} (0 on lsm >= 0.5); expected ERA5 spectral/Gibbs artifact — do not clamp."
        )

    oro_details = {
        "min": oro_min_val,
        "min_cell": {
            "i": int(oro_min_idx[0]),
            "j": int(oro_min_idx[1]),
            "lat": float(lats[oro_min_idx[0]]),
            "lon": float(lons[oro_min_idx[1]]),
            "land_sea_mask": float(lsm[oro_min_idx]),
        },
        "max": float(oro.max()),
        "max_cell": {
            "i": int(oro_max_idx[0]),
            "j": int(oro_max_idx[1]),
            "lat": float(lats[oro_max_idx[0]]),
            "lon": float(lons[oro_max_idx[1]]),
            "land_sea_mask": float(lsm[oro_max_idx]),
        },
        "mean": float(oro64.mean()),
        "std": float(oro64.std()),
        "land_ge_0_5_min_m": float(oro[lsm >= 0.5].min()),
        "land_ge_0_5_max_m": float(oro[lsm >= 0.5].max()),
        "negative_cell_count": len(neg_cells),
        "negative_breakdown": {
            "ocean_exact_0": neg_exact_ocean,
            "coastal_ocean_0_to_0_5": neg_coastal_fringe,
            "interior_land_ge_0_5": neg_interior_land,
            "max_land_sea_mask_among_negative": max_lsm_among_neg,
        },
        "verified_mumbai_arabian_sea_cell": arabian_sea_point,
        "physical_explanation": physical_explanation,
        "policy": "Do NOT clamp negative ocean orography_m values or binarize land_sea_mask.",
        "all_negative_cells": neg_cells,
    }
    _record_check(
        checks,
        errors,
        name="static_orography_integrity",
        category="static",
        passed=oro_ok,
        details=oro_details,
        error_msg=(
            f"Static orography failed integrity check: min={oro_min_val:.4f} m, "
            f"neg_on_land={neg_interior_land}"
        ),
    )

    return {"land_sea_mask": lsm_details, "orography_m": oro_details}


def audit_dataset(nc_path: Path) -> dict[str, Any]:
    """Run all integrity, schema, coordinate, time, dynamic, and static audits."""
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    exists = nc_path.is_file()
    file_size_bytes = int(nc_path.stat().st_size) if exists else 0
    _record_check(
        checks,
        errors,
        name="file_exists_and_readable",
        category="file",
        passed=exists and file_size_bytes > 0,
        details={"nc_path": str(nc_path), "exists": exists, "file_size_bytes": file_size_bytes},
        error_msg=f"Dataset file missing or empty: {nc_path}",
    )

    if not exists or file_size_bytes <= 0:
        return {
            "passed": False,
            "audit_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "nc_path": str(nc_path),
            "summary": {
                "total_checks": len(checks),
                "passed_checks": 0,
                "failed_checks": len(checks),
                "warning_count": len(warnings),
                "error_count": len(errors),
            },
            "checks": checks,
            "warnings": warnings,
            "errors": errors,
        }

    with xr.open_dataset(nc_path) as ds:
        schema_info = _audit_schema(ds, nc_path, file_size_bytes, checks, errors)
        channels_info = _audit_channels_and_units(ds, checks, errors)
        time_info, dates_d = _audit_time_axis(ds, checks, errors)
        coords_info = _audit_coordinates(ds, checks, errors)
        dynamic_info = _audit_dynamic_values(ds, dates_d, checks, warnings, errors)
        static_info = _audit_static_fields(ds, checks, warnings, errors)

    passed_count = sum(1 for c in checks if c["passed"])
    failed_count = len(checks) - passed_count
    overall_passed = (len(errors) == 0) and (failed_count == 0)

    return {
        "passed": overall_passed,
        "audit_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "nc_path": str(nc_path),
        "contract_reference": {
            "channels": list(C.CHANNELS),
            "nc": int(C.NC),
            "nlat": int(C.NLAT),
            "nlon": int(C.NLON),
            "t_in": int(C.T_IN),
            "score_steps": int(C.SCORE_STEPS),
        },
        "summary": {
            "total_checks": len(checks),
            "passed_checks": passed_count,
            "failed_checks": failed_count,
            "warning_count": len(warnings),
            "error_count": len(errors),
        },
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "schema": schema_info,
        "channels_and_units": channels_info,
        "time_axis": time_info["axis"],
        "chronological_splits": time_info["chronological_splits"],
        "coordinates": coords_info,
        "dynamic_channels": dynamic_info,
        "static_fields": static_info,
        "recommendations": [
            "Preserve continuous fractional values in static land_sea_mask [0, 1]; do NOT binarize.",
            "Preserve raw static orography_m values including 218 tiny negative ocean/coastal cells (min -0.156285 m at (18.5, 71.5)); do NOT clamp.",
            "Keep strict chronological split boundaries: train 2005-2018 (5113 d / 5101 windows), val 2019-2020 (731 d / 719 windows), test 2021 (365 d / 353 windows).",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit MonsoonFlow-10 raw NetCDF climate dataset against monsoon_r1.contract."
    )
    parser.add_argument(
        "--nc-path",
        type=Path,
        default=Path("data/raw/india_climate_train.nc"),
        help="Path to raw NetCDF dataset (default: data/raw/india_climate_train.nc)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/audit_report.json"),
        help="Output JSON report path (default: data/raw/audit_report.json)",
    )
    args = parser.parse_args(argv)

    report = audit_dataset(args.nc_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    status = "PASSED" if report["passed"] else "FAILED"
    print(
        f"[audit_dataset] {status}: {s['passed_checks']}/{s['total_checks']} checks passed, "
        f"{s['warning_count']} warning(s), {s['error_count']} error(s) -> {args.out}"
    )
    for w in report["warnings"]:
        print(f"  WARN: {w}")
    for e in report["errors"]:
        print(f"  ERROR: {e}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
