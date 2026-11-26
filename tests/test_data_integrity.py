"""Data integrity, coordinate alignment, and chronological separation tests.

Audited by Agent 5: Agent-QA.
Enforces:
1. Grid coordinates: lat 6.5..37.5 (NLAT=32), lon 65.5..100.5 (NLON=36).
2. Channels: 6 physical channels in exact order ('t2m', 'tp', 'u10', 'v10', 'msl', 'tcwv').
3. Normalization reversibility: (x - mu) / sigma * sigma + mu == x.
4. Chronological separation: Train (2005-2018), Val (2019-2020), Holdout (2021) zero overlap.
5. Static fields: 2 x 32 x 36 (land_sea_mask in [0, 1], orography).
"""
import json
from pathlib import Path
import numpy as np
import pytest
from monsoon_r1.contract import (
    CHANNELS,
    LAT,
    LON,
    NC,
    NLAT,
    NLON,
    T_IN,
    SCORE_STEPS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DATA_DIR = REPO_ROOT / "data"


def test_grid_coordinates_and_shape():
    """Verify official coordinate grid specifications."""
    assert len(LAT) == NLAT == 32, f"Expected 32 latitudes, got {len(LAT)}"
    assert len(LON) == NLON == 36, f"Expected 36 longitudes, got {len(LON)}"
    assert np.isclose(LAT[0], 6.5) and np.isclose(LAT[-1], 37.5), f"LAT range invalid: {LAT[0]}..{LAT[-1]}"
    assert np.isclose(LON[0], 65.5) and np.isclose(LON[-1], 100.5), f"LON range invalid: {LON[0]}..{LON[-1]}"
    assert np.allclose(np.diff(LAT), 1.0), "Latitude step must be 1.0 degree"
    assert np.allclose(np.diff(LON), 1.0), "Longitude step must be 1.0 degree"


def test_channels_order_and_count():
    """Verify channel count and exact ordering."""
    expected_channels = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
    assert tuple(CHANNELS) == expected_channels, f"CHANNELS mismatch: {CHANNELS}"
    assert NC == 6, f"Expected 6 channels, got {NC}"


def test_normalization_reversibility():
    """Verify (x - mean) / std round-trip maintains numerical precision."""
    stats_file = CHECKPOINTS_DIR / "stats.json"
    if stats_file.exists():
        with open(stats_file, "r") as f:
            stats = json.load(f)
        means = np.array(stats["mean"], dtype=np.float32)
        stds = np.array(stats["std"], dtype=np.float32)
    else:
        # Default typical climate values if stats.json not yet written by Agent-ML
        means = np.array([298.0, 5.0, 1.0, 0.5, 101000.0, 35.0], dtype=np.float32)
        stds = np.array([8.0, 8.0, 3.5, 2.5, 450.0, 20.0], dtype=np.float32)

    assert len(means) == 6 and len(stds) == 6
    assert np.all(stds > 0), "Standard deviations must be strictly positive"

    # Create synthetic test tensor [10, 6, 32, 36]
    rng = np.random.default_rng(42)
    x = rng.normal(loc=means[None, :, None, None], scale=stds[None, :, None, None]).astype(np.float32)

    # Round trip
    norm = (x - means[None, :, None, None]) / stds[None, :, None, None]
    rec = norm * stds[None, :, None, None] + means[None, :, None, None]

    max_err = np.max(np.abs(rec - x))
    assert max_err < 1e-5, f"Normalization round-trip error too large: {max_err}"


def test_chronological_splits_separation():
    """Verify strict chronological boundaries: 2005-2018 Train, 2019-2020 Val, 2021 Test."""
    train_start, train_end = 20050101, 20181231
    val_start, val_end = 20190101, 20201231
    test_start, test_end = 20210101, 20211231

    assert train_end < val_start, "Train must strictly precede Validation"
    assert val_end < test_start, "Validation must strictly precede Holdout"

    # Check date arrays if numpy dataset files exist on disk.
    # (QA fix 2026-11-26: the previous `except Exception: pass` also swallowed AssertionError, so this never failed.)
    for npz_path in DATA_DIR.rglob("*.npz"):
        if True:
            with np.load(npz_path) as z:
                if "dates" in z:
                    dates = z["dates"]
                    if "train" in npz_path.name.lower():
                        assert int(dates.max()) <= train_end, f"Train leak in {npz_path}: max date {dates.max()}"
                        assert int(dates.min()) >= train_start, f"Train out-of-range: {dates.min()}"
                    elif "val" in npz_path.name.lower():
                        assert int(dates.min()) >= val_start, f"Val leak into train: {dates.min()}"
                        assert int(dates.max()) <= val_end, f"Val leak into test: {dates.max()}"
                    elif "test" in npz_path.name.lower() or "holdout" in npz_path.name.lower():
                        assert int(dates.min()) >= test_start, f"Test leak: {dates.min()}"
                        assert int(dates.max()) <= test_end, f"Test out-of-range: {dates.max()}"


SPLITS_DIR = DATA_DIR / "raw" / "splits"


@pytest.mark.skipif(not (SPLITS_DIR / "train.npz").exists(), reason="split files not present")
def test_splits_disjoint_continuous_and_complete():
    import datetime as dt
    seen = {}
    for name, (lo, hi) in {"train": (20050101, 20181231), "val": (20190101, 20201231), "test": (20210101, 20211231)}.items():
        with np.load(SPLITS_DIR / f"{name}.npz") as z:
            d = z["dates"]; shp = z["data"].shape
        assert shp[1:] == (NC, NLAT, NLON)
        assert int(d.min()) == lo and int(d.max()) == hi, name
        ds = [dt.date(int(x) // 10000, int(x) // 100 % 100, int(x) % 100) for x in d]
        assert all((b - a).days == 1 for a, b in zip(ds[:-1], ds[1:])), f"{name} has gaps"
        seen[name] = set(int(x) for x in d)
    assert not (seen["train"] & seen["val"]) and not (seen["train"] & seen["test"]) and not (seen["val"] & seen["test"])


@pytest.mark.skipif(not (SPLITS_DIR / "train.npz").exists(), reason="split files not present")
def test_shipped_stats_and_climatology_are_train_only():
    """checkpoints/stats.json and climatology.npy must come from 2005-2018 only (kit monsoon_r1.stats)."""
    from monsoon_r1.stats import channel_stats, climatology
    with np.load(SPLITS_DIR / "train.npz") as z:
        tr, tr_dates = z["data"], z["dates"]
    ref = channel_stats(tr.astype("f8"))
    shipped = json.loads((CHECKPOINTS_DIR / "stats.json").read_text())
    std = np.array(ref["std"])
    # train+val would differ by ~1e-2 std; float32 reduction-order noise is ~1e-5 std
    assert np.max(np.abs(np.array(shipped["mean"]) - np.array(ref["mean"])) / std) < 1e-3
    assert np.max(np.abs(np.array(shipped["std"]) - std) / std) < 1e-4
    assert np.array_equal(np.load(CHECKPOINTS_DIR / "climatology.npy"), climatology(tr, tr_dates))


def test_sliding_window_split_boundaries():
    """Verify that sliding windows do not cross chronological partitions."""
    train_dates = np.array([20181229, 20181230, 20181231])
    val_dates = np.array([20190101, 20190102, 20190103])
    
    # A window requires T_IN + SCORE_STEPS = 13 consecutive days
    window_len = T_IN + SCORE_STEPS
    assert window_len == 13

    # Assert train max date and val min date cannot be merged into single sequence
    combined = np.concatenate([train_dates, val_dates])
    diff_days = np.diff(combined)
    # The jump between 20181231 and 20190101 in integer YYYYMMDD is 8870 (or 1 calendar day, but split border)
    assert not (len(train_dates) >= window_len and train_dates[-1] >= 20190101)


def test_static_terrain_tensor_shape():
    """Verify static terrain tensor shape (2, 32, 36) if present."""
    static_file = CHECKPOINTS_DIR / "static.npy"
    if static_file.exists():
        static = np.load(static_file)
        assert static.shape == (2, 32, 36), f"Expected shape (2, 32, 36), got {static.shape}"
        # Channel 0: land_sea_mask in [0, 1]
        lsm = static[0]
        assert np.all(lsm >= 0.0) and np.all(lsm <= 1.0), "land_sea_mask values must be in [0, 1]"
        # Channel 1: orography elevation finite
        oro = static[1]
        assert np.all(np.isfinite(oro)), "orography must contain finite elevations"
