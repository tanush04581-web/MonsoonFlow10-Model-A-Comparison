"""Phase 2: Real validation baselines (Persistence & 366-day Climatology).

Evaluates strictly on the chronological validation split (2019-2020,
``data/raw/splits/val.npz``) using training-derived 2005-2018 climatology
(``data/raw/splits/train_climatology.npy``) and official ``monsoon_r1.scoring``
conventions. Never touches the sealed 2021 test split.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from monsoon_r1 import contract as C
    from monsoon_r1 import scoring as S
    from monsoon_r1 import stats as ST
except ImportError:  # pragma: no cover
    _KIT_DIR = Path(__file__).resolve().parent.parent / "resources" / "yug-monsoon-kit"
    if _KIT_DIR.exists() and str(_KIT_DIR) not in sys.path:
        sys.path.insert(0, str(_KIT_DIR))
    from monsoon_r1 import contract as C
    from monsoon_r1 import scoring as S
    from monsoon_r1 import stats as ST


def _int_to_date(d_int: int) -> dt.date:
    return dt.date(int(d_int) // 10000, (int(d_int) // 100) % 100, int(d_int) % 100)


def build_validation_rollouts(
    val_npz: Path = Path("data/raw/splits/val.npz"),
    clim_npy: Path = Path("data/raw/splits/train_climatology.npy"),
    t_in: int = C.T_IN,
    steps: int = C.SCORE_STEPS,
    window_stride: int = 1,
) -> dict[str, np.ndarray]:
    """Construct chronological 10-day target, persistence, and climatology rollouts."""
    with np.load(val_npz) as z:
        val_data = np.asarray(z["data"], dtype=np.float32)
        val_dates = np.asarray(z["dates"], dtype=np.int32)

    if int(val_dates.min()) < 20190101 or int(val_dates.max()) > 20201231:
        raise ValueError(f"Validation split out of 2019-2020 range: {val_dates[0]}..{val_dates[-1]}")

    clim = np.load(clim_npy).astype(np.float32)
    if clim.shape != (366, C.NC, C.NLAT, C.NLON):
        raise ValueError(f"Unexpected climatology shape: {clim.shape}")

    n_days = len(val_data)
    win_len = t_in + steps
    indices = list(range(0, n_days - win_len + 1, window_stride))

    y_true = np.empty((len(indices), steps, C.NC, C.NLAT, C.NLON), dtype=np.float32)
    y_persist = np.empty_like(y_true)
    y_clim = np.empty_like(y_true)
    t0_dates = np.empty((len(indices),), dtype=np.int32)

    # Precompute climatology slice per day index in val_dates using official _doy_index
    clim_seq = np.stack([ST.clim_for(clim, _int_to_date(int(d))) for d in val_dates], axis=0)

    for w_i, start in enumerate(indices):
        in_end = start + t_in
        t0_dates[w_i] = val_dates[in_end - 1]
        y_true[w_i] = val_data[in_end : in_end + steps]
        y_persist[w_i] = np.repeat(val_data[in_end - 1 : in_end], steps, axis=0)
        y_clim[w_i] = clim_seq[in_end : in_end + steps]

    return {
        "y_true": y_true,
        "y_persist": y_persist,
        "y_clim": y_clim,
        "clim_seq": clim_seq,
        "val_dates": val_dates,
        "t0_dates": t0_dates,
        "indices": np.asarray(indices, dtype=np.int32),
    }


def _compute_contingency(pred_tp: np.ndarray, true_tp: np.ndarray, thr: float) -> dict[str, float]:
    p = pred_tp > thr
    t = true_tp > thr
    hits = int((p & t).sum())
    misses = int((~p & t).sum())
    false_alarms = int((p & ~t).sum())
    correct_neg = int((~p & ~t).sum())
    pod = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float("nan")
    csi_val = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else float("nan")
    freq_bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else float("nan")
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_neg,
        "csi": float(csi_val),
        "pod": float(pod),
        "far": float(far),
        "frequency_bias": float(freq_bias),
    }


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    af -= af.mean()
    bf -= bf.mean()
    denom = float(np.sqrt((af * af).sum() * (bf * bf).sum()))
    return float((af * bf).sum() / denom) if denom > 1e-12 else float("nan")


def _mae_lat(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    w = C.lat_weights()[None, None, :, None]
    return (np.abs(pred - true) * w).mean(axis=(0, 2, 3))


def evaluate_baseline_predictions(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_clim_target: np.ndarray,
    static_arr: np.ndarray,
    lead_days: tuple[int, ...] = tuple(range(1, 11)),
) -> dict[str, Any]:
    """Compute comprehensive metrics for a baseline forecast [W, 10, 6, 32, 36]."""
    lat_w = C.lat_weights()[None, :, None]  # (1, 32, 36) broadcast via (1, 32, 1)
    lsm = static_arr[0]
    oro = static_arr[1]
    lat_grid = C.LAT[:, None]
    lon_grid = C.LON[None, :]

    regions = {
        "all_grid": np.ones((C.NLAT, C.NLON), dtype=bool),
        "bay_of_bengal": C.box_mask(C.BOX_BAY_OF_BENGAL),
        "arabian_sea": C.box_mask(C.BOX_ARABIAN_SEA),
        "indian_land": lsm >= 0.5,
        "ocean_coastal": lsm < 0.5,
        "monsoon_core_zone": (lat_grid >= 18.0) & (lat_grid <= 28.0) & (lon_grid >= 73.0) & (lon_grid <= 88.0) & (lsm >= 0.5),
        "high_orography_ge_1000m": oro >= 1000.0,
    }

    by_lead: dict[str, Any] = {}
    for day in lead_days:
        k = day - 1
        p_k = y_pred[:, k]  # [W, 6, 32, 36]
        t_k = y_true[:, k]
        c_k = y_clim_target[:, k]

        rmse_lat_ch = S.rmse_lat(p_k, t_k)
        mae_lat_ch = (np.abs(p_k - t_k) * lat_w[:, None, :, :]).mean(axis=(0, 2, 3))
        bias_ch = (p_k - t_k).mean(axis=(0, 2, 3))

        corr_raw = [_pearson_corr(p_k[:, ch], t_k[:, ch]) for ch in range(C.NC)]
        corr_anom = [_pearson_corr(p_k[:, ch] - c_k[:, ch], t_k[:, ch] - c_k[:, ch]) for ch in range(C.NC)]

        vr_list = [S.variance_ratio(p_k[w], t_k[w]) for w in range(p_k.shape[0])]
        vr_mean = np.nanmean(np.asarray(vr_list), axis=0)

        rain_10 = _compute_contingency(p_k[:, 1], t_k[:, 1], 10.0)
        rain_15_6 = _compute_contingency(p_k[:, 1], t_k[:, 1], 15.6)
        rain_64_5 = _compute_contingency(p_k[:, 1], t_k[:, 1], 64.5)

        by_lead[f"day_{day}"] = {
            "rmse_lat": {ch: float(v) for ch, v in zip(C.CHANNELS, rmse_lat_ch)},
            "mae_lat": {ch: float(v) for ch, v in zip(C.CHANNELS, mae_lat_ch)},
            "bias": {ch: float(v) for ch, v in zip(C.CHANNELS, bias_ch)},
            "pearson_r": {ch: float(v) for ch, v in zip(C.CHANNELS, corr_raw)},
            "acc_vs_train_clim": {ch: float(v) for ch, v in zip(C.CHANNELS, corr_anom)},
            "spatial_variance_ratio_mean": {ch: float(v) for ch, v in zip(C.CHANNELS, vr_mean)},
            "rain_ge_10mm": rain_10,
            "rain_ge_15_6mm": rain_15_6,
            "rain_ge_64_5mm": rain_64_5,
        }

    # Regional breakdown on official lead days (1, 4, 8)
    by_region: dict[str, Any] = {}
    for reg_name, mask in regions.items():
        reg_res: dict[str, Any] = {"cell_count": int(mask.sum())}
        for day in C.LEAD_DAYS:
            k = day - 1
            p_k = y_pred[:, k][:, :, mask]  # [W, 6, M]
            t_k = y_true[:, k][:, :, mask]
            rmse_reg = np.sqrt(((p_k - t_k) ** 2).mean(axis=(0, 2)))
            mae_reg = np.abs(p_k - t_k).mean(axis=(0, 2))
            reg_res[f"day_{day}"] = {
                "rmse": {ch: float(v) for ch, v in zip(C.CHANNELS, rmse_reg)},
                "mae": {ch: float(v) for ch, v in zip(C.CHANNELS, mae_reg)},
                "rain_csi_10mm": float(S.csi(p_k[:, 1], t_k[:, 1], thr=10.0)),
            }
        by_region[reg_name] = reg_res

    # Rainfall intensity stratified errors on official lead days (1, 4, 8)
    intensity_bins = (
        ("dry_lt_0_1", 0.0, 0.1),
        ("light_mod_0_1_to_10", 0.1, 10.0),
        ("heavy_10_to_64_5", 10.0, 64.5),
        ("very_heavy_ge_64_5", 64.5, float("inf")),
    )
    by_rain_intensity: dict[str, Any] = {}
    for day in C.LEAD_DAYS:
        k = day - 1
        p_tp = y_pred[:, k, 1]
        t_tp = y_true[:, k, 1]
        day_bins: dict[str, Any] = {}
        for bname, lo, hi in intensity_bins:
            m = (t_tp >= lo) & (t_tp < hi)
            cnt = int(m.sum())
            if cnt > 0:
                diff = p_tp[m] - t_tp[m]
                day_bins[bname] = {
                    "count": cnt,
                    "fraction": float(cnt / t_tp.size),
                    "rmse": float(np.sqrt(np.mean(diff ** 2))),
                    "mae": float(np.mean(np.abs(diff))),
                    "bias": float(np.mean(diff)),
                }
            else:
                day_bins[bname] = {"count": 0, "fraction": 0.0, "rmse": None, "mae": None, "bias": None}
        by_rain_intensity[f"day_{day}"] = day_bins

    return {
        "by_lead_day": by_lead,
        "by_region": by_region,
        "by_rainfall_intensity": by_rain_intensity,
    }


def compute_all_baselines(
    val_npz: Path = Path("data/raw/splits/val.npz"),
    clim_npy: Path = Path("data/raw/splits/train_climatology.npy"),
    static_npy: Path = Path("data/raw/splits/static.npy"),
    out_json: Path = Path("checkpoints/baselines_val.json"),
) -> dict[str, Any]:
    """Run full Phase 2 baseline evaluation and save JSON report."""
    static_arr = np.load(static_npy).astype(np.float32)

    full = build_validation_rollouts(val_npz=val_npz, clim_npy=clim_npy, window_stride=1)
    y_true = full["y_true"]
    y_persist = full["y_persist"]
    y_clim = full["y_clim"]

    persist_metrics = evaluate_baseline_predictions(y_persist, y_true, y_clim, static_arr)
    clim_metrics = evaluate_baseline_predictions(y_clim, y_true, y_clim, static_arr)

    # Identify min(RMSE_persist, RMSE_clim) target floor per lead day & channel
    target_floor: dict[str, Any] = {}
    for day in C.LEAD_DAYS:
        dkey = f"day_{day}"
        p_r = persist_metrics["by_lead_day"][dkey]["rmse_lat"]
        c_r = clim_metrics["by_lead_day"][dkey]["rmse_lat"]
        p_csi = persist_metrics["by_lead_day"][dkey]["rain_ge_10mm"]["csi"]
        c_csi = clim_metrics["by_lead_day"][dkey]["rain_ge_10mm"]["csi"]
        target_floor[dkey] = {
            "best_ref_rmse_lat": {
                ch: {
                    "min_rmse": float(min(p_r[ch], c_r[ch])),
                    "harder_baseline": "persistence" if p_r[ch] <= c_r[ch] else "climatology",
                    "persistence_rmse": p_r[ch],
                    "climatology_rmse": c_r[ch],
                }
                for ch in C.CHANNELS
            },
            "best_ref_csi_10mm": float(np.nanmax([p_csi, c_csi])),
            "persistence_csi_10mm": float(p_csi),
            "climatology_csi_10mm": float(c_csi),
        }

    # Also compute uniform 103-window strided sample (matching evaluation/benchmark_baselines.py stride)
    strided = build_validation_rollouts(
        val_npz=val_npz,
        clim_npy=clim_npy,
        window_stride=max(1, (len(full["val_dates"]) - (C.T_IN + C.SCORE_STEPS) + 1) // 100),
    )
    strided_summary: dict[str, Any] = {"num_windows": int(len(strided["y_true"]))}
    for name, arr in (("persistence", strided["y_persist"]), ("climatology", strided["y_clim"])):
        strided_summary[name] = {}
        for day in C.LEAD_DAYS:
            k = day - 1
            rmses = S.rmse_lat(arr[:, k], strided["y_true"][:, k])
            strided_summary[name][f"day_{day}"] = {
                "rmse_lat": {ch: float(v) for ch, v in zip(C.CHANNELS, rmses)},
                "csi_10mm": float(S.csi(arr[:, k, 1], strided["y_true"][:, k, 1], thr=10.0)),
            }

    report = {
        "split": "validation_2019_2020",
        "val_npz": str(val_npz),
        "climatology_source": str(clim_npy),
        "num_windows_full": int(len(y_true)),
        "t_in": int(C.T_IN),
        "steps": int(C.SCORE_STEPS),
        "t0_date_range": [int(full["t0_dates"][0]), int(full["t0_dates"][-1])],
        "official_target_floor_summary": target_floor,
        "persistence": persist_metrics,
        "climatology": clim_metrics,
        "strided_103_window_reference": strided_summary,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute real 2019-2020 Persistence & Climatology baselines.")
    parser.add_argument("--val-npz", type=Path, default=Path("data/raw/splits/val.npz"))
    parser.add_argument("--clim-npy", type=Path, default=Path("data/raw/splits/train_climatology.npy"))
    parser.add_argument("--static-npy", type=Path, default=Path("data/raw/splits/static.npy"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/baselines_val.json"))
    args = parser.parse_args()

    rep = compute_all_baselines(args.val_npz, args.clim_npy, args.static_npy, args.out)
    print(f"[baselines] Evaluated {rep['num_windows_full']} chronological val windows -> {args.out}")
    for day in C.LEAD_DAYS:
        dkey = f"day_{day}"
        fl = rep["official_target_floor_summary"][dkey]
        print(f"=== Lead Day {day} ===")
        for ch in C.CHANNELS:
            info = fl["best_ref_rmse_lat"][ch]
            print(
                f"  {ch:5s}: persist={info['persistence_rmse']:8.3f} | "
                f"clim={info['climatology_rmse']:8.3f} | harder={info['harder_baseline']}"
            )
        print(
            f"  CSI@10mm: persist={fl['persistence_csi_10mm']:.4f} | "
            f"clim={fl['climatology_csi_10mm']:.4f}"
        )
    print("=== Day-8 Spatial Variance Ratio (Mean across windows) ===")
    for name in ("persistence", "climatology"):
        vr = rep[name]["by_lead_day"]["day_8"]["spatial_variance_ratio_mean"]
        print(f"  {name:11s}: " + ", ".join(f"{c}={vr[c]:.3f}" for c in C.CHANNELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
