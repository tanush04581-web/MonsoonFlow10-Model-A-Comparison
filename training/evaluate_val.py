"""Phase 5 Scientific Validation Suite for MonsoonFlow-10.

Evaluates a trained model checkpoint strictly on the 2019-2020 validation split
(``data/raw/splits/val.npz``) against real Persistence and 2005-2018 Climatology
baselines. Never touches the sealed 2021 test split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from monsoon_r1 import contract as C
    from monsoon_r1 import scoring as S
except ImportError:  # pragma: no cover
    _KIT_DIR = Path(__file__).resolve().parent.parent / "resources" / "yug-monsoon-kit"
    if _KIT_DIR.exists() and str(_KIT_DIR) not in sys.path:
        sys.path.insert(0, str(_KIT_DIR))
    from monsoon_r1 import contract as C
    from monsoon_r1 import scoring as S

from models.backbones import BaseMonsoonForecaster, load_model_from_checkpoint
from training.baselines import build_validation_rollouts, evaluate_baseline_predictions


def _fss_2d(pred_tp: np.ndarray, true_tp: np.ndarray, thr: float = 10.0, size: int = 3) -> float:
    """Fractions Skill Score over [W, H, W_lon] with uniform ``size x size`` neighborhood."""
    p = (pred_tp > thr).astype(np.float64)
    t = (true_tp > thr).astype(np.float64)
    pad = size // 2
    # Compute 2D box average via integral/torch avg_pool2d for speed and clarity
    p_t = torch.from_numpy(p).unsqueeze(1)
    t_t = torch.from_numpy(t).unsqueeze(1)
    fp = torch.nn.functional.avg_pool2d(p_t, kernel_size=size, stride=1, padding=pad, count_include_pad=False).numpy()
    ft = torch.nn.functional.avg_pool2d(t_t, kernel_size=size, stride=1, padding=pad, count_include_pad=False).numpy()
    num = float(np.mean((fp - ft) ** 2))
    den = float(np.mean(fp ** 2) + np.mean(ft ** 2))
    return float(1.0 - num / den) if den > 1e-12 else float("nan")


def _fft_band_ratios(pred_day: np.ndarray, true_day: np.ndarray) -> dict[str, dict[str, float]]:
    """Compare 2D radial power spectrum energy at Day 8 across low/mid/high wavenumbers."""
    # pred_day, true_day: [W, 6, 32, 36]
    fp = np.fft.rfft2(pred_day - pred_day.mean(axis=(-2, -1), keepdims=True), axes=(-2, -1))
    ft = np.fft.rfft2(true_day - true_day.mean(axis=(-2, -1), keepdims=True), axes=(-2, -1))
    pp = np.mean(np.abs(fp) ** 2, axis=0)  # [6, 32, 19]
    pt = np.mean(np.abs(ft) ** 2, axis=0)

    ky = np.fft.fftfreq(C.NLAT)[:, None]
    kx = np.fft.rfftfreq(C.NLON)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    bands = {
        "synoptic_low_k": (kr > 0.0) & (kr <= 0.15),
        "meso_mid_k": (kr > 0.15) & (kr <= 0.35),
        "grid_high_k": kr > 0.35,
    }
    out: dict[str, dict[str, float]] = {}
    for ch_i, ch in enumerate(C.CHANNELS):
        out[ch] = {}
        for bname, bm in bands.items():
            ep = float(pp[ch_i, bm].sum())
            et = float(pt[ch_i, bm].sum())
            out[ch][bname] = float(ep / max(et, 1e-30))
    return out


@torch.no_grad()
def predict_validation_rollouts(
    model: BaseMonsoonForecaster,
    val_npz: Path = Path("data/raw/splits/val.npz"),
    device: str = "cuda",
    batch_size: int = 64,
    steps: int = C.SCORE_STEPS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run autoregressive predictions on all 719 validation windows."""
    roll_data = build_validation_rollouts(val_npz=val_npz, steps=steps, window_stride=1)
    with np.load(val_npz) as z:
        val_data = np.asarray(z["data"], dtype=np.float32)
        val_doy = np.asarray(z["doy"], dtype=np.int32)

    indices = roll_data["indices"]
    n_win = len(indices)
    x_hist = np.empty((n_win, C.T_IN, C.NC, C.NLAT, C.NLON), dtype=np.float32)
    doy_last = np.empty((n_win, 2), dtype=np.float32)
    for w_i, start in enumerate(indices):
        x_hist[w_i] = val_data[start : start + C.T_IN]
        doy_last[w_i] = np.asarray(C.doy_sincos(int(val_doy[start + C.T_IN - 1])), dtype=np.float32)

    was_training = model.training
    model.eval()
    preds = np.empty((n_win, steps, C.NC, C.NLAT, C.NLON), dtype=np.float32)
    dev = torch.device(device)
    model_dev = next(model.parameters()).device
    if model_dev != dev:
        model = model.to(dev)

    for i0 in range(0, n_win, batch_size):
        bx = torch.from_numpy(x_hist[i0 : i0 + batch_size]).to(dev)
        bd = torch.from_numpy(doy_last[i0 : i0 + batch_size]).to(dev)
        out = model(bx, bd, steps=steps, perturbation=None)
        preds[i0 : i0 + batch_size] = out.cpu().numpy()

    if was_training:
        model.train()
    return preds, roll_data


@torch.no_grad()
def evaluate_shocks_and_cpu_latency(
    model: BaseMonsoonForecaster,
    val_npz: Path = Path("data/raw/splits/val.npz"),
    stats_std: np.ndarray | None = None,
    steps: int = C.SCORE_STEPS,
) -> dict[str, Any]:
    """Evaluate official causality checks across representative monsoon windows + 4-core CPU latency."""
    with np.load(val_npz) as z:
        val_data = np.asarray(z["data"], dtype=np.float32)
        val_dates = np.asarray(z["dates"], dtype=np.int32)
        val_doy = np.asarray(z["doy"], dtype=np.int32)

    std_arr = (
        np.asarray(stats_std, dtype=np.float32)
        if stats_std is not None
        else model.stats_std.detach().cpu().numpy().astype(np.float32)
    )

    # Pick 6 diverse seasonal/monsoon validation windows (winter, pre-monsoon, onset, active, retreat, 2020 monsoon)
    n_win = len(val_data) - (C.T_IN + steps) + 1
    probe_starts = sorted({min(idx, n_win - 1) for idx in (15, 110, 165, 210, 260, 560)})

    orig_dev = next(model.parameters()).device
    model_cpu = model.to("cpu").eval()
    torch.set_num_threads(4)

    # 1. Measure 4-core CPU latency on single sample (matching serve.py inference)
    x0 = torch.from_numpy(val_data[probe_starts[2] : probe_starts[2] + C.T_IN].copy())
    d0 = torch.tensor(C.doy_sincos(int(val_doy[probe_starts[2] + C.T_IN - 1])), dtype=torch.float32)
    _ = model_cpu(x0, d0, steps=steps, perturbation=None)
    lat_ms = []
    for _ in range(12):
        ts = time.perf_counter()
        _ = model_cpu(x0, d0, steps=steps, perturbation=None)
        lat_ms.append((time.perf_counter() - ts) * 1000.0)

    # 2. Causality checks across the probe windows
    warm_results = []
    stall_results = []
    null_diffs_abs = []
    null_diffs_norm = []

    for st in probe_starts:
        x_in = torch.from_numpy(val_data[st : st + C.T_IN].copy())
        doy_pair = torch.tensor(C.doy_sincos(int(val_doy[st + C.T_IN - 1])), dtype=torch.float32)
        last_np = val_data[st + C.T_IN - 1]

        base = model_cpu(x_in, doy_pair, steps=steps, perturbation=None).numpy()
        zero_p = torch.zeros((C.NC, C.NLAT, C.NLON), dtype=torch.float32)
        out_zero = model_cpu(x_in, doy_pair, steps=steps, perturbation=zero_p).numpy()

        null_diffs_abs.append(float(np.max(np.abs(out_zero - base))))
        null_diffs_norm.append(float(np.max(np.abs(out_zero - base) / std_arr[None, :, None, None])))

        p_warm = S.make_shock("warm_ocean", last_np, std_arr)
        out_warm = model_cpu(x_in, doy_pair, steps=steps, perturbation=torch.from_numpy(p_warm)).numpy()
        cw = S.causality_checks(base, out_warm, C.BOX_BAY_OF_BENGAL, std_arr)
        cw["t0_date"] = int(val_dates[st + C.T_IN - 1])
        warm_results.append(cw)

        p_stall = S.make_shock("monsoon_stall", last_np, std_arr)
        out_stall = model_cpu(x_in, doy_pair, steps=steps, perturbation=torch.from_numpy(p_stall)).numpy()
        cs = S.causality_checks(base, out_stall, C.BOX_ARABIAN_SEA, std_arr)
        cs["t0_date"] = int(val_dates[st + C.T_IN - 1])
        stall_results.append(cs)

    model.to(orig_dev)

    # Aggregate pass counts and representative score on each window
    window_causality_pts = []
    for cw, cs, nd_norm in zip(warm_results, stall_results, null_diffs_norm):
        null_ok = nd_norm <= S.NULL_TOL
        window_causality_pts.append(S.causality_points([cw, cs], null_ok=null_ok, max_pts=15.0))

    return {
        "cpu_4thread_latency_ms": {
            "median": float(np.median(lat_ms)),
            "p95": float(np.percentile(lat_ms, 95)),
            "min": float(np.min(lat_ms)),
            "max": float(np.max(lat_ms)),
        },
        "zero_shock": {
            "max_abs_diff": float(max(null_diffs_abs)),
            "max_norm_diff_std": float(max(null_diffs_norm)),
            "passes_1e_6_atol": bool(max(null_diffs_abs) < 1e-6),
            "passes_official_null_tol": bool(max(null_diffs_norm) <= S.NULL_TOL),
        },
        "warm_ocean_pass_count": int(sum(all(r[k] for k in ("finite", "nonzero", "localized", "propagates", "stable")) for r in warm_results)),
        "monsoon_stall_pass_count": int(sum(all(r[k] for k in ("finite", "nonzero", "localized", "propagates", "stable")) for r in stall_results)),
        "num_probe_windows": len(probe_starts),
        "mean_causality_pts_out_of_15": float(np.mean(window_causality_pts)),
        "warm_ocean_probes": warm_results,
        "monsoon_stall_probes": stall_results,
    }


def evaluate_model_full(
    model: BaseMonsoonForecaster,
    val_npz: Path = Path("data/raw/splits/val.npz"),
    static_npy: Path = Path("data/raw/splits/static.npy"),
    device: str = "cuda",
    run_shocks: bool = True,
) -> dict[str, Any]:
    """Run complete Phase 5 evaluation on 2019-2020 validation split."""
    static_arr = np.load(static_npy).astype(np.float32)
    y_pred, roll_data = predict_validation_rollouts(model, val_npz=val_npz, device=device)
    y_true = roll_data["y_true"]
    y_persist = roll_data["y_persist"]
    y_clim = roll_data["y_clim"]

    # 1. Validity gate across all 719 windows
    invalid_windows = []
    for w in range(len(y_pred)):
        bad = S.validity(y_pred[w], C.SCORE_STEPS)
        if bad:
            invalid_windows.append({"window": w, "issues": bad})

    # 2. Detailed per-lead, per-channel, regional, intensity breakdown (reuses Phase 2 evaluator)
    detailed = evaluate_baseline_predictions(y_pred, y_true, y_clim, static_arr)

    # 3. Official scoring metrics vs min(Persistence, Climatology)
    ss_per_lead: dict[str, Any] = {}
    ss_all_leads = []
    csi_ss_all_leads = []
    fss_by_lead = {}

    for day in C.LEAD_DAYS:
        k = day - 1
        sk = S.skill_vs_best_ref(y_pred[:, k], y_true[:, k], y_persist[:, k], y_clim[:, k])
        ss_ch = {ch: float(v) for ch, v in zip(C.CHANNELS, sk["ss"])}
        ss_mean = float(np.nanmean(sk["ss"]))
        ss_all_leads.append(ss_mean)

        csi_m = S.csi(y_pred[:, k, 1], y_true[:, k, 1], thr=S.RAIN_THRESH)
        csi_p = S.csi(y_persist[:, k, 1], y_true[:, k, 1], thr=S.RAIN_THRESH)
        csi_c = S.csi(y_clim[:, k, 1], y_true[:, k, 1], thr=S.RAIN_THRESH)
        csi_ss = S.csi_skill(csi_m, csi_p, csi_c)
        csi_ss_all_leads.append(csi_ss)

        fss3 = _fss_2d(y_pred[:, k, 1], y_true[:, k, 1], thr=10.0, size=3)
        fss5 = _fss_2d(y_pred[:, k, 1], y_true[:, k, 1], thr=10.0, size=5)
        fss_by_lead[f"day_{day}"] = {"fss_3x3": fss3, "fss_5x5": fss5}

        ss_per_lead[f"day_{day}"] = {
            "rmse_skill_per_channel": ss_ch,
            "rmse_skill_mean": ss_mean,
            "rmse_skill_vs_persist_mean": float(np.nanmean(sk["ss_persist"])),
            "rmse_skill_vs_clim_mean": float(np.nanmean(sk["ss_clim"])),
            "csi_10mm_model": float(csi_m),
            "csi_10mm_persist": float(csi_p),
            "csi_10mm_clim": float(csi_c),
            "csi_skill": float(csi_ss),
        }

    # Day-4 persistence gate
    d4_pred_rmse = float(np.mean(S.rmse_lat(y_pred[:, 3], y_true[:, 3])))
    d4_pers_rmse = float(np.mean(S.rmse_lat(y_persist[:, 3], y_true[:, 3])))
    d4_pred_norm_rmse = float(np.mean(S.rmse_lat(y_pred[:, 3], y_true[:, 3]) / model.stats_std.detach().cpu().numpy()))
    d4_pers_norm_rmse = float(np.mean(S.rmse_lat(y_persist[:, 3], y_true[:, 3]) / model.stats_std.detach().cpu().numpy()))

    # Day-8 anti-blur check via official monsoon_r1.scoring.anti_blur_points
    ratios_d8 = np.asarray([S.variance_ratio(y_pred[w, 7], y_true[w, 7]) for w in range(len(y_pred))])
    anti_blur_pts = S.anti_blur_points(ratios_d8, max_pts=10.0)
    mean_d8_vr = {ch: float(np.nanmean(ratios_d8[:, i])) for i, ch in enumerate(C.CHANNELS)}

    # Forecast skill points
    fcst_pts = S.forecast_points(ss_all_leads, csi_ss_all_leads)

    # Day-8 spectral ratio
    fft_d8 = _fft_band_ratios(y_pred[:, 7], y_true[:, 7])

    shocks = (
        evaluate_shocks_and_cpu_latency(model, val_npz=val_npz, stats_std=model.stats_std.detach().cpu().numpy())
        if run_shocks
        else None
    )

    causality_pts = shocks["mean_causality_pts_out_of_15"] if shocks else 0.0
    automated_70_score = fcst_pts["points"] + anti_blur_pts + causality_pts

    return {
        "split": "validation_2019_2020",
        "num_windows": int(len(y_pred)),
        "model_config": model.model_config(),
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "physical_validity": {
            "all_windows_valid": len(invalid_windows) == 0,
            "invalid_count": len(invalid_windows),
            "examples": invalid_windows[:5],
        },
        "hard_gate_day4_persistence": {
            "beats_day4_raw_mean_rmse": bool(d4_pred_rmse < d4_pers_rmse),
            "beats_day4_norm_mean_rmse": bool(d4_pred_norm_rmse < d4_pers_norm_rmse),
            "day4_pred_mean_rmse": d4_pred_rmse,
            "day4_persist_mean_rmse": d4_pers_rmse,
            "day4_pred_norm_rmse": d4_pred_norm_rmse,
            "day4_persist_norm_rmse": d4_pers_norm_rmse,
        },
        "official_scoring_summary": {
            "mean_rmse_ss_148": fcst_pts["mean_ss"],
            "mean_csi_ss_148": fcst_pts["mean_csi_skill"],
            "rmse_pts_out_of_30": fcst_pts["rmse_pts"],
            "csi_pts_out_of_15": fcst_pts["csi_pts"],
            "forecast_pts_out_of_45": fcst_pts["points"],
            "anti_blur_pts_out_of_10": anti_blur_pts,
            "mean_day8_variance_ratios": mean_d8_vr,
            "causality_pts_out_of_15": causality_pts,
            "automated_model_score_out_of_70": float(automated_70_score),
            "per_lead": ss_per_lead,
        },
        "fss_rainfall": fss_by_lead,
        "day8_fft_power_ratios": fft_d8,
        "shocks_and_cpu": shocks,
        "detailed": detailed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoint on 2019-2020 validation split.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, _ = load_model_from_checkpoint(args.ckpt, map_location=args.device)
    rep = evaluate_model_full(model, device=args.device, run_shocks=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    s = rep["official_scoring_summary"]
    print(f"[eval] {args.ckpt} | valid={rep['physical_validity']['all_windows_valid']} | "
          f"mean_SS={s['mean_rmse_ss_148']:.4f} ({s['rmse_pts_out_of_30']:.2f}/30) | "
          f"CSI_SS={s['mean_csi_ss_148']:.4f} ({s['csi_pts_out_of_15']:.2f}/15) | "
          f"AntiBlur={s['anti_blur_pts_out_of_10']:.2f}/10 | "
          f"Causality={s['causality_pts_out_of_15']:.2f}/15 | "
          f"Total70={s['automated_model_score_out_of_70']:.2f}/70")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
