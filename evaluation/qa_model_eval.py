"""Phase 3: model evaluation on the 2019-2020 VALIDATION split (never the sealed 2021 test split).

Candidates:
  persistence  - init_states[-1] repeated (reference)
  climatology  - checkpoints/climatology.npy (train-only, kit stats.climatology) at the target days (reference)
  served       - whatever `python serve.py` currently returns, called black-box through the kit client (optional, --server)
  trained      - any checkpoints/*.pt; evaluated ONLY through the server. If none exist the comparison is BLOCKED.

Kit metrics: scoring.rmse_lat, csi, csi_skill, skill_vs_best_ref, variance_ratio, anti_blur_points, validity, forecast_points.
QA-only metrics (not in the kit, clearly labelled): lat-weighted MAE and bias, pattern correlation, anomaly correlation (ACC).
Run:  .\\.venv\\Scripts\\python.exe evaluation\\qa_model_eval.py [--server] [--server-stride 7] [--port 8023]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from qa_common import (C, S, KS, CKPT_DIR, EVAL_DIR, ServerProc, climatology_forecast, doy_sincos_for, load_split,
                       load_std, persistence, simulate, to_date, window_starts, write_json, ymd)

W_LAT = C.lat_weights()[:, None]


def lat_mae(p, t):
    return (np.abs(p - t) * W_LAT).mean(axis=(-2, -1)).reshape(-1, C.NC).mean(axis=0)


def lat_bias(p, t):
    return ((p - t) * W_LAT).mean(axis=(-2, -1)).reshape(-1, C.NC).mean(axis=0)


def _corr(a, b):
    """Lat-weighted centred spatial correlation per [W,6] over the 32x36 grid."""
    w = np.broadcast_to(W_LAT, a.shape[-2:])
    am = a - (a * w).sum((-2, -1), keepdims=True) / w.sum(); bm = b - (b * w).sum((-2, -1), keepdims=True) / w.sum()
    num = (am * bm * w).sum((-2, -1)); den = np.sqrt((am ** 2 * w).sum((-2, -1)) * (bm ** 2 * w).sum((-2, -1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def lead_metrics(pred, true, pers, clim, std):
    """pred/true/pers/clim: [W,6,32,36] at one lead."""
    sk = S.skill_vs_best_ref(pred, true, pers, clim)
    vr = np.stack([S.variance_ratio(pred[w], true[w]) for w in range(len(pred))])
    pc = _corr(pred, true); acc = _corr(pred - clim, true - clim)
    z = np.abs((pred - clim) / std[None, :, None, None])
    return {
        "rmse_lat": dict(zip(C.CHANNELS, sk["rmse"])),
        "ss_vs_best_ref": dict(zip(C.CHANNELS, sk["ss"])),
        "ss_vs_persistence": dict(zip(C.CHANNELS, sk["ss_persist"])),
        "ss_vs_climatology": dict(zip(C.CHANNELS, sk["ss_clim"])),
        "mae_lat_qa": dict(zip(C.CHANNELS, lat_mae(pred, true))),
        "bias_lat_qa": dict(zip(C.CHANNELS, lat_bias(pred, true))),
        "pattern_corr_qa": dict(zip(C.CHANNELS, np.nanmean(pc, 0))),
        "acc_vs_clim_qa": dict(zip(C.CHANNELS, [float(np.nanmean(pc_)) if np.isfinite(pc_).any() else float("nan") for pc_ in acc.T])),
        "csi_10mm": S.csi(pred[:, 1], true[:, 1]),
        "variance_ratio_mean": dict(zip(C.CHANNELS, np.nanmean(vr, 0))),
        "max_abs_anomaly_std": float(np.nanmax(z)) if np.isfinite(z).any() else float("nan"),
        "nonfinite_frac": float(1 - np.isfinite(pred).mean()),
    }


def evaluate(name, get_pred, truth_at, pers_at, clim_at, n_windows, std):
    """get_pred(k) -> [W,6,32,36]. Returns per-lead metrics, validity, day-8 anti-blur, emulated forecast points."""
    t0 = time.perf_counter()
    leads, preds_by_lead = {}, {}
    for k in range(C.SCORE_STEPS):
        p = get_pred(k); preds_by_lead[k] = p
        leads[f"day_{k + 1}"] = lead_metrics(p, truth_at(k), pers_at(k), clim_at(k), std)
    # validity per window across all 10 steps (kit gate)
    viol, n_bad = {}, 0
    for w in range(n_windows):
        roll = np.stack([preds_by_lead[k][w] for k in range(C.SCORE_STEPS)])
        bad = S.validity(roll, C.SCORE_STEPS); n_bad += bool(bad)
        for v in bad:
            viol[v] = viol.get(v, 0) + 1
    # day-8 anti-blur with the kit's own points function
    k8 = 7
    ratios = np.stack([S.variance_ratio(preds_by_lead[k8][w], truth_at(k8)[w]) for w in range(n_windows)])
    # emulated forecast points (the grader's per-window aggregation is hidden; this is the closest kit-only reading)
    ss_w, csi_w = [], []
    for w in range(n_windows):
        ss_l, pm, tm, pp, cc = [], [], [], [], []
        for d in C.LEAD_DAYS:
            k = d - 1
            sk = S.skill_vs_best_ref(preds_by_lead[k][w], truth_at(k)[w], pers_at(k)[w], clim_at(k)[w])
            ss_l.append(np.nanmean(sk["ss"]))
            pm.append(preds_by_lead[k][w, 1]); tm.append(truth_at(k)[w, 1]); pp.append(pers_at(k)[w, 1]); cc.append(clim_at(k)[w, 1])
        ss_w.append(float(np.nanmean(ss_l)))
        pm, tm, pp, cc = map(np.stack, (pm, tm, pp, cc))
        csi_w.append(S.csi_skill(S.csi(pm, tm), S.csi(pp, tm), S.csi(cc, tm)))
    fp = S.forecast_points(ss_w, csi_w)
    # autoregressive instability: growth of the error and of the max anomaly over lead time
    rm = [float(np.mean(list(leads[f"day_{k + 1}"]["rmse_lat"].values()) / std)) for k in range(C.SCORE_STEPS)]
    d4 = leads["day_4"]
    return {
        "candidate": name, "n_windows": n_windows, "eval_s": time.perf_counter() - t0,
        "by_lead": leads,
        "validity": {"windows_with_violation": n_bad, "violation_counts": viol},
        "day8_variance_ratio_mean": dict(zip(C.CHANNELS, np.nanmean(ratios, 0))),
        "day8_anti_blur_points_of_10": S.anti_blur_points(ratios),
        "beats_persistence_day4": {"mean_ss_vs_persistence": float(np.nanmean(list(d4["ss_vs_persistence"].values()))),
                                   "channels_better": [c for c, v in d4["ss_vs_persistence"].items() if v > 0]},
        "emulated_forecast_points_of_45": fp,
        "instability": {"mean_norm_rmse_by_lead": rm,
                        "max_abs_anomaly_std_by_lead": [leads[f"day_{k + 1}"]["max_abs_anomaly_std"] for k in range(C.SCORE_STEPS)],
                        "nonfinite_any": any(leads[f"day_{k + 1}"]["nonfinite_frac"] > 0 for k in range(C.SCORE_STEPS))},
    }


def eval_checkpoint(path, data, dates, starts, last_dates, make, std) -> dict:
    """Offline CPU evaluation of one trained checkpoint (read-only use of models.load_model_from_checkpoint).
    It is NOT what the server returns; serving is checked separately via --server."""
    import hashlib, sys
    import torch
    from qa_common import REPO_ROOT
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from models import load_model_from_checkpoint
    torch.set_num_threads(4)
    raw = path.read_bytes()
    meta = {"path": path.relative_to(CKPT_DIR.parent).as_posix(), "bytes": len(raw), "sha256_16": hashlib.sha256(raw).hexdigest()[:16],
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))}
    model, info = load_model_from_checkpoint(path, map_location="cpu")
    model.eval()
    meta["checkpoint_keys"] = sorted(info.keys()) if isinstance(info, dict) else str(type(info))
    meta["params"] = int(sum(p.numel() for p in model.parameters()))
    st, truth, pers, cl = make(starts)
    preds = np.empty((len(st), C.SCORE_STEPS, C.NC, C.NLAT, C.NLON), "f4")
    with torch.no_grad():
        for b0 in range(0, len(st), 32):
            idx = st[b0:b0 + 32]
            x = torch.from_numpy(np.stack([data[i:i + C.T_IN] for i in idx]))
            d = torch.tensor([doy_sincos_for(last_dates[list(starts).index(i)]) for i in idx], dtype=torch.float32)
            preds[b0:b0 + len(idx)] = model(x, d, steps=C.SCORE_STEPS).numpy()
    res = evaluate(meta["path"], lambda k: preds[:, k], truth, pers, cl, len(st), std)
    res["checkpoint"] = meta
    # CPU latency, single request, 4 threads
    x0 = torch.from_numpy(data[st[180]:st[180] + C.T_IN].copy()); d0 = torch.tensor(doy_sincos_for(last_dates[180]))
    lat = []
    with torch.no_grad():
        model(x0, d0, steps=10)
        for _ in range(10):
            t0 = time.perf_counter(); model(x0, d0, steps=10); lat.append(time.perf_counter() - t0)
    res["cpu4_latency_s"] = {"max": max(lat), "median": float(np.median(lat))}
    # shocks with kit definitions on 3 val windows
    sh = []
    with torch.no_grad():
        for wd in (ymd(2019, 7, 15), ymd(2020, 8, 10), ymd(2019, 1, 15)):
            i_last = int(np.where(dates == wd)[0][0]); init = data[i_last - 2:i_last + 1]
            x = torch.from_numpy(init.copy()); dd = torch.tensor(doy_sincos_for(to_date(wd)))
            base = model(x, dd, steps=10).numpy()
            zero = model(x, dd, steps=10, perturbation=torch.zeros(C.NC, C.NLAT, C.NLON)).numpy()
            rec = {"date": wd, "null_consistent": S.null_consistent(base, zero, std), "null_max_abs": float(np.abs(base - zero).max())}
            ccs = []
            for name, box in (("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA)):
                p = torch.from_numpy(S.make_shock(name, init[-1], std))
                cc = S.causality_checks(base, model(x, dd, steps=10, perturbation=p).numpy(), box, std)
                rec[name] = cc; ccs.append(cc)
            rec["causality_points_of_15"] = S.causality_points(ccs, rec["null_consistent"])
            sh.append(rec)
    res["shocks_offline"] = sh
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", action="store_true", help="also evaluate the currently served model (black-box)")
    ap.add_argument("--server-stride", type=int, default=7)
    ap.add_argument("--port", type=int, default=8023)
    ap.add_argument("--checkpoint", action="append", default=[], help="trained checkpoint(s) under checkpoints/ to evaluate offline on CPU")
    ap.add_argument("--checkpoints-only", action="store_true", help="reuse baselines in model_eval_val.json; only add checkpoint results")
    a = ap.parse_args()
    if a.checkpoints_only:
        import json
        from pathlib import Path
        std = load_std()
        data, dates, _ = load_split("val")
        clim = np.load(CKPT_DIR / "climatology.npy")
        starts = np.array(window_starts(dates))
        last_dates = [to_date(dates[i + C.T_IN - 1]) for i in starts]
        tgt_doy = np.array([[KS._doy_index(to_date(dates[i + C.T_IN + k])) for k in range(C.SCORE_STEPS)] for i in starts])

        def make(idx):
            sel = np.isin(starts, idx); st = starts[sel]; td = tgt_doy[sel]
            return st, (lambda k: data[st + C.T_IN + k]), (lambda k: data[st + C.T_IN - 1]), (lambda k: clim[td[:, k]])
        rp = EVAL_DIR / "model_eval_val.json"
        report = json.loads(rp.read_text())
        report.setdefault("trained_offline", {})
        for c in a.checkpoint:
            r = eval_checkpoint(Path(c).resolve(), data, dates, starts, last_dates, make, std)
            report["trained_offline"][r["checkpoint"]["path"]] = r
            d = r["by_lead"]
            print(r["checkpoint"]["path"], r["checkpoint"]["mtime"], "params", r["checkpoint"]["params"])
            for day in (1, 4, 8):
                print(f"  day{day} rmse", {k: round(v, 3) for k, v in d[f'day_{day}']['rmse_lat'].items()},
                      "ss_best", round(float(np.nanmean(list(d[f'day_{day}']['ss_vs_best_ref'].values()))), 4),
                      "csi", round(d[f'day_{day}']['csi_10mm'], 4))
            print("  day4 vs persistence", r["beats_persistence_day4"], "anti-blur", round(r["day8_anti_blur_points_of_10"], 2),
                  "vr8", {k: round(v, 3) for k, v in r["day8_variance_ratio_mean"].items()})
            print("  validity", r["validity"], "points", r["emulated_forecast_points_of_45"], "cpu", r["cpu4_latency_s"])
            for s in r["shocks_offline"]:
                print("  shock", s["date"], "null", s["null_consistent"], s["null_max_abs"], "pts", s["causality_points_of_15"],
                      {n: {k: s[n][k] for k in ("nonzero", "localized", "propagates", "stable")} | {"near1": round(s[n]["near_frac_day1"], 3), "far": round(s[n]["far_frac_max"], 3)} for n in ("warm_ocean", "monsoon_stall")})
        if report.get("trained", {}).get("status") == "BLOCKED" and report["trained_offline"]:
            report["trained"] = {"status": "EVALUATED_OFFLINE_ONLY",
                                 "reason": "run checkpoints exist under checkpoints/runs/ but no checkpoints/model.pt is served; "
                                           "server still returns persistence"}
        write_json(rp, report)
        return

    std = load_std()
    data, dates, _ = load_split("val")
    clim = np.load(CKPT_DIR / "climatology.npy")
    starts = np.array(window_starts(dates))
    last_dates = [to_date(dates[i + C.T_IN - 1]) for i in starts]
    tgt_doy = np.array([[KS._doy_index(to_date(dates[i + C.T_IN + k])) for k in range(C.SCORE_STEPS)] for i in starts])

    def make(idx):
        sel = np.isin(starts, idx); st = starts[sel]; td = tgt_doy[sel]
        truth = lambda k: data[st + C.T_IN + k]
        pers = lambda k: data[st + C.T_IN - 1]
        cl = lambda k: clim[td[:, k]]
        return st, truth, pers, cl

    report = {"split": "val 2019-2020", "n_windows_total": int(len(starts)),
              "climatology_source": "checkpoints/climatology.npy (bit-identical to kit stats.climatology on 2005-2018 train)",
              "std_source": "checkpoints/stats.json", "test_split": "NOT USED (sealed)"}
    st, truth, pers, cl = make(starts)
    report["persistence"] = evaluate("persistence", pers, truth, pers, cl, len(st), std)
    report["climatology"] = evaluate("climatology", cl, truth, pers, cl, len(st), std)
    for n in ("persistence", "climatology"):
        r = report[n]
        print(n, "d1/d4/d8 rmse t2m,tp:", [(round(r["by_lead"][f"day_{d}"]["rmse_lat"]["t2m"], 3), round(r["by_lead"][f"day_{d}"]["rmse_lat"]["tp"], 3)) for d in (1, 4, 8)],
              "csi:", [round(r["by_lead"][f"day_{d}"]["csi_10mm"], 4) for d in (1, 4, 8)], "anti-blur:", round(r["day8_anti_blur_points_of_10"], 2),
              "validity:", r["validity"])

    ckpts = sorted(p.relative_to(CKPT_DIR).as_posix() for p in CKPT_DIR.rglob("*") if p.suffix in (".pt", ".pth", ".ckpt", ".safetensors", ".onnx"))
    report["checkpoint_inventory"] = {"weights_found": ckpts, "all_files": sorted(p.relative_to(CKPT_DIR).as_posix() for p in CKPT_DIR.rglob("*") if p.is_file())}
    if not ckpts:
        report["trained"] = {"status": "BLOCKED", "reason": "no trained weights in checkpoints/ (no model.pt); nothing to evaluate"}
    print("checkpoints:", report["checkpoint_inventory"])

    if a.server:
        sub = starts[::a.server_stride]
        st, truth, pers, cl = make(sub)
        with ServerProc(a.port, ncores=4) as srv:
            import requests
            health = requests.post(srv.url + "/health", timeout=5).json()
            outs, walls, errs = [], [], []
            for i, d in zip(st, [last_dates[list(starts).index(i)] for i in st]):
                r = simulate(srv.url, data[i:i + C.T_IN], doy_sincos_for(d))
                walls.append(r.wall_s)
                if not r.ok:
                    errs.append(r.error); outs.append(np.full((C.SCORE_STEPS, C.NC, C.NLAT, C.NLON), np.nan, "f4"))
                else:
                    outs.append(r.out)
            outs = np.stack(outs)
        res = evaluate("served", lambda k: outs[:, k], truth, pers, cl, len(st), std)
        pers_full = np.stack([persistence(data[i:i + C.T_IN]) for i in st])
        res["health"] = health
        res["http_errors"] = errs
        res["wall_s"] = {"max": float(max(walls)), "median": float(np.median(walls))}
        res["max_abs_diff_vs_persistence"] = float(np.nanmax(np.abs(outs - pers_full)))
        res["identical_to_persistence"] = bool(np.array_equal(outs, pers_full))
        # same windows for the references so the comparison is like-for-like
        res["refs_on_same_windows"] = {"persistence": evaluate("persistence", pers, truth, pers, cl, len(st), std)["emulated_forecast_points_of_45"],
                                       "climatology": evaluate("climatology", cl, truth, pers, cl, len(st), std)["emulated_forecast_points_of_45"]}
        report["served"] = res
        print("served:", health, "windows", len(st), "identical_to_persistence", res["identical_to_persistence"],
              "maxdiff", res["max_abs_diff_vs_persistence"], "errors", len(errs), "wall", res["wall_s"], "validity", res["validity"])
    write_json(EVAL_DIR / "model_eval_val.json", report)


if __name__ == "__main__":
    main()
