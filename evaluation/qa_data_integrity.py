"""Phase 2: independent data-integrity and leakage audit.

Checks the raw archive and the split files against the kit contract, recomputes the training-only statistics and
climatology WITH THE KIT'S OWN FUNCTIONS (monsoon_r1.stats), and compares them with what is shipped in checkpoints/.
The 2021 test split is only checked for shape, continuity, date isolation and identity with the raw archive.
Run:  .\\.venv\\Scripts\\python.exe evaluation\\qa_data_integrity.py
"""
from __future__ import annotations

import json
import re

import numpy as np

from qa_common import C, KS, REPO_ROOT, SPLITS_DIR, CKPT_DIR, EVAL_DIR, load_split, to_date, window_starts, write_json

SPLIT_POLICY = {"train": (20050101, 20181231), "val": (20190101, 20201231), "test": (20210101, 20211231)}


def nc_audit() -> tuple[dict, dict]:
    import xarray as xr
    ds = xr.open_dataset(REPO_ROOT / "data" / "raw" / "india_climate_train.nc")
    data = ds["data"].values.astype("f4")
    t = ds["time"].values.astype("datetime64[D]")
    dates = np.array([int(str(x).replace("-", "")) for x in t])
    gaps = np.diff(t).astype(int)
    rep = {
        "dims": dict(ds.sizes),
        "channel_coord": [str(x) for x in ds["channel"].values],
        "channel_order_matches_contract": [str(x) for x in ds["channel"].values] == list(C.CHANNELS),
        "lat_matches_contract": bool(np.allclose(ds["lat"].values, C.LAT)),
        "lon_matches_contract": bool(np.allclose(ds["lon"].values, C.LON)),
        "lat_south_to_north": bool(np.all(np.diff(ds["lat"].values) > 0)),
        "units_attr": ds["data"].attrs.get("units"),
        "time_range": [str(t[0]), str(t[-1])], "n_days": int(len(t)),
        "time_daily_continuous": bool(np.all(gaps == 1)), "n_time_gaps": int((gaps != 1).sum()),
        "duplicate_dates": int(len(t) - len(np.unique(t))),
        "nan_count_per_channel": {c: int(np.isnan(data[:, i]).sum()) for i, c in enumerate(C.CHANNELS)},
        "inf_count_per_channel": {c: int(np.isinf(data[:, i]).sum()) for i, c in enumerate(C.CHANNELS)},
        "range_per_channel": {c: [float(np.nanmin(data[:, i])), float(np.nanmax(data[:, i]))] for i, c in enumerate(C.CHANNELS)},
        "tp_negative_cells": int((data[:, 1] < 0).sum()),
        "tcwv_negative_cells": int((data[:, 5] < 0).sum()),
        "t2m_outside_validity_cells": int(((data[:, 0] < 233.15) | (data[:, 0] > 333.15)).sum()),
        "static_channels": [str(x) for x in ds["static_channel"].values],
    }
    st = ds["static"].values.astype("f4")
    rep["static_shape"] = list(st.shape)
    rep["lsm_range"] = [float(st[0].min()), float(st[0].max())]
    rep["orography_range_m"] = [float(st[1].min()), float(st[1].max())]
    rep["static_npy_identical"] = {p: bool(np.array_equal(np.load(d / "static.npy"), st))
                                  for p, d in (("checkpoints", CKPT_DIR), ("splits", SPLITS_DIR))}
    # plausibility of units: t2m mean ~ 290 K (K not C), msl ~ 1e5 Pa (not hPa), tp mm/day (mean few mm)
    means = data.mean(axis=(0, 2, 3))
    rep["unit_plausibility"] = {"t2m_mean_K": float(means[0]), "tp_mean_mm_day": float(means[1]),
                                "msl_mean_Pa": float(means[4]), "tcwv_mean": float(means[5]),
                                "ok": bool(250 < means[0] < 320 and 0 < means[1] < 20 and 9e4 < means[4] < 1.1e5 and 5 < means[5] < 80)}
    ds.close()
    return rep, {"data": data, "dates": dates}


def split_audit(raw: dict) -> dict:
    out, all_dates = {}, {}
    rdates = raw["dates"]
    for name, (lo, hi) in SPLIT_POLICY.items():
        data, dates, doy = load_split(name)
        ds = [to_date(d) for d in dates]
        diffs = np.array([(b - a).days for a, b in zip(ds[:-1], ds[1:])])
        idx = np.searchsorted(rdates, dates)
        same = bool(np.all(rdates[idx] == dates) and np.array_equal(raw["data"][idx], data))
        cal_doy = np.array([d.timetuple().tm_yday for d in ds])
        out[name] = {
            "shape": list(data.shape), "dtype": str(data.dtype),
            "date_range": [int(dates.min()), int(dates.max())], "policy": [lo, hi],
            "within_policy": bool(dates.min() >= lo and dates.max() <= hi),
            "covers_policy_fully": bool(dates.min() == lo and dates.max() == hi),
            "sorted": bool(np.all(np.diff(dates) > 0)), "continuous_daily": bool(np.all(diffs == 1)),
            "n_finite_fraction": float(np.isfinite(data).mean()),
            "bit_identical_to_raw_nc": same,
            "doy_matches_calendar": bool(np.array_equal(cal_doy, doy)),
            "n_windows_13d": len(window_starts(dates)),
        }
        all_dates[name] = set(int(x) for x in dates)
    out["pairwise_overlap"] = {f"{a}&{b}": len(all_dates[a] & all_dates[b])
                               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    out["union_equals_raw"] = (all_dates["train"] | all_dates["val"] | all_dates["test"]) == set(int(x) for x in rdates)
    # a window that starts in one split and ends in the next would leak; windows are built per split so check that
    trd = load_split("train")[1]; vad = load_split("val")[1]
    out["max_train_window_end"] = int(max(trd[i + 12] for i in window_starts(trd)))
    out["min_val_window_start"] = int(min(vad[i] for i in window_starts(vad)))
    return out


def stats_audit(raw: dict) -> dict:
    tr, tr_dates, _ = load_split("train")
    va, _, _ = load_split("val")
    te, _, _ = load_split("test")
    ref_train = KS.channel_stats(tr)
    ref_train64 = KS.channel_stats(tr.astype("f8"))
    candidates = {"train_only_kit_f4": ref_train, "train_only_f8": ref_train64,
                  "train_val": KS.channel_stats(np.concatenate([tr, va])),
                  "all_2005_2021": KS.channel_stats(raw["data"])}
    del te
    shipped = {"checkpoints/stats.json": json.loads((CKPT_DIR / "stats.json").read_text()),
               "splits/train_stats.json": json.loads((SPLITS_DIR / "train_stats.json").read_text())}
    res = {"recomputed": candidates, "comparison": {}}
    for sname, s in shipped.items():
        cmp = {}
        for cname, c in candidates.items():
            dm = np.abs(np.array(s["mean"]) - np.array(c["mean"])) / np.array(c["std"])
            ds_ = np.abs(np.array(s["std"]) - np.array(c["std"])) / np.array(c["std"])
            cmp[cname] = {"max_abs_mean_diff_in_std": float(dm.max()), "max_rel_std_diff": float(ds_.max())}
        best = min(cmp, key=lambda k: cmp[k]["max_abs_mean_diff_in_std"] + cmp[k]["max_rel_std_diff"])
        res["comparison"][sname] = {"vs": cmp, "closest_source": best,
                                    "channels_match": s.get("channels") == list(C.CHANNELS)}
    # climatology: recompute with the kit on TRAIN only and compare with shipped files
    clim_ref = KS.climatology(tr, tr_dates)
    for p in (CKPT_DIR / "climatology.npy", SPLITS_DIR / "train_climatology.npy"):
        cl = np.load(p)
        res[f"climatology:{p.relative_to(REPO_ROOT).as_posix()}"] = {
            "shape": list(cl.shape), "bit_identical_to_kit_train_only": bool(np.array_equal(cl, clim_ref)),
            "max_abs_diff_per_channel": [float(x) for x in np.abs(cl - clim_ref).max(axis=(0, 2, 3))]}
    # discriminating power: how far is the train-only climatology from a train+val one (shows the check can detect leakage)
    va_dates = load_split("val")[1]
    clim_tv = KS.climatology(np.concatenate([tr, va]), np.concatenate([tr_dates, va_dates]))
    res["leak_detectability_train_vs_trainval_clim_maxdiff"] = [float(x) for x in np.abs(clim_tv - clim_ref).max(axis=(0, 2, 3))]
    return res


def code_leakage_audit() -> dict:
    """Static: which code reads which split, and does any inference/model code touch data files or the test split."""
    hits = {}
    pat = re.compile(r"test\.npz|['\"]test['\"]|val\.npz|train\.npz|india_climate_train\.nc|20(19|20|21)\d{4}")
    for d in ("training", "models", "inference", "data", "scripts"):
        for p in (REPO_ROOT / d).rglob("*.py") if (REPO_ROOT / d).is_dir() else []:
            if "__pycache__" in p.parts:
                continue
            lines = [(i + 1, l.strip()[:140]) for i, l in enumerate(p.read_text(errors="ignore").splitlines()) if pat.search(l)]
            if lines:
                hits[p.relative_to(REPO_ROOT).as_posix()] = lines[:25]
    for p in [REPO_ROOT / "serve.py"]:
        lines = [(i + 1, l.strip()[:140]) for i, l in enumerate(p.read_text(errors="ignore").splitlines()) if pat.search(l)]
        if lines:
            hits["serve.py"] = lines
    return {"references": hits,
            "inference_touches_data_files": any(k.startswith("inference/") or k == "serve.py" for k in hits)}


def main():
    nc, raw = nc_audit()
    print("nc:", json.dumps({k: nc[k] for k in ("dims", "channel_order_matches_contract", "time_daily_continuous",
                                                  "nan_count_per_channel", "unit_plausibility")}, default=str))
    sp = split_audit(raw)
    print("splits:", json.dumps({k: v for k, v in sp.items() if not isinstance(v, dict) or k == "pairwise_overlap"}, default=str))
    for k in ("train", "val", "test"):
        print(" ", k, sp[k])
    stt = stats_audit(raw)
    for k, v in stt["comparison"].items():
        print("stats", k, "closest:", v["closest_source"], {kk: vv for kk, vv in v["vs"].items()})
    for k, v in stt.items():
        if k.startswith("climatology"):
            print(k, v)
    print("leak detectability (train vs train+val clim max diff):", stt["leak_detectability_train_vs_trainval_clim_maxdiff"])
    code = code_leakage_audit()
    print("code refs:", json.dumps(code, indent=1)[:4000])
    write_json(EVAL_DIR / "data_integrity_report.json", {"netcdf": nc, "splits": sp, "stats": stt, "code_leakage": code,
                                                         "note": "2021 test split: shape/continuity/isolation only; no metrics computed."})


if __name__ == "__main__":
    main()
