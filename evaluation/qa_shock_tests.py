"""Phase 4: shock / perturbation tests against the RUNNING backend, using only kit definitions.

Official shocks: scoring.make_shock('warm_ocean' | 'monsoon_stall' | 'null'). Custom localized shocks are built from
contract.box_mask. Every response is judged with scoring.causality_checks / null_consistent / causality_points and
the std from checkpoints/stats.json (what the grader substitutes for STD_DEFAULT).
Init states are real 2019-2020 validation windows (never the 2021 test split).
Run:  .\\.venv\\Scripts\\python.exe evaluation\\qa_shock_tests.py [--url http://127.0.0.1:8022]   (else starts its own server)
"""
from __future__ import annotations

import argparse
import contextlib

import numpy as np

from qa_common import (C, S, EVAL_DIR, ROUTE_SIM, ServerProc, doy_sincos_for, load_split, load_std, simulate, to_date,
                       write_json, ymd)

# (name, box (lat0, lat1, lon0, lon1), channel, delta in physical units) - built with the kit's box_mask
CUSTOM = [
    ("central_india_heat_+3K", (20.0, 24.0, 76.0, 80.0), 0, 3.0),
    ("western_ghats_rain_+20mm", (12.0, 18.0, 72.0, 76.0), 1, 20.0),
    ("bay_low_-400Pa", (16.0, 20.0, 86.0, 90.0), 4, -400.0),
    ("single_cell_tcwv_+8", (22.0, 23.0, 80.0, 81.0), 5, 8.0),
]
WINDOW_DATES = tuple(ymd(*t) for t in ((2019, 1, 15), (2019, 4, 20), (2019, 7, 15), (2020, 8, 10), (2020, 10, 15)))


def custom_delta(box, ch, val):
    d = np.zeros((C.NC, C.NLAT, C.NLON), "f4"); d[ch][C.box_mask(box)] = val
    return d


def run(url: str) -> dict:
    std = load_std()
    data, dates, _ = load_split("val")
    out = {"std": std, "windows": {}}
    official_by_window = []
    for wd in WINDOW_DATES:
        i_last = int(np.where(dates == wd)[0][0]); init = data[i_last - 2:i_last + 1]
        doy = doy_sincos_for(to_date(wd)); last = init[-1]
        rec = {}
        base = simulate(url, init, doy, None)
        base2 = simulate(url, init, doy, None)
        zeros = simulate(url, init, doy, S.make_shock("null", last, std))
        omitted = simulate(url, init, doy, omit_perturbation=True)
        if not (base.ok and zeros.ok and omitted.ok and base2.ok):
            rec["error"] = [r.error for r in (base, base2, zeros, omitted)]
            out["windows"][str(wd)] = rec; continue
        rec["validity_base"] = S.validity(base.out, C.SCORE_STEPS)
        rec["deterministic"] = bool(np.array_equal(base.out, base2.out))
        rec["null_consistent_null_vs_zeros"] = S.null_consistent(base.out, zeros.out, std)
        rec["null_consistent_null_vs_omitted"] = S.null_consistent(base.out, omitted.out, std)
        rec["null_max_abs_diff"] = float(np.abs(base.out - zeros.out).max())
        rec["null_within_1e-6"] = rec["null_max_abs_diff"] <= 1e-6
        shocks = {}
        official = []
        for name, box in (("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA)):
            d = S.make_shock(name, last, std)
            r = simulate(url, init, doy, d)
            if not r.ok:
                shocks[name] = {"error": r.error}; continue
            cc = S.causality_checks(base.out, r.out, box, std)
            cc["validity"] = S.validity(r.out, C.SCORE_STEPS)
            dz = np.abs((r.out - base.out) / std[None, :, None, None])
            cc["max_z_by_day"] = dz.max(axis=(1, 2, 3)).tolist()
            near = S.near_mask(box)
            e = ((r.out - base.out) / std[None, :, None, None]) ** 2
            e = e.sum(1)
            cc["outside_energy_by_day"] = [float(e[k][~near].sum()) for k in range(C.SCORE_STEPS)]
            shocks[name] = cc; official.append(cc)
        for name, box, ch, val in CUSTOM:
            r = simulate(url, init, doy, custom_delta(box, ch, val))
            if not r.ok:
                shocks[name] = {"error": r.error}; continue
            cc = S.causality_checks(base.out, r.out, box, std)
            cc["validity"] = S.validity(r.out, C.SCORE_STEPS)
            shocks[name] = cc
        rec["shocks"] = shocks
        if len(official) == 2:
            rec["causality_points_of_15"] = S.causality_points(official, rec["null_consistent_null_vs_zeros"])
            official_by_window.append(rec["causality_points_of_15"])
        out["windows"][str(wd)] = rec
        print(wd, "null_ok", rec["null_consistent_null_vs_zeros"], "omitted_ok", rec["null_consistent_null_vs_omitted"],
              "det", rec["deterministic"], "pts", rec.get("causality_points_of_15"))
        for n, cc in shocks.items():
            print("   ", f"{n:28s}", {k: cc.get(k) for k in ("nonzero", "localized", "propagates", "no_teleport", "stable")},
                  "near1=%.3f far=%.3f" % (cc.get("near_frac_day1", float("nan")), cc.get("far_frac_max", float("nan"))))
    out["causality_points_mean_of_15"] = float(np.mean(official_by_window)) if official_by_window else None
    # adversarial perturbations: invalid values must be rejected, never silently ignored
    i_last = int(np.where(dates == WINDOW_DATES[2])[0][0]); init = data[i_last - 2:i_last + 1]
    import json, requests
    adv = {}
    for label, val in (("nan", float("nan")), ("inf", float("inf"))):
        p = np.zeros((C.NC, C.NLAT, C.NLON), "f4"); p[0, 10, 20] = val
        body = json.dumps({"init_states": C.encode_list(init), "doy_sincos": doy_sincos_for(to_date(WINDOW_DATES[2])),
                           "steps": 10, "perturbation": p.tolist()}, allow_nan=True)
        r = requests.post(url + ROUTE_SIM, data=body, headers={"content-type": "application/json"}, timeout=10)
        adv[f"perturbation_{label}"] = {"http": r.status_code,
                                        "silently_ignored": r.status_code == 200 and S.null_consistent(
                                            np.asarray(r.json()["rollout"], "f4"), simulate(url, init, doy_sincos_for(to_date(WINDOW_DATES[2]))).out, std)}
    out["adversarial"] = adv
    print("adversarial:", adv)
    out["note"] = ("A persistence server returns the perturbed last state unchanged for every day: the response is "
                   "non-zero, localized and stable but cannot propagate (outside-box energy stays 0), so 'propagates' "
                   "is expected False for persistence.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--port", type=int, default=8022)
    a = ap.parse_args()
    ctx = contextlib.nullcontext(None) if a.url else ServerProc(a.port, ncores=4)
    with ctx as srv:
        url = a.url or srv.url
        import requests
        res = run(url)
        res["health"] = requests.post(url + "/health", timeout=5).json()
    write_json(EVAL_DIR / "shock_report.json", res)


if __name__ == "__main__":
    main()
