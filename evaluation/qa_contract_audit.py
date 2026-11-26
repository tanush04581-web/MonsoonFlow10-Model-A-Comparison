"""Phase 1: official contract audit. Records the pinned kit's constants, shock definitions, masks and file hashes.

Read-only: introspects resources/yug-monsoon-kit/monsoon_r1 and writes evaluation/contract_audit.json.
Run:  .\\.venv\\Scripts\\python.exe evaluation\\qa_contract_audit.py
"""
from __future__ import annotations

import hashlib
import subprocess

import numpy as np

from qa_common import C, S, KC, KIT_ROOT, EVAL_DIR, load_std, write_json


def main():
    kit_dir = KIT_ROOT / "monsoon_r1"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(kit_dir.glob("*.py"))}
    try:
        commit = subprocess.run(["git", "-C", str(KIT_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(KIT_ROOT), "status", "--short"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit, dirty = "unknown", "unknown"

    std = load_std()
    rng = np.random.default_rng(0)
    state = (np.array([300.0, 8.0, 3.0, 2.0, 100500.0, 50.0], "f4")[:, None, None]
             + rng.normal(0, 1, (C.NC, C.NLAT, C.NLON)).astype("f4"))
    shocks = {}
    for name, box in (("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA), ("null", None)):
        d = S.make_shock(name, state, std)
        nz = {C.CHANNELS[c]: int((d[c] != 0).sum()) for c in range(C.NC)}
        rec = {"nonzero_cells_per_channel": nz, "max_abs_std_units": float(np.abs(d / std[:, None, None]).max())}
        if box is not None:
            m = C.box_mask(box)
            rec.update({"box_deg": box, "box_cells": int(m.sum()),
                        "box_lat_rows": [float(C.LAT[m.any(1)].min()), float(C.LAT[m.any(1)].max())],
                        "box_lon_cols": [float(C.LON[m.any(0)].min()), float(C.LON[m.any(0)].max())],
                        "near_mask_cells_dilate3": int(S.near_mask(box).sum()),
                        "outside_near_cells": int((~S.near_mask(box)).sum())})
        shocks[name] = rec

    out = {
        "kit_commit": commit, "kit_worktree_dirty": bool(dirty), "kit_file_sha256_16": hashes,
        "grid": {"NC": C.NC, "NLAT": C.NLAT, "NLON": C.NLON, "LAT": [float(C.LAT[0]), float(C.LAT[-1])],
                 "LON": [float(C.LON[0]), float(C.LON[-1])], "channels": list(C.CHANNELS)},
        "io_schema": {
            "health": "POST /health -> 200 {'status':'ready','model_name':str}",
            "simulate_request": {"init_states": [C.T_IN, C.NC, C.NLAT, C.NLON], "doy_sincos": "[sin,cos] of LAST input day (2*pi*doy/365)",
                                 "steps": C.SCORE_STEPS, "perturbation": "null | [6,32,36] physical delta added to init_states[-1]"},
            "simulate_response": {"rollout": [C.SCORE_STEPS, C.NC, C.NLAT, C.NLON], "inference_ms": "float"},
            "wire": "nested JSON lists of finite float32 (b64 tolerated); physical units",
            "client_decode": "client.call_simulate decodes rollout with expect_shape (SCORE_STEPS,6,32,36)",
        },
        "runtime": {"TIME_LIMIT_S": C.TIME_LIMIT_S, "client.LIMIT_S": KC.LIMIT_S, "MAX_RESPONSE_BYTES": KC.MAX_BYTES,
                    "MAX_STEPS": C.MAX_STEPS, "SCORE_STEPS": C.SCORE_STEPS, "LEAD_DAYS": C.LEAD_DAYS,
                    "health_warmup_s": 120, "redirects": "refused", "connect_timeout_s": 3.0,
                    "container": "offline, 4 CPU, 6 GB RAM, no GPU (README / R1_CONTRACT)"},
        "scoring_knobs": {k: getattr(S, k) for k in ("SS_REF", "CSI_SS_REF", "RAIN_THRESH", "BLUR_MIN_RATIO", "BLUR_MAX_RATIO",
                                                      "NULL_TOL", "NONZERO_MIN", "LOCAL_MIN_FRAC", "DILATE",
                                                      "MAX_SPEED_CELLS_PER_DAY", "STABLE_MAX_STD")},
        "STD_DEFAULT": S.STD_DEFAULT, "stats_json_std": std,
        "validity_gate": "shape==(steps,6,32,36); all finite; tp.min() >= -1e-6; 233.15 <= t2m <= 333.15 (strict float compare)",
        "null_requirement": "max |a-b|/std <= NULL_TOL (1e-3) between perturbation=null and perturbation=zeros",
        "causality": "finite; nonzero (max z >= 1e-3); stable (max z <= 5); localized (day-1 energy within DILATE=3 of box >= 60%); "
                     "propagates = outside energy days4-8 mean > 1.5x day1 and > 2% of peak total, AND no_teleport (<5% energy beyond 3+8*(k+1) cells)",
        "causality_points_rule": "per shock 7.5 pts, quarters: nonzero, then localized; propagates and stable count only if localized; null fail halves",
        "shocks": shocks,
        "repo_scan": {"inference_scope": "files matching the simulate route regex + transitive local imports",
                      "critical": ["external weather source pattern in inference code", ">=5 date literals in an inference file",
                                   "npy/npz archive with shape[0]>400 and trailing (32,36) ANYWHERE in the tree",
                                   "no weight file >1KB (WEIGHT_EXT) and no HF link", "no file serves the simulate route"],
                      "warn": ["3-4 date literals or date()/datetime() call with year literal in inference file",
                               "climate data file (.nc/.grib/.zarr) anywhere in the tree"],
                      "skip_dirs": sorted(__import__("monsoon_r1.repo_scan", fromlist=["SKIP_DIRS"]).SKIP_DIRS),
                      "scans_working_tree_not_git": True},
        "hard_gate_round2": "beat persistence at day 4; validity on all 8 cases; non-zero localized response to >=1 shock",
    }
    write_json(EVAL_DIR / "contract_audit.json", out)
    print("kit", commit, "dirty" if dirty else "clean")
    for k, v in shocks.items():
        print(k, v)


if __name__ == "__main__":
    main()
