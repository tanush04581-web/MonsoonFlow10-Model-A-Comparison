"""Phase 5: competition readiness.

- static inference-path audit (kit repo_scan.inference_files + AST imports): CPU-only, thread cap, no network code
- server started through qa_offline_guard.py (outbound network blocked + logged), pinned to 4 cores, CUDA hidden
- latency: 30 scored-shape simulate calls through the kit client (5 s hard deadline), with and without shocks
- memory: peak working set / RSS of the server process
- official schema validator: python -m monsoon_r1.schema_validator <url>
- repo_scan on (a) the working tree, (b) git-tracked files only, (c) tracked + untracked-not-ignored (what `git add -A` would ship)
- git size: git count-objects -vH, tracked and shippable byte totals, largest files
Run:  .\\.venv\\Scripts\\python.exe evaluation\\qa_readiness.py [--port 8020] [--scratch <dir>]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from qa_common import (C, S, PY, REPO_ROOT, EVAL_DIR, KIT_ROOT, ServerProc, doy_sincos_for, load_split, load_std,
                       simulate, to_date, write_json, ymd)
from monsoon_r1 import repo_scan as RS

NET_MODULES = {"requests", "urllib", "urllib3", "httpx", "http.client", "aiohttp", "socket", "ftplib", "smtplib",
               "websocket", "websockets", "boto3", "cdsapi", "huggingface_hub", "gdown"}


def static_inference_audit() -> dict:
    files = sorted(RS.inference_files(REPO_ROOT))
    rep = {"inference_files": [p.relative_to(REPO_ROOT).as_posix() for p in files], "per_file": {}}
    for p in files:
        src = p.read_text(errors="ignore")
        imps = set()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                imps |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                imps.add(n.module)
        net = sorted(i for i in imps if i in NET_MODULES or i.split(".")[0] in NET_MODULES)
        rep["per_file"][p.relative_to(REPO_ROOT).as_posix()] = {
            "imports": sorted(imps), "network_imports": net,
            "urls": re.findall(r"https?://[^\s'\"]+", src),
            "cuda_refs": len(re.findall(r"\.cuda\(|['\"]cuda", src)),
            "map_location_cpu": "map_location=\"cpu\"" in src or "map_location='cpu'" in src,
            "set_num_threads": re.findall(r"set_num_threads\((\d+)\)", src),
            "date_literals": len(RS.DATE_RE.findall(src)),
        }
    ign = {f for f in rep["inference_files"]
           if subprocess.run(["git", "check-ignore", "-q", f], cwd=str(REPO_ROOT)).returncode == 0}
    rep["gitignored_inference_files"] = sorted(ign)
    ship = {k: v for k, v in rep["per_file"].items() if k not in ign}
    rep["shipped_inference_files"] = sorted(ship)
    rep["any_network_import_shipped"] = any(v["network_imports"] for v in ship.values())
    rep["any_cuda_ref_shipped"] = any(v["cuda_refs"] for v in ship.values())
    rep["any_network_import_incl_ignored"] = any(v["network_imports"] for v in rep["per_file"].values())
    return rep


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True).stdout


def git_size() -> dict:
    tracked = [l for l in git("ls-files", "-z").split("\0") if l]
    untracked = [l for l in git("ls-files", "-z", "--others", "--exclude-standard").split("\0") if l]

    def sizes(lst):
        out = []
        for f in lst:
            p = REPO_ROOT / f
            if p.is_file():
                out.append((f, p.stat().st_size))
        return out
    ts, us = sizes(tracked), sizes(untracked)
    tb, ub = sum(s for _, s in ts), sum(s for _, s in us)
    return {"count_objects": git("count-objects", "-vH").strip().splitlines(),
            "tracked_files": len(ts), "tracked_bytes": tb, "tracked_gb": tb / 2**30,
            "untracked_not_ignored_files": len(us), "untracked_not_ignored_bytes": ub,
            "shippable_gb": (tb + ub) / 2**30, "under_2gb": (tb + ub) / 2**30 < 2.0,
            "largest_shippable": sorted(ts + us, key=lambda x: -x[1])[:10],
            "tracked_data_files": [f for f, _ in ts + us if Path(f).suffix in RS.DATA_EXT],
            "tracked_weight_files": [(f, s) for f, s in ts + us if Path(f).suffix in RS.WEIGHT_EXT]}


def scan_copy(file_list: list[str], dest: Path) -> dict:
    if dest.exists():
        shutil.rmtree(dest)
    for f in file_list:
        src = REPO_ROOT / f
        if src.is_file():
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / f)
    return RS.scan(dest)


def repo_scans(scratch: Path) -> dict:
    tracked = [l for l in git("ls-files", "-z").split("\0") if l]
    untracked = [l for l in git("ls-files", "-z", "--others", "--exclude-standard").split("\0") if l]
    wt = RS.scan(REPO_ROOT)
    a = scan_copy(tracked, scratch / "tracked")
    b = scan_copy(tracked + untracked, scratch / "shippable")
    ignored_hits = []
    for f in wt["flags"]:
        what = f.get("what", "") + " " + f.get("file", "")
        m = re.search(r"([\w./\\-]+\.(npz|npy|nc|nc4|grib2?|zarr))", what)
        if m:
            path = m.group(1).replace("\\", "/")
            ignored = subprocess.run(["git", "check-ignore", "-q", path], cwd=str(REPO_ROOT)).returncode == 0
            ignored_hits.append({"flag": f, "path": path, "gitignored": ignored})
    summ = lambda r: {"points": r["points"], "anti_cheat": r["anti_cheat_points"], "code_health": r["code_health_points"],
                      "critical": [f for f in r["flags"] if f["level"] == "critical"],
                      "warn": [f for f in r["flags"] if f["level"] == "warn"], "health": r["health"],
                      "inference_files": r["inference_files"]}
    return {"working_tree": summ(wt), "working_tree_flag_paths": ignored_hits,
            "tracked_only": summ(a), "tracked_plus_untracked_not_ignored": summ(b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--n-calls", type=int, default=30)
    ap.add_argument("--scratch", default=None)
    a = ap.parse_args()
    scratch = Path(a.scratch or tempfile.mkdtemp(prefix="qa_repo_scan_"))
    scratch.mkdir(parents=True, exist_ok=True)
    rep = {"static_inference": static_inference_audit()}
    print("static:", json.dumps({k: v for k, v in rep["static_inference"].items() if k != "per_file"}))

    log = scratch / "offline_guard.log"
    if log.exists():
        log.unlink()
    std = load_std()
    data, dates, _ = load_split("val")
    wd = ymd(2019, 7, 15)
    i_last = int(np.where(dates == wd)[0][0]); init = data[i_last - 2:i_last + 1]; doy = doy_sincos_for(to_date(wd))
    with ServerProc(a.port, ncores=4, launcher=EVAL_DIR / "qa_offline_guard.py", log=scratch / "server.log",
                    env={"QA_OFFLINE_LOG": str(log), "CUDA_VISIBLE_DEVICES": ""}) as srv:
        mem0 = srv.mem()
        walls, srv_ms, errs = [], [], []
        shocks = [None, S.make_shock("warm_ocean", init[-1], std), S.make_shock("monsoon_stall", init[-1], std),
                  np.zeros((C.NC, C.NLAT, C.NLON), "f4")]
        for n in range(a.n_calls):
            r = simulate(srv.url, init, doy, shocks[n % len(shocks)])
            walls.append(r.wall_s); srv_ms.append(r.server_ms)
            if not r.ok:
                errs.append(r.error)
        mem1 = srv.mem()
        val = subprocess.run([PY, "-m", "monsoon_r1.schema_validator", srv.url], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, timeout=300, env={**os.environ, "PYTHONPATH": str(KIT_ROOT)})
        mem2 = srv.mem()
        rep["server"] = {"startup_to_ready_s": srv.startup_s, "affinity_4_cores_set": srv.affinity_ok,
                         "cuda_hidden": True, "mem_after_start": mem0, "mem_after_calls": mem1, "mem_after_validator": mem2,
                         "peak_rss_mb": max(m.get("peak_rss_mb", 0) for m in (mem0, mem1, mem2)), "ram_limit_mb": 6144}
        rep["latency"] = {"n": a.n_calls, "errors": errs, "wall_s_max": max(walls), "wall_s_p50": float(np.median(walls)),
                          "wall_s_p95": float(np.percentile(walls, 95)), "server_ms_max": max(x for x in srv_ms if x is not None) if any(srv_ms) else None,
                          "limit_s": C.TIME_LIMIT_S, "pass": (not errs) and max(walls) <= C.TIME_LIMIT_S}
        rep["schema_validator"] = {"returncode": val.returncode, "stdout": val.stdout.splitlines(), "stderr_tail": val.stderr[-800:]}
    attempts = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    rep["offline_runtime"] = {"blocked_outbound_attempts": len(attempts), "attempts": attempts[:10],
                              "method": "socket.connect/getaddrinfo patched for non-loopback hosts (qa_offline_guard.py)"}
    print("server:", rep["server"]); print("latency:", rep["latency"]); print("validator:", "\n".join(rep["schema_validator"]["stdout"]))
    print("offline attempts:", len(attempts))

    try:
        dv = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        dv = ""
    rep["docker_available"] = dv or False

    rep["git_size"] = git_size()
    print("git size:", {k: v for k, v in rep["git_size"].items() if k not in ("largest_shippable",)})
    rep["repo_scan"] = repo_scans(scratch)
    for k in ("working_tree", "tracked_only", "tracked_plus_untracked_not_ignored"):
        r = rep["repo_scan"][k]
        print(f"repo_scan[{k}]: points={r['points']} anti={r['anti_cheat']} health={r['code_health']} "
              f"critical={[f['what'] for f in r['critical']]} warn={len(r['warn'])} inference={r['inference_files']}")
    rep["scratch_dir"] = str(scratch)
    write_json(EVAL_DIR / "readiness_report.json", rep)


if __name__ == "__main__":
    main()
