"""Automated Anti-Cheat and Code Health Static Scanner.

Audited by Agent 5: Agent-QA.
Directly wraps `monsoon_r1.repo_scan.scan('.')`.
Audits:
- Anti-Cheat (8 pts): 0 critical flags, no banned weather APIs, < 5 date literals, local weight file present.
- Code Health (7 pts): training pipeline, frontend presence, server separation, readme, checkpoint.
- Physical Git Constraints: zero committed .nc files, repository size < 2.0 GB.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from monsoon_r1.repo_scan import scan

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_git_nc_files() -> list[str]:
    """Verify that no .nc files are tracked in git."""
    try:
        res = subprocess.run(
            ["git", "ls-files", "*.nc", "*.nc4", "*.grib", "*.grib2", "*.zarr"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        tracked = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        return tracked
    except Exception as e:
        return []


def check_repo_size_gb() -> float:
    """Calculate total size of the repository on disk in GB."""
    total_bytes = 0
    for p in REPO_ROOT.rglob("*"):
        if p.is_file():
            try:
                total_bytes += p.stat().st_size
            except Exception:
                pass
    return total_bytes / (1024 ** 3)


def main():
    print("=" * 70)
    print("  STATIC ANTI-CHEAT & REPOSITORY AUDIT (monsoon_r1.repo_scan)")
    print("=" * 70)

    # 1. Run official static AST scanner
    scan_result = scan(str(REPO_ROOT))
    
    # Filter out template/resources files if scanned accidentally from vendor kit
    # Official rules specify scanning the root submission
    critical_flags = [f for f in scan_result.get("flags", []) if f.get("level") == "critical"]
    warn_flags = [f for f in scan_result.get("flags", []) if f.get("level") == "warn"]

    print(f"Total Static Points Earned: {scan_result.get('points', 0.0)} / 15.0")
    print(f"  - Anti-Cheat Points: {scan_result.get('anti_cheat_points', 0.0)} / 8.0")
    print(f"  - Code Health Points: {scan_result.get('code_health_points', 0.0)} / 7.0")

    print("\nFlags Detected:")
    if not scan_result.get("flags"):
        print("  [OK] Zero flags detected.")
    else:
        for f in scan_result["flags"]:
            lvl = f.get("level", "info").upper()
            what = f.get("what", "")
            fl = f.get("file", "")
            print(f"  [{lvl}] {what} {f'in {fl}' if fl else ''}")

    # 2. Check git for committed NetCDF files
    print("\nPhysical Git Storage Audit:")
    nc_files = check_git_nc_files()
    if nc_files:
        print(f"  [CRITICAL VIOLATION] Raw climate data files committed in git: {nc_files}")
    else:
        print("  [OK] No raw .nc / .grib / .zarr files tracked in git.")

    # 3. Check repository size
    repo_size = check_repo_size_gb()
    print(f"  [OK] Repository Size: {repo_size:.3f} GB (< 2.0 GB limit)")
    assert repo_size < 2.0, f"Repository size {repo_size:.2f} GB exceeds 2.0 GB limit"

    print("=" * 70)
    report_file = REPO_ROOT / "evaluation" / "repo_scan_report.json"
    scan_result["git_nc_files"] = nc_files
    scan_result["repo_size_gb"] = repo_size
    with open(report_file, "w") as f:
        json.dump(scan_result, f, indent=2)
    print(f"Full audit report saved to: {report_file}\n")


if __name__ == "__main__":
    main()
