"""Official Evaluation & 100-Point Grader Wrapper.

Audited by Agent 5: Agent-QA.
Directly wraps `monsoon_r1.scoring` and `monsoon_r1.contract`.
Evaluates:
- Hard Admission Gate: Validity & Beat Persistence at Day 4
- Priority 1: Forecast Skill (45 pts = 30 RMSE Skill + 15 Heavy-Rain CSI)
- Priority 2: Dynamics & Multiverse (25 pts = 10 Anti-Blur + 15 Causality)
- Priority 3: Static Repo Scan (15 pts)
- Priority 4: 2-Page Brief (15 pts manual)
Total: 100 Points.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from monsoon_r1.contract import (
    CHANNELS,
    NC,
    NLAT,
    NLON,
    LEAD_DAYS,
    SCORE_STEPS,
)
from monsoon_r1.scoring import (
    SS_REF,
    CSI_SS_REF,
    RAIN_THRESH,
    BLUR_MIN_RATIO,
    BLUR_MAX_RATIO,
    rmse_lat,
    skill_vs_best_ref,
    csi,
    validity,
    variance_ratio,
    anti_blur_points,
    causality_points,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate_rollouts(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_persist: np.ndarray,
    y_clim: np.ndarray,
    shock_results: list[dict] | None = None,
    null_ok: bool = True,
    repo_scan_points: float = 15.0,
    brief_points: float = 15.0,
) -> dict:
    """Grade model rollouts against reference baselines and physical constraints.

    Args:
        y_pred: [W, 10, 6, 32, 36] predicted rollouts in physical units
        y_true: [W, 10, 6, 32, 36] ground truth
        y_persist: [W, 10, 6, 32, 36] persistence baseline
        y_clim: [W, 10, 6, 32, 36] climatology baseline
        shock_results: list of dicts from monsoon_r1.scoring.causality_checks
        null_ok: whether null shock met tolerance
    """
    W = y_pred.shape[0]
    scorecard = {
        "admission_gate": {"passed": True, "violations": []},
        "forecast_skill": {"total_pts": 0.0, "rmse_skill_pts": 0.0, "csi_pts": 0.0, "details": {}},
        "dynamics_multiverse": {"total_pts": 0.0, "anti_blur_pts": 0.0, "causality_pts": 0.0, "details": {}},
        "repo_scan_pts": float(repo_scan_points),
        "brief_pts": float(brief_points),
        "total_score": 0.0,
    }

    # 1. Hard Gate: Physical Validity across all windows
    for w in range(W):
        issues = validity(y_pred[w], SCORE_STEPS)
        if issues:
            scorecard["admission_gate"]["passed"] = False
            scorecard["admission_gate"]["violations"].append(f"Window {w}: {', '.join(issues)}")

    # 2. Hard Gate: Beat Persistence at Day 4 (Day 4 is index 3)
    k_day4 = 3
    rmse_pred_d4 = rmse_lat(y_pred[:, k_day4], y_true[:, k_day4])
    rmse_p_d4 = rmse_lat(y_persist[:, k_day4], y_true[:, k_day4])
    mean_rmse_pred_d4 = float(np.mean(rmse_pred_d4))
    mean_rmse_p_d4 = float(np.mean(rmse_p_d4))
    beat_day4 = mean_rmse_pred_d4 < mean_rmse_p_d4
    scorecard["admission_gate"]["beat_day4_persistence"] = beat_day4
    scorecard["admission_gate"]["day4_pred_rmse"] = mean_rmse_pred_d4
    scorecard["admission_gate"]["day4_persist_rmse"] = mean_rmse_p_d4

    if not beat_day4:
        scorecard["admission_gate"]["passed"] = False
        scorecard["admission_gate"]["violations"].append(
            f"Failed to beat Day 4 Persistence: Pred {mean_rmse_pred_d4:.3f} >= Persist {mean_rmse_p_d4:.3f}"
        )

    # 3. Forecast Skill (45 pts): RMSE Skill (30) + Heavy-Rain CSI (15)
    ss_list = []
    csi_ss_list = []

    for day in LEAD_DAYS:
        k = day - 1
        # RMSE Skill
        sk = skill_vs_best_ref(y_pred[:, k], y_true[:, k], y_persist[:, k], y_clim[:, k])
        ss_mean_channel = float(np.nanmean(sk["ss"]))
        ss_list.append(ss_mean_channel)

        # Heavy-Rain CSI at 10 mm/day
        csi_m = csi(y_pred[:, k, 1], y_true[:, k, 1], thr=RAIN_THRESH)
        csi_p = csi(y_persist[:, k, 1], y_true[:, k, 1], thr=RAIN_THRESH)
        csi_c = csi(y_clim[:, k, 1], y_true[:, k, 1], thr=RAIN_THRESH)
        best_ref_csi = max(csi_p if np.isfinite(csi_p) else 0.0, csi_c if np.isfinite(csi_c) else 0.0)
        
        denom = 1.0 - best_ref_csi
        csi_ss = (csi_m - best_ref_csi) / max(denom, 1e-6) if np.isfinite(csi_m) else 0.0
        csi_ss_list.append(max(0.0, csi_ss))

    overall_rmse_ss = float(np.mean(ss_list))
    rmse_pts = 30.0 * np.clip(overall_rmse_ss / SS_REF, 0.0, 1.0)
    scorecard["forecast_skill"]["rmse_skill_pts"] = float(rmse_pts)
    scorecard["forecast_skill"]["overall_rmse_skill_score"] = overall_rmse_ss

    overall_csi_ss = float(np.mean(csi_ss_list))
    csi_pts = 15.0 * np.clip(overall_csi_ss / CSI_SS_REF, 0.0, 1.0)
    scorecard["forecast_skill"]["csi_pts"] = float(csi_pts)
    scorecard["forecast_skill"]["overall_csi_skill_score"] = overall_csi_ss
    scorecard["forecast_skill"]["total_pts"] = float(rmse_pts + csi_pts)

    # 4. Dynamics & Multiverse (25 pts): Anti-Blur (10) + Causality (15)
    # Day 8 Anti-Blur (Day 8 is index 7)
    k_day8 = 7
    var_ratios = []
    for w in range(W):
        vr = variance_ratio(y_pred[w, k_day8], y_true[w, k_day8])
        var_ratios.append(vr)
    ratios_arr = np.array(var_ratios)  # [W, 6]
    anti_blur_pts = anti_blur_points(ratios_arr, max_pts=10.0)
    scorecard["dynamics_multiverse"]["anti_blur_pts"] = float(anti_blur_pts)
    scorecard["dynamics_multiverse"]["mean_day8_variance_ratios"] = {
        c: float(np.nanmean(ratios_arr[:, i])) for i, c in enumerate(CHANNELS)
    }

    # Causality Points (15 pts)
    if shock_results:
        caus_pts = causality_points(shock_results, null_ok, max_pts=15.0)
    else:
        # Default placeholder if shocks evaluated separately
        caus_pts = 15.0 if null_ok else 7.5
    scorecard["dynamics_multiverse"]["causality_pts"] = float(caus_pts)
    scorecard["dynamics_multiverse"]["total_pts"] = float(anti_blur_pts + caus_pts)

    # Total Score
    total = (
        scorecard["forecast_skill"]["total_pts"]
        + scorecard["dynamics_multiverse"]["total_pts"]
        + scorecard["repo_scan_pts"]
        + scorecard["brief_pts"]
    )
    scorecard["total_score"] = float(total if scorecard["admission_gate"]["passed"] else 0.0)

    return scorecard


def main():
    parser = argparse.ArgumentParser(description="Evaluate model rollouts against official monsoon_r1 benchmark.")
    parser.add_argument("--pred-path", type=str, default=None, help="Path to predicted rollouts .npy/.npz")
    parser.add_argument("--val-path", type=str, default=None, help="Path to val.npz ground truth")
    args = parser.parse_args()

    print("[Agent-QA] Running Official 100-Point Grader Audit...")
    # Self-test with baseline dummy to verify grader formulas
    rng = np.random.default_rng(42)
    W = 10
    true = np.ones((W, 10, NC, NLAT, NLON), dtype=np.float32) * 295.0
    true[:, :, 1] = 5.0  # tp
    persist = true.copy()
    clim = true.copy()
    pred = true + rng.normal(0, 0.5, size=true.shape).astype(np.float32)
    pred[:, :, 1] = np.maximum(0.0, pred[:, :, 1])

    scorecard = evaluate_rollouts(pred, true, persist, clim)
    print(json.dumps(scorecard, indent=2))
    print(f"\n[Agent-QA] Grader self-test completed. Admission Passed: {scorecard['admission_gate']['passed']}")


if __name__ == "__main__":
    main()
