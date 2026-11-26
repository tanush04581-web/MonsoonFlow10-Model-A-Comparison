"""Round 1 scoring. Pure numpy, no I/O: every function takes arrays and returns numbers, so each can be tested alone.

Total 100: forecast skill 45 (30 RMSE skill + 15 heavy-rain CSI), dynamics and multiverse 25 (10 anti-blur +
15 perturbation causality), repo integrity 15 (repo_scan.py), brief 15 (manual).
Knobs at the top are set by the organisers and frozen before the event.
"""
from __future__ import annotations
import numpy as np
from .contract import LAT, NC, NLAT, NLON, box_mask, lat_weights, BOX_BAY_OF_BENGAL, BOX_ARABIAN_SEA

# ---- knobs -----------------------------------------------------------------------------------------------
SS_REF = 0.12            # mean RMSE skill (vs the better of persistence / climatology) that earns the full 30 points
CSI_SS_REF = 0.10        # heavy-rain CSI skill (CSI - best_ref) / (1 - best_ref) that earns the full 15 points
RAIN_THRESH = 10.0       # mm/day
BLUR_MIN_RATIO = 0.40    # spec: day-8 spatial variance must keep >= 40% of the real one
BLUR_MAX_RATIO = 2.5     # and not be inflated by injected noise
STD_DEFAULT = np.array([8.0, 8.0, 3.5, 2.5, 450.0, 20.0], "f4")   # per-channel scale (K, mm, m/s, m/s, Pa, kg/m2), replaced by stats.json
NULL_TOL = 1e-3          # null shock: max |delta| / std must stay below this (deterministic or tiny ensemble spread)
NONZERO_MIN = 1e-3       # a shock must move some channel by at least this many stds
LOCAL_MIN_FRAC = 0.60    # day 1: share of the perturbation response energy that must sit near the shock
DILATE = 3               # cells around the box that still count as "near"
MAX_SPEED_CELLS_PER_DAY = 8.0   # ~900 km/day: faster than any atmospheric signal worth the name -> teleportation
STABLE_MAX_STD = 5.0     # |delta| above 5 stds anywhere = numerically exploding


# ---- basic skill -----------------------------------------------------------------------------------------
def rmse_lat(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Latitude-weighted RMSE per channel for [..., 6, 32, 36] arrays (mean over leading dims)."""
    w = lat_weights()[:, None]
    se = (pred - true) ** 2
    return np.sqrt((se * w).mean(axis=(-2, -1)).reshape(-1, NC).mean(axis=0))


def skill_vs_best_ref(pred, true, persist, clim) -> dict:
    """SS = 1 - RMSE / min(RMSE_persistence, RMSE_climatology), per channel; also each reference on its own."""
    r, rp, rc = rmse_lat(pred, true), rmse_lat(persist, true), rmse_lat(clim, true)
    best = np.minimum(rp, rc)
    with np.errstate(divide="ignore", invalid="ignore"):
        ss = np.where(best > 0, 1 - r / best, np.nan)
        sp = np.where(rp > 0, 1 - r / rp, np.nan); sc = np.where(rc > 0, 1 - r / rc, np.nan)
    return {"rmse": r, "ss": ss, "ss_persist": sp, "ss_clim": sc}


def csi(pred_tp: np.ndarray, true_tp: np.ndarray, thr: float = RAIN_THRESH) -> float:
    p, t = pred_tp > thr, true_tp > thr
    h = int((p & t).sum()); m = int((~p & t).sum()); f = int((p & ~t).sum())
    return float("nan") if h + m + f == 0 else h / (h + m + f)


def validity(out: np.ndarray, steps: int) -> list[str]:
    """Physical validity gate for one rollout [steps,6,32,36]. Returns the list of violations (empty = pass)."""
    bad = []
    if out.shape != (steps, NC, NLAT, NLON): return [f"shape {out.shape}"]
    if not np.isfinite(out).all(): bad.append("NaN/Inf")
    else:
        if out[:, 1].min() < -1e-6: bad.append("negative rain")
        t = out[:, 0]
        if t.min() < 233.15 or t.max() > 333.15: bad.append("t2m outside [-40, +60] C")
    return bad


def variance_ratio(pred_day: np.ndarray, true_day: np.ndarray) -> np.ndarray:
    """Spatial variance of the predicted field / of the real field, per channel ([6,32,36] each)."""
    vp = pred_day.reshape(NC, -1).var(axis=1); vt = true_day.reshape(NC, -1).var(axis=1)
    return np.where(vt > 0, vp / np.maximum(vt, 1e-30), np.nan)


def anti_blur_points(ratios: np.ndarray, max_pts: float = 10.0) -> float:
    """ratios [windows, 6]. A channel scores fully at >= 40% of the real variance, 0 if inflated past 2.5x."""
    ok_cnt = np.isfinite(ratios).sum(axis=0)
    r = np.where(ok_cnt > 0, np.nansum(np.where(np.isfinite(ratios), ratios, 0.0), axis=0) / np.maximum(ok_cnt, 1), np.nan)
    ok = np.where(np.isfinite(r), np.clip(r / BLUR_MIN_RATIO, 0, 1) * (r <= BLUR_MAX_RATIO), 0.0)
    return float(max_pts * ok.mean())


# ---- perturbation tests ----------------------------------------------------------------------------------
def _energy(delta: np.ndarray, std: np.ndarray) -> np.ndarray:
    """[steps,6,32,36] -> [steps,32,36] summed squared response in units of channel std."""
    return ((delta / std[None, :, None, None]) ** 2).sum(axis=1)


def near_mask(box, dilate: int = DILATE) -> np.ndarray:
    m = box_mask(box); out = m.copy()
    for _ in range(dilate):
        o = out.copy(); o[1:] |= out[:-1]; o[:-1] |= out[1:]; o[:, 1:] |= out[:, :-1]; o[:, :-1] |= out[:, 1:]; out = o
    return out


def causality_checks(base: np.ndarray, shock: np.ndarray, box, std: np.ndarray) -> dict:
    """The four checks for one shock. base/shock are rollouts [steps,6,32,36]. Returns bools plus the numbers behind them."""
    d = shock - base
    res = {"finite": bool(np.isfinite(d).all())}
    if not res["finite"]:
        return {**res, "nonzero": False, "localized": False, "propagates": False, "stable": False}
    z = np.abs(d / std[None, :, None, None])
    res["nonzero"] = bool(z.max() >= NONZERO_MIN)
    res["stable"] = bool(z.max() <= STABLE_MAX_STD)
    e = _energy(d, std); steps = e.shape[0]
    near = near_mask(box); cen = np.array([LAT[box_mask(box).any(axis=1)].mean(), 0.0])
    tot1 = e[0].sum()
    res["near_frac_day1"] = float(e[0][near].sum() / tot1) if tot1 > 0 else 0.0
    res["localized"] = bool(tot1 > 0 and res["near_frac_day1"] >= LOCAL_MIN_FRAC)
    # outward propagation: energy outside the neighbourhood grows from day 1 to days 4-8, but never faster than light-speed-ish
    out_e = np.array([e[k][~near].sum() for k in range(steps)])
    late = out_e[3:8].mean() if steps >= 8 else out_e[-1]
    grows = bool(late > 1.5 * out_e[0] + 1e-12 and late > 0.02 * max(e.sum(axis=(1, 2)).max(), 1e-12))
    # teleport check: energy beyond the reachable radius at each day must be small
    ii, jj = np.indices((NLAT, NLON)); boxm = box_mask(box)
    bi, bj = np.where(boxm); dist = np.sqrt(((ii[..., None] - bi) ** 2 + (jj[..., None] - bj) ** 2)).min(axis=-1)
    tele = 0.0
    for k in range(steps):
        far = dist > (DILATE + MAX_SPEED_CELLS_PER_DAY * (k + 1))
        tele = max(tele, float(e[k][far].sum() / max(e[k].sum(), 1e-12)))
    res["far_frac_max"] = tele; res["no_teleport"] = bool(tele < 0.05)
    res["propagates"] = bool(grows and res["no_teleport"])
    return res


def causality_points(shock_results: list[dict], null_ok: bool, max_pts: float = 15.0) -> float:
    """Each of 2 shocks is worth max/2, a quarter per check, but checks depend on each other so that behaviour which only
    LOOKS fine earns nothing: no response -> 0; a response that is not localised near the shock -> only the
    'non-zero' quarter (a global flat nudge is not causality); stability and propagation count only for a localised
    response. A failed null-shock consistency halves the total."""
    per = max_pts / max(len(shock_results), 1)
    got = 0.0
    for r in shock_results:
        c = 0
        if r["nonzero"]:
            c += 1
            if r["localized"]:
                c += 1 + int(bool(r["propagates"])) + int(bool(r["stable"]))
        got += per * c / 4.0
    return float(got * (1.0 if null_ok else 0.5))


def null_consistent(a: np.ndarray, b: np.ndarray, std: np.ndarray) -> bool:
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or a.shape != b.shape:
        return False
    return bool((np.abs(a - b) / std[None, :, None, None]).max() <= NULL_TOL)


# ---- shocks ----------------------------------------------------------------------------------------------
def make_shock(name: str, last_state: np.ndarray, std: np.ndarray) -> np.ndarray:
    """delta [6,32,36] added to init_states[-1]. Defined here once so teams and the grader agree."""
    d = np.zeros((NC, NLAT, NLON), "f4")
    if name == "warm_ocean":                       # +3 C and +15% moisture over the Bay of Bengal box
        m = box_mask(BOX_BAY_OF_BENGAL); d[0][m] = 3.0; d[5][m] = 0.15 * last_state[5][m]
    elif name == "monsoon_stall":                  # -40% of the incoming wind over the Arabian Sea box
        m = box_mask(BOX_ARABIAN_SEA); d[2][m] = -0.4 * last_state[2][m]; d[3][m] = -0.4 * last_state[3][m]
    elif name != "null":
        raise ValueError(name)
    return d


# ---- forecast score for one team -------------------------------------------------------------------------
def csi_skill(csi_model: float, csi_persist: float, csi_clim: float) -> float:
    """(CSI - ref) / (1 - ref) where ref is the better of persistence and climatology. NaN when CSI is undefined."""
    ref = np.nanmax([csi_persist, csi_clim]) if np.isfinite([csi_persist, csi_clim]).any() else float("nan")
    if not np.isfinite(csi_model) or not np.isfinite(ref) or ref >= 1:
        return float("nan")
    return float((csi_model - ref) / (1 - ref))


def forecast_points(ss_by_window: list[float], csi_ss_by_window: list[float]) -> dict:
    ss = np.array([x for x in ss_by_window if np.isfinite(x)]); cs = np.array([x for x in csi_ss_by_window if np.isfinite(x)])
    ss_m = float(ss.mean()) if len(ss) else float("nan"); cs_m = float(cs.mean()) if len(cs) else float("nan")
    rmse_pts = 30.0 * float(np.clip(ss_m / SS_REF, 0, 1)) if np.isfinite(ss_m) else 0.0
    csi_pts = 15.0 * float(np.clip(cs_m / CSI_SS_REF, 0, 1)) if np.isfinite(cs_m) else 0.0
    return {"mean_ss": ss_m, "mean_csi_skill": cs_m, "rmse_pts": rmse_pts, "csi_pts": csi_pts, "points": rmse_pts + csi_pts}
