"""Shared, inference-safe tensor transforms for MonsoonFlow-10 candidates.

This module converts between the physical-unit wire format of the official
R1 contract and the normalised feature tensors consumed by the networks, and
applies the final physical safeguards to model output.

Import boundary (anti-cheat, see ``monsoon_r1.repo_scan``)
---------------------------------------------------------
This file is designed to be imported transitively by the root serving
entrypoint. It therefore imports ONLY ``torch`` and the standard ``math`` /
``typing`` modules. It must never import the ``data`` package (whose
``__init__`` pulls in modules containing calendar literals), any calendar or
datetime module, or any network client, and it must contain no hardcoded
calendar dates.

Conventions
-----------
* Dynamic channel order (axis ``-3`` of every state tensor) follows the
  contract: ``t2m [K], tp [mm/day], u10 [m/s], v10 [m/s], msl [Pa],
  tcwv [kg/m2]``.
* Grid: 32 latitude rows (south to north, 6.5..37.5) x 36 longitude columns
  (west to east, 65.5..100.5).
* Normalisation: per-channel z-score ``(x - mean) / max(std, STD_FLOOR)``,
  identical to ``data/dataset.py::MonsoonDataset`` so features stay
  bit-compatible with the existing training pipeline.
* Static terrain: ``[land_sea_mask, orography_m]``. The mask is passed through
  unchanged (already a 0..1 fraction); orography is divided by
  ``OROGRAPHY_SCALE_M`` (8000 m, the convention in ``data/dataset.py``).
* Day of year: the API supplies ``doy_last = [sin(theta), cos(theta)]`` with
  ``theta = 2*pi*doy/365`` for the LAST input day. History frames use offsets
  ``-2, -1, 0`` days; 0-indexed forecast step ``k`` uses offset ``k + 1``.
  Offsets are applied by the rotation matrix of the shared architecture
  document (Source/track-2-multiverse-of-monsoon-tech-stack.md, section
  "Mandatory Cyclic Temporal Conditioning"), so no integer day-of-year,
  calendar or leap-year logic is ever needed.
* Feature layout per frame: ``[6 dynamic | lsm, orography | sin, cos]`` =
  10 channels, matching ``data/dataset.py``.

Precipitation transform
-----------------------
``tp_transform="linear"`` (default) is the plain z-score used by the current
``checkpoints/stats.json``. ``tp_transform="log1p"`` normalises
``log1p(tp)`` instead; this REQUIRES the caller to supply the tp mean/std
computed on ``log1p(tp)`` over the 2005-2018 training split only (stored under
``tp_log1p_mean`` / ``tp_log1p_std`` in the stats dict, see
``stats_tensors_from_dict``). The linear-space tp statistics must never be
reused for log1p space. The transform must be identical at training and
serving time.

Non-finite handling
-------------------
Non-finite values are never silently repaired. ``apply_physical_constraints``
raises ``NonFiniteError`` with a per-channel count; callers decide how to fail
(e.g. return an HTTP error). Clamping is applied only to finite values.
"""

from __future__ import annotations

import math
from typing import Literal, Mapping, Sequence, Tuple, Union

import torch

__all__ = [
    "CHANNELS",
    "N_DYNAMIC",
    "N_STATIC",
    "N_DOY",
    "N_FEATURES",
    "NLAT",
    "NLON",
    "T_IN",
    "DOY_PERIOD_DAYS",
    "HISTORY_OFFSETS",
    "OROGRAPHY_SCALE_M",
    "STD_FLOOR",
    "T2M_MIN_K",
    "T2M_MAX_K",
    "T2M_SAFETY_MARGIN_K",
    "TpTransform",
    "NonFiniteError",
    "stats_tensors_from_dict",
    "normalize_dynamic",
    "denormalize_dynamic",
    "prepare_static",
    "rotate_doy_sincos",
    "make_doy_planes",
    "history_doy_planes",
    "forecast_doy_planes",
    "assemble_frame_features",
    "build_model_inputs",
    "nonfinite_counts",
    "assert_finite",
    "apply_physical_constraints",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CHANNELS: Tuple[str, ...] = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
T2M_IDX, TP_IDX, U10_IDX, V10_IDX, MSL_IDX, TCWV_IDX = range(6)

N_DYNAMIC = 6
N_STATIC = 2
N_DOY = 2
N_FEATURES = N_DYNAMIC + N_STATIC + N_DOY  # 10 channels per frame
NLAT, NLON = 32, 36
T_IN = 3

DOY_PERIOD_DAYS = 365.0
"""Period of the contract's cyclic encoding: theta = 2*pi*doy/365."""

HISTORY_OFFSETS: Tuple[int, ...] = (-2, -1, 0)
"""Day offsets of the three input frames relative to the LAST input day."""

OROGRAPHY_SCALE_M = 8000.0
"""Orography divisor in metres (matches data/dataset.py); maps terrain to ~[0, 1]."""

STD_FLOOR = 1e-6
"""Lower bound applied to std before dividing (matches data/dataset.py)."""

T2M_MIN_K = 233.15
T2M_MAX_K = 333.15
T2M_SAFETY_MARGIN_K = 0.01
"""Clamp t2m to [233.15 + m, 333.15 - m]. float32(233.15) rounds BELOW 233.15,
so clamping exactly at the official bound fails ``scoring.validity`` under
NumPy 1.x. 0.01 K is ~300x the float32 spacing at 333 K and physically
negligible."""

TpTransform = Literal["linear", "log1p"]

ArrayLike = Union[torch.Tensor, Sequence[float]]


class NonFiniteError(ValueError):
    """Raised when a tensor that must be finite contains NaN or Inf."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_tp_transform(tp_transform: str) -> None:
    if tp_transform not in ("linear", "log1p"):
        raise ValueError(f"tp_transform must be 'linear' or 'log1p', got {tp_transform!r}")


def _channel_stat(values: ArrayLike, like: torch.Tensor, name: str) -> torch.Tensor:
    """Return a per-channel statistic shaped (6, 1, 1) on ``like``'s device/dtype."""
    t = torch.as_tensor(values, dtype=like.dtype, device=like.device).reshape(-1)
    if t.numel() != N_DYNAMIC:
        raise ValueError(f"{name} must have {N_DYNAMIC} per-channel values, got {t.numel()}")
    if not bool(torch.isfinite(t).all()):
        raise NonFiniteError(f"{name} contains non-finite values: {t.tolist()}")
    return t.reshape(N_DYNAMIC, 1, 1)


def _check_dynamic(x: torch.Tensor, name: str) -> None:
    if x.dim() < 3 or tuple(x.shape[-3:]) != (N_DYNAMIC, NLAT, NLON):
        raise ValueError(
            f"{name} must end with (6, {NLAT}, {NLON}) [channels, lat, lon], got {tuple(x.shape)}"
        )
    if not x.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor, got {x.dtype}")


def _as_batched_doy(doy: ArrayLike, batch: int | None, like: torch.Tensor | None) -> torch.Tensor:
    """Coerce a (2,) or (B, 2) sin/cos pair to (B, 2)."""
    dtype = like.dtype if like is not None else torch.float32
    device = like.device if like is not None else None
    d = torch.as_tensor(doy, dtype=dtype, device=device)
    if d.dim() == 1:
        if d.shape[0] != 2:
            raise ValueError(f"doy_sincos must have shape (2,) or (B, 2), got {tuple(d.shape)}")
        d = d.unsqueeze(0).expand(batch if batch is not None else 1, 2)
    elif d.dim() == 2 and d.shape[1] == 2:
        if batch is not None and d.shape[0] not in (1, batch):
            raise ValueError(f"doy_sincos batch {d.shape[0]} does not match input batch {batch}")
        if batch is not None and d.shape[0] == 1:
            d = d.expand(batch, 2)
    else:
        raise ValueError(f"doy_sincos must have shape (2,) or (B, 2), got {tuple(d.shape)}")
    if not bool(torch.isfinite(d).all()):
        raise NonFiniteError("doy_sincos contains non-finite values")
    return d


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def stats_tensors_from_dict(
    stats: Mapping[str, object],
    tp_transform: TpTransform = "linear",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract validated (mean, std) tensors of shape (6,) from a stats.json dict.

    The dict must contain ``channels`` (in contract order), ``mean`` and ``std``.
    Statistics are never computed here: they must come from the 2005-2018
    training split.

    For ``tp_transform="log1p"`` the dict must additionally contain
    ``tp_log1p_mean`` and ``tp_log1p_std`` (scalars computed on ``log1p(tp)``
    over the training split); these replace the tp entry. A ``KeyError`` is
    raised if they are missing, rather than silently reusing linear stats.
    """
    _check_tp_transform(tp_transform)
    channels = tuple(stats.get("channels", CHANNELS))  # type: ignore[arg-type]
    if channels != CHANNELS:
        raise ValueError(f"stats channel order {channels} != contract order {CHANNELS}")
    mean = torch.tensor(stats["mean"], dtype=dtype).reshape(-1)  # type: ignore[arg-type]
    std = torch.tensor(stats["std"], dtype=dtype).reshape(-1)  # type: ignore[arg-type]
    if mean.numel() != N_DYNAMIC or std.numel() != N_DYNAMIC:
        raise ValueError("stats mean/std must each have 6 entries")
    if tp_transform == "log1p":
        missing = [k for k in ("tp_log1p_mean", "tp_log1p_std") if k not in stats]
        if missing:
            raise KeyError(
                f"tp_transform='log1p' requires training-split log1p(tp) statistics; missing {missing}"
            )
        mean = mean.clone()
        std = std.clone()
        mean[TP_IDX] = float(stats["tp_log1p_mean"])  # type: ignore[arg-type]
        std[TP_IDX] = float(stats["tp_log1p_std"])  # type: ignore[arg-type]
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise NonFiniteError("stats contain non-finite values")
    if bool((std <= 0).any()):
        raise ValueError(f"stats std must be positive, got {std.tolist()}")
    return mean, std


# --------------------------------------------------------------------------- #
# Dynamic channel normalisation
# --------------------------------------------------------------------------- #
def normalize_dynamic(
    x_phys: torch.Tensor,
    mean: ArrayLike,
    std: ArrayLike,
    tp_transform: TpTransform = "linear",
) -> torch.Tensor:
    """Physical units -> normalised model space.

    Args:
        x_phys: Tensor ``(..., 6, 32, 36)`` in physical units, e.g. ``(B, 3, 6, 32, 36)``
            or ``(B, 6, 32, 36)``. Broadcasting is over all leading dims.
        mean, std: Per-channel statistics, 6 values each, supplied by the caller
            (in log1p space for tp when ``tp_transform="log1p"``).
        tp_transform: ``"linear"`` or ``"log1p"``. With ``"log1p"``, tp is first
            clamped at 0 (log1p is undefined below -1 and rain is non-negative)
            and then transformed as ``log1p(tp)``.

    Returns:
        Tensor of the same shape and dtype: ``(x' - mean) / max(std, STD_FLOOR)``.
    """
    _check_dynamic(x_phys, "x_phys")
    _check_tp_transform(tp_transform)
    m = _channel_stat(mean, x_phys, "mean")
    s = _channel_stat(std, x_phys, "std").clamp_min(STD_FLOOR)
    x = x_phys
    if tp_transform == "log1p":
        tp = torch.log1p(x[..., TP_IDX : TP_IDX + 1, :, :].clamp_min(0.0))
        x = torch.cat([x[..., :TP_IDX, :, :], tp, x[..., TP_IDX + 1 :, :, :]], dim=-3)
    return (x - m) / s


def denormalize_dynamic(
    x_norm: torch.Tensor,
    mean: ArrayLike,
    std: ArrayLike,
    tp_transform: TpTransform = "linear",
) -> torch.Tensor:
    """Normalised model space -> physical units (exact inverse of ``normalize_dynamic``).

    No clamping is done here (apart from the inverse being exact for valid
    inputs); call ``apply_physical_constraints`` on the final output.
    With ``"log1p"``, tp is recovered with ``expm1``; values that overflow
    become ``inf`` and are caught by the finiteness check downstream.
    """
    _check_dynamic(x_norm, "x_norm")
    _check_tp_transform(tp_transform)
    m = _channel_stat(mean, x_norm, "mean")
    s = _channel_stat(std, x_norm, "std").clamp_min(STD_FLOOR)
    x = x_norm * s + m
    if tp_transform == "log1p":
        tp = torch.expm1(x[..., TP_IDX : TP_IDX + 1, :, :])
        x = torch.cat([x[..., :TP_IDX, :, :], tp, x[..., TP_IDX + 1 :, :, :]], dim=-3)
    return x


# --------------------------------------------------------------------------- #
# Static terrain
# --------------------------------------------------------------------------- #
def prepare_static(
    static_raw: torch.Tensor,
    orography_scale_m: float = OROGRAPHY_SCALE_M,
) -> torch.Tensor:
    """Scale raw static fields ``[land_sea_mask, orography_m]`` for the network.

    Args:
        static_raw: ``(2, 32, 36)`` or ``(B, 2, 32, 36)``; channel 0 is the land-sea
            fraction in [0, 1], channel 1 is orography in metres (raw, NOT pre-scaled).
        orography_scale_m: Divisor for orography (default 8000 m).

    Returns:
        Tensor with the same shape: ``[lsm, orography_m / orography_scale_m]``.
    """
    if static_raw.dim() not in (3, 4) or tuple(static_raw.shape[-3:]) != (N_STATIC, NLAT, NLON):
        raise ValueError(f"static_raw must be (2,{NLAT},{NLON}) or (B,2,{NLAT},{NLON}), got {tuple(static_raw.shape)}")
    if orography_scale_m <= 0:
        raise ValueError("orography_scale_m must be positive")
    if not bool(torch.isfinite(static_raw).all()):
        raise NonFiniteError("static_raw contains non-finite values")
    lsm = static_raw[..., 0:1, :, :]
    oro = static_raw[..., 1:2, :, :] / orography_scale_m
    return torch.cat([lsm, oro], dim=-3)


# --------------------------------------------------------------------------- #
# Day-of-year conditioning
# --------------------------------------------------------------------------- #
def rotate_doy_sincos(doy_sincos: ArrayLike, offset_days: Union[int, float, ArrayLike]) -> torch.Tensor:
    """Advance a ``[sin(theta), cos(theta)]`` pair by ``offset_days`` days.

    Uses the rotation from the shared architecture document with
    ``d_theta = 2*pi*offset/365``::

        [s'] = [ cos d  sin d] [s]      s' = sin(theta + d)
        [c']   [-sin d  cos d] [c]      c' = cos(theta + d)

    Args:
        doy_sincos: ``(2,)`` or ``(..., 2)`` pairs (last axis = [sin, cos]).
        offset_days: scalar, or a 1-D sequence/tensor of K offsets (may be negative).

    Returns:
        Scalar offset: same shape as ``doy_sincos``.
        K offsets: ``(*doy_sincos.shape[:-1], K, 2)``.
    The angle is evaluated in float64 for accuracy and cast back to the input dtype.
    """
    d = torch.as_tensor(doy_sincos)
    if not d.is_floating_point():
        d = d.to(torch.float32)
    if d.shape[-1] != 2:
        raise ValueError(f"doy_sincos last dim must be 2, got {tuple(d.shape)}")
    off = torch.as_tensor(offset_days, dtype=torch.float64, device=d.device)
    if off.dim() > 1:
        raise ValueError("offset_days must be a scalar or 1-D")
    delta = off * (2.0 * math.pi / DOY_PERIOD_DAYS)
    cd, sd = torch.cos(delta), torch.sin(delta)
    s0 = d[..., 0].to(torch.float64)
    c0 = d[..., 1].to(torch.float64)
    if off.dim() == 1:
        s0, c0 = s0.unsqueeze(-1), c0.unsqueeze(-1)
    s1 = cd * s0 + sd * c0
    c1 = -sd * s0 + cd * c0
    return torch.stack([s1, c1], dim=-1).to(d.dtype)


def make_doy_planes(
    doy_last: ArrayLike,
    offset_days: Union[int, Sequence[int]],
    height: int = NLAT,
    width: int = NLON,
    batch: int | None = None,
) -> torch.Tensor:
    """Broadcast rotated sin/cos pairs to constant spatial planes.

    Args:
        doy_last: ``(2,)`` or ``(B, 2)`` pair for the LAST input day.
        offset_days: an int (one frame) or a sequence of K ints.
        height, width: spatial size (default 32 x 36).
        batch: batch size used to expand a ``(2,)`` input (default 1).

    Returns:
        int offset -> ``(B, 2, H, W)``; sequence -> ``(B, K, 2, H, W)``.
    """
    like = doy_last if isinstance(doy_last, torch.Tensor) and doy_last.is_floating_point() else None
    d = _as_batched_doy(doy_last, batch, like)  # (B, 2)
    pairs = rotate_doy_sincos(d, offset_days)  # (B, 2) or (B, K, 2)
    return pairs[..., None, None].expand(*pairs.shape, height, width)


def history_doy_planes(doy_last: ArrayLike, batch: int | None = None) -> torch.Tensor:
    """Planes for the three input frames (offsets -2, -1, 0): ``(B, 3, 2, 32, 36)``."""
    return make_doy_planes(doy_last, list(HISTORY_OFFSETS), batch=batch)


def forecast_doy_planes(doy_last: ArrayLike, step: int, batch: int | None = None) -> torch.Tensor:
    """Planes for 0-indexed forecast step ``step`` (offset ``step + 1``): ``(B, 2, 32, 36)``."""
    if step < 0:
        raise ValueError("forecast step must be >= 0")
    return make_doy_planes(doy_last, step + 1, batch=batch)


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #
def assemble_frame_features(
    dyn_norm: torch.Tensor,
    static_prepared: torch.Tensor,
    doy_planes: torch.Tensor,
) -> torch.Tensor:
    """Concatenate ``[dynamic(6) | static(2) | doy(2)]`` along the channel axis.

    Used both for the history frames and for each autoregressive step, where
    ``dyn_norm`` is the model's own normalised prediction.

    Args:
        dyn_norm: ``(B, 6, H, W)`` or ``(B, T, 6, H, W)``, already normalised.
        static_prepared: output of ``prepare_static``: ``(2, H, W)`` or ``(B, 2, H, W)``.
        doy_planes: ``(B, 2, H, W)`` matching a single frame, or ``(B, T, 2, H, W)``.

    Returns:
        ``(B, 10, H, W)`` or ``(B, T, 10, H, W)``.
    """
    _check_dynamic(dyn_norm, "dyn_norm")
    if dyn_norm.dim() not in (4, 5):
        raise ValueError("dyn_norm must be (B,6,H,W) or (B,T,6,H,W)")
    if doy_planes.dim() != dyn_norm.dim() or doy_planes.shape[-3] != N_DOY:
        raise ValueError(f"doy_planes shape {tuple(doy_planes.shape)} incompatible with {tuple(dyn_norm.shape)}")
    lead = dyn_norm.shape[:-3]
    st = static_prepared.to(dtype=dyn_norm.dtype, device=dyn_norm.device)
    if st.dim() == 3:  # (2,H,W)
        st = st.reshape((1,) * len(lead) + tuple(st.shape))
    elif st.dim() == 4 and dyn_norm.dim() == 5:  # (B,2,H,W) -> (B,1,2,H,W)
        st = st.unsqueeze(1)
    st = st.expand(*lead, N_STATIC, dyn_norm.shape[-2], dyn_norm.shape[-1])
    dp = doy_planes.to(dtype=dyn_norm.dtype, device=dyn_norm.device).expand(*lead, N_DOY, *dyn_norm.shape[-2:])
    return torch.cat([dyn_norm, st, dp], dim=-3)


def build_model_inputs(
    x_phys: torch.Tensor,
    static_raw: torch.Tensor,
    doy_last: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    tp_transform: TpTransform = "linear",
    orography_scale_m: float = OROGRAPHY_SCALE_M,
) -> torch.Tensor:
    """Build the history feature tensor from physical inputs.

    Any perturbation must already have been added to ``x_phys[:, -1]`` in
    physical units, before calling this function.

    Args:
        x_phys: ``(B, 3, 6, 32, 36)`` physical-unit states, oldest first.
        static_raw: ``(2, 32, 36)`` or ``(B, 2, 32, 36)`` raw ``[lsm, orography_m]``.
        doy_last: ``(2,)`` or ``(B, 2)`` sin/cos for the LAST input day.
        mean, std: caller-supplied training statistics (6 values each).
        tp_transform: see ``normalize_dynamic``.
        orography_scale_m: see ``prepare_static``.

    Returns:
        ``(B, 3, 10, 32, 36)``: per frame ``[6 normalised dynamic | lsm, oro | sin, cos]``
        with doy offsets ``-2, -1, 0`` days.
    """
    if x_phys.dim() != 5 or x_phys.shape[1] != T_IN:
        raise ValueError(f"x_phys must be (B, {T_IN}, 6, {NLAT}, {NLON}), got {tuple(x_phys.shape)}")
    assert_finite(x_phys, "x_phys")
    batch = x_phys.shape[0]
    dyn = normalize_dynamic(x_phys, mean, std, tp_transform)
    st = prepare_static(static_raw.to(dtype=x_phys.dtype, device=x_phys.device), orography_scale_m)
    if st.dim() == 4 and st.shape[0] not in (1, batch):
        raise ValueError(f"static batch {st.shape[0]} does not match input batch {batch}")
    doy = _as_batched_doy(doy_last, batch, x_phys)
    planes = history_doy_planes(doy, batch=batch)
    return assemble_frame_features(dyn, st, planes)


# --------------------------------------------------------------------------- #
# Finiteness and physical constraints
# --------------------------------------------------------------------------- #
def nonfinite_counts(x: torch.Tensor) -> list[int]:
    """Per-channel count of NaN/Inf for a ``(..., 6, H, W)`` tensor (for logging)."""
    _check_dynamic(x, "x")
    bad = ~torch.isfinite(x)
    return [int(v) for v in bad.movedim(-3, 0).reshape(N_DYNAMIC, -1).sum(dim=1)]


def assert_finite(x: torch.Tensor, name: str = "tensor") -> None:
    """Raise ``NonFiniteError`` (with per-channel counts when possible) if ``x`` has NaN/Inf."""
    if bool(torch.isfinite(x).all()):
        return
    detail = ""
    if x.dim() >= 3 and tuple(x.shape[-3:]) == (N_DYNAMIC, NLAT, NLON):
        counts = nonfinite_counts(x)
        detail = " per-channel " + ", ".join(f"{c}={n}" for c, n in zip(CHANNELS, counts))
    total = int((~torch.isfinite(x)).sum())
    raise NonFiniteError(f"{name} contains {total} non-finite values;{detail}")


def apply_physical_constraints(
    y_phys: torch.Tensor,
    t2m_margin_k: float = T2M_SAFETY_MARGIN_K,
    clamp_tcwv: bool = True,
    check_finite: bool = True,
) -> torch.Tensor:
    """Final physical safeguards on a physical-unit prediction ``(..., 6, 32, 36)``.

    Order of operations:
      1. If ``check_finite``: raise ``NonFiniteError`` on any NaN/Inf. Corrupted
         output is never repaired into a plausible-looking forecast.
      2. ``tp >= 0`` (hard clamp on the FINAL physical value, not Softplus on a
         residual increment, which would add a positive drizzle bias each step).
      3. ``t2m`` clamped to ``[233.15 + m, 333.15 - m]`` K with ``m = t2m_margin_k``
         so float32 rounding can never land outside the official bounds.
      4. ``tcwv >= 0`` if ``clamp_tcwv`` (team rule; not an official check).

    The clamps are differentiable almost everywhere (zero gradient outside the
    bounds), so the function may also be used at the end of a training rollout.
    Returns a new tensor; the input is not modified.
    """
    _check_dynamic(y_phys, "y_phys")
    if not 0.0 <= t2m_margin_k < (T2M_MAX_K - T2M_MIN_K) / 2:
        raise ValueError("t2m_margin_k must be in [0, 50) K")
    if check_finite:
        assert_finite(y_phys, "prediction")
    t2m = y_phys[..., T2M_IDX : T2M_IDX + 1, :, :].clamp(T2M_MIN_K + t2m_margin_k, T2M_MAX_K - t2m_margin_k)
    tp = y_phys[..., TP_IDX : TP_IDX + 1, :, :].clamp_min(0.0)
    mid = y_phys[..., U10_IDX:TCWV_IDX, :, :]  # u10, v10, msl unchanged
    tcwv = y_phys[..., TCWV_IDX : TCWV_IDX + 1, :, :]
    if clamp_tcwv:
        tcwv = tcwv.clamp_min(0.0)
    return torch.cat([t2m, tp, mid, tcwv], dim=-3)
