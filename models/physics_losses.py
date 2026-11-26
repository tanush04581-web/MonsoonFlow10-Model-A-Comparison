"""Physics-aware composite loss for MonsoonFlow-10 autoregressive rollouts.

Aligned directly with ``monsoon_r1.scoring``:
1. Latitude-weighted (cos(lat)) & std-normalized forecast error (Huber + MSE blend).
2. Heavy-rainfall loss (>= 10.0 mm/day CSI target) combining intensity weighting
   and log1p(tp) matching, plus penalty for unphysical negative precipitation.
3. Spatial variance ratio regularizer targeting [0.58, 1.80] across lead steps
   to prevent MSE over-smoothing at Day 8 without noise inflation (> 2.5x).
4. 2D finite-difference spatial gradient preservation loss for coastal/orographic
   rainbands and pressure fronts.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import transforms as T


def _lat_weights_tensor(lat_start: float = 6.5, nlat: int = T.NLAT) -> torch.Tensor:
    lats = torch.arange(nlat, dtype=torch.float32) + lat_start
    w = torch.cos(lats * (math.pi / 180.0))
    return w / w.mean()


class MonsoonPhysicsLoss(nn.Module):
    """Multi-objective spatiotemporal forecast loss."""

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        w_forecast: float = 1.0,
        w_rain: float = 0.55,
        w_var: float = 0.35,
        w_grad: float = 0.25,
        w_neg: float = 0.20,
        var_min_ratio: float = 0.58,
        var_max_ratio: float = 1.80,
        rain_thresh_mm: float = 10.0,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, T.N_DYNAMIC, 1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, T.N_DYNAMIC, 1, 1).clamp_min(T.STD_FLOOR))
        self.register_buffer("lat_w", _lat_weights_tensor().reshape(1, 1, 1, T.NLAT, 1))
        # Slightly emphasize tp/u10/v10/t2m evenly while ensuring balanced multi-channel skill
        self.register_buffer("ch_w", torch.tensor([1.0, 1.15, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).reshape(1, 1, T.N_DYNAMIC, 1, 1))

        self.w_forecast = float(w_forecast)
        self.w_rain = float(w_rain)
        self.w_var = float(w_var)
        self.w_grad = float(w_grad)
        self.w_neg = float(w_neg)
        self.var_min_ratio = float(var_min_ratio)
        self.var_max_ratio = float(var_max_ratio)
        self.rain_thresh_mm = float(rain_thresh_mm)

    def forward(
        self,
        pred_norm_raw: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total physics-aware loss over normalized rollouts ``(B, K, 6, 32, 36)``."""
        # 1. Lead-weighted forecast error (mildly emphasize Day 1..K uniformly with Huber + MSE)
        diff = pred_norm_raw - target_norm
        mse_term = diff.pow(2)
        huber_term = F.smooth_l1_loss(pred_norm_raw, target_norm, reduction="none", beta=0.75)
        per_elem = (0.65 * mse_term + 0.35 * huber_term) * self.lat_w * self.ch_w
        l_forecast = per_elem.mean()

        # Convert precipitation channel to physical mm/day for heavy-rain and non-negativity losses
        tp_mean = self.mean[:, :, T.TP_IDX]
        tp_std = self.std[:, :, T.TP_IDX]
        pred_tp_raw = pred_norm_raw[:, :, T.TP_IDX] * tp_std + tp_mean
        true_tp = (target_norm[:, :, T.TP_IDX] * tp_std + tp_mean).clamp_min(0.0)
        pred_tp_pos = pred_tp_raw.clamp_min(0.0)

        # Penalize negative predicted rain before clamping
        l_neg = F.relu(-pred_tp_raw).pow(2).mean() / (tp_std.squeeze() ** 2 + 1e-6)

        # 2. Heavy-rainfall intensity + log1p + neighborhood CSI-friendly loss
        log_err = F.smooth_l1_loss(torch.log1p(pred_tp_pos), torch.log1p(true_tp), reduction="none", beta=0.5)
        heavy_mask = (true_tp >= self.rain_thresh_mm).to(pred_tp_pos.dtype)
        rain_weights = (1.0 + 2.5 * heavy_mask + 0.03 * true_tp.clamp(max=80.0)) * self.lat_w.squeeze(2)
        l_rain = (log_err * rain_weights).mean()

        # Soft CSI surrogate at 10 mm/day (gentle weight keeps CSI high without over-predicting false alarms)
        p_prob = torch.sigmoid((pred_tp_pos - self.rain_thresh_mm) * 0.5)
        t_bin = heavy_mask
        inter = (p_prob * t_bin).sum()
        union = (p_prob + t_bin - p_prob * t_bin).sum() + 1e-4
        l_soft_csi = 1.0 - (inter + 1e-4) / union
        l_rain_total = l_rain + 0.20 * l_soft_csi

        # 3. Anti-blur spatial variance ratio across rollout days
        # Compute spatial variance per (B, K, C)
        pred_var = pred_norm_raw.var(dim=(-2, -1), unbiased=False)
        true_var = target_norm.var(dim=(-2, -1), unbiased=False)
        ratio = pred_var / (true_var + 0.05)
        under_blur = F.relu(self.var_min_ratio - ratio).pow(2)
        over_inflate = F.relu(ratio - self.var_max_ratio).pow(2)
        l_var = (under_blur + 0.5 * over_inflate).mean()

        # 4. Spatial gradient preservation (finite differences in y and x)
        dy_p = pred_norm_raw[..., 1:, :] - pred_norm_raw[..., :-1, :]
        dy_t = target_norm[..., 1:, :] - target_norm[..., :-1, :]
        dx_p = pred_norm_raw[..., :, 1:] - pred_norm_raw[..., :, :-1]
        dx_t = target_norm[..., :, 1:] - target_norm[..., :, :-1]
        l_grad = 0.5 * (F.l1_loss(dy_p, dy_t) + F.l1_loss(dx_p, dx_t))

        total = (
            self.w_forecast * l_forecast
            + self.w_rain * l_rain_total
            + self.w_var * l_var
            + self.w_grad * l_grad
            + self.w_neg * l_neg
        )
        terms = {
            "total": float(total.detach().item()),
            "forecast": float(l_forecast.detach().item()),
            "rain": float(l_rain_total.detach().item()),
            "var": float(l_var.detach().item()),
            "grad": float(l_grad.detach().item()),
            "neg": float(l_neg.detach().item()),
        }
        return total, terms
