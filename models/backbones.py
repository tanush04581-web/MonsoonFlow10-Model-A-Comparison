"""Lightweight spatiotemporal architectures for MonsoonFlow-10 (32x36 India grid).

All candidate architectures share ``BaseMonsoonForecaster``, guaranteeing:
1. Identical physical-unit wire contract via ``models.transforms``:
   - Inputs: ``init_states_phys`` ``(B, 3, 6, 32, 36)``, ``doy_last`` ``(B, 2)``,
     optional ``perturbation`` ``(6, 32, 36)`` or ``(B, 6, 32, 36)``.
   - Outputs: ``(B, steps, 6, 32, 36)`` in physical units with ``tp >= 0``,
     ``tcwv >= 0``, and ``t2m in [233.16, 333.14] K``.
2. Exact zero-shock consistency (``perturbation=None`` vs ``zeros`` match within
   ``atol < 1e-6``, deterministic ``eval()`` mode with no stochastic layers).
3. Physically bounded causal receptive field per 1-day step (4-6 cells/step,
   well below the 8 cells/day teleportation limit and concentrated within
   ``DILATE=3`` on Day 1) using 3x3 spatial + 1x1 channel-mixing blocks.

Candidates implemented for Phase 3/4 comparison:
- ``mf10_convlstm``: Residual CNN encoder -> 2-layer ConvLSTM -> FiLM decoder.
- ``st_rescnn``: Markovian 3-day sliding-window Residual Spatiotemporal CNN
  with FiLM target-DOY seasonal modulation.
- ``ms_resunet``: Shallow 1-level (32x36 + 16x18) Residual U-Net over the 3-day
  sliding window with decoupled precipitation refinement and FiLM conditioning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import transforms as T

ARCHITECTURES: Tuple[str, ...] = ("mf10_convlstm", "st_rescnn", "ms_resunet")


def _make_coord_grid(height: int = T.NLAT, width: int = T.NLON) -> torch.Tensor:
    """Create normalized static spatial coordinates (2, H, W) in [-1, 1]."""
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([yy, xx], dim=0)


def _norm_groups(channels: int, target_groups: int = 8) -> int:
    """Largest valid GroupNorm group count dividing ``channels`` up to ``target_groups``."""
    for g in range(min(target_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock2d(nn.Module):
    """Resolution-preserving 3x3 + pointwise residual block with GroupNorm and SiLU."""

    def __init__(self, channels: int, dilation: int = 1, gn_groups: int = 8, second_kernel: int = 1) -> None:
        super().__init__()
        g = _norm_groups(channels, gn_groups)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(g, channels)
        pad2 = second_kernel // 2
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=second_kernel, padding=pad2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class FiLMResBlock2d(nn.Module):
    """Residual 3x3 + pointwise block modulated by cyclic DOY conditioning."""

    def __init__(
        self,
        channels: int,
        cond_dim: int = 2,
        dilation: int = 1,
        gn_groups: int = 8,
        second_kernel: int = 1,
    ) -> None:
        super().__init__()
        g = _norm_groups(channels, gn_groups)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(g, channels)
        pad2 = second_kernel // 2
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=second_kernel, padding=pad2)
        self.act = nn.SiLU()
        self.film = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = self.norm1(x) * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.conv1(self.act(h))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class SeasonalAnchorNet(nn.Module):
    """Learned seasonal-topographic equilibrium ``B(static, coords, doy)`` in normalized space."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, T.N_DYNAMIC, kernel_size=1),
        )
        self.alpha_logit = nn.Parameter(torch.full((1, T.N_DYNAMIC, 1, 1), 1.5))

    def background(self, env: torch.Tensor, doy_pair: torch.Tensor) -> torch.Tensor:
        doy_planes = doy_pair[:, :, None, None].expand(env.shape[0], 2, T.NLAT, T.NLON)
        return self.net(torch.cat([env, doy_planes], dim=1))

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)


class ConvLSTMCell(nn.Module):
    """2D Convolutional LSTM cell preserving full 32x36 spatial resolution."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
        )
        self.gate_norm = nn.GroupNorm(_norm_groups(4 * hidden_channels, 8), 4 * hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        if state is None:
            h_prev = x.new_zeros((b, self.hidden_channels, h, w))
            c_prev = x.new_zeros((b, self.hidden_channels, h, w))
        else:
            h_prev, c_prev = state

        combined = torch.cat([x, h_prev], dim=1)
        i, f, g, o = self.gate_norm(self.gates(combined)).chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + 1.0)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class BaseMonsoonForecaster(nn.Module):
    """Shared normalization, static conditioning, DOY rotation, and rollout contract."""

    arch_name: str = "base"

    def __init__(
        self,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        static_raw: torch.Tensor | None = None,
        orography_scale_m: float = T.OROGRAPHY_SCALE_M,
        use_seasonal_anchor: bool = False,
    ) -> None:
        super().__init__()
        self.orography_scale_m = float(orography_scale_m)
        self.use_seasonal_anchor = bool(use_seasonal_anchor)

        default_mean = torch.zeros(T.N_DYNAMIC, dtype=torch.float32)
        default_std = torch.ones(T.N_DYNAMIC, dtype=torch.float32)
        default_static = torch.zeros(T.N_STATIC, T.NLAT, T.NLON, dtype=torch.float32)

        self.register_buffer(
            "stats_mean",
            torch.as_tensor(mean, dtype=torch.float32).reshape(T.N_DYNAMIC) if mean is not None else default_mean,
        )
        self.register_buffer(
            "stats_std",
            torch.as_tensor(std, dtype=torch.float32).reshape(T.N_DYNAMIC) if std is not None else default_std,
        )
        self.register_buffer(
            "static_raw",
            torch.as_tensor(static_raw, dtype=torch.float32).reshape(T.N_STATIC, T.NLAT, T.NLON)
            if static_raw is not None
            else default_static,
        )
        self.register_buffer("coord_grid", _make_coord_grid(T.NLAT, T.NLON))
        self.seasonal_anchor = SeasonalAnchorNet(hidden=32) if self.use_seasonal_anchor else None

    def set_artifacts(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        static_raw: torch.Tensor,
    ) -> None:
        """Bind training-derived channel stats and static terrain buffers."""
        with torch.no_grad():
            self.stats_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).reshape(T.N_DYNAMIC))
            self.stats_std.copy_(torch.as_tensor(std, dtype=torch.float32).reshape(T.N_DYNAMIC))
            self.static_raw.copy_(torch.as_tensor(static_raw, dtype=torch.float32).reshape(T.N_STATIC, T.NLAT, T.NLON))

    def _norm_bounds(self, like: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m = self.stats_mean.to(device=like.device, dtype=like.dtype)
        s = self.stats_std.to(device=like.device, dtype=like.dtype).clamp_min(T.STD_FLOOR)
        t2m_lo = (T.T2M_MIN_K + T.T2M_SAFETY_MARGIN_K - m[T.T2M_IDX]) / s[T.T2M_IDX]
        t2m_hi = (T.T2M_MAX_K - T.T2M_SAFETY_MARGIN_K - m[T.T2M_IDX]) / s[T.T2M_IDX]
        tp_lo = (0.0 - m[T.TP_IDX]) / s[T.TP_IDX]
        tcwv_lo = (0.0 - m[T.TCWV_IDX]) / s[T.TCWV_IDX]
        return t2m_lo, t2m_hi, tp_lo, tcwv_lo

    def _clamp_feedback_norm(self, x_norm: torch.Tensor) -> torch.Tensor:
        t2m_lo, t2m_hi, tp_lo, tcwv_lo = self._norm_bounds(x_norm)
        t2m = x_norm[:, 0:1].clamp(t2m_lo, t2m_hi)
        tp = x_norm[:, 1:2].clamp_min(tp_lo)
        mid = x_norm[:, 2:5]
        tcwv = x_norm[:, 5:6].clamp_min(tcwv_lo)
        return torch.cat([t2m, tp, mid, tcwv], dim=1)

    def _prepared_static(self, batch: int, like: torch.Tensor) -> torch.Tensor:
        st = T.prepare_static(self.static_raw.to(device=like.device, dtype=like.dtype), self.orography_scale_m)
        coords = self.coord_grid.to(device=like.device, dtype=like.dtype)
        env = torch.cat([st, coords], dim=0)
        return env.unsqueeze(0).expand(batch, 4, T.NLAT, T.NLON)

    def _combine_step_update(
        self,
        cur_state: torch.Tensor,
        delta: torch.Tensor,
        env: torch.Tensor,
        cur_doy: torch.Tensor,
        tgt_doy: torch.Tensor,
    ) -> torch.Tensor:
        if self.seasonal_anchor is None:
            return cur_state + delta
        bg_cur = self.seasonal_anchor.background(env, cur_doy)
        bg_tgt = self.seasonal_anchor.background(env, tgt_doy)
        alpha = self.seasonal_anchor.alpha().to(dtype=cur_state.dtype, device=cur_state.device)
        return bg_tgt + alpha * (cur_state - bg_cur) + delta

    def rollout_normalized(
        self,
        hist_dyn_norm: torch.Tensor,
        doy_last: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        init_states_phys: torch.Tensor,
        doy_last: Union[torch.Tensor, Sequence[float]],
        steps: int = 10,
        perturbation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """End-to-end physical-unit forward rollout ``(B, steps, 6, 32, 36)``."""
        single = init_states_phys.dim() == 4
        x_phys = init_states_phys.unsqueeze(0) if single else init_states_phys
        if x_phys.dim() != 5 or tuple(x_phys.shape[1:]) != (T.T_IN, T.N_DYNAMIC, T.NLAT, T.NLON):
            raise ValueError(
                f"init_states_phys must be (B, 3, 6, 32, 36) or (3, 6, 32, 36), got {tuple(init_states_phys.shape)}"
            )
        T.assert_finite(x_phys, "init_states_phys")

        if perturbation is not None:
            p = perturbation.to(device=x_phys.device, dtype=x_phys.dtype)
            if p.dim() == 3:
                p = p.unsqueeze(0)
            if p.dim() != 4 or tuple(p.shape[1:]) != (T.N_DYNAMIC, T.NLAT, T.NLON):
                raise ValueError(f"perturbation must be (6, 32, 36) or (B, 6, 32, 36), got {tuple(perturbation.shape)}")
            T.assert_finite(p, "perturbation")
            x_last = x_phys[:, -1] + p
            x_phys = torch.cat([x_phys[:, :-1], x_last.unsqueeze(1)], dim=1)

        doy_b = T._as_batched_doy(doy_last, x_phys.shape[0], x_phys)
        hist_norm = T.normalize_dynamic(x_phys, self.stats_mean, self.stats_std, tp_transform="linear")
        preds_norm = self.rollout_normalized(hist_norm, doy_b, steps=steps)
        preds_phys = T.denormalize_dynamic(preds_norm, self.stats_mean, self.stats_std, tp_transform="linear")
        preds_phys = T.apply_physical_constraints(preds_phys, check_finite=True)
        return preds_phys.squeeze(0) if single else preds_phys

    def model_config(self) -> dict[str, Any]:
        raise NotImplementedError


class MF10ConvLSTM(BaseMonsoonForecaster):
    """Candidate A: Residual CNN Encoder -> 2-Layer ConvLSTM -> FiLM Decoder (radius=4/step)."""

    arch_name = "mf10_convlstm"

    def __init__(
        self,
        hidden_channels: int = 56,
        enc_blocks: int = 1,
        dec_blocks: int = 1,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        static_raw: torch.Tensor | None = None,
        orography_scale_m: float = T.OROGRAPHY_SCALE_M,
        use_seasonal_anchor: bool = False,
    ) -> None:
        super().__init__(
            mean=mean,
            std=std,
            static_raw=static_raw,
            orography_scale_m=orography_scale_m,
            use_seasonal_anchor=use_seasonal_anchor,
        )
        self.hidden_channels = int(hidden_channels)
        self.enc_blocks = int(enc_blocks)
        self.dec_blocks = int(dec_blocks)

        in_ch = T.N_DYNAMIC + T.N_STATIC + 2 + T.N_DOY
        self.stem = nn.Conv2d(in_ch, hidden_channels, kernel_size=1)
        self.encoder = nn.Sequential(*[ResBlock2d(hidden_channels, second_kernel=1) for _ in range(enc_blocks)])
        self.lstm1 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=3)
        self.lstm2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=3)
        self.decoder_blocks = nn.ModuleList(
            [FiLMResBlock2d(hidden_channels, cond_dim=2, second_kernel=1) for _ in range(dec_blocks)]
        )
        mid_ch = max(28, hidden_channels // 2)
        self.head = nn.Sequential(
            nn.GroupNorm(_norm_groups(hidden_channels, 8), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, mid_ch, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid_ch, T.N_DYNAMIC, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _encode_frame(self, dyn: torch.Tensor, env: torch.Tensor, doy_pair: torch.Tensor) -> torch.Tensor:
        doy_planes = doy_pair[:, :, None, None].expand(dyn.shape[0], 2, T.NLAT, T.NLON)
        x = torch.cat([dyn, env, doy_planes], dim=1)
        return self.encoder(F.silu(self.stem(x)))

    def rollout_normalized(
        self,
        hist_dyn_norm: torch.Tensor,
        doy_last: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        batch = hist_dyn_norm.shape[0]
        env = self._prepared_static(batch, hist_dyn_norm)
        hist_doys = T.rotate_doy_sincos(doy_last, list(T.HISTORY_OFFSETS))

        s1 = s2 = None
        for t_idx in range(T.T_IN):
            feat = self._encode_frame(hist_dyn_norm[:, t_idx], env, hist_doys[:, t_idx])
            s1 = self.lstm1(feat, s1)
            s2 = self.lstm2(s1[0], s2)

        cur_state = hist_dyn_norm[:, -1]
        cur_doy = doy_last
        preds = []
        for k in range(steps):
            tgt_doy = T.rotate_doy_sincos(doy_last, k + 1)
            feat = self._encode_frame(cur_state, env, tgt_doy)
            s1 = self.lstm1(feat, s1)
            s2 = self.lstm2(s1[0], s2)
            h = s2[0]
            for blk in self.decoder_blocks:
                h = blk(h, tgt_doy)
            delta = self.head(h)
            raw_next = self._combine_step_update(cur_state, delta, env, cur_doy, tgt_doy)
            preds.append(raw_next)
            cur_state = self._clamp_feedback_norm(raw_next)
            cur_doy = tgt_doy

        return torch.stack(preds, dim=1)

    def model_config(self) -> dict[str, Any]:
        return {
            "arch": self.arch_name,
            "hidden_channels": self.hidden_channels,
            "enc_blocks": self.enc_blocks,
            "dec_blocks": self.dec_blocks,
            "orography_scale_m": self.orography_scale_m,
            "use_seasonal_anchor": self.use_seasonal_anchor,
        }


class SpatioTemporalResCNN(BaseMonsoonForecaster):
    """Candidate B: 3-day sliding-window Residual Spatiotemporal CNN (radius=4/step)."""

    arch_name = "st_rescnn"

    def __init__(
        self,
        hidden_channels: int = 72,
        num_blocks: int = 4,
        dilations: Sequence[int] = (1, 1, 1, 1),
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        static_raw: torch.Tensor | None = None,
        orography_scale_m: float = T.OROGRAPHY_SCALE_M,
        use_seasonal_anchor: bool = False,
    ) -> None:
        super().__init__(
            mean=mean,
            std=std,
            static_raw=static_raw,
            orography_scale_m=orography_scale_m,
            use_seasonal_anchor=use_seasonal_anchor,
        )
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.dilations = tuple(int(d) for d in dilations)
        if len(self.dilations) != self.num_blocks:
            raise ValueError("len(dilations) must match num_blocks")

        in_ch = T.T_IN * T.N_DYNAMIC + T.N_STATIC + 2 + 2 * T.N_DOY
        self.stem = nn.Conv2d(in_ch, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FiLMResBlock2d(hidden_channels, cond_dim=4, dilation=d, second_kernel=1) for d in self.dilations]
        )
        mid_ch = max(36, hidden_channels // 2)
        self.head = nn.Sequential(
            nn.GroupNorm(_norm_groups(hidden_channels, 8), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, mid_ch, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid_ch, T.N_DYNAMIC, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def rollout_normalized(
        self,
        hist_dyn_norm: torch.Tensor,
        doy_last: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        batch = hist_dyn_norm.shape[0]
        env = self._prepared_static(batch, hist_dyn_norm)
        f0, f1, f2 = hist_dyn_norm[:, 0], hist_dyn_norm[:, 1], hist_dyn_norm[:, 2]

        preds = []
        for k in range(steps):
            cur_doy = T.rotate_doy_sincos(doy_last, k)
            tgt_doy = T.rotate_doy_sincos(doy_last, k + 1)
            cond = torch.cat([cur_doy, tgt_doy], dim=-1)
            doy_planes = cond[:, :, None, None].expand(batch, 4, T.NLAT, T.NLON)

            x = torch.cat([f0, f1, f2, env, doy_planes], dim=1)
            h = F.silu(self.stem(x))
            for blk in self.blocks:
                h = blk(h, cond)
            delta = self.head(h)
            raw_next = self._combine_step_update(f2, delta, env, cur_doy, tgt_doy)
            preds.append(raw_next)
            next_clamped = self._clamp_feedback_norm(raw_next)
            f0, f1, f2 = f1, f2, next_clamped

        return torch.stack(preds, dim=1)

    def model_config(self) -> dict[str, Any]:
        return {
            "arch": self.arch_name,
            "hidden_channels": self.hidden_channels,
            "num_blocks": self.num_blocks,
            "dilations": list(self.dilations),
            "orography_scale_m": self.orography_scale_m,
            "use_seasonal_anchor": self.use_seasonal_anchor,
        }


class MultiScaleResUNet(BaseMonsoonForecaster):
    """Candidate C: Shallow 1-level Residual U-Net (32x36 + 16x18) with decoupled tp head."""

    arch_name = "ms_resunet"

    def __init__(
        self,
        base_channels: int = 48,
        coarse_channels: int = 72,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        static_raw: torch.Tensor | None = None,
        orography_scale_m: float = T.OROGRAPHY_SCALE_M,
        use_seasonal_anchor: bool = False,
    ) -> None:
        super().__init__(
            mean=mean,
            std=std,
            static_raw=static_raw,
            orography_scale_m=orography_scale_m,
            use_seasonal_anchor=use_seasonal_anchor,
        )
        self.base_channels = int(base_channels)
        self.coarse_channels = int(coarse_channels)

        in_ch = T.T_IN * T.N_DYNAMIC + T.N_STATIC + 2 + 2 * T.N_DOY
        self.stem = nn.Conv2d(in_ch, base_channels, kernel_size=1)
        self.fine_enc = FiLMResBlock2d(base_channels, cond_dim=4, dilation=1, second_kernel=1)

        self.down = nn.Conv2d(base_channels, coarse_channels, kernel_size=3, stride=2, padding=1)
        self.coarse1 = FiLMResBlock2d(coarse_channels, cond_dim=4, dilation=1, second_kernel=1)
        self.up_proj = nn.Conv2d(coarse_channels, base_channels, kernel_size=1)

        self.fuse = FiLMResBlock2d(base_channels * 2, cond_dim=4, dilation=1, second_kernel=1)

        self.dyn_head = nn.Sequential(
            nn.GroupNorm(_norm_groups(base_channels * 2, 8), base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, 5, kernel_size=1),
        )
        self.tp_head = nn.Sequential(
            nn.GroupNorm(_norm_groups(base_channels * 2, 8), base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.dyn_head[-1].weight)
        nn.init.zeros_(self.dyn_head[-1].bias)
        nn.init.zeros_(self.tp_head[-1].weight)
        nn.init.zeros_(self.tp_head[-1].bias)

    def rollout_normalized(
        self,
        hist_dyn_norm: torch.Tensor,
        doy_last: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        batch = hist_dyn_norm.shape[0]
        env = self._prepared_static(batch, hist_dyn_norm)
        f0, f1, f2 = hist_dyn_norm[:, 0], hist_dyn_norm[:, 1], hist_dyn_norm[:, 2]

        preds = []
        for k in range(steps):
            cur_doy = T.rotate_doy_sincos(doy_last, k)
            tgt_doy = T.rotate_doy_sincos(doy_last, k + 1)
            cond = torch.cat([cur_doy, tgt_doy], dim=-1)
            doy_planes = cond[:, :, None, None].expand(batch, 4, T.NLAT, T.NLON)

            x = torch.cat([f0, f1, f2, env, doy_planes], dim=1)
            h_fine = self.fine_enc(F.silu(self.stem(x)), cond)

            h_coarse = self.coarse1(F.silu(self.down(h_fine)), cond)
            h_up = F.interpolate(h_coarse, size=(T.NLAT, T.NLON), mode="bilinear", align_corners=False)
            h_up = self.up_proj(h_up)

            h = self.fuse(torch.cat([h_fine, h_up], dim=1), cond)

            d_other = self.dyn_head(h)
            d_tp = self.tp_head(h)
            delta = torch.cat([d_other[:, 0:1], d_tp, d_other[:, 1:5]], dim=1)

            raw_next = self._combine_step_update(f2, delta, env, cur_doy, tgt_doy)
            preds.append(raw_next)
            next_clamped = self._clamp_feedback_norm(raw_next)
            f0, f1, f2 = f1, f2, next_clamped

        return torch.stack(preds, dim=1)

    def model_config(self) -> dict[str, Any]:
        return {
            "arch": self.arch_name,
            "base_channels": self.base_channels,
            "coarse_channels": self.coarse_channels,
            "orography_scale_m": self.orography_scale_m,
            "use_seasonal_anchor": self.use_seasonal_anchor,
        }


def build_model(
    arch: str,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    static_raw: torch.Tensor | None = None,
    **kwargs: Any,
) -> BaseMonsoonForecaster:
    """Instantiate a candidate forecaster by architecture name."""
    if arch == "mf10_convlstm":
        return MF10ConvLSTM(mean=mean, std=std, static_raw=static_raw, **kwargs)
    if arch == "st_rescnn":
        return SpatioTemporalResCNN(mean=mean, std=std, static_raw=static_raw, **kwargs)
    if arch == "ms_resunet":
        return MultiScaleResUNet(mean=mean, std=std, static_raw=static_raw, **kwargs)
    raise ValueError(f"Unknown architecture '{arch}'; expected one of {ARCHITECTURES}")


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    map_location: str = "cpu",
) -> Tuple[BaseMonsoonForecaster, dict[str, Any]]:
    """Load a trained MonsoonFlow-10 model safely from a checkpoint dict."""
    ckpt = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=True)
    cfg = dict(ckpt["model_config"])
    arch = cfg.pop("arch")
    stats = ckpt["stats"]
    model = build_model(
        arch=arch,
        mean=stats["mean"],
        std=stats["std"],
        static_raw=None,
        **cfg,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt
