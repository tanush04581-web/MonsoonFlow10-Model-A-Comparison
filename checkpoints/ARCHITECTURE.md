# MonsoonFlow-10 Primary Model Architecture (`checkpoints/model.pt`)

> **Status:** Provisional promotion pending independent Agent 5 verification  
> **Promoted Checkpoint:** `checkpoints/model.pt` (from `checkpoints/runs/exp4_msresunet_anchor/best.pt`, Epoch 22)  
> **Fallback Checkpoint:** `checkpoints/runs/exp5_msresunet_direct_22ep_control/best.pt`  
> **Architecture:** `MultiScaleResUNet` (`ms_resunet`) + `SeasonalAnchorNet` (`use_seasonal_anchor=True`)  
> **Parameter Count:** `331,786` (`1,362,884` bytes on disk, FP32)

---

## 1. Model Overview

`MultiScaleResUNet` (`models/backbones.py`) is a compact, shallow 1-level spatiotemporal residual U-Net operating autoregressively on the $32 \times 36$ India domain ($6.5^\circ\text{N}..37.5^\circ\text{N}, 65.5^\circ\text{E}..100.5^\circ\text{E}$). It pairs a high-resolution $32 \times 36$ skip pathway with a single $2\times$-downsampled $16 \times 18$ synoptic branch, a decoupled precipitation prediction head (`tp_head`), FiLM cyclic seasonal modulation, and `SeasonalAnchorNet`.

### Forward Rollout Diagram (Per Autoregressive Step $k = 0 \dots \text{steps}-1$)

```text
Inputs (Physical Units on Wire):
  init_states_phys: (B, 3, 6, 32, 36)  [t_-2, t_-1, t_0]
  perturbation:     (6, 32, 36) or None -> added physically to init_states_phys[:, -1]
  doy_last:         (B, 2) [sin(2*pi*doy/365), cos(2*pi*doy/365)] for t_0
        │
        ▼ normalize_dynamic (linear z-score via embedded stats_mean / stats_std)
  3-Day Sliding Window [f0, f1, f2]: each (B, 6, 32, 36)
        │
        ▼ Concat [f0(6), f1(6), f2(6), lsm(1), oro/8000(1), coord_grid(2), cur_doy(2), tgt_doy(2)] = 26 ch
  ┌─ Stem 1x1 (26 -> 48) -> fine_enc: FiLMResBlock2d(48, cond=4) [32x36] ──────────────┐
  │                                                                                    │
  └─▶ down Conv3x3 s=2 (48 -> 72) -> coarse1: FiLMResBlock2d(72) [16x18]               │
                                          │                                            │
                                    bilinear up -> 1x1 proj (72 -> 48)                 │
                                          ▼                                            │
                                    concat [fine(48), up(48)] = 96 ch ◄────────────────┘
                                          │
                                          ▼ fuse: FiLMResBlock2d(96, cond=4)
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                 dyn_head (5 ch)                 tp_head (3x3 conv, 1 ch)
             [t2m, u10, v10, msl, tcwv]                  [tp]
                          └───────────────┬───────────────┘
                                          ▼ delta_norm (B, 6, 32, 36)
  SeasonalAnchorNet:
    B(static, coords, doy) -> bg_cur, bg_tgt; alpha = sigmoid(alpha_logit)
    raw_next = bg_tgt + alpha * (f2 - bg_cur) + delta_norm
        │
        ▼ _clamp_feedback_norm -> slide [f1, f2, next_clamped] for step k+1
        │
        ▼ denormalize_dynamic -> apply_physical_constraints
Output (Physical Units): (B, steps, 6, 32, 36)
```

---

## 2. Key Engineering Properties

1. **Bounded Causal Receptive Field:** Shallow `3x3` + `1x1` blocks keep Day-1 shock influence concentrated inside `DILATE=3` (`near_frac_day1 = 0.968–0.993 >= 0.60`) and prevent teleportation (`far_frac_max <= 0.0143 < 0.05`).
2. **Seasonal-Attractor Anomaly Dynamics (`SeasonalAnchorNet`):** Learns a smooth geographic/seasonal equilibrium `B(static, coords, doy)` and channel-specific persistence factors `alpha = {t2m: 0.8720, tp: 0.7842, u10: 0.8169, v10: 0.7967, msl: 0.8354, tcwv: 0.8398}`, preventing Day-8 drift against Climatology (`Day-8 SS` improved from `-0.0148` in `exp5` to `+0.0070` in `exp4`).
3. **Exact Zero-Shock Parity & Validity:** Deterministic `eval()` mode with zero stochastic layers guarantees `0.0` exact difference for `perturbation=None` vs `zeros`, and `apply_physical_constraints` enforces `tp >= 0`, `tcwv >= 0`, and `t2m in [233.16, 333.14] K`.
4. **Anti-Cheat Import Boundary:** `models/` imports strictly `torch`, `math`, `pathlib`, and `typing` (0 date literals, 0 `datetime` calls, 0 external weather imports).

---

## 3. Training Curriculum & Reproducibility

- **Dataset & Splits:** Strictly chronological `data/raw/splits/`: Train `2005–2018` (`5113` days / `5101` windows), Validation `2019–2020` (`731` days / `719` windows), Sealed Test `2021` (`365` days, untouched).
- **Curriculum (`[4, 4, 4, 10]` = `22` epochs):** Stage 1 (`k=1`, lr `1.5e-3`), Stage 2 (`k=3`, lr `9.0e-4`), Stage 3 (`k=5`, lr `6.0e-4`), Stage 4 (`k=10`, lr `3.5e-4`), cosine decay per stage, `AdamW(weight_decay=1e-4)`, `batch_size=24`, `grad_clip=1.0`, `seed=2026`, FP32 on NVIDIA GeForce RTX 4060 Laptop GPU (`969.7 s` total, `2,676.8 MB` peak VRAM).
- **Composite Loss (`MonsoonPhysicsLoss`):** Latitude/std-weighted Huber+MSE (`1.0`) + heavy-rain `log1p`/`>=10 mm/day` CSI surrogate (`0.45`) + Day-8 spatial variance ratio floor/ceiling `[0.58, 1.80]` (`0.35`) + 2D spatial gradient matching (`0.25`) + negative rain penalty (`0.20`).

---

## 4. Validation Summary (2019–2020, 719 Windows, Provisional)

- **Official Score70:** `66.29 / 70` (`RMSE = 26.61/30 [SS=+0.1064]`, `CSI = 15.00/15 [CSI_SS=+0.1110]`, `AntiBlur = 10.00/10`, `Causality = 14.69/15`).
- **Lead-Day Skill (`SS` vs `min(Persist, Clim)`):** Day 1 `+0.2149` (`CSI=0.5315`), Day 4 `+0.0974` (`CSI=0.3668`), Day 8 `+0.0070` (`CSI=0.3277`).
- **Day-8 Variance Ratios:** `t2m=0.994`, `tp=0.881`, `u10=0.857`, `v10=0.768`, `msl=0.948`, `tcwv=0.920` (all inside `[0.40, 2.50]`).
- **CPU Inference:** `171–200 ms` median on 4 CPU cores (`~403 MiB` RSS), `0.0` zero-shock diff, `719/719` valid rollouts.

---

## 5. Known Limitations

1. **Day-8 Seasonal Bias Drift:** Mean bias at Day 8 shifts slightly cool/dry/low-pressure (`t2m -0.252 K`, `tcwv -0.709 kg/m2`, `msl -57.9 Pa`) due to anchoring toward 2005–2018 climate normals.
2. **Day-8 `t2m`/`tp` Skill:** Day-8 `t2m` (`-0.0053`) and `tp` (`-0.0170`) remain marginally below Climatology, capping `mean_rmse_ss_148` at `0.1064` (below the `0.1200` full-score cap).
3. **Winter `monsoon_stall`:** `monsoon_stall` on winter probe `2019-01-18` fails `propagates` (`5/6` probes pass; `4/4` monsoon-season probes pass).
4. **Provisional Status:** Pending independent Agent 5 evaluation and live server `schema_validator` / `repo_scan` verification once Agent-Backend integrates `checkpoints/model.pt`.
