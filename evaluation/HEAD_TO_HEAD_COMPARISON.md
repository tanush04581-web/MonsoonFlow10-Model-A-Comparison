# Independent Agent-5 QA Head-to-Head Evaluation: Model A vs Model B

**Evaluation Date:** 2026-11-26 / 2026-11-27  
**Evaluator:** Agent 5 (Independent QA)  
**Branch / Remote:** `tanush/model-a` (`335b64cbf0758bdd8998a3f6ba2892878bf80acc`) vs `origin/main` (`80648e83010ad6995e37a42446126d9731286a8f`, `https://github.com/Hardik2811-wq/MonsoonFlow10.git`)  
**Dataset Scope:** Strictly `data/raw/splits/val.npz` (`2019-01-01..2020-12-31`, **719 consecutive 10-day windows**, `t0` in `20190103..20201221`). **Sealed 2021 `data/raw/splits/test.npz` was NEVER accessed or evaluated.**  
**Source of Truth:** Pinned competition kit `resources/yug-monsoon-kit/monsoon_r1/{contract,scoring,stats}.py`, `checkpoints/climatology.npy`, and `checkpoints/stats.json`.  
**Reproducible Script & JSON:** `evaluation/qa_head_to_head.py` → `evaluation/head_to_head_report.json`  
**Status Designations:** Model A (`checkpoints/model.pt`, `exp4_msresunet_anchor`) = **PROVISIONAL PRIMARY**; `checkpoints/runs/exp5_msresunet_direct_22ep_control/best.pt` = **FALLBACK** (pending repository tracking / brief blockers documented in `evaluation/CERTIFICATION_REPORT.md`).

---

## 1. Executive Summary & Official vs Diagnostic Model Scorecard (`/70`)

**Winner: Model A (`exp4_msresunet_anchor`, PROVISIONAL PRIMARY)** decisively outperforms **Model B (`origin/main@80648e83`, `MonsoonFlow10`)** across **every official per-window and diagnostic aggregation mode**, **every forecast lead Day 1–8**, **heavy-rain CSI**, **Day-8 spatial sharpness**, **monsoon shock propagation**, **4-core CPU latency**, **peak RAM**, and **checkpoint disk footprint**.

> **Official Contract vs Diagnostic Separation:**  
> 1. Automated model score (`/70` = `Forecast 45 + Anti-Blur 10 + Causality 15`) is strictly separated from `Repo Integrity /15` + `2-Page Brief /15` (`/30`).  
> 2. `monsoon_r1.scoring.forecast_points(ss_by_window, csi_ss_by_window)` and `docs/R1_CONTRACT.md` line 50 require **per-window scoring** (so any invalid/failed window receives `0.0`). Thus **Modes `P1` and `P2` are contract-compliant per-window interpretations**, whereas **Modes `P3` and `P4` pass pooled aggregates and are DIAGNOSTIC ONLY** (definitively NOT the official judge contract).

### 1.1 All 4 `monsoon_r1.scoring` Aggregation Modes (719 Windows + 5-Window Stratified Subset)

| Aggregation Mode | Contract Compliant? | Max Pts | Persistence | Climatology | **Model A (`exp4` PROVISIONAL PRIMARY)** | **Model B (`80648e83`)** | Delta (`A − B`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode `P1`: Per-Window × Per-Lead (`qa_model_eval.py`)** | **Yes** *(strict lower bound)* | **70.0** | `8.33`*(ab)* | `10.00`*(ab)* | **`50.31 / 70`** (`fcst=25.62` [`11.45+14.17`], `ab=10.00`, `caus=14.69`) | **`29.23 / 70`** (`fcst=5.17` [`0.00+5.17`], `ab=10.00`, `caus=14.06`) | **`+21.08`** |
| **Mode `P2`: Per-Window Stacked `[3,6,32,36]` (`rmse_lat` spec)** | **Yes** *(multi-lead/window)* | **70.0** | `8.33`*(ab)* | `10.00`*(ab)* | **`66.52 / 70`** (`fcst=41.83` [`27.66+14.17`], `ab=10.00`, `caus=14.69`) | **`44.22 / 70`** (`fcst=20.16` [`14.99+5.17`], `ab=10.00`, `caus=14.06`) | **`+22.30`** |
| *Mode `P3`: Pooled-Per-Lead (`evaluate_val.py`)* | **No (DIAGNOSTIC)** | *70.0* | *—* | *—* | *`66.29 / 70`* (`fcst=41.61` [`26.61+15.00`]) | *`40.25 / 70`* (`fcst=16.19` [`7.07+9.12`]) | *`+26.04`* |
| *Mode `P4`: Pooled-All `[W,3,6,32,36]`* | **No (DIAGNOSTIC)** | *70.0* | *—* | *—* | *`69.69 / 70`* (`fcst=45.00` [`30.00+15.00`]) | *`66.20 / 70`* (`fcst=42.14` [`27.14+15.00`]) | *`+3.49`* |
| **5-Window Stratified Subset (`P1` / `P2`)** | **Yes** | **70.0** | — | — | **`60.36` (`P1`) / `69.44` (`P2`)** (`ab=9.75`, `caus=14.69`) | **`29.97` (`P1`) / `41.16` (`P2`)** (`ab=9.43`, `caus=14.06`) | **`+30.39` / `+28.28`** |
| **5-Window Stratified Subset (`P3` / `P4` DIAG)** | **No (DIAGNOSTIC)** | *70.0* | — | — | *`69.44` (`P3`) / `69.44` (`P4`)* | *`43.84` (`P3`) / `67.19` (`P4`)* | *`+25.60` / `+2.25`* |

*(Note: In the 719-window row, `causality` uses all 6 probes: `14.6875/15` for Model A, `14.0625/15` for Model B; using the 5 `qa_shock_tests.py` probes gives `-0.0625` for A and `-0.1875` for B: `P1 = 50.25 vs 29.04`, `P2 = 66.46 vs 44.03`.)*

### 1.2 Hard Admission Gates & Forecast Sub-Metrics (719 Windows)

| Dimension | Persistence | Climatology | **Model A (`exp4`)** | **Model B (`80648e83`)** |
| :--- | :---: | :---: | :---: | :---: |
| **Gate 1: Physical Validity (`719×10×6×32×36`)** | `719/719` PASS | `719/719` PASS | **`719/719` PASS (`0` viol.)** | **`719/719` PASS (`0` viol.)** |
| **Gate 2: Beats Day-4 Persistence (`mean_ss_pers` / `6/6` ch.)** | `0.000` (Ref) | Mixed | **PASS (`+0.2562`, `6/6` ch.)** | **PASS (`+0.2021`, `6/6` ch.)** |
| **Gate 3: Non-Zero Localized Shock Response** | `0/6` prop. | — | **PASS (`6/6` warm, `5/6` stall)** | **PASS (`6/6` warm, `3/6` stall)** |
| **`P1` Mean `SS` / Mean `CSI_SS`** | `-0.2646 / -0.0433` | `-0.2159 / -0.0733` | **`+0.0458 / +0.0945`** | `-0.0224 / +0.0345` |
| **`P2` Mean `SS` / Mean `CSI_SS`** | `-0.1685 / -0.0433` | `-0.0522 / -0.0733` | **`+0.1106 / +0.0945`** | `+0.05996 / +0.0345` |
| **`P3` (DIAG) Mean `SS` / Mean `CSI_SS`** | `-0.1831 / -0.0666` | `-0.1531 / -0.0885` | **`+0.1064 / +0.1109`** | `+0.0283 / +0.0608` |
| **Day-8 Anti-Blur (`scoring.anti_blur_points`)** | `8.33 / 10` (`tp` `2.604`) | `10.00 / 10` | **`10.00 / 10`** (VR `0.768..0.994`) | **`10.00 / 10`** (VR `0.696..0.991`) |

### 1.3 Why `P1` Diverges from `P2`, `P3`, and `P4` (Mathematical Audit)

1. **Why `P3` and `P4` are NOT the official contract:** `scoring.forecast_points(ss_by_window, csi_ss_by_window)` explicitly takes one scalar per window (`n_inputs_ss = W`) so that `R1_CONTRACT.md` line 50 (*"A failed or invalid window scores 0 for it, it is never skipped"*) can be enforced per window. Verified via `invalid_window_guard_test` on 5 stratified windows: corrupting window 0 (`tp = -5.0`) drops `P1` from `35.92 → 31.60` and `P2` from `45.00 → 39.47` (`corrupted_window_scored_zero = True`), which `P3`/`P4` cannot represent.
2. **Jensen's Inequality on `min(RMSE_persist, RMSE_clim)`:** Because `E[min(X, Y)] <= min(E[X], E[Y])`, taking the better reference per `(window, lead)` (`P1`) yields a **3%–20% stricter denominator** than pooled `P3`:
   - `Day 1`: `mean_w(min(rp, rc)) / min(RMSE_p, RMSE_c)` = `0.958` (`t2m`), `0.799` (`tp`), `0.943` (`u10`), `0.937` (`v10`), `0.880` (`msl`), `0.970` (`tcwv`).
   - `Day 4`: `0.886` (`t2m`), `0.847` (`tp`), `0.890` (`u10`), `0.923` (`v10`), `0.858` (`msl`), `0.913` (`tcwv`).
   - `Day 8`: `0.919` (`t2m`), `0.848` (`tp`), `0.907` (`u10`), `0.930` (`v10`), `0.885` (`msl`), `0.939` (`tcwv`).
3. **Single-Day Calm Persistence Small-Denominator Spikes in `P1` vs `P2`:** In `P1`, on individual `(window, lead)` cells where Persistence happens to be unusually close to truth (e.g., `2019-02-04` Lead 8 `msl`: `persist_rmse = 123.65 Pa` vs `clim_rmse = 584.33 Pa`), `SS = 1 - r / best` drops to `-2.782` (Model A, `467.61 Pa`) and `-3.310` (Model B, `532.95 Pa`). Stacking `[0, 3, 7]` per window (`P2`, matching `rmse_lat`'s multi-lead spec) smooths out single-day spikes (`count < -1` drops from `11` to **`0`** for Model A, `15` to `1` for Model B):

| Channel | Model A `P1` (`n=2157`): `min / p05 / med / mean / <#lt-1>` | Model A `P2` (`n=719`): `min / p05 / med / mean / <#lt-1>` | Model B `P1` (`n=2157`): `min / p05 / med / mean / <#lt-1>` | Model B `P2` (`n=719`): `min / p05 / med / mean / <#lt-1>` |
| :--- | :---: | :---: | :---: | :---: |
| `t2m` | `-0.970 / -0.303 / +0.033 / +0.015 / 0` | `-0.576 / -0.124 / +0.096 / +0.089 / 0` | `-1.176 / -0.371 / -0.022 / -0.051 / 1` | `-0.546 / -0.157 / +0.058 / +0.054 / 0` |
| `tp` | `-2.154 / -0.292 / +0.041 / +0.037 / 5` | `-0.611 / -0.103 / +0.080 / +0.076 / 0` | `-2.005 / -0.275 / +0.001 / -0.023 / 5` | `-1.117 / -0.113 / +0.035 / +0.024 / 1` |
| `u10` | `-0.885 / -0.234 / +0.056 / +0.042 / 0` | `-0.457 / -0.085 / +0.114 / +0.109 / 0` | `-0.733 / -0.242 / -0.003 / -0.019 / 0` | `-0.341 / -0.125 / +0.059 / +0.062 / 0` |
| `v10` | `-0.997 / -0.221 / +0.056 / +0.057 / 0` | `-0.295 / -0.049 / +0.113 / +0.120 / 0` | `-0.530 / -0.176 / -0.002 / -0.006 / 0` | `-0.313 / -0.052 / +0.059 / +0.066 / 0` |
| `msl` | `-2.782 / -0.335 / +0.058 / +0.044 / 6` | `-0.640 / -0.109 / +0.129 / +0.127 / 0` | `-3.310 / -0.394 / -0.012 / -0.045 / 9` | `-0.551 / -0.192 / +0.061 / +0.060 / 0` |
| `tcwv` | `-0.600 / -0.188 / +0.091 / +0.080 / 0` | `-0.223 / -0.025 / +0.149 / +0.143 / 0` | `-0.749 / -0.269 / +0.029 / +0.010 / 0` | `-0.301 / -0.088 / +0.097 / +0.094 / 0` |

4. **Why `P4` inflates Model B (`66.20 / 70`):** Pooling all `719 × 3` fields into one call lets Persistence's large Day-4/8 error overwhelm Day-1 Persistence, hiding Model B's Day-1 `t2m` failure and Day-4 `tp` failure (`Climatology` itself yields `P4 mean_ss = 0.0`).

---

## 2. Artifact Integrity, Checksums & Architecture Comparison

### 2.1 Checksums & Companion Artifact Equivalence

Model B artifacts were extracted cleanly from `origin/main@80648e83010ad6995e37a42446126d9731286a8f` into external directory `C:\Users\ASUS\OneDrive\Desktop\Code\Hackathon\DevForge\team_model_ref_80648e`.

| Artifact | Model A Bytes | Model A Full SHA-256 | Model B (`80648e83`) Bytes | Model B Full SHA-256 | Match? |
| :--- | :---: | :--- | :---: | :--- | :---: |
| `checkpoints/model.pt` | `1,362,884` (`1.30 MiB`) | `4f02add2ce5f4aef3b170cd1f1d2352412ceb149daa470bf2c9e303cba30b25b` | `11,651,378` (`11.11 MiB`) | `1a508d7d367d6ba3018d9f5ee40f7046ad57ac462894e6c470fe421791754990` | Different weights |
| `checkpoints/stats.json` | `423` | `928ba456643f9c2f69bd52bcb9bbcb5302ff7cda32ccca239c9d18c6b9a68381` | `423` | `928ba456643f9c2f69bd52bcb9bbcb5302ff7cda32ccca239c9d18c6b9a68381` | **Bit-identical** |
| `checkpoints/climatology.npy` | `10,119,296` | `e34624134e0d4f22986af47748edb6b6a8f8d8138596740edff662ee0f35affa` | `10,119,296` | `e34624134e0d4f22986af47748edb6b6a8f8d8138596740edff662ee0f35affa` | **Bit-identical** |
| `checkpoints/static.npy` | `9,344` | `2adafeb0ff740aa2fb91758c949e261392e5d788cb773830c265501372e69a9f` | `9,344` | `2adafeb0ff740aa2fb91758c949e261392e5d788cb773830c265501372e69a9f` | **Bit-identical** |

### 2.2 `weights_only=True` Security & Architecture Audit

| Property | Model A (`tanush/model-a`) | Model B (`origin/main@80648e83`) |
| :--- | :--- | :--- |
| **`torch.load(..., weights_only=True)`** | **PASS** (dict of `model_state_dict` + config/stats) | **PASS** (raw `OrderedDict` of 33 tensor entries) |
| **`load_state_dict(..., strict=True)`** | **PASS** (`MultiScaleResUNet`) | **PASS** (`MonsoonFlow10`) |
| **Loader Source Hygiene (`inference/loader.py`)** | Uses `weights_only=True` via `load_model_from_checkpoint` | Calls `weights_only=False` in source line 57 (risk flag, though `.pt` itself is safe) |
| **Inference Package Isolation (`models/`)** | Pure `torch`/`math`/`typing`; zero `data` or `numpy` imports | `models/__init__.py` imports `.physics_losses` → `from data.preprocessing import NLAT, NLON` (fails standalone without `data/`!) |
| **Architecture** | Shallow 1-level `MultiScaleResUNet` (`32×36` + `16×18`) + FiLM DOY + decoupled `tp_head` + `SeasonalAnchorNet` | `SpatialEncoder` (`ChannelLayerNorm`, `r=2`) → 2-layer `ConvLSTM` (`3×3` + `1×1`) → `ResidualDecoder` + `AnomalyTransport` (8 substeps/day upwind advection-diffusion) + `clim_norm` table |
| **Parameter Count** | **`331,786`** | `387,736` (`+16.9%` more params) |
| **Normalization & Stats Embedded** | `stats_mean` / `stats_std` (`tp_transform="linear"`), `static_raw`, `coord_grid` in buffers; differs from `checkpoints/stats.json` by `0.273 Pa` on `msl` mean (`4.4e-4` std, float32 vs float64 train reduction) | `mean` / `std` buffers bit-identical to `checkpoints/stats.json`; embeds `(365, 6, 32, 36)` `clim_norm` (`9.60 MiB`, bit-identical to `climatology_to_yday_table(climatology.npy, stats)`) |
| **Learned Relaxation / Transport** | `SeasonalAnchorNet`: `alpha = [0.872, 0.784, 0.817, 0.797, 0.835, 0.840]` | Fixed `clim_norm` + learned `r = [0.965, 0.798, 0.798, 0.797, 0.796, 0.965]` + upwind `speed ≈ 0.994..0.995`, `kappa ≈ 0.190` |

---

## 3. Channel-Wise & Lead-Day Forecast Evaluation (719 Windows, `2019–2020`)

### 3.1 Physical Validity Audit (All `719 × 10 × 6 × 32 × 36 = 49,697,280` Values)

| Candidate | `scoring.validity` Violations | Non-Finite | `tp < 0` | `tcwv < 0` | `t2m < 233.15` | `t2m > 333.15` | `t2m` Range [K] | `tp` Max [mm/d] | `tcwv` Max [kg/m²] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | `0 / 719` | `0` | `0` | `0` | `0` | `0` | `[240.81, 313.80]` | `249.77` | `79.54` |
| **Climatology** | `0 / 719` | `0` | `0` | `0` | `0` | `0` | `[246.96, 309.91]` | `30.99` | `70.14` |
| **Model A** | **`0 / 719`** | **`0`** | **`0`** | **`0`** | **`0`** | **`0`** | `[241.31, 313.44]` | `106.64` | `78.23` |
| **Model B** | **`0 / 719`** | **`0`** | **`0`** | **`0`** | **`0`** | **`0`** | `[242.13, 312.94]` | `60.78` | `75.88` |

### 3.2 Lead-Time Degradation Trajectory (Days 1–10)

*Kit metrics: `ss_best`, `CSI@10mm`, `csi_skill`. QA diagnostic metrics (`_qa`): `Mean Norm RMSE (RMSE/std)`, `POD`, `FAR`, `Freq Bias`.*

| Lead | `Mean Norm RMSE / std` (QA) | | | | `Mean SS vs Best Ref` (Kit) | | `Heavy-Rain CSI@10mm` (Kit) | | | | `CSI Skill` (Kit) | | `POD / FAR` (QA) | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Model A** | **Model B** | Persist | Clim | **Model A** | **Model B** | **Model A** | **Model B** | Persist | Clim | **Model A** | **Model B** | **Model A** | **Model B** |
| **Day 1** | **0.2857** | 0.3426 | 0.3725 | 0.4933 | **+0.2149** | +0.0580 | **0.5315** | 0.4717 | 0.4376 | 0.2882 | **+0.1671** | +0.0606 | **0.740 / 0.346** | 0.707 / 0.414 |
| **Day 2** | **0.3843** | 0.4389 | 0.5278 | 0.4930 | **+0.2079** | +0.0955 | **0.4387** | 0.3724 | 0.2948 | 0.2882 | **+0.2040** | +0.1101 | **0.674 / 0.444** | 0.659 / 0.539 |
| **Day 3** | **0.4278** | 0.4714 | 0.5930 | 0.4929 | **+0.1471** | +0.0599 | **0.3928** | 0.3448 | 0.2479 | 0.2882 | **+0.1470** | +0.0795 | 0.639 / **0.495** | **0.655** / 0.579 |
| **Day 4** | **0.4502** | 0.4823 | 0.6199 | 0.4928 | **+0.0974** | +0.0318 | **0.3668** | 0.3382 | 0.2302 | 0.2882 | **+0.1105** | +0.0702 | 0.620 / **0.527** | **0.655** / 0.588 |
| **Day 5** | **0.4652** | 0.4872 | 0.6352 | 0.4925 | **+0.0642** | +0.0174 | **0.3490** | 0.3343 | 0.2207 | 0.2883 | **+0.0853** | +0.0646 | 0.606 / **0.549** | **0.647** / 0.591 |
| **Day 6** | **0.4765** | 0.4909 | 0.6471 | 0.4923 | **+0.0392** | +0.0069 | **0.3386** | 0.3304 | 0.2110 | 0.2883 | **+0.0706** | +0.0592 | 0.598 / **0.562** | **0.634** / 0.592 |
| **Day 7** | **0.4851** | 0.4935 | 0.6569 | 0.4925 | **+0.0203** | -0.0006 | **0.3327** | 0.3264 | 0.2067 | 0.2883 | **+0.0624** | +0.0535 | 0.595 / **0.570** | **0.618** / 0.591 |
| **Day 8** | **0.4907** | 0.4948 | 0.6643 | 0.4926 | **+0.0070** | -0.0050 | **0.3276** | 0.3250 | 0.2041 | 0.2883 | **+0.0553** | +0.0516 | 0.592 / **0.577** | **0.605** / 0.587 |
| **Day 9** | **0.4950** | 0.4953 | 0.6687 | 0.4927 | **-0.0031** | -0.0072 | **0.3246** | 0.3221 | 0.2055 | 0.2883 | **+0.0509** | +0.0475 | **0.590 / 0.581** | 0.590 / 0.585 |
| **Day 10** | 0.4981 | **0.4956** | 0.6754 | 0.4928 | -0.0106 | **-0.0085** | **0.3214** | 0.3198 | 0.2046 | 0.2883 | **+0.0466** | +0.0444 | **0.588** / 0.585 | 0.577 / **0.582** |

*Note on POD/FAR:* Model B's slightly higher POD on Days 3–8 stems from heavy wet over-prediction (`frequency_bias = 1.47..1.59`, `FAR = 0.587..0.592`), yielding strictly lower `CSI` across all 10 days. Neither model diverges (`nonfinite_any = False`); Model A `max_abs_anomaly_std` stays in `[5.72, 13.20]` (vs Persistence `32.80`), while Model B decays heavily toward climatology (`7.40 → 1.47`).

### 3.3 Per-Channel Breakdown at Official Leads: Day 1, Day 4, Day 8

#### Day 1 (Harder Reference = Persistence for all 6 channels)

| Channel | Kit `rmse_lat` | | | | Kit `ss_vs_best_ref` | | `MAE` (QA) | | `Bias` (QA) | | `ACC vs Clim` (QA) | | `Pattern Corr` (QA) | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Model A** | **Model B** | Persist | Clim | **Model A** | **Model B** | **A** | **B** | **A** | **B** | **A** | **B** | **A** | **B** |
| `t2m` (K) | **0.8626** | 1.0439 | 1.0294 | 1.7423 | **+0.1620** | -0.0140 | **0.588** | 0.690 | -0.066 | **-0.054** | **0.8395** | 0.7593 | **0.9976** | 0.9967 |
| `tp` (mm/d) | **4.5598** | 5.4559 | 6.2671 | 6.7560 | **+0.2724** | +0.1294 | **1.967** | 2.510 | **+0.015** | +0.328 | **0.7128** | 0.5739 | **0.7622** | 0.6543 |
| `u10` (m/s) | **0.9027** | 1.0571 | 1.0890 | 1.6507 | **+0.1710** | +0.0292 | **0.632** | 0.739 | **+0.043** | +0.048 | **0.7687** | 0.6953 | **0.9219** | 0.8960 |
| `v10` (m/s) | **0.8640** | 1.0373 | 1.1082 | 1.4742 | **+0.2204** | +0.0640 | **0.607** | 0.721 | -0.027 | **+0.011** | **0.7586** | 0.6567 | **0.9140** | 0.8755 |
| `msl` (Pa) | **145.83** | 183.15 | 194.15 | 278.53 | **+0.2489** | +0.0567 | **98.00** | 119.41 | **-5.20** | -6.70 | **0.7999** | 0.6684 | **0.9573** | 0.9311 |
| `tcwv` (kg/m²) | **2.8681** | 3.3500 | 3.6520 | 6.2278 | **+0.2146** | +0.0827 | **2.063** | 2.390 | -0.238 | **-0.051** | **0.8571** | 0.8031 | **0.9854** | 0.9799 |

#### Day 4 (Harder Reference = Climatology for all 6 channels)

| Channel | Kit `rmse_lat` | | | | Kit `ss_vs_best_ref` | | `MAE` (QA) | | `Bias` (QA) | | `ACC vs Clim` (QA) | | `Pattern Corr` (QA) | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Model A** | **Model B** | Persist | Clim | **Model A** | **Model B** | **A** | **B** | **A** | **B** | **A** | **B** | **A** | **B** |
| `t2m` (K) | **1.5559** | 1.6637 | 1.9174 | 1.7400 | **+0.1058** | +0.0438 | **1.072** | 1.124 | -0.159 | **-0.088** | **0.4504** | 0.3206 | **0.9932** | 0.9926 |
| `tp` (mm/d) | **6.4809** | 6.8659 | 9.1666 | 6.7496 | **+0.0398** | -0.0172 | **2.940** | 3.542 | **+0.038** | +0.934 | **0.3010** | 0.1457 | **0.4985** | 0.4500 |
| `u10` (m/s) | **1.4680** | 1.5673 | 1.9523 | 1.6491 | **+0.1098** | +0.0496 | **1.006** | 1.069 | +0.133 | **+0.127** | **0.3749** | 0.2094 | **0.8130** | 0.7849 |
| `v10` (m/s) | **1.3390** | 1.4534 | 1.9258 | 1.4729 | **+0.0909** | +0.0132 | **0.924** | 0.987 | **-0.026** | +0.032 | **0.3466** | 0.1140 | **0.7894** | 0.7520 |
| `msl` (Pa) | **246.64** | 266.34 | 332.63 | 278.00 | **+0.1128** | +0.0420 | **170.37** | 185.67 | **-36.39** | -36.64 | **0.3622** | 0.1641 | **0.8963** | 0.8845 |
| `tcwv` (kg/m²) | **5.4478** | 5.8551 | 7.2078 | 6.2268 | **+0.1251** | +0.0597 | **3.930** | 4.222 | -0.614 | **-0.025** | **0.4140** | 0.3036 | **0.9508** | 0.9438 |

#### Day 8 (Harder Reference = Climatology for all 6 channels)

| Channel | Kit `rmse_lat` | | | | Kit `ss_vs_best_ref` | | `MAE` (QA) | | `Bias` (QA) | | `ACC vs Clim` (QA) | | `Pattern Corr` (QA) | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Model A** | **Model B** | Persist | Clim | **Model A** | **Model B** | **A** | **B** | **A** | **B** | **A** | **B** | **A** | **B** |
| `t2m` (K) | **1.7485** | 1.7786 | 2.2542 | 1.7393 | **-0.0053** | -0.0226 | 1.226 | **1.210** | -0.254 | **-0.103** | **0.2585** | 0.1530 | 0.9916 | **0.9920** |
| `tp` (mm/d) | **6.8501** | 6.8733 | 9.4313 | 6.7359 | **-0.0170** | -0.0204 | **3.183** | 3.507 | **+0.110** | +0.771 | **0.1286** | 0.0814 | 0.4390 | **0.4505** |
| `u10` (m/s) | 1.6306 | **1.6273** | 2.1373 | 1.6483 | +0.0107 | **+0.0127** | 1.103 | **1.097** | **+0.151** | +0.171 | **0.2046** | 0.0931 | **0.7802** | 0.7742 |
| `v10` (m/s) | **1.4669** | 1.4751 | 2.0286 | 1.4730 | **+0.0042** | -0.0014 | 0.997 | **0.992** | **-0.017** | +0.026 | **0.1795** | 0.0090 | **0.7493** | 0.7488 |
| `msl` (Pa) | **272.38** | 277.06 | 368.49 | 278.40 | **+0.0216** | +0.0048 | **192.51** | 195.24 | -57.63 | **-53.33** | **0.2038** | 0.0484 | **0.8851** | 0.8840 |
| `tcwv` (kg/m²) | **6.0511** | 6.2437 | 8.0661 | 6.2232 | **+0.0277** | -0.0033 | **4.352** | 4.555 | -0.718 | **-0.122** | **0.2379** | 0.1589 | **0.9430** | 0.9414 |

---

## 4. Day-8 Spatial Variance Ratios & Anti-Blur Analysis

Official gate: `BLUR_MIN_RATIO = 0.40`, `BLUR_MAX_RATIO = 2.50` across all 6 channels at Day 8 (`scoring.anti_blur_points`).

| Candidate | `t2m` VR8 | `tp` VR8 | `u10` VR8 | `v10` VR8 | `msl` VR8 | `tcwv` VR8 | Anti-Blur Pts (`/10`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Persistence | `1.0124` | `2.6040` *(>2.50!)* | `1.1336` | `1.1296` | `1.1798` | `1.0224` | `8.33 / 10` |
| Climatology | `0.9852` | `0.5703` | `0.7563` | `0.6993` | `0.9087` | `0.9081` | `10.00 / 10` |
| **Model A (`exp4`)** | **`0.9935`** | **`0.8814`** | **`0.8568`** | **`0.7682`** | **`0.9482`** | **`0.9203`** | **`10.00 / 10`** |
| **Model B (`80648e83`)** | `0.9908` | `0.6957` | `0.7488` | `0.7070` | `0.9292` | `0.9145` | **`10.00 / 10`** |

---

## 5. Counterfactual Shock & Causality Comparison (6 Probes)

### 5.1 Aggregate Summary (Offline Identical Protocol; Also Verified Live on `serve.py` for Model A)

| Metric | Model A (`exp4`) | Model B (`80648e83`) |
| :--- | :---: | :---: |
| **Zero-Shock Parity (`null_consistent`, worst `max_abs_diff`)** | **`6/6` PASS** (`0.00e+00 <= 1e-6`) | **`6/6` PASS** (`0.00e+00 <= 1e-6`) |
| **Perturbed Rollout Validity (`scoring.validity`)** | **`12/12` PASS** (`0` viol.) | **`12/12` PASS** (`0` viol.) |
| **`warm_ocean` Pass Count (6 probes / 5 QA probes)** | **`6/6`** (`5/5`) | **`6/6`** (`5/5`) |
| **`monsoon_stall` Pass Count (6 probes / 5 QA probes)** | **`5/6`** (`4/5`) — fails Jan 15 only | **`3/6`** (`2/5`) — fails Jan 15, Apr 20, Oct 15 |
| **`warm_ocean` / `monsoon_stall` Day-1 `near_frac` (mean)** | `0.9722` / `0.9881` | **`0.9982` / `0.9989`** |
| **`far_frac_max` (teleportation ceiling `< 0.05`)** | `0.0119` / `0.0156` | **`0.00000` / `0.00000`** |
| **Causality Score (All 6 Probes / 5 QA Probes)** | **`14.69 / 15`** (`14.63 / 15`) | `14.06 / 15` (`13.88 / 15`) |

### 5.2 Per-Probe Breakdown

| `t0` Date | Season | Model | Null `max\|Δ\|` | `warm_ocean` (`near1` / `out1→out48` / `ratio` / `far` / `maxZ`) | `warm` Pass? | `monsoon_stall` (`near1` / `out1→out48` / `ratio` / `far` / `maxZ`) | `stall` Pass? | Pts (`/15`) |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- | :---: | :---: |
| `2019-01-15` | Winter | **A** | `0.0` | `0.988` / `0.22→1.99` (`9.0×`) / `0.004` / `0.67` | **PASS** | `0.988` / `0.39→0.62` (`1.6×`) / `0.011` / `0.68` | **FAIL (`prop`)** | `13.125` |
| `2019-01-15` | Winter | **B** | `0.0` | `0.999` / `0.01→0.52` (`44.1×`) / `0.000` / `0.45` | **PASS** | `1.000` / `0.001→0.001` (`0.8×`) / `0.000` / `0.60` | **FAIL (`prop`)** | `13.125` |
| `2019-04-20` | Pre-Monsoon | **A** | `0.0` | `0.935` / `1.58→2.73` (`1.7×`) / `0.011` / `0.42` | **PASS** | `0.984` / `0.42→0.70` (`1.7×`) / `0.011` / `0.61` | **PASS** | **`15.000`** |
| `2019-04-20` | Pre-Monsoon | **B** | `0.0` | `0.998` / `0.03→1.93` (`71.2×`) / `0.000` / `0.43` | **PASS** | `1.000` / `0.01→0.12` (`11.6×`) / `0.000` / `0.46` | **FAIL (`prop`)** | `13.125` |
| `2019-06-17` | Onset | **A** | `0.0` | `0.982` / `0.93→9.49` (`10.3×`) / `0.005` / `1.04` | **PASS** | `0.984` / `2.71→10.47` (`3.9×`) / `0.014` / `1.43` | **PASS** | **`15.000`** |
| `2019-06-17` | Onset | **B** | `0.0` | `0.998` / `0.04→3.14` (`72.7×`) / `0.000` / `0.46` | **PASS** | `0.998` / `0.23→3.24` (`13.9×`) / `0.000` / `1.16` | **PASS** | **`15.000`** |
| `2019-07-15` | Active Monsoon | **A** | `0.0` | `0.971` / `1.19→4.32` (`3.6×`) / `0.005` / `0.59` | **PASS** | `0.992` / `1.37→3.71` (`2.7×`) / `0.006` / `1.56` | **PASS** | **`15.000`** |
| `2019-07-15` | Active Monsoon | **B** | `0.0` | `0.997` / `0.06→4.83` (`76.5×`) / `0.000` / `0.42` | **PASS** | `0.999` / `0.15→2.97` (`19.5×`) / `0.000` / `1.09` | **PASS** | **`15.000`** |
| `2020-08-10` | Active Monsoon | **A** | `0.0` | `0.988` / `0.62→4.83` (`7.8×`) / `0.003` / `0.68` | **PASS** | `0.987` / `2.60→8.68` (`3.3×`) / `0.013` / `1.51` | **PASS** | **`15.000`** |
| `2020-08-10` | Active Monsoon | **B** | `0.0` | `0.998` / `0.05→3.68` (`69.7×`) / `0.000` / `0.46` | **PASS** | `0.999` / `0.18→4.10` (`23.5×`) / `0.000` / `1.12` | **PASS** | **`15.000`** |
| `2020-10-15` | Retreat | **A** | `0.0` | `0.969` / `2.18→40.97` (`18.8×`) / `0.012` / `1.55` | **PASS** | `0.994` / `0.87→4.67` (`5.4×`) / `0.016` / `2.04` | **PASS** | **`15.000`** |
| `2020-10-15` | Retreat | **B** | `0.0` | `0.998` / `0.05→2.61` (`57.7×`) / `0.000` / `0.44` | **PASS** | `0.999` / `0.07→0.90` (`12.2×`) / `0.000` / `0.88` | **FAIL (`prop`)** | `13.125` |

---

## 6. 4-Core CPU Performance & Memory Benchmark (Isolated Subprocesses)

Measured in **separate isolated Python subprocesses** (`CUDA_VISIBLE_DEVICES=""`, `torch.set_num_threads(4)`, `torch.set_num_interop_threads(1)`, 4-core OS affinity `True`), `5` warm-up calls followed by `30` timed single-request `steps=10` rollouts (`batch_size = 1`, `(3, 6, 32, 36)` → `(10, 6, 32, 36)` `float32`). *(Live HTTP `serve.py` readiness latency/RSS is audited separately in `evaluation/CERTIFICATION_REPORT.md`.)*

| Benchmark Metric | Limit | **Model A (`exp4`)** | **Model B (`80648e83`)** | Ratio / Delta (`A` vs `B`) |
| :--- | :---: | :---: | :---: | :---: |
| **Checkpoint Disk Size** | — | **`1.30 MiB`** (`1,362,884 B`) | `11.11 MiB` (`11,651,378 B`) | **`8.55×` smaller** |
| **Parameter Count** | — | **`331,786`** | `387,736` | **`14.4%` fewer** |
| **Cold Load + Init Time** | — | **`18.60 ms`** | `41.30 ms` | **`2.22×` faster** |
| **Warm-up (5 Calls)** | — | **`444.78 ms`** | `872.17 ms` | **`1.96×` faster** |
| **Single-Request `p50` Latency** | `<= 5000 ms` | **`90.21 ms`** | `174.75 ms` | **`1.94×` faster** |
| **Single-Request `mean` Latency** | `<= 5000 ms` | **`90.19 ms`** (`σ=3.64 ms`) | `175.64 ms` (`σ=3.97 ms`) | **`1.95×` faster** |
| **Single-Request `p95` Latency** | `<= 5000 ms` | **`95.32 ms`** | `183.06 ms` | **`1.92×` faster** |
| **Single-Request `max` Latency** | `<= 5000 ms` | **`98.07 ms`** (`51.0×` headroom) | `185.93 ms` (`26.9×` headroom) | **`1.90×` faster** |
| **Steady-State Process RSS** | `<= 6144 MB` | **`418.40 MiB`** | `433.23 MiB` | **`-14.83 MiB`** |
| **Peak Process RSS** | `<= 6144 MB` | **`418.40 MiB`** | `453.93 MiB` | **`-35.53 MiB`** |
| **Output Tensor Footprint** | `276,480 B` | `276,480 B` (`0.264 MiB`) | `276,480 B` (`0.264 MiB`) | Identical `(10,6,32,36) f4` |

*(719-window batch=32 CPU evaluation wall time: Model A `27.19 s` vs Model B `76.70 s`.)*

---

## 7. Strengths, Weaknesses & Competition Designation

- **Model A (`checkpoints/model.pt`, `exp4_msresunet_anchor` — PROVISIONAL PRIMARY):** Dominates forecast skill under both `P1` (`+21.08/70`) and `P2` (`+22.30/70`), achieves ~2× higher Day-8 ACC, sharper spatial variance (`tp` VR `0.881`), near-zero rain bias, `5/6` `monsoon_stall` propagation (`14.69/15`), `1.94×` faster CPU latency, and `8.55×` smaller disk footprint. Known caveats: slight Day-8 negative drift bias (`t2m -0.254 K`, `tcwv -0.718 kg/m²`, `msl -57.6 Pa`), Day-8 `t2m`/`tp` RMSE slightly trailing Climatology (`-0.0053`/`-0.0170`), and `2019-01-15` winter `monsoon_stall` propagation miss.
- **Model B (`origin/main@80648e83` — NOT RECOMMENDED):** Excels only at zero teleportation (`far_frac_max = 0.00000`), lower Day-8 mean bias on `t2m`/`tcwv`/`msl` (via `clim_norm` damping), and a `0.0033 m/s` (`0.2%`) Day-8 `u10` RMSE edge (`1.6273` vs `1.6306`). Fails Day-1 `t2m` vs Persistence, Day-4 `tp` vs Climatology (`+0.934 mm/d` wet bias, `FAR=0.588`), fails `monsoon_stall` on 3/6 seasons, runs 1.94× slower, and ships `weights_only=False` + `data.preprocessing` coupling in source.
- **Fallback Designation:** Retain `checkpoints/runs/exp5_msresunet_direct_22ep_control/best.pt` as **FALLBACK** (not Model B).
- **Certification Gate Status:** See `evaluation/CERTIFICATION_REPORT.md` — Model A passes all model & live-server gates, while **repository/brief submission blockers** (uncommitted `checkpoints/model.pt` / `models/` on git HEAD `tracked_only repo_scan = 3.5/15`, and missing `project_brief.pdf`) keep Model A **PROVISIONAL PRIMARY** until resolved.
