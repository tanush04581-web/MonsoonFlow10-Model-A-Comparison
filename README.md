# MonsoonFlow-10 — Model A Comparison Package (`exp4_msresunet_anchor`)

> **Status:** **PROVISIONAL PRIMARY** (`exp4_msresunet_anchor`, Epoch 22)  
> **Event / Domain:** Yugaantar DevForge 2026 — Track 2: Multiverse of Monsoon (`6.5°N–37.5°N`, `65.5°E–100.5°E`, `32 × 36` `1° × 1°` grid)  
> **Checkpoint SHA-256 (`checkpoints/model.pt`):** `4f02add2ce5f4aef3b170cd1f1d2352412ceb149daa470bf2c9e303cba30b25b` (`1,362,884` bytes, FP32, `331,786` parameters)

This standalone package contains **Model A** (`MultiScaleResUNet` + `SeasonalAnchorNet`), its safe CPU loading and offline FastAPI serving code, training/validation metadata, support climatology/static arrays, QA scripts, and independent head-to-head comparison & certification reports against the team baseline (`origin/main@80648e83`).

---

## 1. Repository Contents

- [`checkpoints/model.pt`](checkpoints/model.pt) — Model A weights (`exp4_msresunet_anchor`, `weights_only=True` safe state-dict bundle with embedded normalization statistics and static terrain buffers).
- [`checkpoints/ARCHITECTURE.md`](checkpoints/ARCHITECTURE.md) — Architecture specification, forward rollout flow, and curriculum summary.
- [`checkpoints/training_config.json`](checkpoints/training_config.json) — Exact optimizer, 4-stage curriculum `[4, 4, 4, 10]`, composite physics loss weights, hardware, and provenance metadata.
- [`checkpoints/training_history.json`](checkpoints/training_history.json) — Per-epoch loss and validation trajectory across all 22 epochs (`969.71 s` on RTX 4060 Laptop GPU, peak VRAM `2676.8 MB`).
- [`checkpoints/validation_metrics.json`](checkpoints/validation_metrics.json) — Detailed 719-window `2019–2020` validation breakdown, FSS, and Day-8 FFT spectral ratios.
- [`checkpoints/normalization_metadata.json`](checkpoints/normalization_metadata.json) — Exact 2005–2018 training normalization stats, static ranges, DOY conventions, and physical clamps.
- [`checkpoints/stats.json`](checkpoints/stats.json), [`checkpoints/climatology.npy`](checkpoints/climatology.npy), [`checkpoints/static.npy`](checkpoints/static.npy), [`checkpoints/baselines_val.json`](checkpoints/baselines_val.json) — Train-only support statistics, 366-day smoothed climatology (`10,119,296` bytes), `(2, 32, 36)` static land-sea mask & orography (`9,344` bytes), and 2019–2020 baseline metrics.
- [`models/`](models/) — Pure PyTorch inference/loss codebase (`backbones.py`, `transforms.py`, `physics_losses.py`, `__init__.py`), free of external weather APIs, date literals, or `data/` imports.
- [`inference/`](inference/) & [`serve.py`](serve.py) — Offline 4-core CPU FastAPI server (`/health`, `/simulate`, `weights_only=True`, off-event-loop execution, Pydantic v2 validation).
- [`training/`](training/) — Source code for baselines, curriculum trainer, and scientific validation suite (no raw `.nc` data or sealed `2021` holdout data included).
- [`evaluation/`](evaluation/) & [`tests/`](tests/) — Independent Agent-5 QA evaluation suite, contract tests, readiness/shock reports, [`evaluation/HEAD_TO_HEAD_COMPARISON.md`](evaluation/HEAD_TO_HEAD_COMPARISON.md), [`evaluation/head_to_head_report.json`](evaluation/head_to_head_report.json), and [`evaluation/CERTIFICATION_REPORT.md`](evaluation/CERTIFICATION_REPORT.md).
- [`resources/yug-monsoon-kit/`](resources/yug-monsoon-kit/) — Pinned Track-2 competition contract (`monsoon_r1`), scoring primitives, schema validator, and repo scanner.
- [`scripts/verify_model_a.py`](scripts/verify_model_a.py) — Offline CPU smoke-test script (no dataset archive required).

---

## 2. Model Architecture Overview

`MultiScaleResUNet` (`models/backbones.py`) is a shallow 1-level multi-scale residual U-Net (`331,786` parameters) designed for bounded local causality and CPU efficiency:

1. **Physical-Unit I/O Contract:**
   - **Input:** `init_states` `(3, 6, 32, 36)` (3 historical days `t_-2, t_-1, t_0`) + `doy_sincos` `(2,)` for `t_0` + optional physical `perturbation` `(6, 32, 36)` added once to `t_0`.
   - **Channels (strict order):** `t2m` (`K`), `tp` (`mm/day`), `u10` (`m/s`), `v10` (`m/s`), `msl` (`Pa`), `tcwv` (`kg/m^2`).
   - **Output:** `(steps, 6, 32, 36)` in physical units (default `steps=10`, supported `1..14`).
2. **Internal Conditioning & Stabilization:**
   - Embedded 2005–2018 linear z-score normalization buffers (`stats_mean`, `stats_std`, `tp_transform="linear"`) and embedded `(2, 32, 36)` static terrain (`land_sea_mask`, `orography_m / 8000.0`) + normalized `coord_grid`.
   - Fine `32x36` FiLM-modulated residual branch (`48` channels) paired with a `2x` coarse `16x18` synoptic branch (`72` channels) and decoupled `3x3` precipitation head (`tp_head`).
   - `SeasonalAnchorNet`: predicts smooth seasonal-geographic attractors `(bg_cur, bg_tgt)` from `(static, coords, doy)` and updates `raw_next = bg_tgt + alpha * (f2 - bg_cur) + delta_norm` with learned channel persistence `alpha = [0.872, 0.784, 0.817, 0.797, 0.835, 0.840]`.
   - Post-denormalization safeguards (`models/transforms.py::apply_physical_constraints`): `tp >= 0`, `tcwv >= 0`, `t2m in [233.16, 333.14] K` (avoiding float32 boundary underflow against `233.15 K`).

---

## 3. Setup & Dependencies

Tested on **Python 3.11** with PyTorch CPU/CUDA (`torch==2.4.1`, `numpy==1.26.4`, `fastapi==0.114.2`, `uvicorn==0.30.6`, `pydantic==2.14.4`).

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate

pip install -r requirements.txt
pip install -e resources/yug-monsoon-kit --no-deps
```

Run the self-contained offline verification script (checksum, `weights_only=True`, 10-step CPU rollout, zero-shock `0.0` diff, warm-ocean localization):

```bash
python scripts/verify_model_a.py
```

---

## 4. Inference Usage

### 4.1 Direct PyTorch CPU Inference (`weights_only=True`)

Pass inputs in **raw physical units** (`t2m` in K, `tp` in mm/day, `u10`/`v10` in m/s, `msl` in Pa, `tcwv` in kg/m²). Do **not** pre-normalize inputs externally; `model.forward(...)` applies embedded normalization and denormalization automatically.

```python
import numpy as np
import torch
from models.backbones import load_model_from_checkpoint

torch.set_num_threads(4)
model, ckpt = load_model_from_checkpoint("checkpoints/model.pt", map_location="cpu")
model.eval()

# init_phys: float32 tensor/array of shape (3, 6, 32, 36) in physical units
# doy_last:  [sin(2*pi*doy/365), cos(2*pi*doy/365)] for last input day t_0
init_phys = torch.from_numpy(np.zeros((3, 6, 32, 36), dtype=np.float32))
init_phys[:, 0] = 295.0
init_phys[:, 4] = 101100.0
init_phys[:, 5] = 30.0
doy_last = torch.tensor([-0.23030567, -0.97311834], dtype=torch.float32)

with torch.inference_mode():
    rollout = model(init_phys, doy_last, steps=10, perturbation=None)  # (10, 6, 32, 36) physical units
```

### 4.2 Offline FastAPI Server (`serve.py`)

```bash
python serve.py --port 8000
```

- `GET /health` & `POST /health`: returns HTTP `200` with `{"status": "ready", "model_name": "MonsoonFlow-10 (ms_resunet-anchor)", ...}` (`503` if weights are unavailable; no persistence fallback).
- `POST /simulate`: accepts `{"init_states": [3,6,32,36], "doy_sincos": [sin, cos], "steps": 10, "perturbation": null | [6,32,36]}` and returns `{"rollout": [steps,6,32,36], "inference_ms": float}`.
- Validate against the official contract:

```bash
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
pytest tests/test_qa_server_contract.py -v
```

---

## 5. Checkpoint Provenance & Data Isolation

- **Source Run:** `checkpoints/runs/exp4_msresunet_anchor/best.pt` (`epoch=22`, seed `2026`, FP32, batch size `24`, AdamW `weight_decay=1e-4`, 4-stage curriculum `[4, 4, 4, 10]` epochs at `steps=[1, 3, 5, 10]`, controlled against `exp5_msresunet_direct_22ep_control` which shared the identical schedule without `SeasonalAnchorNet`).
- **Strict Temporal Split Hygiene:**
  - Trained strictly on `2005-01-01`..`2018-12-31` (`5,113` days / `5,101` windows).
  - Selected & evaluated strictly on `2019-01-01`..`2020-12-31` (`731` days / `719` windows).
  - The sealed `2021` holdout (`test.npz`) was **never used** for training, tuning, or evaluation.
- **Note on Legacy `checkpoints/stats.json`:** `checkpoints/model.pt` embeds the exact float32-computed 2005–2018 `train_stats.json` buffers used during training (see [`checkpoints/normalization_metadata.json`](checkpoints/normalization_metadata.json)), which differs from `checkpoints/stats.json` (float64 reduction) by `0.273 Pa` on `msl` mean (`0.04%` of std).

---

## 6. Head-to-Head Results & Unresolved Scoring Aggregation Discrepancy

Full results across all `719` validation windows (`2019–2020`), `6` seasonal shock probes, and isolated 4-core CPU benchmarks are recorded in [`evaluation/HEAD_TO_HEAD_COMPARISON.md`](evaluation/HEAD_TO_HEAD_COMPARISON.md) and [`evaluation/CERTIFICATION_REPORT.md`](evaluation/CERTIFICATION_REPORT.md) (comparing **Model A** against teammate commit `Hardik2811-wq/MonsoonFlow10@80648e83`).

### Explicit Disclosure: Scoring Aggregation Discrepancy (`/70` Model Score)

The official kit ([`resources/yug-monsoon-kit/monsoon_r1/scoring.py`](resources/yug-monsoon-kit/monsoon_r1/scoring.py) and [`docs/R1_CONTRACT.md`](resources/yug-monsoon-kit/docs/R1_CONTRACT.md)) supplies pure scoring primitives (`skill_vs_best_ref`, `csi`, `csi_skill`, `anti_blur_points`, `causality_checks`, `causality_points`, and `forecast_points(ss_by_window, csi_ss_by_window)`), while intentionally omitting the organizer's blind outer grading runner over the 5 hidden post-2023 test windows.

This leaves two contract-compliant **per-window** ways (`P1`, `P2`) to invoke `monsoon_r1.scoring` alongside two non-contract **pooled diagnostic** modes (`P3`, `P4`), which remain unresolved until checked against the organizer's hidden outer grader:

| Aggregation Mode | Contract Status | Model A (`exp4`, 719 win) | Model B (`80648e83`, 719 win) | Margin (`A − B`) |
| :--- | :--- | :---: | :---: | :---: |
| **`P1` Per-Window × Per-Lead** (`qa_model_eval.py`) | **Per-window compliant** *(conservative lower bound)* | **`50.31 / 70`** (`fcst=25.62` [`11.45+14.17`], `ab=10.00`, `caus=14.69`) | **`29.23 / 70`** (`fcst=5.17` [`0.00+5.17`], `ab=10.00`, `caus=14.06`) | **`+21.08 / 70`** |
| **`P2` Per-Window Stacked `[3,6,32,36]`** (`rmse_lat` spec) | **Per-window compliant** *(multi-lead per window)* | **`66.52 / 70`** (`fcst=41.83` [`27.66+14.17`], `ab=10.00`, `caus=14.69`) | **`44.22 / 70`** (`fcst=20.16` [`14.99+5.17`], `ab=10.00`, `caus=14.06`) | **`+22.30 / 70`** |
| *`P3` Pooled-Per-Lead* (`training/evaluate_val.py`) | **Diagnostic only** *(passes 3 leads, not `W` windows)* | *`66.29 / 70`* (`fcst=41.61` [`26.61+15.00`]) | *`40.25 / 70`* (`fcst=16.19` [`7.07+9.12`]) | *`+26.04 / 70`* |
| *`P4` Pooled-All `[W,3,6,32,36]`* | **Diagnostic only** | *`69.69 / 70`* | *`66.20 / 70`* | *`+3.49 / 70`* |

*(Automated `/70` model scores exclude the separate `/30` points for `Repo Integrity /15` + `2-Page Brief /15`.)*

- **Why `P1` differs from `P2` & `P3`:** Selecting `min(RMSE_persist, RMSE_clim)` on single `(window, day)` `[6, 32, 36]` slices (`P1`) introduces Jensen's downward denominator shift plus large negative `SS = 1 - r / best` outliers when calm single-day Persistence error is near zero (`msl` down to `-2.782`, `tp` down to `-2.154`). Grouping Leads `(1, 4, 8)` as `[3, 6, 32, 36]` per window (`P2`) eliminates those spikes (`0` values `< -1.0` for Model A).
- **Selected Summary Highlights (719 Val Windows & 4-Core CPU):**
  - **Hard Gates:** `719/719` physical validity (`0` violations), beats Day-4 Persistence on `6/6` channels (`mean_ss_persist = +0.2562`), exact `0.0` zero-shock parity (`<= 1e-6`), and localized causal shock propagation (`warm_ocean 6/6`, `monsoon_stall 5/6`; `4/4` monsoon-season probes pass).
  - **Accuracy:** Leads Model B on `6/6` channels at Days 1 and 4 and `5/6` channels at Day 8 (`CSI@10mm` Days 1/4/8: `0.5315 / 0.3668 / 0.3276` vs `0.4717 / 0.3382 / 0.3250`), with Day-8 spatial variance ratios `[0.768, 0.994]` (`tp` VR `0.8814` vs `0.6957`).
  - **4-Core CPU Latency & Memory (isolated subprocesses, `CUDA_VISIBLE_DEVICES=""`):** `p50 = 90.21 ms`, `p95 = 95.32 ms`, `max = 98.07 ms` (`<< 5000 ms` limit), steady/peak RSS `418.40 MiB` (`<< 6144 MB` limit). Live `qa_readiness.py` HTTP server benchmark (`30` calls + `schema_validator` across JSON serialization): `p50 = 151 ms`, `p95 = 190 ms`, `max = 2388 ms` (`<= 5.0 s`), `peak_rss = 1429.95 MB`, `0` outbound network attempts.

---

## 7. Known Limitations

1. **Unresolved Outer Scoring Loop (`P1` vs `P2`):** The hidden judge runner is outside `resources/yug-monsoon-kit`; both per-window compliant modes leave Model A decisively ahead (`+21.08` to `+22.30` `/70`), but `P1` yields `11.45/30` RMSE points vs `27.66/30` under `P2`.
2. **Day-8 Drift & Climatology Crossover:** Model A exhibits a slight negative Day-8 mean bias relative to 2019–2020 normals (`t2m -0.254 K`, `tcwv -0.718 kg/m²`, `msl -57.6 Pa`), and Day-8 `t2m` (`-0.0053`) and `tp` (`-0.0170`) RMSEs sit slightly behind 2005–2018 Climatology.
3. **Deep-Winter `monsoon_stall` & Custom High-Amplitude Shock:** `monsoon_stall` propagation falls slightly below `0.02 * peak_total` in deep winter (`2019-01-15`: `out48 = 0.617`), and a synthetic custom `+20 mm/day` Western Ghats shock (`2020-08-10`) showed `far = 0.072 > 0.05` (official shocks stay `far <= 0.0156 << 0.05`).
4. **Live Server RSS Growth & Dataset Dependency for Full Val Re-Runs:** Repeated HTTP calls grew peak `serve.py` RSS to `~1.43 GB` (`qa_readiness.py`), still well below `6 GB`. Running full 719-window evaluation scripts (`qa_head_to_head.py`, `qa_model_eval.py`) requires the competition dataset splits under `data/raw/splits/`, which are excluded from this repository.
