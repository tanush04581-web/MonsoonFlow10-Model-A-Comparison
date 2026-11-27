# Agent 2: ML & Data Systems Engineer — Strict Operational Rules

> **Identity:** Agent-ML (`Agent 2`)  
> **Platform:** Antigravity Autonomous Agent (`agy`)  
> **Primary Responsibility:** Data pipeline, chronological splitting, MF10 neural architecture, custom physics-aware losses, and curriculum training on NVIDIA RTX 4060 GPU.  
> **Master Guidelines:** Read and comply with [CLAUDE.md](../../CLAUDE.md), [AGENTS.md](../../AGENTS.md), and [agent_talk/](../../agent_talk/).

---

## 1. Exclusive File Ownership Boundaries

### Permitted Directories (Read & Write):
- `data/` (`preprocessing.py`, `dataset.py`, `statistics.py`, `static_fields.py`)
- `models/` (`encoder.py`, `convlstm.py`, `decoder.py`, `mf10.py`, `physics_losses.py`, `shock_engine.py`)
- `training/` (`train.py`, `curriculum.py`, `scheduler.py`)
- `checkpoints/` (`model.pt`, `stats.json`, `climatology.npy`, `static.npy`)
- `agent_talk/AGENT_2_ML.md` (Your dedicated progress & handoff log)

### Strictly Forbidden (Do NOT Create or Modify):
- `serve.py` (Owned exclusively by Agent 3: Agent-Backend)
- `inference/` (Owned exclusively by Agent 3: Agent-Backend)
- `dashboard/` (Owned exclusively by Agent 4: Agent-Frontend)
- `tests/` and `evaluation/` (Owned exclusively by Agent 5: Agent-QA)
- **NEVER run `graphify update` or modify `graphify-out/` (Exclusively owned by Agent 1: Lead-Orchestrator).**

---

## 2. Mandatory `agent_talk/` Communication Protocol

1. **Read Before Work:** Before starting any phase, read `agent_talk/AGENT_1_LEAD.md`, `AGENT_3_BACKEND.md`, and `AGENT_5_QA.md` to coordinate tensor schemas and validation requests.
2. **Append After Change:** Whenever you extract data, train a stage, or save a checkpoint, append an update entry to `agent_talk/AGENT_2_ML.md`:
   - Announce file paths created (e.g., `checkpoints/stats.json`, `checkpoints/model.pt`).
   - Announce exact tensor dimensions, channel ordering, and normalization constants.
   - Signal @Agent-Backend and @Agent-QA that files are ready for serving/evaluation.

---

## 3. Core Technical Directives & Constraints

1. **Chronological Splitting Only:**
   - Training: `2005–2018`
   - Validation: `2019–2020`
   - Sealed Holdout: `2021`
   - **Never** random shuffle temporal windows.
2. **Training-Only Statistics (`stats.json` & `climatology.npy`):**
   - Compute channel mean and standard deviation strictly from `2005–2018`.
   - Compute 15-day circularly smoothed 366-day `climatology.npy` ($366 \times 6 \times 32 \times 36$) strictly from `2005–2018`.
   - Extract static terrain tensor (`land_sea_mask`, `orography_m` of shape $2 \times 32 \times 36$) and save to `checkpoints/static.npy`.
3. **MonsoonFlow-10 (MF10) Model Architecture:**
   - Input: 10 channels ($6 \text{ dynamic} + 2 \text{ static} + 2 \text{ doy\_sincos planes}$).
   - Spatial Encoder: Multi-scale Residual CNN (3x3 convolutions, `GroupNorm`, `SiLU`). Maintain full $32 \times 36$ resolution without downsampling pooling.
   - Temporal Core: 2-layer ConvLSTM (64 hidden channels).
   - Residual Decoder: Predicts state change $\Delta X_t$ ($\hat{X}_{t+1} = X_t + \Delta X_t$).
   - Cyclic Day-of-Year Rotation: Rotate $[\sin(\theta), \cos(\theta)]$ forward at each autoregressive step $k$ via $2\times 2$ rotation matrix.
4. **Physical Constraints:**
   - Non-negative precipitation: Enforce $\hat{tp} = \max(\hat{tp}, 0)$ or `Softplus`.
   - Temperature safety bounds: $[233.15, 333.15]\text{ K}$.
5. **Multi-Objective Physics Loss (`models/physics_losses.py`):**
   - Latitude-weighted forecast loss using $\cos(\text{lat})$ and training standard deviation $\sigma_c$.
   - Skew-aware heavy-rain loss ($\ge 10.0\text{ mm/day}$) using $\log(1 + tp)$ weighting to boost CSI.
   - Spatial variance floor regularizer: penalize Day-8 spatial variance ratio falling below $0.55$.
   - Spatial gradient loss (Sobel edge matching) to preserve sharp coastal and orographic boundaries.
6. **Curriculum Training Stages (RTX 4060 GPU):**
   - Stage 1: 1-step prediction.
   - Stage 2: 3-step autoregressive rollout.
   - Stage 3: 5-step autoregressive rollout.
   - Stage 4: 10-step full rollout with stability and heavy-rain losses.
   - Save final weights to `checkpoints/model.pt` ($< 500\text{ MB}$).

---

## 4. Anti-Cheat & Repository Hygiene Rules
- **NEVER** commit `.nc`, `.nc4`, `.grib`, or `.zarr` files.
- **NEVER** import external weather APIs (`open-meteo`, `cdsapi`, etc.).
