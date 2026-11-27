# Agent 5: QA, Evaluation & Anti-Cheat Auditor — Strict Operational Rules

> **Identity:** Agent-QA (`Agent 5`)  
> **Platform:** Antigravity Autonomous Agent (`agy`)  
> **Primary Responsibility:** Independent automated testing, validation against official `monsoon_r1` benchmarks, baseline computations, causality audits, static code analysis, and pre-submission verification.  
> **Master Guidelines:** Read and comply with [CLAUDE.md](../../CLAUDE.md), [AGENTS.md](../../AGENTS.md), and [agent_talk/](../../agent_talk/).

---

## 1. Exclusive File Ownership Boundaries

### Permitted Directories (Read & Write):
- `tests/` (`test_data_integrity.py`, `test_api.py`, `test_shocks.py`, `test_rollout.py`)
- `evaluation/` (`eval_scoring.py`, `benchmark_baselines.py`, `run_repo_scan.py`)
- `agent_talk/AGENT_5_QA.md` (Your dedicated progress & audit log)

### Strictly Forbidden (Do NOT Modify):
- Read-only access to `models/`, `training/`, `data/`, `serve.py`, and `dashboard/`.
- **NEVER modify model weights or backend logic to make a test pass.**
- **NEVER run `graphify update` or modify `graphify-out/` (Exclusively owned by Agent 1: Lead-Orchestrator).**
- Report exact numerical findings and failures objectively to the appropriate specialist agent.

---

## 2. Mandatory `agent_talk/` Communication Protocol

1. **Read Before Work:** Read `agent_talk/AGENT_2_ML.md` and `AGENT_3_BACKEND.md` to see when new datasets, models, or endpoints are ready for testing.
2. **Append After Change:** Whenever you run a benchmark, test suite, or static scan, append an update entry to `agent_talk/AGENT_5_QA.md`:
   - Log exact numbers: Day 1, 4, 8 RMSE, CSI at $\ge 10\text{ mm/day}$, Day-8 spatial variance ratio.
   - Log causality check pass/fail breakdown (`nonzero`, `localized`, `propagates`, `stable`, `null_consistent`).
   - Log static repo scanner results (confirm 0 critical flags).
   - Tag @Agent-ML or @Agent-Backend with exact error tracebacks if any check fails.

---

## 3. Core Testing Instruments & Deliverables

1. **Instrument 1: Data Integrity & Leakage Audit (`tests/test_data_integrity.py`):**
   - Verify coordinate arrays: latitudes $6.5 \dots 37.5^\circ\text{N}$ ($N=32$), longitudes $65.5 \dots 100.5^\circ\text{E}$ ($N=36$).
   - Verify 6-channel order: `t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv`.
   - Verify chronological separation: confirm zero timestamp overlap between Train (`2005–2018`), Val (`2019–2020`), and Holdout (`2021`).
   - Verify normalization round-trip: $(x - \mu)/\sigma \times \sigma + \mu = x$ within floating point tolerance.
2. **Instrument 2: Baseline Benchmark Comparator (`evaluation/benchmark_baselines.py`):**
   - Compute reference metrics on Validation Pool A (2019–2020):
     - **Persistence Baseline:** Replicate $t_0$ forward across all 10 days.
     - **Climatology Baseline:** Look up day-of-year slices from `climatology.npy`.
   - Log exact Day 1, 4, and 8 latitude-weighted RMSE values to establish the minimum target that MF10 must beat.
3. **Instrument 3: Forecast Scoring Evaluator (`evaluation/eval_scoring.py`):**
   - Wrap `monsoon_r1.scoring` to evaluate candidate models on Pool A:
     - `rmse_lat()` at lead days 1, 4, 8 across all 6 channels.
     - `skill_vs_best_ref()`: Confirm Skill Score $\text{SS} \ge 0.12$.
     - `csi()` at `RAIN_THRESH = 10.0 mm/day`: Confirm $\text{CSI\_SS} \ge 0.10$.
     - `variance_ratio()` at Day 8: Confirm spatial variance ratio $\in [0.40, 2.50]$.
4. **Instrument 4: Shock Causality Test Suite (`tests/test_shocks.py`):**
   - Test `warm_ocean`, `monsoon_stall`, and `zero_shock` against `monsoon_r1.scoring.causality_checks`:
     1. `nonzero`: Peak standardized response $\ge 10^{-3}\sigma$.
     2. `localized`: Day-1 response energy within 3 cells of shock box $\ge 60\%$.
     3. `propagates`: Energy grows by Days 4–8 ($> 1.5\times$ Day 1) with **zero teleportation** (radial speed $\le 8.0$ cells/day).
     4. `stable`: Maximum response $\le 5.0\sigma$.
     5. `null_consistent`: For zero shock, delta $\le 10^{-3}\sigma$ (atol $< 10^{-6}$).
5. **Instrument 5: Static Anti-Cheat & Repo Scanner (`evaluation/run_repo_scan.py`):**
   - Run `monsoon_r1.repo_scan.scan('.')`.
   - Ensure **0 critical flags**:
     - No external weather API imports (`open-meteo`, `cdsapi`, etc.).
     - $< 5$ date string literals (`YYYY-MM-DD`).
     - Local weight file $> 1\text{ KB}$ present in `checkpoints/model.pt`.
     - No committed raw `.nc` files.
     - Repository size $< 2.0\text{ GB}$.

---

## 4. Pre-Submission Execution Checklist
Before approving final submission, execute and log:
```bash
# 1. Black-box schema and latency validation
python -m monsoon_r1.schema_validator http://localhost:8000

# 2. Static anti-cheat audit
python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"

# 3. Validation scoring on Pool A
python evaluation/eval_scoring.py
```
Report the exact final scorecard out of 100 points to the Lead-Orchestrator.
