# Independent Agent-5 QA Certification & Submission Gate Report

**Date:** 2026-11-27  
**Auditor:** Agent 5 (Independent QA)  
**Branch / Commit:** `tanush/model-a` (`HEAD 335b64cbf0758bdd8998a3f6ba2892878bf80acc`), remote `https://github.com/Hardik2811-wq/MonsoonFlow10.git`  
**Kit Truth Source:** `resources/yug-monsoon-kit` @ `689c8798bbff7f318a1e74878a6985abc646080a` (`clean`)  
**Primary Candidate:** `checkpoints/model.pt` (`exp4_msresunet_anchor`, SHA-256 `4f02add2ce5f4aef3b170cd1f1d2352412ceb149daa470bf2c9e303cba30b25b`, `1,362,884` bytes) — **PROVISIONAL PRIMARY**  
**Fallback Candidate:** `checkpoints/runs/exp5_msresunet_direct_22ep_control/best.pt` (`1,316,270` bytes) — **FALLBACK**  
**Holdout Discipline:** Sealed 2021 `data/raw/splits/test.npz` was **NEVER accessed or evaluated**. All model metrics use strictly `data/raw/splits/val.npz` (`2019-01-01..2020-12-31`, `719` windows).

---

## 1. Executive Certification Verdict

> **VERDICT: `PROVISIONAL PRIMARY — CONDITIONAL PASS (MODEL & SERVING READY; SUBMISSION BLOCKED ON GIT COMMIT & BRIEF)`**

1. **Model & Live Serving Gates: ALL PASS.** Model A (`checkpoints/model.pt`) passes all 3 Hard Admission Gates (`Validity 719/719`, `Day-4 > Persistence` on raw/norm/all-6 channels, `Non-zero localized & propagating shocks` `5/5` warm / `4/5` stall on live `serve.py`), passes `monsoon_r1.schema_validator` `6/6`, passes `qa_offline_guard.py` (`0` outbound attempts), passes 4-core CPU limits (`p50=0.151 s`, `max=2.388 s <= 5.0 s`; `peak_rss=1429.95 MB <= 6144 MB`), and passes `41/41` executed `pytest` checks (`1` skipped intentionally for unstarted port `8000`).
2. **Why NOT Fully Certified for Submission Yet:**
   - **CRITICAL BLOCKER 1 (`repo_scan[tracked_only] = 3.5 / 15`):** `checkpoints/model.pt`, `models/`, `training/`, and `serve.py`/`inference/`/`dashboard/` edits are uncommitted/untracked on `tanush/model-a`. Scanning git-tracked files fails critically (`"no model checkpoint in the repo and no Hugging Face link"`).
   - **CRITICAL BLOCKER 2 (`project_brief.pdf` missing):** `project_brief.pdf` (`15 / 100` manual brief score) does not exist at the repository root.
   - **AMBIGUITY NOTE (Hidden Judge Aggregation):** While `P1` and `P2` are both faithful per-window `monsoon_r1.scoring` interpretations, the organizer's blind judge script is intentionally unshipped, leaving a `50.31` (`P1`) vs `66.52` (`P2`) spread on 719 windows (`60.36` vs `69.44` on 5 stratified windows).

---

## 2. Exact Pinned Contract & Scoring Data-Flow Trace

Inspected `resources/yug-monsoon-kit/monsoon_r1/{contract,scoring,client,schema_validator,repo_scan,stats}.py` and `resources/yug-monsoon-kit/docs/R1_CONTRACT.md`:

### 2.1 End-to-End Judge Pipeline (Kit-Verified vs Organizer-Hidden)

```text
[HIDDEN ORGANIZER JUDGE] (not shipped in monsoon_r1)
  ├─ Selects 5 blind post-2023-01-10 windows + 1 shock window (warm_ocean, monsoon_stall, null)
  ├─ Polls POST /health (up to 120s) via client.wait_ready(...)
  ├─ Calls POST /simulate 8 times (limit_s = 5.0 s, JSON lists) via client.call_simulate(...)
  │     └─ Returns CaseResult(ok, out=(10,6,32,36) f4, wall_s, server_ms)
  ▼
[EXACT PINNED PRIMITIVES IN monsoon_r1/scoring.py & docs/R1_CONTRACT.md]
  ├─ Per Window w (Leads d in (1, 4, 8) => indices k in [0, 3, 7]):
  │     ├─ Gate: S.validity(rollout, 10) -> [] (else window scores 0.0; NEVER skipped per R1_CONTRACT L50)
  │     ├─ RMSE Skill: S.skill_vs_best_ref(pred_w, true_w, pers_w, clim_w)
  │     │     └─ uses S.rmse_lat -> sqrt((se * lat_weights()[:,None]).mean((-2,-1)).reshape(-1,6).mean(0))
  │     │     └─ SS = 1 - rmse_model / min(rmse_persist, rmse_clim) -> scalar mean ss_w
  │     ├─ Heavy-Rain CSI (> 10.0 mm/day): S.csi(pred_tp, true_tp) -> S.csi_skill(csi_m, csi_p, csi_c) -> csi_w
  │     └─ Day-8 Anti-Blur (index 7): S.variance_ratio(pred[7], true[7]) -> ratios[w, 6]
  ├─ Aggregate Forecast (/45): S.forecast_points(ss_by_window, csi_ss_by_window)
  │     └─ 30 * clip(mean(ss_by_window) / 0.12, 0, 1) + 15 * clip(mean(csi_ss_by_window) / 0.10, 0, 1)
  ├─ Aggregate Anti-Blur (/10): S.anti_blur_points(ratios[W, 6])
  │     └─ 10 * mean_ch(clip(mean_w(ratios) / 0.40, 0, 1) * (mean_w(ratios) <= 2.50))
  ├─ Causality (/15, 1 shock window):
  │     ├─ S.null_consistent(base, zero, std) <= 1e-3 std (else halves total)
  │     ├─ S.causality_checks(base, shock, box, std) for warm_ocean (BoB) & monsoon_stall (Arabian Sea)
  │     │     └─ finite; nonzero (>=1e-3 z); stable (<=5.0 z); localized (day1 near_frac >= 0.60, DILATE=3);
  │     │        propagates (late_d4_d8 > 1.5*day1_out & > 0.02*peak_total) AND no_teleport (far_frac_max < 0.05)
  │     └─ S.causality_points([warm_cc, stall_cc], null_ok)
  └─ Repo Integrity (/15): monsoon_r1.repo_scan.scan(repo_root) -> anti_cheat(8) + code_health(7)
```

### 2.2 What Is Shipped vs Intentionally Hidden
- **Shipped in `monsoon_r1`:** Grid/channel wire contract (`contract.py`), HTTP timeout/decode (`client.py`), pre-submission smoke tester (`schema_validator.py`), static scanner (`repo_scan.py`), train-only clim/stats builder (`stats.py`), and all metric primitives (`scoring.py`).
- **Intentionally Hidden (noted in `client.py` docstring):** Which 5 post-`2023-01-10` windows + 1 shock window the judge picks, and the exact inner loop nesting inside the organizer's blind judge script (whether each window's 3 leads `[0, 3, 7]` are averaged after per-lead `skill_vs_best_ref` [**`P1`**] or passed stacked as `[3, 6, 32, 36]` to `rmse_lat` [**`P2`**]).

---

## 3. Resolution of Scoring Aggregation Discrepancy (`P1`–`P4`)

Evaluated identically across all `719` validation windows (`20190103..20201221`) and a `5`-window seasonal stratified subset (`2019-01-15, 2019-04-20, 2019-06-17, 2019-07-15, 2020-10-15`) in `evaluation/qa_head_to_head.py`:

| Mode | Official Contract Compliant? | Model A (`719` win) `fcst /70` | Model B (`719` win) `fcst /70` | Model A (`5` strat.) `fcst /70` | Model B (`5` strat.) `fcst /70` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`P1` Per-Window × Per-Lead (`qa_model_eval.py`)** | **Yes** *(lower bound)* | `25.62` / **`50.31`** | `5.17` / **`29.23`** | `35.92` / **`60.36`** | `6.48` / **`29.97`** |
| **`P2` Per-Window Stacked `[3,6,32,36]` (`rmse_lat` spec)** | **Yes** *(multi-lead spec)* | `41.83` / **`66.52`** | `20.16` / **`44.22`** | `45.00` / **`69.44`** | `17.66` / **`41.16`** |
| *`P3` Pooled-Per-Lead (`evaluate_val.py`)* | **No (DIAGNOSTIC)** | *`41.61` / `66.29`* | *`16.19` / `40.25`* | *`45.00` / `69.44`* | *`20.35` / `43.84`* |
| *`P4` Pooled-All `[W,3,6,32,36]`* | **No (DIAGNOSTIC)** | *`45.00` / `69.69`* | *`42.14` / `66.20`* | *`45.00` / `69.44`* | *`43.69` / `67.19`* |

### 3.1 Definitive Verdict on the Discrepancy
1. **Why `P3` (`66.29` / `40.25`) and `P4` (`69.69` / `66.20`) are DEFINITIVELY NOT the official contract:**
   - `S.forecast_points(ss_by_window, csi_ss_by_window)` explicitly takes **per-window** skill lists (`len = W`), not `3` lead-level scalars (`P3`) or `1` pooled scalar (`P4`).
   - `R1_CONTRACT.md` line 50 mandates: *"A failed or invalid window scores 0 for it, it is never skipped."* Pooled modes (`P3`/`P4`) cannot isolate an invalid window to score `0.0` (either propagating `NaN` or skewing pooled RMSE). Our `invalid_window_guard_test` confirms `P1` and `P2` assign `0.0` strictly when window 0 is corrupted (`clean_P1 = 35.92 → 31.60`, `clean_P2 = 45.00 → 39.47`, `corrupted_window_scored_zero = True`).
   - `P3` and `P4` are therefore labeled **DIAGNOSTIC ONLY**.
2. **Why `P1` (`50.31` / `29.23`) diverges from `P2` (`66.52` / `44.22`) and `P3` (`66.29` / `40.25`):**
   - **Jensen's Inequality (`E[min(X,Y)] <= min(E[X],E[Y])`):** Choosing `min(RMSE_persist, RMSE_clim)` inside every `(window, lead)` reduces the reference denominator by `3.0%–20.1%` vs pooled (`Day 1 tp` ratio `0.7994`, `Day 4 msl` ratio `0.8576`, `Day 8 tp` ratio `0.8484`).
   - **Single-Day Calm Persistence Spikes:** On individual `(window, lead)` cells where Persistence is briefly near zero error (worst case: `2019-02-04` Lead 8 `msl`, `persist_rmse = 123.65 Pa` vs `clim_rmse = 584.33 Pa`), `1 - r / min(rp, rc)` drops to `-2.782` (Model A, `467.61 Pa`) and `-3.310` (Model B, `532.95 Pa`), pulling `P1` `mean_ss` down (`+0.0458` A / `-0.0224` B) with `11` (`A`) and `15` (`B`) cells `< -1.0`. Stacking `[0, 3, 7]` per window in `P2` eliminates all `< -1.0` spikes in Model A (`0 / 4,314`), lifting `mean_ss` to `+0.1106` (`A`) / `+0.0600` (`B`).
   - **Contract Verdict:** Both `P1` (strict conservative lower bound) and `P2` (faithful multi-lead-per-window `rmse_lat` reading) satisfy `scoring.py` and `R1_CONTRACT.md`; exact loop nesting on the hidden judge remains organizer-side. Under **both** compliant modes, **Model A leads Model B by `+21.08` (`P1`) and `+22.30` (`P2`) points `/70`**.

---

## 4. Hard Admission Gates Audit (Round 2 Eligibility)

| Hard Admission Gate (`R1_CONTRACT.md` L53) | Requirement | Observed Result (Model A / Live `serve.py`) | Status |
| :--- | :--- | :--- | :---: |
| **Gate 1: Physical Validity** | All rollouts finite, `tp >= -1e-6`, `233.15 <= t2m <= 333.15` | `719/719` offline val (`0` viol.); `6/6` `schema_validator`; `5/5` live shock probes (`0` viol.) | **PASS** |
| **Gate 2: Beat Persistence @ Day 4** | Better than Persistence at Day 4 | Raw mean RMSE `43.822 < 59.133`; Norm mean RMSE `0.4502 < 0.6199`; **`6/6` channels better** (`mean_ss_pers = +0.2562`) | **PASS** |
| **Gate 3: Non-Zero Localized Shock Response** | `>= 1` shock nonzero + localized | Live `serve.py` passes **`5/5` `warm_ocean`** and **`4/5` `monsoon_stall`** (`near1 = 0.935..0.994 >= 0.60`, `far <= 0.0156 < 0.05`, mean **`14.625/15`**; 6 offline probes **`14.6875/15`**) | **PASS** |

---

## 5. Live Backend Serving, Offline Isolation & Contract Test Results

All live gates executed **after** Agent-3 backend wiring of `checkpoints/model.pt` (`MonsoonFlow-10 (ms_resunet-anchor)`, `CUDA_VISIBLE_DEVICES=""`, 4 CPU cores pinned), with zero lingering server processes verified afterward.

| Suite / Command | Report / Output | Key Verified Measurements | Status |
| :--- | :--- | :--- | :---: |
| `qa_readiness.py --port 8020` | `evaluation/readiness_report.json` | Startup `2.73 s`; `affinity_4_cores_set=True`; `cuda_hidden=True`; `offline_attempts=0` (`qa_offline_guard.py`); `schema_validator` **`6/6` OK** (`0.13s–0.17s`); 30-call CPU latency `p50=0.151 s`, `p95=0.190 s`, `max=2.388 s <= 5.0 s` (`server_ms_max=105.38 ms`); `peak_rss_mb=1429.95 MB <= 6144 MB` | **PASS** |
| `qa_shock_tests.py --port 8022` | `evaluation/shock_report.json` | `health` reports `MonsoonFlow-10 (ms_resunet-anchor)` (`epoch=22`); `null_ok=True`, `omitted_ok=True`, `det=True` (`max_abs=0.0`); `warm_ocean` `5/5`, `monsoon_stall` `4/5` (winter `2019-01-15` fails `propagates`), mean **`14.625/15`**; adversarial `NaN`/`Inf` rejected with **`HTTP 422`** (`silently_ignored=False`). *(Note: custom `western_ghats_rain_+20mm` on `2020-08-10` showed `far=0.072`, `no_teleport=False`.)* | **PASS (Official)** |
| `pytest tests/test_data_integrity.py tests/test_shocks.py tests/test_qa_server_contract.py -v` | Live stdout (`tests/` untouched) | **`41 passed, 1 skipped` in `12.58 s`** (`8/8` data integrity PASS; `4/4` offline shocks PASS, `test_live_server_shocks_and_null` skipped since port `8000` inactive; **`29/29` `test_qa_server_contract.py` PASS**, including `233.15 K`, `NaN/Inf` `4xx`, `steps=5000` `4xx`, `propagates`, and `not_persistence`) | **PASS** |
| `qa_contract_audit.py` & `qa_data_integrity.py` | `evaluation/contract_audit.json`, `evaluation/data_integrity_report.json` | Pinned kit `689c8798` clean; `train` (`5113d`), `val` (`731d`), `test` (`365d`) disjoint, continuous, bit-identical to NetCDF; `stats.json` & `climatology.npy` strictly 2005–2018; `test.npz` untouched | **PASS** |

---

## 6. Repository Hygiene, `repo_scan` & Submission Size Audit

From refreshed `evaluation/readiness_report.json` (`monsoon_r1.repo_scan`):

| Scan Scope | Total `/15` | Anti-Cheat `/8` | Code Health `/7` | Critical Flags | Warnings | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`tracked_only` (`git ls-files`)** | **`3.5 / 15`** | **`0.0 / 8`** | `3.5 / 7` | **1 Critical: `"no model checkpoint in the repo and no Hugging Face link"`** | `0` | **SUBMISSION BLOCKER:** `checkpoints/model.pt`, `models/`, `training/` are untracked; `training_pipeline=False`, `checkpoint=False`. |
| **`tracked_plus_untracked_not_ignored`** | **`15.0 / 15`** | **`8.0 / 8`** | **`7.0 / 7`** | **`0`** | **`0`** | Clean once tracked (`9` shipped inference files: `serve.py`, `inference/*`, `models/*`, zero network/CUDA/date flags). |
| **`working_tree` (incl. gitignored)** | `5.0 / 15` | `0.0 / 8` | `5.0 / 7` | `14` *(all gitignored)* | `1` *(gitignored)* | All 14 criticals + 1 warning come solely from gitignored `resources/yug-monsoon-kit/monsoon_r1/repo_scan.py` regex list and `data/raw/splits/*.npz` / `.nc`. |

- **Repository Size (`git_size`):** Tracked = `11.00 MB` (`0.0102 GB`); Tracked + untracked-not-ignored = `43.32 MB` (`0.0403 GB << 2.0 GB`). Note: `11` weight files totaling `~29 MB` sit in `checkpoints/runs/` (untracked, not ignored) alongside `checkpoints/model.pt` (`1.36 MB`); prune or gitignore unused `exp1..exp4` intermediates (`last_state.pt`) before committing if keeping git history lean (preserving `exp5` fallback as desired).

---

## 7. Complete Actionable Blocker & Risk Register

### Remaining Submission Blockers (Must Resolve Before Final Submission)
1. **[CRITICAL — Repo Scan]** Commit `checkpoints/model.pt`, `models/`, `training/`, `serve.py`, `inference/`, `dashboard/`, and companion metadata on `tanush/model-a` so `repo_scan` on tracked git files passes **`15.0 / 15`** instead of `3.5 / 15` (decide whether to keep `checkpoints/runs/exp5_msresunet_direct_22ep_control/best.pt` or gitignore `checkpoints/runs/`).
2. **[CRITICAL — Brief Score `15 / 100`]** Author and add the required 2-page `project_brief.pdf` at repository root (currently absent).

### Non-Blocking Findings / Hygiene Items to Note
3. **[HIGH — Stale Pre-Wiring JSON/TXT]** `evaluation/model_eval_val.json` (still records pre-wiring `served.identical_to_persistence = True`) and `evaluation/pytest_results.txt` (still records pre-wiring `19 failed`) were preserved untouched per Agent-5 scope boundaries; live neural server verification was completed cleanly via refreshed `shock_report.json`, `readiness_report.json`, and live `pytest` (`41 passed, 1 skipped`). Refresh `qa_model_eval.py --server` / `pytest_results.txt` if desired before archiving QA evidence.
4. **[INFO — Memory Growth Under Repeated Calls]** `qa_readiness.py` observed `serve.py` peak RSS grow from `417.6 MB` at boot to `1429.95 MB` across 30 calls + `schema_validator` (still `4.3×` under the `6144 MB` limit, vs `418.4 MB` in raw isolated `torch` benchmark). Acceptable for the 8-call blind evaluation, worth monitoring if long-lived stress traffic is expected.
5. **[INFO — Scientific / Custom-Shock Caveats]** Model A fails `propagates` on deep-winter `2019-01-15` `monsoon_stall` (`13.125/15`), shows mild Day-8 cool/dry bias (`t2m -0.254 K`, `tcwv -0.718 kg/m²`, `msl -57.6 Pa`), and exceeded `far < 0.05` on custom `western_ghats_rain_+20mm` on `2020-08-10` (`far = 0.072`). Official shocks stay `far <= 0.0156 << 0.05`.
6. **[VERIFIED CLEAN]** Zero lingering server processes on any port; both `checkpoints/model.pt` (`4f02add2...`) and `team_model_ref_80648e/checkpoints/model.pt` (`1a508d7d...`) unmodified; `data/raw/splits/test.npz` untouched.
