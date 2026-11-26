# Agent-5 QA Report — MonsoonFlow-10 (2026-11-26, branch tanush/model-a)

Source of truth: pinned kit `resources/yug-monsoon-kit` @ `689c879` (clean). All metrics from `monsoon_r1.scoring`, except
the QA-labelled ones: lat-weighted MAE and bias, pattern correlation and ACC. Inputs come from the 2019–2020 validation
split. The 2021 test split was checked only for shape, continuity and isolation; no metrics were computed on it.

## Artefacts
| Phase | Script | Output |
|---|---|---|
| 1 contract | `evaluation/qa_contract_audit.py` | `evaluation/contract_audit.json` |
| 2 data | `evaluation/qa_data_integrity.py` | `evaluation/data_integrity_report.json` |
| 3 models | `evaluation/qa_model_eval.py [--server] [--checkpoints-only --checkpoint ...]` | `evaluation/model_eval_val.json` |
| 4 shocks | `evaluation/qa_shock_tests.py` | `evaluation/shock_report.json` |
| 5 readiness | `evaluation/qa_readiness.py` (+ `qa_offline_guard.py`) | `evaluation/readiness_report.json` |
| gates | `tests/test_qa_server_contract.py`, `tests/test_data_integrity.py`, `tests/test_shocks.py` | `evaluation/pytest_results.txt` |

## Phase 2 — data integrity: PASS
- NetCDF: 6209 days, 2005-01-01..2021-12-31, daily with no gaps or duplicates, 0 NaN and 0 Inf. Channel order, lat/lon and units all match the contract.
- Splits: train 5113 days (2005–2018), val 731 (2019–2020), test 365 (2021). Each is continuous and bit-identical to the NetCDF slice. Pairwise overlap is 0 and the union equals the raw archive. Windows: 5101 / 719 / 353.
- `checkpoints/stats.json` is train-only: it matches a float64 train recomputation to 1.4e-5 std. A train+val source would be off by 1.2e-2 std.
- `checkpoints/climatology.npy` is bit-identical to `kit stats.climatology(train)`.
- Inference code (`serve.py`, `inference/`) never references data files. No training code references `test.npz`.
- Low-severity mismatch: training normalises with `splits/train_stats.json` (float32 reduction) while `checkpoints/stats.json` was reduced in float64. The gap is at most 4.4e-4 std (msl 0.27 Pa).

## Phase 3 — validation 2019–2020 (719 windows), kit `rmse_lat`
| Lead | Persist t2m / tp | Clim t2m / tp | exp3 ms_resunet t2m / tp | exp3 mean SS vs best ref | exp3 CSI@10 (P / C) |
|---|---|---|---|---|---|
| 1 | 1.029 / 6.267 | 1.742 / 6.756 | 0.880 / 4.556 | +0.209 | 0.530 (0.438 / 0.288) |
| 4 | 1.917 / 9.167 | 1.740 / 6.750 | 1.635 / 6.510 | +0.085 | 0.367 (0.230 / 0.288) |
| 8 | 2.254 / 9.431 | 1.739 / 6.736 | 1.884 / 6.892 | **−0.014** | 0.329 (0.204 / 0.288) |

| Candidate | Day-4 SS vs persistence | Day-8 anti-blur /10 | Emulated forecast pts /45 | Validity | Shock pts /15 (3 windows) | CPU 10-step |
|---|---|---|---|---|---|---|
| persistence | 0 | 8.33 (tp ratio 2.60 > 2.5) | 0 | 719/719 OK | — | — |
| climatology | — | 10.0 | 0 | 719/719 OK | — | — |
| served (`serve.py`) | 0: identical to persistence (max diff 0.0) | 8.33 | 0 | OK | 11.25 | 0.16–0.46 s |
| exp1 convlstm (offline) | +0.228 | 10.0 | 15.5 | OK | 13.1 / 15 / 13.1 | 0.24 s |
| exp2 st_rescnn (offline) | +0.234 | 10.0 | 18.6 | OK | 13.1 / 13.1 / 11.25 | 0.18 s |
| **exp3 ms_resunet (offline)** | **+0.245** | 10.0 | **22.2** | OK | 15 / 15 / 13.1 | 0.16 s |
| exp4 ms_resunet_anchor | NOT EVALUATED: still training at audit time | | | | | |

- Emulated points use kit `skill_vs_best_ref`, `csi_skill` and `forecast_points` per window over leads 1/4/8. The grader's own per-window aggregation is hidden, so treat these as estimates.
- No autoregressive instability. Normalised RMSE rises smoothly and saturates (exp3: 0.29 → 0.51 std), with zero non-finite values.
- Every trained model is worse than climatology at day 8. This caps the 30-point RMSE skill term at about 8 points.

## Phase 4 — shocks against the running server (5 val windows, kit `make_shock`, `causality_checks`)
- Null equals zeros equals an omitted perturbation; the outputs are deterministic and the max diff is 0.0.
- warm_ocean, monsoon_stall and 4 custom `box_mask` shocks all come back nonzero, localized (near_frac_day1 = 1.0), stable and no_teleport. `propagates` is False in every case, as expected for persistence: the response never leaves the box. The server scores 11.25/15 per window.
- A NaN perturbation returns HTTP 200 and is silently ignored. An Inf perturbation returns 200 and is silently clamped to 333.15 K.

## Phase 5 — readiness (server pinned to 4 cores, CUDA hidden, outbound sockets blocked)
- The schema validator passes 6/6. 30 calls finish in at most 0.68 s (p50 0.18 s) while a GPU training job ran concurrently. Peak RSS is 70 MB (persistence path, no torch loaded). The offline guard logged 0 outbound attempts. The shipped inference files have no network imports or CUDA references.
- Git size: 9.1 MiB packed; tracked 10.4 MB; tracked plus untracked-not-ignored 33.6 MB. Both are far below 2 GB, and no `.nc` file is tracked.
- `repo_scan`:
  - On the working tree: 0/8 anti-cheat with 14 criticals. All of them come from gitignored content: the kit's own `repo_scan.py` pattern list under `resources/`, and `data/raw/splits/{train,val}.npz`.
  - On the git-tracked tree (HEAD content): 3.5/15, critical "no model checkpoint".
  - On tracked plus untracked-not-ignored: 15/15 with 0 flags, but only because `checkpoints/runs/*.pt` would be committed.
