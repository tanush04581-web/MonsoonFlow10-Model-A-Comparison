# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
- Primary (about 70%): Yugaantar DevForge 2026 Track 2 judges with climate and ML backgrounds, evaluating whether the model and its what-if lab behave like real weather. They watch a live demo on a hall projector and later click through on their own laptops.
- Secondary (about 30%): students and members of the public with no meteorology background who want to see what a changed sea or wind does to India's monsoon.

## Product Purpose
MonsoonFlow-10 (MF10) is a compact, physics-constrained neural forecaster for the Indian monsoon. From the last 3 days of 6 variables on a 32 x 36 grid of 1 degree boxes (6.5-37.5 N, 65.5-100.5 E) it rolls the atmosphere forward 10 days, and runs a second, perturbed future for any shock the visitor applies. Success: a judge understands the mechanism in one screen and can run a shock in the lab within a minute.

## Positioning
Two futures from one model run side by side: the difference between the baseline rollout and the shocked rollout is exactly what the shock caused, localized on day 1 and spreading at physical speed afterwards.

## Operating Context
- Lab: full-bleed Lambert conformal map of India, shock presets (warm Bay of Bengal +3 K t2m / +15% tcwv on 8-20N 82-94E; Arabian Sea monsoon stall -40% u10/v10 on 8-20N 65-75E), custom Gaussian brush, day 0-10 timeline, cell probe, downstream basin impacts.
- DEMO mode uses a synthetic toy engine and must be labelled as such; LIVE mode calls serve.py on :8000.
- Runs offline: no external weather services, bundled coastlines (Natural Earth, public domain).

## Capabilities and Constraints
- Six channels: t2m (K, shown in C), tp (mm/day, >= 0), u10, v10 (m/s), msl (Pa, shown in hPa), tcwv (kg/m2).
- Chronological split: train 2005-2018, validate 2019-2020, sealed holdout 2021.
- Zero shock equals baseline within 1e-6. Temperature bounded -40 C to +60 C.
- CPU only, 4 cores, under 0.5 s target per 10-day rollout.
- Frontend lives only in dashboard/ (team ownership rule).

## Evidence on Hand
- The model's own rollouts (live) and the bundled initial state (2019-07-15, validation split).
- No testimonials, customers, pricing or benchmarks exist; none may be invented.

## Product Principles
- Every number on screen is computed from the tensors being drawn.
- Synthetic data is always labelled synthetic.
- The map is the product; chrome serves it.
- Plain-language labels first, technical ids available for experts.
