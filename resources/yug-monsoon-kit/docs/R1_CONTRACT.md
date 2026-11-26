# Round 1: exact API contract (addendum to the PS)

The PS gives the shape of the API. This pins down everything the automated judge depends on, so nothing is a surprise.

## Grid, channels, units (physical units on the wire, never normalised)
32 x 36 cells of 1 deg, centres lat 6.5..37.5 (south to north), lon 65.5..100.5 (west to east). Each cell is the mean of the 4
surrounding 1 deg ERA5 points. Channel order, axis 1 of every state array:

| i | name | meaning | unit |
|---|---|---|---|
| 0 | t2m | 2 m temperature | K |
| 1 | tp | daily total precipitation | mm/day |
| 2 | u10 | 10 m eastward wind | m/s |
| 3 | v10 | 10 m northward wind | m/s |
| 4 | msl | mean sea level pressure | Pa |
| 5 | tcwv | total column water vapour | kg/m2 |

Daily fields: t2m, u10, v10, msl, tcwv are the mean of the 00, 06, 12, 18 UTC samples; tp is the sum over the 24 h of the day.
Normalisation statistics (mean, std per channel) ship in `stats.json` for your own use; the wire is always physical.

## Endpoints
`POST /health` -> `200 {"status": "ready", "model_name": "..."}` (anything else = not ready).

`POST /simulate`, JSON body:
```
{"init_states": [3,6,32,36], "doy_sincos": [sin, cos], "steps": 10, "perturbation": null | [6,32,36]}
```
`init_states` are 3 consecutive days, oldest first; the forecast starts the day after the last one. `doy_sincos = [sin(2 pi doy/365), cos(2 pi doy/365)]` for the LAST input day. `perturbation` is a delta in physical units added to `init_states[-1]` before you simulate (null = none).
Response: `{"rollout": [steps,6,32,36], "inference_ms": float}`; `rollout[k]` is day k+1 after the last input day.

**Wire format (one canonical format).** Plain nested JSON lists of finite floats, in both directions: `init_states` is a 3 x 6 x 32 x 36 list, `perturbation` a 6 x 32 x 36 list (or null), `rollout` a steps x 6 x 32 x 36 list. 10 x 6 x 32 x 36 is 69,120 numbers, about 1 MB of JSON, far under the 5 s limit. The judge always sends lists. (A `{"b64": ..., "shape": [...]}` response is tolerated, but lists are the contract: do not rely on anything else.) NaN and Inf are not valid JSON and fail the validity gate. `python -m monsoon_r1.schema_validator http://localhost:8000` checks all of this against your server.

**Units: physical, always.** `/simulate` receives and returns raw physical units (K, mm/day, m/s, m/s, Pa, kg/m2) and the perturbation is a physical delta. The normalisation constants in `stats.json` are for use INSIDE your model only; normalise on the way in and de-normalise on the way out. The automated checks (rain above 10 mm/day, temperature within -40 and +60 C) use the same physical units.

**Channels and static fields.** The 6 dynamic channels and their order are the table above. The 2 static channels (`land_sea_mask`, `orography_m`) are shipped inside `india_climate_train.nc` (variable `static`, shape 2 x 32 x 36) for training; the API does not send them, bundle them with your model.

**`doy_sincos` refers to the LAST input day** (`init_states[-1]`), not the first forecast day. Forecast step k is therefore day-of-year + k.

**Hosting and warm-up.** The judge calls `/health` repeatedly (every 2 s, up to 120 s) and only starts scoring once it reports `{"status": "ready"}`, so cold starts belong there, not in a scored call. After that the 8 `/simulate` calls run once each, back to back, each with its own 5.0 s limit, no retries. The endpoint must stay up from submission until results are published (assume 48 hours). Free tunnels and sleeping free-tier hosts drop connections: test yours with the schema validator over the public URL, not localhost. **Default: the organisers run your repo themselves, offline, and you need no public URL.** Your repo must have `serve.py` at its root that starts the server on `0.0.0.0` with `python serve.py --port 8000` (a working FastAPI starter is in `monsoon_r1/template/serve.py`). There is no internet during judging, so everything the server needs must be inside the repo: weights (a Hugging Face link is not enough, put the files in), static maps, vendored code. Available libraries: torch (CPU), numpy, scipy, pandas, xarray, scikit-learn, lightgbm, onnxruntime, FastAPI, Flask, uvicorn, gunicorn. Limits: 6 GB RAM, 4 CPU cores, no GPU. Test it the same way: `docker run --rm --network none -v $PWD:/sub:ro -w /sub -p 8000:8000 ...` or simply `python serve.py` and the schema validator.

**Limits.** 10 steps in under 5.0 s wall time as measured by the judge, response under 80 MB, no redirects. The judge only calls public addresses and refuses private or loopback hosts.

## What is tested (8 blind calls)
5 real windows (hidden dates after 2023-01-10, never sent, only state grids and day-of-year), then 3 shocks on one real window:
- **warm_ocean**: +3.0 K on t2m and +15% of the current tcwv over the Bay of Bengal box, lat 8..20, lon 82..94.
- **monsoon_stall**: -40% of the current u10 and v10 over the Arabian Sea box, lat 8..20, lon 65..75.
- **null**: `perturbation` = zeros. The output must equal the unperturbed call (max difference below 1e-3 of the channel std; deterministic or a very small ensemble spread). If it does not, half the causality points are lost.

## Scoring (100)
Forecast skill 45 = 30 (lat-weighted RMSE skill vs the better of persistence and climatology, lead days 1, 4, 8, all 6 channels, 5 windows) + 15 (heavy-rain CSI at 10 mm/day, same leads). A failed or invalid window scores 0 for it, it is never skipped.
Dynamics 25 = 10 anti-blur (day-8 spatial variance between 40% and 250% of the real one, per channel) + 15 perturbation causality (two shocks: response non-zero; localised near the box on day 1, at least 60% of the response energy within 3 cells of it; propagates outward by day 4 to 8 without exceeding 8 cells/day; numerically stable, never above 5 std). A model that does not respond at all gets 0 of these 15, however stable it looks.
Repo 15 and brief 15: see the PS. You can run the same repo scan yourself (`monsoon_r1.repo_scan.scan`, see the README).
Hard gate for Round 2: beat persistence at day 4, pass the validity gate on all 8 cases (no NaN or Inf, rain never negative, t2m within -40 and +60 C), and give a non-zero localised response to at least one shock.
