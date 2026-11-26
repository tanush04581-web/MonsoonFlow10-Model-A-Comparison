# Multiverse of Monsoon: Round 1 team kit

Everything you need to build and self-test your Round 1 submission. Read `docs/R1_CONTRACT.md` first: it is the exact API contract.

## Install
```bash
pip install monsoon_r1_kit-1.0.0-py3-none-any.whl          # from the Release page, or: pip install .
pip install "monsoon-r1-kit[serve]"                         # adds FastAPI and uvicorn if you use the starter server
```

## Start here
1. Download `india_climate_train.nc` and `stats.json` from the Release (2005-2021 daily, 6 channels, 32 x 36, plus 2 static channels).
2. Copy the starter server: `python -c "import monsoon_r1, pathlib; print(pathlib.Path(monsoon_r1.__file__).parent / 'template' / 'serve.py')"`. Put it at the root of your repo as `serve.py` and replace `simulate()` with your model.
3. Run it: `python serve.py --port 8000`, then check it: `python -m monsoon_r1.schema_validator http://localhost:8000`.
4. Check your repo the way the judge will: `python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('path/to/your/repo'), indent=1))"`.

## What the judge does (summary, the contract has the detail)
Your repo is run **offline** in an organiser container (no internet, 6 GB RAM, 4 CPU cores, no GPU) with `python serve.py --port 8000`, so every weight and static file must be inside the repo. The judge sends 8 blind requests (5 real windows from dates after 2023-01-10, 3 shocks) and scores forecast skill (45), dynamics and perturbation causality (25), repo integrity (15) and your brief (15).

Units on the wire are physical (K, mm/day, m/s, m/s, Pa, kg/m2). Format: nested JSON lists. `doy_sincos` is for the last input day.

## Licence and data
Code: MIT. Data in `india_climate_train.nc`: see `NOTICE`.
