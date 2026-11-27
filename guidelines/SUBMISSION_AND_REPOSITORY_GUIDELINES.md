# Yugaantar DevForge 2026 — Repository & Submission Guidelines

> **Event:** Yugaantar DevForge 2026 — Track 2: Multiverse of Monsoon  
> **Source Reference:** DevForge Tracks Brief (Pages 1, 8–10) & Round 1 Contract  
> **Official Repo:** [Hardik2811-wq/MonsoonFlow10](https://github.com/Hardik2811-wq/MonsoonFlow10)

---

## 1. Executive Summary & Repository Naming
- **Repository Name Flexibility:** According to the official DevForge guidelines (Pages 1, 8–10), there is **no naming restriction** imposed on the repository (e.g., `MonsoonFlow10` is fully compliant).
- **Public Visibility:** The repository **must be public** throughout the hackathon and evaluation cycle.
- **Creation & Timing:** The repository must be created during the active hackathon period and will be frozen at the submission deadline. **No commits made after the official deadline will be evaluated.**

---

## 2. Non-Negotiable Repository Layout & Root Entrypoint

The automated evaluation harness executes directly against the root of the repository inside an isolated, unnetworked container.

### 2.1 Root `serve.py` Requirement
- **Exact File Path:** `serve.py` **must live at the repository root** (`/serve.py`), not inside a subfolder such as `server/` or `api/`.
- **Startup Command:** The automated judge executes:
  ```bash
  python serve.py --port 8000
  ```
- **Network Interface:** Must listen on `0.0.0.0:8000`.
- **Pre-execution / Cold Start:**
  - All model weights, normalization statistics (`stats.json`), and static terrain tensors must be loaded at import time or during the FastAPI startup event.
  - The judge polls `POST /health` every 2 seconds for up to 120 seconds. Once `200 OK` is returned, simulation tests commence.

### 2.2 Decoupled Frontend & Backend
- **Separation of Concerns:** The interactive visualization dashboard must be kept completely separate from backend serving code in a dedicated `dashboard/` directory.
- **Communication:** The frontend communicates with the backend solely via standard HTTP requests (`http://localhost:8000/simulate` and `/health`).
- **Offline Parity:** The backend must remain completely functional and verifiable on its own without requiring the frontend dev server to be running.

---

## 3. Model Weights & Offline Container Constraints

The evaluation environment runs inside an offline container with strict hardware and security limits:

| Parameter | Constraint | Compliance Architecture |
| :--- | :--- | :--- |
| **Network Access** | `--network none` (Strictly Offline) | Zero Hugging Face downloads, zero external weather APIs (`open-meteo`, `cdsapi`, etc.). |
| **Compute Limit** | 4 CPU cores, 6.0 GB RAM | Optimized PyTorch inference with `torch.set_num_threads(4)`. |
| **Inference Latency** | $\le 5.0$ seconds per 10-step call | Compact MF10 model runs in $< 0.5$ s on CPU. |
| **Model Weight Storage** | Committed inside the repository | Saved to `checkpoints/model.pt` (Git LFS is supported). |
| **Total Repository Size** | Strictly $< 2.0$ GB | Raw `.nc` datasets excluded via `.gitignore`; only bundle compact checkpoints. |

---

## 4. Anti-Cheat & Clean AST Directives (`repo_scan.py` — 15 Points)

The automated grading harness includes a static AST and binary scanner (`monsoon_r1.repo_scan.scan('.')`) that awards **15 points** for code health and anti-cheat compliance. The following triggers result in an immediate loss of points:

1. **No External Weather Imports:**
   - Any import of `open-meteo`, `cdsapi`, `weatherbench2`, `imdlib`, `openweathermap`, `nomads`, or similar external packages will cause immediate disqualification.
2. **Zero Hardcoded Date Literals:**
   - Less than 5 date string literals (`YYYY-MM-DD`) permitted in the entire codebase. Inference code must never inspect timestamps or rely on calendar lookups.
3. **No Multi-Year Precomputed Archives:**
   - Pre-computed target lookups (`.npy` arrays with $> 400$ days) are strictly forbidden. Only a 1-year (366-day) `climatology.npy` is permitted as a seasonal baseline.
4. **No Raw NetCDF Files in Git:**
   - Raw climate datasets (`.nc`, `.nc4`, `.grib`, `.zarr`) must remain in `.gitignore`. Shipping multi-gigabyte raw data triggers static scan warnings and risks breaching the 2 GB repository ceiling.

---

## 5. Scoring Breakdown & Deliverables Summary

| Priority | Category | Points | Core Deliverables |
| :---: | :--- | :---: | :--- |
| **1** | **Forecast Skill** | **45** | • Latitude-weighted RMSE ($\text{SS} \ge 0.12$ vs. min(persist, clim))<br>• Heavy-rain CSI ($\ge 10.0\text{ mm/day}$, $\text{CSI\_SS} \ge 0.10$) |
| **2** | **Dynamics & Causality** | **25** | • Anti-blur Day-8 spatial variance ratio $\in [0.40, 2.50]$<br>• Localized Day-1 response ($\ge 60\%$) with outward propagation ($\le 8\text{ cells/day}$)<br>• Zero-shock exact consistency ($\text{atol} < 10^{-6}$) |
| **3** | **Repository Correctness** | **15** | • Zero critical flags in `repo_scan.py`<br>• Real training code & runnable `serve.py`<br>• Repo size $< 2.0\text{ GB}$ |
| **4** | **Project Brief** | **15** | • 2-page PDF brief (`project_brief.pdf`) documenting architecture, curriculum, physics losses, and humanitarian/flood impact analysis |

---

## 6. Official Verification Gate Commands

Before any final submission or tag, run these commands to confirm complete compliance:

```bash
# 1. Start root server
python serve.py --port 8000

# 2. Run official black-box schema & latency validator
python -m monsoon_r1.schema_validator http://localhost:8000

# 3. Run static anti-cheat & repo scan
python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"
```
