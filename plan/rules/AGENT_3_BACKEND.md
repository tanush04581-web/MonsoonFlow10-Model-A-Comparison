# Agent 3: Backend & Inference Systems Engineer — Strict Operational Rules

> **Identity:** Agent-Backend (`Agent 3`)  
> **Platform:** Antigravity Autonomous Agent (`agy`)  
> **Primary Responsibility:** Offline CPU inference serving, root `serve.py` implementation, physical de-normalization pipeline, perturbation input handling, and CPU latency optimization.  
> **Master Guidelines:** Read and comply with [CLAUDE.md](../../CLAUDE.md), [AGENTS.md](../../AGENTS.md), and [agent_talk/](../../agent_talk/).

---

## 1. Exclusive File Ownership Boundaries

### Permitted Directories (Read & Write):
- `serve.py` (**CRITICAL: Must live at the REPOSITORY ROOT!**)
- `inference/` (`predictor.py`, `benchmark.py`, `loader.py`, `schemas.py`)
- `agent_talk/AGENT_3_BACKEND.md` (Your dedicated progress & handoff log)

### Strictly Forbidden (Do NOT Create or Modify):
- `models/` and `training/` (Owned exclusively by Agent 2: Agent-ML)
- `data/` (Owned exclusively by Agent 2: Agent-ML)
- `dashboard/` (Owned exclusively by Agent 4: Agent-Frontend)
- `tests/` and `evaluation/` (Owned exclusively by Agent 5: Agent-QA)
- **NEVER move `serve.py` into `server/` or any subfolder.**
- **NEVER run `graphify update` or modify `graphify-out/` (Exclusively owned by Agent 1: Lead-Orchestrator).**

---

## 2. Mandatory `agent_talk/` Communication Protocol

1. **Read Before Work:** Read `agent_talk/AGENT_2_ML.md` to see when `checkpoints/model.pt` and `stats.json` are generated and their exact channel orders.
2. **Append After Change:** Whenever you update `serve.py`, run benchmarks, or verify `/simulate`, append an entry to `agent_talk/AGENT_3_BACKEND.md`:
   - State server health (`http://localhost:8000/health`).
   - Report 4-core CPU latency in milliseconds.
   - Notify @Agent-Frontend that the API is ready for frontend consumption.
   - Notify @Agent-QA that the server is ready for `schema_validator`.

---

## 3. Core Technical Directives & Constraints

1. **Root `serve.py` Execution Contract:**
   - Must be runnable via: `python serve.py --port 8000` listening on `0.0.0.0:8000`.
   - Framework: FastAPI + Uvicorn.
   - Operates in offline container: `--network none`, 4 CPU cores, 6.0 GB RAM, zero GPU.
   - Wall-time limit: $\le 5.0\text{ s}$ per 10-step simulation call (target $< 0.5\text{ s}$).
2. **Cold-Start Preloading (Import Time):**
   - Load `checkpoints/model.pt`, `checkpoints/stats.json`, and static terrain tensor into memory at module import or startup lifespan.
   - **Never** re-read files or re-instantiate models inside `/simulate` request handlers.
   - Set `torch.set_num_threads(4)` at startup.
   - Set model to deterministic evaluation mode: `model.eval()`.
3. **Wire Format & Physical Units Boundary:**
   - `POST /health` returns: `200 {"status": "ready", "model_name": "MonsoonFlow-10"}`
   - `POST /simulate` receives:
     ```json
     {
       "init_states": [[[[... 36] ... 32] ... 6] ... 3],
       "doy_sincos": [sin_val, cos_val],
       "steps": 10,
       "perturbation": null or [[[[... 36] ... 32] ... 6]]
     }
     ```
   - Input values are in **raw physical units** ($K, \text{mm/day}, \text{m/s}, \text{Pa}, \text{kg/m}^2$).
   - Normalize on input using `stats.json` (`mean`, `std`).
   - Run 10-step forward rollout using `torch.no_grad()`.
   - **De-normalize back to physical units** before returning plain nested JSON lists.
   - Clamp $tp \ge 0$ and $t2m \in [233.15, 333.15]\text{ K}$.
4. **Perturbation Handling & Zero-Shock Contract:**
   - If `perturbation` is provided, add it as a physical delta directly to `init_states[-1]`: $X'_t = X_t + P$.
   - If `perturbation` is `null` or all zeros, the output must match the unperturbed baseline within $10^{-6}$ numerical tolerance (`atol < 1e-6`).
5. **Anti-Cheat & Clean AST Directives:**
   - **NEVER** import `datetime`, `date`, `Timestamp`, or date formatting libraries in `serve.py` or `inference/`.
   - **NEVER** write string literals matching `YYYY-MM-DD`.
   - **NEVER** make outbound HTTP calls or import external weather packages.

---

## 4. Definition of Done
Your task is complete ONLY when you run and pass:
```bash
python -m monsoon_r1.schema_validator http://localhost:8000
```
Verifying: `/health` 200, valid JSON and base64 formats, zero-shock parity, and latency $< 5.0\text{ s}$.
