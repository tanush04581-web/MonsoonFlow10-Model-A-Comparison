# CLAUDE.md — Antigravity Multi-Agent Rules & Guardrails

> **System:** Antigravity Autonomous Agent Workspace (`agy`)  
> **Project:** MonsoonFlow-10 (MF10) & Monsoon Multiverse  
> **Event:** Yugaantar DevForge 2026 — Track 2: Multiverse of Monsoon  
> **Topology:** 5 Antigravity Sessions Running in Parallel  
> **Detailed Guides:** See [AGENTS.md](AGENTS.md), [agent_talk/](agent_talk/), and [plan/](plan/)

---

## 1. Five-Agent Team Topology & Ownership Boundaries

Every Antigravity agent operating in this repository MUST strictly respect its exclusive file ownership:

| Agent | Identity & Role | Exclusive Write Ownership | Forbidden Actions |
| :--- | :--- | :--- | :--- |
| **Agent 1** | **Lead-Orchestrator** (Current session) | `CLAUDE.md`, `AGENTS.md`, `README.md`, `plan/`, `docs/`, `agent_talk/AGENT_1_LEAD.md`, `graphify-out/`, `project_brief.pdf` | Does not touch model math or frontend directly |
| **Agent 2** | **Agent-ML** (Data & Modeling) | `data/`, `models/`, `training/`, `checkpoints/`, `agent_talk/AGENT_2_ML.md` | Never touches `serve.py`, `dashboard/`, or `graphify`; no raw `.nc` in git; no external APIs |
| **Agent 3** | **Agent-Backend** (Serving & CPU) | `serve.py` (ROOT!), `inference/`, `agent_talk/AGENT_3_BACKEND.md` | Must keep `serve.py` at root; never touches `models/`, `dashboard/`, or `graphify`; no date literals |
| **Agent 4** | **Agent-Frontend** (Digital Twin UI) | `dashboard/`, `agent_talk/AGENT_4_FRONTEND.md` | Strictly within `dashboard/`; never touches ANY Python file (*.py) or `graphify`; strictly no AI slop |
| **Agent 5** | **Agent-QA** (Auditor & Grader) | `tests/`, `evaluation/`, `agent_talk/AGENT_5_QA.md` | Read-only on rest of repo; never modifies model code, server, or `graphify` |

---

## 2. Graphify Knowledge Graph Ownership Rule (STRICT)

- **EXCLUSIVELY OWNED BY AGENT 1:** Only **Agent 1 (Lead-Orchestrator)** is authorized to run `graphify update .` or modify `graphify-out/`.
- **FORBIDDEN FOR ALL OTHER AGENTS:** Agents 2, 3, 4, and 5 must **NEVER execute `graphify update`** or touch `graphify-out/`. Agent 1 maintains the AST graph at milestone checkpoints.

---

## 3. Mandatory Asynchronous Communication Protocol (`agent_talk/`)

Since all 5 Antigravity sessions run independently, agents coordinate via the shared blackboard in `agent_talk/`:

1. **Read Before Work:** Before starting any phase, an agent MUST read partner logs in `agent_talk/` to discover latest tensor contracts, model checkpoints, and server endpoints.
2. **Append After Change:** Immediately after creating or modifying files, an agent MUST append a status entry to its own log file (`agent_talk/AGENT_X_*.md`).
3. **No Cross-Log Editing:** Agents write ONLY to their assigned log file to prevent file-locking collisions.

---

## 4. Non-Negotiable Competition Constraints

1. **Root Serving Entrypoint:**
   - `serve.py` MUST live at the **repository root** (never in `server/` or a subfolder).
   - Starts via `python serve.py --port 8000` on `0.0.0.0:8000`.
   - Must run in an offline container (`--network none`, 4 CPU cores, 6 GB RAM) and complete a 10-step simulation in $\le 5.0$ seconds (target $< 0.5$ s).
2. **Wire Format & Physical Units:**
   - `POST /simulate` receives and returns **physical units** (`t2m` in K, `tp` in mm/day, `u10`/`v10` in m/s, `msl` in Pa, `tcwv` in kg/m²) in plain JSON lists.
   - Normalization using `stats.json` happens **inside** `simulate()`; predictions must be de-normalized before returning.
   - Output shape: strictly `(10, 6, 32, 36)`. All values finite (zero NaNs, zero Infs).
3. **Physical Constraints:**
   - Precipitation is non-negative: $tp \ge 0$.
   - Temperature is bounded: $t2m \in [233.15, 333.15]$ K ($[-40, +60]^\circ\text{C}$).
4. **Causality & Zero Shock:**
   - Zero shock ($P = 0$ or null) must equal unperturbed baseline within $10^{-6}$ numerical tolerance.
   - Non-zero physical shocks must be localized on Day 1 ($\ge 60\%$ within 3 grid cells), propagate outwards by Day 4–8 ($> 1.5\times$ Day 1), and exhibit zero teleportation ($\le 8.0$ cells/day).
5. **Anti-Leakage & Anti-Cheat:**
   - Chronological split ONLY: 2005–2018 train, 2019–2020 val, 2021 sealed holdout.
   - Zero external weather imports (`open-meteo`, `cdsapi`, `weatherbench2`, `imdlib`, etc.).
   - Zero hardcoded date literals in inference code ($< 5$ in entire codebase).
   - Never commit raw `.nc` files to git. Repository size must remain strictly $< 2$ GB with trained weights inside `checkpoints/model.pt`.

---

## 5. Definition of Done (Executable Gates)

No task is considered complete until it passes the official kit verification commands:
1. `python -m monsoon_r1.schema_validator http://localhost:8000`
2. `python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"` (0 critical flags)
3. Validation against `monsoon_r1.scoring` (Beat Day 4 persistence; Day-8 spatial variance ratio $\ge 0.40$; Heavy-Rain CSI improvement $\ge 0.10$).
