# AGENTS.md — DevFORGE Project & Antigravity Agent Architecture

> **Platform:** Antigravity Multi-Agent Swarm  
> **Project:** MonsoonFlow-10 (MF10) & Monsoon Multiverse  
> **Event:** Yugaantar DevForge 2026 — Track 2: Multiverse of Monsoon  
> **Core Objective:** Compact, physics-aware spatiotemporal climate digital twin producing 10-day autoregressive rollouts and localized counterfactual shock simulations under offline CPU constraints.

---

## 1. Graphify Knowledge Graph Rules (AGENT 1 ONLY)
- This project is equipped with `graphify`.
- **STRICT OWNERSHIP:** Only **Agent 1: Lead-Orchestrator** is allowed to run `graphify update .` or modify `graphify-out/`.
- **Agents 2, 3, 4, 5 MUST NEVER run `graphify update`**.
- To query architecture without modifying the graph, agents may run `graphify query "<question>"`.
- After substantial code milestones, Agent 1 will run `graphify update .` to keep the AST graph current.

---

## 2. Unslop UI Craft Standards
- All frontend UI (`dashboard/`) must comply with the `unslop` skill (`.agents/skills/unslop/`).
- **No AI Tropes:** Zero generic purple/violet gradients, no glowing blobs, no emoji badges in navigation/cards, no micro-font tracked labels.
- **Meteorological Instrument Aesthetic:** Quiet slate/charcoal chrome (`#0B0F17`, `#111827`, `#1E293B`), 4px spacing grid, tabular numbers, clear interactive 2D geospatial weather visualization with crisp meteorological color scales.

---

## 3. Five-Agent Team Topology & Strict Boundaries

| Agent | Responsibility | Exclusive File Ownership | Constraints |
| :--- | :--- | :--- | :--- |
| **Agent 1: Lead-Orchestrator** | Coordination, scoring gates, briefing, Graphify | `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `README.md`, `plan/`, `docs/`, `agent_talk/AGENT_1_LEAD.md`, `graphify-out/`, `project_brief.pdf` | Does not touch model math or frontend directly |
| **Agent 2: Agent-ML** | Data pipeline, MF10 architecture, GPU training | `data/`, `models/`, `training/`, `checkpoints/`, `agent_talk/AGENT_2_ML.md` | Never touches `serve.py`, `dashboard/`, or `graphify`; no raw `.nc` in git; no external weather APIs |
| **Agent 3: Agent-Backend** | Fast offline inference, REST API | `serve.py`, `inference/`, `agent_talk/AGENT_3_BACKEND.md` | Must keep `serve.py` at root; never touches `models/`, `dashboard/`, or `graphify`; no datetime literals |
| **Agent 4: Agent-Frontend** | Interactive React 19 UI, map, shocks | `dashboard/`, `agent_talk/AGENT_4_FRONTEND.md` | Strictly within `dashboard/`; never touches ANY Python file (*.py) or `graphify`; strictly no AI slop |
| **Agent 5: Agent-QA** | Independent evaluation & audit | `tests/`, `evaluation/`, `agent_talk/AGENT_5_QA.md` | Read-only on `models/`, `serve.py`, and `graphify`; never modifies weights to pass tests |

---

## 4. Asynchronous Communication Blackboard (`agent_talk/`)
- Every agent has an assigned blackboard log in `agent_talk/`.
- **Mandatory Read-Before-Work:** Inspect partner logs in `agent_talk/` before starting work.
- **Mandatory Append-After-Change:** Append action summaries, file changes, and interface contracts to your own log after any modification.

---

## 5. Non-Negotiable Competition Constraints
1. **Offline CPU Serving:** `python serve.py --port 8000` must run with `--network none`, 4 CPU cores, $\le 6$ GB RAM, and complete a 10-step simulation in $\le 5.0$ seconds.
2. **Physical Units & Validity:**
   - Output shape: strictly `(10, 6, 32, 36)`.
   - Channels: `t2m` ($K$), `tp` ($\text{mm/day}$ $\ge 0$), `u10` ($\text{m/s}$), `v10` ($\text{m/s}$), `msl` ($\text{Pa}$), `tcwv` ($\text{kg/m}^2 \ge 0$).
   - Finite values only (zero NaNs, Infs).
3. **Causality & Zero Shock:**
   - Zero shock ($P = 0$) output must equal baseline within $10^{-6}$ numerical tolerance.
   - Non-zero physical shocks must be localized on Day 1, propagate outwards by Day 4-8 ($\le 8.0$ grid cells/day), and exhibit zero teleportation.
4. **Anti-Leakage & Anti-Cheat:**
   - Strict chronological split: 2005–2018 train, 2019–2020 val, 2021 holdout.
   - Zero external weather packages (`open-meteo`, `cdsapi`, `weatherbench2`, etc.).
   - Repository size strictly $< 2$ GB.
