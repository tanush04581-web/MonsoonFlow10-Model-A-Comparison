# MonsoonFlow-10 (MF10) & Monsoon Multiverse — Master Plan

Welcome to the central planning directory for **Track 2: Multiverse of Monsoon** (Yugaantar DevForge 2026).

This folder contains the complete, battle-tested blueprint developed from deep analysis of the problem statement, competition rules, the official `yug-monsoon-kit` benchmark, and the companion research roadmap.

---

## Plan Structure & Document Index

| Document | Purpose & Key Contents |
| :--- | :--- |
| **[01_PROJECT_EXECUTION_PLAN.md](file:///c:/CS/CS%20Projects/Hackathon/DevFORGE/plan/01_PROJECT_EXECUTION_PLAN.md)** | **End-to-End Hackathon Project Plan:**<br>• Problem statement & constraints analysis ($32 \times 36$ grid, 6 dynamic + 2 static channels)<br>• 100-point competition scoring rubric & win conditions<br>• MonsoonFlow-10 (MF10) neural architecture design<br>• Physics-aware loss formulation (lat-weighted, heavy rain, variance floor)<br>• 5-phase execution roadmap from bootstrap to 2-page brief<br>• Critical traps & anti-cheat disqualification prevention |
| **[02_MULTI_AI_AGENTS_ORCHESTRATION_PLAN.md](file:///c:/CS/CS%20Projects/Hackathon/DevFORGE/plan/02_MULTI_AI_AGENTS_ORCHESTRATION_PLAN.md)** | **Multi-AI-Agent Orchestration Playbook:**<br>• 5-Agent team topology (Lead, ML, Backend, Frontend, QA)<br>• Strict file ownership matrix (preventing file collisions and scope creep)<br>• Inter-agent communication contracts (weights, HTTP wire format, pass/fail gates)<br>• Copy-paste ready initial prompts for each specialist agent<br>• Parallel execution timeline with synchronization barriers<br>• Fail-safe recovery playbook for test failures |
| **[03_DEPENDENCIES_AND_SETUP_PLAN.md](file:///c:/CS/CS%20Projects/Hackathon/DevFORGE/plan/03_DEPENDENCIES_AND_SETUP_PLAN.md)** | **Dependencies & Scaffolding Blueprint:**<br>• Hardware specs (RTX 4060 GPU training vs. 4-core offline CPU serving)<br>• Python dependency tiers (`requirements.txt` vs. `requirements-dev.txt`)<br>• Frontend packages (`React 19 + TypeScript + Vite + Tailwind + Leaflet + Recharts`)<br>• Complete planned repository directory tree<br>• Step-by-step setup runbook (ready for execution once approved) |

---

## Core Guiding Principles

1. **Strict Offline Parity:** The final inference pipeline runs via `python serve.py --port 8000` under `--network none`, 4 CPU cores, and 6 GB RAM, taking $\le 5.0\text{ s}$ per 10-step simulation.
2. **Deterministic Shock Consistency:** Zero shock ($P = 0$) must match baseline forecasts within $10^{-6}$ numerical tolerance.
3. **Anti-Leakage Discipline:** Strict chronological splitting (2005–2018 train, 2019–2020 val, 2021 holdout). Normalization parameters are calculated exclusively on training data.
4. **Physical Reality & Non-Negativity:** Precipitation is strictly non-negative ($tp \ge 0$), temperatures are bounded to physical ranges ($[233.15, 333.15]\text{ K}$), and units on the wire are always physical units.
5. **Separation of Concerns:** Frontend development (`dashboard/`) is completely decoupled from Python backend serving, while ML model training runs on GPU without polluting server logic.
