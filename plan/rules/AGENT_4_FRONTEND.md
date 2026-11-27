# Agent 4: Digital Twin Frontend Developer — Strict Operational Rules

> **Identity:** Agent-Frontend (`Agent 4`)  
> **Platform:** Antigravity Autonomous Agent (`agy`)  
> **Primary Responsibility:** Interactive web application in `dashboard/` (Vite, React 19, TypeScript, Tailwind CSS, Leaflet, Recharts), India map layers, Day 0–10 time scrubber, Baseline vs. Shock anomaly diff view, interactive Gaussian-smoothed shock drawing brush, and downstream impact module.  
> **Master Guidelines:** Read and comply with [CLAUDE.md](../../CLAUDE.md), [AGENTS.md](../../AGENTS.md), and [agent_talk/](../../agent_talk/).

---

## 1. Exclusive File Ownership Boundaries

### Permitted Directory (Read & Write):
- `dashboard/` (Entire directory and all its subfolders)
- `agent_talk/AGENT_4_FRONTEND.md` (Your dedicated progress & handoff log)

### Strictly Forbidden (Do NOT Create or Modify):
- **NEVER touch ANY Python file (`*.py`) anywhere in the repository.**
- Never modify `serve.py`, `models/`, `training/`, `data/`, or `checkpoints/`.
- Communicate with backend **solely** via HTTP calls to `http://localhost:8000`.
- **NEVER run `graphify update` or modify `graphify-out/` (Exclusively owned by Agent 1: Lead-Orchestrator).**

---

## 2. Mandatory `agent_talk/` Communication Protocol

1. **Read Before Work:** Read `agent_talk/AGENT_3_BACKEND.md` to see server endpoint updates, sample payload formats, and live status.
2. **Append After Change:** Whenever you build UI components, wire API calls, or finish features, append an update entry to `agent_talk/AGENT_4_FRONTEND.md`:
   - Report component status (Map, Time Scrubber, Shock Lab, Impact Module).
   - Report build health (`npm run build` pass/fail).
   - Request mock or real data from @Agent-Backend when ready.

---

## 3. Design Standards & Unslop Aesthetic (`.agents/skills/unslop/`)

The dashboard must reflect a **domain-authentic meteorological instrument aesthetic**, NOT generic AI SaaS tropes:

- **Color Palette:** Quiet slate/charcoal chrome:
  - Background: `#0B0F17` (Deep Night Slate)
  - Card/Panel Background: `#111827` (Charcoal)
  - Borders: `#1E293B` (Muted Steel Slate)
  - Text: `#F1F5F9` (Primary), `#94A3B8` (Muted Secondary)
- **Forbidden AI Tropes:**
  - ❌ NO generic purple/violet gradients or glowing blobs.
  - ❌ NO decorative emoji badges in navigation tabs or stat cards.
  - ❌ NO micro-font tracked all-caps labels.
- **Craft Principles:**
  - 4px spacing grid.
  - Tabular monospace numbers (`font-mono` / `tabular-nums`) for coordinates, temperatures, and timestamps.
  - Crisp, standard meteorological colormaps:
    - Temperature (`t2m`): Thermal palette (blue $\rightarrow$ green $\rightarrow$ yellow $\rightarrow$ dark red).
    - Precipitation (`tp`): White $\rightarrow$ light blue $\rightarrow$ dark blue $\rightarrow$ violet.
    - Anomaly Diff ($\Delta X_t$): Divergent blue-white-red palette centered at 0.

---

## 4. Core Functional Modules to Implement

1. **India Climate Map Viewer:**
   - Coordinate bounding box: $6.5^\circ\text{N} \dots 37.5^\circ\text{N}$, $65.5^\circ\text{E} \dots 100.5^\circ\text{E}$ ($32 \times 36$ grid cells).
   - Render weather channels: `t2m` (K or °C toggle), `tp` (mm/day), `u10`/`v10` wind vectors, `msl` (hPa), `tcwv` (kg/m²).
   - Show terrain orography as an optional backdrop layer.
2. **Forecast Timeline Scrubber:**
   - Interactive slider from Day 0 (initial condition) through Day 10.
   - Smooth play/pause forward animation with speed control.
3. **Multiverse Shock Laboratory:**
   - **Preset Shock Buttons:**
     - `Warm Bay of Bengal` ($+3.0\text{ K}$ over $8.0\dots 20.0^\circ\text{N}, 82.0\dots 94.0^\circ\text{E}$)
     - `Monsoon Stall` ($-40\%$ wind speed over $8.0\dots 20.0^\circ\text{N}, 65.0\dots 75.0^\circ\text{E}$)
     - `Reset / Zero Shock`
   - **Custom Interactive Shock Brush:**
     - User clicks or paints a localized circle on the map.
     - **Mandatory 2D Gaussian Smoothing:** Apply $\sigma \approx 300\text{ km}$ (~2–3 grid cells) so the shock is physically coherent.
     - Shock Realism Warning: Display visual badge if user perturbation exceeds training bounds ($|\Delta T| > 5\text{ K}$ or negative total rain).
4. **Baseline vs. Shock Anomaly Diff View:**
   - Split-screen comparison or single toggle view rendering $\Delta X_t = X_t^{\text{shock}} - X_t^{\text{baseline}}$.
5. **Downstream Impact Module (Humanitarian & Agricultural Lens):**
   - **Flood Risk Index:** Computes regional flood risk indicators based on heavy-rainfall clusters ($tp \ge 10\text{ mm/day}$ and $\ge 50\text{ mm/day}$).
   - **Crop Moisture Stress Index:** Highlights agricultural zones experiencing severe dry spells or sudden excess deluge.
6. **Live Backend API Integration:**
   - Client calling `GET /health` and `POST /simulate` against `http://localhost:8000`.
   - Display live inference latency (e.g., `Inference: 142 ms (Offline CPU)`).
