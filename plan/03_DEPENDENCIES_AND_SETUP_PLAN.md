# Dependencies & Setup Plan

> **Hackathon:** Yugaantar DevForge 2026 — Track 2: Multiverse of Monsoon  
> **Status:** Planning Document (Do not install or download yet — ready for execution once approved)  
> **Target Hardware:**  
> • Training: Local Workstation with NVIDIA GeForce RTX 4060 (8 GB VRAM, CUDA)  
> • Serving/Judging Parity: Offline 4-Core CPU, 6 GB RAM, no GPU, `--network none`

---

## 1. Environment & Hardware Requirements

### 1.1 Software Runtimes
| Runtime | Recommended Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.10.x` or `3.11.x` | Core ML training, data processing, offline serving, evaluation |
| **Node.js** | `v18.x` or `v20.x` LTS | Frontend build environment (Vite, TypeScript, React) |
| **npm** | `v9.x` or `v10.x` | Package management for dashboard |
| **Git** | `2.40+` | Version control, tracking commit sizes (`< 2 GB`) |
| **CUDA Toolkit** | `12.1` or `11.8` | GPU-accelerated PyTorch training on RTX 4060 |

### 1.2 Target Hardware Profile
- **Training (Local Dev):**
  - GPU: NVIDIA GeForce RTX 4060 Laptop/Desktop (8 GB GDDR6 VRAM)
  - Memory: 16 GB+ RAM
  - Storage: Fast NVMe SSD (~10 GB free for raw NetCDF data and PyTorch checkpoints)
- **Judging / Serving Container Profile (Organizer Environment):**
  - CPU: 4 vCPU cores (Intel/AMD x86_64)
  - RAM: 6.0 GB ceiling
  - Network: Fully disconnected (`--network none`)
  - Inference Latency Limit: $\le 5.0\text{ s}$ per 10-step simulation
  - Weight File Limit: Submissions must be self-contained (`< 2.0 GB` total repo size)

---

## 2. Python Dependencies Breakdown

To ensure zero friction between local GPU training and offline CPU judging, we split Python dependencies into two specifications:
1. `requirements.txt`: The lean, judge-compliant package list matching pre-installed container libraries.
2. `requirements-dev.txt`: Extended packages for local GPU training, dataset inspection, testing, and diagnostics.

### 2.1 Lean Serving Dependencies (`requirements.txt`)
These represent the exact offline container runtime required to run `python serve.py --port 8000`:

```text
# --- Deep Learning & Numerical Core (CPU-compatible) ---
torch>=2.0.0,<2.5.0
numpy>=1.23.0,<2.0.0
scipy>=1.10.0,<1.15.0

# --- High-Performance Offline Web Server ---
fastapi>=0.100.0,<0.115.0
uvicorn[standard]>=0.22.0,<0.31.0
pydantic>=2.0.0,<3.0.0

# --- Evaluation & Baseline Verification ---
scikit-learn>=1.2.0,<1.6.0
xarray>=2023.1.0
pandas>=2.0.0,<2.3.0
```

### 2.2 Development & Training Dependencies (`requirements-dev.txt`)
Required on the local machine for data preparation, GPU acceleration, and test execution:

```text
# Inherit base requirements
-r requirements.txt

# --- GPU Acceleration (Installed with CUDA index) ---
# Note: Install via: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- Climate Data & NetCDF Backend ---
netCDF4>=1.6.2
h5netcdf>=1.2.0
cftime>=1.6.2

# --- Experiment Diagnostics & Visualization ---
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0
tqdm>=4.65.0

# --- Automated Testing & HTTP Client Testing ---
pytest>=7.3.0
pytest-cov>=4.1.0
httpx>=0.24.0
requests>=2.31.0

# --- Code Quality & Anti-Cheat Audit Tools ---
ruff>=0.1.0
```

---

## 3. Frontend Dashboard Dependencies (`dashboard/package.json`)

The frontend is completely isolated inside `dashboard/` and built with **React 19 + TypeScript + Vite + Tailwind CSS**:

### 3.1 Production Packages (`dependencies`)
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `react` | `^18.3.1` or `^19.0.0` | Declarative UI component library |
| `react-dom` | `^18.3.1` or `^19.0.0` | DOM rendering layer |
| `leaflet` | `^1.9.4` | Interactive mapping engine for India regional grid |
| `react-leaflet` | `^4.2.1` or `^5.0.0` | React wrapper for Leaflet map components |
| `recharts` | `^2.12.7` | Time-series and rollout stability diagnostic charts |
| `lucide-react` | `^0.395.0` | Clean iconography for climate layers & controls |
| `clsx` | `^2.1.1` | Utility for constructing conditional class names |
| `tailwind-merge` | `^2.3.0` | Conflict-free Tailwind CSS class merging |

### 3.2 Development Packages (`devDependencies`)
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `vite` | `^5.3.0` | Ultra-fast frontend build tool and dev server |
| `@vitejs/plugin-react` | `^4.3.0` | React JSX/TSX fast-refresh plugin |
| `typescript` | `^5.2.2` | Static type safety and contract enforcement |
| `tailwindcss` | `^3.4.4` | Utility-first CSS framework |
| `postcss` | `^8.4.38` | CSS processing tool |
| `autoprefixer` | `^10.4.19` | Automatic vendor prefixing for CSS |
| `@types/react` | `^18.3.3` | TypeScript definitions for React |
| `@types/react-dom` | `^18.3.0` | TypeScript definitions for React DOM |
| `@types/leaflet` | `^1.9.12` | TypeScript definitions for Leaflet |

---

## 4. Planned Repository Directory Structure

```text
DevFORGE/
├── .gitignore                   # Critical: ignores *.nc, node_modules/, large caches
├── CLAUDE.md                    # Core operational rules for AI coding agents
├── README.md                    # Master submission guide and instructions
├── requirements.txt             # Lean offline serving requirements
├── requirements-dev.txt         # GPU training & development requirements
├── serve.py                     # ROOT INFERENCE ENTRYPOINT (Mandatory at root!)
│
├── plan/                        # Strategic planning documents
│   ├── README.md
│   ├── 01_PROJECT_EXECUTION_PLAN.md
│   ├── 02_MULTI_AI_AGENTS_ORCHESTRATION_PLAN.md
│   └── 03_DEPENDENCIES_AND_SETUP_PLAN.md
│
├── resources/                   # External reference kits & data tools
│   └── yug-monsoon-kit/         # Official Round 1 grading & validation kit
│       └── monsoon_r1/          # scoring.py, contract.py, repo_scan.py, validator
│
├── data/                        # Climate data processing pipeline
│   ├── __init__.py
│   ├── preprocessing.py         # NetCDF slicing & window construction
│   ├── dataset.py               # Chronological PyTorch Dataset & DataLoader
│   ├── statistics.py            # Computes stats.json & climatology.npy
│   └── static_fields.py         # Extracts & bundles 2x32x36 terrain tensor
│
├── models/                      # Deep learning architecture (MF10)
│   ├── __init__.py
│   ├── encoder.py               # Multi-scale Residual CNN spatial encoder
│   ├── convlstm.py              # 2-layer 64-channel ConvLSTM temporal core
│   ├── decoder.py               # Residual prediction head (Delta X)
│   ├── mf10.py                  # Complete MonsoonFlow-10 model wrapper
│   ├── physics_losses.py        # Lat-weighted, heavy-rain CSI, & variance losses
│   └── shock_engine.py          # Spatial perturbation injection & smoothing
│
├── training/                    # Curriculum training scripts
│   ├── __init__.py
│   ├── train.py                 # Multi-stage rollout curriculum trainer
│   ├── validate.py              # Validation loop with monsoon_r1.scoring
│   └── scheduler.py             # Learning rate & loss weight progression
│
├── inference/                   # CPU execution & benchmark helpers
│   ├── __init__.py
│   ├── predictor.py             # Inference pipeline with normalization
│   └── benchmark.py             # 4-core CPU latency validation
│
├── checkpoints/                 # Saved model weights (< 500 MB)
│   ├── .gitkeep
│   ├── model.pt                 # Trained PyTorch checkpoint (COMMITTED)
│   ├── stats.json               # Channel mean & std (COMMITTED)
│   ├── climatology.npy          # 366-day climatology tensor (COMMITTED)
│   └── static.npy               # 2x32x36 land_sea & orography (COMMITTED)
│
├── evaluation/                  # Grader wrappers & diagnostic suite
│   ├── __init__.py
│   ├── eval_scoring.py          # Wraps monsoon_r1.scoring on Pool A
│   └── run_repo_scan.py         # Static anti-cheat & code health audit
│
├── tests/                       # Unit & integration tests
│   ├── test_data_integrity.py   # Coordinate alignment, shapes, zero-leakage
│   ├── test_api.py              # /health & /simulate contract verification
│   └── test_shocks.py           # Causality checks & null-consistency
│
└── dashboard/                   # Interactive React Digital Twin
    ├── package.json             # Frontend dependencies
    ├── vite.config.ts           # Vite bundler config
    ├── tailwind.config.js       # Tailwind theme & color definitions
    ├── tsconfig.json            # TypeScript configuration
    ├── index.html               # Web root
    └── src/
        ├── main.tsx             # Application mount
        ├── App.tsx              # Root component & state coordinator
        ├── components/
        │   ├── MapViewer.tsx    # Leaflet 32x36 spatial grid overlay
        │   ├── TimeScrubber.tsx # Day 0 to 10 forecast scrubber
        │   ├── ShockLab.tsx     # Presets (Bay of Bengal, Arabian Sea) & Brush
        │   ├── AnomalyView.tsx  # Side-by-side Delta X_t difference map
        │   ├── MetricsBar.tsx   # Latency, variance ratio, physical checks
        │   └── ImpactCards.tsx  # Flood risk & crop moisture stress indicators
        ├── services/
        │   └── api.ts           # Live HTTP client to POST /simulate
        └── utils/
            ├── colormaps.ts     # Weather & anomaly color palettes
            └── coordinates.ts   # 32x36 India grid mapping
```

---

## 5. Setup Runbook (Step-by-Step Execution Guide)

When ready to begin setup, execute the following steps in sequence:

### Step 1: Initialize Git & Lock `.gitignore`
Prevent accidental commitment of multi-gigabyte `.nc` climate files:
```bash
# Verify .gitignore contains:
*.nc
*.nc4
*.grib
*.grib2
*.zarr
data/raw/
node_modules/
__pycache__/
*.pyc
.env
dist/
build/
```

### Step 2: Set Up Python Virtual Environment
```powershell
# In Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install PyTorch with CUDA for RTX 4060
```powershell
# Install GPU-accelerated PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA detects the RTX 4060:
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step 4: Install Remaining Python Packages
```powershell
# Install development and climate data packages:
pip install -r requirements-dev.txt
```

### Step 5: Verify the Official Reference Kit
```powershell
# Clone or position yug-monsoon-kit:
git clone https://github.com/OfficialAbhinavSingh/yug-monsoon-kit resources/yug-monsoon-kit

# Test monsoon_r1 import:
python -c "import sys; sys.path.append('resources/yug-monsoon-kit'); from monsoon_r1 import contract, scoring; print('Kit version verified')"
```

### Step 6: Scaffold & Setup Frontend Dashboard
```powershell
# Create Vite React-TS app in dashboard/:
npm create vite@latest dashboard -- --template react-ts

# Install frontend dependencies:
cd dashboard
npm install
npm install leaflet react-leaflet recharts lucide-react clsx tailwind-merge
npm install -D tailwindcss postcss autoprefixer @types/leaflet
npx tailwindcss init -p
cd ..
```

---

## 6. Pre-Flight Verification Checklist

Before starting any neural model training:
- [ ] PyTorch detects local NVIDIA RTX 4060 with CUDA enabled.
- [ ] Raw `.nc` dataset is accessible locally under `data/raw/` but safely gitignored.
- [ ] `checkpoints/` directory is prepared to store `model.pt`, `stats.json`, and `climatology.npy`.
- [ ] `python -m monsoon_r1.repo_scan` runs without crashing.
- [ ] Dashboard builds cleanly with `npm run build` inside `dashboard/`.
