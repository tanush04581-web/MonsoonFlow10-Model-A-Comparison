"""Static scan of a team's repo (never executes anything from it). 15 points: anti-cheat 8 + code health 7.

Anti-cheat looks only at the INFERENCE code: the file that serves /simulate plus every local module it imports,
transitively. Training code may legitimately read ERA5 or call imdlib; the server may not.
Flags are 'critical' (cost all anti-cheat points) or 'warn' (cost part, and go to a human).
"""
from __future__ import annotations
import ast, re, zipfile
from pathlib import Path
import numpy as np

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".ipynb_checkpoints", "dist", "build"}
CODE_EXT = {".py", ".ipynb"}
EXTERNAL_WEATHER = [r"open-meteo", r"openweathermap", r"api\.weather", r"weatherapi", r"visualcrossing", r"tomorrow\.io",
                    r"api\.met\.no", r"meteomatics", r"worldweatheronline", r"power\.larc\.nasa", r"cdsapi", r"cds\.climate\.copernicus",
                    r"gcp-public-data-arco-era5", r"weatherbench2", r"mosdac", r"noaa-gfs", r"imdlib", r"nomads\.ncep", r"mausam\.imd"]
WEIGHT_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5", ".keras", ".joblib", ".pkl", ".pickle", ".npz", ".msgpack", ".bin"}
DATA_EXT = {".nc", ".nc4", ".grib", ".grib2", ".zarr"}
DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/]?(?:0[1-9]|1[0-2])[-/]?(?:0[1-9]|[12]\d|3[01])\b")
DATETIME_CALL = re.compile(r"(?:date|datetime|Timestamp)\(\s*['\"]?(?:19|20)\d{2}")
ROUTE_RE = re.compile(r"""[/'"]simulate['"]|/simulate\b""")
HF_RE = re.compile(r"huggingface\.co/[\w.-]+/[\w.-]+")


def _files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and not (set(p.relative_to(root).parts[:-1]) & SKIP_DIRS):
            yield p


def _text(p: Path) -> str:
    try:
        t = p.read_text(errors="ignore")
    except Exception:
        return ""
    if p.suffix == ".ipynb":
        import json
        try:
            return "\n".join("".join(c.get("source", [])) for c in json.loads(t).get("cells", []) if c.get("cell_type") == "code")
        except Exception:
            return t
    return t


def _local_imports(src: str, here: Path, root: Path) -> set[Path]:
    out = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.append(n.module); names += [f"{n.module}.{a.name}" for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level:
            names += [a.name for a in n.names]
    for nm in names:
        parts = nm.split(".")
        for base in (here.parent, root):
            for k in range(len(parts), 0, -1):
                for cand in (base.joinpath(*parts[:k]).with_suffix(".py"), base.joinpath(*parts[:k], "__init__.py")):
                    if cand.is_file() and root in cand.resolve().parents:
                        out.add(cand.resolve())
    return out


def inference_files(root: Path) -> set[Path]:
    """Files that serve /simulate plus every local module they import (transitively)."""
    root = root.resolve()
    seeds = [p.resolve() for p in _files(root) if p.suffix == ".py" and ROUTE_RE.search(_text(p))]
    seen, stack = set(), list(seeds)
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack += list(_local_imports(_text(p), p, root))
    return seen


def _npy_shape(fobj):
    try:
        v = np.lib.format.read_magic(fobj)
        return (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(fobj)[0]
    except Exception:
        return None


def archive_arrays(root: Path) -> list[str]:
    """npy/npz files holding a multi-year daily archive: shape (n > 400, ..., 32, 36) (a one-year climatology is fine)."""
    hits = []
    for p in _files(root):
        shapes = []
        try:
            if p.suffix == ".npy":
                with open(p, "rb") as f: shapes.append(_npy_shape(f))
            elif p.suffix == ".npz":
                with zipfile.ZipFile(p) as z:
                    for n in z.namelist():
                        if n.endswith(".npy"):
                            with z.open(n) as f: shapes.append(_npy_shape(f))
        except Exception:
            continue
        for s in shapes:
            if s and len(s) >= 3 and s[0] > 400 and tuple(s[-2:]) == (32, 36):
                hits.append(f"{p.relative_to(root)} shape {tuple(s)}")
    return hits


def _bad_repo(why: str) -> dict:
    return {"points": 0.0, "anti_cheat_points": 0.0, "code_health_points": 0.0, "flags": [{"level": "critical", "what": why}],
            "health": {}, "inference_files": []}


def scan(root: Path) -> dict:
    """Never raises: an unreadable or missing repo scores 0 with a critical flag instead of stopping the grading run."""
    try:
        root = Path(root).resolve()
        if not root.is_dir():
            return _bad_repo(f"repo path is not a directory: {root.name}")
        return _scan(root)
    except Exception as e:
        return _bad_repo(f"repo scan failed: {type(e).__name__}: {str(e)[:120]}")


def _scan(root: Path) -> dict:
    flags, critical, warn = [], 0, 0
    inf = inference_files(root)
    if not inf:
        flags.append({"level": "critical", "what": "no file serves /simulate"}); critical += 1
    for p in inf:
        t = _text(p); rel = p.relative_to(root)
        for pat in EXTERNAL_WEATHER:
            if re.search(pat, t, re.I):
                flags.append({"level": "critical", "what": f"external weather/data source '{pat}' in inference code", "file": str(rel)}); critical += 1
        dates = DATE_RE.findall(t)
        if len(dates) >= 3 or DATETIME_CALL.search(t):
            flags.append({"level": "critical" if len(dates) >= 5 else "warn", "what": f"hardcoded dates in inference code ({len(dates)} literals)", "file": str(rel)})
            critical += len(dates) >= 5; warn += len(dates) < 5
    for h in archive_arrays(root):
        flags.append({"level": "critical", "what": f"multi-year daily archive in the repo: {h}"}); critical += 1
    for p in _files(root):
        if p.suffix in DATA_EXT or p.name.endswith(".zarr"):
            flags.append({"level": "warn", "what": "climate data file in repo", "file": str(p.relative_to(root))}); warn += 1
    weights = [p for p in _files(root) if p.suffix in WEIGHT_EXT and p.stat().st_size > 1024]
    hf = any(HF_RE.search(_text(p)) for p in _files(root) if p.suffix in (".md", ".py", ".txt", ".ipynb"))
    has_ckpt = bool(weights) or hf
    if not has_ckpt:
        flags.append({"level": "critical", "what": "no model checkpoint in the repo and no Hugging Face link"}); critical += 1

    code = {p: _text(p) for p in _files(root) if p.suffix in CODE_EXT}
    train = [p for p, t in code.items() if re.search(r"\.backward\(|optimizer\.step|\.fit\(|model\.fit|train_step|loss\.backward|lightning|Trainer\(", t)
             and p.resolve() not in inf]
    server_names = re.compile(r"fastapi|flask|uvicorn|starlette|aiohttp|http\.server|bottle", re.I)
    front = any(p.name in ("package.json", "index.html", "streamlit_app.py", "app.js", "dashboard.py") or "streamlit" in t.lower() or "plotly" in t.lower()
                or "leaflet" in t.lower() for p, t in [(q, _text(q)) for q in _files(root) if q.suffix in (".json", ".html", ".py", ".js", ".ts", ".tsx", ".jsx")][:400])
    server_has_ui = any(re.search(r"streamlit|gradio|dash\(|plotly|matplotlib\.pyplot", code.get(p, ""), re.I) for p in inf)
    readme = any(p.name.lower().startswith("readme") for p in root.iterdir())
    health = {"training_pipeline": bool(train), "frontend_present": bool(front), "server_separate_from_ui": bool(inf) and not server_has_ui,
              "readme": readme, "checkpoint": has_ckpt}
    anti = max(0.0, 8.0 - (8.0 if critical else 0.0) - 2.0 * warn)
    code_pts = 3.0 * health["training_pipeline"] + 2.0 * health["server_separate_from_ui"] + 1.0 * health["frontend_present"] + 0.5 * health["readme"] + 0.5 * health["checkpoint"]
    return {"points": round(anti + code_pts, 2), "anti_cheat_points": round(anti, 2), "code_health_points": round(code_pts, 2),
            "flags": flags, "health": health, "inference_files": sorted(str(p.relative_to(root)) for p in inf)}
