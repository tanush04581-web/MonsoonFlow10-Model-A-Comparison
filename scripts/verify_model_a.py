"""Standalone self-check for MonsoonFlow-10 Model A (checkpoints/model.pt).

Verifies:
1. Artifact checksum (SHA-256) and weights_only=True loading on CPU.
2. End-to-end physical-unit inference (3, 6, 32, 36) -> (10, 6, 32, 36).
3. Physical bounds (finite values, tp >= 0, tcwv >= 0, t2m in [233.15, 333.15] K).
4. Exact 0.0 zero-shock parity (perturbation=None vs all-zero perturbation).
5. Non-zero localized Day-1 warm-ocean Bay of Bengal shock response.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KIT = ROOT / "resources" / "yug-monsoon-kit"
if KIT.is_dir() and str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from models.backbones import load_model_from_checkpoint
from monsoon_r1 import contract as C, scoring as S

EXPECTED_SHA256 = "4f02add2ce5f4aef3b170cd1f1d2352412ceb149daa470bf2c9e303cba30b25b"


def main() -> int:
    ckpt_path = ROOT / "checkpoints" / "model.pt"
    raw = ckpt_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    print(f"Checkpoint: {ckpt_path} ({len(raw)} bytes, sha256={sha})")
    assert sha == EXPECTED_SHA256, f"SHA-256 mismatch: {sha} != {EXPECTED_SHA256}"

    torch.set_num_threads(4)
    model, ckpt = load_model_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    print(f"Loaded architecture={ckpt.get('architecture')}, epoch={ckpt.get('epoch')}, params={sum(p.numel() for p in model.parameters())}")

    stats = json.loads((ROOT / "checkpoints" / "stats.json").read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
    init_np = np.broadcast_to(mean, (3, 6, 32, 36)).copy()
    init_np[:, 1] = np.maximum(init_np[:, 1], 0.0)
    x = torch.from_numpy(init_np)
    doy = torch.tensor(C.doy_sincos(196), dtype=torch.float32)

    with torch.inference_mode():
        _ = model(x, doy, steps=10, perturbation=None)  # warm-up
        t0 = time.perf_counter()
        out_base = model(x, doy, steps=10, perturbation=None).numpy()
        ms = (time.perf_counter() - t0) * 1000.0
        out_zero = model(x, doy, steps=10, perturbation=torch.zeros((6, 32, 36), dtype=torch.float32)).numpy()

        std = np.asarray(stats["std"], dtype=np.float32)
        p_warm = S.make_shock("warm_ocean", init_np[-1], std)
        out_warm = model(x, doy, steps=10, perturbation=torch.from_numpy(p_warm)).numpy()

    assert out_base.shape == (10, 6, 32, 36), f"Unexpected shape: {out_base.shape}"
    assert S.validity(out_base, 10) == [], f"Validity failure: {S.validity(out_base, 10)}"
    assert out_base[:, 5].min() >= 0.0, f"Negative tcwv: {out_base[:, 5].min()}"
    zero_diff = float(np.max(np.abs(out_base - out_zero)))
    assert zero_diff == 0.0, f"Non-zero null diff: {zero_diff}"

    cc = S.causality_checks(out_base, out_warm, C.BOX_BAY_OF_BENGAL, std)
    assert cc["nonzero"] and cc["localized"] and cc["no_teleport"] and cc["stable"], cc
    print(f"Inference 10-step CPU latency: {ms:.2f} ms | validity=PASS | zero_shock_diff={zero_diff:.1f}")
    print(f"Warm ocean check: near_frac_day1={cc['near_frac_day1']:.4f}, far_frac_max={cc['far_frac_max']:.5f}, propagates={cc['propagates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
