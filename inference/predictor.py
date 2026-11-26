"""Neural autoregressive prediction engine with physical units and zero-shock parity."""
from __future__ import annotations

import threading
import time
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .loader import registry
from .schemas import (
    DEFAULT_STEPS,
    INIT_STATE_SHAPE,
    MAX_STEPS,
    MIN_STEPS,
    NUM_CHANNELS,
    NUM_LATS,
    NUM_LONS,
    PERTURBATION_SHAPE,
    decode_array,
)

# Serialize multi-threaded (4-core) CPU forward passes to avoid intra-op thread contention
_INFER_LOCK = threading.Lock()


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested while no valid model checkpoint is loaded."""


def is_ready() -> bool:
    return registry.is_ready


def get_model_name() -> str:
    return registry.model_name


def simulate(
    init_states_raw: Any,
    doy_sincos: Sequence[float],
    steps: int = DEFAULT_STEPS,
    perturbation_raw: Optional[Any] = None,
) -> tuple[np.ndarray, float]:
    """Execute physical-unit autoregressive rollout using the loaded MonsoonFlow-10 model.

    All inputs and outputs remain strictly in physical units:
      t2m [K], tp [mm/day], u10 [m/s], v10 [m/s], msl [Pa], tcwv [kg/m^2].
    Normalization, static terrain fusion, DOY rotation, and physical clamping
    are executed inside ``registry.model.forward(...)``.
    """
    model = registry.model
    if not registry.is_ready or model is None:
        raise ModelNotReadyError(registry.load_error or "Model checkpoint is not loaded")

    t0 = time.perf_counter()

    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise ValueError("steps must be an integer")
    steps_int = int(steps)
    if not (MIN_STEPS <= steps_int <= MAX_STEPS):
        raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}, got {steps_int}")

    init_states = decode_array(init_states_raw, expect_shape=INIT_STATE_SHAPE, require_finite=True)

    if len(doy_sincos) != 2:
        raise ValueError(f"doy_sincos must have length 2, got {len(doy_sincos)}")
    doy_arr = np.asarray([float(doy_sincos[0]), float(doy_sincos[1])], dtype=np.float32)
    if not bool(np.isfinite(doy_arr).all()):
        raise ValueError("doy_sincos contains non-finite values")

    pert_tensor: Optional[torch.Tensor] = None
    if perturbation_raw is not None:
        pert = decode_array(perturbation_raw, expect_shape=PERTURBATION_SHAPE, require_finite=True)
        # Preserve exact bit-for-bit equality between None and all-zero perturbation
        if bool(np.any(pert != 0.0)):
            pert_tensor = torch.from_numpy(np.ascontiguousarray(pert, dtype=np.float32))

    x_tensor = torch.from_numpy(np.ascontiguousarray(init_states, dtype=np.float32))
    doy_tensor = torch.from_numpy(doy_arr)

    with _INFER_LOCK:
        with torch.inference_mode():
            out_tensor = model(
                x_tensor,
                doy_tensor,
                steps=steps_int,
                perturbation=pert_tensor,
            )

    out = np.ascontiguousarray(out_tensor.detach().cpu().numpy(), dtype=np.float32)
    expected_out_shape = (steps_int, NUM_CHANNELS, NUM_LATS, NUM_LONS)
    if out.shape != expected_out_shape:
        raise RuntimeError(f"Unexpected rollout shape {out.shape}; expected {expected_out_shape}")
    if not bool(np.isfinite(out).all()):
        raise RuntimeError("Model produced non-finite rollout values")

    inference_ms = (time.perf_counter() - t0) * 1000.0
    return out, inference_ms
