"""Pydantic schemas and strict array decoding for MonsoonFlow-10 serving."""
from __future__ import annotations

import base64
import math
from typing import Any, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Constants matching competition contract (monsoon_r1.contract)
NUM_CHANNELS = 6
NUM_LATS = 32
NUM_LONS = 36
INPUT_DAYS = 3
DEFAULT_STEPS = 10
MIN_STEPS = 1
MAX_STEPS = 14

INIT_STATE_SHAPE = (INPUT_DAYS, NUM_CHANNELS, NUM_LATS, NUM_LONS)
PERTURBATION_SHAPE = (NUM_CHANNELS, NUM_LATS, NUM_LONS)


def decode_array(
    x: Any,
    expect_shape: Optional[tuple[int, ...]] = None,
    *,
    require_finite: bool = True,
) -> np.ndarray:
    """Decode nested numeric lists, ndarray, or {'b64': str, 'shape': list[int]} into float32 ndarray."""
    if isinstance(x, dict):
        if "b64" not in x or "shape" not in x:
            raise ValueError("array dict requires 'b64' and 'shape'")
        b64_val = x["b64"]
        shape_raw = x["shape"]
        if not isinstance(b64_val, str) or not isinstance(shape_raw, (list, tuple)):
            raise ValueError("array dict 'b64' must be str and 'shape' must be list of ints")
        shape_list: list[int] = []
        for s in shape_raw:
            if isinstance(s, bool) or not isinstance(s, int):
                raise ValueError("shape dimensions must be integers")
            shape_list.append(s)
        shape = tuple(shape_list)
        if any(s < 0 or s > 100000 for s in shape) or int(np.prod(shape)) > 20_000_000:
            raise ValueError("array dimensions exceed safety threshold")
        if expect_shape is not None and shape != tuple(expect_shape):
            raise ValueError(f"Shape {shape} does not match expected {tuple(expect_shape)}")
        raw = base64.b64decode(b64_val, validate=True)
        if len(raw) != int(np.prod(shape)) * 4:
            raise ValueError("base64 byte buffer length does not match specified shape")
        arr = np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32, copy=True)
    elif isinstance(x, np.ndarray):
        if x.dtype == np.bool_ or not np.issubdtype(x.dtype, np.number):
            raise ValueError(f"Array dtype {x.dtype} is not numeric")
        arr = np.asarray(x, dtype=np.float32)
    elif isinstance(x, (list, tuple)):
        try:
            arr = np.asarray(x, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid nested numeric array: {exc}") from exc
    else:
        raise ValueError(f"Expected nested list or b64 dict, got {type(x).__name__}")

    if expect_shape is not None and arr.shape != tuple(expect_shape):
        raise ValueError(f"Shape {arr.shape} does not match expected {tuple(expect_shape)}")
    if require_finite and not bool(np.isfinite(arr).all()):
        raise ValueError("Array contains non-finite values (NaN or Inf)")
    return arr


class HealthResponse(BaseModel):
    status: str = "ready"
    model_name: str = "MonsoonFlow-10"
    device: str = "cpu"
    threads: int = 4
    uptime_seconds: float = 0.0
    error: Optional[str] = None


class SimulateRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    init_states: Any = Field(
        ...,
        description="Nested lists or b64 dict for (3, 6, 32, 36) physical initial conditions",
    )
    doy_sincos: List[float] = Field(
        ...,
        description="[sin, cos] representing day of year cyclic encoding for last input day",
    )
    steps: int = Field(
        default=DEFAULT_STEPS,
        ge=MIN_STEPS,
        le=MAX_STEPS,
        description="Number of autoregressive forecast steps (1..14)",
    )
    perturbation: Optional[Any] = Field(
        default=None,
        description="Optional physical shock delta of shape (6, 32, 36)",
    )

    @field_validator("init_states", mode="before")
    @classmethod
    def _validate_init_states(cls, v: Any) -> np.ndarray:
        return decode_array(v, expect_shape=INIT_STATE_SHAPE, require_finite=True)

    @field_validator("doy_sincos", mode="before")
    @classmethod
    def _validate_doy_sincos(cls, v: Any) -> List[float]:
        if isinstance(v, np.ndarray):
            if v.ndim != 1 or v.shape[0] != 2 or v.dtype == np.bool_ or not np.issubdtype(v.dtype, np.number):
                raise ValueError("doy_sincos must be a 2-element numeric list [sin, cos]")
            vals = [float(v[0]), float(v[1])]
        elif isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"doy_sincos must have length 2, got {len(v)}")
            vals = []
            for item in v:
                if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
                    raise ValueError("doy_sincos elements must be finite numbers")
                vals.append(float(item))
        else:
            raise ValueError("doy_sincos must be a 2-element list [sin, cos]")

        if not all(math.isfinite(x) for x in vals):
            raise ValueError("doy_sincos contains non-finite values")
        return vals

    @field_validator("steps", mode="before")
    @classmethod
    def _validate_steps(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError("steps must be an integer")
        iv = int(v)
        if not (MIN_STEPS <= iv <= MAX_STEPS):
            raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}, got {iv}")
        return iv

    @field_validator("perturbation", mode="before")
    @classmethod
    def _validate_perturbation(cls, v: Any) -> Optional[np.ndarray]:
        if v is None:
            return None
        return decode_array(v, expect_shape=PERTURBATION_SHAPE, require_finite=True)


class SimulateResponse(BaseModel):
    rollout: Any = Field(..., description="Forecast tensor of shape (steps, 6, 32, 36) in physical units")
    inference_ms: float = Field(..., description="Execution latency in milliseconds")
