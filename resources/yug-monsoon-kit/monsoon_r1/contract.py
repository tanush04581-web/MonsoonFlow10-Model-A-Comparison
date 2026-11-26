"""Round 1 API contract and the physical conventions every team and the grader share.

Grid: 32 x 36 cells of 1 x 1 degree, centres lat 6.5..37.5 (south -> north), lon 65.5..100.5 (west -> east).
Channels (axis 1 of every state array), PHYSICAL units, never normalised on the wire:
  0 t2m  2 m temperature (K)          1 tp   daily total precipitation (mm/day)
  2 u10  10 m eastward wind (m/s)     3 v10  10 m northward wind (m/s)
  4 msl  mean sea level pressure (Pa) 5 tcwv total column water vapour (kg/m2)
Daily fields: t2m/u10/v10/msl/tcwv = mean of the 00, 06, 12, 18 UTC samples; tp = sum over the 24 h of the day.
"""
from __future__ import annotations
import base64
import numpy as np

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
NC, NLAT, NLON = 6, 32, 36
LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0)
T_IN = 3
MAX_STEPS = 14
TIME_LIMIT_S = 5.0
LEAD_DAYS = (1, 4, 8)
SCORE_STEPS = 10

# boxes as (lat_min, lat_max, lon_min, lon_max) in degrees
BOX_BAY_OF_BENGAL = (8.0, 20.0, 82.0, 94.0)
BOX_ARABIAN_SEA = (8.0, 20.0, 65.0, 75.0)


def box_mask(box) -> np.ndarray:
    la0, la1, lo0, lo1 = box
    m = ((LAT >= la0) & (LAT <= la1))[:, None] & ((LON >= lo0) & (LON <= lo1))[None, :]
    return m


def lat_weights() -> np.ndarray:
    w = np.cos(np.deg2rad(LAT)); return w / w.mean()


def encode_list(a: np.ndarray) -> list:
    """THE canonical wire format: plain nested JSON lists of finite floats (10 x 6 x 32 x 36 = 69,120 numbers, about 1 MB of JSON)."""
    return np.asarray(a, dtype="f4").tolist()


def encode_array(a: np.ndarray) -> dict:
    """Compact wire form: base64 little-endian float32 plus shape (JSON lists are also accepted)."""
    a = np.ascontiguousarray(a, dtype="<f4")
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape)}


def decode_array(x, expect_shape=None) -> np.ndarray:
    """Accept {'b64','shape'} or nested lists. Raises ValueError on anything malformed (never trusts the sender)."""
    if isinstance(x, dict):
        if "b64" not in x or "shape" not in x:
            raise ValueError("array dict needs 'b64' and 'shape'")
        shape = tuple(int(s) for s in x["shape"])
        if any(s < 0 or s > 100000 for s in shape) or int(np.prod(shape)) > 20_000_000:
            raise ValueError("array too large")
        raw = base64.b64decode(x["b64"], validate=True)
        if len(raw) != int(np.prod(shape)) * 4:
            raise ValueError("b64 length does not match shape")
        a = np.frombuffer(raw, dtype="<f4").reshape(shape).astype("f4")
    else:
        a = np.asarray(x, dtype="f4")
    if expect_shape is not None and a.shape != tuple(expect_shape):
        raise ValueError(f"shape {a.shape} != expected {tuple(expect_shape)}")
    return a


def doy_sincos(doy: int) -> list[float]:
    return [float(np.sin(2 * np.pi * doy / 365.0)), float(np.cos(2 * np.pi * doy / 365.0))]
