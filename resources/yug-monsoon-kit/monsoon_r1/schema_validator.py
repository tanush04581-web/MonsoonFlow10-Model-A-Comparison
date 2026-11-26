"""Teams run this against their own server before submitting:  python -m monsoon_r1.schema_validator http://localhost:8000
Checks /health, /simulate with random-but-plausible inputs in BOTH wire formats (base64 and nested lists), a perturbation,
the null perturbation, output shape/values, and the 5 s limit. It does not know the hidden cases and has no scoring."""
from __future__ import annotations
import sys, time
import numpy as np
from . import contract as C, scoring as S
from .client import call_simulate, wait_ready


def _state(seed=0):
    r = np.random.default_rng(seed)
    base = np.array([295.0, 4.0, 2.0, 1.0, 100500.0, 35.0], "f4")[:, None, None]
    return (base + r.normal(0, 1, (C.T_IN, C.NC, C.NLAT, C.NLON)) * np.array([3, 3, 2, 2, 300, 8], "f4")[None, :, None, None]).astype("f4")


def main(argv=None) -> int:
    url = (argv or sys.argv[1:])[0]; ok_all = True
    ok, why = wait_ready(url, 120, allow_private=True); print("health:", "OK" if ok else f"FAIL ({why})"); ok_all &= ok
    st = _state(); st[:, 1] = np.abs(st[:, 1])
    for label, enc in (("JSON lists (canonical)", C.encode_list), ("base64 (also tolerated)", C.encode_array)):
        for pert_name, pert in (("no perturbation", None), ("warm_ocean", S.make_shock("warm_ocean", st[-1], S.STD_DEFAULT)), ("zeros", np.zeros((C.NC, C.NLAT, C.NLON), "f4"))):
            payload = {"init_states": enc(st), "doy_sincos": C.doy_sincos(160), "steps": C.SCORE_STEPS, "perturbation": None if pert is None else enc(pert)}
            r = call_simulate(url, payload, allow_private=True)
            bad = S.validity(r.out, C.SCORE_STEPS) if r.ok else []
            good = r.ok and not bad
            print(f"  [{label:12s}] {pert_name:16s} {'OK ' if good else 'FAIL'} {r.wall_s:5.2f}s {r.error or ''} {bad or ''}"); ok_all &= good
    print("RESULT:", "all checks passed" if ok_all else "FIX THE FAILURES ABOVE"); return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
