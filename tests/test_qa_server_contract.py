"""Agent-5 black-box contract gates for the root serve.py (started here on port 8021, pinned to 4 cores).

All judgements use the pinned kit (monsoon_r1.scoring.validity / null_consistent / causality_checks / make_shock and
monsoon_r1.client.call_simulate). Inputs are real 2019-2020 validation windows. Tests that FAIL are real defects:
they are not xfail'ed on purpose so the suite stays a gate for the backend.
Run:  .\\.venv\\Scripts\\python.exe -m pytest tests\\test_qa_server_contract.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from qa_common import (C, S, ROUTE_SIM, ServerProc, doy_sincos_for, load_split, load_std, persistence,  # noqa: E402
                       simulate, to_date, ymd)

PORT = 8021
WD = ymd(2019, 7, 15)


@pytest.fixture(scope="module")
def srv():
    with ServerProc(PORT, ncores=4) as s:
        yield s


@pytest.fixture(scope="module")
def window():
    data, dates, _ = load_split("val")
    i = int(np.where(dates == WD)[0][0])
    return data[i - 2:i + 1].copy(), doy_sincos_for(to_date(WD))


def post_raw(url, body: str):
    return requests.post(url + ROUTE_SIM, data=body, headers={"content-type": "application/json"}, timeout=15)


def payload(init, doy, pert=None, steps=10):
    return {"init_states": np.asarray(init, "f4").tolist(), "doy_sincos": doy, "steps": steps,
            "perturbation": None if pert is None else np.asarray(pert, "f4").tolist()}


# ---- happy path ----------------------------------------------------------------------------------------------
def test_health_ready(srv):
    r = requests.post(srv.url + "/health", timeout=5)
    assert r.status_code == 200 and r.json().get("status") == "ready" and isinstance(r.json().get("model_name"), str)


def test_simulate_shape_finite_valid_and_fast(srv, window):
    init, doy = window
    r = simulate(srv.url, init, doy)
    assert r.ok, r.error
    assert r.out.shape == (10, 6, 32, 36)
    assert S.validity(r.out, 10) == []
    assert r.wall_s < C.TIME_LIMIT_S


def test_null_zeros_and_omitted_are_identical(srv, window):
    init, doy = window; std = load_std()
    a = simulate(srv.url, init, doy, None).out
    b = simulate(srv.url, init, doy, np.zeros((6, 32, 36), "f4")).out
    c = simulate(srv.url, init, doy, omit_perturbation=True).out
    assert S.null_consistent(a, b, std) and S.null_consistent(a, c, std)
    assert np.abs(a - b).max() <= 1e-6


def test_deterministic(srv, window):
    init, doy = window
    assert np.array_equal(simulate(srv.url, init, doy).out, simulate(srv.url, init, doy).out)


@pytest.mark.parametrize("name,box", [("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA)])
def test_official_shock_nonzero_localized_stable(srv, window, name, box):
    init, doy = window; std = load_std()
    base = simulate(srv.url, init, doy).out
    shk = simulate(srv.url, init, doy, S.make_shock(name, init[-1], std)).out
    cc = S.causality_checks(base, shk, box, std)
    assert cc["nonzero"] and cc["localized"] and cc["stable"] and cc["no_teleport"], cc


@pytest.mark.parametrize("name,box", [("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA)])
def test_official_shock_propagates(srv, window, name, box):
    """Fails for any persistence server by construction (the response can never leave the box)."""
    init, doy = window; std = load_std()
    base = simulate(srv.url, init, doy).out
    shk = simulate(srv.url, init, doy, S.make_shock(name, init[-1], std)).out
    assert S.causality_checks(base, shk, box, std)["propagates"]


def test_served_model_is_not_persistence(srv, window):
    """/health advertises a model; the rollout must not be the last input repeated (cannot beat day-4 persistence)."""
    init, doy = window
    out = simulate(srv.url, init, doy).out
    assert not np.allclose(out, persistence(init), atol=1e-6), f"served '{requests.post(srv.url + '/health').json()}' returns persistence"


# ---- physical clamp edge -------------------------------------------------------------------------------------
@pytest.mark.parametrize("t2m", [220.0, 233.15, 345.0])
def test_t2m_clamp_passes_kit_validity(srv, window, t2m):
    init, doy = window; init = init.copy(); init[:, 0] = t2m
    r = simulate(srv.url, init, doy)
    assert r.ok, r.error
    assert S.validity(r.out, 10) == [], f"t2m min {r.out[:, 0].min()!r} max {r.out[:, 0].max()!r}"


def test_negative_rain_input_never_yields_negative_rain(srv, window):
    init, doy = window; init = init.copy(); init[:, 1] = -5.0
    r = simulate(srv.url, init, doy)
    assert r.ok and r.out[:, 1].min() >= 0.0


# ---- robustness: invalid input must be a 4xx, never a 500, never silently accepted -------------------------
@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_nonfinite_init_states_rejected_4xx(srv, window, bad):
    init, doy = window; init = init.copy(); init[2, 0, 5, 5] = float(bad)
    r = post_raw(srv.url, json.dumps(payload(init, doy), allow_nan=True))
    assert 400 <= r.status_code < 500, r.status_code


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_nonfinite_perturbation_rejected_4xx(srv, window, bad):
    init, doy = window; p = np.zeros((6, 32, 36), "f4"); p[0, 10, 20] = float(bad)
    r = post_raw(srv.url, json.dumps(payload(init, doy, p), allow_nan=True))
    assert 400 <= r.status_code < 500, f"HTTP {r.status_code} (perturbation {bad} accepted)"


@pytest.mark.parametrize("case", ["missing_init", "wrong_shape", "string_init", "not_json", "bad_pert_shape", "doy_len_1"])
def test_malformed_requests_rejected_4xx(srv, window, case):
    init, doy = window
    body = payload(init, doy)
    if case == "missing_init":
        body.pop("init_states"); raw = json.dumps(body)
    elif case == "wrong_shape":
        body["init_states"] = init[:2].tolist(); raw = json.dumps(body)
    elif case == "string_init":
        body["init_states"] = "hello"; raw = json.dumps(body)
    elif case == "not_json":
        raw = "{not json"
    elif case == "bad_pert_shape":
        body["perturbation"] = np.zeros((6, 32, 35)).tolist(); raw = json.dumps(body)
    else:
        body["doy_sincos"] = [0.5]; raw = json.dumps(body)
    r = post_raw(srv.url, raw)
    assert 400 <= r.status_code < 500, f"{case}: HTTP {r.status_code}"


@pytest.mark.parametrize("steps", [1, C.MAX_STEPS])
def test_in_range_steps_accepted(srv, window, steps):
    init, doy = window
    r = post_raw(srv.url, json.dumps(payload(init, doy, steps=steps)))
    assert r.status_code == 200 and np.asarray(r.json()["rollout"]).shape == (steps, 6, 32, 36)


# kept LAST: an unbounded server spends ~80 s and ~600 MB of JSON on steps=5000, blocking its event loop
@pytest.mark.parametrize("steps", [0, -1, C.MAX_STEPS + 1, 5000])
def test_out_of_range_steps_rejected_4xx(srv, window, steps):
    init, doy = window
    try:
        r = requests.post(srv.url + ROUTE_SIM, data=json.dumps(payload(init, doy, steps=steps)),
                          headers={"content-type": "application/json"}, timeout=30)
    except requests.exceptions.ReadTimeout:
        pytest.fail(f"steps={steps}: no answer within 30 s (unbounded work)")
    assert 400 <= r.status_code < 500, f"steps={steps}: HTTP {r.status_code}"
