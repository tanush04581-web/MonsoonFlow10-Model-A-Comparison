"""Starter server for Round 1. This is API plumbing only: `simulate()` below is a placeholder (it repeats the last day).
Replace it with YOUR model. The judge runs this exact file in an offline container:  python serve.py --port 8000

Rules: listen on 0.0.0.0 (not 127.0.0.1), load weights from files next to this one (there is no internet), keep
the model loaded in memory (a 10-step rollout must return in under 5 s), return finite numbers in physical units.
"""
import argparse
import numpy as np
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()
MODEL_NAME = "placeholder-persistence"
# TODO load your weights here, once, at import time, e.g. model = torch.load(Path(__file__).parent / "weights.pt")


def _arr(x):
    """Accept nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated)."""
    if isinstance(x, dict):
        import base64
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical units, oldest first. perturbation [6,32,36] delta or None.
    Return [steps,6,32,36] physical units. doy_sincos = [sin, cos] of the day of year of init_states[-1]."""
    last = init_states[-1].copy()
    if perturbation is not None:
        last = last + perturbation                     # the shock is added to the last input day before simulating
    return np.repeat(last[None], steps, axis=0)        # PLACEHOLDER: replace with your model


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    import time
    t0 = time.perf_counter(); body = await request.json()
    init = _arr(body["init_states"]); pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
    out = simulate(init, body["doy_sincos"], int(body["steps"]), pert)
    return {"rollout": np.asarray(out, dtype="f4").tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
