"""Root inference server for MonsoonFlow-10.

Listens on 0.0.0.0:8000 (configurable via --port).
Provides /health and /simulate endpoints under strict offline CPU constraints.
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import math
import time
from typing import Any

import anyio
import uvicorn
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.loader import registry
from inference.predictor import ModelNotReadyError, get_model_name, simulate
from inference.schemas import SimulateRequest


def _sanitize_validation_errors(errors: Any) -> list[dict[str, Any]]:
    """Build JSON-safe validation details without echoing NaN/Inf or huge arrays."""
    clean: list[dict[str, Any]] = []
    for err in errors:
        item = {
            "loc": list(err.get("loc", ())),
            "msg": str(err.get("msg", "Invalid input")),
            "type": str(err.get("type", "value_error")),
        }
        ctx = err.get("ctx")
        if isinstance(ctx, dict):
            safe_ctx = {}
            for k, v in ctx.items():
                if isinstance(v, float) and not math.isfinite(v):
                    safe_ctx[k] = str(v)
                elif isinstance(v, (str, int, float, bool)):
                    safe_ctx[k] = v
            if safe_ctx:
                item["ctx"] = safe_ctx
        clean.append(item)
    return clean


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload trained weights on CPU at server startup."""
    registry.load_artifacts()
    yield


app = FastAPI(
    title="MonsoonFlow-10 Inference Service",
    description="Offline spatiotemporal climate digital twin inference server",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local frontend communication (e.g. Vite on localhost / 127.0.0.1)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": _sanitize_validation_errors(exc.errors())},
    )


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(_, exc: ModelNotReadyError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


@app.get("/health", status_code=status.HTTP_200_OK)
@app.post("/health", status_code=status.HTTP_200_OK)
async def health_endpoint():
    """Health check endpoint required by competition grader and frontend."""
    uptime = max(0.0, time.monotonic() - registry.start_time)
    payload: dict[str, Any] = {
        "status": "ready" if registry.is_ready else "unavailable",
        "model_name": get_model_name(),
        "device": "cpu",
        "threads": registry.num_threads,
        "uptime_seconds": round(uptime, 3),
    }
    if registry.checkpoint_meta:
        payload.update(registry.checkpoint_meta)
    if not registry.is_ready:
        payload["error"] = registry.load_error or "Model checkpoint is not loaded"
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
    return payload


@app.post("/simulate", status_code=status.HTTP_200_OK)
async def simulate_endpoint(req: SimulateRequest):
    """Execute physical-unit spatiotemporal weather rollout off the event loop."""
    if not registry.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": registry.load_error or "Model checkpoint is not loaded"},
        )

    out, inference_ms = await anyio.to_thread.run_sync(
        simulate,
        req.init_states,
        req.doy_sincos,
        req.steps,
        req.perturbation,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "rollout": out.tolist(),
            "inference_ms": float(inference_ms),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="MonsoonFlow-10 Root Serving Entrypoint")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
