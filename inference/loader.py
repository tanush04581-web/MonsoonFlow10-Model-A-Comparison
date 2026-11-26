"""Model artifact loader for offline CPU serving."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import torch

from models.backbones import BaseMonsoonForecaster, load_model_from_checkpoint

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
DEFAULT_MODEL_PATH = CHECKPOINTS_DIR / "model.pt"
CPU_THREADS = 4


def configure_cpu_threads(num_threads: int = CPU_THREADS) -> int:
    """Configure deterministic, resource-bounded CPU thread pools."""
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch raises RuntimeError if interop threads are already initialized
        pass
    return int(torch.get_num_threads())


class ArtifactRegistry:
    """Singleton container preloading trained model weights on CPU at boot."""

    def __init__(self) -> None:
        self.model: Optional[BaseMonsoonForecaster] = None
        self.checkpoint_meta: dict[str, Any] = {}
        self.stats: Optional[dict[str, Any]] = None
        self.model_name: str = "unavailable"
        self.num_threads: int = CPU_THREADS
        self.load_error: Optional[str] = "Model not loaded yet"
        self.start_time: float = time.monotonic()
        self._is_ready: bool = False

    def reset(self) -> None:
        self.model = None
        self.checkpoint_meta = {}
        self.stats = None
        self.model_name = "unavailable"
        self.load_error = "Model not loaded yet"
        self._is_ready = False

    def load_artifacts(self, checkpoint_path: Optional[Path] = None) -> bool:
        """Load trained checkpoint safely on CPU during server lifespan startup."""
        self.reset()
        self.start_time = time.monotonic()
        self.num_threads = configure_cpu_threads(CPU_THREADS)

        model_path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_MODEL_PATH
        if not model_path.is_file():
            self.load_error = f"Checkpoint file not found: {model_path.name}"
            return False

        stats_path = CHECKPOINTS_DIR / "stats.json"
        if stats_path.is_file():
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    self.stats = json.load(f)
            except Exception:
                self.stats = None

        try:
            model, ckpt = load_model_from_checkpoint(model_path, map_location="cpu")
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

            cfg = ckpt.get("model_config", {})
            arch = str(ckpt.get("architecture") or cfg.get("arch") or "ms_resunet")
            use_anchor = bool(cfg.get("use_seasonal_anchor", False))
            run_name = str(ckpt.get("run_config", {}).get("run_name") or "")
            epoch = ckpt.get("epoch")

            # Warm up CPU kernels with a 1-step realistic mean-state pass
            mean_state = model.stats_mean.detach().reshape(1, 6, 1, 1).expand(3, 6, 32, 36).clone()
            with torch.inference_mode():
                _ = model(mean_state, [0.0, 1.0], steps=1, perturbation=None)

            self.model = model
            self.stats = ckpt.get("stats", self.stats)
            self.checkpoint_meta = {
                "checkpoint_file": model_path.name,
                "architecture": arch,
                "use_seasonal_anchor": use_anchor,
                "run_name": run_name,
                "epoch": int(epoch) if isinstance(epoch, int) else None,
                "tp_transform": ckpt.get("tp_transform", "linear"),
            }
            variant = f"{arch}-anchor" if use_anchor else arch
            self.model_name = f"MonsoonFlow-10 ({variant})"
            self.load_error = None
            self._is_ready = True
            return True
        except Exception as exc:
            self.model = None
            self.model_name = "unavailable"
            self.load_error = f"Failed to load {model_path.name}: {type(exc).__name__}: {exc}"
            self._is_ready = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self.model is not None


registry = ArtifactRegistry()
