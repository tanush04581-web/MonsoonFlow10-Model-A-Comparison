"""MonsoonFlow-10 neural architectures, physics losses, and transforms.

Strict import boundary (for offline serving anti-cheat compliance):
This package imports ONLY ``torch``, ``math``, ``pathlib``, and ``typing``.
It must never import ``data``, ``training``, ``datetime``, or external weather APIs,
and must contain no hardcoded calendar date literals.
"""
from . import transforms
from .backbones import (
    ARCHITECTURES,
    BaseMonsoonForecaster,
    MF10ConvLSTM,
    MultiScaleResUNet,
    SpatioTemporalResCNN,
    build_model,
    load_model_from_checkpoint,
)
from .physics_losses import MonsoonPhysicsLoss

__all__ = [
    "transforms",
    "ARCHITECTURES",
    "BaseMonsoonForecaster",
    "MF10ConvLSTM",
    "SpatioTemporalResCNN",
    "MultiScaleResUNet",
    "build_model",
    "load_model_from_checkpoint",
    "MonsoonPhysicsLoss",
]
