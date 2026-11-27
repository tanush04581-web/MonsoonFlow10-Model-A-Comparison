"""MonsoonFlow-10 (MF10) Data Engineering Module.
Strictly offline, chronological splitting and feature engineering.
"""
from .preprocessing import load_raw_dataset, chronological_split
from .static_fields import extract_static_fields
from .statistics import compute_training_stats, compute_climatology

try:
    from .dataset import MonsoonDataset
except ImportError:
    MonsoonDataset = None

__all__ = [
    "load_raw_dataset",
    "chronological_split",
    "extract_static_fields",
    "compute_training_stats",
    "compute_climatology",
    "MonsoonDataset",
]
