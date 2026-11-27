"""PyTorch Dataset with sliding-window autoregressive rollout formulation for MonsoonFlow-10.

Input per step (10 channels):
- 6 dynamic weather channels (normalized: (X - mean) / std)
- 2 static environmental channels (land_sea_mask, orography_m normalized)
- 2 day-of-year cyclic planes [sin(2*pi*doy/365), cos(2*pi*doy/365)]

Temporal structure:
- Input horizon: T_in = 3 consecutive days (t_-2, t_-1, t_0)
- Target horizon: T_rollout = K days (t_1 ... t_K), K in {1, 3, 5, 10}
"""
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

NLAT, NLON = 32, 36


def get_doy_sincos(date_int: int) -> Tuple[float, float]:
    """Calculate [sin, cos] for a given date YYYYMMDD."""
    year = int(date_int) // 10000
    month = (int(date_int) // 100) % 100
    day = int(date_int) % 100
    d = dt.date(year, month, day)
    doy = d.timetuple().tm_yday
    rad = 2.0 * np.pi * doy / 365.0
    return float(np.sin(rad)), float(np.cos(rad))


class MonsoonDataset(Dataset):
    """Sliding-window spatiotemporal climate dataset."""
    
    def __init__(
        self,
        data: np.ndarray,
        dates: np.ndarray,
        stats: dict,
        static_fields: np.ndarray,
        t_in: int = 3,
        t_rollout: int = 1,
        step_stride: int = 1,
    ):
        """
        Args:
            data: Raw physical data array of shape (N, 6, 32, 36)
            dates: Integer array of dates YYYYMMDD of shape (N,)
            stats: Dict with 'mean' and 'std' for the 6 dynamic channels
            static_fields: Terrain array (2, 32, 36) containing [lsm, orography]
            t_in: Consecutive input days (default: 3)
            t_rollout: Rollout forecast steps (default: 1, up to 10)
            step_stride: Step stride between consecutive sliding windows (default: 1)
        """
        super().__init__()
        self.t_in = t_in
        self.t_rollout = t_rollout
        self.dates = dates
        
        # 1. Normalize dynamic weather fields
        mean = np.array(stats["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
        std = np.array(stats["std"], dtype=np.float32).reshape(1, 6, 1, 1)
        self.norm_data = (data.astype(np.float32) - mean) / np.maximum(std, 1e-6)
        
        # 2. Normalize static fields: lsm in [0, 1], orography normalized by max elevation ~8000m
        lsm = static_fields[0:1].astype(np.float32)
        oro = static_fields[1:2].astype(np.float32)
        oro_norm = oro / 8000.0  # Safe elevation scaling
        self.static = np.concatenate([lsm, oro_norm], axis=0) # (2, 32, 36)
        
        # 3. Precompute day-of-year sin/cos planes for each day in dates
        N = len(dates)
        self.doy_planes = np.zeros((N, 2, NLAT, NLON), dtype=np.float32)
        for i, d in enumerate(dates):
            s, c = get_doy_sincos(d)
            self.doy_planes[i, 0] = s
            self.doy_planes[i, 1] = c
            
        # 4. Valid sliding-window starting indices
        # Window requires (t_in + t_rollout) consecutive days
        window_len = t_in + t_rollout
        self.valid_indices = list(range(0, N - window_len + 1, step_stride))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs: Tensor of shape (t_in, 10, 32, 36)
            targets: Tensor of shape (t_rollout, 6, 32, 36)
            t0_doy: Tensor of shape (2,) representing [sin, cos] of day t_0
        """
        start_idx = self.valid_indices[idx]
        in_end = start_idx + self.t_in
        target_end = in_end + self.t_rollout
        
        # Input 6 dynamic channels
        in_dynamic = self.norm_data[start_idx:in_end] # (t_in, 6, 32, 36)
        
        # Static fields tiled over t_in
        t_static = np.tile(self.static[None, ...], (self.t_in, 1, 1, 1)) # (t_in, 2, 32, 36)
        
        # DOY sin/cos planes over t_in
        in_doy = self.doy_planes[start_idx:in_end] # (t_in, 2, 32, 36)
        
        # Concatenate along channel dimension -> (t_in, 10, 32, 36)
        in_tensor = np.concatenate([in_dynamic, t_static, in_doy], axis=1)
        
        # Target dynamic normalized states
        target_tensor = self.norm_data[in_end:target_end] # (t_rollout, 6, 32, 36)
        
        # DOY sin/cos for the LAST input day (t_0)
        t0_doy = self.doy_planes[in_end - 1, :, 0, 0] # (2,)
        
        return (
            torch.from_numpy(in_tensor),
            torch.from_numpy(target_tensor),
            torch.from_numpy(t0_doy),
        )
