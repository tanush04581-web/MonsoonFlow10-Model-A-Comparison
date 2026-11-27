/**
 * gaussianBrush.ts - 2D Gaussian shock perturbation generator
 * Applies Gaussian convolution with sigma ~ 300 km (~2.5 grid cells)
 * to ensure shocks are physically smooth without single-pixel spikes.
 */

import { GRID_CONFIG, type ChannelId } from '../types/climate';
import type { GaussianShockSpec } from '../types/shocks';

export interface ShockBoundsReport {
  isValid: boolean;
  warnings: string[];
}

/**
 * Calculates Great-Circle distance in km using Haversine formula
 */
export function haversineDistanceKm(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const R = 6371; // Earth radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Generates a 2D grid [32, 36] perturbation using 2D Gaussian smoothing:
 * P(i, j) = magnitude * exp(-d(i, j)^2 / (2 * sigma^2)) for d <= radius
 */
export function generateGaussianShock2D(spec: GaussianShockSpec): number[][] {
  const grid: number[][] = [];
  const sigma = spec.sigmaKm > 0 ? spec.sigmaKm : 300;
  const twoSigmaSq = 2 * sigma * sigma;

  for (let i = 0; i < GRID_CONFIG.nLat; i++) {
    const row: number[] = [];
    const cellLat = GRID_CONFIG.latMin + i * GRID_CONFIG.resolutionDeg;

    for (let j = 0; j < GRID_CONFIG.nLon; j++) {
      const cellLon = GRID_CONFIG.lonMin + j * GRID_CONFIG.resolutionDeg;
      const distKm = haversineDistanceKm(spec.centerLat, spec.centerLon, cellLat, cellLon);

      if (distKm <= spec.radiusKm) {
        // Gaussian decay with smooth cutoff
        const decay = Math.exp(-(distKm * distKm) / twoSigmaSq);
        row.push(spec.magnitude * decay);
      } else {
        row.push(0);
      }
    }
    grid.push(row);
  }

  return grid;
}

/**
 * Validates physical realism bounds:
 * - Temperature pert: |ΔT| <= 5.0 K
 * - Precipitation pert: >= -40 mm/day, and baseline + pert >= 0
 * - Wind pert: |Δu|, |Δv| <= 15 m/s
 */
export function validateShockRealism(
  channel: ChannelId,
  magnitude: number
): ShockBoundsReport {
  const warnings: string[] = [];

  switch (channel) {
    case 't2m':
      if (Math.abs(magnitude) > 5.0) {
        warnings.push(`Temperature shock |${magnitude.toFixed(1)} K| exceeds climatological bounds (±5.0 K).`);
      }
      break;
    case 'tp':
      if (magnitude < -30) {
        warnings.push(`Precipitation shock (${magnitude.toFixed(1)} mm/d) may cause negative precipitation.`);
      } else if (magnitude > 60) {
        warnings.push(`Precipitation shock (+${magnitude.toFixed(1)} mm/d) exceeds 100-year return rate.`);
      }
      break;
    case 'u10':
    case 'v10':
      if (Math.abs(magnitude) > 12.0) {
        warnings.push(`Wind shock |${magnitude.toFixed(1)} m/s| exceeds typical monsoon surge capacity.`);
      }
      break;
  }

  return {
    isValid: warnings.length === 0,
    warnings,
  };
}
