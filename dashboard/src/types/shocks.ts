/**
 * shocks.ts - Shock configurations
 *
 * Two kinds of perturbation:
 * - 'box': the official graded shocks, reproduced exactly from monsoon_r1/scoring.py make_shock():
 *     warm_ocean    : box (8..20N, 82..94E), t2m += 3.0 K, tcwv += 15% of the Day-0 tcwv
 *     monsoon_stall : box (8..20N, 65..75E), u10 -= 40% and v10 -= 40% of the Day-0 wind
 *   Box membership uses inclusive bounds on cell centres (LAT >= lat0 && LAT <= lat1), as box_mask() does.
 * - 'gaussian': dashboard-only scenarios (preset or interactive brush), one channel, additive.
 */

import type { ChannelId } from './climate';

export interface GaussianShockSpec {
  centerLat: number;
  centerLon: number;
  radiusKm: number;
  sigmaKm: number; // default ~300 km (~2.5 grid cells)
  channel: ChannelId;
  magnitude: number; // in physical wire units (e.g. +3.0 K, -5 m/s)
}

/** (lat_min, lat_max, lon_min, lon_max) in degrees, inclusive on cell centres. */
export type LatLonBox = [number, number, number, number];

export interface BoxShockTerm {
  channel: ChannelId;
  /** 'add': delta = value (physical units). 'fraction': delta = value * Day-0 state. */
  mode: 'add' | 'fraction';
  value: number;
}

export interface BoxShockSpec {
  box: LatLonBox;
  terms: BoxShockTerm[];
}

export type PresetShockId =
  | 'WARM_BAY_OF_BENGAL'
  | 'MONSOON_STALL'
  | 'WESTERN_GHATS_DELUGE'
  | 'ZERO_SHOCK'
  | 'CUSTOM';

export interface ShockPreset {
  id: PresetShockId;
  name: string;
  category: 'Thermal' | 'Dynamical' | 'Hydrological' | 'Baseline' | 'Custom';
  description: string;
  /** True for the two shocks the competition grader applies (monsoon_r1.scoring.make_shock). */
  graded: boolean;
  spec: GaussianShockSpec | null;
  box: BoxShockSpec | null;
  boundsWarning?: string;
}

// Grader boxes, mirrored from monsoon_r1/contract.py
export const BOX_BAY_OF_BENGAL: LatLonBox = [8.0, 20.0, 82.0, 94.0];
export const BOX_ARABIAN_SEA: LatLonBox = [8.0, 20.0, 65.0, 75.0];

export const SHOCK_PRESETS: ShockPreset[] = [
  {
    id: 'WARM_BAY_OF_BENGAL',
    name: 'Warm Bay of Bengal',
    category: 'Thermal',
    description: 'Graded shock: +3.0 K t2m and +15% tcwv over the Bay of Bengal box (8–20°N, 82–94°E).',
    graded: true,
    spec: null,
    box: {
      box: BOX_BAY_OF_BENGAL,
      terms: [
        { channel: 't2m', mode: 'add', value: 3.0 },
        { channel: 'tcwv', mode: 'fraction', value: 0.15 },
      ],
    },
  },
  {
    id: 'MONSOON_STALL',
    name: 'Arabian Sea Monsoon Stall',
    category: 'Dynamical',
    description: 'Graded shock: −40% of the Day-0 u10 and v10 over the Arabian Sea box (8–20°N, 65–75°E).',
    graded: true,
    spec: null,
    box: {
      box: BOX_ARABIAN_SEA,
      terms: [
        { channel: 'u10', mode: 'fraction', value: -0.4 },
        { channel: 'v10', mode: 'fraction', value: -0.4 },
      ],
    },
  },
  {
    id: 'WESTERN_GHATS_DELUGE',
    name: 'Western Ghats Deluge',
    category: 'Hydrological',
    description: 'Dashboard scenario (not graded): +30 mm/day Gaussian rain pulse over the Konkan/Goa coast (15.5°N, 73.5°E).',
    graded: false,
    spec: {
      centerLat: 15.5,
      centerLon: 73.5,
      radiusKm: 350,
      sigmaKm: 220,
      channel: 'tp',
      magnitude: 30.0,
    },
    box: null,
  },
  {
    id: 'ZERO_SHOCK',
    name: 'Zero Shock / Reset',
    category: 'Baseline',
    description: 'No counterfactual perturbation (P = null). The shock rollout equals the baseline rollout.',
    graded: false,
    spec: null,
    box: null,
  },
];

/** Synthetic entry describing the interactive brush (not listed as a preset button). */
export const CUSTOM_PRESET: ShockPreset = {
  id: 'CUSTOM',
  name: 'Custom Gaussian Brush',
  category: 'Custom',
  description: 'User-painted single-channel Gaussian perturbation.',
  graded: false,
  spec: null,
  box: null,
};

/** Per-channel amplitude slider ranges in wire units (msl is in Pa, so it needs a wide range). */
export const AMPLITUDE_RANGE: Record<ChannelId, { min: number; max: number; step: number }> = {
  t2m: { min: -10, max: 10, step: 0.1 },
  tp: { min: -30, max: 60, step: 0.5 },
  u10: { min: -15, max: 15, step: 0.5 },
  v10: { min: -15, max: 15, step: 0.5 },
  msl: { min: -1500, max: 1500, step: 25 },
  tcwv: { min: -20, max: 20, step: 0.5 },
};
