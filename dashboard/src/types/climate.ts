/**
 * climate.ts - Physical channel specifications and domain definitions
 * India Domain: 32x36 grid (6.5°N - 37.5°N, 65.5°E - 100.5°E) at 1.0° resolution
 */

export const GRID_CONFIG = {
  nLat: 32,
  nLon: 36,
  latMin: 6.5,
  latMax: 37.5,
  lonMin: 65.5,
  lonMax: 100.5,
  resolutionDeg: 1.0,
  approxKmPerDeg: 111.0,
} as const;

export type ChannelId = 't2m' | 'tp' | 'u10' | 'v10' | 'msl' | 'tcwv';

export interface ChannelMeta {
  id: ChannelId;
  index: number;
  label: string;
  wireUnit: string;
  displayUnit: string;
  min: number;
  max: number;
  step: number;
  description: string;
}

export const CHANNELS: Record<ChannelId, ChannelMeta> = {
  t2m: {
    id: 't2m',
    index: 0,
    label: '2m Temperature',
    wireUnit: 'K',
    displayUnit: '°C',
    min: 273.15 + 10, // 10°C
    max: 273.15 + 46, // 46°C
    step: 0.1,
    description: 'Near-surface air temperature at 2 metres',
  },
  tp: {
    id: 'tp',
    index: 1,
    label: 'Precipitation',
    wireUnit: 'mm/day',
    displayUnit: 'mm/day',
    min: 0,
    max: 80,
    step: 0.5,
    description: 'Daily accumulated total precipitation',
  },
  u10: {
    id: 'u10',
    index: 2,
    label: 'Zonal Wind (U10)',
    wireUnit: 'm/s',
    displayUnit: 'm/s',
    min: -20,
    max: 20,
    step: 0.2,
    description: 'Eastward 10m wind component',
  },
  v10: {
    id: 'v10',
    index: 3,
    label: 'Meridional Wind (V10)',
    wireUnit: 'm/s',
    displayUnit: 'm/s',
    min: -20,
    max: 20,
    step: 0.2,
    description: 'Northward 10m wind component',
  },
  msl: {
    id: 'msl',
    index: 4,
    label: 'Mean Sea Level Pressure',
    wireUnit: 'Pa',
    displayUnit: 'hPa',
    min: 99000,
    max: 101800,
    step: 10,
    description: 'Atmospheric pressure normalized to sea level',
  },
  tcwv: {
    id: 'tcwv',
    index: 5,
    label: 'Water Vapour (TCWV)',
    wireUnit: 'kg/m²',
    displayUnit: 'kg/m²',
    min: 10,
    max: 75,
    step: 0.5,
    description: 'Total column integrated atmospheric water vapour',
  },
};

export const CHANNEL_LIST: ChannelMeta[] = Object.values(CHANNELS);

// Physical domain grid coordinates
export const LATS: number[] = Array.from(
  { length: GRID_CONFIG.nLat },
  (_, i) => GRID_CONFIG.latMin + i * GRID_CONFIG.resolutionDeg
);

export const LONS: number[] = Array.from(
  { length: GRID_CONFIG.nLon },
  (_, j) => GRID_CONFIG.lonMin + j * GRID_CONFIG.resolutionDeg
);

// Tensor types: [time (10), channel (6), lat (32), lon (36)]
export type Grid2D = number[][];
export type ChannelTensor = Grid2D[]; // [6, 32, 36]
export type SimulationRollout = ChannelTensor[]; // [10, 6, 32, 36]
