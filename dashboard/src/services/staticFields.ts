/**
 * staticFields.ts - Real time-invariant terrain for relief shading.
 * public/data/static_fields.json is exported from checkpoints/static.npy (2, 32, 36):
 * land_sea_mask (0 ocean .. 1 land) and orography_m, rows south -> north like every other tensor.
 * If it is missing, the map simply draws no relief (nothing is invented).
 */

import type { Grid2D } from '../types/climate';
import { GRID_CONFIG } from '../types/climate';

export interface StaticFields {
  landSea: Grid2D;
  orography: Grid2D;
}

const isGrid = (g: unknown): g is Grid2D =>
  Array.isArray(g) &&
  g.length === GRID_CONFIG.nLat &&
  g.every((row) => Array.isArray(row) && row.length === GRID_CONFIG.nLon && row.every(Number.isFinite));

export async function loadStaticFields(): Promise<StaticFields | null> {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}data/static_fields.json`);
    if (!res.ok) return null;
    const raw = (await res.json()) as { land_sea_mask?: unknown; orography_m?: unknown };
    if (!isGrid(raw.land_sea_mask) || !isGrid(raw.orography_m)) return null;
    return { landSea: raw.land_sea_mask, orography: raw.orography_m };
  } catch {
    return null;
  }
}
