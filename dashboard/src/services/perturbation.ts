/**
 * perturbation.ts - Builds the (6, 32, 36) physical perturbation tensor sent to POST /simulate.
 * Box shocks reproduce monsoon_r1.scoring.make_shock() exactly, using the real Day-0 state.
 */

import { CHANNELS, GRID_CONFIG, LATS, LONS, type ChannelTensor } from '../types/climate';
import type { BoxShockSpec, GaussianShockSpec, LatLonBox, ShockPreset } from '../types/shocks';
import { generateGaussianShock2D } from './gaussianBrush';
import { createZeroTensor } from './syntheticData';

/** Inclusive box mask on cell centres, identical to monsoon_r1.contract.box_mask(). */
export function boxMask(box: LatLonBox): boolean[][] {
  const [la0, la1, lo0, lo1] = box;
  return LATS.map((lat) => LONS.map((lon) => lat >= la0 && lat <= la1 && lon >= lo0 && lon <= lo1));
}

export function buildBoxPerturbation(spec: BoxShockSpec, day0: ChannelTensor): ChannelTensor {
  const out = createZeroTensor();
  const mask = boxMask(spec.box);
  for (const term of spec.terms) {
    const c = CHANNELS[term.channel].index;
    for (let i = 0; i < GRID_CONFIG.nLat; i++) {
      for (let j = 0; j < GRID_CONFIG.nLon; j++) {
        if (!mask[i][j]) continue;
        out[c][i][j] += term.mode === 'add' ? term.value : term.value * day0[c][i][j];
      }
    }
  }
  return out;
}

export function buildGaussianPerturbation(spec: GaussianShockSpec): ChannelTensor | null {
  if (spec.magnitude === 0) return null;
  const out = createZeroTensor();
  out[CHANNELS[spec.channel].index] = generateGaussianShock2D(spec);
  return out;
}

/**
 * Resolve the active shock to a perturbation tensor, or null for "no shock".
 * ZERO_SHOCK -> null; box presets -> exact grader shock; gaussian preset / CUSTOM -> brush spec.
 */
export function resolvePerturbation(
  preset: ShockPreset,
  customSpec: GaussianShockSpec,
  day0: ChannelTensor
): ChannelTensor | null {
  if (preset.id === 'ZERO_SHOCK') return null;
  if (preset.box) return buildBoxPerturbation(preset.box, day0);
  if (preset.id === 'CUSTOM') return buildGaussianPerturbation(customSpec);
  if (preset.spec) return buildGaussianPerturbation(preset.spec);
  return null;
}
