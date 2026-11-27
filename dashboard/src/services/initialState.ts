/**
 * initialState.ts - Loads the real 3-day initial condition shipped in public/data/initial_state.json
 * (a validation-split window exported from data/india_climate_train.nc). Falls back to the
 * synthetic analytic field ONLY if that file cannot be loaded, and says so.
 */

import { GRID_CONFIG, type ChannelTensor } from '../types/climate';
import { generate3DayClimatologicalHistory } from './syntheticData';

export interface InitialState {
  history3Day: ChannelTensor[]; // (3, 6, 32, 36) physical units
  doySincos: [number, number]; // [sin, cos] of Day 0, matching monsoon_r1.contract.doy_sincos
  label: string; // human-readable Day-0 description
  isSynthetic: boolean;
}

interface InitialStateFile {
  source: string;
  t0_label: string;
  t0_doy: number;
  doy_sincos: [number, number];
  init_states: number[][][][];
}

function hasGridShape(x: number[][][][]): boolean {
  return (
    x.length === 3 &&
    x.every(
      (t) =>
        t.length === 6 &&
        t.every((c) => c.length === GRID_CONFIG.nLat && c.every((row) => row.length === GRID_CONFIG.nLon))
    )
  );
}

function syntheticFallback(reason: string): InitialState {
  // DOY 196 (mid-July), same formula as the backend contract: 2*pi*doy/365
  const theta = (2 * Math.PI * 196) / 365;
  return {
    history3Day: generate3DayClimatologicalHistory(),
    doySincos: [Math.sin(theta), Math.cos(theta)],
    label: `SYNTHETIC analytic field (${reason})`,
    isSynthetic: true,
  };
}

export async function loadInitialState(): Promise<InitialState> {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}data/initial_state.json`);
    if (!res.ok) return syntheticFallback(`initial_state.json HTTP ${res.status}`);
    const file: InitialStateFile = await res.json();
    if (!hasGridShape(file.init_states)) return syntheticFallback('initial_state.json has wrong shape');
    return {
      history3Day: file.init_states,
      doySincos: file.doy_sincos,
      label: `${file.t0_label} (DOY ${file.t0_doy}) · validation data`,
      isSynthetic: false,
    };
  } catch {
    return syntheticFallback('initial_state.json not reachable');
  }
}
