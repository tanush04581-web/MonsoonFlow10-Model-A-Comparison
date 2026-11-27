/**
 * diagnostics.ts - Human-readable facts derived from the displayed tensors (never invented):
 *   - where the shock starts (from the active preset / brush),
 *   - how far its footprint has spread on each lead day (drives the map's ripple rings),
 *   - pressure centres (H / L symbols) from local msl extrema,
 *   - a one-sentence headline for the current day and view.
 */

import { CHANNELS, GRID_CONFIG, LATS, LONS, type ChannelTensor, type Grid2D, type SimulationRollout } from '../types/climate';
import type { GaussianShockSpec, ShockPreset } from '../types/shocks';
import { regionName } from '../geo/basemap';

const NLAT = GRID_CONFIG.nLat;
const NLON = GRID_CONFIG.nLon;
const KM_PER_DEG = GRID_CONFIG.approxKmPerDeg;
const D2R = Math.PI / 180;

export interface ShockOrigin {
  lat: number;
  lon: number;
  /** Footprint radius at Day 0 in km (Gaussian radius, or half-diagonal of a box). */
  radiusKm: number;
  region: string;
}

export function shockOrigin(preset: ShockPreset, customSpec: GaussianShockSpec): ShockOrigin | null {
  if (preset.id === 'ZERO_SHOCK') return null;
  if (preset.box) {
    const [la0, la1, lo0, lo1] = preset.box.box;
    const lat = (la0 + la1) / 2;
    const lon = (lo0 + lo1) / 2;
    const half = Math.hypot(((la1 - la0) / 2) * KM_PER_DEG, ((lo1 - lo0) / 2) * KM_PER_DEG * Math.cos(lat * D2R));
    return { lat, lon, radiusKm: half, region: regionName(lat, lon) };
  }
  const spec = preset.id === 'CUSTOM' ? customSpec : preset.spec;
  if (!spec || spec.magnitude === 0) return null;
  return { lat: spec.centerLat, lon: spec.centerLon, radiusKm: spec.radiusKm, region: regionName(spec.centerLat, spec.centerLon) };
}

export function distanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const a =
    Math.sin(((lat2 - lat1) * D2R) / 2) ** 2 +
    Math.cos(lat1 * D2R) * Math.cos(lat2 * D2R) * Math.sin(((lon2 - lon1) * D2R) / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** "Noticeable change" per channel, in wire units. */
const NOTICEABLE: Record<string, number> = { t2m: 0.3, tp: 1.0, u10: 0.7, v10: 0.7, msl: 60, tcwv: 1.5 };

/** max over channels of |Δ| / noticeable — ≥ 1 means a person-scale change in at least one variable. */
export function changeMagnitude(shock: ChannelTensor, base: ChannelTensor): Grid2D {
  const out: Grid2D = Array.from({ length: NLAT }, () => new Array<number>(NLON).fill(0));
  for (const ch of Object.values(CHANNELS)) {
    const s = shock[ch.index];
    const b = base[ch.index];
    const scale = NOTICEABLE[ch.id];
    for (let i = 0; i < NLAT; i++) {
      for (let j = 0; j < NLON; j++) {
        const m = Math.abs(s[i][j] - b[i][j]) / scale;
        if (m > out[i][j]) out[i][j] = m;
      }
    }
  }
  return out;
}

export interface SpreadDay {
  day: number;
  /** 90th-percentile distance (km) of noticeably-changed cells from the shock centre; 0 if none. */
  frontKm: number;
  /** Area (km²) of noticeably-changed cells. */
  areaKm2: number;
  cells: number;
}

/** Day-by-day footprint of the shock (Day 0 = the perturbation itself). */
export function spreadByDay(
  day0Base: ChannelTensor,
  day0Shock: ChannelTensor | null,
  base: SimulationRollout,
  shock: SimulationRollout,
  origin: ShockOrigin | null
): SpreadDay[] {
  if (!origin) return [];
  const frames: [ChannelTensor, ChannelTensor][] = [];
  if (day0Shock) frames.push([day0Shock, day0Base]);
  else frames.push([day0Base, day0Base]);
  const steps = Math.min(base.length, shock.length);
  for (let t = 0; t < steps; t++) frames.push([shock[t], base[t]]);
  return frames.map(([s, b], day) => {
    const mag = changeMagnitude(s, b);
    const dists: number[] = [];
    let area = 0;
    for (let i = 0; i < NLAT; i++) {
      const cellArea = KM_PER_DEG * KM_PER_DEG * Math.cos(LATS[i] * D2R);
      for (let j = 0; j < NLON; j++) {
        if (mag[i][j] < 1) continue;
        dists.push(distanceKm(origin.lat, origin.lon, LATS[i], LONS[j]));
        area += cellArea;
      }
    }
    dists.sort((a, c) => a - c);
    const frontKm = dists.length ? dists[Math.min(dists.length - 1, Math.floor(dists.length * 0.9))] : 0;
    return { day, frontKm, areaKm2: Math.round(area), cells: dists.length };
  });
}

export interface PressureCentre {
  lat: number;
  lon: number;
  kind: 'H' | 'L';
  hPa: number;
  contrast: number;
}

/** Local msl extrema (5x5 window) that stand ≥ 1 hPa out from their 7x7 surroundings. */
export function pressureCentres(msl: Grid2D, maxEach = 3): PressureCentre[] {
  const found: PressureCentre[] = [];
  for (let i = 2; i < NLAT - 2; i++) {
    for (let j = 2; j < NLON - 2; j++) {
      const v = msl[i][j];
      let isMax = true;
      let isMin = true;
      for (let di = -2; di <= 2 && (isMax || isMin); di++) {
        for (let dj = -2; dj <= 2; dj++) {
          if (di === 0 && dj === 0) continue;
          const w = msl[i + di][j + dj];
          if (w >= v) isMax = false;
          if (w <= v) isMin = false;
        }
      }
      if (!isMax && !isMin) continue;
      let sum = 0;
      let n = 0;
      for (let di = -3; di <= 3; di++) {
        for (let dj = -3; dj <= 3; dj++) {
          const ii = i + di;
          const jj = j + dj;
          if (ii < 0 || jj < 0 || ii >= NLAT || jj >= NLON) continue;
          sum += msl[ii][jj];
          n++;
        }
      }
      const contrast = Math.abs(v - sum / n);
      if (contrast < 100) continue;
      found.push({ lat: LATS[i], lon: LONS[j], kind: isMax ? 'H' : 'L', hPa: Math.round(v / 100), contrast });
    }
  }
  const pick = (kind: 'H' | 'L') =>
    found
      .filter((c) => c.kind === kind)
      .sort((a, b) => b.contrast - a.contrast)
      .slice(0, maxEach);
  return [...pick('H'), ...pick('L')];
}

// ---------------------------------------------------------------------------------------------
// Headline
// ---------------------------------------------------------------------------------------------

function smooth3(g: Grid2D, i: number, j: number): number {
  let s = 0;
  let n = 0;
  for (let di = -1; di <= 1; di++) {
    for (let dj = -1; dj <= 1; dj++) {
      const ii = i + di;
      const jj = j + dj;
      if (ii < 0 || jj < 0 || ii >= NLAT || jj >= NLON) continue;
      s += g[ii][jj];
      n++;
    }
  }
  return s / n;
}

const fmt = (v: number, digits = 1) => (Math.abs(v) >= 10 ? Math.round(v).toString() : v.toFixed(digits));
const km = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} thousand` : `${Math.round(v / 10) * 10}`);

export interface Headline {
  title: string;
  detail: string;
  tone: 'neutral' | 'wet' | 'dry' | 'warm' | 'cool';
}

const dayLabel = (d: number) => (d === 0 ? 'Today (Day 0)' : `Day ${d}`);

export function buildHeadline(args: {
  day: number;
  base: ChannelTensor;
  shock: ChannelTensor;
  origin: ShockOrigin | null;
  presetName: string;
  spread: SpreadDay | null;
  showingShock: boolean;
}): Headline {
  const { day, base, shock, origin, presetName, spread, showingShock } = args;
  const tpIdx = CHANNELS.tp.index;
  const t2mIdx = CHANNELS.t2m.index;

  if (!origin || !showingShock) {
    const tp = base[tpIdx];
    let bi = 0;
    let bj = 0;
    let best = -Infinity;
    let heavy = 0;
    for (let i = 0; i < NLAT; i++) {
      for (let j = 0; j < NLON; j++) {
        const s = smooth3(tp, i, j);
        if (s > best) {
          best = s;
          bi = i;
          bj = j;
        }
        if (tp[i][j] >= 10) heavy++;
      }
    }
    return {
      title: `${dayLabel(day)}: the heaviest rain is over ${regionName(LATS[bi], LONS[bj])}, about ${fmt(best)} mm a day`,
      detail:
        heavy > 0
          ? `${heavy} of ${NLAT * NLON} grid boxes get heavy rain (10 mm/day or more).`
          : 'No grid box reaches heavy rain (10 mm/day) on this day.',
      tone: 'neutral',
    };
  }

  if (day === 0) {
    return {
      title: `“${presetName}” is applied over ${origin.region}`,
      detail: 'Press play to watch the next 10 days split away from the normal forecast.',
      tone: 'neutral',
    };
  }

  // Largest smoothed rain change
  const dTp: Grid2D = shock[tpIdx].map((row, i) => row.map((v, j) => v - base[tpIdx][i][j]));
  let bi = 0;
  let bj = 0;
  let best = 0;
  for (let i = 0; i < NLAT; i++) {
    for (let j = 0; j < NLON; j++) {
      const s = smooth3(dTp, i, j);
      if (Math.abs(s) > Math.abs(best)) {
        best = s;
        bi = i;
        bj = j;
      }
    }
  }
  const dT: Grid2D = shock[t2mIdx].map((row, i) => row.map((v, j) => v - base[t2mIdx][i][j]));
  let ti = 0;
  let tj = 0;
  let bestT = 0;
  for (let i = 0; i < NLAT; i++) {
    for (let j = 0; j < NLON; j++) {
      const s = smooth3(dT, i, j);
      if (Math.abs(s) > Math.abs(bestT)) {
        bestT = s;
        ti = i;
        tj = j;
      }
    }
  }

  const spreadText =
    spread && spread.cells > 0
      ? `The change now reaches about ${km(spread.frontKm)} km from where the shock began and covers ${(spread.areaKm2 / 1e6).toFixed(2)} million km².`
      : 'The shock has almost faded: this future is close to the normal forecast.';

  if (Math.abs(best) >= 1) {
    const baseHere = smooth3(base[tpIdx], bi, bj);
    const pct = baseHere >= 2 ? ` (${best > 0 ? '+' : '−'}${Math.round((Math.abs(best) / baseHere) * 100)}%)` : '';
    const tempBit =
      Math.abs(bestT) >= 0.5
        ? ` Air over ${regionName(LATS[ti], LONS[tj])} is ${Math.abs(bestT).toFixed(1)} °C ${bestT > 0 ? 'warmer' : 'cooler'}.`
        : '';
    return {
      title: `${dayLabel(day)}: ${best > 0 ? 'more' : 'less'} rain over ${regionName(LATS[bi], LONS[bj])}, ${best > 0 ? '+' : '−'}${fmt(Math.abs(best))} mm a day${pct} compared with the normal forecast`,
      detail: `${spreadText}${tempBit}`,
      tone: best > 0 ? 'wet' : 'dry',
    };
  }
  if (Math.abs(bestT) >= 0.3) {
    return {
      title: `${dayLabel(day)}: air over ${regionName(LATS[ti], LONS[tj])} is ${Math.abs(bestT).toFixed(1)} °C ${bestT > 0 ? 'warmer' : 'cooler'} than in the normal forecast`,
      detail: spreadText,
      tone: bestT > 0 ? 'warm' : 'cool',
    };
  }
  return {
    title: `${dayLabel(day)}: this future is almost the same as the normal forecast`,
    detail: spreadText,
    tone: 'neutral',
  };
}
