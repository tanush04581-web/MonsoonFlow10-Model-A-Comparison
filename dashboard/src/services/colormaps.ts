/**
 * colormaps.ts - Meteorological palettes (one per channel) and the divergent anomaly scale.
 *
 * Every palette is a list of [physical value (wire units), colour] stops, linearly interpolated.
 * Transparency is part of the scale: dry cells (tp < ~0.5 mm/d) and near-zero anomalies are
 * see-through so the basemap and terrain show where nothing is happening.
 */

import type { ChannelId } from '../types/climate';

export interface RgbColor {
  r: number;
  g: number;
  b: number;
}

export interface RgbaColor extends RgbColor {
  /** 0..1 */
  a: number;
}

type Stops = [number, RgbColor][];

const hex = (h: string): RgbColor => ({
  r: parseInt(h.slice(1, 3), 16),
  g: parseInt(h.slice(3, 5), 16),
  b: parseInt(h.slice(5, 7), 16),
});

export function interpolateRgb(c1: RgbColor, c2: RgbColor, t: number): RgbColor {
  const k = Math.max(0, Math.min(1, t));
  return {
    r: Math.round(c1.r + (c2.r - c1.r) * k),
    g: Math.round(c1.g + (c2.g - c1.g) * k),
    b: Math.round(c1.b + (c2.b - c1.b) * k),
  };
}

export function rgbToCss(c: RgbColor, alpha = 1): string {
  return `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha})`;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

function evaluateStops(value: number, stops: Stops): RgbColor {
  if (!(value > stops[0][0])) return stops[0][1]; // also catches NaN
  const last = stops[stops.length - 1];
  if (value >= last[0]) return last[1];
  for (let i = 0; i < stops.length - 1; i++) {
    const [v0, c0] = stops[i];
    const [v1, c1] = stops[i + 1];
    if (value <= v1) return interpolateRgb(c0, c1, (value - v0) / (v1 - v0));
  }
  return last[1];
}

// ---------------------------------------------------------------------------------------------
// Palettes (wire units: K, mm/day, m/s, Pa, kg/m^2)
// ---------------------------------------------------------------------------------------------

/** t2m: deep arctic blue -> teal -> emerald -> amber -> scorched red */
const T2M_STOPS: Stops = [
  [273.15, hex('#0B2E6F')], // 0 C   arctic blue
  [283.15, hex('#1D5FB4')], // 10 C
  [292.15, hex('#0D9488')], // 19 C  teal
  [299.15, hex('#10B981')], // 26 C  emerald
  [305.15, hex('#F59E0B')], // 32 C  amber
  [311.15, hex('#EA580C')], // 38 C
  [316.15, hex('#B91C1C')], // 43 C  scorched red
  [321.15, hex('#6B0F0F')], // 48 C
];

/** tp: (transparent dry) -> crisp cyan -> deep cobalt -> electric violet -> crimson at >= 50 mm/d */
const TP_STOPS: Stops = [
  [0.5, hex('#A5F3FC')],
  [2.5, hex('#22D3EE')], // crisp cyan
  [10, hex('#1D4ED8')], // deep cobalt (heavy-rain threshold)
  [25, hex('#1E3A8A')],
  [38, hex('#7C3AED')], // electric violet
  [50, hex('#DC143C')], // crimson (extreme threshold)
  [80, hex('#FF5C7A')],
];

/** |u10|, |v10| magnitude: slate -> cyan -> emerald -> gold */
const WIND_STOPS: Stops = [
  [0, hex('#334155')],
  [4, hex('#22D3EE')],
  [9, hex('#10B981')],
  [15, hex('#FACC15')],
  [20, hex('#FDE68A')],
];

/** msl: deep cyclone indigo -> steel blue -> emerald -> high-pressure amber */
const MSL_STOPS: Stops = [
  [99000, hex('#1E1B4B')],
  [99500, hex('#312E81')], // cyclone indigo
  [100000, hex('#4682B4')], // steel blue
  [100700, hex('#10B981')], // emerald
  [101400, hex('#F59E0B')], // high-pressure amber
  [101800, hex('#FBBF24')],
];

/** tcwv: dry desert ochre -> mint -> cerulean -> saturated sapphire */
const TCWV_STOPS: Stops = [
  [10, hex('#A16207')], // desert ochre
  [22, hex('#CA8A04')],
  [34, hex('#6EE7B7')], // mint
  [48, hex('#0EA5E9')], // cerulean
  [62, hex('#1D4ED8')],
  [75, hex('#0F2A7A')], // saturated sapphire
];

const PALETTES: Record<ChannelId, Stops> = {
  t2m: T2M_STOPS,
  tp: TP_STOPS,
  u10: WIND_STOPS,
  v10: WIND_STOPS,
  msl: MSL_STOPS,
  tcwv: TCWV_STOPS,
};

export function isWindChannel(channel: ChannelId): boolean {
  return channel === 'u10' || channel === 'v10';
}

/** Opaque palette colour for a physical value (wind channels use |component|). */
export function getColorForChannel(value: number, channel: ChannelId): RgbColor {
  const v = isWindChannel(channel) ? Math.abs(value) : value;
  return evaluateStops(v, PALETTES[channel]);
}

const FIELD_ALPHA = 0.5;

/** Palette colour + transparency rule for a physical value. */
export function channelRgba(value: number, channel: ChannelId): RgbaColor {
  const c = getColorForChannel(value, channel);
  if (channel === 'tp') {
    // Dry -> fully transparent; heavier rain -> more opaque
    const a = smoothstep(0.2, 1.5, value) * (0.34 + 0.38 * Math.min(1, value / 30));
    return { ...c, a };
  }
  return { ...c, a: FIELD_ALPHA };
}

// ---------------------------------------------------------------------------------------------
// Anomaly Diff: blue <- transparent 0 -> orange/red
// ---------------------------------------------------------------------------------------------

const NEG_STOPS: Stops = [
  [0, hex('#BAE6FD')],
  [0.35, hex('#38BDF8')],
  [0.7, hex('#2563EB')],
  [1, hex('#1E3A8A')],
];
const POS_STOPS: Stops = [
  [0, hex('#FED7AA')],
  [0.35, hex('#FB923C')],
  [0.7, hex('#EF4444')],
  [1, hex('#991B1B')],
];

/** Opaque divergent colour for Delta X_t = X_t(shock) - X_t(baseline). */
export function getAnomalyDivergentColor(delta: number, maxAbsDelta: number): RgbColor {
  const n = maxAbsDelta > 0 ? Math.max(-1, Math.min(1, delta / maxAbsDelta)) : 0;
  return n < 0 ? evaluateStops(-n, NEG_STOPS) : evaluateStops(n, POS_STOPS);
}

/** Divergent colour with transparency centred at 0 (|delta| < 1% of max is invisible). */
export function anomalyRgba(delta: number, maxAbsDelta: number): RgbaColor {
  const n = maxAbsDelta > 0 ? Math.min(1, Math.abs(delta) / maxAbsDelta) : 0;
  const a = smoothstep(0.01, 0.12, n) * (0.34 + 0.42 * n);
  return { ...getAnomalyDivergentColor(delta, maxAbsDelta), a };
}

// ---------------------------------------------------------------------------------------------
// Legend scales (wire units) with meteorologically meaningful ticks (display units)
// ---------------------------------------------------------------------------------------------

export interface LegendScale {
  min: number;
  max: number;
  title: string;
  unit: string;
  ticks: { value: number; label: string }[];
}

const kToC = (k: number) => `${Math.round(k - 273.15)}°`;

export const LEGEND_SCALES: Record<ChannelId, LegendScale> = {
  t2m: {
    min: 273.15,
    max: 321.15,
    title: '2m Temperature',
    unit: '°C',
    ticks: [273.15, 283.15, 293.15, 303.15, 313.15, 321.15].map((v) => ({ value: v, label: kToC(v) })),
  },
  tp: {
    min: 0,
    max: 80,
    title: 'Precipitation',
    unit: 'mm/day',
    ticks: [
      { value: 0, label: 'dry' },
      { value: 10, label: '10' },
      { value: 25, label: '25' },
      { value: 50, label: '50' },
      { value: 80, label: '80' },
    ],
  },
  u10: {
    min: 0,
    max: 20,
    title: '|U10| zonal wind (arrows: direction)',
    unit: 'm/s',
    ticks: [0, 5, 10, 15, 20].map((v) => ({ value: v, label: `${v}` })),
  },
  v10: {
    min: 0,
    max: 20,
    title: '|V10| meridional wind (arrows: direction)',
    unit: 'm/s',
    ticks: [0, 5, 10, 15, 20].map((v) => ({ value: v, label: `${v}` })),
  },
  msl: {
    min: 99000,
    max: 101800,
    title: 'Mean Sea-Level Pressure',
    unit: 'hPa',
    ticks: [99000, 99600, 100200, 100800, 101800].map((v) => ({ value: v, label: `${v / 100}` })),
  },
  tcwv: {
    min: 10,
    max: 75,
    title: 'Water Vapour (TCWV)',
    unit: 'kg/m²',
    ticks: [10, 25, 40, 55, 75].map((v) => ({ value: v, label: `${v}` })),
  },
};
