/**
 * fieldRaster.ts - Turns live 32x36 tensors into map imagery.
 *
 * Grid convention (same as monsoon_r1.contract): row i -> lat 6.5 + i (south -> north),
 * col j -> lon 65.5 + j (west -> east); cell (i, j) spans its centre +/- 0.5 deg.
 * "Fractional grid coords" (fi, fj) put cell centres on integers.
 */

import type { Grid2D } from '../types/climate';
import { GRID_CONFIG } from '../types/climate';
import type { RgbaColor } from './colormaps';

const NLAT = GRID_CONFIG.nLat;
const NLON = GRID_CONFIG.nLon;
const HALF = GRID_CONFIG.resolutionDeg / 2;

/** Bilinear sample at fractional grid coords, clamped to the outermost cell centres. */
export function sampleBilinear(grid: Grid2D, fi: number, fj: number): number {
  const i = Math.max(0, Math.min(NLAT - 1, fi));
  const j = Math.max(0, Math.min(NLON - 1, fj));
  const i0 = Math.min(NLAT - 2, Math.floor(i));
  const j0 = Math.min(NLON - 2, Math.floor(j));
  const ti = i - i0;
  const tj = j - j0;
  const r0 = grid[i0];
  const r1 = grid[i0 + 1];
  const top = r0[j0] + (r0[j0 + 1] - r0[j0]) * tj;
  const bot = r1[j0] + (r1[j0 + 1] - r1[j0]) * tj;
  return top + (bot - top) * ti;
}

export function gridDelta(a: Grid2D, b: Grid2D): Grid2D {
  return a.map((row, i) => row.map((v, j) => v - b[i][j]));
}

// ---------------------------------------------------------------------------------------------
// Projection: screen projectors come from geo/lcc.ts (Lambert Conformal Conic MapView)
// ---------------------------------------------------------------------------------------------

export type Projector = (lat: number, lon: number) => [number, number];

export const fracToLat = (fi: number) => GRID_CONFIG.latMin + fi * GRID_CONFIG.resolutionDeg;
export const fracToLon = (fj: number) => GRID_CONFIG.lonMin + fj * GRID_CONFIG.resolutionDeg;

// ---------------------------------------------------------------------------------------------
// Field image: S pixels per cell. S = 1 -> exact cell values (blocky); S > 1 -> bilinear.
// Pixel row 0 is the NORTH edge of the image.
//
// `margin` (whole cells) adds a soft edge OUTSIDE the domain: it repeats the nearest edge colour
// and fades it to fully transparent within EDGE_FADE_CELLS, so the field dissolves into the
// basemap instead of ending in a hard square. It carries no new data (the outline marks the
// valid area); every pixel inside the domain is unchanged.
// ---------------------------------------------------------------------------------------------

/** Width of the outside fade, in cells (degrees). */
export const EDGE_FADE_CELLS = 1.5;

function smoothstep01(x: number): number {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3 - 2 * t);
}

/** 1 inside the domain, fading to 0 at EDGE_FADE_CELLS outside it (fractional grid coords). */
export function edgeFade(fi: number, fj: number): number {
  const di = Math.max(-0.5 - fi, fi - (NLAT - 0.5), 0);
  const dj = Math.max(-0.5 - fj, fj - (NLON - 0.5), 0);
  const dist = Math.hypot(di, dj);
  return dist === 0 ? 1 : 1 - smoothstep01(dist / EDGE_FADE_CELLS);
}

export function buildFieldImage(
  grid: Grid2D,
  colour: (v: number) => RgbaColor,
  S: number,
  margin = 0
): HTMLCanvasElement {
  const W = (NLON + 2 * margin) * S;
  const H = (NLAT + 2 * margin) * S;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  const img = ctx.createImageData(W, H);
  const d = img.data;
  for (let py = 0; py < H; py++) {
    const fi = NLAT + margin - (py + 0.5) / S - 0.5;
    for (let px = 0; px < W; px++) {
      const fj = (px + 0.5) / S - 0.5 - margin;
      const fade = margin > 0 ? edgeFade(fi, fj) : 1;
      if (fade <= 0) continue;
      const v = S === 1 && margin === 0 ? grid[NLAT - 1 - py][px] : sampleBilinear(grid, fi, fj);
      const c = colour(v);
      const k = (py * W + px) * 4;
      d[k] = c.r;
      d[k + 1] = c.g;
      d[k + 2] = c.b;
      d[k + 3] = Math.round(c.a * fade * 255);
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/**
 * Draw a field image onto the map canvas, one grid row (S pixel rows) per call so the
 * Mercator stretch in latitude is respected.
 */
export function drawFieldImage(
  ctx: CanvasRenderingContext2D,
  image: HTMLCanvasElement,
  project: Projector,
  smooth: boolean,
  margin = 0
): void {
  const rows = NLAT + 2 * margin;
  const S = image.height / rows;
  const north = GRID_CONFIG.latMax + HALF + margin;
  ctx.imageSmoothingEnabled = smooth;
  // Snap strip edges to device pixels: adjacent strips share an edge exactly, so there is
  // neither a gap nor a double-painted (brighter) seam with semi-transparent colours.
  const dpr = window.devicePixelRatio || 1;
  const snap = (v: number) => Math.round(v * dpr) / dpr;
  const [x0] = project(0, GRID_CONFIG.lonMin - HALF - margin);
  const [x1] = project(0, GRID_CONFIG.lonMax + HALF + margin);
  const left = snap(x0);
  const width = snap(x1) - left;
  let yTop = snap(project(north, 0)[1]);
  for (let r = 0; r < rows; r++) {
    const yBot = snap(project(north - r - 1, 0)[1]);
    if (yBot > yTop) ctx.drawImage(image, 0, r * S, image.width, S, left, yTop, width, yBot - yTop);
    yTop = yBot;
  }
}

// ---------------------------------------------------------------------------------------------
// Hillshade from real orography (checkpoints/static.npy -> public/data/static_fields.json)
// ---------------------------------------------------------------------------------------------

export function buildHillshadeImage(orography: Grid2D, landSea: Grid2D, S: number): HTMLCanvasElement {
  const W = NLON * S;
  const H = NLAT * S;
  const elev = new Float32Array(W * H);
  const land = new Float32Array(W * H);
  for (let py = 0; py < H; py++) {
    const fi = NLAT - (py + 0.5) / S - 0.5;
    for (let px = 0; px < W; px++) {
      const fj = (px + 0.5) / S - 0.5;
      elev[py * W + px] = sampleBilinear(orography, fi, fj);
      land[py * W + px] = sampleBilinear(landSea, fi, fj);
    }
  }

  // Bilinear surfaces have piecewise-constant slopes (blocky shading): smooth first with
  // three box passes (~Gaussian, sigma ~ 0.6 cell) in each direction.
  const radius = Math.max(1, Math.round(S * 0.45));
  boxBlur(elev, W, H, radius, 3);

  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  const img = ctx.createImageData(W, H);
  const d = img.data;

  const EXAGGERATION = 30; // 1-deg orography is heavily smoothed; exaggerate so relief reads
  const az = (315 * Math.PI) / 180; // light from the north-west (clockwise from north)
  const alt = (42 * Math.PI) / 180;
  const lx = Math.sin(az) * Math.cos(alt);
  const ly = Math.cos(az) * Math.cos(alt);
  const lz = Math.sin(alt);
  const flat = lz;
  const dy = (111_000 * GRID_CONFIG.resolutionDeg) / S;

  for (let py = 1; py < H - 1; py++) {
    const lat = fracToLat(NLAT - (py + 0.5) / S - 0.5);
    const dx = dy * Math.cos((lat * Math.PI) / 180);
    for (let px = 1; px < W - 1; px++) {
      const k = py * W + px;
      // Image y grows southwards; gradients in (east, north)
      const dzdx = ((elev[k + 1] - elev[k - 1]) / (2 * dx)) * EXAGGERATION;
      const dzdy = ((elev[k - W] - elev[k + W]) / (2 * dy)) * EXAGGERATION;
      // Lambert: unit surface normal (east, north, up) . unit light vector
      const norm = Math.hypot(dzdx, dzdy, 1);
      const shade = (-dzdx * lx - dzdy * ly + lz) / norm;
      const diff = shade - flat;
      // Ocean stays untouched; land gets light/dark relief
      const w = Math.max(0, Math.min(1, (land[k] - 0.2) / 0.5));
      const strength = Math.min(1, Math.abs(diff) * 2.2) * w;
      const o = k * 4;
      const v = diff > 0 ? 255 : 0;
      d[o] = v;
      d[o + 1] = v;
      d[o + 2] = v;
      d[o + 3] = Math.round(strength * (diff > 0 ? 75 : 140));
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** In-place separable box blur (edge-clamped), `passes` times. */
function boxBlur(a: Float32Array, W: number, H: number, r: number, passes: number): void {
  const tmp = new Float32Array(a.length);
  const n = 2 * r + 1;
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let s = 0;
        for (let k = -r; k <= r; k++) s += a[y * W + Math.max(0, Math.min(W - 1, x + k))];
        tmp[y * W + x] = s / n;
      }
    }
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let s = 0;
        for (let k = -r; k <= r; k++) s += tmp[Math.max(0, Math.min(H - 1, y + k)) * W + x];
        a[y * W + x] = s / n;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Contours (marching squares on a bilinearly refined grid) -> segments in fractional grid coords
// ---------------------------------------------------------------------------------------------

export type Segment = [number, number, number, number]; // fi1, fj1, fi2, fj2

export function contourSegments(grid: Grid2D, level: number, refine = 4): Segment[] {
  const R = (NLAT - 1) * refine + 1;
  const C = (NLON - 1) * refine + 1;
  const f = new Float32Array(R * C);
  for (let r = 0; r < R; r++) {
    for (let c = 0; c < C; c++) f[r * C + c] = sampleBilinear(grid, r / refine, c / refine);
  }
  const segs: Segment[] = [];
  const lerp = (a: number, b: number) => (a === b ? 0.5 : (level - a) / (b - a));
  for (let r = 0; r < R - 1; r++) {
    for (let c = 0; c < C - 1; c++) {
      const v00 = f[r * C + c];
      const v01 = f[r * C + c + 1];
      const v10 = f[(r + 1) * C + c];
      const v11 = f[(r + 1) * C + c + 1];
      const code = (v00 >= level ? 1 : 0) | (v01 >= level ? 2 : 0) | (v11 >= level ? 4 : 0) | (v10 >= level ? 8 : 0);
      if (code === 0 || code === 15) continue;
      // Edge points (row, col) in refined units
      const eB: [number, number] = [r, c + lerp(v00, v01)]; // between 00-01
      const eR: [number, number] = [r + lerp(v01, v11), c + 1]; // between 01-11
      const eT: [number, number] = [r + 1, c + lerp(v10, v11)]; // between 10-11
      const eL: [number, number] = [r + lerp(v00, v10), c]; // between 00-10
      const pairs: [[number, number], [number, number]][] = [];
      switch (code) {
        case 1: case 14: pairs.push([eL, eB]); break;
        case 2: case 13: pairs.push([eB, eR]); break;
        case 3: case 12: pairs.push([eL, eR]); break;
        case 4: case 11: pairs.push([eR, eT]); break;
        case 6: case 9: pairs.push([eB, eT]); break;
        case 7: case 8: pairs.push([eL, eT]); break;
        case 5: pairs.push([eL, eT], [eB, eR]); break;
        case 10: pairs.push([eL, eB], [eR, eT]); break;
      }
      for (const [p, q] of pairs) segs.push([p[0] / refine, p[1] / refine, q[0] / refine, q[1] / refine]);
    }
  }
  return segs;
}

export interface ContourStyle {
  color: string;
  width: number;
  glow: number; // shadow blur in px; 0 = plain isoline
  dash?: number[];
}

export function strokeSegments(
  ctx: CanvasRenderingContext2D,
  segs: Segment[],
  project: Projector,
  style: ContourStyle
): void {
  if (segs.length === 0) return;
  ctx.beginPath();
  for (const [i1, j1, i2, j2] of segs) {
    const [x1, y1] = project(fracToLat(i1), fracToLon(j1));
    const [x2, y2] = project(fracToLat(i2), fracToLon(j2));
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
  }
  ctx.save();
  ctx.lineCap = 'round';
  ctx.setLineDash(style.dash ?? []);
  ctx.strokeStyle = style.color;
  ctx.lineWidth = style.width;
  if (style.glow > 0) {
    ctx.shadowColor = style.color;
    ctx.shadowBlur = style.glow;
    ctx.stroke(); // halo
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 0.9;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = Math.max(0.6, style.width * 0.45);
  }
  ctx.stroke();
  ctx.restore();
}
