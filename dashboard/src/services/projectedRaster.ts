/**
 * projectedRaster.ts - Renders 32x36 tensors as images in Lambert Conformal Conic space.
 *
 * A fixed bitmap covers the projected domain (plus a soft fade margin). For every bitmap pixel the
 * inverse projection is solved ONCE and cached as fractional grid coords (fi, fj) + an edge-fade
 * weight. Painting a new field is then just a bilinear lookup + colour LUT per pixel (~5–10 ms),
 * and pan/zoom is a single affine drawImage, because LCC space -> screen is affine.
 *
 * Nothing outside the domain is invented: the margin only fades the nearest edge colour to zero.
 */

import type { Grid2D } from '../types/climate';
import { GRID_CONFIG } from '../types/climate';
import { lccInverse, projectedBounds, type Bounds, type MapView } from '../geo/lcc';
import { edgeFade } from './fieldRaster';
import type { RgbaColor } from './colormaps';

const NLAT = GRID_CONFIG.nLat;
const NLON = GRID_CONFIG.nLon;
const HALF = GRID_CONFIG.resolutionDeg / 2;
const LUT_SIZE = 1024;

export class ProjectedRaster {
  readonly W: number;
  readonly H: number;
  readonly bounds: Bounds;
  private readonly fi: Float32Array;
  private readonly fj: Float32Array;
  private readonly fade: Float32Array;
  private readonly inside: Uint8Array;

  constructor(widthPx = 760, marginCells = 2) {
    this.bounds = projectedBounds(
      GRID_CONFIG.latMin - HALF - marginCells,
      GRID_CONFIG.latMax + HALF + marginCells,
      GRID_CONFIG.lonMin - HALF - marginCells,
      GRID_CONFIG.lonMax + HALF + marginCells
    );
    const b = this.bounds;
    const res = (b.x1 - b.x0) / widthPx;
    this.W = widthPx;
    this.H = Math.ceil((b.y1 - b.y0) / res);
    const n = this.W * this.H;
    this.fi = new Float32Array(n);
    this.fj = new Float32Array(n);
    this.fade = new Float32Array(n);
    this.inside = new Uint8Array(n);
    for (let py = 0; py < this.H; py++) {
      const y = b.y1 - (py + 0.5) * res;
      for (let px = 0; px < this.W; px++) {
        const x = b.x0 + (px + 0.5) * res;
        const [lat, lon] = lccInverse(x, y);
        const fi = (lat - GRID_CONFIG.latMin) / GRID_CONFIG.resolutionDeg;
        const fj = (lon - GRID_CONFIG.lonMin) / GRID_CONFIG.resolutionDeg;
        const k = py * this.W + px;
        this.fi[k] = fi;
        this.fj[k] = fj;
        this.fade[k] = edgeFade(fi, fj);
        this.inside[k] = fi >= -0.5 && fi <= NLAT - 0.5 && fj >= -0.5 && fj <= NLON - 0.5 ? 1 : 0;
      }
    }
  }

  private newCanvas(): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; img: ImageData } | null {
    const canvas = document.createElement('canvas');
    canvas.width = this.W;
    canvas.height = this.H;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    return { canvas, ctx, img: ctx.createImageData(this.W, this.H) };
  }

  /**
   * Colour a field. smooth = bilinear + soft fade past the domain edge; otherwise exact 1° cells
   * with a hard edge (what the model actually outputs).
   */
  paintField(grid: Grid2D, colour: (v: number) => RgbaColor, smooth: boolean): HTMLCanvasElement | null {
    const out = this.newCanvas();
    if (!out) return null;
    const flat = new Float32Array(NLAT * NLON);
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < NLAT; i++) {
      for (let j = 0; j < NLON; j++) {
        const v = grid[i][j];
        flat[i * NLON + j] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return out.canvas;
    const span = hi - lo > 1e-9 ? hi - lo : 1;
    // Colour LUT over the field's own range (bilinear values never leave [lo, hi])
    const lut = new Uint8ClampedArray(LUT_SIZE * 4);
    for (let s = 0; s < LUT_SIZE; s++) {
      const c = colour(lo + (span * s) / (LUT_SIZE - 1));
      lut[s * 4] = c.r;
      lut[s * 4 + 1] = c.g;
      lut[s * 4 + 2] = c.b;
      lut[s * 4 + 3] = Math.round(c.a * 255);
    }
    const d = out.img.data;
    const n = this.W * this.H;
    const scale = (LUT_SIZE - 1) / span;
    for (let k = 0; k < n; k++) {
      let v: number;
      let w: number;
      if (smooth) {
        w = this.fade[k];
        if (w <= 0) continue;
        const i = Math.max(0, Math.min(NLAT - 1, this.fi[k]));
        const j = Math.max(0, Math.min(NLON - 1, this.fj[k]));
        const i0 = Math.min(NLAT - 2, Math.floor(i));
        const j0 = Math.min(NLON - 2, Math.floor(j));
        const ti = i - i0;
        const tj = j - j0;
        const r0 = i0 * NLON;
        const r1 = r0 + NLON;
        const top = flat[r0 + j0] + (flat[r0 + j0 + 1] - flat[r0 + j0]) * tj;
        const bot = flat[r1 + j0] + (flat[r1 + j0 + 1] - flat[r1 + j0]) * tj;
        v = top + (bot - top) * ti;
      } else {
        if (!this.inside[k]) continue;
        w = 1;
        const i = Math.max(0, Math.min(NLAT - 1, Math.round(this.fi[k])));
        const j = Math.max(0, Math.min(NLON - 1, Math.round(this.fj[k])));
        v = flat[i * NLON + j];
      }
      const s = Math.max(0, Math.min(LUT_SIZE - 1, Math.round((v - lo) * scale))) * 4;
      const o = k * 4;
      d[o] = lut[s];
      d[o + 1] = lut[s + 1];
      d[o + 2] = lut[s + 2];
      d[o + 3] = lut[s + 3] * w;
    }
    out.ctx.putImageData(out.img, 0, 0);
    return out.canvas;
  }

  /** Re-project a grid-space image (north-up, S px per cell, no margin) into this raster. */
  warpGridImage(src: HTMLCanvasElement, S: number): HTMLCanvasElement | null {
    const sctx = src.getContext('2d');
    const out = this.newCanvas();
    if (!sctx || !out) return null;
    const sw = src.width;
    const sh = src.height;
    const sd = sctx.getImageData(0, 0, sw, sh).data;
    const d = out.img.data;
    const n = this.W * this.H;
    for (let k = 0; k < n; k++) {
      if (!this.inside[k]) continue;
      const px = Math.round((this.fj[k] + 0.5) * S - 0.5);
      const py = Math.round((NLAT - 0.5 - this.fi[k]) * S - 0.5);
      if (px < 0 || py < 0 || px >= sw || py >= sh) continue;
      const si = (py * sw + px) * 4;
      const o = k * 4;
      d[o] = sd[si];
      d[o + 1] = sd[si + 1];
      d[o + 2] = sd[si + 2];
      d[o + 3] = sd[si + 3];
    }
    out.ctx.putImageData(out.img, 0, 0);
    return out.canvas;
  }

  /** Draw a raster produced by this instance through the current view. */
  draw(ctx: CanvasRenderingContext2D, view: MapView, image: HTMLCanvasElement, alpha = 1, smooth = true): void {
    const b = this.bounds;
    const [x0, y0] = view.toScreen(b.x0, b.y1);
    const [x1, y1] = view.toScreen(b.x1, b.y0);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.imageSmoothingEnabled = smooth;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(image, x0, y0, x1 - x0, y1 - y0);
    ctx.restore();
  }
}

let shared: ProjectedRaster | null = null;
/** One shared lookup (the inverse projection is the expensive part). */
export function getProjectedRaster(): ProjectedRaster {
  if (!shared) shared = new ProjectedRaster();
  return shared;
}
