/**
 * lcc.ts - Lambert Conformal Conic projection for the India domain + a pan/zoom view on top of it.
 *
 * Standard parallels 12°N / 32°N bracket the 6.5–37.5°N domain (scale error < 1.5 % everywhere),
 * central meridian 83°E, origin 22°N. This is the family of projection used for Indian
 * meteorological charts: meridians converge northwards, parallels are concentric arcs.
 *
 * Projected units are on the unit sphere (1 unit ≈ 6371 km at the standard parallels); y points
 * north. The view maps them to CSS pixels with a uniform scale k and a translation (tx, ty).
 */

import { GRID_CONFIG } from '../types/climate';

const D2R = Math.PI / 180;
const PHI1 = 12 * D2R;
const PHI2 = 32 * D2R;
const PHI0 = 22 * D2R;
export const LAMBDA0_DEG = 83;
const LAMBDA0 = LAMBDA0_DEG * D2R;
export const EARTH_RADIUS_KM = 6371;

const tanHalf = (phi: number) => Math.tan(Math.PI / 4 + phi / 2);
const N = Math.log(Math.cos(PHI1) / Math.cos(PHI2)) / Math.log(tanHalf(PHI2) / tanHalf(PHI1));
const F = (Math.cos(PHI1) * Math.pow(tanHalf(PHI1), N)) / N;
const RHO0 = F / Math.pow(tanHalf(PHI0), N);

/** (lat, lon) in degrees -> projected (x, y), y north-up. */
export function lccForward(lat: number, lon: number): [number, number] {
  const phi = Math.max(-89, Math.min(89, lat)) * D2R;
  const rho = F / Math.pow(tanHalf(phi), N);
  const theta = N * (lon * D2R - LAMBDA0);
  return [rho * Math.sin(theta), RHO0 - rho * Math.cos(theta)];
}

/** Projected (x, y) -> (lat, lon) in degrees. */
export function lccInverse(x: number, y: number): [number, number] {
  const dy = RHO0 - y;
  const rho = Math.sign(N) * Math.hypot(x, dy);
  const theta = Math.atan2(x, dy);
  const lon = (LAMBDA0 + theta / N) / D2R;
  const lat = (2 * Math.atan(Math.pow(F / rho, 1 / N)) - Math.PI / 2) / D2R;
  return [lat, lon];
}

export interface Bounds {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** Projected bounding box of a lat/lon rectangle (edges are curves, so sample them). */
export function projectedBounds(latMin: number, latMax: number, lonMin: number, lonMax: number): Bounds {
  const b: Bounds = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity };
  const add = (lat: number, lon: number) => {
    const [x, y] = lccForward(lat, lon);
    b.x0 = Math.min(b.x0, x);
    b.x1 = Math.max(b.x1, x);
    b.y0 = Math.min(b.y0, y);
    b.y1 = Math.max(b.y1, y);
  };
  for (let s = 0; s <= 40; s++) {
    const t = s / 40;
    add(latMin, lonMin + (lonMax - lonMin) * t);
    add(latMax, lonMin + (lonMax - lonMin) * t);
    add(latMin + (latMax - latMin) * t, lonMin);
    add(latMin + (latMax - latMin) * t, lonMax);
  }
  return b;
}

const HALF = GRID_CONFIG.resolutionDeg / 2;
/** Outer edge of the 32x36 model domain (cell edges, not centres). */
export const DOMAIN_BOUNDS = projectedBounds(
  GRID_CONFIG.latMin - HALF,
  GRID_CONFIG.latMax + HALF,
  GRID_CONFIG.lonMin - HALF,
  GRID_CONFIG.lonMax + HALF
);

export interface Padding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export type Projector = (lat: number, lon: number) => [number, number];

const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * Pan/zoom view over LCC space. Holds no React state: listeners are notified on every change
 * so canvases can redraw synchronously inside the same animation frame.
 */
export class MapView {
  k = 1;
  tx = 0;
  ty = 0;
  width = 1;
  height = 1;
  minK = 1;
  maxK = 1;
  private padding: Padding = { top: 0, right: 0, bottom: 0, left: 0 };
  private listeners = new Set<() => void>();
  private anim = 0;

  onChange(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(): void {
    this.listeners.forEach((fn) => fn());
  }

  /** Screen projector: (lat, lon) -> CSS px. */
  readonly project: Projector = (lat, lon) => {
    const [x, y] = lccForward(lat, lon);
    return [this.tx + this.k * x, this.ty - this.k * y];
  };

  /** Projected units -> CSS px. */
  toScreen(x: number, y: number): [number, number] {
    return [this.tx + this.k * x, this.ty - this.k * y];
  }

  /** CSS px -> (lat, lon). */
  invert(sx: number, sy: number): [number, number] {
    return lccInverse((sx - this.tx) / this.k, (this.ty - sy) / this.k);
  }

  /** CSS px per km at the standard parallels (good to ~1.5 % across the domain). */
  pxPerKm(): number {
    return this.k / EARTH_RADIUS_KM;
  }

  private fitParams(b: Bounds, pad: Padding): { k: number; tx: number; ty: number } {
    const availW = Math.max(40, this.width - pad.left - pad.right);
    const availH = Math.max(40, this.height - pad.top - pad.bottom);
    const k = Math.min(availW / (b.x1 - b.x0), availH / (b.y1 - b.y0));
    const cx = (b.x0 + b.x1) / 2;
    const cy = (b.y0 + b.y1) / 2;
    return {
      k,
      tx: pad.left + availW / 2 - k * cx,
      ty: pad.top + availH / 2 + k * cy,
    };
  }

  /** Resize the viewport; keeps the domain fitted to the free area inside `padding`. */
  resize(width: number, height: number, padding: Padding, refit: boolean): void {
    this.width = width;
    this.height = height;
    this.padding = padding;
    const fit = this.fitParams(DOMAIN_BOUNDS, padding);
    this.minK = fit.k * 0.8;
    this.maxK = fit.k * 6;
    if (refit) {
      this.k = fit.k;
      this.tx = fit.tx;
      this.ty = fit.ty;
    }
    this.clamp();
    this.emit();
  }

  /** Animate to the fitted domain. `fromScale` < 1 starts zoomed out for a fly-in. */
  flyToDomain(durationMs = 900, fromScale?: number): void {
    const target = this.fitParams(DOMAIN_BOUNDS, this.padding);
    if (fromScale !== undefined) {
      const cx = this.width / 2;
      const cy = this.height / 2;
      this.k = target.k * fromScale;
      this.tx = cx - (cx - target.tx) * fromScale;
      this.ty = cy - (cy - target.ty) * fromScale;
    }
    this.animateTo(target, durationMs);
  }

  private animateTo(target: { k: number; tx: number; ty: number }, durationMs: number): void {
    cancelAnimationFrame(this.anim);
    const start = { k: this.k, tx: this.tx, ty: this.ty };
    const reduced =
      typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduced || durationMs <= 0) {
      Object.assign(this, target);
      this.emit();
      return;
    }
    const t0 = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - t0) / durationMs);
      const e = easeInOutCubic(t);
      // Interpolate zoom geometrically so the fly feels uniform
      this.k = start.k * Math.pow(target.k / start.k, e);
      this.tx = start.tx + (target.tx - start.tx) * e;
      this.ty = start.ty + (target.ty - start.ty) * e;
      this.emit();
      if (t < 1) this.anim = requestAnimationFrame(step);
    };
    this.anim = requestAnimationFrame(step);
  }

  panBy(dx: number, dy: number): void {
    cancelAnimationFrame(this.anim);
    this.tx += dx;
    this.ty += dy;
    this.clamp();
    this.emit();
  }

  zoomAt(factor: number, sx: number, sy: number): void {
    cancelAnimationFrame(this.anim);
    const k = Math.max(this.minK, Math.min(this.maxK, this.k * factor));
    const f = k / this.k;
    this.tx = sx - (sx - this.tx) * f;
    this.ty = sy - (sy - this.ty) * f;
    this.k = k;
    this.clamp();
    this.emit();
  }

  /** Keep at least part of the domain on screen. */
  private clamp(): void {
    const b = DOMAIN_BOUNDS;
    const left = this.tx + this.k * b.x0;
    const right = this.tx + this.k * b.x1;
    const top = this.ty - this.k * b.y1;
    const bottom = this.ty - this.k * b.y0;
    const mx = this.width * 0.35;
    const my = this.height * 0.35;
    if (left > this.width - mx) this.tx -= left - (this.width - mx);
    if (right < mx) this.tx += mx - right;
    if (top > this.height - my) this.ty -= top - (this.height - my);
    if (bottom < my) this.ty += my - bottom;
  }

  dispose(): void {
    cancelAnimationFrame(this.anim);
    this.listeners.clear();
  }
}
