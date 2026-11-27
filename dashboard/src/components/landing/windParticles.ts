/**
 * windParticles.ts - Landing-page wind particles advected through the model's real (u10, v10).
 *
 * Shared by the 2D story canvas and the 3D orbit scene. Each particle keeps a short history of
 * positions in fractional grid coords, so trails are redrawn from scratch every frame (no
 * accumulating trail canvas, hence no ghost haze over the domain).
 *
 * Edge feathering: particle density and opacity both fall smoothly to zero towards the edges of
 * the 32 x 36 domain, so the weather emerges over India instead of filling a visible rectangle.
 * Nothing is invented outside the domain: particles there are simply not drawn.
 */

import { GRID_CONFIG, type Grid2D } from '../../types/climate';
import { sampleBilinear } from '../../services/fieldRaster';

const NLAT = GRID_CONFIG.nLat;
const NLON = GRID_CONFIG.nLon;

/** 0 at the domain edge, rising smoothly to 1 at `cells` inside it (fractional grid coords). */
export function domainFeather(fi: number, fj: number, cells: number): number {
  const d = Math.min(fi + 0.5, NLAT - 0.5 - fi, fj + 0.5, NLON - 0.5 - fj);
  if (d <= 0) return 0;
  const t = Math.min(1, d / cells);
  return t * t * (3 - 2 * t);
}

/** Display-only taper of a grid towards zero at the edges (for fields whose alpha follows value). */
export function featherGrid(g: Grid2D, cells: number): Grid2D {
  return g.map((row, i) => row.map((v, j) => v * domainFeather(i, j, cells)));
}

export interface WindGrids {
  u: Grid2D;
  v: Grid2D;
}

export class WindParticles {
  readonly count: number;
  readonly trail: number;
  /** [count * trail * 2] (fi, fj) pairs; slot 0 of each particle is the newest point. */
  readonly hist: Float32Array;
  readonly len: Uint8Array;
  readonly speed: Float32Array;
  private readonly age: Float32Array;
  private readonly life: Float32Array;
  private readonly feather: number;

  constructor(count: number, trail: number, featherCells: number) {
    this.count = count;
    this.trail = trail;
    this.feather = featherCells;
    this.hist = new Float32Array(count * trail * 2);
    this.len = new Uint8Array(count);
    this.speed = new Float32Array(count);
    this.age = new Float32Array(count);
    this.life = new Float32Array(count);
    for (let k = 0; k < count; k++) this.spawn(k, true);
  }

  private spawn(k: number, randomAge: boolean): void {
    // Rejection sampling: fewer particles near the edges, so density tapers with the feather.
    let fi = 0;
    let fj = 0;
    for (let tries = 0; tries < 16; tries++) {
      fi = Math.random() * NLAT - 0.5;
      fj = Math.random() * NLON - 0.5;
      if (Math.random() < domainFeather(fi, fj, this.feather)) break;
    }
    const b = k * this.trail * 2;
    this.hist[b] = fi;
    this.hist[b + 1] = fj;
    this.len[k] = 1;
    this.life[k] = 1.4 + Math.random() * 1.8;
    this.age[k] = randomAge ? Math.random() * this.life[k] : 0;
  }

  /** Advance by dt seconds through the wind (m/s). degPerSecPerMs sets the on-screen speed. */
  step(dtIn: number, f: WindGrids, degPerSecPerMs = 0.27): void {
    const dt = Math.min(0.05, Math.max(0, dtIn));
    const T = this.trail;
    for (let k = 0; k < this.count; k++) {
      this.age[k] += dt;
      if (this.age[k] > this.life[k]) {
        this.spawn(k, false);
        continue;
      }
      const b = k * T * 2;
      const fi = this.hist[b];
      const fj = this.hist[b + 1];
      if (fi < -0.5 || fi > NLAT - 0.5 || fj < -0.5 || fj > NLON - 0.5) {
        this.spawn(k, false);
        continue;
      }
      const u = sampleBilinear(f.u, fi, fj);
      const v = sampleBilinear(f.v, fi, fj);
      const sp = Math.hypot(u, v);
      this.speed[k] = sp;
      if (sp < 0.3) {
        this.spawn(k, false);
        continue;
      }
      const lat = GRID_CONFIG.latMin + fi * GRID_CONFIG.resolutionDeg;
      const cosLat = Math.max(0.2, Math.cos((lat * Math.PI) / 180));
      this.hist.copyWithin(b + 2, b, b + (T - 1) * 2);
      this.hist[b] = fi + v * degPerSecPerMs * dt;
      this.hist[b + 1] = fj + (u * degPerSecPerMs * dt) / cosLat;
      if (this.len[k] < T) this.len[k] += 1;
    }
  }

  /** Opacity of particle k: fade in/out over its life x edge feather at its head. */
  alpha(k: number): number {
    const a = this.age[k] / this.life[k];
    const lifeFade = Math.max(0, Math.min(1, a * 5, (1 - a) * 4));
    const b = k * this.trail * 2;
    return lifeFade * domainFeather(this.hist[b], this.hist[b + 1], this.feather);
  }
}
