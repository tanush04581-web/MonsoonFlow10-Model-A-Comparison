/**
 * windFlow.ts - Particle advection through the live (u10, v10) field, plus static arrow glyphs.
 *
 * Particles live in fractional grid coords (fi, fj) so they stay geo-referenced while the map
 * pans. Velocity is bilinearly interpolated from the displayed frame and converted from m/s
 * to deg/frame with the cos(lat) metric, so direction and relative speed are physical.
 */

import type { Grid2D } from '../types/climate';
import { GRID_CONFIG } from '../types/climate';
import { edgeFade, fracToLat, fracToLon, sampleBilinear, type Projector } from './fieldRaster';
import { getColorForChannel, rgbToCss } from './colormaps';

export interface WindField {
  u: Grid2D;
  v: Grid2D;
  /** Multiplies m/s before advection (1 for real winds; > 1 to make small Δwind visible). */
  speedScale: number;
  /** Particles in weaker raw wind (m/s, before scaling) are not drawn. */
  minSpeed: number;
}

interface Particle {
  fi: number;
  fj: number;
  age: number;
  life: number;
  /** Last in-domain velocity (m/s, already scaled): carried unchanged once the particle leaves. */
  u: number;
  v: number;
}

const NLAT = GRID_CONFIG.nLat;
const NLON = GRID_CONFIG.nLon;
/** deg per frame for 1 m/s (10 m/s ~ 0.045 deg/frame ~ 2.7 deg/s at 60 fps) */
const DEG_PER_MS_FRAME = 0.0045;
const SPEED_BUCKETS = [0, 3, 6, 9, 12, 15];
/** Opacity levels for particles drifting out past the domain edge */
const FADE_LEVELS = [1, 0.6, 0.3];

function spawn(p: Particle, randomAge: boolean): void {
  p.fi = Math.random() * NLAT - 0.5;
  p.fj = Math.random() * NLON - 0.5;
  p.life = 50 + Math.random() * 60;
  p.age = randomAge ? Math.random() * p.life : 0;
}

export class WindFlow {
  private particles: Particle[] = [];

  constructor(count: number) {
    for (let k = 0; k < count; k++) {
      const p: Particle = { fi: 0, fj: 0, age: 0, life: 1, u: 0, v: 0 };
      spawn(p, true);
      this.particles.push(p);
    }
  }

  /**
   * Advance one frame and draw the new trail segments.
   * fieldAt(x) picks the wind field for a screen x (lets SPLIT view use baseline/shock per side).
   */
  step(
    ctx: CanvasRenderingContext2D,
    project: Projector,
    fieldAt: (screenX: number) => WindField,
    colourBySpeed: boolean
  ): void {
    // buckets[fadeLevel][speedBucket] -> flat list of x0, y0, x1, y1
    const buckets: number[][][] = FADE_LEVELS.map(() => SPEED_BUCKETS.map(() => []));
    for (const p of this.particles) {
      p.age += 1;
      if (p.age > p.life) {
        spawn(p, false);
        continue;
      }
      const lat = fracToLat(p.fi);
      const lon = fracToLon(p.fj);
      const [x0, y0] = project(lat, lon);
      const inside = p.fi >= -0.5 && p.fi <= NLAT - 0.5 && p.fj >= -0.5 && p.fj <= NLON - 0.5;

      if (inside) {
        // Live model wind at this point
        const f = fieldAt(x0);
        const rawU = sampleBilinear(f.u, p.fi, p.fj);
        const rawV = sampleBilinear(f.v, p.fi, p.fj);
        if (Math.hypot(rawU, rawV) < f.minSpeed) {
          spawn(p, false);
          continue;
        }
        p.u = rawU * f.speedScale;
        p.v = rawV * f.speedScale;
      }
      // Outside the domain there is no data: the particle keeps its last in-domain velocity
      // and fades out over EDGE_FADE_CELLS (no wind is sampled or invented there).
      const speed = Math.hypot(p.u, p.v);
      if (speed < 0.05) {
        spawn(p, false);
        continue;
      }
      const cosLat = Math.max(0.2, Math.cos((lat * Math.PI) / 180));
      p.fi += p.v * DEG_PER_MS_FRAME;
      p.fj += (p.u * DEG_PER_MS_FRAME) / cosLat;

      const fade = edgeFade(p.fi, p.fj);
      if (fade <= 0.05) {
        spawn(p, false);
        continue;
      }
      const level = fade > 0.85 ? 0 : fade > 0.45 ? 1 : 2;
      const [x1, y1] = project(fracToLat(p.fi), fracToLon(p.fj));
      let b = SPEED_BUCKETS.length - 1;
      while (b > 0 && speed < SPEED_BUCKETS[b]) b--;
      buckets[level][b].push(x0, y0, x1, y1);
    }

    ctx.lineWidth = 1.25;
    ctx.lineCap = 'round';
    buckets.forEach((bySpeed, level) => {
      bySpeed.forEach((segs, b) => {
        if (segs.length === 0) return;
        const mid = SPEED_BUCKETS[b] + 1.5;
        const fadeAlpha = FADE_LEVELS[level];
        ctx.strokeStyle = colourBySpeed
          ? rgbToCss(getColorForChannel(mid, 'u10'), 0.75 * fadeAlpha)
          : `rgba(226, 232, 240, ${(0.25 + 0.07 * b) * fadeAlpha})`;
        ctx.beginPath();
        for (let k = 0; k < segs.length; k += 4) {
          ctx.moveTo(segs[k], segs[k + 1]);
          ctx.lineTo(segs[k + 2], segs[k + 3]);
        }
        ctx.stroke();
      });
    });
  }
}

/** Static arrows every `every` cells (used for ARROWS mode and prefers-reduced-motion). */
export function drawWindArrows(
  ctx: CanvasRenderingContext2D,
  project: Projector,
  fieldAt: (screenX: number) => WindField,
  colourBySpeed: boolean,
  every = 2
): void {
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (let i = 0; i < NLAT; i += every) {
    for (let j = 0; j < NLON; j += every) {
      const lat = fracToLat(i);
      const lon = fracToLon(j);
      const [x, y] = project(lat, lon);
      const f = fieldAt(x);
      if (Math.hypot(f.u[i][j], f.v[i][j]) < f.minSpeed) continue;
      const u = f.u[i][j] * f.speedScale;
      const v = f.v[i][j] * f.speedScale;
      const speed = Math.hypot(u, v);
      if (speed < 0.3) continue;
      // Screen direction from projecting a short step along (u, v): correct for any projection
      // (meridians converge in Lambert conic, so "north" is not straight up everywhere).
      const eps = 0.05;
      const [px, py] = project(lat + (v / speed) * eps, lon + ((u / speed) * eps) / Math.cos((lat * Math.PI) / 180));
      const sl = Math.hypot(px - x, py - y) || 1;
      const len = Math.min(22, 5 + speed * 1.2);
      const dx = ((px - x) / sl) * len;
      const dy = ((py - y) / sl) * len;
      const tx = x + dx / 2;
      const ty = y + dy / 2;
      const hx = x - dx / 2;
      const hy = y - dy / 2;
      const ang = Math.atan2(dy, dx);
      ctx.strokeStyle = colourBySpeed
        ? rgbToCss(getColorForChannel(speed, 'u10'), 0.95)
        : 'rgba(226, 232, 240, 0.75)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(tx, ty);
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - 5 * Math.cos(ang - 0.45), ty - 5 * Math.sin(ang - 0.45));
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - 5 * Math.cos(ang + 0.45), ty - 5 * Math.sin(ang + 0.45));
      ctx.stroke();
    }
  }
}
