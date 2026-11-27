/**
 * StoryStage.tsx - The sticky canvas behind the landing story and Lab assembly.
 *
 * Scroll progress p (0 … STORY_STEPS) drives everything:
 *   step 0  WINDOW   - slowly turning orthographic globe; today's rain glows as dots
 *   step 1  TODAY    - globe unrolls into the Lambert conic chart; dots become a field
 *   step 2  NORMAL   - the normal 10-day forecast, time-lapsed by scroll
 *   step 3  SHOCK    - subtle, curvy Bay of Bengal warming (no harsh rectangular box)
 *   step 4  RIPPLE   - rain change (shock − normal) for Day 1 → 10 with measured spread rings
 *   step 5  MEANING  - final state, dimmed behind the summary
 *   step 6  CENTER   - all colored zones, rings, and overlays clear out; India glides to center
 *   step 7  LAYERS   - relief + channel/layer live demonstration (Rain ↔ Temp → Rain)
 *   step 8  WHATIF   - animated probe & subtle shock brush demonstration as left panel docks
 *   step 9  TIMELINE - Day 0 → Day 10 smooth rollout demonstration as bottom scrubber docks
 *   step 10 IMPACTS  - right panel docks & map settles into exact Lab geometry
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import worldGeo from '../../assets/geo/world110.json';
import coastGeo from '../../assets/geo/coast.json';
import { CHANNELS, GRID_CONFIG, LATS, LONS, type Grid2D } from '../../types/climate';
import { MapView, type Padding, type Projector } from '../../geo/lcc';
import { drawCoastline, drawDomainVeil, drawPlaceLabels } from '../../geo/basemap';
import { getProjectedRaster } from '../../services/projectedRaster';
import { anomalyRgba, channelRgba, type RgbaColor } from '../../services/colormaps';
import { buildHillshadeImage, contourSegments, fracToLat, fracToLon, gridDelta, strokeSegments } from '../../services/fieldRaster';
import { loadStaticFields, type StaticFields } from '../../services/staticFields';
import { shockOrigin, spreadByDay } from '../../services/diagnostics';
import type { GaussianShockSpec } from '../../types/shocks';
import type { StoryData } from './useStoryData';
import { WindParticles, domainFeather, featherGrid, type WindGrids } from './windParticles';
import { heroGlobeFrame, type GlobeFrame } from './globeFrame';

export const STORY_STEPS = 12;

interface Props {
  data: StoryData | null;
  progressRef: React.RefObject<number>;
  isDesktop: boolean;
  /** Receives the globe framing every frame (for the 3D hand-off). */
  frameOut?: React.RefObject<GlobeFrame | null>;
  /** While true the stage is fully covered: keep timing, skip drawing. */
  idleRef?: React.RefObject<boolean>;
  /** Skip the 2D globe + unroll and always draw the chart (the 3D scene and a cloud pass cover it). */
  mapOnly?: boolean;
}

// Monochrome palette transitioning into Lab instrument slate-navy (#03060C)
const C_DARK = '#171717';
const C_CHAR = '#262626';
const C_GRAPH = '#404040';
const OFFWHITE = '229, 229, 229';

const D2R = Math.PI / 180;
const WORLD: number[][] = (worldGeo as { rings: number[][] }).rings.filter((r) => {
  let maxLat = -90;
  for (let k = 1; k < r.length; k += 2) maxLat = Math.max(maxLat, r[k]);
  return maxLat > -55;
});
const COAST: number[][] = (coastGeo as { rings: number[][] }).rings;
const UNUSED_SPEC: GaussianShockSpec = { centerLat: 15, centerLon: 88, radiusKm: 450, sigmaKm: 300, channel: 't2m', magnitude: 0 };
const MAP_FEATHER = 3;
const GLOBE_FEATHER = 6;
const WIND_LEVELS = 6;
const RELIEF_S = 8;

/** Natural, organic Bay of Bengal basin perimeter (lat, lon) replacing rigid box edges for display. */
const BAY_CURVE: [number, number][] = [
  [20.2, 89.1],
  [19.7, 86.6],
  [17.8, 84.2],
  [15.2, 81.9],
  [12.3, 81.2],
  [9.6, 81.8],
  [8.4, 84.5],
  [8.3, 88.2],
  [9.3, 91.4],
  [11.6, 93.0],
  [14.7, 93.6],
  [17.8, 93.2],
  [19.6, 91.6],
];

const onClip = (lon: number, lat: number) =>
  Math.abs(lon - 62) < 0.02 || Math.abs(lon - 104) < 0.02 || Math.abs(lat - 3.5) < 0.02 || Math.abs(lat - 40.5) < 0.02;
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const smooth = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};
const easeOut = (t: number) => 1 - Math.pow(1 - clamp01(t), 3);
const reducedMotion = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

function lerpPad(a: Padding, b: Padding, t: number): Padding {
  return {
    top: lerp(a.top, b.top, t),
    right: lerp(a.right, b.right, t),
    bottom: lerp(a.bottom, b.bottom, t),
    left: lerp(a.left, b.left, t),
  };
}

function lerpHex(hexA: string, hexB: string, t: number): string {
  const k = clamp01(t);
  const pa = [parseInt(hexA.slice(1, 3), 16), parseInt(hexA.slice(3, 5), 16), parseInt(hexA.slice(5, 7), 16)];
  const pb = [parseInt(hexB.slice(1, 3), 16), parseInt(hexB.slice(3, 5), 16), parseInt(hexB.slice(5, 7), 16)];
  return `rgb(${Math.round(lerp(pa[0], pb[0], k))}, ${Math.round(lerp(pa[1], pb[1], k))}, ${Math.round(lerp(pa[2], pb[2], k))})`;
}

/** Smooth 3x3 separable blur passes so display shock glow has soft, organic edges. */
function smoothDisplayGrid(grid: Grid2D, passes: number): Grid2D {
  const nLat = grid.length;
  const nLon = grid[0]?.length ?? 0;
  let cur = grid.map((r) => [...r]);
  for (let pass = 0; pass < passes; pass++) {
    const tmp = cur.map((r) => [...r]);
    for (let i = 0; i < nLat; i++) {
      for (let j = 0; j < nLon; j++) {
        const l = cur[i][Math.max(0, j - 1)];
        const c = cur[i][j];
        const r = cur[i][Math.min(nLon - 1, j + 1)];
        tmp[i][j] = 0.25 * l + 0.5 * c + 0.25 * r;
      }
    }
    const next = tmp.map((r) => [...r]);
    for (let i = 0; i < nLat; i++) {
      for (let j = 0; j < nLon; j++) {
        const t = tmp[Math.max(0, i - 1)][j];
        const c = tmp[i][j];
        const b = tmp[Math.min(nLat - 1, i + 1)][j];
        next[i][j] = 0.25 * t + 0.5 * c + 0.25 * b;
      }
    }
    cur = next;
  }
  return cur;
}

/** Sculpted organic Bay of Bengal display envelope (soft elliptical + curvy modulation). */
function organicBayEnvelope(dT0: Grid2D): Grid2D {
  const smoothed = smoothDisplayGrid(dT0, 3);
  const cLat = 14.2;
  const cLon = 87.7;
  return smoothed.map((row, i) =>
    row.map((v, j) => {
      const lat = LATS[i];
      const lon = LONS[j];
      const dy = (lat - cLat) / 6.0;
      // Narrow slightly toward the northern head of the bay around 20N
      const northTaper = lat > 16 ? 1 - 0.26 * ((lat - 16) / 5) : 1;
      const dx = (lon - cLon) / (6.0 * northTaper);
      const ang = Math.atan2(dy, dx);
      const wave = 1 + 0.08 * Math.sin(ang * 3 + 0.7) + 0.05 * Math.cos(ang * 2 - 0.4);
      const r = Math.hypot(dx, dy) / wave;
      const env = 1 - smooth(0.58, 1.08, r);
      return v * env;
    })
  );
}

/** Soft, subtle warm amber-coral palette for the Bay warming preview (avoids harsh brick red). */
function subtleBayShockRgba(delta: number): RgbaColor {
  const n = Math.max(0, Math.min(1, delta / 3.0));
  const a = smooth(0.06, 0.42, n) * (0.14 + 0.30 * n);
  const r = Math.round(lerp(253, 234, n));
  const g = Math.round(lerp(186, 88, n));
  const b = Math.round(lerp(116, 40, n));
  return { r, g, b, a };
}

/** Trace a C1-smooth closed organic curve through projected screen points. */
function traceSmoothClosedCurve(ctx: CanvasRenderingContext2D, pts: [number, number][]): void {
  const n = pts.length;
  if (n < 3) return;
  const midX = (a: [number, number], b: [number, number]) => (a[0] + b[0]) * 0.5;
  const midY = (a: [number, number], b: [number, number]) => (a[1] + b[1]) * 0.5;
  ctx.beginPath();
  ctx.moveTo(midX(pts[n - 1], pts[0]), midY(pts[n - 1], pts[0]));
  for (let i = 0; i < n; i++) {
    const cur = pts[i];
    const nxt = pts[(i + 1) % n];
    ctx.quadraticCurveTo(cur[0], cur[1], midX(cur, nxt), midY(cur, nxt));
  }
  ctx.closePath();
}

function prepare(canvas: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D | null {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const bw = Math.round(w * dpr);
  const bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** Draw particle trails (newest point brightest), bucketed by opacity to keep stroke calls few. */
function drawWind(ctx: CanvasRenderingContext2D, wp: WindParticles, project: Projector, alpha: number): void {
  const buckets: number[][] = Array.from({ length: WIND_LEVELS }, () => []);
  const T = wp.trail;
  for (let k = 0; k < wp.count; k++) {
    const n = wp.len[k];
    if (n < 2) continue;
    const a0 = wp.alpha(k) * (0.35 + 0.65 * Math.min(1, wp.speed[k] / 12));
    if (a0 < 0.03) continue;
    const b = k * T * 2;
    let [px, py] = project(fracToLat(wp.hist[b]), fracToLon(wp.hist[b + 1]));
    for (let s = 1; s < n; s++) {
      const [x, y] = project(fracToLat(wp.hist[b + s * 2]), fracToLon(wp.hist[b + s * 2 + 1]));
      const lvl = Math.min(WIND_LEVELS - 1, Math.floor(a0 * (1 - s / n) * WIND_LEVELS));
      if (lvl >= 0) buckets[lvl].push(px, py, x, y);
      px = x;
      py = y;
    }
  }
  ctx.lineWidth = 1.2;
  ctx.lineCap = 'round';
  buckets.forEach((segs, lvl) => {
    if (segs.length === 0) return;
    ctx.strokeStyle = `rgba(${OFFWHITE}, ${(((lvl + 1) / WIND_LEVELS) * 0.75 * alpha).toFixed(3)})`;
    ctx.beginPath();
    for (let q = 0; q < segs.length; q += 4) {
      ctx.moveTo(segs[q], segs[q + 1]);
      ctx.lineTo(segs[q + 2], segs[q + 3]);
    }
    ctx.stroke();
  });
}

export const StoryStage: React.FC<Props> = ({ data, progressRef, isDesktop, frameOut, idleRef, mapOnly = false }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trailRef = useRef<HTMLCanvasElement>(null);
  const [staticFields, setStaticFields] = useState<StaticFields | null>(null);

  useEffect(() => {
    let alive = true;
    loadStaticFields().then((s) => {
      if (alive) setStaticFields(s);
    });
    return () => {
      alive = false;
    };
  }, []);

  // Precomputed rasters (re-projected once). Fields are tapered towards the domain edges.
  const assets = useMemo(() => {
    if (!data) return null;
    const raster = getProjectedRaster();
    const tp = CHANNELS.tp.index;
    const t2m = CHANNELS.t2m.index;
    const baseFrames = [data.day0, ...data.base];
    const baseImgs = baseFrames.map((f) => raster.paintField(featherGrid(f[tp], MAP_FEATHER), (v) => channelRgba(v, 'tp'), true));
    const t2mImg = raster.paintField(featherGrid(data.day0[t2m], MAP_FEATHER), (v) => channelRgba(v, 't2m'), true);
    const dT0 = gridDelta(data.day0Shocked[t2m], data.day0[t2m]);
    const shockImg = raster.paintField(featherGrid(organicBayEnvelope(dT0), MAP_FEATHER), subtleBayShockRgba, true);
    const dTp: Grid2D[] = data.shock.map((f, k) => gridDelta(f[tp], data.base[k][tp]));
    let maxAbs = 1;
    for (const g of dTp) for (const row of g) for (const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));
    const rippleImgs = dTp.map((g) => raster.paintField(featherGrid(g, MAP_FEATHER), (d) => anomalyRgba(d, maxAbs), true));
    const rainContour10 = contourSegments(data.day0[tp], 10);
    const origin = shockOrigin(data.preset, UNUSED_SPEC);
    const spread = spreadByDay(data.day0, data.day0Shocked, data.base, data.shock, origin);
    return { baseImgs, t2mImg, shockImg, rippleImgs, rainContour10, origin, spread, baseFrames };
  }, [data]);

  const reliefImage = useMemo(() => {
    if (!staticFields) return null;
    const grid = buildHillshadeImage(staticFields.orography, staticFields.landSea, RELIEF_S);
    return getProjectedRaster().warpGridImage(grid, RELIEF_S);
  }, [staticFields]);

  const assetsRef = useRef(assets);
  useEffect(() => {
    assetsRef.current = assets;
  }, [assets]);

  const reliefRef = useRef(reliefImage);
  useEffect(() => {
    reliefRef.current = reliefImage;
  }, [reliefImage]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const trail = trailRef.current;
    if (!canvas || !trail) return;
    const view = new MapView();
    const wind = new WindParticles(isDesktop ? 900 : 520, 16, GLOBE_FEATHER);
    const stars = Array.from({ length: 160 }, () => ({ x: Math.random(), y: Math.random(), r: Math.random() * 1.1 + 0.2, a: Math.random() * 0.6 + 0.2 }));
    const raster = getProjectedRaster();
    const reduced = reducedMotion();
    let w = 0;
    let h = 0;
    let raf = 0;
    let last = performance.now();
    const t0 = performance.now();

    const paddingForProgress = (p: number): Padding => {
      const storyPad: Padding = isDesktop
        ? { top: 92, right: 40, bottom: 56, left: Math.min(560, w * 0.4) }
        : { top: 64, right: 12, bottom: h * 0.42, left: 12 };
      if (p <= 5.75) return storyPad;

      const centerPad: Padding = isDesktop
        ? { top: 76, right: 40, bottom: 76, left: 40 }
        : { top: 64, right: 16, bottom: 64, left: 16 };
      const labPad: Padding = isDesktop
        ? { top: 174, right: 356, bottom: 108, left: 356 }
        : { top: 150, right: 12, bottom: 70, left: 12 };

      const cMove = smooth(5.75, 6.65, p);
      let pad = lerpPad(storyPad, centerPad, cMove);
      if (p > 7.0) {
        const topDemo = smooth(7.05, 7.6, p);
        const lDock = smooth(8.2, 8.9, p);
        const bDock = smooth(9.15, 9.75, p);
        const rDock = smooth(10.12, 10.82, p);
        const midTop = isDesktop ? 132 : 120;
        pad = {
          top: lerp(lerp(centerPad.top, midTop, topDemo), labPad.top, rDock),
          left: lerp(centerPad.left, labPad.left, lDock),
          bottom: lerp(centerPad.bottom, labPad.bottom, bDock),
          right: lerp(centerPad.right, labPad.right, rDock),
        };
      }
      return pad;
    };

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      w = r.width;
      h = r.height;
      view.resize(w, h, paddingForProgress(Math.max(0, progressRef.current ?? 0)), true);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const frame = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      const a = assetsRef.current;
      const relief = reliefRef.current;
      const p = Math.max(0, progressRef.current ?? 0);
      const step = Math.min(STORY_STEPS - 1, Math.floor(p));
      const t = clamp01(p - step);
      const m = mapOnly ? 1 : step === 0 ? 0 : step === 1 ? easeOut(t * 1.7) : 1;
      const fieldIn = smooth(0.55, 0.95, m);

      if (p > 5.75 && w >= 2 && h >= 2) {
        view.resize(w, h, paddingForProgress(p), true);
      }

      // Orthographic globe parameters
      const hero = heroGlobeFrame(w, h, isDesktop);
      const R0 = hero.R;
      const R = R0 * (1 + (step === 0 ? 0.18 * t : 0.18));
      const cx = hero.cx;
      const cy = hero.cy;
      const spin = reduced ? 0 : 1 - easeOut((now - t0) / 4200);
      const lamC = (82 + 70 * spin - (reduced ? 0 : 5 * Math.sin((now - t0) / 7000)) * (1 - m)) * D2R;
      const phiC = 17 * D2R;
      if (frameOut) frameOut.current = { cx, cy, R, lon: lamC / D2R, lat: 17 };

      if (idleRef?.current || w < 2) {
        raf = requestAnimationFrame(frame);
        return;
      }
      const ctx = prepare(canvas, w, h);
      const tctx = prepare(trail, w, h);
      if (!ctx || !tctx) {
        raf = requestAnimationFrame(frame);
        return;
      }
      const sinPC = Math.sin(phiC);
      const cosPC = Math.cos(phiC);

      const ortho = (lat: number, lon: number): [number, number, boolean] => {
        const phi = lat * D2R;
        const dl = lon * D2R - lamC;
        const cosPhi = Math.cos(phi);
        const x = R * cosPhi * Math.sin(dl);
        const y = R * (cosPC * Math.sin(phi) - sinPC * cosPhi * Math.cos(dl));
        const vis = sinPC * Math.sin(phi) + cosPC * cosPhi * Math.cos(dl) > 0;
        if (!vis) {
          const n = Math.hypot(x, y) || 1;
          return [cx + (x / n) * R, cy - (y / n) * R, false];
        }
        return [cx + x, cy - y, true];
      };
      const project: Projector = (lat, lon) => {
        const [ox, oy] = ortho(lat, lon);
        if (m <= 0) return [ox, oy];
        const [lx, ly] = view.project(lat, lon);
        return [ox + (lx - ox) * m, oy + (ly - oy) * m];
      };

      const labTheme = smooth(5.85, 6.75, p);
      const storyExit = 1 - smooth(5.6, 6.2, p);

      // --- Background: story charcoal smoothly deepening to Lab ocean slate
      ctx.clearRect(0, 0, w, h);
      const bg = ctx.createRadialGradient(
        lerp(cx, w * 0.5, labTheme),
        lerp(cy, h * 0.45, labTheme),
        0,
        lerp(cx, w * 0.5, labTheme),
        lerp(cy, h * 0.45, labTheme),
        Math.max(w, h) * lerp(1, 0.75, labTheme)
      );
      bg.addColorStop(0, lerpHex(C_CHAR, '#0A1424', labTheme));
      bg.addColorStop(0.55, lerpHex(C_DARK, '#03060C', labTheme));
      bg.addColorStop(1, lerpHex(C_DARK, '#03060C', labTheme));
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);
      if (m < 1) {
        for (const s of stars) {
          ctx.fillStyle = `rgba(${OFFWHITE}, ${s.a * (1 - m) * (0.6 + 0.4 * Math.sin(now / 900 + s.x * 40))})`;
          ctx.fillRect(s.x * w, s.y * h, s.r, s.r);
        }
      }

      // --- Globe body + rim light
      if (m < 1) {
        const k = 1 - m;
        ctx.save();
        ctx.globalAlpha = k;
        const glow = ctx.createRadialGradient(cx, cy, R * 0.96, cx, cy, R * 1.22);
        glow.addColorStop(0, `rgba(${OFFWHITE}, 0.22)`);
        glow.addColorStop(1, `rgba(${OFFWHITE}, 0)`);
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(cx, cy, R * 1.22, 0, Math.PI * 2);
        ctx.fill();
        const sea = ctx.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R);
        sea.addColorStop(0, C_CHAR);
        sea.addColorStop(1, C_DARK);
        ctx.fillStyle = sea;
        ctx.beginPath();
        ctx.arc(cx, cy, R, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // --- Graticule (globe when m < 0.5, subtle Lab graticule when centered)
      if (m < 0.5) {
        ctx.save();
        ctx.strokeStyle = `rgba(115, 115, 115, ${0.18 * (1 - m * 2)})`;
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        for (let lon = -180; lon < 180; lon += 15) {
          let pen = false;
          for (let lat = -80; lat <= 80; lat += 4) {
            const [x, y, vis] = ortho(lat, lon);
            if (vis && pen) ctx.lineTo(x, y);
            else if (vis) ctx.moveTo(x, y);
            pen = vis;
          }
        }
        for (let lat = -60; lat <= 75; lat += 15) {
          let pen = false;
          for (let lon = -180; lon <= 180; lon += 4) {
            const [x, y, vis] = ortho(lat, lon);
            if (vis && pen) ctx.lineTo(x, y);
            else if (vis) ctx.moveTo(x, y);
            pen = vis;
          }
        }
        ctx.stroke();
        ctx.restore();
      } else if (labTheme > 0.02) {
        ctx.save();
        ctx.strokeStyle = `rgba(148, 163, 184, ${(0.08 * labTheme).toFixed(3)})`;
        ctx.lineWidth = 0.75;
        ctx.beginPath();
        for (let lon = 60; lon <= 105; lon += 5) {
          const [ax, ay] = view.project(2, lon);
          const [bx, by] = view.project(42, lon);
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
        }
        for (let lat = 5; lat <= 40; lat += 5) {
          for (let lon = 58; lon <= 107; lon += 1) {
            const [x, y] = view.project(lat, lon);
            if (lon === 58) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
        ctx.restore();
      }

      // --- Land: world (globe) cross-fading to the 1:50m coast (map)
      const worldA = 1 - smooth(0.04, 0.3, m);
      const coastA = smooth(0.12, 0.5, m);
      const fillRings = (rings: number[][], alpha: number, fill: string | CanvasGradient, stroke: string | null, visibleOnly: boolean) => {
        if (alpha <= 0.01) return;
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        const pts: number[] = [];
        for (const r of rings) {
          let anyVis = m >= 0.5 && !visibleOnly;
          pts.length = 0;
          for (let k = 0; k < r.length; k += 2) {
            const lat = r[k + 1];
            const lon = r[k];
            const vis = ortho(lat, lon)[2];
            if (visibleOnly && !vis) continue;
            const [x, y] = project(lat, lon);
            if (vis) anyVis = true;
            pts.push(x, y);
          }
          if (!anyVis || pts.length < 6) continue;
          ctx.moveTo(pts[0], pts[1]);
          for (let k = 2; k < pts.length; k += 2) ctx.lineTo(pts[k], pts[k + 1]);
          ctx.closePath();
        }
        ctx.fillStyle = fill;
        ctx.fill();
        if (stroke) {
          ctx.strokeStyle = stroke;
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }
        ctx.restore();
      };
      if (m < 0.3) fillRings(WORLD, worldA, C_GRAPH, `rgba(${OFFWHITE}, 0.16)`, true);
      if (coastA > 0.01) {
        const landColor = lerpHex(C_CHAR, '#121B2A', labTheme);
        if (labTheme > 0.85) {
          fillRings(COAST, coastA, landColor, null, false);
        } else {
          const [ax, ay] = project(3.5, 62);
          const [bx, by] = project(40.5, 104);
          const [gx, gy] = project(22, 83);
          const rad = 0.5 * Math.min(Math.abs(bx - ax), Math.abs(by - ay));
          const land = ctx.createRadialGradient(gx, gy, 0, gx, gy, Math.max(1, rad));
          land.addColorStop(0, landColor);
          land.addColorStop(0.65, landColor);
          land.addColorStop(1, labTheme > 0.2 ? landColor : 'rgba(38, 38, 38, 0)');
          fillRings(COAST, coastA, land, null, false);
        }
      }

      // Subtle terrain hillshade fades in during Step 7 (Layers demo)
      const reliefAlpha = relief ? smooth(7.06, 7.5, p) : 0;
      if (relief && reliefAlpha > 0.01) {
        raster.draw(ctx, view, relief, reliefAlpha, true);
      }

      // --- Data fields
      if (a) {
        const tpIdx = CHANNELS.tp.index;
        if (fieldIn < 1) {
          const g = a.baseFrames[0][tpIdx];
          ctx.save();
          ctx.globalCompositeOperation = 'lighter';
          const sz = (m < 1 ? R / 160 : 1) * (1 + m * 2);
          for (let i = 0; i < GRID_CONFIG.nLat; i++) {
            for (let j = 0; j < GRID_CONFIG.nLon; j++) {
              const v = g[i][j];
              if (v < 5) continue;
              const fe = domainFeather(i, j, GLOBE_FEATHER);
              if (fe <= 0.01) continue;
              const [x, y] = project(GRID_CONFIG.latMin + i, GRID_CONFIG.lonMin + j);
              const k = Math.min(1, (v - 5) / 35);
              ctx.fillStyle = `rgba(${OFFWHITE}, ${((0.05 + 0.2 * k) * (1 - fieldIn) * fe).toFixed(3)})`;
              ctx.beginPath();
              ctx.arc(x, y, sz * (0.7 + 1.6 * k), 0, Math.PI * 2);
              ctx.fill();
            }
          }
          ctx.restore();
        }

        if (fieldIn > 0) {
          const draw = (img: HTMLCanvasElement | null, alpha: number, mult = 0.8) => {
            if (img && alpha > 0.01) raster.draw(ctx, view, img, alpha * fieldIn * mult, true);
          };

          if (p < 6.4) {
            if (step <= 1) {
              draw(a.baseImgs[0], 1);
            } else if (step === 2) {
              const d = t * 10;
              const i0 = Math.min(10, Math.floor(d));
              const f = d - i0;
              draw(a.baseImgs[i0], 1 - f);
              if (i0 < 10) draw(a.baseImgs[i0 + 1], f);
            } else if (step === 3) {
              draw(a.baseImgs[0], 0.32);
              draw(a.shockImg, easeOut(t * 2), 1);
            } else {
              const d = step === 4 ? t * 9 : 9;
              const i0 = Math.min(9, Math.floor(d));
              const f = d - i0;
              const dim = (step >= 5 ? 0.6 : 1) * storyExit;
              draw(a.rippleImgs[i0], (1 - f) * dim);
              if (i0 < 9) draw(a.rippleImgs[i0 + 1], f * dim);
            }
          } else if (p >= 7.05) {
            // Lab assembly live demonstrations:
            // p 7..8: Layers & Channel switch (tp -> t2m -> tp)
            const layerTp =
              smooth(7.12, 7.38, p) * (1 - smooth(7.42, 7.56, p)) +
              smooth(7.72, 7.9, p) * (1 - smooth(8.4, 8.65, p)) +
              smooth(10.38, 10.78, p);
            const layerT2m = smooth(7.42, 7.56, p) * (1 - smooth(7.72, 7.9, p));
            const demoShockWarm = smooth(8.28, 8.48, p) * (1 - smooth(8.62, 8.85, p));
            const demoAnomaly = smooth(8.5, 8.78, p) * (1 - smooth(10.35, 10.75, p));

            if (layerTp > 0.01) draw(a.baseImgs[0], layerTp, 0.92);
            if (layerT2m > 0.01) draw(a.t2mImg, layerT2m, 0.85);
            if (demoShockWarm > 0.01) draw(a.shockImg, demoShockWarm, 1);
            if (demoAnomaly > 0.01) {
              const demoDay = p < 9.05 ? smooth(8.5, 9.05, p) * 2 : 2 + smooth(9.08, 9.92, p) * 7;
              const i0 = Math.min(9, Math.floor(demoDay));
              const f = demoDay - i0;
              draw(a.rippleImgs[i0], (1 - f) * demoAnomaly, 0.9);
              if (i0 < 9) draw(a.rippleImgs[i0 + 1], f * demoAnomaly, 0.9);
            }
          }
        }
      }

      // Domain veil fades in softly as Lab settles into final alignment
      const veilAlpha = smooth(10.35, 10.85, p);
      if (veilAlpha > 0.01) {
        ctx.save();
        ctx.globalAlpha = veilAlpha;
        drawDomainVeil(ctx, view, w, h);
        ctx.restore();
      }

      // Subtle heavy-rain 10 mm/day contour during final Lab alignment & Layers demo
      const contourAlpha =
        a && p >= 7.75
          ? smooth(7.75, 8.0, p) * (1 - smooth(8.42, 8.65, p)) + smooth(10.45, 10.85, p)
          : 0;
      if (a && contourAlpha > 0.01) {
        ctx.save();
        ctx.globalAlpha = contourAlpha;
        strokeSegments(ctx, a.rainContour10, project, { color: '#22D3EE', width: 1.2, glow: 5 });
        ctx.restore();
      }

      // Coastline + labels on the map
      if (m > 0.6) {
        ctx.save();
        ctx.globalAlpha = smooth(0.6, 1, m);
        drawCoastline(ctx, view, lerp(0.5, 0.55, labTheme));
        ctx.restore();
        if (m > 0.9) {
          ctx.save();
          ctx.globalAlpha = smooth(0.9, 1, m) * lerp(0.85, 0.95, labTheme);
          drawPlaceLabels(ctx, view, 1);
          ctx.restore();
        }
      }

      // Curvy organic Bay of Bengal contour + spread rings (story shock/ripple + timeline demo)
      const storyContourWeight = step >= 3 && p < 6.25 ? storyExit : 0;
      const demoContourWeight = smooth(8.32, 8.56, p) * (1 - smooth(10.2, 10.55, p));
      const bayWeight = Math.max(storyContourWeight, demoContourWeight);
      if (a && a.origin && bayWeight > 0.01) {
        const pulse = reduced ? 0.5 : 0.5 + 0.5 * Math.sin(now / 420);
        const bayPts: [number, number][] = BAY_CURVE.map(([lat, lon], idx) => {
          const wave = reduced ? 0 : Math.sin(now / 1900 + idx * 1.15) * 0.16;
          return view.project(lat + wave, lon - wave * 0.7);
        });
        ctx.save();
        ctx.globalAlpha = bayWeight;
        traceSmoothClosedCurve(ctx, bayPts);
        ctx.fillStyle = `rgba(251, 146, 60, ${(0.04 + 0.02 * pulse).toFixed(3)})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(251, 146, 60, ${(0.30 + 0.18 * pulse).toFixed(3)})`;
        ctx.lineWidth = 1.15;
        ctx.setLineDash([4, 6]);
        ctx.lineDashOffset = reduced ? 0 : -now / 55;
        ctx.stroke();
        ctx.restore();

        // Spread ring during Ripple (step >= 4 in story) or Timeline demo (p 8.75..10.45)
        let ringAlpha = 0;
        let d = 10;
        if (step >= 4 && p < 6.25) {
          ringAlpha = storyExit;
          d = step === 4 ? 1 + t * 9 : 10;
        } else if (p >= 8.75 && p < 10.5) {
          ringAlpha = smooth(8.75, 9.05, p) * (1 - smooth(10.15, 10.45, p));
          d = p < 9.05 ? 1 : 1 + smooth(9.08, 9.92, p) * 9;
        }
        if (ringAlpha > 0.01) {
          const i0 = Math.min(10, Math.floor(d));
          const f = d - i0;
          const r0 = a.spread[i0]?.frontKm ?? 0;
          const r1 = a.spread[Math.min(10, i0 + 1)]?.frontKm ?? r0;
          const km = r0 + (r1 - r0) * f;
          if (km > 1) {
            const [ox, oy] = view.project(a.origin.lat, a.origin.lon);
            const rp = km * view.pxPerKm();
            ctx.save();
            ctx.globalAlpha = ringAlpha;
            ctx.strokeStyle = '#FBBF24';
            ctx.shadowColor = 'rgba(251, 191, 36, 0.5)';
            ctx.shadowBlur = 7;
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            ctx.arc(ox, oy, rp, 0, Math.PI * 2);
            ctx.stroke();
            if (!reduced) {
              for (let q = 0; q < 2; q++) {
                const ph = (((now - t0) / 2400 + q * 0.5) % 1 + 1) % 1;
                ctx.globalAlpha = ringAlpha * 0.4 * (1 - ph);
                ctx.lineWidth = 1.3;
                ctx.beginPath();
                ctx.arc(ox, oy, easeOut(ph) * rp, 0, Math.PI * 2);
                ctx.stroke();
              }
            }
            ctx.restore();
          }
        }
      }

      // Animated probe crosshair & Gaussian brush preview during Step 8 (What-if demo)
      const probeAlpha = smooth(8.05, 8.22, p) * (1 - smooth(8.62, 8.85, p));
      if (probeAlpha > 0.01) {
        const moveT = smooth(8.08, 8.42, p);
        const pLat = lerp(21.5, 14.5, moveT);
        const pLon = lerp(74.5, 87.8, moveT);
        const [px, py] = view.project(pLat, pLon);
        const brushR = 320 * view.pxPerKm();
        ctx.save();
        ctx.globalAlpha = probeAlpha;
        ctx.strokeStyle = 'rgba(251, 146, 60, 0.78)';
        ctx.fillStyle = 'rgba(251, 146, 60, 0.12)';
        ctx.lineWidth = 1.4;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.arc(px, py, brushR, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.moveTo(px, py - 14);
        ctx.lineTo(px, py - 9);
        ctx.moveTo(px, py + 9);
        ctx.lineTo(px, py + 14);
        ctx.moveTo(px - 14, py);
        ctx.lineTo(px - 9, py);
        ctx.moveTo(px + 9, py);
        ctx.lineTo(px + 14, py);
        ctx.stroke();
        // Soft ripple pulse once pointer reaches the Bay
        if (p >= 8.38) {
          const rp = clamp01((p - 8.38) / 0.32);
          ctx.globalAlpha = probeAlpha * (1 - rp) * 0.8;
          ctx.strokeStyle = '#FB923C';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(px, py, 10 + rp * brushR * 1.15, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
      }

      // Wind particles: trails rebuilt from particle history on a cleared canvas every frame
      tctx.clearRect(0, 0, w, h);
      if (a && data && !reduced) {
        const u = CHANNELS.u10.index;
        const v = CHANNELS.v10.index;
        let f = a.baseFrames[0];
        let windAlpha = 1;
        if (step === 2) {
          f = a.baseFrames[Math.min(10, Math.round(t * 10))];
        } else if (step >= 3 && p < 6.1) {
          f = step === 3 ? data.day0Shocked : data.shock[Math.min(9, Math.round(step === 4 ? t * 9 : 9))];
        } else if (p >= 6.1 && p < 7.2) {
          f = a.baseFrames[0];
          windAlpha = 0.45;
        } else if (p >= 8.45 && p < 10.45) {
          const demoDay = p < 9.05 ? 1 : Math.min(9, Math.round(1 + smooth(9.08, 9.92, p) * 8));
          f = data.shock[demoDay];
        } else {
          f = a.baseFrames[0];
        }
        const grids: WindGrids = { u: f[u], v: f[v] };
        wind.step(dt, grids);
        drawWind(tctx, wind, project, windAlpha);
      }

      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      view.dispose();
    };
  }, [isDesktop, progressRef, data, frameOut, idleRef, mapOnly]);

  return (
    <div className="absolute inset-0">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      <canvas ref={trailRef} className="absolute inset-0 w-full h-full pointer-events-none" style={{ width: '100%', height: '100%' }} />
    </div>
  );
};
