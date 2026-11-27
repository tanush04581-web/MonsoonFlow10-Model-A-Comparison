/**
 * IndiaClimateMap.tsx - Lambert Conformal Conic weather map for the 32x36 India domain.
 *
 * No tiles and no Leaflet: a custom pan/zoom view over LCC space (geo/lcc.ts) drives five canvases
 *   base  : offline vector basemap (land, curved graticule) + terrain relief from static.npy
 *   field : the displayed tensor, re-projected once per frame change, cross-faded between days
 *   fx    : domain veil, isolines, coastline, place labels, H/L pressure centres, shock footprint,
 *           day-by-day spread rings, pinned cell, split divider
 *   wind  : particle trails advected by the displayed (u10, v10)
 *   sfx   : rain streak motion inside heavy-rain areas + sonar pulse from the shock origin
 * and a DOM "probe" cursor that reads values under the pointer (or previews the brush footprint).
 * Every pixel, ring and symbol is computed from the live tensors; nothing is decorative data.
 */

import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { CHANNELS, GRID_CONFIG, type ChannelId, type ChannelTensor, type Grid2D } from '../../types/climate';
import { useClimateData } from '../../context/ClimateDataContext';
import { useTimeline } from '../../context/TimelineContext';
import { useShockLab } from '../../context/ShockLabContext';
import { anomalyRgba, channelRgba, isWindChannel } from '../../services/colormaps';
import {
  buildHillshadeImage,
  contourSegments,
  gridDelta,
  sampleBilinear,
  strokeSegments,
  type ContourStyle,
  type Segment,
} from '../../services/fieldRaster';
import { getProjectedRaster } from '../../services/projectedRaster';
import { loadStaticFields, type StaticFields } from '../../services/staticFields';
import { WindFlow, drawWindArrows, type WindField } from '../../services/windFlow';
import { displayDeltaString, latLonToGrid, wireToDisplay } from '../../services/physicalUnits';
import { resolvePerturbation } from '../../services/perturbation';
import { buildHeadline, pressureCentres, shockOrigin, spreadByDay, type PressureCentre } from '../../services/diagnostics';
import { MapView, type Padding, type Projector } from '../../geo/lcc';
import { drawBasemap, drawCoastline, drawDomainVeil, drawPlaceLabels, regionName } from '../../geo/basemap';
import { CHANNEL_HUMAN } from '../ui/labels';
import { ColorbarLegend } from './ColorbarLegend';
import { ChannelSelector } from './ChannelSelector';
import { NarrativeBanner } from './NarrativeBanner';
import {
  AlertTriangle,
  Box,
  CloudRain,
  Grid3x3,
  Layers,
  Locate,
  MapPin,
  Minus,
  Mountain,
  Plus,
  Radar,
  RefreshCw,
  Spline,
  Tag,
  Waves,
  Wind,
  X,
} from 'lucide-react';

type WindMode = 'FLOW' | 'ARROWS' | 'OFF';
type Side = 'all' | 'left' | 'right';

interface ContourLayer {
  segs: Segment[];
  style: ContourStyle;
  side: Side;
}

interface PinnedCell {
  lat: number;
  lon: number;
  gridI: number;
  gridJ: number;
}

interface FieldImages {
  left: HTMLCanvasElement | null;
  right: HTMLCanvasElement | null;
}

interface Ripple {
  id: number;
  x: number;
  y: number;
  color: string;
}

interface Props {
  /** Free space reserved for the floating side panels on desktop (px). */
  insetLeft: number;
  insetRight: number;
  /** Skip the zoomed-out fly-in when mounted seamlessly from the landing assembly. */
  skipIntroFlyIn?: boolean;
}

const NLAT = GRID_CONFIG.nLat;
const NLON = GRID_CONFIG.nLon;
const HALF = GRID_CONFIG.resolutionDeg / 2;
const TILT_DEG = 38;
const TILT_PERSPECTIVE = 1400;
const STAGE_TRANSFORM = `perspective(${TILT_PERSPECTIVE}px) rotateX(${TILT_DEG}deg)`;
const RELIEF_S = 8;
const CROSSFADE_MS = 420;
const RING_MS = 750;
const PULSE_PERIOD_MS = 2600;

const prefersReducedMotion = (): boolean =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);

/** Size a canvas to CSS px with a devicePixelRatio backing store; returns a CSS-px context. */
function prepareCanvas(canvas: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  const bw = Math.round(w * dpr);
  const bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function clipSide(ctx: CanvasRenderingContext2D, side: Side, w: number, h: number): void {
  if (side === 'all') return;
  ctx.beginPath();
  if (side === 'left') ctx.rect(0, 0, w / 2, h);
  else ctx.rect(w / 2, 0, w / 2, h);
  ctx.clip();
}

/** A lat/lon rectangle, traced with sampled edges so it bends with the conic grid. */
function traceLatLonRect(ctx: CanvasRenderingContext2D, project: Projector, la0: number, la1: number, lo0: number, lo1: number): void {
  const pts: [number, number][] = [];
  const n = 16;
  for (let s = 0; s <= n; s++) pts.push(project(la0, lo0 + ((lo1 - lo0) * s) / n));
  for (let s = 1; s <= n; s++) pts.push(project(la0 + ((la1 - la0) * s) / n, lo1));
  for (let s = 1; s <= n; s++) pts.push(project(la1, lo1 - ((lo1 - lo0) * s) / n));
  for (let s = 1; s < n; s++) pts.push(project(la1 - ((la1 - la0) * s) / n, lo0));
  pts.forEach(([x, y], k) => (k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  ctx.closePath();
}

/** Isolines + glowing extreme cores for one displayed grid (physical thresholds). */
function contourLayersFor(grid: Grid2D, channel: ChannelId, side: Side): ContourLayer[] {
  if (channel === 'tp') {
    return [
      { segs: contourSegments(grid, 10), style: { color: '#22D3EE', width: 1.3, glow: 6 }, side },
      { segs: contourSegments(grid, 50), style: { color: '#FF3355', width: 2.0, glow: 12 }, side },
    ];
  }
  if (channel === 'msl') {
    const layers: ContourLayer[] = [];
    for (let p = 98400; p <= 102400; p += 200) {
      const deepLow = p <= 100000;
      layers.push({
        segs: contourSegments(grid, p),
        style: deepLow ? { color: '#A5B4FC', width: 1.2, glow: 6 } : { color: 'rgba(226, 232, 240, 0.30)', width: 0.8, glow: 0 },
        side,
      });
    }
    return layers;
  }
  return [];
}

/** Value of a grid at (lat, lon), bilinear, for the probe readout. */
function probeValue(grid: Grid2D, lat: number, lon: number): number {
  return sampleBilinear(grid, (lat - GRID_CONFIG.latMin) / GRID_CONFIG.resolutionDeg, (lon - GRID_CONFIG.lonMin) / GRID_CONFIG.resolutionDeg);
}

const insideDomain = (lat: number, lon: number) =>
  lat >= GRID_CONFIG.latMin - HALF && lat <= GRID_CONFIG.latMax + HALF && lon >= GRID_CONFIG.lonMin - HALF && lon <= GRID_CONFIG.lonMax + HALF;

/** Deltas smaller than this are not shown in the probe (wire units). */
const PROBE_DELTA_MIN: Record<ChannelId, number> = { t2m: 0.05, tp: 0.2, u10: 0.1, v10: 0.1, msl: 10, tcwv: 0.2 };

const LayerToggle: React.FC<{
  on: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  hint: string;
  disabled?: boolean;
}> = ({ on, onClick, icon, label, hint, disabled }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    title={hint}
    role="switch"
    aria-checked={on}
    className={`mf-press w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-left text-[12.5px] transition-colors disabled:opacity-40 ${
      on ? 'bg-white/[0.08] text-white' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04]'
    }`}
  >
    <span className={on ? 'text-sky-300' : 'text-zinc-500'}>{icon}</span>
    <span className="flex-1">{label}</span>
    <span className={`w-7 h-4 rounded-full relative transition-colors ${on ? 'bg-sky-500/70' : 'bg-white/10'}`}>
      <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${on ? 'left-3.5' : 'left-0.5'}`} />
    </span>
  </button>
);

export const IndiaClimateMap: React.FC<Props> = ({ insetLeft, insetRight, skipIntroFlyIn = false }) => {
  const outerRef = useRef<HTMLDivElement>(null);
  const surfaceRef = useRef<HTMLDivElement>(null);
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const fieldCanvasRef = useRef<HTMLCanvasElement>(null);
  const fxCanvasRef = useRef<HTMLCanvasElement>(null);
  const windCanvasRef = useRef<HTMLCanvasElement>(null);
  const sfxCanvasRef = useRef<HTMLCanvasElement>(null);
  const probeRef = useRef<HTMLDivElement>(null);
  const probeTextRef = useRef<HTMLDivElement>(null);
  const brushRingRef = useRef<HTMLDivElement>(null);
  const [view] = useState(() => new MapView());
  const windFlowRef = useRef<WindFlow | null>(null);
  const rafRef = useRef(0);
  const drawAllRef = useRef<(now: number) => boolean>(() => false);
  const fittedRef = useRef(false);

  const {
    baselineRollout,
    shockRollout,
    initialCondition,
    selectedChannel,
    viewMode,
    dataMode,
    setDataMode,
    apiStatus,
    executeSimulation,
  } = useClimateData();
  const { currentDay } = useTimeline();
  const { customSpec, updateCustomSpec, isBrushMode, activePreset } = useShockLab();

  const [pinnedCell, setPinnedCell] = useState<PinnedCell | null>(null);
  const [showGridLines, setShowGridLines] = useState(false);
  const [smooth, setSmooth] = useState(true);
  const [showRelief, setShowRelief] = useState(true);
  const [showContours, setShowContours] = useState(true);
  const [showSymbols, setShowSymbols] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [rainMotion, setRainMotion] = useState(true);
  const [windMode, setWindMode] = useState<WindMode>('FLOW');
  const [tilt, setTilt] = useState(false);
  const [layersOpen, setLayersOpen] = useState(false);
  const [staticFields, setStaticFields] = useState<StaticFields | null>(null);
  const [ripples, setRipples] = useState<Ripple[]>([]);
  const [isDesktop, setIsDesktop] = useState<boolean>(() =>
    typeof window !== 'undefined' ? window.matchMedia('(min-width: 1024px)').matches : true
  );

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 1024px)');
    const onChange = () => setIsDesktop(mq.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const chIdx = CHANNELS[selectedChannel].index;
  const isAnomaly = viewMode === 'ANOMALY_DIFF';

  const hasForecastForDay =
    currentDay === 0 || (baselineRollout.length >= currentDay && shockRollout.length >= currentDay);
  const liveForecastUnavailable =
    dataMode === 'LIVE' && (Boolean(apiStatus.errorMessage) || (currentDay > 0 && !hasForecastForDay));
  const noData = liveForecastUnavailable && currentDay > 0;

  // ---- Frames for the current lead day --------------------------------------------------------
  const baseFrame: ChannelTensor = useMemo(
    () => (currentDay === 0 ? initialCondition : baselineRollout[currentDay - 1] ?? initialCondition),
    [currentDay, baselineRollout, initialCondition]
  );
  const shockFrame: ChannelTensor = useMemo(
    () => (currentDay === 0 ? initialCondition : shockRollout[currentDay - 1] ?? initialCondition),
    [currentDay, shockRollout, initialCondition]
  );
  const currentBaselineGrid = baseFrame[chIdx];
  const currentShockGrid = shockFrame[chIdx];
  const deltaGrid = useMemo(() => gridDelta(currentShockGrid, currentBaselineGrid), [currentShockGrid, currentBaselineGrid]);
  const rawMaxAbsDelta = useMemo(() => {
    let m = 0;
    for (const row of deltaGrid) for (const d of row) m = Math.max(m, Math.abs(d));
    return m;
  }, [deltaGrid]);
  const maxAbsDelta = Math.max(1.0, rawMaxAbsDelta);

  // ---- Shock origin, spread rings, headline -----------------------------------------------------
  const origin = useMemo(() => shockOrigin(activePreset, customSpec), [activePreset, customSpec]);
  const day0Shock = useMemo(() => {
    const p = resolvePerturbation(activePreset, customSpec, initialCondition);
    if (!p) return null;
    return initialCondition.map((g, c) => g.map((row, i) => row.map((v, j) => v + p[c][i][j])));
  }, [activePreset, customSpec, initialCondition]);
  const spread = useMemo(
    () => spreadByDay(initialCondition, day0Shock, baselineRollout, shockRollout, origin),
    [initialCondition, day0Shock, baselineRollout, shockRollout, origin]
  );
  const showingShock = viewMode !== 'BASELINE';
  const headline = useMemo(
    () =>
      buildHeadline({
        day: currentDay,
        base: baseFrame,
        shock: shockFrame,
        origin,
        presetName: activePreset.name,
        spread: spread[currentDay] ?? null,
        showingShock,
      }),
    [currentDay, baseFrame, shockFrame, origin, activePreset.name, spread, showingShock]
  );
  const ringTargetKm =
    origin && showingShock && !noData ? (currentDay === 0 ? origin.radiusKm : spread[currentDay]?.frontKm ?? 0) : 0;

  // ---- Static terrain -------------------------------------------------------------------------
  useEffect(() => {
    let alive = true;
    loadStaticFields().then((s) => {
      if (alive) setStaticFields(s);
    });
    return () => {
      alive = false;
    };
  }, []);

  const reliefImage = useMemo(() => {
    if (!staticFields) return null;
    const grid = buildHillshadeImage(staticFields.orography, staticFields.landSea, RELIEF_S);
    return getProjectedRaster().warpGridImage(grid, RELIEF_S);
  }, [staticFields]);

  // ---- Field imagery (re-projected) -------------------------------------------------------------
  const fieldImages = useMemo<FieldImages | null>(() => {
    if (noData) return null;
    const raster = getProjectedRaster();
    const colour = (v: number) => channelRgba(v, selectedChannel);
    if (isAnomaly) return { left: raster.paintField(deltaGrid, (d) => anomalyRgba(d, maxAbsDelta), smooth), right: null };
    if (viewMode === 'BASELINE') return { left: raster.paintField(currentBaselineGrid, colour, smooth), right: null };
    if (viewMode === 'SPLIT') {
      return { left: raster.paintField(currentBaselineGrid, colour, smooth), right: raster.paintField(currentShockGrid, colour, smooth) };
    }
    return { left: raster.paintField(currentShockGrid, colour, smooth), right: null };
  }, [noData, selectedChannel, isAnomaly, deltaGrid, maxAbsDelta, smooth, viewMode, currentBaselineGrid, currentShockGrid]);

  // ---- Isolines ---------------------------------------------------------------------------------
  const contourLayers = useMemo<ContourLayer[]>(() => {
    if (!showContours || noData) return [];
    if (isAnomaly) {
      if (rawMaxAbsDelta <= 1e-6) return [];
      const core = 0.5 * rawMaxAbsDelta;
      return [
        { segs: contourSegments(deltaGrid, core), style: { color: '#FB923C', width: 1.6, glow: 10 }, side: 'all' },
        { segs: contourSegments(deltaGrid.map((r) => r.map((d) => -d)), core), style: { color: '#38BDF8', width: 1.6, glow: 10 }, side: 'all' },
      ];
    }
    if (viewMode === 'SPLIT') {
      return [...contourLayersFor(currentBaselineGrid, selectedChannel, 'left'), ...contourLayersFor(currentShockGrid, selectedChannel, 'right')];
    }
    return contourLayersFor(viewMode === 'BASELINE' ? currentBaselineGrid : currentShockGrid, selectedChannel, 'all');
  }, [showContours, noData, isAnomaly, rawMaxAbsDelta, deltaGrid, viewMode, currentBaselineGrid, currentShockGrid, selectedChannel]);

  // ---- H / L pressure centres -------------------------------------------------------------------
  const centres = useMemo<{ list: PressureCentre[]; side: Side }[]>(() => {
    if (!showSymbols || noData || isAnomaly) return [];
    const m = CHANNELS.msl.index;
    if (viewMode === 'SPLIT') {
      return [
        { list: pressureCentres(baseFrame[m]), side: 'left' },
        { list: pressureCentres(shockFrame[m]), side: 'right' },
      ];
    }
    return [{ list: pressureCentres((viewMode === 'BASELINE' ? baseFrame : shockFrame)[m]), side: 'all' }];
  }, [showSymbols, noData, isAnomaly, viewMode, baseFrame, shockFrame]);

  // ---- Wind fields ----------------------------------------------------------------------------------
  const windFields = useMemo(() => {
    const u = CHANNELS.u10.index;
    const v = CHANNELS.v10.index;
    const base: WindField = { u: baseFrame[u], v: baseFrame[v], speedScale: 1, minSpeed: 0 };
    const shock: WindField = { u: shockFrame[u], v: shockFrame[v], speedScale: 1, minSpeed: 0 };
    const du = gridDelta(shockFrame[u], baseFrame[u]);
    const dv = gridDelta(shockFrame[v], baseFrame[v]);
    let peak = 0;
    for (let i = 0; i < du.length; i++) for (let j = 0; j < du[i].length; j++) peak = Math.max(peak, Math.hypot(du[i][j], dv[i][j]));
    const deltaScale = peak > 1e-3 ? Math.min(10, 8 / peak) : 0;
    const delta: WindField = { u: du, v: dv, speedScale: deltaScale, minSpeed: 0.15 * peak };
    return { base, shock, delta, deltaPeak: peak, deltaScale };
  }, [baseFrame, shockFrame]);

  // ---- Latest scene for the imperative renderer -------------------------------------------------
  const rainGrid =
    selectedChannel === 'tp' && !isAnomaly && !noData ? (viewMode === 'BASELINE' ? baseFrame : shockFrame)[CHANNELS.tp.index] : null;
  const scene = {
    fieldImages,
    reliefImage,
    showRelief,
    smooth,
    contourLayers,
    centres,
    showGridLines,
    showLabels,
    showSymbols,
    viewMode,
    pinnedCell,
    activePreset,
    customSpec,
    isBrushMode,
    origin,
    spread,
    currentDay,
    ringTargetKm,
    rainGrid,
    rainMotion,
  };
  const sceneRef = useRef(scene);
  const fadeRef = useRef<{ prev: FieldImages | null; start: number }>({ prev: null, start: 0 });
  const lastImagesRef = useRef<FieldImages | null>(null);
  const ringRef = useRef({ from: 0, to: 0, start: 0, shown: 0 });

  const requestDraw = useCallback(() => {
    const schedule = () => {
      if (rafRef.current) return;
      rafRef.current = requestAnimationFrame((now) => {
        rafRef.current = 0;
        if (drawAllRef.current(now)) schedule();
      });
    };
    schedule();
  }, []);

  // Field cross-fade between days / views
  useEffect(() => {
    if (lastImagesRef.current && fieldImages && lastImagesRef.current !== fieldImages && !prefersReducedMotion()) {
      fadeRef.current = { prev: lastImagesRef.current, start: performance.now() };
    }
    lastImagesRef.current = fieldImages;
  }, [fieldImages]);

  // Spread ring grows from its last radius to the new day's reach
  useEffect(() => {
    const r = ringRef.current;
    if (r.to === ringTargetKm) return;
    ringRef.current = { from: r.shown, to: ringTargetKm, start: performance.now(), shown: r.shown };
  }, [ringTargetKm]);

  useLayoutEffect(() => {
    sceneRef.current = scene;
    requestDraw();
  });

  // ---- Imperative drawing ---------------------------------------------------------------------------
  useLayoutEffect(() => {
    drawAllRef.current = (now: number): boolean => {
      const base = baseCanvasRef.current;
      const field = fieldCanvasRef.current;
      const fx = fxCanvasRef.current;
      if (!base || !field || !fx) return false;
      const w = view.width;
      const h = view.height;
      if (w <= 1 || h <= 1) return false;
      const s = sceneRef.current;
      const raster = getProjectedRaster();
      const project = view.project;
      let again = false;

      // Base: vector basemap + relief
      const bctx = prepareCanvas(base, w, h);
      if (bctx) {
        bctx.clearRect(0, 0, w, h);
        drawBasemap(bctx, view, w, h);
        if (s.showRelief && s.reliefImage) raster.draw(bctx, view, s.reliefImage, 1, true);
      }

      // Field with cross-fade
      const fctx = prepareCanvas(field, w, h);
      if (fctx) {
        fctx.clearRect(0, 0, w, h);
        const fade = fadeRef.current;
        const t = fade.prev ? Math.min(1, (now - fade.start) / CROSSFADE_MS) : 1;
        const e = easeOutCubic(t);
        const paint = (imgs: FieldImages, alpha: number) => {
          if (imgs.right) {
            fctx.save();
            clipSide(fctx, 'left', w, h);
            if (imgs.left) raster.draw(fctx, view, imgs.left, alpha, s.smooth);
            fctx.restore();
            fctx.save();
            clipSide(fctx, 'right', w, h);
            raster.draw(fctx, view, imgs.right, alpha, s.smooth);
            fctx.restore();
          } else if (imgs.left) {
            raster.draw(fctx, view, imgs.left, alpha, s.smooth);
          }
        };
        const fading = Boolean(fade.prev) && t < 1;
        if (fading && fade.prev) {
          paint(fade.prev, 1 - e);
          again = true;
        } else {
          fade.prev = null;
        }
        if (s.fieldImages) paint(s.fieldImages, fading ? e : 1);
      }

      // FX
      const xctx = prepareCanvas(fx, w, h);
      if (!xctx) return again;
      xctx.clearRect(0, 0, w, h);
      drawDomainVeil(xctx, view, w, h);

      if (s.showGridLines) {
        xctx.save();
        xctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
        xctx.lineWidth = 0.5;
        xctx.beginPath();
        for (let k = 0; k <= NLON; k++) {
          const lon = GRID_CONFIG.lonMin - HALF + k;
          const [xa, ya] = project(GRID_CONFIG.latMin - HALF, lon);
          const [xb, yb] = project(GRID_CONFIG.latMax + HALF, lon);
          xctx.moveTo(xa, ya);
          xctx.lineTo(xb, yb);
        }
        for (let k = 0; k <= NLAT; k++) {
          const lat = GRID_CONFIG.latMin - HALF + k;
          for (let q = 0; q <= NLON; q++) {
            const [x, y] = project(lat, GRID_CONFIG.lonMin - HALF + q);
            if (q === 0) xctx.moveTo(x, y);
            else xctx.lineTo(x, y);
          }
        }
        xctx.stroke();
        xctx.restore();
      }

      for (const layer of s.contourLayers) {
        xctx.save();
        clipSide(xctx, layer.side, w, h);
        strokeSegments(xctx, layer.segs, project, layer.style);
        xctx.restore();
      }

      drawCoastline(xctx, view);
      const zoomRatio = view.k / (view.minK / 0.8);
      if (s.showLabels) drawPlaceLabels(xctx, view, zoomRatio);

      // H / L pressure centres
      for (const group of s.centres) {
        xctx.save();
        clipSide(xctx, group.side, w, h);
        for (const c of group.list) {
          const [x, y] = project(c.lat, c.lon);
          const col = c.kind === 'H' ? '#FBBF24' : '#F87171';
          xctx.beginPath();
          xctx.arc(x, y, 13, 0, Math.PI * 2);
          xctx.fillStyle = 'rgba(3, 6, 12, 0.6)';
          xctx.fill();
          xctx.lineWidth = 1.2;
          xctx.strokeStyle = col;
          xctx.shadowColor = col;
          xctx.shadowBlur = 8;
          xctx.stroke();
          xctx.shadowBlur = 0;
          xctx.fillStyle = col;
          xctx.font = '700 15px "Inter Variable", "Segoe UI", system-ui, sans-serif';
          xctx.textAlign = 'center';
          xctx.textBaseline = 'middle';
          xctx.fillText(c.kind, x, y + 0.5);
          xctx.font = '600 10px "JetBrains Mono Variable", ui-monospace, monospace';
          xctx.fillStyle = 'rgba(241, 245, 249, 0.85)';
          xctx.fillText(`${c.hPa}`, x, y + 23);
        }
        xctx.restore();
      }

      // Shock footprint (box or Gaussian radius)
      if (s.activePreset.id !== 'ZERO_SHOCK' || s.isBrushMode) {
        xctx.save();
        xctx.strokeStyle = 'rgba(125, 211, 252, 0.85)';
        xctx.fillStyle = 'rgba(56, 189, 248, 0.07)';
        xctx.lineWidth = 1.3;
        xctx.setLineDash([5, 4]);
        xctx.beginPath();
        if (s.activePreset.box) {
          const [la0, la1, lo0, lo1] = s.activePreset.box.box;
          traceLatLonRect(xctx, project, la0 - HALF, la1 + HALF, lo0 - HALF, lo1 + HALF);
        } else {
          const spec = s.activePreset.id === 'CUSTOM' || s.isBrushMode ? s.customSpec : s.activePreset.spec ?? s.customSpec;
          const [cx, cy] = project(spec.centerLat, spec.centerLon);
          xctx.arc(cx, cy, spec.radiusKm * view.pxPerKm(), 0, Math.PI * 2);
        }
        xctx.fill();
        xctx.stroke();
        xctx.restore();
      }

      // Day-by-day spread rings ("tree rings" of the shock)
      if (s.showSymbols && s.origin && s.viewMode !== 'BASELINE') {
        const [ox, oy] = project(s.origin.lat, s.origin.lon);
        const pk = view.pxPerKm();
        xctx.save();
        xctx.setLineDash([2, 5]);
        xctx.lineWidth = 1;
        for (let d = 1; d < s.currentDay; d++) {
          const r = s.spread[d]?.frontKm ?? 0;
          if (r <= 0) continue;
          xctx.strokeStyle = `rgba(251, 191, 36, ${0.12 + (0.25 * d) / Math.max(1, s.currentDay)})`;
          xctx.beginPath();
          xctx.arc(ox, oy, r * pk, 0, Math.PI * 2);
          xctx.stroke();
        }
        xctx.restore();

        const ring = ringRef.current;
        const rt = Math.min(1, (now - ring.start) / RING_MS);
        ring.shown = ring.from + (ring.to - ring.from) * (prefersReducedMotion() ? 1 : easeOutCubic(rt));
        if (rt < 1) again = true;
        if (ring.shown > 1) {
          const r = ring.shown * pk;
          xctx.save();
          xctx.strokeStyle = '#FBBF24';
          xctx.lineWidth = 1.8;
          xctx.shadowColor = '#FBBF24';
          xctx.shadowBlur = 12;
          xctx.beginPath();
          xctx.arc(ox, oy, r, 0, Math.PI * 2);
          xctx.stroke();
          xctx.shadowBlur = 0;
          const label = s.currentDay === 0 ? 'Shock footprint' : `Day ${s.currentDay} · reach ≈ ${Math.round(ring.shown / 10) * 10} km`;
          xctx.font = '600 11px "Inter Variable", "Segoe UI", system-ui, sans-serif';
          const tw = xctx.measureText(label).width;
          const ly = Math.max(14, oy - r - 11);
          xctx.fillStyle = 'rgba(28, 20, 4, 0.88)';
          xctx.beginPath();
          xctx.roundRect(ox - tw / 2 - 8, ly - 10, tw + 16, 20, 10);
          xctx.fill();
          xctx.strokeStyle = 'rgba(251, 191, 36, 0.6)';
          xctx.lineWidth = 1;
          xctx.stroke();
          xctx.fillStyle = '#FDE68A';
          xctx.textAlign = 'center';
          xctx.textBaseline = 'middle';
          xctx.fillText(label, ox, ly + 0.5);
          xctx.beginPath();
          xctx.arc(ox, oy, 3.5, 0, Math.PI * 2);
          xctx.fillStyle = '#FBBF24';
          xctx.fill();
          xctx.restore();
        }
      }

      // Pinned cell (curved quad)
      if (s.pinnedCell) {
        const cLat = GRID_CONFIG.latMin + s.pinnedCell.gridI * GRID_CONFIG.resolutionDeg;
        const cLon = GRID_CONFIG.lonMin + s.pinnedCell.gridJ * GRID_CONFIG.resolutionDeg;
        xctx.save();
        xctx.strokeStyle = '#FACC15';
        xctx.lineWidth = 1.75;
        xctx.shadowColor = '#FACC15';
        xctx.shadowBlur = 8;
        xctx.beginPath();
        traceLatLonRect(xctx, project, cLat - HALF, cLat + HALF, cLon - HALF, cLon + HALF);
        xctx.stroke();
        xctx.restore();
      }

      if (s.viewMode === 'SPLIT') {
        xctx.save();
        xctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        xctx.lineWidth = 1.5;
        xctx.setLineDash([6, 4]);
        xctx.beginPath();
        xctx.moveTo(w / 2, 0);
        xctx.lineTo(w / 2, h);
        xctx.stroke();
        xctx.restore();
      }
      return again;
    };
  }, [view]);

  // ---- Viewport sizing + intro fly-in ------------------------------------------------------------
  const padding = useMemo<Padding>(
    () =>
      isDesktop
        ? { top: 118, right: insetRight + 24, bottom: 24, left: insetLeft + 24 }
        : { top: 150, right: 12, bottom: 70, left: 12 },
    [isDesktop, insetLeft, insetRight]
  );

  useEffect(() => {
    const outer = outerRef.current;
    if (!outer) return;
    let lastW = 0;
    let lastH = 0;
    const apply = (refit: boolean) => {
      const r = outer.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return;
      const first = !fittedRef.current;
      view.resize(r.width, r.height, padding, refit || first);
      if (first) {
        fittedRef.current = true;
        if (!skipIntroFlyIn) view.flyToDomain(1500, 0.55);
      } else if (refit) {
        view.flyToDomain(500);
      }
      lastW = r.width;
      lastH = r.height;
    };
    apply(true);
    const ro = new ResizeObserver(() => {
      const r = outer.getBoundingClientRect();
      if (Math.abs(r.width - lastW) > 1 || Math.abs(r.height - lastH) > 1) apply(true);
    });
    ro.observe(outer);
    return () => ro.disconnect();
  }, [padding, view, skipIntroFlyIn]);

  useEffect(() => view.onChange(requestDraw), [view, requestDraw]);
  useEffect(() => () => view.dispose(), [view]);

  // ---- Wind particles ------------------------------------------------------------------------------
  useEffect(() => {
    const canvas = windCanvasRef.current;
    if (!canvas) return;
    const colourBySpeed = isWindChannel(selectedChannel) && !isAnomaly;
    const fieldAt = (x: number): WindField => {
      if (isAnomaly) return windFields.delta;
      if (viewMode === 'BASELINE') return windFields.base;
      if (viewMode === 'SPLIT') return x < view.width / 2 ? windFields.base : windFields.shock;
      return windFields.shock;
    };
    const clear = () => {
      const ctx = prepareCanvas(canvas, view.width, view.height);
      ctx?.clearRect(0, 0, view.width, view.height);
      return ctx;
    };
    if (noData || windMode === 'OFF' || (isAnomaly && windFields.deltaPeak <= 1e-3)) {
      clear();
      return;
    }
    if (windMode === 'ARROWS' || prefersReducedMotion()) {
      const drawArrows = () => {
        const ctx = clear();
        if (ctx) drawWindArrows(ctx, view.project, fieldAt, colourBySpeed);
      };
      drawArrows();
      return view.onChange(drawArrows);
    }
    if (!windFlowRef.current) windFlowRef.current = new WindFlow(1500);
    const flow = windFlowRef.current;
    const off = view.onChange(() => {
      clear();
    });
    let raf = 0;
    const tick = () => {
      const ctx = prepareCanvas(canvas, view.width, view.height);
      if (ctx) {
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.globalCompositeOperation = 'destination-in';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
        flow.step(ctx, view.project, fieldAt, colourBySpeed);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      off();
    };
  }, [noData, windMode, windFields, viewMode, isAnomaly, selectedChannel, view]);

  // ---- SFX: rain streaks + sonar pulse from the shock origin ---------------------------------------
  useEffect(() => {
    const canvas = sfxCanvasRef.current;
    if (!canvas || prefersReducedMotion()) return;
    interface Drop {
      lat: number;
      lon: number;
      age: number;
      life: number;
      k: number;
    }
    const drops: Drop[] = [];
    const spawn = (grid: Grid2D): Drop | null => {
      for (let tries = 0; tries < 10; tries++) {
        const i = Math.floor(Math.random() * NLAT);
        const j = Math.floor(Math.random() * NLON);
        const tp = grid[i][j];
        if (tp < 10 || Math.random() > Math.min(1, tp / 45)) continue;
        return {
          lat: GRID_CONFIG.latMin + i + (Math.random() - 0.5),
          lon: GRID_CONFIG.lonMin + j + (Math.random() - 0.5),
          age: 0,
          life: 26 + Math.random() * 22,
          k: Math.min(1, tp / 50),
        };
      }
      return null;
    };
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const ctx = prepareCanvas(canvas, view.width, view.height);
      if (ctx) {
        ctx.clearRect(0, 0, view.width, view.height);
        const s = sceneRef.current;
        if (s.rainGrid && s.rainMotion) {
          for (let n = 0; n < 6 && drops.length < 420; n++) {
            const d = spawn(s.rainGrid);
            if (d) drops.push(d);
          }
          ctx.lineCap = 'round';
          ctx.lineWidth = 1;
          const zoom = Math.sqrt(view.k / (view.minK / 0.8));
          for (let n = drops.length - 1; n >= 0; n--) {
            const d = drops[n];
            d.age += 1;
            if (d.age > d.life) {
              drops.splice(n, 1);
              continue;
            }
            const [x, y] = view.project(d.lat, d.lon);
            const fall = d.age * 0.7 * zoom;
            const a = Math.sin((Math.PI * d.age) / d.life) * (0.22 + 0.45 * d.k);
            ctx.strokeStyle = `rgba(207, 250, 254, ${a})`;
            ctx.beginPath();
            ctx.moveTo(x - 1.5 * zoom + fall * 0.25, y - 6 * zoom + fall);
            ctx.lineTo(x + fall * 0.25, y + fall);
            ctx.stroke();
          }
        } else if (drops.length) {
          drops.length = 0;
        }
        // Sonar pulse: rings travel from the origin out to today's measured reach
        if (s.showSymbols && s.origin && s.viewMode !== 'BASELINE' && s.ringTargetKm > 1) {
          const [ox, oy] = view.project(s.origin.lat, s.origin.lon);
          const maxR = s.ringTargetKm * view.pxPerKm();
          for (let p = 0; p < 2; p++) {
            const phase = ((((now - t0) / PULSE_PERIOD_MS + p * 0.5) % 1) + 1) % 1;
            const r = easeOutCubic(phase) * maxR;
            ctx.strokeStyle = `rgba(251, 191, 36, ${0.45 * (1 - phase)})`;
            ctx.lineWidth = 2 * (1 - phase) + 0.5;
            ctx.beginPath();
            ctx.arc(ox, oy, r, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [view]);

  // ---- Pointer: probe cursor, pan, wheel/pinch zoom, click to pin / place brush -----------------------
  /** Screen point -> view point, inverting the CSS perspective tilt when active. */
  const toContainerPoint = useCallback(
    (clientX: number, clientY: number): [number, number] | null => {
      const outer = outerRef.current;
      if (!outer) return null;
      const r = outer.getBoundingClientRect();
      const x = clientX - r.left;
      const y = clientY - r.top;
      if (!tilt) return [x, y];
      const th = (TILT_DEG * Math.PI) / 180;
      const P = TILT_PERSPECTIVE;
      // transform-origin is 50% 60%
      const xs = x - r.width / 2;
      const ys = y - r.height * 0.6;
      const denom = P * Math.cos(th) + ys * Math.sin(th);
      if (denom <= 0) return null;
      const vy = (ys * P) / denom;
      const vx = (xs * (P - vy * Math.sin(th))) / P;
      return [vx + r.width / 2, vy + r.height * 0.6];
    },
    [tilt]
  );

  const probeState = { baseFrame, shockFrame, deltaGrid, selectedChannel, viewMode, isBrushMode, customSpec, noData };
  const probeStateRef = useRef(probeState);
  useLayoutEffect(() => {
    probeStateRef.current = probeState;
  });

  const updateProbe = useCallback(
    (clientX: number, clientY: number) => {
      const probe = probeRef.current;
      const text = probeTextRef.current;
      const ring = brushRingRef.current;
      const outer = outerRef.current;
      if (!probe || !text || !ring || !outer) return;
      const r = outer.getBoundingClientRect();
      probe.style.opacity = '1';
      probe.style.transform = `translate3d(${clientX - r.left}px, ${clientY - r.top}px, 0)`;
      const pt = toContainerPoint(clientX, clientY);
      const ps = probeStateRef.current;
      if (ps.isBrushMode) {
        const d = 2 * ps.customSpec.radiusKm * view.pxPerKm();
        ring.style.width = `${d}px`;
        ring.style.height = `${d}px`;
        ring.style.opacity = '1';
        ring.dataset.sign = ps.customSpec.magnitude >= 0 ? 'pos' : 'neg';
      } else {
        ring.style.opacity = '0';
      }
      if (!pt) {
        text.innerHTML = '';
        return;
      }
      const [lat, lon] = view.invert(pt[0], pt[1]);
      const coord = `${Math.abs(lat).toFixed(1)}°${lat >= 0 ? 'N' : 'S'} · ${lon.toFixed(1)}°E`;
      if (!insideDomain(lat, lon)) {
        text.innerHTML = `<div class="mf-probe-coord">${coord}</div><div class="mf-probe-muted">Outside the model area</div>`;
        return;
      }
      const where = regionName(lat, lon);
      if (ps.isBrushMode) {
        text.innerHTML = `<div class="mf-probe-coord">${coord}</div><div class="mf-probe-val">Click to place the shock</div><div class="mf-probe-muted">near ${where}</div>`;
        return;
      }
      if (ps.noData) {
        text.innerHTML = `<div class="mf-probe-coord">${coord}</div><div class="mf-probe-muted">No forecast for this day</div>`;
        return;
      }
      const ch = ps.selectedChannel;
      const human = CHANNEL_HUMAN[ch].short;
      const leftSide = pt[0] < view.width / 2;
      let html = `<div class="mf-probe-coord">${coord}</div>`;
      if (ps.viewMode === 'ANOMALY_DIFF') {
        const d = probeValue(ps.deltaGrid, lat, lon);
        const cls = d > 0 ? 'mf-probe-up' : d < 0 ? 'mf-probe-down' : 'mf-probe-muted';
        html += `<div class="mf-probe-val">${human} change <span class="${cls}">${displayDeltaString(d, ch)}</span></div>`;
      } else {
        const showBase = ps.viewMode === 'BASELINE' || (ps.viewMode === 'SPLIT' && leftSide);
        const g = (showBase ? ps.baseFrame : ps.shockFrame)[CHANNELS[ch].index];
        const v = wireToDisplay(probeValue(g, lat, lon), ch);
        html += `<div class="mf-probe-val">${human} <b>${v.str}</b> <span class="mf-probe-unit">${v.unit}</span></div>`;
        if (!showBase) {
          const d = probeValue(ps.deltaGrid, lat, lon);
          if (Math.abs(d) >= PROBE_DELTA_MIN[ch]) {
            const cls = d > 0 ? 'mf-probe-up' : 'mf-probe-down';
            html += `<div class="mf-probe-delta"><span class="${cls}">${displayDeltaString(d, ch)}</span> vs normal forecast</div>`;
          }
        }
      }
      html += `<div class="mf-probe-muted">near ${where}</div>`;
      text.innerHTML = html;
    },
    [toContainerPoint, view]
  );

  const hideProbe = () => {
    if (probeRef.current) probeRef.current.style.opacity = '0';
  };

  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const dragRef = useRef({ moved: false, startX: 0, startY: 0, pinchDist: 0 });

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    e.currentTarget.setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size === 1) dragRef.current = { moved: false, startX: e.clientX, startY: e.clientY, pinchDist: 0 };
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      dragRef.current.pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
      dragRef.current.moved = true;
    }
    setLayersOpen(false);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.pointerType === 'mouse') updateProbe(e.clientX, e.clientY);
    const prev = pointers.current.get(e.pointerId);
    if (!prev) return;
    const next = { x: e.clientX, y: e.clientY };
    pointers.current.set(e.pointerId, next);
    if (tilt) return; // pan/zoom locked in 3D
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const r = outerRef.current?.getBoundingClientRect();
      if (dragRef.current.pinchDist > 0 && r) {
        view.zoomAt(dist / dragRef.current.pinchDist, (a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top);
      }
      dragRef.current.pinchDist = dist;
      return;
    }
    if (!dragRef.current.moved && Math.hypot(e.clientX - dragRef.current.startX, e.clientY - dragRef.current.startY) > 4) {
      dragRef.current.moved = true;
      surfaceRef.current?.classList.add('mf-grabbing');
    }
    if (dragRef.current.moved) view.panBy(next.x - prev.x, next.y - prev.y);
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    surfaceRef.current?.classList.remove('mf-grabbing');
    const had = pointers.current.delete(e.pointerId);
    if (!had || dragRef.current.moved || pointers.current.size > 0) return;
    const pt = toContainerPoint(e.clientX, e.clientY);
    if (!pt) return;
    const [lat, lon] = view.invert(pt[0], pt[1]);
    const grid = latLonToGrid(lat, lon);
    if (!grid || !insideDomain(lat, lon)) return;
    const outerRect = outerRef.current?.getBoundingClientRect();
    if (isBrushMode) {
      if (lat >= GRID_CONFIG.latMin && lat <= GRID_CONFIG.latMax && lon >= GRID_CONFIG.lonMin && lon <= GRID_CONFIG.lonMax) {
        updateCustomSpec({ centerLat: Math.round(lat * 10) / 10, centerLon: Math.round(lon * 10) / 10 });
        if (outerRect && !prefersReducedMotion()) {
          const id = performance.now();
          const color = customSpec.magnitude >= 0 ? '251, 146, 60' : '56, 189, 248';
          setRipples((rs) => [...rs, { id, x: e.clientX - outerRect.left, y: e.clientY - outerRect.top, color }]);
        }
      }
      return;
    }
    setPinnedCell((prev) =>
      prev && prev.gridI === grid.i && prev.gridJ === grid.j
        ? null
        : { lat: Math.round(lat * 10) / 10, lon: Math.round(lon * 10) / 10, gridI: grid.i, gridJ: grid.j }
    );
  };

  // Wheel zoom (non-passive so the page never scrolls under the map)
  useEffect(() => {
    const el = surfaceRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (tilt) return;
      const r = el.getBoundingClientRect();
      view.zoomAt(Math.exp(-e.deltaY * 0.0016), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [tilt, view]);

  const zoomButtons = (factor: number) => view.zoomAt(factor, view.width / 2, view.height / 2);

  // ---- Pinned inspector values ---------------------------------------------------------------------
  const inspected = useMemo(() => {
    if (!pinnedCell || noData) return null;
    const { gridI: i, gridJ: j } = pinnedCell;
    const baseVal = currentBaselineGrid[i][j];
    const shockVal = currentShockGrid[i][j];
    return {
      ...pinnedCell,
      region: regionName(pinnedCell.lat, pinnedCell.lon),
      baseVal,
      shockVal,
      delta: shockVal - baseVal,
      allChannels: (Object.keys(CHANNELS) as ChannelId[]).map((id) => ({
        id,
        shockVal: shockFrame[CHANNELS[id].index][i][j],
      })),
    };
  }, [pinnedCell, noData, currentBaselineGrid, currentShockGrid, shockFrame]);

  const sideLeft = isDesktop ? insetLeft + 12 : 12;
  const sideRight = isDesktop ? insetRight + 12 : 12;
  const nextWindMode: Record<WindMode, WindMode> = { FLOW: 'ARROWS', ARROWS: 'OFF', OFF: 'FLOW' };
  const windLabel: Record<WindMode, string> = { FLOW: 'Wind: flowing', ARROWS: 'Wind: arrows', OFF: 'Wind: off' };

  return (
    <div ref={outerRef} className="relative w-full h-full bg-[#03060C] overflow-hidden select-none">
      {/* Tiltable stage: all canvases move together */}
      <div
        className="absolute inset-0 transition-transform duration-700 ease-out"
        style={{ transform: tilt ? STAGE_TRANSFORM : 'none', transformOrigin: '50% 60%' }}
      >
        <canvas ref={baseCanvasRef} className="absolute inset-0 pointer-events-none" />
        <canvas ref={fieldCanvasRef} className="absolute inset-0 pointer-events-none" style={{ filter: 'saturate(0.92)' }} />
        <canvas ref={fxCanvasRef} className="absolute inset-0 pointer-events-none" />
        <canvas ref={windCanvasRef} className="absolute inset-0 pointer-events-none" />
        <canvas ref={sfxCanvasRef} className="absolute inset-0 pointer-events-none" />
      </div>
      {tilt && (
        <div className="absolute inset-x-0 top-0 h-2/5 z-[5] pointer-events-none bg-gradient-to-b from-[#03060C] via-[#03060C]/60 to-transparent" />
      )}

      {/* Interaction surface: the custom probe cursor replaces the system cursor here only */}
      <div
        ref={surfaceRef}
        data-lenis-prevent
        className={`absolute inset-0 z-[6] touch-none ${isBrushMode ? 'mf-cursor-brush' : 'mf-cursor-probe'}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={(e) => {
          pointers.current.delete(e.pointerId);
          surfaceRef.current?.classList.remove('mf-grabbing');
        }}
        onPointerLeave={hideProbe}
        onDoubleClick={(e) => {
          if (tilt) return;
          const r = outerRef.current?.getBoundingClientRect();
          if (r) view.zoomAt(1.8, e.clientX - r.left, e.clientY - r.top);
        }}
      />

      {/* Probe cursor */}
      <div ref={probeRef} className="mf-probe pointer-events-none absolute left-0 top-0 z-[7]" style={{ opacity: 0 }} aria-hidden>
        <div ref={brushRingRef} className="mf-brush-ring" style={{ opacity: 0 }} />
        <svg className="mf-probe-cross" width="34" height="34" viewBox="-17 -17 34 34">
          <circle r="7" fill="none" stroke="rgba(255,255,255,0.9)" strokeWidth="1.2" />
          <circle r="1.6" fill="#fff" />
          <path d="M0 -16 V-10 M0 10 V16 M-16 0 H-10 M10 0 H16" stroke="rgba(255,255,255,0.85)" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
        <div ref={probeTextRef} className="mf-probe-card" />
      </div>

      {/* Click ripples (brush placement) */}
      {ripples.map((rp) => (
        <span
          key={rp.id}
          className="mf-ripple pointer-events-none absolute z-[7]"
          style={{ left: rp.x, top: rp.y, ['--rip' as string]: rp.color } as React.CSSProperties}
          onAnimationEnd={() => setRipples((rs) => rs.filter((x) => x.id !== rp.id))}
        />
      ))}

      {/* Top overlays: headline, then channel + layers row */}
      <div className="absolute z-20 pointer-events-none flex flex-col gap-2" style={{ top: isDesktop ? 12 : 44, left: sideLeft, right: sideRight }}>
        <NarrativeBanner headline={headline} busy={apiStatus.isSimulating} unavailable={noData} synthetic={dataMode === 'DEMO'} />
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="pointer-events-auto flex items-start gap-2 max-w-full">
            <ChannelSelector />
            <div className="relative">
              <button
                onClick={() => setLayersOpen((o) => !o)}
                className={`mf-glass mf-press h-[34px] px-3 rounded-xl flex items-center gap-1.5 text-[12.5px] ${layersOpen ? 'text-white' : 'text-zinc-300 hover:text-white'}`}
                aria-expanded={layersOpen}
              >
                <Layers className="w-4 h-4" />
                <span className="hidden sm:inline">Layers</span>
              </button>
              {layersOpen && (
                <div className="mf-glass mf-pop absolute left-0 top-[40px] w-60 rounded-xl p-1.5 space-y-0.5 shadow-2xl">
                  <LayerToggle on={smooth} onClick={() => setSmooth((p) => !p)} icon={<Waves className="w-4 h-4" />} label="Smooth shading" hint="On: bilinear blend between 1° cells. Off: the exact 1° model cells." />
                  <LayerToggle on={showRelief} onClick={() => setShowRelief((p) => !p)} icon={<Mountain className="w-4 h-4" />} label="Terrain" hint="Hill shading from the model's orography (static.npy)" disabled={!staticFields} />
                  <LayerToggle on={showContours} onClick={() => setShowContours((p) => !p)} icon={<Spline className="w-4 h-4" />} label="Contours" hint="Heavy (10) and extreme (50 mm/day) rain lines; isobars on Pressure" />
                  <LayerToggle on={windMode !== 'OFF'} onClick={() => setWindMode((m) => nextWindMode[m])} icon={<Wind className="w-4 h-4" />} label={windLabel[windMode]} hint="Cycles flowing particles → arrows → off (from the displayed u10/v10)" />
                  <LayerToggle on={rainMotion} onClick={() => setRainMotion((p) => !p)} icon={<CloudRain className="w-4 h-4" />} label="Falling rain" hint="Animated streaks where rain ≥ 10 mm/day (Rain view only)" />
                  <LayerToggle on={showSymbols} onClick={() => setShowSymbols((p) => !p)} icon={<Radar className="w-4 h-4" />} label="Symbols & spread rings" hint="H / L pressure centres and the day-by-day reach of the shock" />
                  <LayerToggle on={showLabels} onClick={() => setShowLabels((p) => !p)} icon={<Tag className="w-4 h-4" />} label="Place names" hint="Countries, seas and cities" />
                  <LayerToggle on={showGridLines} onClick={() => setShowGridLines((p) => !p)} icon={<Grid3x3 className="w-4 h-4" />} label="Model grid (1°)" hint="The 32 × 36 cells the model predicts" />
                  <LayerToggle on={tilt} onClick={() => setTilt((p) => !p)} icon={<Box className="w-4 h-4" />} label="3D tilt" hint="Perspective view (pan and zoom are locked while tilted)" />
                </div>
              )}
            </div>
          </div>

          <div className="pointer-events-auto mf-glass rounded-xl flex items-center p-0.5 ml-auto">
            <button onClick={() => zoomButtons(1.4)} disabled={tilt} className="mf-press w-8 h-8 grid place-items-center rounded-lg text-zinc-300 hover:text-white hover:bg-white/10 disabled:opacity-40" aria-label="Zoom in">
              <Plus className="w-4 h-4" />
            </button>
            <button onClick={() => zoomButtons(1 / 1.4)} disabled={tilt} className="mf-press w-8 h-8 grid place-items-center rounded-lg text-zinc-300 hover:text-white hover:bg-white/10 disabled:opacity-40" aria-label="Zoom out">
              <Minus className="w-4 h-4" />
            </button>
            <button onClick={() => view.flyToDomain(700)} className="mf-press w-8 h-8 grid place-items-center rounded-lg text-zinc-300 hover:text-white hover:bg-white/10" aria-label="Fit the whole model area" title="Fit the whole model area">
              <Locate className="w-4 h-4" />
            </button>
          </div>
        </div>
        {dataMode === 'LIVE' && apiStatus.isLive && apiStatus.isStubResponse && (
          <div data-testid="stub-warning-chip" className="pointer-events-auto self-end px-2.5 py-1 rounded-lg bg-amber-950/80 border border-amber-700/70 text-[11px] text-amber-200 max-w-xs text-right">
            Stub detected · {apiStatus.stubReason}
          </div>
        )}
        {isAnomaly && windMode !== 'OFF' && windFields.deltaPeak > 1e-3 && !noData && (
          <div className="self-start px-2.5 py-1 rounded-lg mf-glass text-[11px] text-zinc-400">
            Wind shows the change only · peak {windFields.deltaPeak.toFixed(2)} m/s, drawn ×{windFields.deltaScale.toFixed(1)}
          </div>
        )}
      </div>

      {viewMode === 'SPLIT' && !noData && (
        <div className="absolute z-[8] flex justify-between pointer-events-none text-[12px]" style={{ bottom: isDesktop ? 112 : 150, left: sideLeft, right: sideRight }}>
          <div className="mf-glass px-3 py-1 rounded-full text-zinc-300">← Normal forecast</div>
          <div className="mf-glass px-3 py-1 rounded-full text-amber-200">With shock →</div>
        </div>
      )}

      {/* LIVE error: never substitutes synthetic forecasts */}
      {dataMode === 'LIVE' && apiStatus.errorMessage && (
        <div
          role="alert"
          data-testid="live-error-banner"
          className="absolute z-30 left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2 w-[calc(100vw-32px)] max-w-md bg-[#160B0C]/95 backdrop-blur-2xl border border-rose-700/70 rounded-2xl p-5 shadow-2xl text-[13px] text-zinc-200 mf-pop"
        >
          <div className="flex items-center gap-2 text-rose-300 font-semibold text-sm mb-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>The live model is not reachable</span>
          </div>
          <p className="text-zinc-300 leading-relaxed mb-2 break-words font-mono text-[11.5px]">{apiStatus.errorMessage}</p>
          <p className="text-[12px] text-rose-200/80 mb-4">
            Live mode never substitutes synthetic data, so nothing is shown until MonsoonFlow-10 answers.
          </p>
          <div className="flex flex-wrap gap-2">
            <button onClick={() => void executeSimulation()} className="mf-press px-3 py-1.5 rounded-lg bg-rose-900/60 hover:bg-rose-800/70 border border-rose-600/60 text-white flex items-center gap-1.5">
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Try again</span>
            </button>
            <button onClick={() => setDataMode('DEMO')} className="mf-press px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20 text-zinc-100">
              Switch to demo data
            </button>
          </div>
        </div>
      )}

      {/* Colour legend */}
      <div className="absolute z-20" style={{ bottom: 12, left: sideLeft }}>
        <ColorbarLegend isAnomaly={isAnomaly} maxAbsDelta={maxAbsDelta} contoursOn={showContours} />
      </div>

      {/* Pinned cell inspector */}
      {inspected && (
        <div
          style={{ right: sideRight, bottom: isDesktop ? 12 : 128 }}
          className="absolute z-20 mf-glass mf-pop rounded-2xl p-3.5 text-[12px] tabular-nums w-[280px] max-w-[calc(100vw-24px)]"
          data-testid="cell-inspector"
        >
          <div className="flex items-start justify-between gap-2 mb-2">
            <div>
              <div className="flex items-center gap-1.5 text-zinc-100 font-semibold">
                <MapPin className="w-3.5 h-3.5 text-amber-300" />
                <span>Near {inspected.region}</span>
              </div>
              <div className="text-[11px] text-zinc-500 font-mono mt-0.5">
                {inspected.lat.toFixed(1)}°N, {inspected.lon.toFixed(1)}°E · cell [{inspected.gridI},{inspected.gridJ}]
              </div>
            </div>
            <button onClick={() => setPinnedCell(null)} className="mf-press p-1 rounded-md text-zinc-400 hover:text-white hover:bg-white/10" title="Unpin" aria-label="Unpin cell">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="grid grid-cols-3 gap-2 mb-2.5 p-2 rounded-xl bg-white/[0.04]">
            <div>
              <span className="text-zinc-500 block text-[10.5px]">Normal</span>
              <span className="text-zinc-200 font-medium">
                {wireToDisplay(inspected.baseVal, selectedChannel).str}
                <span className="text-zinc-500 text-[10px] ml-0.5">{wireToDisplay(inspected.baseVal, selectedChannel).unit}</span>
              </span>
            </div>
            <div>
              <span className="text-zinc-500 block text-[10.5px]">With shock</span>
              <span className="text-white font-medium">
                {wireToDisplay(inspected.shockVal, selectedChannel).str}
                <span className="text-zinc-500 text-[10px] ml-0.5">{wireToDisplay(inspected.shockVal, selectedChannel).unit}</span>
              </span>
            </div>
            <div>
              <span className="text-zinc-500 block text-[10.5px]">Change</span>
              <span className={`font-semibold ${inspected.delta > 0 ? 'text-orange-300' : inspected.delta < 0 ? 'text-sky-300' : 'text-zinc-400'}`}>
                {displayDeltaString(inspected.delta, selectedChannel)}
              </span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
            {inspected.allChannels.map((entry) => {
              const disp = wireToDisplay(entry.shockVal, entry.id);
              return (
                <div key={entry.id} className="flex justify-between gap-1 text-zinc-400" title={CHANNEL_HUMAN[entry.id].tech}>
                  <span className={entry.id === selectedChannel ? 'text-white font-semibold' : ''}>{CHANNEL_HUMAN[entry.id].short}</span>
                  <span className="text-zinc-200">
                    {disp.str}
                    <span className="text-zinc-500 ml-0.5">{disp.unit}</span>
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
