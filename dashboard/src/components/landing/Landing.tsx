/**
 * Landing.tsx - Scroll story in chapters:
 *   3D             00 Earth from deep space → 01 the monsoon seen from orbit → 02 the model's window (dive)
 *   cloud pass     the camera dives through a cloud bank; under full cover the 3D Earth is swapped for the 2D chart
 *   story (2D)     02 the model's window → 03 today → 04 normal forecast → 05 one shock
 *                  → 06 the ripple → 07 what it means
 * then "how it works", design guarantees and the way into the lab.
 * Every number and field in the story is computed from the same tensors the canvases draw.
 */

import React, { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CHANNELS, GRID_CONFIG, LATS, LONS } from '../../types/climate';
import { buildHeadline, shockOrigin, spreadByDay } from '../../services/diagnostics';
import { regionName } from '../../geo/basemap';
import type { GaussianShockSpec } from '../../types/shocks';
import { navigate } from '../../router';
import { BrandMark } from '../ui/BrandMark';
import { CountUp } from '../ui/CountUp';
import { SplitReveal, type RevealSegment } from '../ui/SplitReveal';
import { StoryStage, STORY_STEPS } from './StoryStage';
import { CLOUD, heroGlobeFrame } from './globeFrame';
import { CursorFollower } from './CursorFollower';
import { ChapterRail, type Chapter } from './ChapterRail';
import { Preloader } from './Preloader';
import { useSmoothScroll } from './useSmoothScroll';
import { useStoryData, type StoryData } from './useStoryData';
import { ArrowRight, ChevronDown, Cpu, GitFork, Gauge, Lock, Orbit, Radar, ShieldCheck, Thermometer, Waves } from 'lucide-react';

const loadOrbit = () => import('./OrbitScene');
const OrbitScene = lazy(loadOrbit);

/** Screens of 3D prologue before the 2D story starts. */
const PROLOGUE = 2;
const TOTAL = PROLOGUE + STORY_STEPS;
const CHAPTERS: Chapter[] = [
  { label: 'Orbit', screen: 0 },
  { label: 'Monsoon', screen: 1 },
  { label: 'Window', screen: 2 },
  { label: 'Today', screen: 3 },
  { label: 'Forecast', screen: 4 },
  { label: 'Shock', screen: 5 },
  { label: 'Ripple', screen: 6 },
  { label: 'Meaning', screen: 7 },
];

const UNUSED_SPEC: GaussianShockSpec = { centerLat: 15, centerLon: 88, radiusKm: 450, sigmaKm: 300, channel: 't2m', magnitude: 0 };
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const SERIF = 'font-display italic font-normal';

const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const smooth = (e0: number, e1: number, x: number) => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};

function startDate(label: string): string | null {
  const m = /(\d{4})-(\d{2})-(\d{2})/.exec(label);
  if (!m) return null;
  return `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]} ${m[1]}`;
}

function useIsDesktop(): boolean {
  const [d, setD] = useState(() => typeof window !== 'undefined' && window.matchMedia('(min-width: 1024px)').matches);
  useEffect(() => {
    const mq = window.matchMedia('(min-width: 1024px)');
    const on = () => setD(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return d;
}

/** Reveal-on-scroll for static sections. */
function useReveal(): void {
  useEffect(() => {
    const els = document.querySelectorAll<HTMLElement>('.mf-reveal');
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && e.target.classList.add('is-in')),
      { threshold: 0.15 }
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);
}

interface Summary {
  heavyToday: number;
  rainyToday: number;
  heavyDelta: number;
  heavyDeltaDay: number;
  reachKm: number;
  peakDelta: number;
  peakDay: number;
  peakRegion: string;
  windMax: number;
  windRegion: string;
  tcwvMax: number;
  tcwvRegion: string;
}

function summarise(s: StoryData): Summary {
  const tp = CHANNELS.tp.index;
  const count = (g: number[][], thr: number) => g.reduce((n, row) => n + row.filter((v) => v >= thr).length, 0);
  let heavyDelta = 0;
  let heavyDeltaDay = 1;
  let peakDelta = 0;
  let peakDay = 1;
  let peakI = 0;
  let peakJ = 0;
  s.shock.forEach((f, k) => {
    const dh = count(f[tp], 10) - count(s.base[k][tp], 10);
    if (Math.abs(dh) > Math.abs(heavyDelta)) {
      heavyDelta = dh;
      heavyDeltaDay = k + 1;
    }
    for (let i = 1; i < GRID_CONFIG.nLat - 1; i++) {
      for (let j = 1; j < GRID_CONFIG.nLon - 1; j++) {
        let sum = 0;
        for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) sum += f[tp][i + di][j + dj] - s.base[k][tp][i + di][j + dj];
        const d = sum / 9;
        if (Math.abs(d) > Math.abs(peakDelta)) {
          peakDelta = d;
          peakDay = k + 1;
          peakI = i;
          peakJ = j;
        }
      }
    }
  });
  // Day-0 extremes for the orbit chapter
  const u = s.day0[CHANNELS.u10.index];
  const v = s.day0[CHANNELS.v10.index];
  const q = s.day0[CHANNELS.tcwv.index];
  let windMax = 0;
  let wi = 0;
  let wj = 0;
  let tcwvMax = 0;
  let qi = 0;
  let qj = 0;
  for (let i = 0; i < GRID_CONFIG.nLat; i++) {
    for (let j = 0; j < GRID_CONFIG.nLon; j++) {
      const sp = Math.hypot(u[i][j], v[i][j]);
      if (sp > windMax) [windMax, wi, wj] = [sp, i, j];
      if (q[i][j] > tcwvMax) [tcwvMax, qi, qj] = [q[i][j], i, j];
    }
  }
  const origin = shockOrigin(s.preset, UNUSED_SPEC);
  const spread = spreadByDay(s.day0, s.day0Shocked, s.base, s.shock, origin);
  return {
    heavyToday: count(s.day0[tp], 10),
    rainyToday: count(s.day0[tp], 5),
    heavyDelta,
    heavyDeltaDay,
    reachKm: Math.max(0, ...spread.map((d) => d.frontKm)),
    peakDelta,
    peakDay,
    peakRegion: regionName(LATS[peakI], LONS[peakJ]),
    windMax,
    windRegion: regionName(LATS[wi], LONS[wj]),
    tcwvMax,
    tcwvRegion: regionName(LATS[qi], LONS[qj]),
  };
}

const StepCard: React.FC<{
  num: number;
  eyebrow: string;
  title: RevealSegment[];
  children: React.ReactNode;
  active: boolean;
}> = ({ num, eyebrow, title, children, active }) => (
  <div className="relative max-w-md">
    <span className="mf-numeral" style={{ '--n': num } as React.CSSProperties} aria-hidden="true">
      {String(num).padStart(2, '0')}
    </span>
    <div className={`mf-card relative rounded-[28px] p-6 sm:p-7 transition-all duration-700 ${active ? 'opacity-100 translate-y-0' : 'opacity-50 translate-y-3'}`}>
      <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#E5E5E5] mb-3">
        <span className="text-[#737373]">{String(num).padStart(2, '0')} —</span> {eyebrow}
      </p>
      {/* Reveals once 30 % of the title is on screen, independent of the active chapter */}
      <SplitReveal as="h2" segments={title} className="text-[27px] sm:text-[32px] leading-[1.08] text-[#FAFAFA] tracking-[-0.02em] text-balance" />
      <div className="mt-4 text-[15px] leading-relaxed font-light text-[#E5E5E5] space-y-3">{children}</div>
    </div>
  </div>
);

const Chip: React.FC<{ children: React.ReactNode; mono?: boolean }> = ({ children, mono }) => (
  <span className={`px-2.5 py-1 rounded-full border border-[#404040] bg-[#262626] text-[#E5E5E5] flex items-center gap-1.5 ${mono ? 'font-mono' : ''}`}>{children}</span>
);

export default function Landing() {
  const story = useStoryData();
  const isDesktop = useIsDesktop();
  const [reduced] = useState(() => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const rawRef = useRef(0);
  const progressRef = useRef(0);
  const idleRef = useRef(true);
  const storyRef = useRef<HTMLDivElement>(null);
  const stage2dRef = useRef<HTMLDivElement>(null);
  const altRef = useRef<HTMLSpanElement>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const [chapter, setChapter] = useState(0);
  /** Screen closest to the viewport: drives which card is emphasised (reads earlier than floor). */
  const [nearest, setNearest] = useState(0);
  const [step, setStep] = useState(-1);
  const [day, setDay] = useState(0);
  const [scrolled, setScrolled] = useState(false);
  const [engineLoaded, setEngineLoaded] = useState(false);
  const [webgl, setWebgl] = useState<'loading' | 'ready' | 'failed'>('loading');
  const [introDone, setIntroDone] = useState(false);
  const lenisRef = useSmoothScroll(!reduced && introDone);
  useReveal();

  useEffect(() => {
    let alive = true;
    loadOrbit()
      .then(() => alive && setEngineLoaded(true))
      .catch(() => {
        if (!alive) return;
        setEngineLoaded(true);
        setWebgl('failed');
      });
    return () => {
      alive = false;
    };
  }, []);

  // Hold the page still while the preloader is up
  useEffect(() => {
    if (introDone) return;
    const html = document.documentElement;
    html.style.overflow = 'hidden';
    return () => {
      html.style.overflow = '';
    };
  }, [introDone]);

  const use3d = webgl !== 'failed';

  useEffect(() => {
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const el = storyRef.current;
        if (!el) return;
        const vh = window.innerHeight;
        const r = el.getBoundingClientRect();
        const raw = clamp(-r.top / vh, 0, TOTAL - 0.001);
        rawRef.current = raw;
        const p = clamp(raw - PROLOGUE, 0, STORY_STEPS - 0.001);
        progressRef.current = p;
        // With 3D on, the chart is switched on while the cloud bank fully covers the screen
        const show2d = use3d ? smooth(CLOUD.swap - 0.03, CLOUD.swap + 0.01, raw) : 1;
        if (stage2dRef.current) stage2dRef.current.style.opacity = show2d.toFixed(3);
        idleRef.current = show2d <= 0.001;
        el.style.setProperty('--mf-raw', raw.toFixed(3));
        if (barRef.current) barRef.current.style.transform = `scaleX(${(clamp(-r.top / vh, 0, TOTAL) / TOTAL).toFixed(4)})`;
        const s = raw < PROLOGUE ? -1 : Math.floor(p);
        const t = p - Math.floor(p);
        setChapter(r.bottom < vh * 0.5 ? -1 : Math.min(TOTAL - 1, Math.floor(raw)));
        setNearest(r.bottom < vh * 0.5 ? -1 : Math.min(TOTAL - 1, Math.round(raw)));
        setStep(s);
        setDay(s === 2 ? Math.min(10, Math.round(t * 10)) : s === 4 ? Math.min(10, 1 + Math.round(t * 9)) : s >= 5 ? 10 : 0);
        setScrolled(window.scrollY > 40);
      });
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, [use3d]);

  const summary = useMemo(() => (story ? summarise(story) : null), [story]);
  const origin = useMemo(() => (story ? shockOrigin(story.preset, UNUSED_SPEC) : null), [story]);
  const spread = useMemo(
    () => (story && origin ? spreadByDay(story.day0, story.day0Shocked, story.base, story.shock, origin) : []),
    [story, origin]
  );
  const live = useMemo(() => {
    if (!story) return null;
    const d = Math.max(0, Math.min(10, day));
    const base = d === 0 ? story.day0 : story.base[d - 1];
    const shock = d === 0 ? story.day0Shocked : story.shock[d - 1];
    return buildHeadline({
      day: d,
      base,
      shock,
      origin,
      presetName: 'Warmer Bay of Bengal',
      spread: spread[d] ?? null,
      showingShock: step >= 3,
    });
  }, [story, day, step, origin, spread]);
  const date = story ? startDate(story.startLabel) : null;

  const scrollToY = useCallback(
    (y: number) => {
      const l = lenisRef.current;
      if (l) l.scrollTo(y, { duration: 1.6 });
      else window.scrollTo({ top: y, behavior: reduced ? 'auto' : 'smooth' });
    },
    [lenisRef, reduced]
  );
  const jumpToScreen = useCallback(
    (screen: number) => {
      const el = storyRef.current;
      if (!el) return;
      scrollToY(el.getBoundingClientRect().top + window.scrollY + screen * window.innerHeight + 2);
    },
    [scrollToY]
  );
  const jumpToHow = () => {
    const el = document.getElementById('how');
    if (el) scrollToY(el.getBoundingClientRect().top + window.scrollY);
  };
  const enterLab = () => {
    navigate('lab');
    window.scrollTo(0, 0);
  };

  const onOrbitStatus = useCallback((s: 'ready' | 'failed') => setWebgl(s), []);
  const onIntroDone = useCallback(() => setIntroDone(true), []);
  const preOrigin = typeof window !== 'undefined' ? heroGlobeFrame(window.innerWidth, window.innerHeight, isDesktop) : { cx: 0, cy: 0 };

  return (
    <div className="mf-landing bg-[#171717] text-[#E5E5E5] min-h-screen">
      <CursorFollower />
      {!introDone && (
        <Preloader
          origin={{ x: preOrigin.cx, y: preOrigin.cy }}
          reduced={reduced}
          onDone={onIntroDone}
          steps={[
            { label: '3D engine', done: engineLoaded },
            { label: 'Earth imagery · NASA', done: webgl !== 'loading', note: webgl === 'failed' ? '2D fallback' : 'July 2004' },
            { label: 'Day-0 atmosphere', done: Boolean(story), note: story?.syntheticStart ? 'synthetic' : 'real tensors' },
            { label: '10-day forecasts ×2', done: Boolean(story), note: story?.live ? 'live model' : 'demo engine' },
          ]}
        />
      )}

      {/* Nav */}
      <nav
        className={`fixed top-0 inset-x-0 z-50 h-16 px-4 sm:px-8 flex items-center justify-between transition-colors duration-500 ${
          scrolled ? 'bg-[#171717]/80 backdrop-blur-xl border-b border-[#262626]' : ''
        }`}
      >
        <a href="#/" className="flex items-center gap-2.5" onClick={() => scrollToY(0)}>
          <BrandMark tone="mono" className="w-7 h-7" />
          <span className="text-[#FAFAFA] text-[16px]">
            <span className="font-extrabold tracking-tight">MonsoonFlow</span>
            <span className="font-display italic text-[19px]">-10</span>
          </span>
        </a>
        <div className="flex items-center gap-2 sm:gap-6 text-[14px]">
          <a
            href="#how"
            onClick={(e) => {
              e.preventDefault();
              jumpToHow();
            }}
            className="hidden sm:inline font-light text-[#E5E5E5] hover:text-[#FAFAFA]"
          >
            How it works
          </a>
          <button onClick={enterLab} data-cursor="open" data-magnetic className="mf-press h-10 px-4 rounded-full bg-[#FAFAFA] text-[#171717] font-semibold flex items-center gap-1.5 hover:bg-[#E5E5E5]">
            Open the lab <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </nav>

      <ChapterRail chapters={CHAPTERS} active={chapter} barRef={barRef} onJump={jumpToScreen} />

      {/* Scroll story */}
      <section ref={storyRef} className="relative" style={{ height: `${TOTAL * 100}vh` }}>
        <div className="sticky top-0 h-screen overflow-hidden">
          <div ref={stage2dRef} className="absolute inset-0" style={{ opacity: use3d ? 0 : 1 }}>
            <StoryStage data={story} progressRef={progressRef} isDesktop={isDesktop} idleRef={idleRef} mapOnly={use3d} />
          </div>
          {use3d && (
            <Suspense fallback={null}>
              <OrbitScene
                rawRef={rawRef}
                prologue={PROLOGUE}
                data={story}
                isDesktop={isDesktop}
                reduced={reduced}
                altitudeRef={altRef}
                onStatus={onOrbitStatus}
              />
            </Suspense>
          )}
          <div className="mf-vignette" aria-hidden="true" />
          <div className="mf-grain" aria-hidden="true" />
          <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-[#171717] to-transparent pointer-events-none" />

          {/* Day badge during the time-lapses */}
          {(step === 2 || step === 4) && (
            <div className="absolute top-20 right-4 sm:right-8 lg:right-44 mf-card rounded-2xl px-4 py-2 text-right mf-pop">
              <div className="font-mono text-[10.5px] uppercase tracking-[0.18em] text-[#E5E5E5]">{step === 2 ? 'Normal forecast' : 'Change caused by the shock'}</div>
              <div className="text-[30px] leading-tight text-[#FAFAFA] tabular-nums">
                {day === 0 ? <span className={SERIF}>Today</span> : <><span className="font-extralight">Day</span> <span className="font-extrabold">{day}</span></>}
              </div>
            </div>
          )}
          {use3d && chapter >= 0 && chapter <= PROLOGUE && (
            <div className="hidden sm:block absolute bottom-4 right-4 sm:right-8 font-mono text-[10.5px] uppercase tracking-[0.14em] text-[#E5E5E5] px-3 py-1.5 rounded-full mf-card" title="Camera position (geometry, not data)">
              <span ref={altRef}>ALT · km</span>
            </div>
          )}
          {story && (
            <div className="absolute bottom-4 left-4 sm:left-8 text-[11.5px] px-3 py-1.5 rounded-full mf-card flex items-center gap-2 text-[#E5E5E5]">
              <span className={`w-2 h-2 rounded-full ${story.live ? 'bg-[#FAFAFA] mf-ping' : 'border border-[#E5E5E5]'}`} />
              {story.live ? 'Live MonsoonFlow-10 output' : 'Demo data (toy engine). Start serve.py for the real model'}
            </div>
          )}
        </div>

        <div className="relative -mt-[100vh] z-10">
          {/* 00 · Hero */}
          <div className="h-screen flex items-center px-5 sm:px-10 lg:px-16">
            <div className={`max-w-[640px] ${introDone ? 'mf-hero-in' : 'mf-hero-wait'}`}>
              <p className="mf-hero-item font-mono text-[11px] uppercase tracking-[0.22em] text-[#E5E5E5] mb-7" style={{ '--i': 0 } as React.CSSProperties}>
                <span className="text-[#737373]">00 —</span> An AI weather model for India’s monsoon
              </p>
              <SplitReveal
                as="h1"
                active={introDone}
                delayMs={120}
                className="text-[44px] sm:text-[62px] lg:text-[76px] leading-[0.98] tracking-[-0.035em] text-[#FAFAFA] text-balance"
                segments={[
                  { t: 'What if', c: 'font-extrabold' },
                  { t: 'the Bay of Bengal were', c: 'font-extralight' },
                  { t: '3 °C warmer?', c: `${SERIF} tracking-[-0.01em] text-[1.12em]` },
                ]}
              />
              <p className="mf-hero-item mt-7 text-[17px] sm:text-[19px] leading-relaxed font-light text-[#E5E5E5] max-w-lg" style={{ '--i': 3 } as React.CSSProperties}>
                MonsoonFlow-10 forecasts the next 10 days of the monsoon, and every{' '}
                <span className="font-display italic text-[1.15em] text-[#FAFAFA]">what-if</span> version of them. Change the weather today and watch where the rain goes.
              </p>
              <div className="mf-hero-item mt-9 flex flex-wrap items-center gap-3" style={{ '--i': 4 } as React.CSSProperties}>
                <button onClick={enterLab} data-cursor="go" data-magnetic className="mf-press h-12 px-6 rounded-full bg-[#FAFAFA] text-[#171717] font-semibold text-[15px] flex items-center gap-2 hover:bg-[#E5E5E5]">
                  Explore the multiverse <ArrowRight className="w-4 h-4" />
                </button>
                <button
                  onClick={() => jumpToScreen(1)}
                  data-magnetic
                  className="mf-press h-12 px-5 rounded-full border border-[#404040] text-[#E5E5E5] hover:bg-[#262626] hover:border-[#737373] text-[15px] font-light"
                >
                  Watch the story
                </button>
              </div>
              <div className="mf-hero-item mt-14 flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.16em] text-[#E5E5E5]" style={{ '--i': 5 } as React.CSSProperties}>
                <ChevronDown className="w-4 h-4 mf-scroll-cue" /> Scroll to fly down to India
              </div>
            </div>
          </div>

          {/* 01 · From orbit */}
          <div className="h-screen flex items-end lg:items-center px-4 sm:px-10 lg:px-16 pb-8 lg:pb-0">
            <StepCard
              num={1}
              eyebrow="From orbit"
              active={nearest === 1}
              title={[
                { t: 'Every summer, the ocean', c: 'font-extralight' },
                { t: 'breathes', c: SERIF },
                { t: 'on India.', c: 'font-extrabold' },
              ]}
            >
              <p>
                From June to September, south-west winds carry moisture off the Arabian Sea and the Bay of Bengal and across the subcontinent. The streaks are those winds
                {date ? <> on {date}</> : null}; the haze is water vapour and the glow is rain, read straight from the model’s Day-0 tensors.
              </p>
              {summary && (
                <p className="flex flex-wrap gap-2 text-[12.5px]">
                  <Chip>
                    <Waves className="w-3.5 h-3.5" />
                    <span>
                      Wind up to <b className="font-bold text-[#FAFAFA]">{summary.windMax.toFixed(1)} m/s</b> over {summary.windRegion}
                    </span>
                  </Chip>
                  <Chip>
                    <span>
                      Water vapour up to <b className="font-bold text-[#FAFAFA]">{Math.round(summary.tcwvMax)} kg/m²</b> over {summary.tcwvRegion}
                    </span>
                  </Chip>
                </p>
              )}
            </StepCard>
          </div>

          {/* 02 · The model's window (2D globe, step 0) */}
          <div className="h-screen flex items-end lg:items-center px-4 sm:px-10 lg:px-16 pb-8 lg:pb-0">
            <StepCard
              num={2}
              eyebrow="The model’s window"
              active={nearest === 2}
              title={[
                { t: 'The model sees India as', c: 'font-extralight' },
                { t: `${(GRID_CONFIG.nLat * GRID_CONFIG.nLon).toLocaleString('en-US')}`, c: 'font-extrabold' },
                { t: 'boxes.', c: SERIF },
              ]}
            >
              <p>
                A {GRID_CONFIG.nLat} × {GRID_CONFIG.nLon} grid of 1° boxes, roughly 110 km a side, from {GRID_CONFIG.latMin}° to {GRID_CONFIG.latMax}° N and {GRID_CONFIG.lonMin}° to{' '}
                {GRID_CONFIG.lonMax}° E.
              </p>
              {summary && (
                <p className="text-[14px]">
                  The soft glow is rain: <b className="font-bold text-[#FAFAFA]"><CountUp value={summary.rainyToday} /></b> boxes have more than 5 mm today.
                </p>
              )}
            </StepCard>
          </div>

          {/* 03 · Today */}
          <div className="h-screen flex items-end lg:items-center px-4 sm:px-10 lg:px-16 pb-8 lg:pb-0">
            <StepCard
              num={3}
              eyebrow="Today"
              active={nearest === 3}
              title={date ? [{ t: 'This is the monsoon', c: 'font-extralight' }, { t: `on ${date}.`, c: SERIF }] : [{ t: 'This is the monsoon', c: 'font-extralight' }, { t: 'today.', c: SERIF }]}
            >
              <p>Below the clouds, the same day as a weather chart. Colour is the rain the model sees today; the faint streaks are the south-west monsoon winds.</p>
              {summary && (
                <p className="text-[14px]">
                  <b className="font-bold text-[#FAFAFA]"><CountUp value={summary.heavyToday} /></b> of {GRID_CONFIG.nLat * GRID_CONFIG.nLon} grid boxes have heavy rain (10 mm/day or more).
                </p>
              )}
            </StepCard>
          </div>

          {/* 04 · Normal forecast */}
          <div className="h-screen flex items-end lg:items-center px-4 sm:px-10 lg:px-16 pb-8 lg:pb-0">
            <StepCard
              num={4}
              eyebrow="The normal forecast"
              active={nearest === 4}
              title={[{ t: 'Left alone,', c: SERIF }, { t: 'the next 10 days', c: 'font-extrabold' }, { t: 'look like this.', c: 'font-extralight' }]}
            >
              <p>Keep scrolling to fast-forward. The model rolls the atmosphere forward one day at a time; each day feeds the next.</p>
              {live && step === 2 && (
                <p key={live.title} className="mf-rise text-[14px] text-[#FAFAFA] border-l border-[#E5E5E5] pl-3">
                  {live.title}
                </p>
              )}
            </StepCard>
          </div>

          {/* 05 · The shock */}
          <div className="h-screen flex items-end lg:items-center px-4 sm:px-10 lg:px-16 pb-8 lg:pb-0">
            <StepCard
              num={5}
              eyebrow="Change one thing"
              active={nearest === 5}
              title={[{ t: 'Warm the Bay of Bengal', c: 'font-extralight' }, { t: 'by 3 °C.', c: SERIF }]}
            >
              <p>
                We also add 15 % more moisture to the air above it. This is one of the two official stress tests the model is graded on. Everything else about today stays
                exactly the same.
              </p>
              <p className="flex flex-wrap gap-2 text-[12px]">
                <Chip>
                  <Thermometer className="w-3.5 h-3.5" /> +3 °C air temperature
                </Chip>
                <Chip>
                  <Waves className="w-3.5 h-3.5" /> +15 % water vapour
                </Chip>
                <Chip mono>8–20°N · 82–94°E</Chip>
              </p>
            </StepCard>
          </div>

          {/* 06 · Ripple */}
          <div className="h-screen flex items-end lg:items-center px-4 sm:px-10 lg:px-16 pb-8 lg:pb-0">
            <StepCard
              num={6}
              eyebrow="Watch the ripple"
              active={nearest === 6}
              title={[{ t: 'The change starts', c: 'font-extralight' }, { t: 'small,', c: SERIF }, { t: 'then travels.', c: 'font-extrabold' }]}
            >
              <p>
                Orange means more rain than the normal forecast, blue means less. The gold ring marks how far a noticeable change has reached: it starts at the bay and
                spreads at the speed weather actually moves.
              </p>
              {live && step === 4 && (
                <p key={live.title} className="mf-rise text-[14px] text-[#FAFAFA] border-l border-[#FBBF24]/70 pl-3">
                  {live.title}
                </p>
              )}
            </StepCard>
          </div>

          {/* 07 · Meaning */}
          <div className="h-screen flex items-center px-4 sm:px-10 lg:px-16">
            <div className={`relative max-w-3xl transition-all duration-700 ${nearest >= 7 ? 'opacity-100' : 'opacity-50'}`}>
              <span className="mf-numeral" style={{ '--n': 7 } as React.CSSProperties} aria-hidden="true">
                07
              </span>
              <p className="relative font-mono text-[11px] uppercase tracking-[0.2em] text-[#E5E5E5] mb-4">
                <span className="text-[#737373]">07 —</span> What one warm sea changes
              </p>
              <SplitReveal
                as="h2"
                className="relative text-[36px] sm:text-[52px] leading-[1.02] tracking-[-0.03em] text-[#FAFAFA] text-balance"
                segments={[{ t: 'One change today.', c: 'font-extralight' }, { t: 'A different monsoon', c: 'font-extrabold' }, { t: 'in ten days.', c: SERIF }]}
              />
              {summary ? (
                <div className="relative mt-9 grid sm:grid-cols-3 gap-3">
                  {[
                    {
                      value: <CountUp value={summary.peakDelta} decimals={1} signed />,
                      unit: 'mm/day',
                      body: `Biggest rain change, over ${summary.peakRegion} on day ${summary.peakDay}.`,
                    },
                    {
                      value: <CountUp value={summary.heavyDelta} signed />,
                      unit: 'boxes',
                      body: `Change in grid boxes with heavy rain, at its largest (day ${summary.heavyDeltaDay}).`,
                    },
                    {
                      value: <CountUp value={Math.round(summary.reachKm / 10) * 10} />,
                      unit: 'km',
                      body: 'How far from the bay a noticeable change travelled within 10 days.',
                    },
                  ].map((c) => (
                    <div key={c.unit} className="mf-card rounded-2xl p-5">
                      <div className="text-[38px] font-extrabold text-[#FAFAFA] leading-none tracking-[-0.02em]">
                        {c.value}
                        <span className="font-display italic font-normal text-[18px] text-[#E5E5E5]"> {c.unit}</span>
                      </div>
                      <p className="mt-3 text-[13.5px] font-light text-[#E5E5E5]">{c.body}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-8 h-28 rounded-2xl mf-skeleton" />
              )}
              <button onClick={enterLab} data-cursor="try it" data-magnetic className="relative mt-9 mf-press h-12 px-6 rounded-full bg-[#FAFAFA] text-[#171717] font-semibold text-[15px] inline-flex items-center gap-2 hover:bg-[#E5E5E5]">
                Try your own what-if <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="relative px-5 sm:px-10 lg:px-16 py-28 border-t border-[#262626] bg-[#171717]">
        <div className="max-w-6xl mx-auto">
          <p className="mf-reveal font-mono text-[11px] uppercase tracking-[0.2em] text-[#E5E5E5] mb-4">
            <span className="text-[#737373]">08 —</span> How it works
          </p>
          <SplitReveal
            as="h2"
            className="text-[34px] sm:text-[46px] leading-[1.04] tracking-[-0.03em] max-w-3xl text-[#FAFAFA] text-balance"
            segments={[{ t: 'Three steps from', c: 'font-extralight' }, { t: 'today’s sky', c: SERIF }, { t: 'to a multiverse of futures.', c: 'font-extrabold' }]}
          />
          <div className="mt-14 grid md:grid-cols-3 gap-4">
            {[
              {
                icon: <Radar className="w-5 h-5" />,
                title: 'Read today’s sky',
                body: 'Six variables (rain, temperature, wind east–west and north–south, pressure, water vapour) over the last 3 days, on a 32 × 36 grid of 1° boxes (~110 km).',
              },
              {
                icon: <Orbit className="w-5 h-5" />,
                title: 'Roll forward 10 days',
                body: 'The model predicts tomorrow, then uses that to predict the day after, ten times. It runs offline on an ordinary 4-core CPU.',
              },
              {
                icon: <GitFork className="w-5 h-5" />,
                title: 'Split the future',
                body: 'It runs twice: once as is, once with your change. The difference between the two is exactly what your change caused.',
              },
            ].map((c, i) => (
              <div key={c.title} className="mf-reveal mf-tile rounded-[28px] p-7" style={{ transitionDelay: `${i * 90}ms` }}>
                <div className="w-10 h-10 rounded-xl grid place-items-center bg-[#262626] text-[#FAFAFA] border border-[#404040]">{c.icon}</div>
                <div className="mt-6 font-mono text-[12px] text-[#737373]">0{i + 1}</div>
                <h3 className="mt-1 text-[21px] font-bold text-[#FAFAFA] tracking-tight">{c.title}</h3>
                <p className="mt-2 text-[14.5px] leading-relaxed font-light text-[#E5E5E5]">{c.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Guarantees */}
      <section className="px-5 sm:px-10 lg:px-16 py-24 bg-[#171717] border-t border-[#262626]">
        <div className="max-w-6xl mx-auto grid lg:grid-cols-[1fr_1.2fr] gap-12 items-start">
          <div>
            <p className="mf-reveal font-mono text-[11px] uppercase tracking-[0.2em] text-[#E5E5E5] mb-4">
              <span className="text-[#737373]">09 —</span> Built to be trusted
            </p>
            <SplitReveal
              as="h2"
              className="text-[32px] sm:text-[42px] leading-[1.04] tracking-[-0.03em] text-[#FAFAFA] text-balance"
              segments={[{ t: 'Rules the model is', c: 'font-extralight' }, { t: 'held to.', c: SERIF }]}
            />
            <p className="mf-reveal mt-5 text-[15px] font-light text-[#E5E5E5] leading-relaxed max-w-md">
              A what-if is only useful if it behaves like real weather. These are the design rules the model and this dashboard follow.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 gap-3">
            {[
              { icon: <Lock className="w-4 h-4" />, title: 'Never peeks at the future', body: 'Trained on 2005–2018, tuned on 2019–2020, tested on a sealed 2021.' },
              { icon: <ShieldCheck className="w-4 h-4" />, title: 'No shock, no change', body: 'With nothing changed, both futures match to within one part in a million.' },
              { icon: <Gauge className="w-4 h-4" />, title: 'Physically possible', body: 'Rain is never negative; temperatures stay within −40 °C and +60 °C.' },
              { icon: <Cpu className="w-4 h-4" />, title: 'Fully offline', body: 'No weather services are called at run time. Even the globe and map are drawn from bundled data.' },
            ].map((g, i) => (
              <div key={g.title} className="mf-reveal mf-tile rounded-2xl p-5" style={{ transitionDelay: `${i * 80}ms` }}>
                <div className="flex items-center gap-2 text-[#FAFAFA]">
                  {g.icon}
                  <h3 className="text-[15px] font-bold">{g.title}</h3>
                </div>
                <p className="mt-2 text-[14px] font-light text-[#E5E5E5] leading-relaxed">{g.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="relative px-5 py-32 text-center overflow-hidden bg-[#171717] border-t border-[#262626]">
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,#262626,transparent_62%)]" />
        <SplitReveal
          as="h2"
          className="relative text-[40px] sm:text-[64px] leading-[1] tracking-[-0.035em] text-[#FAFAFA] text-balance"
          segments={[{ t: 'Your turn.', c: 'font-extrabold' }, { t: 'Change the', c: 'font-extralight' }, { t: 'weather.', c: SERIF }]}
        />
        <p className="mf-reveal relative mt-5 text-[16px] font-light text-[#E5E5E5] max-w-xl mx-auto">Pick a scenario or paint your own shock anywhere on the map, then press play.</p>
        <button onClick={enterLab} data-cursor="let’s go" data-magnetic className="mf-reveal relative mt-9 mf-press h-12 px-7 rounded-full bg-[#FAFAFA] text-[#171717] font-semibold text-[15px] inline-flex items-center gap-2 hover:bg-[#E5E5E5]">
          Open the lab <ArrowRight className="w-4 h-4" />
        </button>
      </section>

      <footer className="px-5 sm:px-10 py-8 border-t border-[#262626] text-[12.5px] font-light text-[#E5E5E5] flex flex-wrap items-center justify-between gap-3">
        <span className="flex items-center gap-2">
          <BrandMark tone="mono" className="w-5 h-5" /> MonsoonFlow-10 · Yugaantar DevForge 2026 · Multiverse of Monsoon
        </span>
        <span>Earth imagery: NASA Blue Marble Next Generation, July 2004 (public domain) · Coastlines: Natural Earth (public domain)</span>
      </footer>
    </div>
  );
}
