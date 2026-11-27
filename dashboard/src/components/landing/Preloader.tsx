/**
 * Preloader.tsx - Honest loading screen: each line is a real task (3D engine chunk, NASA Earth
 * imagery, the Day-0 atmosphere + both 10-day forecasts). Leaves with an iris that opens on the
 * Earth's position. Never blocks for long: a hard cap reveals the page anyway; Skip is always
 * available. Returning visitors (same session) get a shorter hold.
 */

import React, { useEffect, useRef, useState } from 'react';
import { BrandMark } from '../ui/BrandMark';

export interface LoadStep {
  label: string;
  done: boolean;
  note?: string;
}

interface Props {
  steps: LoadStep[];
  /** Iris centre in CSS px. */
  origin: { x: number; y: number };
  reduced: boolean;
  onDone: () => void;
}

const HARD_CAP_MS = 7000;
const OUT_MS = 1150;

function seenBefore(): boolean {
  try {
    const s = sessionStorage.getItem('mf-intro-seen') === '1';
    sessionStorage.setItem('mf-intro-seen', '1');
    return s;
  } catch {
    return false;
  }
}

export const Preloader: React.FC<Props> = ({ steps, origin, reduced, onDone }) => {
  const [minHold] = useState(() => (seenBefore() || reduced ? 250 : 1100));
  const [t0] = useState(() => performance.now());
  const [leaving, setLeaving] = useState(false);
  const [capped, setCapped] = useState(false);
  const doneRef = useRef(onDone);
  useEffect(() => {
    doneRef.current = onDone;
  }, [onDone]);

  const done = steps.filter((s) => s.done).length;
  const allDone = done === steps.length;
  const pct = Math.round((done / steps.length) * 100);

  useEffect(() => {
    const id = window.setTimeout(() => setCapped(true), HARD_CAP_MS);
    return () => window.clearTimeout(id);
  }, []);

  useEffect(() => {
    if (leaving || !(allDone || capped)) return;
    const wait = Math.max(0, minHold - (performance.now() - t0));
    const id = window.setTimeout(() => setLeaving(true), wait);
    return () => window.clearTimeout(id);
  }, [allDone, capped, leaving, minHold, t0]);

  useEffect(() => {
    if (!leaving) return;
    const id = window.setTimeout(() => doneRef.current(), reduced ? 200 : OUT_MS);
    return () => window.clearTimeout(id);
  }, [leaving, reduced]);

  return (
    <div
      className={`mf-preloader ${leaving ? 'is-out' : ''}`}
      style={{ '--ox': `${origin.x}px`, '--oy': `${origin.y}px` } as React.CSSProperties}
      role="status"
      aria-live="polite"
      aria-label={`Loading MonsoonFlow-10, ${pct} percent`}
    >
      <div className="w-[min(420px,86vw)]">
        <div className="flex items-center gap-3">
          <BrandMark tone="mono" className="w-9 h-9" />
          <div className="text-[22px] leading-none text-[#FAFAFA]">
            <span className="font-extrabold tracking-tight">MonsoonFlow</span>
            <span className="font-display italic text-[26px] ml-1">-10</span>
          </div>
        </div>
        <p className="mt-6 text-[15px] font-extralight text-[#E5E5E5]">
          Assembling <span className="font-display italic text-[18px] text-[#FAFAFA]">today’s monsoon</span> from the model’s own tensors.
        </p>
        <div className="mt-6 h-px bg-[#404040] overflow-hidden">
          <div className="h-full bg-[#FAFAFA] transition-[width] duration-700 ease-out" style={{ width: `${Math.max(6, pct)}%` }} />
        </div>
        <ul className="mt-5 space-y-2 font-mono text-[11.5px] tracking-[0.08em] uppercase">
          {steps.map((s) => (
            <li key={s.label} className={`flex items-center justify-between gap-4 ${s.done ? 'text-[#E5E5E5]' : 'text-[#737373]'}`}>
              <span className="flex items-center gap-2.5">
                <span className={`w-1.5 h-1.5 rounded-full ${s.done ? 'bg-[#FAFAFA]' : 'border border-[#737373] mf-blink'}`} />
                {s.label}
              </span>
              <span className="tabular-nums">{s.done ? (s.note ?? 'ready') : '···'}</span>
            </li>
          ))}
        </ul>
        <div className="mt-8 flex items-center justify-between">
          <span className="font-mono text-[11px] tabular-nums text-[#737373]">{String(pct).padStart(3, '0')}%</span>
          <button type="button" onClick={() => setLeaving(true)} className="mf-press text-[12.5px] text-[#E5E5E5] underline underline-offset-4 decoration-[#737373] hover:decoration-[#FAFAFA]">
            Skip intro
          </button>
        </div>
      </div>
    </div>
  );
};
