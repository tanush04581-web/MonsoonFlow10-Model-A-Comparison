/**
 * NarrativeBanner.tsx - One plain-language sentence about what the map shows right now,
 * generated from the displayed tensors (services/diagnostics.ts). Re-animates when it changes.
 */

import React from 'react';
import type { Headline } from '../../services/diagnostics';
import { CloudRain, Flame, Loader2, Snowflake, Sparkles, Sun } from 'lucide-react';

interface Props {
  headline: Headline;
  busy: boolean;
  unavailable: boolean;
  synthetic: boolean;
}

const TONE: Record<Headline['tone'], { icon: React.ReactNode; ring: string }> = {
  neutral: { icon: <Sparkles className="w-4 h-4 text-sky-300" />, ring: 'from-sky-400/40 to-transparent' },
  wet: { icon: <CloudRain className="w-4 h-4 text-cyan-300" />, ring: 'from-cyan-400/50 to-transparent' },
  dry: { icon: <Sun className="w-4 h-4 text-amber-300" />, ring: 'from-amber-400/50 to-transparent' },
  warm: { icon: <Flame className="w-4 h-4 text-orange-300" />, ring: 'from-orange-400/50 to-transparent' },
  cool: { icon: <Snowflake className="w-4 h-4 text-sky-300" />, ring: 'from-sky-400/50 to-transparent' },
};

export const NarrativeBanner: React.FC<Props> = ({ headline, busy, unavailable, synthetic }) => {
  const tone = TONE[headline.tone];
  const title = unavailable ? 'No forecast for this day yet' : headline.title;
  const detail = unavailable
    ? 'The live model has not returned a rollout. Nothing is shown rather than guessing.'
    : headline.detail;

  return (
    <div className="pointer-events-auto mx-auto w-full max-w-3xl">
      <div className={`relative rounded-2xl p-px bg-gradient-to-r ${tone.ring}`}>
        <div className="mf-glass-strong relative rounded-2xl px-4 py-2.5 flex items-start gap-3 overflow-hidden">
          <div className="mt-0.5 shrink-0">{busy ? <Loader2 className="w-4 h-4 text-zinc-300 animate-spin" /> : tone.icon}</div>
          <div key={title} className="min-w-0 flex-1 mf-rise">
            <p className="text-[14px] sm:text-[15px] leading-snug font-semibold text-zinc-50 text-balance">{title}</p>
            <p className="text-[12px] sm:text-[12.5px] leading-snug text-zinc-400 mt-0.5">{detail}</p>
          </div>
          {synthetic && (
            <span
              className="shrink-0 mt-0.5 text-[10.5px] px-2 py-0.5 rounded-full border border-amber-500/40 bg-amber-500/10 text-amber-200"
              title="Demo mode uses a simple local advection toy model, not the trained MonsoonFlow-10. Switch to Live for real model output."
            >
              demo data
            </span>
          )}
          {busy && <div className="absolute inset-x-0 bottom-0 h-px mf-shimmer-line" />}
        </div>
      </div>
    </div>
  );
};
