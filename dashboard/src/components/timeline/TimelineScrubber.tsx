/**
 * TimelineScrubber.tsx - Day 0 → Day 10 playback. Days are labelled with real calendar dates
 * derived from the Day-0 initial-state label (no dates are hard-coded), and the day ticks light up
 * in sequence whenever a new 10-day rollout arrives.
 */

import React, { useMemo } from 'react';
import { useTimeline } from '../../context/TimelineContext';
import { useClimateData } from '../../context/ClimateDataContext';
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from 'lucide-react';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** Parse a leading ISO date (YYYY-MM-DD) from the initial-state label, if present. */
function parseStart(label: string): Date | null {
  const m = /(\d{4})-(\d{2})-(\d{2})/.exec(label);
  if (!m) return null;
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
  return Number.isNaN(d.getTime()) ? null : d;
}

const fmtDay = (d: Date) => `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;

export const TimelineScrubber: React.FC = () => {
  const { currentDay, isPlaying, playbackSpeed, setCurrentDay, togglePlay, setPlaybackSpeed, stepForward, stepBackward, resetToDay0 } = useTimeline();
  const { apiStatus } = useClimateData();

  const start = useMemo(() => parseStart(apiStatus.initialStateLabel), [apiStatus.initialStateLabel]);
  const dateOf = (day: number) => (start ? fmtDay(new Date(start.getTime() + day * 86_400_000)) : null);
  const flashKey = apiStatus.lastSimulatedAt ? apiStatus.lastSimulatedAt.getTime() : 0;
  const days = Array.from({ length: 11 }, (_, i) => i);
  const pct = (currentDay / 10) * 100;

  return (
    <div className="px-3 lg:px-4 py-2.5 border-t border-white/[0.07] bg-[#03060C]/85 backdrop-blur-2xl flex flex-wrap lg:flex-nowrap items-center gap-3 lg:gap-6 select-none">
      {/* Transport */}
      <div className="flex items-center gap-1.5">
        <button onClick={resetToDay0} className="mf-press w-8 h-8 grid place-items-center rounded-lg text-zinc-400 hover:text-white hover:bg-white/10" title="Back to today (Day 0)" aria-label="Back to Day 0">
          <RotateCcw className="w-4 h-4" />
        </button>
        <button onClick={stepBackward} className="mf-press w-8 h-8 grid place-items-center rounded-lg text-zinc-400 hover:text-white hover:bg-white/10" title="One day back" aria-label="One day back">
          <SkipBack className="w-4 h-4" />
        </button>
        <button
          onClick={togglePlay}
          className={`mf-press h-9 px-4 rounded-xl font-medium text-[13px] flex items-center gap-2 transition-colors ${
            isPlaying ? 'bg-white/15 text-white' : 'bg-sky-400 text-[#03111f] hover:bg-sky-300 shadow-[0_0_24px_-6px_rgba(56,189,248,0.8)]'
          }`}
          title={isPlaying ? 'Pause' : 'Play the 10-day forecast'}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button onClick={stepForward} className="mf-press w-8 h-8 grid place-items-center rounded-lg text-zinc-400 hover:text-white hover:bg-white/10" title="One day forward" aria-label="One day forward">
          <SkipForward className="w-4 h-4" />
        </button>
        <div className="ml-1 mf-glass rounded-lg p-0.5 flex items-center text-[11.5px]" role="group" aria-label="Playback speed">
          {[0.5, 1, 2].map((s) => (
            <button
              key={s}
              onClick={() => setPlaybackSpeed(s)}
              className={`mf-press px-2 py-0.5 rounded-md ${playbackSpeed === s ? 'bg-white/[0.12] text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      {/* Track */}
      <div className="w-full lg:w-auto lg:flex-1 order-last lg:order-none">
        <div className="relative h-6 flex items-center">
          <div className="absolute inset-x-2 h-1.5 rounded-full bg-white/[0.08] overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-sky-400 via-cyan-300 to-amber-300 transition-[width] duration-500 ease-out"
              style={{ width: `${pct}%` }}
            />
          </div>
          <input
            type="range"
            min={0}
            max={10}
            step={1}
            value={currentDay}
            onChange={(e) => setCurrentDay(parseInt(e.target.value, 10))}
            aria-label="Forecast day"
            aria-valuetext={currentDay === 0 ? 'Today' : `Day ${currentDay}`}
            className="relative w-full h-6 appearance-none bg-transparent cursor-pointer opacity-0"
          />
          <div
            className="pointer-events-none absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-white shadow-[0_0_0_4px_rgba(56,189,248,0.25),0_0_16px_rgba(56,189,248,0.7)] transition-[left] duration-500 ease-out"
            style={{ left: `calc(8px + (100% - 16px) * ${currentDay / 10})` }}
          />
        </div>
        <div key={flashKey} className="flex justify-between mt-0.5 px-0.5">
          {days.map((d) => (
            <button
              key={d}
              onClick={() => setCurrentDay(d)}
              style={{ ['--i' as string]: d } as React.CSSProperties}
              className={`${flashKey ? 'mf-tick-flash' : ''} rounded-md px-1 py-0.5 text-center leading-tight transition-colors ${
                currentDay === d ? 'text-white' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              <span className={`block text-[11px] ${currentDay === d ? 'font-semibold' : ''}`}>{d === 0 ? 'Today' : `D${d}`}</span>
              {dateOf(d) && <span className="hidden md:block text-[9.5px] text-zinc-600">{dateOf(d)}</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Context */}
      <div className="hidden sm:block text-right shrink-0 min-w-[150px]">
        <div className="text-[13px] font-semibold text-white tabular-nums">
          {currentDay === 0 ? 'Today' : `Day ${currentDay}`}
          <span className="text-zinc-500 font-normal"> · +{currentDay * 24} h</span>
        </div>
        <div className={`text-[11px] ${apiStatus.initialStateIsSynthetic ? 'text-rose-300' : 'text-zinc-500'}`} title={apiStatus.initialStateLabel}>
          {start ? `Starts ${fmtDay(start)} ${start.getUTCFullYear()} · validation data` : apiStatus.initialStateLabel}
        </div>
      </div>
    </div>
  );
};
