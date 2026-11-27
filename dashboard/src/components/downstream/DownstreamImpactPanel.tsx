/**
 * DownstreamImpactPanel.tsx - "What it means on the ground": flood-risk and crop-stress indices,
 * shown as a level AND as the change versus the normal forecast (so "100/100, Δ 0" can never
 * look like an alarm caused by the shock), rain counts, and region cards.
 * All numbers are computed from the displayed rollout tensors (services/impactCalculator.ts).
 */

import React, { useMemo } from 'react';
import { useClimateData } from '../../context/ClimateDataContext';
import { useTimeline } from '../../context/TimelineContext';
import { useShockLab } from '../../context/ShockLabContext';
import { calculateDownstreamImpacts } from '../../services/impactCalculator';
import { ShockTrajectoryChart } from '../anomaly/ShockTrajectoryChart';
import { CountUp } from '../ui/CountUp';
import { AlertTriangle, ArrowDownRight, ArrowUpRight, CloudRain, Info, Minus, PanelRightClose, Sprout, X } from 'lucide-react';

interface DownstreamImpactPanelProps {
  onClose?: () => void;
  onCollapse?: () => void;
}

const STATUS_STYLE: Record<string, string> = {
  Critical: 'bg-red-500/15 text-red-300 border-red-500/30',
  Warning: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  Advisory: 'bg-yellow-500/10 text-yellow-200 border-yellow-500/25',
  Watch: 'bg-sky-500/10 text-sky-200 border-sky-500/25',
  Normal: 'bg-white/[0.05] text-zinc-400 border-white/10',
};

const Change: React.FC<{ delta: number; unit?: string; decimals?: number }> = ({ delta, unit = '', decimals = 0 }) => {
  if (Math.abs(delta) < Math.pow(10, -decimals) / 2) {
    return (
      <span className="inline-flex items-center gap-0.5 text-zinc-500">
        <Minus className="w-3 h-3" /> no change
      </span>
    );
  }
  const up = delta > 0;
  return (
    <span className={`inline-flex items-center gap-0.5 font-medium ${up ? 'text-orange-300' : 'text-sky-300'}`}>
      {up ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
      <CountUp value={delta} decimals={decimals} signed suffix={unit} />
    </span>
  );
};

const Meter: React.FC<{ value: number; tone: string }> = ({ value, tone }) => (
  <div className="w-full bg-white/[0.07] h-1.5 rounded-full overflow-hidden">
    <div className={`h-full rounded-full transition-[width] duration-700 ease-out ${tone}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
  </div>
);

const stagger = (i: number) => ({ ['--i' as string]: i }) as React.CSSProperties;

export const DownstreamImpactPanel: React.FC<DownstreamImpactPanelProps> = ({ onClose, onCollapse }) => {
  const { baselineRollout, shockRollout, initialCondition, dataMode, apiStatus } = useClimateData();
  const { currentDay } = useTimeline();
  const { activePresetId } = useShockLab();
  const hasShock = activePresetId !== 'ZERO_SHOCK';

  const hasForecastForDay = currentDay === 0 || (baselineRollout.length >= currentDay && shockRollout.length >= currentDay);
  const liveUnavailable = dataMode === 'LIVE' && (Boolean(apiStatus.errorMessage) || (currentDay > 0 && !hasForecastForDay));

  const shockFrame = useMemo(
    () => (currentDay === 0 ? initialCondition : shockRollout[currentDay - 1] || initialCondition),
    [currentDay, shockRollout, initialCondition]
  );
  const baseFrame = useMemo(
    () => (currentDay === 0 ? initialCondition : baselineRollout[currentDay - 1] || initialCondition),
    [currentDay, baselineRollout, initialCondition]
  );

  const impacts = useMemo(() => calculateDownstreamImpacts(shockFrame, baseFrame), [shockFrame, baseFrame]);
  const normal = useMemo(() => calculateDownstreamImpacts(baseFrame, baseFrame), [baseFrame]);

  const floodTone = impacts.floodRiskIndex >= 70 ? 'bg-red-500' : impacts.floodRiskIndex >= 40 ? 'bg-amber-400' : 'bg-sky-400';
  const cropTone = impacts.cropMoistureStressIndex >= 60 ? 'bg-amber-400' : 'bg-emerald-400';

  return (
    <aside className="mf-glass mf-slide-right w-80 max-w-[calc(100vw-24px)] h-full rounded-2xl flex flex-col overflow-hidden select-none text-[13px]">
      <div className="px-4 pt-4 pb-3 flex items-start justify-between gap-2 border-b border-white/[0.07]">
        <div>
          <h2 className="text-[15px] font-semibold text-zinc-50 tracking-tight">What it means on the ground</h2>
          <p className="text-[12px] text-zinc-400 mt-0.5">
            {currentDay === 0 ? 'Today' : `Day ${currentDay}`}
            {hasShock ? ' · with shock, vs the normal forecast' : ' · normal forecast'}
          </p>
        </div>
        <div className="flex items-center gap-1">
          {onCollapse && (
            <button onClick={onCollapse} className="hidden lg:block mf-press p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-white/10" title="Hide panel" aria-label="Hide panel">
              <PanelRightClose className="w-4 h-4" />
            </button>
          )}
          {onClose && (
            <button onClick={onClose} className="lg:hidden mf-press p-1.5 rounded-lg bg-white/10 text-zinc-300 hover:text-white" aria-label="Close panel">
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      <div data-lenis-prevent className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
        {liveUnavailable && currentDay > 0 ? (
          <div className="rounded-xl border border-rose-500/30 bg-rose-500/[0.07] p-3 text-rose-200 space-y-1.5">
            <div className="flex items-center gap-1.5 font-semibold">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>Waiting for the live model</span>
            </div>
            <p className="text-[12px] text-rose-200/80 leading-relaxed">Impacts appear once {apiStatus.endpoint}/simulate returns a valid 10-day rollout.</p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-2">
              <div className="mf-enter rounded-xl border border-white/[0.07] bg-white/[0.03] p-3" style={stagger(0)}>
                <div className="flex items-center justify-between text-zinc-400 text-[12px] mb-1">
                  <span className="flex items-center gap-1">
                    Flood risk
                    <span title="Heuristic 0–100 index from heavy/extreme rain cells and basin-mean rain. For comparing scenarios; not an IMD warning.">
                      <Info className="w-3 h-3 text-zinc-600" />
                    </span>
                  </span>
                  <CloudRain className="w-4 h-4 text-sky-300" />
                </div>
                <div className="text-[24px] font-semibold text-zinc-50 leading-none">
                  <CountUp value={impacts.floodRiskIndex} />
                  <span className="text-[12px] font-normal text-zinc-500">/100</span>
                </div>
                <div className="my-2">
                  <Meter value={impacts.floodRiskIndex} tone={floodTone} />
                </div>
                <div className="text-[11.5px]">
                  {hasShock ? <Change delta={impacts.floodRiskIndex - normal.floodRiskIndex} /> : <span className="text-zinc-500">normal forecast</span>}
                </div>
              </div>
              <div className="mf-enter rounded-xl border border-white/[0.07] bg-white/[0.03] p-3" style={stagger(1)}>
                <div className="flex items-center justify-between text-zinc-400 text-[12px] mb-1">
                  <span className="flex items-center gap-1">
                    Crop stress
                    <span title="Heuristic 0–100 index: drought stress when basin rain < 5 mm/day, waterlogging when > 40 mm/day.">
                      <Info className="w-3 h-3 text-zinc-600" />
                    </span>
                  </span>
                  <Sprout className="w-4 h-4 text-emerald-300" />
                </div>
                <div className="text-[24px] font-semibold text-zinc-50 leading-none">
                  <CountUp value={impacts.cropMoistureStressIndex} />
                  <span className="text-[12px] font-normal text-zinc-500">/100</span>
                </div>
                <div className="my-2">
                  <Meter value={impacts.cropMoistureStressIndex} tone={cropTone} />
                </div>
                <div className="text-[11.5px]">
                  {hasShock ? (
                    <Change delta={impacts.cropMoistureStressIndex - normal.cropMoistureStressIndex} />
                  ) : (
                    <span className="text-zinc-500">normal forecast</span>
                  )}
                </div>
              </div>
            </div>

            <ShockTrajectoryChart />

            <div className="mf-enter rounded-xl border border-white/[0.07] bg-white/[0.03] p-3 space-y-2" style={stagger(3)}>
              <p className="text-[12px] font-medium text-zinc-300">Rain across the map</p>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-zinc-400 text-[12px]">Heavy rain (10+ mm/day)</span>
                <span className="text-right">
                  <span className="font-semibold text-sky-300">
                    <CountUp value={impacts.heavyRainCells} />
                  </span>
                  <span className="text-zinc-500 text-[11px]"> boxes</span>
                  {hasShock && (
                    <span className="block text-[11px]">
                      <Change delta={impacts.heavyRainCells - normal.heavyRainCells} />
                    </span>
                  )}
                </span>
              </div>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-zinc-400 text-[12px]">Extreme rain (50+ mm/day)</span>
                <span className="text-right">
                  <span className="font-semibold text-rose-300">
                    <CountUp value={impacts.extremeRainCells} />
                  </span>
                  <span className="text-zinc-500 text-[11px]"> boxes · {impacts.highRiskAreaKm2.toLocaleString()} km²</span>
                </span>
              </div>
            </div>

            <div className="space-y-1.5">
              <p className="px-1 text-[11px] uppercase tracking-[0.12em] text-zinc-500">Regions</p>
              {impacts.basins.map((basin, idx) => (
                <div key={basin.regionId} className="mf-enter rounded-xl border border-white/[0.07] bg-white/[0.025] p-2.5" style={stagger(4 + idx)}>
                  <div className="flex justify-between items-start gap-2 mb-1.5">
                    <span className="font-medium text-zinc-100 text-[12.5px] leading-snug">{basin.name}</span>
                    <span className={`shrink-0 text-[10.5px] px-1.5 py-0.5 rounded-full border ${STATUS_STYLE[basin.status]}`}>{basin.status}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1">
                      <Meter value={(basin.meanPrecipMm / 40) * 100} tone="bg-gradient-to-r from-cyan-400 to-blue-500" />
                    </div>
                    <span className="text-[11.5px] text-zinc-300 tabular-nums w-[78px] text-right">{basin.meanPrecipMm} mm/day</span>
                  </div>
                  {hasShock && (
                    <div className="mt-1 text-[11px]">
                      <Change delta={basin.anomalyPrecipMm} unit=" mm/day" decimals={1} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="px-4 py-2.5 border-t border-white/[0.07] text-[11px] text-zinc-500 leading-relaxed">
        {dataMode === 'DEMO' ? (
          <span className="text-amber-300/90">Demo data: a toy local engine, not the trained model. </span>
        ) : apiStatus.isStubResponse ? (
          <span className="text-amber-300/90">The live backend is returning a persistence stub. </span>
        ) : null}
        Indices are heuristics for comparing scenarios, not official warnings.
      </div>
    </aside>
  );
};
