/**
 * ColorbarLegend.tsx - Quantitative colour scale for the displayed channel (or anomaly),
 * with ticks at meteorologically meaningful values and a key for the active isolines.
 */

import React from 'react';
import { CHANNELS } from '../../types/climate';
import { useClimateData } from '../../context/ClimateDataContext';
import { LEGEND_SCALES, anomalyRgba, channelRgba, getColorForChannel, rgbToCss } from '../../services/colormaps';
import { CHANNEL_HUMAN } from '../ui/labels';

interface Props {
  isAnomaly?: boolean;
  /** Colour-scale half-range in wire units */
  maxAbsDelta?: number;
  contoursOn?: boolean;
}

const STEPS = 64;

const Swatch: React.FC<{ color: string; label: string }> = ({ color, label }) => (
  <span className="flex items-center gap-1">
    <span className="inline-block w-3 h-0.5 rounded" style={{ background: color, boxShadow: `0 0 4px ${color}` }} />
    {label}
  </span>
);

export const ColorbarLegend: React.FC<Props> = ({ isAnomaly = false, maxAbsDelta = 5, contoursOn = true }) => {
  const { selectedChannel } = useClimateData();
  const meta = CHANNELS[selectedChannel];
  const scale = LEGEND_SCALES[selectedChannel];

  const samples: string[] = [];
  for (let s = 0; s <= STEPS; s++) {
    if (isAnomaly) {
      const d = -maxAbsDelta + (2 * maxAbsDelta * s) / STEPS;
      const c = anomalyRgba(d, maxAbsDelta);
      samples.push(rgbToCss(c, c.a === 0 ? 0 : Math.max(0.35, c.a)));
    } else {
      const v = scale.min + ((scale.max - scale.min) * s) / STEPS;
      // Precipitation shows its real transparency for dry cells; other channels are opaque here
      if (selectedChannel === 'tp') {
        const c = channelRgba(v, 'tp');
        samples.push(rgbToCss(c, v < 0.5 ? 0 : Math.max(0.5, c.a)));
      } else {
        samples.push(rgbToCss(getColorForChannel(v, selectedChannel)));
      }
    }
  }
  const gradientCss = `linear-gradient(to right, ${samples.join(', ')})`;

  // Display-unit half range for the anomaly scale (msl is shown in hPa)
  const anomalyHalf = selectedChannel === 'msl' ? maxAbsDelta / 100 : maxAbsDelta;
  const anomalyDigits = anomalyHalf < 10 ? 2 : 1;

  return (
    <div className="mf-glass rounded-2xl p-2.5 sm:p-3 select-none text-xs w-64 sm:w-72" title={CHANNEL_HUMAN[meta.id].tech}>
      <div className="flex justify-between items-center mb-1.5 text-zinc-300">
        <span className="font-semibold text-zinc-100 truncate text-[12.5px]">
          {isAnomaly ? `${CHANNEL_HUMAN[meta.id].short} change vs normal` : CHANNEL_HUMAN[meta.id].long}
        </span>
        <span className="text-zinc-400 shrink-0 ml-2">{isAnomaly ? `Δ ${meta.displayUnit}` : scale.unit}</span>
      </div>

      <div
        className="h-3 w-full rounded border border-white/[0.12] bg-black/30"
        style={{ backgroundImage: gradientCss }}
      />

      <div className="relative h-4 mt-0.5 text-[11px] text-zinc-400 tabular-nums">
        {isAnomaly ? (
          <>
            <span className="absolute left-0">-{anomalyHalf.toFixed(anomalyDigits)}</span>
            <span className="absolute left-1/2 -translate-x-1/2">0</span>
            <span className="absolute right-0">+{anomalyHalf.toFixed(anomalyDigits)}</span>
          </>
        ) : (
          scale.ticks.map((t, k) => {
            const f = (t.value - scale.min) / (scale.max - scale.min);
            const align = k === 0 ? 'translate-x-0' : k === scale.ticks.length - 1 ? '-translate-x-full' : '-translate-x-1/2';
            return (
              <span key={t.value} className={`absolute ${align}`} style={{ left: `${f * 100}%` }}>
                {t.label}
              </span>
            );
          })
        )}
      </div>

      {contoursOn && (isAnomaly || selectedChannel === 'tp' || selectedChannel === 'msl') && (
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1 pt-1 border-t border-white/[0.12] text-[10px] text-zinc-400">
          {isAnomaly ? (
            <>
              <Swatch color="#FB923C" label="+ plume core (½ peak |Δ|)" />
              <Swatch color="#38BDF8" label="− plume core" />
            </>
          ) : selectedChannel === 'tp' ? (
            <>
              <Swatch color="#22D3EE" label="≥ 10 mm/d heavy" />
              <Swatch color="#FF3355" label="≥ 50 mm/d extreme" />
            </>
          ) : (
            <>
              <Swatch color="rgba(226, 232, 240, 0.6)" label="isobars every 2 hPa" />
              <Swatch color="#A5B4FC" label="≤ 1000 hPa deep low" />
            </>
          )}
        </div>
      )}
    </div>
  );
};
