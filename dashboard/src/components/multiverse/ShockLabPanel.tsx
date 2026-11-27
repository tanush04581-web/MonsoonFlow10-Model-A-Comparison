/**
 * ShockLabPanel.tsx - "Create a what-if": preset scenarios in plain words, plus a design-your-own
 * shock (Gaussian brush). Exact technical definitions stay one hover away.
 */

import React, { useState } from 'react';
import { AMPLITUDE_RANGE, SHOCK_PRESETS, type PresetShockId } from '../../types/shocks';
import { useShockLab } from '../../context/ShockLabContext';
import { useClimateData } from '../../context/ClimateDataContext';
import { CHANNEL_LIST, CHANNELS, type ChannelId } from '../../types/climate';
import { CHANNEL_HUMAN } from '../ui/labels';
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  CircleSlash,
  CloudRainWind,
  Crosshair,
  PanelLeftClose,
  RotateCcw,
  ShieldCheck,
  ThermometerSun,
  Wind,
  X,
} from 'lucide-react';

interface ShockLabPanelProps {
  onClose?: () => void;
  onCollapse?: () => void;
}

const PRESET_COPY: Record<Exclude<PresetShockId, 'CUSTOM'>, { title: string; plain: string; icon: React.ReactNode; accent: string }> = {
  WARM_BAY_OF_BENGAL: {
    title: 'Warmer Bay of Bengal',
    plain: 'The bay gets 3 °C warmer and 15 % more humid. Where does the extra rain go?',
    icon: <ThermometerSun className="w-4 h-4" />,
    accent: 'text-orange-300 bg-orange-500/10 border-orange-500/30',
  },
  MONSOON_STALL: {
    title: 'Monsoon winds stall',
    plain: 'Winds over the Arabian Sea weaken by 40 %. Does the west coast dry out?',
    icon: <Wind className="w-4 h-4" />,
    accent: 'text-sky-300 bg-sky-500/10 border-sky-500/30',
  },
  WESTERN_GHATS_DELUGE: {
    title: 'Western Ghats cloudburst',
    plain: 'A burst of +30 mm/day rain on the Konkan–Goa coast. How far inland does it reach?',
    icon: <CloudRainWind className="w-4 h-4" />,
    accent: 'text-cyan-300 bg-cyan-500/10 border-cyan-500/30',
  },
  ZERO_SHOCK: {
    title: 'No shock',
    plain: 'The normal forecast. Both futures are identical.',
    icon: <CircleSlash className="w-4 h-4" />,
    accent: 'text-zinc-300 bg-white/[0.05] border-white/15',
  },
};

/** Amplitude in display units (msl is Pa on the wire, shown in hPa; a K delta equals a °C delta). */
function amplitudeLabel(channel: ChannelId, value: number): string {
  const sign = value > 0 ? '+' : value < 0 ? '−' : '';
  const abs = Math.abs(value);
  if (channel === 'msl') return `${sign}${(abs / 100).toFixed(abs % 100 === 0 ? 0 : 1)} hPa`;
  if (channel === 't2m') return `${sign}${abs.toFixed(1)} °C`;
  return `${sign}${abs} ${CHANNELS[channel].displayUnit}`;
}

export const ShockLabPanel: React.FC<ShockLabPanelProps> = ({ onClose, onCollapse }) => {
  const { activePresetId, customSpec, isBrushMode, boundsReport, selectPreset, updateCustomSpec, toggleBrushMode, resetShock } = useShockLab();
  const { viewMode, setViewMode } = useClimateData();
  const [designOpen, setDesignOpen] = useState(activePresetId === 'CUSTOM');
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const choose = (id: PresetShockId) => {
    selectPreset(id);
    // Picking a shock while exploring the normal forecast should show the shock
    if (id !== 'ZERO_SHOCK' && viewMode === 'BASELINE') setViewMode('SHOCK');
  };

  const range = AMPLITUDE_RANGE[customSpec.channel];

  return (
    <aside className="mf-glass mf-slide-left w-80 max-w-[calc(100vw-24px)] h-full rounded-2xl flex flex-col overflow-hidden select-none text-[13px]">
      {/* Header */}
      <div className="px-4 pt-4 pb-3 flex items-start justify-between gap-2 border-b border-white/[0.07]">
        <div>
          <h2 className="text-[15px] font-semibold text-zinc-50 tracking-tight">Create a what-if</h2>
          <p className="text-[12px] text-zinc-400 mt-0.5">Change today’s weather and watch 10 days unfold.</p>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={resetShock} className="mf-press p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-white/10" title="Reset: no shock (P = 0)" aria-label="Reset shock">
            <RotateCcw className="w-4 h-4" />
          </button>
          {onCollapse && (
            <button onClick={onCollapse} className="hidden lg:block mf-press p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-white/10" title="Hide panel" aria-label="Hide panel">
              <PanelLeftClose className="w-4 h-4" />
            </button>
          )}
          {onClose && (
            <button onClick={onClose} className="lg:hidden mf-press p-1.5 rounded-lg bg-white/10 text-zinc-300 hover:text-white" aria-label="Close panel">
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      <div data-lenis-prevent className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
        <p className="px-1 text-[11px] uppercase tracking-[0.12em] text-zinc-500">Scenarios</p>
        {SHOCK_PRESETS.map((preset, idx) => {
          const copy = PRESET_COPY[preset.id as Exclude<PresetShockId, 'CUSTOM'>];
          const isSelected = activePresetId === preset.id;
          return (
            <button
              key={preset.id}
              onClick={() => choose(preset.id)}
              title={preset.description}
              aria-pressed={isSelected}
              style={{ ['--i' as string]: idx } as React.CSSProperties}
              className={`mf-enter mf-press w-full text-left p-3 rounded-xl border transition-colors relative overflow-hidden ${
                isSelected ? 'bg-white/[0.08] border-white/25' : 'bg-white/[0.025] border-white/[0.07] hover:bg-white/[0.05] hover:border-white/15'
              }`}
            >
              {isSelected && <span className="absolute left-0 top-3 bottom-3 w-[3px] rounded-r bg-sky-400" />}
              <div className="flex items-start gap-2.5">
                <span className={`mt-0.5 shrink-0 w-7 h-7 rounded-lg grid place-items-center border ${copy.accent}`}>{copy.icon}</span>
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="font-semibold text-zinc-100">{copy.title}</span>
                    {preset.graded && (
                      <span
                        className="inline-flex items-center gap-0.5 text-[10px] px-1.5 py-px rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/25"
                        title="One of the two shocks the competition grader applies (monsoon_r1.scoring.make_shock)"
                      >
                        <ShieldCheck className="w-3 h-3" /> official test
                      </span>
                    )}
                  </div>
                  <p className="text-[12px] text-zinc-400 leading-relaxed mt-0.5">{copy.plain}</p>
                </div>
              </div>
            </button>
          );
        })}

        {/* Design your own */}
        <div className={`rounded-xl border mt-3 ${activePresetId === 'CUSTOM' ? 'border-sky-500/40 bg-sky-500/[0.04]' : 'border-white/[0.07] bg-white/[0.02]'}`}>
          <button onClick={() => setDesignOpen((o) => !o)} className="mf-press w-full flex items-center justify-between px-3 py-2.5 text-left" aria-expanded={designOpen}>
            <span className="font-semibold text-zinc-100">
              Design your own shock
              {activePresetId === 'CUSTOM' && <span className="ml-1.5 text-[11px] font-normal text-sky-300">active</span>}
            </span>
            <ChevronDown className={`w-4 h-4 text-zinc-400 transition-transform ${designOpen ? 'rotate-180' : ''}`} />
          </button>
          {designOpen && (
            <div className="px-3 pb-3 space-y-3 mf-pop">
              <button
                onClick={toggleBrushMode}
                className={`mf-press w-full h-9 rounded-lg border flex items-center justify-center gap-2 font-medium transition-colors ${
                  isBrushMode ? 'bg-orange-500/20 border-orange-400/50 text-orange-100' : 'bg-white/[0.05] border-white/15 text-zinc-200 hover:bg-white/10'
                }`}
              >
                <Crosshair className="w-4 h-4" />
                {isBrushMode ? 'Click the map to place it…' : 'Place it on the map'}
              </button>

              <label className="block">
                <span className="text-[12px] text-zinc-400">What changes</span>
                <select
                  value={customSpec.channel}
                  onChange={(e) => {
                    const channel = e.target.value as ChannelId;
                    const r = AMPLITUDE_RANGE[channel];
                    updateCustomSpec({ channel, magnitude: Math.max(r.min, Math.min(r.max, customSpec.magnitude)) });
                  }}
                  className="mt-1 w-full h-9 bg-black/40 border border-white/[0.12] rounded-lg px-2 text-zinc-100"
                >
                  {CHANNEL_LIST.map((c) => (
                    <option key={c.id} value={c.id}>
                      {CHANNEL_HUMAN[c.id].long}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block">
                <span className="flex justify-between text-[12px] text-zinc-400">
                  <span>Strength</span>
                  <span className={`font-semibold tabular-nums ${customSpec.magnitude > 0 ? 'text-orange-300' : customSpec.magnitude < 0 ? 'text-sky-300' : 'text-zinc-300'}`}>
                    {amplitudeLabel(customSpec.channel, customSpec.magnitude)}
                  </span>
                </span>
                <input
                  type="range"
                  min={range.min}
                  max={range.max}
                  step={range.step}
                  value={customSpec.magnitude}
                  onChange={(e) => updateCustomSpec({ magnitude: parseFloat(e.target.value) })}
                  className="mt-1.5 w-full cursor-pointer"
                  aria-label="Shock strength"
                />
              </label>

              <label className="block">
                <span className="flex justify-between text-[12px] text-zinc-400">
                  <span>Size</span>
                  <span className="text-zinc-200 tabular-nums">{customSpec.sigmaKm} km</span>
                </span>
                <input
                  type="range"
                  min="100"
                  max="600"
                  step="25"
                  value={customSpec.sigmaKm}
                  onChange={(e) => updateCustomSpec({ sigmaKm: parseInt(e.target.value, 10) })}
                  className="mt-1.5 w-full cursor-pointer"
                  aria-label="Shock size (Gaussian sigma)"
                  title="Gaussian σ. About 300 km (~2.5 grid cells) keeps the shock smooth."
                />
              </label>

              <button onClick={() => setAdvancedOpen((o) => !o)} className="text-[12px] text-zinc-500 hover:text-zinc-300 flex items-center gap-1" aria-expanded={advancedOpen}>
                <ChevronDown className={`w-3.5 h-3.5 transition-transform ${advancedOpen ? 'rotate-180' : ''}`} />
                Exact position
              </button>
              {advancedOpen && (
                <div className="grid grid-cols-2 gap-2 mf-pop">
                  <label className="block">
                    <span className="text-[11px] text-zinc-500">Latitude °N</span>
                    <input
                      type="number"
                      step="0.5"
                      min="6.5"
                      max="37.5"
                      value={customSpec.centerLat}
                      onChange={(e) => updateCustomSpec({ centerLat: parseFloat(e.target.value) || 20 })}
                      className="mt-0.5 w-full h-8 bg-black/40 border border-white/[0.12] rounded-lg px-2 text-zinc-100 font-mono text-[12px]"
                    />
                  </label>
                  <label className="block">
                    <span className="text-[11px] text-zinc-500">Longitude °E</span>
                    <input
                      type="number"
                      step="0.5"
                      min="65.5"
                      max="100.5"
                      value={customSpec.centerLon}
                      onChange={(e) => updateCustomSpec({ centerLon: parseFloat(e.target.value) || 80 })}
                      className="mt-0.5 w-full h-8 bg-black/40 border border-white/[0.12] rounded-lg px-2 text-zinc-100 font-mono text-[12px]"
                    />
                  </label>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Physical realism */}
      <div className="px-3 pb-3 pt-2 border-t border-white/[0.07]">
        {boundsReport.isValid ? (
          <div className="flex items-start gap-2 p-2.5 rounded-xl bg-emerald-500/[0.07] border border-emerald-500/20 text-emerald-200 text-[12px]">
            <CheckCircle2 className="w-4 h-4 shrink-0 mt-px" />
            <span>Physically plausible: this shock stays within realistic bounds.</span>
          </div>
        ) : (
          <div className="p-2.5 rounded-xl bg-amber-500/[0.08] border border-amber-500/25 text-amber-200 text-[12px]">
            <div className="flex items-center gap-1.5 font-semibold mb-1">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>Stronger than nature usually allows</span>
            </div>
            {boundsReport.warnings.map((w, idx) => (
              <p key={idx} className="text-[11.5px] text-amber-200/80 leading-snug">
                • {w}
              </p>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
};
