/**
 * Header.tsx - Brand, three plain-language modes (Explore / What-if / Compare), data source
 * switch, and ONE provenance chip (DEMO vs LIVE is stated once, not three times).
 */

import React from 'react';
import { useClimateData, type ViewMode } from '../../context/ClimateDataContext';
import { AlertTriangle, ArrowLeft, Columns2, Compass, Diff, FlaskConical, GitCompareArrows, Zap } from 'lucide-react';
import { BrandMark } from '../ui/BrandMark';

type Mode = 'EXPLORE' | 'WHATIF' | 'COMPARE';

const modeOf = (v: ViewMode): Mode => (v === 'BASELINE' ? 'EXPLORE' : v === 'SHOCK' ? 'WHATIF' : 'COMPARE');

export const Header: React.FC<{ onBackToStory?: () => void }> = ({ onBackToStory }) => {
  const { apiStatus, dataMode, setDataMode, viewMode, setViewMode } = useClimateData();
  const mode = modeOf(viewMode);

  const modes: { id: Mode; label: string; hint: string; icon: React.ReactNode; to: ViewMode }[] = [
    { id: 'EXPLORE', label: 'Explore', hint: 'The normal forecast, with no shock (P = 0)', icon: <Compass className="w-4 h-4" />, to: 'BASELINE' },
    { id: 'WHATIF', label: 'What-if', hint: 'The forecast after your shock is applied', icon: <FlaskConical className="w-4 h-4" />, to: 'SHOCK' },
    {
      id: 'COMPARE',
      label: 'Compare',
      hint: 'Normal vs shocked, side by side or as a difference',
      icon: <GitCompareArrows className="w-4 h-4" />,
      to: viewMode === 'ANOMALY_DIFF' ? 'ANOMALY_DIFF' : 'SPLIT',
    },
  ];

  const status = (() => {
    if (dataMode === 'DEMO') {
      return {
        tone: 'border-amber-500/40 bg-amber-500/10 text-amber-200',
        dot: 'bg-amber-300',
        text: 'Demo data',
        title: `Synthetic toy advection engine, not the trained MF10 model. Backend ${apiStatus.endpoint} is ${apiStatus.serverOnline ? 'online' : 'offline'}.`,
        warn: false,
      };
    }
    if (!apiStatus.isLive || apiStatus.errorMessage) {
      return {
        tone: 'border-rose-500/40 bg-rose-500/10 text-rose-200',
        dot: 'bg-rose-400',
        text: apiStatus.isSimulating ? 'Connecting…' : 'Model offline',
        title: apiStatus.errorMessage || `Waiting for ${apiStatus.endpoint}`,
        warn: true,
      };
    }
    if (apiStatus.isStubResponse) {
      return {
        tone: 'border-amber-500/40 bg-amber-500/10 text-amber-200',
        dot: 'bg-amber-300',
        text: `Stub · ${apiStatus.modelName}`,
        title: apiStatus.stubReason || 'Backend returned a constant persistence rollout',
        warn: true,
      };
    }
    return {
      tone: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
      dot: 'bg-emerald-400 mf-ping',
      text: apiStatus.modelName,
      title: `Live model at ${apiStatus.endpoint}`,
      warn: false,
    };
  })();

  return (
    <header className="h-auto lg:h-14 px-3 lg:px-4 py-2 lg:py-0 flex flex-wrap lg:flex-nowrap items-center justify-between gap-2 border-b border-white/[0.07] bg-[#03060C]/80 backdrop-blur-2xl select-none">
      {/* Brand */}
      <div className="flex items-center gap-2.5 shrink-0">
        {onBackToStory ? (
          <button
            type="button"
            onClick={onBackToStory}
            className="mf-press flex items-center gap-1.5 text-zinc-500 hover:text-zinc-200 text-[12px] pr-2.5 border-r border-white/10"
            title="Back to the story"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span className="hidden md:inline">Story</span>
          </button>
        ) : (
          <a href="#/" className="mf-press flex items-center gap-1.5 text-zinc-500 hover:text-zinc-200 text-[12px] pr-2.5 border-r border-white/10" title="Back to the story">
            <ArrowLeft className="w-3.5 h-3.5" />
            <span className="hidden md:inline">Story</span>
          </a>
        )}
        <BrandMark className="w-6 h-6" />
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight text-zinc-50">MonsoonFlow-10</div>
          <div className="hidden xl:block text-[10.5px] text-zinc-500">10-day monsoon multiverse · India</div>
        </div>
      </div>

      {/* Modes */}
      <div className="flex items-center gap-2 order-last lg:order-none w-full lg:w-auto justify-center">
        <nav className="mf-glass rounded-xl p-0.5 flex items-center" aria-label="View mode">
          {modes.map((m) => (
            <button
              key={m.id}
              onClick={() => setViewMode(m.to)}
              title={m.hint}
              aria-pressed={mode === m.id}
              className={`mf-press h-8 px-3 rounded-[10px] text-[13px] flex items-center gap-1.5 transition-colors ${
                mode === m.id ? 'bg-white/[0.12] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.12)]' : 'text-zinc-400 hover:text-zinc-100'
              }`}
            >
              <span className={mode === m.id ? 'text-sky-300' : ''}>{m.icon}</span>
              {m.label}
            </button>
          ))}
        </nav>
        {mode === 'COMPARE' && (
          <div className="mf-glass mf-pop rounded-xl p-0.5 flex items-center" aria-label="Compare style">
            <button
              onClick={() => setViewMode('SPLIT')}
              title="Normal forecast on the left, shocked on the right"
              className={`mf-press h-8 px-2.5 rounded-[10px] text-[12.5px] flex items-center gap-1.5 ${viewMode === 'SPLIT' ? 'bg-white/[0.12] text-white' : 'text-zinc-400 hover:text-zinc-100'}`}
            >
              <Columns2 className="w-3.5 h-3.5" />
              Side by side
            </button>
            <button
              onClick={() => setViewMode('ANOMALY_DIFF')}
              title="Only the change: shocked minus normal (ΔXt)"
              className={`mf-press h-8 px-2.5 rounded-[10px] text-[12.5px] flex items-center gap-1.5 ${viewMode === 'ANOMALY_DIFF' ? 'bg-white/[0.12] text-white' : 'text-zinc-400 hover:text-zinc-100'}`}
            >
              <Diff className="w-3.5 h-3.5" />
              Difference
            </button>
          </div>
        )}
      </div>

      {/* Data source + status */}
      <div className="flex items-center gap-2 text-[12px]">
        <div className="mf-glass rounded-xl p-0.5 flex items-center" role="group" aria-label="Data source">
          <button
            onClick={() => setDataMode('DEMO')}
            className={`mf-press h-7 px-2.5 rounded-[9px] ${dataMode === 'DEMO' ? 'bg-amber-500/20 text-amber-100' : 'text-zinc-400 hover:text-zinc-100'}`}
            title="Synthetic local toy model (clearly marked as demo)"
          >
            Demo
          </button>
          <button
            onClick={() => setDataMode('LIVE')}
            className={`mf-press h-7 px-2.5 rounded-[9px] flex items-center gap-1.5 ${dataMode === 'LIVE' ? 'bg-emerald-500/20 text-emerald-100' : 'text-zinc-400 hover:text-zinc-100'}`}
            title={`Use the trained model at ${apiStatus.endpoint}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${apiStatus.serverOnline ? 'bg-emerald-400' : 'bg-zinc-500'}`} />
            Live model
          </button>
        </div>

        {dataMode === 'LIVE' && apiStatus.latencyMs !== null && (
          <div className="hidden md:flex items-center gap-1 px-2 h-8 rounded-xl mf-glass text-zinc-300 tabular-nums" title="inference_ms reported by serve.py /simulate">
            <Zap className="w-3.5 h-3.5 text-amber-300" />
            {apiStatus.latencyMs} ms
          </div>
        )}

        <div
          data-testid={dataMode === 'DEMO' ? 'demo-watermark' : undefined}
          className={`flex items-center gap-1.5 h-8 px-2.5 rounded-xl border ${status.tone}`}
          title={status.title}
        >
          {status.warn ? <AlertTriangle className="w-3.5 h-3.5 shrink-0" /> : <span className={`w-2 h-2 rounded-full ${status.dot}`} />}
          <span className="font-medium max-w-[11rem] truncate">{status.text}</span>
        </div>
      </div>
    </header>
  );
};
