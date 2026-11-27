/**
 * Lab.tsx - The interactive dashboard: full-bleed LCC map with collapsible glass side panels
 * (desktop) or drawers (mobile), header and timeline.
 */

import { useState } from 'react';
import { ShockLabProvider } from '../../context/ShockLabContext';
import { ClimateDataProvider } from '../../context/ClimateDataContext';
import { TimelineProvider } from '../../context/TimelineContext';
import { Header } from './Header';
import { ShockLabPanel } from '../multiverse/ShockLabPanel';
import { IndiaClimateMap } from '../map/IndiaClimateMap';
import { DownstreamImpactPanel } from '../downstream/DownstreamImpactPanel';
import { TimelineScrubber } from '../timeline/TimelineScrubber';
import { FlaskConical, PanelLeftOpen, PanelRightOpen, TrendingUp } from 'lucide-react';

const PANEL_W = 320;
const GAP = 12;

export interface LabProps {
  onBackToStory?: () => void;
  skipIntroFlyIn?: boolean;
}

export default function Lab({ onBackToStory, skipIntroFlyIn = false }: LabProps = {}) {
  const [mobilePanel, setMobilePanel] = useState<'NONE' | 'SHOCK' | 'IMPACTS'>('NONE');
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);

  const insetLeft = leftOpen ? PANEL_W + GAP : 0;
  const insetRight = rightOpen ? PANEL_W + GAP : 0;

  return (
    <ShockLabProvider>
      <ClimateDataProvider>
        <TimelineProvider>
          <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#03060C] text-zinc-100 select-none">
            <div className="shrink-0 z-40">
              <Header onBackToStory={onBackToStory} />
            </div>

            <main className="relative flex-1 min-h-0 overflow-hidden">
              <div className="absolute inset-0">
                <IndiaClimateMap insetLeft={insetLeft} insetRight={insetRight} skipIntroFlyIn={skipIntroFlyIn} />
              </div>

              {/* Mobile drawer toggles */}
              <div className="lg:hidden absolute top-2 left-3 z-30 flex items-center gap-2 text-[12px]">
                <button
                  onClick={() => setMobilePanel((p) => (p === 'SHOCK' ? 'NONE' : 'SHOCK'))}
                  className={`mf-glass mf-press px-3 h-8 rounded-xl flex items-center gap-1.5 ${mobilePanel === 'SHOCK' ? 'text-white' : 'text-zinc-300'}`}
                >
                  <FlaskConical className="w-4 h-4" />
                  What-if
                </button>
                <button
                  onClick={() => setMobilePanel((p) => (p === 'IMPACTS' ? 'NONE' : 'IMPACTS'))}
                  className={`mf-glass mf-press px-3 h-8 rounded-xl flex items-center gap-1.5 ${mobilePanel === 'IMPACTS' ? 'text-white' : 'text-zinc-300'}`}
                >
                  <TrendingUp className="w-4 h-4" />
                  Impacts
                </button>
              </div>

              {/* Left panel */}
              {(leftOpen || mobilePanel === 'SHOCK') && (
                <div className={`${mobilePanel === 'SHOCK' ? 'flex' : 'hidden lg:flex'} absolute left-3 top-3 bottom-3 z-[35] lg:z-30`}>
                  <ShockLabPanel onClose={() => setMobilePanel('NONE')} onCollapse={() => setLeftOpen(false)} />
                </div>
              )}
              {!leftOpen && (
                <button
                  onClick={() => setLeftOpen(true)}
                  className="hidden lg:flex mf-glass mf-press mf-slide-left absolute left-3 top-1/2 -translate-y-1/2 z-30 flex-col items-center gap-2 px-2 py-3 rounded-xl text-zinc-300 hover:text-white"
                  title="Show the what-if panel"
                >
                  <PanelLeftOpen className="w-4 h-4" />
                  <span className="text-[11.5px] [writing-mode:vertical-rl] rotate-180">What-if</span>
                </button>
              )}

              {/* Right panel */}
              {(rightOpen || mobilePanel === 'IMPACTS') && (
                <div className={`${mobilePanel === 'IMPACTS' ? 'flex' : 'hidden lg:flex'} absolute right-3 top-3 bottom-3 z-[35] lg:z-30`}>
                  <DownstreamImpactPanel onClose={() => setMobilePanel('NONE')} onCollapse={() => setRightOpen(false)} />
                </div>
              )}
              {!rightOpen && (
                <button
                  onClick={() => setRightOpen(true)}
                  className="hidden lg:flex mf-glass mf-press mf-slide-right absolute right-3 top-1/2 -translate-y-1/2 z-30 flex-col items-center gap-2 px-2 py-3 rounded-xl text-zinc-300 hover:text-white"
                  title="Show the impacts panel"
                >
                  <PanelRightOpen className="w-4 h-4" />
                  <span className="text-[11.5px] [writing-mode:vertical-rl]">Impacts</span>
                </button>
              )}
            </main>

            <div className="shrink-0 z-40">
              <TimelineScrubber />
            </div>
          </div>
        </TimelineProvider>
      </ClimateDataProvider>
    </ShockLabProvider>
  );
}
