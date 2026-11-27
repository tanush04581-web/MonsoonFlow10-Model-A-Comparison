/**
 * ChannelSelector.tsx - Which variable the map shows. Plain names first; the technical id and
 * unit live in the tooltip.
 */

import React from 'react';
import { CHANNEL_LIST, type ChannelId } from '../../types/climate';
import { useClimateData } from '../../context/ClimateDataContext';
import { CHANNEL_HUMAN } from '../ui/labels';
import { CloudRain, Droplets, Gauge, MoveHorizontal, MoveVertical, Thermometer } from 'lucide-react';

const ICONS: Record<ChannelId, React.ReactNode> = {
  tp: <CloudRain className="w-3.5 h-3.5" />,
  t2m: <Thermometer className="w-3.5 h-3.5" />,
  u10: <MoveHorizontal className="w-3.5 h-3.5" />,
  v10: <MoveVertical className="w-3.5 h-3.5" />,
  msl: <Gauge className="w-3.5 h-3.5" />,
  tcwv: <Droplets className="w-3.5 h-3.5" />,
};

export const ChannelSelector: React.FC = () => {
  const { selectedChannel, setSelectedChannel } = useClimateData();

  return (
    <div className="mf-glass rounded-xl p-0.5 flex flex-wrap items-center gap-0.5 max-w-full" role="radiogroup" aria-label="Map variable">
      {CHANNEL_LIST.map((ch) => {
        const isSelected = selectedChannel === ch.id;
        return (
          <button
            key={ch.id}
            role="radio"
            aria-checked={isSelected}
            onClick={() => setSelectedChannel(ch.id)}
            title={CHANNEL_HUMAN[ch.id].tech}
            className={`mf-press h-[30px] px-2.5 text-[12.5px] rounded-[10px] transition-colors flex items-center gap-1.5 ${
              isSelected ? 'bg-white/[0.12] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.12)]' : 'text-zinc-400 hover:text-zinc-100'
            }`}
          >
            <span className={isSelected ? 'text-sky-300' : ''}>{ICONS[ch.id]}</span>
            <span className={isSelected ? 'inline' : 'hidden xl:inline'}>{CHANNEL_HUMAN[ch.id].short}</span>
          </button>
        );
      })}
    </div>
  );
};
