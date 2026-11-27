/**
 * ChapterRail.tsx - Chapter progress: a vertical rail of ticks on desktop (click to jump) and a
 * hairline progress bar under the nav on small screens.
 */

import React from 'react';

export interface Chapter {
  label: string;
  screen: number;
}

interface Props {
  chapters: Chapter[];
  active: number;
  /** Mobile progress bar element; its scaleX is written directly by the scroll handler. */
  barRef: React.RefObject<HTMLDivElement | null>;
  onJump: (screen: number) => void;
}

export const ChapterRail: React.FC<Props> = ({ chapters, active, barRef, onJump }) => (
  <>
    <nav aria-label="Story chapters" className="hidden lg:flex fixed right-6 top-1/2 -translate-y-1/2 z-40 flex-col items-end gap-3.5">
      {chapters.map((c, i) => {
        const on = i === active;
        return (
          <button
            key={c.label}
            type="button"
            onClick={() => onJump(c.screen)}
            aria-label={`Chapter ${String(i).padStart(2, '0')}: ${c.label}`}
            aria-current={on ? 'step' : undefined}
            className="group flex items-center gap-3 h-5"
            data-cursor={c.label.toLowerCase()}
          >
            <span
              className={`font-mono text-[10.5px] uppercase tracking-[0.14em] transition-all duration-500 ${
                on ? 'opacity-100 text-[#FAFAFA] translate-x-0' : 'opacity-0 translate-x-2 text-[#E5E5E5] group-hover:opacity-100 group-hover:translate-x-0 group-focus-visible:opacity-100'
              }`}
            >
              <span className="text-[#737373] mr-1.5">{String(i).padStart(2, '0')}</span>
              {c.label}
            </span>
            <span className={`block h-px transition-all duration-500 ${on ? 'w-9 bg-[#FAFAFA]' : i < active ? 'w-4 bg-[#E5E5E5]' : 'w-4 bg-[#404040] group-hover:bg-[#737373]'}`} />
          </button>
        );
      })}
    </nav>
    <div className="lg:hidden fixed top-16 inset-x-0 z-40 h-px bg-[#262626]" aria-hidden="true">
      <div ref={barRef} className="h-full bg-[#E5E5E5] origin-left" style={{ transform: 'scaleX(0)' }} />
    </div>
  </>
);
