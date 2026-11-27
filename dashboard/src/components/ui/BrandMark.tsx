/**
 * BrandMark.tsx - MonsoonFlow-10 mark: a raindrop whose interior is a branching flow line
 * (one baseline, one diverging "multiverse" path). `tone="mono"` renders it in the neutral
 * landing palette.
 */

import React from 'react';

export const BrandMark: React.FC<{ className?: string; tone?: 'color' | 'mono' }> = ({ className, tone = 'color' }) => {
  const mono = tone === 'mono';
  const id = mono ? 'mf-drop-mono' : 'mf-drop';
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={mono ? '#FAFAFA' : '#7DD3FC'} />
          <stop offset="1" stopColor={mono ? '#737373' : '#2563EB'} />
        </linearGradient>
      </defs>
      <path d="M16 2.5C16 2.5 6 13.2 6 19.6A10 10 0 0 0 26 19.6C26 13.2 16 2.5 16 2.5Z" fill={`url(#${id})`} opacity="0.95" />
      <path d="M10.5 21.5C13 21.5 14.2 19.5 16 17.5S19.5 14.5 21.5 14.5" stroke={mono ? '#171717' : '#fff'} strokeWidth="1.6" fill="none" strokeLinecap="round" />
      <path d="M16 17.5C17.6 19.4 19 21.5 21.5 21.5" stroke={mono ? '#404040' : '#FBBF24'} strokeWidth="1.6" fill="none" strokeLinecap="round" />
    </svg>
  );
};
