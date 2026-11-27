/**
 * SplitReveal.tsx - Headline that rises word by word out of a mask (staggered, eased).
 * Segments let one heading mix styles, e.g. bold sans + light sans + an italic serif phrase.
 * Controlled with `active`, or self-triggers once when 30 % of it is in view. Screen readers get
 * the plain sentence via aria-label; the animated words are aria-hidden. Reduced motion shows
 * text at once.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';

export type RevealSegment = string | { t: string; c?: string };

interface Props {
  as?: 'h1' | 'h2' | 'h3' | 'p';
  segments: RevealSegment[];
  className?: string;
  /** Controlled visibility; omit to reveal once on scroll-in. */
  active?: boolean;
  delayMs?: number;
  staggerMs?: number;
}

interface Seg {
  c: string;
  words: { w: string; i: number }[];
}

function layout(segments: RevealSegment[]): { segs: Seg[]; full: string } {
  const segs: Seg[] = [];
  const texts: string[] = [];
  let i = 0;
  for (const s of segments) {
    const t = typeof s === 'string' ? s : s.t;
    const c = typeof s === 'string' ? '' : (s.c ?? '');
    texts.push(t);
    const words = t.split(/\s+/).filter(Boolean).map((w) => ({ w, i: i++ }));
    segs.push({ c, words });
  }
  return { segs, full: texts.join(' ') };
}

export const SplitReveal: React.FC<Props> = ({ as = 'h2', segments, className = '', active, delayMs = 0, staggerMs = 42 }) => {
  const ref = useRef<HTMLHeadingElement>(null);
  const [seen, setSeen] = useState(false);
  const { segs, full } = useMemo(() => layout(segments), [segments]);

  useEffect(() => {
    if (active !== undefined || seen) return;
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { threshold: 0.3 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [active, seen]);

  const on = active ?? seen;
  // One element type for typing; the rendered tag still follows `as`.
  const Tag = as as 'h2';

  return (
    // The accessible name is the plain sentence; the per-word spans are hidden from assistive
    // tech, so the text is announced once and innerText holds it once.
    <Tag ref={ref} aria-label={full} className={`mf-split ${on ? 'is-in' : ''} ${className}`}>
      <span aria-hidden="true">
        {segs.map((s, si) => (
          <span key={si} className={s.c}>
            {s.words.map(({ w, i }) => (
              <React.Fragment key={i}>
                <span className="mf-word">
                  <span style={{ transitionDelay: `${delayMs + i * staggerMs}ms` }}>{w}</span>
                </span>{' '}
              </React.Fragment>
            ))}
          </span>
        ))}
      </span>
    </Tag>
  );
};
