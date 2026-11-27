/**
 * CountUp.tsx - Animates a number from its previous value to the new one (eased, ~600 ms).
 * Honours prefers-reduced-motion by jumping straight to the value.
 */

import React, { useEffect, useRef, useState } from 'react';

interface Props {
  value: number;
  decimals?: number;
  durationMs?: number;
  prefix?: string;
  suffix?: string;
  signed?: boolean;
  className?: string;
}

const reduced = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

export const CountUp: React.FC<Props> = ({ value, decimals = 0, durationMs = 650, prefix = '', suffix = '', signed = false, className }) => {
  const [shown, setShown] = useState(value);
  const fromRef = useRef(value);
  const shownRef = useRef(value);

  const noMotion = reduced();

  useEffect(() => {
    if (noMotion) {
      shownRef.current = value;
      return;
    }
    fromRef.current = shownRef.current;
    const t0 = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - t0) / durationMs);
      const e = 1 - Math.pow(1 - t, 3);
      const v = fromRef.current + (value - fromRef.current) * e;
      shownRef.current = v;
      setShown(v);
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value, durationMs, noMotion]);

  const rounded = Number((noMotion ? value : shown).toFixed(decimals));
  const sign = signed && rounded > 0 ? '+' : '';
  const text = `${prefix}${sign}${rounded.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`;
  return <span className={`tabular-nums ${className ?? ''}`}>{text}</span>;
};
