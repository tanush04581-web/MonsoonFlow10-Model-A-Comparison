/**
 * CursorFollower.tsx - Landing-page cursor: an exact dot plus a softly lagging ring that grows
 * over links/buttons and shows a hint label on elements with data-cursor="…".
 * Fine pointers only; disabled for touch and prefers-reduced-motion (the native cursor returns).
 */

import React, { useEffect, useRef, useState } from 'react';

export const CursorFollower: React.FC = () => {
  const dotRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const labelRef = useRef<HTMLSpanElement>(null);
  const [enabled] = useState(
    () =>
      typeof window !== 'undefined' &&
      window.matchMedia('(pointer: fine)').matches &&
      !window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );

  useEffect(() => {
    if (!enabled) return;
    document.documentElement.classList.add('mf-hide-cursor');
    const pos = { x: -100, y: -100 };
    const ring = { x: -100, y: -100, s: 1 };
    let target = 1;
    let raf = 0;
    // Magnetic elements ([data-magnetic]) lean towards the pointer. Uses the CSS `translate`
    // property so it composes with the press-scale `transform`.
    let mag: { el: HTMLElement; cx: number; cy: number } | null = null;
    const releaseMag = () => {
      if (mag) mag.el.style.translate = '';
      mag = null;
    };
    const onMove = (e: PointerEvent) => {
      pos.x = e.clientX;
      pos.y = e.clientY;
      const m = (e.target as HTMLElement | null)?.closest<HTMLElement>('[data-magnetic]') ?? null;
      if (m !== (mag?.el ?? null)) {
        releaseMag();
        if (m) {
          const r = m.getBoundingClientRect();
          mag = { el: m, cx: r.left + r.width / 2, cy: r.top + r.height / 2 };
        }
      }
      const el = (e.target as HTMLElement | null)?.closest<HTMLElement>('a, button, [data-cursor]');
      target = el ? 1.9 : 1;
      const hint = el?.dataset.cursor ?? '';
      if (labelRef.current && labelRef.current.textContent !== hint) labelRef.current.textContent = hint;
      ringRef.current?.classList.toggle('has-label', Boolean(hint));
    };
    const onDown = () => (target = 0.8);
    const onUp = () => (target = 1);
    const onLeave = () => {
      pos.x = -100;
      pos.y = -100;
    };
    const tick = () => {
      ring.x += (pos.x - ring.x) * 0.18;
      ring.y += (pos.y - ring.y) * 0.18;
      ring.s += (target - ring.s) * 0.2;
      if (dotRef.current) dotRef.current.style.transform = `translate3d(${pos.x}px, ${pos.y}px, 0)`;
      if (ringRef.current) ringRef.current.style.transform = `translate3d(${ring.x}px, ${ring.y}px, 0) scale(${ring.s})`;
      if (mag) mag.el.style.translate = `${((pos.x - mag.cx) * 0.22).toFixed(1)}px ${((pos.y - mag.cy) * 0.3).toFixed(1)}px`;
      raf = requestAnimationFrame(tick);
    };
    const onScroll = () => releaseMag();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerdown', onDown);
    window.addEventListener('pointerup', onUp);
    document.addEventListener('pointerleave', onLeave);
    raf = requestAnimationFrame(tick);
    return () => {
      document.documentElement.classList.remove('mf-hide-cursor');
      cancelAnimationFrame(raf);
      releaseMag();
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointerup', onUp);
      document.removeEventListener('pointerleave', onLeave);
    };
  }, [enabled]);

  if (!enabled) return null;
  return (
    <>
      <div ref={ringRef} className="mf-cursor-ring" aria-hidden>
        <span ref={labelRef} className="mf-cursor-label" />
      </div>
      <div ref={dotRef} className="mf-cursor-dot" aria-hidden />
    </>
  );
};
