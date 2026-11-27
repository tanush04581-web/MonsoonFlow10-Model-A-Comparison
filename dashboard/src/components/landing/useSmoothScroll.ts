/**
 * useSmoothScroll.ts - Lenis inertial scrolling for the landing page only (destroyed on unmount,
 * so the lab keeps native scrolling). Disabled for prefers-reduced-motion. Touch keeps native
 * scrolling (Lenis default).
 */

import { useEffect, useRef } from 'react';
import Lenis from 'lenis';

export function useSmoothScroll(enabled: boolean): React.RefObject<Lenis | null> {
  const ref = useRef<Lenis | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const lenis = new Lenis({ autoRaf: true, lerp: 0.09, smoothWheel: true });
    ref.current = lenis;
    return () => {
      lenis.destroy();
      ref.current = null;
    };
  }, [enabled]);
  return ref;
}
