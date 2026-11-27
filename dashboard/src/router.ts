/**
 * router.ts - Tiny hash router: "#/" = landing story, "#/lab" = the interactive dashboard.
 * Hash routing works from any static host and from `vite preview` with no server config.
 */

import { useSyncExternalStore } from 'react';

export type Route = 'landing' | 'lab';

const parse = (): Route => (window.location.hash.replace(/^#/, '').startsWith('/lab') ? 'lab' : 'landing');

const subscribe = (cb: () => void) => {
  window.addEventListener('hashchange', cb);
  return () => window.removeEventListener('hashchange', cb);
};

export function useHashRoute(): Route {
  return useSyncExternalStore(subscribe, parse, () => 'landing');
}

export function navigate(route: Route): void {
  window.location.hash = route === 'lab' ? '#/lab' : '#/';
}
