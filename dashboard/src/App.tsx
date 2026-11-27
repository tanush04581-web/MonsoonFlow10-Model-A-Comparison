/**
 * App.tsx - Routes between the landing story ("#/") and the interactive lab ("#/lab").
 * Both routes are code-split so the story loads fast and the lab only pays for what it uses.
 */

import { Suspense, lazy } from 'react';
import { useHashRoute } from './router';

const Landing = lazy(() => import('./components/landing/Landing'));
const Lab = lazy(() => import('./components/layout/Lab'));

const Splash = () => (
  <div className="h-screen w-screen grid place-items-center bg-[#171717]">
    <div className="w-40 h-px overflow-hidden bg-[#262626]">
      <div className="h-full w-full mf-shimmer-line" />
    </div>
  </div>
);

export function App() {
  const route = useHashRoute();
  return <Suspense fallback={<Splash />}>{route === 'lab' ? <Lab /> : <Landing />}</Suspense>;
}

export default App;
