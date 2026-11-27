/**
 * useStoryData.ts - Data for the landing story: the real Day-0 state (initial_state.json) and two
 * 10-day futures, normal (P = 0) and the graded "warm Bay of Bengal" shock.
 * Uses the live MF10 backend when /health answers; otherwise the clearly labelled DEMO engine.
 */

import { useEffect, useState } from 'react';
import type { ChannelTensor, SimulationRollout } from '../../types/climate';
import { SHOCK_PRESETS, type GaussianShockSpec, type ShockPreset } from '../../types/shocks';
import { loadInitialState } from '../../services/initialState';
import { resolvePerturbation } from '../../services/perturbation';
import { checkBackendHealth, runDemoSimulation, runLiveSimulation } from '../../services/api';

export interface StoryData {
  day0: ChannelTensor;
  day0Shocked: ChannelTensor;
  base: SimulationRollout;
  shock: SimulationRollout;
  preset: ShockPreset;
  live: boolean;
  startLabel: string;
  syntheticStart: boolean;
}

const UNUSED_SPEC: GaussianShockSpec = { centerLat: 15, centerLon: 88, radiusKm: 450, sigmaKm: 300, channel: 't2m', magnitude: 0 };

export function useStoryData(): StoryData | null {
  const [data, setData] = useState<StoryData | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      const init = await loadInitialState();
      const day0 = init.history3Day[2];
      const preset = SHOCK_PRESETS.find((p) => p.id === 'WARM_BAY_OF_BENGAL') as ShockPreset;
      const P = resolvePerturbation(preset, UNUSED_SPEC, day0);
      const day0Shocked = P ? day0.map((g, c) => g.map((row, i) => row.map((v, j) => v + P[c][i][j]))) : day0;

      let base = runDemoSimulation(init.history3Day, null);
      let shock = P ? runDemoSimulation(init.history3Day, P) : base;
      let live = false;

      const health = await checkBackendHealth();
      if (health.isOnline) {
        const [b, s] = await Promise.all([
          runLiveSimulation(init.history3Day, init.doySincos, null),
          runLiveSimulation(init.history3Day, init.doySincos, P),
        ]);
        if (b.ok && s.ok) {
          base = b.rollout;
          shock = s.rollout;
          live = true;
        }
      }
      if (!alive) return;
      setData({ day0, day0Shocked, base, shock, preset, live, startLabel: init.label, syntheticStart: init.isSynthetic });
    })();
    return () => {
      alive = false;
    };
  }, []);

  return data;
}
