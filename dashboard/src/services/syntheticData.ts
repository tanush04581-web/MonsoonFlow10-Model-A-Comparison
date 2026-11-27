/**
 * syntheticData.ts - OFFLINE FALLBACK ONLY.
 * Analytic (made-up) monsoon-like fields and a toy diffusion/advection rollout, used only when
 * public/data/initial_state.json or the live serve.py cannot be reached. Never model output;
 * the UI labels anything produced here as SYNTHETIC.
 */

import { GRID_CONFIG, type ChannelTensor, type SimulationRollout } from '../types/climate';

export function createZeroTensor(): ChannelTensor {
  return Array.from({ length: 6 }, () =>
    Array.from({ length: GRID_CONFIG.nLat }, () =>
      Array.from({ length: GRID_CONFIG.nLon }, () => 0)
    )
  );
}

/**
 * Generates an authentic Day 0 monsoon synoptic state:
 * - Arabian Sea low-level jet (strong south-westerly winds)
 * - Monsoon trough over Indo-Gangetic plain (lower MSL)
 * - Heavy orographic rainfall along Western Ghats and Meghalaya
 * - High tropical temperatures in northwest (Rajasthan)
 */
export function generateClimatologicalInitialCondition(offsetDay = 0): ChannelTensor {
  const t2m: number[][] = [];
  const tp: number[][] = [];
  const u10: number[][] = [];
  const v10: number[][] = [];
  const msl: number[][] = [];
  const tcwv: number[][] = [];

  for (let i = 0; i < GRID_CONFIG.nLat; i++) {
    const lat = GRID_CONFIG.latMin + i * GRID_CONFIG.resolutionDeg;
    const rowT2m: number[] = [];
    const rowTp: number[] = [];
    const rowU10: number[] = [];
    const rowV10: number[] = [];
    const rowMsl: number[] = [];
    const rowTcwv: number[] = [];

    for (let j = 0; j < GRID_CONFIG.nLon; j++) {
      const lon = GRID_CONFIG.lonMin + j * GRID_CONFIG.resolutionDeg;

      // 1. Temperature
      const distNW = Math.hypot(lat - 28.0, lon - 72.0);
      const tempK = 299.15 + 8.0 * Math.exp(-(distNW * distNW) / 100) - (lat - 20) * 0.25 + offsetDay * 0.1;
      rowT2m.push(Math.round(tempK * 10) / 10);

      // 2. Precipitation: Western Ghats and Northeast
      const distWG = Math.hypot(lat - 14.5, lon - 74.5);
      const distNE = Math.hypot(lat - 25.5, lon - 92.5);
      const precip =
        35.0 * Math.exp(-(distWG * distWG) / 12) +
        28.0 * Math.exp(-(distNE * distNE) / 18) +
        Math.max(0, 3.0 * Math.sin((lon - 70) * 0.15 + offsetDay * 0.1));
      rowTp.push(Math.max(0, Math.round(precip * 10) / 10));

      // 3. Low-Level Jet (Findlater Jet)
      const distLLJ = Math.hypot(lat - 14.0, lon - 68.0);
      const u = 6.0 + 10.0 * Math.exp(-(distLLJ * distLLJ) / 45);
      const v = 4.0 + 7.0 * Math.exp(-(distLLJ * distLLJ) / 45);
      rowU10.push(Math.round(u * 10) / 10);
      rowV10.push(Math.round(v * 10) / 10);

      // 4. Monsoon Trough (MSL)
      const distTrough = Math.hypot(lat - 24.0, lon - 80.0);
      const pressure = 100800 - 800 * Math.exp(-(distTrough * distTrough) / 80);
      rowMsl.push(Math.round(pressure));

      // 5. TCWV
      const moist = 52.0 + 12.0 * Math.sin(lat * 0.08) - (lat > 28 ? 15 : 0);
      rowTcwv.push(Math.round(Math.max(15, moist) * 10) / 10);
    }

    t2m.push(rowT2m);
    tp.push(rowTp);
    u10.push(rowU10);
    v10.push(rowV10);
    msl.push(rowMsl);
    tcwv.push(rowTcwv);
  }

  return [t2m, tp, u10, v10, msl, tcwv];
}

/**
 * Generates (3, 6, 32, 36) historical conditions corresponding to Day -2, Day -1, Day 0
 */
export function generate3DayClimatologicalHistory(): ChannelTensor[] {
  return [
    generateClimatologicalInitialCondition(-2),
    generateClimatologicalInitialCondition(-1),
    generateClimatologicalInitialCondition(0),
  ];
}

/**
 * Autoregressively rolls out 10 steps of atmospheric state.
 */
export function simulateOfflineRollout(
  initialState: ChannelTensor,
  perturbation: ChannelTensor | null
): SimulationRollout {
  const rollout: SimulationRollout = [];
  const currentStateInit: ChannelTensor = JSON.parse(JSON.stringify(initialState));
  let currentState = currentStateInit;

  if (perturbation) {
    for (let c = 0; c < 6; c++) {
      for (let i = 0; i < GRID_CONFIG.nLat; i++) {
        for (let j = 0; j < GRID_CONFIG.nLon; j++) {
          currentState[c][i][j] += perturbation[c][i][j];
          if (c === 1 || c === 5) currentState[c][i][j] = Math.max(0, currentState[c][i][j]);
        }
      }
    }
  }

  for (let day = 1; day <= 10; day++) {
    const nextState: ChannelTensor = JSON.parse(JSON.stringify(currentState));

    for (let c = 0; c < 6; c++) {
      for (let i = 0; i < GRID_CONFIG.nLat; i++) {
        for (let j = 0; j < GRID_CONFIG.nLon; j++) {
          const val = currentState[c][i][j];
          const westVal = j > 0 ? currentState[c][i][j - 1] : val;
          const southVal = i > 0 ? currentState[c][i - 1][j] : val;
          const eastVal = j < GRID_CONFIG.nLon - 1 ? currentState[c][i][j + 1] : val;
          const northVal = i < GRID_CONFIG.nLat - 1 ? currentState[c][i + 1][j] : val;

          const laplacian = 0.25 * (westVal + southVal + eastVal + northVal) - val;
          const advection = 0.08 * (westVal - val);

          let newVal = val + 0.3 * laplacian + advection;
          if (c === 0) {
            newVal = Math.max(233.15, Math.min(333.15, newVal));
          } else if (c === 1 || c === 5) {
            newVal = Math.max(0, newVal);
          }

          nextState[c][i][j] = Math.round(newVal * 100) / 100;
        }
      }
    }

    rollout.push(nextState);
    currentState = nextState;
  }

  return rollout;
}
