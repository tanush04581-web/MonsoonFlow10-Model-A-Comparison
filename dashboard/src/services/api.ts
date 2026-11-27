/**
 * api.ts - HTTP client for root serve.py
 * - POST /simulate with init_states: (3, 6, 32, 36), doy_sincos: [sin, cos], steps: 10, perturbation: (6, 32, 36) | null
 * - GET /health returning {"status": "ready", "model_name": ...}
 *
 * Honesty contract:
 * - LIVE mode calls the backend and NEVER silently substitutes synthetic forecasts on error.
 * - DEMO mode explicitly uses the local synthetic advection/diffusion engine.
 * - Backend responses are validated for exact (10, 6, 32, 36) shape, finiteness, and persistence-stub behavior.
 */

import { GRID_CONFIG, type ChannelTensor, type SimulationRollout } from '../types/climate';
import type { HealthResponse, SimulateRequest, SimulateResponse } from '../types/api';
import { simulateOfflineRollout } from './syntheticData';

// 127.0.0.1 default avoids Windows IPv6 ::1 stall before IPv4 fallback
export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, '') || 'http://127.0.0.1:8000';

const HEALTH_TIMEOUT_MS = 2000;
const SIMULATE_TIMEOUT_MS = 6500;
const STEPS = 10;
const NC = 6;

export interface HealthCheckResult {
  isOnline: boolean;
  info?: HealthResponse;
  reason?: string;
}

export async function checkBackendHealth(): Promise<HealthCheckResult> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);

    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      return { isOnline: false, reason: `HTTP ${res.status} from /health` };
    }

    const data: HealthResponse = await res.json();
    if (!data || typeof data !== 'object' || data.status !== 'ready') {
      return {
        isOnline: false,
        info: data,
        reason: `backend status is "${String(data?.status ?? 'unknown')}" (expected "ready")`,
      };
    }

    return { isOnline: true, info: data };
  } catch (err: unknown) {
    const reason =
      err instanceof DOMException && err.name === 'AbortError'
        ? 'health check timed out'
        : 'backend unreachable';
    return { isOnline: false, reason };
  }
}

export function validateRolloutShape(rollout: unknown, expectedSteps = STEPS): string | null {
  if (!Array.isArray(rollout) || rollout.length !== expectedSteps) {
    return `expected ${expectedSteps} forecast steps, got ${Array.isArray(rollout) ? rollout.length : typeof rollout}`;
  }
  for (let t = 0; t < expectedSteps; t++) {
    const frame = rollout[t];
    if (!Array.isArray(frame) || frame.length !== NC) {
      return `step ${t}: expected ${NC} channels`;
    }
    for (let c = 0; c < NC; c++) {
      const grid = frame[c];
      if (!Array.isArray(grid) || grid.length !== GRID_CONFIG.nLat) {
        return `step ${t}, channel ${c}: expected ${GRID_CONFIG.nLat} latitudes`;
      }
      for (let i = 0; i < GRID_CONFIG.nLat; i++) {
        const row = grid[i];
        if (!Array.isArray(row) || row.length !== GRID_CONFIG.nLon) {
          return `step ${t}, channel ${c}, lat ${i}: expected ${GRID_CONFIG.nLon} longitudes`;
        }
        for (let j = 0; j < GRID_CONFIG.nLon; j++) {
          if (typeof row[j] !== 'number' || !Number.isFinite(row[j])) {
            return `step ${t}, channel ${c}, cell [${i}, ${j}]: non-finite value`;
          }
        }
      }
    }
  }
  return null;
}

/**
 * Detects placeholder/persistence backend behavior via either reported model_name
 * or step-invariant rollout output (all forecast steps identical to Day 1).
 */
export function detectPersistenceStub(rollout: SimulationRollout, modelName?: string | null): string | null {
  const name = (modelName || '').trim();
  if (/pending|stub|persistence/i.test(name)) {
    return `Backend reported stub/pending model_name ("${name}")`;
  }
  if (rollout.length <= 1) return null;

  let maxStepDiff = 0;
  const first = rollout[0];
  for (let t = 1; t < rollout.length; t++) {
    const step = rollout[t];
    for (let c = 0; c < NC; c++) {
      for (let i = 0; i < GRID_CONFIG.nLat; i++) {
        for (let j = 0; j < GRID_CONFIG.nLon; j++) {
          const d = Math.abs(step[c][i][j] - first[c][i][j]);
          if (d > maxStepDiff) maxStepDiff = d;
        }
      }
    }
  }
  if (maxStepDiff < 1e-6) {
    return `Backend returned identical constant fields across all ${rollout.length} forecast steps (persistence stub; inference/predictor.py not executing checkpoints/model.pt)`;
  }
  return null;
}

export type LiveSimulationResult =
  | {
      ok: true;
      rollout: SimulationRollout;
      inferenceTimeMs: number | null;
    }
  | {
      ok: false;
      error: string;
    };

export async function runLiveSimulation(
  history3Day: ChannelTensor[],
  doySincos: [number, number],
  perturbation: ChannelTensor | null
): Promise<LiveSimulationResult> {
  const payload: SimulateRequest = {
    init_states: history3Day,
    doy_sincos: doySincos,
    steps: STEPS,
    perturbation,
  };

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), SIMULATE_TIMEOUT_MS);

    const res = await fetch(`${API_BASE_URL}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      let detail = '';
      try {
        detail = (await res.text()).slice(0, 160);
      } catch {
        /* ignore unreadable body */
      }
      return {
        ok: false,
        error: `POST /simulate returned HTTP ${res.status}${detail ? `: ${detail}` : ''}`,
      };
    }

    let data: SimulateResponse;
    try {
      data = (await res.json()) as SimulateResponse;
    } catch {
      return {
        ok: false,
        error: 'POST /simulate returned invalid JSON (NaN/Inf are invalid JSON)',
      };
    }

    const shapeProblem = validateRolloutShape(data?.rollout, STEPS);
    if (shapeProblem) {
      return {
        ok: false,
        error: `Invalid rollout schema from backend: ${shapeProblem}`,
      };
    }

    const inferenceTimeMs =
      typeof data.inference_ms === 'number' && Number.isFinite(data.inference_ms)
        ? Math.round(data.inference_ms * 10) / 10
        : null;

    return {
      ok: true,
      rollout: data.rollout,
      inferenceTimeMs,
    };
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      return {
        ok: false,
        error: `POST /simulate timed out (> ${(SIMULATE_TIMEOUT_MS / 1000).toFixed(1)} s) at ${API_BASE_URL}`,
      };
    }
    return {
      ok: false,
      error: `Cannot reach ${API_BASE_URL}/simulate — start serve.py on port 8000 or switch to DEMO mode.`,
    };
  }
}

export function runDemoSimulation(
  history3Day: ChannelTensor[],
  perturbation: ChannelTensor | null
): SimulationRollout {
  return simulateOfflineRollout(history3Day[2], perturbation);
}
