/**
 * api.ts - Wire interface schemas for root serve.py backend
 * Exactly matches root serve.py (FastAPI) and inference/schemas.py
 */

export type DataMode = 'DEMO' | 'LIVE';

export interface HealthResponse {
  status: string;
  model_name?: string;
  device?: string;
  threads?: number;
  uptime_seconds?: number;
}

export interface SimulateRequest {
  init_states: number[][][][];        // (3, 6, 32, 36) in physical units
  doy_sincos: [number, number];        // [sin, cos] cyclic day-of-year representation
  steps: number;                      // 10
  perturbation?: number[][][] | null; // (6, 32, 36) physical perturbation delta
}

export interface SimulateResponse {
  rollout: number[][][][];            // (steps, 6, 32, 36) in physical units
  inference_ms?: number;              // Execution latency in milliseconds
}

export interface ApiStatus {
  /** Active simulation mode selected by the user. */
  mode: DataMode;
  /** Health endpoint reachable with status === "ready" right now. */
  serverOnline: boolean;
  /** Both displayed baseline AND shock rollouts came from the live server in LIVE mode. */
  isLive: boolean;
  /** True if backend metadata or rollout behavior indicates a persistence/stub placeholder. */
  isStubResponse: boolean;
  /** Detailed reason when a persistence/stub response is detected. */
  stubReason: string | null;
  /** Reason synthetic data is active (DEMO mode) or why LIVE failed. */
  fallbackReason: string | null;
  /** Day-0 initial condition description and whether it is synthetic. */
  initialStateLabel: string;
  initialStateIsSynthetic: boolean;
  isSimulating: boolean;
  /** Backend-reported inference_ms in LIVE mode only; null in DEMO or on LIVE failure. */
  latencyMs: number | null;
  endpoint: string;
  modelName: string;
  lastSimulatedAt: Date | null;
  errorMessage: string | null;
}
