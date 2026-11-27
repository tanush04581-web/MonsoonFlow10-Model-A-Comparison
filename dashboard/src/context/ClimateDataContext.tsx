/**
 * ClimateDataContext.tsx - Master atmospheric simulation state coordinator
 *
 * - Day 0 comes from public/data/initial_state.json (2019-2020 validation split), with its true doy_sincos.
 * - Explicit DEMO vs LIVE mode separation:
 *   - DEMO: uses syntheticData.ts local advection/diffusion engine, clearly badged SYNTHETIC, never claims model latency.
 *   - LIVE: calls serve.py (/health + /simulate), NEVER falls back to synthetic data on failure,
 *     validates (10, 6, 32, 36) finiteness, detects persistence stubs, and uses real backend inference_ms only.
 */

import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import type { ChannelId, ChannelTensor, SimulationRollout } from '../types/climate';
import type { ApiStatus, DataMode } from '../types/api';
import {
  API_BASE_URL,
  checkBackendHealth,
  detectPersistenceStub,
  runDemoSimulation,
  runLiveSimulation,
  type LiveSimulationResult,
} from '../services/api';
import { loadInitialState, type InitialState } from '../services/initialState';
import { resolvePerturbation } from '../services/perturbation';
import { createZeroTensor } from '../services/syntheticData';
import { useShockLab } from './ShockLabContext';

export type ViewMode = 'BASELINE' | 'SHOCK' | 'SPLIT' | 'ANOMALY_DIFF';

interface ClimateDataContextType {
  initialCondition: ChannelTensor; // Day 0 state [6, 32, 36]
  history3Day: ChannelTensor[]; // 3-day history [3, 6, 32, 36]
  baselineRollout: SimulationRollout;
  shockRollout: SimulationRollout;
  selectedChannel: ChannelId;
  viewMode: ViewMode;
  dataMode: DataMode;
  apiStatus: ApiStatus;
  setSelectedChannel: (channel: ChannelId) => void;
  setViewMode: (mode: ViewMode) => void;
  setDataMode: (mode: DataMode) => void;
  executeSimulation: () => Promise<void>;
}

const ClimateDataContext = createContext<ClimateDataContextType | undefined>(undefined);

const PLACEHOLDER_HISTORY: ChannelTensor[] = [createZeroTensor(), createZeroTensor(), createZeroTensor()];

export const ClimateDataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { activePreset, customSpec } = useShockLab();

  const [initState, setInitState] = useState<InitialState | null>(null);
  const history3Day = initState?.history3Day ?? PLACEHOLDER_HISTORY;
  const initialCondition = history3Day[2]; // Day 0 state

  const [baselineRollout, setBaselineRollout] = useState<SimulationRollout>([]);
  const [shockRollout, setShockRollout] = useState<SimulationRollout>([]);
  const [selectedChannel, setSelectedChannel] = useState<ChannelId>('tp');
  const [viewMode, setViewMode] = useState<ViewMode>('SHOCK');
  const [dataMode, setDataModeState] = useState<DataMode>('DEMO');

  const [apiStatus, setApiStatus] = useState<ApiStatus>({
    mode: 'DEMO',
    serverOnline: false,
    isLive: false,
    isStubResponse: false,
    stubReason: null,
    fallbackReason: 'DEMO mode active (synthetic advection/diffusion)',
    initialStateLabel: 'loading initial state…',
    initialStateIsSynthetic: false,
    isSimulating: false,
    latencyMs: null,
    endpoint: API_BASE_URL,
    modelName: 'MonsoonFlow-10',
    lastSimulatedAt: null,
    errorMessage: null,
  });

  const modelNameRef = useRef<string>('MonsoonFlow-10');
  const requestIdRef = useRef(0);
  const hasLiveRolloutRef = useRef(false);

  // 1. Load the real Day-0 initial condition once
  useEffect(() => {
    let isMounted = true;
    loadInitialState().then((s) => {
      if (!isMounted) return;
      setInitState(s);
      setApiStatus((prev) => ({
        ...prev,
        initialStateLabel: s.label,
        initialStateIsSynthetic: s.isSynthetic,
      }));
    });
    return () => {
      isMounted = false;
    };
  }, []);

  // 2. Periodic health check (active in LIVE mode only so offline DEMO never spams console errors)
  useEffect(() => {
    if (dataMode !== 'LIVE') return;
    let isMounted = true;
    const pollHealth = async () => {
      const { isOnline, info } = await checkBackendHealth();
      if (!isMounted) return;
      if (info?.model_name) {
        modelNameRef.current = info.model_name;
      }
      setApiStatus((prev) => ({
        ...prev,
        serverOnline: isOnline,
        modelName: info?.model_name || prev.modelName,
      }));
    };
    void pollHealth();
    const interval = setInterval(pollHealth, 3500);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [dataMode]);

  const setDataMode = useCallback((nextMode: DataMode) => {
    requestIdRef.current += 1;
    setDataModeState(nextMode);
    if (nextMode === 'LIVE') {
      // Never leak synthetic DEMO rollouts into LIVE mode
      hasLiveRolloutRef.current = false;
      setBaselineRollout([]);
      setShockRollout([]);
      setApiStatus((prev) => ({
        ...prev,
        mode: 'LIVE',
        isLive: false,
        isStubResponse: false,
        stubReason: null,
        fallbackReason: null,
        isSimulating: true,
        latencyMs: null,
        errorMessage: null,
      }));
    } else {
      setApiStatus((prev) => ({
        ...prev,
        mode: 'DEMO',
        isLive: false,
        isStubResponse: false,
        stubReason: null,
        fallbackReason: 'DEMO mode active (synthetic advection/diffusion)',
        latencyMs: null,
        errorMessage: null,
      }));
    }
  }, []);

  // 3. Execute simulation according to active dataMode
  const executeSimulation = useCallback(async () => {
    if (!initState) return;
    const requestId = ++requestIdRef.current;
    const { history3Day: hist, doySincos } = initState;
    const perturbation = resolvePerturbation(activePreset, customSpec, hist[2]);

    if (dataMode === 'DEMO') {
      const baseRoll = runDemoSimulation(hist, null);
      const shockRoll = perturbation ? runDemoSimulation(hist, perturbation) : baseRoll;
      await Promise.resolve();
      if (requestId !== requestIdRef.current) return;

      hasLiveRolloutRef.current = false;
      setBaselineRollout(baseRoll);
      setShockRollout(shockRoll);
      setApiStatus((prev) => ({
        ...prev,
        mode: 'DEMO',
        isSimulating: false,
        isLive: false,
        isStubResponse: false,
        stubReason: null,
        fallbackReason: 'DEMO mode active (synthetic advection/diffusion)',
        latencyMs: null,
        lastSimulatedAt: new Date(),
        errorMessage: null,
      }));
      return;
    }

    // LIVE mode: strict backend execution, zero synthetic fallback
    await Promise.resolve();
    if (requestId !== requestIdRef.current) return;
    setApiStatus((prev) => ({ ...prev, mode: 'LIVE', isSimulating: true, errorMessage: null }));

    const healthPromise = checkBackendHealth();
    const basePromise = runLiveSimulation(hist, doySincos, null);
    const shockPromise: Promise<LiveSimulationResult> = perturbation
      ? runLiveSimulation(hist, doySincos, perturbation)
      : basePromise;

    const [health, base, shock] = await Promise.all([healthPromise, basePromise, shockPromise]);
    if (requestId !== requestIdRef.current) return;

    const reportedModel = health.info?.model_name || modelNameRef.current;
    if (health.info?.model_name) modelNameRef.current = health.info.model_name;

    if (!base.ok || !shock.ok) {
      const errMsg = !base.ok ? base.error : !shock.ok ? shock.error : 'Backend simulation failed';
      hasLiveRolloutRef.current = false;
      setBaselineRollout([]);
      setShockRollout([]);
      setApiStatus((prev) => ({
        ...prev,
        mode: 'LIVE',
        serverOnline: health.isOnline,
        modelName: reportedModel,
        isSimulating: false,
        isLive: false,
        isStubResponse: false,
        stubReason: null,
        fallbackReason: errMsg,
        latencyMs: null,
        errorMessage: errMsg,
      }));
      return;
    }

    const stubReason = detectPersistenceStub(base.rollout, reportedModel);
    const backendLatency = shock.inferenceTimeMs ?? base.inferenceTimeMs;
    hasLiveRolloutRef.current = true;

    setBaselineRollout(base.rollout);
    setShockRollout(shock.rollout);
    setApiStatus((prev) => ({
      ...prev,
      mode: 'LIVE',
      serverOnline: health.isOnline,
      modelName: reportedModel,
      isSimulating: false,
      isLive: true,
      isStubResponse: Boolean(stubReason),
      stubReason,
      fallbackReason: null,
      latencyMs: backendLatency,
      lastSimulatedAt: new Date(),
      errorMessage: null,
    }));
  }, [initState, activePreset, customSpec, dataMode]);

  // Re-run whenever inputs, shock, or mode change (and auto-recover if server transitions online)
  const serverOnline = apiStatus.serverOnline;
  const prevServerOnlineRef = useRef(false);
  useEffect(() => {
    const transitionedOnline = serverOnline && !prevServerOnlineRef.current;
    prevServerOnlineRef.current = serverOnline;
    if (!initState) return;
    if (transitionedOnline && hasLiveRolloutRef.current) return;
    const id = setTimeout(() => {
      void executeSimulation();
    }, 0);
    return () => clearTimeout(id);
  }, [initState, executeSimulation, serverOnline]);

  return (
    <ClimateDataContext.Provider
      value={{
        initialCondition,
        history3Day,
        baselineRollout,
        shockRollout,
        selectedChannel,
        viewMode,
        dataMode,
        apiStatus,
        setSelectedChannel,
        setViewMode,
        setDataMode,
        executeSimulation,
      }}
    >
      {children}
    </ClimateDataContext.Provider>
  );
};

export function useClimateData(): ClimateDataContextType {
  const ctx = useContext(ClimateDataContext);
  if (!ctx) throw new Error('useClimateData must be used within ClimateDataProvider');
  return ctx;
}
