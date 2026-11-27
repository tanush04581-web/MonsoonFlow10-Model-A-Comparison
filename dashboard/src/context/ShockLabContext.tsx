/**
 * ShockLabContext.tsx - Counterfactual perturbation configuration state
 *
 * activePresetId semantics:
 * - 'ZERO_SHOCK'           : no perturbation (null) is sent
 * - graded box presets     : exact monsoon_r1 make_shock() perturbation, built from the Day-0 state
 * - 'WESTERN_GHATS_DELUGE' : its Gaussian spec
 * - 'CUSTOM'               : the interactive Gaussian brush (any edit to the brush switches to CUSTOM)
 */

import React, { createContext, useContext, useState, useMemo, useCallback } from 'react';
import type { GaussianShockSpec, PresetShockId, ShockPreset } from '../types/shocks';
import { CUSTOM_PRESET, SHOCK_PRESETS } from '../types/shocks';
import { type ShockBoundsReport, validateShockRealism } from '../services/gaussianBrush';

interface ShockLabContextType {
  activePresetId: PresetShockId;
  activePreset: ShockPreset;
  customSpec: GaussianShockSpec;
  isBrushMode: boolean;
  boundsReport: ShockBoundsReport;
  selectPreset: (presetId: PresetShockId) => void;
  updateCustomSpec: (partial: Partial<GaussianShockSpec>) => void;
  toggleBrushMode: () => void;
  setBrushMode: (active: boolean) => void;
  resetShock: () => void;
}

const DEFAULT_CUSTOM_SPEC: GaussianShockSpec = {
  centerLat: 15.0,
  centerLon: 88.0,
  radiusKm: 450,
  sigmaKm: 300,
  channel: 't2m',
  magnitude: 3.0,
};

const ZERO_PRESET = SHOCK_PRESETS.find((p) => p.id === 'ZERO_SHOCK') as ShockPreset;

const ShockLabContext = createContext<ShockLabContextType | undefined>(undefined);

export const ShockLabProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activePresetId, setActivePresetId] = useState<PresetShockId>('ZERO_SHOCK');
  const [customSpec, setCustomSpec] = useState<GaussianShockSpec>(DEFAULT_CUSTOM_SPEC);
  const [isBrushMode, setIsBrushMode] = useState<boolean>(false);

  const activePreset = useMemo<ShockPreset>(() => {
    if (activePresetId === 'CUSTOM') return CUSTOM_PRESET;
    return SHOCK_PRESETS.find((p) => p.id === activePresetId) || ZERO_PRESET;
  }, [activePresetId]);

  const selectPreset = useCallback((presetId: PresetShockId) => {
    setActivePresetId(presetId);
    setIsBrushMode(false);
    const preset = SHOCK_PRESETS.find((p) => p.id === presetId);
    if (preset?.spec) {
      // Gaussian presets seed the brush so they can be tweaked afterwards
      setCustomSpec({ ...preset.spec });
    }
  }, []);

  const updateCustomSpec = useCallback((partial: Partial<GaussianShockSpec>) => {
    setCustomSpec((prev) => ({ ...prev, ...partial }));
    // Any brush edit makes the brush the active shock (previously this fell back to
    // ZERO_SHOCK, so custom shocks were silently never sent to the model).
    setActivePresetId('CUSTOM');
  }, []);

  const toggleBrushMode = useCallback(() => {
    setIsBrushMode((prev) => !prev);
  }, []);

  const setBrushMode = useCallback((active: boolean) => {
    setIsBrushMode(active);
  }, []);

  const resetShock = useCallback(() => {
    setActivePresetId('ZERO_SHOCK');
    setIsBrushMode(false);
    setCustomSpec(DEFAULT_CUSTOM_SPEC);
  }, []);

  const boundsReport = useMemo<ShockBoundsReport>(() => {
    if (activePreset.box) {
      // Graded box shocks are defined by the organisers; check each additive term
      const warnings = activePreset.box.terms
        .filter((t) => t.mode === 'add')
        .flatMap((t) => validateShockRealism(t.channel, t.value).warnings);
      return { isValid: warnings.length === 0, warnings };
    }
    if (activePresetId === 'ZERO_SHOCK') return { isValid: true, warnings: [] };
    const spec = activePresetId === 'CUSTOM' ? customSpec : activePreset.spec ?? customSpec;
    return validateShockRealism(spec.channel, spec.magnitude);
  }, [activePreset, activePresetId, customSpec]);

  return (
    <ShockLabContext.Provider
      value={{
        activePresetId,
        activePreset,
        customSpec,
        isBrushMode,
        boundsReport,
        selectPreset,
        updateCustomSpec,
        toggleBrushMode,
        setBrushMode,
        resetShock,
      }}
    >
      {children}
    </ShockLabContext.Provider>
  );
};

export function useShockLab(): ShockLabContextType {
  const ctx = useContext(ShockLabContext);
  if (!ctx) throw new Error('useShockLab must be used within ShockLabProvider');
  return ctx;
}
