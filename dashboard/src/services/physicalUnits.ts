/**
 * physicalUnits.ts - Conversions between wire units and meteorological display values
 */

import { GRID_CONFIG, type ChannelId } from '../types/climate';

export function wireToDisplay(value: number, channel: ChannelId): { val: number; str: string; unit: string } {
  switch (channel) {
    case 't2m': {
      // Wire is Kelvin, display as Celsius
      const celsius = value - 273.15;
      return { val: celsius, str: celsius.toFixed(1), unit: '°C' };
    }
    case 'tp': {
      // Wire is mm/day (strictly >= 0)
      const clamped = Math.max(0, value);
      return { val: clamped, str: clamped.toFixed(1), unit: 'mm/day' };
    }
    case 'u10':
    case 'v10': {
      return { val: value, str: value.toFixed(1), unit: 'm/s' };
    }
    case 'msl': {
      // Wire is Pa, display as hPa
      const hPa = value / 100.0;
      return { val: hPa, str: hPa.toFixed(1), unit: 'hPa' };
    }
    case 'tcwv': {
      const clamped = Math.max(0, value);
      return { val: clamped, str: clamped.toFixed(1), unit: 'kg/m²' };
    }
    default:
      return { val: value, str: (value as number).toFixed(2), unit: '' };
  }
}

export function displayDeltaString(delta: number, channel: ChannelId): string {
  const sign = delta >= 0 ? '+' : '';
  switch (channel) {
    case 't2m':
      return `${sign}${delta.toFixed(2)} K`;
    case 'tp':
      return `${sign}${delta.toFixed(1)} mm/d`;
    case 'u10':
    case 'v10':
      return `${sign}${delta.toFixed(1)} m/s`;
    case 'msl':
      return `${sign}${(delta / 100.0).toFixed(1)} hPa`;
    case 'tcwv':
      return `${sign}${delta.toFixed(1)} kg/m²`;
    default:
      return `${sign}${delta.toFixed(2)}`;
  }
}

// Convert between lat/lon and 32x36 grid index
export function latLonToGrid(lat: number, lon: number): { i: number; j: number } | null {
  if (
    lat < GRID_CONFIG.latMin - 0.5 ||
    lat > GRID_CONFIG.latMax + 0.5 ||
    lon < GRID_CONFIG.lonMin - 0.5 ||
    lon > GRID_CONFIG.lonMax + 0.5
  ) {
    return null;
  }
  const i = Math.round((lat - GRID_CONFIG.latMin) / GRID_CONFIG.resolutionDeg);
  const j = Math.round((lon - GRID_CONFIG.lonMin) / GRID_CONFIG.resolutionDeg);
  const clampedI = Math.max(0, Math.min(GRID_CONFIG.nLat - 1, i));
  const clampedJ = Math.max(0, Math.min(GRID_CONFIG.nLon - 1, j));
  return { i: clampedI, j: clampedJ };
}

export function gridToLatLon(i: number, j: number): { lat: number; lon: number } {
  return {
    lat: GRID_CONFIG.latMin + i * GRID_CONFIG.resolutionDeg,
    lon: GRID_CONFIG.lonMin + j * GRID_CONFIG.resolutionDeg,
  };
}
