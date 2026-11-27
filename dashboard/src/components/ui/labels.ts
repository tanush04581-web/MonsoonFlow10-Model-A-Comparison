/**
 * labels.ts - Plain-language names for the six model variables. Technical ids (t2m, tp, …) stay
 * available in tooltips so experts lose nothing.
 */

import type { ChannelId } from '../../types/climate';

export const CHANNEL_HUMAN: Record<ChannelId, { short: string; long: string; tech: string }> = {
  tp: { short: 'Rain', long: 'Rainfall', tech: 'tp · total precipitation (mm/day)' },
  t2m: { short: 'Temperature', long: 'Air temperature', tech: 't2m · 2 m temperature (wire: K, shown in °C)' },
  u10: { short: 'Wind E–W', long: 'East–west wind', tech: 'u10 · 10 m zonal wind (m/s, + = from west)' },
  v10: { short: 'Wind N–S', long: 'North–south wind', tech: 'v10 · 10 m meridional wind (m/s, + = from south)' },
  msl: { short: 'Pressure', long: 'Sea-level pressure', tech: 'msl · mean sea-level pressure (wire: Pa, shown in hPa)' },
  tcwv: { short: 'Moisture', long: 'Water vapour in the air', tech: 'tcwv · total column water vapour (kg/m²)' },
};
