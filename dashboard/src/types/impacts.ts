/**
 * impacts.ts - Humanitarian and Agricultural downstream indicators
 */

export interface BasinImpact {
  regionId: string;
  name: string;
  latRange: [number, number];
  lonRange: [number, number];
  floodRiskScore: number;      // 0 - 100
  cropMoistureStress: number;  // 0 - 100
  meanPrecipMm: number;
  anomalyPrecipMm: number;
  status: 'Normal' | 'Watch' | 'Advisory' | 'Warning' | 'Critical';
}

export interface DownstreamImpactSummary {
  floodRiskIndex: number;          // 0 - 100 composite index
  cropMoistureStressIndex: number; // 0 - 100 composite index
  extremeRainCells: number;        // count of cells with tp >= 50 mm/day
  heavyRainCells: number;          // count of cells with tp >= 10 mm/day (incl. extreme)
  drySpellCells: number;           // count of cells with tp < 2 mm/day under t2m > 305 K
  highRiskAreaKm2: number;
  basins: BasinImpact[];
}

export const MONSOON_BASINS: Omit<BasinImpact, 'floodRiskScore' | 'cropMoistureStress' | 'meanPrecipMm' | 'anomalyPrecipMm' | 'status'>[] = [
  {
    regionId: 'western_ghats',
    name: 'Western Ghats Coastal Belt',
    latRange: [8.5, 19.5],
    lonRange: [72.5, 76.5],
  },
  {
    regionId: 'gangetic_plains',
    name: 'Indo-Gangetic Basin',
    latRange: [23.5, 29.5],
    lonRange: [77.5, 88.5],
  },
  {
    regionId: 'central_india',
    name: 'Central Agricultural Belt (MP/Vidarbha)',
    latRange: [18.5, 24.5],
    lonRange: [74.5, 82.5],
  },
  {
    regionId: 'northeast_hills',
    name: 'Northeast India & Brahmaputra',
    latRange: [23.5, 28.5],
    lonRange: [89.5, 96.5],
  },
  {
    regionId: 'southern_peninsula',
    name: 'Southern Rain Shadow (Deccan)',
    latRange: [11.5, 17.5],
    lonRange: [76.5, 80.5],
  },
];
