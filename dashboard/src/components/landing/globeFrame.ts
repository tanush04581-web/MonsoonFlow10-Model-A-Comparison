/**
 * globeFrame.ts - Where the 2D story globe sits on screen. The 3D orbit scene lands on exactly
 * this framing, so the two canvases can cross-fade without a jump.
 */

/**
 * Cloud pass between the 3D Earth and the 2D weather chart, in story-section screens (raw).
 * Clouds roll in, the canvases swap while the screen is fully covered, then the clouds part.
 */
export const CLOUD = {
  in0: 2.68, // clouds start to gather
  full: 3.0, // fully covered
  swap: 3.04, // 3D Earth off, 2D map on (hidden under the clouds)
  out0: 3.1, // clouds start to part
  out1: 3.5, // clear: the map is fully revealed
} as const;

/** Globe framing in CSS px and degrees. */
export interface GlobeFrame {
  cx: number;
  cy: number;
  R: number;
  lon: number;
  lat: number;
}

/** Globe framing at story progress p = 0 for a viewport. */
export function heroGlobeFrame(w: number, h: number, isDesktop: boolean): GlobeFrame {
  return {
    cx: isDesktop ? w * 0.66 : w * 0.5,
    cy: isDesktop ? h * 0.52 : h * 0.33,
    R: isDesktop ? Math.min(w * 0.3, h * 0.4) : Math.min(w * 0.42, h * 0.27),
    lon: 82,
    lat: 17,
  };
}
