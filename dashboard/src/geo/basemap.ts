/**
 * basemap.ts - Fully offline vector basemap for the LCC map: land, coastline, curved graticule,
 * domain veil and human-readable place labels. No tiles, no network.
 *
 * Coastline: Natural Earth 1:50m land (public domain), clipped to 62–104°E / 3.5–40.5°N
 * (src/assets/geo/coast.json). Ring edges that lie ON that clip box are not stroked, so the
 * clip rectangle never shows up as a fake coastline.
 */

import coastGeo from '../assets/geo/coast.json';
import { GRID_CONFIG } from '../types/climate';
import { lccForward, type MapView } from './lcc';

const HALF = GRID_CONFIG.resolutionDeg / 2;
const CLIP = { lonMin: 62, lonMax: 104, latMin: 3.5, latMax: 40.5 };
const onClip = (lon: number, lat: number) =>
  Math.abs(lon - CLIP.lonMin) < 0.02 ||
  Math.abs(lon - CLIP.lonMax) < 0.02 ||
  Math.abs(lat - CLIP.latMin) < 0.02 ||
  Math.abs(lat - CLIP.latMax) < 0.02;

interface ProjectedRing {
  xy: Float64Array; // projected x, y pairs
  clipEdge: Uint8Array; // 1 if segment k -> k+1 lies on the clip box
}

const RINGS: ProjectedRing[] = (coastGeo as { rings: number[][] }).rings
  .filter((r) => r.length >= 6)
  .map((r) => {
    const n = r.length / 2;
    const xy = new Float64Array(n * 2);
    const clipEdge = new Uint8Array(n);
    for (let k = 0; k < n; k++) {
      const lon = r[2 * k];
      const lat = r[2 * k + 1];
      const [x, y] = lccForward(lat, lon);
      xy[2 * k] = x;
      xy[2 * k + 1] = y;
      const nk = (k + 1) % n;
      clipEdge[k] = onClip(lon, lat) && onClip(r[2 * nk], r[2 * nk + 1]) ? 1 : 0;
    }
    return { xy, clipEdge };
  });

/** Curved outline of the 32x36 model domain (cell edges) as projected points. */
const DOMAIN_OUTLINE: [number, number][] = (() => {
  const la0 = GRID_CONFIG.latMin - HALF;
  const la1 = GRID_CONFIG.latMax + HALF;
  const lo0 = GRID_CONFIG.lonMin - HALF;
  const lo1 = GRID_CONFIG.lonMax + HALF;
  const pts: [number, number][] = [];
  for (let lon = lo0; lon <= lo1; lon += 0.5) pts.push(lccForward(la0, lon));
  for (let lat = la0; lat <= la1; lat += 0.5) pts.push(lccForward(lat, lo1));
  for (let lon = lo1; lon >= lo0; lon -= 0.5) pts.push(lccForward(la1, lon));
  for (let lat = la1; lat >= la0; lat -= 0.5) pts.push(lccForward(lat, lo0));
  return pts;
})();

export function traceDomainOutline(ctx: CanvasRenderingContext2D, view: MapView): void {
  DOMAIN_OUTLINE.forEach(([x, y], k) => {
    const [sx, sy] = view.toScreen(x, y);
    if (k === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  });
  ctx.closePath();
}

export function drawBasemap(ctx: CanvasRenderingContext2D, view: MapView, w: number, h: number): void {
  ctx.save();
  // Ocean: deep navy with a soft centre lift
  const g = ctx.createRadialGradient(w * 0.5, h * 0.45, 0, w * 0.5, h * 0.45, Math.max(w, h) * 0.75);
  g.addColorStop(0, '#0A1424');
  g.addColorStop(1, '#03060C');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  // Land
  ctx.beginPath();
  for (const ring of RINGS) {
    const n = ring.xy.length / 2;
    for (let k = 0; k < n; k++) {
      const [sx, sy] = view.toScreen(ring.xy[2 * k], ring.xy[2 * k + 1]);
      if (k === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.closePath();
  }
  ctx.fillStyle = '#121B2A';
  ctx.fill();

  // Graticule: meridians are straight in LCC, parallels are arcs
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.09)';
  ctx.lineWidth = 0.75;
  ctx.beginPath();
  for (let lon = 60; lon <= 105; lon += 5) {
    const [ax, ay] = view.project(2, lon);
    const [bx, by] = view.project(42, lon);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  for (let lat = 5; lat <= 40; lat += 5) {
    for (let lon = 58; lon <= 107; lon += 1) {
      const [x, y] = view.project(lat, lon);
      if (lon === 58) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
  ctx.restore();
}

/** Coastline stroke (drawn above the data so shorelines stay readable through the field). */
export function drawCoastline(ctx: CanvasRenderingContext2D, view: MapView, alpha = 0.55): void {
  ctx.save();
  ctx.strokeStyle = `rgba(203, 213, 225, ${alpha})`;
  ctx.lineWidth = 0.9;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (const ring of RINGS) {
    const n = ring.xy.length / 2;
    let pen = false;
    for (let k = 0; k <= n; k++) {
      const kk = k % n;
      const [sx, sy] = view.toScreen(ring.xy[2 * kk], ring.xy[2 * kk + 1]);
      if (!pen) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
      pen = k < n && ring.clipEdge[kk] === 0;
    }
  }
  ctx.stroke();
  ctx.restore();
}

/** Darken everything outside the model domain (curved, so it follows the conic grid). */
export function drawDomainVeil(ctx: CanvasRenderingContext2D, view: MapView, w: number, h: number): void {
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, w, h);
  traceDomainOutline(ctx, view);
  ctx.fillStyle = 'rgba(2, 4, 9, 0.55)';
  ctx.fill('evenodd');
  ctx.restore();
}

// ---------------------------------------------------------------------------------------------
// Places
// ---------------------------------------------------------------------------------------------

interface Place {
  name: string;
  lat: number;
  lon: number;
}

const COUNTRIES: Place[] = [
  { name: 'INDIA', lat: 22.6, lon: 78.6 },
  { name: 'PAKISTAN', lat: 28.4, lon: 69.0 },
  { name: 'NEPAL', lat: 28.5, lon: 83.6 },
  { name: 'BANGLADESH', lat: 24.0, lon: 90.2 },
  { name: 'MYANMAR', lat: 21.6, lon: 95.8 },
  { name: 'SRI LANKA', lat: 7.6, lon: 80.7 },
  { name: 'CHINA', lat: 34.0, lon: 89.0 },
  { name: 'AFGHANISTAN', lat: 34.4, lon: 66.6 },
];

const SEAS: Place[] = [
  { name: 'Arabian Sea', lat: 14.5, lon: 66.5 },
  { name: 'Bay of Bengal', lat: 15.0, lon: 88.0 },
  { name: 'Andaman Sea', lat: 11.5, lon: 96.5 },
  { name: 'Indian Ocean', lat: 7.2, lon: 72.0 },
];

export const CITIES: (Place & { major: boolean })[] = [
  { name: 'Delhi', lat: 28.61, lon: 77.21, major: true },
  { name: 'Mumbai', lat: 19.08, lon: 72.88, major: true },
  { name: 'Kolkata', lat: 22.57, lon: 88.36, major: true },
  { name: 'Chennai', lat: 13.08, lon: 80.27, major: true },
  { name: 'Bengaluru', lat: 12.97, lon: 77.59, major: true },
  { name: 'Hyderabad', lat: 17.39, lon: 78.49, major: true },
  { name: 'Dhaka', lat: 23.81, lon: 90.41, major: true },
  { name: 'Karachi', lat: 24.86, lon: 67.0, major: true },
  { name: 'Ahmedabad', lat: 23.02, lon: 72.57, major: false },
  { name: 'Lucknow', lat: 26.85, lon: 80.95, major: false },
  { name: 'Patna', lat: 25.59, lon: 85.14, major: false },
  { name: 'Guwahati', lat: 26.14, lon: 91.74, major: false },
  { name: 'Bhubaneswar', lat: 20.3, lon: 85.82, major: false },
  { name: 'Kathmandu', lat: 27.72, lon: 85.32, major: false },
  { name: 'Nagpur', lat: 21.15, lon: 79.09, major: false },
  { name: 'Jaipur', lat: 26.91, lon: 75.79, major: false },
  { name: 'Lahore', lat: 31.55, lon: 74.34, major: false },
  { name: 'Yangon', lat: 16.84, lon: 96.17, major: false },
  { name: 'Thiruvananthapuram', lat: 8.52, lon: 76.94, major: false },
  { name: 'Colombo', lat: 6.93, lon: 79.86, major: false },
];

function haloText(ctx: CanvasRenderingContext2D, text: string, x: number, y: number): void {
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}

export function drawPlaceLabels(ctx: CanvasRenderingContext2D, view: MapView, zoomRatio: number): void {
  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(3, 6, 12, 0.85)';
  ctx.lineWidth = 3;

  // Countries: wide-tracked caps, quiet
  ctx.font = '600 11px "Inter Variable", "Segoe UI", system-ui, sans-serif';
  ctx.letterSpacing = '3px';
  ctx.fillStyle = 'rgba(226, 232, 240, 0.42)';
  for (const p of COUNTRIES) {
    const [x, y] = view.project(p.lat, p.lon);
    haloText(ctx, p.name, x, y);
  }

  // Seas: italic, ocean blue
  ctx.font = 'italic 500 12px "Inter Variable", "Segoe UI", system-ui, sans-serif';
  ctx.letterSpacing = '1px';
  ctx.fillStyle = 'rgba(125, 175, 230, 0.55)';
  for (const p of SEAS) {
    const [x, y] = view.project(p.lat, p.lon);
    haloText(ctx, p.name, x, y);
  }

  // Cities: major always, the rest once zoomed in
  ctx.letterSpacing = '0px';
  ctx.textAlign = 'left';
  ctx.font = '500 11px "Inter Variable", "Segoe UI", system-ui, sans-serif';
  for (const c of CITIES) {
    if (!c.major && zoomRatio < 1.6) continue;
    const [x, y] = view.project(c.lat, c.lon);
    ctx.fillStyle = 'rgba(248, 250, 252, 0.9)';
    ctx.beginPath();
    ctx.arc(x, y, c.major ? 2.4 : 1.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(3, 6, 12, 0.85)';
    ctx.lineWidth = 3;
    ctx.fillStyle = c.major ? 'rgba(241, 245, 249, 0.85)' : 'rgba(203, 213, 225, 0.7)';
    haloText(ctx, c.name, x + 5, y);
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------------------------
// Plain-language region names for narrative text
// ---------------------------------------------------------------------------------------------

const REGIONS: Place[] = [
  { name: 'the Bay of Bengal', lat: 15, lon: 88 },
  { name: 'the northern Bay of Bengal', lat: 19.5, lon: 89 },
  { name: 'the Arabian Sea', lat: 15, lon: 66.5 },
  { name: 'the Konkan–Goa coast', lat: 16.5, lon: 73.3 },
  { name: 'the Kerala coast', lat: 10, lon: 76.2 },
  { name: 'Tamil Nadu', lat: 11, lon: 78.6 },
  { name: 'the Deccan plateau', lat: 16.5, lon: 77 },
  { name: 'Telangana', lat: 18, lon: 79.2 },
  { name: 'the Odisha coast', lat: 20, lon: 85.8 },
  { name: 'central India', lat: 22.5, lon: 79 },
  { name: 'Gujarat', lat: 22.6, lon: 71.6 },
  { name: 'Rajasthan', lat: 26.6, lon: 73.5 },
  { name: 'the Punjab plains', lat: 30.8, lon: 75.3 },
  { name: 'the upper Ganges plain', lat: 27, lon: 80 },
  { name: 'Bihar and the lower Ganges', lat: 25.6, lon: 85.5 },
  { name: 'West Bengal', lat: 23, lon: 87.8 },
  { name: 'Bangladesh', lat: 23.8, lon: 90.3 },
  { name: 'Northeast India', lat: 26, lon: 92.8 },
  { name: 'the Himalaya', lat: 30.5, lon: 80.5 },
  { name: 'the Tibetan Plateau', lat: 33.5, lon: 88 },
  { name: 'Myanmar', lat: 21, lon: 95.8 },
  { name: 'the Andaman Sea', lat: 12, lon: 96.5 },
  { name: 'Sri Lanka', lat: 7.8, lon: 80.7 },
  { name: 'Pakistan', lat: 28, lon: 68.5 },
  { name: 'Nepal', lat: 28.3, lon: 84 },
  { name: 'Afghanistan', lat: 34.3, lon: 67 },
];

/** Nearest named region (great-circle-ish distance in degrees with cos-lat correction). */
export function regionName(lat: number, lon: number): string {
  let best = REGIONS[0];
  let bestD = Infinity;
  const c = Math.cos((lat * Math.PI) / 180);
  for (const r of REGIONS) {
    const d = (r.lat - lat) ** 2 + ((r.lon - lon) * c) ** 2;
    if (d < bestD) {
      bestD = d;
      best = r;
    }
  }
  return best.name;
}
