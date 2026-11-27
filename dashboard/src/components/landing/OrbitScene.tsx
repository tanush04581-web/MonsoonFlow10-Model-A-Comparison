/**
 * OrbitScene.tsx - The 3D prologue: a lit, monochrome Earth seen from deep space. Scroll scrubs a
 * long-lens camera along an arc that ends on India, then keeps diving towards it. A procedural
 * cloud bank (fbm shader, two parallax layers) rolls in; while the screen is fully covered the Earth
 * is switched off and the 2D weather chart is switched on underneath, then the clouds part over it.
 *
 * Real data only: the moisture haze (tcwv), rain glow (tp) and wind streamlines (u10, v10) are the
 * model's Day-0 tensors, feathered to zero at the domain edges. The Earth surface is NASA Blue
 * Marble NG (July 2004, public domain), used only as a land/ocean mask + relief and shaded in greys.
 *
 * Lazy-loaded (three.js stays out of the lab bundle). DPR capped, rendering stops while hidden,
 * every GPU resource is disposed on unmount. prefers-reduced-motion: static camera at the hand-off
 * framing, no drift, frozen streamlines, render-on-change only.
 */

import { useEffect, useRef } from 'react';
import {
  AdditiveBlending,
  BackSide,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Points,
  Scene,
  ShaderMaterial,
  SphereGeometry,
  SRGBColorSpace,
  type Texture,
  TextureLoader,
  Vector3,
  WebGLRenderer,
} from 'three';
import worldGeo from '../../assets/geo/world110.json';
import { CHANNELS, GRID_CONFIG, type Grid2D } from '../../types/climate';
import { sampleBilinear } from '../../services/fieldRaster';
import { CLOUD, heroGlobeFrame } from './globeFrame';
import type { StoryData } from './useStoryData';
import { WindParticles, domainFeather } from './windParticles';

export interface OrbitSceneProps {
  /** Scroll position in screens from the top of the story section. */
  rawRef: React.RefObject<number>;
  /** Screens of 3D prologue (the orbit arc ends LOCK_BEFORE screens before this). */
  prologue: number;
  data: StoryData | null;
  isDesktop: boolean;
  reduced: boolean;
  /** Optional element that shows the camera altitude / sub-camera point. */
  altitudeRef?: React.RefObject<HTMLElement | null>;
  onStatus: (s: 'ready' | 'failed') => void;
}

const D2R = Math.PI / 180;
const EARTH_KM = 6371;
// Camera distance in Earth radii. A very long lens at the hand-off makes the perspective globe
// match the 2D orthographic globe to ~1 % of R, so the cross-fade does not double coastlines.
const D_START = 150;
const D_END = 56;
const STAR_R = 3000;
/** The orbit arc settles on India this many screens before the prologue ends. */
const LOCK_BEFORE = 0.34;
/** Camera distance multiplier reached when the clouds close (globe ~2.4x larger: a dive). */
const DIVE_SCALE = 0.42;
const LON_START = -28;
const LAT_START = 3;
const ROLL_START = 0.2;
const FEATHER = 6;
const TEXTURE_URL = `${import.meta.env.BASE_URL}textures/earth-bmng-july-4k.jpg`;

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
const smooth = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};
const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Unit vector for (lat, lon) in the frame of three's SphereGeometry + equirectangular UVs. */
function dirOf(lat: number, lon: number, out = new Vector3()): Vector3 {
  const p = lat * D2R;
  const l = lon * D2R;
  return out.set(Math.cos(p) * Math.cos(l), Math.sin(p), -Math.cos(p) * Math.sin(l));
}

// ---------------------------------------------------------------------------------------------
// Shaders (monochrome palette in linear space: #171717 .0086, #262626 .0194, #404040 .0513,
// #737373 .1714, #E5E5E5 .7835)
// ---------------------------------------------------------------------------------------------

const EARTH_VS = /* glsl */ `
varying vec2 vUv; varying vec3 vN; varying vec3 vP;
void main() {
  vUv = uv;
  vN = normalize(mat3(modelMatrix) * normal);
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vP = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

// Land/ocean comes from uMask (the same Natural Earth rings the 2D globe draws), never from the
// imagery's colours: the Blue Marble texture has shaded bathymetry (shelves, slopes, ridges) that
// a colour test mistakes for land, which showed as offset "shadow" copies of the continents.
// The imagery only modulates relief INSIDE the land mask; the ocean is flat.
const EARTH_FS = /* glsl */ `
uniform sampler2D uDay; uniform sampler2D uMask; uniform float uHasTex; uniform vec3 uSun; uniform vec3 uCam; uniform float uChart;
varying vec2 vUv; varying vec3 vN; varying vec3 vP;
void main() {
  float water = 1.0 - smoothstep(0.3, 0.7, texture2D(uMask, vUv).r);
  vec3 tex = uHasTex > 0.5 ? texture2D(uDay, vUv).rgb : vec3(0.06);
  float lum = dot(tex, vec3(0.2126, 0.7152, 0.0722));
  float land = mix(0.0513, 0.1714, smoothstep(0.03, 0.42, lum));
  land = mix(land, 0.5, smoothstep(0.55, 0.9, lum));            // ice and snow
  float ocean = 0.0194;                                         // flat Charcoal
  float albedo = mix(land, ocean, water);

  vec3 N = normalize(vN);
  vec3 V = normalize(uCam - vP);
  float ndl = dot(N, uSun);
  float dayK = smoothstep(-0.12, 0.32, ndl);
  float lit = albedo * (0.3 + 1.05 * max(ndl, 0.0));
  float night = 0.0086 + albedo * 0.1;
  float g = mix(night, lit, dayK);
  vec3 H = normalize(uSun + V);
  g += pow(max(dot(N, H), 0.0), 70.0) * water * 0.32 * dayK;     // sun glint on the ocean
  g += 0.7835 * pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.32 * (0.25 + 0.75 * dayK); // rim light

  float chart = mix(0.0513, 0.0194, water);                       // the 2D globe's flat greys
  g = mix(g, chart, uChart);
  gl_FragColor = vec4(vec3(g), 1.0);
  #include <colorspace_fragment>
}`;

const ATMO_VS = EARTH_VS;
const ATMO_FS = /* glsl */ `
uniform vec3 uSun; uniform vec3 uCam; uniform float uStrength;
varying vec2 vUv; varying vec3 vN; varying vec3 vP;
void main() {
  vec3 N = normalize(vN);
  vec3 V = normalize(uCam - vP);
  float k = clamp(-dot(N, V), 0.0, 1.0);
  float glow = pow(clamp(k / 0.43, 0.0, 1.0), 4.0);
  float sunF = smoothstep(-0.4, 0.6, dot(N, uSun));
  float a = glow * (0.12 + 0.88 * sunF) * uStrength;
  gl_FragColor = vec4(vec3(0.7835) * a, a);
  #include <colorspace_fragment>
}`;

const STAR_VS = /* glsl */ `
attribute float aSize; attribute float aPhase;
uniform float uTime; uniform float uPx;
varying float vA;
void main() {
  vA = 0.55 + 0.45 * sin(uTime * (0.6 + aPhase) + aPhase * 40.0);
  gl_PointSize = aSize * uPx;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const STAR_FS = /* glsl */ `
uniform float uOpacity; varying float vA;
void main() {
  float d = length(gl_PointCoord - 0.5);
  float a = smoothstep(0.5, 0.0, d) * vA * uOpacity;
  gl_FragColor = vec4(vec3(0.898) * a, a);
}`;

// Full-screen cloud bank: two fbm layers that expand at different rates as the camera "flies
// through" (uDepth), lit from the upper left, in the neutral palette. Straight alpha output.
const CLOUD_VS = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const CLOUD_FS = /* glsl */ `
uniform float uCover; uniform float uTime; uniform float uDepth; uniform float uAspect;
varying vec2 vUv;
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float noise(vec2 p) {
  vec2 i = floor(p); vec2 f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0; float a = 0.5; mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
  for (int i = 0; i < 5; i++) { v += a * noise(p); p = m * p; a *= 0.5; }
  return v;
}
// fbm clusters around 0.5 (a sum of octaves is near-Gaussian); stretch it to 0..1 so a threshold
// yields real gaps between clouds instead of a uniform veil.
float layer(vec2 uv, float scale, float speed, float zoom, float seed) {
  vec2 c = (uv - 0.5) * vec2(uAspect, 1.0) / zoom;
  return smoothstep(0.28, 0.72, fbm(c * scale + vec2(uTime * speed, -uTime * speed * 0.6) + seed));
}
void main() {
  float zFar = 1.0 + uDepth * 0.9;
  float zNear = 1.0 + uDepth * 2.6;
  float nFar = layer(vUv, 1.3, 0.015, zFar, 3.1);
  float nNear = layer(vUv, 2.1, 0.03, zNear, 7.7);
  // Coverage 0 = clear sky, 1 = solid bank. A gap needs BOTH layers clear, so the far layer is
  // kept sparser and coverage is eased, otherwise the bank looks closed until the very end.
  float thr = mix(1.02, -0.2, pow(uCover, 1.8));
  float aFar = smoothstep(thr + 0.14, thr + 0.3, nFar);
  float aNear = smoothstep(thr, thr + 0.16, nNear);
  // Fake sun from the upper left: brighter where the cloud thickens towards the light
  float nLight = layer(vUv + vec2(-0.02, 0.024), 2.1, 0.03, zNear, 7.7);
  float lit = clamp(0.5 + (nNear - nLight) * 3.0, 0.0, 1.0);
  float thick = smoothstep(0.2, 0.9, nNear);
  vec3 cFar = vec3(mix(0.0194, 0.17, nFar));
  vec3 cNear = vec3(mix(0.0194, 0.7835, pow(lit, 1.8) * (0.35 + 0.65 * thick)));
  float a = aNear + aFar * (1.0 - aNear);
  vec3 col = (cFar * aFar * (1.0 - aNear) + cNear * aNear) / max(a, 1e-3);
  gl_FragColor = vec4(col, a);
  #include <colorspace_fragment>
}`;

// ---------------------------------------------------------------------------------------------

/** Paint a feathered neutral overlay (north-up, S px per cell) from a real grid. */
function paintLayer(grid: Grid2D, alphaOf: (v: number) => number, S: number): HTMLCanvasElement {
  const W = GRID_CONFIG.nLon * S;
  const H = GRID_CONFIG.nLat * S;
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const ctx = c.getContext('2d');
  if (!ctx) return c;
  const img = ctx.createImageData(W, H);
  for (let y = 0; y < H; y++) {
    const fi = GRID_CONFIG.nLat - 0.5 - (y + 0.5) / S;
    for (let x = 0; x < W; x++) {
      const fj = (x + 0.5) / S - 0.5;
      const a = alphaOf(sampleBilinear(grid, fi, fj)) * domainFeather(fi, fj, FEATHER);
      const o = (y * W + x) * 4;
      img.data[o] = 229;
      img.data[o + 1] = 229;
      img.data[o + 2] = 229;
      img.data[o + 3] = Math.round(255 * Math.max(0, Math.min(1, a)));
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

/**
 * Equirectangular land mask (white = land) from the bundled Natural Earth 1:110m rings, so the 3D
 * coastlines are identical to the 2D globe's at the hand-off. Longitudes are unwrapped so rings
 * that cross the antimeridian stay continuous, and each ring is drawn at -360/0/+360 to wrap.
 * Non-zero fill keeps lakes (opposite winding) as water.
 */
function landMask(W: number, H: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const ctx = c.getContext('2d');
  if (!ctx) return c;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  const sx = W / 360;
  const sy = H / 180;
  ctx.beginPath();
  for (const r of (worldGeo as { rings: number[][] }).rings) {
    if (r.length < 6) continue;
    const pts: number[] = [];
    let off = 0;
    let prev = r[0];
    for (let k = 0; k < r.length; k += 2) {
      let lon = r[k] + off;
      if (lon - prev > 180) {
        off -= 360;
        lon -= 360;
      } else if (lon - prev < -180) {
        off += 360;
        lon += 360;
      }
      prev = lon;
      pts.push(lon, r[k + 1]);
    }
    for (const shift of [-360, 0, 360]) {
      for (let k = 0; k < pts.length; k += 2) {
        const x = (pts[k] + shift + 180) * sx;
        const y = (90 - pts[k + 1]) * sy;
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
    }
  }
  ctx.fillStyle = '#fff';
  ctx.fill();
  return c;
}

/** Sphere patch covering exactly the model domain (cell edges), UVs north-up. */
function domainPatch(radius: number): SphereGeometry {
  const half = GRID_CONFIG.resolutionDeg / 2;
  const lon0 = GRID_CONFIG.lonMin - half;
  const lon1 = GRID_CONFIG.lonMax + half;
  const lat0 = GRID_CONFIG.latMin - half;
  const lat1 = GRID_CONFIG.latMax + half;
  return new SphereGeometry(radius, 48, 44, (lon0 + 180) * D2R, (lon1 - lon0) * D2R, (90 - lat1) * D2R, (lat1 - lat0) * D2R);
}

export default function OrbitScene({ rawRef, prologue, data, isDesktop, reduced, altitudeRef, onStatus }: OrbitSceneProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const dataRef = useRef<StoryData | null>(data);
  const setDataRef = useRef<((d: StoryData) => void) | null>(null);
  const statusRef = useRef(onStatus);

  useEffect(() => {
    statusRef.current = onStatus;
  }, [onStatus]);

  useEffect(() => {
    dataRef.current = data;
    if (data) setDataRef.current?.(data);
  }, [data]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    // Each run owns a fresh canvas: the cleanup below force-loses its context, and a canvas
    // whose context was lost cannot host a new renderer (StrictMode / breakpoint re-runs).
    const canvas = document.createElement('canvas');
    canvas.className = 'absolute inset-0 w-full h-full';
    canvas.setAttribute('aria-hidden', 'true');
    host.appendChild(canvas);
    let renderer: WebGLRenderer;
    try {
      renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    } catch {
      canvas.remove();
      statusRef.current('failed');
      return;
    }
    renderer.setClearColor(0x000000, 0);
    const maxDpr = isDesktop ? 2 : 1.5;
    renderer.setPixelRatio(Math.min(maxDpr, window.devicePixelRatio || 1));

    const scene = new Scene();
    const camera = new PerspectiveCamera(4, 1, 5, STAR_R * 2);
    const disposables: { dispose: () => void }[] = [];
    const track = <T extends { dispose: () => void }>(x: T): T => {
      disposables.push(x);
      return x;
    };

    // Sun fixed in the world: lights India from the upper left at the hand-off framing
    const cEnd = dirOf(17, 82);
    const left = new Vector3().crossVectors(cEnd, new Vector3(0, 1, 0)).normalize();
    const sun = cEnd.clone().multiplyScalar(0.72).addScaledVector(left, 0.55).add(new Vector3(0, 0.38, 0)).normalize();

    // Earth
    const maskTex = track(new CanvasTexture(landMask(isDesktop ? 4096 : 2048, isDesktop ? 2048 : 1024)));
    maskTex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    const earthUniforms = {
      uDay: { value: null as Texture | null },
      uMask: { value: maskTex as Texture },
      uHasTex: { value: 0 },
      uSun: { value: sun },
      uCam: { value: new Vector3() },
      uChart: { value: 0 },
    };
    const earthGeo = track(new SphereGeometry(1, isDesktop ? 160 : 96, isDesktop ? 112 : 64));
    const earthMat = track(new ShaderMaterial({ uniforms: earthUniforms, vertexShader: EARTH_VS, fragmentShader: EARTH_FS }));
    const earth = new Mesh(earthGeo, earthMat);
    scene.add(earth);

    // Atmosphere rim (back faces, additive)
    const atmoUniforms = { uSun: { value: sun }, uCam: earthUniforms.uCam, uStrength: { value: 0.75 } };
    const atmoGeo = track(new SphereGeometry(1.11, 96, 64));
    const atmoMat = track(
      new ShaderMaterial({
        uniforms: atmoUniforms,
        vertexShader: ATMO_VS,
        fragmentShader: ATMO_FS,
        side: BackSide,
        transparent: true,
        depthWrite: false,
        blending: AdditiveBlending,
      })
    );
    const atmo = new Mesh(atmoGeo, atmoMat);
    scene.add(atmo);

    // Stars: concentrated behind the Earth along the camera path (the long lens sees a narrow sky)
    const nStars = isDesktop ? 3200 : 1700;
    const sp = new Float32Array(nStars * 3);
    const ss = new Float32Array(nStars);
    const sph = new Float32Array(nStars);
    const tmp = new Vector3();
    for (let i = 0; i < nStars; i++) {
      const u = Math.random();
      dirOf(LAT_START + (17 - LAT_START) * u, LON_START + (82 - LON_START) * u, tmp).negate();
      tmp.x += (Math.random() - 0.5) * 0.9;
      tmp.y += (Math.random() - 0.5) * 0.7;
      tmp.z += (Math.random() - 0.5) * 0.9;
      tmp.normalize().multiplyScalar(STAR_R);
      sp.set([tmp.x, tmp.y, tmp.z], i * 3);
      ss[i] = Math.random() < 0.06 ? 2.2 + Math.random() * 1.6 : 0.7 + Math.random() * 1.3;
      sph[i] = Math.random();
    }
    const starGeo = track(new BufferGeometry());
    starGeo.setAttribute('position', new BufferAttribute(sp, 3));
    starGeo.setAttribute('aSize', new BufferAttribute(ss, 1));
    starGeo.setAttribute('aPhase', new BufferAttribute(sph, 1));
    const starUniforms = { uTime: { value: 0 }, uPx: { value: renderer.getPixelRatio() }, uOpacity: { value: 1 } };
    const starMat = track(
      new ShaderMaterial({ uniforms: starUniforms, vertexShader: STAR_VS, fragmentShader: STAR_FS, transparent: true, depthWrite: false, blending: AdditiveBlending })
    );
    const stars = new Points(starGeo, starMat);
    scene.add(stars);

    // Cloud bank (drawn last, over everything, ignores the camera)
    const cloudUniforms = { uCover: { value: 0 }, uTime: { value: 0 }, uDepth: { value: 0 }, uAspect: { value: 1 } };
    const cloudGeo = track(new PlaneGeometry(2, 2));
    const cloudMat = track(
      new ShaderMaterial({ uniforms: cloudUniforms, vertexShader: CLOUD_VS, fragmentShader: CLOUD_FS, transparent: true, depthTest: false, depthWrite: false })
    );
    const clouds = new Mesh(cloudGeo, cloudMat);
    clouds.frustumCulled = false;
    clouds.renderOrder = 999;
    clouds.visible = false;
    scene.add(clouds);

    // Data layers (built when data arrives)
    const moistMat = track(new MeshBasicMaterial({ transparent: true, depthWrite: false, opacity: 0, blending: AdditiveBlending }));
    const rainMat = track(new MeshBasicMaterial({ transparent: true, depthWrite: false, opacity: 0, blending: AdditiveBlending }));
    const moistGeo = track(domainPatch(1.016));
    const rainGeo = track(domainPatch(1.008));
    const moist = new Mesh(moistGeo, moistMat);
    const rain = new Mesh(rainGeo, rainMat);
    moist.visible = false;
    rain.visible = false;
    scene.add(moist, rain);
    const layerTextures: Texture[] = [];

    // Wind streamlines (real Day-0 u10/v10)
    const wind = new WindParticles(isDesktop ? 1100 : 600, 16, FEATHER);
    const segs = wind.count * (wind.trail - 1);
    const lPos = new Float32Array(segs * 6);
    const lCol = new Float32Array(segs * 6);
    const lineGeo = track(new BufferGeometry());
    lineGeo.setAttribute('position', new BufferAttribute(lPos, 3));
    lineGeo.setAttribute('color', new BufferAttribute(lCol, 3));
    const lineMat = track(new LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, blending: AdditiveBlending }));
    const lines = new LineSegments(lineGeo, lineMat);
    lines.frustumCulled = false;
    lines.visible = false;
    scene.add(lines);
    let windGrids: { u: Grid2D; v: Grid2D } | null = null;
    let dirty = true;

    const setData = (d: StoryData) => {
      layerTextures.forEach((t) => t.dispose());
      layerTextures.length = 0;
      const tcwv = d.day0[CHANNELS.tcwv.index];
      const tp = d.day0[CHANNELS.tp.index];
      const mk = (c: HTMLCanvasElement) => {
        const t = new CanvasTexture(c);
        t.colorSpace = SRGBColorSpace;
        layerTextures.push(t);
        return t;
      };
      moistMat.map = mk(paintLayer(tcwv, (v) => smooth(40, 68, v) * 0.14, 8));
      rainMat.map = mk(paintLayer(tp, (v) => Math.pow(smooth(2, 45, v), 1.15) * 0.42, 10));
      moistMat.needsUpdate = true;
      rainMat.needsUpdate = true;
      moist.visible = true;
      rain.visible = true;
      windGrids = { u: d.day0[CHANNELS.u10.index], v: d.day0[CHANNELS.v10.index] };
      lines.visible = true;
      if (reduced) for (let s = 0; s < 40; s++) wind.step(1 / 60, windGrids, 0.5); // frozen streamlines
      dirty = true;
    };
    setDataRef.current = setData;

    // Earth imagery
    let texture: Texture | null = null;
    let alive = true;
    new TextureLoader().load(
      TEXTURE_URL,
      (t) => {
        if (!alive) {
          t.dispose();
          return;
        }
        t.colorSpace = SRGBColorSpace;
        t.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
        texture = t;
        earthUniforms.uDay.value = t;
        earthUniforms.uHasTex.value = 1;
        dirty = true;
        statusRef.current('ready');
      },
      undefined,
      () => {
        if (alive) statusRef.current('ready'); // the grey procedural sphere still works
      }
    );

    // Size
    let w = 1;
    let h = 1;
    const resize = () => {
      const r = canvas.getBoundingClientRect();
      w = Math.max(1, r.width);
      h = Math.max(1, r.height);
      renderer.setPixelRatio(Math.min(maxDpr, window.devicePixelRatio || 1));
      renderer.setSize(w, h, false);
      starUniforms.uPx.value = renderer.getPixelRatio();
      dirty = true;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    // Pointer parallax (depth between Earth and stars)
    const ptr = { x: 0, y: 0, sx: 0, sy: 0 };
    const onPointer = (e: PointerEvent) => {
      ptr.x = (e.clientX / window.innerWidth) * 2 - 1;
      ptr.y = (e.clientY / window.innerHeight) * 2 - 1;
    };
    if (!reduced) window.addEventListener('pointermove', onPointer, { passive: true });

    const camPos = new Vector3();
    let raf = 0;
    let last = performance.now();
    let lastRaw = -1;
    let lastAlt = 0;
    const t0 = performance.now();

    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const dt = (now - last) / 1000;
      last = now;
      const raw = Math.max(0, rawRef.current ?? 0);
      if (raw >= CLOUD.out1) {
        canvas.style.opacity = '0';
        lastRaw = raw;
        return; // clouds have parted over the chart: no GPU work
      }
      canvas.style.opacity = '1';
      const cover = smooth(CLOUD.in0, CLOUD.full, raw) * (1 - smooth(CLOUD.out0, CLOUD.out1, raw));
      const earthOn = raw < CLOUD.swap;
      if (reduced && !dirty && Math.abs(raw - lastRaw) < 1e-4) return;
      lastRaw = raw;
      dirty = false;

      const fr = heroGlobeFrame(w, h, isDesktop);
      const arcEnd = prologue - LOCK_BEFORE;
      const u = reduced ? 1 : clamp01(raw / arcEnd);
      const e = easeInOutCubic(u);
      // After the arc, keep diving towards India until the clouds close
      const dive = reduced ? 0 : Math.pow(clamp01((raw - arcEnd) / (CLOUD.full - arcEnd)), 1.6);
      const tm = (now - t0) / 1000;
      const settle = 1 - smooth(0.6, 0.95, u); // drift and parallax settle as the arc ends
      ptr.sx += (ptr.x - ptr.sx) * 0.05;
      ptr.sy += (ptr.y - ptr.sy) * 0.05;
      const drift = reduced ? 0 : settle;
      const lat = LAT_START + (fr.lat - LAT_START) * e + 4 * dive + drift * (1.2 * Math.sin(tm * 0.05) - ptr.sy * 1.4);
      const lon = LON_START + (fr.lon - LON_START) * e - 2 * dive + drift * (3 * Math.sin(tm * 0.04) + ptr.sx * 2.2);
      const D = D_START * Math.pow(D_END / D_START, e) * Math.pow(DIVE_SCALE, dive);
      dirOf(lat, lon, camPos).multiplyScalar(D);
      camera.position.copy(camPos);
      camera.up.set(0, 1, 0);
      camera.lookAt(0, 0, 0);
      camera.rotateZ(ROLL_START * (1 - e) * (reduced ? 0 : 1));

      // Long lens: the globe has radius R at the end of the arc (then grows during the dive)
      const tanHalf = (Math.tan(Math.asin(1 / D_END)) * (h / 2)) / Math.max(20, fr.R);
      camera.fov = (2 * Math.atan(tanHalf)) / D2R;
      camera.aspect = w / h;
      camera.setViewOffset(w, h, w / 2 - fr.cx, h / 2 - fr.cy, w, h);
      camera.updateProjectionMatrix();
      earthUniforms.uCam.value.copy(camPos);

      // The Earth is only switched off while the clouds fully cover the screen
      earth.visible = earthOn;
      atmo.visible = earthOn;
      stars.visible = earthOn;
      moist.visible = earthOn && windGrids !== null;
      rain.visible = earthOn && windGrids !== null;
      starUniforms.uTime.value = reduced ? 0 : tm;
      starUniforms.uOpacity.value = 1 - dive;
      const dataIn = smooth(0.35, 1.05, raw);
      moistMat.opacity = dataIn;
      rainMat.opacity = dataIn;

      clouds.visible = cover > 0.002;
      cloudUniforms.uCover.value = cover;
      cloudUniforms.uTime.value = reduced ? 0 : tm;
      cloudUniforms.uDepth.value = smooth(CLOUD.in0, CLOUD.out1, raw);
      cloudUniforms.uAspect.value = w / h;

      if (earthOn && windGrids && dataIn > 0.01) {
        if (!reduced) wind.step(dt, windGrids, 0.5);
        const T = wind.trail;
        let o = 0;
        const v0 = new Vector3();
        const v1 = new Vector3();
        for (let k = 0; k < wind.count; k++) {
          const n = wind.len[k];
          const a0 = wind.alpha(k) * (0.35 + 0.65 * Math.min(1, wind.speed[k] / 12)) * dataIn;
          const b = k * T * 2;
          for (let s = 0; s < T - 1; s++) {
            if (s + 1 < n && a0 > 0.02) {
              const h0 = wind.hist[b + s * 2];
              const g0 = wind.hist[b + s * 2 + 1];
              const h1 = wind.hist[b + s * 2 + 2];
              const g1 = wind.hist[b + s * 2 + 3];
              dirOf(GRID_CONFIG.latMin + h0, GRID_CONFIG.lonMin + g0, v0).multiplyScalar(1.012);
              dirOf(GRID_CONFIG.latMin + h1, GRID_CONFIG.lonMin + g1, v1).multiplyScalar(1.012);
              const c0 = a0 * (1 - s / n);
              const c1 = a0 * (1 - (s + 1) / n);
              lPos[o] = v0.x;
              lPos[o + 1] = v0.y;
              lPos[o + 2] = v0.z;
              lPos[o + 3] = v1.x;
              lPos[o + 4] = v1.y;
              lPos[o + 5] = v1.z;
              lCol[o] = lCol[o + 1] = lCol[o + 2] = c0;
              lCol[o + 3] = lCol[o + 4] = lCol[o + 5] = c1;
            } else {
              lCol.fill(0, o, o + 6);
            }
            o += 6;
          }
        }
        (lineGeo.attributes.position as BufferAttribute).needsUpdate = true;
        (lineGeo.attributes.color as BufferAttribute).needsUpdate = true;
        lines.visible = true;
      } else {
        lines.visible = false;
      }

      // Telemetry: camera altitude above the surface + sub-camera point (geometry, not data)
      const el = altitudeRef?.current;
      if (el && now - lastAlt > 120) {
        lastAlt = now;
        const alt = Math.round(((D - 1) * EARTH_KM) / 100) * 100;
        el.textContent = `ALT ${alt.toLocaleString('en-US')} km · ${Math.abs(lat).toFixed(1)}°${lat >= 0 ? 'N' : 'S'} ${Math.abs(((lon + 540) % 360) - 180).toFixed(1)}°${((lon + 540) % 360) - 180 >= 0 ? 'E' : 'W'}`;
      }

      renderer.render(scene, camera);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('pointermove', onPointer);
      setDataRef.current = null;
      layerTextures.forEach((t) => t.dispose());
      texture?.dispose();
      disposables.forEach((d) => d.dispose());
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    };
  }, [isDesktop, reduced, prologue, rawRef, altitudeRef]);

  // Apply data that arrived before the scene existed
  useEffect(() => {
    if (dataRef.current) setDataRef.current?.(dataRef.current);
  }, [isDesktop, reduced]);

  return <div ref={hostRef} className="absolute inset-0" aria-hidden="true" />;
}
