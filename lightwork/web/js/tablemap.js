// The 3D map on the table (C2 `map`). MapLibre GL draws it, in a div of its own over the work area; the camera comes from
// the server in every scene, because the hands that move it are seen there. Loaded from the CDN the first time a map is
// asked for: nothing of it runs otherwise. The style's attribution stays on screen, as OpenFreeMap requires.
const VERSION = '4.7.1';
const BASE = `https://unpkg.com/maplibre-gl@${VERSION}/dist/`;
const STYLE = 'https://tiles.openfreemap.org/styles/liberty';

let el = null;
let map = null;
let loading = null;
let flight = 0;
let flyEnd = 0;

function load() {
  if (loading) return loading;
  loading = new Promise((resolve, reject) => {
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = `${BASE}maplibre-gl.css`;
    document.head.appendChild(css);
    const js = document.createElement('script');
    js.src = `${BASE}maplibre-gl.js`;
    js.onload = resolve;
    js.onerror = reject;
    document.head.appendChild(js);
  });
  return loading;
}

async function create(prim) {
  await load();
  map = new window.maplibregl.Map({
    container: el,
    style: STYLE,
    center: [prim.center[0], 20],
    zoom: 1.4,
    interactive: false,
    fadeDuration: 0,
    maxPitch: 75,
  });
  map.on('error', (e) => console.warn('map:', e && e.error && e.error.message));
  place(el, prim.place); // its size may have changed while the library loaded
  map.resize();
}

function place(div, rect) {
  div.style.left = `${rect.left}px`;
  div.style.top = `${rect.top}px`;
  div.style.width = `${rect.w}px`;
  div.style.height = `${rect.h}px`;
  div.style.transform = rect.turned ? 'rotate(180deg)' : '';
}

// `prim` is the scene's map or null; `toPage` maps camera px to page px (unturned), W x H the page, `rotate` its turn.
export function syncMap(prim, toPage, W, H, rotate) {
  if (!prim) {
    if (el) el.style.display = 'none';
    return;
  }
  const pts = prim.corners.map((c) => toPage(c[0], c[1]));
  const x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
  const y0 = Math.min(...pts.map((p) => p[1])), y1 = Math.max(...pts.map((p) => p[1]));
  const turned = rotate === 180;
  const rect = { left: turned ? W - x1 : x0, top: turned ? H - y1 : y0, w: x1 - x0, h: y1 - y0, turned };
  if (!el) {
    el = document.createElement('div');
    el.id = 'tablemap';
    el.style.position = 'fixed';
    el.style.zIndex = '1';
    document.body.appendChild(el);
  }
  const hidden = el.style.display === 'none';
  el.style.display = 'block';
  place(el, rect);
  prim = { ...prim, place: rect };
  if (!map) {
    if (!loading) create(prim).catch((e) => console.warn('map library:', e));
    return;
  }
  if (hidden) map.resize();
  const now = performance.now();
  const camera = { center: prim.center, zoom: prim.zoom, bearing: prim.bearing, pitch: prim.pitch };
  if (prim.flight !== flight) {
    flight = prim.flight;
    flyEnd = now + prim.fly_ms;
    map.flyTo({ ...camera, duration: prim.fly_ms, essential: true });
  } else if (now >= flyEnd) {
    map.jumpTo(camera);
  }
}

// For the checks.
export const mapState = () => (map ? { loaded: map.loaded(), center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing(),
                                       pitch: map.getPitch(), attribution: !!document.querySelector('.maplibregl-ctrl-attrib') } : null);
window.lightworkMapState = mapState;
