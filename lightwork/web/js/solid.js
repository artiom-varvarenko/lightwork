// A 3D model on the table (C2 `solid`): Three.js draws it, in a div of its own over the work area, and does the slicing
// (a clipping plane) and the moving through (a camera that can go inside). The model is loaded once by URL; a scene only
// says how it is seen, because the hands that move it are seen by the server.
const THREE_URL = '/static/vendor/three.module.min.js';
const FOV = 35; // degrees; helper.MODEL_FOV
const NEAR = 0.01;

let THREE = null;
let starting = null;
let el = null;
let renderer = null;
let world = null;
let camera = null;
let group = null;
let plane = null;
let blade = null;
let model = null; // {src, half, parts: [{name, mesh, lines, tri_n, line_n}], t0}
let loading = '';

async function start(rect) {
  THREE = await import(THREE_URL);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
  renderer.setClearColor(0x000000, 1);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.localClippingEnabled = true;
  el.appendChild(renderer.domElement);
  world = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(FOV, rect.w / rect.h, NEAR, 60);
  camera.up.set(0, 0, 1);
  world.add(new THREE.HemisphereLight(0xffffff, 0x8888aa, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.0);
  sun.position.set(-1.5, -2.5, 3);
  world.add(sun);
  group = new THREE.Group();
  world.add(group);
  plane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 2);
  const edge = new THREE.BufferGeometry();
  edge.setAttribute('position', new THREE.BufferAttribute(new Float32Array(15), 3));
  blade = new THREE.Line(edge, new THREE.LineBasicMaterial({ color: 0xffffff }));
  blade.frustumCulled = false;
  blade.visible = false;
  world.add(blade);
}

function dispose() {
  if (!model) return;
  for (const p of model.parts) {
    for (const o of [p.mesh, p.lines]) {
      if (!o) continue;
      group.remove(o);
      o.geometry.dispose();
      o.material.dispose();
    }
  }
  model = null;
}

async function load(src) {
  const manifest = await (await fetch(src)).json();
  const floats = new Float32Array(await (await fetch(manifest.bin)).arrayBuffer());
  dispose();
  const parts = manifest.parts.map((part) => {
    const p = { name: part.name, mesh: null, lines: null, tri_n: part.tri_n, line_n: part.line_n };
    const colour = new THREE.Color(part.color);
    if (part.tri_n) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(floats.subarray(part.tri_at, part.tri_at + part.tri_n * 9), 3));
      geo.computeVertexNormals();
      p.mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: colour, side: THREE.DoubleSide, clippingPlanes: [plane], flatShading: true }));
      group.add(p.mesh);
    }
    if (part.line_n) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(floats.subarray(part.line_at, part.line_at + part.line_n * 6), 3));
      p.lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: colour, clippingPlanes: [plane] }));
      group.add(p.lines);
    }
    return p;
  });
  model = { src, half: manifest.half, parts, t0: 0 };
}

function fade(o, faded) {
  if (!o) return;
  o.material.transparent = faded;
  o.material.opacity = faded ? 0.1 : 1;
  o.material.depthWrite = !faded;
}

// The parts come up one after the other, in the order the program built them.
function reveal(p, i, n, f) {
  const k = Math.min(1, Math.max(0, f * n - i));
  const visible = k > 0;
  if (p.mesh) {
    p.mesh.visible = visible;
    p.mesh.geometry.setDrawRange(0, Math.ceil(k * p.tri_n) * 3);
  }
  if (p.lines) {
    p.lines.visible = visible;
    p.lines.geometry.setDrawRange(0, Math.ceil(k * p.line_n) * 2);
  }
}

function setBlade(prim) {
  const on = prim.cut !== null && prim.cut !== undefined;
  blade.visible = on;
  const a = prim.cut_axis;
  const normal = [0, 0, 0];
  normal[a] = -1;
  plane.normal.set(normal[0], normal[1], normal[2]);
  plane.constant = on ? prim.cut : 100; // keeps what lies below the cut
  if (!on) return;
  const b = (a + 1) % 3, c = (a + 2) % 3;
  const hb = model.half[b] * 1.08, hc = model.half[c] * 1.08;
  const pos = blade.geometry.attributes.position;
  [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]].forEach(([sb, sc], i) => {
    const v = [0, 0, 0];
    v[a] = prim.cut;
    v[b] = sb * hb;
    v[c] = sc * hc;
    pos.setXYZ(i, v[0], v[1], v[2]);
  });
  pos.needsUpdate = true;
}

function draw(prim, rect) {
  renderer.setSize(rect.w, rect.h);
  camera.aspect = rect.w / rect.h;
  camera.updateProjectionMatrix();
  const c = Math.cos(prim.tilt), s = Math.sin(prim.tilt);
  camera.position.set(prim.target[0] - prim.dist * c * Math.sin(prim.yaw), prim.target[1] - prim.dist * c * Math.cos(prim.yaw),
                      prim.target[2] + prim.dist * s);
  camera.lookAt(prim.target[0], prim.target[1], prim.target[2]);
  setBlade(prim);
  const now = performance.now();
  if (!model.t0) model.t0 = now;
  const f = prim.draw_ms ? Math.min(1, (now - model.t0) / prim.draw_ms) : 1;
  const low = (prim.only || '').toLowerCase();
  model.parts.forEach((p, i) => {
    reveal(p, i, model.parts.length, f);
    const faded = !!low && p.name.toLowerCase() !== low;
    fade(p.mesh, faded);
    fade(p.lines, faded);
  });
  renderer.render(world, camera);
  return f < 1;
}

// `prim` is the scene's solid or null; `toPage` maps camera px to page px (unturned), W x H the page, `rotate` its turn.
// True while it is still building itself: the caller should keep drawing.
export function syncSolid(prim, toPage, W, H, rotate) {
  if (!prim) {
    if (el) el.style.display = 'none';
    return false;
  }
  const pts = prim.corners.map((c) => toPage(c[0], c[1]));
  const x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
  const y0 = Math.min(...pts.map((p) => p[1])), y1 = Math.max(...pts.map((p) => p[1]));
  const turned = rotate === 180;
  const rect = { left: turned ? W - x1 : x0, top: turned ? H - y1 : y0, w: Math.round(x1 - x0), h: Math.round(y1 - y0) };
  if (!el) {
    el = document.createElement('div');
    el.id = 'tablesolid';
    el.style.position = 'fixed';
    el.style.zIndex = '1';
    el.style.background = '#000';
    document.body.appendChild(el);
  }
  el.style.display = 'block';
  el.style.left = `${rect.left}px`;
  el.style.top = `${rect.top}px`;
  el.style.width = `${rect.w}px`;
  el.style.height = `${rect.h}px`;
  el.style.transform = turned ? 'rotate(180deg)' : '';
  if (!renderer) {
    if (!starting) starting = start(rect).catch((e) => console.warn('3D:', e));
    return true;
  }
  if ((!model || model.src !== prim.src) && loading !== prim.src) {
    loading = prim.src;
    load(prim.src).catch((e) => console.warn('model:', e)).finally(() => { loading = ''; });
  }
  if (!model || model.src !== prim.src) {
    return true;
  }
  return draw(prim, rect);
}

// For the checks.
window.lightworkSolidState = () => (model ? { src: model.src, parts: model.parts.map((p) => p.name), half: model.half,
                                              calls: renderer.info.render.calls, triangles: renderer.info.render.triangles } : null);
