// Projector page (C2 role "projector"): draws scenes through H_cam_to_page, measurement and pattern modes, speaks
// `say` lines and plays `sfx` sounds (both only after the tap to start), ACKs every drawn scene after
// requestAnimationFrame. A `view` with rotate 180 turns everything drawn, pattern included, about the page centre;
// page CSS px (calib, ACK rects) stay in the unturned frame.
import { drawScene, forgetReveals, homographyMap, MEASURE_FILL, PALETTES } from './scene.js';
import { drawMarkers } from './aruco.js';
import { openSocket } from './link.js';
import * as sfx from './sfx.js';
import { Speaker } from './tts.js';
import { syncMap } from './tablemap.js';
import { syncSolid } from './solid.js';

const PROJ_W = 854; // DLP native resolution (rig brief §1); line and text sizes are set in projector px
const PROJ_H = 480;
const LINE_PROJ_PX = 5;
const TEXT_PROJ_PX = 28;
const LINK_LOST_MS = 2000;

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d', { alpha: false });
const tap = document.getElementById('tap');
const diag = document.getElementById('diag');

let dpr = 1;
let W = 0;
let H = 0;
let scene = null;
let calib = null;
let lastMsg = performance.now();
let linkLost = false;
let ackedSeq = 0;
let frameRequested = false;
let animating = false;
let wake = null;
let started = false;
let rotate = 0; // C2 view: 0 or 180 degrees
let smooth = false; // C2 view.smooth: ease moves between scenes, fade what comes and goes (hand games)

// ---- smoothing (view.smooth). Scenes arrive up to 30 times a second over Wi-Fi with uneven gaps; drawn as they come,
// moving things jump and a cursor blinks whenever the hand tracker misses a frame (3 Nov: "make the projector
// smoother and less flickery"). With smoothing on, a prim that moved glides from where it was drawn to its new place
// over TWEEN_MS, a new one fades in over FADE_IN_MS and one that left fades out over FADE_OUT_MS. Off while the
// table is scanned: the server turns it off then (a fading outline would look like an object to the camera).
const TWEEN_MS = 45;
const FADE_IN_MS = 90;
const FADE_OUT_MS = 180;
let shown = new Map(); // key -> prim as drawn last frame (eased positions)
let from = new Map(); // key -> prim where the current scene's ease starts
let born = new Map(); // key -> ms the prim first appeared
let ghosts = new Map(); // key -> {prim, t}: prims that left, fading out
let easeT0 = 0;

function keyOf(p) {
  return `${p.kind} ${p.id}`;
}

function forgetSmooth() {
  shown = new Map();
  from = new Map();
  born = new Map();
  ghosts = new Map();
}

function sceneArrived(msg, now) {
  if (msg.mode !== 'normal' || !scene || scene.mode !== 'normal') {
    forgetSmooth();
    return;
  }
  from = new Map(shown); // ease from what is on the table right now, not from the last scene: no jumps mid-ease
  const next = new Set((msg.prims || []).map(keyOf));
  for (const [k, p] of shown) if (!next.has(k)) ghosts.set(k, { prim: p, t: now });
  easeT0 = now;
}

const lerp = (a, b, f) => a + (b - a) * f;
const lerp2 = (a, b, f) => [lerp(a[0], b[0], f), lerp(a[1], b[1], f)];

function eased(a, b, f) {
  if (!a || a.kind !== b.kind) return b;
  if (b.kind === 'poly' && a.pts && a.pts.length === b.pts.length) return { ...b, pts: b.pts.map((pt, i) => lerp2(a.pts[i], pt, f)) };
  if (b.kind === 'ring' && a.c) return { ...b, c: lerp2(a.c, b.c, f), r: lerp(a.r, b.r, f) };
  if (b.kind === 'dot' && a.c) return { ...b, c: lerp2(a.c, b.c, f) };
  if (b.kind === 'label' && a.at) return { ...b, at: lerp2(a.at, b.at, f) };
  if (b.kind === 'arrow' && a.a) return { ...b, a: lerp2(a.a, b.a, f), b: lerp2(a.b, b.b, f) };
  return b;
}

// The scene to draw now, eased and faded, and whether it is still changing.
function smoothScene(now) {
  const f = Math.min(1, (now - easeT0) / TWEEN_MS);
  const e = 1 - (1 - f) * (1 - f); // ease-out
  let moving = f < 1;
  const prims = [];
  const drawn = new Map();
  for (const p of scene.prims || []) {
    const k = keyOf(p);
    ghosts.delete(k);
    if (!born.has(k)) born.set(k, now);
    const q = eased(from.get(k), p, e);
    drawn.set(k, q);
    const a = Math.min(1, (now - born.get(k)) / FADE_IN_MS);
    if (a < 1) moving = true;
    prims.push(a < 1 ? { ...q, alpha: a } : q);
  }
  for (const [k, g] of ghosts) {
    const a = 1 - (now - g.t) / FADE_OUT_MS;
    if (a <= 0) {
      ghosts.delete(k);
      born.delete(k);
      continue;
    }
    moving = true;
    prims.unshift({ ...g.prim, alpha: a }); // under what is still there
  }
  shown = drawn;
  return { scene: { ...scene, prims }, moving };
}

const speaker = new Speaker((m) => sock.send(m));

const sock = openSocket('/ws/render', {
  onopen: () => {
    ackedSeq = 0; // wire seqs restart per connection
    rotate = 0; // a view belongs to its server: a new one sends its own before anything it turns
    document.documentElement.classList.remove('rot180');
    sendHello();
  },
  onmessage: (data) => {
    lastMsg = performance.now();
    if (linkLost) {
      linkLost = false;
      requestDraw();
    }
    const msg = JSON.parse(data);
    if (msg.type === 'scene') {
      if (smooth) sceneArrived(msg, performance.now());
      scene = msg;
      requestDraw();
    } else if (msg.type === 'calib') {
      calib = { map: homographyMap(msg.H_cam_to_page), size: msg.page_size };
      requestDraw();
    } else if (msg.type === 'say') {
      speaker.say(msg);
    } else if (msg.type === 'sfx') {
      sfx.play(msg.name, msg.gain); // silent until the tap to start unlocks audio
    } else if (msg.type === 'view') {
      smooth = !!msg.smooth;
      if (!smooth) forgetSmooth();
      rotate = msg.rotate === 180 ? 180 : 0;
      document.documentElement.classList.toggle('rot180', rotate === 180);
      requestDraw();
    }
  },
});

// Device-px transform for drawing at `scale` device px per unit, turned by the current view.
function setView(scale) {
  if (rotate === 180) ctx.setTransform(-scale, 0, 0, -scale, canvas.width, canvas.height);
  else ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

function sendHello() {
  sock.send({
    type: 'hello',
    role: 'projector',
    innerWidth: Math.round(innerWidth),
    innerHeight: Math.round(innerHeight),
    dpr: devicePixelRatio,
    screen: { width: screen.width, height: screen.height, availWidth: screen.availWidth, availHeight: screen.availHeight },
    orientation: (screen.orientation && screen.orientation.type) || '',
    fullscreen: !!document.fullscreenElement,
    secure: isSecureContext,
    wakelock: !!wake && !wake.released,
    ua: navigator.userAgent,
  });
}

function showDiag() {
  diag.textContent = [
    `secure       ${isSecureContext}`,
    `mediaDevices ${!!navigator.mediaDevices}`,
    `wakeLock     ${'wakeLock' in navigator}`,
    `AudioWorklet ${!!window.AudioWorklet}`,
    `speech       ${'speechSynthesis' in window}`,
    `page         ${innerWidth}x${innerHeight} @${devicePixelRatio}`,
    `screen       ${screen.width}x${screen.height}`,
    `link         ${sock.connected ? 'up' : 'down'}`,
  ].join('\n');
}

function resize() {
  dpr = devicePixelRatio || 1;
  W = innerWidth;
  H = innerHeight;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  sendHello();
  showDiag();
  requestDraw();
}

function requestDraw() {
  if (!frameRequested) {
    frameRequested = true;
    requestAnimationFrame(draw);
  }
}

function draw(ts) {
  frameRequested = false;
  // No scene on the table (link lost, measurement or pattern mode, no mapping): when it is back, it reveals itself
  // again (draw_ms outlines, type_ms labels).
  if (linkLost || !scene || scene.mode !== 'normal' || !calib) forgetReveals();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const live = !linkLost && scene && scene.mode === 'normal' && calib;
  syncMap(live ? (scene.prims || []).find((q) => q.kind === 'map') || null : null, live ? calib.map : null, W, H, rotate);
  const building = syncSolid(live ? (scene.prims || []).find((q) => q.kind === 'solid') || null : null, live ? calib.map : null, W, H, rotate);
  if (linkLost) {
    drawLinkLost();
    return;
  }
  if (!scene) return;
  let drawn;
  animating = false;
  if (scene.mode === 'pattern') {
    setView(1); // drawMarkers works in device px; turned, whole device px stay whole
    drawn = drawMarkers(ctx, scene.pattern.markers, dpr);
  } else if (scene.mode !== 'normal') {
    ctx.fillStyle = MEASURE_FILL[scene.mode];
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  } else if (calib) {
    setView(dpr);
    const unit = Math.max(W / PROJ_W, H / PROJ_H);
    let toDraw = scene;
    let moving = false;
    if (smooth) ({ scene: toDraw, moving } = smoothScene(performance.now()));
    animating = drawScene(ctx, toDraw, {
      map: calib.map,
      palette: PALETTES.projector,
      lineWidth: LINE_PROJ_PX * unit,
      fontPx: TEXT_PROJ_PX * unit,
      t: ts / 1000,
    }) || moving;
  }
  if (scene.seq !== ackedSeq) {
    const ack = { type: 'ack', seq: scene.seq };
    if (drawn) ack.drawn = drawn;
    ackedSeq = scene.seq;
    setTimeout(() => sock.send(ack), 0);
  }
  if (animating || building) requestDraw();
}

function drawLinkLost() {
  setView(dpr);
  const unit = Math.max(W / PROJ_W, H / PROJ_H);
  ctx.strokeStyle = PALETTES.projector.error;
  ctx.lineWidth = 12 * unit;
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, Math.min(W, H) * 0.3, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.fillStyle = PALETTES.projector.error;
  ctx.font = `bold ${TEXT_PROJ_PX * 1.5 * unit}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('LINK LOST', W / 2, H / 2);
  ctx.textAlign = 'start';
}

// The screen going off for a moment stops requestAnimationFrame, so scenes go unacknowledged, so the server stops sending
// (it keeps at most 2 unacknowledged), so there is never a scene to acknowledge again: only a new connection gets out of that.
let lastRetry = 0;
setInterval(() => {
  const now = performance.now();
  if (!linkLost && now - lastMsg > LINK_LOST_MS) {
    linkLost = true;
    scene = null;
    ackedSeq = 0;
    requestDraw();
    lastRetry = 0;
  }
  if (linkLost && now - lastRetry > 3000) {
    lastRetry = now;
    sock.reconnect();
  }
  if (!tap.classList.contains('hidden')) showDiag();
}, 250);

async function requestWake() {
  if (!('wakeLock' in navigator) || document.visibilityState !== 'visible') return;
  try {
    wake = await navigator.wakeLock.request('screen');
    wake.addEventListener('release', sendHello);
  } catch (e) {
    wake = null;
  }
  sendHello();
}

// Tap to start: the gesture grants fullscreen, the landscape lock, speech and sound (an `adb shell input tap` counts).
tap.addEventListener('pointerup', async () => {
  started = true;
  speaker.activate();
  sfx.unlock(); // inside the gesture, before any await
  tap.classList.add('hidden');
  try {
    await document.documentElement.requestFullscreen({ navigationUI: 'hide' });
    if (screen.orientation && screen.orientation.lock) await screen.orientation.lock('landscape').catch(() => {});
  } catch (e) {
    // no fullscreen (desktop testing, or the gesture was not trusted): hello reports fullscreen:false
  }
  await requestWake();
});

document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement && started) tap.classList.remove('hidden');
  sendHello();
});
// A hidden page (an old tab behind the one on the table) draws nothing and ACKs nothing; connected, it would be the newest
// projector page, the one the server sends calibration patterns to (4 Nov: three projector tabs, every pattern unanswered).
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    sock.resume();
    if (started) requestWake();
  } else {
    sock.pause();
  }
});
if (document.visibilityState !== 'visible') sock.pause();
addEventListener('resize', resize);
if (screen.orientation) screen.orientation.addEventListener('change', resize);
resize();
