// Projector page (C2 role "projector"): draws scenes through H_cam_to_page, measurement and pattern modes, speaks
// `say` lines, ACKs every drawn scene after requestAnimationFrame.
import { drawScene, homographyMap, MEASURE_FILL, PALETTES } from './scene.js';
import { drawMarkers } from './aruco.js';
import { openSocket } from './link.js';
import { Speaker } from './tts.js';

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

const speaker = new Speaker((m) => sock.send(m));

const sock = openSocket('/ws/render', {
  onopen: () => {
    ackedSeq = 0; // wire seqs restart per connection
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
      scene = msg;
      requestDraw();
    } else if (msg.type === 'calib') {
      calib = { map: homographyMap(msg.H_cam_to_page), size: msg.page_size };
      requestDraw();
    } else if (msg.type === 'say') {
      speaker.say(msg);
    }
  },
});

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
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (linkLost) {
    drawLinkLost();
    return;
  }
  if (!scene) return;
  let drawn;
  animating = false;
  if (scene.mode === 'pattern') {
    drawn = drawMarkers(ctx, scene.pattern.markers, dpr);
  } else if (scene.mode !== 'normal') {
    ctx.fillStyle = MEASURE_FILL[scene.mode];
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  } else if (calib) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const unit = Math.max(W / PROJ_W, H / PROJ_H);
    animating = drawScene(ctx, scene, {
      map: calib.map,
      palette: PALETTES.projector,
      lineWidth: LINE_PROJ_PX * unit,
      fontPx: TEXT_PROJ_PX * unit,
      t: ts / 1000,
    });
  }
  if (scene.seq !== ackedSeq) {
    const ack = { type: 'ack', seq: scene.seq };
    if (drawn) ack.drawn = drawn;
    ackedSeq = scene.seq;
    setTimeout(() => sock.send(ack), 0);
  }
  if (animating) requestDraw();
}

function drawLinkLost() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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

setInterval(() => {
  if (!linkLost && performance.now() - lastMsg > LINK_LOST_MS) {
    linkLost = true;
    scene = null;
    ackedSeq = 0;
    requestDraw();
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

// Tap to start: the gesture grants fullscreen, the landscape lock and speech (an `adb shell input tap` counts).
tap.addEventListener('pointerup', async () => {
  started = true;
  speaker.activate();
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
document.addEventListener('visibilitychange', () => {
  if (started && document.visibilityState === 'visible') requestWake();
});
addEventListener('resize', resize);
if (screen.orientation) screen.orientation.addEventListener('change', resize);
resize();
