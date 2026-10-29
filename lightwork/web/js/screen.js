// Screen page (C2 role "screen"): camera preview with the scene overlay on one canvas, step list with evidence
// chips, model trace, voice HUD, Start button, layout click tools. State comes from /ws/control (keys: hub.py).
import { drawScene, fitRect, rectMap, PALETTES } from './scene.js';
import { openSocket } from './link.js';
import { Speaker } from './tts.js';

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const ctx = canvas.getContext('2d', { alpha: false });
const stage = $('stage');

const DEFAULT_CAMERA = { w: 1920, h: 1080 };
const TRACE_ROWS = 50;
const MODE_BADGE = { pattern: 'CALIBRATION PATTERN', grey: 'MEASURING (grey)', dark: 'MEASURING (dark)', white: 'MEASURING (white)' };

let state = {};
let scene = null;
let ackedSeq = 0;
let bitmap = null;
let bitmapId = -1;
let frameToAck = null;
let frameRequested = false;
let map = null;
let tool = null; // 'corners' | 'rect' | 'area'
let points = [];
let drag = null;
const stats = { framesDrawn: 0, scenesDrawn: 0, fps: 0, lastFrameId: null };
window.lwStats = stats;

// ------------------------------------------------------------------ sockets

const speaker = new Speaker((m) => render.send(m));

const render = openSocket('/ws/render', {
  onopen: () => {
    ackedSeq = 0; // wire seqs restart per connection
    sendHello();
    showLink();
  },
  onclose: showLink,
  onmessage: (data) => {
    const msg = JSON.parse(data);
    if (msg.type === 'scene') {
      scene = msg;
      showBanner();
      requestDraw();
    } else if (msg.type === 'say') {
      speaker.say(msg);
    }
  },
});

const camera = openSocket('/ws/camera', {
  binary: true,
  onmessage: async (buf) => {
    const dv = new DataView(buf);
    const id = dv.getUint32(0, true);
    const bmp = await createImageBitmap(new Blob([new Uint8Array(buf, 12)], { type: 'image/jpeg' }));
    if (id <= bitmapId) {
      bmp.close();
      return;
    }
    if (bitmap) bitmap.close();
    bitmap = bmp;
    bitmapId = id;
    frameToAck = id;
    requestDraw();
  },
  onopen: () => {
    bitmapId = -1; // a new server may restart frame ids
  },
});

const control = openSocket('/ws/control', {
  onmessage: (data) => {
    const msg = JSON.parse(data);
    if (msg.type === 'state') applyState(msg.state);
  },
});

function sendHello() {
  render.send({
    type: 'hello',
    role: 'screen',
    innerWidth: Math.round(innerWidth),
    innerHeight: Math.round(innerHeight),
    dpr: devicePixelRatio,
    screen: { width: screen.width, height: screen.height, availWidth: screen.availWidth, availHeight: screen.availHeight },
    orientation: (screen.orientation && screen.orientation.type) || '',
    fullscreen: !!document.fullscreenElement,
    secure: isSecureContext,
    wakelock: false,
    ua: navigator.userAgent,
  });
}

function showLink() {
  $('link').textContent = render.connected ? 'linked' : 'offline';
}

function sendUi(action, args = {}) {
  control.send({ type: 'ui', action, args });
}

// ------------------------------------------------------------------ canvas

function requestDraw() {
  if (!frameRequested) {
    frameRequested = true;
    requestAnimationFrame(draw);
  }
}

function draw(ts) {
  frameRequested = false;
  const dpr = devicePixelRatio || 1;
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0b0d10';
  ctx.fillRect(0, 0, w, h);

  const cam = state.camera || DEFAULT_CAMERA;
  const src = state.preview_xywh || [0, 0, cam.w, cam.h];
  const content = fitRect(src, w, h);
  map = rectMap(src, content);

  if (bitmap) {
    ctx.drawImage(bitmap, content[0], content[1], content[2], content[3]);
    if (stats.lastFrameId !== bitmapId) {
      stats.framesDrawn++;
      stats.lastFrameId = bitmapId;
    }
  } else {
    ctx.strokeStyle = '#2a313b';
    ctx.strokeRect(content[0] + 0.5, content[1] + 0.5, content[2] - 1, content[3] - 1);
    ctx.fillStyle = '#8a94a3';
    ctx.font = '16px system-ui, sans-serif';
    ctx.fillText('no camera preview', content[0] + 16, content[1] + 28);
  }
  if (frameToAck !== null) {
    const id = frameToAck;
    frameToAck = null;
    setTimeout(() => camera.send({ ack: id }), 0);
  }

  drawLayout();
  let animating = false;
  if (scene) {
    if (scene.mode === 'normal') {
      animating = drawScene(ctx, scene, { map, palette: PALETTES.screen, lineWidth: 3, fontPx: 16, t: ts / 1000, outline: true });
    } else {
      badge(MODE_BADGE[scene.mode], content);
    }
    if (scene.seq !== ackedSeq) {
      ackedSeq = scene.seq;
      stats.scenesDrawn++;
      const ack = { type: 'ack', seq: scene.seq };
      setTimeout(() => render.send(ack), 0);
    }
  }
  if (animating) requestDraw();
}

function badge(text, content) {
  ctx.font = 'bold 15px system-ui, sans-serif';
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = 'rgba(0,0,0,0.7)';
  ctx.fillRect(content[0] + 8, content[1] + 8, tw + 16, 26);
  ctx.fillStyle = '#ffd60a';
  ctx.fillText(text, content[0] + 16, content[1] + 26);
}

function drawLayout() {
  const objects = (state.layout && state.layout.objects) || [];
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = PALETTES.screen.info;
  ctx.fillStyle = PALETTES.screen.info;
  ctx.font = '12px system-ui, sans-serif';
  for (const o of objects) {
    const pts = o.corners_cam_px || o.polygon_cam_px;
    if (!pts || !pts.length) continue;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const [x, y] = map(p[0], p[1]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
    const [lx, ly] = map(pts[0][0], pts[0][1]);
    ctx.fillText(o.id, lx + 3, ly - 4);
  }
  const wa = state.layout && state.layout.work_area_xywh;
  if (wa) rectOutline(wa, PALETTES.screen.warn);
  ctx.setLineDash([]);
  ctx.fillStyle = PALETTES.screen.target;
  points.forEach((p, i) => {
    const [x, y] = map(p[0], p[1]);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillText(['TL', 'TR', 'BR', 'BL'][i], x + 6, y - 6);
  });
  if (drag) rectOutline(dragRect(), PALETTES.screen.target);
  ctx.restore();
}

function rectOutline(xywh, colour) {
  const [x0, y0] = map(xywh[0], xywh[1]);
  const [x1, y1] = map(xywh[0] + xywh[2], xywh[1] + xywh[3]);
  ctx.strokeStyle = colour;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
}

// ------------------------------------------------------------------ layout tools

const q1 = (v) => Math.round(v * 10) / 10;

function camPoint(e) {
  const [x, y] = map.inverse(e.offsetX, e.offsetY);
  return [q1(x), q1(y)];
}

function dragRect() {
  const [ax, ay] = drag.a;
  const [bx, by] = drag.b;
  return [Math.round(Math.min(ax, bx)), Math.round(Math.min(ay, by)), Math.round(Math.abs(bx - ax)), Math.round(Math.abs(by - ay))];
}

function setTool(t) {
  tool = tool === t ? null : t;
  points = [];
  drag = null;
  canvas.classList.toggle('picking', tool !== null);
  for (const [id, name] of [['t-corners', 'corners'], ['t-rect', 'rect'], ['t-area', 'area']]) {
    $(id).classList.toggle('on', tool === name);
  }
  $('hint').textContent = { corners: 'click TL, TR, BR, BL', rect: 'drag a rectangle', area: 'drag the work area' }[tool] || '';
  requestDraw();
}

$('t-corners').onclick = () => setTool('corners');
$('t-rect').onclick = () => setTool('rect');
$('t-area').onclick = () => setTool('area');
$('t-accept').onclick = () => sendUi('layout_accept');

canvas.addEventListener('pointerdown', (e) => {
  if (!tool || !map) return;
  if (tool === 'corners') {
    points.push(camPoint(e));
    if (points.length === 4) {
      sendUi('layout_corners', { object: $('obj').value, corners_cam_px: points });
      setTool(null);
    }
  } else {
    drag = { a: camPoint(e), b: camPoint(e) };
    canvas.setPointerCapture(e.pointerId);
  }
  requestDraw();
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  drag.b = camPoint(e);
  requestDraw();
});
canvas.addEventListener('pointerup', () => {
  if (!drag) return;
  const xywh = dragRect();
  if (xywh[2] > 0 && xywh[3] > 0) {
    if (tool === 'area') sendUi('layout_work_area', { xywh });
    else sendUi('layout_rect', { object: $('obj').value, xywh });
  }
  setTool(null);
});

// ------------------------------------------------------------------ side panel

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function showBanner() {
  const banner = (scene && scene.banner) || state.banner || '';
  $('banner').textContent = banner;
  $('banner').classList.toggle('on', !!banner);
}

function applyState(s) {
  state = s;
  const run = s.run || {};
  showBanner();
  $('runinfo').textContent = [run.pack, run.mode, run.backend, run.provenance === 'dev-standin' ? 'DEV STAND-IN' : null]
    .filter(Boolean)
    .join(' · ');
  $('cost').textContent = typeof s.cost_usd === 'number' ? `$${s.cost_usd.toFixed(3)}` : '';

  const hud = s.hud || {};
  setHud('hud-voice', hud.voice || 'off', hud.voice);
  setHud('hud-turn', hud.turn || '-', hud.turn);
  setHud('hud-heard', hud.heard || '-', null);

  $('say').textContent = s.last_say ? `“${s.last_say}”` : '';
  const chips = s.chips || [];
  const steps = $('steps');
  steps.replaceChildren(
    ...(s.steps || []).map((st) => {
      const li = el('li', st.state + (st.id === s.active ? ' active' : ''));
      li.append(el('span', 'id', st.id), el('span', 'st', st.state), document.createTextNode(st.say || ''));
      if (st.reason && (st.state === 'failed' || st.state === 'recovery' || st.state === 'unverified')) {
        li.append(el('div', 'reason', st.reason));
      }
      const mine = chips.filter((c) => c.step === st.id);
      if (mine.length) {
        const box = el('div', 'chips');
        for (const c of mine) {
          const chip = el('span', `chip ${c.status}`, `${c.check}: ${c.status}`);
          chip.title = c.reason || '';
          box.append(chip);
        }
        li.append(box);
      }
      return li;
    }),
  );

  $('trace').replaceChildren(
    ...(s.trace || []).slice(-TRACE_ROWS).reverse().map((r) => {
      const tr = el('tr');
      tr.append(
        el('td', '', r.role),
        el('td', 'model', r.model),
        el('td', r.status, r.status),
        el('td', 'num', r.latency_ms == null ? '' : String(r.latency_ms)),
        el('td', 'num', r.cost_usd == null ? '' : r.cost_usd.toFixed(4)),
      );
      tr.title = r.intent_id;
      return tr;
    }),
  );

  const objects = (s.layout && s.layout.objects) || [];
  $('tools').classList.toggle('on', !!s.layout);
  const sel = $('obj');
  const keep = sel.value;
  sel.replaceChildren(...objects.map((o) => el('option', '', o.id)));
  if (objects.some((o) => o.id === keep)) sel.value = keep;
  requestDraw();
}

function setHud(id, text, cls) {
  const e = $(id);
  e.textContent = text;
  e.className = cls || '';
}

// ------------------------------------------------------------------ controls

document.querySelectorAll('button[data-action]').forEach((b) => {
  b.onclick = () => sendUi(b.dataset.action);
});

const KEYS = { ' ': 'confirm', r: 'repeat', w: 'why', b: 'back' };
addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLSelectElement || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
  const action = KEYS[e.key];
  if (action) {
    e.preventDefault();
    sendUi(action);
  } else if (e.key === 'Escape') {
    setTool(null);
  }
});

$('start').onclick = async () => {
  const tts = speaker.activate();
  let mic = false;
  try {
    const { startMic } = await import('/static/js/mic.js');
    await startMic({ onState: (s) => setHud('hud-mic', s.label ? `${s.state} (${s.label})` : s.state, s.state) });
    mic = true;
  } catch (e) {
    setHud('hud-mic', `error: ${e.message || e}`, 'error');
  }
  $('start').classList.add('done');
  $('start').textContent = tts ? 'Started' : 'Started (no speech in this browser)';
  sendUi('start', { tts, mic });
};

let lastCount = 0;
setInterval(() => {
  stats.fps = stats.framesDrawn - lastCount;
  lastCount = stats.framesDrawn;
  $('stats').textContent = `preview ${stats.fps} fps · frame ${stats.lastFrameId ?? '-'} · scenes ${stats.scenesDrawn}`;
}, 1000);

new ResizeObserver(() => requestDraw()).observe(stage);
addEventListener('resize', sendHello);
