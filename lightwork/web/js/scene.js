// Shared C2 scene renderer (projector, screen, viewer). Primitives are in camera px; `map(x, y)` turns camera px
// into the canvas' drawing units (CSS px when the context is scaled by devicePixelRatio).

export const PALETTES = {
  // 100 lm DLP on pure black: saturated primaries plus white, never dark or desaturated tones.
  projector: { target: '#00ff00', source: '#00ffff', error: '#ff0000', warn: '#ffff00', done: '#0000ff', info: '#ffffff' },
  // Over the camera image: bright tones with a dark outline for contrast on any background.
  screen: { target: '#39ff14', source: '#00e5ff', error: '#ff3b3b', warn: '#ffd60a', done: '#6b93ff', info: '#ffffff' },
};

export const MEASURE_FILL = { dark: '#000000', grey: '#a6a6a6', white: '#ffffff' }; // grey = 65 % (D7)

export function homographyMap(H) {
  return (x, y) => {
    const w = H[2][0] * x + H[2][1] * y + H[2][2];
    return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
  };
}

// Maps the camera-px rect `src` [x, y, w, h] onto the canvas rect `dst` [x, y, w, h].
export function rectMap(src, dst) {
  const sx = dst[2] / src[2], sy = dst[3] / src[3];
  const map = (x, y) => [dst[0] + (x - src[0]) * sx, dst[1] + (y - src[1]) * sy];
  map.inverse = (u, v) => [src[0] + (u - dst[0]) / sx, src[1] + (v - dst[1]) / sy];
  return map;
}

// Largest rect with the aspect of `src` centred inside a w x h area.
export function fitRect(src, w, h) {
  const s = Math.min(w / src[2], h / src[3]);
  const cw = src[2] * s, ch = src[3] * s;
  return [(w - cw) / 2, (h - ch) / 2, cw, ch];
}

// Mirrored in lightwork/server/projector_style.py (B's overlay mask predicts these pixels).
const RING_SEGMENTS = 64;
const PULSE_HZ = 2;
const PULSE_GAIN = 0.6;
const DOT_RADIUS_LINES = 1.5;
const ARROW_HEAD_LINES = 4;
const ARROW_HEAD_RAD = 0.5;
const GLOW_WIDTH_LINES = 3.2;
const GLOW_ALPHA = 0.28;

// Reveals of `draw_ms` outlines and `type_ms` labels: kind + id -> {key, t0, ms}. Only the ids of the scene drawn
// last are kept, so an outline that leaves and comes back (the server blanks them for a moment after a scan) draws
// itself again. A new `key` restarts a reveal: a poly's draw_ms, a label's text.
let reveals = new Map();

// The page showed no scene (link lost, measurement or pattern mode): what comes back reveals itself again.
export function forgetReveals() {
  reveals = new Map();
}

// Fraction 0..1 of prim `id`'s reveal, `ms` long from when it first showed with `key`; 1 without a clock. A running
// reveal keeps the length it started with.
function revealed(seen, id, key, ms, t) {
  let r = reveals.get(id);
  if (!r || r.key !== key || !(t >= r.t0)) r = { key, t0: t, ms }; // also when the clock went back: start again
  seen.set(id, r);
  if (t === undefined || !(r.ms > 0)) return 1;
  return Math.min(1, ((t - r.t0) * 1000) / r.ms);
}

// The first `frac` of the outline through mapped points `pts`, by length from the first point (closing edge last).
function outlinePath(ctx, pts, closed, frac) {
  ctx.beginPath();
  if (!pts.length) return;
  ctx.moveTo(pts[0][0], pts[0][1]);
  if (frac >= 1) {
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    if (closed) ctx.closePath();
    return;
  }
  const route = closed ? [...pts, pts[0]] : pts;
  const lens = [];
  let total = 0;
  for (let i = 1; i < route.length; i++) {
    lens.push(Math.hypot(route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1]));
    total += lens[i - 1];
  }
  let left = frac * total;
  for (let i = 1; i < route.length && left > 0; i++) {
    const [ax, ay] = route[i - 1];
    const [bx, by] = route[i];
    const k = Math.min(1, left / lens[i - 1]);
    ctx.lineTo(ax + (bx - ax) * k, ay + (by - ay) * k);
    left -= lens[i - 1];
  }
}

// A wide faint stroke of the current path under the line: saturated colour on black reads as a soft halo through a
// 100 lm DLP.
function glow(ctx, colour, width) {
  const alpha = ctx.globalAlpha;
  ctx.globalAlpha = alpha * GLOW_ALPHA;
  ctx.strokeStyle = colour;
  ctx.lineWidth = width * GLOW_WIDTH_LINES;
  ctx.stroke();
  ctx.globalAlpha = alpha;
}

function stroke(ctx, outline, colour, width) {
  if (outline) {
    ctx.strokeStyle = 'rgba(0,0,0,0.75)';
    ctx.lineWidth = width + 3;
    ctx.stroke();
  }
  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  ctx.stroke();
}

function ringPath(ctx, map, c, r) {
  ctx.beginPath();
  for (let i = 0; i <= RING_SEGMENTS; i++) {
    const a = (i / RING_SEGMENTS) * 2 * Math.PI;
    const [x, y] = map(c[0] + r * Math.cos(a), c[1] + r * Math.sin(a));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

// Pictures come by URL, never in a scene (C2 `picture`): each is loaded once and stays for the page's lifetime.
const pictures = new Map(); // src -> HTMLImageElement

function pictureOf(src) {
  let img = pictures.get(src);
  if (!img) {
    img = new Image();
    img.src = src;
    pictures.set(src, img);
  }
  return img;
}

// Wireframes come by URL too (C2 `wire`): polylines of [x, y, z] become one flat list of segments, loaded once.
const wires = new Map(); // src -> {segs: Float32Array | null, failed}
const WIRE_PERSPECTIVE = 3.5;
const WIRE_DEPTHS = ['#0030ff', '#0088ff', '#00ffff']; // far to near, on black: the far side dims away
const WIRE_MIN_WIDTH = 1.3;

function wireOf(src) {
  let w = wires.get(src);
  if (!w) {
    w = { segs: null, failed: false };
    wires.set(src, w);
    fetch(src).then((r) => r.json()).then((lines) => {
      const flat = [];
      for (const line of lines) {
        for (let i = 1; i < line.length; i++) flat.push(...line[i - 1], ...line[i]);
      }
      w.segs = new Float32Array(flat);
    }).catch(() => { w.failed = true; });
  }
  return w;
}

// Each segment is drawn in the colour of how far back it lies. Paths are stroked far to near: what is near stays on top.
function drawWire(ctx, segs, p, map, lineWidth, frac) {
  const [ox, oy] = map(p.origin[0], p.origin[1]);
  const [ex, ey] = map(p.ex[0], p.ex[1]);
  const [ux, uy] = map(p.ey[0], p.ey[1]);
  const ax = ex - ox, ay = ey - oy, bx = ux - ox, by = uy - oy;
  const cy = Math.cos(p.yaw), sy = Math.sin(p.yaw), ct = Math.cos(p.tilt), st = Math.sin(p.tilt);
  const total = segs.length / 6;
  const n = Math.floor(frac * total);
  const paths = WIRE_DEPTHS.map(() => new Path2D());
  for (let i = 0; i < n; i++) {
    const k = i * 6;
    const x0 = segs[k] * cy - segs[k + 1] * sy, y0 = segs[k] * sy + segs[k + 1] * cy, z0 = segs[k + 2];
    const x1 = segs[k + 3] * cy - segs[k + 4] * sy, y1 = segs[k + 3] * sy + segs[k + 4] * cy, z1 = segs[k + 5];
    const d0 = y0 * ct - z0 * st, d1 = y1 * ct - z1 * st;
    const f0 = WIRE_PERSPECTIVE / (WIRE_PERSPECTIVE + d0), f1 = WIRE_PERSPECTIVE / (WIRE_PERSPECTIVE + d1);
    const v0 = y0 * st + z0 * ct, v1 = y1 * st + z1 * ct;
    const d = (d0 + d1) / 2;
    const path = paths[d > 0.3 ? 0 : d > -0.2 ? 1 : 2];
    path.moveTo(ox + f0 * (x0 * ax + v0 * bx), oy + f0 * (x0 * ay + v0 * by));
    path.lineTo(ox + f1 * (x1 * ax + v1 * bx), oy + f1 * (x1 * ay + v1 * by));
  }
  ctx.lineWidth = Math.max(WIRE_MIN_WIDTH, lineWidth * Math.min(1, Math.sqrt(300 / Math.max(total, 1))));
  paths.forEach((path, b) => {
    ctx.strokeStyle = WIRE_DEPTHS[b];
    ctx.stroke(path);
  });
}

// Draws `scene.prims`. opts: {map, palette, lineWidth, fontPx, t (seconds: pulse and reveal clock), outline}.
// Returns true while something moves (a pulsing ring, an outline drawing itself, a label typing): the caller should
// keep requesting frames.
export function drawScene(ctx, scene, opts) {
  const { map, palette, lineWidth, fontPx, t = 0, outline = false } = opts;
  const clock = typeof opts.t === 'number' ? opts.t : undefined; // no clock: reveals show at once
  const seen = new Map();
  let animated = false;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const p of scene.prims || []) {
    const colour = palette[p.role] || palette.info;
    ctx.globalAlpha = typeof p.alpha === 'number' ? Math.max(0, Math.min(1, p.alpha)) : 1; // fades (projector.js)
    if (p.kind === 'ring') {
      let w = lineWidth;
      if (p.pulse) {
        animated = true;
        w *= 1 + PULSE_GAIN * (0.5 + 0.5 * Math.sin(2 * Math.PI * PULSE_HZ * t));
      }
      ringPath(ctx, map, p.c, p.r);
      if (p.glow) glow(ctx, colour, w);
      stroke(ctx, outline, colour, w);
    } else if (p.kind === 'arrow') {
      const [ax, ay] = map(p.a[0], p.a[1]);
      const [bx, by] = map(p.b[0], p.b[1]);
      const ang = Math.atan2(by - ay, bx - ax);
      const head = lineWidth * ARROW_HEAD_LINES;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.moveTo(bx - head * Math.cos(ang - ARROW_HEAD_RAD), by - head * Math.sin(ang - ARROW_HEAD_RAD));
      ctx.lineTo(bx, by);
      ctx.lineTo(bx - head * Math.cos(ang + ARROW_HEAD_RAD), by - head * Math.sin(ang + ARROW_HEAD_RAD));
      stroke(ctx, outline, colour, lineWidth);
    } else if (p.kind === 'poly') {
      const f = revealed(seen, `poly ${p.id}`, p.draw_ms || 0, p.draw_ms, clock);
      if (f < 1) animated = true;
      if (f <= 0) continue;
      outlinePath(ctx, p.pts.map((pt) => map(pt[0], pt[1])), p.closed, 1 - (1 - f) ** 3); // ease-out cubic
      if (p.glow) glow(ctx, colour, lineWidth);
      stroke(ctx, outline, colour, lineWidth);
    } else if (p.kind === 'dot') {
      const [x, y] = map(p.c[0], p.c[1]);
      ctx.beginPath();
      ctx.arc(x, y, lineWidth * DOT_RADIUS_LINES, 0, 2 * Math.PI);
      if (outline) {
        ctx.strokeStyle = 'rgba(0,0,0,0.75)';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      ctx.fillStyle = colour;
      ctx.fill();
    } else if (p.kind === 'wire') {
      const w = wireOf(p.src);
      if (!w.segs) { // still loading: keep drawing until it is here
        animated = !w.failed || animated;
        continue;
      }
      const f = revealed(seen, `wire ${p.id}`, p.src, p.draw_ms, clock);
      if (f < 1) animated = true;
      if (f > 0) drawWire(ctx, w.segs, p, map, lineWidth, f);
    } else if (p.kind === 'picture') {
      const img = pictureOf(p.src);
      if (!img.complete) { // still loading: keep drawing until it is here
        animated = true;
        continue;
      }
      if (!img.naturalWidth) continue; // it failed to load
      const f = revealed(seen, `picture ${p.id}`, p.src, p.draw_ms, clock);
      if (f < 1) animated = true;
      if (f <= 0) continue;
      const [ax, ay] = map(p.corners[0][0], p.corners[0][1]);
      const [bx, by] = map(p.corners[1][0], p.corners[1][1]);
      const [dx, dy] = map(p.corners[3][0], p.corners[3][1]);
      const w = img.naturalWidth, h = img.naturalHeight;
      const rows = Math.max(1, Math.round(h * (1 - (1 - f) ** 3)));
      ctx.save(); // on top of the page's turn, so the picture's own up stays the design's up
      ctx.transform((bx - ax) / w, (by - ay) / w, (dx - ax) / h, (dy - ay) / h, ax, ay);
      ctx.drawImage(img, 0, 0, w, rows, 0, 0, w, rows);
      ctx.restore();
    } else if (p.kind === 'label') {
      const [x, y] = map(p.at[0], p.at[1]);
      ctx.font = `bold ${fontPx}px system-ui, sans-serif`;
      ctx.textBaseline = 'middle';
      let text = p.text;
      let cursor = null; // [x offset, width] of the cell the next character will fill
      const f = revealed(seen, `label ${p.id}`, p.text, p.type_ms, clock);
      if (f < 1) {
        animated = true;
        const chars = Array.from(p.text); // code points: never half an emoji
        const n = Math.floor(f * chars.length);
        text = chars.slice(0, n).join('');
        if (n < chars.length) {
          const before = ctx.measureText(text).width;
          cursor = [before, ctx.measureText(text + chars[n]).width - before];
        }
      }
      if (outline) {
        ctx.strokeStyle = 'rgba(0,0,0,0.85)';
        ctx.lineWidth = Math.max(3, fontPx / 5);
        ctx.strokeText(text, x, y);
      }
      ctx.fillStyle = colour;
      ctx.fillText(text, x, y);
      // A block cursor inside the next character's cell: never outside the finished label (the overlay mask's box).
      if (cursor) ctx.fillRect(x + cursor[0], y - 0.35 * fontPx, Math.max(cursor[1], 0.2 * fontPx), 0.7 * fontPx);
    }
  }
  ctx.restore();
  reveals = seen;
  return animated;
}
