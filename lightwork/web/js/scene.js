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

// Draws `scene.prims`. opts: {map, palette, lineWidth, fontPx, t (seconds, for pulse), outline}.
// Returns true when a primitive pulses, i.e. the caller should keep animating.
export function drawScene(ctx, scene, opts) {
  const { map, palette, lineWidth, fontPx, t = 0, outline = false } = opts;
  let animated = false;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const p of scene.prims || []) {
    const colour = palette[p.role] || palette.info;
    if (p.kind === 'ring') {
      let w = lineWidth;
      if (p.pulse) {
        animated = true;
        w *= 1 + PULSE_GAIN * (0.5 + 0.5 * Math.sin(2 * Math.PI * PULSE_HZ * t));
      }
      ringPath(ctx, map, p.c, p.r);
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
      ctx.beginPath();
      p.pts.forEach((pt, i) => {
        const [x, y] = map(pt[0], pt[1]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      if (p.closed) ctx.closePath();
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
    } else if (p.kind === 'label') {
      const [x, y] = map(p.at[0], p.at[1]);
      ctx.font = `bold ${fontPx}px system-ui, sans-serif`;
      ctx.textBaseline = 'middle';
      if (outline) {
        ctx.strokeStyle = 'rgba(0,0,0,0.85)';
        ctx.lineWidth = Math.max(3, fontPx / 5);
        ctx.strokeText(p.text, x, y);
      }
      ctx.fillStyle = colour;
      ctx.fillText(p.text, x, y);
    }
  }
  ctx.restore();
  return animated;
}
