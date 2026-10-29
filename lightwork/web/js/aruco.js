// Calibration pattern (D5, rig brief §4.1): ArUco DICT_4X4_50 ids 0-14 drawn module by module.
import { MEASURE_FILL } from './scene.js';

// "White" modules use the measurement grey: the locked exposure fits it, while full white can clip and bias corners.
const LIGHT = MEASURE_FILL.grey;
// CODES[id] = inner 4x4 bits, row-major, MSB = top-left module, 1 = white; generated with OpenCV 5.0
// cv2.aruco.generateImageMarker(getPredefinedDictionary(DICT_4X4_50), id, 6).
const CODES = [
  0xb532, 0x0f9a, 0x332d, 0x9946, 0x549e, 0x79cd, 0x9e2e, 0xc4f2,
  0xfeda, 0xcf56, 0xf991, 0x11a7, 0x0eb7, 0x2a0f, 0x24b1,
];

// Draws `markers` [{id, x, y, size}] (CSS px; x, y = top-left of the outer black border, size = its side) on a
// context with the identity transform (device px). Corners and module size snap to whole device px; each marker
// gets a 1-module white quiet zone. Returns the rects as drawn, in CSS px, for the `ack`.
export function drawMarkers(ctx, markers, dpr) {
  const drawn = [];
  for (const m of markers) {
    const code = CODES[m.id];
    if (code === undefined) continue;
    const mod = Math.max(1, Math.round((m.size * dpr) / 6));
    const x = Math.round(m.x * dpr);
    const y = Math.round(m.y * dpr);
    ctx.fillStyle = LIGHT;
    ctx.fillRect(x - mod, y - mod, 8 * mod, 8 * mod);
    ctx.fillStyle = '#000000';
    ctx.fillRect(x, y, 6 * mod, 6 * mod);
    ctx.fillStyle = LIGHT;
    for (let i = 0; i < 16; i++) {
      if ((code >> (15 - i)) & 1) ctx.fillRect(x + ((i % 4) + 1) * mod, y + (Math.floor(i / 4) + 1) * mod, mod, mod);
    }
    drawn.push({ id: m.id, x: x / dpr, y: y / dpr, size: (6 * mod) / dpr });
  }
  return drawn;
}
