// Short synthesized UI sounds for C2 `sfx` (WebAudio oscillators, no files). A page is silent until a user gesture:
// `unlock()` must run inside one (the projector's tap to start); before that `play()` does nothing. Neither throws:
// a sound must never cost a frame.
const LEVEL = 0.3; // master gain at sfx gain 1: the phone speaker is loud
const REPEAT_S = 0.06; // the same sound again this soon would only be louder: skipped

let ac = null;
let out = null;
const lastStart = new Map(); // sound name -> AudioContext time it last started

export function unlock() {
  try {
    if (!ac) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return false;
      ac = new Ctx({ latencyHint: 'interactive' });
      out = ac.createGain();
      out.gain.value = LEVEL;
      out.connect(ac.destination);
    }
    if (ac.state !== 'running') ac.resume().catch(() => {});
    // A silent one-sample source started inside the gesture: some engines unlock only on a played source.
    const src = ac.createBufferSource();
    src.buffer = ac.createBuffer(1, 1, ac.sampleRate);
    src.connect(ac.destination);
    src.start();
    return true;
  } catch (e) {
    return false;
  }
}

export function play(name, gain = 1) {
  try {
    const sound = Object.hasOwn(SOUNDS, name) ? SOUNDS[name] : null;
    const g = Math.min(1, Math.max(0, Number(gain)));
    if (!ac || !sound || !(g > 0)) return;
    if (ac.state === 'suspended') ac.resume().catch(() => {}); // e.g. after the screen was off
    const t = ac.currentTime + 0.01;
    if (t - (lastStart.get(name) ?? -Infinity) < REPEAT_S) return;
    lastStart.set(name, t);
    sound(t, g);
  } catch (e) {
    // no sound rather than a broken page
  }
}

// One note from `t`: a `type` oscillator gliding f0 -> f1 Hz over `dur` s, rising to `peak` in `attack` s, then an
// exponential decay to silence at `dur`. `lowpass` (Hz) softens a square.
function note(t, g, { type = 'sine', f0, f1 = f0, dur, attack = 0.004, peak, lowpass = 0 }) {
  const osc = ac.createOscillator();
  const env = ac.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(f0, t);
  if (f1 !== f0) osc.frequency.exponentialRampToValueAtTime(f1, t + dur);
  env.gain.setValueAtTime(0, t);
  env.gain.linearRampToValueAtTime(peak * g, t + attack);
  env.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  let src = osc;
  if (lowpass) {
    const lp = ac.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = lowpass;
    src = src.connect(lp);
  }
  src.connect(env).connect(out);
  osc.onended = () => env.disconnect();
  osc.start(t);
  osc.stop(t + dur + 0.02);
}

// All under 350 ms; two-note sounds step up a fifth, so they read as UI, never as a tune.
const SOUNDS = {
  // the camera looks again: a quick rising sweep with a soft attack and a faint octave above
  scan: (t, g) => {
    note(t, g, { f0: 600, f1: 1400, dur: 0.28, attack: 0.07, peak: 0.5 });
    note(t, g, { f0: 1200, f1: 2800, dur: 0.24, attack: 0.07, peak: 0.08 });
  },
  // something new on the table: a two-note chime
  found: (t, g) => {
    note(t, g, { type: 'triangle', f0: 880, dur: 0.14, peak: 0.55 });
    note(t + 0.08, g, { type: 'triangle', f0: 1320, dur: 0.24, peak: 0.55 });
  },
  // a projected button was pressed: a short soft tick
  press: (t, g) => {
    note(t, g, { type: 'triangle', f0: 1700, f1: 900, dur: 0.05, attack: 0.002, peak: 0.5 });
  },
  // something left the table: a falling tone
  gone: (t, g) => {
    note(t, g, { type: 'triangle', f0: 700, f1: 350, dur: 0.28, attack: 0.01, peak: 0.5 });
  },
  // two low buzzes, low-passed so they warn without rasping
  error: (t, g) => {
    for (const dt of [0, 0.14]) note(t + dt, g, { type: 'square', f0: 220, dur: 0.1, peak: 0.32, lowpass: 1400 });
  },
  // the page is listening: a soft rising two-tone
  listen: (t, g) => {
    note(t, g, { f0: 587, dur: 0.14, attack: 0.02, peak: 0.4 });
    note(t + 0.1, g, { f0: 880, dur: 0.2, attack: 0.02, peak: 0.4 });
  },
  // an answer is ready: one soft bell, a fundamental and a faster-fading inharmonic partial
  answer: (t, g) => {
    note(t, g, { f0: 990, dur: 0.27, attack: 0.003, peak: 0.45 });
    note(t, g, { f0: 990 * 2.76, dur: 0.1, attack: 0.003, peak: 0.06 });
  },
};
