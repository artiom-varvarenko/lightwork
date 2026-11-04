// Spoken output via speechSynthesis (D9, webvoice brief §2). Reports `tts` {id, state: start|end|error} through
// `report`; the server mutes the mic between start and end/error.
const MAX_CHUNK = 200; // chars; Chrome desktop stops long utterances after ~14 s
const CHARS_PER_S = 10;

function chunks(text) {
  const sentences = text.match(/[^.!?]+[.!?]*\s*/g) || [text];
  const out = [];
  let cur = '';
  for (const s of sentences) {
    if (cur && (cur + s).length > MAX_CHUNK) {
      out.push(cur.trim());
      cur = '';
    }
    cur += s;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

export class Speaker {
  constructor(report) {
    this.report = report;
    this.voice = null;
    this.active = false;
    if ('speechSynthesis' in window) {
      this.pickVoice();
      speechSynthesis.addEventListener('voiceschanged', () => this.pickVoice());
    }
  }

  get available() {
    return 'speechSynthesis' in window;
  }

  pickVoice() {
    const en = speechSynthesis.getVoices().filter((v) => v.lang.replace('_', '-').toLowerCase().startsWith('en'));
    this.voice = en.find((v) => v.localService) || en[0] || null;
  }

  // Must run inside a user gesture: unlocks speak() for the document's lifetime.
  activate() {
    if (!this.available) return false;
    const u = new SpeechSynthesisUtterance('');
    u.volume = 0;
    speechSynthesis.speak(u);
    this.active = true;
    return true;
  }

  say(msg) {
    if (!this.available) {
      this.report({ type: 'tts', id: msg.id, state: 'error' });
      return;
    }
    if (msg.priority === 'interrupt') speechSynthesis.cancel();
    const parts = chunks(msg.text);
    if (!parts.length) { // nothing to say: it only stopped what was being said
      this.report({ type: 'tts', id: msg.id, state: 'end' });
      return;
    }
    let started = false;
    let finished = false;
    let watchdog = null;
    const finish = (state) => {
      if (finished) return;
      finished = true;
      clearTimeout(watchdog);
      this.report({ type: 'tts', id: msg.id, state });
    };
    // If speech never starts (no activation, engine stuck) report an error; once started, end it after the
    // expected duration even when onend never fires, so the mic is never left muted.
    watchdog = setTimeout(() => finish('error'), 10000 + (msg.text.length / CHARS_PER_S) * 1000);
    parts.forEach((text, i) => {
      const u = new SpeechSynthesisUtterance(text);
      if (this.voice) {
        u.voice = this.voice;
        u.lang = this.voice.lang;
      }
      u.onstart = () => {
        if (!started) {
          started = true;
          this.report({ type: 'tts', id: msg.id, state: 'start' });
        }
        clearTimeout(watchdog);
        watchdog = setTimeout(() => finish('end'), (text.length / CHARS_PER_S + 2) * 1000);
      };
      u.onerror = () => finish('error');
      if (i === parts.length - 1) u.onend = () => finish('end');
      speechSynthesis.speak(u);
    });
  }
}
