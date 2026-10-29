// AudioWorklet: mono Float32 render quanta -> PCM16 LE at 24 kHz, posted as 80 ms (1920-sample, 3840-byte) buffers.
// When the context runs at another rate (Firefox < 148 fallback) samples are linearly resampled to 24 kHz.
const OUT_RATE = 24000;
const FRAME_SAMPLES = 1920;

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / OUT_RATE; // input samples per output sample
    this.pos = this.step; // position of the next output sample, relative to `prev` (0) .. current input (1)
    this.prev = 0;
    this.buf = new Int16Array(FRAME_SAMPLES);
    this.n = 0;
  }

  emit(x) {
    const s = x > 1 ? 1 : x < -1 ? -1 : x;
    this.buf[this.n++] = s < 0 ? s * 32768 : s * 32767;
    if (this.n === FRAME_SAMPLES) {
      this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
      this.buf = new Int16Array(FRAME_SAMPLES);
      this.n = 0;
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0]; // block size is not guaranteed to be 128: use ch.length
    if (!ch) return true;
    if (this.step === 1) {
      for (let i = 0; i < ch.length; i++) this.emit(ch[i]);
      return true;
    }
    for (let i = 0; i < ch.length; i++) {
      const x = ch[i];
      while (this.pos <= 1) {
        this.emit(this.prev + (x - this.prev) * this.pos);
        this.pos += this.step;
      }
      this.pos -= 1;
      this.prev = x;
    }
    return true;
  }
}

registerProcessor("lw-pcm-capture", PcmCapture);
