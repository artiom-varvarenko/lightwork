// Browser mic capture for Lightwork voice input (briefs/webvoice.md §1.1).
// AudioWorklet -> PCM16 LE mono 24 kHz -> 80 ms binary frames -> ws://<host>/ws/mic.
// Call startMic() from a user gesture (the screen page's Start button). TTS muting happens on the server.
//
//   import { startMic } from "/static/js/mic.js";
//   const mic = await startMic({ onState: (s) => hud.voiceMic(s) });   // s = {state, label, error}
//   mic.stop();
//
// state: "off" (stopped, or the server has no voice service) | "connecting" | "live" | "reconnecting" | "error".

const RATE = 24000;
const VOICE_OFF_CLOSE = 4003; // server/routes/voice.py
const RETRY_MS = 1000;
const RETRY_OFF_MS = 5000;
const MAX_BUFFERED = 2 * 48000; // bytes; beyond ~2 s queued the socket is stuck: drop frames, the server pads silence
const LABEL_KEY = "lw.mic.label";
const CONSTRAINTS = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };

async function openStream(labelPattern) {
  // Labels are empty until permission is granted: open the default device first, then pick by label.
  let stream = await navigator.mediaDevices.getUserMedia({ audio: CONSTRAINTS });
  const inputs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
  const saved = localStorage.getItem(LABEL_KEY);
  const pick = inputs.find((d) => saved && d.label === saved) || inputs.find((d) => labelPattern.test(d.label));
  const current = stream.getAudioTracks()[0];
  if (pick && pick.deviceId !== current.getSettings().deviceId) {
    stream.getTracks().forEach((t) => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({ audio: { ...CONSTRAINTS, deviceId: { exact: pick.deviceId } } });
  }
  const label = stream.getAudioTracks()[0].label;
  localStorage.setItem(LABEL_KEY, label);
  return { stream, label };
}

async function openGraph(stream) {
  // Prefer a 24 kHz context (the browser resamples the mic); fall back to the native rate and resample in the worklet.
  let ctx = new AudioContext({ sampleRate: RATE });
  let source;
  try {
    source = ctx.createMediaStreamSource(stream);
  } catch (e) {
    await ctx.close();
    ctx = new AudioContext();
    source = ctx.createMediaStreamSource(stream);
  }
  await ctx.audioWorklet.addModule(new URL("./mic-worklet.js", import.meta.url));
  const node = new AudioWorkletNode(ctx, "lw-pcm-capture", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
  const sink = ctx.createGain();
  sink.gain.value = 0; // keeps the node pulled by the graph without playing the mic back
  source.connect(node).connect(sink).connect(ctx.destination);
  if (ctx.state !== "running") await ctx.resume();
  return { ctx, node };
}

export async function startMic({ onState = () => {}, labelPattern = /DJI|C920|Webcam/i, url } = {}) {
  const wsUrl = url || `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/mic`;
  let label = "";
  let ws = null;
  let stopped = false;
  let retry = null;
  const report = (state, error = null) => onState({ state, label, error });

  report("connecting");
  let stream, ctx, node;
  try {
    ({ stream, label } = await openStream(labelPattern));
    ({ ctx, node } = await openGraph(stream));
  } catch (e) {
    report("error", `${e.name || "Error"}: ${e.message}`);
    throw e;
  }

  const connect = () => {
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => report("live");
    ws.onclose = (ev) => {
      ws = null;
      if (stopped) return;
      const off = ev.code === VOICE_OFF_CLOSE;
      report(off ? "off" : "reconnecting", off ? ev.reason || "voice not running" : null);
      retry = setTimeout(connect, off ? RETRY_OFF_MS : RETRY_MS);
    };
  };

  node.port.onmessage = (ev) => {
    if (ws && ws.readyState === WebSocket.OPEN && ws.bufferedAmount < MAX_BUFFERED) ws.send(ev.data);
  };
  stream.getAudioTracks()[0].addEventListener("ended", () => {
    if (!stopped) report("error", "microphone disconnected");
  });
  connect();

  return {
    get label() {
      return label;
    },
    stop() {
      stopped = true;
      clearTimeout(retry);
      if (ws) ws.close(1000);
      node.port.onmessage = null;
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
      report("off");
    },
  };
}
