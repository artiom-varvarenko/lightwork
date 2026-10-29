import {drawScene, fitRect, rectMap, PALETTES} from './scene.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const base = new URL('./', location.href);
const httpViewer = location.pathname.endsWith('/viewer');
let bundle, data, mediaBase, time = 0, duration = 0, playing = false;
let snapshot = null, snapshotIndex = -1, image = null, imagePath = null, imageGeneration = 0;
let lastTick = null, loadGeneration = 0;

function localURL(path, root = base) {
  if (typeof path !== 'string' || !path || path.includes('\\') || path.split('/').some(p => p === '..' || p === '.') || /[%?#]/.test(path)) throw new Error('Unsafe artifact path');
  const url = new URL(path, root);
  if (url.origin !== root.origin || !url.pathname.startsWith(root.pathname)) throw new Error('Artifact outside recorded run');
  return url;
}
async function json(url) {
  const response = await fetch(url, {credentials: 'omit', redirect: 'error'});
  if (!response.ok) throw new Error(`${response.status}: ${url.pathname}`);
  return response.json();
}
function text(tag, value, parent) {
  const node = document.createElement(tag);
  node.textContent = value;
  parent.append(node);
  return node;
}
function pause() {
  playing = false;
  $('play').textContent = 'Play';
}
function fail(error) {
  pause();
  $('play').disabled = true;
  $('error').textContent = error.message;
}
function panels(state) {
  $('steps').replaceChildren();
  for (const step of state.steps || []) {
    const li = text('li', step.say || step.id, $('steps'));
    li.dataset.state = step.state;
    text('span', step.state, li).className = 'badge';
    if (step.reason) text('div', step.reason, li).className = 'muted';
  }
  $('chips').replaceChildren();
  for (const chip of state.chips || []) {
    const node = text('div', `${chip.step} / ${chip.check}: ${chip.status}`, $('chips'));
    node.className = 'chip'; node.dataset.status = chip.status;
    if (chip.reason) text('div', chip.reason, node);
  }
  if (!state.chips?.length) text('span', 'No checks yet', $('chips')).className = 'muted';
  $('trace').replaceChildren();
  for (const row of state.trace || []) {
    const tr = document.createElement('tr'); $('trace').append(tr);
    text('td', `${row.model} / ${row.role}`, tr);
    text('td', row.status, tr);
    text('td', row.latency_ms == null ? '—' : `${row.latency_ms.toFixed(1)} ms`, tr);
    text('td', row.cost_usd == null ? '—' : `$${row.cost_usd.toFixed(6)}`, tr);
    const cell = text('td', '', tr);
    const record = data.trace.find(r => r.intent_id === row.intent_id && r.t <= time);
    if (record) { const link = text('a', 'Call metadata', cell); link.href = localURL(`${record.call_dir}/meta.json`, mediaBase); }
  }
  $('cost').textContent = `· total $${(state.cost_usd || 0).toFixed(6)}`;
  $('instruction').textContent = state.last_say || 'Awaiting recorded procedure input';
}
function draw() {
  const box = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const w = Math.round(box.width * dpr), h = Math.round(box.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {canvas.width = w; canvas.height = h;}
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  if (!snapshot) return;
  const crop = snapshot.image?.crop || data.crop;
  const dest = fitRect(crop, box.width, box.height);
  if (image) ctx.drawImage(image, ...dest);
  drawScene(ctx, snapshot.scene, {map: rectMap(crop, dest), palette: PALETTES.screen, lineWidth: 3, fontPx: 20, t: time, outline: true});
}
function seek(value) {
  time = Math.max(0, Math.min(duration, value));
  $('seek').value = time;
  $('clock').textContent = `${time.toFixed(3)} / ${duration.toFixed(3)} s`;
  let lo = 0, hi = data.timeline.length;
  while (lo < hi) {const mid = (lo + hi) >>> 1; if (data.timeline[mid].t <= time) lo = mid + 1; else hi = mid;}
  const index = lo - 1;
  if (index !== snapshotIndex) {
    snapshotIndex = index; snapshot = index < 0 ? null : data.timeline[index];
    panels(snapshot?.state || {});
    const path = snapshot?.image?.path || null;
    if (path !== imagePath) {
      imagePath = path; image = null;
      const generation = ++imageGeneration;
      if (path) {
        const next = new Image();
        next.onload = () => {if (generation === imageGeneration) {image = next; draw();}};
        next.onerror = () => {if (generation === imageGeneration) fail(new Error(`Cannot load recorded image: ${path}`));};
        next.src = localURL(path, mediaBase);
      }
    }
  }
  draw();
}
async function loadRun(id) {
  const generation = ++loadGeneration;
  pause(); $('play').disabled = true; $('error').textContent = '';
  if (!/^[A-Za-z0-9_-]+$/.test(id)) throw new Error('Invalid run id');
  const entry = bundle.runs.find(run => run.id === id);
  if (!entry) throw new Error('Run is not in this bundle');
  const next = await json(httpViewer ? localURL(`api/viewer/${id}`) : localURL(entry.path));
  if (generation !== loadGeneration) return;
  if (!next.verify.ok || next.verify.diffs.length) throw new Error('This run does not have a 0-diff verification');
  if (!next.timeline.length) throw new Error('Run has no replay snapshots');
  data = next;
  mediaBase = localURL(httpViewer ? `api/runs/${id}/` : `runs/${id}/`);
  duration = data.timeline.reduce((last, item) => Math.max(last, item.t), 0);
  snapshotIndex = -2; snapshot = null; imagePath = null; image = null; imageGeneration++;
  $('seek').max = duration; $('runs').value = id;
  $('banner').textContent = data.banner;
  $('banner').classList.toggle('synthetic', data.manifest.provenance !== 'meta');
  $('verify').textContent = `verify: 0 diffs · ${data.verify.n_events} events`;
  for (const [element, path] of [['manifest','manifest.json'], ['report','verify.json'], ['events','events.jsonl']]) $(element).href = localURL(path, mediaBase);
  // HTTP verification is computed without mutating the source run, so expose the fresh report itself.
  if (httpViewer) {
    if ($('report').dataset.blob) URL.revokeObjectURL($('report').dataset.blob);
    const url = URL.createObjectURL(new Blob([JSON.stringify(data.verify, null, 2)], {type:'application/json'}));
    $('report').href = url; $('report').dataset.blob = url;
  }
  const url = new URL(location.href); url.searchParams.set('run', id); history.replaceState(null, '', url);
  $('play').disabled = false;
  seek(0);
}
$('play').addEventListener('click', () => {
  if (playing) pause();
  else {if (time >= duration) seek(0); playing = true; lastTick = null; $('play').textContent = 'Pause';}
});
$('seek').addEventListener('input', () => {pause(); seek(Number($('seek').value));});
$('runs').addEventListener('change', () => loadRun($('runs').value).catch(fail));
window.addEventListener('resize', draw);
function tick(now) {
  if (playing && data) {
    const delta = lastTick == null ? 0 : (now - lastTick) / 1000;
    seek(time + delta * Number($('speed').value));
    if (time >= duration) pause();
  }
  lastTick = now;
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
async function start() {
  if (httpViewer) {
    const runs = await json(localURL('api/runs'));
    bundle = {runs: runs.map(run => ({...run, id: run.id}))};
  } else bundle = await json(localURL('bundle.json'));
  for (const run of bundle.runs) {const option = text('option', `${run.id} · ${run.provenance || 'recorded'}`, $('runs')); option.value = run.id;}
  if (!bundle.runs.length) throw new Error('No recorded runs available');
  await loadRun(new URLSearchParams(location.search).get('run') || bundle.runs[0].id);
}
start().catch(fail);
