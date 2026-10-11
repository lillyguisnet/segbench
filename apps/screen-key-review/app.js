'use strict';
// Screen answer-key review. Boxes are screenshot pixels [left, top, right, bottom],
// edges included (a click on an edge counts), exactly as the scorer reads them
// (scripts/score_screens.py). Autosaves in this browser; Export writes the file
// scripts/accept_screen_key.py turns into the answer key.
const seed = window.SCREEN_SEED;
const NS = 'http://www.w3.org/2000/svg';
const storageKey = `segbench-screen-key-v1:${seed.draft_sha256}`;
const tabId = Math.random().toString(36).slice(2);
const $ = s => document.querySelector(s);
const now = () => new Date().toISOString();
const clone = x => JSON.parse(JSON.stringify(x));
const byKey = Object.fromEntries(seed.tasks.map(t => [t.key, t]));

function fresh() {
  const tasks = {};
  for (const t of seed.tasks) tasks[t.key] = {targets: clone(t.targets), accepted: !!t.accepted, accepted_at: t.accepted_at || null, notes: t.notes || ''};
  return {reviewer: '', tasks, log: []};
}

let state = fresh(), revision = 0, conflict = false;
try {
  const env = JSON.parse(localStorage.getItem(storageKey) || 'null');
  if (env && env.data && env.data.tasks) { state = env.data; revision = env.revision || 0; }
} catch (e) { /* a broken save starts fresh; the export files are the record */ }

let current = seed.tasks[0].key, mode = 'select', selected = null, hlModel = null, showClicks = true;
const history = {}, future = {};
for (const t of seed.tasks) { history[t.key] = []; future[t.key] = []; }
const view = {};  // per task: {x, y, w, h} in screenshot pixels

function save(action) {
  if (action) state.log.push({at: now(), task: current, action});
  revision += 1;
  try {
    localStorage.setItem(storageKey, JSON.stringify({revision, tab: tabId, data: state}));
    setStatus(conflict ? 'Another tab changed this draft. Export here before reloading.' : `Saved in this browser · ${new Date().toLocaleTimeString()}`, conflict);
  } catch (e) { setStatus('Could not save in this browser: export now.', true); }
}
window.addEventListener('storage', e => {
  if (e.key !== storageKey || !e.newValue) return;
  const env = JSON.parse(e.newValue);
  if (env.tab !== tabId && env.revision > revision) { conflict = true; setStatus('Another tab changed this draft. Export here before reloading.', true); }
});
function setStatus(text, warn) { const s = $('#saveStatus'); s.textContent = text; s.style.color = warn ? 'var(--bad)' : ''; }
let toastTimer;
function toast(text) { const t = $('#toast'); t.textContent = text; t.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => t.hidden = true, 3200); }

// ---------- editing, with undo ----------
const task = () => state.tasks[current];
function snapshot() { history[current].push(clone(task().targets)); if (history[current].length > 200) history[current].shift(); future[current] = []; }
function edited(action) {
  const t = task();
  if (t.accepted) { t.accepted = false; t.accepted_at = null; toast('Changed after accepting: accept this screenshot again when it is right.'); }
  save(action); render();
}
function undo() { const h = history[current]; if (!h.length) return; future[current].push(clone(task().targets)); task().targets = h.pop(); selected = null; edited('undo'); }
function redo() { const f = future[current]; if (!f.length) return; history[current].push(clone(task().targets)); task().targets = f.pop(); selected = null; edited('redo'); }
const isCircle = g => Array.isArray(g.circle);
const bounds = g => isCircle(g) ? [g.circle[0] - g.circle[2], g.circle[1] - g.circle[2], g.circle[0] + g.circle[2], g.circle[1] + g.circle[2]] : g.box;
const shapeText = g => isCircle(g) ? `circle ${JSON.stringify(g.circle)}` : JSON.stringify(g.box);
const r1 = v => Math.round(v * 10) / 10;
function normalise(b) { const [l, t, r, bt] = b.map(Math.round); return [Math.min(l, r), Math.min(t, bt), Math.max(l, r), Math.max(t, bt)]; }
function clampBox(b) { const [W, H] = byKey[current].image_size; return [Math.max(0, b[0]), Math.max(0, b[1]), Math.min(W - 1, b[2]), Math.min(H - 1, b[3])]; }
function newId() { const ids = new Set(task().targets.map(g => g.id)); let i = 1; while (ids.has(`target-${i}`)) i++; return `target-${i}`; }

// ---------- scoring, same rule as scripts/score_screens.py ----------
const inShape = (c, g) => isCircle(g) ? (c.x - g.circle[0]) ** 2 + (c.y - g.circle[1]) ** 2 <= g.circle[2] ** 2
  : g.box[0] <= c.x && c.x <= g.box[2] && g.box[1] <= c.y && c.y <= g.box[3];
function counts(c, g, all) {  // same rule as segbench/screens.py hits(): a target drawn on top takes the click
  if (!inShape(c, g)) return false;
  return !(g.covered_by || []).some(id => { const o = all.find(t => t.id === id); return o && inShape(c, o); });
}
const anyHit = (c, all) => all.some(g => counts(c, g, all));
function matched(clicks, targets) {  // maximum one-to-one matching, augmenting paths
  const owner = new Array(targets.length).fill(-1);
  const edges = clicks.map(c => targets.map((g, j) => counts(c, g, targets) ? j : -1).filter(j => j >= 0));
  const augment = (p, seen) => { for (const j of edges[p]) { if (seen.has(j)) continue; seen.add(j); if (owner[j] < 0 || augment(owner[j], seen)) { owner[j] = p; return true; } } return false; };
  clicks.forEach((_, p) => augment(p, new Set()));
  return owner.filter(o => o >= 0).length;
}
function f1(clicks, targets) { const tp = matched(clicks, targets); return clicks.length + targets.length ? 2 * tp / (clicks.length + targets.length) : 1; }

// ---------- view ----------
const svg = $('#svg');
function fit() { const [W, H] = byKey[current].image_size; view[current] = {x: 0, y: 0, w: W, h: H}; applyView(); }
function scale() { const r = svg.getBoundingClientRect(), v = view[current]; return Math.max(Math.min(r.width / v.w, r.height / v.h), 1e-3); }
function applyView() {
  const v = view[current]; svg.setAttribute('viewBox', `${v.x} ${v.y} ${v.w} ${v.h}`);
  $('#zoomLabel').textContent = `${Math.round(scale() * 100)}%`;
  drawOverlay();
}
function toImage(e) { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); }
function zoomAt(f, cx, cy) {
  const v = view[current], [W, H] = byKey[current].image_size;
  const w = Math.min(Math.max(v.w * f, 20), W * 4), k = w / v.w;
  view[current] = {x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k, w, h: v.h * k}; applyView();
}
function zoomTo(b) {
  const pad = Math.max(40, (b[2] - b[0]) * 1.5, (b[3] - b[1]) * 1.5), r = svg.getBoundingClientRect();
  let w = b[2] - b[0] + 2 * pad, h = b[3] - b[1] + 2 * pad;
  if (w / h < r.width / r.height) w = h * r.width / r.height; else h = w * r.height / r.width;
  view[current] = {x: (b[0] + b[2]) / 2 - w / 2, y: (b[1] + b[3]) / 2 - h / 2, w, h}; applyView();
}

// ---------- drawing ----------
function el(name, attrs, parent) { const e = document.createElementNS(NS, name); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (parent) parent.append(e); return e; }
let imageEl, overlay;
function buildStage() {
  svg.replaceChildren();
  const t = byKey[current], [W, H] = t.image_size;
  imageEl = el('image', {href: t.image, x: 0, y: 0, width: W, height: H}, svg);
  overlay = el('g', {}, svg);
  if (!view[current]) fit(); else applyView();
}
function handlesOf(g) {  // a circle has one handle: its right edge sets the radius
  return isCircle(g) ? {r: [g.circle[0] + g.circle[2], g.circle[1]]} : handles(g.box);
}
function handles(b) {
  const [l, t, r, bt] = b, mx = (l + r) / 2, my = (t + bt) / 2;
  return {nw: [l, t], n: [mx, t], ne: [r, t], e: [r, my], se: [r, bt], s: [mx, bt], sw: [l, bt], w: [l, my]};
}
function drawOverlay() {
  if (!overlay) return;
  overlay.replaceChildren();
  const px = 1 / scale(), all = task().targets;
  imageEl.style.imageRendering = scale() >= 2 ? 'pixelated' : 'auto';
  task().targets.forEach((g, i) => {
    const [l, t] = bounds(g), sel = i === selected;
    const look = {fill: sel ? 'rgba(0,170,70,.16)' : 'rgba(0,170,70,.07)', stroke: '#00a046', 'stroke-width': sel ? 3 : 2, 'vector-effect': 'non-scaling-stroke'};
    if (isCircle(g)) el('circle', {cx: g.circle[0], cy: g.circle[1], r: g.circle[2], ...look}, overlay);
    else el('rect', {x: l, y: t, width: g.box[2] - l, height: g.box[3] - t, ...look}, overlay);
    if (sel) {  // only the selected box is named: names would hide the screen being checked
      const label = el('text', {x: l, y: t - 7 * px, 'font-size': 12 * px, fill: '#00843a', 'font-weight': 700,
        'paint-order': 'stroke', stroke: 'white', 'stroke-width': 3 * px}, overlay);
      label.textContent = g.id;
    }
  });
  if (showClicks) {
    for (const c of byKey[current].clicks) {
      const hit = anyHit(c, all), hl = hlModel === c.model, dim = hlModel && !hl;
      const dot = el('circle', {cx: c.x, cy: c.y, r: (hl ? 7 : 5) * px, fill: hit ? '#00be46' : '#e61e1e',
        stroke: hl ? '#111' : 'white', 'stroke-width': (hl ? 2.5 : 1.5) * px, opacity: dim ? .25 : 1}, overlay);
      el('title', {}, dot).textContent = `${c.label} · try ${c.try}${c.n > 1 ? ` · click ${c.n}` : ''} · (${Math.round(c.x)}, ${Math.round(c.y)})`;
    }
  }
  if (selected !== null && task().targets[selected]) {
    for (const [x, y] of Object.values(handlesOf(task().targets[selected])))
      el('rect', {x: x - 4.5 * px, y: y - 4.5 * px, width: 9 * px, height: 9 * px, fill: 'white', stroke: '#00a046', 'stroke-width': 1.5 * px}, overlay);
  }
  if (drag && drag.kind === 'draw' && drag.circle) {
    el('circle', {cx: drag.start.x, cy: drag.start.y, r: Math.hypot(drag.at.x - drag.start.x, drag.at.y - drag.start.y), fill: 'rgba(0,170,70,.12)', stroke: '#00a046',
      'stroke-dasharray': '6 4', 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke'}, overlay);
  } else if (drag && drag.kind === 'draw') {
    const b = normalise([drag.start.x, drag.start.y, drag.at.x, drag.at.y]);
    el('rect', {x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1], fill: 'rgba(0,170,70,.12)', stroke: '#00a046',
      'stroke-dasharray': '6 4', 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke'}, overlay);
  }
}

// ---------- pointer ----------
let drag = null, spaceDown = false;
function hitHandle(p) {
  if (selected === null || !task().targets[selected]) return null;
  const tol = 8 / scale();
  for (const [name, [x, y]] of Object.entries(handlesOf(task().targets[selected])))
    if (Math.abs(p.x - x) <= tol && Math.abs(p.y - y) <= tol) return name;
  return null;
}
function hitBox(p) {  // the smallest box under the pointer (a little slack at small zoom)
  const tol = 3 / scale();
  const area = g => { const b = bounds(g); return (b[2] - b[0]) * (b[3] - b[1]); };
  const near = g => isCircle(g) ? Math.hypot(p.x - g.circle[0], p.y - g.circle[1]) <= g.circle[2] + tol
    : g.box[0] - tol <= p.x && p.x <= g.box[2] + tol && g.box[1] - tol <= p.y && p.y <= g.box[3] + tol;
  const under = task().targets.map((g, i) => [g, i]).filter(([g]) => near(g));
  under.sort((a, b) => area(a[0]) - area(b[0]));
  return under.length ? under[0][1] : null;
}
svg.addEventListener('pointerdown', e => {
  svg.focus();
  const p = toImage(e);
  const pan = mode === 'pan' || spaceDown || e.button === 1;
  const shapeOf = g => [...(isCircle(g) ? g.circle : g.box)];
  if (!pan && (mode === 'draw' || mode === 'circle')) { drag = {kind: 'draw', circle: mode === 'circle', start: p, at: p}; }
  else if (!pan && mode === 'select' && hitHandle(p)) { snapshot(); drag = {kind: 'resize', handle: hitHandle(p), box: shapeOf(task().targets[selected]), start: p, moved: false}; }
  else if (!pan && mode === 'select' && hitBox(p) !== null) { selected = hitBox(p); snapshot(); drag = {kind: 'move', box: shapeOf(task().targets[selected]), start: p, moved: false}; renderSide(); }
  else { if (mode === 'select' && !pan) { selected = null; renderSide(); } drag = {kind: 'pan', sx: e.clientX, sy: e.clientY, v: {...view[current]}}; svg.classList.add('dragging'); }
  svg.setPointerCapture(e.pointerId); drawOverlay();
});
svg.addEventListener('pointermove', e => {
  const p = toImage(e);
  $('#coords').textContent = `x ${Math.round(p.x)} · y ${Math.round(p.y)}`;
  if (!drag) return;
  if (drag.kind === 'pan') {
    const k = 1 / scale();
    view[current] = {...drag.v, x: drag.v.x - (e.clientX - drag.sx) * k, y: drag.v.y - (e.clientY - drag.sy) * k}; applyView(); return;
  }
  if (drag.kind === 'draw') { drag.at = p; drawOverlay(); return; }
  const g = task().targets[selected];
  if (isCircle(g)) {  // circles move and resize in tenths of a pixel
    const dx = r1(p.x - drag.start.x), dy = r1(p.y - drag.start.y);
    if (dx || dy) drag.moved = true;
    const c = [...drag.box];
    if (drag.kind === 'move') { c[0] = r1(c[0] + dx); c[1] = r1(c[1] + dy); }
    else c[2] = Math.max(0.5, r1(Math.hypot(p.x - c[0], p.y - c[1])));
    g.circle = c; drawOverlay(); renderSide(true); return;
  }
  const dx = Math.round(p.x - drag.start.x), dy = Math.round(p.y - drag.start.y);
  if (dx || dy) drag.moved = true;
  const b = [...drag.box];
  if (drag.kind === 'move') { b[0] += dx; b[2] += dx; b[1] += dy; b[3] += dy; }
  else {
    if (drag.handle.includes('w')) b[0] += dx; if (drag.handle.includes('e')) b[2] += dx;
    if (drag.handle.includes('n')) b[1] += dy; if (drag.handle.includes('s')) b[3] += dy;
  }
  task().targets[selected].box = b; drawOverlay(); renderSide(true);
});
svg.addEventListener('pointerup', () => {
  if (!drag) return;
  const d = drag; drag = null; svg.classList.remove('dragging');
  if (d.kind === 'draw' && d.circle) {
    const r = r1(Math.hypot(d.at.x - d.start.x, d.at.y - d.start.y));
    if (r >= 1.5) {
      snapshot(); task().targets.push({id: newId(), circle: [r1(d.start.x), r1(d.start.y), r], why: 'Added by the reviewer.'}); selected = task().targets.length - 1;
      setMode('select'); edited(`added circle ${task().targets[selected].id} ${JSON.stringify(task().targets[selected].circle)}`); return;
    }
    drawOverlay(); return;
  }
  if (d.kind === 'draw') {
    const b = clampBox(normalise([d.start.x, d.start.y, d.at.x, d.at.y]));
    if (b[2] - b[0] >= 2 && b[3] - b[1] >= 2) {
      snapshot(); task().targets.push({id: newId(), box: b, why: 'Added by the reviewer.'}); selected = task().targets.length - 1;
      setMode('select'); edited(`added box ${task().targets[selected].id} ${JSON.stringify(b)}`); return;
    }
    drawOverlay(); return;
  }
  if (d.kind === 'move' || d.kind === 'resize') {
    const g = task().targets[selected];
    if (!d.moved) { history[current].pop(); if (isCircle(g)) g.circle = d.box; else g.box = d.box; render(); return; }
    if (!isCircle(g)) g.box = clampBox(normalise(g.box));
    edited(`${d.kind === 'move' ? 'moved' : 'resized'} ${g.id} to ${shapeText(g)}`);
  }
});
svg.addEventListener('wheel', e => { e.preventDefault(); const p = toImage(e); zoomAt(Math.exp(e.deltaY * 0.0015), p.x, p.y); }, {passive: false});
svg.addEventListener('dblclick', e => { const i = hitBox(toImage(e)); if (i !== null) zoomTo(bounds(task().targets[i])); });

// ---------- keys ----------
document.addEventListener('keydown', e => {
  const typing = /INPUT|TEXTAREA/.test(document.activeElement.tagName);
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { if (typing) return; e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { if (typing) return; e.preventDefault(); redo(); return; }
  if (typing || e.ctrlKey || e.metaKey) return;
  if (e.key === ' ') { spaceDown = true; svg.classList.add('panning'); e.preventDefault(); return; }
  const k = e.key.toLowerCase();
  if (k === 'v') setMode('select'); else if (k === 'b') setMode('draw'); else if (k === 'o') setMode('circle'); else if (k === 'h') setMode('pan');
  else if (k === 'f') fit(); else if (k === 'n') { const i = seed.tasks.findIndex(t => t.key === current); go(seed.tasks[(i + 1) % seed.tasks.length].key); }
  else if ((e.key === 'Delete' || e.key === 'Backspace') && selected !== null) { e.preventDefault(); removeBox(selected); }
  else if (e.key.startsWith('Arrow') && selected !== null) {
    e.preventDefault();
    const g = task().targets[selected];
    if (isCircle(g)) {  // 0.5 px steps (Shift: 5); Alt + up/down changes the radius
      const st = e.shiftKey ? 5 : 0.5, c = [...g.circle];
      const ddx = e.key === 'ArrowLeft' ? -st : e.key === 'ArrowRight' ? st : 0, ddy = e.key === 'ArrowUp' ? -st : e.key === 'ArrowDown' ? st : 0;
      if (e.altKey) c[2] = Math.max(0.5, r1(c[2] - ddy + ddx)); else { c[0] = r1(c[0] + ddx); c[1] = r1(c[1] + ddy); }
      snapshot(); g.circle = c; edited(`${e.altKey ? 'resized' : 'nudged'} ${g.id} to ${shapeText(g)}`); return;
    }
    const s = e.shiftKey ? 10 : 1, b = [...task().targets[selected].box];
    const dx = e.key === 'ArrowLeft' ? -s : e.key === 'ArrowRight' ? s : 0, dy = e.key === 'ArrowUp' ? -s : e.key === 'ArrowDown' ? s : 0;
    if (e.altKey) { b[2] += dx; b[3] += dy; } else { b[0] += dx; b[2] += dx; b[1] += dy; b[3] += dy; }
    snapshot(); task().targets[selected].box = clampBox(normalise(b));
    edited(`${e.altKey ? 'resized' : 'nudged'} ${task().targets[selected].id} to ${JSON.stringify(task().targets[selected].box)}`);
  }
});
document.addEventListener('keyup', e => { if (e.key === ' ') { spaceDown = false; svg.classList.remove('panning'); } });

function setMode(m) { mode = m; document.querySelectorAll('[data-mode]').forEach(b => b.classList.toggle('active', b.dataset.mode === m)); svg.className.baseVal = `mode-${m}`; }
document.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => setMode(b.dataset.mode));
$('#undo').onclick = undo; $('#redo').onclick = redo;
$('#zoomIn').onclick = () => { const v = view[current]; zoomAt(1 / 1.5, v.x + v.w / 2, v.y + v.h / 2); };
$('#zoomOut').onclick = () => { const v = view[current]; zoomAt(1.5, v.x + v.w / 2, v.y + v.h / 2); };
$('#fit').onclick = fit;
$('#showClicks').onchange = e => { showClicks = e.target.checked; drawOverlay(); };
$('#resetTask').onclick = () => {
  if (!confirm('Throw away every change to this screenshot and go back to the draft boxes?')) return;
  snapshot(); task().targets = clone(byKey[current].targets); selected = null; edited('reset to draft');
};
function removeBox(i) { const g = task().targets[i]; snapshot(); task().targets.splice(i, 1); task().targets.forEach(o => { if (o.covered_by) o.covered_by = o.covered_by.filter(id => id !== g.id); }); selected = null; edited(`removed box ${g.id}`); }

// ---------- side panel ----------
function render() { renderNav(); renderSide(); drawOverlay(); $('#undo').disabled = !history[current].length; $('#redo').disabled = !future[current].length; }
function renderNav() {
  const nav = $('#tasks'); nav.replaceChildren();
  for (const t of seed.tasks) {
    const s = state.tasks[t.key], b = document.createElement('button');
    const changed = JSON.stringify(s.targets) !== JSON.stringify(t.targets);
    b.innerHTML = `${t.key[0].toUpperCase() + t.key.slice(1)}<small class="${s.accepted ? 'ok' : changed ? 'warn' : ''}">${s.accepted ? '✓ accepted' : changed ? 'edited, not accepted' : 'not accepted yet'}</small>`;
    b.classList.toggle('active', t.key === current); b.onclick = () => go(t.key); nav.append(b);
  }
}
function scoresByAnswer() {
  const targets = task().targets, out = new Map();
  for (const a of byKey[current].answers) out.set(`${a.model}|${a.try}`, {...a, clicks: []});
  for (const c of byKey[current].clicks) out.get(`${c.model}|${c.try}`).clicks.push(c);
  for (const a of out.values()) a.f1 = f1(a.clicks, targets);
  return [...out.values()];
}
function renderSide(boxesOnlyNumbers) {
  const t = byKey[current], s = task();
  const clicks = t.clicks, inn = clicks.filter(c => anyHit(c, s.targets)).length;
  $('#tally').innerHTML = `<b>${inn}</b> of ${clicks.length} model clicks are on a target · ${s.targets.length} target${s.targets.length === 1 ? '' : 's'}`;
  if (boxesOnlyNumbers) {  // while dragging: only refresh the numbers, keep focus
    s.targets.forEach((g, i) => document.querySelectorAll(`[data-box="${i}"] .edges input`).forEach((inp, k) => inp.value = (isCircle(g) ? g.circle : g.box)[k]));
    return;
  }
  $('#instruction').textContent = t.instruction; $('#what').textContent = t.what;
  const list = $('#boxes'); list.replaceChildren();
  s.targets.forEach((g, i) => {
    const d = document.createElement('div'); d.className = 'box' + (i === selected ? ' sel' : ''); d.dataset.box = i;
    const n = clicks.filter(c => counts(c, g, s.targets)).length, circ = isCircle(g), vals = circ ? g.circle : g.box;
    const fields = circ ? ['centre x', 'centre y', 'radius'] : ['left', 'top', 'right', 'bottom'];
    d.innerHTML = `<div class="row"><input class="name" aria-label="Box name"><button class="small" data-act="zoom" title="Zoom to this box">Zoom</button><button class="small danger" data-act="del" title="Delete this box">✕</button></div>
      <div class="edges">${fields.map(k => `<label>${k}<input type="number" step="${circ ? 0.1 : 1}"></label>`).join('')}</div>
      <div class="hits">${n} click${n === 1 ? '' : 's'} count · ${circ ? `round, ${r1(2 * g.circle[2])} px across` : `${g.box[2] - g.box[0] + 1} × ${g.box[3] - g.box[1] + 1} px`}${(g.covered_by || []).length ? ` · under ${g.covered_by.join(', ')}` : ''}</div>
      <textarea rows="2" aria-label="Why this box" placeholder="Why this box"></textarea>`;
    const name = d.querySelector('.name'); name.value = g.id;
    name.onchange = () => { const v = name.value.trim(); if (!v || s.targets.some((o, j) => j !== i && o.id === v)) { name.value = g.id; toast('Names must be unique and not empty.'); return; } snapshot(); const old = g.id; g.id = v; s.targets.forEach(o => { if (o.covered_by) o.covered_by = o.covered_by.map(id => id === old ? v : id); }); edited(`renamed ${old} to ${v}`); };
    d.querySelectorAll('.edges input').forEach((inp, k) => { inp.value = vals[k]; inp.onchange = () => { const v = Number(inp.value); if (!Number.isFinite(v) || (circ && k === 2 && v <= 0)) { inp.value = vals[k]; return; } snapshot();
      if (circ) { const c = [...g.circle]; c[k] = r1(v); g.circle = c; } else { const b = [...g.box]; b[k] = v; g.box = clampBox(normalise(b)); }
      edited(`set ${g.id} to ${shapeText(g)}`); }; });
    const why = d.querySelector('textarea'); why.value = g.why || ''; why.onchange = () => { g.why = why.value; save(`note on ${g.id}`); };
    d.querySelector('[data-act=zoom]').onclick = ev => { ev.stopPropagation(); selected = i; zoomTo(bounds(g)); renderSide(); };
    d.querySelector('[data-act=del]').onclick = ev => { ev.stopPropagation(); removeBox(i); };
    d.onclick = ev => { if (ev.target.closest('input,textarea,button')) return; selected = i; renderSide(); drawOverlay(); };
    list.append(d);
  });
  if (!s.targets.length) list.innerHTML = '<p class="muted small">No target: no click can be right. Use “Draw box” or “Draw circle” to add one.</p>';
  const acc = $('#acceptBox');
  if (s.accepted) {
    acc.innerHTML = `<div class="accepted"><span>✓ Accepted ${new Date(s.accepted_at).toLocaleString()}</span><button class="small">Reopen</button></div>`;
    acc.querySelector('button').onclick = () => { s.accepted = false; s.accepted_at = null; save('reopened'); render(); };
  } else {
    acc.innerHTML = `<button class="accept-btn">✓ Accept this screenshot's answer key</button>`;
    acc.querySelector('button').onclick = () => {
      if (!s.targets.length && !confirm('No target: every click on this screenshot will count as wrong. Accept anyway?')) return;
      s.accepted = true; s.accepted_at = now(); save(`accepted with ${JSON.stringify(s.targets.map(g => [g.id, g.box || g.circle]))}`); render();
      const next = seed.tasks.find(x => !state.tasks[x.key].accepted);
      toast(next ? `Accepted. Next: ${next.instruction}` : `All ${seed.tasks.length} accepted. Export the answer key (button at the top).`);
    };
  }
  $('#notes').value = s.notes || '';
  const table = $('#models'); table.replaceChildren();
  const rows = new Map();
  for (const a of scoresByAnswer()) { if (!rows.has(a.model)) rows.set(a.model, {label: a.label, tries: []}); rows.get(a.model).tries[a.try - 1] = a; }
  const sorted = [...rows.entries()].sort((x, y) => mean(y[1].tries) - mean(x[1].tries) || x[1].label.localeCompare(y[1].label));
  for (const [model, r] of sorted) {
    const tr = document.createElement('tr'); tr.classList.toggle('hl', hlModel === model);
    tr.innerHTML = `<td>${r.label}</td>` + r.tries.map(a => { const v = a.f1; return `<td class="s ${v === 1 ? 'ok' : v === 0 ? 'bad' : 'part'}" title="${a.clicks.length} click${a.clicks.length === 1 ? '' : 's'}">${v === 1 ? '✓' : v === 0 ? '✗' : Math.round(v * 100)}</td>`; }).join('');
    tr.onclick = () => { hlModel = hlModel === model ? null : model; renderSide(); drawOverlay(); };
    table.append(tr);
  }
}
const mean = tries => tries.reduce((s, a) => s + a.f1, 0) / tries.length;
$('#notes').onchange = e => { task().notes = e.target.value; save('notes'); };
$('#reviewer').value = state.reviewer || '';
$('#reviewer').onchange = e => { state.reviewer = e.target.value.trim(); save(); };

function go(key) { current = key; selected = null; hlModel = null; buildStage(); render(); }

// ---------- export / import ----------
$('#export').onclick = () => {
  const done = seed.tasks.filter(t => state.tasks[t.key].accepted).length;
  if (!state.reviewer) { toast('Write your name in “Reviewed by” first.'); $('#reviewer').focus(); return; }
  const out = {
    schema: 'segbench-screen-key/v1',
    status: done === seed.tasks.length ? `accepted by ${state.reviewer}` : `partly reviewed by ${state.reviewer}: ${done} of ${seed.tasks.length} screenshots accepted`,
    reviewer: state.reviewer, exported_at: now(), draft: seed.draft, draft_sha256: seed.draft_sha256, run: seed.run,
    coordinates: 'screenshot pixels, [left, top, right, bottom], edges included',
    scoring: 'a click is right when it falls inside a box no earlier click has claimed; score = F1 per answer',
    tasks: Object.fromEntries(seed.tasks.map(t => { const s = state.tasks[t.key]; return [t.key, {
      instruction: t.instruction, image: t.image.replace(/^images\//, 'screenshots/'), image_size: t.image_size, image_sha256: t.image_sha256,
      accepted: s.accepted, accepted_at: s.accepted_at, notes: s.notes, targets: s.targets}]; })),
    log: state.log,
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'}), url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `segbench-screen-key-${seed.draft_sha256.slice(0, 8)}-${now().replace(/[:.]/g, '-')}.json`;
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 60000);
  toast(done === seed.tasks.length ? `Exported: all ${seed.tasks.length} accepted.` : `Exported: ${done} of ${seed.tasks.length} accepted so far.`);
};
$('#import').onclick = () => $('#file').click();
$('#file').onchange = async e => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    const d = JSON.parse(await f.text());
    if (d.schema !== 'segbench-screen-key/v1') throw new Error('not a screen answer-key export');
    if (d.draft_sha256 !== seed.draft_sha256 && !confirm('This file was made from another draft. Load it anyway?')) return;
    const next = fresh();
    for (const t of seed.tasks) if (d.tasks[t.key]) Object.assign(next.tasks[t.key], {targets: d.tasks[t.key].targets, accepted: !!d.tasks[t.key].accepted, accepted_at: d.tasks[t.key].accepted_at || null, notes: d.tasks[t.key].notes || ''});
    next.reviewer = d.reviewer || ''; next.log = [...(d.log || []), {at: now(), action: `imported ${f.name}`}];
    state = next; $('#reviewer').value = state.reviewer; for (const t of seed.tasks) { history[t.key] = []; future[t.key] = []; }
    save(); go(current); toast(`Loaded ${f.name}`);
  } catch (err) { toast(`Could not load: ${err.message}`); }
};

window.addEventListener('resize', () => applyView());
setMode('select'); go(current);
setStatus(revision ? 'Restored your work from this browser' : 'Draft loaded · saves in this browser as you go');
