'use strict';
(() => {
const seed = window.MEASURE_SEED;
const $ = s => document.querySelector(s);
const clone = x => JSON.parse(JSON.stringify(x));
const now = () => new Date().toISOString();
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const storageKey = `segbench-measure-clicks-v1:${seed.questions_sha256}`;
const Q = seed.questions, byKey = Object.fromEntries(Q.map(q => [q.key, q]));
const canvas = $('#canvas'), ctx = canvas.getContext('2d');
const images = new Map();
let qi = 0, selected = null, draft = null;          // draft: points of the item being clicked
let scale = 1, fitScale = 1, tx = 0, ty = 0, cw = 0, ch = 0, image = null, gen = 0;
let gesture = null, space = false, cursor = null, lastAdd = {t: 0, x: 0, y: 0};
let server = false, revision = 0, saveTimer, conflict = false, saveError = '', toastTimer;
const undos = [], redos = [];

function emptyState() {
 return {schema: 'segbench-measure-clicks/v1', questions_sha256: seed.questions_sha256,
  coordinate_system: 'original upright photo pixels, [x, y]',
  purpose: 'reference measurements for outline-from-a-dot questions: a line along the middle of a part, or two points; dot = what the models are shown',
  created_at: now(), updated_at: now(), items: [],
  questions: Object.fromEntries(Q.map(q => [q.key, {done: false, dropped: false, notes: ''}])), audit: []};
}
function valid(d) {
 if (!d || d.schema !== 'segbench-measure-clicks/v1' || d.questions_sha256 !== seed.questions_sha256) throw Error('This file is not a click file for these questions.');
 for (const q of Q) d.questions[q.key] ||= {done: false, dropped: false, notes: ''};
 return d;
}
let state = emptyState();
const q = () => Q[qi];
const photo = () => seed.photos[q().photo];
const items = () => state.items.filter(it => it.q === q().key);
const sel = () => state.items.find(it => it.id === selected);
const isDraft = it => it.status === 'draft';
const drafts = () => items().filter(isDraft);

// --- measurements -------------------------------------------------------------
function length(pts) { let s = 0; for (let i = 1; i < pts.length; i++) s += Math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]); return s; }
function middle(pts) {
 const half = length(pts) / 2; let s = 0;
 for (let i = 1; i < pts.length; i++) {
  const d = Math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]);
  if (s + d >= half && d > 0) { const f = (half - s) / d; return [pts[i-1][0] + f * (pts[i][0] - pts[i-1][0]), pts[i-1][1] + f * (pts[i][1] - pts[i-1][1])].map(v => Math.round(v * 10) / 10); }
  s += d;
 }
 return pts[0].slice();
}
function angle(pts) { const [a, b] = [pts[0], pts[pts.length - 1]]; let d = Math.atan2(-(b[1] - a[1]), b[0] - a[0]) * 180 / Math.PI; if (d < 0) d += 180; return d; }
function describe(it) {
 const L = length(it.points).toFixed(1);
 return byKey[it.q].kind === 'span' ? `${L} px · ${angle(it.points).toFixed(1)}° from horizontal` : `${L} px along ${it.points.length} clicks`;
}
const dotRadius = () => Math.max(4, Math.round(Math.min(photo().width, photo().height) * 0.008)); // as scripts/probe_point_segment.py draws it

// --- saving -------------------------------------------------------------------
function status(s, bad) { $('#saveStatus').textContent = s; $('#saveStatus').style.color = bad ? 'var(--bad)' : ''; }
function showStatus() {
 if (conflict) return status('Another window saved newer clicks. Export yours, then reload.', true);
 if (saveError) return status(saveError, true);
 status(server ? `Saved on lambda (revision ${revision})` : 'Saved in this browser only: Export to keep your work');
}
function local() { try { localStorage.setItem(storageKey, JSON.stringify({revision, data: state})); } catch (e) { saveError = 'Could not save in this browser: export a backup'; } }
async function push() {
 local();
 if (!server || conflict) return showStatus();
 try {
  const r = await fetch('api/save', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({base_revision: revision, data: state})});
  const j = await r.json();
  if (r.status === 409) conflict = true;
  else if (!r.ok) saveError = `Not saved on lambda: ${j.error}`;
  else { revision = j.revision; saveError = ''; local(); }
 } catch (e) { saveError = 'Lambda unreachable: saved in this browser only'; }
 showStatus();
}
function changed(action, details = {}) {
 state.updated_at = now(); state.audit.push({at: now(), action, q: q().key, ...details});
 status('Saving…'); clearTimeout(saveTimer); saveTimer = setTimeout(push, 400); update();
}
function mutate(action, fn, details) {
 const before = clone({items: state.items, questions: state.questions});
 fn(); undos.push(before); if (undos.length > 200) undos.shift(); redos.length = 0; changed(action, details);
}
function history(redo) {
 const from = redo ? redos : undos, to = redo ? undos : redos; if (!from.length) return;
 to.push(clone({items: state.items, questions: state.questions})); Object.assign(state, from.pop());
 if (!sel()) selected = null; changed(redo ? 'redo' : 'undo');
}
window.addEventListener('beforeunload', e => { clearTimeout(saveTimer); local(); if (conflict || saveError || !server) { e.preventDefault(); e.returnValue = ''; } });

// --- view ---------------------------------------------------------------------
function fit() { if (!cw) return; fitScale = Math.min(cw / photo().width, ch / photo().height) * .96; scale = fitScale; tx = (cw - photo().width * scale) / 2; ty = (ch - photo().height * scale) / 2; draw(); }
function zoom(f, x = cw / 2, y = ch / 2) { const ns = Math.max(fitScale * .5, Math.min(fitScale * 80, scale * f)); tx = x - (x - tx) * ns / scale; ty = y - (y - ty) * ns / scale; scale = ns; draw(); }
function focusOn(it) {
 const xs = it.points.map(p => p[0]), ys = it.points.map(p => p[1]);
 const w = Math.max(60, Math.max(...xs) - Math.min(...xs)), h = Math.max(60, Math.max(...ys) - Math.min(...ys));
 scale = Math.max(fitScale, Math.min(fitScale * 40, Math.min(cw / (w * 3), ch / (h * 3))));
 tx = cw / 2 - (Math.min(...xs) + Math.max(...xs)) / 2 * scale; ty = ch / 2 - (Math.min(...ys) + Math.max(...ys)) / 2 * scale; draw();
}
function resize() { const r = $('#viewport').getBoundingClientRect(), first = !cw; cw = r.width; ch = r.height; const d = devicePixelRatio || 1;
 canvas.width = Math.round(cw * d); canvas.height = Math.round(ch * d); ctx.setTransform(d, 0, 0, d, 0, 0); if (first) fit(); else { fitScale = Math.min(cw / photo().width, ch / photo().height) * .96; draw(); } }
new ResizeObserver(resize).observe($('#viewport'));
const sp = ev => { const r = canvas.getBoundingClientRect(); return {x: ev.clientX - r.left, y: ev.clientY - r.top}; };
const toImg = p => [(p.x - tx) / scale, (p.y - ty) / scale];
const toScr = p => [p[0] * scale + tx, p[1] * scale + ty];

function path(pts, color, width, dash) {
 if (pts.length < 1) return;
 ctx.save(); ctx.setLineDash(dash || []); ctx.lineJoin = ctx.lineCap = 'round';
 ctx.beginPath(); pts.forEach((p, i) => { const [x, y] = toScr(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
 ctx.strokeStyle = 'rgba(0,0,0,.75)'; ctx.lineWidth = width + 2.5; ctx.stroke(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke(); ctx.restore();
}
function handles(pts, color) { for (const p of pts) { const [x, y] = toScr(p); ctx.fillStyle = '#111'; ctx.fillRect(x - 4, y - 4, 8, 8); ctx.fillStyle = color; ctx.fillRect(x - 3, y - 3, 6, 6); } }
function dot(p, strong) {
 const [x, y] = toScr(p), r = Math.max(3.5, dotRadius() * scale), ring = Math.max(1.2, r / 3);
 ctx.globalAlpha = strong ? 1 : .8; ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(x, y, r + ring, 0, 7); ctx.fill();
 ctx.fillStyle = '#e60000'; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
}
function draw() {
 ctx.clearRect(0, 0, cw, ch);
 if (image?.complete && image.naturalWidth) ctx.drawImage(image, tx, ty, photo().width * scale, photo().height * scale);
 if ($('#others').checked) for (const it of state.items) if (it.q !== q().key && byKey[it.q].photo === q().photo) path(it.points, 'rgba(200,200,200,.75)', 1.4, [4, 4]);
 for (const it of items()) {
  const s = it.id === selected;
  const c = s ? '#ffd23f' : isDraft(it) ? '#ff8c1a' : '#3fd5ff';
  path(it.points, c, s ? 2.6 : 1.8, isDraft(it) && !s ? [7, 4] : null); handles(it.points, c); dot(it.dot, s);
 }
 if (draft) {
  const pts = cursor && !gesture ? [...draft, cursor] : draft;
  path(pts, '#7dff9a', 2, [6, 5]); handles(draft, '#7dff9a');
 }
 $('#zoomLevel').textContent = `${Math.round(scale / fitScale * 100)}%`;
}
function update() {
 let html = '', last = '';
 Q.forEach((x, i) => {
  if (x.photo !== last) { html += `<div class="photo">${esc(x.photo.replace('.jpg', '').replaceAll('_', ' '))}</div>`; last = x.photo; }
  const st = state.questions[x.key], mine = state.items.filter(it => it.q === x.key), d = mine.filter(isDraft).length;
  html += `<button data-i="${i}" class="${i === qi ? 'active' : ''} ${st.done ? 'done' : ''} ${st.dropped ? 'dropped' : ''}">${esc(x.title)}<small>${st.done ? '✓ ' : ''}${d ? `${d} to check · ` : ''}${mine.length - d} kept</small></button>`;
 });
 $('#questions').innerHTML = html;
 $('#questions').querySelectorAll('button').forEach(b => b.onclick = () => go(Number(b.dataset.i)));
 const x = q(), st = state.questions[x.key], list = items();
 $('#qPhoto').textContent = `${x.photo} · ${x.kind === 'span' ? 'SPAN: 2 clicks' : 'LINE: along the middle'}`;
 $('#qTitle').textContent = x.title; $('#qHow').textContent = x.how; $('#qNote').textContent = x.note;
 $('#qPrompt').textContent = seed.prompt.replace('{part}', x.part).replace(/\{\{/g, '{').replace(/\}\}/g, '}');
 const nd = list.filter(isDraft).length;
 $('#qCount').textContent = list.length - nd; $('#qSuggest').textContent = `kept${nd ? ` · ${nd} drafts to check` : ''} · about ${x.suggest} wanted (fewer is fine; only clear ones)`;
 $('#items').innerHTML = list.map(it => `<li data-id="${it.id}" class="${it.id === selected ? 'sel' : ''}">${isDraft(it) ? '<b style="color:#e07000">draft</b> · ' : ''}${describe(it)}${it.dot_moved ? ' <small>· dot moved</small>' : ''}</li>`).join('');
 $('#items').querySelectorAll('li').forEach(li => li.onclick = () => { selected = li.dataset.id; focusOn(sel()); update(); });
 $('#finish').hidden = x.kind !== 'line'; $('#finish').disabled = !draft || draft.length < 2;
 $('#keep').disabled = !sel() || !isDraft(sel()); $('#nextDraft').disabled = !nd; $('#cancel').disabled = !draft;
 $('#delete').disabled = !sel(); $('#resetDot').disabled = !sel()?.dot_moved;
 $('#undo').disabled = !undos.length; $('#redo').disabled = !redos.length;
 $('#done').checked = st.done; $('#dropped').checked = st.dropped;
 if (document.activeElement !== $('#qNotes')) $('#qNotes').value = st.notes;
 $('#hint').textContent = draft ? (x.kind === 'span' ? 'Click the second point.' : `${draft.length} point(s): keep clicking along the middle; Enter to finish.`)
  : sel() && isDraft(sel()) ? 'Draft: drag its points or dot where they belong, then Enter to keep it; Delete if it is wrong or unclear.'
  : sel() ? 'Drag its points or its red dot; Delete removes it.' : nd ? 'J: next draft to check. Or click to add one yourself.' : 'Zoom in, then click the first point.';
 draw();
}
function go(i) {
 if (draft) { draft = null; }
 qi = (i + Q.length) % Q.length; selected = null; const g = ++gen; const src = photo().src;
 const im = images.get(src) || new Image(); images.set(src, im);
 const ready = () => { if (g !== gen) return; const same = image === im; image = im; $('#loading').hidden = true; if (!same) fit();
  const d0 = order().find(isDraft); if (d0) { selected = d0.id; focusOn(d0); } update(); };
 if (im.complete && im.naturalWidth) ready(); else { $('#loading').hidden = false; im.onload = ready; im.src = src; }
 update();
}

// --- clicking -----------------------------------------------------------------
function commit() {
 const x = q(); if (!draft || draft.length < 2) return;
 const pts = draft.map(p => p.map(v => Math.round(v * 10) / 10)); draft = null;
 const id = `${x.key}-${Date.now().toString(36)}`;
 mutate('add', () => state.items.push({id, q: x.key, points: pts, dot: middle(pts), dot_moved: false, status: 'kept', origin: 'human', created_at: now(), edited_at: now()}), {id});
 selected = id; update();
}
function addPoint(p) {
 const W = photo().width, H = photo().height;
 if (p[0] < 0 || p[1] < 0 || p[0] > W || p[1] > H) { toast('Outside the photo.'); return; }
 (draft ||= []).push(p); selected = null;
 if (q().kind === 'span' && draft.length === 2) commit(); else update();
}
function hit(p) {
 let best = null, bd = 10;
 for (const it of items()) {
  const cand = [['dot', -1, it.dot], ...it.points.map((pt, i) => ['point', i, pt])];
  for (const [type, i, pt] of cand) {
   const [x, y] = toScr(pt), d = Math.hypot(x - p.x, y - p.y) - (type === 'dot' && it.id === selected ? 1 : 0);
   if (d < bd) { bd = d; best = {it, type, i}; }
  }
 }
 return best;
}
canvas.addEventListener('pointerdown', ev => {
 ev.preventDefault(); canvas.focus(); canvas.setPointerCapture(ev.pointerId);
 const p = sp(ev);
 if (space || ev.button === 1 || ev.button === 2) { gesture = {type: 'pan', start: p, tx, ty}; return; }
 if (ev.button !== 0) return;
 const h = !draft && hit(p);
 if (h) { selected = h.it.id; gesture = {type: 'drag', h, start: p, before: clone({items: state.items, questions: state.questions}), moved: false}; update(); return; }
 gesture = {type: 'click', start: p, tx, ty};
});
canvas.addEventListener('pointermove', ev => {
 const p = sp(ev); cursor = toImg(p);
 if (!gesture) { if (draft) draw(); return; }
 if (gesture.type === 'click' && Math.hypot(p.x - gesture.start.x, p.y - gesture.start.y) > 5) gesture.type = 'pan';
 if (gesture.type === 'pan') { tx = gesture.tx + p.x - gesture.start.x; ty = gesture.ty + p.y - gesture.start.y; draw(); return; }
 if (gesture.type === 'drag') {
  if (!gesture.moved && Math.hypot(p.x - gesture.start.x, p.y - gesture.start.y) < 3) return;
  gesture.moved = true; const it = gesture.h.it, W = photo().width, H = photo().height;
  const v = [Math.max(0, Math.min(W, cursor[0])), Math.max(0, Math.min(H, cursor[1]))].map(c => Math.round(c * 10) / 10);
  if (gesture.h.type === 'dot') { it.dot = v; it.dot_moved = true; } else { it.points[gesture.h.i] = v; if (!it.dot_moved) it.dot = middle(it.points); }
  draw();
 }
});
canvas.addEventListener('pointerup', ev => {
 const g = gesture; gesture = null; if (!g) return; const p = sp(ev);
 if (g.type === 'click') {
  const t = performance.now();
  if (draft && q().kind === 'line' && t - lastAdd.t < 350 && Math.hypot(p.x - lastAdd.x, p.y - lastAdd.y) < 6) { commit(); return; } // double-click ends a line
  lastAdd = {t, x: p.x, y: p.y}; addPoint(toImg(p)); return;
 }
 if (g.type === 'drag' && g.moved) {
  g.h.it.edited_at = now(); undos.push(g.before); redos.length = 0;
  changed(g.h.type === 'dot' ? 'move dot' : 'move point', {id: g.h.it.id});
 }
});
canvas.addEventListener('pointerleave', () => { cursor = null; draw(); });
canvas.addEventListener('wheel', ev => { ev.preventDefault(); const p = sp(ev); zoom(Math.exp(-Math.max(-200, Math.min(200, ev.deltaY)) * .003), p.x, p.y); }, {passive: false});
canvas.addEventListener('contextmenu', ev => ev.preventDefault());

function order() { return items().slice().sort((a, b) => Math.round(a.dot[1] / 150) - Math.round(b.dot[1] / 150) || a.dot[0] - b.dot[0]); }
function nextDraft() {
 const list = order(), ds = list.filter(isDraft);
 if (!ds.length) { selected = null; update(); toast('No drafts left here. Add any clear one that is missing, then tick Done.'); return; }
 const i = list.findIndex(it => it.id === selected), it = list.slice(i + 1).find(isDraft) || ds[0];
 selected = it.id; focusOn(it); update();
}
function keep() { const it = sel(); if (!it || !isDraft(it)) return; mutate('keep', () => { it.status = 'kept'; it.edited_at = now(); }, {id: it.id}); nextDraft(); }
function del() { const it = sel(); if (!it) return; const was = isDraft(it); mutate(was ? 'delete draft' : 'delete', () => { state.items = state.items.filter(x => x.id !== it.id); }, {id: it.id}); if (was) nextDraft(); else { selected = null; update(); } }
document.addEventListener('keydown', ev => {
 if (ev.target.tagName === 'TEXTAREA' || ev.target.tagName === 'INPUT') return;
 const k = ev.key.toLowerCase();
 if ((ev.ctrlKey || ev.metaKey) && k === 'z') { ev.preventDefault(); history(ev.shiftKey); return; }
 if (k === ' ') { space = true; ev.preventDefault(); return; }
 if (k === 'enter') { ev.preventDefault(); if (draft) commit(); else keep(); }
 else if (k === 'j') nextDraft();
 else if (k === 'escape') { draft = null; selected = null; update(); }
 else if (k === 'backspace' && draft) { ev.preventDefault(); draft.pop(); if (!draft.length) draft = null; update(); }
 else if ((k === 'delete' || k === 'backspace') && sel()) { ev.preventDefault(); del(); }
 else if (k === 'f') fit();
 else if (k === 'n') go(qi + 1);
 else if (k === 'p') go(qi - 1);
 else if (k === '+' || k === '=') zoom(1.4);
 else if (k === '-') zoom(1 / 1.4);
});
document.addEventListener('keyup', ev => { if (ev.key === ' ') space = false; });
$('#finish').onclick = commit; $('#keep').onclick = keep; $('#nextDraft').onclick = nextDraft; $('#cancel').onclick = () => { draft = null; update(); };
$('#delete').onclick = del; $('#undo').onclick = () => history(false); $('#redo').onclick = () => history(true);
$('#resetDot').onclick = () => { const it = sel(); if (it) mutate('dot to middle', () => { it.dot = middle(it.points); it.dot_moved = false; }, {id: it.id}); };
$('#zoomIn').onclick = () => zoom(1.4); $('#zoomOut').onclick = () => zoom(1 / 1.4); $('#fit').onclick = fit; $('#others').onchange = draw;
$('#done').onchange = e => { if (e.target.checked && drafts().length) toast(`${drafts().length} drafts still to check here: only kept ones count.`);
 mutate(e.target.checked ? 'done' : 'not done', () => { state.questions[q().key].done = e.target.checked; }); };
$('#dropped').onchange = e => mutate(e.target.checked ? 'dropped' : 'kept', () => { state.questions[q().key].dropped = e.target.checked; });
$('#qNotes').oninput = e => { state.questions[q().key].notes = e.target.value; state.updated_at = now(); clearTimeout(saveTimer); saveTimer = setTimeout(push, 800); };

function toast(s) { $('#toast').textContent = s; $('#toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').hidden = true, 3000); }
$('#export').onclick = () => {
 const blob = new Blob([JSON.stringify(state, null, 1)], {type: 'application/json'});
 const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
 a.download = `segbench-measure-clicks-${seed.questions_sha256.slice(0, 8)}-${now().replace(/[:.]/g, '-')}.json`; a.click(); URL.revokeObjectURL(a.href);
};
$('#import').onclick = () => $('#file').click();
$('#file').onchange = async e => {
 const f = e.target.files[0]; if (!f) return;
 try { const d = valid(JSON.parse(await f.text())); mutate('import', () => { state.items = d.items; state.questions = d.questions; }); toast('Imported.'); }
 catch (err) { toast(err.message); }
 e.target.value = '';
};

// --- start: the copy on lambda wins; else this browser's -----------------------
(async () => {
 $('#source').textContent = `Questions ${seed.questions_sha256.slice(0, 12)}.`;
 let localEnv = null; try { localEnv = JSON.parse(localStorage.getItem(storageKey) || 'null'); } catch (e) {}
 try {
  const r = await fetch('api/state', {cache: 'no-store'}); if (!r.ok) throw Error();
  const j = await r.json(); server = true; revision = j.revision;
  if (j.data) state = valid(j.data);
  else if (localEnv?.data) { state = valid(localEnv.data); push(); }
 } catch (e) { server = false; if (localEnv?.data) try { state = valid(localEnv.data); } catch (err) {} }
 showStatus(); go(0);
})();
})();
