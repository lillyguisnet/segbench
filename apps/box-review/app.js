'use strict';
(() => {
const seed = window.BOX_SEED;
const $ = s => document.querySelector(s);
const clone = x => JSON.parse(JSON.stringify(x));
const now = () => new Date().toISOString();
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const storageKey = `segbench-box-review-v1:${seed.source_sha256}`;
const tabId = Math.random().toString(36).slice(2);
const drafts = new Map(seed.tasks.flatMap(t => t.objects.map(o => [o.id, o])));
const imageCache = new Map(), undos = {}, redos = {};
let taskIndex = 0, selected = null, mode = 'select', space = false, peek = false;
let scale = 1, fitScale = 1, tx = 0, ty = 0, cw = 0, ch = 0, image = null, gen = 0;
let gesture = null, lastHandle = null, saveTimer, saveFailed = false, conflict = false, revision = 0, lastExport = null, toastTimer;
const canvas = $('#canvas'), ctx = canvas.getContext('2d');
const HANDLES = ['nw','n','ne','e','se','s','sw','w'];

function defaultState() {
 return {schema:'segbench-box-review/v1',source_sha256:seed.source_sha256,source_file:seed.source,
  coordinate_system:'original upright photo pixels, box = [x1, y1, x2, y2]',
  purpose:'answer-key boxes for the box benchmark: smallest rectangle around the visible part of each object',
  created_at:now(),updated_at:now(),
  tasks:Object.fromEntries(seed.tasks.map(t=>[t.key,{width:t.width,height:t.height,image_sha256:t.image_sha256,full_photo_checked:false,notes:'',
   objects:t.objects.map(o=>({id:o.id,box:o.box?o.box.slice():null,status:'pending',label:o.label??null,origin:'sam-draft',edited_at:null}))}])),
  audit:[]};
}
function validate(v) {
 if(!v||v.schema!=='segbench-box-review/v1'||v.source_sha256!==seed.source_sha256)throw Error('This file belongs to another draft or is not a box review export.');
 for(const t of seed.tasks){
  const s=v.tasks?.[t.key];
  if(!s||s.width!==t.width||s.height!==t.height||s.image_sha256!==t.image_sha256||!Array.isArray(s.objects))throw Error(`Invalid ${t.key}.`);
  const ids=new Set(s.objects.map(o=>o.id));
  if(t.objects.some(o=>!ids.has(o.id)))throw Error(`Missing drafted boxes in ${t.key}.`);
  for(const o of s.objects){
   if(!['pending','accepted','rejected'].includes(o.status))throw Error(`Invalid status in ${t.key}.`);
   if(o.box!==null&&!(Array.isArray(o.box)&&o.box.length===4&&o.box.every(Number.isFinite)))throw Error(`Invalid box in ${t.key}.`);
  }
 }
 return v;
}
let state = defaultState();
try{const raw=localStorage.getItem(storageKey);if(raw){const env=JSON.parse(raw);state=validate(env.data);revision=env.revision||0;}}catch(e){saveFailed=true;}
const t = () => seed.tasks[taskIndex];
const ts = () => state.tasks[t().key];
const sel = () => ts().objects.find(o=>o.id===selected);
function toast(s){$('#toast').textContent=s;$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').hidden=true,3000);}
function setSaveMessage(){$('#saveStatus').textContent=conflict?'Another tab changed this draft. Export before reloading.':saveFailed?'Not autosaved — export a backup':`Saved in this browser${lastExport?' · exported':''}`;$('#saveStatus').style.color=saveFailed||conflict?'var(--bad)':'';}
function save(){clearTimeout(saveTimer);if(conflict){setSaveMessage();return;}
 try{const prev=localStorage.getItem(storageKey);if(prev){const env=JSON.parse(prev);if(env.revision>revision&&env.tab!==tabId){conflict=true;setSaveMessage();return;}}
  revision++;localStorage.setItem(storageKey,JSON.stringify({revision,tab:tabId,data:state}));saveFailed=false;}catch(e){saveFailed=true;}
 setSaveMessage();}
function dirty(){state.updated_at=now();lastExport=null;$('#saveStatus').textContent='Saving…';clearTimeout(saveTimer);saveTimer=setTimeout(save,200);}
window.addEventListener('pagehide',save);
window.addEventListener('beforeunload',e=>{save();if(saveFailed||conflict){e.preventDefault();e.returnValue='';}});
window.addEventListener('storage',e=>{if(e.key===storageKey&&e.newValue){const env=JSON.parse(e.newValue);if(env.tab!==tabId&&env.revision>revision){conflict=true;setSaveMessage();}}});
function finish(before,action,details={}){
 if(JSON.stringify(before)===JSON.stringify(ts()))return;
 (undos[t().key]||=[]).push(before);if(undos[t().key].length>80)undos[t().key].shift();redos[t().key]=[];
 ts().full_photo_checked=false;state.audit.push({at:now(),task:t().key,action,...details});dirty();update();
}
function mutate(action,fn,details){const before=clone(ts());fn();finish(before,action,details);}
function history(redo){const from=(redo?redos:undos)[t().key]||[],to=(redo?undos:redos)[t().key]||=[];if(!from.length)return;to.push(clone(ts()));state.tasks[t().key]=from.pop();state.audit.push({at:now(),task:t().key,action:redo?'redo':'undo'});dirty();update();}
function norm(b){return [Math.min(b[0],b[2]),Math.min(b[1],b[3]),Math.max(b[0],b[2]),Math.max(b[1],b[3])];}
function clampBox(b){const W=t().width,H=t().height;return norm(b).map((v,i)=>Math.round(Math.max(0,Math.min(i%2?H:W,v))));}
function order(){return ts().objects.slice().sort((a,b)=>{const ra=drafts.get(a.id),rb=drafts.get(b.id);const ay=ra?ra.y:a.box[1],by=rb?rb.y:b.box[1],ax=ra?ra.x:a.box[0],bx=rb?rb.x:b.box[0];return Math.round(ay/150)-Math.round(by/150)||ax-bx;});}
function focus(o){if(!o.box)return;const [x1,y1,x2,y2]=o.box,w=Math.max(40,x2-x1),h=Math.max(40,y2-y1);
 scale=Math.max(fitScale,Math.min(fitScale*30,Math.min(cw/(w*2.6),ch/(h*2.6))));tx=cw/2-(x1+x2)/2*scale;ty=ch/2-(y1+y2)/2*scale;draw();}
function sureness(o){const d=drafts.get(o.id);if(!d)return 2;if(d.source&&d.source!=='models')return -1;return d.agreement??1;}
function goUnsure(){canvas.focus({preventScroll:true});const pend=ts().objects.filter(o=>o.status==='pending').sort((a,b)=>sureness(a)-sureness(b));
 if(!pend.length){selected=null;update();toast('Every drafted box is checked.');return;}const o=pend[0];selected=o.id;lastHandle=null;focus(o);update();}
function go(step){
 canvas.focus({preventScroll:true});
 const list=order();
 if(step==='next'){
  const pend=list.filter(o=>o.status==='pending');
  if(!pend.length){selected=null;update();toast('Every drafted box is checked. Look over the whole photo for missing objects.');return;}
  const i=list.findIndex(o=>o.id===selected);
  const after=list.slice(i+1).find(o=>o.status==='pending')||pend[0];selected=after.id;lastHandle=null;focus(after);update();return;
 }
 const i=list.findIndex(o=>o.id===selected);const o=list[i<=0?list.length-1:i-1];if(o){selected=o.id;lastHandle=null;focus(o);update();}
}
function decide(status){const o=sel();if(!o)return;if(status==='accepted'&&(!o.box||o.box[2]-o.box[0]<1||o.box[3]-o.box[1]<1)){toast('This box is empty: draw it first.');return;}
 mutate(status,()=>{o.status=status;o.edited_at=now();},{id:o.id});go('next');}
function changeLabel(l){const o=sel();if(!o||t().key!=='dishes')return;mutate('label',()=>{o.label=l;o.edited_at=now();},{id:o.id,label:l});}
function setMode(m){mode=m;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));canvas.style.cursor=m==='pan'?'grab':m==='add'?'crosshair':'default';}
function fit(){if(!cw)return;fitScale=Math.min(cw/t().width,ch/t().height)*.95;scale=fitScale;tx=(cw-t().width*scale)/2;ty=(ch-t().height*scale)/2;draw();}
function zoom(f,x=cw/2,y=ch/2){const ns=Math.max(fitScale*.5,Math.min(fitScale*60,scale*f));tx=x-(x-tx)*ns/scale;ty=y-(y-ty)*ns/scale;scale=ns;draw();}
function resize(){const r=$('#viewport').getBoundingClientRect(),ow=cw,oh=ch;cw=r.width;ch=r.height;const d=devicePixelRatio||1;canvas.width=Math.round(cw*d);canvas.height=Math.round(ch*d);ctx.setTransform(d,0,0,d,0,0);if(!ow)fit();else{tx+=(cw-ow)/2;ty+=(ch-oh)/2;fitScale=Math.min(cw/t().width,ch/t().height)*.95;draw();}}
new ResizeObserver(resize).observe($('#viewport'));
const sp=ev=>{const r=canvas.getBoundingClientRect();return {x:ev.clientX-r.left,y:ev.clientY-r.top};};
const pp=p=>({x:(p.x-tx)/scale,y:(p.y-ty)/scale});
const S=b=>[b[0]*scale+tx,b[1]*scale+ty,b[2]*scale+tx,b[3]*scale+ty];
function handlePoints(b){const [x1,y1,x2,y2]=S(b),mx=(x1+x2)/2,my=(y1+y2)/2;return {nw:[x1,y1],n:[mx,y1],ne:[x2,y1],e:[x2,my],se:[x2,y2],s:[mx,y2],sw:[x1,y2],w:[x1,my]};}
function hitHandle(p,o){if(!o?.box)return null;const hp=handlePoints(o.box);let best=null,bd=11;for(const h of HANDLES){const d=Math.hypot(hp[h][0]-p.x,hp[h][1]-p.y);if(d<bd){bd=d;best=h;}}
 if(best)return best;const [x1,y1,x2,y2]=S(o.box);const near=7;
 if(p.y>y1&&p.y<y2){if(Math.abs(p.x-x1)<near)return 'w';if(Math.abs(p.x-x2)<near)return 'e';}
 if(p.x>x1&&p.x<x2){if(Math.abs(p.y-y1)<near)return 'n';if(Math.abs(p.y-y2)<near)return 's';}
 return null;}
function hitBox(p){const q=pp(p);let best=null,ba=Infinity;for(const o of ts().objects){if(!o.box||(o.status==='rejected'&&o.id!==selected))continue;const [x1,y1,x2,y2]=o.box;if(q.x>=x1&&q.x<=x2&&q.y>=y1&&q.y<=y2){const a=(x2-x1)*(y2-y1);if(a<ba){ba=a;best=o;}}}return best;}
function applyHandle(b,h,dx,dy){b=b.slice();if(h==='move'){b[0]+=dx;b[2]+=dx;b[1]+=dy;b[3]+=dy;return b;}
 if(h.includes('w'))b[0]+=dx;if(h.includes('e'))b[2]+=dx;if(h.includes('n'))b[1]+=dy;if(h.includes('s'))b[3]+=dy;return b;}
function draw(){
 ctx.clearRect(0,0,cw,ch);
 if(image?.complete&&image.naturalWidth)ctx.drawImage(image,tx,ty,t().width*scale,t().height*scale);
 if(peek||!$('#overlay').checked){$('#zoomLevel').textContent=`${Math.round(scale/fitScale*100)}%`;return;}
 const showOthers=$('#others').checked||!selected;
 for(const o of ts().objects){
  if(o.id!==selected&&!showOthers)continue;
  const d=drafts.get(o.id);
  if(o.box){const [x1,y1,x2,y2]=S(o.box);
   const color=o.status==='accepted'?'#4de697':o.status==='rejected'?'#9a9a9a':'#ffc25a';
   ctx.save();if(o.status==='rejected')ctx.setLineDash([5,4]);
   ctx.lineWidth=o.id===selected?3:1.6;ctx.strokeStyle='#111';ctx.strokeRect(x1-.5,y1-.5,x2-x1+1,y2-y1+1);
   ctx.strokeStyle=o.id===selected?'#fff':color;ctx.lineWidth=o.id===selected?2:1.4;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
   if(o.id===selected){ctx.strokeStyle=color;ctx.setLineDash([6,6]);ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);
    const hp=handlePoints(o.box);for(const h of HANDLES){ctx.fillStyle=h===lastHandle?'#ff4d6d':'#fff';ctx.strokeStyle='#111';ctx.lineWidth=1;ctx.fillRect(hp[h][0]-4,hp[h][1]-4,8,8);ctx.strokeRect(hp[h][0]-4,hp[h][1]-4,8,8);}}
   if(t().key==='dishes'){ctx.font='bold 11px system-ui';ctx.fillStyle='#111';ctx.fillRect(x1,y1-14,14,14);ctx.fillStyle=o.label==='dirty'?'#ffaaaa':o.label==='clean'?'#a8d5ff':'#ffe1a3';ctx.fillText(o.label==='dirty'?'D':o.label==='clean'?'C':'?',x1+3,y1-3);}
   ctx.restore();}
  if(d){const x=d.x*scale+tx,y=d.y*scale+ty;ctx.fillStyle='#111';ctx.beginPath();ctx.arc(x,y,o.id===selected?5:3.5,0,7);ctx.fill();ctx.fillStyle=o.id===selected?'#fff':'#4de697';ctx.beginPath();ctx.arc(x,y,o.id===selected?3.5:2.2,0,7);ctx.fill();}
 }
 if(gesture?.type==='draw'){const b=S(norm([gesture.a.x,gesture.a.y,gesture.b.x,gesture.b.y]));ctx.strokeStyle='#4de697';ctx.lineWidth=2;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}
 $('#zoomLevel').textContent=`${Math.round(scale/fitScale*100)}%`;
}
function update(){
 $('#tasks').innerHTML=seed.tasks.map((x,i)=>{const s=state.tasks[x.key],k=s.objects.filter(o=>o.status==='accepted').length,p=s.objects.filter(o=>o.status==='pending').length;
  return `<button class="${i===taskIndex?'active':''}" data-task="${i}">${esc(x.title)}<small>${s.full_photo_checked?'✓ photo checked':p?`${p} to check`:`${k} kept`}</small></button>`;}).join('');
 $('#tasks').querySelectorAll('button').forEach(b=>b.onclick=()=>switchTask(Number(b.dataset.task)));
 $('#taskTitle').textContent=t().title;$('#target').textContent=`Box ${t().target}: the visible part only.`;
 const objs=ts().objects;
 $('#counts').innerHTML=[['Kept',objs.filter(o=>o.status==='accepted').length],['To check',objs.filter(o=>o.status==='pending').length],['Removed',objs.filter(o=>o.status==='rejected').length]].map(([a,b])=>`<div class="stat"><b>${b}</b>${a}</div>`).join('');
 const o=sel(),d=o?drafts.get(o.id):null;
 $('#selection').innerHTML=o?`<b>${o.status==='accepted'?'Kept by you':o.status==='rejected'?'Removed':'Not checked yet'}${o.origin==='human'?' · drawn by you':''}</b><small>${o.box?`${o.box[2]-o.box[0]} × ${o.box[3]-o.box[1]} px at ${o.box[0]}, ${o.box[1]}`:'no box'}${d?`<br>${d.source==='models'?`${d.support} model answers agree (median overlap ${Math.round(d.agreement*100)}%)`:d.source?'<span style="color:var(--warn)">Few model boxes: SAM draft</span>':`SAM confidence ${d.sam_score}`}`:''}</small>`:`<b>Press N to start, or U for the least sure first</b><small>Each amber box is a draft around one of your dots.</small>`;
 ['accept','reject','reset'].forEach(k=>$('#'+k).disabled=!o);$('#reset').disabled=!o||!d;
 $('#labels').hidden=t().key!=='dishes';
 document.querySelectorAll('[data-label]').forEach(b=>{b.disabled=!o;b.classList.toggle('active',o?.label===b.dataset.label);});
 $('#reviewed').checked=ts().full_photo_checked;if(document.activeElement!==$('#notes'))$('#notes').value=ts().notes;
 $('#undo').disabled=!undos[t().key]?.length;$('#redo').disabled=!redos[t().key]?.length;
 setMode(mode);draw();
}
function switchTask(i){gesture=null;taskIndex=i;selected=null;lastHandle=null;image=null;const g=++gen;$('#loading').hidden=false;
 const im=imageCache.get(t().photo)||new Image();imageCache.set(t().photo,im);
 const ready=()=>{if(g!==gen)return;image=im;$('#loading').hidden=true;fit();update();};
 im.onload=ready;if(im.complete&&im.naturalWidth)ready();else im.src=t().photo;update();}
canvas.addEventListener('pointerdown',ev=>{
 if(ev.button!==0&&ev.button!==1)return;ev.preventDefault();canvas.focus();canvas.setPointerCapture(ev.pointerId);
 const p=sp(ev),q=pp(p);
 if(space||mode==='pan'||ev.button===1){gesture={type:'pan',start:p,tx,ty};return;}
 if(mode==='add'){gesture={type:'draw',a:q,b:q};return;}
 const h=hitHandle(p,sel());
 if(h){gesture={type:'edit',handle:h,before:clone(ts()),start:q,box:sel().box.slice()};lastHandle=h;draw();return;}
 const o=hitBox(p);
 if(o){if(o.id!==selected){selected=o.id;lastHandle=null;update();}gesture={type:'edit',handle:'move',before:clone(ts()),start:q,box:o.box.slice(),moved:false};return;}
 selected=null;lastHandle=null;gesture={type:'pan',start:p,tx,ty};update();
});
canvas.addEventListener('pointermove',ev=>{
 if(!gesture)return;const p=sp(ev),q=pp(p);
 if(gesture.type==='pan'){tx=gesture.tx+p.x-gesture.start.x;ty=gesture.ty+p.y-gesture.start.y;draw();return;}
 if(gesture.type==='draw'){gesture.b=q;draw();return;}
 if(gesture.type==='edit'){const dx=q.x-gesture.start.x,dy=q.y-gesture.start.y;if(gesture.handle==='move'&&Math.hypot(dx,dy)*scale<4&&!gesture.moved)return;gesture.moved=true;
  sel().box=clampBox(applyHandle(gesture.box,gesture.handle,dx,dy));draw();}
});
function end(){if(!gesture)return;const g=gesture;gesture=null;
 if(g.type==='draw'){const b=clampBox([g.a.x,g.a.y,g.b.x,g.b.y]);if(b[2]-b[0]<3||b[3]-b[1]<3){draw();return;}
  const id=`${t().key}-human-${Date.now().toString(36)}`;
  mutate('add',()=>{ts().objects.push({id,box:b,status:'accepted',label:t().key==='dishes'?'unsure':null,origin:'human',edited_at:now()});selected=id;},{id});
  setMode('select');toast(t().key==='dishes'?'Added. Choose dirty, clean or unsure.':'Box added and kept.');return;}
 if(g.type==='edit'&&g.before){const o=sel();if(o){o.edited_at=now();}finish(g.before,g.handle==='move'?'move':'resize',{id:selected,handle:g.handle});}
}
canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',()=>{if(gesture?.before){state.tasks[t().key]=gesture.before;}gesture=null;update();});
canvas.addEventListener('wheel',ev=>{ev.preventDefault();const p=sp(ev);zoom(Math.exp(-Math.max(-200,Math.min(200,ev.deltaY))*.003),p.x,p.y);},{passive:false});
canvas.addEventListener('contextmenu',ev=>ev.preventDefault());
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{setMode(b.dataset.mode);canvas.focus();});
$('#zoomIn').onclick=()=>zoom(1.4);$('#zoomOut').onclick=()=>zoom(1/1.4);$('#fit').onclick=fit;
$('#undo').onclick=()=>history(false);$('#redo').onclick=()=>history(true);
$('#accept').onclick=()=>decide('accepted');$('#reject').onclick=()=>decide('rejected');
$('#next').onclick=()=>go('next');$('#unsure').onclick=goUnsure;$('#prev').onclick=()=>go('prev');
$('#reset').onclick=()=>{const o=sel(),d=o&&drafts.get(o.id);if(!d)return;mutate('reset',()=>{o.box=d.box?d.box.slice():null;o.status='pending';o.edited_at=now();},{id:o.id});};
document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>changeLabel(b.dataset.label));
$('#overlay').onchange=draw;$('#others').onchange=draw;
$('#reviewed').onchange=()=>{const before=clone(ts());ts().full_photo_checked=$('#reviewed').checked;(undos[t().key]||=[]).push(before);state.audit.push({at:now(),task:t().key,action:'full_photo_check',checked:ts().full_photo_checked});dirty();update();};
$('#notes').onchange=()=>{const before=clone(ts());ts().notes=$('#notes').value.slice(0,10000);finish(before,'notes');};
let nudgeBefore=null,nudgeTimer;
function nudge(dx,dy){const o=sel();if(!o?.box)return;if(!nudgeBefore)nudgeBefore=clone(ts());o.box=clampBox(applyHandle(o.box,lastHandle||'move',dx,dy));o.edited_at=now();draw();
 clearTimeout(nudgeTimer);nudgeTimer=setTimeout(()=>{const b=nudgeBefore;nudgeBefore=null;finish(b,'nudge',{id:o.id,handle:lastHandle||'move'});},500);}
window.addEventListener('keydown',ev=>{
 if(['INPUT','TEXTAREA','BUTTON'].includes(document.activeElement?.tagName)&&ev.key!=='Escape')return;
 const k=ev.key.toLowerCase();
 if((ev.ctrlKey||ev.metaKey)&&k==='z'){ev.preventDefault();history(ev.shiftKey);return;}
 if((ev.ctrlKey||ev.metaKey)&&k==='y'){ev.preventDefault();history(true);return;}
 if(ev.ctrlKey||ev.metaKey||ev.altKey)return;
 const step=ev.shiftKey?10:1;
 if(k==='arrowleft'){ev.preventDefault();nudge(-step,0);return;}if(k==='arrowright'){ev.preventDefault();nudge(step,0);return;}
 if(k==='arrowup'){ev.preventDefault();nudge(0,-step);return;}if(k==='arrowdown'){ev.preventDefault();nudge(0,step);return;}
 if(k===' '){ev.preventDefault();space=true;return;}
 if(k==='h'){peek=true;draw();return;}
 if(k==='f'){fit();return;}
 if(k==='escape'){gesture=null;selected=null;lastHandle=null;update();return;}
 if(k==='v')setMode('select');if(k==='a')setMode('add');
 if(k==='n'){ev.preventDefault();go('next');}if(k==='u'){ev.preventDefault();goUnsure();}if(k==='p'){ev.preventDefault();go('prev');}
 if(k==='enter'){ev.preventDefault();decide('accepted');}
 if(k==='delete'||k==='backspace'){ev.preventDefault();decide('rejected');}
 if(k==='d')changeLabel('dirty');if(k==='c')changeLabel('clean');if(k==='s')changeLabel('unsure');
});
window.addEventListener('keyup',ev=>{if(ev.key===' ')space=false;if(ev.key.toLowerCase()==='h'){peek=false;draw();}});
window.addEventListener('blur',()=>{space=false;peek=false;});
$('#export').onclick=()=>{save();const out=clone(state);out.exported_at=now();
 const blob=new Blob([JSON.stringify(out,null,1)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=url;a.download=`segbench-boxes-${seed.source_sha256.slice(0,8)}-${now().replace(/[:.]/g,'-')}.json`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
 lastExport=now();setSaveMessage();toast('Exported. Send me this file when you are done.');};
$('#import').onclick=()=>$('#file').click();
$('#file').onchange=async()=>{const f=$('#file').files[0];$('#file').value='';if(!f)return;
 try{const v=validate(JSON.parse(await f.text()));if(!confirm('Replace this browser’s boxes with the imported file?'))return;state=v;for(const k in undos)delete undos[k];for(const k in redos)delete redos[k];selected=null;dirty();save();update();toast('Restored.');}catch(e){toast(e.message);}};
$('#source').textContent=`Draft: ${seed.source} · ${seed.source_sha256.slice(0,12)}`;
setSaveMessage();switchTask(0);
window.boxTest={getState:()=>clone(state),switchTask,validate};
})();
