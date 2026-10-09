'use strict';
(() => {
const seed = window.REVIEW_SEED;
const $ = s => document.querySelector(s);
const clone = x => JSON.parse(JSON.stringify(x));
const escapeHTML = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const now = () => new Date().toISOString();
const sourceMap = new Map(seed.tasks.map(t => [t.key, t]));
const originals = new Map(seed.tasks.flatMap(t => (t.objects || []).map(o => [o.id, o])));
const storageKey = `segbench-review-v1:${seed.source_sha256}`;
const tabId = Math.random().toString(36).slice(2);
const imageCache = new Map(), undos = {}, redos = {};
let taskIndex = 0, selected = null, mode = 'select', space = false, peek = false, showOriginal = false;
let scale = 1, fitScale = 1, tx = 0, ty = 0, cw = 0, ch = 0, image = null;
let gesture = null, pinch = null, brushMask = null, maskCanvas = null, imageGeneration = 0;
let saveTimer, saveFailed = false, conflict = false, revision = 0, lastExport = null;
let toastTimer;
const pointers = new Map();
const canvas = $('#canvas'), ctx = canvas.getContext('2d');
const regionOffscreen = document.createElement('canvas');
function defaultState() {
 return {schema:'segbench-consensus-review/v1',source_sha256:seed.source_sha256,source_file:seed.source,
  coordinate_system:'original upright photo pixels; region masks have their own explicit grid',
  purpose:'human-reviewed annotation hints; NOT benchmark ground truth',created_at:now(),updated_at:now(),
  tasks:Object.fromEntries(seed.tasks.map(t => [t.key, {kind:t.kind,width:t.width,height:t.height,image_sha256:t.image_sha256,
   full_photo_checked:false,notes:'',...(t.kind==='objects' ? {objects:t.objects.map(o=>({id:o.id,x:o.x,y:o.y,
    status:'pending',label:o.label,origin:'consensus',edited_at:null}))} : {grid:t.grid.slice(),mask:t.mask,mask_edited:false})} ])), audit:[]};
}
let state = defaultState();
function validate(value) {
 if (!value || value.schema !== 'segbench-consensus-review/v1' || value.source_sha256 !== seed.source_sha256)
  throw Error('This file belongs to a different consensus snapshot or is not a review export.');
 if (!value.tasks || !Array.isArray(value.audit) || value.audit.length > 100000) throw Error('Invalid review file.');
 for (const t of seed.tasks) {
  const v = value.tasks[t.key];
  if (!v || v.kind!==t.kind || v.image_sha256!==t.image_sha256 || v.width!==t.width || v.height!==t.height ||
    typeof v.full_photo_checked!=='boolean' || typeof v.notes!=='string' || v.notes.length>10000) throw Error(`Invalid ${t.key} metadata.`);
  if (t.kind==='objects') {
   if (!Array.isArray(v.objects) || v.objects.length>10000) throw Error('Invalid object list.');
   const ids = new Set();
   for (const o of v.objects) {
    const orig = originals.get(o.id);
    const validOrigin = orig ? t.objects.some(s=>s.id===o.id) && o.origin==='consensus' : typeof o.id==='string' && o.id.startsWith(`${t.key}-human-`) && o.origin==='human';
    const inBounds = o.x>=0 && o.x<=t.width && o.y>=0 && o.y<=t.height;
    const untouchedOutside = orig && o.x===orig.x && o.y===orig.y && o.status!=='accepted';
    if (!validOrigin || ids.has(o.id) || !Number.isFinite(o.x) || !Number.isFinite(o.y) || (!inBounds && !untouchedOutside) ||
      !['pending','accepted','rejected','unsure'].includes(o.status) || !(t.key==='dishes'?['dirty','clean','unsure'].includes(o.label):o.label===null))
      throw Error(`Invalid object in ${t.key}.`);
    ids.add(o.id);
   }
   if (t.objects.some(o=>!ids.has(o.id))) throw Error(`Missing original candidates in ${t.key}; reject them rather than deleting records.`);
  } else if (!Array.isArray(v.grid) || v.grid.join()!==t.grid.join() || typeof v.mask!=='string' || v.mask.length!==t.mask.length || /[^01]/.test(v.mask) || typeof v.mask_edited!=='boolean') {
   throw Error(`Invalid region mask for ${t.key}.`);
  }
 }
 return value;
}
try {
 const raw = localStorage.getItem(storageKey);
 if (raw) {const envelope = JSON.parse(raw);state=validate(envelope.data);revision=envelope.revision||0;}
} catch(e) {saveFailed=true; $('#saveStatus').textContent='Browser save unavailable — use Export';}
function t() {return seed.tasks[taskIndex];}
function taskState() {return state.tasks[t().key];}
function getSelected() {return taskState().objects?.find(o=>o.id===selected);}
function toast(text) {$('#toast').textContent=text;$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').hidden=true,3500);}
function setSaveMessage() {
 $('#saveStatus').textContent=conflict?'Another tab changed this draft. Export before reloading.':saveFailed?'Not autosaved — export a backup':`Saved in this browser${lastExport?' · backup exported':''}`;
 $('#saveStatus').style.color=saveFailed||conflict?'var(--bad)':'';
}
function save() {
 clearTimeout(saveTimer);
 if(conflict){setSaveMessage();return;}
 try {
  const previous=localStorage.getItem(storageKey);
  if(previous){const env=JSON.parse(previous);if(env.revision>revision && env.tab!==tabId){conflict=true;setSaveMessage();return;}}
  revision++;
  localStorage.setItem(storageKey,JSON.stringify({revision,tab:tabId,data:state}));saveFailed=false;
 } catch(e){saveFailed=true;}
 setSaveMessage();
}
function dirty() {state.updated_at=now();lastExport=null;$('#saveStatus').textContent='Saving…';clearTimeout(saveTimer);saveTimer=setTimeout(save,200);}
window.addEventListener('storage', e => {if(e.key===storageKey && e.newValue){const env=JSON.parse(e.newValue);if(env.tab!==tabId&&env.revision>revision){conflict=true;setSaveMessage();toast('This draft changed in another tab. Export here before reloading.');}}});
window.addEventListener('pagehide',save);
window.addEventListener('beforeunload', e=>{save();if(saveFailed||conflict){e.preventDefault();e.returnValue='';}});
function audit(action,details={}) {state.audit.push({at:now(),task:t().key,action,...details});}
function finishChange(before,action,details={}) {
 if(JSON.stringify(before)===JSON.stringify(taskState()))return;
 (undos[t().key] ||= []).push(before);if(undos[t().key].length>60)undos[t().key].shift();redos[t().key]=[];
 taskState().full_photo_checked=false;audit(action,details);dirty();update();
}
function mutate(action,fn,details={}) {const before=clone(taskState());fn();finishChange(before,action,details);}
function history(redo) {
 const from=(redo?redos:undos)[t().key]||[],to=(redo?undos:redos)[t().key]||=[];
 if(!from.length)return;
 to.push(clone(taskState()));state.tasks[t().key]=from.pop();audit(redo?'redo':'undo');brushMask=null;maskCanvas=null;dirty();update();
}
function candidateVisible(o) {
 if(o.status==='accepted'||o.status==='unsure')return true;
 if(o.status==='rejected')return $('#weak').checked||o.id===selected;
 return $('#weak').checked||o.id===selected||(originals.get(o.id)?.support??1)>=.2;
}
function pendingObjects() {return taskState().objects.filter(o=>o.status==='pending'&&(originals.get(o.id)?.support??1)>=.2);}
function uncertain(o) {
 const src=originals.get(o.id);return o.status==='unsure'||(o.status==='pending'&&src&&((src.support>=.2&&src.support<.8)||(src.label_support>.25&&src.label_support<.75)));
}
function next(onlyUncertain=false, advance=true) {
 canvas.focus({preventScroll:true});
 const list=taskState().objects.filter(o=>onlyUncertain?uncertain(o):o.status==='pending'&&candidateVisible(o));
 list.sort((a,b)=>a.y-b.y||a.x-b.x);
 if(!list.length){selected=null;update();toast(onlyUncertain?'No uncertain candidates left. Still check the entire photo.':'No visible unreviewed dots. Check hidden suggestions and missed objects.');return;}
 const i=list.findIndex(o=>o.id===selected);
 const o=list[i<0?0:advance?(i+1)%list.length:i];selected=o.id;focusObject(o);update();
}
function decide(status) {
 const o=getSelected();if(!o)return;
 if(status==='accepted'&&!inPhoto(o)){toast('This suggestion is outside the photo. Move it onto the object first.');return;}
 mutate(status,()=>{o.status=status;o.edited_at=now();},{id:o.id});
 if(status!=='unsure')next(false);else update();
}
function changeLabel(label) {const o=getSelected();if(!o||t().key!=='dishes')return;mutate('label',()=>{o.label=label;o.edited_at=now();},{id:o.id,label});canvas.focus({preventScroll:true});}
function focusObject(o) {
 scale=Math.max(fitScale, Math.min(fitScale*8, Math.min(cw,ch)/Math.max(t().width,t().height)*6));
 tx=cw/2-o.x*scale;ty=ch/2-o.y*scale;draw();
}
function setMode(m) {mode=m;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));canvas.style.cursor=m==='pan'?'grab':m==='select'?'default':'crosshair';}
function fit() {if(!cw||!ch)return;fitScale=Math.min(cw/t().width,ch/t().height)*.95;scale=fitScale;tx=(cw-t().width*scale)/2;ty=(ch-t().height*scale)/2;draw();}
function zoom(factor,x=cw/2,y=ch/2) {const ns=Math.max(fitScale*.5,Math.min(fitScale*35,scale*factor));tx=x-(x-tx)*ns/scale;ty=y-(y-ty)*ns/scale;scale=ns;draw();}
function resize() {const rect=$('#viewport').getBoundingClientRect();const oldw=cw,oldh=ch;cw=rect.width;ch=rect.height;const dpr=window.devicePixelRatio||1;canvas.width=Math.round(cw*dpr);canvas.height=Math.round(ch*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);if(!oldw){fit();}else{tx+=(cw-oldw)/2;ty+=(ch-oldh)/2;fitScale=Math.min(cw/t().width,ch/t().height)*.95;draw();}}
new ResizeObserver(resize).observe($('#viewport'));
function screenPoint(ev){const r=canvas.getBoundingClientRect();return {x:ev.clientX-r.left,y:ev.clientY-r.top};}
function photoPoint(p){return {x:(p.x-tx)/scale,y:(p.y-ty)/scale};}
function inPhoto(p){return p.x>=0&&p.y>=0&&p.x<=t().width&&p.y<=t().height;}
function hit(p) {
 let result=null,best=18;
 for(const o of taskState().objects||[]){if(!candidateVisible(o))continue;const d=Math.hypot(o.x*scale+tx-p.x,o.y*scale+ty-p.y);if(d<best){best=d;result=o;}}
 return result;
}
function regionCanvas() {
 if(maskCanvas)return maskCanvas;
 const src=showOriginal?t().mask:brushMask||taskState().mask,[w,h]=t().grid;
 regionOffscreen.width=w;regionOffscreen.height=h;
 const rc=regionOffscreen.getContext('2d'),pixels=rc.createImageData(w,h);
 for(let i=0;i<src.length;i++)if(src[i]==='1'||src[i]===1){pixels.data.set([255,90,60,255],i*4);}
 rc.putImageData(pixels,0,0);maskCanvas=regionOffscreen;return maskCanvas;
}
function draw() {
 ctx.clearRect(0,0,cw,ch);
 if(image?.complete&&image.naturalWidth)ctx.drawImage(image,tx,ty,t().width*scale,t().height*scale);
 if(!peek&&$('#overlay').checked){
  if(t().kind==='region') {
   ctx.save();ctx.globalAlpha=Number($('#opacity').value)/100;ctx.imageSmoothingEnabled=false;ctx.drawImage(regionCanvas(),tx,ty,t().width*scale,t().height*scale);ctx.restore();
  } else {
   for(const o of taskState().objects){
    if(!candidateVisible(o))continue;
    const x=o.x*scale+tx,y=o.y*scale+ty;if(x<-20||x>cw+20||y<-20||y>ch+20)continue;
    const src=originals.get(o.id),low=o.status==='rejected'||(o.status==='pending'&&(src?.support??1)<.2);
    const color=low?'#ababab':o.status==='accepted'?'#4de697':'#ffc25a';
    ctx.save();ctx.globalAlpha=low?.65:1;ctx.lineWidth=o.id===selected?3:2;
    ctx.strokeStyle='#151515';ctx.beginPath();ctx.arc(x,y,o.id===selected?11:8,0,Math.PI*2);ctx.stroke();
    ctx.strokeStyle=color;if(o.status==='unsure')ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.arc(x,y,o.id===selected?10:7,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
    if(o.status==='accepted'||o.origin==='human'){ctx.fillStyle=t().key==='dishes'?(o.label==='dirty'?'#ff7272':o.label==='clean'?'#6bbdff':'#ffc25a'):color;ctx.beginPath();ctx.arc(x,y,3.5,0,Math.PI*2);ctx.fill();}
    if(o.status==='rejected'){ctx.beginPath();ctx.moveTo(x-5,y-5);ctx.lineTo(x+5,y+5);ctx.stroke();}
    if(o.id===selected){ctx.strokeStyle='#fff';ctx.beginPath();ctx.arc(x,y,14,0,Math.PI*2);ctx.stroke();}
    ctx.restore();
   }
  }
 }
 $('#zoomLevel').textContent=`${Math.round(scale/fitScale*100)}%`;
}
function update() {
 $('#tasks').innerHTML=seed.tasks.map((x,i)=>{const s=state.tasks[x.key],count=s.objects?.filter(o=>o.status==='accepted').length;return `<button class="${i===taskIndex?'active':''}" data-task="${i}">${escapeHTML(x.title)}<small>${s.full_photo_checked?'✓ Photo checked':s.kind==='objects'?`${count} kept by you`:'Region draft'}</small></button>`;}).join('');
 $('#tasks').querySelectorAll('button').forEach(b=>b.onclick=()=>switchTask(Number(b.dataset.task)));
 $('#taskTitle').textContent=t().title;$('#target').textContent=`Find ${t().target}.`;
 $('#taskKind').textContent=t().kind==='objects'?'OBJECT REVIEW':'REGION REVIEW';
 const isObject=t().kind==='objects';
 $('#objectTools').hidden=!isObject;$('#objectReview').hidden=!isObject;$('#weakLabel').hidden=!isObject;
 $('#regionTools').hidden=isObject;$('#regionReview').hidden=isObject;
 $('#reviewed').checked=taskState().full_photo_checked;
 if(document.activeElement!==$('#notes'))$('#notes').value=taskState().notes;
 $('#undo').disabled=!(undos[t().key]?.length);$('#redo').disabled=!(redos[t().key]?.length);
 if(isObject){
  const objects=taskState().objects;
  const counts=[['Kept',objects.filter(o=>o.status==='accepted').length],['Unreviewed',pendingObjects().length],['Unsure',objects.filter(o=>o.status==='unsure').length]];
  $('#counts').innerHTML=counts.map(([a,b])=>`<div class="stat"><b>${b}</b>${a}</div>`).join('');
  const o=getSelected(),src=o?originals.get(o.id):null;
  const hiddenCount=objects.filter(o=>o.status==='pending'&&!candidateVisible(o)).length;
  $('#selection').innerHTML=o?`<b>${escapeHTML(o.status==='accepted'?'Kept by you':o.status==='rejected'?'Rejected by you':o.status==='unsure'?'Needs another look':'Not yet reviewed')}</b><small>${src?`${src.votes.length} model answers · agreement index ${src.support.toFixed(2)} (not a calibrated probability)`:'Added by you'}<br>${Math.round(o.x)}, ${Math.round(o.y)} photo pixels</small>`:`<b>Pick a dot or start reviewing</b><small>${hiddenCount} low-support suggestions hidden. Green dots mean you confirmed them—not just that models agreed.</small>`;
  ['accept','reject','unsure'].forEach(k=>$('#'+k).disabled=!o);
  $('#labels').hidden=t().key!=='dishes'||!o;
  document.querySelectorAll('[data-label]').forEach(b=>b.classList.toggle('active',o?.label===b.dataset.label));
  $('#votes').textContent=src?src.votes.join('\n'):'No original model votes for this selection.';
 } else {
  const n=[...taskState().mask].filter(x=>x==='1').length;
  $('#counts').innerHTML=`<div class="stat"><b>${(100*n/taskState().mask.length).toFixed(1)}%</b>Photo area</div><div class="stat"><b>${t().grid[0]}</b>Grid width</div><div class="stat"><b>${t().grid[1]}</b>Grid height</div>`;
  $('#regionInfo').textContent=`Edits use the original ${t().grid.join(' × ')} consensus grid. Zooming doesn't increase mask detail. Red foliage area is relative to the photo, not the tree crown.`;
 }
 setMode(mode);draw();
}
function endGesture(cancel=false){
 if(!gesture)return;
 const g=gesture;gesture=null;
 if(g.type==='tap'&&!cancel){
  if(g.tapMode==='remove'){if(g.id){selected=g.id;decide('rejected');}return;}
  const id=`${t().key}-human-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,8)}`;
  mutate('add',()=>{taskState().objects.push({id,x:Math.round(g.photo.x),y:Math.round(g.photo.y),status:'accepted',label:t().key==='dishes'?'unsure':null,origin:'human',edited_at:now()});selected=id;},{id});
  toast(t().key==='dishes'?'Added. Choose dirty, clean, or unsure.':'Added object.');return;
 }
 if(g.before){
  if(cancel){state.tasks[t().key]=g.before;brushMask=null;maskCanvas=null;update();return;}
  if(g.type==='brush'){taskState().mask=Array.from(brushMask).join('');taskState().mask_edited=true;brushMask=null;maskCanvas=null;}
  finishChange(g.before,g.type==='brush'?mode:'move',{...(g.id?{id:g.id}:{})});
 }
}
function switchTask(i){endGesture();taskIndex=i;selected=null;brushMask=null;maskCanvas=null;image=null;const gen=++imageGeneration;setMode(t().kind==='objects'?'select':'paint');$('#loading').hidden=false;
 const photo=t().photo;
 const im=imageCache.get(photo)||new Image();imageCache.set(photo,im);
 const ready=()=>{if(gen!==imageGeneration)return;image=im;$('#loading').hidden=true;fit();update();};
 im.onload=ready;im.onerror=()=>{if(gen===imageGeneration){$('#loading').hidden=false;$('#loading').textContent='Photo failed to load. Export your work before refreshing.';}};
 if(im.complete&&im.naturalWidth)ready();else im.src=photo;
 update();
}
function brushAt(p,previous){
 const [w,h]=t().grid,fx=w/t().width,fy=h/t().height,r=Number($('#brush').value)/2;
 const x=p.x*fx,y=p.y*fy,px=previous?previous.x*fx:x,py=previous?previous.y*fy:y;
 const steps=Math.max(1,Math.ceil(Math.hypot(x-px,y-py)/Math.max(1,r/3)));
 for(let k=0;k<=steps;k++){
  const cx=px+(x-px)*k/steps,cy=py+(y-py)*k/steps;
  for(let yy=Math.max(0,Math.floor(cy-r));yy<=Math.min(h-1,Math.ceil(cy+r));yy++)
   for(let xx=Math.max(0,Math.floor(cx-r));xx<=Math.min(w-1,Math.ceil(cx+r));xx++)
    if(Math.hypot(xx+.5-cx,yy+.5-cy)<=r)brushMask[yy*w+xx]=mode==='erase'?0:1;
 }
 maskCanvas=null;draw();
}
canvas.addEventListener('pointerdown',ev=>{
 if(ev.button!==0&&ev.button!==1)return;
 ev.preventDefault();canvas.focus();canvas.setPointerCapture(ev.pointerId);
 const p=screenPoint(ev);pointers.set(ev.pointerId,p);
 if(pointers.size===2){endGesture(true);const [a,b]=[...pointers.values()];pinch={distance:Math.max(1,Math.hypot(a.x-b.x,a.y-b.y)),mid:{x:(a.x+b.x)/2,y:(a.y+b.y)/2}};return;}
 if(pointers.size>2)return;
 const pp=photoPoint(p);
 if(space||mode==='pan'||ev.button===1){gesture={type:'pan',start:p,tx,ty};return;}
 if(!inPhoto(pp))return;
 if(t().kind==='region'){
  gesture={type:'brush',before:clone(taskState()),last:pp};brushMask=Uint8Array.from(taskState().mask,Number);brushAt(pp);return;
 }
 const obj=hit(p);
 if(mode==='remove'){gesture={type:'tap',tapMode:mode,id:obj?.id,start:p,photo:pp,tx,ty};return;}
 if(obj){selected=obj.id;gesture={type:'move',before:clone(taskState()),id:obj.id,start:p,x:obj.x,y:obj.y};update();return;}
 if(mode==='add'){gesture={type:'tap',tapMode:mode,start:p,photo:pp,tx,ty};return;}
 selected=null;gesture={type:'pan',start:p,tx,ty};update();
});
canvas.addEventListener('pointermove',ev=>{
 const p=screenPoint(ev);if(pointers.has(ev.pointerId))pointers.set(ev.pointerId,p);
 if(pinch&&pointers.size>=2){const[a,b]=[...pointers.values()],d=Math.max(1,Math.hypot(a.x-b.x,a.y-b.y)),mid={x:(a.x+b.x)/2,y:(a.y+b.y)/2};zoom(d/pinch.distance,pinch.mid.x,pinch.mid.y);tx+=mid.x-pinch.mid.x;ty+=mid.y-pinch.mid.y;pinch={distance:d,mid};draw();return;}
 if(!gesture)return;
 if(gesture.type==='tap'){
  if(Math.hypot(p.x-gesture.start.x,p.y-gesture.start.y)>7)gesture.type='pan';
  else return;
 }
 if(gesture.type==='pan'){tx=gesture.tx+p.x-gesture.start.x;ty=gesture.ty+p.y-gesture.start.y;draw();return;}
 if(gesture.type==='move'){
  if(Math.hypot(p.x-gesture.start.x,p.y-gesture.start.y)<3)return;
  const o=getSelected();o.x=Math.round(Math.max(0,Math.min(t().width,gesture.x+(p.x-gesture.start.x)/scale)));o.y=Math.round(Math.max(0,Math.min(t().height,gesture.y+(p.y-gesture.start.y)/scale)));o.edited_at=now();draw();return;
 }
 const pp=photoPoint(p);brushAt(pp,gesture.last);gesture.last=pp;
});
function pointerEnd(ev,cancel=false){pointers.delete(ev.pointerId);if(pinch){if(pointers.size<2)pinch=null;gesture=null;return;}endGesture(cancel);}
canvas.addEventListener('pointerup',ev=>pointerEnd(ev));
canvas.addEventListener('pointercancel',ev=>pointerEnd(ev,true));
canvas.addEventListener('wheel',ev=>{ev.preventDefault();endGesture();const p=screenPoint(ev);zoom(Math.exp(-Math.max(-200,Math.min(200,ev.deltaY))*.003),p.x,p.y);},{passive:false});
canvas.addEventListener('contextmenu',ev=>ev.preventDefault());
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{endGesture();setMode(b.dataset.mode);canvas.focus({preventScroll:true});});
$('#zoomIn').onclick=()=>zoom(1.4);$('#zoomOut').onclick=()=>zoom(1/1.4);$('#fit').onclick=fit;
$('#undo').onclick=()=>history(false);$('#redo').onclick=()=>history(true);
$('#accept').onclick=()=>decide('accepted');$('#reject').onclick=()=>decide('rejected');$('#unsure').onclick=()=>decide('unsure');
$('#next').onclick=()=>next();$('#nextUncertain').onclick=()=>next(true);
document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>changeLabel(b.dataset.label));
$('#weak').onchange=update;$('#overlay').onchange=draw;$('#opacity').oninput=draw;
$('#brush').oninput=()=>$('#brushSize').textContent=$('#brush').value;
$('#reviewed').onchange=()=>{const checked=$('#reviewed').checked;const before=clone(taskState());taskState().full_photo_checked=checked;(undos[t().key] ||= []).push(before);redos[t().key]=[];audit('full_photo_check',{checked});dirty();update();};
$('#notes').onchange=()=>{const before=clone(taskState());taskState().notes=$('#notes').value.slice(0,10000);finishChange(before,'notes');};
$('#originalRegion').onpointerdown=ev=>{ev.currentTarget.setPointerCapture(ev.pointerId);showOriginal=true;maskCanvas=null;draw();};
['pointerup','pointercancel','lostpointercapture'].forEach(type=>$('#originalRegion').addEventListener(type,()=>{showOriginal=false;maskCanvas=null;draw();}));
window.addEventListener('keydown',ev=>{
 if(['INPUT','TEXTAREA','SELECT','BUTTON'].includes(document.activeElement?.tagName))return;
 const k=ev.key.toLowerCase();
 if((ev.ctrlKey||ev.metaKey)&&k==='z'){ev.preventDefault();history(ev.shiftKey);return;}
 if((ev.ctrlKey||ev.metaKey)&&k==='y'){ev.preventDefault();history(true);return;}
 if(ev.ctrlKey||ev.metaKey||ev.altKey)return;
 if(k===' '){ev.preventDefault();space=true;return;}
 if(k==='h'){peek=true;draw();return;}
 if(k==='f'){fit();return;}
 if(k==='escape'){endGesture(true);selected=null;update();return;}
 if(t().kind!=='objects')return;
 if(k==='v')setMode('select');if(k==='a')setMode('add');if(k==='x')setMode('remove');
 if(k==='n'){ev.preventDefault();next();}if(k==='u'){ev.preventDefault();next(true);}
 if(k==='enter'){ev.preventDefault();decide('accepted');}
 if(k==='delete'||k==='backspace'){ev.preventDefault();decide('rejected');}
 if(k==='?')decide('unsure');
 if(k==='d')changeLabel('dirty');if(k==='c')changeLabel('clean');if(k==='s')changeLabel('unsure');
});
window.addEventListener('keyup',ev=>{if(ev.key===' ')space=false;if(ev.key.toLowerCase()==='h'){peek=false;draw();}});
window.addEventListener('blur',()=>{space=false;peek=false;endGesture();draw();});
function exportDraft(){
 endGesture();$('#notes').blur();save();
 const exported=clone(state);exported.exported_at=now();exported.source_snapshot=seed;
 const blob=new Blob([JSON.stringify(exported,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob);
 const a=document.createElement('a');a.href=url;a.download=`segbench-review-${seed.source_sha256.slice(0,8)}-${now().replace(/[:.]/g,'-')}.json`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
 lastExport=now();setSaveMessage();toast('Backup exported. Keep this JSON file to restore or transfer corrections.');
}
$('#export').onclick=exportDraft;
$('#import').onclick=()=>$('#file').click();
$('#file').onchange=async()=>{
 const file=$('#file').files[0];$('#file').value='';if(!file)return;
 try{
  if(file.size>15000000)throw Error('File is too large (15 MB limit).');
  const imported=validate(JSON.parse(await file.text()));delete imported.source_snapshot;
  if(!confirm('Replace this browser’s corrections with the imported file? Export your current work first if you want to keep both.'))return;
  state=clone(imported);for(const k of Object.keys(undos))delete undos[k];for(const k of Object.keys(redos))delete redos[k];
  selected=null;maskCanvas=null;brushMask=null;audit('import',{file:file.name});dirty();save();update();toast('Corrections restored.');
 }catch(e){toast(e.message);}
};
$('#source').textContent=`Source: ${seed.source} · ${seed.source_sha256.slice(0,12)}. This snapshot does not yet include specialists.`;
setSaveMessage();switchTask(0);
// Small surface for deterministic geometry and import tests; no network APIs.
window.reviewTest={validate,getState:()=>clone(state),getSource:()=>seed,getView:()=>({scale,tx,ty,cw,ch}),switchTask,save};
})();
