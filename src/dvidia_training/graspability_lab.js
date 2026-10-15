/* DVIDIA prerequisite experiment viewer. MIT. Recorded native poses only. */
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const $=id=>document.getElementById(id), MAX=64*1024*1024;
const assert=(ok,message)=>{if(!ok)throw new Error(message);};
const shapeNames={'easy-block':'Small block','simplified-shoe':'Narrow shoe proxy','current-shoe':'Full shoe proxy'};
const loadNames={'lightweight-control':'30 g · light control','reference-load':'300 g · reference load'};
const toolName=t=>t==='mounted'?'Rake mounted':'Rake absent';
let report,task='in-reach-graspability',trial,model,motion,pose,renderer,scene,camera,orbit,bodyGroups=[],visual=[],collision=[],frame=0,playing=false,started=0,queued=false,generation=0,resizeObserver;
const geometryCache=new Map();
function localPath(path){assert(typeof path==='string'&&/^[a-zA-Z0-9_.-]+(?:\/[a-zA-Z0-9_.-]+)*$/.test(path)&&!path.split('/').some(p=>p==='.'||p==='..'),'Invalid export path');return path;}
async function bounded(stream,limit){assert(stream,'Missing response body');const reader=stream.getReader(),chunks=[];let length=0;try{for(;;){const {done,value}=await reader.read();if(done)break;length+=value.byteLength;assert(length<=limit,'Asset exceeds byte limit');chunks.push(value);}}finally{await reader.cancel().catch(()=>{});reader.releaseLock();}const out=new Uint8Array(length);let offset=0;for(const c of chunks){out.set(c,offset);offset+=c.length;}return out.buffer;}
async function resource(path,limit){
  const local=new URL(localPath(path),location.href), hosted=location.hostname==='dvidia-robot-simulation-lab.static.hf.space';
  const urls=hosted&&path.endsWith('.bin.gz')?[new URL('https://research.dvidia.org/downloads/topics/physical-grounding/runs/2026-10-14-shoe-graspability-v03/lab/'+path),local]:[local];
  let last;
  for(const [i,url] of urls.entries()){
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),i===0&&urls.length>1?15000:45000);
    try{
      const r=await fetch(url,{signal:controller.signal,credentials:'omit',referrerPolicy:'no-referrer',redirect:hosted?'follow':'error'});
      const final=new URL(r.url);
      assert(final.origin===url.origin||(hosted&&final.protocol==='https:'&&(final.hostname.endsWith('.hf.co')||final.hostname==='huggingface.co')),'Unexpected asset redirect');
      assert(r.ok,'HTTP '+r.status+' loading '+path);
      return await bounded(r.body,limit);
    }catch(error){last=error;}finally{clearTimeout(timeout);}
  }
  throw new Error('Could not load '+path+'. '+last.message);
}
async function json(path){return JSON.parse(new TextDecoder().decode(await resource(path,8*1024*1024)));}
async function binary(meta){
  assert(meta?.compression==='gzip'&&Number.isInteger(meta.bytes)&&meta.bytes>0&&meta.bytes<=MAX&&Number.isInteger(meta.compressed_bytes)&&meta.compressed_bytes>0&&meta.compressed_bytes<=MAX,'Invalid binary descriptor');
  const bytes=await resource(meta.href,meta.compressed_bytes);assert(bytes.byteLength===meta.compressed_bytes,'Compressed length mismatch');
  const unpacked=await bounded(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')),meta.bytes);assert(unpacked.byteLength===meta.bytes,'Decoded length mismatch');
  assert(crypto?.subtle,'This viewer requires a secure context or localhost');
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',unpacked))].map(x=>x.toString(16).padStart(2,'0')).join('');
  assert(hash===meta.sha256,'Binary checksum mismatch');return unpacked;
}
function validate(m,p,g,data){
  assert(m.schema==='dvidia.prerequisite-scene/v1'&&p.schema==='dvidia.prerequisite-motion/v1','Unsupported replay schema');
  assert(Number.isInteger(p.body_count)&&p.body_count>0&&p.body_count<=128&&m.bodies.length===p.body_count&&Number.isInteger(p.frame_count)&&p.frame_count>=2&&p.frame_count<=10000&&p.times.length===p.frame_count&&data.byteLength===p.frame_count*p.body_count*28,'Invalid pose layout');
  assert(p.times.every((t,i)=>Number.isFinite(t)&&t>=0&&(!i||t>p.times[i-1])),'Invalid timestamps');
  assert(m.meshes.length<=128&&m.geoms.length<=512,'Scene too large');
  const dv=new DataView(g),ids=new Set();
  for(const mesh of m.meshes){const n=mesh.vertex_count,k=mesh.index_count,v=mesh.position_byte_offset,f=mesh.index_byte_offset;assert([mesh.id,n,k,v,f].every(Number.isInteger)&&!ids.has(mesh.id)&&n>0&&k>=0&&k%3===0&&v>=0&&f>=0&&v+n*12<=g.byteLength&&f+k*4<=g.byteLength,'Invalid mesh range');ids.add(mesh.id);for(let i=0;i<n*3;i++)assert(Number.isFinite(dv.getFloat32(v+i*4,true)),'Invalid mesh vertex');for(let i=0;i<k;i++)assert(dv.getUint32(f+i*4,true)<n,'Invalid mesh index');}
  for(const geom of m.geoms){assert(Number.isInteger(geom.body_id)&&geom.body_id>=0&&geom.body_id<p.body_count&&['mesh','box','sphere','ellipsoid','capsule','cylinder','plane'].includes(geom.type),'Invalid geometry');for(const [key,n] of [['position',3],['quaternion_wxyz',4],['rgba',4],['size',3]])assert(Array.isArray(geom[key])&&geom[key].length===n&&geom[key].every(Number.isFinite),'Invalid geom values');assert(geom.type!=='mesh'||ids.has(geom.mesh_id),'Missing mesh');}
  const pv=new Float32Array(data);for(let i=0;i<pv.length;i+=7){assert(Array.from(pv.slice(i,i+7)).every(Number.isFinite)&&Math.abs(Math.hypot(...pv.slice(i+3,i+7))-1)<.001,'Invalid body pose');}
}
function primitive(g,meshes){switch(g.type){case'mesh':return meshes.get(g.mesh_id);case'box':return new THREE.BoxGeometry(...g.size.map(x=>x*2));case'sphere':return new THREE.SphereGeometry(g.size[0],20,12);case'ellipsoid':{const shape=new THREE.SphereGeometry(1,24,16);shape.scale(...g.size);return shape;}case'capsule':{const shape=new THREE.CapsuleGeometry(g.size[0],2*g.size[1],6,18);shape.rotateX(Math.PI/2);return shape;}case'cylinder':{const shape=new THREE.CylinderGeometry(g.size[0],g.size[0],g.size[1]*2,24);shape.rotateX(Math.PI/2);return shape;}case'plane':return new THREE.PlaneGeometry(5,5);}}
function requestRender(){if(!queued&&renderer){queued=true;requestAnimationFrame(render);}}
function showFrame(index){
  frame=Math.min(motion.frame_count-1,Math.max(0,index));$('timeline').value=frame;const t=motion.times[frame];$('clock').textContent=t.toFixed(2)+' / '+motion.times.at(-1).toFixed(2)+' s';
  for(let i=0;i<bodyGroups.length;i++){const o=(frame*bodyGroups.length+i)*7;bodyGroups[i].position.set(pose[o],pose[o+1],pose[o+2]);bodyGroups[i].quaternion.set(pose[o+4],pose[o+5],pose[o+6],pose[o+3]);}
  $('events').querySelectorAll('button').forEach(button=>button.classList.toggle('current',Math.abs(Number(button.dataset.time)-t)<.15));
  const row=motion.rows[frame];
  $('phase').textContent=human(row.phase||'unrecorded');
  $('fingers').textContent=(row.finger_normal_forces_n||[]).map(x=>x.toFixed(2)).join(' / ')||'—';
  $('support').textContent=Number.isFinite(row.world_object_force_n)?row.world_object_force_n.toFixed(2):'—';
  const reference=trial.independent_score?.reference;
  $('height').textContent=row.object_position&&reference&&t>=reference.t_s?((row.object_position[2]-reference.object_z_m)*1000).toFixed(1):'—';
  $('drive').textContent=Number.isFinite(row.gripper_actuator_force_n)?row.gripper_actuator_force_n.toFixed(2):'—';
  $('jaw').textContent=Number.isFinite(row.jaw_opening_m)?(row.jaw_opening_m*1000).toFixed(1):'—';
  $('stage').dataset.frame=String(frame);requestRender();
}
function stop(){playing=false;$('play').textContent='Play trial';$('play').setAttribute('aria-pressed','false');}
function render(now){queued=false;if(playing&&motion){const t=Math.max(0,(now-started)/1000);let next=frame;while(next+1<motion.frame_count&&motion.times[next+1]<=t)next++;if(next!==frame)showFrame(next);if(next===motion.frame_count-1)stop();}renderer.render(scene,camera);if(playing)requestRender();}
function cameraView(view){const target=new THREE.Vector3(.36,0,.15), positions={home:[1.2,-1.35,1.05],top:[.4,0,1.8],front:[.4,-1.8,.48]};camera.position.fromArray(positions[view]);orbit.target.copy(target);orbit.update();requestRender();}
function disposeScene(){if(!scene)return;const geometries=new Set(),materials=new Set();scene.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)materials.add(o.material);});geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());}
function setup(geometry){
  if(!renderer){assert($('canvas').getContext('webgl2'),'WebGL2 is unavailable');renderer=new THREE.WebGLRenderer({canvas:$('canvas'),antialias:true,powerPreference:'low-power'});renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.25;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;camera=new THREE.PerspectiveCamera(38,1,.01,20);camera.up.set(0,0,1);orbit=new OrbitControls(camera,$('canvas'));orbit.minDistance=.3;orbit.maxDistance=4;orbit.addEventListener('change',requestRender);resizeObserver=new ResizeObserver(()=>{const r=$('stage').getBoundingClientRect();camera.aspect=r.width/r.height;camera.updateProjectionMatrix();renderer.setSize(r.width,r.height,false);requestRender();});resizeObserver.observe($('stage'));}
  disposeScene();scene=new THREE.Scene();scene.background=new THREE.Color(0xe9eadf);scene.add(new THREE.HemisphereLight(0xf7fbff,0x949c7d,2.6));const sun=new THREE.DirectionalLight(0xfff7e7,3.2);sun.position.set(1,-1,3);sun.castShadow=true;sun.shadow.mapSize.set(1024,1024);sun.shadow.camera.left=-1.4;sun.shadow.camera.right=1.4;sun.shadow.camera.top=1.4;sun.shadow.camera.bottom=-1.4;sun.shadow.normalBias=.003;scene.add(sun);const fill=new THREE.DirectionalLight(0xe4edff,.9);fill.position.set(-1,1,2);scene.add(fill);
  bodyGroups=model.bodies.map(()=>{const group=new THREE.Group();scene.add(group);return group;});visual=[];collision=[];const meshes=new Map(),dv=new DataView(geometry);
  for(const mesh of model.meshes){const p=new Float32Array(mesh.vertex_count*3),idx=new Uint32Array(mesh.index_count);for(let i=0;i<p.length;i++)p[i]=dv.getFloat32(mesh.position_byte_offset+i*4,true);for(let i=0;i<idx.length;i++)idx[i]=dv.getUint32(mesh.index_byte_offset+i*4,true);const shape=new THREE.BufferGeometry();shape.setAttribute('position',new THREE.BufferAttribute(p,3));shape.setIndex(new THREE.BufferAttribute(idx,1));shape.computeVertexNormals();meshes.set(mesh.id,shape);}
  for(const geom of model.geoms){const shape=primitive(geom,meshes),rgba=geom.rgba;const add=(material)=>{const mesh=new THREE.Mesh(shape,material);mesh.position.fromArray(geom.position);const q=geom.quaternion_wxyz;mesh.quaternion.set(q[1],q[2],q[3],q[0]);bodyGroups[geom.body_id].add(mesh);return mesh;};if(rgba[3]>0&&geom.group<=2){const mesh=add(new THREE.MeshStandardMaterial({color:new THREE.Color().setRGB(...rgba.slice(0,3),THREE.LinearSRGBColorSpace),roughness:.68,metalness:geom.type==='mesh'?.12:0,transparent:rgba[3]<1,opacity:rgba[3],side:THREE.DoubleSide}));mesh.castShadow=geom.type!=='plane';mesh.receiveShadow=true;visual.push(mesh);}if(geom.collision){const mesh=add(new THREE.MeshStandardMaterial({color:0xb9774b,transparent:true,opacity:.4,depthWrite:false}));mesh.visible=false;collision.push(mesh);}}
  const r=$('stage').getBoundingClientRect();camera.aspect=r.width/r.height;camera.updateProjectionMatrix();renderer.setSize(r.width,r.height,false);showFrame(0);cameraView('home');$('collision').setAttribute('aria-pressed','false');$('scene-state').textContent='NATIVE PHYSICS · RECORDED POSES';
}
function human(key){return key.replaceAll('_',' ');}
function scalar(value){return typeof value==='number'?(Number.isInteger(value)?String(value):value.toFixed(2)):typeof value==='boolean'?(value?'Yes':'No'):String(value??'—');}
function eventTime(event){return Number(event.t??event.time_s??event.time??0);}
function eventLabel(event){const label=event.message||human(event.type||'event');return ({grip_loss_detected:'Grip check: ',grip_recovery_started:'Lowering: ',help_requested:'Stopped: '})[event.type]?({grip_loss_detected:'Grip check: ',grip_recovery_started:'Lowering: ',help_requested:'Stopped: '})[event.type]+label:label;}
function renderEvents(){const ul=$('events');ul.replaceChildren();for(const event of trial.events){if(['phase','assumptions'].includes(event.type)&&$('show-phases').getAttribute('aria-pressed')!=='true')continue;const li=document.createElement('li'),button=document.createElement('button'),time=document.createElement('time'),text=document.createElement('span');const t=eventTime(event);time.textContent=t.toFixed(2)+' s';text.textContent=eventLabel(event);button.dataset.time=t;button.append(time,text);button.addEventListener('click',()=>{stop();let i=motion.times.findIndex(x=>x>=t);showFrame(i<0?motion.frame_count-1:i);});li.append(button);ul.append(li);}if(!trial.events.length){const li=document.createElement('li');li.className='events-empty';li.textContent='No discrete intervention events were recorded.';ul.append(li);}}
function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o);}
function fillResults(){
  const rows=$('results');rows.replaceChildren();
  for(const shape of Object.keys(shapeNames))for(const load of Object.keys(loadNames))for(const tool of ['mounted','absent']){
    const group=report.attempts.filter(r=>r.case.object_shape===shape&&r.case.load_band===load&&r.case.tool_condition===tool);if(!group.length)continue;
    const tr=document.createElement('tr');for(const value of [shapeNames[shape],loadNames[load],toolName(tool),group.filter(r=>r.success&&r.valid).length+' / '+group.length,group.filter(r=>r.valid).length+' / '+group.length]){const td=document.createElement('td');td.textContent=value;tr.append(td);}rows.append(tr);
  }
}
function controls(disabled){document.querySelectorAll('#play,#timeline,#collision,#case,#shape,#load,#tool,#show-phases,#events button').forEach(el=>el.disabled=disabled);}
function chooseLayouts(){
  const previous=$('case').value;const cases=report.attempts.filter(r=>r.case.object_shape===$('shape').value&&r.case.load_band===$('load').value);const ids=[...new Set(cases.map(r=>r.case.replicate_id))];$('case').replaceChildren();for(const id of ids)option($('case'),id,human(id));if(ids.includes(previous))$('case').value=previous;
}
async function loadTrial(){
  const gen=++generation;stop();controls(true);$('loading').hidden=false;$('loading-message').textContent='Loading geometry and recorded motion.';$('stage').dataset.ready='loading';
  for(const id of ['phase','fingers','support','height','drive','jaw','duration'])$(id).textContent='—';$('events').replaceChildren();
  trial=report.attempts.find(row=>row.case.object_shape===$('shape').value&&row.case.load_band===$('load').value&&row.case.replicate_id===$('case').value&&row.case.tool_condition===$('tool').value);assert(trial,'Trial not found');
  const score=trial.independent_score,metrics=trial.metrics;
  $('scene-title').textContent=shapeNames[trial.case.object_shape];
  $('outcome').textContent=trial.valid?(trial.success?'Lift retained':'Lift not retained'):'Invalid attempt';$('outcome').dataset.success=trial.success&&trial.valid;$('outcome').dataset.valid=trial.valid;$('success').textContent=trial.success&&trial.valid?'Yes':'No';
  $('details').textContent=JSON.stringify({case:trial.case,metrics,independent_score:score,errors:trial.errors},null,2);$('receipt').href='trials/'+localPath(trial.id)+'.json';
  $('metric-a').textContent=(trial.case.mass*1000).toFixed(0)+' g';$('metric-b').textContent=Number.isFinite(score?.terminal_relative_lift_m)?(score.terminal_relative_lift_m*1000).toFixed(1)+' mm':'—';
  $('task-description').textContent='The same fixed pickup sequence runs on three rigid shapes at two loads, with and without the wrist-mounted rake.';
  $('scope-note').textContent='The simulated jaw command spans 70 mm; the full shoe sole is 100 mm wide. The position drive has a gain of 40 N/m. Its 10 N force cap is a limit, not the force delivered at contact. These settings are not a hardware calibration. Removing the rake changes both clearance and wrist inertia.';
  if(!trial.replay)throw new Error('This attempt has no valid replay. Its failure receipt remains downloadable.');
  const [m,p]=await Promise.all([json(trial.replay.model),json(trial.replay.motion)]);let geometry=geometryCache.get(m.geometry_binary.sha256);if(!geometry){geometry=await binary(m.geometry_binary);geometryCache.set(m.geometry_binary.sha256,geometry);}const positions=await binary(p.motion_binary);if(gen!==generation)return;validate(m,p,geometry,positions);model=m;motion=p;pose=new Float32Array(positions);setup(geometry);renderEvents();$('timeline').max=String(motion.frame_count-1);$('duration').textContent=motion.times.at(-1).toFixed(2)+' s';$('status').textContent=(report.split==='heldout'?'Fresh evaluation':'Development controls')+' · '+human(trial.case.replicate_id)+' · '+toolName(trial.case.tool_condition);$('stage').dataset.ready='true';$('loading').hidden=true;controls(false);
}
function fail(error){stop();$('loading').hidden=false;$('loading-message').textContent=error.message;$('stage').dataset.ready='error';$('status').textContent='Replay unavailable. The protocol and trial receipts remain accessible.';for(const id of ['case','shape','load','tool'])$(id).disabled=false;console.error(error);}
async function main(){
  assert(location.protocol!=='file:','Serve the export over HTTP');report=await json('benchmark.json');assert(report.schema==='dvidia.graspability/v3'&&Array.isArray(report.attempts)&&report.attempts.length>0&&report.attempts.length<=100,'Invalid benchmark');fillResults();
  const commit=report.source.commit;if(commit){assert(/^[a-f0-9]{40}$/.test(commit),'Invalid source commit');$('source').href='https://github.com/Dvidia-Inference/dvidia-training/tree/'+commit;}
  $('command').textContent='git clone https://github.com/Dvidia-Inference/dvidia-training.git\ncd dvidia-training\n'+(commit?'git checkout '+commit+'\n':'')+'python -m venv .venv\n# Activate .venv using your shell, then:\npython -m pip install ".[arm]"\ndvidia-robot-sim fetch piper --output robot-assets\ndvidia-graspability-benchmark --assets robot-assets --output results --split '+report.split+(commit?' --source-commit '+commit:'')+'\npython -m http.server 8000 --directory results';
  for(const [id,labels,key] of [['shape',shapeNames,'object_shape'],['load',loadNames,'load_band']]){for(const value of [...new Set(report.attempts.map(r=>r.case[key]))])option($(id),value,labels[value]||human(value));$(id).addEventListener('change',()=>{chooseLayouts();loadTrial().catch(fail);});}
  for(const id of ['case','tool'])$(id).addEventListener('change',()=>loadTrial().catch(fail));
  document.querySelectorAll('[data-camera]').forEach(button=>button.addEventListener('click',()=>{if(camera)cameraView(button.dataset.camera);}));
  $('show-phases').addEventListener('click',()=>{$('show-phases').setAttribute('aria-pressed',String($('show-phases').getAttribute('aria-pressed')!=='true'));if(motion)renderEvents();});
  $('play').addEventListener('click',()=>{if(playing){stop();return;}if(frame===motion.frame_count-1)showFrame(0);playing=true;started=performance.now()-motion.times[frame]*1000;$('play').textContent='Pause trial';$('play').setAttribute('aria-pressed','true');requestRender();});
  $('timeline').addEventListener('input',()=>{stop();showFrame(Number($('timeline').value));});
  $('collision').addEventListener('click',()=>{const on=$('collision').getAttribute('aria-pressed')!=='true';$('collision').setAttribute('aria-pressed',String(on));visual.forEach(m=>m.visible=!on);collision.forEach(m=>m.visible=on);$('scene-state').textContent=on?'COLLISION GEOMETRY · RECORDED POSES':'NATIVE PHYSICS · RECORDED POSES';requestRender();});
  $('canvas').addEventListener('keydown',event=>{if(event.key==='Home'||event.key==='h'){event.preventDefault();cameraView('home');}});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});chooseLayouts();await loadTrial();
}
main().catch(fail);
