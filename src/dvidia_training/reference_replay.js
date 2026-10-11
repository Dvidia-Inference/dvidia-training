/* DVIDIA native reference replay. MIT. No remote imports, telemetry or physics execution. */
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const $ = id => document.getElementById(id), finite = n => typeof n === 'number' && Number.isFinite(n);
const MAX_BYTES = 64 * 1024 * 1024, MAX_JSON = 12 * 1024 * 1024;
const signed = (n, digits = 2) => `${n >= 0 ? '+' : ''}${n.toFixed(digits)}`;
const radiansToDegrees = 180 / Math.PI;
let renderer, controls, scene, camera, selected, metadata, model, poseBytes, bodies, tcpAxes, trail;
let playing = false, startedAt = 0, startTime = 0, requested = false, currentFrame = 0, renderCount = 0;
let floor, grid, visual = [], fitMeshes = [], proxies = [], markers = [], markerLabels = [];
const focus = new THREE.Vector3(0, 0, .38), extent = {radius: 1.6};
const vector = new THREE.Vector3(), quaternion = new THREE.Quaternion(), euler = new THREE.Euler(0, 0, 0, 'XYZ');

function fail(error) {
  playing = false;
  if (renderer) renderer.dispose();
  $('loading').hidden = false; $('loading').classList.add('failure');
  $('loading').replaceChildren();
  const content = document.createElement('div'), title = document.createElement('strong'), details = document.createElement('p');
  title.textContent = 'The 3D replay could not open.';
  details.id = 'error-details'; details.textContent = `${error.message}\n\nThis replay needs WebGL2 and a local or hosted HTTP URL. Downloaded files must be served together, including geometry and motion. The numerical receipt remains available below.`;
  content.append(title, details); $('loading').append(content);
  $('status').textContent = 'Replay unavailable. No robot commands were sent.';
  document.querySelectorAll('button,select,input').forEach(element => { element.disabled = true; });
  $('stage').dataset.ready = 'error';
}
function assert(condition, message) { if (!condition) throw new Error(message); }
function array(value, length, name) { assert(Array.isArray(value) && value.length === length && value.every(finite), `Invalid ${name}.`); }
function integer(value, min, max, name) { assert(Number.isInteger(value) && value >= min && value <= max, `Invalid ${name}.`); }
function localHref(value) {
  assert(typeof value === 'string' && /^[a-zA-Z0-9_.-]+(?:\/[a-zA-Z0-9_.-]+)*$/.test(value) && !value.split('/').some(p => p === '..' || p === '.'), 'Replay file link must stay within this directory.');
  return value;
}
async function response(href) {
  const result = await fetch(localHref(href), {credentials: 'omit', redirect: 'error'});
  assert(result.ok, `Could not load ${href} (HTTP ${result.status}).`); return result;
}
async function readBounded(stream, limit, label) {
  assert(stream,`Empty ${label} response.`);const reader=stream.getReader(),chunks=[];let total=0;
  try{for(;;){const {done,value}=await reader.read();if(done)break;total+=value.byteLength;if(total>limit){await reader.cancel();throw new Error(`${label} exceeds its byte bounds.`);}chunks.push(value);}}finally{reader.releaseLock();}
  const bytes=new Uint8Array(total);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.byteLength;}return bytes.buffer;
}
async function json(href) {
  const text = new TextDecoder().decode(await readBounded((await response(href)).body,MAX_JSON,href)); return JSON.parse(text);
}
async function binary(meta, defaultName) {
  assert(meta && meta.href === defaultName, `Invalid ${defaultName} metadata.`);
  integer(meta.bytes, 0, MAX_BYTES, `${defaultName} byte count`);
  let bytes;
  if (meta.compressed_href && typeof DecompressionStream === 'function') {
    assert(meta.compression === 'gzip', 'Unsupported binary compression.');
    integer(meta.compressed_bytes, 0, MAX_BYTES, 'compressed byte count');
    const packed = await readBounded((await response(meta.compressed_href)).body,meta.compressed_bytes,meta.compressed_href);
    assert(packed.byteLength === meta.compressed_bytes, `Compressed ${defaultName} size mismatch.`);
    bytes = await readBounded(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip')),meta.bytes,defaultName);
  } else bytes = await readBounded((await response(meta.href)).body,meta.bytes,defaultName);
  assert(bytes.byteLength === meta.bytes && bytes.byteLength <= MAX_BYTES, `${defaultName} size mismatch.`);
  return bytes;
}
function checkSchema(sceneData, motionData, geometry, motion) {
  assert(sceneData.schema === 'dvidia.reference-replay-scene/v1' && motionData.schema === 'dvidia.reference-replay-motion/v1', 'Unsupported replay schema.');
  assert(Array.isArray(sceneData.bodies) && sceneData.bodies.length >= 1 && sceneData.bodies.length <= 256, 'Invalid body list.');
  integer(motionData.body_count, 1, 256, 'body count');
  assert(sceneData.bodies.length === motionData.body_count && motionData.frame_stride_floats === motionData.body_count * 7, 'Body pose layout mismatch.');
  sceneData.bodies.forEach((body, index) => assert(body.id === index && typeof body.name === 'string', 'Bodies must use ordered native IDs.'));
  assert(Array.isArray(sceneData.meshes) && sceneData.meshes.length <= 512 && Array.isArray(sceneData.geoms) && sceneData.geoms.length <= 2048, 'Invalid geometry list.');
  const meshIds = new Set(), dv = new DataView(geometry);
  for (const mesh of sceneData.meshes) {
    integer(mesh.id, 0, 512, 'mesh ID'); assert(!meshIds.has(mesh.id), 'Duplicate mesh ID.'); meshIds.add(mesh.id);
    integer(mesh.vertex_count, 1, MAX_BYTES / 12, 'vertex count'); integer(mesh.index_count, 3, MAX_BYTES / 4, 'index count');
    integer(mesh.position_byte_offset, 0, geometry.byteLength, 'vertex offset'); integer(mesh.index_byte_offset, 0, geometry.byteLength, 'index offset');
    assert(mesh.position_byte_offset % 4 === 0 && mesh.index_byte_offset % 4 === 0 && mesh.index_count % 3 === 0 && mesh.position_byte_offset + mesh.vertex_count * 12 <= geometry.byteLength && mesh.index_byte_offset + mesh.index_count * 4 <= geometry.byteLength, 'Mesh binary bounds mismatch.');
    for (let i = 0; i < mesh.vertex_count * 3; ++i) assert(finite(dv.getFloat32(mesh.position_byte_offset + i * 4, true)), 'Non-finite mesh coordinate.');
    for (let i = 0; i < mesh.index_count; ++i) assert(dv.getUint32(mesh.index_byte_offset + i * 4, true) < mesh.vertex_count, 'Mesh index outside vertices.');
  }
  for (const geom of sceneData.geoms) {
    integer(geom.body_id, 0, motionData.body_count - 1, 'geom body');
    integer(geom.group, 0, 5, 'native visibility group');
    assert(['mesh', 'sphere', 'box', 'capsule', 'cylinder', 'plane'].includes(geom.type), 'Unsupported native geometry.');
    array(geom.position, 3, 'local position'); array(geom.quaternion_wxyz, 4, 'local quaternion'); array(geom.size, 3, 'geom size'); array(geom.rgba, 4, 'geom color');
    assert(typeof geom.collision === 'boolean' && geom.size.every(n => n >= 0), 'Invalid collision geometry.');
    assert(geom.type !== 'mesh' || meshIds.has(geom.mesh_id), 'Missing mesh.');
  }
  assert(Array.isArray(motionData.arm_joint_names) && motionData.arm_joint_names.length === 6 && motionData.arm_joint_names.every(n => typeof n === 'string'), 'Expected six named arm joints.');
  assert(Array.isArray(motionData.cases) && motionData.cases.length >= 1 && motionData.cases.length <= 100, 'Invalid recorded motions.');
  const poses = new DataView(motion), ranges = [];
  for (const run of motionData.cases) {
    assert(typeof run.name === 'string' && finite(run.timestep_seconds) && run.timestep_seconds > 0, 'Invalid motion identity.');
    integer(run.frame_count, 2, 100000, 'frame count'); integer(run.pose_byte_offset, 0, motion.byteLength, 'pose offset');
    const end = run.pose_byte_offset + run.frame_count * motionData.frame_stride_floats * 4;
    assert(run.pose_byte_offset % 4 === 0 && end <= motion.byteLength && Array.isArray(run.rows) && run.rows.length === run.frame_count, 'Motion binary bounds mismatch.');
    assert(!ranges.some(([a,b]) => run.pose_byte_offset < b && end > a), 'Recorded motion byte ranges overlap.'); ranges.push([run.pose_byte_offset,end]);
    let previous = -Infinity;
    for (const row of run.rows) {
      assert(finite(row.t_seconds) && row.t_seconds >= 0 && row.t_seconds > previous, 'Invalid recorded timestamps.'); previous = row.t_seconds;
      array(row.joint_position_rad, 6, 'joint position'); array(row.issued_joint_targets_rad, 6, 'joint targets'); array(row.linkage_position_rad, 8, 'gripper linkage');
      array(row.tcp_position_m, 3, 'tool position'); array(row.tcp_quaternion_wxyz, 4, 'tool orientation');
      assert(finite(row.issued_gripper_0_255) && finite(row.pad_center_separation_m), 'Invalid gripper observation.'); integer(row.contact_count, 0, 100000, 'contact count');
    }
    for (let offset = run.pose_byte_offset; offset < end; offset += 28) {
      for (let i = 0; i < 7; ++i) assert(finite(poses.getFloat32(offset + i * 4, true)), 'Non-finite recorded body pose.');
      const norm = Math.hypot(...[3,4,5,6].map(i => poses.getFloat32(offset + i * 4, true)));
      assert(Math.abs(norm - 1) < .001, 'Recorded body quaternion is not normalized.');
    }
  }
}
function nativeQuaternion(q, target) { return target.set(q[1], q[2], q[3], q[0]); }
function geometryFor(geom, meshes) {
  switch (geom.type) {
    case 'mesh': return meshes.get(geom.mesh_id);
    case 'box': return new THREE.BoxGeometry(...geom.size.map(x => x * 2));
    case 'sphere': return new THREE.SphereGeometry(geom.size[0], 24, 16);
    case 'capsule': { const g = new THREE.CapsuleGeometry(geom.size[0], 2 * geom.size[1], 6, 20); g.rotateX(Math.PI / 2); return g; }
    case 'cylinder': { const g = new THREE.CylinderGeometry(geom.size[0], geom.size[0], 2 * geom.size[1], 28); g.rotateX(Math.PI / 2); return g; }
    case 'plane': return new THREE.PlaneGeometry(Math.max(1,geom.size[0] * 2), Math.max(1,geom.size[1] * 2));
  }
}
function initWorld(geometryBytes) {
  assert($('canvas').getContext('webgl2') !== null, 'WebGL2 is unavailable. Enable hardware acceleration or use a browser with WebGL2 support.');
  renderer = new THREE.WebGLRenderer({canvas: $('canvas'), antialias: true, alpha: true, powerPreference: 'low-power'});
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2)); renderer.setClearColor(0xf6f7f1, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.3;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  scene = new THREE.Scene(); camera = new THREE.PerspectiveCamera(38, 1, .005, 30); camera.up.set(0,0,1);
  controls = new OrbitControls(camera, $('canvas')); controls.enableDamping = false; controls.minDistance = .22; controls.maxDistance = 6; controls.target.copy(focus); controls.addEventListener('change', requestRender);
  scene.add(new THREE.HemisphereLight(0xf4fbff, 0xc6cab5, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 3.4); key.position.set(2,-2,4); key.castShadow = true; key.shadow.mapSize.set(1024,1024); Object.assign(key.shadow.camera,{left:-1.6,right:1.6,top:1.6,bottom:-1.6,near:.1,far:8}); key.shadow.bias = -.00015; scene.add(key);
  const fill = new THREE.DirectionalLight(0xd8e7f4,.8); fill.position.set(-2,3,2); scene.add(fill);
  floor = new THREE.Mesh(new THREE.PlaneGeometry(5,5), new THREE.MeshStandardMaterial({color:0xf0f2e9,roughness:.95})); floor.position.z = -.007; floor.receiveShadow = true; scene.add(floor);
  grid = new THREE.GridHelper(3,30,0xb4bdaa,0xd9dfd2); grid.rotation.x = Math.PI / 2; grid.position.z = -.006; grid.material.transparent = true; grid.material.opacity = .48; scene.add(grid);
  const worldAxes = new THREE.AxesHelper(.22); worldAxes.position.set(-.65,-.5,.002); scene.add(worldAxes);
  bodies = model.bodies.map(() => { const group = new THREE.Group(); scene.add(group); return group; });
  const meshMap = new Map(), dv = new DataView(geometryBytes);
  for (const mesh of model.meshes) {
    const points = new Float32Array(mesh.vertex_count * 3), indices = new Uint32Array(mesh.index_count);
    for (let i=0;i<points.length;++i) points[i] = dv.getFloat32(mesh.position_byte_offset + i * 4,true);
    for (let i=0;i<indices.length;++i) indices[i] = dv.getUint32(mesh.index_byte_offset + i * 4,true);
    const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position',new THREE.BufferAttribute(points,3)); geometry.setIndex(new THREE.BufferAttribute(indices,1)); geometry.computeVertexNormals(); meshMap.set(mesh.id,geometry);
  }
  for (const geom of model.geoms) {
    const geometry = geometryFor(geom,meshMap), rgba = geom.rgba;
    // Upstream group/contact geoms can be invisible; preserve native alpha in visual mode.
    if (rgba[3] > 0 && geom.group <= 2) {
      const material = new THREE.MeshStandardMaterial({color:new THREE.Color().setRGB(rgba[0],rgba[1],rgba[2],THREE.LinearSRGBColorSpace),metalness:geom.type === 'mesh' ? .22 : .1,roughness:.43,transparent:rgba[3] < 1,opacity:rgba[3],side:THREE.DoubleSide});
      const mesh = new THREE.Mesh(geometry,material); mesh.position.fromArray(geom.position); nativeQuaternion(geom.quaternion_wxyz,mesh.quaternion); mesh.castShadow = geom.type !== 'plane'; mesh.receiveShadow = true; bodies[geom.body_id].add(mesh); visual.push(mesh); if(geom.type !== 'plane')fitMeshes.push(mesh); else {floor.visible=false;grid.position.z=geom.position[2]+.0004;}
    }
    if (geom.collision) {
      const proxy = new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0xc27644,transparent:true,opacity:.44,roughness:.8,depthWrite:false,side:THREE.DoubleSide}));
      proxy.position.fromArray(geom.position); nativeQuaternion(geom.quaternion_wxyz,proxy.quaternion); proxy.visible = false; bodies[geom.body_id].add(proxy); proxies.push(proxy);
      const lines = new THREE.LineSegments(new THREE.EdgesGeometry(geometry,25),new THREE.LineBasicMaterial({color:0x8c5b36,transparent:true,opacity:.55})); proxy.add(lines);
    }
  }
  const bodyNames = ['shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link'];
  bodyNames.forEach((name,index) => {
    const body = model.bodies.find(b => b.name === name || b.name.endsWith('/'+name)); if (!body) return;
    const marker = new THREE.Mesh(new THREE.SphereGeometry(.012,16,12),new THREE.MeshBasicMaterial({color:0x35736b,depthTest:false})); marker.visible = false; marker.renderOrder = 9; bodies[body.id].add(marker); markers.push(marker);
    const label = document.createElement('span'); label.className='joint-label'; label.textContent = `J${index+1}`; Object.assign(label.style,{position:'absolute',font:'11px system-ui',color:'#254e47',background:'#ffffffda',padding:'3px 5px',border:'1px solid #bed0c3',borderRadius:'3px',pointerEvents:'none',display:'none'}); $('stage').append(label); markerLabels.push({marker,label});
  });
  tcpAxes = new THREE.AxesHelper(.105); tcpAxes.material.depthTest = false; tcpAxes.renderOrder = 10; scene.add(tcpAxes);
  trail = new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x528b79,transparent:true,opacity:.55})); scene.add(trail);
  $('canvas').addEventListener('webglcontextlost',event => {event.preventDefault();fail(new Error('The WebGL graphics context was lost. Reload this page to reopen the replay.'));});
  new ResizeObserver(() => { const bounds = $('stage').getBoundingClientRect(); if(bounds.width && bounds.height){const aspect=bounds.width / bounds.height,scale=Math.max(1,1/aspect)/Math.max(1,1/camera.aspect);camera.position.sub(controls.target).multiplyScalar(scale).add(controls.target);renderer.setSize(bounds.width,bounds.height,false);camera.aspect=aspect;camera.updateProjectionMatrix();controls.update();requestRender();}}).observe($('stage'));
}
function createReadouts() {
  metadata.arm_joint_names.forEach((name,index) => {
    const row = document.createElement('div'); row.className = 'joint';
    const heading = document.createElement('div'); heading.className = 'joint-heading';
    const label = document.createElement('span'); label.textContent = `J${index+1} · ${['Base','Shoulder','Elbow','Wrist 1','Wrist 2','Wrist 3'][index]}`;
    const value = document.createElement('span'); value.id = `joint-${index}`; value.className = 'joint-value';
    const native = document.createElement('div'); native.className = 'joint-name'; native.textContent = name;
    const track = document.createElement('div'); track.className='joint-track'; const bar=document.createElement('div');bar.className='joint-fill';bar.id=`joint-fill-${index}`; const target=document.createElement('span');target.className='target-dot';target.id=`joint-target-${index}`;
    track.append(bar,target);heading.append(label,value);row.append(heading,native,track);$('joint-readouts').append(row);
  });
}
function cameraPreset(view) {
  const r = extent.radius * Math.max(1,1/camera.aspect);
  const offsets = {home:[1.1,-1.25,.8],top:[.001,.001,1.7],front:[0,-1.7,.15],side:[1.7,0,.15]};
  const p = offsets[view] || offsets.home; controls.target.copy(focus);camera.position.copy(focus).add(new THREE.Vector3(...p).multiplyScalar(r));controls.update();requestRender();
  document.querySelectorAll('[data-camera]').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.camera === view)));
}
function drawFrame(index) {
  currentFrame = Math.max(0,Math.min(selected.frame_count-1,index)); const row = selected.rows[currentFrame];
  const start = selected.pose_byte_offset + currentFrame * metadata.frame_stride_floats * 4;
  bodies.forEach((body,id) => { const offset=start+id*28;body.position.set(poseBytes.getFloat32(offset,true),poseBytes.getFloat32(offset+4,true),poseBytes.getFloat32(offset+8,true));body.quaternion.set(poseBytes.getFloat32(offset+16,true),poseBytes.getFloat32(offset+20,true),poseBytes.getFloat32(offset+24,true),poseBytes.getFloat32(offset+12,true)); });
  $('timeline').value=String(currentFrame);$('clock').textContent=`${row.t_seconds.toFixed(2)} / ${selected.rows.at(-1).t_seconds.toFixed(2)} s`;
  row.joint_position_rad.forEach((angle,i) => { $('joint-'+i).textContent=`${signed(angle*radiansToDegrees,1)}°`;const range=model.joints.find(j=>j.name===metadata.arm_joint_names[i])?.range || [-Math.PI*2,Math.PI*2];const proportion=n=>100*Math.max(0,Math.min(1,(n-range[0])/(range[1]-range[0]||1)));$('joint-fill-'+i).style.width=`${proportion(angle)}%`;$('joint-target-'+i).style.left=`${proportion(row.issued_joint_targets_rad[i])}%`; });
  $('tcp-position').textContent=row.tcp_position_m.map(n=>signed(n,3)).join(' · ');nativeQuaternion(row.tcp_quaternion_wxyz,quaternion);euler.setFromQuaternion(quaternion,'XYZ');$('tcp-rotation').textContent=[euler.x,euler.y,euler.z].map(n=>`${signed(n*radiansToDegrees,1)}°`).join(' · ');
  $('gripper-command').textContent=row.issued_gripper_0_255.toFixed(0);$('gripper-gap').textContent=`${(row.pad_center_separation_m*1000).toFixed(1)} mm`;$('contacts').textContent=String(row.contact_count);
  tcpAxes.position.fromArray(row.tcp_position_m);tcpAxes.quaternion.copy(quaternion);
  const points = new Float32Array((currentFrame+1)*3);for(let i=0;i<=currentFrame;++i)points.set(selected.rows[i].tcp_position_m,i*3);trail.geometry.dispose();trail.geometry=new THREE.BufferGeometry();trail.geometry.setAttribute('position',new THREE.BufferAttribute(points,3));
  $('status').textContent=`Recorded frame ${currentFrame+1}/${selected.frame_count} · ${selected.timestep_seconds*1000} ms native physics step · ${metadata.body_count} body poses · Browser physics: none.`;
  requestRender();
}
function chooseMotion(reset=true) {
  selected=metadata.cases.find(c=>c.name===$('case').value && c.timestep_seconds===Number($('step').value));
  assert(selected,'Requested recorded motion is missing.');$('timeline').max=String(selected.frame_count-1);drawFrame(reset?0:Math.min(currentFrame,selected.frame_count-1));
  if(playing){startTime=selected.rows[currentFrame].t_seconds;startedAt=performance.now();}
}
function updateSteps(preferred=Number($('step').value)) {
  $('step').replaceChildren();const steps=[...new Set(metadata.cases.filter(c=>c.name===$('case').value).map(c=>c.timestep_seconds))].sort((a,b)=>b-a);
  for(const step of steps){const option=document.createElement('option');option.value=String(step);option.textContent=`${step*1000} ms`;$('step').append(option);}
  $('step').value=String(steps.includes(preferred)?preferred:steps.includes(.001)?.001:steps[0]);
}
function requestRender() { if(!requested && renderer){requested=true;requestAnimationFrame(render);} }
function render(now) {
  requested=false;if(!renderer)return;
  if(playing){const first=selected.rows[0].t_seconds,last=selected.rows.at(-1).t_seconds;let t=startTime+(now-startedAt)/1000*Number($('speed').value);if(t>last){t=first+(t-first)%(last-first);startTime=t;startedAt=now;}let index=0;while(index+1<selected.rows.length&&selected.rows[index+1].t_seconds<=t)index++;if(index!==currentFrame)drawFrame(index);}
  renderer.render(scene,camera);$('canvas').dataset.renderCount=String(++renderCount);
  markerLabels.forEach(({marker,label})=>{if(!marker.visible){label.style.display='none';return;}marker.getWorldPosition(vector);vector.project(camera);const bounds=$('stage').getBoundingClientRect();const visible=vector.z>=-1&&vector.z<=1&&Math.abs(vector.x)<1&&Math.abs(vector.y)<1;label.style.display=visible?'block':'none';label.style.left=`${(vector.x*.5+.5)*bounds.width+8}px`;label.style.top=`${(-vector.y*.5+.5)*bounds.height-8}px`;});
  if(playing)requestRender();
}
function togglePlay() {playing=!playing;$('play').textContent=playing?'Pause motion':'Play motion';$('play').setAttribute('aria-pressed',String(playing));if(playing){startedAt=performance.now();startTime=selected.rows[currentFrame].t_seconds;requestRender();}}
function connectControls() {
  $('play').addEventListener('click',togglePlay);$('timeline').addEventListener('input',()=>{drawFrame(Number($('timeline').value));if(playing){startTime=selected.rows[currentFrame].t_seconds;startedAt=performance.now();}});
  $('case').addEventListener('change',()=>{updateSteps();chooseMotion();});$('step').addEventListener('change',()=>chooseMotion());
  $('speed').addEventListener('change',()=>{startTime=selected.rows[currentFrame].t_seconds;startedAt=performance.now();});
  document.querySelectorAll('[data-camera]').forEach(b=>b.addEventListener('click',()=>cameraPreset(b.dataset.camera)));
  $('collision').addEventListener('click',()=>{const active=$('collision').getAttribute('aria-pressed')!=='true';$('collision').setAttribute('aria-pressed',String(active));visual.forEach(m=>m.visible=!active);proxies.forEach(m=>m.visible=active);$('mode-label').textContent=active?'Native collision proxies':'Native visual geometry';requestRender();});
  for(const [id,objects] of [['trail',[trail]],['axes',[tcpAxes]],['markers',markers]])$(id).addEventListener('click',()=>{const active=$(id).getAttribute('aria-pressed')!=='true';$(id).setAttribute('aria-pressed',String(active));objects.forEach(m=>m.visible=active);requestRender();});
  $('canvas').addEventListener('keydown',event=>{const key=event.key;if(key==='h'||key==='H'||key==='Home'){event.preventDefault();cameraPreset('home');return;}if(key===' '){event.preventDefault();togglePlay();return;}if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','_'].includes(key))return;event.preventDefault();const offset=camera.position.clone().sub(controls.target);const spherical=new THREE.Spherical().setFromVector3(offset.applyAxisAngle(new THREE.Vector3(1,0,0),-Math.PI/2));if(key==='ArrowLeft')spherical.theta-=.1;if(key==='ArrowRight')spherical.theta+=.1;if(key==='ArrowUp')spherical.phi=Math.max(.01,spherical.phi-.1);if(key==='ArrowDown')spherical.phi=Math.min(Math.PI-.01,spherical.phi+.1);if(key==='+'||key==='=')spherical.radius=Math.max(controls.minDistance,spherical.radius*.9);if(key==='-'||key==='_')spherical.radius=Math.min(controls.maxDistance,spherical.radius/ .9);offset.setFromSpherical(spherical).applyAxisAngle(new THREE.Vector3(1,0,0),Math.PI/2);camera.position.copy(controls.target).add(offset);controls.update();requestRender();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing)togglePlay();});
}
async function main() {
  if(location.protocol==='file:')throw new Error('Local file URLs cannot fetch the binary replay. Serve this directory with a local static HTTP server.');
  [model,metadata]=await Promise.all([json('scene.json'),json('motion.json')]);
  $('loading').textContent='Opening pinned geometry and recorded motion…';
  const [geometry,motion]=await Promise.all([binary(model.geometry_binary,'geometry.bin'),binary(metadata.motion_binary,'motion.bin')]);checkSchema(model,metadata,geometry,motion);poseBytes=new DataView(motion);
  initWorld(geometry);createReadouts();
  for(const name of [...new Set(metadata.cases.map(c=>c.name))]){const option=document.createElement('option');option.value=name;option.textContent=name.replaceAll('_',' ');$('case').append(option);}
  $('case').value=metadata.cases.some(c=>c.name==='arm_sweep')?'arm_sweep':metadata.cases[0].name;updateSteps(.001);chooseMotion();
  const bounds=new THREE.Box3(); // Fit the arm and gripper, excluding native and viewing floor planes.
  for(const mesh of fitMeshes){mesh.updateWorldMatrix(true,false);bounds.expandByObject(mesh);}if(!bounds.isEmpty()){bounds.getCenter(focus);const size=bounds.getSize(new THREE.Vector3());extent.radius=Math.max(.84,size.length()*.84);}
  cameraPreset('home');connectControls();
  document.querySelectorAll('button,select,input').forEach(element=>{element.disabled=false;});$('loading').hidden=true;$('stage').dataset.ready='true';$('stage').dataset.visualGeoms=String(visual.length);$('stage').dataset.collisionGeoms=String(proxies.length);
  const source=metadata.source||model.source||{};
  if(source.report_href)$('report-link').href=localHref(source.report_href);if(source.receipt_href)$('receipt-link').href=localHref(source.receipt_href);
  $('provenance').textContent=`Reference: ${source.profile||'UR5e + Robotiq 2F-85'}. Native meshes are separately licensed by their authors; notices are included with this replay. All rendering code and replay data are served locally. Physical accuracy remains unmeasured.`;
}
main().catch(fail);
