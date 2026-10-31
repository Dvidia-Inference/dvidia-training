/* DVIDIA environment builder. MIT. Draft editor and recorded native replay. */
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {TransformControls} from 'three/addons/controls/TransformControls.js';
const $=id=>document.getElementById(id),assert=(ok,message)=>{if(!ok)throw Error(message)};
const MAX_BYTES=64*1024*1024,rootURL=new URL(document.documentElement.dataset.artifactRoot||'./',location.href);
let renderer,world,camera,orbit,transform,draftGroup,robotGroup,replayGroup,objects=new Map();
let recipe,originalRecipe,selectedId,local=false,indexData,indexURL,currentCase,poseData,bodyGroups=[];
let mode='edit',frame=0,playing=false,startTime=0,loadEpoch=0,selectedOutline,localBusy=false;
const copy=value=>JSON.parse(JSON.stringify(value)),finiteArray=(value,n)=>Array.isArray(value)&&value.length===n&&value.every(Number.isFinite);
function resourceURL(path,base=rootURL,boundary=rootURL){
  assert(path instanceof URL||(typeof path==='string'&&path.length<512&&!path.startsWith('//')&&!/^[a-z][a-z0-9+.-]*:/i.test(path)&&!/[\\?#%]/.test(path)),'Invalid local resource path.');
  const url=new URL(path,base),scope=new URL('./',boundary);
  assert(url.origin===location.origin&&url.protocol===location.protocol&&!url.username&&!url.password&&!url.search&&!url.hash&&!url.pathname.includes('%')&&url.pathname.startsWith(scope.pathname),'Resource escapes its local artifact directory.');return url;
}
async function getBytes(path,base=rootURL,limit=8*1024*1024,optional=false,boundary=rootURL){
  const url=resourceURL(path,base,boundary);
  for(let attempt=0;attempt<3;attempt++){
    if(attempt)await new Promise(resolve=>setTimeout(resolve,200*attempt));
    try{
      const response=await fetch(url,{credentials:'omit',referrerPolicy:'no-referrer'});
      const final=new URL(response.url),allowed=final.origin===location.origin||(location.hostname==='dvidia-robot-simulation-lab.static.hf.space'&&final.protocol==='https:'&&final.hostname.endsWith('.hf.co'));
      assert(allowed,'Unexpected resource host.');
      if(!response.ok&&[408,425,429,500,502,503,504].includes(response.status)&&attempt<2){if(response.body)await response.body.cancel().catch(()=>{});continue}
      if(optional&&!response.ok)return null;
      assert(response.ok,'Could not open '+url.pathname);assert(response.body,'Missing resource body.');
      const reader=response.body.getReader(),chunks=[];let length=0;
      try{for(;;){const {done,value}=await reader.read();if(done)break;length+=value.length;assert(length<=limit,'Resource exceeds the byte limit.');chunks.push(value)}}finally{await reader.cancel().catch(()=>{});reader.releaseLock()}
      const bytes=new Uint8Array(length);let offset=0;for(const c of chunks){bytes.set(c,offset);offset+=c.length}return bytes;
    }catch(error){if(attempt===2||!(error instanceof TypeError||error.name==='NetworkError'))throw error}
  }
}
async function getJSON(path,base=rootURL,optional=false,boundary=rootURL){const bytes=await getBytes(path,base,8*1024*1024,optional,boundary);return bytes===null?null:JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes))}
async function unpack(path,identity,base,boundary=rootURL){
  assert(identity&&Number.isInteger(identity.bytes)&&identity.bytes>0&&identity.bytes<=MAX_BYTES,'Invalid native resource byte count.');
  assert(typeof identity.sha256==='string'&&/^[a-f0-9]{64}$/.test(identity.sha256),'Invalid native digest.');
  const packed=await getBytes(path,base,16*1024*1024,false,boundary),reader=new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip')).getReader(),chunks=[];let length=0;
  try{for(;;){const {done,value}=await reader.read();if(done)break;length+=value.length;assert(length<=identity.bytes,'Inflated native data exceeds its byte count.');chunks.push(value)}}finally{await reader.cancel();reader.releaseLock()}
  const raw=new Uint8Array(length);let offset=0;for(const c of chunks){raw.set(c,offset);offset+=c.length}
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),x=>x.toString(16).padStart(2,'0')).join('');
  assert(raw.length===identity.bytes&&hash===identity.sha256,'Native data failed SHA-256 verification.');return raw.buffer;
}
const geometryCache=new Map();let geometryCacheBytes=0;
async function cachedGeometry(path,identity,base,boundary=rootURL){
  const url=resourceURL(path,base,boundary);
  assert(identity&&Number.isInteger(identity.bytes)&&identity.bytes>0&&identity.bytes<=MAX_BYTES&&typeof identity.sha256==='string'&&/^[a-f0-9]{64}$/.test(identity.sha256),'Invalid geometry identity.');
  const key=url.href+'|'+identity.sha256+'|'+identity.bytes;
  if(geometryCache.has(key)){const value=geometryCache.get(key);geometryCache.delete(key);geometryCache.set(key,value);return value}
  const value=await unpack(url,identity,base,boundary);
  if(geometryCache.has(key))return geometryCache.get(key);
  while(geometryCacheBytes+value.byteLength>MAX_BYTES){const oldest=geometryCache.keys().next().value;geometryCacheBytes-=geometryCache.get(oldest).byteLength;geometryCache.delete(oldest)}
  geometryCache.set(key,value);geometryCacheBytes+=value.byteLength;return value;
}
function status(message,kind=''){$('status').textContent=message;$('status').dataset.kind=kind}
function render(){renderer.render(world,camera)}
function view(name='home'){
  if(name==='top'){camera.position.set(.43,.01,1.9);orbit.target.set(.43,0,.01)}
  else if(name==='front'){camera.position.set(.43,-1.8,.6);orbit.target.set(.43,0,.16)}
  else{camera.position.set(1.5,-1.55,1.12);orbit.target.set(.43,0,.18)}
  orbit.update();render();
}
function setup(){
  renderer=new THREE.WebGLRenderer({canvas:$('canvas'),antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  world=new THREE.Scene();world.background=new THREE.Color('#edeee7');world.up.set(0,0,1);camera=new THREE.PerspectiveCamera(37,1,.01,30);camera.up.set(0,0,1);
  world.add(new THREE.HemisphereLight('#fffef6','#687568',2.5));const light=new THREE.DirectionalLight('#fff4dc',3);light.position.set(-1,-2,4);light.castShadow=true;light.shadow.mapSize.set(2048,2048);light.shadow.camera.left=-1.5;light.shadow.camera.right=1.5;light.shadow.camera.top=1.5;light.shadow.camera.bottom=-1.5;light.shadow.normalBias=.002;world.add(light);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(12,12),new THREE.MeshStandardMaterial({color:'#e2e5dc',roughness:1}));floor.position.z=-.09;floor.receiveShadow=true;world.add(floor);
  orbit=new OrbitControls(camera,$('canvas'));orbit.minDistance=.35;orbit.maxDistance=4;orbit.addEventListener('change',render);
  transform=new TransformControls(camera,$('canvas'));transform.setSize(.8);transform.showZ=false;world.add(transform.getHelper());
  transform.addEventListener('dragging-changed',event=>{orbit.enabled=!event.value});
  transform.addEventListener('change',()=>{if(transform.dragging&&mode==='edit')updateFromHandle();render()});
  $('canvas').addEventListener('pointerup',event=>{if(mode!=='edit'||transform.dragging||transform.axis)return;const rect=$('canvas').getBoundingClientRect();const pointer=new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1),ray=new THREE.Raycaster();ray.setFromCamera(pointer,camera);const hits=ray.intersectObjects([...objects.values()],true);if(hits.length){let node=hits[0].object;while(node&&!node.userData.objectId)node=node.parent;if(node)selectObject(node.userData.objectId)}});
  $('canvas').addEventListener('keydown',event=>{if(event.key.toLowerCase()==='h'){event.preventDefault();view()}if(event.code==='Space'&&mode==='replay'){event.preventDefault();$('play').click()}});
  new ResizeObserver(()=>{const rect=$('stage').getBoundingClientRect();camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();renderer.setSize(rect.width,rect.height,false);render()}).observe($('stage'));view();
}
function box(size,position,color,parent,yaw=0){
  const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),new THREE.MeshStandardMaterial({color,roughness:.83,metalness:.025}));mesh.position.fromArray(position);mesh.rotation.z=yaw;mesh.castShadow=true;mesh.receiveShadow=true;parent.add(mesh);return mesh;
}
function removeGroup(group){if(!group)return;world.remove(group);const geometries=new Set(),materials=new Set();group.traverse(child=>{if(child.geometry)geometries.add(child.geometry);if(child.material)materials.add(child.material)});for(const geometry of geometries)geometry.dispose();for(const material of materials)material.dispose()}
function pieces(obj){const [x,y,z]=obj.size_m;return obj.kind==='box'?[[[x,y,z],[0,0,z/2]]]:[[[x,y,.18*z],[0,0,.09*z]],[[.85*x,.85*y,.5*z],[0,0,.43*z]],[[.6*x,.7*y,.32*z],[0,0,.84*z]]]}
function basicRecipe(value){
  assert(value&&value.schema==='dvidia.task-scene-recipe/v1'&&value.robot_profile==='piper','Expected a versioned PiPER task scene recipe.');
  assert(value.table&&finiteArray(value.table.center_m,3)&&finiteArray(value.table.size_m,3),'Missing table geometry.');
  assert(value.bin&&finiteArray(value.bin.center_m,3)&&finiteArray(value.bin.inner_size_m,3)&&Number.isFinite(value.bin.wall_m)&&Number.isFinite(value.bin.yaw_rad),'Missing open bin geometry.');
  assert(Array.isArray(value.objects)&&value.objects.length>0&&value.objects.length<=4,'Use between one and four objects.');
  assert(value.table.center_m.every(x=>Math.abs(x)<=4)&&value.bin.center_m.every(x=>Math.abs(x)<=4)&&Math.abs(value.bin.yaw_rad)<=100*Math.PI,'Draft poses exceed the preview limits.');
  const ids=new Set();for(const obj of value.objects){assert(typeof obj.id==='string'&&/^[a-z][a-z0-9_-]{0,31}$/.test(obj.id)&&!ids.has(obj.id),'Invalid or duplicate object ID.');ids.add(obj.id);assert(['box','rigid_shoe_surrogate'].includes(obj.kind)&&finiteArray(obj.position_m,3)&&finiteArray(obj.size_m,3)&&Number.isFinite(obj.yaw_rad),'Invalid object geometry.');assert(obj.size_m.every(x=>x>0&&x<=2)&&obj.position_m.every(x=>Math.abs(x)<=4)&&value.table.size_m.every(x=>x>0&&x<=4)&&value.bin.inner_size_m.every(x=>x>0&&x<=2)&&value.bin.wall_m>0&&value.bin.wall_m<=.1,'Draft dimensions exceed the preview limits.');}
  return copy(value);
}
function drawDraft(){
  transform.detach();removeGroup(draftGroup);draftGroup=new THREE.Group();world.add(draftGroup);objects=new Map();
  box(recipe.table.size_m,recipe.table.center_m,'#cfc0a0',draftGroup);
  const bin=new THREE.Group();bin.position.fromArray(recipe.bin.center_m);bin.rotation.z=recipe.bin.yaw_rad;draftGroup.add(bin);const [x,y,h]=recipe.bin.inner_size_m,w=recipe.bin.wall_m;
  for(const [size,pos] of [[[x+2*w,y+2*w,w],[0,0,w/2]],[[w,y+2*w,h],[-(x+w)/2,0,w+h/2]],[[w,y+2*w,h],[(x+w)/2,0,w+h/2]],[[x,w,h],[0,-(y+w)/2,w+h/2]],[[x,w,h],[0,(y+w)/2,w+h/2]]])box(size,pos,'#a3afa0',bin);
  for(const obj of recipe.objects){const group=new THREE.Group();group.userData.objectId=obj.id;group.position.fromArray(obj.position_m);group.rotation.z=obj.yaw_rad;for(const [size,pos] of pieces(obj))box(size,pos,obj.kind==='box'?'#b58555':'#304b49',group);objects.set(obj.id,group);draftGroup.add(group)}
  $('object').replaceChildren(...recipe.objects.map(obj=>{const option=document.createElement('option');option.value=obj.id;option.textContent=obj.id+(obj.kind==='box'?' · block':' · rigid shoe');return option}));
  selectObject(objects.has(selectedId)?selectedId:recipe.objects[0].id);draftGroup.visible=mode==='edit';render();
}
function selectObject(id){
  selectedId=id;$('object').value=id;const obj=recipe.objects.find(o=>o.id===id),group=objects.get(id);assert(obj&&group,'Unknown selected object.');
  removeGroup(selectedOutline);selectedOutline=new THREE.BoxHelper(group,'#b7984f');world.add(selectedOutline);selectedOutline.visible=mode==='edit';
  if(mode==='edit')transform.attach(group);syncFields();render();
}
function syncFields(){
  const obj=recipe.objects.find(o=>o.id===selectedId);for(const [i,axis] of ['x','y','z'].entries())$('position-'+axis).value=(obj.position_m[i]*100).toFixed(2);
  $('yaw').value=(obj.yaw_rad*180/Math.PI).toFixed(1);$('kind').textContent=obj.kind==='box'?'One rigid box':'Three-box rigid surrogate';$('size').textContent=obj.size_m.map(x=>(x*100).toFixed(1)).join(' × ');$('mechanics').textContent=`${(obj.mass_kg*1000).toFixed(0)} g / μ ${obj.sliding_friction}`;
}
function changed(){if(selectedOutline)selectedOutline.update();status('Draft changed. Validate it before simulation.');render()}
function updateFromHandle(){
  const obj=recipe.objects.find(o=>o.id===selectedId),group=objects.get(selectedId);obj.position_m=group.position.toArray().map(x=>Math.round(x*1e6)/1e6);obj.yaw_rad=Math.atan2(Math.sin(group.rotation.z),Math.cos(group.rotation.z));syncFields();changed();
}
function updateFromFields(){
  const obj=recipe.objects.find(o=>o.id===selectedId),position=['x','y','z'].map(axis=>Number($('position-'+axis).value)/100),yaw=Number($('yaw').value)*Math.PI/180;
  if(!position.every(Number.isFinite)||position.some(v=>Math.abs(v)>4)||!Number.isFinite(yaw)){status('Enter finite metric positions and a rotation.','error');return}
  obj.position_m=position;obj.yaw_rad=Math.atan2(Math.sin(yaw),Math.cos(yaw));const group=objects.get(selectedId);group.position.fromArray(position);group.rotation.z=obj.yaw_rad;syncFields();changed();
}
function pause(){playing=false;$('play').textContent='Play recording';$('play').setAttribute('aria-pressed','false')}
function setMode(value){
  mode=value;pause();$('edit-mode').setAttribute('aria-pressed',String(value==='edit'));$('replay-mode').setAttribute('aria-pressed',String(value==='replay'));
  $('edit-panel').hidden=value!=='edit';$('replay-panel').hidden=value!=='replay';$('transport').hidden=value!=='replay';$('draft-caption').hidden=value!=='edit';
  $('mode-label').textContent=value==='edit'?'DRAFT PREVIEW · NO PHYSICS':'RECORDED NATIVE PHYSICS · ARM HELD';$('scene-hint').textContent=value==='edit'?'Select an object. Drag its arrows on the table plane. No automatic height clamping. Drag empty space to orbit · Scroll to zoom · H home':'Recorded body poses; the browser does not run physics. Drag to orbit · Scroll to zoom · Space play/pause';
  if(draftGroup)draftGroup.visible=value==='edit';if(robotGroup)robotGroup.visible=value==='edit';if(replayGroup)replayGroup.visible=value==='replay';if(selectedOutline)selectedOutline.visible=value==='edit';
  if(value==='edit'){transform.attach(objects.get(selectedId));transform.enabled=true}else{transform.detach();transform.enabled=false}render();
}
function nativeScene(model,geometry,robotOnly=false){
  assert(Array.isArray(model.bodies)&&model.bodies.length>0&&model.bodies.length<=128&&Array.isArray(model.geoms)&&model.geoms.length<=1024&&Array.isArray(model.meshes)&&model.meshes.length<=128,'Invalid native model layout.');
  const group=new THREE.Group(),bodies=model.bodies.map((body,i)=>{assert(body.id===i,'Native body IDs are not contiguous.');const b=new THREE.Group();group.add(b);return b});
  const meshes=model.meshes.map(mesh=>{for(const k of ['position_byte_offset','vertex_count','index_byte_offset','index_count'])assert(Number.isInteger(mesh[k])&&mesh[k]>=0,'Invalid native mesh offset.');assert(mesh.position_byte_offset+mesh.vertex_count*12<=geometry.byteLength&&mesh.index_byte_offset+mesh.index_count*4<=geometry.byteLength,'Native mesh exceeds its data.');const shape=new THREE.BufferGeometry(),positions=new Float32Array(geometry.slice(mesh.position_byte_offset,mesh.position_byte_offset+mesh.vertex_count*12)),indices=new Uint32Array(geometry.slice(mesh.index_byte_offset,mesh.index_byte_offset+mesh.index_count*4));assert(positions.every(Number.isFinite)&&indices.every(x=>x<mesh.vertex_count),'Invalid native mesh values.');shape.setAttribute('position',new THREE.BufferAttribute(positions,3));shape.setIndex(new THREE.BufferAttribute(indices,1));shape.computeVertexNormals();return shape});
  for(const geom of model.geoms){
    assert(finiteArray(geom.rgba,4)&&finiteArray(geom.position,3)&&finiteArray(geom.quaternion_wxyz,4)&&finiteArray(geom.size,3)&&Number.isInteger(geom.body_id)&&bodies[geom.body_id],'Invalid native geometry.');
    const task=String(geom.name).startsWith('task_');if(geom.rgba[3]===0||(geom.group===3&&!task)||(robotOnly&&task))continue;
    let shape;if(geom.type==='mesh'){shape=meshes[geom.mesh_id];assert(shape,'Native mesh ID missing.')}
    else if(geom.type==='box')shape=new THREE.BoxGeometry(...geom.size.map(x=>2*x));else if(geom.type==='sphere')shape=new THREE.SphereGeometry(geom.size[0],20,12);
    else if(geom.type==='ellipsoid'){shape=new THREE.SphereGeometry(1,20,12);shape.scale(...geom.size)}
    else if(geom.type==='cylinder'){shape=new THREE.CylinderGeometry(geom.size[0],geom.size[0],2*geom.size[1],24);shape.rotateX(Math.PI/2)}
    else if(geom.type==='capsule'){shape=new THREE.CapsuleGeometry(geom.size[0],geom.size[1]*2,8,16);shape.rotateX(Math.PI/2)}
    else if(geom.type==='plane'){if(robotOnly)continue;shape=new THREE.PlaneGeometry(4,4)}else throw Error('Unsupported native geometry: '+geom.type);
    const mesh=new THREE.Mesh(shape,new THREE.MeshStandardMaterial({color:new THREE.Color(...geom.rgba.slice(0,3)),roughness:.8,metalness:.05}));mesh.position.fromArray(geom.position);const q=geom.quaternion_wxyz;mesh.quaternion.set(q[1],q[2],q[3],q[0]);mesh.castShadow=true;mesh.receiveShadow=true;bodies[geom.body_id].add(mesh);
  }
  return {group,bodies};
}
function applyPoses(groups,view,index){const start=index*groups.length*28;for(let body=0;body<groups.length;body++){const offset=start+body*28,p=Array.from({length:7},(_,j)=>view.getFloat32(offset+j*4,true));assert(p.every(Number.isFinite),'Nonfinite native body pose.');groups[body].position.set(p[0],p[1],p[2]);groups[body].quaternion.set(p[4],p[5],p[6],p[3])}}
async function loadRobot(path){
  const url=resourceURL(path),model=await getJSON(url),geometry=await cachedGeometry(model.geometry_binary.href,model.geometry_binary,url),native=nativeScene(model,geometry,true);
  assert(Array.isArray(model.initial_poses)&&model.initial_poses.length===native.bodies.length,'Native robot initial poses missing.');
  model.initial_poses.forEach((p,i)=>{assert(finiteArray(p,7),'Invalid robot initial pose.');native.bodies[i].position.set(...p.slice(0,3));native.bodies[i].quaternion.set(p[4],p[5],p[6],p[3])});
  removeGroup(robotGroup);robotGroup=native.group;world.add(robotGroup);robotGroup.visible=mode==='edit';$('arm-label').textContent='PiPER · pinned native geometry';render();
}
function drawFrame(value){
  if(!currentCase||!poseData)return;frame=Math.max(0,Math.min(currentCase.frame_count-1,value));applyPoses(bodyGroups,poseData,frame);$('timeline').value=String(frame);$('time').textContent=(frame*currentCase.dt_s).toFixed(2)+' s';render();
}
function animate(time){if(!playing)return;drawFrame(Math.floor((time-startTime)/(currentCase.dt_s*1000)));if(frame>=currentCase.frame_count-1){pause();return}requestAnimationFrame(animate)}
async function selectRecording(){
  const epoch=++loadEpoch;pause();currentCase=null;poseData=null;bodyGroups=[];frame=0;removeGroup(replayGroup);replayGroup=null;
  $('play').disabled=true;$('timeline').disabled=true;$('timeline').value='0';$('timeline').max='0';$('time').textContent='0.00 s';
  for(const id of ['result','goal-result','expected-result','duration'])$(id).textContent='—';
  $('integrity').textContent='Loading and verifying…';$('retry-recording').hidden=true;$('replay-error').hidden=true;$('stage').dataset.replayReady='loading';
  $('loading').hidden=false;$('loading').textContent='Verifying native geometry and body poses…';render();
  try{
    const item=indexData.cases.find(c=>c.id===$('recording').value);assert(item,'Unknown recording.');assert(Number.isInteger(item.frame_count)&&item.frame_count>0&&item.frame_count<=10000&&Number.isInteger(item.body_count)&&item.body_count>0&&item.body_count<=128&&Number.isFinite(item.dt_s)&&item.dt_s>0&&item.dt_s<=1,'Invalid native motion dimensions.');
    const modelURL=resourceURL(item.scene,indexURL,indexURL),model=await getJSON(modelURL,indexURL,false,indexURL),geometry=await cachedGeometry(model.geometry_binary.href,model.geometry_binary,modelURL,indexURL),raw=await unpack(item.poses,{bytes:item.frame_count*item.body_count*28,sha256:item.sha256},indexURL,indexURL);if(epoch!==loadEpoch)return;
    assert(model.bodies.length===item.body_count,'Native motion and geometry body counts differ.');const native=nativeScene(model,geometry);removeGroup(replayGroup);replayGroup=native.group;bodyGroups=native.bodies;world.add(replayGroup);replayGroup.visible=mode==='replay';currentCase=item;poseData=new DataView(raw);
    if(!robotGroup){const robot=nativeScene(model,geometry,true);applyPoses(robot.bodies,poseData,0);robotGroup=robot.group;world.add(robotGroup);robotGroup.visible=mode==='edit';$('arm-label').textContent='PiPER · pinned native geometry'}
    const metrics=item.metrics||{};$('result').textContent=metrics.checks_passed===true?'Check passed':metrics.checks_passed===false?'Check failed':'Recorded run';$('goal-result').textContent=metrics.native_goal_success===true?'Complete':metrics.native_goal_success===false?'Incomplete':'Not reported';$('expected-result').textContent=metrics.expected_goal_success===true?'Complete':metrics.expected_goal_success===false?'Incomplete':'Exploratory run';$('duration').textContent=((item.frame_count-1)*item.dt_s).toFixed(2)+' s';$('integrity').textContent='Geometry + poses · SHA-256 verified';
    $('timeline').max=String(item.frame_count-1);$('play').disabled=false;$('timeline').disabled=false;drawFrame(0);$('loading').hidden=true;$('stage').dataset.replayReady='true';
  }catch(error){if(epoch!==loadEpoch)return;$('loading').hidden=true;$('result').textContent='Recording unavailable';$('integrity').textContent='Not verified';$('replay-error').textContent='The selected recording could not load: '+error.message;$('replay-error').hidden=false;$('retry-recording').hidden=false;$('stage').dataset.replayReady='error';render();throw error}
}
async function loadIndex(path){
  const url=resourceURL(path),data=await getJSON(url);assert(data.schema==='dvidia.task-scene-replay/v2'&&Array.isArray(data.cases)&&data.cases.length>0&&data.cases.length<=128,'Unsupported recorded benchmark.');
  const ids=new Set();for(const item of data.cases){assert(typeof item.id==='string'&&item.id.length<=80&&!ids.has(item.id),'Duplicate or invalid recording.');ids.add(item.id)}
  indexData=data;indexURL=url;$('recording').replaceChildren(...data.cases.map(item=>{const option=document.createElement('option');option.value=item.id;option.textContent=item.label||item.id;return option}));$('replay-mode').disabled=false;$('index-link').href=url.href;$('index-link').hidden=false;
  const checked=data.cases.filter(c=>c.metrics?.checks_passed===true).length;$('study-summary').textContent=`${checked}/${data.cases.length} recorded case checks passed. ${Array.isArray(data.prechecks)?data.prechecks.length:0} recipe prechecks are listed in the benchmark data.`;if(!local)setMode('replay');await selectRecording();
}
async function simulate(){
  if(localBusy||!local)return;pause();localBusy=true;$('simulate').disabled=true;$('simulate').textContent='Simulating…';status('Validating the recipe, then running native physics.');
  try{
    const response=await fetch(resourceURL('/api/simulate'),{method:'POST',headers:{'Content-Type':'application/json'},credentials:'omit',body:JSON.stringify({recipe})});assert(response.body,'Simulation response missing.');const text=await response.text();assert(text.length<=65536,'Simulation response too large.');let result;try{result=JSON.parse(text)}catch{throw Error('The local simulation returned an unreadable response.')};assert(response.ok,result.error||'The recipe could not be simulated.');assert(typeof result.index_url==='string','Simulation response has no evidence link.');
    await loadIndex(result.index_url);setMode('replay');status('A fresh native run is ready. The original draft remains editable.','success');
  }catch(error){setMode('edit');status(error.message,'error')}finally{localBusy=false;$('simulate').disabled=false;$('simulate').textContent='Validate & simulate'}
}
function bind(){
  $('object').addEventListener('change',()=>selectObject($('object').value));for(const id of ['position-x','position-y','position-z','yaw'])$(id).addEventListener('change',updateFromFields);
  $('rotate').addEventListener('click',()=>{const rotating=transform.mode!=='rotate';transform.setMode(rotating?'rotate':'translate');transform.showX=!rotating;transform.showY=!rotating;transform.showZ=rotating;$('rotate').setAttribute('aria-pressed',String(rotating));$('rotate').textContent=rotating?'Move handle':'Rotate handle';render()});
  $('edit-mode').addEventListener('click',()=>setMode('edit'));$('replay-mode').addEventListener('click',()=>setMode('replay'));document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>view(button.dataset.view)));
  $('recording').addEventListener('change',()=>selectRecording().catch(error=>status(error.message,'error')));$('play').addEventListener('click',()=>{if(playing){pause();return}if(!currentCase)return;if(frame===currentCase.frame_count-1)drawFrame(0);playing=true;startTime=performance.now()-frame*currentCase.dt_s*1000;$('play').textContent='Pause';$('play').setAttribute('aria-pressed','true');requestAnimationFrame(animate)});$('timeline').addEventListener('input',()=>{pause();drawFrame(Number($('timeline').value))});document.addEventListener('visibilitychange',()=>{if(document.hidden)pause()});
  $('retry-recording').addEventListener('click',()=>selectRecording().catch(error=>status(error.message,'error')));
  $('simulate').addEventListener('click',simulate);$('import').addEventListener('click',()=>$('recipe-file').click());$('recipe-file').addEventListener('change',async()=>{const file=$('recipe-file').files[0];if(!file)return;try{assert(file.size<=65536,'The recipe must be at most 64 KB.');const value=basicRecipe(JSON.parse(await file.text()));recipe=value;setMode('edit');drawDraft();status('Recipe imported as an unvalidated draft.','success')}catch(error){status(error.message,'error')}finally{$('recipe-file').value=''}});
  $('export').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(recipe,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),anchor=document.createElement('a');anchor.href=url;anchor.download='recipe.json';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status('Draft recipe exported. Server validation has not been implied.');});
  $('reset-draft').addEventListener('click',()=>{recipe=copy(originalRecipe);setMode('edit');drawDraft();status('Draft reset to the source recipe.')});
}
async function main(){
  assert(location.protocol!=='file:','Serve this directory over HTTP, or start the local CLI.');setup();bind();let config=null;
  if(['127.0.0.1','localhost','[::1]'].includes(location.hostname)){try{config=await getJSON('/api/config',rootURL,true)}catch{config=null}}
  recipe=basicRecipe(config?.recipe||await getJSON('recipe.json'));originalRecipe=copy(recipe);$('recipe-link').href=resourceURL('recipe.json').href;drawDraft();setMode('edit');local=config?.local===true;
  if(local){$('simulate').disabled=false;$('local-note').textContent='Runs a fresh CPU simulation on this machine. The arm stays at home; this checks the environment and placement goal.';$('local-command').textContent='dvidia-environment studio --assets ./robot-assets --output ./local-scenes --port '+location.port;}
  $('loading').hidden=true;$('stage').dataset.ready='true';
  if(config?.robot_scene){try{await loadRobot(config.robot_scene)}catch(error){$('arm-label').textContent='Native arm geometry unavailable';status(error.message,'error')}}
  try{await loadIndex(config?.index_url||'benchmark/index.json');if(!local)setMode('replay')}catch(error){if(!robotGroup)$('arm-label').textContent='Native arm geometry unavailable';if(!local)status('Published replay unavailable: '+error.message,'error');$('loading').hidden=true;render()}
}
main().catch(error=>{$('loading').hidden=false;$('loading').textContent='The workstation could not open: '+error.message;$('stage').dataset.ready='error'});
