/* DVIDIA USD workstation. MIT. Recorded native physics; no robot commands. */
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const $=id=>document.getElementById(id),assert=(ok,message)=>{if(!ok)throw Error(message)};
let renderer,camera,world,controls,bodies,meta,selected,poses,index=0,playing=false,began=0,epoch=0;
async function bytes(name,limit=8*1024*1024){
  assert(/^[a-zA-Z0-9_.-]+(?:\/[a-zA-Z0-9_.-]+)*$/.test(name)&&!name.split('/').includes('..'),'Invalid local resource.');
  const r=await fetch(new URL(name,location.href),{credentials:'omit',referrerPolicy:'no-referrer'}),url=new URL(r.url);
  assert(url.origin===location.origin||(location.host==='dvidia-robot-simulation-lab.static.hf.space'&&url.protocol==='https:'&&url.hostname.endsWith('.hf.co')),'Unexpected resource host.');
  assert(r.ok,'Could not open '+name);const reader=r.body.getReader(),chunks=[];let length=0;
  try{for(;;){const {done,value}=await reader.read();if(done)break;length+=value.byteLength;assert(length<=limit,'Resource exceeds its byte limit.');chunks.push(value)}}finally{reader.cancel();reader.releaseLock()}
  const out=new Uint8Array(length);let offset=0;for(const chunk of chunks){out.set(chunk,offset);offset+=chunk.byteLength}return out;
}
const json=async name=>JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(await bytes(name)));
async function unpack(name,identity){
  assert(Number.isInteger(identity.bytes)&&identity.bytes>0&&identity.bytes<=64*1024*1024,'Invalid native byte count.');
  const packed=await bytes(name,16*1024*1024),reader=new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip')).getReader();
  const parts=[];let length=0;try{for(;;){const {done,value}=await reader.read();if(done)break;length+=value.length;assert(length<=identity.bytes,'Inflated native resource too large.');parts.push(value)}}finally{reader.cancel();reader.releaseLock()}
  const raw=new Uint8Array(length);let offset=0;for(const p of parts){raw.set(p,offset);offset+=p.length}
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(x=>x.toString(16).padStart(2,'0')).join('');
  assert(raw.length===identity.bytes&&hash===identity.sha256,'Native bytes failed verification.');return raw.buffer;
}
function render(){renderer.render(world,camera)}
function home(){camera.position.set(1.45,-1.45,1.0);controls.target.set(.36,.08,.22);controls.update();render()}
function build(model,geometry){
  renderer=new THREE.WebGLRenderer({canvas:$('canvas'),antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=true;
  world=new THREE.Scene();world.background=new THREE.Color('#e9e8e0');world.up.set(0,0,1);
  camera=new THREE.PerspectiveCamera(37,1,.01,30);camera.up.set(0,0,1);
  world.add(new THREE.HemisphereLight('#ffffff','#62685f',2));const light=new THREE.DirectionalLight('#fff4dc',3);light.position.set(-1,-2,4);world.add(light);
  bodies=model.bodies.map((b,i)=>{assert(b.id===i,'Invalid native body order.');const g=new THREE.Group();world.add(g);return g});
  const meshes=model.meshes.map(m=>{const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(geometry.slice(m.position_byte_offset,m.position_byte_offset+m.vertex_count*12)),3));g.setIndex(new THREE.BufferAttribute(new Uint32Array(geometry.slice(m.index_byte_offset,m.index_byte_offset+m.index_count*4)),1));g.computeVertexNormals();return g});
  for(const g of model.geoms){
    if(g.rgba[3]===0||(g.group===3&&g.name!=='usd_object_geom'&&g.name!=='usd_table'))continue;
    let shape;
    if(g.type==='mesh')shape=meshes[g.mesh_id];
    else if(g.type==='box')shape=new THREE.BoxGeometry(...g.size.map(x=>x*2));
    else if(g.type==='sphere')shape=new THREE.SphereGeometry(g.size[0],20,12);
    else if(g.type==='cylinder'){shape=new THREE.CylinderGeometry(g.size[0],g.size[0],g.size[1]*2,20);shape.rotateX(Math.PI/2)}
    else if(g.type==='capsule'){shape=new THREE.CapsuleGeometry(g.size[0],g.size[1]*2,8,16);shape.rotateX(Math.PI/2)}
    else if(g.type==='plane')shape=new THREE.PlaneGeometry(4,4);
    else throw Error('Unsupported native geometry.');
    const m=new THREE.Mesh(shape,new THREE.MeshStandardMaterial({color:new THREE.Color(...g.rgba.slice(0,3)),roughness:.8,metalness:.05}));
    m.position.fromArray(g.position);m.quaternion.set(g.quaternion_wxyz[1],g.quaternion_wxyz[2],g.quaternion_wxyz[3],g.quaternion_wxyz[0]);bodies[g.body_id].add(m);
  }
  controls=new OrbitControls(camera,$('canvas'));controls.addEventListener('change',render);home();
  new ResizeObserver(()=>{const r=$('viewport').getBoundingClientRect();camera.aspect=r.width/r.height;camera.updateProjectionMatrix();renderer.setSize(r.width,r.height,false);render()}).observe($('viewport'));
}
function draw(i){
  index=Math.max(0,Math.min(selected.rows.length-1,i));const row=selected.rows[index],start=index*bodies.length*28;
  for(let b=0;b<bodies.length;b++){const p=start+b*28;const values=Array.from({length:7},(_,j)=>poses.getFloat32(p+j*4,true));assert(values.every(Number.isFinite),'Nonfinite native pose.');bodies[b].position.set(...values.slice(0,3));bodies[b].quaternion.set(values[4],values[5],values[6],values[3])}
  $('timeline').value=String(index);$('time').textContent=row.t_s.toFixed(2)+' s';$('height').textContent=(row.position_m[2]*100).toFixed(1)+' cm';$('speed').textContent=Math.hypot(...row.linear_velocity_m_s).toFixed(3)+' m/s';$('contact').textContent=row.table_contact?'Table contact':'No table contact';render();
}
function pause(){playing=false;$('play').textContent='Play recording'}
function animate(now){if(!playing)return;draw(Math.floor((now-began)/10));if(index>=selected.rows.length-1){pause();return}requestAnimationFrame(animate)}
async function select(){
  const current=++epoch;pause();$('play').disabled=true;$('loading').hidden=false;
  const c=meta.cases.find(c=>c.case.id===$('case').value);assert(c,'Unknown case.');assert(/^cases\/case-[0-9]{2}\/poses\.bin\.gz$/.test(c.pose_file),'Invalid pose path.');
  const raw=await unpack(c.pose_file,c.pose_identity);if(current!==epoch)return;
  assert(raw.byteLength===c.rows.length*meta.body_count*28,'Native frame layout differs.');selected=c;poses=new DataView(raw);
  $('timeline').max=String(c.rows.length-1);$('result').textContent=c.result.success?'Physics checks passed':'Physics checks failed';$('result').dataset.passed=String(c.result.success);
  $('properties').textContent=`${(c.case.mass_kg*1000).toFixed(0)} g · friction ${c.case.friction} · ${c.case.control.replaceAll('_',' ')}`;
  $('recipe').href='cases/'+c.case.id+'/input/recipe.json';$('usd').href='cases/'+c.case.id+'/input/workstation.usda';$('trace').href='cases/'+c.case.id+'/trajectory.json';
  $('play').disabled=false;$('timeline').disabled=false;$('loading').hidden=true;draw(0);
}
async function main(){
  assert(location.protocol!=='file:','Serve the exported folder over HTTP.');const [model,motion,report]=await Promise.all(['model.json','motion.json','report.json'].map(json));
  assert(model.schema==='dvidia.usd-native-replay/v1'&&motion.schema==='dvidia.usd-workstation-benchmark/v1','Unsupported USD replay.');meta=motion;
  assert(meta.body_count===model.bodies.length&&meta.cases.length===45,'Incomplete native replay roster.');
  build(model,await unpack('geometry.bin.gz',model.geometry_binary));$('summary').textContent=`${report.passed}/${report.scheduled} physics checks · ${report.negative_passed}/${report.negative_scheduled} negative controls · CPU simulation`;
  for(const c of meta.cases){const option=document.createElement('option');option.value=c.case.id;option.textContent=`${c.case.kind==='drop'?'Drop':'Slide'} · ${(c.case.mass_kg*1000).toFixed(0)} g · μ ${c.case.friction} · ${c.case.control.replaceAll('_',' ')}`;$('case').append(option)}
  $('case').disabled=false;$('case').addEventListener('change',()=>select().catch(fail));$('home').addEventListener('click',home);
  $('play').addEventListener('click',()=>{if(playing){pause();return}if(index===selected.rows.length-1)draw(0);playing=true;began=performance.now()-index*10;$('play').textContent='Pause';requestAnimationFrame(animate)});
  $('timeline').addEventListener('input',()=>{pause();draw(Number($('timeline').value))});$('canvas').addEventListener('keydown',e=>{if(e.key===' '){e.preventDefault();$('play').click()}if(e.key.toLowerCase()==='h'){e.preventDefault();home()}});document.addEventListener('visibilitychange',()=>{if(document.hidden)pause()});
  await select();$('viewport').dataset.ready='true';
}
function fail(e){pause();$('loading').hidden=false;$('loading').textContent='The recording could not open: '+e.message+' Numerical evidence is available below.';$('viewport').dataset.ready='error'}
main().catch(fail);
