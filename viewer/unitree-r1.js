import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const canvas = document.querySelector('#view');
const position = document.querySelector('#position');
const status = document.querySelector('#status');
const sliders = document.querySelector('#joints');
const buttons = [...document.querySelectorAll('[data-side]')];
const materials = [0x526059, 0xcbd3c9, 0xc4cfc7, 0xaebfb5, 0xc8d4cc, 0xa9b8ae, 0xc5d3c9, 0x52665b]
  .map(color => new THREE.MeshStandardMaterial({color, roughness:.58, metalness:.24}));
const openingPose = [0,1.3,2.6,1.6,0,.12,0];
const radians = Math.PI / 180;
let model, renderer, camera, controls, scene, loadId = 0;
const models = new Map();

function draw() {
  renderer.render(scene, camera);
}

function start() {
  renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0xe9ece6);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(35, 1, .01, 10);
  camera.up.set(0,0,1);
  camera.position.set(.77,-.83,.59);
  controls = new OrbitControls(camera, canvas);
  controls.target.set(0,0,.24);
  controls.enablePan = false;
  controls.minDistance = .26;
  controls.maxDistance = 2.8;
  controls.addEventListener('change', draw);
  controls.update();
  scene.add(new THREE.HemisphereLight(0xfffefa, 0xc1cac2, 2.2));
  const key = new THREE.DirectionalLight(0xfff9ed, 2.2);
  key.position.set(1,-1.2,2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 1);
  fill.position.set(-1,1,.6);
  scene.add(fill);
  const grid = new THREE.GridHelper(1.6,32,0xb6c2b8,0xd1dbd2);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -.002;
  scene.add(grid);
  new ResizeObserver(() => {
    const {width,height} = canvas.getBoundingClientRect();
    if (!height || !width) return;
    renderer.setSize(width,height,false);
    camera.aspect = width/height;
    camera.updateProjectionMatrix();
    draw();
  }).observe(canvas.parentNode);
}

function setOrigin(object, item) {
  object.position.fromArray(item.xyz);
  object.rotation.set(...item.rpy,'ZYX');
}

async function getAsset(path, method) {
  const response = await fetch(path);
  if (!response.ok) throw new Error('Arm asset is unavailable');
  return response[method]();
}

async function build(side) {
  const spec = await getAsset(side+'.json','json');
  const data = await getAsset(spec.binary,'arrayBuffer');
  const group = new THREE.Group(), frames = new Map();
  spec.links.forEach((link, i) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(data,link.position_byte_offset,3*link.vertex_count),3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(data,link.index_byte_offset,link.index_count),1));
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry,materials[i]);
    setOrigin(mesh,link);
    const frame = new THREE.Group();
    frame.add(mesh);
    frames.set(link.name,frame);
  });
  group.add(frames.get('base_link'));
  const joints = spec.joints.map(joint => {
    const origin = new THREE.Group(), rotating = new THREE.Group();
    setOrigin(origin,joint);
    frames.get(joint.parent).add(origin);
    origin.add(rotating);
    rotating.add(frames.get(joint.child));
    return {...joint,rotating,unitAxis:new THREE.Vector3(...joint.axis).normalize()};
  });
  const wrist = frames.get('Link7');
  wrist.add(new THREE.AxesHelper(.045));
  return {group,joints,wrist,side};
}

function updateJoint(joint, angle) {
  joint.rotating.quaternion.setFromAxisAngle(joint.unitAxis,angle);
  joint.slider.value = String(angle);
  joint.output.value = (angle / radians).toFixed(1)+'°';
}

function showPose() {
  const p = model.wrist.getWorldPosition(new THREE.Vector3());
  position.textContent = 'Link7 · '+p.toArray().map((v,i)=> 'XYZ'[i]+' '+v.toFixed(3)).join('   ')+' m';
  draw();
}

function resetPose() {
  if (!model) return;
  model.joints.forEach((joint,i) => updateJoint(joint,openingPose[i]));
  showPose();
}

async function select(side) {
  const token = ++loadId;
  for (const button of buttons) button.setAttribute('aria-pressed',String(button.dataset.side===side));
  if (model) scene.remove(model.group);
  model = null;
  sliders.replaceChildren();
  position.textContent='Wrist position · metres';
  status.textContent='Loading '+side+' arm…';
  draw();
  try {
    if (!models.has(side)) models.set(side,build(side));
    const next = await models.get(side);
    if (token !== loadId) return;
    scene.add(next.group);
    model = next;
    for (const joint of next.joints) {
      const row = document.createElement('label');
      row.className='joint';
      const name=document.createElement('span');
      name.textContent=joint.name;
      const slider=document.createElement('input'), output=document.createElement('output');
      slider.type='range'; slider.min=joint.lower; slider.max=joint.upper; slider.step='0.001';
      slider.setAttribute('aria-label',joint.name+' angle');
      slider.addEventListener('input',()=>{ updateJoint(joint,slider.valueAsNumber); showPose(); });
      joint.slider=slider; joint.output=output;
      row.append(name,slider,output);
      sliders.append(row);
    }
    const receipt=document.querySelector('#receipt');
    receipt.href='benchmark-'+side+'.json';
    receipt.textContent=(side==='left'?'Left':'Right')+'-arm benchmark';
    status.textContent='';
    resetPose();
  } catch {
    models.delete(side);
    if (token===loadId) status.textContent='Model could not load. Choose an arm to retry.';
  }
}

try {
  start();
  buttons.forEach(button=>button.addEventListener('click',()=>select(button.dataset.side)));
  document.querySelector('#reset').addEventListener('click',resetPose);
  select('right');
} catch {
  status.textContent='This viewer requires WebGL 2. The numeric benchmark is available below.';
}
