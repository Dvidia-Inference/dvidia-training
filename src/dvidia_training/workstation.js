/* DVIDIA live workstation. MIT. Rendering follows native worker body poses. */
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {TransformControls} from 'three/addons/controls/TransformControls.js';

const $ = id => document.getElementById(id);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const MAX_BYTES = 64 * 1024 * 1024;
const sessionStart = performance.now();
const openedAt = new Date().toISOString();
const receipt = {commands: [], samples: [], commands_dropped: 0, samples_dropped: 0};
let manifest, model, catalog, worker, renderer, world, camera, orbit, transform, transformHelper;
let bodies = [], collisionMeshes = [], target, actualTool, targetLine, shoeBody;
let latest, sceneReady = false, workerReady = false, ready = false, failed = false;
let scheduled = false, renderCount = 0, stateCount = 0, lastStateAt = 0, lastMetricsAt = -Infinity;
let lastSampleAt = -Infinity, lastSentTargetAt = -Infinity, pendingTarget = null, targetTimer = null;
let manipulating = false, settingTarget = false, awaitingPause = false, lastPause = null;
let manifestDigest = null, sceneBounds, jawRange, renderedPoses = 0;
let trackingErrorSince = null, reportedTarget = null, trackingLimited = false;
let targetDraftDirty = false;

function vec(value, count) {
  assert((Array.isArray(value) || ArrayBuffer.isView(value)) && value.length === count
    && Array.from(value).every(Number.isFinite), 'The simulation supplied an invalid vector.');
  return value;
}
function relative(path) {
  assert(typeof path === 'string' && /^[a-zA-Z0-9_.-]+(?:\/[a-zA-Z0-9_.-]+)*$/.test(path)
    && !path.split('/').some(part => part === '.' || part === '..'), 'Invalid local resource path.');
  return path;
}
async function bounded(stream, limit) {
  assert(stream, 'The server returned an empty resource.');
  const reader = stream.getReader(), chunks = [];
  let length = 0;
  try {
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > limit) { await reader.cancel(); throw new Error('Resource exceeds the workstation byte limit.'); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const output = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.length; }
  return output.buffer;
}
async function fetchBytes(path, limit) {
  const url = new URL(relative(path), location.href);
  const hosted = location.protocol === 'https:' && location.host === 'dvidia-robot-simulation-lab.static.hf.space';
  const response = await fetch(url, {credentials: 'omit', referrerPolicy: 'no-referrer', redirect: hosted ? 'follow' : 'error'});
  const final = new URL(response.url);
  assert(final.origin === location.origin || (hosted && final.protocol === 'https:' && final.host === 'us.aws.cdn.hf.co'),
    'A resource redirected outside the workstation host.');
  assert(response.ok, 'Could not load ' + path + ' (HTTP ' + response.status + ').');
  return bounded(response.body, limit);
}
async function digest(bytes) {
  if (!globalThis.crypto?.subtle) return null;
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), n => n.toString(16).padStart(2, '0')).join('');
}
async function loadJSON(path) {
  return JSON.parse(new TextDecoder().decode(await fetchBytes(path, 4 * 1024 * 1024)));
}
async function loadGeometry(meta) {
  assert(meta && meta.compression === 'gzip' && meta.href === manifest.geometry_url
    && [meta.bytes, meta.compressed_bytes].every(x => Number.isInteger(x) && x > 0 && x <= MAX_BYTES),
  'Invalid compiled geometry descriptor.');
  assert(typeof DecompressionStream === 'function', 'This browser needs gzip stream support.');
  const packed = await fetchBytes(meta.href, meta.compressed_bytes);
  assert(packed.byteLength === meta.compressed_bytes, 'The geometry download is incomplete.');
  const bytes = await bounded(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip')), meta.bytes);
  assert(bytes.byteLength === meta.bytes, 'The decoded geometry has an unexpected size.');
  const checksum = await digest(bytes);
  assert(checksum === null || checksum === meta.sha256, 'The compiled geometry checksum does not match.');
  return bytes;
}
function validateGeometry(bytes) {
  assert(model.schema === 'dvidia.live-workstation-model/v1' && Array.isArray(model.bodies)
    && model.bodies.length > 0 && model.bodies.length <= 128, 'Unsupported native model geometry.');
  model.bodies.forEach((body, index) => assert(body.id === index && typeof body.name === 'string', 'Invalid body identity.'));
  assert(Array.isArray(model.meshes) && model.meshes.length <= 128 && Array.isArray(model.geoms)
    && model.geoms.length <= 512, 'Invalid geometry inventory.');
  const ids = new Set(), view = new DataView(bytes);
  for (const mesh of model.meshes) {
    const {vertex_count: n, index_count: m, position_byte_offset: v, index_byte_offset: f} = mesh;
    assert(Number.isInteger(mesh.id) && !ids.has(mesh.id) && [n, m, v, f].every(Number.isInteger)
      && n > 0 && m >= 0 && m % 3 === 0 && v >= 0 && f >= 0 && v % 4 === 0 && f % 4 === 0
      && v + n * 12 <= bytes.byteLength && f + m * 4 <= bytes.byteLength, 'Invalid native mesh range.');
    ids.add(mesh.id);
    for (let i = 0; i < n * 3; i++) assert(Number.isFinite(view.getFloat32(v + i * 4, true)), 'Nonfinite native mesh.');
    for (let i = 0; i < m; i++) assert(view.getUint32(f + i * 4, true) < n, 'Mesh index outside vertices.');
  }
  for (const geom of model.geoms) {
    assert(Number.isInteger(geom.body_id) && geom.body_id >= 0 && geom.body_id < model.bodies.length
      && ['mesh', 'box', 'sphere', 'ellipsoid', 'capsule', 'cylinder', 'plane'].includes(geom.type), 'Invalid native geometry.');
    for (const [key, n] of [['position', 3], ['quaternion_wxyz', 4], ['size', 3], ['rgba', 4]]) vec(geom[key], n);
    assert(geom.type !== 'mesh' || ids.has(geom.mesh_id), 'Missing native mesh.');
  }
}
function nativeQuaternion(q, output) { output.set(q[1], q[2], q[3], q[0]); }
function primitive(geom, meshes) {
  switch (geom.type) {
    case 'mesh': return meshes.get(geom.mesh_id);
    case 'box': return new THREE.BoxGeometry(...geom.size.map(n => n * 2));
    case 'sphere': return new THREE.SphereGeometry(geom.size[0], 28, 20);
    case 'ellipsoid': { const shape = new THREE.SphereGeometry(1, 48, 32); shape.scale(...geom.size); return shape; }
    case 'capsule': { const shape = new THREE.CapsuleGeometry(geom.size[0], 2 * geom.size[1], 8, 24); shape.rotateX(Math.PI / 2); return shape; }
    case 'cylinder': { const shape = new THREE.CylinderGeometry(geom.size[0], geom.size[0], geom.size[1] * 2, 36); shape.rotateX(Math.PI / 2); return shape; }
    case 'plane': return new THREE.PlaneGeometry(12, 12);
  }
}
function texture(kind) {
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 512;
  const ctx = canvas.getContext('2d');
  let seed = 2718;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  ctx.fillStyle = kind === 'wood' ? '#cdb794' : '#d6d6ce'; ctx.fillRect(0, 0, 512, 512);
  if (kind === 'wood') {
    for (let y = 0; y < 512; y++) {
      const alpha = .018 + random() * .075;
      ctx.strokeStyle = 'rgba(89,60,35,' + alpha + ')'; ctx.lineWidth = .4 + random() * 1.4;
      ctx.beginPath(); ctx.moveTo(0, y);
      for (let x = 0; x <= 512; x += 16) ctx.lineTo(x, y + Math.sin(x * .016 + y * .045) * (1 + random() * 2));
      ctx.stroke();
    }
    for (const y of [126, 256, 386]) {
      ctx.fillStyle = 'rgba(70,53,32,.13)'; ctx.fillRect(0, y, 512, 1);
      ctx.fillStyle = 'rgba(255,255,235,.2)'; ctx.fillRect(0, y + 1, 512, 1);
    }
  } else {
    for (let y = 0; y < 512; y += 4) for (let x = 0; x < 512; x += 4) {
      ctx.fillStyle = 'rgba(' + (random() > .5 ? '255,255,255,' : '40,43,42,') + (.025 + random() * .085) + ')';
      ctx.fillRect(x, y, 2, 3);
    }
  }
  const map = new THREE.CanvasTexture(canvas);
  map.colorSpace = THREE.SRGBColorSpace; map.wrapS = map.wrapT = THREE.RepeatWrapping;
  map.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  if (kind !== 'wood') map.repeat.set(3, 2);
  return map;
}
function labelTexture(text, detail) {
  const canvas = document.createElement('canvas'); canvas.width = 1024; canvas.height = 256;
  const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, 1024, 256);
  ctx.fillStyle = '#526657'; ctx.font = '500 43px sans-serif'; ctx.fillText(text, 40, 113);
  ctx.fillStyle = '#75816a'; ctx.font = '25px monospace'; ctx.fillText(detail, 41, 167);
  const map = new THREE.CanvasTexture(canvas); map.colorSpace = THREE.SRGBColorSpace; return map;
}
function dressShoe(shoe, fabric) {
  // Original visual mesh stays within the catalog shoe's 280 x 100 x 90 mm
  // bounds. Native box/ellipsoid collision, inertia and dynamics are unchanged.
  const asset = catalog.shoes.find(item => item.id === manifest.shoe_asset_ids[0]);
  const upper = asset?.geometry.find(item => item.name === 'upper');
  if (!upper) return;
  const material = color => new THREE.MeshStandardMaterial({color, roughness: .92, metalness: 0});
  const add = (geometry, surface, z = 0) => {
    const mesh = new THREE.Mesh(geometry, surface); mesh.position.z = z;
    mesh.castShadow = true; mesh.receiveShadow = true; shoe.add(mesh); return mesh;
  };
  const outline = new THREE.Shape();
  outline.moveTo(-.131, -.03);
  outline.bezierCurveTo(-.139, -.023, -.139, .027, -.129, .035);
  outline.bezierCurveTo(-.106, .046, -.065, .044, -.035, .041);
  outline.bezierCurveTo(.007, .044, .045, .048, .085, .047);
  outline.bezierCurveTo(.122, .046, .137, .029, .136, .003);
  outline.bezierCurveTo(.136, -.026, .125, -.044, .101, -.045);
  outline.bezierCurveTo(.05, -.047, .017, -.041, -.03, -.037);
  outline.bezierCurveTo(-.071, -.038, -.113, -.043, -.131, -.03);
  const extrude = (depth, bevel) => new THREE.ExtrudeGeometry(outline, {
    depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel,
    bevelSegments: 3, steps: 1, curveSegments: 18
  });
  add(extrude(.003, .001), material(0x4a5756), .001);
  add(extrude(.011, .002), material(0xe8e4d3), .006);
  const thread = material(0xa4afa4);
  const rim = outline.getPoints(90).map(point => new THREE.Vector3(point.x, point.y, .0155));
  add(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(rim, true), 100, .0006, 4, true), thread);
  // Loft an upper over the two authored ellipsoid profiles, with an actual
  // open visual collar instead of drawing a black decal on a closed surface.
  const cy = upper.position_m[1], base = .019;
  const profile = x => {
    let height = base, width = .0001;
    for (const component of asset.geometry.filter(item => item.type === 'ellipsoid')) {
      const u = (x - component.position_m[0]) / component.size_m[0];
      if (Math.abs(u) <= 1) {
        const k = Math.sqrt(1 - u * u);
        width = Math.max(width, component.size_m[1] * k);
        height = Math.max(height, component.position_m[2] + component.size_m[2] * k);
      }
    }
    return {height, width};
  };
  const roof = (x, y) => {
    const p = profile(x), cross = Math.sqrt(Math.max(0, 1 - ((y - cy) / p.width) ** 2));
    return base + (p.height - base) * Math.pow(cross, .72);
  };
  const hole = (x, y) => ((x + .079) / .027) ** 2 + ((y - cy) / .017) ** 2 < 1;
  const position = [], uv = [], index = [], colors = [];
  const nx = 96, ny = 48;
  const navy = new THREE.Color(0x334d5a), heel = new THREE.Color(0x718279), toe = new THREE.Color(0xc4cdc2);
  for (let i = 0; i <= nx; i++) {
    const x = -.127 + .262 * i / nx, p = profile(x);
    for (let j = 0; j <= ny; j++) {
      const angle = Math.PI * j / ny, y = cy + p.width * Math.cos(angle);
      const z = base + (p.height - base) * Math.pow(Math.sin(angle), .72);
      position.push(x, y, z); uv.push(i / nx, j / ny);
      const color = x < -.046 ? heel : x > .064 ? toe : navy;
      colors.push(color.r, color.g, color.b);
    }
  }
  const includeTriangle = (a, b, c) => {
    const x = (position[a * 3] + position[b * 3] + position[c * 3]) / 3;
    const y = (position[a * 3 + 1] + position[b * 3 + 1] + position[c * 3 + 1]) / 3;
    if (!hole(x, y)) index.push(a, b, c);
  };
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    const a = i * (ny + 1) + j, b = a + ny + 1;
    includeTriangle(a, b, a + 1); includeTriangle(b, b + 1, a + 1);
  }
  const upperGeometry = new THREE.BufferGeometry();
  upperGeometry.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
  upperGeometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  upperGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  upperGeometry.setIndex(index); upperGeometry.computeVertexNormals();
  add(upperGeometry, new THREE.MeshStandardMaterial({map: fabric, vertexColors: true, roughness: .95, side: THREE.DoubleSide}));
  const collar = [];
  for (let i = 0; i <= 64; i++) {
    const angle = i / 64 * Math.PI * 2, x = -.079 + .027 * Math.cos(angle), y = cy + .017 * Math.sin(angle);
    collar.push(new THREE.Vector3(x, y, roof(x, y) + .0003));
  }
  add(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(collar, true), 80, .0022, 6, true), material(0x9daaa0));
  const inner = new THREE.SphereGeometry(1, 30, 20); inner.scale(.026, .016, .012);
  const lining = add(inner, material(0x243532)); lining.position.set(-.079, cy, .047);
  const tongue = [], tongueUV = [], tongueIndices = [];
  for (let i = 0; i <= 14; i++) for (let j = 0; j <= 6; j++) {
    const x = -.047 + i * .006, y = cy - .016 + j * .032 / 6;
    tongue.push(x, y, Math.min(.087, roof(x, y) + .0006)); tongueUV.push(i / 14, j / 6);
  }
  for (let i = 0; i < 14; i++) for (let j = 0; j < 6; j++) {
    const a = i * 7 + j, b = a + 7; tongueIndices.push(a, b, a + 1, b, b + 1, a + 1);
  }
  const tongueGeometry = new THREE.BufferGeometry();
  tongueGeometry.setAttribute('position', new THREE.Float32BufferAttribute(tongue, 3));
  tongueGeometry.setAttribute('uv', new THREE.Float32BufferAttribute(tongueUV, 2));
  tongueGeometry.setIndex(tongueIndices); tongueGeometry.computeVertexNormals();
  add(tongueGeometry, new THREE.MeshStandardMaterial({map: fabric, color: 0x7d8e85, roughness: 1, side: THREE.DoubleSide}));
  const laces = material(0xeae7d5), eyeletMaterial = new THREE.MeshStandardMaterial({color: 0x8e9d92, metalness: .25, roughness: .6});
  for (let row = 0; row < 5; row++) {
    const x = -.038 + row * .015, width = .020;
    for (const side of [-1, 1]) {
      const y = cy + side * width;
      const eyelet = add(new THREE.TorusGeometry(.0023, .00055, 5, 12), eyeletMaterial);
      eyelet.position.set(x, y, roof(x, y) + .001);
    }
    const points = [
      new THREE.Vector3(x, cy - width, roof(x, cy - width) + .0015),
      new THREE.Vector3(x + .004, cy, Math.min(.088, roof(x + .004, cy) + .0013)),
      new THREE.Vector3(x + .005, cy + width, roof(x + .005, cy + width) + .0015)
    ];
    add(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 14, .0012, 5, false), laces);
  }
  for (const x of [-.046, .064]) {
    const p = profile(x), panel = [];
    for (let i = 0; i <= 32; i++) {
      const angle = i / 32 * Math.PI, y = cy + p.width * Math.cos(angle);
      panel.push(new THREE.Vector3(x, y, roof(x, y) + .0005));
    }
    add(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(panel), 36, .00055, 4, false), thread);
  }
}
function authoredDetails(shoe, fabric) {
  dressShoe(shoe, fabric);
  // Original screen-printed worktop markings; physical surface remains the native box.
  const print = new THREE.Mesh(new THREE.PlaneGeometry(.32, .08),
    new THREE.MeshBasicMaterial({map: labelTexture('D V I D I A', 'MANIPULATION / STATION 01'), transparent: true, depthWrite: false}));
  print.position.set(-.18, -.34, .0008); world.add(print);
  const ruler = new THREE.Group();
  const rulerMat = new THREE.LineBasicMaterial({color: 0x8c896e, transparent: true, opacity: .6});
  const points = [];
  for (let i = 0; i <= 30; i++) {
    const x = .15 + i * .018;
    points.push(new THREE.Vector3(x, -.405, .0008), new THREE.Vector3(x, -.405 + (i % 5 === 0 ? .016 : .007), .0008));
  }
  ruler.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), rulerMat)); world.add(ruler);
  const trayOutline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(.19, .255, .113), new THREE.Vector3(.55, .255, .113),
    new THREE.Vector3(.55, .401, .113), new THREE.Vector3(.19, .401, .113)
  ]), new THREE.LineBasicMaterial({color: 0x8c795e, transparent: true, opacity: .4}));
  world.add(trayOutline);
}
function requestRender() {
  if (scheduled || !renderer || failed || document.hidden) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    if (failed) return;
    renderer.render(world, camera);
    $('canvas').dataset.renderCount = String(++renderCount);
  });
}
function cameraPreset(name) {
  if (!camera) return;
  const center = new THREE.Vector3(...manifest.camera.target);
  const portrait = Math.max(1, .95 / camera.aspect);
  const offsets = {
    perspective: new THREE.Vector3(...manifest.camera.position).sub(center),
    top: new THREE.Vector3(.001, -.001, 1.8),
    front: new THREE.Vector3(0, -1.85, .38)
  };
  camera.position.copy(center).add((offsets[name] || offsets.perspective).clone().multiplyScalar(portrait));
  orbit.target.copy(center); orbit.update();
  document.querySelectorAll('[data-camera]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.camera === name)));
  requestRender();
}
function setupGeometry(bytes) {
  assert($('canvas').getContext('webgl2'), 'WebGL2 is unavailable in this browser.');
  renderer = new THREE.WebGLRenderer({canvas: $('canvas'), antialias: true, powerPreference: 'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  world = new THREE.Scene(); world.background = new THREE.Color(0xe9e9de);
  world.fog = new THREE.Fog(0xe9e9de, 4, 11);
  camera = new THREE.PerspectiveCamera(37, 1, .01, 20); camera.up.set(0, 0, 1);
  orbit = new OrbitControls(camera, $('canvas'));
  orbit.minDistance = .25; orbit.maxDistance = 4.5; orbit.maxPolarAngle = Math.PI * .48;
  orbit.screenSpacePanning = true; orbit.addEventListener('change', requestRender);
  world.add(new THREE.HemisphereLight(0xfff8e9, 0xb1b5a5, 2.3));
  const sun = new THREE.DirectionalLight(0xfff1d6, 2.7);
  sun.position.set(-1.1, -1.3, 3.5); sun.target.position.set(.2, 0, 0);
  sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -1.4; sun.shadow.camera.right = 1.4;
  sun.shadow.camera.top = 1.4; sun.shadow.camera.bottom = -1.4;
  sun.shadow.camera.near = .1; sun.shadow.camera.far = 7;
  sun.shadow.normalBias = .0008; sun.shadow.bias = -.00008;
  sun.shadow.radius = 4; world.add(sun, sun.target);
  const fill = new THREE.DirectionalLight(0xdce9f5, .9);
  fill.position.set(1, 1, 2); world.add(fill);
  bodies = model.bodies.map(() => { const group = new THREE.Group(); world.add(group); return group; });
  const meshes = new Map(), view = new DataView(bytes), wood = texture('wood'), fabric = texture('fabric');
  for (const source of model.meshes) {
    const positions = new Float32Array(source.vertex_count * 3), indices = new Uint32Array(source.index_count);
    for (let i = 0; i < positions.length; i++) positions[i] = view.getFloat32(source.position_byte_offset + i * 4, true);
    for (let i = 0; i < indices.length; i++) indices[i] = view.getUint32(source.index_byte_offset + i * 4, true);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1)); geometry.computeVertexNormals();
    meshes.set(source.id, geometry);
  }
  let visualCount = 0;
  for (const geom of model.geoms) {
    const geometry = primitive(geom, meshes), name = geom.name || '';
    const isShoe = manifest.shoe_body_names.includes(model.bodies[geom.body_id].name);
    if (geom.rgba[3] > 0 && geom.group <= 2 && !isShoe) {
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color().setRGB(...geom.rgba.slice(0, 3), THREE.LinearSRGBColorSpace),
        metalness: .05, roughness: .66, transparent: geom.rgba[3] < 1, opacity: geom.rgba[3]
      });
      if (name === manifest.environment_geoms.table || name === manifest.environment_geoms.rack) {
        material.color.set(0xe9dcc1); material.map = wood; material.roughness = .78; material.metalness = 0;
      } else if (name === manifest.environment_geoms.floor) {
        material.color.set(0xd9ded1); material.roughness = 1; material.metalness = 0;
      } else if (isShoe) {
        material.color.set(name.includes('sole') ? 0xece8d7 : name.includes('heel') ? 0x788785 : 0x334d5e);
        material.roughness = .97; material.metalness = 0;
        if (!name.includes('sole')) material.map = fabric;
      } else if (!name.startsWith('workstation_') && geom.type === 'mesh') {
        const brightness = (geom.rgba[0] + geom.rgba[1] + geom.rgba[2]) / 3;
        material.color.set(brightness > .35 ? 0xe3e1d7 : 0x223c4c);
        material.roughness = .5; material.metalness = .15;
      }
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.fromArray(geom.position); nativeQuaternion(geom.quaternion_wxyz, mesh.quaternion);
      mesh.castShadow = geom.type !== 'plane'; mesh.receiveShadow = true;
      bodies[geom.body_id].add(mesh); visualCount++;
    }
    if (geom.collision && geom.type !== 'plane') {
      const wire = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
        color: isShoe ? 0xe3a66d : 0x649d91, wireframe: true, transparent: true, opacity: .38, depthWrite: false
      }));
      wire.position.fromArray(geom.position); nativeQuaternion(geom.quaternion_wxyz, wire.quaternion);
      wire.visible = false; wire.renderOrder = 3; bodies[geom.body_id].add(wire); collisionMeshes.push(wire);
    }
  }
  const shoeIndex = model.bodies.findIndex(body => body.name === manifest.shoe_body_names[0]);
  assert(shoeIndex >= 0, 'The native shoe is missing.'); shoeBody = bodies[shoeIndex];
  authoredDetails(shoeBody, fabric);
  target = new THREE.Group(); world.add(target);
  const marker = new THREE.Mesh(new THREE.SphereGeometry(.008, 20, 14),
    new THREE.MeshBasicMaterial({color: 0x769f74, transparent: true, opacity: .7, depthTest: false}));
  marker.renderOrder = 10; target.add(marker);
  actualTool = new THREE.Mesh(new THREE.SphereGeometry(.0035, 16, 12), new THREE.MeshBasicMaterial({color: 0xdee9da}));
  world.add(actualTool);
  targetLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
    new THREE.LineDashedMaterial({color: 0x819b71, dashSize: .008, gapSize: .005, transparent: true, opacity: .6}));
  world.add(targetLine);
  transform = new TransformControls(camera, $('canvas')); transform.setMode('translate'); transform.setSpace('world'); transform.setSize(.65);
  transform.attach(target); transformHelper = transform.getHelper(); world.add(transformHelper);
  transform.addEventListener('dragging-changed', event => {
    manipulating = event.value; orbit.enabled = !event.value;
    if (!event.value) flushTarget();
    requestRender();
  });
  transform.addEventListener('objectChange', () => {
    if (settingTarget || !ready) return;
    const position = boundedTarget(target.position.toArray());
    targetDraftDirty = false; $('target-error').textContent = '';
    target.position.fromArray(position); showTarget(position, true); queueTarget(position);
    requestRender();
  });
  transform.addEventListener('change', requestRender);
  new ResizeObserver(() => {
    const rect = $('stage').getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;
    const old = camera.aspect; camera.aspect = rect.width / rect.height;
    camera.position.sub(orbit.target).multiplyScalar(Math.max(1, .95 / camera.aspect) / Math.max(1, .95 / old)).add(orbit.target);
    camera.updateProjectionMatrix(); renderer.setSize(rect.width, rect.height, false); requestRender();
  }).observe($('stage'));
  const rect = $('stage').getBoundingClientRect(); camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix(); renderer.setSize(rect.width, rect.height, false); cameraPreset('perspective');
  $('canvas').addEventListener('webglcontextlost', event => { event.preventDefault(); fail(new Error('WebGL was interrupted. Reload the workstation to restart.')); });
  $('stage').dataset.visualGeoms = String(visualCount); $('stage').dataset.collisionGeoms = String(collisionMeshes.length);
  sceneReady = true;
}
function appendBounded(array, value, limit, droppedKey) {
  if (array.length === limit) { array.shift(); receipt[droppedKey]++; }
  array.push(value);
}
function command(value) {
  if (!ready || failed) return;
  worker.postMessage(value);
  appendBounded(receipt.commands, {wall_ms: performance.now() - sessionStart, sim_time: latest?.sim_time ?? null, ...value}, 2000, 'commands_dropped');
}
function boundedTarget(position) { return position.map((value, axis) => clamp(value, ...sceneBounds[axis])); }
function showTarget(position, force = false) {
  for (const [index, axis] of ['x', 'y', 'z'].entries()) {
    const input = $('target-' + axis);
    if (force || (!targetDraftDirty && document.activeElement !== input)) input.value = String(Math.round(position[index] * 1000));
  }
  $('height').value = String(Math.round(position[2] * 1000));
  $('height-value').textContent = (position[2] * 1000).toFixed(0) + ' mm';
}
function setTarget(position, send = true) {
  vec(position, 3);
  const bounded = boundedTarget(position);
  targetDraftDirty = false;
  settingTarget = true; target.position.fromArray(bounded); settingTarget = false;
  showTarget(bounded, true); $('target-error').textContent = '';
  if (send) queueTarget(bounded);
  requestRender();
}
function queueTarget(position) {
  pendingTarget = position.slice();
  const wait = Math.max(0, 34 - (performance.now() - lastSentTargetAt));
  if (!targetTimer) targetTimer = setTimeout(flushTarget, wait);
}
function flushTarget() {
  clearTimeout(targetTimer); targetTimer = null;
  if (pendingTarget) { command({type: 'target', position: pendingTarget}); pendingTarget = null; lastSentTargetAt = performance.now(); }
}
function setJaw(value, send = true) {
  const jaw = clamp(value, ...jawRange);
  $('gripper').value = String(jaw * 1000); $('gripper-value').textContent = (jaw * 1000).toFixed(1) + ' mm';
  document.querySelector('.gripper-diagram').style.setProperty('--jaw-visual', (7 + jaw / jawRange[1] * 18) + 'px');
  if (send) command({type: 'gripper', jaw});
}
function pause(paused) {
  if (!ready || awaitingPause) return;
  awaitingPause = true; $('pause').disabled = true;
  command({type: 'pause', paused});
}
function format(value, digits = 1) { return Number.isFinite(value) ? value.toFixed(digits) : '—'; }
function setStatus(state) {
  if (failed) return;
  const paused = state.paused === true;
  const mode = paused ? 'paused' : 'running';
  if ($('session').dataset.state !== mode) {
    $('session').dataset.state = mode;
    $('session-label').textContent = paused ? 'Simulation paused' : 'Simulation running';
  }
  $('pause').setAttribute('aria-pressed', String(paused));
  $('pause-label').textContent = paused ? 'Resume' : 'Pause';
  $('pause-icon').setAttribute('d', paused ? 'M6 4 16 10 6 16Z' : 'M7 5v10M13 5v10');
  $('live-status').textContent = paused ? 'Physics paused · camera stays active' : 'Live native physics';
  if (lastPause !== paused) { $('announcer').textContent = paused ? 'Simulation paused.' : 'Simulation running.'; lastPause = paused; }
  awaitingPause = false; $('pause').disabled = !ready;
}
function applyState(state) {
  const positions = state.body_pos, quaternions = state.body_quat;
  vec(positions, bodies.length * 3); vec(quaternions, bodies.length * 4);
  vec(state.tcp_position, 3); vec(state.target_position, 3);
  assert(Number.isFinite(state.sim_time) && state.sim_time >= 0, 'Invalid native simulation time.');
  if (!reportedTarget || state.target_position.some((value, index) => Math.abs(value - reportedTarget[index]) > 1e-9)) {
    reportedTarget = state.target_position.slice(); trackingErrorSince = null;
  }
  if (state.tracking_error_m > .005) {
    if (trackingErrorSince === null || state.sim_time < trackingErrorSince) trackingErrorSince = state.sim_time;
  } else trackingErrorSince = null;
  const limited = trackingErrorSince !== null && state.sim_time - trackingErrorSince >= 4;
  if (limited && !trackingLimited) $('announcer').textContent = 'The tool has not reached within five millimeters of this target after four simulated seconds. Try a closer target.';
  trackingLimited = limited;
  bodies.forEach((body, index) => {
    body.position.fromArray(positions, index * 3);
    const offset = index * 4;
    const norm = Math.hypot(...quaternions.slice(offset, offset + 4));
    assert(Math.abs(norm - 1) < .002, 'Invalid native body rotation.');
    body.quaternion.set(quaternions[offset + 1], quaternions[offset + 2], quaternions[offset + 3], quaternions[offset]);
  });
  actualTool.position.fromArray(state.tcp_position);
  if (!manipulating && pendingTarget === null && performance.now() - lastSentTargetAt > 150) {
    settingTarget = true; target.position.fromArray(state.target_position); settingTarget = false;
    showTarget(state.target_position);
  }
  const points = targetLine.geometry.attributes.position;
  points.setXYZ(0, ...state.tcp_position); points.setXYZ(1, ...target.position.toArray());
  points.needsUpdate = true; targetLine.computeLineDistances(); targetLine.geometry.computeBoundingSphere();
  renderedPoses++;
  $('stage').dataset.stateCount = String(stateCount); $('stage').dataset.simTime = String(state.sim_time);
  $('stage').dataset.renderedPoses = String(renderedPoses);
  const now = performance.now();
  if (now - lastSampleAt >= 100) {
    appendBounded(receipt.samples, {
      wall_ms: now - sessionStart, sim_time: state.sim_time, paused: state.paused,
      tcp_position: state.tcp_position, target_position: state.target_position,
      body_pos: state.body_pos, body_quat: state.body_quat, qpos: state.qpos, qvel: state.qvel,
      ctrl: state.ctrl, actuator_force: state.actuator_force, jaw_target_m: state.jaw,
      jaw_actual_m: state.actual_jaw, jaw_commanded_m: state.commanded_jaw,
      contacts: state.contacts, contact_normal_force_total_n: state.contact_normal_force_total_n,
      contact_normal_force_max_n: state.contact_normal_force_max_n,
      tracking_error_m: state.tracking_error_m, metrics: state.metrics, warnings: state.warnings
    }, 3000, 'samples_dropped');
    lastSampleAt = now;
  }
  setStatus(state);
  if (now - lastMetricsAt >= 150) {
    const time = state.sim_time, minutes = Math.floor(time / 60);
    $('sim-time').textContent = String(minutes).padStart(2, '0') + ':' + (time % 60).toFixed(2).padStart(5, '0');
    $('tracking-error').textContent = format(state.tracking_error_m * 1000);
    $('contacts').textContent = Number.isInteger(state.contacts) ? String(state.contacts) : '—';
    const metrics = state.metrics || {};
    $('step-p95').textContent = format(metrics.step_ms_p95, 2);
    $('rtf').textContent = format(metrics.rtf, 2);
    $('control-p95').textContent = format(metrics.control_ms_p95, 2) + ' ms';
    $('lag').textContent = format(metrics.lag_ms, 0) + ' ms';
    $('actual-jaw').textContent = format(state.actual_jaw * 1000, 1) + ' mm';
    $('contact-force').textContent = format(state.contact_normal_force_total_n, 2) + ' N';
    $('state-count').textContent = String(stateCount);
    $('receipt-count').textContent = String(receipt.samples.length);
    $('engine-version').textContent = state.engine || 'MuJoCo WASM';
    const warnings = (state.warnings || []).map(String).join(' · ');
    $('warning').hidden = !warnings; $('warning').textContent = warnings;
    $('tool-state').lastChild.textContent = limited ? ' Target not reached' : state.tracking_error_m > .005 ? ' Approaching' : ' Tracking';
    $('tool-state').dataset.state = limited ? 'limited' : 'tracking';
    $('tool-state').title = limited ? 'Target error remains above 5 mm after 4 simulated seconds. Workspace bounds do not guarantee reachability.' : 'Measured native tool position compared with the requested target.';
    lastMetricsAt = now;
  }
  requestRender();
}
function becomeReady() {
  if (!sceneReady || !workerReady || ready || failed) return;
  ready = true;
  document.querySelectorAll('button,input').forEach(element => { element.disabled = false; });
  setTarget(manifest.initial_target_xyz, false);
  setJaw(latest?.jaw ?? manifest.home_ctrl[manifest.gripper_actuator_ids[0]], false);
  $('loading').hidden = true; $('stage').dataset.ready = 'true';
  if (latest) applyState(latest);
  if (document.hidden) pause(true);
}
function receiptDownload() {
  flushTarget();
  const output = {
    schema: 'dvidia.live-workstation-session/v1', opened_at: openedAt, exported_at: new Date().toISOString(),
    source: {manifest_sha256: manifestDigest, compiled_model_sha256: manifest.compiled_model_sha256,
      geometry_sha256: model.geometry_binary.sha256, native_version: manifest.native_version,
      robot: manifest.robot, shoe_asset_ids: manifest.shoe_asset_ids},
    scope: 'Local browser interaction with privileged native simulator state. No learned policy, physical robot control, or grasp qualification.',
    controller: manifest.controller, bounds_xyz: manifest.bounds_xyz, timestep: manifest.timestep,
    control_dt: manifest.control_dt, interaction_limits: manifest.interaction_limits,
    recording: {sample_wall_interval_ms: 100, sample_limit: 3000, command_limit: 2000,
      received_states: stateCount, samples_dropped: receipt.samples_dropped, commands_dropped: receipt.commands_dropped,
      note: 'Bounded recent samples and commands; wall timestamps continue across reset. Simulation time restarts on reset. UI samples may omit intervening physics steps.'},
    latest_metrics: latest?.metrics ?? null, commands: receipt.commands, samples: receipt.samples
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(output, null, 2) + '\n'], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url;
  link.download = 'dvidia-workstation-session-' + new Date().toISOString().replace(/[:.]/g, '-') + '.json';
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
  $('announcer').textContent = 'Session receipt downloaded. Measurements remain on your device.';
}
function connect() {
  for (const axis of ['x', 'y', 'z']) {
    $('target-' + axis).addEventListener('input', () => {
      targetDraftDirty = true;
      $('target-error').textContent = 'Coordinates edited. Apply target to send.';
    });
  }
  document.querySelectorAll('[data-camera]').forEach(button => button.addEventListener('click', () => cameraPreset(button.dataset.camera)));
  $('home-camera').addEventListener('click', () => cameraPreset('perspective'));
  $('focus-shoe').addEventListener('click', () => {
    orbit.target.copy(shoeBody.position).add(new THREE.Vector3(0, 0, .05));
    camera.position.copy(orbit.target).add(new THREE.Vector3(.48, -.65, .42).multiplyScalar(Math.max(1, .7 / camera.aspect)));
    orbit.update(); document.querySelectorAll('[data-camera]').forEach(button => button.setAttribute('aria-pressed', 'false'));
    requestRender();
  });
  $('gizmo').addEventListener('click', () => {
    const visible = $('gizmo').getAttribute('aria-pressed') !== 'true';
    $('gizmo').setAttribute('aria-pressed', String(visible));
    transformHelper.visible = visible; transform.enabled = visible; target.visible = visible; targetLine.visible = visible;
    requestRender();
  });
  $('collision').addEventListener('click', () => {
    const visible = $('collision').getAttribute('aria-pressed') !== 'true';
    $('collision').setAttribute('aria-pressed', String(visible));
    collisionMeshes.forEach(mesh => { mesh.visible = visible; });
    $('scene-caption').textContent = visible ? 'Native collision wireframes over visual geometry.' : 'One arm. One object. Interactive local simulation.';
    requestRender();
  });
  $('target-form').addEventListener('submit', event => {
    event.preventDefault();
    const position = ['x', 'y', 'z'].map(axis => $('target-' + axis).valueAsNumber / 1000);
    if (!position.every((value, axis) => Number.isFinite(value) && value >= sceneBounds[axis][0] - 1e-9 && value <= sceneBounds[axis][1] + 1e-9)) {
      $('target-error').textContent = 'Use coordinates inside the displayed workspace bounds.'; return;
    }
    setTarget(position); flushTarget();
  });
  $('reset-target').addEventListener('click', () => {
    if (!latest) return; setTarget(latest.tcp_position); flushTarget();
    if (latest.tcp_position.some((n, i) => n < sceneBounds[i][0] || n > sceneBounds[i][1])) {
      $('target-error').textContent = 'The current tool is outside the command volume; target moved to the nearest allowed position.';
    }
  });
  $('height').addEventListener('input', () => setTarget([target.position.x, target.position.y, $('height').valueAsNumber / 1000]));
  $('height').addEventListener('change', flushTarget);
  $('gripper').addEventListener('input', () => setJaw($('gripper').valueAsNumber / 1000));
  $('pause').addEventListener('click', () => pause(!latest?.paused));
  $('reset').addEventListener('click', () => {
    clearTimeout(targetTimer); targetTimer = null; pendingTarget = null;
    command({type: 'reset'}); setTarget(manifest.initial_target_xyz, false);
    setJaw(manifest.home_ctrl[manifest.gripper_actuator_ids[0]], false);
    $('announcer').textContent = 'Reset requested. The original arm and shoe state will be restored.';
  });
  $('push').addEventListener('click', () => {
    const impulse = Math.min(.12, manifest.interaction_limits.push_impulse_norm_max_ns);
    command({type: 'push', body: manifest.shoe_body_names[0], impulse: [0, impulse, 0]});
    $('announcer').textContent = latest?.paused
      ? 'Nudge queued. Resume the simulation to apply it.'
      : 'Nudge requested: ' + impulse.toFixed(2) + ' newton seconds toward positive Y.';
  });
  $('receipt').addEventListener('click', receiptDownload);
  $('canvas').addEventListener('keydown', event => {
    if (!ready) return;
    if (['h', 'H', 'Home'].includes(event.key)) { event.preventDefault(); cameraPreset('perspective'); }
    else if (event.key === ' ') { event.preventDefault(); pause(!latest?.paused); }
    else if (['+', '=', '-', '_'].includes(event.key)) {
      event.preventDefault();
      const offset = camera.position.clone().sub(orbit.target);
      offset.setLength(clamp(offset.length() * (['+', '='].includes(event.key) ? .9 : 1 / .9), orbit.minDistance, orbit.maxDistance));
      camera.position.copy(orbit.target).add(offset); orbit.update(); requestRender();
    }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && ready && !latest?.paused) pause(true);
    if (!document.hidden) requestRender();
  });
  window.addEventListener('pagehide', () => worker?.terminate());
}
function fail(error) {
  if (failed) return;
  failed = true; ready = false; worker?.terminate(); clearTimeout(targetTimer);
  $('session').dataset.state = 'error'; $('session-label').textContent = 'Simulation unavailable';
  $('loading').hidden = false; $('loading').classList.add('error');
  $('loading-title').textContent = 'The workstation could not continue';
  $('loading-message').textContent = error?.message || String(error);
  $('stage').dataset.ready = 'error';
  $('live-status').textContent = 'Physics stopped';
  document.querySelectorAll('button,input').forEach(element => { element.disabled = true; });
  // Preserve the session's evidence if an engine or renderer failure occurs.
  if (receipt.samples.length) $('receipt').disabled = false;
  $('announcer').textContent = $('loading-message').textContent;
}
async function main() {
  assert(['http:', 'https:'].includes(location.protocol), 'Serve the exported workstation over local HTTP or HTTPS.');
  assert(typeof Worker === 'function', 'This browser does not support local physics workers.');
  const manifestBytes = await fetchBytes('workstation.json', 4 * 1024 * 1024);
  manifest = JSON.parse(new TextDecoder().decode(manifestBytes)); manifestDigest = await digest(manifestBytes);
  assert(manifest.schema === 'dvidia.live-workstation/v1' && manifest.physical_robot_ready === false
    && manifest.learned_policy === false, 'Unsupported workstation manifest.');
  assert(Array.isArray(manifest.bounds_xyz) && manifest.bounds_xyz.length === 3, 'Missing target workspace bounds.');
  sceneBounds = manifest.bounds_xyz.map(bounds => { vec(bounds, 2); assert(bounds[1] > bounds[0], 'Invalid workspace bounds.'); return bounds; });
  jawRange = vec(manifest.controller.gripper_range_m, 2);
  vec(manifest.initial_target_xyz, 3); vec(manifest.camera.position, 3); vec(manifest.camera.target, 3);
  assert(Number.isFinite(manifest.interaction_limits.push_impulse_norm_max_ns) && manifest.interaction_limits.push_impulse_norm_max_ns > 0,
    'Missing physical disturbance limit.');
  for (const [index, axis] of ['x', 'y', 'z'].entries()) {
    $('target-' + axis).min = String(sceneBounds[index][0] * 1000);
    $('target-' + axis).max = String(sceneBounds[index][1] * 1000);
    $('target-' + axis).title = Math.round(sceneBounds[index][0] * 1000) + ' to ' + Math.round(sceneBounds[index][1] * 1000) + ' mm';
  }
  $('height').min = String(sceneBounds[2][0] * 1000); $('height').max = String(sceneBounds[2][1] * 1000);
  $('height-min').textContent = Math.round(sceneBounds[2][0] * 1000) + ' mm';
  $('height-max').textContent = Math.round(sceneBounds[2][1] * 1000) + ' mm';
  $('gripper').min = String(jawRange[0] * 1000); $('gripper').max = String(jawRange[1] * 1000);
  $('physics-step').textContent = format(manifest.timestep * 1000, 0) + ' ms';
  worker = new Worker('./workstation_worker.bundle.js', {type: 'module'});
  worker.addEventListener('error', event => fail(new Error(event.message || 'The local physics worker stopped unexpectedly.')));
  worker.addEventListener('message', event => {
    try {
      const message = event.data;
      if (message.type === 'error') { fail(new Error(message.message)); return; }
      if (message.type === 'ready' || message.type === 'state') {
        if (!message.state) return;
        latest = message.state; stateCount++; lastStateAt = performance.now();
        if (message.type === 'ready') workerReady = true;
        if (ready) applyState(latest); else becomeReady();
        worker.postMessage({type: 'ack'});
      } else if (message.type === 'progress') {
        $('loading-message').textContent = String(message.message || 'Preparing the native physics engine.');
      }
    } catch (error) { fail(error); }
  });
  worker.postMessage({type: 'init', manifest_url: new URL('workstation.json', location.href).href});
  [model, catalog] = await Promise.all([loadJSON(manifest.model_url), loadJSON('catalog.json')]);
  const geometry = await loadGeometry(model.geometry_binary); validateGeometry(geometry);
  setupGeometry(geometry); connect();
  const asset = catalog.shoes.find(item => item.id === manifest.shoe_asset_ids[0]);
  assert(asset, 'The shoe is missing from the asset catalog.');
  $('object-name').textContent = asset.name;
  $('object-mass').textContent = format(asset.properties.mass_kg.value * 1000, 0) + ' g';
  $('object-friction').textContent = format(asset.properties.sliding_friction.value, 2) + ' μ';
  becomeReady();
  setInterval(() => {
    if (!ready || failed || latest?.paused || document.hidden) return;
    if (lastStateAt && performance.now() - lastStateAt > 2000) {
      $('live-status').textContent = 'Waiting for physics updates';
      $('session').dataset.state = 'loading'; $('session-label').textContent = 'Waiting for physics';
    }
  }, 1000);
}
main().catch(fail);
