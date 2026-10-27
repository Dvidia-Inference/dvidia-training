/* Recorded native PiPER states only. No physics, inference, telemetry or remote imports. */
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const $ = id => document.getElementById(id);
const MAX_BYTES = 64 * 1024 * 1024;
const labels = {learned: 'Learned policy', teacher: 'Authored teacher', open_jaw: 'Learned arm · open jaw', hold: 'Hold still · open jaw', shuffled_labels: 'Shuffled-label learner'};
const finite = value => typeof value === 'number' && Number.isFinite(value);
let model, metadata, poseBytes, renderer, camera, scene, controls, bodies, selected;
let frame = 0, playing = false, requested = false, originTime = 0, originClock = 0;
const focus = new THREE.Vector3(.25, 0, .25);

function assert(condition, message) { if (!condition) throw new Error(message); }
function vector(value, size, label) { assert(Array.isArray(value) && value.length === size && value.every(finite), `Invalid ${label}.`); }
function integer(value, low, high, label) { assert(Number.isInteger(value) && value >= low && value <= high, `Invalid ${label}.`); }
function localHref(href) {
  assert(typeof href === 'string' && /^[a-zA-Z0-9_.-]+(?:\/[a-zA-Z0-9_.-]+)*$/.test(href) && !href.split('/').some(p => p === '.' || p === '..'), 'Study links must stay inside the exported directory.');
  return href;
}
const HF_SPACE_HOST = 'dvidia-robot-simulation-lab.static.hf.space';
const HF_VIEWER_PATH = '/learned-placement/v01/viewer/';
const HF_BINARY_ROOT = 'https://huggingface.co/spaces/Dvidia/robot-simulation-lab/resolve/main/learned-placement/v01/viewer/';
function studyUrl(href) {
  const path = localHref(href);
  const hosted = location.protocol === 'https:' && location.host === HF_SPACE_HOST && location.pathname.startsWith(HF_VIEWER_PATH);
  // Static Spaces do not expand Git LFS pointers. The Hub resolves these
  // public binary files; local and research exports remain self-contained.
  return new URL(path, hosted && /\.(?:gz|bin|npz)$/.test(path) ? HF_BINARY_ROOT : location.href);
}
function trustedResponseUrl(final, requested) {
  if (final.username || final.password) return false;
  if (requested.origin === location.origin) return final.origin === location.origin;
  if (!requested.href.startsWith(HF_BINARY_ROOT)) return false;
  return final.href === requested.href || (final.protocol === 'https:' && final.host === 'us.aws.cdn.hf.co');
}
async function response(href, signal) {
  const requestedUrl = studyUrl(href);
  const sameOrigin = requestedUrl.origin === location.origin;
  const result = await abortable(fetch(requestedUrl, {credentials: sameOrigin ? 'same-origin' : 'omit', referrerPolicy: 'no-referrer', redirect: sameOrigin ? 'error' : 'follow', signal}), signal);
  const final = new URL(result.url);
  assert(trustedResponseUrl(final, requestedUrl), 'Unexpected study download origin.');
  assert(result.ok, `Could not load ${href} (${result.status}).`);
  return result;
}
async function abortable(promise, signal) {
  if (!signal) return await promise;
  if (signal.aborted) throw signal.reason;
  let listener;
  const aborted = new Promise((resolve, reject) => {
    listener = () => reject(signal.reason);
    signal.addEventListener('abort', listener, {once: true});
  });
  try { return await Promise.race([promise, aborted]); }
  finally { signal.removeEventListener('abort', listener); }
}
async function deadline(href, milliseconds, operation) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error(`Timed out loading ${href} after ${milliseconds / 1000} s. Check the connection or open the complete download locally.`)), milliseconds);
  try { return await abortable(operation(controller.signal), controller.signal); }
  finally { clearTimeout(timer); }
}
async function bounded(stream, limit, signal) {
  assert(stream, 'Study response is empty.');
  const reader = stream.getReader(), chunks = []; let size = 0, finished = false;
  try {
    for (;;) {
      const part = await abortable(reader.read(), signal); if (part.done) { finished = true; break; }
      size += part.value.byteLength;
      if (size > limit) throw new Error('Study file exceeds its byte bound.');
      chunks.push(part.value);
    }
  } finally {
    // A stalled source must not keep startup waiting for cancellation either.
    if (!finished) void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
  const bytes = new Uint8Array(size); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return bytes.buffer;
}
async function json(href) {
  return deadline(href, 30000, async signal => JSON.parse(new TextDecoder().decode(await bounded((await response(href, signal)).body, MAX_BYTES, signal))));
}
async function verifiedBinary(bytes, meta) {
  assert(bytes.byteLength === meta.bytes, 'Binary byte count changed.');
  assert(globalThis.crypto?.subtle, 'SHA-256 verification requires HTTPS or a localhost server.');
  const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(n => n.toString(16).padStart(2, '0')).join('');
  assert(hash === meta.sha256, 'Study binary SHA256 changed.');
  return bytes;
}
async function binary(meta, expected) {
  assert(meta && meta.href === expected, 'Unexpected binary file.');
  integer(meta.bytes, 0, MAX_BYTES, 'binary byte count');
  if (typeof DecompressionStream === 'function' && meta.compressed_href) {
    assert(meta.compression === 'gzip', 'Unsupported binary compression.');
    integer(meta.compressed_bytes, 0, MAX_BYTES, 'compressed byte count');
    try {
      return await deadline(meta.compressed_href, 8000, async signal => {
        const compressed = await bounded((await response(meta.compressed_href, signal)).body, meta.compressed_bytes, signal);
        assert(compressed.byteLength === meta.compressed_bytes, 'Compressed byte count changed.');
        const bytes = await bounded(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'), {signal}), meta.bytes, signal);
        return verifiedBinary(bytes, meta);
      });
    } catch (error) {
      // The raw bytes have their own deadline and must pass the same digest.
      // A blocked or stalled compressed response cannot strand the replay.
    }
  }
  return deadline(meta.href, 30000, async signal => verifiedBinary(await bounded((await response(meta.href, signal)).body, meta.bytes, signal), meta));
}
function validate(geometry, motion) {
  assert(model.schema === 'dvidia.piper-imitation-scene/v1' && metadata.schema === 'dvidia.piper-imitation-viewer/v1', 'Unsupported study replay format.');
  integer(metadata.body_count, 1, 128, 'body count');
  assert(Array.isArray(model.bodies) && model.bodies.length === metadata.body_count && metadata.frame_stride_floats === metadata.body_count * 7, 'Body pose layout mismatch.');
  model.bodies.forEach((body, index) => assert(body.id === index && typeof body.name === 'string', 'Body IDs must match native order.'));
  assert(Array.isArray(model.meshes) && model.meshes.length <= 128 && Array.isArray(model.geoms) && model.geoms.length <= 512, 'Geometry inventory exceeds bounds.');
  const dv = new DataView(geometry), meshIds = new Set();
  for (const mesh of model.meshes) {
    integer(mesh.id, 0, 128, 'mesh id'); assert(!meshIds.has(mesh.id), 'Duplicate mesh ID.'); meshIds.add(mesh.id);
    integer(mesh.vertex_count, 1, MAX_BYTES / 12, 'vertex count'); integer(mesh.index_count, 3, MAX_BYTES / 4, 'index count');
    integer(mesh.position_byte_offset, 0, geometry.byteLength, 'vertex offset'); integer(mesh.index_byte_offset, 0, geometry.byteLength, 'index offset');
    assert(mesh.position_byte_offset % 4 === 0 && mesh.index_byte_offset % 4 === 0 && mesh.index_count % 3 === 0 && mesh.position_byte_offset + mesh.vertex_count * 12 <= geometry.byteLength && mesh.index_byte_offset + mesh.index_count * 4 <= geometry.byteLength, 'Invalid mesh binary bounds.');
    for (let i = 0; i < mesh.vertex_count * 3; i++) assert(finite(dv.getFloat32(mesh.position_byte_offset + i * 4, true)), 'Nonfinite mesh vertex.');
    for (let i = 0; i < mesh.index_count; i++) assert(dv.getUint32(mesh.index_byte_offset + i * 4, true) < mesh.vertex_count, 'Mesh index outside vertices.');
  }
  for (const geom of model.geoms) {
    integer(geom.body_id, 0, metadata.body_count - 1, 'geometry body');
    vector(geom.position, 3, 'geometry position'); vector(geom.quaternion_wxyz, 4, 'geometry rotation'); vector(geom.rgba, 4, 'geometry color'); vector(geom.size, 3, 'geometry size');
    assert(['mesh', 'box', 'sphere', 'capsule', 'cylinder', 'plane'].includes(geom.type) && (geom.type !== 'mesh' || meshIds.has(geom.mesh_id)), 'Unsupported native geometry.');
  }
  assert(Array.isArray(metadata.cases) && metadata.cases.length > 0 && metadata.cases.length <= 100, 'Invalid replay case count.');
  const poses = new DataView(motion);
  for (const item of metadata.cases) {
    assert(typeof item.id === 'string' && Object.hasOwn(labels, item.variant), 'Invalid replay identity.');
    integer(item.frame_count, 0, 1501, 'frame count'); integer(item.pose_byte_offset, 0, motion.byteLength, 'pose offset');
    assert(item.pose_byte_offset % 4 === 0 && item.pose_byte_offset + item.frame_count * metadata.frame_stride_floats * 4 <= motion.byteLength, 'Recorded motion exceeds binary bounds.');
    assert(Array.isArray(item.rows) && item.rows.length === item.frame_count, 'Recorded row count mismatch.');
    item.rows.forEach((row, index) => {
      assert(finite(row.t_seconds) && Math.abs(row.t_seconds - index * .02) < 1e-8, 'Recorded time cadence changed.');
      vector(row.joint_position_rad, 6, 'six arm axes'); vector(row.gripper_joint_position_m, 2, 'jaw linkage positions');
    });
    localHref(item.episode);
    for (let offset = item.pose_byte_offset; offset < item.pose_byte_offset + item.frame_count * metadata.frame_stride_floats * 4; offset += 4) assert(finite(poses.getFloat32(offset, true)), 'Nonfinite body pose.');
  }
  assert(Array.isArray(metadata.all_attempts) && metadata.all_attempts.length <= 300, 'Invalid complete result list.');
}
function nativeQuaternion(q, target) { target.set(q[1], q[2], q[3], q[0]); }
function geometryFor(geom, meshes) {
  switch (geom.type) {
    case 'mesh': return meshes.get(geom.mesh_id);
    case 'box': return new THREE.BoxGeometry(...geom.size.map(v => v * 2));
    case 'sphere': return new THREE.SphereGeometry(geom.size[0], 24, 16);
    case 'capsule': { const g = new THREE.CapsuleGeometry(geom.size[0], geom.size[1] * 2, 6, 20); g.rotateX(Math.PI / 2); return g; }
    case 'cylinder': { const g = new THREE.CylinderGeometry(geom.size[0], geom.size[0], geom.size[1] * 2, 28); g.rotateX(Math.PI / 2); return g; }
    case 'plane': return new THREE.PlaneGeometry(Math.max(2, geom.size[0] * 2), Math.max(2, geom.size[1] * 2));
  }
}
function initWorld(geometry) {
  assert($('viewport').getContext('webgl2'), 'WebGL2 is unavailable.');
  renderer = new THREE.WebGLRenderer({canvas: $('viewport'), antialias: true, alpha: true, powerPreference: 'low-power'});
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  scene = new THREE.Scene(); camera = new THREE.PerspectiveCamera(36, 1, .005, 12); camera.up.set(0, 0, 1);
  camera.position.set(.95, -1.10, .80);
  controls = new OrbitControls(camera, $('viewport')); controls.target.copy(focus); controls.enableDamping = false; controls.minDistance = .35; controls.maxDistance = 3.; controls.addEventListener('change', requestRender); controls.update();
  scene.add(new THREE.HemisphereLight(0xfaf7ef, 0xc3c1ad, 2.6));
  const light = new THREE.DirectionalLight(0xffffff, 3.1); light.position.set(-1, -2, 4); light.castShadow = true; light.shadow.mapSize.set(1024, 1024); Object.assign(light.shadow.camera, {left: -1, right: 1, top: 1, bottom: -1, near: .1, far: 7}); light.shadow.bias = -.0002; scene.add(light);
  const fill = new THREE.DirectionalLight(0xd9e7ed, 1.0); fill.position.set(1, 2, 2); scene.add(fill);
  bodies = model.bodies.map(() => { const body = new THREE.Group(); scene.add(body); return body; });
  const meshes = new Map(), dv = new DataView(geometry);
  for (const mesh of model.meshes) {
    const xyz = new Float32Array(mesh.vertex_count * 3), indices = new Uint32Array(mesh.index_count);
    for (let i = 0; i < xyz.length; i++) xyz[i] = dv.getFloat32(mesh.position_byte_offset + i * 4, true);
    for (let i = 0; i < indices.length; i++) indices[i] = dv.getUint32(mesh.index_byte_offset + i * 4, true);
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(xyz, 3)); g.setIndex(new THREE.BufferAttribute(indices, 1)); g.computeVertexNormals(); meshes.set(mesh.id, g);
  }
  for (const geom of model.geoms) {
    if (geom.rgba[3] <= 0 || geom.group > 2) continue;
    const rgba = geom.rgba, material = new THREE.MeshStandardMaterial({color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.LinearSRGBColorSpace), metalness: geom.type === 'mesh' ? .18 : .03, roughness: .58, transparent: rgba[3] < 1, opacity: rgba[3], side: THREE.DoubleSide});
    const mesh = new THREE.Mesh(geometryFor(geom, meshes), material); mesh.position.fromArray(geom.position); nativeQuaternion(geom.quaternion_wxyz, mesh.quaternion); mesh.castShadow = geom.type !== 'plane'; mesh.receiveShadow = true; bodies[geom.body_id].add(mesh);
  }
  new ResizeObserver(() => {
    const bounds = $('stage').getBoundingClientRect(); if (!bounds.width || !bounds.height) return;
    renderer.setSize(bounds.width, bounds.height, false); camera.aspect = bounds.width / bounds.height; camera.updateProjectionMatrix(); requestRender();
  }).observe($('stage'));
  $('viewport').addEventListener('webglcontextlost', event => { event.preventDefault(); fail(new Error('WebGL context was lost. Reload to reopen the saved replay.')); });
}
function requestRender() { if (!requested && renderer) { requested = true; requestAnimationFrame(render); } }
function render(now) {
  requested = false;
  if (playing && selected.frame_count > 1) {
    const target = originClock + (now - originTime) / 1000 * Number($('speed').value);
    let index = frame; while (index + 1 < selected.frame_count && selected.rows[index + 1].t_seconds <= target) index++;
    if (index !== frame) draw(index);
    if (target >= selected.rows.at(-1).t_seconds) setPlaying(false);
  }
  renderer.render(scene, camera); $('viewport').dataset.renderCount = String(Number($('viewport').dataset.renderCount || 0) + 1);
  if (playing) requestRender();
}
function draw(index) {
  frame = Math.max(0, Math.min(selected.frame_count - 1, index)); const row = selected.rows[frame];
  const start = selected.pose_byte_offset + frame * metadata.frame_stride_floats * 4;
  bodies.forEach((body, id) => {
    const p = start + id * 28;
    body.position.set(poseBytes.getFloat32(p, true), poseBytes.getFloat32(p + 4, true), poseBytes.getFloat32(p + 8, true));
    body.quaternion.set(poseBytes.getFloat32(p + 16, true), poseBytes.getFloat32(p + 20, true), poseBytes.getFloat32(p + 24, true), poseBytes.getFloat32(p + 12, true));
  });
  $('timeline').value = String(frame); $('clock').textContent = `${row.t_seconds.toFixed(2)} / ${selected.rows.at(-1).t_seconds.toFixed(2)} s`;
  $('frame').textContent = `Frame ${frame + 1} of ${selected.frame_count}`;
  $('joint-positions').textContent = row.joint_position_rad.map((v, i) => `J${i + 1} ${v.toFixed(3)}`).join(' · ') + ' rad';
  const command = finite(row.issued_gripper_position_m) ? ` · requested ${row.issued_gripper_position_m.toFixed(4)} m` : '';
  $('jaw-position').textContent = `${row.gripper_joint_position_m[0].toFixed(4)} m${command}`;
  const status = `${labels[selected.variant]} · saved 20 ms states · drag to orbit, scroll to zoom.`;
  if ($('status').textContent !== status) $('status').textContent = status;
  requestRender();
}
function setPlaying(value) {
  playing = value && selected?.frame_count > 1; $('play').textContent = playing ? 'Pause replay' : 'Play replay'; $('play').setAttribute('aria-pressed', String(playing));
  if (playing) {
    if (frame === selected.frame_count - 1) draw(0);
    originClock = selected.rows[frame].t_seconds; originTime = performance.now(); requestRender();
  }
}
function updateVariants(preferred = $('variant-select').value) {
  $('variant-select').replaceChildren();
  for (const variant of Object.keys(labels)) if (metadata.cases.some(item => item.id === $('case-select').value && item.variant === variant)) {
    const option = document.createElement('option'); option.value = variant; option.textContent = labels[variant]; $('variant-select').append(option);
  }
  if ([...$('variant-select').options].some(option => option.value === preferred)) $('variant-select').value = preferred;
}
function choose() {
  setPlaying(false); selected = metadata.cases.find(item => item.id === $('case-select').value && item.variant === $('variant-select').value);
  assert(selected, 'Requested replay is unavailable.');
  $('case-title').textContent = selected.id.replace(/^confirmation-/, 'Held-out start ');
  $('outcome').textContent = selected.score.success ? 'Completed' : 'Did not complete'; $('outcome').dataset.success = String(selected.score.success);
  $('goal-error').textContent = finite(selected.score.terminal_xy_error_m) ? `${(selected.score.terminal_xy_error_m * 1000).toFixed(1)} mm` : 'Unavailable';
  $('timeline').max = String(Math.max(0, selected.frame_count - 1)); $('play').disabled = selected.frame_count < 2; $('timeline').disabled = !selected.frame_count;
  bodies.forEach(body => { body.visible = selected.frame_count > 0; });
  if (selected.frame_count) draw(0);
  else { $('clock').textContent = 'No saved frames'; $('frame').textContent = 'Replay unavailable'; $('status').textContent = 'This attempt failed before a native state was saved. Its numerical result remains in the report.'; requestRender(); }
}
function evidence() {
  const learned = metadata.all_attempts.filter(item => item.variant === 'learned');
  const teacher = metadata.all_attempts.filter(item => item.variant === 'teacher');
  $('aggregate').textContent = `Full confirmation: learned ${learned.filter(item => item.score.success).length}/${learned.length} completed · teacher ${teacher.filter(item => item.score.success).length}/${teacher.length} · simulation only.`;
  $('selection-note').textContent = metadata.selection_notice;
  $('counts').replaceChildren();
  for (const variant of Object.keys(labels)) {
    const attempts = metadata.all_attempts.filter(item => item.variant === variant), tr = document.createElement('tr');
    for (const value of [labels[variant], attempts.filter(item => item.score.success).length, attempts.length]) { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); }
    $('counts').append(tr);
  }
  if (Array.isArray(metadata.limitations) && metadata.limitations.length) $('limitations').replaceChildren();
  for (const text of metadata.limitations || []) { const li = document.createElement('li'); li.textContent = text; $('limitations').append(li); }
  const details = document.createElement('details'), summary = document.createElement('summary'), list = document.createElement('ul');
  summary.textContent = `All ${metadata.all_attempts.length} attempt traces`;
  for (const item of metadata.all_attempts) {
    const li = document.createElement('li'), a = document.createElement('a'); a.href = studyUrl(item.episode).href; a.download = ''; a.referrerPolicy = 'no-referrer'; a.rel = 'noreferrer'; a.textContent = `${item.id} · ${labels[item.variant]} · ${item.score.success ? 'completed' : 'failed'} ↓`; li.append(a); list.append(li);
  }
  details.append(summary, list); $('selection-note').after(details);
}
function connect() {
  $('play').addEventListener('click', () => setPlaying(!playing));
  $('timeline').addEventListener('input', () => { draw(Number($('timeline').value)); originClock = selected.rows[frame].t_seconds; originTime = performance.now(); });
  $('speed').addEventListener('change', () => { originClock = selected.rows[frame]?.t_seconds || 0; originTime = performance.now(); });
  $('case-select').addEventListener('change', () => { updateVariants(); choose(); });
  $('variant-select').addEventListener('change', choose);
  document.addEventListener('visibilitychange', () => { if (document.hidden) setPlaying(false); });
  $('viewport').addEventListener('keydown', event => {
    if (event.key === ' ') { event.preventDefault(); setPlaying(!playing); }
    if (event.key === 'Home' || event.key.toLowerCase() === 'h') { event.preventDefault(); camera.position.set(.95, -1.10, .80); controls.target.copy(focus); controls.update(); requestRender(); }
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', '_'].includes(event.key)) {
      event.preventDefault();
      const offset = camera.position.clone().sub(controls.target).applyAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      if (event.key === 'ArrowLeft') spherical.theta -= .1;
      if (event.key === 'ArrowRight') spherical.theta += .1;
      if (event.key === 'ArrowUp') spherical.phi = Math.max(.01, spherical.phi - .1);
      if (event.key === 'ArrowDown') spherical.phi = Math.min(Math.PI - .01, spherical.phi + .1);
      if (event.key === '+' || event.key === '=') spherical.radius = Math.max(controls.minDistance, spherical.radius * .9);
      if (event.key === '-' || event.key === '_') spherical.radius = Math.min(controls.maxDistance, spherical.radius / .9);
      offset.setFromSpherical(spherical).applyAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
      camera.position.copy(controls.target).add(offset); controls.update(); requestRender();
    }
  });
}
function fail(error) {
  setPlaying(false); $('loading').hidden = false; $('loading').textContent = `Replay could not open: ${error.message} Serve the complete export over HTTP. The report remains available below.`;
  $('status').textContent = 'Recorded replay unavailable.';
  for (const element of document.querySelectorAll('button,select,input')) element.disabled = true;
  $('stage').dataset.ready = 'error';
}
async function main() {
  assert(location.protocol !== 'file:', 'Saved replay files need a local or hosted HTTP URL.');
  [model, metadata] = await Promise.all([json('scene.json'), json('motion.json')]); evidence();
  const [geometry, motion] = await Promise.all([binary(model.geometry_binary, 'geometry.bin'), binary(metadata.motion_binary, 'motion.bin')]);
  validate(geometry, motion); poseBytes = new DataView(motion); initWorld(geometry);
  $('case-select').replaceChildren();
  for (const id of [...new Set(metadata.cases.map(item => item.id))]) { const option = document.createElement('option'); option.value = id; option.textContent = id.replace(/^confirmation-/, 'Held-out '); $('case-select').append(option); }
  const first = metadata.cases.find(item => item.variant === 'learned' && item.frame_count > 0) || metadata.cases[0]; $('case-select').value = first.id;
  for (const element of document.querySelectorAll('button,select,input')) element.disabled = false;
  updateVariants('learned'); choose(); connect(); $('loading').hidden = true; $('stage').dataset.ready = 'true';
  $('replay-note').textContent = `Native PiPER geometry · six arm axes and one jaw command · ${metadata.body_count} recorded bodies. Browser physics: none.`;
}
main().catch(fail);
