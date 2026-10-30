const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');
const zlib = require('node:zlib');
const path = require('node:path');

const file = path.join(__dirname, '../src/dvidia_training/piper_grip_clock_viewer.js');
let source = fs.readFileSync(file, 'utf8')
  .replace(/^import .*;\n/gm, '')
  .replace('const CLOCK_RESOURCES = __DVIDIA_CLOCK_RESOURCE_MANIFEST__;', 'const CLOCK_RESOURCES = {};');
source = source.slice(0, source.lastIndexOf('\nmain().catch('));
source += '\nglobalThis.api={resourceName,local,descriptor,decode,payload,receive,evidenceHref,replayKey,validateRows,setMeta:value=>meta=value};';
const context = vm.createContext({
  crypto: {subtle: {digest: async (_, bytes) => crypto.webcrypto.subtle.digest('SHA-256', bytes)}},
  DecompressionStream, Blob, Response, TextDecoder, URL, AbortController, setTimeout, clearTimeout,
  atob, btoa,
});
vm.runInContext(source, context);
const api = context.api;

test('three digit replay resource indices cover the complete declared index space', () => {
  for (const i of [99, 100, 257, 319]) {
    assert.equal(api.resourceName('frames', i), `frames-${String(i).padStart(3, '0')}.json`);
    assert.equal(api.resourceName('poses', i), `poses-${String(i).padStart(3, '0')}.bin`);
  }
  assert.throws(() => api.resourceName('frames', 320), /replay index/);
  assert.throws(() => api.resourceName('frames', -1), /replay index/);
});

test('resource paths and evidence URLs reject traversal, foreign hosts, and mutable revisions', () => {
  assert.throws(() => api.local('../frames-000.json'), /Unexpected local/);
  assert.throws(() => api.local('a/../frames-000.json'), /Unexpected local/);
  api.setMeta({native_dataset: {revision: 'a'.repeat(40)}});
  const item = {case_id: 'confirmation-xy01-g0-nominal', variant: 'clock', evidence: {
    'native.json': {bytes: 2, sha256: 'b'.repeat(64), href: `https://huggingface.co/datasets/Dvidia/piper-grip-clock-demonstrations-v03/resolve/${'a'.repeat(40)}/study/confirmation/confirmation-xy01-g0-nominal/clock/native.json`},
  }};
  assert.equal(api.evidenceHref(item, 'native.json'), item.evidence['native.json'].href);
  for (const href of [item.evidence['native.json'].href.replace('huggingface.co', 'example.com'), item.evidence['native.json'].href.replace('a'.repeat(40), 'main'), item.evidence['native.json'].href + '?download=1']) {
    item.evidence['native.json'].href = href;
    assert.throws(() => api.evidenceHref(item, 'native.json'), /pinned/);
  }
});

function resource(raw, href = 'frames-000.json') {
  const packed = zlib.gzipSync(raw);
  const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
  return {href, bytes: raw.length, sha256: digest(raw), compression: 'gzip', compressed_bytes: packed.length,
    compressed_sha256: digest(packed), data: packed.toString('base64')};
}

test('resource payload exact keys, canonical base64, compressed and raw hashes are checked', async () => {
  const raw = Buffer.from('[{"ok":true}]');
  const value = resource(raw);
  const descriptor = {...value}; delete descriptor.data;
  assert.deepEqual(JSON.parse(Buffer.from(await api.decode(value, descriptor)).toString()), [{ok: true}]);
  assert.throws(() => api.payload({schema: 'dvidia.piper-clock-resource-data/v1', id: 'replay-000', resources: [value, value], extra: 1}, {id: 'replay-000', resources: [descriptor, descriptor]}), /fields/);
  const noncanonical = {...value, data: value.data.replace(/.$/, 'B')};
  await assert.rejects(api.decode(noncanonical, descriptor));
  const altered = {...value, compressed_sha256: '0'.repeat(64)};
  await assert.rejects(api.decode(altered, {...descriptor, compressed_sha256: '0'.repeat(64)}), /Compressed SHA256/);
  const inflated = resource(Buffer.alloc(128 * 1024, 65));
  const underdeclared = {...inflated, bytes: 1, sha256: crypto.createHash('sha256').update(Buffer.from('A')).digest('hex')};
  const underdeclaredMeta = {...underdeclared}; delete underdeclaredMeta.data;
  await assert.rejects(api.decode(underdeclared, underdeclaredMeta), /decompression bound/);
});

test('large bounded base64 descriptors do not depend on the regular expression stack', () => {
  const value = resource(Buffer.from('fixture'));
  // A syntax-only descriptor fixture at the compressed-byte bound. Gzip and
  // canonical pad bits are checked separately by decode before use.
  value.compressed_bytes = 3 * 1024 * 1024;
  value.data = 'QUJD'.repeat(value.compressed_bytes / 3);
  assert.doesNotThrow(() => api.descriptor(value, value));
  for (const data of [value.data.slice(0, 12) + '=' + value.data.slice(13),
      value.data.slice(0, -1) + ' ', value.data.slice(0, -1) + '＝']) {
    assert.throws(() => api.descriptor({...value, data}, {...value, data}), /base64/);
  }
});

test('zero byte pose resources retain valid empty-gzip identity and stale callbacks do nothing', () => {
  const packed = zlib.gzipSync(Buffer.alloc(0));
  const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
  const emptyPose = {href: 'poses-000.bin', bytes: 0, sha256: digest(Buffer.alloc(0)), compression: 'gzip', compressed_bytes: packed.length, compressed_sha256: digest(packed), data: packed.toString('base64')};
  assert.doesNotThrow(() => api.descriptor(emptyPose, emptyPose));
  const later = {...emptyPose, href: 'poses-001.bin'};
  assert.doesNotThrow(() => api.descriptor(later, later));
  assert.throws(() => api.descriptor({...emptyPose, href: 'frames-000.json'}, {...emptyPose, href: 'frames-000.json'}), /raw size/);
  assert.equal(api.receive('replay-319', {schema: 'dvidia.piper-clock-resource-data/v1'}), false);
});

test('full and blind replays of the same scene never share cached frames', () => {
  const scene = 'confirmation-xy01-g0-nominal';
  assert.notEqual(api.replayKey({case_id: scene, variant: 'clock'}), api.replayKey({case_id: scene, variant: 'blind_clock'}));
  assert.throws(() => api.replayKey({case_id: scene, variant: 'unknown'}), /identity/);
});

test('saved clock rows bind runtime origins, causal progress and terminal no-command state', () => {
  api.setMeta({frame_stride_floats: 7});
  const first = {
    t_seconds: 0, joint_position_rad: [0, 0, 0, 0, 0, 0], gripper_joint_position_m: [0, 0],
    object_position_m: [0, 0, 0], tcp_position_m: [0, 0, 0], native_actuator_controls: [0, 0, 0, 0, 0, 0, 0],
    issued_gripper_position_m: 0, virtual_progress_s: 0, clock_debt_s: 0,
    grasp_contact_counts: {left: 0, right: 0}, pad_normal_forces_n: {left: 0, right: 0}, table_object_contact_count: 1,
    next_clock_decision: {increment_s: 0, progress_after_s: 0, origin: 'learned_grip_clock', advance_probability: 0, secure_grip: null},
  };
  const terminal = {...first, t_seconds: .02, clock_debt_s: .02, issued_gripper_position_m: null, next_clock_decision: null};
  const item = {frame_count: 2, variant: 'clock', score: {valid: true}};
  const poses = new ArrayBuffer(2 * 7 * 4);
  assert.doesNotThrow(() => api.validateRows(item, [first, terminal], poses));
  assert.throws(() => api.validateRows(item, [{...first, next_clock_decision: {...first.next_clock_decision, origin: 'model'}}, terminal], poses), /decision/);
  assert.throws(() => api.validateRows(item, [{...first, next_clock_decision: {...first.next_clock_decision, advance_probability: 1}}, terminal], poses), /probability/);
  assert.throws(() => api.validateRows(item, [first, {...terminal, virtual_progress_s: .02}], poses), /progress/);
  assert.doesNotThrow(() => api.validateRows({...item, frame_count: 1, score: {valid: false}}, [first], new ArrayBuffer(7 * 4)));
  assert.throws(() => api.validateRows({...item, frame_count: 1}, [first], new ArrayBuffer(7 * 4)), /terminal/);
});
