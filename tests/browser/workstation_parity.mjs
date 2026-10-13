/* Reproducible native/WASM parity, including force-only shoe perturbation. */
import assert from 'node:assert/strict';
import {readFile, writeFile, mkdtemp, rm} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {build} from '../../viewer/node_modules/esbuild/lib/main.js';
import loadMujoco from '../../viewer/node_modules/@mujoco/mujoco/mujoco.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const [sceneArg, fixtureArg, outputArg] = process.argv.slice(2);
if (!sceneArg || !fixtureArg || !outputArg) throw new Error('Usage: node tests/browser/workstation_parity.mjs SCENE FIXTURE OUTPUT');
const scene = path.resolve(sceneArg);
const manifestRaw = await readFile(path.join(scene, 'workstation.json'));
const manifest = JSON.parse(manifestRaw);
const fixture = JSON.parse(await readFile(fixtureArg, 'utf8'));
const xml = await readFile(path.join(scene, 'scene.xml'));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
assert.equal(digest(xml), fixture.native_scene_sha256, 'Fixture and WASM must use identical XML');
assert.equal(digest(manifestRaw), fixture.manifest_sha256, 'Fixture and WASM must use identical settings');
assert.equal(fixture.valid, true, 'Native simulation must be valid');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'dvidia-workstation-parity-'));
let engine;
try {
  const compiled = path.join(temporary, 'worker.mjs');
  await build({entryPoints: [path.join(root, 'src/dvidia_training/workstation_worker.js')], outfile: compiled,
    bundle: true, platform: 'node', format: 'esm', nodePaths: [path.join(root, 'viewer/node_modules')]});
  const {WorkstationEngine} = await import(pathToFileURL(compiled));
  const mj = await loadMujoco({wasmBinary: await readFile(path.join(root, 'viewer/node_modules/@mujoco/mujoco/mujoco.wasm'))});
  const vfs = new mj.MjVFS();
  let model;
  try {
    for (const asset of manifest.assets) {
      const bytes = await readFile(path.join(scene, asset.path));
      assert.equal(bytes.length, asset.bytes); assert.equal(digest(bytes), asset.sha256);
      vfs.addBuffer(asset.path, new Uint8Array(bytes));
    }
    model = mj.MjModel.from_xml_string(xml.toString('utf8'), vfs);
  } finally { vfs.delete(); }
  engine = new WorkstationEngine(mj, model, manifest);
  const max = {qpos: 0, qvel: 0, ctrl: 0, tcp_position: 0, shoe_position: 0};
  let differingContacts = 0; let differingActiveContacts = 0; let error = null; let completed = 0;
  const contactDifferences = [];
  const schedule = new Map();
  for (const command of fixture.commands) schedule.set(command.step, [...(schedule.get(command.step) ?? []), command]);
  const started = performance.now();
  const samples = []; const fullStepMs = [];
  try {
    for (let index = 0; index < fixture.physics_steps; index += 1) {
      for (const command of schedule.get(index) ?? []) engine.applyCommand(command);
      const stepStarted = performance.now();
      engine.step(); completed += 1;
      fullStepMs.push(performance.now() - stepStarted);
      if ((index + 1) % fixture.control_stride !== 0) continue;
      const native = fixture.trace[(index + 1) / fixture.control_stride - 1];
      const state = engine.state();
      const fields = {qpos: state.qpos, qvel: state.qvel, ctrl: state.ctrl,
        tcp_position: state.tcp_position, shoe_position: state.shoes[manifest.shoe_body_names[0]].position};
      for (const [name, values] of Object.entries(fields)) {
        assert.equal(values.length, native[name].length);
        max[name] = Math.max(max[name], ...values.map((value, item) => Math.abs(value - native[name][item])));
      }
      if (state.contacts !== native.contacts) differingContacts += 1;
      if (state.active_contacts !== native.active_contacts) differingActiveContacts += 1;
      if (state.contacts !== native.contacts || state.active_contacts !== native.active_contacts) {
        contactDifferences.push({completed_steps: index + 1, native_contacts: native.contacts, wasm_contacts: state.contacts,
          native_active_contacts: native.active_contacts, wasm_active_contacts: state.active_contacts,
          native_normal_forces_n: native.contact_normal_forces_n, wasm_normal_forces_n: state.contact_normal_forces_n});
      }
      samples.push({completed_steps: index + 1, tcp_position: state.tcp_position,
        shoe_position: fields.shoe_position, contacts: state.contacts, active_contacts: state.active_contacts,
        tracking_error_m: state.tracking_error_m});
    }
  } catch (exception) { error = exception.message; }
  const wallSeconds = (performance.now() - started) / 1000;
  const thresholds = {qpos: 1e-6, qvel: 1e-5, ctrl: 1e-6, tcp_position: 1e-6, shoe_position: 1e-6};
  const orderedStepMs = fullStepMs.slice().sort((a, b) => a - b);
  const quantile = fraction => orderedStepMs[Math.floor((orderedStepMs.length - 1) * fraction)] ?? null;
  const numericallyValid = !error && completed === fixture.physics_steps && engine.warnings.length === 0;
  const numericalParityPassed = numericallyValid && Object.entries(max).every(([field, value]) => value <= thresholds[field]);
  const parityPassed = numericalParityPassed && differingContacts === 0;
  const result = {schema: 'dvidia.workstation-wasm-parity/v1', valid: numericallyValid, error,
    strict_parity_passed: parityPassed, numerical_parity_passed: numericalParityPassed,
    active_contact_count_parity_passed: differingActiveContacts === 0,
    engine: mj.mj_versionString(), node: process.version, platform: `${os.platform()} ${os.release()} ${os.arch()}`,
    hardware: {cpu: os.cpus()[0]?.model ?? null, logical_cpus: os.cpus().length, total_memory_bytes: os.totalmem()},
    native_scene_sha256: fixture.native_scene_sha256, manifest_sha256: fixture.manifest_sha256,
    simulated_seconds: completed * manifest.timestep, completed_steps: completed,
    compared_states: samples.length, maximum_absolute_errors: max, thresholds,
    differing_contact_count_states: differingContacts, warnings: engine.warnings,
    differing_active_contact_count_states: differingActiveContacts, active_contact_threshold_n: .001,
    contact_differences: contactDifferences,
    wall_seconds: wallSeconds, real_time_factor: completed * manifest.timestep / wallSeconds,
    step_ms_p50: quantile(.5), step_ms_p95: quantile(.95), step_ms_p99: quantile(.99), step_ms_max: quantile(1),
    control_ms_p95: engine.controlSamples.quantile(.95),
    timing_window: 'All 15,000 physics steps including startup; all 1,500 control updates. Comparison overhead included in wall time.',
    process_memory: process.memoryUsage(), trace: samples,
    scope: 'Native-versus-WASM deterministic controller/physics parity; not a shoe grasp success benchmark or real-robot validation.'};
  await writeFile(outputArg, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(Object.fromEntries(Object.entries(result).filter(([key]) => !['trace', 'contact_differences'].includes(key))), null, 2));
  if (!parityPassed) process.exitCode = 1;
} finally { engine?.dispose(); await rm(temporary, {recursive: true, force: true}); }
