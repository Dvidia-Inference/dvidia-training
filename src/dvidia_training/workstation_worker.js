/* DVIDIA live PiPER workstation. MIT. MuJoCo 3.15.0 runs entirely in this worker. */
import loadMujoco from '@mujoco/mujoco';
import {boundedTarget, clamp, dampedTargets, finiteVector, RollingSamples} from './workstation_control.js';

const ENGINE_VERSION = '3.15.0';
const MAX_ASSETS = 256;
const MAX_RESOURCE_BYTES = 64 * 1024 * 1024;
const MAX_TOTAL_BYTES = 192 * 1024 * 1024;
const MAX_CATCHUP_SECONDS = 0.1;
const MAX_STEPS_PER_TICK = 50;
// The event loop may deliver fewer states than requested. Backpressure retains
// at most one queued state while physics continues independently at 500 Hz.
const STATE_INTERVAL_MS = 1000 / 60;

function localResource(path, base) {
  if (typeof path !== 'string' || path.length > 1024 || path.startsWith('/')
      || path.split('/').some(part => part === '..') || path.includes('\\')) {
    throw new Error('Invalid local asset path');
  }
  const url = new URL(path, base);
  if (url.origin !== base.origin || !['http:', 'https:'].includes(url.protocol)) {
    throw new Error('Workstation resources must be same-origin HTTP assets');
  }
  return url;
}

async function resourceBytes(url, expected) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Resource fetch failed (${response.status}): ${url.pathname}`);
  const advertised = Number(response.headers.get('content-length'));
  if (advertised > MAX_RESOURCE_BYTES) throw new Error('Resource is too large');
  if (!response.body) throw new Error('Streaming resource responses are required');
  const reader = response.body.getReader();
  const chunks = []; let received = 0;
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      received += value.byteLength;
      if (received > MAX_RESOURCE_BYTES) {
        await reader.cancel();
        throw new Error('Resource is too large');
      }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(received); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  if (expected) {
    if (!Number.isInteger(expected.bytes) || bytes.length !== expected.bytes || !/^[a-f0-9]{64}$/.test(expected.sha256)) {
      throw new Error(`Asset size or digest descriptor mismatch: ${url.pathname}`);
    }
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
      value => value.toString(16).padStart(2, '0')).join('');
    if (digest !== expected.sha256) throw new Error(`Asset checksum mismatch: ${url.pathname}`);
  }
  return bytes;
}

function validateManifest(manifest) {
  if (!manifest || manifest.schema !== 'dvidia.live-workstation/v1') throw new Error('Unsupported workstation schema');
  if (manifest.native_version !== ENGINE_VERSION) throw new Error('Native and browser MuJoCo versions must match');
  if (manifest.timestep !== 0.002 || manifest.control_dt !== 0.02) throw new Error('Unsupported physics/control timestep');
  if (!Array.isArray(manifest.assets) || manifest.assets.length > MAX_ASSETS) throw new Error('Invalid asset manifest');
  if (!Array.isArray(manifest.bounds_xyz) || manifest.bounds_xyz.length !== 3) throw new Error('Invalid workspace bounds');
  for (const bounds of manifest.bounds_xyz) {
    finiteVector(bounds, 2, 'Workspace bounds');
    if (bounds[0] >= bounds[1] || bounds[0] < -2 || bounds[1] > 2) throw new Error('Invalid workspace bounds');
  }
  boundedTarget(manifest.initial_target_xyz, manifest.bounds_xyz);
  const orientation = finiteVector(manifest.orientation_wxyz, 4, 'Tool orientation');
  if (Math.abs(Math.hypot(...orientation) - 1) > 1e-8) throw new Error('Tool orientation must be normalized');
  for (const field of ['joint_names', 'actuator_names']) {
    if (!Array.isArray(manifest[field]) || manifest[field].length !== 6
        || !manifest[field].every(name => typeof name === 'string' && name.length < 100)) throw new Error(`Invalid ${field}`);
  }
  const config = manifest.controller;
  for (const name of ['orientation_weight', 'translation_error_cap_m', 'rotation_error_cap_rad', 'damping',
    'max_joint_delta_rad', 'joint_margin_rad', 'joint_rate_rad_s', 'gripper_rate_m_s']) {
    if (!Number.isFinite(config?.[name]) || config[name] <= 0 || config[name] > 5) throw new Error(`Invalid controller ${name}`);
  }
  finiteVector(config.gripper_range_m, 2, 'Gripper range');
  if (config.gripper_range_m[0] !== 0 || config.gripper_range_m[1] !== 0.035) throw new Error('Unsupported gripper range');
  if (!Array.isArray(manifest.shoe_body_names) || manifest.shoe_body_names.length > 12) throw new Error('Invalid shoe body list');
  const pushLimit = manifest.interaction_limits?.push_impulse_norm_max_ns;
  if (!Number.isFinite(pushLimit) || pushLimit <= 0 || pushLimit > .12) throw new Error('Invalid push impulse limit');
  return manifest;
}

export class WorkstationEngine {
  constructor(mujoco, model, manifest, emit = () => {}) {
    this.mj = mujoco; this.model = model; this.manifest = validateManifest(manifest); this.emit = emit;
    if (model.nq !== 15 || model.nv !== 14 || model.nu !== 7 || model.neq !== 1 || model.nmocap !== 0
        || model.opt.timestep !== manifest.timestep) throw new Error('Native PiPER scene contract mismatch');
    this.data = new mujoco.MjData(model);
    this.buffers = [];
    const buffer = size => { const value = new mujoco.DoubleBuffer(size); this.buffers.push(value); return value; };
    this.jacp = buffer(3 * model.nv); this.jacr = buffer(3 * model.nv);
    this.quaternion = buffer(4); this.rotation = buffer(3); this.contactForce = buffer(6);
    const id = (type, name) => {
      const value = mujoco.mj_name2id(model, mujoco.mjtObj[type].value, name);
      if (value < 0) throw new Error(`Missing native model element: ${name}`);
      return value;
    };
    this.tcp = id('mjOBJ_SITE', manifest.tcp_site_name);
    this.joints = manifest.joint_names.map(name => id('mjOBJ_JOINT', name));
    this.actuators = manifest.actuator_names.map(name => id('mjOBJ_ACTUATOR', name));
    this.jawActuator = id('mjOBJ_ACTUATOR', manifest.gripper_actuator_names[0]);
    this.jawQpos = model.jnt_qposadr[id('mjOBJ_JOINT', 'joint7')];
    this.qposIds = this.joints.map(joint => model.jnt_qposadr[joint]);
    this.dofIds = this.joints.map(joint => model.jnt_dofadr[joint]);
    this.ctrlRange = this.actuators.map(actuator => Array.from(model.actuator_ctrlrange.slice(actuator * 2, actuator * 2 + 2)));
    this.shoes = Object.fromEntries(manifest.shoe_body_names.map(name => [name, id('mjOBJ_BODY', name)]));
    // These are borrowed views owned by MjData in the pinned 3.15 bindings.
    // Retain once; deleting either view independently corrupts native MjData.
    this.nativeWarnings = this.data.warning;
    this.nativeWarningEntries = Array.from({length: this.nativeWarnings.size()}, (_, index) => this.nativeWarnings.get(index));
    this.dt = manifest.timestep; this.controlStride = Math.round(manifest.control_dt / this.dt);
    this.running = true; this.benchmarking = false; this.timer = null; this.statePending = false;
    this.reset();
  }

  reset() {
    this.mj.mj_resetData(this.model, this.data);
    const qpos = finiteVector(this.manifest.home_qpos, this.model.nq, 'Initial positions');
    const controls = finiteVector(this.manifest.home_ctrl, this.model.nu, 'Initial controls');
    this.data.qpos.set(qpos); this.data.ctrl.set(controls);
    this.mj.mj_forward(this.model, this.data);
    this.target = this.manifest.initial_target_xyz.slice();
    this.jaw = controls[this.jawActuator];
    this.desiredJaw = this.jaw;
    this.desired = this.actuators.map(index => controls[index]);
    this.steps = 0; this.pushes = new Map(); this.warnings = [];
    this.stepSamples = new RollingSamples(); this.controlSamples = new RollingSamples();
    this.startWall = performance.now(); this.lastWall = this.startWall; this.lastStateWall = 0;
    this.activeWallMs = 0; this.lagMs = 0; this.accumulator = 0;
    this.peakContacts = 0; this.invalidCommands = 0;
    this.statePending = false;
    this.initialShoePositions = Object.fromEntries(Object.entries(this.shoes).map(([name, body]) => [name,
      Array.from(this.data.xpos.slice(3 * body, 3 * body + 3))]));
  }

  controller() {
    const started = performance.now();
    const matrix = Array.from(this.data.site_xmat.slice(this.tcp * 9, this.tcp * 9 + 9));
    this.mj.mj_jacSite(this.model, this.data, this.jacp, this.jacr, this.tcp);
    this.mj.mju_mat2Quat(this.quaternion, matrix);
    this.mj.mju_subQuat(this.rotation, this.manifest.orientation_wxyz, Array.from(this.quaternion.GetView()));
    const localRotation = this.rotation.GetView();
    const rotationWorld = [0, 1, 2].map(row => matrix[row * 3] * localRotation[0]
      + matrix[row * 3 + 1] * localRotation[1] + matrix[row * 3 + 2] * localRotation[2]);
    this.desired = dampedTargets({jacp: this.jacp.GetView(), jacr: this.jacr.GetView(), nv: this.model.nv,
      dofIds: this.dofIds, qpos: this.data.qpos, qposIds: this.qposIds,
      target: this.target, current: this.data.site_xpos.slice(this.tcp * 3, this.tcp * 3 + 3),
      rotationWorld, ctrlRange: this.ctrlRange, config: this.manifest.controller});
    this.desiredJaw = this.jaw;
    this.controlSamples.add(performance.now() - started);
  }

  step() {
    const started = performance.now();
    if (this.steps % this.controlStride === 0) this.controller();
    const config = this.manifest.controller;
    const ctrl = this.data.ctrl;
    this.actuators.forEach((actuator, index) => {
      ctrl[actuator] += clamp(this.desired[index] - ctrl[actuator], -config.joint_rate_rad_s * this.dt, config.joint_rate_rad_s * this.dt);
    });
    ctrl[this.jawActuator] += clamp(this.desiredJaw - ctrl[this.jawActuator], -config.gripper_rate_m_s * this.dt, config.gripper_rate_m_s * this.dt);
    this.data.xfrc_applied.fill(0);
    for (const [body, push] of this.pushes) {
      if (push.remainingSteps <= 0) this.pushes.delete(body);
      else { this.data.xfrc_applied.set(push.force, body * 6); push.remainingSteps -= 1; }
    }
    this.mj.mj_step(this.model, this.data);
    this.data.xfrc_applied.fill(0);
    this.mj.mj_forward(this.model, this.data);
    this.steps += 1;
    this.peakContacts = Math.max(this.peakContacts, this.data.ncon);
    if (![...this.data.qpos, ...this.data.qvel, ...this.data.qacc, ...this.data.ctrl].every(Number.isFinite)) {
      throw new Error('Non-finite native physics state; simulation paused');
    }
    this.checkWarnings();
    this.stepSamples.add(performance.now() - started);
  }

  checkWarnings() {
    for (let index = 0; index < this.nativeWarningEntries.length; index += 1) {
      const warning = this.nativeWarningEntries[index];
      if (warning.number > 0) {
        const text = this.mj.mju_warningText(index, warning.lastinfo);
        if (!this.warnings.includes(text)) this.warnings.push(text);
      }
    }
    if (this.warnings.length) throw new Error(`Native solver warning; simulation paused: ${this.warnings[0]}`);
  }

  applyCommand(command) {
    switch (command.type) {
      case 'target': this.target = boundedTarget(command.position, this.manifest.bounds_xyz); break;
      case 'gripper': {
        const range = this.manifest.controller.gripper_range_m;
        if (!Number.isFinite(command.jaw) || command.jaw < range[0] || command.jaw > range[1]) throw new Error('Invalid gripper position');
        this.jaw = command.jaw; break;
      }
      case 'pause':
        if (typeof command.paused !== 'boolean') throw new Error('Pause requires a boolean');
        this.running = !command.paused; this.lastWall = performance.now(); this.accumulator = 0; break;
      case 'reset': this.reset(); break;
      case 'push': {
        const body = this.shoes[command.body];
        if (body === undefined) throw new Error('Push is restricted to a named shoe');
        const impulse = finiteVector(command.impulse, 3, 'Push impulse');
        const limit = this.manifest.interaction_limits.push_impulse_norm_max_ns;
        if (Math.hypot(...impulse) > limit + 1e-9) throw new Error(`Push exceeds the ${limit} N·s impulse limit`);
        this.pushes.set(body, {force: impulse.map(value => value / this.dt), remainingSteps: 1}); break;
      }
      default: throw new Error(`Unsupported command: ${String(command.type).slice(0, 60)}`);
    }
  }

  state() {
    const position = Array.from(this.data.site_xpos.slice(this.tcp * 3, this.tcp * 3 + 3));
    const shoes = Object.fromEntries(Object.entries(this.shoes).map(([name, body]) => {
      const position = Array.from(this.data.xpos.slice(body * 3, body * 3 + 3));
      return [name, {position, height_change_m: position[2] - this.initialShoePositions[name][2]}];
    }));
    let contactForceTotal = 0; let contactForceMax = 0; let activeContacts = 0;
    const contactForces = [];
    for (let contact = 0; contact < this.data.ncon; contact += 1) {
      this.mj.mj_contactForce(this.model, this.data, contact, this.contactForce);
      const normal = Math.max(0, this.contactForce.GetView()[0]);
      contactForceTotal += normal; contactForceMax = Math.max(contactForceMax, normal);
      contactForces.push(normal); activeContacts += Number(normal > .001);
    }
    return {body_pos: Array.from(this.data.xpos), body_quat: Array.from(this.data.xquat),
      qpos: Array.from(this.data.qpos), qvel: Array.from(this.data.qvel), ctrl: Array.from(this.data.ctrl),
      tcp_position: position, target_position: this.target.slice(), jaw: this.jaw,
      actual_jaw: this.data.qpos[this.jawQpos], commanded_jaw: this.data.ctrl[this.jawActuator], sim_time: this.data.time,
      paused: !this.running, tracking_error_m: Math.hypot(...position.map((value, axis) => value - this.target[axis])),
      contacts: this.data.ncon, contact_normal_force_total_n: contactForceTotal, contact_normal_force_max_n: contactForceMax, shoes,
      active_contacts: activeContacts, contact_normal_forces_n: contactForces,
      actuator_force: Array.from(this.data.actuator_force),
      metrics: {step_ms_p50: this.stepSamples.quantile(.5), step_ms_p95: this.stepSamples.quantile(.95),
        control_ms_p95: this.controlSamples.quantile(.95), rtf: this.activeWallMs > 0 ? this.data.time * 1000 / this.activeWallMs : 0,
        lag_ms: this.lagMs, physics_steps: this.steps, active_wall_ms: this.activeWallMs,
        peak_contacts: this.peakContacts, invalid_commands: this.invalidCommands,
        samples_retained: this.stepSamples.values.length},
      warnings: this.warnings.slice(), engine: `MuJoCo ${ENGINE_VERSION} WASM`,
      observation_origin: 'privileged_live_simulator_state'};
  }

  start() {
    const tick = () => {
      const now = performance.now();
      const elapsed = Math.max(0, (now - this.lastWall) / 1000); this.lastWall = now;
      if (this.running && !this.benchmarking) {
        this.activeWallMs += elapsed * 1000;
        const available = this.accumulator + elapsed;
        this.accumulator = Math.min(available, MAX_CATCHUP_SECONDS);
        this.lagMs += Math.max(0, available - this.accumulator) * 1000;
        let steps = 0;
        try {
          while (this.accumulator >= this.dt && steps < MAX_STEPS_PER_TICK) {
            this.step(); this.accumulator -= this.dt; steps += 1;
          }
        } catch (error) { this.running = false; this.emit({type: 'error', message: error.message}); }
      }
      if (now - this.lastStateWall >= STATE_INTERVAL_MS && !this.benchmarking && !this.statePending) {
        this.publishState(); this.lastStateWall = now;
      }
      this.timer = setTimeout(tick, 4);
    };
    this.lastWall = performance.now(); this.timer = setTimeout(tick, 0);
  }

  publishState() {
    if (this.statePending) return;
    this.statePending = true;
    this.emit({type: 'state', state: this.state()});
  }

  async benchmark(command) {
    if (!Number.isInteger(command.steps) || command.steps < 1 || command.steps > 30000) throw new Error('Benchmark requires 1–30,000 physics steps');
    const commands = command.commands ?? [];
    if (!Array.isArray(commands) || commands.length > 1000) throw new Error('Too many benchmark commands');
    if (commands.some(item => !Number.isInteger(item.step) || item.step < 0 || item.step >= command.steps
        || !['target', 'gripper', 'push'].includes(item.type))) throw new Error('Invalid benchmark schedule');
    const wasRunning = this.running; this.benchmarking = true;
    this.reset(); this.running = false;
    const started = performance.now(); let error = null;
    const schedule = new Map();
    for (const item of commands) schedule.set(item.step, [...(schedule.get(item.step) ?? []), item]);
    const trace = [];
    try {
      for (let step = 0; step < command.steps; step += 1) {
        for (const item of schedule.get(step) ?? []) this.applyCommand(item);
        this.step();
        if (step % 50 === 0 || step === command.steps - 1) {
          const state = this.state();
          trace.push({step, sim_time: state.sim_time, tcp_position: state.tcp_position,
            tracking_error_m: state.tracking_error_m, contacts: state.contacts, shoes: state.shoes});
        }
        // Yield to the worker event loop, retaining a bounded, deterministic physics schedule.
        if (step % 250 === 249) await new Promise(resolve => setTimeout(resolve, 0));
      }
    } catch (exception) { error = exception.message; }
    const wallMs = performance.now() - started;
    this.activeWallMs = wallMs;
    const finalState = this.state();
    this.benchmarking = false; this.running = wasRunning && !error;
    this.lastWall = performance.now(); this.accumulator = 0; this.statePending = false;
    return {schema: 'dvidia.live-workstation-benchmark/v1', engine_version: ENGINE_VERSION,
      requested_steps: command.steps, completed_steps: this.steps, simulated_seconds: this.data.time,
      wall_seconds: wallMs / 1000, real_time_factor: this.data.time * 1000 / wallMs,
      valid: !error && !this.warnings.length, error, final_state: finalState, trace,
      timestep: this.dt, control_dt: this.manifest.control_dt,
      scene_sha256: this.manifest.native_scene.sha256,
      limitations: ['Rigid synthetic shoe; material properties are uncalibrated.', 'Benchmark state is privileged simulator state.']};
  }

  dispose() {
    if (this.timer !== null) clearTimeout(this.timer);
    for (const buffer of this.buffers) buffer.delete();
    this.data.delete(); this.model.delete();
  }
}

let engine = null;
let initializing = false;

async function initialize(message) {
  if (engine || initializing) throw new Error('The worker has already been initialized');
  initializing = true;
  let model = null; let vfs = null;
  try {
    const base = new URL(message.manifest_url, self.location.href);
    if (base.origin !== self.location.origin || !['http:', 'https:'].includes(base.protocol)) throw new Error('Manifest must be same-origin');
    const manifestBytes = await resourceBytes(base);
    if (manifestBytes.length > 2 * 1024 * 1024) throw new Error('Manifest is too large');
    const manifest = validateManifest(JSON.parse(new TextDecoder().decode(manifestBytes)));
    const mujoco = await loadMujoco({locateFile: file => new URL(file, self.location.href).href,
      printErr: text => { if (engine && engine.warnings.length < 20) engine.warnings.push(String(text).slice(0, 300)); }});
    if (mujoco.mj_versionString() !== ENGINE_VERSION) throw new Error('Unexpected MuJoCo WASM version');
    vfs = new mujoco.MjVFS();
    let totalBytes = 0;
    for (const asset of manifest.assets) {
      const bytes = await resourceBytes(localResource(asset.path, base), asset);
      totalBytes += bytes.length;
      if (totalBytes > MAX_TOTAL_BYTES) throw new Error('Scene assets exceed worker memory budget');
      vfs.addBuffer(asset.path, bytes);
    }
    if (!manifest.native_scene || manifest.native_scene.path !== manifest.scene_url) throw new Error('Scene checksum descriptor is missing');
    const scene = await resourceBytes(localResource(manifest.scene_url, base), manifest.native_scene);
    model = mujoco.MjModel.from_xml_string(new TextDecoder().decode(scene), vfs);
    engine = new WorkstationEngine(mujoco, model, manifest, data => self.postMessage(data));
    model = null;
    self.postMessage({type: 'ready', manifest, state: engine.state()});
    engine.start();
  } finally { vfs?.delete(); model?.delete(); initializing = false; }
}

if (typeof self !== 'undefined' && typeof self.postMessage === 'function' && typeof document === 'undefined') {
  self.onmessage = async event => {
    const message = event.data;
    try {
      if (!message || typeof message !== 'object' || typeof message.type !== 'string') throw new Error('Invalid worker message');
      if (message.type === 'init') { await initialize(message); return; }
      if (!engine) throw new Error('Workstation is not initialized');
      if (message.type === 'ack') { engine.statePending = false; return; }
      if (engine.benchmarking) throw new Error('Interactive commands are disabled during deterministic benchmarking');
      if (message.type === 'benchmark') {
        const result = await engine.benchmark(message);
        self.postMessage({type: 'benchmark_result', request_id: message.request_id, result});
      } else if (message.type === 'state') engine.publishState();
      else { engine.applyCommand(message); engine.publishState(); }
    } catch (error) {
      if (engine) engine.invalidCommands += 1;
      self.postMessage({type: 'error', request_id: message?.request_id, message: String(error.message).slice(0, 500)});
    }
  };
}
