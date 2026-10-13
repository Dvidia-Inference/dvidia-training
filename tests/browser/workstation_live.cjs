/* Actual-UI live physics acceptance and foreground stability benchmark.
 * DVIDIA_PLAYWRIGHT_MODULE=/absolute/path/to/playwright DVIDIA_BENCH_SECONDS=600
 * node tests/browser/workstation_live.cjs URL OUTPUT_DIRECTORY
 * Default renderer is explicitly software SwiftShader; never call it a device-GPU result.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const {createHash} = require('node:crypto');
const {chromium} = require(process.env.DVIDIA_PLAYWRIGHT_MODULE || path.join(__dirname, '../../viewer/node_modules/playwright'));

const percentile = (values, fraction) => {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)];
};
const distribution = values => ({count: values.length, p50: percentile(values, .5), p95: percentile(values, .95),
  p99: percentile(values, .99), min: values.length ? Math.min(...values) : null, max: values.length ? Math.max(...values) : null});
const distance = (a, b) => Math.hypot(...a.map((value, index) => value - b[index]));

function installProbe() {
  const probe = window.__dvidiaLiveProbe = {
    started_at: new Date().toISOString(), commands: [], states: [], latencies: [], errors: [], latest: null,
    state_count: 0, commands_dropped: 0, states_dropped: 0, frame_samples_dropped: 0,
    worker_count: 0, warnings: [], recent: [], raf_times: [], render_times: [], worker_times: [],
    visibility_events: [], measuring: false, control_ack_count: 0, native_finite: true,
    latest_sample_ms: -Infinity, state_schema: null,
  };
  const bounded = (array, value, maximum, dropped) => {
    if (array.length === maximum) { array.shift(); if (dropped) probe[dropped] += 1; }
    array.push(value);
  };
  const norm = (a, b) => Math.hypot(...a.map((value, index) => value - b[index]));
  const same = (a, b) => a?.length === b?.length && a.every((value, index) => Math.abs(value - b[index]) < 1e-9);
  const NativeWorker = window.Worker;
  window.Worker = class extends NativeWorker {
    constructor(...args) {
      super(...args); probe.worker_count += 1;
      this.addEventListener('message', event => {
        const message = event.data;
        if (message.type === 'error') bounded(probe.errors, {wall_ms: performance.now(), message: message.message}, 100);
        if (!['ready', 'state'].includes(message.type) || !message.state) return;
        const state = message.state; const now = performance.now();
        probe.latest = state; probe.state_count += 1;
        probe.native_finite &&= ['qpos', 'qvel', 'ctrl', 'body_pos', 'body_quat'].every(key => state[key]?.every(Number.isFinite));
        for (const warning of state.warnings || []) if (!probe.warnings.includes(warning)) bounded(probe.warnings, warning, 100);
        bounded(probe.recent, {wall_ms: now, tcp: state.tcp_position, target: state.target_position}, 30);
        if (probe.measuring) bounded(probe.worker_times, now, 45000, 'frame_samples_dropped');
        if (now - probe.latest_sample_ms >= 100) {
          bounded(probe.states, {wall_ms: now, sim_time: state.sim_time, paused: state.paused,
            tcp_position: state.tcp_position, target_position: state.target_position,
            qpos: state.qpos, qvel: state.qvel, ctrl: state.ctrl, actual_jaw: state.actual_jaw,
            shoes: state.shoes, contacts: state.contacts, active_contacts: state.active_contacts,
            contact_normal_force_total_n: state.contact_normal_force_total_n,
            tracking_error_m: state.tracking_error_m, metrics: state.metrics, warnings: state.warnings}, 15000, 'states_dropped');
          probe.latest_sample_ms = now;
        }
        for (const latency of probe.latencies) {
          if (latency.superseded_at_ms !== null || now - latency.command_wall_ms > 5000
              || now < latency.command_wall_ms || !same(state.target_position, latency.target)) continue;
          if (latency.target_echo_ms === null) latency.target_echo_ms = now - latency.command_wall_ms;
          if (latency.physical_response_ms === null && latency.command_distance_m > .005) {
            const projected = state.tcp_position.reduce((sum, value, index) => sum
              + (value - latency.baseline_tcp[index]) * (latency.target[index] - latency.baseline_tcp[index]), 0) / latency.command_distance_m;
            if (projected >= .001) {
              latency.physical_response_ms = now - latency.command_wall_ms;
              latency.response_tcp = state.tcp_position;
              latency.response_sim_time = state.sim_time;
            }
          }
        }
      });
    }
    postMessage(message, ...rest) {
      const now = performance.now();
      if (message?.type === 'ack') probe.control_ack_count += 1;
      if (message && !['ack', 'state'].includes(message.type)) {
        bounded(probe.commands, {wall_ms: now, sim_time: probe.latest?.sim_time ?? null, ...structuredClone(message)}, 2000, 'commands_dropped');
        if (message.type === 'target' && probe.latest) {
          for (const previous of probe.latencies) {
            if (previous.physical_response_ms === null && previous.superseded_at_ms === null) previous.superseded_at_ms = now;
          }
          const baseline = probe.latest.tcp_position.slice();
          const recent = probe.recent.filter(state => now - state.wall_ms < 350);
          const baselineSpread = Math.max(0, ...recent.map(state => norm(state.tcp, baseline)));
          bounded(probe.latencies, {command_wall_ms: now, baseline_sim_time: probe.latest.sim_time,
            target: message.position.slice(), baseline_tcp: baseline, command_distance_m: norm(message.position, baseline),
            baseline_spread_m: baselineSpread,
            baseline_stable: recent.length >= 5 && baselineSpread < .0005,
            target_echo_ms: null, physical_response_ms: null, superseded_at_ms: null}, 300);
        }
      }
      return super.postMessage(message, ...rest);
    }
  };
  document.addEventListener('visibilitychange', () => bounded(probe.visibility_events,
    {wall_ms: performance.now(), visibility: document.visibilityState}, 100));
  const frame = timestamp => {
    if (probe.measuring) bounded(probe.raf_times, timestamp, 90000, 'frame_samples_dropped');
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  const observeCanvas = () => {
    const canvas = document.querySelector('#canvas');
    if (!canvas) { setTimeout(observeCanvas, 20); return; }
    new MutationObserver(records => {
      if (probe.measuring && records.some(record => record.attributeName === 'data-render-count')) {
        bounded(probe.render_times, performance.now(), 90000, 'frame_samples_dropped');
      }
    }).observe(canvas, {attributes: true, attributeFilter: ['data-render-count']});
  };
  observeCanvas();
  probe.begin = () => {
    probe.measuring = true; probe.measure_start_ms = performance.now();
    probe.measure_start_sim = probe.latest.sim_time;
    probe.measure_start_lag_ms = probe.latest.metrics.lag_ms;
    probe.measure_start_steps = probe.latest.metrics.physics_steps;
    probe.measure_start_render_count = Number(document.querySelector('#canvas').dataset.renderCount || 0);
    probe.raf_times = []; probe.render_times = []; probe.worker_times = [];
  };
  probe.end = () => {
    probe.measuring = false; probe.measure_end_ms = performance.now();
    probe.measure_end_sim = probe.latest.sim_time;
    probe.measure_end_lag_ms = probe.latest.metrics.lag_ms;
    probe.measure_end_steps = probe.latest.metrics.physics_steps;
    probe.measure_end_render_count = Number(document.querySelector('#canvas').dataset.renderCount || 0);
    return {start: probe.measure_start_ms, end: probe.measure_end_ms};
  };
}

async function setTarget(page, target) {
  for (const [index, axis] of ['X', 'Y', 'Z'].entries()) {
    await page.getByRole('spinbutton', {name: `Tool target ${axis} in millimeters`}).fill(String(target[index] * 1000));
  }
  await page.getByRole('button', {name: 'Apply target', exact: true}).click();
  await page.waitForFunction(value => window.__dvidiaLiveProbe.latest.target_position.every((coordinate, axis) => Math.abs(coordinate - value[axis]) < 1e-9), target);
}

async function main() {
  const [url, output] = process.argv.slice(2);
  assert(url && output, 'Usage: node tests/browser/workstation_live.cjs URL OUTPUT_DIRECTORY');
  const seconds = Number(process.env.DVIDIA_BENCH_SECONDS || 20);
  assert(Number.isFinite(seconds) && seconds >= 10 && seconds <= 1200, 'Benchmark duration must be 10–1200 seconds');
  const headless = process.env.DVIDIA_BENCH_HEADLESS !== '0';
  const software = process.env.DVIDIA_BENCH_RENDERER !== 'device';
  const channel = process.env.DVIDIA_BROWSER_CHANNEL || undefined;
  await fs.mkdir(output, {recursive: true});
  const errors = [], consoleErrors = [], unexpectedRequests = [], failedRequests = [], responseErrors = [], requestOrigins = new Set();
  const origin = new URL(url).origin;
  const allowedCDN = host => /^(?:[a-z0-9-]+\.)*cdn\.hf\.co$/.test(host)
    || /^cdn-lfs(?:-[a-z0-9-]+)?\.hf\.co$/.test(host) || host === 'cas-bridge.xethub.hf.co';
  const browser = await chromium.launch({headless, channel,
    args: software ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] : []});
  let page; let probe; let summary;
  const controls = {};
  try {
    page = await browser.newPage({viewport: {width: 1440, height: 1080}, deviceScaleFactor: 1, acceptDownloads: true});
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', event => { if (event.type() === 'error') consoleErrors.push(event.text()); });
    page.on('requestfailed', request => failedRequests.push({url: request.url().split('?')[0], failure: request.failure()?.errorText}));
    page.on('response', response => { if (response.status() >= 400) responseErrors.push({url: response.url().split('?')[0], status: response.status()}); });
    page.on('request', request => {
      const target = new URL(request.url());
      if (['blob:', 'data:'].includes(target.protocol)) return;
      requestOrigins.add(target.origin);
      if (target.origin !== origin && !allowedCDN(target.hostname)) unexpectedRequests.push(target.origin + target.pathname);
    });
    await page.addInitScript(installProbe);
    const loadStarted = performance.now();
    await page.goto(url, {waitUntil: 'domcontentloaded'});
    await page.bringToFront();
    await page.waitForFunction(() => document.querySelector('#stage')?.dataset.ready === 'true', {}, {timeout: 120000});
    const loadMs = performance.now() - loadStarted;
    await page.waitForFunction(() => window.__dvidiaLiveProbe.latest?.sim_time >= 1);
    assert.equal(await page.evaluate(() => document.visibilityState), 'visible');
    const manifestResponse = await page.request.get(new URL('workstation.json', url).href);
    assert(manifestResponse.ok());
    const manifestBytes = await manifestResponse.body(); const manifest = JSON.parse(manifestBytes);
    const manifestSha256 = createHash('sha256').update(manifestBytes).digest('hex');
    const renderer = await page.evaluate(() => {
      const gl = document.querySelector('#canvas').getContext('webgl2');
      const debug = gl.getExtension('WEBGL_debug_renderer_info');
      return {vendor: debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
        renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
        version: gl.getParameter(gl.VERSION), shading_language: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
        user_agent: navigator.userAgent, hardware_concurrency: navigator.hardwareConcurrency,
        device_memory_gib: navigator.deviceMemory ?? null};
    });

    // Numeric controls provide a keyboard-accessible alternative to the drag gizmo.
    const before = await page.evaluate(() => window.__dvidiaLiveProbe.latest.tcp_position);
    await setTarget(page, [.38, .02, .25]);
    await page.waitForFunction(initial => Math.hypot(...window.__dvidiaLiveProbe.latest.tcp_position.map((value, index) => value - initial[index])) > .003, before, {timeout: 5000});
    controls.target_motion_m = distance(before, await page.evaluate(() => window.__dvidiaLiveProbe.latest.tcp_position));
    controls.target_apply = true;
    const slider = page.getByRole('slider', {name: 'Finger travel target'});
    await slider.focus(); await page.keyboard.press('Home');
    await page.waitForFunction(() => window.__dvidiaLiveProbe.latest.actual_jaw < .01, {}, {timeout: 5000});
    controls.gripper_closed_m = await page.evaluate(() => window.__dvidiaLiveProbe.latest.actual_jaw);
    await page.keyboard.press('End');
    await page.waitForFunction(() => window.__dvidiaLiveProbe.latest.actual_jaw > .025, {}, {timeout: 5000});
    controls.gripper_open_m = await page.evaluate(() => window.__dvidiaLiveProbe.latest.actual_jaw);
    await page.getByRole('button', {name: 'Pause', exact: true}).click();
    await page.waitForFunction(() => window.__dvidiaLiveProbe.latest.paused);
    const pausedTime = await page.evaluate(() => window.__dvidiaLiveProbe.latest.sim_time);
    await page.waitForTimeout(400);
    assert.equal(await page.evaluate(() => window.__dvidiaLiveProbe.latest.sim_time), pausedTime);
    controls.pause_preserved_sim_time = true;
    await page.getByRole('button', {name: 'Reset', exact: true}).click();
    await page.waitForFunction(() => window.__dvidiaLiveProbe.latest.sim_time === 0 && window.__dvidiaLiveProbe.latest.paused);
    controls.reset_restored_qpos = await page.evaluate(home => window.__dvidiaLiveProbe.latest.qpos.every((value, index) => Math.abs(value - home[index]) < 1e-12), manifest.home_qpos);
    assert(controls.reset_restored_qpos);
    await page.getByRole('button', {name: 'Resume', exact: true}).click();
    await page.waitForFunction(() => !window.__dvidiaLiveProbe.latest.paused && window.__dvidiaLiveProbe.latest.sim_time > .5);
    const shoe = manifest.shoe_body_names[0];
    const shoeBefore = await page.evaluate(name => window.__dvidiaLiveProbe.latest.shoes[name].position, shoe);
    await page.getByRole('button', {name: 'Nudge shoe', exact: true}).click();
    await page.waitForFunction(({name, initial}) => Math.hypot(...window.__dvidiaLiveProbe.latest.shoes[name].position.map((value, index) => value - initial[index])) > .0005,
      {name: shoe, initial: shoeBefore}, {timeout: 4000});
    controls.nudge_displacement_m = distance(shoeBefore, await page.evaluate(name => window.__dvidiaLiveProbe.latest.shoes[name].position, shoe));
    controls.nudge_physical_response = true;

    // Isolate stability/throughput from initialization, resets and acceptance actions.
    await page.getByRole('button', {name: 'Pause', exact: true}).click();
    await page.waitForFunction(() => window.__dvidiaLiveProbe.latest.paused);
    await page.getByRole('button', {name: 'Reset', exact: true}).click();
    await page.waitForFunction(() => window.__dvidiaLiveProbe.latest.sim_time === 0 && window.__dvidiaLiveProbe.latest.paused);
    await page.getByRole('button', {name: 'Resume', exact: true}).click();
    await page.waitForFunction(() => window.__dvidiaLiveProbe.latest.sim_time > .6);
    await page.evaluate(() => window.__dvidiaLiveProbe.begin());
    const stabilityStart = performance.now();
    const targets = [[.38, .02, .25], [.32, -.12, .20], [.40, .10, .28], manifest.initial_target_xyz];
    let nextTarget = 0; let lastLog = 0;
    while ((performance.now() - stabilityStart) / 1000 < seconds) {
      const elapsed = (performance.now() - stabilityStart) / 1000;
      if (elapsed >= nextTarget * 10) {
        await setTarget(page, targets[nextTarget % targets.length]);
        nextTarget += 1;
      }
      if (elapsed - lastLog >= 60) {
        const status = await page.evaluate(() => ({sim: window.__dvidiaLiveProbe.latest.sim_time,
          states: window.__dvidiaLiveProbe.state_count, errors: window.__dvidiaLiveProbe.errors.length}));
        console.log(JSON.stringify({progress_seconds: Math.round(elapsed), ...status})); lastLog = elapsed;
      }
      await page.waitForTimeout(Math.min(500, Math.max(1, (seconds - elapsed) * 1000)));
      assert.equal(await page.evaluate(() => document.visibilityState), 'visible', 'Stability benchmark must remain foreground');
      assert.equal(await page.evaluate(() => window.__dvidiaLiveProbe.errors.length), 0, 'Worker reported an error');
    }
    await page.evaluate(() => window.__dvidiaLiveProbe.end());
    probe = await page.evaluate(() => {
      const source = window.__dvidiaLiveProbe;
      return Object.fromEntries(Object.entries(source).filter(([, value]) => typeof value !== 'function'));
    });
    assert(probe.latest.metrics.samples_retained <= 2048, 'Worker timing retention must remain bounded');
    assert(probe.states.length <= 15000 && probe.commands.length <= 2000, 'Harness retention must remain bounded');
    const downloadPending = page.waitForEvent('download');
    await page.getByRole('button', {name: 'Download session data', exact: true}).click();
    const download = await downloadPending;
    const receiptPath = path.join(output, 'session-receipt.json'); await download.saveAs(receiptPath);
    const receipt = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
    assert.equal(receipt.source.manifest_sha256, manifestSha256);
    assert(receipt.samples.length <= 3000 && receipt.commands.length <= 2000);
    controls.receipt_export = true;
    controls.receipt_retention = receipt.recording;
    await page.screenshot({path: path.join(output, 'final-workstation.png'), fullPage: true});
    const wallSeconds = (probe.measure_end_ms - probe.measure_start_ms) / 1000;
    const simSeconds = probe.measure_end_sim - probe.measure_start_sim;
    const intervals = values => values.slice(1).map((value, index) => value - values[index]);
    const stableLatencies = probe.latencies.filter(item => item.baseline_stable && item.command_distance_m > .005);
    const measuredStates = probe.states.filter(state => state.wall_ms >= probe.measure_start_ms && state.wall_ms <= probe.measure_end_ms);
    const actualSoftware = /swiftshader|llvmpipe|software/i.test(renderer.renderer);
    const renderFps = (probe.measure_end_render_count - probe.measure_start_render_count) / wallSeconds;
    const numericalValidity = probe.native_finite && !probe.errors.length && !probe.warnings.length;
    const acceptancePassed = numericalValidity && !errors.length && !consoleErrors.length && !unexpectedRequests.length
      && !failedRequests.length && Object.values(controls).every(value => value !== false) && stableLatencies.some(item => item.physical_response_ms !== null);
    summary = {schema: 'dvidia.workstation-browser-benchmark/v1', passed: acceptancePassed,
      passed_scope: 'Functional UI acceptance and numerical validity; performance targets are reported separately.',
      functional_acceptance_passed: acceptancePassed,
      quantile_method: 'Nearest-rank empirical quantiles; sample counts are reported for each distribution.',
      run_started_at: probe.started_at, url, manifest_sha256: manifestSha256,
      native_scene_sha256: manifest.native_scene.sha256, engine: probe.latest.engine,
      browser: browser.version(), browser_channel: channel ?? 'playwright-chromium', headless,
      requested_renderer: software ? 'SwiftShader software' : 'device default', renderer,
      renderer_classification: actualSoftware ? 'software renderer; not a device GPU benchmark' : 'renderer reported above; hardware backing not independently established',
      host: {os: `${os.platform()} ${os.release()} ${os.arch()}`, cpu: os.cpus()[0]?.model,
        logical_cpus: os.cpus().length, ram_bytes: os.totalmem()},
      viewport: {width: 1440, height: 1080, device_scale_factor: 1}, load_to_ready_ms: loadMs,
      controls, numerical_validity: numericalValidity,
      performance_targets: {render_30_to_60_fps_met: renderFps >= 30 && renderFps <= 60.5,
        real_time_physics_met: simSeconds / wallSeconds >= .99,
        note: 'Performance depends on the recorded renderer, device and browser; failed targets do not change functional acceptance.'},
      stability: {requested_wall_seconds: seconds, measured_wall_seconds: wallSeconds, simulated_seconds: simSeconds,
        real_time_factor: simSeconds / wallSeconds, physics_steps: probe.measure_end_steps - probe.measure_start_steps,
        lag_added_ms: probe.measure_end_lag_ms - probe.measure_start_lag_ms,
        foreground_throughout: !probe.visibility_events.some(event => event.wall_ms >= probe.measure_start_ms
          && event.wall_ms <= probe.measure_end_ms && event.visibility !== 'visible'),
        foreground_definition: 'document.visibilityState remained visible in the isolated browser; headless runs have no OS-visible window.',
        warning_count: probe.warnings.length, native_state_samples: measuredStates.length,
        cyclic_target_interval_seconds: 10, cyclic_target_commands: nextTarget},
      rendering: {render_invocations: probe.measure_end_render_count - probe.measure_start_render_count,
        render_invocations_per_second: renderFps, render_interval_ms: distribution(intervals(probe.render_times)),
        raf_callbacks_per_second: probe.raf_times.length / wallSeconds, raf_interval_ms: distribution(intervals(probe.raf_times)),
        worker_states_per_second: probe.worker_times.length / wallSeconds, worker_state_interval_ms: distribution(intervals(probe.worker_times)),
        target_30_to_60_render_fps_met: renderFps >= 30 && renderFps <= 60.5,
        note: 'RAF cadence, Three.js render invocations and worker-state delivery are separate measurements. Render invocations do not measure GPU presentation completion.'},
      response_latency: {measurement: 'Main-thread command post to first received target echo, separately from first >=1 mm TCP movement projected toward the target.',
        stable_baseline: 'At least 5 recent states within 350 ms, spread below 0.5 mm; commanded distance exceeds 5 mm.',
        response_timeout_ms: 5000, superseded_commands: 'Unfinished responses remain missing when a newer target is issued.',
        all_target_commands: probe.latencies.length, stable_baseline_commands: stableLatencies.length,
        target_echo_ms: distribution(stableLatencies.map(item => item.target_echo_ms).filter(Number.isFinite)),
        physical_response_ms: distribution(stableLatencies.map(item => item.physical_response_ms).filter(Number.isFinite)),
        missing_physical_response: stableLatencies.filter(item => item.physical_response_ms === null).length},
      observed_worker_timing: {step_ms_p95: distribution(measuredStates.map(state => state.metrics.step_ms_p95)),
        control_ms_p95: distribution(measuredStates.map(state => state.metrics.control_ms_p95)),
        note: 'Distribution of worker rolling p95 snapshots, not an all-step timing distribution.'},
      memory: {browser_heap: await page.evaluate(() => performance.memory ? {used_js_heap_size: performance.memory.usedJSHeapSize,
        total_js_heap_size: performance.memory.totalJSHeapSize, js_heap_size_limit: performance.memory.jsHeapSizeLimit} : null),
        receipt_samples: receipt.samples.length, receipt_sample_limit: 3000, receipt_samples_dropped: receipt.recording.samples_dropped,
        harness_states_retained: probe.states.length, harness_state_limit: 15000, harness_states_dropped: probe.states_dropped,
        harness_frame_samples_dropped: probe.frame_samples_dropped, worker_timing_samples_retained: probe.latest.metrics.samples_retained},
      request_origins: [...requestOrigins], errors, console_errors: consoleErrors, failed_requests: failedRequests,
      response_errors: responseErrors,
      unexpected_requests: unexpectedRequests, warnings: probe.warnings,
      limitations: ['Rigid synthetic shoe with uncalibrated materials.', 'No shoe-grasp or real-robot skill success claim.',
        'Browser instrumentation adds overhead; software rendering results do not establish device-GPU performance.']};
    await fs.writeFile(path.join(output, 'browser-benchmark.json'), JSON.stringify(summary, null, 2) + '\n');
    await fs.writeFile(path.join(output, 'browser-raw-samples.json'), JSON.stringify(probe, null, 2) + '\n');
    console.log(JSON.stringify(summary, null, 2));
    if (!acceptancePassed) process.exitCode = 1;
  } catch (error) {
    if (page) {
      probe = await page.evaluate(() => window.__dvidiaLiveProbe ? Object.fromEntries(Object.entries(window.__dvidiaLiveProbe)
        .filter(([, value]) => typeof value !== 'function')) : null).catch(() => null);
      await fs.writeFile(path.join(output, 'browser-failure.json'), JSON.stringify({passed: false, error: error.stack,
        errors, consoleErrors, unexpectedRequests, failedRequests, responseErrors, probe}, null, 2) + '\n');
    }
    throw error;
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
