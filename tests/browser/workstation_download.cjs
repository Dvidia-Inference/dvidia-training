/* DVIDIA workstation transport recovery acceptance. MIT.
 * Usage: node tests/browser/workstation_download.cjs URL NEW_OUTPUT_DIRECTORY
 * Uses the real rendered scene and native MuJoCo worker. Only HTTP requests
 * are fault-injected; no model, controller, state or physics responses are mocked.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require(process.env.DVIDIA_PLAYWRIGHT_MODULE
  || path.join(__dirname, '../../viewer/node_modules/playwright'));

const ASSETS = ['geometry.bin.gz', 'native-assets.bin.gz', 'mujoco.wasm.gz'];
const MAX_ATTEMPTS = 3;
const cleanURL = value => { const url = new URL(value); return url.origin + url.pathname; };
const assetName = value => new URL(value).pathname.split('/').pop();
const getState = page => page.evaluate(() => window.__workstationState);

async function instrument(context) {
  await context.addInitScript(() => {
    window.__downloadFaultEvidence = {loading: [], worker: []};
    let lastText;
    new MutationObserver(() => {
      const text = document.querySelector('#loading-message')?.textContent;
      if (text && text !== lastText) {
        lastText = text;
        if (window.__downloadFaultEvidence.loading.length < 500) {
          window.__downloadFaultEvidence.loading.push({ms: performance.now(), text});
        }
      }
    }).observe(document, {subtree: true, childList: true, characterData: true});
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', event => {
          const message = event.data;
          if (message?.state) window.__workstationState = message.state;
          if (message?.type === 'progress' && message.download) {
            // Retain every retry and completion, plus the start of each attempt.
            const item = message.download;
            if (item.phase !== 'downloading' || item.receivedBytes === 0) {
              window.__downloadFaultEvidence.worker.push({ms: performance.now(), ...item});
            }
          }
        });
      }
    };
  });
}

function capture(page) {
  const evidence = {pageErrors: [], httpErrors: [], requestFailures: []};
  page.on('pageerror', error => evidence.pageErrors.push(error.message));
  page.on('response', response => {
    if (response.status() >= 400) evidence.httpErrors.push({url: cleanURL(response.url()), status: response.status()});
  });
  page.on('requestfailed', request => evidence.requestFailures.push({
    url: cleanURL(request.url()), error: request.failure()?.errorText,
  }));
  return evidence;
}

async function ready(page) {
  await page.waitForFunction(() => ['true', 'error'].includes(document.querySelector('#stage')?.dataset.ready),
    {}, {timeout: 120000});
  assert.equal(await page.locator('#stage').getAttribute('data-ready'), 'true',
    await page.locator('#loading-message').textContent());
  await page.waitForFunction(() => window.__workstationState?.sim_time > .3);
}

async function nativeControls(page, directory) {
  const before = await getState(page);
  await page.locator('#pause').click();
  await page.waitForFunction(() => window.__workstationState?.paused === true);
  const paused = await getState(page);
  await page.waitForTimeout(300);
  assert.equal((await getState(page)).sim_time, paused.sim_time, 'Paused native simulation must stay still');
  await page.locator('#pause').click();
  await page.waitForFunction(() => window.__workstationState?.paused === false);
  for (const [axis, value] of [['x', '370'], ['y', '-100'], ['z', '240']]) {
    await page.locator('#target-' + axis).fill(value);
  }
  await page.locator('#apply-target').click();
  await page.waitForFunction(() => Math.abs(window.__workstationState.target_position[0] - .37) < 1e-8
    && Math.abs(window.__workstationState.target_position[1] + .1) < 1e-8);
  await page.waitForFunction(() => window.__workstationState.tracking_error_m < .01, {}, {timeout: 15000});
  const reached = await getState(page);
  assert(reached.qpos.some((value, index) => Math.abs(value - before.qpos[index]) > .01),
    'Native joints must move in response to the target');
  await page.locator('#gripper').evaluate(element => {
    element.value = '20'; element.dispatchEvent(new Event('input', {bubbles: true}));
  });
  await page.waitForFunction(() => Math.abs(window.__workstationState.jaw - .02) < 1e-8
    && Math.abs(window.__workstationState.actual_jaw - .02) < .002, {}, {timeout: 10000});
  const prePush = (await getState(page)).shoes.shoe_trainer_left.position;
  await page.locator('#push').click();
  await page.waitForFunction(position => Math.hypot(...window.__workstationState.shoes.shoe_trainer_left.position
    .map((value, index) => value - position[index])) > .001, prePush, {timeout: 10000});
  const afterPush = await getState(page);
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#receipt').click();
  const download = await downloadPromise;
  await download.saveAs(path.join(directory, 'session.json'));
  const receipt = JSON.parse(await fs.readFile(path.join(directory, 'session.json'), 'utf8'));
  assert(receipt.commands.some(command => command.type === 'target'));
  assert(receipt.commands.some(command => command.type === 'push' && command.impulse));
  assert(receipt.samples.length > 0 && receipt.samples.some(sample => sample.body_pos.length > 3));
  return {pauseStable: true, targetErrorM: reached.tracking_error_m,
    actualJawM: afterPush.actual_jaw, shoeBefore: prePush, shoeAfter: afterPush.shoes.shoe_trainer_left.position,
    receiptCommands: receipt.commands.length, receiptSamples: receipt.samples.length};
}

async function transientCase(browser, url, directory) {
  const context = await browser.newContext({viewport: {width: 1440, height: 1100}, serviceWorkers: 'block'});
  await instrument(context);
  const counts = Object.fromEntries(ASSETS.map(name => [name, 0]));
  await context.route('**/*', async route => {
    const name = assetName(route.request().url());
    if (Object.hasOwn(counts, name) && ++counts[name] === 1) return route.abort('failed');
    return route.continue();
  });
  const page = await context.newPage(); const network = capture(page); const started = performance.now();
  try {
    const navigation = await page.goto(url, {waitUntil: 'domcontentloaded', timeout: 60000});
    assert(navigation?.ok());
    await ready(page);
    const readyMs = performance.now() - started;
    for (const name of ASSETS) assert.equal(counts[name], 2, name + ' must recover on exactly one retry');
    const progress = await page.evaluate(() => window.__downloadFaultEvidence);
    for (const name of ['native-assets.bin.gz', 'mujoco.wasm.gz']) {
      assert(progress.worker.some(item => item.phase === 'retrying' && item.path.endsWith('/' + name)),
        name + ' worker retry must emit structured progress');
    }
    assert(progress.loading.some(item => /retry/i.test(item.text) && item.text.includes('Scene geometry')),
      'Geometry retry must be visible in the loading status');
    const controls = await nativeControls(page, directory);
    assert.deepEqual(network.pageErrors, []); assert.deepEqual(network.httpErrors, []);
    await page.screenshot({path: path.join(directory, 'recovered.png'), fullPage: true});
    return {passed: true, fault: 'First request for each large binary aborted before headers', counts,
      readyMs, progress, controls, network};
  } catch (error) {
    await page.screenshot({path: path.join(directory, 'failure.png'), fullPage: true}).catch(() => {});
    throw Object.assign(error, {caseEvidence: {counts, network,
      progress: await page.evaluate(() => window.__downloadFaultEvidence).catch(() => null)}});
  } finally { await context.close(); }
}

async function terminalCase(browser, url, directory) {
  const context = await browser.newContext({viewport: {width: 1440, height: 1100}, serviceWorkers: 'block'});
  await instrument(context);
  let blocked = true, requests = 0;
  await context.route('**/*', async route => {
    if (assetName(route.request().url()) === 'geometry.bin.gz') {
      requests += 1;
      if (blocked) return route.abort('failed');
    }
    return route.continue();
  });
  const page = await context.newPage(); const network = capture(page);
  try {
    const navigation = await page.goto(url, {waitUntil: 'domcontentloaded', timeout: 60000});
    assert(navigation?.ok());
    await page.waitForFunction(() => document.querySelector('#stage')?.dataset.ready === 'error', {}, {timeout: 120000});
    assert.equal(requests, MAX_ATTEMPTS, 'Persistent transport failure must stop after three attempts');
    const message = await page.locator('#loading-message').textContent();
    assert.match(message, /geometry\.bin\.gz/);
    assert.match(message, /attempt 3\/3/);
    assert(await page.locator('#retry-download').isVisible(), 'A retry control must remain visible after terminal failure');
    assert(await page.locator('#retry-download').isEnabled(), 'A retry control must remain enabled after terminal failure');
    const failedProgress = await page.evaluate(() => window.__downloadFaultEvidence);
    await page.screenshot({path: path.join(directory, 'named-failure-and-retry.png'), fullPage: true});
    blocked = false;
    const started = performance.now();
    await Promise.all([page.waitForEvent('domcontentloaded'), page.locator('#retry-download').click()]);
    await ready(page);
    assert.equal(requests, MAX_ATTEMPTS + 1, 'User retry must start one new successful geometry request');
    assert.deepEqual(network.pageErrors, []); assert.deepEqual(network.httpErrors, []);
    await page.locator('#pause').click();
    await page.waitForFunction(() => window.__workstationState?.paused === true);
    await page.screenshot({path: path.join(directory, 'manual-retry-recovered.png'), fullPage: true});
    return {passed: true, fault: 'Every geometry request aborted until explicit user retry', requests,
      terminalMessage: message, retryEnabled: true, recoveredReadyMs: performance.now() - started,
      failedProgress, recoveredNativeSimTime: (await getState(page)).sim_time, network};
  } catch (error) {
    await page.screenshot({path: path.join(directory, 'failure.png'), fullPage: true}).catch(() => {});
    throw Object.assign(error, {caseEvidence: {requests, blocked, network,
      progress: await page.evaluate(() => window.__downloadFaultEvidence).catch(() => null)}});
  } finally { await context.close(); }
}

(async () => {
  const [url, output] = process.argv.slice(2);
  assert(url && output, 'Provide a workstation URL and a new output directory');
  assert(['http:', 'https:'].includes(new URL(url).protocol));
  await fs.mkdir(output, {recursive: true});
  // Refuse to silently overwrite a prior verdict, especially a recorded failure.
  await fs.access(path.join(output, 'acceptance.json')).then(() => {
    throw new Error('Use a fresh output directory; acceptance.json already exists');
  }, () => {});
  const browser = await chromium.launch({headless: true});
  const result = {passed: false, url: cleanURL(url), scope: 'Real-browser transport recovery with native MuJoCo physics; synthetic HTTP faults only.', cases: {}};
  try {
    for (const [name, run] of [['transient', transientCase], ['terminal', terminalCase]]) {
      const directory = path.join(output, name); await fs.mkdir(directory, {recursive: true});
      try { result.cases[name] = await run(browser, url, directory); }
      catch (error) { result.cases[name] = {passed: false, error: error.message, ...error.caseEvidence}; throw error; }
    }
    result.passed = true;
  } catch (error) { result.error = error.message; process.exitCode = 1; }
  finally {
    await fs.writeFile(path.join(output, 'acceptance.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result, null, 2)); await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
