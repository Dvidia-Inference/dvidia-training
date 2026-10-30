/* Browser-independent report contracts. Synthetic data only; no files are served. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, 'piper_action_capture_report.html'), 'utf8');
const source = fs.readFileSync(path.join(__dirname, 'piper_action_capture_report.js'), 'utf8');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);

class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.listeners = {};
    this.attrs = {};
    this.dataset = {};
    this.hidden = false;
    this.currentTime = 0;
    this.sourceWrites = [];
    this.playCalls = 0;
    this.pauseCalls = 0;
    this.classList = { toggle() {} };
  }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  getAttribute(key) { return this.attrs[key] ?? null; }
  removeAttribute(key) { delete this.attrs[key]; }
  set src(value) { this.attrs.src = value; this.sourceWrites.push(value); }
  get src() { return this.attrs.src; }
  set href(value) { this.attrs.href = value; }
  get href() { return this.attrs.href; }
  addEventListener(event, callback) { this.listeners[event] = callback; }
  querySelectorAll(tag) { return this.children.filter((child) => child.tag === tag); }
  emit(event) { this.listeners[event]?.(); }
  pause() { this.pauseCalls += 1; }
  play() { this.playCalls += 1; return Promise.resolve(); }
  load() { this.currentTime = 0; }
}

function makeEpisode(id, snapshots, status) {
  return {
    id, label: id, status, snapshots, steps: snapshots - 1,
    native_success: status === 'complete',
    recorded_task_result: status === 'complete' ? 'unverified' : 'unknown',
    wall_seconds: 1, warnings: [],
    rows: Array.from({ length: snapshots }, (_, index) => ({
      t: index * 0.02,
      rgb: `evidence/registered/${id}/frames/rgb-${index}.png`,
      depth: `evidence/registered/${id}/frames/depth-${index}.npy`,
      measured: [1, 2, 3, 4, 5, 6, 0.01], jaw_pair: [0.01, -0.01],
      velocity: [1, 2, 3, 4, 5, 6, 0.001, -0.001],
      issued: index === 0 ? null : [1, 2, 3, 4, 5, 6, 0.02],
      ramped: [1, 2, 3, 4, 5, 6, 0.015],
      native: index === 0 ? null : [1, 2, 3, 4, 5, 6, 0.019]
    }))
  };
}

function fixture(compact = false) {
  const data = {
    summary: {
      repeat: {
        snapshot_pairs: 1001, identical_rgb_and_depth_frame_pairs: 1001,
        max_difference: 0, same_commands: true, same_controls: true
      },
      missing_tracker_channels: 4
    },
    episodes: [
      makeEpisode('nominal-full', 1001, 'complete'),
      makeEpisode('nominal-repeat', 1001, 'complete'),
      makeEpisode('interrupted-prefix', 3, 'interrupted')
    ]
  };
  if (compact) {
    data.download_archive_url = 'https://huggingface.co/datasets/Dvidia/example/resolve/main/original.zip';
    data.episodes.forEach((episode) => { episode.video = `videos/${episode.id}.mp4`; });
  }
  return data;
}

async function boot(data) {
  const elements = Object.fromEntries(ids.map((id) => [id, new Element()]));
  elements.report.hidden = true;
  const document = {
    getElementById(id) { assert.ok(elements[id], `Missing HTML element: ${id}`); return elements[id]; },
    createElement(tag) { return new Element(tag); },
    addEventListener() {}, hidden: false
  };
  let scheduledFrame;
  const window = {
    matchMedia() { return { matches: true, addEventListener() {} }; },
    setTimeout, clearTimeout, cancelAnimationFrame() {},
    requestAnimationFrame(callback) { scheduledFrame = callback; return 1; }
  };
  vm.runInNewContext(source, {
    document, window, Intl, URL, AbortController, performance: { now: () => 0 },
    fetch: async (url) => {
      assert.equal(url, './viewer-data.json', 'The report fetches only its local index');
      return { ok: true, json: async () => data };
    }
  });
  await new Promise(setImmediate);
  return { elements, tick: (now) => scheduledFrame(now) };
}

function seek(elements, index) {
  elements.timeline.value = String(index);
  elements.timeline.emit('input');
}

test('HTML IDs are unique and depth is never described as a PNG preview', () => {
  assert.equal(new Set(ids).size, ids.length);
  assert.match(html, /id="show-depth"[^>]*disabled[^>]*Raw depth is an NPY array, no image preview/);
  assert.doesNotMatch(html + source, /\bdepth\s+(?:PNG|NPY\/PNG)/i);
});

test('every full-kit snapshot displays only RGB and links its exact raw NPY', async () => {
  const data = fixture();
  const { elements: e } = await boot(data);
  assert.equal(e.report.hidden, false);
  assert.equal(e.play.textContent, 'Play');
  for (const [episodeIndex, episode] of data.episodes.entries()) {
    e['episode-list'].children[episodeIndex].emit('click');
    assert.equal(e['show-depth'].disabled, true);
    assert.equal(e['show-depth'].title, 'Raw depth is an NPY array, no image preview');
    assert.equal(e['recording-link'].href, `evidence/registered/${episode.id}/recording.json`);
    for (const [index, row] of episode.rows.entries()) {
      seek(e, index);
      e['show-depth'].emit('click'); // Even a programmatic click cannot select raw depth.
      assert.equal(e['frame-image'].src, row.rgb);
      assert.equal(e['frame-source'].href, row.rgb);
      assert.equal(e['depth-source'].href, row.depth);
      assert.equal(e['depth-source'].hidden, false);
      assert.match(e['depth-source'].textContent, /raw depth NPY/);
      assert.match(e['camera-label'].textContent, /^RGB camera/);
    }
  }
  assert.ok(e['frame-image'].sourceWrites.length > 2000);
  assert.ok(e['frame-image'].sourceWrites.every((value) => value.includes('/rgb-') && value.endsWith('.png')));
  assert.ok(e['frame-image'].sourceWrites.every((value) => !value.endsWith('.npy')));
  assert.equal(e.q8.textContent, '-0.01000');
  assert.equal(e.qd8.textContent, '-0.00100');
});

test('every compact snapshot loads no raw image and routes depth/recordings to the ZIP', async () => {
  const data = fixture(true);
  const { elements: e } = await boot(data);
  assert.equal(e.report.hidden, false);
  for (const [episodeIndex, episode] of data.episodes.entries()) {
    e['episode-list'].children[episodeIndex].emit('click');
    assert.equal(e['frame-video'].src, episode.video);
    assert.equal(e['show-depth'].disabled, true);
    assert.equal(e['recording-link'].href, data.download_archive_url);
    assert.equal(e['capture-link'].href, data.download_archive_url);
    assert.equal(e['audit-link'].href, `audits/${episode.id}-audit.json`);
    for (const [index, row] of episode.rows.entries()) {
      seek(e, index);
      e['show-depth'].emit('click');
      assert.equal(e['frame-video'].currentTime, row.t);
      assert.equal(e['frame-source'].href, data.download_archive_url);
      assert.equal(e['depth-source'].href, data.download_archive_url);
      assert.equal(e['depth-source'].hidden, false);
      assert.match(e['depth-source'].textContent, /original ZIP/);
    }
  }
  assert.deepEqual(e['frame-image'].sourceWrites, []);
  assert.ok(e['frame-video'].sourceWrites.every((value) => value.endsWith('.mp4')));
  assert.match(e['camera-label'].textContent, /lossy derived MP4/);
});

test('video playback follows media time and does not claim an early end is the final snapshot', async () => {
  const { elements: e, tick } = await boot(fixture(true));
  e.play.emit('click');
  e['frame-video'].currentTime = 13.001;
  tick(999999);
  assert.equal(e.timeline.value, '650');
  assert.equal(e['frame-video'].currentTime, 13.001);
  e['frame-video'].currentTime = 20.02;
  tick(1000000);
  assert.equal(e.timeline.value, '1000');
  assert.equal(e.play.textContent, 'Play');
  e['episode-list'].children[2].emit('click');
  e['frame-video'].currentTime = 0.03;
  e['frame-video'].emit('ended');
  assert.equal(e.timeline.value, '1');
  assert.equal(e['image-error'].hidden, false);
  assert.match(e['image-error'].textContent, /ended before/);
  assert.deepEqual(e['frame-image'].sourceWrites, []);
});

test('compact mode without an archive hides the depth link instead of requesting absent arrays', async () => {
  const data = fixture(true);
  delete data.download_archive_url;
  const { elements: e } = await boot(data);
  assert.equal(e['depth-source'].hidden, true);
  assert.equal(e['depth-source'].getAttribute('href'), null);
  assert.equal(e['frame-source'].href, data.episodes[0].video);
  assert.deepEqual(e['frame-image'].sourceWrites, []);
});

test('external media paths are refused before playback', async () => {
  const data = fixture(true);
  data.episodes[0].video = 'https://example.invalid/capture.mp4';
  const { elements: e } = await boot(data);
  assert.equal(e.report.hidden, true);
  assert.equal(e.retry.hidden, false);
  assert.deepEqual(e['frame-image'].sourceWrites, []);
  assert.deepEqual(e['frame-video'].sourceWrites, []);
});
