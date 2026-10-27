import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Exercise the exact viewer transport without starting Three.js or the DOM.
const source = readFileSync(new URL('../../src/dvidia_training/piper_imitation_viewer.js', import.meta.url), 'utf8');
const transport = source.slice(source.indexOf('function assert('), source.indexOf('async function bounded('));
const base = 'https://dvidia-robot-simulation-lab.static.hf.space/learned-placement/v01/viewer/';
const hub = 'https://huggingface.co/spaces/Dvidia/robot-simulation-lab/resolve/main/learned-placement/v01/viewer/';
function viewer(url, fetch = () => { throw new Error('Unexpected fetch'); }) {
  const context = vm.createContext({URL, location: new URL(url), fetch});
  vm.runInContext(transport, context);
  return context;
}

test('only this hosted viewer maps binary resources to the fixed public Hub path', () => {
  const app = viewer(base + 'index.html');
  for (const path of ['geometry.bin', 'motion.bin.gz', 'episodes/learned.json.gz', 'data/actions.npz']) {
    assert.equal(app.studyUrl(path).href, hub + path);
  }
  for (const path of ['scene.json', 'motion.json', 'report.json', 'README.md', 'viewer.js']) {
    assert.equal(app.studyUrl(path).href, base + path);
  }
});

test('local, research, unrelated Space and unrelated namespaces retain relative URLs', () => {
  for (const url of [
    'http://127.0.0.1:8303/index.html',
    'https://research.dvidia.org/downloads/topics/physical-grounding/run/viewer/',
    'https://unrelated.static.hf.space/learned-placement/v01/viewer/',
    'https://dvidia-robot-simulation-lab.static.hf.space/workstation/v01/',
    'https://dvidia-robot-simulation-lab.static.hf.space/learned-placement/v010/viewer/',
    'https://dvidia-robot-simulation-lab.static.hf.space:8443/learned-placement/v01/viewer/',
  ]) assert.equal(viewer(url).studyUrl('motion.bin.gz').href, new URL('motion.bin.gz', url).href);
});

test('resource paths reject traversal, absolute URLs, encoded escapes and queries', () => {
  const app = viewer(base);
  for (const path of ['../model.json', './motion.bin', '/motion.bin', 'https://example.org/a.bin',
    'episodes/../x.bin', 'episodes/%2e%2e/x.bin', 'episodes\\x.bin', 'motion.bin?x=1', 'motion.bin#fragment']) {
    assert.throws(() => app.studyUrl(path), /inside the exported directory/);
  }
});

test('only the exact Hub request or observed HTTPS CDN can finish a mapped download', () => {
  const app = viewer(base), request = app.studyUrl('motion.bin.gz');
  assert.equal(app.trustedResponseUrl(new URL(request), request), true);
  assert.equal(app.trustedResponseUrl(new URL('https://us.aws.cdn.hf.co/xet-bridge-us/blob?signature=opaque'), request), true);
  for (const url of [
    hub + 'another.bin.gz',
    'https://huggingface.co/spaces/Other/space/resolve/main/motion.bin.gz',
    'https://us.aws.cdn.hf.co.evil.test/blob',
    'https://evil.us.aws.cdn.hf.co/blob',
    'https://us.aws.cdn.hf.co:8443/blob',
    'https://user:password@us.aws.cdn.hf.co/blob',
    'http://us.aws.cdn.hf.co/blob',
    'https://cdn.hf.co/blob',
    base + 'motion.bin.gz',
  ]) assert.equal(app.trustedResponseUrl(new URL(url), request), false, url);
});

test('ordinary relative downloads cannot redirect to the Hub or CDN', () => {
  for (const url of ['http://127.0.0.1:8303/', base]) {
    const app = viewer(url), request = app.studyUrl('scene.json');
    assert.equal(app.trustedResponseUrl(new URL(request), request), true);
    assert.equal(app.trustedResponseUrl(new URL(hub + 'scene.json'), request), false);
    assert.equal(app.trustedResponseUrl(new URL('https://us.aws.cdn.hf.co/blob'), request), false);
  }
});

test('mapped fetch omits credentials and referrer and validates its final URL', async () => {
  const calls = [];
  const app = viewer(base, async (url, options) => {
    calls.push({url: url.href, options});
    return {url: 'https://us.aws.cdn.hf.co/xet-bridge-us/blob', ok: true};
  });
  await app.response('motion.bin.gz');
  assert.equal(calls[0].url, hub + 'motion.bin.gz');
  assert.deepEqual({...calls[0].options}, {credentials: 'omit', referrerPolicy: 'no-referrer', redirect: 'follow'});
});

test('same-origin fetch rejects redirects and untrusted final destinations', async () => {
  const calls = [];
  const app = viewer(base, async (url, options) => {
    calls.push({url: url.href, options});
    return {url: 'https://evil.test/scene.json', ok: true};
  });
  await assert.rejects(app.response('scene.json'), /Unexpected study download origin/);
  assert.equal(calls[0].url, base + 'scene.json');
  assert.equal(calls[0].options.redirect, 'error');
});
