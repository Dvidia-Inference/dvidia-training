import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {webcrypto} from 'node:crypto';
import test from 'node:test';
import vm from 'node:vm';
import {gzipSync} from 'node:zlib';

// Exercise the exact viewer transport without starting Three.js or the DOM.
const source = readFileSync(new URL('../../src/dvidia_training/piper_imitation_viewer.js', import.meta.url), 'utf8');
const transport = source.slice(source.indexOf('function assert('), source.indexOf('function validate('));
const base = 'https://dvidia-robot-simulation-lab.static.hf.space/learned-placement/v01/viewer/';
const hub = 'https://huggingface.co/spaces/Dvidia/robot-simulation-lab/resolve/main/learned-placement/v01/viewer/';
function viewer(url, fetch = () => { throw new Error('Unexpected fetch'); }, overrides = {}) {
  const context = vm.createContext({URL, location: new URL(url), fetch, MAX_BYTES: 64 * 1024 * 1024,
    AbortController, Blob, TextDecoder, Uint8Array, crypto: webcrypto, DecompressionStream,
    setTimeout, clearTimeout, ...overrides});
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
  assert.deepEqual({...calls[0].options}, {credentials: 'omit', referrerPolicy: 'no-referrer', redirect: 'follow', signal: undefined});
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
  assert.equal(calls[0].options.credentials, 'same-origin');
});

const localBase = 'https://research.dvidia.org/study/viewer/';
const raw = new TextEncoder().encode('A small native binary fixture.');
const compressed = gzipSync(raw);
const sha256 = Buffer.from(await webcrypto.subtle.digest('SHA-256', raw)).toString('hex');
const meta = {href: 'motion.bin', bytes: raw.byteLength, sha256, compressed_href: 'motion.bin.gz',
  compressed_bytes: compressed.byteLength, compression: 'gzip'};
const quickDeadlines = {setTimeout: (callback, ms) => setTimeout(callback, ms === 8000 ? 15 : 100)};
function fixture(url, data) {
  return {url: url.href, ok: true, body: new Blob([data]).stream()};
}

test('a valid compressed fixture is length- and SHA-verified without fetching the raw copy', async () => {
  const calls = [], app = viewer(localBase, async (url, options) => {
    calls.push({url: url.href, options}); return fixture(url, compressed);
  });
  const bytes = await app.binary(meta, 'motion.bin');
  assert.deepEqual(Buffer.from(bytes), Buffer.from(raw));
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, localBase + 'motion.bin.gz');
  assert.equal(calls[0].options.credentials, 'same-origin');
  assert.equal(calls[0].options.signal.aborted, true);
});

test('hung gzip headers abort at the compressed deadline and fall back to verified raw bytes', async () => {
  const calls = [], app = viewer(localBase, async (url, options) => {
    calls.push({url: url.href, signal: options.signal});
    return url.pathname.endsWith('.gz') ? new Promise(() => {}) : fixture(url, raw);
  }, quickDeadlines);
  assert.deepEqual(Buffer.from(await app.binary(meta, 'motion.bin')), Buffer.from(raw));
  assert.deepEqual(calls.map(call => call.url), [localBase + 'motion.bin.gz', localBase + 'motion.bin']);
  assert.equal(calls[0].signal.aborted, true);
});

test('an HTTP error aborts its unread response before the raw fallback begins', async () => {
  let failedSignal;
  const app = viewer(localBase, async (url, options) => {
    if (url.pathname.endsWith('.gz')) {
      failedSignal = options.signal;
      return {url: url.href, ok: false, status: 403, body: new ReadableStream()};
    }
    assert.equal(failedSignal.aborted, true);
    return fixture(url, raw);
  });
  assert.deepEqual(Buffer.from(await app.binary(meta, 'motion.bin')), Buffer.from(raw));
});

test('hosted gzip network errors fall back to the fixed Hub raw URL with credentials omitted', async () => {
  const calls = [], app = viewer(base + 'index.html', async (url, options) => {
    calls.push({url: url.href, options});
    if (url.pathname.endsWith('.gz')) throw new TypeError('Network error');
    return {url: 'https://us.aws.cdn.hf.co/xet-bridge-us/verified-fixture', ok: true, body: new Blob([raw]).stream()};
  });
  assert.deepEqual(Buffer.from(await app.binary(meta, 'motion.bin')), Buffer.from(raw));
  assert.deepEqual(calls.map(call => call.url), [hub + 'motion.bin.gz', hub + 'motion.bin']);
  for (const call of calls) {
    assert.equal(call.options.credentials, 'omit');
    assert.equal(call.options.referrerPolicy, 'no-referrer');
    assert.equal(call.options.redirect, 'follow');
    assert.equal(call.options.signal.aborted, true);
  }
});

test('hung gzip body reads are cancelled and raw fallback can complete', async () => {
  let cancelled = false;
  const app = viewer(localBase, async url => url.pathname.endsWith('.gz') ?
    {url: url.href, ok: true, body: new ReadableStream({cancel() { cancelled = true; }})} : fixture(url, raw), quickDeadlines);
  assert.deepEqual(Buffer.from(await app.binary(meta, 'motion.bin')), Buffer.from(raw));
  assert.equal(cancelled, true);
});

test('the gzip deadline also covers decompression, even if stream cancellation never completes', async () => {
  class StalledDecompression {
    constructor() { return new TransformStream({transform() { return new Promise(() => {}); }}); }
  }
  const calls = [], app = viewer(localBase, async url => {
    calls.push(url.href); return fixture(url, url.pathname.endsWith('.gz') ? compressed : raw);
  }, {...quickDeadlines, DecompressionStream: StalledDecompression});
  assert.deepEqual(Buffer.from(await app.binary(meta, 'motion.bin')), Buffer.from(raw));
  assert.equal(calls.length, 2);
});

test('corrupt raw fallback cannot pass the original binary SHA-256', async () => {
  const corrupt = raw.slice(); corrupt[0] ^= 1;
  const app = viewer(localBase, async url => url.pathname.endsWith('.gz') ?
    new Promise(() => {}) : fixture(url, corrupt), quickDeadlines);
  await assert.rejects(app.binary(meta, 'motion.bin'), /Study binary SHA256 changed/);
});

test('a raw copy exceeding its declared size is rejected', async () => {
  const app = viewer(localBase, async url => fixture(url, new Uint8Array(raw.length + 1)), {DecompressionStream: undefined});
  await assert.rejects(app.binary(meta, 'motion.bin'), /exceeds its byte bound/);
});

test('a stalled raw fallback ends with an actionable error rather than indefinite loading', async () => {
  const app = viewer(localBase, async () => new Promise(() => {}), quickDeadlines);
  await assert.rejects(app.binary(meta, 'motion.bin'), /Timed out loading motion\.bin after 30 s.*open the complete download locally/);
});

test('the metadata deadline covers a stalled body after response headers arrive', async () => {
  const app = viewer(localBase, async url => ({url: url.href, ok: true, body: new ReadableStream()}), quickDeadlines);
  await assert.rejects(app.json('motion.json'), /Timed out loading motion\.json after 30 s/);
});
