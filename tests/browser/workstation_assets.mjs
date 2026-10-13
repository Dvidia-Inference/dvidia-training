import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createHash} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import {unpackNativeAssets, loadNativeAssets, loadWasmBytes, resourceBytes, boundedStreamBytes, MAX_RESOURCE_BYTES} from '../../src/dvidia_training/workstation_assets.js';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const original = [new TextEncoder().encode('first exact mesh\n'), new Uint8Array([0, 2, 4, 255])];
const assets = original.map((bytes, index) => ({path: `assets/${index}.obj`, bytes: bytes.length, sha256: hash(bytes)}));
const combined = Buffer.concat(original);
const pack = bytes => {
  const packed = gzipSync(bytes);
  return {packed, bundle: {path: 'native-assets.bin.gz', compression: 'gzip', bytes: packed.length,
    sha256: hash(packed), uncompressed_bytes: combined.length}};
};
const base = new URL('https://example.org/workstation/workstation.json');
const fast = {idleTimeoutMs: 100, timeoutMs: 1000, retryDelayMs: 0};
const response = (body, url, init) => {
  const value = new Response(body, init);
  Object.defineProperty(value, 'url', {value: String(url)});
  return value;
};
async function usingFetch(mock, run) {
  const originalFetch = globalThis.fetch;
  try { globalThis.fetch = mock; return await run(); }
  finally { globalThis.fetch = originalFetch; }
}

test('one gzip bundle restores original mesh bytes and ordered native paths', async () => {
  const {packed, bundle} = pack(combined);
  const restored = await unpackNativeAssets(packed, bundle, assets);
  assert.deepEqual(restored.map(item => item.path), assets.map(item => item.path));
  restored.forEach((item, index) => assert.deepEqual(item.bytes, original[index]));
});

test('bundle and individual mesh checksums both gate model loading', async () => {
  const {packed, bundle} = pack(combined);
  await assert.rejects(unpackNativeAssets(packed, {...bundle, sha256: '0'.repeat(64)}, assets), /checksum/);
  const corrupt = Buffer.from(combined); corrupt[1] ^= 1;
  const resealed = pack(corrupt);
  await assert.rejects(unpackNativeAssets(resealed.packed, resealed.bundle, assets), /checksum/);
});

test('truncated, oversized, trailing and malformed compressed payloads fail', async () => {
  for (const bytes of [combined.subarray(0, -1), Buffer.concat([combined, Buffer.alloc(200000)]), Buffer.from('bad')]) {
    const {packed, bundle} = pack(bytes);
    await assert.rejects(unpackNativeAssets(packed, bundle, assets));
  }
  const {packed, bundle} = pack(combined);
  const truncated = packed.subarray(0, -8);
  await assert.rejects(unpackNativeAssets(truncated, {...bundle, bytes: truncated.length, sha256: hash(truncated)}, assets));
  await assert.rejects(unpackNativeAssets(packed, {...bundle, uncompressed_bytes: MAX_RESOURCE_BYTES * 4}, assets));
});

test('decoded streaming bound cancels excessive output before retaining it', async () => {
  let cancelled = false;
  const stream = new ReadableStream({start(controller) { controller.enqueue(new Uint8Array(20)); },
    cancel() { cancelled = true; }});
  await assert.rejects(boundedStreamBytes(stream, 10), /too large/);
  assert.equal(cancelled, true);
});

test('loader requests only the bundle; old manifests retain raw-file loading', async () => {
  const {packed, bundle} = pack(combined);
  const originalFetch = globalThis.fetch; const calls = [];
  try {
    globalThis.fetch = async url => {
      calls.push(url.pathname);
      const index = assets.findIndex(asset => url.pathname.endsWith(asset.path));
      return response(index >= 0 ? original[index] : packed, url);
    };
    const base = new URL('https://example.org/workstation/workstation.json');
    const result = await loadNativeAssets({assets, native_asset_bundle: bundle}, base);
    assert.deepEqual(calls, ['/workstation/native-assets.bin.gz']);
    assert.equal(result.length, 2);
    calls.length = 0;
    const legacy = await loadNativeAssets({assets}, base);
    assert.deepEqual(calls, assets.map(asset => '/workstation/' + asset.path));
    legacy.forEach((item, index) => assert.deepEqual(item.bytes, original[index]));
    await assert.rejects(loadNativeAssets({assets: [{...assets[0], path: '../escape'}], native_asset_bundle: bundle}, base));
  } finally { globalThis.fetch = originalFetch; }
});

test('interrupted body retries from the start and verifies only the completed download', async () => {
  const progress = [], options = [], calls = [];
  await usingFetch(async (url, init) => {
    calls.push(url.pathname); options.push(init);
    if (calls.length === 1) {
      let reads = 0;
      return response(new ReadableStream({pull(controller) {
        if (reads++ === 0) controller.enqueue(original[0].subarray(0, 3));
        else controller.error(new TypeError('network error'));
      }}), url);
    }
    return response(original[0], url);
  }, async () => {
    assert.deepEqual(await resourceBytes(new URL(assets[0].path, base), assets[0],
      {...fast, onProgress: value => progress.push(value)}), original[0]);
  });
  assert.equal(calls.length, 2);
  assert.equal(options[0].credentials, 'omit'); assert.equal(options[0].redirect, 'error');
  assert.equal(options[1].cache, 'reload');
  assert(progress.some(value => value.phase === 'retrying' && value.attempt === 1));
  assert(progress.some(value => value.attempt === 2 && value.receivedBytes === 0));
  assert.equal(progress.at(-1).phase, 'complete');
  assert.equal(progress.at(-1).receivedBytes, original[0].length);
});

test('transient HTTP and fetch errors retry, terminal HTTP does not', async () => {
  for (const status of [408, 429, 502, 503, 504, 'network']) {
    let calls = 0;
    await usingFetch(async url => {
      if (++calls === 1) {
        if (status === 'network') throw new TypeError('Failed to fetch');
        return response('Temporary outage', url, {status});
      }
      return response(original[0], url);
    }, async () => assert.deepEqual(await resourceBytes(new URL(assets[0].path, base), assets[0], fast), original[0]));
    assert.equal(calls, 2, String(status));
  }
  let calls = 0;
  await usingFetch(async url => { calls++; return response('Missing', url, {status: 404}); }, async () => {
    await assert.rejects(resourceBytes(new URL(assets[0].path, base), assets[0], fast), /0.obj.*HTTP 404/);
  });
  assert.equal(calls, 1);
});

test('progress uses a valid content length only when it describes decoded transfer bytes', async () => {
  for (const [headers, total] of [
    [{'content-length': String(original[0].length)}, original[0].length],
    [{'content-length': '5', 'content-encoding': 'gzip'}, null],
    [{'content-length': 'invalid'}, null],
  ]) {
    const progress = [];
    await usingFetch(async url => response(original[0], url, {headers}), async () => {
      await resourceBytes(new URL('geometry.bin.gz', base), undefined,
        {...fast, maxBytes: 100, onProgress: value => progress.push(value)});
    });
    assert.equal(progress.at(-1).totalBytes, total);
  }
});

test('idle timeout covers headers and a stalled body; attempts remain bounded', async () => {
  for (const stage of ['headers', 'body']) {
    let calls = 0, cancelled = 0; const signals = [];
    const started = performance.now();
    await usingFetch(async (url, init) => {
      calls++; signals.push(init.signal);
      if (stage === 'headers') return new Promise(() => {});
      return response(new ReadableStream({cancel() { cancelled++; }}), url);
    }, async () => {
      await assert.rejects(resourceBytes(new URL('waiting.bin', base), undefined,
        {...fast, idleTimeoutMs: 10}), /waiting.bin.*attempt 3\/3.*stalled/);
    });
    assert.equal(calls, 3); assert(signals.every(signal => signal.aborted));
    if (stage === 'body') assert.equal(cancelled, 3);
    assert(performance.now() - started < 1000);
  }
});

test('total deadline stops a trickling response even while the idle timer resets', async () => {
  let calls = 0, cancelled = 0, interval;
  const started = performance.now();
  await usingFetch(async url => {
    calls++;
    return response(new ReadableStream({start(controller) {
      interval = setInterval(() => controller.enqueue(new Uint8Array([1])), 3);
    }, cancel() { cancelled++; clearInterval(interval); }}), url);
  }, async () => {
    await assert.rejects(resourceBytes(new URL('trickle.bin', base), undefined,
      {...fast, timeoutMs: 35, idleTimeoutMs: 20}), /deadline exceeded/);
  });
  assert.equal(calls, 1); assert.equal(cancelled, 1); assert(performance.now() - started < 1000);
});

test('size, hash and manifest violations never retry or relax byte bounds', async () => {
  for (const [bytes, init, expected, message] of [
    [original[0].subarray(0, -1), {}, assets[0], /size mismatch/],
    [new Uint8Array(original[0].length), {}, assets[0], /checksum mismatch/],
    [new Uint8Array(original[0].length + 1), {}, assets[0], /too large/],
    [original[0], {headers: {'content-length': String(MAX_RESOURCE_BYTES + 1)}}, assets[0], /too large/],
  ]) {
    let calls = 0;
    await usingFetch(async url => { calls++; return response(bytes, url, init); }, async () => {
      await assert.rejects(resourceBytes(new URL('bounded.bin', base), expected, fast), message);
    });
    assert.equal(calls, 1);
  }
  let calls = 0;
  await usingFetch(async () => { calls++; throw new Error('must not fetch'); }, async () => {
    for (const options of [{maxBytes: MAX_RESOURCE_BYTES + 1}, {maxAttempts: 4}, {timeoutMs: 0}, {idleTimeoutMs: 0}]) {
      await assert.rejects(resourceBytes(new URL('invalid.bin', base), undefined, options), /transport option/);
    }
    await assert.rejects(resourceBytes(new URL('invalid.bin', base), {...assets[0], bytes: -1}), /descriptor/);
  });
  assert.equal(calls, 0);
});

test('oversized body is cancelled and limit errors are not retried', async () => {
  let calls = 0, cancelled = false;
  await usingFetch(async url => {
    calls++;
    return response(new ReadableStream({start(controller) { controller.enqueue(new Uint8Array(20)); },
      cancel() { cancelled = true; }}), url);
  }, async () => {
    await assert.rejects(resourceBytes(new URL('large.bin', base), undefined, {...fast, maxBytes: 10}), /too large/);
  });
  assert.equal(calls, 1); assert.equal(cancelled, true);
});

test('only the exact public Space may follow its observed HTTPS CDN origin', async () => {
  for (const [source, final, permitted] of [
    ['https://example.org/file', 'https://example.org/file', true],
    ['https://example.org/file', 'https://us.aws.cdn.hf.co/file', false],
    ['https://dvidia-robot-simulation-lab.static.hf.space/file', 'https://us.aws.cdn.hf.co/file', true],
    ['https://dvidia-robot-simulation-lab.static.hf.space/file', 'https://other.static.hf.space/file', false],
    ['https://dvidia-robot-simulation-lab.static.hf.space/file', 'http://us.aws.cdn.hf.co/file', false],
    ['https://dvidia-robot-simulation-lab.static.hf.space.evil.example/file', 'https://us.aws.cdn.hf.co/file', false],
  ]) {
    let calls = 0;
    await usingFetch(async (url, options) => {
      calls++;
      assert.equal(options.redirect, new URL(source).host === 'dvidia-robot-simulation-lab.static.hf.space' ? 'follow' : 'error');
      return response(original[0], final);
    }, async () => {
      if (permitted) assert.deepEqual(await resourceBytes(source, assets[0], fast), original[0]);
      else await assert.rejects(resourceBytes(source, assets[0], fast), /permitted workstation host/);
    });
    assert.equal(calls, 1);
  }
});

const mirrorAssets = [
  {name: 'geometry.bin.gz', bytes: 3574720, sha256: '51365a5a661d8062e28c66abc6bdb843ee410811f84b49d62c5878c265b3631f'},
  {name: 'native-assets.bin.gz', bytes: 7318481, sha256: '9c960598f9b2c0c7459a8bdfc8a544513b4c6ffaff2c8787fa49f728774eb9e7'},
  {name: 'mujoco.wasm.gz', bytes: 2543112, sha256: 'ad01772e00939bf6300a953b396d38aa9ec03c05fcda46c3c8d940d56f772b3b'},
];
const spaceAsset = item => 'https://dvidia-robot-simulation-lab.static.hf.space/workstation/v01/' + item.name;
const mirrorAsset = item => 'https://research.dvidia.org/runtime/workstation/' + item.sha256 + '/' + item.name;

test('three exact approved hosted assets use their content-addressed mirror first', async () => {
  for (const item of mirrorAssets) {
    const calls = [];
    await usingFetch(async (url, options) => {
      calls.push(String(url));
      assert.equal(options.credentials, 'omit'); assert.equal(options.referrerPolicy, 'no-referrer');
      assert.equal(options.redirect, 'manual');
      return response('Denied', url, {status: 403});
    }, async () => await assert.rejects(resourceBytes(spaceAsset(item), item, fast), /HTTP 403/));
    assert.deepEqual(calls, [mirrorAsset(item)]);
  }
});

test('mirror routing requires exact origin, path, bytes and digest; other exports stay local', async () => {
  const item = mirrorAssets[0];
  for (const [url, expected] of [
    [spaceAsset(item), undefined],
    [spaceAsset(item), {...item, bytes: item.bytes + 1}],
    [spaceAsset(item), {...item, sha256: '0'.repeat(64)}],
    [spaceAsset(item) + '?revision=future', item],
    [spaceAsset(item).replace('/v01/', '/v02/'), item],
    [spaceAsset(item).replace('https:', 'http:'), item],
    [spaceAsset(item).replace('.static.hf.space', '.static.hf.space.evil.example'), item],
    [spaceAsset(item).replace('dvidia-robot-simulation-lab.static.hf.space', 'research.dvidia.org'), item],
    ['http://127.0.0.1:8295/' + item.name, item],
  ]) {
    const calls = [];
    await usingFetch(async requested => {
      calls.push(String(requested)); return response('Missing', requested, {status: 404});
    }, async () => await assert.rejects(resourceBytes(url, expected, fast), /HTTP 404/));
    assert.deepEqual(calls, [url]);
  }
});

test('mirror network or availability errors fall back to the original within the same attempt budget', async () => {
  const item = mirrorAssets[2];
  for (const failure of ['network', 'body', 404, 410, 503]) {
    const calls = [];
    await usingFetch(async url => {
      calls.push(String(url));
      if (calls.length === 1) {
        if (failure === 'network') throw new TypeError('Failed to fetch');
        if (failure === 'body') return response(new ReadableStream({start(controller) {
          controller.error(new TypeError('network error'));
        }}), url);
        return response('Unavailable', url, {status: failure});
      }
      // The origin's own permanent failure must remain terminal after fallback.
      return response('Missing', url, {status: 404});
    }, async () => await assert.rejects(resourceBytes(spaceAsset(item), item, fast), /attempt 2\/3.*HTTP 404/));
    assert.deepEqual(calls, [mirrorAsset(item), spaceAsset(item)], String(failure));
  }
  const calls = [];
  await usingFetch(async url => { calls.push(String(url)); throw new TypeError('Failed to fetch'); },
    async () => await assert.rejects(resourceBytes(spaceAsset(item), item, fast), /attempt 3\/3/));
  assert.deepEqual(calls, [mirrorAsset(item), spaceAsset(item), spaceAsset(item)]);
});

test('mirror integrity, size and redirect failures never fall back or relax verification', async () => {
  const item = mirrorAssets[2];
  for (const [makeResponse, message] of [
    [url => response(new Uint8Array(item.bytes), url), /checksum mismatch/],
    [url => response(new Uint8Array(1), url), /size mismatch/],
    [url => response(new Uint8Array(item.bytes + 1), url), /too large/],
    [url => response('Redirect', url, {status: 302}), /may not redirect/],
    [url => { const value = response(null, url); Object.defineProperty(value, 'type', {value: 'opaqueredirect'}); return value; }, /may not redirect/],
    [() => response(null, 'https://untrusted.example/geometry.bin.gz'), /permitted workstation host/],
    [url => response(null, String(url) + '?changed=1'), /permitted workstation host/],
  ]) {
    const calls = [];
    await usingFetch(async url => { calls.push(String(url)); return makeResponse(url); },
      async () => await assert.rejects(resourceBytes(spaceAsset(item), item, fast), message));
    assert.deepEqual(calls, [mirrorAsset(item)]);
  }
});

const wasm = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]);
const wasmManifest = () => {
  const packed = gzipSync(wasm);
  return {packed, manifest: {native_wasm: {path: 'mujoco.wasm', bytes: wasm.length, sha256: hash(wasm)},
    wasm_bundle: {path: 'mujoco.wasm.gz', compression: 'gzip', bytes: packed.length,
      sha256: hash(packed), uncompressed_bytes: wasm.length}}};
};

test('compressed WASM restores exact native bytes while legacy exports stay readable', async () => {
  const {packed, manifest} = wasmManifest(); const calls = [];
  await usingFetch(async url => {
    calls.push(url.pathname);
    return response(url.pathname.endsWith('.gz') ? packed : wasm, url);
  }, async () => {
    assert.deepEqual(await loadWasmBytes(manifest, base, fast), wasm);
    assert.deepEqual(await loadWasmBytes({native_wasm: manifest.native_wasm}, base, fast), wasm);
    assert.deepEqual(await loadWasmBytes({}, base, fast), wasm);
  });
  assert.deepEqual(calls, ['/workstation/mujoco.wasm.gz', '/workstation/mujoco.wasm', '/workstation/mujoco.wasm']);
});

test('WASM checks compressed and decoded integrity, truncation, expansion and descriptors', async () => {
  const {packed, manifest} = wasmManifest();
  const cases = [
    [packed, {...manifest, wasm_bundle: {...manifest.wasm_bundle, sha256: '0'.repeat(64)}}, /checksum/],
    [packed, {...manifest, native_wasm: {...manifest.native_wasm, sha256: '0'.repeat(64)}}, /checksum/],
    [packed.subarray(0, -4), manifest, /size mismatch/],
  ];
  for (const decoded of [wasm.subarray(0, -1), new Uint8Array(10000)]) {
    const changed = gzipSync(decoded);
    cases.push([changed, {...manifest, wasm_bundle: {...manifest.wasm_bundle, bytes: changed.length, sha256: hash(changed)}},
      decoded.length > wasm.length ? /too large/ : /size mismatch/]);
  }
  const truncated = packed.subarray(0, -4);
  cases.push([truncated, {...manifest, wasm_bundle: {...manifest.wasm_bundle, bytes: truncated.length, sha256: hash(truncated)}}, /./]);
  for (const [bytes, input, error] of cases) {
    let calls = 0;
    await usingFetch(async url => { calls++; return response(bytes, url); }, async () => {
      await assert.rejects(loadWasmBytes(input, base, fast), error);
    });
    assert.equal(calls, 1);
  }
  let calls = 0;
  await usingFetch(async () => { calls++; throw new Error('must not fetch'); }, async () => {
    for (const input of [{wasm_bundle: manifest.wasm_bundle},
      {...manifest, wasm_bundle: {...manifest.wasm_bundle, uncompressed_bytes: 123}},
      {...manifest, native_wasm: {...manifest.native_wasm, path: 'other.wasm'}},
      {...manifest, wasm_bundle: {...manifest.wasm_bundle, path: 'other.gz'}}]) {
      await assert.rejects(loadWasmBytes(input, base, fast), /descriptor/);
    }
  });
  assert.equal(calls, 0);
});
