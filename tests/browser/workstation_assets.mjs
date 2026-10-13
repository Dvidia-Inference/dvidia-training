import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createHash} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import {unpackNativeAssets, loadNativeAssets, boundedStreamBytes, MAX_RESOURCE_BYTES} from '../../src/dvidia_training/workstation_assets.js';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const original = [new TextEncoder().encode('first exact mesh\n'), new Uint8Array([0, 2, 4, 255])];
const assets = original.map((bytes, index) => ({path: `assets/${index}.obj`, bytes: bytes.length, sha256: hash(bytes)}));
const combined = Buffer.concat(original);
const pack = bytes => {
  const packed = gzipSync(bytes);
  return {packed, bundle: {path: 'native-assets.bin.gz', compression: 'gzip', bytes: packed.length,
    sha256: hash(packed), uncompressed_bytes: combined.length}};
};

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
      return new Response(index >= 0 ? original[index] : packed);
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
