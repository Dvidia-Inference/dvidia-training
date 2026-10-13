/* Verified delivery of the exact native mesh bytes. MIT. */
export const MAX_RESOURCE_BYTES = 64 * 1024 * 1024;
export const MAX_TOTAL_BYTES = 192 * 1024 * 1024;

export function localResource(path, base) {
  if (typeof path !== 'string' || path.length > 1024 || path.startsWith('/')
      || path.split('/').some(part => part === '..') || path.includes('\\')) throw new Error('Invalid local asset path');
  const url = new URL(path, base);
  if (url.origin !== base.origin || !['http:', 'https:'].includes(url.protocol)) {
    throw new Error('Workstation resources must be same-origin HTTP assets');
  }
  return url;
}

function descriptor(expected, limit = MAX_RESOURCE_BYTES) {
  if (!expected || !Number.isSafeInteger(expected.bytes) || expected.bytes < 0 || expected.bytes > limit
      || typeof expected.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(expected.sha256)) {
    throw new Error('Asset size or digest descriptor mismatch');
  }
}

async function verify(bytes, expected) {
  descriptor(expected);
  if (bytes.length !== expected.bytes) throw new Error('Asset size mismatch');
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    value => value.toString(16).padStart(2, '0')).join('');
  if (digest !== expected.sha256) throw new Error('Asset checksum mismatch');
}

export async function boundedStreamBytes(stream, limit) {
  if (!stream) throw new Error('Streaming resource responses are required');
  if (!Number.isSafeInteger(limit) || limit < 0 || limit > MAX_TOTAL_BYTES) throw new Error('Invalid resource limit');
  const reader = stream.getReader();
  const chunks = []; let received = 0;
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      received += value.byteLength;
      if (received > limit) {
        await reader.cancel();
        throw new Error('Resource is too large');
      }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(received); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return bytes;
}

export async function resourceBytes(url, expected) {
  if (expected) descriptor(expected);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Resource fetch failed (${response.status}): ${url.pathname}`);
  if (Number(response.headers.get('content-length')) > MAX_RESOURCE_BYTES) throw new Error('Resource is too large');
  const bytes = await boundedStreamBytes(response.body, expected?.bytes ?? MAX_RESOURCE_BYTES);
  if (expected) await verify(bytes, expected);
  return bytes;
}

export function nativeAssetBytes(assets) {
  if (!Array.isArray(assets) || assets.length > 256) throw new Error('Invalid asset manifest');
  let total = 0;
  const paths = new Set();
  for (const asset of assets) {
    descriptor(asset);
    if (paths.has(asset.path)) throw new Error('Duplicate native asset path');
    paths.add(asset.path);
    total += asset.bytes;
    if (total > MAX_TOTAL_BYTES) throw new Error('Scene assets exceed worker memory budget');
  }
  return total;
}

export function validateBundle(bundle, assets) {
  descriptor(bundle);
  const total = nativeAssetBytes(assets);
  if (bundle.compression !== 'gzip' || bundle.path !== 'native-assets.bin.gz'
      || !Number.isSafeInteger(bundle.uncompressed_bytes) || bundle.uncompressed_bytes !== total) {
    throw new Error('Invalid native asset bundle descriptor');
  }
  return total;
}

export async function unpackNativeAssets(packed, bundle, assets) {
  const total = validateBundle(bundle, assets);
  await verify(packed, bundle);
  // Bound decoded output independently of the compressed download to reject
  // truncation, trailing members and compression bombs before model loading.
  const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'));
  const unpacked = await boundedStreamBytes(stream, total);
  if (unpacked.byteLength !== total) throw new Error('Native asset bundle is truncated');
  const result = []; let offset = 0;
  for (const asset of assets) {
    const bytes = unpacked.subarray(offset, offset + asset.bytes);
    await verify(bytes, asset);
    result.push({path: asset.path, bytes}); offset += asset.bytes;
  }
  return result;
}

export async function loadNativeAssets(manifest, base) {
  nativeAssetBytes(manifest.assets);
  for (const asset of manifest.assets) localResource(asset.path, base);
  if (manifest.native_asset_bundle !== undefined) {
    const bundle = manifest.native_asset_bundle;
    validateBundle(bundle, manifest.assets);
    const packed = await resourceBytes(localResource(bundle.path, base), bundle);
    return unpackNativeAssets(packed, bundle, manifest.assets);
  }
  // Older exports remain readable. Newly exported scenes use one compressed
  // request while keeping these original files for independent reproduction.
  const result = [];
  for (const asset of manifest.assets) result.push({path: asset.path,
    bytes: await resourceBytes(localResource(asset.path, base), asset)});
  return result;
}
