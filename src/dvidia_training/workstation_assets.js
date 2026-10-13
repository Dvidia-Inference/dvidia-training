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

const HF_SPACE_HOST = 'dvidia-robot-simulation-lab.static.hf.space';
const HF_CDN_HOST = 'us.aws.cdn.hf.co';
const RETRY_HTTP_STATUS = new Set([408, 429, 502, 503, 504]);

class ResourceDownloadError extends Error {
  constructor(message, retryable = false) {
    super(message); this.name = 'ResourceDownloadError'; this.retryable = retryable;
  }
}

function transportOptions(options, expected) {
  const settings = {maxBytes: expected?.bytes ?? MAX_RESOURCE_BYTES, idleTimeoutMs: 30000,
    timeoutMs: 120000, maxAttempts: 3, retryDelayMs: 250, ...options};
  for (const [name, maximum, minimum] of [['maxBytes', MAX_RESOURCE_BYTES, 0],
    ['idleTimeoutMs', 120000, 1], ['timeoutMs', 120000, 1], ['maxAttempts', 3, 1], ['retryDelayMs', 2000, 0]]) {
    if (!Number.isSafeInteger(settings[name]) || settings[name] < minimum || settings[name] > maximum) {
      throw new Error(`Invalid resource transport option: ${name}`);
    }
  }
  if (expected && settings.maxBytes < expected.bytes) throw new Error('Resource byte limit is smaller than its descriptor');
  if (settings.onProgress !== undefined && typeof settings.onProgress !== 'function') throw new Error('Invalid resource progress callback');
  return settings;
}

// Fetch resolves at headers. Cover every body read too so a stalled CDN
// response cannot leave startup hanging indefinitely.
async function withAbort(promise, signal) {
  if (signal.aborted) throw signal.reason;
  let listener;
  const interrupted = new Promise((resolve, reject) => {
    listener = () => reject(signal.reason);
    signal.addEventListener('abort', listener, {once: true});
  });
  try { return await Promise.race([promise, interrupted]); }
  finally { signal.removeEventListener('abort', listener); }
}

async function downloadBody(stream, limit, signal, onChunk) {
  if (!stream) throw new ResourceDownloadError('The server returned an empty resource');
  const reader = stream.getReader(); const chunks = []; let received = 0; let finished = false;
  try {
    for (;;) {
      let result;
      try { result = await withAbort(reader.read(), signal); }
      catch (error) {
        if (signal.aborted) throw signal.reason;
        throw new ResourceDownloadError(`Body download interrupted: ${error.message || 'network error'}`, true);
      }
      if (result.done) { finished = true; break; }
      received += result.value.byteLength;
      if (received > limit) throw new ResourceDownloadError('Resource is too large');
      chunks.push(result.value);
      if (result.value.byteLength) onChunk(received);
    }
  } finally {
    // Do not wait for a broken underlying source to acknowledge cancellation.
    if (!finished) void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
  const bytes = new Uint8Array(received); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return bytes;
}

export async function resourceBytes(input, expected, options = {}) {
  if (expected) descriptor(expected);
  const settings = transportOptions(options, expected);
  const url = new URL(input);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('Invalid resource URL');
  const hosted = url.protocol === 'https:' && url.host === HF_SPACE_HOST;
  const deadline = performance.now() + settings.timeoutMs;
  const label = url.pathname.split('/').pop() || 'resource';
  let totalBytes = expected?.bytes ?? null;
  const progress = (phase, attempt, receivedBytes = 0) => settings.onProgress?.({
    path: url.pathname, phase, attempt, receivedBytes, totalBytes});
  const expired = () => new ResourceDownloadError('Total download deadline exceeded');
  for (let attempt = 1; attempt <= settings.maxAttempts; attempt += 1) {
    const remaining = deadline - performance.now();
    if (remaining <= 0) throw new Error(`Could not download ${label}: total download deadline exceeded`);
    const controller = new AbortController();
    const totalTimer = setTimeout(() => controller.abort(expired()), remaining);
    let idleTimer;
    const active = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => controller.abort(new ResourceDownloadError('Download stalled without receiving data', true)),
        settings.idleTimeoutMs);
    };
    try {
      totalBytes = expected?.bytes ?? null;
      progress('downloading', attempt); active();
      let response;
      try {
        response = await withAbort(fetch(url, {credentials: 'omit', referrerPolicy: 'no-referrer',
          redirect: hosted ? 'follow' : 'error', signal: controller.signal,
          ...(attempt > 1 ? {cache: 'reload'} : {})}), controller.signal);
      } catch (error) {
        if (controller.signal.aborted) throw controller.signal.reason;
        throw new ResourceDownloadError(`Network request failed: ${error.message || 'network error'}`, true);
      }
      active();
      const final = new URL(response.url);
      if (final.origin !== url.origin && !(hosted && final.protocol === 'https:' && final.host === HF_CDN_HOST)) {
        void response.body?.cancel().catch(() => {});
        throw new ResourceDownloadError('Resource redirected outside the permitted workstation host');
      }
      if (!response.ok) {
        void response.body?.cancel().catch(() => {});
        throw new ResourceDownloadError(`HTTP ${response.status}`, RETRY_HTTP_STATUS.has(response.status));
      }
      if (Number(response.headers.get('content-length')) > settings.maxBytes) {
        void response.body?.cancel().catch(() => {});
        throw new ResourceDownloadError('Resource is too large');
      }
      const lengthHeader = response.headers.get('content-length');
      const encoding = response.headers.get('content-encoding');
      if (!expected && (!encoding || encoding === 'identity') && /^\d+$/.test(lengthHeader ?? '')
          && Number.isSafeInteger(Number(lengthHeader)) && Number(lengthHeader) <= settings.maxBytes) {
        totalBytes = Number(lengthHeader); progress('downloading', attempt);
      }
      const bytes = await downloadBody(response.body, expected?.bytes ?? settings.maxBytes, controller.signal, received => {
        active(); progress('downloading', attempt, received);
      });
      if (expected) await verify(bytes, expected);
      progress('complete', attempt, bytes.byteLength);
      return bytes;
    } catch (error) {
      const canRetry = error instanceof ResourceDownloadError && error.retryable
        && attempt < settings.maxAttempts && performance.now() < deadline;
      if (!canRetry) throw new Error(`Could not download ${label} (attempt ${attempt}/${settings.maxAttempts}): ${error.message}`);
      progress('retrying', attempt);
      const delay = Math.min(settings.retryDelayMs * attempt, Math.max(0, deadline - performance.now()));
      if (delay) await new Promise(resolve => setTimeout(resolve, delay));
    } finally { clearTimeout(totalTimer); clearTimeout(idleTimer); }
  }
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

async function decodeGzip(packed, limit, path) {
  try {
    const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'));
    return await boundedStreamBytes(stream, limit);
  } catch (error) { throw new Error(`Could not decode ${path}: ${error.message}`); }
}

export async function unpackNativeAssets(packed, bundle, assets) {
  const total = validateBundle(bundle, assets);
  await verify(packed, bundle);
  // Bound decoded output independently of the compressed download to reject
  // truncation, trailing members and compression bombs before model loading.
  const unpacked = await decodeGzip(packed, total, bundle.path);
  if (unpacked.byteLength !== total) throw new Error('Native asset bundle is truncated');
  const result = []; let offset = 0;
  for (const asset of assets) {
    const bytes = unpacked.subarray(offset, offset + asset.bytes);
    await verify(bytes, asset);
    result.push({path: asset.path, bytes}); offset += asset.bytes;
  }
  return result;
}

export async function loadNativeAssets(manifest, base, options = {}) {
  nativeAssetBytes(manifest.assets);
  for (const asset of manifest.assets) localResource(asset.path, base);
  if (manifest.native_asset_bundle !== undefined) {
    const bundle = manifest.native_asset_bundle;
    validateBundle(bundle, manifest.assets);
    const packed = await resourceBytes(localResource(bundle.path, base), bundle, options);
    return unpackNativeAssets(packed, bundle, manifest.assets);
  }
  // Older exports remain readable. Newly exported scenes use one compressed
  // request while keeping these original files for independent reproduction.
  const result = [];
  for (const asset of manifest.assets) result.push({path: asset.path,
    bytes: await resourceBytes(localResource(asset.path, base), asset, options)});
  return result;
}

export async function loadWasmBytes(manifest, base, options = {}) {
  const native = manifest.native_wasm;
  if (native !== undefined) {
    descriptor(native);
    if (native.path !== 'mujoco.wasm') throw new Error('Invalid native WASM descriptor');
  }
  const bundle = manifest.wasm_bundle;
  if (bundle === undefined) return resourceBytes(localResource('mujoco.wasm', base), native, options);
  descriptor(bundle);
  if (!native || bundle.path !== 'mujoco.wasm.gz' || bundle.compression !== 'gzip'
      || !Number.isSafeInteger(bundle.uncompressed_bytes) || bundle.uncompressed_bytes !== native.bytes) {
    throw new Error('Invalid WASM bundle descriptor');
  }
  const packed = await resourceBytes(localResource(bundle.path, base), bundle, options);
  const unpacked = await decodeGzip(packed, native.bytes, bundle.path);
  try { await verify(unpacked, native); }
  catch (error) { throw new Error(`Decoded ${native.path} from ${bundle.path}: ${error.message}`); }
  return unpacked;
}
