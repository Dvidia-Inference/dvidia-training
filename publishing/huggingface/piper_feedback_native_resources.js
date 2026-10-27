/* Browser-managed immutable data resources; no downloaded policy or physics. */
const RESOURCE_SCHEMA = 'dvidia.piper-feedback-resource-data/v1';
const RESOURCE_CALLBACK = '__dvidiaPiperFeedbackResourcesV1';
const resourcePending = new Map();
let resourceSelection;

function resourceKeys(value, keys, label) {
  assert(value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).sort().join(',') === keys, `Invalid ${label} fields.`);
}
function resourceDescriptor(value, expected) {
  resourceKeys(value, 'bytes,compressed_bytes,compressed_sha256,compression,data,href,sha256', 'saved data resource');
  for (const key of ['href', 'bytes', 'sha256', 'compressed_bytes', 'compressed_sha256', 'compression']) {
    assert(value[key] === expected[key], `Saved data resource ${key} changed.`);
  }
  localHref(value.href);
  integer(value.bytes, 1, 8 * 1024 * 1024, 'resource raw size');
  integer(value.compressed_bytes, 1, 4 * 1024 * 1024, 'resource compressed size');
  assert(value.compression === 'gzip' && /^[0-9a-f]{64}$/.test(value.sha256) && /^[0-9a-f]{64}$/.test(value.compressed_sha256), 'Invalid resource digest or compression.');
  assert(typeof value.data === 'string' && value.data.length === 4 * Math.ceil(value.compressed_bytes / 3) && /^[A-Za-z0-9+/]*={0,2}$/.test(value.data), 'Invalid bounded resource base64.');
}
async function resourceDeadline(name, operation, outerSignal) {
  localHref(name);
  const controller = new AbortController();
  const propagate = () => controller.abort(outerSignal.reason);
  if (outerSignal?.aborted) throw outerSignal.reason;
  outerSignal?.addEventListener('abort', propagate, {once: true});
  const timer = setTimeout(() => controller.abort(new Error(`Timed out opening saved resource ${name} after 30 s.`)), 30000);
  try { return await abortable(operation(controller.signal), controller.signal); }
  finally { clearTimeout(timer); outerSignal?.removeEventListener('abort', propagate); controller.abort(); }
}
async function resourceDecode(value, expected, signal) {
  resourceDescriptor(value, expected);
  assert(typeof DecompressionStream === 'function', 'This browser needs native gzip decompression to open the saved replay.');
  let decoded;
  try { decoded = atob(value.data); } catch (error) { throw new Error('Saved resource base64 could not be decoded.'); }
  assert(decoded.length === value.compressed_bytes && btoa(decoded) === value.data, 'Saved resource base64 length or canonical bits changed.');
  const compressed = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; i++) compressed[i] = decoded.charCodeAt(i);
  await abortable(verifiedBinary(compressed.buffer, {bytes:value.compressed_bytes, sha256:value.compressed_sha256}), signal);
  const raw = await bounded(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'), {signal}), value.bytes, signal);
  await abortable(verifiedBinary(raw, value), signal);
  return raw;
}
function resourcePayload(payload, descriptor) {
  resourceKeys(payload, 'id,resources,schema', 'replay data wrapper');
  assert(payload.schema === RESOURCE_SCHEMA && payload.id === descriptor.id, 'Unexpected replay data identity.');
  assert(Array.isArray(payload.resources) && payload.resources.length === 2 && descriptor.resources.length === 2, 'Invalid replay resource count.');
  payload.resources.forEach((value, index) => resourceDescriptor(value, descriptor.resources[index]));
  return payload;
}
function resourceReceive(id, payload) {
  const pending = resourcePending.get(id);
  // A removed, cancelled script may finish late. It cannot populate another
  // case or persistent cache; only the exact currently pending id is accepted.
  if (!pending) return false;
  try { pending.accept(resourcePayload(payload, pending.descriptor)); }
  catch (error) { pending.reject(error); }
  return true;
}
function resourceScript(descriptor, signal) {
  assert(typeof descriptor.id === 'string' && /^replay-[0-9]{2}$/.test(descriptor.id), 'Unexpected replay script identity.');
  resourceKeys(descriptor.script, 'bytes,href,integrity', 'replay script');
  assert(/^replay-data-[0-9]{2}\.js$/.test(descriptor.script.href) && descriptor.script.href === `replay-data-${descriptor.id.slice(-2)}.js`, 'Unexpected replay script path.');
  integer(descriptor.script.bytes, 1, 1024 * 1024, 'replay script byte count');
  assert(/^sha384-[A-Za-z0-9+/]{64}$/.test(descriptor.script.integrity), 'Invalid replay script SHA384 integrity.');
  const url = new URL(localHref(descriptor.script.href), location.href);
  assert(url.origin === location.origin && !url.username && !url.password && ['http:', 'https:'].includes(url.protocol), 'Replay scripts require the existing HTTP origin.');
  assert(!resourcePending.has(descriptor.id), 'Replay resource is already pending.');
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise((resolve, reject) => {
    const script = document.createElement('script'); let received = false, complete = false, payload;
    const cleanup = () => { resourcePending.delete(descriptor.id); signal.removeEventListener('abort', aborted); script.onload = script.onerror = null; script.remove(); };
    const finish = (error, value) => { if (complete) return; complete = true; cleanup(); error ? reject(error) : resolve(value); };
    const aborted = () => finish(signal.reason);
    resourcePending.set(descriptor.id, {descriptor, accept(value) { assert(!received, 'Duplicate replay resource callback.'); received = true; payload = value; }, reject:error=>finish(error)});
    script.async = true; script.src = url.href; script.integrity = descriptor.script.integrity;
    script.crossOrigin = 'anonymous'; script.referrerPolicy = 'no-referrer';
    script.onload = () => received ? finish(null, payload) : finish(new Error('The expected replay data callback was not received.'));
    script.onerror = () => finish(new Error('Saved replay data script failed to load or pass SHA384 integrity.'));
    signal.addEventListener('abort', aborted, {once:true});
    try { document.head.append(script); } catch (error) { finish(error); }
  });
}
assert(!Object.prototype.hasOwnProperty.call(globalThis, RESOURCE_CALLBACK), 'Saved replay callback is already registered.');
Object.defineProperty(globalThis, RESOURCE_CALLBACK, {value:resourceReceive, writable:false, configurable:false, enumerable:false});

async function resourceInitial(href) {
  const value = BROWSER_RESOURCES.embedded[href];
  assert(value && value.href === href, 'Unexpected initial saved resource.');
  return resourceDeadline(href, signal => resourceDecode(value, value, signal));
}
async function resourceJson(href) {
  const raw = await resourceInitial(href);
  return JSON.parse(new TextDecoder('utf-8', {fatal:true}).decode(raw));
}
async function browserCase(item, outerSignal) {
  const descriptor = BROWSER_RESOURCES.cases.find(value => value.case_id === item.id && value.variant === item.variant);
  assert(descriptor && descriptor.resources[0].href === item.rows_file.href && descriptor.resources[1].href === item.pose_binary.href, 'Requested saved replay is unavailable.');
  for (const [expected, meta] of [[descriptor.resources[0], item.rows_file], [descriptor.resources[1], item.pose_binary]]) {
    assert(expected.bytes === meta.bytes && expected.sha256 === meta.sha256, 'Replay metadata differs from its embedded identity.');
  }
  return resourceDeadline(descriptor.id, async signal => {
    const payload = descriptor.id === BROWSER_RESOURCES.first.id ? resourcePayload(BROWSER_RESOURCES.first, descriptor) : await resourceScript(descriptor, signal);
    const [rows, motion] = await Promise.all(payload.resources.map((value, index) => resourceDecode(value, descriptor.resources[index], signal)));
    return {rows:JSON.parse(new TextDecoder('utf-8', {fatal:true}).decode(rows)), motion};
  }, outerSignal);
}
