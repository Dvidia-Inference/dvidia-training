import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {webcrypto} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import vm from 'node:vm';
import test from 'node:test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const derived = execFileSync('python3', ['-c', "from pathlib import Path; import sys; from derive_piper_feedback_geometry_chunks import patched_source; sys.stdout.buffer.write(patched_source(Path('src/dvidia_training/piper_feedback_viewer.js').read_bytes()))"],
  {cwd: root, env: {...process.env, PYTHONPATH: path.join(root, 'publishing/huggingface')}, encoding: 'utf8'});
const transport = derived.slice(derived.indexOf('function assert('), derived.indexOf('function validate('));
const base = 'https://research.dvidia.org/study/viewer/';
const digest = async raw => Buffer.from(await webcrypto.subtle.digest('SHA-256', raw)).toString('hex');
const pieces = [Buffer.alloc(262144, 19), Buffer.alloc(262144, 47), Buffer.alloc(1234, 83)];
const joined = Buffer.concat(pieces);
const meta = {href: 'geometry.bin', bytes: joined.length, sha256: await digest(joined), chunks: await Promise.all(pieces.map(async (piece, index) => ({
  href: `geometry-part-${String(index).padStart(3, '0')}.bin`, bytes: piece.length, sha256: await digest(piece),
  compressed_href: `geometry-part-${String(index).padStart(3, '0')}.bin.gz`, compressed_bytes: gzipSync(piece).length, compression: 'gzip'})))};
function viewer(fetch, options={}) {
  const loading = {};
  const context = vm.createContext({URL, location: new URL(base), fetch, MAX_BYTES: 64*1024*1024,
    AbortController, Blob, TextDecoder, Uint8Array, crypto: webcrypto, DecompressionStream,
    setTimeout, clearTimeout, $: () => loading, ...options});
  vm.runInContext(transport, context); return {context, loading};
}
function response(url, raw) { return {url:url.href, ok:true, body:new Blob([raw]).stream()}; }
function indexOf(url) { return Number(/geometry-part-([0-9]{3})/.exec(url.pathname)[1]); }

test('small transfers reconstruct and verify the exact original geometry with bounded concurrency', async () => {
  const calls=[], signals=[]; let active=0, max=0;
  const {context,loading}=viewer(async (url,options)=>{
    calls.push(url.href); signals.push(options.signal); max=Math.max(max,++active);
    await new Promise(resolve=>setTimeout(resolve,2)); active--;
    return response(url,url.pathname.endsWith('.gz')?gzipSync(pieces[indexOf(url)]):pieces[indexOf(url)]);
  });
  assert.deepEqual(Buffer.from(await context.geometryChunks(meta)),joined);
  assert.equal(calls.length,3); assert.ok(max<=3); assert.ok(loading.textContent.includes('3/3 pieces verified'));
  assert.ok(signals.every(signal=>signal.aborted));
});

test('chunk count, order, byte bounds and total coverage reject before any request', async () => {
  const {context}=viewer(()=>{throw new Error('must not fetch');});
  for(const change of [value=>value.chunks=[],value=>value.chunks[0].bytes=262145,
    value=>value.chunks[0].href='geometry-part-001.bin',value=>value.bytes++,value=>value.chunks[0].sha256='bad']) {
    const altered=JSON.parse(JSON.stringify(meta));change(altered);await assert.rejects(context.geometryChunks(altered));
  }
});

test('per-chunk success cannot bypass the original whole-geometry digest', async () => {
  const {context}=viewer(async url=>response(url,gzipSync(pieces[indexOf(url)])));
  await assert.rejects(context.geometryChunks({...meta,sha256:'0'.repeat(64)}),/SHA256/);
});

test('one corrupt chunk cancels other active transfers without replacing their failures with raw retries', async () => {
  const calls=[];
  const {context}=viewer(async (url,options)=>{
    calls.push({url:url.href,signal:options.signal});
    if(indexOf(url)!==0)return new Promise(()=>{});
    const bad=Buffer.from(pieces[0]);bad[0]^=1;
    return response(url,url.pathname.endsWith('.gz')?gzipSync(bad):bad);
  });
  await assert.rejects(context.geometryChunks(meta));
  assert.ok(calls.every(call=>call.signal.aborted));
  assert.equal(calls.filter(call=>!call.url.endsWith('.gz')).length,1);
});

test('the whole transport deadline aborts all pending headers and prevents raw fallback after cancellation', async () => {
  const calls=[];
  const {context}=viewer(async (url,options)=>{calls.push({url:url.href,signal:options.signal});return new Promise(()=>{});},
    {setTimeout:(callback,ms)=>setTimeout(callback,ms===120000?10:500)});
  await assert.rejects(context.geometryChunks(meta),/Timed out loading chunked geometry/);
  assert.equal(calls.length,3);assert.ok(calls.every(call=>call.signal.aborted));
});
