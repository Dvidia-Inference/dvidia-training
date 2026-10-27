import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {webcrypto} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import vm from 'node:vm';
import test from 'node:test';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const source=execFileSync('python3',['-c',"from pathlib import Path;import sys;import derive_piper_feedback_geometry_chunks as g;import derive_piper_feedback_transport_diagnostics as d;sys.stdout.buffer.write(d.patched_source(g.patched_source(Path('src/dvidia_training/piper_feedback_viewer.js').read_bytes())))"],
  {cwd:root,env:{...process.env,PYTHONPATH:path.join(root,'publishing/huggingface')},encoding:'utf8'});
const transport=source.slice(source.indexOf('const transportRequests'),source.indexOf('function validate('));
const raw=new TextEncoder().encode('a known complete fixture');
const meta={href:'geometry-part-006.bin',bytes:raw.length,sha256:Buffer.from(await webcrypto.subtle.digest('SHA-256',raw)).toString('hex')};
function viewer(fetch,overrides={}) {
  const nodes={'transport-details':{},loading:{}};
  const context=vm.createContext({URL,location:new URL('https://research.dvidia.org/viewer/'),fetch,
    MAX_BYTES:64*1024*1024,AbortController,Blob,TextDecoder,Uint8Array,crypto:webcrypto,DecompressionStream,
    setTimeout:(cb,ms)=>setTimeout(cb,ms===8000?10:25),clearTimeout,$:id=>nodes[id],...overrides});
  vm.runInContext(transport,context);return{context,nodes};
}
test('headers stall reports the local asset, pending HTTP state and exact zero byte count',async()=>{
  const calls=[];const{context,nodes}=viewer(async(url,options)=>{calls.push(options.signal);return new Promise(()=>{});});
  await assert.rejects(context.binary(meta,meta.href),error=>error.message.includes('awaiting headers, HTTP pending, 0/'+raw.length+' bytes'));
  assert.ok(nodes['transport-details'].textContent.includes('failed after awaiting headers'));
  assert.equal(calls.length,1);assert.ok(calls[0].aborted);
});
test('partial body stall reports HTTP200 and received/expected bytes without making extra requests',async()=>{
  let calls=0;const{context,nodes}=viewer(async url=>{calls++;return{url:url.href,ok:true,status:200,
    body:new ReadableStream({start(controller){controller.enqueue(raw.slice(0,5));}})};});
  await assert.rejects(context.binary(meta,meta.href),error=>error.message.includes('reading body, HTTP 200, 5/'+raw.length+' bytes'));
  assert.ok(nodes['transport-details'].textContent.includes('failed after reading body'));
  assert.equal(calls,1);
});
test('decompression stall remains visible when the bounded raw fallback also stalls',async()=>{
  class Hung{constructor(){return new TransformStream({transform(){return new Promise(()=>{});}});}}
  const packed=gzipSync(raw),compressed={...meta,compressed_href:meta.href+'.gz',compressed_bytes:packed.length,compression:'gzip'};
  const{context,nodes}=viewer(async url=>url.pathname.endsWith('.gz')?{url:url.href,ok:true,status:200,body:new Blob([packed]).stream()}:new Promise(()=>{}),{DecompressionStream:Hung});
  await assert.rejects(context.binary(compressed,compressed.href),/awaiting headers/);
  assert.ok(nodes['transport-details'].textContent.includes('failed after decompressing'));
  assert.ok(nodes['transport-details'].textContent.includes('HTTP 200 · 0/'+raw.length+' bytes'));
});
test('digest stall is distinguished from transfer and decompression',async()=>{
  const{context}=viewer(async url=>({url:url.href,ok:true,status:200,body:new Blob([raw]).stream()}),
    {crypto:{subtle:{digest:()=>new Promise(()=>{})}}});
  await assert.rejects(context.binary(meta,meta.href),error=>error.message.includes('verifying, HTTP 200, '+raw.length+'/'+raw.length+' bytes'));
});
test('completed metadata and binary requests are shown locally without exposing response URLs or starting telemetry',async()=>{
  const calls=[];const{context,nodes}=viewer(async url=>{calls.push(url.href);return{url:url.href+'?signed=not-displayed',ok:true,status:200,
    body:new Blob([url.pathname.endsWith('.json')?'{}':raw]).stream()};});
  await context.json('scene.json');await context.binary(meta,meta.href);
  assert.ok(nodes['transport-details'].textContent.includes('scene.json · complete'));
  assert.ok(nodes['transport-details'].textContent.includes(meta.href+' · complete'));
  assert.ok(!nodes['transport-details'].textContent.includes('signed'));
  assert.equal(calls.length,2);
});
