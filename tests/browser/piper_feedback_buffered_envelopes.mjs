import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {webcrypto} from 'node:crypto';
import vm from 'node:vm';
import test from 'node:test';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const source=execFileSync('python3',['-c',"from pathlib import Path;import sys;import derive_piper_feedback_geometry_chunks as g;import derive_piper_feedback_transport_diagnostics as d;import derive_piper_feedback_text_transport as t;import derive_piper_feedback_buffered_envelopes as b;sys.stdout.buffer.write(b.patched_source(t.patched_source(d.patched_source(g.patched_source(Path('src/dvidia_training/piper_feedback_viewer.js').read_bytes())))))"],
  {cwd:root,env:{...process.env,PYTHONPATH:path.join(root,'publishing/huggingface')},encoding:'utf8'});
const transport=source.slice(source.indexOf('const transportRequests'),source.indexOf('function validate('));
const raw=Buffer.from('original raw pose bytes');
const hash=async value=>Buffer.from(await webcrypto.subtle.digest('SHA-256',value)).toString('hex');
const rawSha=await hash(raw);
function canonical(value){return Buffer.from(JSON.stringify({bytes:value.bytes,data:value.data,encoding:value.encoding,schema:value.schema,sha256:value.sha256})+'\n');}
const packed=canonical({bytes:raw.length,data:raw.toString('base64'),encoding:'base64',schema:'dvidia.binary-text-envelope/v1',sha256:rawSha});
const meta={href:'poses-00.bin',bytes:raw.length,sha256:rawSha,json_envelope:{href:'poses-00.bin.json',bytes:packed.length,sha256:await hash(packed)}};
function ab(bytes){return bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);}
function viewer(fetch,overrides={}){
  const nodes={loading:{},'transport-details':{}};
  const context=vm.createContext({URL,location:new URL('https://research.dvidia.org/viewer/'),fetch,MAX_BYTES:64*1024*1024,AbortController,Blob,TextDecoder,Uint8Array,
    crypto:webcrypto,DecompressionStream,setTimeout,clearTimeout,atob,btoa,$:id=>nodes[id],...overrides});
  vm.runInContext(transport,context);return{context,nodes};
}
function response(url,bytes,{length=String(bytes.length),buffer=()=>Promise.resolve(ab(bytes))}={}){
  return{url:url.href,ok:true,status:200,headers:new Headers(length===null?{}:{'content-length':length}),
    body:{getReader(){throw new Error('Streaming reader must not be used in native envelope experiment.');}},arrayBuffer:buffer};
}
test('known hosted envelope uses native body buffering without touching the streaming reader',async()=>{
  const calls=[];const{context,nodes}=viewer(async(url,options)=>{calls.push({url:url.href,options});return response(url,packed);});
  assert.deepEqual(Buffer.from(await context.binary(meta,meta.href)),raw);
  assert.equal(calls.length,1);assert.equal(calls[0].options.redirect,'error');assert.equal(calls[0].options.credentials,'same-origin');
  assert.ok(nodes['transport-details'].textContent.includes('complete'));
});
test('missing or compressed-wire Content-Length keeps exact postread count and SHA checks',async()=>{
  for(const length of [null,'5',String(packed.length)]){
    const{context}=viewer(async url=>response(url,packed,{length}));
    assert.deepEqual(Buffer.from(await context.binary(meta,meta.href)),raw);
  }
});
test('malformed or oversized declared lengths reject before native body allocation or raw fallback',async()=>{
  for(const length of ['0','-1','1.5','NaN','9007199254740993',String(packed.length+1),'1048577']){
    let calls=0,buffered=0;
    const{context}=viewer(async url=>{calls++;return response(url,packed,{length,buffer:()=>{buffered++;return Promise.resolve(ab(packed));}});});
    await assert.rejects(context.binary(meta,meta.href),/Content-Length/);assert.equal(calls,1);assert.equal(buffered,0);
  }
});
test('completed native bytes cannot bypass expected count, envelope hash, decoded hash or schema',async()=>{
  for(const bytes of [packed.subarray(0,-1),Buffer.concat([packed,Buffer.from('x')]),Buffer.alloc(packed.length,65)]){
    let calls=0;const{context}=viewer(async url=>{calls++;return response(url,bytes,{length:null});});
    await assert.rejects(context.binary(meta,meta.href));assert.equal(calls,1);
  }
  const changed=canonical({bytes:raw.length,data:Buffer.alloc(raw.length,1).toString('base64'),encoding:'base64',schema:'dvidia.binary-text-envelope/v1',sha256:rawSha});
  const changedMeta={...meta,json_envelope:{...meta.json_envelope,bytes:changed.length,sha256:await hash(changed)}};
  const{context}=viewer(async url=>response(url,changed));await assert.rejects(context.binary(changedMeta,meta.href),/Decoded binary SHA256/);
});
test('a hanging native body is bounded by the unchanged request deadline and aborts fetch',async()=>{
  const calls=[];const{context,nodes}=viewer(async(url,options)=>{calls.push(options.signal);return response(url,packed,{buffer:()=>new Promise(()=>{})});},
    {setTimeout:(callback,ms)=>setTimeout(callback,ms===30000?10:ms)});
  await assert.rejects(context.envelopedBinary(meta,meta.href),/native body buffering, HTTP 200, 0\//);
  assert.equal(calls.length,1);assert.ok(calls[0].aborted);assert.ok(nodes['transport-details'].textContent.includes('failed after native body buffering'));
});
test('parent cancellation ends pending native body without binary fallback',async()=>{
  const calls=[];const{context}=viewer(async(url,options)=>{calls.push(options.signal);return response(url,packed,{buffer:()=>new Promise(()=>{})});});
  const controller=new AbortController(),pending=context.binary(meta,meta.href,controller.signal);
  await Promise.resolve();controller.abort(new Error('parent cancelled'));
  await assert.rejects(pending,/parent cancelled/);assert.equal(calls.length,1);assert.ok(calls[0].aborted);
});
test('same-origin and redirect validation run before native buffering',async()=>{
  let buffered=0;const{context}=viewer(async url=>({...response(url,packed,{buffer:()=>{buffered++;return Promise.resolve(ab(packed));}}),url:'https://untrusted.example/asset.json'}));
  await assert.rejects(context.envelopedBinary(meta,meta.href),/origin/);assert.equal(buffered,0);
});
