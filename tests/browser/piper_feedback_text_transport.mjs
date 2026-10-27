import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {webcrypto} from 'node:crypto';
import vm from 'node:vm';
import test from 'node:test';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const source=execFileSync('python3',['-c',"from pathlib import Path;import sys;import derive_piper_feedback_geometry_chunks as g;import derive_piper_feedback_transport_diagnostics as d;import derive_piper_feedback_text_transport as t;sys.stdout.buffer.write(t.patched_source(d.patched_source(g.patched_source(Path('src/dvidia_training/piper_feedback_viewer.js').read_bytes()))))"],
  {cwd:root,env:{...process.env,PYTHONPATH:path.join(root,'publishing/huggingface')},encoding:'utf8'});
const transport=source.slice(source.indexOf('const transportRequests'),source.indexOf('function validate('));
const raw=Buffer.from('original raw pose bytes');
const hash=async value=>Buffer.from(await webcrypto.subtle.digest('SHA-256',value)).toString('hex');
const rawSha=await hash(raw);
function canonical(value){return Buffer.from(JSON.stringify({bytes:value.bytes,data:value.data,encoding:value.encoding,schema:value.schema,sha256:value.sha256})+'\n');}
const original={bytes:raw.length,data:raw.toString('base64'),encoding:'base64',schema:'dvidia.binary-text-envelope/v1',sha256:rawSha};
async function metadata(packed){return{href:'poses-00.bin',bytes:raw.length,sha256:rawSha,json_envelope:{href:'poses-00.bin.json',bytes:packed.length,sha256:await hash(packed)}};}
function viewer(fetch,url='https://research.dvidia.org/viewer/'){
  const nodes={loading:{},'transport-details':{}};
  const context=vm.createContext({URL,location:new URL(url),fetch,MAX_BYTES:64*1024*1024,AbortController,Blob,TextDecoder,Uint8Array,
    crypto:webcrypto,DecompressionStream,setTimeout,clearTimeout,atob,btoa,$:id=>nodes[id]});
  vm.runInContext(transport,context);return context;
}
function response(url,bytes){return{url:url.href,ok:true,status:200,body:new Blob([bytes]).stream()};}
test('hosted pose bytes prefer bounded JSON and retain both envelope and original raw digests',async()=>{
  const packed=canonical(original),meta=await metadata(packed),calls=[];
  const context=viewer(async(url,options)=>{calls.push({url:url.href,options});return response(url,packed);});
  assert.deepEqual(Buffer.from(await context.binary(meta,meta.href)),raw);
  assert.deepEqual(calls.map(call=>call.url),['https://research.dvidia.org/viewer/poses-00.bin.json']);
  assert.equal(calls[0].options.credentials,'same-origin');
});
test('the exact HF viewer serves text relatively rather than routing it through binary Hub downloads',async()=>{
  const packed=canonical(original),meta=await metadata(packed),calls=[];
  const context=viewer(async url=>{calls.push(url.href);return response(url,packed);},'https://dvidia-robot-simulation-lab.static.hf.space/learned-placement/v02/viewer/index.html');
  assert.deepEqual(Buffer.from(await context.binary(meta,meta.href)),raw);
  assert.equal(calls[0],'https://dvidia-robot-simulation-lab.static.hf.space/learned-placement/v02/viewer/poses-00.bin.json');
});
test('local and unrelated hosts retain the original verified binary path',async()=>{
  const meta=await metadata(canonical(original));
  for(const url of ['http://127.0.0.1:8305/','https://unrelated.example/viewer/']){
    const calls=[],context=viewer(async target=>{calls.push(target.href);return response(target,raw);},url);
    assert.deepEqual(Buffer.from(await context.binary(meta,meta.href)),raw);
    assert.deepEqual(calls,[new URL('poses-00.bin',url).href]);
  }
});
test('network-unavailable JSON may use the original raw fallback while all bytes remain verified',async()=>{
  const meta=await metadata(canonical(original)),calls=[];
  const context=viewer(async url=>{calls.push(url.href);if(url.pathname.endsWith('.json'))throw new TypeError('network');return response(url,raw);});
  assert.deepEqual(Buffer.from(await context.binary(meta,meta.href)),raw);assert.equal(calls.length,2);
});
test('malformed envelopes fail closed even when their changed JSON outer hash is correctly bound',async()=>{
  const changes=[value=>value.encoding='hex',value=>value.bytes++,value=>value.sha256='0'.repeat(64),
    value=>value.data='A'.repeat(value.data.length),value=>value.data=value.data.slice(0,-1)+' ',
    value=>value.extra='executable fields forbidden'];
  for(const change of changes){
    const value={...original};change(value);
    const packed=Object.hasOwn(value,'extra')?Buffer.from(JSON.stringify(value)+'\n'):canonical(value),meta=await metadata(packed),calls=[];
    const context=viewer(async url=>{calls.push(url.href);return response(url,packed);});
    await assert.rejects(context.binary(meta,meta.href));assert.equal(calls.length,1);
  }
});
test('duplicate JSON fields and noncanonical base64 unused bits are rejected',async()=>{
  const duplicate=Buffer.from(canonical(original).toString().replace('"bytes":','"bytes":999,"bytes":'));
  let context=viewer(async url=>response(url,duplicate));await assert.rejects(context.binary(await metadata(duplicate),'poses-00.bin'),/canonical finite JSON/);
  const one=Buffer.from([0]),value={bytes:1,data:'AB==',encoding:'base64',schema:original.schema,sha256:await hash(one)},packed=canonical(value);
  const meta={href:'poses-00.bin',bytes:1,sha256:value.sha256,json_envelope:{href:'poses-00.bin.json',bytes:packed.length,sha256:await hash(packed)}};
  context=viewer(async url=>response(url,packed));await assert.rejects(context.binary(meta,meta.href),/canonical base64 bits/);
});
test('envelope byte bounds, digest and decoded raw hash independently reject corruption',async()=>{
  const packed=canonical(original),meta=await metadata(packed),context=viewer(async url=>response(url,packed));
  await assert.rejects(context.envelopedBinary({...meta,json_envelope:{...meta.json_envelope,bytes:1048577}},meta.href),/size or digest/);
  await assert.rejects(context.envelopedBinary({...meta,json_envelope:{...meta.json_envelope,href:'../poses-00.bin.json'}},meta.href),/path or fields/);
  await assert.rejects(context.envelopedBinary({...meta,json_envelope:{...meta.json_envelope,sha256:'0'.repeat(64)}},meta.href),/digest or JSON/);
  const bad={...original,data:Buffer.alloc(raw.length,12).toString('base64')},badPacked=canonical(bad),badMeta=await metadata(badPacked);
  const badContext=viewer(async url=>response(url,badPacked));await assert.rejects(badContext.binary(badMeta,badMeta.href),/Decoded binary SHA256/);
});
test('parent cancellation aborts pending text transfer without starting binary fallback',async()=>{
  const meta=await metadata(canonical(original)),calls=[];
  const context=viewer(async(url,options)=>{calls.push(options.signal);return new Promise(()=>{});});
  const controller=new AbortController(),promise=context.binary(meta,meta.href,controller.signal);
  controller.abort(new Error('parent cancelled'));
  await assert.rejects(promise,/parent cancelled/);assert.equal(calls.length,1);assert.ok(calls[0].aborted);
});
