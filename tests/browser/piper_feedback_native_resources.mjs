import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {createHash,webcrypto} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import vm from 'node:vm';
import test from 'node:test';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const canonical=readFileSync(path.join(root,'src/dvidia_training/piper_feedback_viewer.js'),'utf8');
const transport=canonical.slice(canonical.indexOf('function assert('),canonical.indexOf('function validate('));
const addon=readFileSync(path.join(root,'publishing/huggingface/piper_feedback_native_resources.js'),'utf8');
const sha=value=>createHash('sha256').update(value).digest('hex');
const sri=value=>'sha384-'+createHash('sha384').update(value).digest('base64');
const callback='__dvidiaPiperFeedbackResourcesV1';
function resource(href,raw){const packed=gzipSync(raw);return{href,bytes:raw.length,sha256:sha(raw),compression:'gzip',compressed_bytes:packed.length,compressed_sha256:sha(packed),data:packed.toString('base64')};}
function fixture(){
  const rows=Buffer.from('[{"t_seconds":0}]'),pose=Buffer.from('exact saved pose');
  const payloads=[0,1].map(i=>({schema:'dvidia.piper-feedback-resource-data/v1',id:`replay-0${i}`,resources:[resource(`frames-0${i}.json`,rows),resource(`poses-0${i}.bin`,pose)]}));
  const scripts=new Map();
  const cases=payloads.map((value,i)=>{
    const source=`globalThis[${JSON.stringify(callback)}](${JSON.stringify(value.id)},${JSON.stringify(value)});\n`,href=`replay-data-0${i}.js`;
    scripts.set(href,source);
    return{id:value.id,case_id:`case${i}`,variant:'feedback',resources:value.resources.map(({data,...meta})=>meta),script:i?{href,bytes:Buffer.byteLength(source),integrity:sri(source)}:null};
  });
  const manifest={embedded:{'geometry.bin':resource('geometry.bin',Buffer.from('original geometry'))},first:payloads[0],cases};
  const items=cases.map(value=>({id:value.case_id,variant:value.variant,rows_file:value.resources[0],pose_binary:value.resources[1]}));
  return{manifest,payloads,scripts,items,rows,pose};
}
function viewer(fx,behavior){
  const calls=[],removed=[];let context;
  const document={head:{append(script){calls.push(script);behavior?.(script,context,fx);}},createElement(tag){assert.equal(tag,'script');return{remove(){removed.push(this);}};}};
  context=vm.createContext({URL,location:new URL('https://research.dvidia.org/viewer/index.html'),MAX_BYTES:64*1024*1024,AbortController,Blob,TextDecoder,Uint8Array,
    crypto:webcrypto,DecompressionStream,setTimeout,clearTimeout,atob,btoa,document,fetch(){throw new Error('Playback must not fetch a body.');}});
  vm.runInContext(transport+'\nconst BROWSER_RESOURCES='+JSON.stringify(fx.manifest)+';\n'+addon,context);
  return{context,calls,removed};
}
function browserExecute(script,context,fx){
  queueMicrotask(()=>{
    const source=fx.scripts.get(new URL(script.src).pathname.split('/').at(-1));
    if(!source || sri(source)!==script.integrity){script.onerror?.();return;}
    vm.runInContext(source,context);script.onload?.();
  });
}
test('embedded geometry and first scheduled replay open with no body fetch or data script',async()=>{
  const fx=fixture(),{context,calls}=viewer(fx,browserExecute);
  assert.equal(Buffer.from(await context.resourceInitial('geometry.bin')).toString(),'original geometry');
  const loaded=await context.browserCase(fx.items[0],new AbortController().signal);
  assert.equal(JSON.stringify(loaded.rows),fx.rows.toString());assert.deepEqual(Buffer.from(loaded.motion),fx.pose);assert.equal(calls.length,0);
});
test('other cases load only on selection with fixed same-origin SRI/anonymous/no-referrer script',async()=>{
  const fx=fixture(),{context,calls,removed}=viewer(fx,browserExecute);
  assert.equal(calls.length,0);const loaded=await context.browserCase(fx.items[1],new AbortController().signal);
  assert.deepEqual(Buffer.from(loaded.motion),fx.pose);assert.equal(calls.length,1);assert.equal(removed.length,1);
  assert.equal(calls[0].src,'https://research.dvidia.org/viewer/replay-data-01.js');assert.equal(calls[0].integrity,fx.manifest.cases[1].script.integrity);
  assert.equal(calls[0].crossOrigin,'anonymous');assert.equal(calls[0].referrerPolicy,'no-referrer');
});
test('unexpected cases, paths, oversized script or malformed SRI reject before browser request',async()=>{
  let fx=fixture(),v=viewer(fx,browserExecute);
  await assert.rejects(v.context.browserCase({...fx.items[1],id:'other-case'},new AbortController().signal),/unavailable/);assert.equal(v.calls.length,0);
  for(const change of [d=>d.script.href='../replay-data-01.js',d=>d.script.href='https://other.example/replay-data-01.js',d=>d.script.bytes=1048577,d=>d.script.integrity='sha256-wrong']){
    fx=fixture();change(fx.manifest.cases[1]);v=viewer(fx,browserExecute);
    await assert.rejects(v.context.browserCase(fx.items[1],new AbortController().signal));assert.equal(v.calls.length,0);
  }
});
test('a script rejected by native SRI cannot deliver changed data',async()=>{
  const fx=fixture();fx.scripts.set('replay-data-01.js',fx.scripts.get('replay-data-01.js')+'globalThis.UNEXPECTED=true;');
  const{context}=viewer(fx,browserExecute);
  await assert.rejects(context.browserCase(fx.items[1],new AbortController().signal),/SHA384 integrity/);assert.equal(context.UNEXPECTED,undefined);
});
test('unexpected wrapper id, extra fields, wrong resource identity and oversized data fail closed',async()=>{
  const mutations=[p=>p.id='replay-00',p=>p.extra=true,p=>p.resources[1].href='poses-00.bin',p=>p.resources[0].data+='AAAA'];
  for(const mutate of mutations){
    const fx=fixture();mutate(fx.payloads[1]);
    const{context}=viewer(fx,(script,context)=>queueMicrotask(()=>{context[callback]('replay-01',fx.payloads[1]);script.onload?.();}));
    await assert.rejects(context.browserCase(fx.items[1],new AbortController().signal));
  }
});
test('validly wrapped bytes still require compressed and original raw SHA plus canonical base64',async()=>{
  for(const change of [r=>r.data=Buffer.alloc(r.compressed_bytes,65).toString('base64'),r=>r.sha256='0'.repeat(64),r=>r.data=r.data.slice(0,-1)+'!']){
    const fx=fixture();change(fx.manifest.first.resources[1]);const{context}=viewer(fx,browserExecute);
    await assert.rejects(context.browserCase(fx.items[0],new AbortController().signal));
  }
});
test('bounded local gzip expansion and original raw digest cannot be bypassed by valid compressed bytes',async()=>{
  const fx=fixture(),{context}=viewer(fx,browserExecute),signal=new AbortController().signal;
  const changed=resource('poses-01.bin',Buffer.from('other saved pose'));changed.sha256=sha(fx.pose);
  await assert.rejects(context.resourceDecode(changed,changed,signal),/SHA256/);
  const expansion=resource('poses-01.bin',fx.pose);expansion.bytes=1;
  await assert.rejects(context.resourceDecode(expansion,expansion,signal),/byte bound/);
  const oversized=resource('poses-01.bin',fx.pose);oversized.bytes=8*1024*1024+1;
  await assert.rejects(context.resourceDecode(oversized,oversized,signal),/raw size/);
});
test('parent cancellation removes pending script, rejects, and ignores late payload without caching it',async()=>{
  const fx=fixture(),{context,calls,removed}=viewer(fx,()=>{}),controller=new AbortController();
  const pending=context.browserCase(fx.items[1],controller.signal);controller.abort(new Error('selection changed'));
  await assert.rejects(pending,/selection changed/);assert.equal(calls.length,1);assert.equal(removed.length,1);
  assert.equal(context[callback]('replay-01',fx.payloads[1]),false);
});
test('script deadline cancels its pending DOM resource without fetch fallback',async()=>{
  const fx=fixture(),{context,removed}=viewer(fx,()=>{});context.setTimeout=cb=>setTimeout(cb,10);
  await assert.rejects(context.browserCase(fx.items[1],new AbortController().signal),/after 30 s/);assert.equal(removed.length,1);
});
test('a loaded script with missing or unexpected callback cannot open another case',async()=>{
  const fx=fixture(),{context}=viewer(fx,(script,context)=>queueMicrotask(()=>{assert.equal(context[callback]('replay-99',fx.payloads[1]),false);script.onload?.();}));
  await assert.rejects(context.browserCase(fx.items[1],new AbortController().signal),/expected replay data callback/);
});
