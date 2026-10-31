import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../src/dvidia_training/task_scene_viewer.js',import.meta.url),'utf8');
function block(start,end){const a=source.indexOf(start),b=source.indexOf(end,a);assert.ok(a>=0&&b>a);return source.slice(a,b)}
const rootURL=new URL('https://example.test/artifact/');
const guard=(ok,message)=>{if(!ok)throw Error(message)};
const resourceURL=(path,base=rootURL)=>new URL(path,base);
function response(status,bytes=new Uint8Array([1,2])){return {status,ok:status===200,url:rootURL.href+'file',body:new ReadableStream({start(controller){controller.enqueue(bytes);controller.close()}})}}
function byteReader(fetch){const context={URL,rootURL,location:{origin:rootURL.origin,hostname:rootURL.hostname},resourceURL,assert:guard,Uint8Array,TypeError,fetch,setTimeout:callback=>callback()};vm.createContext(context);vm.runInContext(block('async function getBytes(','\nasync function getJSON')+'\nthis.read=getBytes;',context);return context.read}

test('idempotent reads retry transport failures at most three times',async()=>{
  let count=0;const read=byteReader(async()=>{if(++count<3)throw new TypeError('Failed to fetch');return response(200)});
  assert.deepEqual(Array.from(await read('file')),[1,2]);assert.equal(count,3);
  count=0;const unavailable=byteReader(async()=>{count++;throw new TypeError('Failed to fetch')});
  await assert.rejects(unavailable('file'),/Failed to fetch/);assert.equal(count,3);
});
test('transient HTTP statuses retry, permanent missing files do not',async()=>{
  let count=0;const transient=byteReader(async()=>response(++count===1?503:200));await transient('file');assert.equal(count,2);
  count=0;const missing=byteReader(async()=>{count++;return response(404)});await assert.rejects(missing('file'),/Could not open/);assert.equal(count,1);
});
test('byte bounds and unexpected redirects are not retried',async()=>{
  let count=0;const oversized=byteReader(async()=>{count++;return response(200,new Uint8Array(8))});await assert.rejects(oversized('file',rootURL,4),/byte limit/);assert.equal(count,1);
  count=0;const foreign=byteReader(async()=>{count++;return {...response(200),url:'https://foreign.test/file'}});await assert.rejects(foreign('file'),/Unexpected resource host/);assert.equal(count,1);
});
test('geometry cache keys exact URL, hash and byte count, and evicts within its byte budget',async()=>{
  let verified=0;const context={URL,rootURL,resourceURL,assert:guard,MAX_BYTES:8,Map,unpack:async(_url,identity)=>{verified++;return new ArrayBuffer(identity.bytes)}};
  vm.createContext(context);vm.runInContext(block('const geometryCache=','\nfunction status')+'\nthis.cached=cachedGeometry;this.total=()=>geometryCacheBytes;',context);
  const identity={bytes:3,sha256:'a'.repeat(64)};const first=await context.cached('geometry/a.gz',identity,rootURL);
  assert.equal(await context.cached('geometry/a.gz',identity,rootURL),first);assert.equal(verified,1);
  await context.cached('geometry/a.gz',{...identity,sha256:'b'.repeat(64)},rootURL);assert.equal(verified,2);
  await context.cached('geometry/b.gz',{...identity,bytes:5},rootURL);assert.equal(context.total(),8);
  await context.cached('geometry/a.gz',identity,rootURL);assert.equal(verified,4);assert.ok(context.total()<=8);
});
test('unverified geometry never enters the cache',async()=>{
  let verified=0;const context={URL,rootURL,resourceURL,assert:guard,MAX_BYTES:8,Map,unpack:async()=>{verified++;throw Error('SHA-256 mismatch')}};
  vm.createContext(context);vm.runInContext(block('const geometryCache=','\nfunction status')+'\nthis.cached=cachedGeometry;',context);
  const identity={bytes:3,sha256:'a'.repeat(64)};await assert.rejects(context.cached('geometry.gz',identity,rootURL),/SHA-256/);await assert.rejects(context.cached('geometry.gz',identity,rootURL),/SHA-256/);assert.equal(verified,2);
});
test('a failed selection clears old model, poses, metrics and playback, and offers retry',async()=>{
  const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{dataset:{},textContent:'old result',value:'new',hidden:false,disabled:false});return nodes.get(id)};
  let removed=0;const context={$,assert:guard,loadEpoch:0,pause:()=>{},currentCase:{id:'old'},poseData:{old:true},bodyGroups:[1],frame:12,replayGroup:{old:true},removeGroup:()=>removed++,render:()=>{},indexURL:new URL('benchmark/index.json',rootURL),indexData:{cases:[{id:'new',scene:'cases/new/model.json',frame_count:2,body_count:1,dt_s:.01}]},resourceURL,getJSON:async()=>{throw Error('Network unavailable')}};
  vm.createContext(context);vm.runInContext(block('async function selectRecording(','\nasync function loadIndex')+'\nthis.select=selectRecording;',context);
  await assert.rejects(context.select(),/Network unavailable/);assert.equal(context.currentCase,null);assert.equal(context.poseData,null);assert.equal(context.replayGroup,null);assert.equal(removed,1);
  assert.equal($('play').disabled,true);assert.equal($('timeline').disabled,true);assert.equal($('goal-result').textContent,'—');assert.equal($('expected-result').textContent,'—');assert.equal($('integrity').textContent,'Not verified');assert.equal($('retry-recording').hidden,false);assert.equal($('replay-error').hidden,false);assert.equal($('stage').dataset.replayReady,'error');
});
