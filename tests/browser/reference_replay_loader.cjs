/* Offline checks of the actual loader functions; no server, browser or network. */
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const {gzipSync} = require('node:zlib');

async function runLoaderChecks() {
  const source = fs.readFileSync(path.resolve(__dirname,'../../src/dvidia_training/reference_replay.js'),'utf8');
  const start = source.indexOf('function assert('), end = source.indexOf('function checkSchema(');
  assert.ok(start>0 && end>start, 'Expected the real loader function boundary');
  const requests = [];
  let finalURL, payload, status, fetchError;
  const context = vm.createContext({URL,TextDecoder,Blob,DecompressionStream,Uint8Array,
    MAX_BYTES:64*1024*1024,MAX_JSON:12*1024*1024,finite:n=>typeof n==='number'&&Number.isFinite(n),
    location:new URL('https://dvidia-robot-simulation-lab.static.hf.space/replay/index.html'),
    fetch:async (url,options)=>{
      requests.push({url,options});
      if(fetchError)throw fetchError;
      return {url:finalURL||url,ok:status>=200&&status<300,status,
        body:new Blob([payload]).stream()};
    }});
  vm.runInContext(source.slice(start,end),context);
  const reset = (location,final,payloadBytes=Uint8Array.of(1,2,3))=>{
    context.location=new URL(location);finalURL=final;payload=payloadBytes;status=200;fetchError=undefined;requests.length=0;
  };
  const space='https://dvidia-robot-simulation-lab.static.hf.space/replay/index.html';
  reset(space,'https://us.aws.cdn.hf.co/xet-bridge-us/fixture?download=1');
  await context.response('geometry.bin.gz');
  assert.equal(requests[0].url,'https://dvidia-robot-simulation-lab.static.hf.space/replay/geometry.bin.gz');
  assert.equal(requests[0].options.redirect,'follow');
  assert.equal(requests[0].options.credentials,'omit');
  assert.equal(requests[0].options.referrerPolicy,'no-referrer');
  reset(space);
  await context.response('scene.json');
  for(const location of ['http://127.0.0.1:8291/replay/index.html','https://research.dvidia.org/replay/index.html',
    'http://dvidia-robot-simulation-lab.static.hf.space/replay/index.html',
    'https://dvidia-robot-simulation-lab.static.hf.space:444/replay/index.html',
    'https://dvidia-robot-simulation-lab.static.hf.space.evil.example/replay/index.html']) {
    reset(location);
    await context.response('geometry.bin');
    assert.equal(requests[0].options.redirect,'error');
    reset(location,'https://us.aws.cdn.hf.co/xet-bridge-us/fixture');
    await assert.rejects(()=>context.response('geometry.bin'),/Unexpected replay download origin/);
  }
  for(const final of ['https://evil.example/geometry.bin','http://us.aws.cdn.hf.co/geometry.bin',
    'https://us.aws.cdn.hf.co.evil.example/geometry.bin','https://us.aws.cdn.hf.co:444/geometry.bin',
    'https://embedded:credential@us.aws.cdn.hf.co/geometry.bin']) {
    reset(space,final);
    await assert.rejects(()=>context.response('geometry.bin'),/Unexpected replay download origin/);
  }
  for(const href of ['https://us.aws.cdn.hf.co/geometry.bin','//us.aws.cdn.hf.co/geometry.bin',
    '../geometry.bin','receipt/../geometry.bin','/geometry.bin']) {
    reset(space);
    await assert.rejects(()=>context.response(href),/Replay file link must stay within this directory/);
    assert.equal(requests.length,0);
  }
  const native=Uint8Array.of(1,2,3,4,5,6,7,8),packed=gzipSync(native);
  const meta={href:'geometry.bin',compressed_href:'geometry.bin.gz',bytes:native.byteLength,
    compressed_bytes:packed.byteLength,compression:'gzip'};
  reset(space,'https://us.aws.cdn.hf.co/xet-bridge-us/fixture',packed);
  assert.deepEqual(new Uint8Array(await context.binary(meta,'geometry.bin')),native);
  reset(space,'https://us.aws.cdn.hf.co/xet-bridge-us/fixture',packed);
  await assert.rejects(()=>context.binary({...meta,bytes:7},'geometry.bin'),/exceeds its byte bounds/);
  reset(space,'https://us.aws.cdn.hf.co/xet-bridge-us/fixture',packed.subarray(1));
  await assert.rejects(()=>context.binary(meta,'geometry.bin'),/Compressed geometry.bin size mismatch/);
  reset(space,'https://us.aws.cdn.hf.co/xet-bridge-us/fixture',Buffer.alloc(packed.length));
  await assert.rejects(()=>context.binary(meta,'geometry.bin'));
  reset(space);
  fetchError=new TypeError('Fixture network failure');
  await assert.rejects(()=>context.response('geometry.bin.gz'),/Fixture network failure/);
  assert.equal(requests.length,1,'Failed redirect/network routes are not silently replaced');
  reset(space);status=404;
  await assert.rejects(()=>context.response('geometry.bin'),/HTTP 404/);
  return {scope:'Offline native replay loader origin and byte boundaries',checks:8,networkRequests:0};
}

module.exports=runLoaderChecks;
if(require.main===module)runLoaderChecks().then(result=>console.log(JSON.stringify(result))).catch(error=>{console.error(error);process.exitCode=1;});
