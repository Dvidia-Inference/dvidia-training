/* Exercise the actual browser adapter with synthetic browser APIs; no network. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {webcrypto} = require('node:crypto');
const {TextEncoder} = require('node:util');
const source = fs.readFileSync(`${__dirname}/templates/demo-adapter.js`,'utf8');
const episode = {id:'visible',duration_seconds:8,source:{sha256:'a'.repeat(64)},
  observations:[{id:'sample-1',label:'Blue shape',start_seconds:0,end_seconds:8,boxes:[]}],
  review:{revision:0,status:'unreviewed',outcome:'unknown',events:[
    {id:'sample-1',observation_id:'sample-1',label:'Blue shape',start_seconds:0,end_seconds:8,decision:'pending'}]}};
const copy = value => JSON.parse(JSON.stringify(value));
function storageObject() {
  const storage = {};
  Object.defineProperties(storage,{
    setItem:{value(key,value){storage[key]=String(value);}},
    getItem:{value(key){return Object.hasOwn(storage,key)?storage[key]:null;}},
    removeItem:{value(key){delete storage[key];}}
  });
  return storage;
}
function lockObject() {
  const chains = new Map();
  return {request(key,callback) {
    const promise = (chains.get(key) || Promise.resolve()).then(callback);
    chains.set(key,promise.catch(()=>{})); return promise;
  }};
}
function browser(storage=storageObject(),locks=lockObject(),blocked=false) {
  const requests = [], notice = {textContent:''};
  const context = {window:{},navigator:{locks},crypto:webcrypto,TextEncoder,
    localStorage:blocked?{setItem(){throw new Error('blocked');}}:storage,
    document:{getElementById(){return notice;}},fetch:async(path,options)=>{
      requests.push({path,options});
      return {ok:true,json:async()=>path==='demo/episodes.json'?{episodes:[{id:'visible'}],failures:[]}:copy(episode)};
    }};
  vm.runInNewContext(source,context);
  return {api:context.window.DvidiaObservationDemo,requests,notice,storage};
}
const body = decision => ({base_revision:0,status:'in_review',outcome:'unknown',
  events:episode.review.events.map(event=>({...event,label:'Corrected blue shape',decision}))});
(async()=>{
  const b=browser(); await b.api.list();
  assert.match(b.notice.textContent,/stay in this browser/);
  await b.api.save('visible',body('approved'));
  const saved=await b.api.episode('visible');
  assert.equal(saved.review.revision,1);assert.equal(saved.review.events[0].label,'Corrected blue shape');
  assert.equal(saved.observations[0].label,'Blue shape');assert.equal(saved.review.scope,'synthetic_workflow_demo');
  assert.match(saved.review.parent_sha256,/^[a-f0-9]{64}$/);
  assert(b.requests.every(row=>row.options.method===undefined));
  const reload=browser(b.storage);assert.equal((await reload.api.episode('visible')).review.revision,1);
  await assert.rejects(()=>reload.api.save('visible',body('approved')),/changed in another/);
  assert.equal((await browser().api.episode('visible')).review.revision,0);
  const shared=storageObject(),locks=lockObject(),one=browser(shared,locks),two=browser(shared,locks);
  const race=await Promise.allSettled([one.api.save('visible',body('approved')),two.api.save('visible',body('rejected'))]);
  assert.equal(race.filter(row=>row.status==='fulfilled').length,1);
  assert.equal(race.filter(row=>row.status==='rejected').length,1);
  assert.equal((await one.api.episode('visible')).review.revision,1);
  const fresh=browser();
  const bad=body('pending');bad.status='reviewed';
  await assert.rejects(()=>fresh.api.save('visible',bad),/decision and timing/);
  await assert.rejects(()=>fresh.api.save('visible',{...body('approved'),events:[]}),/every original/);
  const invalid=body('approved');invalid.events[0].end_seconds=9;
  await assert.rejects(()=>fresh.api.save('visible',invalid),/decision and timing/);
  const manual=body('approved');manual.events.push({id:'manual-test',observation_id:null,label:'Manual event',
    start_seconds:2,end_seconds:3,decision:'approved'});manual.status='reviewed';
  await fresh.api.save('visible',manual);assert.equal((await fresh.api.episode('visible')).review.events.length,2);
  await assert.rejects(()=>fresh.api.episode('../../private'),/included synthetic/);
  const key=Object.keys(fresh.storage)[0],history=JSON.parse(fresh.storage[key]);
  history[0].parent_sha256='b'.repeat(64);fresh.storage[key]=JSON.stringify(history);
  await assert.rejects(()=>fresh.api.episode('visible'),/history changed/);
  const blocked=browser(undefined,undefined,true);await blocked.api.list();
  assert.match(blocked.notice.textContent,/only while this page stays open/);
  await blocked.api.save('visible',body('approved'));assert.equal((await blocked.api.episode('visible')).review.revision,1);
  assert.equal((await browser(undefined,undefined,true).api.episode('visible')).review.revision,0);
  await b.api.reset();assert.equal((await b.api.episode('visible')).review.revision,0);
  console.log('Observation demo adapter: browser isolation, revision conflicts, no remote writes, validation and memory fallback passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
