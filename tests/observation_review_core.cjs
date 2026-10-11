/* Pure review invariants. No model-accuracy claim. */
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'../src/dvidia_training/observation_studio.html'),'utf8');
const code=html.split('// BEGIN REVIEW CORE')[1].split('// END REVIEW CORE')[0];
const context={}; vm.createContext(context);vm.runInContext('//'+code,context);
const event=(id,start,end,label='Hand visible',decision='pending')=>({id,observation_id:id,label,decision,start_seconds:start,end_seconds:end});
const rows=[event('a',0,1),event('b',.5,1.5),event('c',2,3),event('d',0,1,'Cup visible')];
const raw=rows.map(e=>({...e,evidence:'detector'})), original=JSON.stringify({rows,raw});
let groups=context.groupObservations(rows,raw); assert.equal(groups.length,3); assert.equal(groups.find(g=>g.id==='a').ids.length,2);
assert.equal(JSON.stringify({rows,raw}),original,'Grouping never mutates evidence');
rows[1].decision='rejected'; assert.equal(context.groupObservations(rows,raw).length,4,'Preserve separately decided evidence');
rows[1].decision='pending'; rows[1].label='Corrected hand label'; assert.equal(context.groupObservations(rows,raw).length,4,'Preserve individual corrections');
rows[1].label='Hand visible';raw[1].evidence='overlap';assert.equal(context.groupObservations(rows,raw).length,4,'Keep evidence types distinct');
const manual=[{...event('manual-a',0,1),observation_id:null},{...event('manual-b',0,1),observation_id:null}];
assert.equal(context.groupObservations(manual,[]).length,2,'Manual annotations stay independent');
const boxes=[{time_seconds:0},{time_seconds:.5},{time_seconds:2}];
for(const [time,want] of [[0,0],[.25,0],[.26,.5],[.75,.5],[.76,null],[1.5,null],[1.75,2],[2.25,2],[2.26,null]])
 assert.equal(context.nearestSample(boxes,time)?.time_seconds??null,want);
assert.equal(context.nearestSample([],0),null);
console.log('Grouping, evidence preservation and sample-boundary invariants passed.');
