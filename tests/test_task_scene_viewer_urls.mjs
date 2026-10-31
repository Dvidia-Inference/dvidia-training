import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

// Execute the exact browser helper without constructing a WebGL context.
const source=readFileSync(new URL('../src/dvidia_training/task_scene_viewer.js',import.meta.url),'utf8');
const start=source.indexOf('function resourceURL('),end=source.indexOf('\nasync function getBytes',start);
assert.ok(start>=0&&end>start);
const rootURL=new URL('https://example.test/releases/workstation/');
const context={URL,rootURL,location:{origin:rootURL.origin,protocol:rootURL.protocol},assert:(ok,message)=>{if(!ok)throw Error(message)}};
vm.createContext(context);vm.runInContext(source.slice(start,end)+'\nthis.resolveResource=resourceURL;',context);
const resolve=context.resolveResource,index=new URL('benchmark/index.json',rootURL),model=new URL('cases/case-00/model.json',index);

test('deduplicated geometry may traverse to its own artifact root',()=>{
  assert.equal(resolve('../../geometry/abc.bin.gz',model,index).href,'https://example.test/releases/workstation/benchmark/geometry/abc.bin.gz');
  assert.equal(resolve('poses.bin.gz',model,index).pathname,'/releases/workstation/benchmark/cases/case-00/poses.bin.gz');
});
test('native artifacts cannot escape to website or API files',()=>{
  for(const path of ['../../../secret.json','/api/config','/releases/workstation/benchmark-other/file','../../../../api/config'])assert.throws(()=>resolve(path,model,index));
});
test('ambiguous paths and remote origins are refused',()=>{
  for(const path of ['//foreign.test/file','https://example.test/file','file.bin?token=x','file.bin#anchor','%2e%2e/file','..\\file'])assert.throws(()=>resolve(path,model,index));
  assert.throws(()=>resolve(new URL('https://foreign.test/file'),model,index));
  assert.throws(()=>resolve(new URL('https://user:pass@example.test/releases/workstation/benchmark/file'),model,index));
});
test('separate local configuration flow can resolve its named API resource',()=>{
  const localRoot=new URL('https://example.test/');
  assert.equal(resolve('/api/robot-geometry.bin.gz',localRoot,localRoot).pathname,'/api/robot-geometry.bin.gz');
});
