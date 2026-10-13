import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {readFile, writeFile, copyFile} from 'node:fs/promises';
import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(path.join(root,'viewer/package.json'));
const {build}=require('esbuild');
const source=path.join(root,'src/dvidia_training');
const modules=path.join(root,'viewer/node_modules');
const packageInfo=JSON.parse(await readFile(path.join(modules,'@mujoco/mujoco/package.json'),'utf8'));
if(packageInfo.version!=='3.15.0')throw new Error('The live workstation requires the pinned MuJoCo 3.15.0 runtime.');
const options={bundle:true,minify:true,target:['es2022'],platform:'browser',
  nodePaths:[modules],legalComments:'none',charset:'utf8',
  alias:{three:path.join(modules,'three/build/three.module.js'),
    'three/addons/controls/OrbitControls.js':path.join(modules,'three/examples/jsm/controls/OrbitControls.js'),
    'three/addons/controls/TransformControls.js':path.join(modules,'three/examples/jsm/controls/TransformControls.js'),
    '@mujoco/mujoco':path.join(modules,'@mujoco/mujoco/mujoco.js')}};
await build({...options,entryPoints:[path.join(source,'workstation.js')],
  outfile:path.join(source,'workstation.bundle.js'),format:'esm',
  banner:{js:'/* DVIDIA live workstation. MIT. Three.js r180 MIT. See workstation.third-party.txt. */'}});
await build({...options,entryPoints:[path.join(source,'workstation_worker.js')],
  outfile:path.join(source,'workstation_worker.bundle.js'),format:'esm',external:['module'],
  banner:{js:'/* DVIDIA live workstation worker. MIT. MuJoCo 3.15.0 Apache-2.0. See workstation.third-party.txt. */'}});
await copyFile(path.join(modules,'@mujoco/mujoco/mujoco.wasm'),path.join(source,'mujoco.wasm'));
const notices=await readFile(path.join(source,'reference_replay.third-party.txt'),'utf8');
const engineLicense=await readFile(path.join(root,'third_party/mujoco/LICENSE.txt'),'utf8');
const engineNotices=await readFile(path.join(root,'third_party/mujoco/THIRD_PARTY_NOTICES.txt'),'utf8');
await writeFile(path.join(source,'workstation.third-party.txt'),notices+'\n\nMuJoCo 3.15.0, Google DeepMind\nhttps://github.com/google-deepmind/mujoco/tree/3.15.0\nOfficial WASM distribution, unmodified binary; JavaScript bundled/minified by esbuild.\n\n'+engineLicense+'\n\n'+engineNotices);
console.log('Built live workstation UI, worker and pinned MuJoCo 3.15.0 WASM.');
