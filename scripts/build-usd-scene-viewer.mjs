import {readFile,writeFile} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const vendor=existsSync(path.join(root,'viewer/node_modules/esbuild'))?root:path.resolve(root,'../dvidia-training');
const require=createRequire(path.join(vendor,'viewer/package.json'));
for(const [name,version] of [['three','0.180.0'],['esbuild','0.25.10']]){
  if(JSON.parse(await readFile(path.join(vendor,'viewer/node_modules',name,'package.json'),'utf8')).version!==version)throw Error('Unexpected viewer dependency version.');
}
await require('esbuild').build({entryPoints:[path.join(root,'src/dvidia_training/usd_scene_viewer.js')],outfile:path.join(root,'src/dvidia_training/usd_scene_viewer.bundle.js'),bundle:true,minify:true,format:'iife',target:['es2020'],platform:'browser',legalComments:'none',charset:'utf8',nodePaths:[path.join(vendor,'viewer/node_modules')],alias:{three:path.join(vendor,'viewer/node_modules/three/build/three.module.js'),'three/addons/controls/OrbitControls.js':path.join(vendor,'viewer/node_modules/three/examples/jsm/controls/OrbitControls.js')},banner:{js:'/* DVIDIA USD workstation. MIT. Bundled Three.js r180 and OrbitControls, MIT; see reference_replay.third-party.txt. */'}});
console.log('Built offline USD scene viewer.');
