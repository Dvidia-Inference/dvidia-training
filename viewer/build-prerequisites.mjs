import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(path.join(root,'viewer/package.json'));
const {build}=require('esbuild');
await build({entryPoints:[path.join(root,'src/dvidia_training/prerequisite_lab.js')],outfile:path.join(root,'src/dvidia_training/prerequisite_lab.bundle.js'),bundle:true,minify:true,format:'iife',target:['es2020'],platform:'browser',alias:{'three':path.join(root,'viewer/node_modules/three/build/three.module.js'),'three/addons/controls/OrbitControls.js':path.join(root,'viewer/node_modules/three/examples/jsm/controls/OrbitControls.js')},nodePaths:[path.join(root,'viewer/node_modules')],legalComments:'none',charset:'utf8',banner:{js:'/* DVIDIA prerequisite experiments. MIT; bundled Three.js r180, MIT. See reference_replay.third-party.txt. */'}});
console.log('Built prerequisite_lab.bundle.js using pinned local dependencies.');
