import {createRequire} from 'node:module';
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const candidates = [process.env.DVIDIA_VIEWER_NODE_MODULES,
  path.join(root, 'viewer/node_modules'), path.join(root, '../dvidia-training/viewer/node_modules')].filter(Boolean);
const modules = candidates.find(candidate => existsSync(path.join(candidate, 'esbuild/lib/main.js')) && existsSync(path.join(candidate, 'three/build/three.module.js')));
if (!modules) throw new Error('Pinned local viewer dependencies are missing. Set DVIDIA_VIEWER_NODE_MODULES to an existing node_modules directory.');
const require = createRequire(path.join(modules, 'esbuild/package.json'));
const {build} = require(path.join(modules, 'esbuild/lib/main.js'));
const outfile = path.join(root, 'src/dvidia_training/piper_feedback_viewer.bundle.js');
await build({
  entryPoints: [path.join(root, 'src/dvidia_training/piper_feedback_viewer.js')],
  outfile,
  bundle: true, minify: true, format: 'iife', target: ['es2020'], platform: 'browser',
  alias: {'three': path.join(modules, 'three/build/three.module.js'),
    'three/addons/controls/OrbitControls.js': path.join(modules, 'three/examples/jsm/controls/OrbitControls.js')},
  nodePaths: [modules], legalComments: 'none', charset: 'utf8',
  banner: {js: '/* DVIDIA PiPER recorded feedback replay. MIT; Three.js r180/OrbitControls, MIT. See reference_replay.third-party.txt. */'},
});
// Three.js embeds GLSL templates whose line-end spaces survive minification.
// Strip those insignificant spaces in every build for a reproducible clean diff.
writeFileSync(outfile, readFileSync(outfile, 'utf8').replace(/[\t ]+$/gm, ''));
console.log('Built the self-hosted PiPER feedback viewer with pinned local dependencies.');
