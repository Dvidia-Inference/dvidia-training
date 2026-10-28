import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {existsSync} from 'node:fs';

const [sourceArg, outputArg] = process.argv.slice(2);
if (!sourceArg || !outputArg || process.argv.length !== 4) {
  throw new Error('Usage: node scripts/build-piper-grip-clock-viewer.mjs <source.js> <output.js>');
}
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.resolve(sourceArg), outfile = path.resolve(outputArg);
if (source === outfile || /piper_feedback_viewer\.bundle\.js$/.test(outfile)) {
  throw new Error('The v0.3 build requires a distinct output path.');
}
const vendorRoot = existsSync(path.join(root, 'viewer/node_modules/esbuild'))
  ? root
  : path.resolve(root, '../dvidia-training');
const require = createRequire(path.join(vendorRoot, 'viewer/package.json'));
for (const [name, version] of [['esbuild', '0.25.10'], ['three', '0.180.0']]) {
  const packagePath = path.join(vendorRoot, 'viewer/node_modules', name, 'package.json');
  const installed = JSON.parse(await readFile(packagePath, 'utf8')).version;
  if (installed !== version) throw new Error(`Expected pinned ${name} ${version}; found ${installed}.`);
}
const {build} = require('esbuild');
await build({
  entryPoints: [source], outfile, bundle: true, minify: true, format: 'iife',
  target: ['es2020'], platform: 'browser',
  alias: {
    three: path.join(vendorRoot, 'viewer/node_modules/three/build/three.module.js'),
    'three/addons/controls/OrbitControls.js': path.join(vendorRoot, 'viewer/node_modules/three/examples/jsm/controls/OrbitControls.js'),
  },
  nodePaths: [path.join(vendorRoot, 'viewer/node_modules')], legalComments: 'none', charset: 'utf8',
  banner: {js: '/* DVIDIA PiPER grip clock replay v0.3. MIT. Bundled Three.js r180, MIT; https://github.com/mrdoob/three.js/tree/r180 */'},
});
const output = await readFile(outfile);
if (output.byteLength > 8 * 1024 * 1024) throw new Error('Compiled viewer exceeds the 8 MiB bundle limit.');
console.log(`Built ${outfile} (${output.byteLength} bytes).`);
