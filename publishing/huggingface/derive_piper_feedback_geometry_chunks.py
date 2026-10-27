"""Derive reusable, hash-bound small geometry transfers from a sealed viewer.

Offline only. The original geometry and all scientific evidence stay byte
identical. Output is a fresh hosting projection, never a mutation of its source.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import publish_piper_feedback_hosting as hosting


SCHEMA = 'dvidia.piper-feedback-geometry-transport/v1'
CHUNK_BYTES = 262144
CHANGED = {'scene.json', 'piper_feedback_viewer.js', 'piper_feedback_viewer.bundle.js', 'viewer-integrity.json'}


NODE_BUILD = r"""
import {createRequire} from 'node:module';
import {existsSync,readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
const [root,source,outfile] = process.argv.slice(1);
const candidates=[path.join(root,'viewer/node_modules'),path.join(root,'../dvidia-training/viewer/node_modules')];
const modules=candidates.find(p=>existsSync(path.join(p,'esbuild/lib/main.js'))&&existsSync(path.join(p,'three/build/three.module.js')));
if(!modules)throw new Error('Pinned viewer dependencies are unavailable.');
const require=createRequire(path.join(modules,'esbuild/package.json'));
await require(path.join(modules,'esbuild/lib/main.js')).build({entryPoints:[source],outfile,bundle:true,minify:true,format:'iife',target:['es2020'],platform:'browser',alias:{three:path.join(modules,'three/build/three.module.js'),'three/addons/controls/OrbitControls.js':path.join(modules,'three/examples/jsm/controls/OrbitControls.js')},nodePaths:[modules],legalComments:'none',charset:'utf8',banner:{js:'/* DVIDIA PiPER recorded feedback replay. MIT; Three.js r180/OrbitControls, MIT. See reference_replay.third-party.txt. */'}});
writeFileSync(outfile,readFileSync(outfile,'utf8').replace(/[\t ]+$/gm,''));
"""


def _once(source, old, new):
    if source.count(old) != 1:
        raise hosting.PublicationError('Source transport differs from the reviewed patch anchor.')
    return source.replace(old, new, 1)


def patched_source(raw):
    source = raw.decode('utf8')
    source = _once(source, 'async function deadline(href, milliseconds, operation) {',
        'async function deadline(href, milliseconds, operation, outerSignal) {')
    source = _once(source, '  const timer = setTimeout(() => controller.abort(',
        "  const propagate = () => controller.abort(outerSignal.reason);\n  if (outerSignal?.aborted) throw outerSignal.reason;\n  outerSignal?.addEventListener('abort', propagate, {once: true});\n  const timer = setTimeout(() => controller.abort(")
    source = _once(source, '    clearTimeout(timer);',
        "    clearTimeout(timer);\n    outerSignal?.removeEventListener('abort', propagate);")
    source = _once(source, 'async function binary(meta, expected) {', 'async function binary(meta, expected, outerSignal) {')
    source = _once(source, '        return verifiedBinary(bytes, meta);\n      });',
        '        return verifiedBinary(bytes, meta);\n      }, outerSignal);')
    source = _once(source, '    } catch (error) {\n      // The raw bytes',
        '    } catch (error) {\n      if (outerSignal?.aborted) throw error;\n      // The raw bytes')
    source = _once(source, "return deadline(meta.href, 30000, async signal => verifiedBinary(await bounded((await response(meta.href, signal)).body, meta.bytes, signal), meta));",
        "return deadline(meta.href, 30000, async signal => verifiedBinary(await bounded((await response(meta.href, signal)).body, meta.bytes, signal), meta), outerSignal);")
    addon = Path(__file__).with_name('piper_feedback_geometry_transport.js').read_text()
    source = _once(source, 'function validate(geometry) {', addon+'\nfunction validate(geometry) {')
    source = _once(source, "const geometry = await binary(model.geometry_binary, 'geometry.bin');",
        'const geometry = await geometryChunks(model.geometry_binary);')
    return source.encode('utf8')


def derive(viewer, output, *, expected_viewer_sha256):
    viewer, output = map(hosting.study._plain, (viewer, output))
    if output.exists():
        raise hosting.PublicationError('Geometry hosting projection must use a fresh output.')
    original, parent = hosting._source_viewer(viewer, expected_viewer_sha256)
    scene = json.loads(original['scene.json'])
    meta = scene.get('geometry_binary')
    raw = original['geometry.bin']
    if (not isinstance(meta, dict) or meta.get('href') != 'geometry.bin'
            or hosting._receipt(raw) != {k: meta[k] for k in ('bytes', 'sha256')}
            or 'chunks' in meta):
        raise hosting.PublicationError('Expected unchanged original geometry with a whole-file digest.')
    derived_source = patched_source(original['piper_feedback_viewer.js'])
    shutil.copytree(viewer, output)
    chunks = []
    for index, offset in enumerate(range(0, len(raw), CHUNK_BYTES)):
        piece = raw[offset:offset+CHUNK_BYTES]
        name = f'geometry-part-{index:03}.bin'
        packed = gzip.compress(piece, mtime=0)
        (output/name).write_bytes(piece); (output/(name+'.gz')).write_bytes(packed)
        chunks.append(dict(href=name, bytes=len(piece), sha256=sha256(piece).hexdigest(),
                           compressed_href=name+'.gz', compressed_bytes=len(packed), compression='gzip'))
    # These same original bytes remain the authority for every mesh/index bound.
    scene['geometry_binary']['chunks'] = chunks
    (output/'scene.json').write_bytes(hosting.study._encoded(scene))
    (output/'piper_feedback_viewer.js').write_bytes(derived_source)
    subprocess.run(['node', '--input-type=module', '-e', NODE_BUILD, str(Path(__file__).resolve().parents[2]),
                    str(output/'piper_feedback_viewer.js'), str(output/'piper_feedback_viewer.bundle.js')], check=True)
    note = dict(schema=SCHEMA, parent_viewer_integrity_sha256=expected_viewer_sha256,
                original_geometry_sha256=meta['sha256'], original_geometry_bytes=meta['bytes'],
                chunk_bytes=CHUNK_BYTES, chunks=len(chunks), parallel_fetches=3,
                compressed_deadline_s=8, raw_deadline_s=30, total_geometry_deadline_s=120,
                whole_geometry_verification='Per-chunk SHA256 and byte count, then exact original whole-file SHA256; unchanged original scene mesh/index validation follows.',
                observed_transport='One initial public research replay load reported the geometry.bin raw fallback exceeding its 30-second deadline. A subsequent intentional reload succeeded with the original viewer. Local gzip/raw bytes and digests agree. The precise network/server cause was not measured; this is bounded cold-transfer robustness, not a claim of universal failure or data corruption.',
                scope='Hosting transport only. No changes to physics, policy, model, scoring, recorded states, original geometry or evidence. Original sealed viewer remains untouched.',
                changed_original_paths=sorted(CHANGED),
                implementation_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__), Path(__file__).with_name('piper_feedback_geometry_transport.js'))})
    (output/'GEOMETRY-TRANSPORT.json').write_bytes(hosting.study._encoded(note))
    seal = deepcopy(parent)
    seal['geometry_transport_derivation'] = note
    seal['files'] = {p.relative_to(output).as_posix(): hosting._receipt(p.read_bytes())
                     for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'viewer-integrity.json'}
    (output/'viewer-integrity.json').write_bytes(hosting.study._encoded(seal))
    for name, before in original.items():
        if name not in CHANGED and (output/name).read_bytes() != before:
            raise hosting.PublicationError('An original scientific or rendering artifact changed.')
    joined = b''.join((output/chunk['href']).read_bytes() for chunk in chunks)
    if joined != raw:
        raise hosting.PublicationError('Derived chunks differ from original geometry bytes.')
    return dict(schema=SCHEMA, output=str(output), chunks=len(chunks), original_geometry_sha256=meta['sha256'],
                parent_viewer_integrity_sha256=expected_viewer_sha256,
                derived_viewer_integrity_sha256=sha256((output/'viewer-integrity.json').read_bytes()).hexdigest(),
                changed_original_paths=sorted(CHANGED), scientific_evidence_unchanged=True,
                files=len(seal['files'])+1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-viewer-sha256', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(derive(args.viewer, args.output, expected_viewer_sha256=args.expected_viewer_sha256), indent=2))


if __name__ == '__main__':
    main()
