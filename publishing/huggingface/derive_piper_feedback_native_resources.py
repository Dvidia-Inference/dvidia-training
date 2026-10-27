"""Offline browser-managed data resource experiment; original science stays fixed."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import gzip
from io import BytesIO
from hashlib import sha256, sha384
import json
from pathlib import Path
import re
import shutil
import subprocess

import derive_piper_feedback_geometry_chunks as geometry
import publish_piper_feedback_hosting as hosting


SCHEMA = 'dvidia.piper-feedback-native-resources/v1'
PAYLOAD_SCHEMA = 'dvidia.piper-feedback-resource-data/v1'
CALLBACK = '__dvidiaPiperFeedbackResourcesV1'
CHANGED = {'index.html', 'piper_feedback_viewer.js', 'piper_feedback_viewer.bundle.js', 'viewer-integrity.json'}
MAX_SCRIPT = 1024 * 1024
MAX_BUNDLE = 8 * 1024 * 1024


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def resource(name, raw, packed=None):
    if not re.fullmatch(r'(?:geometry\.bin|scene\.json|motion\.json|frames-[0-9]{2}\.json|poses-[0-9]{2}\.bin)', name):
        raise hosting.PublicationError('Unexpected browser-managed data resource path.')
    packed = gzip.compress(raw, mtime=0) if packed is None else packed
    if not 0 < len(raw) <= 8*1024*1024 or not 0 < len(packed) <= 4*1024*1024:
        raise hosting.PublicationError('Browser-managed resource exceeds its bounds or differs from original bytes.')
    try:
        with gzip.GzipFile(fileobj=BytesIO(packed)) as stream:
            expanded = stream.read(len(raw)+1)
    except (OSError, EOFError) as error:
        raise hosting.PublicationError('Saved resource gzip is invalid.') from error
    if expanded != raw:
        raise hosting.PublicationError('Browser-managed resource expansion differs from its original bytes.')
    return dict(href=name, **hosting._receipt(raw), compression='gzip', compressed_bytes=len(packed),
                compressed_sha256=sha256(packed).hexdigest(), data=base64.b64encode(packed).decode('ascii'))


def script(payload):
    if (set(payload) != {'schema', 'id', 'resources'} or payload.get('schema') != PAYLOAD_SCHEMA
            or not re.fullmatch(r'replay-[0-9]{2}', payload.get('id', ''))
            or not isinstance(payload.get('resources'), list) or len(payload['resources']) != 2):
        raise hosting.PublicationError('Unexpected literal data wrapper identity.')
    raw = b'globalThis['+encoded(CALLBACK)+b']('+encoded(payload['id'])+b','+encoded(payload)+b');\n'
    if len(raw) > MAX_SCRIPT:
        raise hosting.PublicationError('Literal replay data script exceeds 1 MiB.')
    # Enforce the exact generated grammar, containing only two JSON literals.
    prefix = b'globalThis['+encoded(CALLBACK)+b']('+encoded(payload['id'])+b','
    if not raw.startswith(prefix) or json.loads(raw[len(prefix):-3]) != payload:
        raise hosting.PublicationError('Literal data wrapper changed.')
    return raw


def patched_source(raw, manifest):
    source = raw.decode('utf8')
    addon = Path(__file__).with_name('piper_feedback_native_resources.js').read_text()
    source = geometry._once(source, 'function validate(geometry) {',
        'const BROWSER_RESOURCES = '+encoded(manifest).decode('ascii')+';\n'+addon+'\nfunction validate(geometry) {')
    source = geometry._once(source,
        "  [model, metadata] = await Promise.all([json('scene.json'), json('motion.json')]); evidence();",
        "  [model, metadata] = await Promise.all([resourceJson('scene.json'), resourceJson('motion.json')]); evidence();")
    source = geometry._once(source, "  const geometry = await binary(model.geometry_binary, 'geometry.bin');",
        "  const geometry = await resourceInitial('geometry.bin');\n  assert(geometry.byteLength === model.geometry_binary.bytes, 'Original geometry byte count changed.');\n  await verifiedBinary(geometry, model.geometry_binary);")
    source = geometry._once(source, 'async function choose() {\n  setPlaying(false);',
        "async function choose() {\n  resourceSelection?.abort(new Error('Replay selection changed.'));\n  const selectionController = new AbortController(); resourceSelection = selectionController;\n  setPlaying(false);")
    old = """  if (!caseCache.has(key)) caseCache.set(key, Promise.all([verifiedJson(item.rows_file), binary(item.pose_binary, key)]).then(([rows, motion]) => {
    validateCaseRows(item, rows, motion); return {rows, motion};
  }));
  let loaded;
  try { loaded = await caseCache.get(key); }
  catch (error) { caseCache.delete(key); if (epoch !== selectionEpoch) return; loadingCase = false; throw error; }"""
    new = """  let loaded = caseCache.get(key);
  try {
    if (!loaded) {
      loaded = await browserCase(item, selectionController.signal);
      validateCaseRows(item, loaded.rows, loaded.motion); caseCache.set(key, loaded);
    }
  } catch (error) { if (epoch !== selectionEpoch) return; loadingCase = false; throw error; }"""
    return geometry._once(source, old, new).encode('utf8')


def derive(viewer, output, *, expected_viewer_sha256, dataset_revision=None):
    viewer, output = map(hosting.study._plain, (viewer, output))
    if output.exists():
        raise hosting.PublicationError('Native-resource experiment must use a fresh output.')
    original, parent = hosting._source_viewer(viewer, expected_viewer_sha256)
    if any(key.endswith('_derivation') for key in parent):
        raise hosting.PublicationError('Use the original scientific viewer, leaving prior transport experiments sealed.')
    if dataset_revision is not None and (not isinstance(dataset_revision, str) or re.fullmatch(r'[0-9a-f]{40}', dataset_revision) is None):
        raise hosting.PublicationError('Dataset revision must be an exact lowercase 40-hex commit.')
    motion_raw = original['motion.json'] if dataset_revision is None else hosting._pin_motion(original['motion.json'], dataset_revision)[0]
    scene, motion = json.loads(original['scene.json']), json.loads(motion_raw)
    if len(motion.get('cases', [])) != 28 or len(motion.get('all_attempts', [])) != 120:
        raise hosting.PublicationError('Expected all 28 scheduled replays and all 120 attempts.')
    first = next((c for c in motion['cases'] if c['variant'] == 'feedback' and c['frame_count'] > 0), motion['cases'][0])
    manifest = dict(schema=SCHEMA, embedded={}, cases=[])
    for name in ('scene.json', 'motion.json', 'geometry.bin'):
        raw = motion_raw if name == 'motion.json' else original[name]
        manifest['embedded'][name] = resource(name, raw, original.get(name+'.gz'))
    scripts = {}
    for index, case in enumerate(motion['cases']):
        data = []
        for meta in (case['rows_file'], case['pose_binary']):
            name = meta['href']; raw = original[name]
            if hosting._receipt(raw) != {key:meta[key] for key in ('bytes', 'sha256')}:
                raise hosting.PublicationError('Original replay descriptor differs from its saved bytes.')
            data.append(resource(name, raw, original.get(name+'.gz')))
        payload = dict(schema=PAYLOAD_SCHEMA, id=f'replay-{index:02}', resources=data)
        descriptor = dict(id=payload['id'], case_id=case['id'], variant=case['variant'],
                          resources=[{key:value for key,value in item.items() if key != 'data'} for item in data])
        if case is first:
            manifest['first'] = payload
            descriptor['script'] = None
        else:
            raw = script(payload); name = f'replay-data-{index:02}.js'; scripts[name] = raw
            descriptor['script'] = dict(href=name, bytes=len(raw), integrity='sha384-'+base64.b64encode(sha384(raw).digest()).decode('ascii'))
        manifest['cases'].append(descriptor)
    source = patched_source(original['piper_feedback_viewer.js'], manifest)
    shutil.copytree(viewer, output)
    if dataset_revision is not None: (output/'motion.json').write_bytes(motion_raw)
    for name, raw in scripts.items(): (output/name).write_bytes(raw)
    (output/'piper_feedback_viewer.js').write_bytes(source)
    subprocess.run(['node', '--input-type=module', '-e', geometry.NODE_BUILD, str(Path(__file__).resolve().parents[2]),
                    str(output/'piper_feedback_viewer.js'), str(output/'piper_feedback_viewer.bundle.js')], check=True)
    bundle = (output/'piper_feedback_viewer.bundle.js').read_bytes()
    if len(bundle) > MAX_BUNDLE:
        raise hosting.PublicationError('Compiled browser-managed resource bundle exceeds 8 MiB.')
    html = geometry._once(original['index.html'].decode('utf8'), '<script src="piper_feedback_viewer.bundle.js" defer></script>',
        '<script src="piper_feedback_viewer.bundle.js" integrity="sha384-'+base64.b64encode(sha384(bundle).digest()).decode('ascii')+'" crossorigin="anonymous" referrerpolicy="no-referrer" defer></script>')
    html = geometry._once(html, '<head>', '<head>\n  <meta http-equiv="Content-Security-Policy" content="script-src \'self\'; object-src \'none\'; base-uri \'none\'">')
    (output/'index.html').write_text(html)
    note = dict(schema=SCHEMA, parent_viewer_integrity_sha256=expected_viewer_sha256,
        observed_transport='Original, chunked, text-first and native-buffered fetch-body strategies have each timed out during controlled public loads. Native buffering reached 22/26 chunks before the 120-second geometry deadline. They are not established fixes. Script/HTML browser-managed resources have loaded reliably in the observed sessions; the precise transport cause remains undiagnosed.',
        strategy='Embed the exact original gzip geometry, original scene/motion JSON and the first scheduled feedback replay in the compiled native script. Load each other case only on selection using one fixed same-origin generated literal data script with SHA384 SRI, anonymous CORS, no-referrer and a cancellable 30-second deadline. No runtime fetch-body reads are needed for playback.',
        security='Compiled main bundle has HTML SHA384 SRI. Case scripts have fixed filenames, at most 1 MiB source and manifest-bound SHA384 SRI. A restrictive native script-src self policy prevents external script targets, including redirects; no inline script is used. Each generated wrapper contains only literals delivered to one fixed callback; only an actively requested exact case identity can be accepted. Every payload checks exact keys, bounded canonical base64, compressed SHA256, bounded gzip expansion and unchanged original raw SHA256. Original whole-geometry and mesh/index bounds remain.',
        limits='SRI is enforced by the browser before a data script executes; this is immutable generated data delivery, not arbitrary policy loading. Download byte caps are declared and pinned by SRI rather than incrementally enforced on native script responses. Main bundle is below 8 MiB, each selected-case script below 1 MiB. Browser gzip support is required. No new hosts, authentication or telemetry. This is experimental until an independent controlled public load.',
        first_embedded_case=dict(id=first['id'], variant=first['variant']), replay_count=28, attempts=120,
        selected_case_scripts=len(scripts), largest_script_bytes=max(map(len, scripts.values())), bundle_bytes=len(bundle),
        original_geometry_sha256=scene['geometry_binary']['sha256'], original_geometry_bytes=scene['geometry_binary']['bytes'],
        changed_original_paths=sorted(CHANGED | ({'motion.json'} if dataset_revision is not None else set())), scientific_bytes_unchanged=True,
        native_dataset_revision=dataset_revision or 'main (explicitly unpinned experimental transport test)',
        original_motion_sha256=sha256(original['motion.json']).hexdigest(), derived_motion_sha256=sha256(motion_raw).hexdigest(),
        motion_publication_change='Only native evidence URLs and dataset publication metadata may be pinned to the supplied exact commit. Embedded and external motion JSON are byte-identical. The original scientific viewer and original motion remain sealed and untouched.',
        implementation_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('piper_feedback_native_resources.js'))})
    (output/'BROWSER-MANAGED-RESOURCES.json').write_bytes(hosting.study._encoded(note))
    seal = deepcopy(parent); seal['browser_managed_resources_derivation'] = note
    seal['files'] = {p.relative_to(output).as_posix():hosting._receipt(p.read_bytes()) for p in sorted(output.rglob('*'))
                     if p.is_file() and p.name != 'viewer-integrity.json'}
    (output/'viewer-integrity.json').write_bytes(hosting.study._encoded(seal))
    for name, raw in original.items():
        if name not in (CHANGED | ({'motion.json'} if dataset_revision is not None else set())) and (output/name).read_bytes() != raw:
            raise hosting.PublicationError('An original scientific or geometry file changed.')
    return dict(schema=SCHEMA, output=str(output), files=len(seal['files'])+1, bundle_bytes=len(bundle),
        selected_case_scripts=len(scripts), largest_script_bytes=max(map(len,scripts.values())),
        parent_viewer_integrity_sha256=expected_viewer_sha256,
        derived_viewer_integrity_sha256=sha256((output/'viewer-integrity.json').read_bytes()).hexdigest(),
        scientific_bytes_unchanged=True, remote_requests=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-viewer-sha256', required=True)
    parser.add_argument('--dataset-revision', help='Optional exact 40-hex dataset commit, pinned in both embedded and external motion.')
    args = parser.parse_args(argv)
    print(json.dumps(derive(args.viewer, args.output, expected_viewer_sha256=args.expected_viewer_sha256, dataset_revision=args.dataset_revision), indent=2))


if __name__ == '__main__': main()
