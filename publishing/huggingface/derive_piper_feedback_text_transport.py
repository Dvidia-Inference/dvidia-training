"""Offline, derived JSON/base64 transport for original geometry and pose bytes.

This is a hosting hypothesis, not an established public transport fix. Original
binary/gzip files remain available, and all decoded/whole-geometry SHA checks,
mesh bounds, case laziness, deadlines and origin restrictions are preserved.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import derive_piper_feedback_geometry_chunks as geometry
import publish_piper_feedback_hosting as hosting


SCHEMA = 'dvidia.piper-feedback-text-transport/v1'
ENVELOPE_SCHEMA = 'dvidia.binary-text-envelope/v1'
MAX_ENVELOPE = 1048576
MAX_RAW = 786432
CHANGED = {'scene.json', 'motion.json', 'piper_feedback_viewer.js', 'piper_feedback_viewer.bundle.js', 'viewer-integrity.json'}


def envelope(raw):
    if not 0 < len(raw) <= MAX_RAW:
        raise hosting.PublicationError('Raw bytes exceed the bounded text envelope size.')
    document = dict(schema=ENVELOPE_SCHEMA, encoding='base64', bytes=len(raw), sha256=sha256(raw).hexdigest(),
                    data=base64.b64encode(raw).decode('ascii'))
    packed = (json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')
    if len(packed) > MAX_ENVELOPE:
        raise hosting.PublicationError('Encoded JSON envelope exceeds 1 MiB.')
    return packed


def patched_source(raw):
    source = raw.decode('utf8')
    addon = Path(__file__).with_name('piper_feedback_text_transport.js').read_text()
    source = geometry._once(source, 'async function binary(meta, expected, outerSignal) {',
        addon+'\nasync function binary(meta, expected, outerSignal) {')
    source = geometry._once(source, "  integer(meta.bytes, 0, MAX_BYTES, 'binary byte count');",
        "  integer(meta.bytes, 0, MAX_BYTES, 'binary byte count');\n  if (preferTextTransport() && meta.json_envelope) {\n    try { return await envelopedBinary(meta, expected, outerSignal); }\n    catch (error) { if (error instanceof EnvelopeValidationError || outerSignal?.aborted) throw error; }\n  }")
    return source.encode('utf8')


def derive(viewer, output, *, expected_viewer_sha256):
    viewer, output = map(hosting.study._plain, (viewer, output))
    if output.exists():
        raise hosting.PublicationError('Text-transport projection must use a fresh output.')
    original, parent = hosting._source_viewer(viewer, expected_viewer_sha256)
    scene, motion = (json.loads(original[name]) for name in ('scene.json', 'motion.json'))
    pieces = scene.get('geometry_binary', {}).get('chunks')
    if (not isinstance(pieces, list) or len(pieces) != 26 or len(motion.get('cases', [])) != 28
            or len(motion.get('all_attempts', [])) != 120):
        raise hosting.PublicationError('Expected the complete chunked geometry and 28-case replay projection.')
    source = patched_source(original['piper_feedback_viewer.js'])
    descriptors = pieces+[case['pose_binary'] for case in motion['cases']]
    additions = {}
    for descriptor in descriptors:
        name = descriptor.get('href')
        if not isinstance(name, str) or name not in original or 'json_envelope' in descriptor:
            raise hosting.PublicationError('Raw piece identity changed or is already enveloped.')
        raw = original[name]
        if hosting._receipt(raw) != {key: descriptor[key] for key in ('bytes', 'sha256')}:
            raise hosting.PublicationError('Raw piece differs from its original size or digest.')
        packed = envelope(raw)
        additions[name+'.json'] = packed
        descriptor['json_envelope'] = dict(href=name+'.json', **hosting._receipt(packed))
    shutil.copytree(viewer, output)
    for name, packed in additions.items():
        (output/name).write_bytes(packed)
    (output/'scene.json').write_bytes(hosting.study._encoded(scene))
    (output/'motion.json').write_bytes(hosting.study._encoded(motion))
    (output/'piper_feedback_viewer.js').write_bytes(source)
    subprocess.run(['node', '--input-type=module', '-e', geometry.NODE_BUILD, str(Path(__file__).resolve().parents[2]),
                    str(output/'piper_feedback_viewer.js'), str(output/'piper_feedback_viewer.bundle.js')], check=True)
    note = dict(schema=SCHEMA, parent_viewer_integrity_sha256=expected_viewer_sha256,
        original_scientific_viewer_integrity_sha256=parent.get('geometry_transport_derivation', {}).get('parent_viewer_integrity_sha256'),
        observed_transport='Controlled public diagnostics observed HTTP200 geometry binary/gzip bodies stalling after partial transfer while scene.json and motion.json completed. No header, decompression or hash-stage failure was demonstrated in that load. JSON envelopes are an unproven hosting hypothesis until a separate public cold-load test.',
        envelopes=len(additions), max_envelope_bytes=MAX_ENVELOPE, max_decoded_bytes=MAX_RAW,
        preference='Same-origin JSON first only on the exact existing HTTPS research domain or existing HF Space viewer namespace. Local runs retain binary/gzip behavior; network-unavailable text requests may fall back to original binary/gzip. Malformed or mismatched envelopes fail closed.',
        verification='Canonical bounded JSON, exact schema/keys, exact base64 length/padding/canonical bits, envelope SHA, then unchanged decoded raw SHA; geometry retains its unchanged whole-file SHA and original mesh bounds.',
        preserved='All original raw/gzip files, scientific results, policy/model, states, evidence, deadlines, origin checks, case laziness and cancellation. No external hosts, credentials or telemetry added.',
        changed_original_paths=sorted(CHANGED),
        implementation_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('piper_feedback_text_transport.js'))})
    (output/'TEXT-TRANSPORT.json').write_bytes(hosting.study._encoded(note))
    seal=deepcopy(parent);seal['text_transport_derivation']=note
    seal['files']={p.relative_to(output).as_posix():hosting._receipt(p.read_bytes()) for p in sorted(output.rglob('*'))
                   if p.is_file() and p.name!='viewer-integrity.json'}
    (output/'viewer-integrity.json').write_bytes(hosting.study._encoded(seal))
    for name,raw in original.items():
        if name not in CHANGED and (output/name).read_bytes()!=raw:
            raise hosting.PublicationError('An immutable binary or scientific artifact changed.')
    for descriptor in descriptors:
        packed=additions[descriptor['json_envelope']['href']]
        decoded=base64.b64decode(json.loads(packed)['data'],validate=True)
        if decoded!=original[descriptor['href']]:
            raise hosting.PublicationError('Decoded text envelope differs from its original raw bytes.')
    return dict(schema=SCHEMA,output=str(output),envelopes=len(additions),
        largest_envelope_bytes=max(map(len,additions.values())),
        parent_viewer_integrity_sha256=expected_viewer_sha256,
        derived_viewer_integrity_sha256=sha256((output/'viewer-integrity.json').read_bytes()).hexdigest(),
        files=len(seal['files'])+1,scientific_binary_bytes_unchanged=True,remote_requests=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-viewer-sha256',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(derive(args.viewer,args.output,expected_viewer_sha256=args.expected_viewer_sha256),indent=2))


if __name__=='__main__':main()
