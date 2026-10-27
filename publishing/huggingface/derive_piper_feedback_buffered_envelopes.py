"""Offline compatibility experiment using native buffering for known text envelopes.

Native buffering is not a streaming memory cap: absent a Content-Length header,
the browser may allocate a changed body before the post-read size check rejects
it. This explicit hosting experiment preserves original bytes and verification;
it does not claim to diagnose or resolve the observed public partial-body stalls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import derive_piper_feedback_geometry_chunks as geometry
import publish_piper_feedback_hosting as hosting


SCHEMA = 'dvidia.piper-feedback-native-buffered-envelopes/v1'
CHANGED = {'piper_feedback_viewer.js', 'piper_feedback_viewer.bundle.js', 'viewer-integrity.json'}


def patched_source(raw):
    source = raw.decode('utf8')
    addon = Path(__file__).with_name('piper_feedback_buffered_envelopes.js').read_text()
    source = geometry._once(source, 'async function envelopedBinary(meta, expected, outerSignal) {',
        addon+'\nasync function envelopedBinary(meta, expected, outerSignal) {')
    source = geometry._once(source,
        '    const packed = await bounded((await response(descriptor.href, signal, descriptor.bytes)).body, descriptor.bytes, signal, descriptor.href);',
        '    const result = await response(descriptor.href, signal, descriptor.bytes);\n    const packed = await nativeBufferedEnvelope(result, descriptor, signal);')
    return source.encode('utf8')


def derive(viewer, output, *, expected_viewer_sha256):
    viewer, output = map(hosting.study._plain, (viewer, output))
    if output.exists():
        raise hosting.PublicationError('Native-buffering experiment must use a fresh output.')
    original, parent = hosting._source_viewer(viewer, expected_viewer_sha256)
    if parent.get('text_transport_derivation', {}).get('schema') != 'dvidia.piper-feedback-text-transport/v1':
        raise hosting.PublicationError('Expected the original sealed text-transport projection.')
    source = patched_source(original['piper_feedback_viewer.js'])
    shutil.copytree(viewer, output)
    (output/'piper_feedback_viewer.js').write_bytes(source)
    subprocess.run(['node', '--input-type=module', '-e', geometry.NODE_BUILD, str(Path(__file__).resolve().parents[2]),
                    str(output/'piper_feedback_viewer.js'), str(output/'piper_feedback_viewer.bundle.js')], check=True)
    note = dict(schema=SCHEMA, parent_viewer_integrity_sha256=expected_viewer_sha256,
        original_scientific_viewer_integrity_sha256=parent.get('text_transport_derivation', {}).get('original_scientific_viewer_integrity_sha256'),
        observed_transport='A controlled text-first public load also stalled during HTTP 200 envelope bodies: parts 014/015 reached 122508/349693 bytes and part 016 reached 56972/349693 bytes at 14/26 completion; earlier parts 009–013 timed out partway before existing gzip fallback completed. JSON encoding alone did not establish a fix. No header/decompression/hash cause was demonstrated.',
        code_review='The existing reader loop serially awaits reader.read(); it does not cancel successful streams. On failure or deadline its finally block cancels the unfinished reader without awaiting cancellation. This inspection alone does not establish a browser, reader-loop or server cause.',
        strategy='native-buffered-json/v1: only the existing known same-origin JSON envelopes use Response.arrayBuffer(). Fetch redirects, origin restrictions, 30-second request and 120-second whole-geometry deadlines, shared cancellation, exact post-read byte count, envelope and decoded raw SHA checks, base64 validation and original whole-geometry/mesh checks remain. Other metadata/binary/gzip reads are unchanged.',
        bounds='Known expected envelope sizes remain at most 1 MiB. Reject malformed, zero, unsafe, or above-expected Content-Length before native buffering. Content-Length may be absent or describe compressed wire bytes; exact decoded size is checked after buffering. Native buffering cannot enforce a streaming allocation cap on an absent or dishonest Content-Length before the body completes; this is an explicit compatibility experiment, not a claim of identical streaming memory behavior.',
        diagnostics='Local native body buffering phase reports 0/expected bytes until the native body finishes, then actual bytes. It cannot report incremental body bytes. No telemetry, credentials or external hosts are added.',
        status='Unproven transport hypothesis; requires one separate controlled public cold-load test. Existing fallback behavior is preserved, with no retry-until-success rule.',
        changed_original_paths=sorted(CHANGED),
        implementation_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(__file__).with_name('piper_feedback_buffered_envelopes.js'))})
    (output/'NATIVE-BUFFERED-ENVELOPES.json').write_bytes(hosting.study._encoded(note))
    seal = deepcopy(parent); seal['native_buffered_envelopes_derivation'] = note
    seal['files'] = {p.relative_to(output).as_posix():hosting._receipt(p.read_bytes()) for p in sorted(output.rglob('*'))
                     if p.is_file() and p.name != 'viewer-integrity.json'}
    (output/'viewer-integrity.json').write_bytes(hosting.study._encoded(seal))
    for name, raw in original.items():
        if name not in CHANGED and (output/name).read_bytes() != raw:
            raise hosting.PublicationError('A preserved scientific, envelope or geometry artifact changed.')
    return dict(schema=SCHEMA, output=str(output), files=len(seal['files'])+1,
        parent_viewer_integrity_sha256=expected_viewer_sha256,
        derived_viewer_integrity_sha256=sha256((output/'viewer-integrity.json').read_bytes()).hexdigest(),
        original_data_and_envelopes_unchanged=True, remote_requests=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-viewer-sha256', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(derive(args.viewer, args.output, expected_viewer_sha256=args.expected_viewer_sha256), indent=2))


if __name__ == '__main__': main()
