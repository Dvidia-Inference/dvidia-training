"""Fresh hosting projection with local request-phase/byte diagnostics only.

No telemetry is transmitted. Original and geometry-chunked projections remain
untouched; byte bounds, deadlines, SHA checks and origin rules are preserved.
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


SCHEMA = 'dvidia.piper-feedback-local-transport-diagnostics/v1'
CHANGED = {'index.html', 'piper_feedback_viewer.js', 'piper_feedback_viewer.bundle.js', 'viewer-integrity.json'}


def patched_source(raw):
    source = raw.decode('utf8')
    once = geometry._once
    addon = Path(__file__).with_name('piper_feedback_transport_diagnostics.js').read_text()
    source = once(source, 'function assert(condition, message)', addon+'\nfunction assert(condition, message)')
    source = once(source, 'async function response(href, signal) {', 'async function response(href, signal, expectedBytes) {\n  transportBegin(href, expectedBytes);')
    source = once(source, '  const final = new URL(result.url);', "  transportPhase(href, 'reading body', {status: Number.isInteger(result.status) ? result.status : null});\n  const final = new URL(result.url);")
    source = once(source, '${milliseconds / 1000} s. Check the connection', '${milliseconds / 1000} s.${transportSummary(href)} Check the connection')
    source = once(source, '  try { return await abortable(operation(controller.signal), controller.signal); }\n  finally {',
        '  try { return await abortable(operation(controller.signal), controller.signal); }\n  catch (error) { transportFailed(href); throw error; }\n  finally {')
    source = once(source, 'async function bounded(stream, limit, signal) {', 'async function bounded(stream, limit, signal, transportHref) {')
    source = once(source, '      size += part.value.byteLength;', '      size += part.value.byteLength; transportBytes(transportHref, size);')
    source = once(source, 'bounded((await response(href, signal)).body, MAX_BYTES, signal)',
        'bounded((await response(href, signal)).body, MAX_BYTES, signal, href)')
    source = once(source,
        '  return deadline(href, 30000, async signal => JSON.parse(new TextDecoder().decode(await bounded((await response(href, signal)).body, MAX_BYTES, signal, href))));',
        "  return deadline(href, 30000, async signal => {\n    const bytes = await bounded((await response(href, signal)).body, MAX_BYTES, signal, href);\n    const value = JSON.parse(new TextDecoder().decode(bytes));\n    transportPhase(href, 'complete');\n    return value;\n  });")
    source = once(source, 'async function verifiedBinary(bytes, meta) {',
        "async function verifiedBinary(bytes, meta, transportHref = meta.href) {\n  transportPhase(transportHref, 'verifying', {received: bytes.byteLength, expected: meta.bytes});")
    source = once(source, "  assert(hash === meta.sha256, 'Study binary SHA256 changed.');\n  return bytes;",
        "  assert(hash === meta.sha256, 'Study binary SHA256 changed.');\n  transportPhase(transportHref, 'complete');\n  return bytes;")
    source = once(source, 'bounded((await response(meta.compressed_href, signal)).body, meta.compressed_bytes, signal)',
        'bounded((await response(meta.compressed_href, signal, meta.compressed_bytes)).body, meta.compressed_bytes, signal, meta.compressed_href)')
    source = once(source, "        const bytes = await bounded(new Blob([compressed])", "        transportPhase(meta.compressed_href, 'decompressing', {received: 0, expected: meta.bytes});\n        const bytes = await bounded(new Blob([compressed])")
    source = once(source, ".pipeThrough(new DecompressionStream('gzip'), {signal}), meta.bytes, signal);",
        ".pipeThrough(new DecompressionStream('gzip'), {signal}), meta.bytes, signal, meta.compressed_href);")
    source = once(source, '        return verifiedBinary(bytes, meta);', '        return verifiedBinary(bytes, meta, meta.compressed_href);')
    # Two identical raw/verified-JSON read sites intentionally share this patch.
    old = 'bounded((await response(meta.href, signal)).body, meta.bytes, signal)'
    if source.count(old) != 2:
        raise hosting.PublicationError('Source raw/JSON transfer sites differ from the reviewed diagnostic patch.')
    source = source.replace(old, 'bounded((await response(meta.href, signal, meta.bytes)).body, meta.bytes, signal, meta.href)')
    return source.encode('utf8')


def derive(viewer, output, *, expected_viewer_sha256):
    viewer, output = map(hosting.study._plain, (viewer, output))
    if output.exists():
        raise hosting.PublicationError('Diagnostic projection must use a fresh output.')
    original, parent = hosting._source_viewer(viewer, expected_viewer_sha256)
    source = patched_source(original['piper_feedback_viewer.js'])
    html = original['index.html'].decode('utf8')
    html = geometry._once(html, '      <details class="study-details">',
        '      <details class="study-details"><summary>Transfer details</summary><p>Local request state only. No diagnostics are sent.</p><pre id="transport-details" style="white-space:pre-wrap;overflow-wrap:anywhere">Waiting for saved assets…</pre></details>\n\n      <details class="study-details">')
    shutil.copytree(viewer, output)
    (output/'piper_feedback_viewer.js').write_bytes(source)
    (output/'index.html').write_text(html)
    subprocess.run(['node', '--input-type=module', '-e', geometry.NODE_BUILD, str(Path(__file__).resolve().parents[2]),
                    str(output/'piper_feedback_viewer.js'), str(output/'piper_feedback_viewer.bundle.js')], check=True)
    note = dict(schema=SCHEMA, parent_viewer_integrity_sha256=expected_viewer_sha256,
        original_scientific_viewer_integrity_sha256=parent.get('geometry_transport_derivation', {}).get('parent_viewer_integrity_sha256'),
        observed_transport='The original large transfer timed out once and an intentional reload succeeded. The first chunked public cold load also timed out on geometry-part-006.bin after 30 seconds. Chunking alone has not established a public transport fix; exact network/server cause remains unmeasured.',
        change='Display local relative asset name, request phase, HTTP status and received/expected bytes. Include the observed stalled phase/bytes in timeout errors. No cookies, authentication, signed URLs or network telemetry are recorded or transmitted.',
        preserved='All deadlines, cancellation, SHA/byte bounds and exact origin restrictions; every scientific/model/state/geometry file unchanged.',
        changed_original_paths=sorted(CHANGED),
        implementation_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('piper_feedback_transport_diagnostics.js'))})
    (output/'TRANSPORT-DIAGNOSTICS.json').write_bytes(hosting.study._encoded(note))
    seal=deepcopy(parent);seal['transport_diagnostics_derivation']=note
    seal['files']={p.relative_to(output).as_posix():hosting._receipt(p.read_bytes()) for p in sorted(output.rglob('*'))
                   if p.is_file() and p.name!='viewer-integrity.json'}
    (output/'viewer-integrity.json').write_bytes(hosting.study._encoded(seal))
    for name,raw in original.items():
        if name not in CHANGED and (output/name).read_bytes()!=raw:
            raise hosting.PublicationError('A preserved scientific or geometry artifact changed.')
    return dict(schema=SCHEMA,output=str(output),parent_viewer_integrity_sha256=expected_viewer_sha256,
        derived_viewer_integrity_sha256=sha256((output/'viewer-integrity.json').read_bytes()).hexdigest(),
        files=len(seal['files'])+1,scientific_evidence_unchanged=True,remote_requests=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-viewer-sha256',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(derive(args.viewer,args.output,expected_viewer_sha256=args.expected_viewer_sha256),indent=2))


if __name__=='__main__':main()
