"""Export the complete frozen v0.3 clock study as a lean recorded 3D replay.

Verification recomputes stored numeric scores and source bindings, but does not
step physics, run a policy or fit a model. Rendering poses use recorded native
states and forward kinematics only. Full episodes and 1 ms arrays stay in the
separately published, exact-commit-pinned dataset.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import gzip
from hashlib import sha256, sha384
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

from . import piper_grip_clock_cli as lifecycle
from . import piper_grip_clock_benchmark as runner
from . import piper_grip_clock_model as clock_model
from . import piper_imitation as core
from . import reference_sim as sim
from .piper_imitation_viewer import _inside, _json, _poses
from .reference_replay import _decode, _read, _scene, MIT_LICENSE


SCHEMA = 'dvidia.piper-grip-clock-viewer/v0.3'
SCENE_SCHEMA = 'dvidia.piper-grip-clock-scene/v0.3'
RESOURCE_SCHEMA = 'dvidia.piper-clock-native-resources/v1'
PAYLOAD_SCHEMA = 'dvidia.piper-clock-resource-data/v1'
CALLBACK = '__dvidiaPiperClockResourcesV1'
DATASET_REPO = 'Dvidia/piper-grip-clock-demonstrations-v03'
DATASET_ROOT = 'https://huggingface.co/datasets/'+DATASET_REPO+'/resolve/'
VARIANTS = ('clock', 'blind_clock', 'fixed', 'teacher')
CONDITIONS = ('nominal', 'post_grip')
MAX_REPLAYS = 320
MAX_FILE = 8*1024*1024
MAX_SCRIPT = 1024*1024
MAX_PACKED_RESOURCE = 4*1024*1024
MAX_NAMESPACE = 192*1024*1024
MAX_JSON = 64*1024*1024
MAX_EPISODE_PACKED = 32*1024*1024
MAX_EPISODE = 128*1024*1024
MAX_NATIVE = 64*1024*1024
ATTEMPT_FILES = {'started.json', 'episode.json.gz', 'native.npz', 'native.json', 'score.json', 'post-grip.json'}
INPUT_DOCUMENTS = ('protocol.json', 'frozen.json', 'report.json', 'base-model.json', 'jaw-model.json', 'selected.json', 'blind.json')
RESOURCE_NAME = re.compile(r'(?:geometry\.bin|scene\.json|motion\.json|frames-[0-9]{3}\.json|poses-[0-9]{3}\.bin)\Z')
CASE_ID = re.compile(r'confirmation-xy[0-9]{2}-g[01]-(?:nominal|post_grip)\Z')
ORIGINS = dict(clock='learned_grip_clock', blind_clock='learned_blind_clock',
               fixed='fixed_unit_progress', teacher='authored_causal_grip_teacher')
SCOPE = ('Recorded synthetic PiPER hold/advance-clock simulation. All seven original primitive arm/jaw '
         'references share learned virtual progress; no new path geometry or jaw policy is learned. '
         'Clock sensing is privileged native state. The separately fitted blind clock masks its live '
         'grip/history inputs while retaining context, clocks and the same native command limits. '
         'Native recorded poses use forward kinematics only: no browser physics, live inference, '
         'human-video learning or physical robot qualification.')


def _plain(path):
    path = Path(path).absolute()
    if any(part.is_symlink() and part not in (Path('/var'), Path('/tmp')) for part in (path, *path.parents)):
        raise ValueError('Clock replay paths may not traverse artifact-owned symlinks.')
    return path.resolve()


def _hash(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _revision(value):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{40}', value) is None:
        raise ValueError('Clock replay dataset revision must be an exact lowercase 40-hex commit.')
    return value


def _receipt(raw):
    return dict(bytes=len(raw), sha256=sha256(raw).hexdigest())


def resource_id(index):
    if type(index) is not int or not 0 <= index < MAX_REPLAYS:
        raise ValueError('Replay resource index exceeds the literal 0..319 bound.')
    return f'replay-{index:03}'


def _native_href(revision, case_id, variant, name):
    _revision(revision)
    if (type(case_id) is not str or CASE_ID.fullmatch(case_id) is None
            or variant not in VARIANTS or name not in ATTEMPT_FILES|{'receipt.json'}):
        raise ValueError('Clock evidence URL has an unexpected case, variant or file.')
    return DATASET_ROOT+revision+'/study/confirmation/'+case_id+'/'+variant+'/'+name


def validate_evidence_href(href, revision, case_id, variant, name):
    if type(href) is not str or href != _native_href(revision, case_id, variant, name):
        raise ValueError('Clock evidence must use the exact pinned public dataset path.')
    return href


def _verified_study(root):
    """Verify an already-complete report using the frozen numeric lifecycle.

    The existing lifecycle's private report function rechecks all 512 attempt
    receipts, independent numeric scores, pre-confirmation files and source
    snapshots. It is allowed here only when report.json already exists; the
    before/after byte equality ensures export never creates or changes science.
    """
    root = _plain(root)
    if not root.is_dir():
        raise ValueError('Clock replay requires an existing completed study directory.')
    raw = {name: _read(_inside(root, name), MAX_JSON) for name in INPUT_DOCUMENTS}
    protocol, frozen, report = (_decode(raw[name]) for name in INPUT_DOCUMENTS[:3])
    if any(type(value) is not dict or value.get('schema') != runner.SCHEMA for value in (protocol, frozen, report)):
        raise ValueError('Clock replay requires complete frozen v0.3 study documents.')
    if (protocol.get('physical_robot_ready') is not False or frozen.get('physical_robot_ready') is not False
            or report.get('physical_robot_ready') is not False or report.get('qualified_robot_skill') is not False
            or any(value.get('jaw_mode') != 'coordinated_reference' for value in (protocol, frozen, report))):
        raise ValueError('This viewer requires the declared simulation-only coordinated-reference architecture.')
    protocol_sha, frozen_sha = sha256(raw['protocol.json']).hexdigest(), sha256(raw['frozen.json']).hexdigest()
    if (frozen.get('protocol_sha256') != protocol_sha or report.get('protocol_sha256') != protocol_sha
            or report.get('frozen_sha256') != frozen_sha):
        raise ValueError('Clock report/protocol/frozen source hashes disagree.')
    sources, runtime = lifecycle.source_hashes(), runner.runtime_hashes()
    if (protocol.get('source_sha256') != sources or frozen.get('lifecycle_source_sha256') != sources
            or frozen.get('runtime_sha256') != runtime):
        raise ValueError('Clock replay source differs from the exact frozen lifecycle/runtime.')
    if any(_receipt(_read(_inside(root, 'sources/'+name), MAX_FILE))['sha256'] != digest
           for name, digest in sources.items()):
        raise ValueError('A frozen clock-study source snapshot changed.')
    for name, key in (('base-model.json', 'base_model_raw_sha256'), ('jaw-model.json', 'jaw_model_raw_sha256'),
                      ('selected.json', 'selected_sha256'), ('blind.json', 'blind_sha256')):
        if not _hash(frozen.get(key)) or sha256(raw[name]).hexdigest() != frozen[key]:
            raise ValueError('A frozen clock/base/jaw model or manifest changed.')
    cases = protocol.get('cases', {}).get('confirmation') if type(protocol.get('cases')) is dict else None
    if type(cases) is not list or len(cases) != 128 or protocol.get('variants') != list(VARIANTS):
        raise ValueError('Clock replay requires the full 128-case/four-variant confirmation schedule.')
    ids = [case.get('id') if type(case) is dict else None for case in cases]
    if (any(type(case_id) is not str or CASE_ID.fullmatch(case_id) is None for case_id in ids)
            or len(set(ids)) != 128 or any(case.get('condition') not in CONDITIONS for case in cases)):
        raise ValueError('Clock confirmation cases are malformed or duplicated.')
    expected = [(case['id'], case['condition'], variant) for case in cases for variant in VARIANTS]
    rows = report.get('rows')
    if type(rows) is not list or len(rows) != 512:
        raise ValueError('Clock replay requires all 512 scheduled outcomes, including failures.')
    actual = []
    for row in rows:
        score = row.get('score') if type(row) is dict else None
        if type(score) is not dict or type(score.get('valid')) is not bool or type(score.get('success')) is not bool:
            raise ValueError('Clock report has a malformed recorded score.')
        actual.append((row.get('case_id'), row.get('condition'), row.get('variant')))
    if actual != expected:
        raise ValueError('Clock report must retain every scheduled outcome in exact order.')
    if type(report.get('variants')) is not dict or set(report['variants']) != set(VARIANTS):
        raise ValueError('Clock report lacks all four complete variant summaries.')
    for variant in VARIANTS:
        subset = [row for row in rows if row['variant'] == variant]
        summary = report['variants'][variant]
        if (type(summary) is not dict or summary.get('scheduled') != 128
                or summary.get('valid') != sum(row['score']['valid'] for row in subset)
                or summary.get('strict_successes') != sum(row['score']['valid'] and row['score']['success'] for row in subset)):
            raise ValueError('Clock report aggregate differs from complete scheduled outcomes.')
    # This only recomputes stored numerical telemetry; never runner.execute(),
    # lifecycle.confirm(), native mj_step(), inference or teacher construction.
    recomputed = lifecycle._report(root)
    if recomputed != report or any(_read(_inside(root, name), MAX_JSON) != value for name, value in raw.items()):
        raise ValueError('Clock study changed or disagreed with independent numeric report verification.')
    return raw, protocol, frozen, report


def selected_replays(rows):
    """All 256 learned rows plus first fixed/teacher, without outcome filtering."""
    selected, controls = [], set()
    for row in rows:
        if row['variant'] in ('clock', 'blind_clock') or row['variant'] not in controls:
            selected.append(row)
            controls.add(row['variant'])
    if len(selected) != 258 or len({(row['case_id'], row['variant']) for row in selected}) != 258:
        raise ValueError('Clock replay roster must retain 258 outcome-independent attempts.')
    return selected


def _unpack(raw):
    try:
        with gzip.GzipFile(fileobj=BytesIO(raw)) as stream:
            body = stream.read(MAX_EPISODE+1)
    except (OSError, EOFError) as error:
        raise ValueError('Clock episode is not valid bounded gzip.') from error
    if len(body) > MAX_EPISODE:
        raise ValueError('Clock episode exceeds the decoded byte bound.')
    return _decode(body)


def _attempt(root, row, case, frozen, revision, *, decode_episode):
    prefix = 'confirmation/'+case['id']+'/'+row['variant']
    folder = _inside(root, prefix)
    if not folder.is_dir() or {path.name for path in folder.iterdir()} != ATTEMPT_FILES|{'receipt.json'}:
        raise ValueError('Clock confirmation attempt has an incomplete or foreign file inventory.')
    receipt_raw = _read(_inside(root, prefix+'/receipt.json'), MAX_JSON)
    receipt = _decode(receipt_raw)
    binding = receipt.get('binding') if type(receipt) is dict else None
    expected_model = frozen['selected_sha256'] if row['variant'] == 'clock' else frozen['blind_sha256'] if row['variant'] == 'blind_clock' else None
    if (sha256(receipt_raw).hexdigest() != row.get('receipt_sha256')
            or receipt.get('case_id') != case['id'] or receipt.get('variant') != row['variant']
            or type(receipt.get('files')) is not dict or set(receipt['files']) != ATTEMPT_FILES
            or type(binding) is not dict or binding.get('case') != case
            or binding.get('schema') != runner.SCHEMA or binding.get('split') != 'confirmation'
            or binding.get('variant') != row['variant'] or binding.get('collect_labels') is not False
            or binding.get('jaw_mode') != 'coordinated_reference'
            or binding.get('protocol_sha256') != frozen['protocol_sha256']
            or binding.get('runtime_sha256') != frozen['runtime_sha256']
            or binding.get('model_raw_sha256') != expected_model
            or binding.get('base_model_raw_sha256') != frozen['base_model_raw_sha256']
            or binding.get('jaw_model_raw_sha256') != frozen['jaw_model_raw_sha256']):
        raise ValueError('Clock attempt differs from its exact frozen source/model/request identity.')
    evidence, inputs, episode = {}, {}, None
    for name in sorted(ATTEMPT_FILES):
        limit = MAX_NATIVE if name == 'native.npz' else MAX_EPISODE_PACKED if name.endswith('.gz') else MAX_JSON
        raw = _read(_inside(root, prefix+'/'+name), limit)
        meta = _receipt(raw)
        if not _hash(receipt['files'].get(name)) or meta['sha256'] != receipt['files'][name]:
            raise ValueError('An original clock confirmation file differs from its receipt.')
        inputs[prefix+'/'+name] = meta
        evidence[name] = dict(href=_native_href(revision, case['id'], row['variant'], name), **meta)
        if name == 'started.json' and _decode(raw) != binding:
            raise ValueError('Clock started marker differs from its original request binding.')
        if name == 'score.json' and _decode(raw) != row['score']:
            raise ValueError('Clock score bytes differ from the complete report row.')
        if name == 'post-grip.json' and _decode(raw) != row['score'].get('post_grip_analysis'):
            raise ValueError('Clock intervention analysis differs from the scored result.')
        if name == 'episode.json.gz' and decode_episode:
            episode = _unpack(raw)
    inputs[prefix+'/receipt.json'] = _receipt(receipt_raw)
    evidence['receipt.json'] = dict(href=_native_href(revision, case['id'], row['variant'], 'receipt.json'), **_receipt(receipt_raw))
    if episode is not None:
        if (type(episode) is not dict or episode.get('schema') != runner.EPISODE_SCHEMA or episode.get('case_id') != case['id']
                or episode.get('policy_kind') != row['variant'] or episode.get('condition') != case['condition']
                or episode.get('seed') != case['seed'] or episode.get('jaw_mode') != 'coordinated_reference'
                or episode.get('physical_robot_ready') is not False
                or episode.get('implementation_sha256') != frozen['runtime_sha256']
                or any(episode.get(key) != binding.get(key) for key in ('model_sha256', 'base_model_sha256', 'jaw_model_sha256'))):
            raise ValueError('Clock replay episode differs from its original frozen identity.')
    return evidence, inputs, episode


def _clock_rows(episode, rows, variant):
    states, actions = episode.get('trajectory'), episode.get('actions')
    if (type(states) is not list or type(actions) is not list or len(actions) > 1000
            or len(rows) != len(states)
            or not (len(states) == len(actions)+1
                    or len(states) == len(actions) and episode.get('valid') is False)):
        raise ValueError('Clock replay requires the complete causal action/state prefix.')
    encoder = clock_model.ClockFeatures(blind=variant == 'blind_clock')
    progress = 0.
    for index, (row, state) in enumerate(zip(rows, states)):
        if index < len(actions):
            action = actions[index]
            decision = action.get('clock_decision') if type(action) is dict else None
            if type(decision) is not dict or decision.get('decision_origin') != ORIGINS[variant]:
                raise ValueError('Clock frame lacks its exact variant-specific pre-action decision.')
            elapsed = core._number(action.get('t_seconds'), 'action time', 0., 20.)
            physical = core._number(decision.get('physical_time_s'), 'physical time', 0., 20.)
            before = core._number(decision.get('progress_before_s'), 'virtual progress before', 0., 20.)
            after = core._number(decision.get('progress_after_s'), 'virtual progress after', 0., 20.)
            increment = core._number(decision.get('increment_s'), 'clock increment', 0., .02)
            if (increment not in (0., .02) or abs(elapsed-index*.02) > 1e-8
                    or abs(elapsed-state['t_seconds']) > 1e-8 or abs(physical-elapsed) > 1e-8
                    or abs(before-progress) > 1e-8 or abs(after-before-increment) > 1e-8
                    or decision.get('jaw_mode') != 'coordinated_reference'):
                raise ValueError('Clock decision time, causal progress or architecture changed.')
            nominal_jaw = core._number(decision.get('nominal_physical_jaw_m'), 'physical nominal jaw', 0., core.JAW_MAX_M)
            feature = encoder(state, {'gripper_position_m': nominal_jaw}, before)
            recorded = core._array(decision.get('features'), (58,), 'recorded clock feature', 10.)
            if not np.array_equal(feature, recorded):
                raise ValueError('Clock features differ from exact causal saved measurements.')
            requested_jaw = core._number(action.get('gripper_position_m'), 'issued jaw', 0., core.JAW_MAX_M)
            if (decision.get('frozen_jaw_target_m') != requested_jaw or state.get('command') != action
                    or state.get('issued_gripper_position_m') != requested_jaw
                    or state.get('issued_joint_targets_rad') != action.get('joint_targets_rad')):
                raise ValueError('Clock decision does not describe the command issued from this saved state.')
            probability = (core._number(decision.get('advance_probability'), 'advance probability', 0., 1.)
                           if variant in ('clock', 'blind_clock') else None)
            if probability is not None and (probability > .5) != (increment == .02):
                raise ValueError('Clock decision differs from its recorded classification probability.')
            secure = decision.get('secure_grip') if variant == 'teacher' else None
            if variant == 'teacher' and type(secure) is not bool:
                raise ValueError('Teacher frame lacks its authored readiness label.')
            next_decision = dict(increment_s=increment, progress_after_s=after,
                                 origin=ORIGINS[variant], advance_probability=probability, secure_grip=secure)
            current_progress, progress = before, after
        else:
            if any(state.get(key) is not None for key in ('command', 'issued_gripper_position_m', 'issued_joint_targets_rad')):
                raise ValueError('Terminal clock state must not fabricate an issued action.')
            current_progress, next_decision = progress, None
        debt = state['t_seconds']-current_progress
        if debt < -1e-8:
            raise ValueError('Recorded path progress uses future physical time.')
        row.update(grasp_contact_counts=deepcopy(state.get('grasp_contact_counts')),
                   pad_normal_forces_n=deepcopy(state.get('pad_normal_forces_n')),
                   table_object_contact_count=state.get('table_object_contact_count'),
                   native_actuator_controls=deepcopy(state.get('native_actuator_controls')),
                   virtual_progress_s=current_progress, clock_debt_s=max(0., debt), next_clock_decision=next_decision)
    return rows


def resource(name, raw, packed=None):
    if type(name) is not str or RESOURCE_NAME.fullmatch(name) is None:
        raise ValueError('Unexpected clock replay resource name.')
    if name.startswith(('poses-', 'frames-')):
        resource_id(int(name[6:9] if name.startswith('poses-') else name[7:10]))
    if not isinstance(raw, bytes) or len(raw) > MAX_FILE or (not raw and not name.startswith('poses-')):
        raise ValueError('Clock replay resource exceeds its raw byte bound.')
    packed = gzip.compress(raw, mtime=0) if packed is None else packed
    if not isinstance(packed, bytes) or not 0 < len(packed) <= MAX_PACKED_RESOURCE:
        raise ValueError('Clock replay resource exceeds its compressed byte bound.')
    try:
        with gzip.GzipFile(fileobj=BytesIO(packed)) as stream:
            expanded = stream.read(len(raw)+1)
    except (OSError, EOFError) as error:
        raise ValueError('Clock replay resource has invalid gzip bytes.') from error
    if expanded != raw:
        raise ValueError('Clock replay resource differs from its preserved raw bytes.')
    return dict(href=name, **_receipt(raw), compression='gzip', compressed_bytes=len(packed),
                compressed_sha256=sha256(packed).hexdigest(), data=base64.b64encode(packed).decode('ascii'))


def payload(index, frames, poses):
    resource_id(index)
    if not isinstance(frames, bytes) or not isinstance(poses, bytes):
        raise ValueError('Replay payload resources must be exact bytes.')
    return dict(schema=PAYLOAD_SCHEMA, id=resource_id(index), resources=[
        resource(f'frames-{index:03}.json', frames), resource(f'poses-{index:03}.bin', poses)])


def script(value):
    if (type(value) is not dict or set(value) != {'schema', 'id', 'resources'}
            or value.get('schema') != PAYLOAD_SCHEMA or type(value.get('id')) is not str
            or re.fullmatch(r'replay-[0-9]{3}', value['id']) is None
            or value['id'] != resource_id(int(value['id'][-3:]))
            or type(value.get('resources')) is not list or len(value['resources']) != 2):
        raise ValueError('Unexpected literal clock data wrapper identity.')
    index = int(value['id'][-3:])
    for resource_meta, expected in zip(value['resources'], (f'frames-{index:03}.json', f'poses-{index:03}.bin')):
        if (type(resource_meta) is not dict
                or set(resource_meta) != {'href', 'bytes', 'sha256', 'compression', 'compressed_bytes', 'compressed_sha256', 'data'}
                or resource_meta.get('href') != expected or resource_meta.get('compression') != 'gzip'
                or type(resource_meta.get('bytes')) is not int or not 0 <= resource_meta['bytes'] <= MAX_FILE
                or resource_meta['bytes'] == 0 and not expected.startswith('poses-')
                or type(resource_meta.get('compressed_bytes')) is not int or not 0 < resource_meta['compressed_bytes'] <= MAX_PACKED_RESOURCE
                or not _hash(resource_meta.get('sha256')) or not _hash(resource_meta.get('compressed_sha256'))
                or type(resource_meta.get('data')) is not str
                or len(resource_meta['data']) != 4*((resource_meta['compressed_bytes']+2)//3)):
            raise ValueError('Unexpected clock script resource fields, bounds or identity.')
        try:
            compressed = base64.b64decode(resource_meta['data'], validate=True)
            with gzip.GzipFile(fileobj=BytesIO(compressed)) as stream:
                expanded = stream.read(resource_meta['bytes']+1)
        except (ValueError, OSError, EOFError) as error:
            raise ValueError('Clock script has invalid bounded base64/gzip resources.') from error
        if (base64.b64encode(compressed).decode('ascii') != resource_meta['data']
                or len(compressed) != resource_meta['compressed_bytes']
                or sha256(compressed).hexdigest() != resource_meta['compressed_sha256']
                or len(expanded) != resource_meta['bytes'] or sha256(expanded).hexdigest() != resource_meta['sha256']):
            raise ValueError('Clock script resources differ from their exact byte/hash receipts.')
    raw = b'globalThis['+core.canonical(CALLBACK)+b']('+core.canonical(value['id'])+b','+core.canonical(value)+b');\n'
    if len(raw) >= MAX_SCRIPT:
        raise ValueError('Clock selected-case script must remain below 1 MiB.')
    prefix = b'globalThis['+core.canonical(CALLBACK)+b']('+core.canonical(value['id'])+b','
    if not raw.startswith(prefix) or _decode(raw[len(prefix):-3]) != value:
        raise ValueError('Clock literal wrapper grammar changed.')
    return raw


def _write(root, name, raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_FILE:
        raise ValueError('Clock viewer file exceeds the 8 MiB public bound.')
    path = _inside(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def _finalize(root, manifest, motion, scene, scientific_sources, source, residuals):
    source_root = Path(__file__).parent
    template = _read(source_root/'piper_grip_clock_viewer.js', MAX_FILE)
    token = b'__DVIDIA_CLOCK_RESOURCE_MANIFEST__'
    if template.count(token) != 1:
        raise ValueError('Clock viewer template requires exactly one resource-manifest token.')
    manifest_raw = _json(manifest)
    _write(root, 'replay-resources.json', manifest_raw)
    generated = template.replace(token, core.canonical(manifest))
    # The expanded generated source is a private build intermediate. Preserve
    # the small source template and exact manifest in the sealed public export.
    with tempfile.TemporaryDirectory(prefix='.dvidia-clock-build-', dir=root.parent) as temporary:
        entry = Path(temporary)/'clock.js'
        entry.write_bytes(generated)
        build_script = Path(__file__).resolve().parents[2]/'scripts'/'build-piper-grip-clock-viewer.mjs'
        subprocess.run(['node', str(build_script), str(entry), str(root/'piper_grip_clock_viewer.bundle.js')], check=True)
    bundle = _read(root/'piper_grip_clock_viewer.bundle.js', MAX_FILE)
    html = _read(source_root/'piper_grip_clock_viewer.html', MAX_FILE)
    marker = b'<script src="piper_grip_clock_viewer.bundle.js" defer></script>'
    if html.count(marker) != 1:
        raise ValueError('Clock HTML requires its one exact native script marker.')
    script_tag = ('<script src="piper_grip_clock_viewer.bundle.js?sha256='+sha256(bundle).hexdigest()
                  +'" integrity="sha384-'+base64.b64encode(sha384(bundle).digest()).decode('ascii')
                  +'" crossorigin="anonymous" referrerpolicy="no-referrer" defer></script>').encode()
    html = html.replace(marker, script_tag)
    if b"script-src 'self'" not in html or b"object-src 'none'" not in html:
        raise ValueError('Clock HTML must retain the native self-only script policy.')
    _write(root, 'index.html', html)
    _write(root, 'piper_grip_clock_viewer.js', template)
    _write(root, 'motion.json', _json(motion))
    _write(root, 'scene.json', _json(scene))
    _write(root, 'scientific-sources.json', _json(scientific_sources))
    _write(root, 'README.md', ('# PiPER learned grip clock replay v0.3\n\n'+SCOPE+'\n\n'
        'Serve this complete directory over HTTP; open index.html. Playback uses native browser-managed '
        'scripts and exact hash-checked gzip resources, with only one case loaded on demand. The first '
        'scheduled clock attempt is embedded, including any failure. All 258 selected replays and all '
        '512 outcomes are retained. Full 20 ms episodes and 1 ms arrays are external exact-commit-pinned '
        'dataset evidence; the full study is not duplicated in this viewer.\n\n'
        'Physical time controls playback. Path progress controls the seven original primitive references. '
        'Next command is the decision issued from the currently displayed pre-action state; terminal '
        'states issue no new command. Clock output probability is not a hardware safety confidence. '
        'The blind clock ablates its grip/history inputs only. Native joint command caps and context remain.\n\n'
        'The exporter rechecks frozen sources, exact requests, full outcome counts and numeric score '
        'reconstruction before native forward-kinematics pose export. Checksums establish artifact '
        'consistency, not authenticity or physical accuracy. Publication verification is separate; '
        'consult the publication receipt for verified upload commits and anonymous readback.\n\n'
        'MIT source and upstream PiPER/Three.js notices are included. replay-resources.json plus the '
        'small JS template preserves the exact inputs used for the self-contained compiled bundle.\n').encode())
    inventory = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            raw = _read(path, MAX_FILE)
            inventory[path.relative_to(root).as_posix()] = _receipt(raw)
    seal = dict(schema=SCHEMA, scope='Recorded viewer projection consistency; external dataset science remains outside this file inventory.',
        source=source, scientific_sources_sha256=inventory['scientific-sources.json']['sha256'],
        physical_robot_ready=False, qualified_robot_skill=False, files=inventory,
        replay_cases=258, all_attempts=512, reconstruction_max_absolute_residual_m=residuals,
        generated_source_sha256=sha256(generated).hexdigest(), bundle_sha256=sha256(bundle).hexdigest(),
        first_embedded_case=dict(id=motion['cases'][0]['id'], variant=motion['cases'][0]['variant']),
        native_dataset=motion['native_dataset'], namespace_byte_limit=MAX_NAMESPACE,
        file_byte_limit=MAX_FILE, lazy_script_byte_limit_exclusive=MAX_SCRIPT,
        startup='Exact gzip geometry, scene/motion and first scheduled clock replay embedded; no fetch-body reads for playback.')
    _write(root, 'viewer-integrity.json', _json(seal))
    total = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
    if total > MAX_NAMESPACE:
        raise ValueError('The complete clock viewer exceeds its 192 MiB namespace budget.')
    return dict(files=len(inventory)+1, bytes=total, bundle_bytes=len(bundle),
                viewer_integrity_sha256=sha256((root/'viewer-integrity.json').read_bytes()).hexdigest())


def export_viewer(study, output, assets, dataset_revision):
    """Verify all 512 outcomes; export all 256 learned and two control replays."""
    from .piper_task import PiperTaskConfig, _asset_root, build_task_model
    from . import robot_profiles
    root, destination = _plain(study), _plain(output)
    _revision(dataset_revision)
    if destination.exists() or not destination.parent.is_dir() or destination == root or destination.is_relative_to(root):
        raise ValueError('Use a fresh viewer directory outside the sealed study, with an existing parent.')
    raw, protocol, frozen, report = _verified_study(root)
    asset_root = _asset_root(assets)
    asset_info = robot_profiles.verify_assets('piper', asset_root)
    _, asset_blobs = robot_profiles._payload('piper', asset_root)
    source = dict(report_sha256=sha256(raw['report.json']).hexdigest(), protocol_sha256=sha256(raw['protocol.json']).hexdigest(),
        frozen_sha256=sha256(raw['frozen.json']).hexdigest(), runtime_sha256=frozen['runtime_sha256'],
        lifecycle_source_sha256=frozen['lifecycle_source_sha256'], asset_provenance=asset_info,
        exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        pose_exporter_sha256=sha256(Path(__file__).with_name('piper_imitation_viewer.py').read_bytes()).hexdigest(),
        geometry_exporter_sha256=sha256(Path(__file__).with_name('reference_replay.py').read_bytes()).hexdigest(),
        browser_source_sha256=sha256(Path(__file__).with_name('piper_grip_clock_viewer.js').read_bytes()).hexdigest(),
        html_source_sha256=sha256(Path(__file__).with_name('piper_grip_clock_viewer.html').read_bytes()).hexdigest(),
        bundle_builder_sha256=sha256((Path(__file__).resolve().parents[2]/'scripts'/'build-piper-grip-clock-viewer.mjs').read_bytes()).hexdigest(),
        base_model_raw_sha256=frozen['base_model_raw_sha256'], jaw_model_raw_sha256=frozen['jaw_model_raw_sha256'],
        selected_sha256=frozen['selected_sha256'], blind_sha256=frozen['blind_sha256'], jaw_mode='coordinated_reference',
        physical_robot_ready=False)
    scientific_sources = dict(schema=SCHEMA+'/source-inventory',
        documents={name: _receipt(value) for name, value in raw.items()},
        source_snapshots={name: dict(sha256=value) for name, value in frozen['lifecycle_source_sha256'].items()},
        confirmation={}, native_dataset_revision=dataset_revision,
        note='All original scientific files remain in the pinned dataset study/ directory; only display resources are derived here.')
    selection = {(row['case_id'], row['variant']) for row in selected_replays(report['rows'])}
    by_id = {case['id']: case for case in protocol['cases']['confirmation']}
    all_attempts, cases, descriptors = [], [], []
    residuals, warnings = dict(tcp_position_m=0., object_position_m=0.), []
    scene = geometry = first_payload = previous_config = native_model = None
    with tempfile.TemporaryDirectory(prefix='.dvidia-clock-export-', dir=destination.parent) as temporary:
        staging = Path(temporary)/'viewer'
        staging.mkdir()
        with sim.native_warnings(warnings):
            for row in report['rows']:
                selected = (row['case_id'], row['variant']) in selection
                evidence, bindings, episode = _attempt(root, row, by_id[row['case_id']], frozen, dataset_revision,
                                                     decode_episode=selected)
                scientific_sources['confirmation'].update(bindings)
                item = dict(id=row['case_id'], case_id=row['case_id'], variant=row['variant'], condition=row['condition'],
                            score=row['score'], evidence=evidence)
                all_attempts.append(item)
                if not selected:
                    continue
                index = len(cases)
                config = episode.get('config')
                if type(config) is not dict:
                    raise ValueError('Clock replay lacks its recorded native configuration.')
                if config != previous_config:
                    native_model = build_task_model(asset_root, PiperTaskConfig(**config))
                    previous_config = deepcopy(config)
                    candidate, candidate_geometry = _scene(native_model, source)
                    candidate.update(schema=SCENE_SCHEMA, scope=SCOPE)
                    if scene is None:
                        scene, geometry = candidate, candidate_geometry
                    elif candidate != scene or candidate_geometry != geometry:
                        raise ValueError('Native geometry/body layout differs across clock replay configurations.')
                compiled = sim.model_digest(native_model)
                if episode.get('trajectory') and episode.get('compiled_model_sha256') != compiled:
                    raise ValueError('Clock recorded compiled model differs from the rebuilt native model.')
                poses, frame_rows = _poses(native_model, episode, residuals)
                frame_rows = _clock_rows(episode, frame_rows, row['variant'])
                frame_raw = _json(frame_rows)
                packed = payload(index, frame_raw, poses)
                descriptor = dict(id=resource_id(index), case_id=row['case_id'], variant=row['variant'],
                    resources=[{key: value for key, value in resource_meta.items() if key != 'data'}
                               for resource_meta in packed['resources']], script=None)
                if index == 0:
                    if row['variant'] != 'clock':
                        raise ValueError('The first scheduled learned-clock attempt must be embedded without outcome selection.')
                    first_payload = packed
                else:
                    script_raw = script(packed)
                    name = f'replay-data-{index:03}.js'
                    _write(staging, name, script_raw)
                    descriptor['script'] = dict(href=name, bytes=len(script_raw),
                        integrity='sha384-'+base64.b64encode(sha384(script_raw).digest()).decode('ascii'))
                descriptors.append(descriptor)
                cases.append(dict(item, frame_count=len(frame_rows), compiled_model_sha256=compiled,
                    pose_binary=dict(href=f'poses-{index:03}.bin', **_receipt(poses)),
                    rows_file=dict(href=f'frames-{index:03}.json', **_receipt(frame_raw))))
        if warnings or len(all_attempts) != 512 or len(cases) != 258:
            raise ValueError('Clock export has native warnings or incomplete outcome/replay counts.')
        scene['geometry_binary'] = dict(href='geometry.bin', **_receipt(geometry))
        motion = dict(schema=SCHEMA, scope=SCOPE, source=source, cases=cases, all_attempts=all_attempts,
            body_count=len(scene['bodies']), frame_stride_floats=len(scene['bodies'])*7,
            variants=report['variants'], contrasts=report['contrasts'], gates=report['gates'],
            advance_to_followup_simulation=report['advance_to_followup_simulation'],
            qualified_robot_skill=False, physical_robot_ready=False, limitations=report['limitations'],
            native_dataset=dict(repo_id=DATASET_REPO, revision=dataset_revision, publication_readback_verified=False,
                note='Exact dataset commit configured offline; consult the publication receipt for verified uploads and anonymous readback.'),
            local_resources=dict(report='report.json', protocol='protocol.json', frozen_references='frozen.json'),
            selection_notice='All 128 clock and 128 blind-clock confirmation attempts plus the first scheduled fixed/teacher controls are replayable. Selection is independent of outcome. All 512 scheduled results and exact pinned evidence links remain available.',
            clock_notice='Physical time drives playback; the current command uses virtual progress before the next hold/advance increment. Terminal states issue no new action. Clock probability is not a physical safety confidence.',
            reconstruction_max_absolute_residual_m=residuals,
            binary_pose_precision='little-endian Float32 native body xyz metres and quaternion wxyz')
        manifest = dict(schema=RESOURCE_SCHEMA, embedded={name: resource(name, value) for name, value in
            (('scene.json', _json(scene)), ('motion.json', _json(motion)), ('geometry.bin', geometry))},
            first=first_payload, cases=descriptors)
        for name in ('protocol.json', 'frozen.json', 'report.json'):
            _write(staging, name, raw[name])
        _write(staging, 'LICENSE', MIT_LICENSE.encode())
        _write(staging, 'reference_replay.third-party.txt', _read(Path(__file__).with_name('reference_replay.third-party.txt'), MAX_FILE))
        for name in ('LICENSE', 'README.md'):
            key = 'agilex_piper/'+name
            if key in asset_blobs:
                _write(staging, 'licenses/'+key, asset_blobs[key])
        result = _finalize(staging, manifest, motion, scene, scientific_sources, source, residuals)
        if (any(_read(_inside(root, name), MAX_JSON) != value for name, value in raw.items())
                or lifecycle.source_hashes() != frozen['lifecycle_source_sha256']
                or runner.runtime_hashes() != frozen['runtime_sha256']):
            raise ValueError('Clock source or original science changed during recorded replay export.')
        # A fresh directory rename publishes the complete local sealed export;
        # the study and any failed previous export remain untouched.
        if destination.exists() or destination.is_symlink():
            raise ValueError('Clock replay destination appeared during export.')
        staging.rename(destination)
    return dict(schema=SCHEMA, output=str(destination), replay_cases=258, report_attempts=512,
                frames=sum(case['frame_count'] for case in cases), physical_robot_ready=False,
                namespace_byte_limit=MAX_NAMESPACE, dataset_revision=dataset_revision,
                reconstruction_max_absolute_residual_m=residuals, **result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--dataset-revision', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_viewer(args.study, args.output, args.assets, args.dataset_revision), indent=2))


if __name__ == '__main__':
    main()
