"""Read-only forensic audit of the lineage-bound learned PiPER grip study.

No physics is run, no model is trained and no network/credential is accessed.
Saved numeric sidecars are independently rescored by the frozen scoring source.
A passing audit means consistency of preserved evidence, not robot qualification.
Only the final fresh audit JSON is written, after every check has completed.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import zipfile

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY/'src'))
from dvidia_training.piper_feedback_score import score_feedback_episode

SCHEMA = 'dvidia.piper-feedback-audit/v0.2'
STUDY_SCHEMA = 'dvidia.piper-feedback-study/v0.2'
PARENT_SCHEMA = 'dvidia.piper-feedback-training-parent/v1'
BASE_SHA = '2976a605575d9edefb5db366e72c5e006c0ce85a5531ad3d61512436864c8ded'
VARIANTS = ('feedback', 'blind', 'feedforward', 'teacher', 'open_jaw')
NEGATIVE_CANDIDATES = ('dag256', 'dag512', 'dagstrong128', 'full256', 'full512')
SHA = re.compile(r'[a-f0-9]{64}\Z')
MAX_FILE = 32*1024*1024
MAX_DECODED = 128*1024*1024
MAX_FILES = 10000
MAX_TOTAL = 3*1024*1024*1024
DEADLINE = .02


class AuditError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise AuditError(message)


def _plain(path):
    path = Path(path).absolute()
    _require(not any(p.is_symlink() and p not in {Path('/tmp'), Path('/var')} for p in (path, *path.parents)),
             'Audit inputs and output must not traverse artifact-owned symlinks.')
    return path.resolve()


def _relative(name):
    _require(isinstance(name, str) and name and '\\' not in name, 'Malformed manifest path.')
    path = PurePosixPath(name)
    _require(not path.is_absolute() and path.as_posix() == name and all(not p.startswith('.') for p in path.parts)
             and '__pycache__' not in path.parts, 'Unsafe manifest path.')
    return path


def _read(path, limit=MAX_FILE):
    path = _plain(path)
    _require(path.is_file() and path.stat().st_size <= limit, 'Missing or oversized audit input.')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    _require(len(raw) <= limit, 'Audit input grew beyond its byte bound.')
    return raw


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON key in evidence.')
        result[key] = value
    return result


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(AuditError('Nonfinite JSON evidence.')))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError('Invalid JSON evidence.') from exc


def _document(path):
    value = _json(_read(path))
    _require(isinstance(value, dict), 'Expected a JSON object.')
    return value


def _encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(path):
    return sha256(_read(path)).hexdigest()


def _inventory(root):
    root = _plain(root)
    _require(root.is_dir(), 'Expected a plain study directory.')
    files, total = {}, 0
    for path in sorted(root.rglob('*')):
        _plain(path)
        if path.is_dir():
            continue
        _require(path.is_file(), 'Non-file entry in study evidence.')
        name = path.relative_to(root).as_posix()
        _relative(name)
        raw = _read(path)
        files[name] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
        total += len(raw)
        _require(len(files) <= MAX_FILES and total <= MAX_TOTAL, 'Study inventory exceeds audit bounds.')
    return files


def _npz(raw):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            _require(len(members) <= 100 and len({m.filename for m in members}) == len(members), 'Malformed NPZ member inventory.')
            _require(sum(m.file_size for m in members) <= MAX_DECODED, 'NPZ decompression exceeds its bound.')
            for member in members:
                p = _relative(member.filename)
                _require(len(p.parts) == 1 and p.suffix == '.npy' and not member.flag_bits & 1,
                         'Unsafe or encrypted native archive member.')
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
        _require(all(a.dtype.kind in 'fiuUS' and a.dtype.kind != 'O' for a in result.values()), 'Executable/object arrays are forbidden.')
        return result
    except AuditError:
        raise
    except (ValueError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise AuditError('Native/sample archive cannot be decoded safely.') from exc


def _episode(raw):
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            decoded = stream.read(MAX_DECODED+1)
    except (OSError, EOFError) as exc:
        raise AuditError('Episode compression is invalid.') from exc
    _require(len(decoded) <= MAX_DECODED, 'Episode decompression exceeds its bound.')
    value = _json(decoded)
    _require(isinstance(value, dict), 'Episode is not a JSON object.')
    return value


def _sources(root, runtime):
    _require(isinstance(runtime, dict) and runtime, 'Missing frozen runtime source inventory.')
    source_root = REPOSITORY/'src/dvidia_training'
    for name, digest in runtime.items():
        _require(isinstance(name, str) and len(_relative(name).parts) == 1
                 and isinstance(digest, str) and SHA.fullmatch(digest), 'Invalid runtime source record.')
        _require(_sha(root/'sources'/name) == digest, 'Saved source snapshot differs from the bound runtime.')
        _require(_sha(source_root/name) == digest, 'Trusted local audit/runtime source differs from the bound source.')
    _require({'piper_feedback_score.py', 'piper_imitation_score.py'} <= set(runtime), 'Independent scoring source is not bound.')


def _summary(samples):
    values = np.asarray(samples, dtype=np.float64)
    if not len(values):
        return dict(samples=0, p50_s=None, p95_s=None, p99_s=None, maximum_s=None)
    return dict(samples=len(values), p50_s=float(np.percentile(values, 50)), p95_s=float(np.percentile(values, 95)),
                p99_s=float(np.percentile(values, 99)), maximum_s=float(np.max(values)))


def _timing(episode):
    timing = episode.get('timing')
    _require(isinstance(timing, dict), 'Attempt lacks saved timing samples.')
    result = {}
    arrays = {}
    for name, field in (('inference', 'inference_samples_s'), ('control_step', 'control_step_samples_s')):
        values = timing.get(field)
        _require(isinstance(values, list) and len(values) <= 1000
                 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 3600 for v in values),
                 'Malformed timing samples.')
        arrays[name] = values
        result[name] = _summary(values)
        saved = timing.get(name)
        if values:
            _require(isinstance(saved, dict) and set(saved) == set(result[name])
                     and saved['samples'] == len(values)
                     and all(math.isclose(saved[k], result[name][k], rel_tol=0, abs_tol=1e-12)
                             for k in ('p50_s', 'p95_s', 'p99_s', 'maximum_s')), 'Saved timing summary differs from raw samples.')
        else:
            _require(saved is None, 'Empty timing samples have an unexpected summary.')
    misses = sum(v > DEADLINE for v in arrays['control_step'])
    _require(type(timing.get('missed_20ms_boundaries')) is int and timing['missed_20ms_boundaries'] == misses,
             'Saved 20 ms deadline count differs from raw step timings.')
    wall = timing.get('wall_seconds')
    peak = timing.get('process_peak_rss_bytes')
    _require(type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall <= 86400
             and type(peak) is int and peak > 0, 'Malformed process timing or memory evidence.')
    if episode.get('valid') is True:
        expected_actions = round(episode['config']['timeout_s']/.02)
        _require(len(arrays['inference']) == len(episode['actions']) == len(arrays['control_step']) == expected_actions,
                 'Valid full-duration attempt lacks a timing measurement per control action.')
    result.update(missed_20ms_boundaries=misses, inference_over_20ms=sum(v > DEADLINE for v in arrays['inference']),
                  wall_seconds=float(wall), process_peak_rss_bytes=peak)
    return result, arrays


def _audit_attempt(root, relative, case, variant, protocol_sha, runtime, base_raw_sha, base_semantic_sha,
                   model_raw_sha=None, model_semantic_sha=None, collect=False, expert_probability=1., *, require_full_time=True):
    """Check raw hashes, literal identities, then recompute native numeric results."""
    folder = root/_relative(relative)
    receipt = _document(folder/'receipt.json')
    bound = _document(folder/'started.json')
    names = {'started.json', 'episode.json.gz', 'native.npz', 'native.json', 'score.json'} | ({'samples.npz'} if collect else set())
    _require(set(p.name for p in folder.iterdir()) == names | {'receipt.json'}, 'Attempt folder is incomplete or contains unbound files.')
    _require(receipt.get('binding') == bound and receipt.get('case_id') == case['id'] and receipt.get('variant') == variant,
             'Attempt receipt identity differs from its literal request.')
    expected = dict(case=case, variant=variant, collect_labels=collect, expert_probability=float(expert_probability),
        protocol_sha256=protocol_sha, base_model_raw_sha256=base_raw_sha, base_model_sha256=base_semantic_sha,
        model_raw_sha256=model_raw_sha, model_sha256=model_semantic_sha, runtime_sha256=runtime)
    _require(bound == expected, 'Attempt model, protocol, source, case or supervision binding differs.')
    raw_files = {name: _read(folder/name) for name in names}
    _require(receipt.get('files') == {name: sha256(raw).hexdigest() for name, raw in raw_files.items()}, 'Attempt raw bytes differ from its receipt.')
    episode = _episode(raw_files['episode.json.gz'])
    metadata, saved_score = _json(raw_files['native.json']), _json(raw_files['score.json'])
    _require(episode.get('schema') == 'dvidia.piper-feedback-episode/v0.2'
             and episode.get('case_id') == case['id'] and episode.get('seed') == case['seed']
             and episode.get('policy_kind') == variant and episode.get('condition') == case['condition']
             and episode.get('pulses') == case['pulses'] and metadata.get('pulses') == case['pulses']
             and episode.get('base_model_sha256') == base_semantic_sha and episode.get('model_sha256') == model_semantic_sha
             and episode.get('implementation_sha256') == runtime and episode.get('physical_robot_ready') is False,
             'Episode identity, intervention, model or source differs from its receipt.')
    config = episode.get('config', {})
    _require(config.get('object_position_m') == [*case['start_xy'], .082]
             and config.get('target_position_m') == [*case['goal_xy'], .08], 'Episode configuration differs from literal scheduled geometry.')
    if require_full_time:
        _require(config.get('timeout_s') == 20. and config.get('control_dt') == .02 and config.get('timestep') == .001,
                 'Attempt changes the fixed 20 s / 20 ms / 1 ms protocol.')
    arrays = _npz(raw_files['native.npz'])
    actual_score = score_feedback_episode(episode, arrays, metadata, case['goal_xy'])
    _require(actual_score == saved_score, 'Independent native reconstruction differs from the saved score.')
    if collect:
        samples = _npz(raw_files['samples.npz'])
        _require(set(samples) == {'features', 'blind_features', 'targets', 'groups', 'weights'}, 'Original label archive fields changed.')
        count = len(episode.get('actions', []))
        _require(samples['features'].shape == samples['blind_features'].shape == (count, 183)
                 and samples['targets'].shape == (count, 7) and samples['weights'].shape == samples['groups'].shape == (count,)
                 and samples['groups'].dtype.kind in 'US' and np.all(samples['groups'] == case['id'])
                 and all(np.isfinite(samples[k]).all() for k in ('features', 'blind_features', 'targets', 'weights')),
                 'Original causal label archive is incomplete or malformed.')
        _require(all(isinstance(a.get('expert_action'), dict) for a in episode.get('actions', [])), 'Original teacher labels are missing.')
    timing, samples = _timing(episode)
    row = dict(path=relative, case_id=case['id'], condition=case['condition'], variant=variant,
        valid=actual_score.get('valid') is True, success=actual_score.get('success') is True,
        reason=actual_score.get('reason'), score_sha256=sha256(raw_files['score.json']).hexdigest(),
        episode_sha256=sha256(raw_files['episode.json.gz']).hexdigest(), native_npz_sha256=sha256(raw_files['native.npz']).hexdigest(),
        native_metric_mismatches=actual_score.get('native_metrics_mismatches', []),
        strict_v01_success=actual_score.get('strict_v01', {}).get('success') is True,
        terminal_only_success=actual_score.get('terminal_only', {}).get('success') is True,
        terminal_xy_error_m=actual_score.get('terminal_only', {}).get('terminal_xy_error_m'),
        retention_lost=actual_score.get('native_recomputed', {}).get('retention_lost') is True,
        grip_events=actual_score.get('grip_events', []), timing=timing,
        native_rows=metadata.get('rows'), asset_provenance_sha256=_digest(episode.get('asset_provenance')),
        compiled_model_sha256=episode.get('compiled_model_sha256'), platform=episode.get('platform'))
    return row, samples, actual_score


def _aggregate(rows, timing_samples):
    events = [event for row in rows for event in row['grip_events']]
    supported = [event for event in events if event['supported_at_onset']]
    airborne = [event for event in events if not event['supported_at_onset']]
    latency = [event['onset_to_qualified_restoration_s'] for event in events if event['restored']]
    xy = [r['terminal_xy_error_m'] for r in rows if type(r['terminal_xy_error_m']) in (int, float) and math.isfinite(r['terminal_xy_error_m'])]
    xy_summary = _summary(xy)
    xy_summary = {key.removesuffix('_s'): value for key, value in xy_summary.items()}
    return dict(scheduled=len(rows), valid=sum(r['valid'] for r in rows), successes=sum(r['success'] for r in rows),
        invalid=sum(not r['valid'] for r in rows), native_metric_mismatch_invalid=sum(bool(r['native_metric_mismatches']) for r in rows),
        strict_v01_successes=sum(r['strict_v01_success'] for r in rows), terminal_only_successes=sum(r['terminal_only_success'] for r in rows),
        terminal_only_without_strict_success=sum(r['terminal_only_success'] and not r['success'] for r in rows),
        retention_lost_attempts=sum(r['retention_lost'] for r in rows),
        terminal_xy_error_m=xy_summary,
        grip=dict(loss_events=len(events), supported_loss_events=len(supported), airborne_loss_events=len(airborne),
            supported_restored_events=sum(e['restored'] for e in supported), airborne_restored_events=sum(e['restored'] for e in airborne),
            unresolved_events=sum(not e['restored'] for e in events), qualification_latency=_summary(latency)),
        timing=dict(inference=_summary(timing_samples['inference']), control_step=_summary(timing_samples['control_step']),
            control_step_over_20ms=sum(r['timing']['missed_20ms_boundaries'] for r in rows),
            inference_over_20ms=sum(r['timing']['inference_over_20ms'] for r in rows),
            episode_wall=_summary([r['timing']['wall_seconds'] for r in rows]),
            maximum_cumulative_process_rss_bytes=max((r['timing']['process_peak_rss_bytes'] for r in rows), default=None)))


def audit(run, *, progress=False):
    from dvidia_training.piper_grip_adapter import validate_model as validate_grip
    from dvidia_training.piper_feedback_model import validate_model as validate_residual
    from dvidia_training.piper_motor_primitive import validate_model as validate_base
    root = _plain(run)
    _require(all((root/n).is_file() for n in ('report.json', 'protocol.json', 'frozen.json', 'training-parent.json')),
             'Audit requires a complete frozen confirmation report; no partial report is accepted.')
    inventory = _inventory(root)
    protocol, frozen, report, parent = [_document(root/n) for n in ('protocol.json', 'frozen.json', 'report.json', 'training-parent.json')]
    bindings = {key: inventory[name]['sha256'] for key, name in
        (('report_sha256', 'report.json'), ('protocol_sha256', 'protocol.json'), ('frozen_sha256', 'frozen.json'), ('training_parent_sha256', 'training-parent.json'))}
    _require(protocol.get('schema') == frozen.get('schema') == report.get('schema') == STUDY_SCHEMA
             and protocol.get('physical_robot_ready') is False and report.get('physical_robot_ready') is False
             and protocol.get('variants') == list(VARIANTS)
             and frozen.get('protocol_sha256') == report.get('protocol_sha256') == bindings['protocol_sha256']
             and report.get('frozen_sha256') == bindings['frozen_sha256']
             and protocol.get('training_parent_manifest_sha256') == frozen.get('training_parent_sha256') == bindings['training_parent_sha256'],
             'Active protocol/report/frozen/parent bindings disagree.')
    _require(inventory['base-model.json']['sha256'] == frozen.get('base_model_sha256') == protocol.get('base_model_raw_sha256') == BASE_SHA,
             'The raw frozen v0.1 base bytes changed.')
    base = validate_base(_read(root/'base-model.json'))
    base_semantic = _digest(base)
    runtime = frozen.get('runtime_sha256')
    _sources(root, runtime)
    cases = protocol.get('cases')
    _require(isinstance(cases, dict) and set(cases) == {'train', 'development', 'confirmation'}
             and all(isinstance(cases[k], list) and len(cases[k]) == count for k, count in (('train', 64), ('development', 16), ('confirmation', 24))),
             'Active split schedule is incomplete.')
    for split, case_rows in cases.items():
        ids = [c.get('id') for c in case_rows]
        _require(all(isinstance(v, str) and len(_relative(v).parts) == 1 and v.startswith(split+'-') for v in ids)
                 and len(set(ids)) == len(ids), 'Duplicate or unsafe scheduled case identity.')
    xy_sets = {}
    for split, expected_xy in (('train', 16), ('development', 4), ('confirmation', 6)):
        scene_roster = set()
        for case in cases[split]:
            xy, goal, condition = case.get('start_xy'), case.get('goal_xy'), case.get('condition')
            _require(isinstance(xy, list) and len(xy) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in xy)
                     and .284 <= xy[0] <= .314 and -.045 <= xy[1] <= -.015
                     and goal in ([.35, .08], [.40, -.08]) and condition in ('nominal', 'jaw_interruption'),
                     'Scene is outside the declared narrow geometry or condition roster.')
            identity = (*xy, *goal, condition)
            _require(identity not in scene_roster, 'Repeated scene geometry/condition within a split.')
            scene_roster.add(identity)
        xy_sets[split] = {tuple(c['start_xy']) for c in cases[split]}
        _require(len(xy_sets[split]) == expected_xy and len(scene_roster) == expected_xy*4, 'Split does not cover two goals and both conditions at every starting XY.')
    _require(not (xy_sets['train'] & xy_sets['development'] or xy_sets['train'] & xy_sets['confirmation']
                  or xy_sets['development'] & xy_sets['confirmation']), 'Train, development and confirmation starting XY overlap.')
    oldroot = root/'prior-studies/study01'
    oldfiles = {n.removeprefix('prior-studies/study01/'): value for n, value in inventory.items() if n.startswith('prior-studies/study01/')}
    _require(parent.get('schema') == PARENT_SCHEMA and parent.get('path') == 'prior-studies/study01'
             and parent.get('files') == oldfiles and parent.get('confirmation_attempts') == 0
             and not any(n.startswith('prior-studies/') and not n.startswith('prior-studies/study01/') for n in inventory)
             and not any(n.startswith(('demonstrations/', 'dagger/')) for n in inventory)
             and not any(n.startswith('confirmation/') or n in {'frozen.json', 'report.json'} for n in oldfiles),
             'Original unfrozen negative-study inventory differs from its exact parent manifest.')
    oldprotocol = _document(oldroot/'protocol.json')
    oldruntime = parent.get('runtime_sha256')
    _require(oldprotocol.get('schema') == STUDY_SCHEMA and oldfiles['protocol.json']['sha256'] == parent.get('protocol_sha256')
             and oldfiles['base-model.json']['sha256'] == BASE_SHA
             and oldprotocol['cases']['train'] == cases['train'] and oldprotocol['cases']['development'] == cases['development'],
             'Original protocol or reused train/development geometry was relabeled.')
    _sources(oldroot, oldruntime)
    samples = {f"demonstrations/{c['id']}/samples.npz": oldfiles.get(f"demonstrations/{c['id']}/samples.npz", {}).get('sha256') for c in cases['train']}
    _require(parent.get('teacher_sample_files') == samples and all(isinstance(v, str) and SHA.fullmatch(v) for v in samples.values()),
             'Manifest does not bind all 64 original teacher sample archives.')
    _require(sorted(parent.get('negative_candidates', [])) == list(NEGATIVE_CANDIDATES), 'Negative candidate lineage differs from the recorded five fits.')
    oldcandidate_names = {PurePosixPath(n).parts[1] for n in oldfiles if n.startswith('candidates/')}
    _require(oldcandidate_names == set(NEGATIVE_CANDIDATES), 'Original negative candidate files were omitted or added.')
    groups, timing_groups, checked_paths = {}, {}, set()
    condition_groups, condition_timings = {}, {}
    count = 0

    def check(scope, folder_root, relative, case, variant, proto_sha, source_sha, model=None, *, collect=False, probability=1.):
        nonlocal count
        raw_sha, semantic_sha = (_sha(model), _digest(_document(model))) if model else (None, None)
        row, times, score = _audit_attempt(folder_root, relative, case, variant, proto_sha, source_sha,
            BASE_SHA, base_semantic, raw_sha, semantic_sha, collect=collect, expert_probability=probability)
        fullpath = ('prior-studies/study01/' if folder_root == oldroot else '')+relative
        _require(fullpath not in checked_paths, 'An attempt was counted twice.')
        checked_paths.add(fullpath)
        row['path'] = fullpath
        groups.setdefault(scope, []).append(row)
        aggregate_times = timing_groups.setdefault(scope, {'inference': [], 'control_step': []})
        for name in aggregate_times:
            aggregate_times[name].extend(times[name])
        if scope.startswith('confirmation_'):
            condition_key = scope+'/'+case['condition']
            condition_groups.setdefault(condition_key, []).append(row)
            conditional_times = condition_timings.setdefault(condition_key, {'inference': [], 'control_step': []})
            for name in conditional_times:
                conditional_times[name].extend(times[name])
        count += 1
        if progress and count % 16 == 0:
            print(json.dumps(dict(phase='native-evidence-rescored', attempts=count, scope=scope)), flush=True)
        return score

    oldsha = parent['protocol_sha256']
    for case in oldprotocol['cases']['train']:
        check('prior_teacher', oldroot, 'demonstrations/'+case['id'], case, 'teacher', oldsha, oldruntime, collect=True)
    dagger_cases = [c for c in oldprotocol['cases']['train'] if c['condition'] == 'jaw_interruption']
    _require(len(dagger_cases) == 32, 'Original DAgger scene roster changed.')
    for case in dagger_cases:
        bound = _document(oldroot/'dagger'/case['id']/'started.json')
        digest = bound.get('model_raw_sha256')
        matches = [oldroot/'candidates'/n/'model.json' for n in NEGATIVE_CANDIDATES if _sha(oldroot/'candidates'/n/'model.json') == digest]
        _require(len(matches) == 1, 'Original mixed-policy attempt is not bound to a retained candidate.')
        check('prior_dagger', oldroot, 'dagger/'+case['id'], case, 'mixture', oldsha, oldruntime, matches[0], collect=True, probability=.5)
    for name in NEGATIVE_CANDIDATES:
        directory = oldroot/'candidates'/name
        model = validate_residual(_read(directory/'model.json'))
        receipt, development = _document(directory/'receipt.json'), _document(directory/'development.json')
        _require(receipt.get('model_sha256') == development.get('model_sha256') == _sha(directory/'model.json')
                 and receipt.get('protocol_sha256') == development.get('protocol_sha256') == oldsha
                 and receipt.get('source_sha256') == development.get('runtime_sha256') == oldruntime
                 and receipt.get('training') == model['training'] and model['base_sha256'] == base_semantic,
                 'An original negative model/training/development/source binding changed.')
        evaluated = []
        variant = 'blind' if model['blind'] else 'feedback'
        for case in oldprotocol['cases']['development']:
            score = check('prior_candidate_'+name, oldroot, f"candidates/{name}/development/{case['id']}", case, variant,
                          oldsha, oldruntime, directory/'model.json')
            evaluated.append(dict(case_id=case['id'], condition=case['condition'], score=score))
        _require(development.get('rows') == evaluated and development.get('scheduled') == 16
                 and development.get('successes') == sum(r['score']['success'] for r in evaluated) == 0,
                 'Original negative development aggregate differs from all retained attempts.')
    candidates = sorted({PurePosixPath(n).parts[1] for n in inventory if n.startswith('candidates/')})
    _require(candidates and frozen.get('selected') in candidates and frozen.get('blind') in candidates
             and frozen['selected'] != frozen['blind'], 'Frozen selected and blind candidates are missing or identical.')
    models = {}
    for name in candidates:
        directory = root/'candidates'/name
        model = validate_grip(_read(directory/'model.json'))
        receipt, development = _document(directory/'receipt.json'), _document(directory/'development.json')
        _require(receipt.get('training_parent_sha256') == bindings['training_parent_sha256']
                 and receipt.get('model_sha256') == development.get('model_sha256') == _sha(directory/'model.json')
                 and receipt.get('protocol_sha256') == development.get('protocol_sha256') == bindings['protocol_sha256']
                 and receipt.get('source_sha256') == development.get('runtime_sha256') == runtime
                 and receipt.get('training') == model['training'] and receipt.get('dagger') is False
                 and model['base_sha256'] == base_semantic, 'Active candidate model, parent, recipe or source binding differs.')
        lineage = model['training'].get('parent_artifacts')
        _require(isinstance(lineage, dict) and lineage.get('protocol_sha256') == oldsha
                 and lineage.get('base_model_raw_sha256') == BASE_SHA and model['training']['sample_count'] == 64000
                 and model['training']['train_groups'] == [c['id'] for c in oldprotocol['cases']['train']]
                 and lineage.get('episodes') == [dict(case_id=c['id'], episode_sha256=_sha(oldroot/'demonstrations'/c['id']/'episode.json.gz'), samples=1000) for c in oldprotocol['cases']['train']],
                 'New learned model does not bind exactly the 64 original teacher episodes.')
        evaluated = []
        variant = 'blind' if model['blind'] else 'feedback'
        for case in cases['development']:
            score = check('active_candidate_'+name, root, f"candidates/{name}/development/{case['id']}", case, variant,
                          bindings['protocol_sha256'], runtime, directory/'model.json')
            evaluated.append(dict(case_id=case['id'], condition=case['condition'], score=score))
        _require(development.get('rows') == evaluated and development.get('scheduled') == 16
                 and development.get('successes') == sum(r['score']['success'] for r in evaluated),
                 'Active candidate development aggregate differs from saved native evidence.')
        models[name] = model
    recipe_keys = ('algorithm', 'max_depth', 'min_leaf', 'max_bins', 'sample_count', 'scene_count', 'train_groups', 'weight_sum')
    selected, blind = models[frozen['selected']], models[frozen['blind']]
    _require(selected['blind'] is False and blind['blind'] is True
             and all(selected['training'][k] == blind['training'][k] for k in recipe_keys), 'Feedback and blind controls differ in feature mode or training budget.')
    for prefix, name in (('selected', frozen['selected']), ('blind', frozen['blind'])):
        for suffix, filename in (('', 'model.json'), ('_receipt', 'receipt.json'), ('_development', 'development.json')):
            _require(frozen.get(prefix+suffix+'_sha256') == _sha(root/'candidates'/name/filename), 'Frozen candidate artifact changed.')
    confirmation = []
    for case in cases['confirmation']:
        for variant in VARIANTS:
            name = frozen['blind'] if variant == 'blind' else frozen['selected']
            model_path = root/'candidates'/name/'model.json' if variant in ('feedback', 'blind', 'open_jaw') else None
            score = check('confirmation_'+variant, root, f"confirmation/{case['id']}/{variant}", case, variant,
                          bindings['protocol_sha256'], runtime, model_path)
            confirmation.append(dict(case_id=case['id'], condition=case['condition'], variant=variant, score=score))
    _require(report.get('rows') == confirmation, 'Report order/outcomes differ from all 120 scheduled confirmation attempts.')
    counts = {v: dict(scheduled=24, valid=sum(r['score']['valid'] for r in confirmation if r['variant'] == v),
                     successes=sum(r['score']['success'] for r in confirmation if r['variant'] == v)) for v in VARIANTS}
    _require(report.get('variants') == counts, 'Published variant counts differ from independent native rescoring.')
    actual_attempts = {str(PurePosixPath(n).parent) for n in inventory if n.endswith('/started.json')}
    _require(actual_attempts == checked_paths, 'An original or active attempt was omitted, duplicated or left unscored.')
    _require(_inventory(root) == inventory, 'Study files changed while the read-only audit was running.')
    aggregates = {name: _aggregate(rows, timing_groups[name]) for name, rows in groups.items()}
    conditions = {name: _aggregate(rows, condition_timings[name]) for name, rows in condition_groups.items()}
    by_case = {(row['case_id'], row['variant']): row['score'] for row in confirmation}
    paired = {}
    for control in ('blind', 'feedforward', 'teacher', 'open_jaw'):
        pairs = [(by_case[c['id'], 'feedback'], by_case[c['id'], control]) for c in cases['confirmation']]
        paired[control] = dict(scheduled_pairs=24, both_success=sum(a['success'] and b['success'] for a, b in pairs),
            feedback_only_success=sum(a['success'] and not b['success'] for a, b in pairs),
            control_only_success=sum(not a['success'] and b['success'] for a, b in pairs),
            both_unsuccessful=sum(not a['success'] and not b['success'] for a, b in pairs),
            invalid_in_either=sum(not a['valid'] or not b['valid'] for a, b in pairs))
    return dict(schema=SCHEMA, status='passed', **bindings,
        auditor_sha256=_sha(Path(__file__)), scoring_source_sha256={n: runtime[n] for n in ('piper_feedback_score.py', 'piper_imitation_score.py')},
        physical_robot_ready=False, native_physics_run=False, model_retrained=False, credentials_read=False, network=False,
        audit_scope='Hash/identity/source/roster consistency and independent reconstruction from saved numeric native evidence. No physical rerun, hardware attestation or proof of optimization.',
        restoration_scope='Loaded bilateral contact restored for 100 ms after sustained loss; supported restoration is not proof of airborne regrasp or task recovery.',
        timing_scope='Shared-host recorded CPU inference and full native step; excludes cameras, transport and physical actuation. Percentiles are pooled recorded samples, not hard real-time bounds; RSS is cumulative process peak.',
        evidence=dict(prior_teacher=64, prior_dagger=32, prior_negative_candidates=5, prior_development=80,
                      active_candidates=len(candidates), active_development=16*len(candidates), confirmation=120, total_rescored=count),
        geometry_scope='Numerical terminal XY error relative to the simulated target, including unsuccessful attempts with a valid terminal diagnostic; not physical positioning accuracy.',
        platform=dict(python=platform.python_version(), numpy=np.__version__), variants=counts, aggregates=aggregates,
        confirmation_conditions=conditions, paired_feedback_comparison=paired,
        attempts=[row for rows in groups.values() for row in rows], audited_files=inventory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='Fresh audit file; default RUN/audit.json. Existing files are never overwritten.')
    args = parser.parse_args(argv)
    try:
        output = _plain(args.output or args.run/'audit.json')
        root = _plain(args.run)
        _require(not output.exists() and not output.is_symlink(), 'Audit output already exists; refusing to overwrite it.')
        _require(root not in output.parents or output == root/'audit.json', 'Audit may only create audit.json inside the study; evidence subdirectories are read-only.')
        result = audit(root, progress=True)
        raw = _encoded(result)
        _require(len(raw) <= 8*1024*1024, 'Audit report exceeds the publication size bound.')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(dict(schema=SCHEMA, status=result['status'], evidence=result['evidence'], variants=result['variants'],
                             audit_sha256=sha256(raw).hexdigest(), native_physics_run=False, network=False), indent=2, sort_keys=True))
    except Exception as exc:
        print(str(exc) if isinstance(exc, AuditError) else f'Audit stopped: {type(exc).__name__}; raw error suppressed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
