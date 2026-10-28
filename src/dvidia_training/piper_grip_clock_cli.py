"""Finite, source-bound lifecycle for the exploratory PiPER progress-clock study.

Terminal and Python entry points share the same immutable attempts. No command
automatically selects a model, changes a protocol, retries interrupted physics,
or treats successful installation as a qualified robot skill.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from . import piper_grip_clock_benchmark as runner
from . import piper_grip_clock_model as model
from . import piper_motor_primitive as primitive
from .piper_grip_clock import TeacherConfig
from .piper_grip_clock_env import PostGripParameters
from .piper_imitation import digest
from .piper_imitation_benchmark import read, write, sha, load_episode, _atomic_create


SEED = 273031
MINIMUM_GROUP_DISTANCE_M = .00075
GROUP_COUNTS = {'train': 16, 'development': 8, 'confirmation': 32}
CONFIRM_VARIANTS = ('clock', 'blind_clock', 'fixed', 'teacher')
CANDIDATES = {'full4': (False, 4), 'full6': (False, 6),
              'blind4': (True, 4), 'blind6': (True, 6)}
# Student-t .9875 quantile, df31: two-sided 97.5% intervals for each of two
# contrasts provide Bonferroni simultaneous 95% coverage under the t model.
T975_DF31 = 2.3555682821598367
LIMITATIONS = [
    'Exploratory simulation-only comparison; 32 independent XY groups do not establish adequate statistical power.',
    'Synthetic direct hold/advance labels and privileged simulator sensing, not human videos or camera learning.',
    'Jaw architecture is declared and hash-bound separately; physical-jaw and coordinated-reference studies are not pooled.',
    'Blind clock masks only its live clock inputs; it has the same declared path/jaw architecture as the full clock.',
    'Acquisition-only teacher releases its wait latch once; this is not continuous learned grip recovery.',
    'Same rigid 40 g block, two known destinations and narrow PiPER start envelope; no physical robot qualification.',
    'Simulator computation timing excludes camera, communication and physical actuators; no hard real-time guarantee.',
]


def source_hashes():
    return {**runner.runtime_hashes(), Path(__file__).name: sha(__file__)}


def limitations(jaw_mode):
    extra = ('Frozen v0.2 live-feedback jaw uses physical elapsed time; only six-arm-reference timing changes.'
             if jaw_mode == 'physical_adapter' else
             'All seven frozen v0.1 raw references, including the learned base jaw reference, share virtual progress; no v0.2 jaw adapter.')
    return [*LIMITATIONS, extra]


def historical_starts():
    """Literal union of all v0.1/v0.2 partitions and the known pilot starts."""
    values = {(x, y) for x in (.284, .294, .304, .314) for y in (-.045, -.035, -.025, -.015)}
    values |= {(x, y) for x in (.285, .295, .305, .313) for y in (-.044, -.034, -.024, -.016)}
    values |= {(x, y) for x in (.289, .309) for y in (-.04, -.02)}
    values |= {(x, y) for x in (.290, .310) for y in (-.039, -.019)}
    values |= {(.287, -.032), (.291, -.018), (.299, -.041), (.303, -.028), (.311, -.038), (.307, -.022),
               (.286, -.031), (.292, -.017), (.298, -.042), (.302, -.027), (.312, -.037), (.306, -.021)}
    return sorted(values)


def make_cases():
    rng = np.random.default_rng(SEED)
    excluded, cases, accepted = historical_starts(), {}, []
    for split, count in GROUP_COUNTS.items():
        points = []
        for _ in range(100000):
            point = np.round(rng.uniform((.284, -.045), (.314, -.015)), 6).tolist()
            if any(math.dist(point, other) < MINIMUM_GROUP_DISTANCE_M-1e-12
                   for other in (*excluded, *accepted)):
                continue
            points.append(point)
            accepted.append(point)
            if len(points) == count:
                break
        if len(points) != count:
            raise ValueError('Finite grouped-roster generation failed.')
        first = sum(GROUP_COUNTS[s] for s in list(GROUP_COUNTS)[:list(GROUP_COUNTS).index(split)])
        cases[split] = [dict(id=f'{split}-xy{index:02}-g{goal}-{condition}', start_xy=point,
                            goal_xy=list(destination), seed=273100+(first+index)*2+goal,
                            condition=condition)
                        for index, point in enumerate(points)
                        for goal, destination in enumerate(primitive.GOALS)
                        for condition in ('nominal', 'post_grip')]
    return cases


@contextmanager
def _locked(root):
    path = Path(root)/'.lifecycle.lock'
    if path.is_symlink():
        raise ValueError('Lifecycle lock cannot be a symlink.')
    descriptor = os.open(path, os.O_RDWR|os.O_CREAT|getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        if os.name == 'nt':
            import msvcrt
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b'0')
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _same_or_create(path, value):
    path = Path(path)
    if path.exists():
        if path.is_symlink() or read(path) != value:
            raise ValueError('An existing immutable aggregate differs: '+path.name)
    else:
        write(path, value)
    return value


def prepare(output, base_model, jaw_model, *, teacher_config=None, intervention_parameters=None):
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise ValueError('Prepare requires a fresh study directory.')
    base, jaw = Path(base_model), Path(jaw_model)
    if base.is_symlink() or jaw.is_symlink() or sha(base) != runner.BASE_SHA:
        raise ValueError('Prepare requires exact raw bytes of the frozen v0.1 base.')
    jaw_mode, _ = runner.jaw_contract(read(base), read(jaw))
    if jaw_mode == 'physical_adapter' and sha(jaw) != runner.JAW_SHA:
        raise ValueError('Physical-adapter studies require the exact frozen full jaw4 bytes.')
    runner._documents(read(base), read(jaw), None, 'teacher')
    teacher = teacher_config if teacher_config is not None else asdict(
        TeacherConfig(gate_offset_s=.32 if jaw_mode == 'physical_adapter' else .16))
    teacher, intervention = runner._options(teacher, intervention_parameters)
    sources = source_hashes()
    protocol = dict(schema=runner.SCHEMA, cases=make_cases(), variants=list(CONFIRM_VARIANTS),
        base_model_raw_sha256=runner.BASE_SHA, jaw_model_raw_sha256=sha(jaw), jaw_mode=jaw_mode,
        source_sha256=sources, teacher_config=asdict(teacher), post_grip_parameters=intervention,
        roster=dict(seed=SEED, groups=GROUP_COUNTS, rounding_decimals=6,
                    minimum_group_distance_m=MINIMUM_GROUP_DISTANCE_M,
                    historical_exclusions_xy_m=[list(p) for p in historical_starts()],
                    case_pairing='Both goals and both conditions in each whole-XY split; conditions share reset seed.'),
        training=dict(episodes=64, label='Actual causal pre-issue teacher progress increment: 0 or .02 seconds.',
                      hold_weight=20., advance_weight=1., selection='Every finite prefix, without outcome filtering.',
                      candidate_recipes={name: dict(blind=blind, max_depth=depth, min_leaf=64, max_bins=31)
                                         for name, (blind, depth) in CANDIDATES.items()}),
        analysis=dict(primary_denominator='All scheduled attempts; invalid/aborted/acquisition failures remain failures.',
            nonactivation='Score observed strict outcome; do not discard or automatically fail an unactivated condition.',
            primary_contrasts=['clock-minus-blind_clock', 'clock-minus-fixed'],
            independent_unit='Whole initial-XY group; four paired goal/condition cases.',
            confirmation_groups=32, cases_per_variant=128, scheduled_confirmation_attempts=512,
            interval='Mean group difference ± t(.9875,31)*sampleSD(group difference)/sqrt(32).',
            t_critical=T975_DF31, simultaneous_coverage=.95, comparison_interval_coverage=.975,
            minimum_mean_benefit=.10, lower_interval_bound_must_exceed=0.,
            nominal_difference_floor=-.05, critical_contact_attempt_increase_allowed=0,
            invalid_or_aborted_allowed=0, post_grip_effective_fraction_for_recovery_interpretation=.75,
            full_step_over_20ms_fraction_limit=.01,
            status='Exploratory; no adequate-power, hardware qualification or general recovery claim.'),
        strict_endpoint='Unchanged v0.1/v0.2 native lift/carry/retention/release/settling scorer; no relaxed recovery endpoint.',
        physical_robot_ready=False, limitations=limitations(jaw_mode))
    root.mkdir(parents=True)
    _atomic_create(root/'base-model.json', base.read_bytes())
    _atomic_create(root/'jaw-model.json', jaw.read_bytes())
    for name, expected in sources.items():
        source = Path(__file__).with_name(name)
        _atomic_create(root/'sources'/name, source.read_bytes())
        if sha(root/'sources'/name) != expected:
            raise ValueError('Source changed during preparation.')
    write(root/'protocol.json', protocol)
    write(root/'prepared.json', dict(schema=runner.SCHEMA, protocol_sha256=sha(root/'protocol.json'),
        source_sha256=sources, base_model_raw_sha256=runner.BASE_SHA, jaw_model_raw_sha256=sha(jaw), jaw_mode=jaw_mode))
    return protocol


def _study(root, *, mutable=False):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Study must be an ordinary prepared directory.')
    protocol, prepared = read(root/'protocol.json'), read(root/'prepared.json')
    sources = source_hashes()
    if (protocol.get('schema') != runner.SCHEMA or prepared.get('schema') != runner.SCHEMA
            or protocol.get('cases') != make_cases()
            or prepared.get('protocol_sha256') != sha(root/'protocol.json')
            or protocol.get('source_sha256') != sources or prepared.get('source_sha256') != sources
            or sha(root/'base-model.json') != runner.BASE_SHA
            or sha(root/'jaw-model.json') != protocol.get('jaw_model_raw_sha256')
            or prepared.get('jaw_model_raw_sha256') != protocol.get('jaw_model_raw_sha256')
            or prepared.get('jaw_mode') != protocol.get('jaw_mode')):
        raise ValueError('Prepared protocol, grouped roster, frozen model or lifecycle/runtime source changed.')
    jaw_mode, _ = runner.jaw_contract(read(root/'base-model.json'), read(root/'jaw-model.json'))
    if jaw_mode != protocol.get('jaw_mode') or jaw_mode == 'physical_adapter' and sha(root/'jaw-model.json') != runner.JAW_SHA:
        raise ValueError('Declared jaw architecture or exact physical-adapter identity changed.')
    if any((root/'sources'/name).is_symlink() or sha(root/'sources'/name) != expected
           for name, expected in sources.items()):
        raise ValueError('A prepared source snapshot changed.')
    if mutable and ((root/'frozen.json').exists() or (root/'confirmation').exists()):
        raise ValueError('Collection/training/development are closed once confirmation is frozen or started.')
    return protocol


def _attempt(root, case, variant, folder, model_path=None, *, labels=False):
    return runner._completed_attempt(folder, runner._request_binding(root, case, variant, model_path, labels))


def _folders_exact(directory, case_ids):
    if directory.is_symlink() or not directory.is_dir() or {p.name for p in directory.iterdir()} != set(case_ids):
        raise ValueError('Attempt inventory differs from the complete literal scheduled roster.')


def _teachers(root, protocol):
    root = Path(root)
    cases = protocol['cases']['train']
    _folders_exact(root/'demonstrations', [case['id'] for case in cases])
    rows, datasets = [], []
    for case in cases:
        folder = root/'demonstrations'/case['id']
        score = _attempt(root, case, 'teacher', folder, labels=True)
        arrays = runner._load_npz(folder/'samples.npz')
        # Independently checks all finite rows, 20/1 weights and exact masking.
        arrays = runner._sample_arrays(arrays, case['id'])
        rows.append(dict(case_id=case['id'], valid=score['valid'], success=score['success'],
                         samples=len(arrays['targets']), receipt_sha256=sha(folder/'receipt.json'),
                         samples_sha256=sha(folder/'samples.npz')))
        datasets.append(arrays)
    manifest = dict(schema=runner.SCHEMA, protocol_sha256=sha(root/'protocol.json'),
                    source_sha256=source_hashes(), scheduled=64, rows=rows,
                    valid=sum(row['valid'] for row in rows), successes=sum(row['success'] for row in rows),
                    samples=sum(row['samples'] for row in rows),
                    selection='Every finite teacher prefix, without success/validity filtering.')
    return manifest, datasets


def collect(root, assets):
    root = Path(root)
    with _locked(root):
        protocol = _study(root, mutable=True)
        for case in protocol['cases']['train']:
            runner.execute(root, assets, case, 'teacher', root/'demonstrations'/case['id'], collect_labels=True)
        manifest, _ = _teachers(root, protocol)
        return _same_or_create(root/'teacher-manifest.json', manifest)


def train(root, name, *, blind=False, max_depth=4, min_leaf=64, max_bins=31):
    root = Path(root)
    with _locked(root):
        protocol = _study(root, mutable=True)
        if name not in CANDIDATES or CANDIDATES[name] != (blind, max_depth) or min_leaf != 64 or max_bins != 31:
            raise ValueError('Use one of the four literal predeclared matched candidate recipes.')
        directory = root/'candidates'/name
        if directory.exists() or directory.is_symlink():
            raise ValueError('A candidate fit is never overwritten or silently resumed.')
        manifest, datasets = _teachers(root, protocol)
        if read(root/'teacher-manifest.json') != manifest:
            raise ValueError('Teacher manifest differs from complete independently checked training evidence.')
        feature_key = 'blind_features' if blind else 'features'
        features = np.concatenate([data[feature_key] for data in datasets], axis=0)
        targets = np.concatenate([data['targets'] for data in datasets])
        weights = np.concatenate([data['weights'] for data in datasets])
        groups = [str(group) for data in datasets for group in data['groups']]
        began = time.perf_counter()
        parents = dict(protocol_sha256=sha(root/'protocol.json'),
                       teacher_manifest_sha256=sha(root/'teacher-manifest.json'))
        document = model.train(dict(features=features, targets=targets, groups=groups),
            base_sha256=digest(read(root/'base-model.json')), jaw_sha256=digest(read(root/'jaw-model.json')),
            blind=blind, max_depth=max_depth, min_leaf=min_leaf, max_bins=max_bins,
            sample_weights=weights, parent_artifacts=parents, jaw_mode=protocol['jaw_mode'])
        write(directory/'model.json', document)
        receipt = dict(schema=runner.SCHEMA, name=name, blind=blind, protocol_sha256=sha(root/'protocol.json'),
            teacher_manifest_sha256=sha(root/'teacher-manifest.json'), source_sha256=source_hashes(),
            model_sha256=sha(directory/'model.json'), training_seconds=time.perf_counter()-began,
            model_bytes=(directory/'model.json').stat().st_size, training=document['training'])
        write(directory/'receipt.json', receipt)
        return receipt


def _candidate(root, name):
    root = Path(root)
    if name not in CANDIDATES:
        raise ValueError('Unknown declared candidate.')
    directory = root/'candidates'/name
    document, receipt = model.validate_model(read(directory/'model.json')), read(directory/'receipt.json')
    blind, depth = CANDIDATES[name]
    parents = dict(protocol_sha256=sha(root/'protocol.json'), teacher_manifest_sha256=sha(root/'teacher-manifest.json'))
    if (receipt.get('name') != name or receipt.get('blind') is not blind
            or document['blind'] is not blind or document['training']['max_depth'] != depth
            or document['base_sha256'] != digest(read(root/'base-model.json'))
            or document['jaw_sha256'] != digest(read(root/'jaw-model.json'))
            or document['training']['parent_artifacts'] != parents
            or receipt.get('protocol_sha256') != parents['protocol_sha256']
            or receipt.get('teacher_manifest_sha256') != parents['teacher_manifest_sha256']
            or receipt.get('source_sha256') != source_hashes()
            or receipt.get('model_sha256') != sha(directory/'model.json')
            or receipt.get('training') != document['training']):
        raise ValueError('Candidate model, recipe, source or training lineage changed.')
    return directory, document


def _evaluation(root, protocol, variant, directory, model_path=None):
    cases = protocol['cases']['development']
    _folders_exact(directory, [case['id'] for case in cases])
    rows = [dict(case_id=case['id'], condition=case['condition'],
                 score=_attempt(root, case, variant, directory/case['id'], model_path)) for case in cases]
    return dict(schema=runner.SCHEMA, protocol_sha256=sha(Path(root)/'protocol.json'),
        source_sha256=source_hashes(), model_sha256=sha(model_path) if model_path else None,
        variant=variant, scheduled=len(rows), valid=sum(row['score']['valid'] for row in rows),
        successes=sum(row['score']['success'] for row in rows), rows=rows)


def development(root, assets, name=None):
    root = Path(root)
    with _locked(root):
        protocol = _study(root, mutable=True)
        requests = [('fixed', root/'development'/'fixed', None),
                    ('teacher', root/'development'/'teacher', None)]
        if name is not None:
            directory, document = _candidate(root, name)
            requests.append(('blind_clock' if document['blind'] else 'clock', directory/'development', directory/'model.json'))
        results = {}
        for variant, directory, model_path in requests:
            for case in protocol['cases']['development']:
                runner.execute(root, assets, case, variant, directory/case['id'], model_path=model_path)
            result = _evaluation(root, protocol, variant, directory, model_path)
            aggregate = (directory.parent/'development.json' if model_path is not None
                         else root/'development'/(variant+'.json'))
            results[variant] = _same_or_create(aggregate, result)
        return results


def _bound_files(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        relative = path.relative_to(root).as_posix()
        if relative.split('/')[0] == 'confirmation' or relative in ('.lifecycle.lock', 'frozen.json', 'report.json'):
            continue
        if path.is_symlink():
            raise ValueError('Study evidence cannot contain symlinks.')
        if path.is_file():
            result[relative] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def freeze(root, selected, blind):
    root = Path(root)
    with _locked(root):
        protocol = _study(root, mutable=True)
        if selected not in ('full4', 'full6') or blind not in ('blind4', 'blind6'):
            raise ValueError('Select one full clock and one matched blind clock explicitly.')
        if CANDIDATES[selected][1] != CANDIDATES[blind][1]:
            raise ValueError('Selected full/blind clocks must have the same tree recipe.')
        manifest, _ = _teachers(root, protocol)
        if read(root/'teacher-manifest.json') != manifest:
            raise ValueError('Training evidence changed before freeze.')
        _folders_exact(root/'candidates', CANDIDATES)
        models = {}
        for name in CANDIDATES:
            directory, document = _candidate(root, name)
            variant = 'blind_clock' if document['blind'] else 'clock'
            expected = _evaluation(root, protocol, variant, directory/'development', directory/'model.json')
            if read(directory/'development.json') != expected:
                raise ValueError('Candidate development aggregate changed or is incomplete.')
            models[name] = document
        for variant in ('fixed', 'teacher'):
            if read(root/'development'/(variant+'.json')) != _evaluation(root, protocol, variant, root/'development'/variant):
                raise ValueError('The full fixed/teacher development anchors are required.')
        recipe = ('algorithm', 'max_depth', 'min_leaf', 'max_bins', 'sample_count',
                  'scene_count', 'train_groups', 'group_samples', 'groups_sha256', 'weight_sum')
        if any(models[selected]['training'][key] != models[blind]['training'][key] for key in recipe):
            raise ValueError('Selected full/blind clocks differ in training budget or recipe.')
        _atomic_create(root/'selected.json', (root/'candidates'/selected/'model.json').read_bytes())
        _atomic_create(root/'blind.json', (root/'candidates'/blind/'model.json').read_bytes())
        frozen = dict(schema=runner.SCHEMA, protocol_sha256=sha(root/'protocol.json'),
            runtime_sha256=runner.runtime_hashes(), lifecycle_source_sha256=source_hashes(),
            base_model_raw_sha256=runner.BASE_SHA, jaw_model_raw_sha256=protocol['jaw_model_raw_sha256'],
            jaw_mode=protocol['jaw_mode'],
            selected=selected, blind=blind, selected_sha256=sha(root/'selected.json'),
            blind_sha256=sha(root/'blind.json'), files=_bound_files(root), physical_robot_ready=False)
        write(root/'frozen.json', frozen)
        return frozen


def _frozen(root):
    protocol = _study(root)
    frozen = read(Path(root)/'frozen.json')
    fields = {'schema', 'protocol_sha256', 'runtime_sha256', 'lifecycle_source_sha256',
              'base_model_raw_sha256', 'jaw_model_raw_sha256', 'jaw_mode', 'selected', 'blind',
              'selected_sha256', 'blind_sha256', 'files', 'physical_robot_ready'}
    if (type(frozen) is not dict or set(frozen) != fields
            or frozen.get('selected') not in ('full4', 'full6')
            or frozen.get('blind') not in ('blind4', 'blind6')
            or CANDIDATES[frozen['selected']][1] != CANDIDATES[frozen['blind']][1]
            or frozen.get('physical_robot_ready') is not False
            or frozen.get('base_model_raw_sha256') != runner.BASE_SHA
            or frozen.get('jaw_model_raw_sha256') != protocol['jaw_model_raw_sha256']
            or frozen.get('jaw_mode') != protocol['jaw_mode']
            or frozen.get('schema') != runner.SCHEMA or frozen.get('protocol_sha256') != sha(Path(root)/'protocol.json')
            or frozen.get('runtime_sha256') != runner.runtime_hashes()
            or frozen.get('lifecycle_source_sha256') != source_hashes()
            or frozen.get('selected_sha256') != sha(Path(root)/'selected.json')
            or frozen.get('blind_sha256') != sha(Path(root)/'blind.json')
            or frozen.get('selected_sha256') != sha(Path(root)/'candidates'/frozen['selected']/'model.json')
            or frozen.get('blind_sha256') != sha(Path(root)/'candidates'/frozen['blind']/'model.json')
            or frozen.get('files') != _bound_files(root)):
        raise ValueError('Frozen source, protocol, selected models or pre-confirmation evidence changed.')
    return protocol, frozen


def confirm(root, assets):
    root = Path(root)
    with _locked(root):
        protocol, _ = _frozen(root)
        models = {'clock': root/'selected.json', 'blind_clock': root/'blind.json'}
        for case in protocol['cases']['confirmation']:
            for variant in CONFIRM_VARIANTS:
                runner.execute(root, assets, case, variant, root/'confirmation'/case['id']/variant,
                               model_path=models.get(variant))
        return _report(root)


def _paired(groups, variant, comparator):
    values = [sum(group[variant])/4.-sum(group[comparator])/4. for group in groups]
    if len(values) != 32:
        raise ValueError('Paired intervals require exactly 32 complete start groups.')
    mean = float(np.mean(values))
    half = float(T975_DF31*np.std(values, ddof=1)/math.sqrt(32))
    return dict(comparator=comparator, group_differences=values, mean_difference=mean,
                interval=[mean-half, mean+half], interval_coverage=.975,
                simultaneous_coverage=.95, independent_groups=32,
                method='Two-sided t interval across XY group differences; Bonferroni for two primary contrasts.')


def _timing(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all() or np.any(array < 0.):
        raise ValueError('Timing rows must be finite nonnegative seconds.')
    return dict(count=len(array), p50_s=float(np.quantile(array, .50)) if len(array) else None,
                p95_s=float(np.quantile(array, .95)) if len(array) else None,
                p99_s=float(np.quantile(array, .99)) if len(array) else None,
                maximum_s=float(np.max(array)) if len(array) else None,
                over_20ms=int(np.count_nonzero(array > .02)),
                over_20ms_fraction=float(np.mean(array > .02)) if len(array) else None)


def _report(root):
    root = Path(root)
    protocol, frozen = _frozen(root)
    cases = protocol['cases']['confirmation']
    _folders_exact(root/'confirmation', [case['id'] for case in cases])
    rows, groups = [], {}
    timings = {variant: dict(inference=[], step=[], wall=[], platform=[]) for variant in CONFIRM_VARIANTS}
    summaries = {variant: dict(scheduled=0, valid=0, strict_successes=0, nominal_successes=0,
        post_grip_successes=0, terminal_only_successes=0, acquisition_failures=0, critical_contact_attempts=0,
        qualifying_lift_failures=0, retention_failures=0, loaded_loss_events=0, supported_loss_events=0, airborne_loss_events=0,
        restored_contact_events=0, post_grip_eligible=0, post_grip_activated=0, post_grip_effective=0,
        clock_hold_decisions=0, clock_advance_decisions=0, episodes_reaching_path_end=0,
        maximum_episode_clock_debt_s=0., nonactivation_reasons={}) for variant in CONFIRM_VARIANTS}
    for case in cases:
        _folders_exact(root/'confirmation'/case['id'], CONFIRM_VARIANTS)
        group = groups.setdefault(tuple(case['start_xy']), {v: [] for v in CONFIRM_VARIANTS})
        for variant in CONFIRM_VARIANTS:
            folder = root/'confirmation'/case['id']/variant
            path = root/('blind.json' if variant == 'blind_clock' else 'selected.json') if variant in ('clock', 'blind_clock') else None
            score = _attempt(root, case, variant, folder, path)
            episode = load_episode(folder/'episode.json.gz')
            native, grip_summary = score.get('native_recomputed', {}), score.get('grip_summary', {})
            intervention = score.get('post_grip_analysis', {})
            success = score['valid'] is True and score['success'] is True
            row = dict(case_id=case['id'], start_xy=case['start_xy'], goal_xy=case['goal_xy'],
                       condition=case['condition'], variant=variant, score=score,
                       receipt_sha256=sha(folder/'receipt.json'))
            rows.append(row)
            group[variant].append(int(success))
            summary = summaries[variant]
            summary['scheduled'] += 1
            summary['valid'] += score['valid'] is True
            summary['strict_successes'] += success
            summary['nominal_successes' if case['condition'] == 'nominal' else 'post_grip_successes'] += success
            summary['terminal_only_successes'] += bool(score.get('terminal_only', {}).get('success'))
            summary['acquisition_failures'] += not bool(native.get('grasp_seen'))
            summary['qualifying_lift_failures'] += not bool(native.get('lift_seen'))
            summary['critical_contact_attempts'] += native.get('unexpected_contact_native_ticks', 0) > 0
            summary['retention_failures'] += bool(native.get('retention_lost'))
            decisions = [action.get('clock_decision', {}) for action in episode.get('actions', [])]
            summary['clock_hold_decisions'] += sum(d.get('increment_s') == 0. for d in decisions)
            summary['clock_advance_decisions'] += sum(d.get('increment_s') == .02 for d in decisions)
            if decisions:
                summary['episodes_reaching_path_end'] += decisions[-1].get('progress_after_s') == 20.
                debt = max((d.get('physical_time_s', 0.)-d.get('progress_before_s', 0.) for d in decisions), default=0.)
                summary['maximum_episode_clock_debt_s'] = max(summary['maximum_episode_clock_debt_s'], debt)
            for target, source in (('loaded_loss_events', 'loss_events'), ('supported_loss_events', 'supported_events'),
                                   ('airborne_loss_events', 'airborne_events'), ('restored_contact_events', 'restored_events')):
                summary[target] += int(grip_summary.get(source, 0))
            if case['condition'] == 'post_grip':
                for target, source in (('post_grip_eligible', 'eligible'), ('post_grip_activated', 'activated'),
                                       ('post_grip_effective', 'effective')):
                    summary[target] += intervention.get(source) is True
                if not intervention.get('activated'):
                    reason = intervention.get('nonactivation_reason') or 'unexplained_nonactivation'
                    summary['nonactivation_reasons'][reason] = summary['nonactivation_reasons'].get(reason, 0)+1
            timing = episode.get('timing', {})
            timings[variant]['inference'].extend(timing.get('inference_samples_s', []))
            timings[variant]['step'].extend(timing.get('control_step_samples_s', []))
            timings[variant]['wall'].append(timing.get('wall_seconds', 0.))
            platform = episode.get('platform')
            if platform is not None and platform not in timings[variant]['platform']:
                timings[variant]['platform'].append(platform)
    if len(groups) != 32 or any(len(values) != 4 for group in groups.values() for values in group.values()):
        raise ValueError('Confirmation does not retain complete four-case start groups.')
    contrasts = [_paired(list(groups.values()), 'clock', other) for other in ('blind_clock', 'fixed')]
    for variant, summary in summaries.items():
        summary.update(nominal_scheduled=64, post_grip_scheduled=64,
                       invalid_or_aborted=128-summary['valid'], strict_success_rate=summary['strict_successes']/128.,
                       post_grip_effective_fraction=summary['post_grip_effective']/64.)
        values = timings[variant]
        summary['timing'] = dict(inference=_timing(values['inference']), full_step=_timing(values['step']),
                                episode_wall=_timing(values['wall']), platforms=values['platform'],
                                scope='Dense simulator CPU computation; excludes camera, transport and real actuation.')
    clock = summaries['clock']
    gates = dict(
        paired_benefit=all(c['mean_difference'] >= .10 and c['interval'][0] > 0. for c in contrasts),
        nominal=all((clock['nominal_successes']-summaries[v]['nominal_successes'])/64. >= -.05
                    for v in ('blind_clock', 'fixed')),
        critical_contacts=all(clock['critical_contact_attempts'] <= summaries[v]['critical_contact_attempts']
                              for v in ('blind_clock', 'fixed')),
        validity=all(summary['invalid_or_aborted'] == 0 for summary in summaries.values()),
        post_grip_exposure=all(summary['post_grip_effective_fraction'] >= .75 for summary in summaries.values()),
        compute=all(summary['timing']['full_step']['count'] == 128000
                    and summary['timing']['full_step']['over_20ms_fraction'] <= .01
                    for summary in summaries.values()))
    result = dict(schema=runner.SCHEMA, frozen_sha256=sha(root/'frozen.json'),
        protocol_sha256=sha(root/'protocol.json'), variants=summaries, contrasts=contrasts,
        start_groups=[dict(start_xy=list(point), successes=group) for point, group in groups.items()],
        rows=rows, gates=gates, advance_to_followup_simulation=all(gates.values()),
        qualified_robot_skill=False, physical_robot_ready=False,
        teacher_training=read(root/'teacher-manifest.json'),
        development={name: {k: v for k, v in read(root/'candidates'/name/'development.json').items() if k != 'rows'}
                     for name in CANDIDATES},
        development_anchors={name: {k: v for k, v in read(root/'development'/(name+'.json')).items() if k != 'rows'}
                             for name in ('fixed', 'teacher')},
        jaw_mode=protocol['jaw_mode'], limitations=protocol['limitations'])
    return _same_or_create(root/'report.json', result)


def report(root):
    with _locked(root):
        return _report(Path(root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'collect', 'train', 'development', 'freeze', 'confirm', 'report'))
    parser.add_argument('run', type=Path)
    parser.add_argument('--base-model', type=Path)
    parser.add_argument('--jaw-model', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--blind-name')
    parser.add_argument('--blind', action='store_true')
    parser.add_argument('--max-depth', type=int, choices=(4, 6), default=4)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        if args.base_model is None or args.jaw_model is None:
            parser.error('prepare needs --base-model and --jaw-model')
        result = prepare(args.run, args.base_model, args.jaw_model)
    elif args.command in ('collect', 'development', 'confirm'):
        if args.assets is None:
            parser.error(args.command+' needs --assets')
        if args.command == 'collect':
            result = collect(args.run, args.assets)
        elif args.command == 'development':
            result = development(args.run, args.assets, args.name)
        else:
            result = confirm(args.run, args.assets)
    elif args.command == 'train':
        if args.name is None:
            parser.error('train needs an explicit --name')
        result = train(args.run, args.name, blind=args.blind, max_depth=args.max_depth)
    elif args.command == 'freeze':
        if args.name is None or args.blind_name is None:
            parser.error('freeze needs --name and --blind-name')
        result = freeze(args.run, args.name, args.blind_name)
    else:
        result = report(args.run)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('rows', 'files', 'cases', 'source_sha256', 'training')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
