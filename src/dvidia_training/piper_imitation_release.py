"""Audit, summarize and seal the immutable PiPER imitation evidence.

This reads saved tapes; it neither executes policies nor advances physics.
The integrity manifest detects subsequent changes, not dishonest simulation data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import shutil

import numpy as np

from .piper_imitation_benchmark import (VARIANTS, RUNTIME_FILES, read, write, sha,
                                      load_episode, runtime_hashes)
from .piper_imitation_score import score_episode


def _inside(root, name):
    if not isinstance(name, str) or '\\' in name:
        raise ValueError('Expected a relative evidence path.')
    p = Path(name)
    if p.is_absolute() or any(s in ('', '.', '..') for s in p.parts):
        raise ValueError('Unsafe evidence path.')
    result = root/p
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Evidence must remain inside this study.')
    return result


def _stats(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Timing measurements must be finite nonnegative numbers.')
    return dict(samples=len(values), p50_s=float(np.percentile(a, 50)),
                p95_s=float(np.percentile(a, 95)), p99_s=float(np.percentile(a, 99)), maximum_s=float(max(a)))


def _wilson(successes, count):
    z = 1.959963984540054
    p = successes/count
    d = 1+z*z/count
    center = (p+z*z/(2*count))/d
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/d
    return [max(0., center-half), min(1., center+half)]


def audit(root):
    """Recompute every scheduled endpoint and verify the frozen bindings."""
    root = Path(root)
    report, protocol, frozen = (read(root/name) for name in ('report.json','protocol.json','frozen.json'))
    if report.get('status') != 'completed' or report.get('physical_robot_ready') is not False:
        raise ValueError('Only a completed simulation study may be audited.')
    for name, key in (('model/model.json','model_sha256'),('model/shuffled-model.json','shuffled_model_sha256'),
                      ('training/dataset.json','dataset_sha256'),('training/samples.npz','samples_sha256'),('protocol.json','protocol_sha256')):
        if sha(root/name) != frozen[key]:
            raise ValueError('Frozen artifact changed: '+name)
    if runtime_hashes() != frozen['runtime_sha256']:
        raise ValueError('Frozen runtime differs from the installed implementation.')
    if report['frozen_sha256'] != sha(root/'frozen.json') or report['protocol_sha256'] != sha(root/'protocol.json'):
        raise ValueError('Report is not bound to this freeze.')
    starts = {split:{tuple(c['start_xy']) for c in rows} for split,rows in protocol['cases'].items()}
    if any(starts[a]&starts[b] for a,b in (('train','development'),('train','confirmation'),('development','confirmation'))):
        raise ValueError('An initial position leaked across splits.')
    training = read(root/'training/dataset.json')
    if training != report['training']:
        raise ValueError('Report training receipt changed.')
    for row in training['rows']:
        p = _inside(root,row['episode'])
        if sha(p) != row['sha256'] or score_episode(load_episode(p),row['goal_xy']) != row['score']:
            raise ValueError('Teacher hash or independently recomputed score changed.')
    cases = {c['id']:c for c in protocol['cases']['confirmation']}
    expected = {(c,v) for c in cases for v in VARIANTS}
    observed = set()
    variants = {v:dict(scheduled=0, valid=0, successes=0, failures=0, reasons=Counter(),
                       inference=[], steps=[], wall=[], missed=0, rss=[]) for v in VARIANTS}
    for row in report['attempts']:
        identity = (row['id'],row['variant'])
        if identity not in expected or identity in observed:
            raise ValueError('Missing, duplicate or unexpected confirmation attempt.')
        observed.add(identity)
        case = cases[row['id']]
        if any(row[k] != case[k] for k in ('start_xy','goal_xy','seed')):
            raise ValueError('Attempt identity differs from the frozen case.')
        p = _inside(root,row['episode'])
        episode = load_episode(p)
        score = score_episode(episode,case['goal_xy'])
        saved = read(root/'attempt-scores'/(row['id']+'-'+row['variant']+'.json'))
        if sha(p)!=row['sha256'] or row != saved or score!=row['score']:
            raise ValueError('Attempt bytes or independently recomputed endpoint changed.')
        if row['variant']!='teacher' and episode.get('reason')!='interrupted':
            key = 'shuffled_model_sha256' if row['variant']=='shuffled_labels' else 'model_sha256'
            if episode.get('model_sha256')!=frozen[key] or episode.get('implementation_sha256')!=frozen['runtime_sha256']:
                raise ValueError('Attempt model/runtime binding differs from the freeze.')
        v = variants[row['variant']]
        v['scheduled']+=1; v['valid']+=int(episode.get('valid') is True)
        v['successes']+=int(score['success']); v['failures']+=int(not score['success'])
        v['reasons'][score['reason']]+=1
        timing = episode.get('timing',{})
        v['inference'].extend(timing.get('inference_samples_s',[]))
        v['steps'].extend(timing.get('control_step_samples_s',[]))
        if 'wall_seconds' in timing: v['wall'].append(timing['wall_seconds'])
        v['missed']+=timing.get('missed_20ms_boundaries',0)
        if timing.get('process_peak_rss_bytes') is not None: v['rss'].append(timing['process_peak_rss_bytes'])
    if observed != expected:
        raise ValueError('The full frozen confirmation roster has not completed.')
    for name,v in variants.items():
        if report['variants'][name]!={'scheduled':v['scheduled'],'successes':v['successes']}:
            raise ValueError('Published numerator or denominator differs from saved tapes.')
        v['descriptive_wilson_95_interval'] = _wilson(v['successes'],v['scheduled'])
        v['inference_compute'] = _stats(v.pop('inference'))
        v['control_step_compute'] = _stats(v.pop('steps'))
        v['episode_wall'] = _stats(v.pop('wall'))
        v['missed_20ms_compute_boundaries'] = v.pop('missed')
        v['process_peak_rss_bytes'] = max(v.pop('rss'), default=None)
        v['reasons'] = dict(v['reasons'])
    return dict(schema='dvidia.piper-imitation-audit/v1',report_sha256=sha(root/'report.json'),
                verified_attempts=len(observed),split_overlap=False,variants=variants,
                confidence_scope='Descriptive Wilson intervals only; these twelve deterministic nearby cases are not independent draws from a real-world task distribution.',
                timing_scope='CPU compute only; includes source-process overhead, excludes camera, transport and physical actuation. Peak RSS is cumulative process high-water mark.',
                physical_robot_ready=False)


def seal(root):
    root = Path(root)
    if (root/'integrity.json').exists():
        raise ValueError('Study is already sealed; do not overwrite its integrity manifest.')
    summary = audit(root)
    write(root/'audit.json',summary)
    model=read(root/'model/model.json'); frozen=read(root/'frozen.json')
    write(root/'model/skill.json',dict(schema='dvidia.learned-simulation-skill/v1',
        id='piper-block-placement-v01',publisher='DVIDIA',license='MIT',release_stage='research',
        model_file='model.json',model_sha256=sha(root/'model/model.json'),model_schema=model['schema'],
        feature_contract=model['feature_contract'],supervision_origin='synthetic_authored_controller',
        embodiment='piper',profile_lock_sha256=frozen['runtime_sha256']['robot_profiles.lock.json'],
        simulation_only=True,physical_robot_ready=False,learned_contact_recovery=False,
        native_control=dict(arm='six joint position targets, radians',jaw='one coupled jaw position target, metres',
                            control_dt_s=.02,timestep_s=.001,authored_native_servos=True),
        scope=dict(object='one 40 g rigid block',start_x_m=[.284,.314],start_y_m=[-.045,-.015],
                   known_goal_xy_m=[[.35,.08],[.40,-.08]],goal_z_m=.08,
                   novel_goals_tested=False,novel_objects_tested=False,disturbance_recovery_tested=False),
        confirmation=summary['variants']['learned'],report_sha256=sha(root/'report.json')))
    source = root/'sources'; source.mkdir()
    package = Path(__file__).parent
    extras = ('piper_imitation_release.py','piper_imitation_driver.py','piper_learned_skill.py','piper_imitation_viewer.py',
              'piper_imitation_viewer.js','piper_imitation_viewer.html','piper_imitation_viewer.bundle.js',
              'reference_replay.py')
    for name in (*RUNTIME_FILES,*extras):
        shutil.copyfile(package/name,source/name)
    repository = package.parents[1]
    shutil.copyfile(repository/'LICENSE',source/'LICENSE')
    shutil.copyfile(repository/'pyproject.toml',source/'pyproject.toml.txt')
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('Sealed evidence cannot contain symlinks.')
        if path.is_file(): files[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size,sha256=sha(path))
    write(root/'integrity.json',dict(schema='dvidia.piper-imitation-integrity/v1',files=files))
    return dict(files=len(files),integrity_sha256=sha(root/'integrity.json'),audit=summary)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','seal')); p.add_argument('--run',type=Path,required=True)
    args=p.parse_args(argv)
    import json
    print(json.dumps(audit(args.run) if args.command=='audit' else seal(args.run),indent=2,allow_nan=False))


if __name__=='__main__': main()
