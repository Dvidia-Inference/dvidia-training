"""Resume a frozen study without repeating completed native attempts.

The original driver compares tuple-valued live configuration to its JSON list
representation after saving a tape. This adapter handles only that serialization
guard, verifies the persisted identity, then resumes through the original API.
It never changes frozen files, policy commands, simulation or scoring rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import piper_imitation_benchmark as study


def complete(root, assets):
    root=Path(root)
    if (root/'report.json').exists():
        raise ValueError('Confirmation already complete; no attempt will be repeated.')
    protocol=study.read(root/'protocol.json')
    frozen=study.read(root/'frozen.json')
    schedule=[(c,v) for c in protocol['cases']['confirmation'] for v in study.VARIANTS]
    for _ in range(len(schedule)+1):
        try:
            result=study.confirm(root,assets,resume=(root/'confirmation-opened.json').exists())
            records=list((root/'driver-recovery').glob('*.json')) if (root/'driver-recovery').exists() else []
            study.write(root/'driver-recovery.json',dict(
                schema='dvidia.piper-imitation-driver-recovery/v1', frozen_sha256=study.sha(root/'frozen.json'),
                adapter_sha256=study.sha(Path(__file__)), serialization_guard_resumes=len(records),
                completed_tapes_reexecuted=0, frozen_runtime_changed=False,
                issue='Live configuration tuples compare unequal to persisted JSON lists; resume scores the same saved tape.',
                records=[p.relative_to(root).as_posix() for p in sorted(records)]))
            return result
        except ValueError as exc:
            if str(exc)!='Confirmation tape identity, model or runtime differs from the schedule.':
                raise
            if study.runtime_hashes()!=frozen['runtime_sha256']:
                raise ValueError('Frozen runtime changed; recovery refused.') from exc
            unscored=[]
            for case,variant in schedule:
                stem=case['id']+'-'+variant
                tape=root/'attempts'/(stem+'.json.gz')
                score=root/'attempt-scores'/(stem+'.json')
                if tape.exists() and not score.exists(): unscored.append((case,variant,tape,stem))
            if len(unscored)!=1:
                raise ValueError('Serialization recovery requires exactly one completed unscored tape.') from exc
            case,variant,tape,stem=unscored[0]
            episode=study.load_episode(tape)
            marker=study.read(root/'attempt-starts'/(stem+'.json'))
            expected_runtime=({n:study.sha(Path(study.__file__).with_name(n)) for n in study.TEACHER_FILES}
                              if variant=='teacher' else frozen['runtime_sha256'])
            expected_config=json.loads(json.dumps(study.asdict(study.config(case))))
            key='shuffled_model_sha256' if variant=='shuffled_labels' else 'model_sha256'
            if (marker.get('frozen_sha256')!=study.sha(root/'frozen.json')
                    or episode.get('config')!=expected_config or episode.get('seed')!=case['seed']
                    or episode.get('implementation_sha256')!=expected_runtime
                    or (variant=='teacher' and (episode.get('control')!='placement' or episode.get('policy_mode')!='baseline'))
                    or (variant!='teacher' and (episode.get('policy_kind')!=variant or episode.get('model_sha256')!=frozen[key]))):
                raise ValueError('Persisted tape identity does not qualify for serialization recovery.') from exc
            record=root/'driver-recovery'/(stem+'.json')
            if record.exists():
                raise ValueError('The same serialization guard repeated without forward progress.') from exc
            study.write(record,dict(id=case['id'],variant=variant,tape_sha256=study.sha(tape),
                frozen_sha256=study.sha(root/'frozen.json'),adapter_sha256=study.sha(Path(__file__)),
                reason='configuration_tuple_list_comparison',physical_attempt_repeated=False))
    raise ValueError('Exceeded the bounded frozen confirmation schedule.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True); p.add_argument('--assets',type=Path,required=True)
    a=p.parse_args(argv)
    result=complete(a.run,a.assets)
    print(json.dumps(result['variants']))


if __name__=='__main__': main()
