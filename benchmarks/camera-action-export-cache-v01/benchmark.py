#!/usr/bin/env python3
"""One registered training episode: strict export, cache build, three verified hits.

No fitting, teacher execution, simulator or confirmation-scene reads. Output is
fresh and outside the source study. Wall times include validation, not imports.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np

from dvidia_training import camera_action_cache as cache
from dvidia_training import camera_action_export as export
from dvidia_training import camera_action_model as model


def array_hash(value):
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study'); parser.add_argument('output'); parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    root = Path(args.study).resolve(); out = Path(args.output).absolute()
    if root == out or root in out.parents:
        raise ValueError('Proof output must be outside source study.')
    out.mkdir()  # fresh output, no rerun/overwrite
    def progress(event):
        with (out/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(event, sort_keys=True) + '\n')
    protocol = json.loads((root/'protocol.json').read_text())
    if sha256((root/'protocol.json').read_bytes()).hexdigest() != args.protocol_sha256:
        raise ValueError('Unexpected protocol identity.')
    entry = next(r for r in protocol['rosters']['demonstrations'] if r['case']['split'] == 'train')
    relative = 'episodes/demonstrations/' + entry['id']
    directory = root/relative
    metadata_before = {name: sha256((directory/name).read_bytes()).hexdigest()
                       for name in ('recording.json','integrity.json','rollout.json')}
    start = time.perf_counter(); strict = export.export_episode(directory); strict_s = time.perf_counter()-start
    progress(dict(stage='strict_export', seconds=strict_s))
    start = time.perf_counter()
    pins = cache.build_cache(root, out/'cache', expected_protocol_sha256=args.protocol_sha256,
                             split='train', stage='demonstrations', attempt_ids=[entry['id']])
    build_s = time.perf_counter()-start
    progress(dict(stage='strict_cache_build', seconds=build_s))
    (out/'pins.json').write_text(json.dumps(pins, sort_keys=True, indent=2)+'\n')
    (out/'pins.json').chmod(0o444)
    loads, checks = [], []
    for _ in range(3):
        start = time.perf_counter()
        loaded = cache.load_cache(out/'cache', expected_manifest_sha256=pins['manifest_sha256'],
             expected_protocol_sha256=args.protocol_sha256, expected_split='train', expected_source_sha256=pins['source_sha256'])
        loads.append(time.perf_counter()-start)
        progress(dict(stage='verified_hit', index=len(loads), seconds=loads[-1]))
        # Equality comparisons are outside the timed hit; byte/hash/semantic
        # validation performed by load_cache itself is inside every timed hit.
        checks.append(dict(features=np.array_equal(strict['features'], loaded['features']),
            targets=np.array_equal(strict['targets'], loaded['targets']), groups=strict['groups']==loaded['groups'],
            provenance=strict['provenance']==loaded['provenance'],
            dataset_sha256=model._dataset(strict, blind=False)[-1]==model._dataset(loaded, blind=False)[-1]))
    if not all(all(c.values()) for c in checks):
        raise ValueError('Cache output differs from strict export.')
    metadata_after = {name: sha256((directory/name).read_bytes()).hexdigest() for name in metadata_before}
    if metadata_before != metadata_after:
        raise ValueError('Source episode metadata changed during benchmark.')
    hardware = dict(platform=platform.platform(), machine=platform.machine(), python=platform.python_version(), numpy=np.__version__)
    if platform.system() == 'Darwin':
        for name in ('machdep.cpu.brand_string','hw.physicalcpu','hw.memsize'):
            query = subprocess.run(['sysctl','-n',name], check=False, text=True, capture_output=True)
            hardware[name] = query.stdout.strip() if query.returncode == 0 else {'status':'unavailable', 'reason':'host metadata query refused'}
    files = [p for p in directory.rglob('*') if p.is_file()]
    result = dict(schema='dvidia.camera-action-export-cache-benchmark/v1', scope=__doc__.strip(),
        attempt_id=entry['id'], split='train', protocol_sha256=args.protocol_sha256,
        source_identity=cache.source_identity(), pins=pins, episode_metadata_sha256=metadata_after,
        samples=len(strict['groups']), features_sha256=array_hash(strict['features']), targets_sha256=array_hash(strict['targets']),
        groups_sha256=model.digest(strict['groups']), provenance_sha256=model.digest(strict['provenance']),
        equality_checks=checks, strict_export_seconds=strict_s, strict_cache_build_seconds=build_s,
        verified_hit_seconds=loads, verified_hit_median_seconds=float(np.median(loads)),
        strict_to_hit_median_ratio=strict_s/float(np.median(loads)),
        source_episode_files=len(files), source_episode_bytes=sum(p.stat().st_size for p in files),
        cache_bytes=sum(p.stat().st_size for p in (out/'cache').iterdir()),
        peak_process_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_process_rss_units='bytes' if platform.system()=='Darwin' else 'KiB',
        hardware=hardware, raw_source_reverified_on_hit=False, native_runs=0, fitting_runs=0,
        qualification_changed=False,
        timing_caveats=['One fixed training episode, one strict export and one strict cache build, then three sequential cache hits.',
                       'Existing OS file cache was not flushed; shared laptop, no exclusive CPU reservation.',
                       'Imports are excluded; validation and source-hash checks are included. Equality comparison is outside timed hit.',
                       'RSS is cumulative process high water including strict export and build, not isolated cache memory.',
                       'No evidence of faster inference, improved control, whole-dataset throughput or hardware readiness.'])
    (out/'benchmark.json').write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('attempt_id','samples','strict_export_seconds','strict_cache_build_seconds',
                      'verified_hit_seconds','strict_to_hit_median_ratio','cache_bytes')}, indent=2))


if __name__ == '__main__':
    main()
