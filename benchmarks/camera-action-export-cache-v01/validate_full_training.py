"""One bounded full-training cache validation; no fitting or native simulation.

The unchanged strict exporter runs once per selected training recording, inside
each cache build. Previously sealed model dataset identities are the independent
golden references. No redundant strict export or held-out recording is read.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

import numpy as np
from dvidia_training import camera_action_cache as cache
from dvidia_training import camera_action_export as export
from dvidia_training import camera_action_model as model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('study', type=Path, help='Exact sealed v0.1 study (protocol and episode hashes remain pinned).')
parser.add_argument('output', type=Path, help='Fresh output directory outside the study; its parent must exist.')
parser.add_argument('--expected-venv', required=True, type=Path, help='Independently verified dev5 environment prefix.')
parser.add_argument('--working-directory', required=True, type=Path, help='Expected neutral directory outside the checkout.')
args = parser.parse_args()
if not all(path.is_absolute() for path in (args.study, args.output, args.expected_venv, args.working_directory)):
    parser.error('All paths must be absolute; use the real virtual-environment and working-directory paths.')
STUDY = args.study
OUTPUT = args.output
PROOFS = OUTPUT.parent
VENV = args.expected_venv
WORKING_DIRECTORY = args.working_directory
SOURCE_COMMIT = 'cc6746a95669cfd69a49e32a3e6c2e5a42ce2546'
SOURCE_SHA = '0ab01145c66d651eaca96d763b4bec1706327f2cdd4bd8665d05056cbdd6f677'
CACHE_SHA = '9191f2eb9f96bd09d4f711c31966a583bd938853a874e6018f8ddebfb0bc00d8'
PROTOCOL_SHA = '576851e321034a0f6d3cb4d0818b3a5ca97975fcc6f2c7fa30b07ffd203a2d76'
STAGE_SHA = {
    'demonstrations': '5d57eac599f1686d2bdb15bb53ceebe09e899ec71dfe87039aa85de7f853c06b',
    'corrections': 'ae0a5653f728f147f929325c60318cb3d164c6d97d846fb96ac07ee9d27ad5cb',
    'train-initial': '86be5d9c17af43c35e97677d819b1433faed54ccf930078afc0f785139acc6a9',
    'train-corrected': '9b37a37d7b49fceb595f776d59579986f7579c7135f810b8d706199d5e4d6ddc',
}
GOLDEN = {
    'initial/full': {
        'model_sha256': '1d1f80376c89995513b518b36c4f8eca9e78efec3d3e68a10ee9988e84f4f038',
        'dataset_sha256': '07b175f78e4e77cbdde9a461a4bcae4dcb0db7d389ee4f2e702ff28db8bb3dfc',
        'provenance_sha256': '7103ad53111cb114cdcd164cfa15bbbf359ebc4ae66e7456ab7c1f38820b2864',
        'samples': 64000,
    },
    'initial/blind': {
        'model_sha256': 'd30b0ae10a4fbdc34a73122a34d73fd9246ee4b87c90c2b3b1eb3a98888d707e',
        'dataset_sha256': '759fe1c0610c1768f1455ed6cdcc6305be0d463b2b1e9c48a327f4f45b9c50c7',
        'provenance_sha256': '7d5279ffd186f018a41641a18bf5d6bbf010036114abb9e902d25aadc9ca3716',
        'samples': 64000,
    },
    'corrected/full': {
        'model_sha256': 'd176b391b61b5e331bcfe1676107c53f4c0ae6110414dfed4b07d57cbb883b0e',
        'dataset_sha256': 'b500b3ba348bba44cc79e377266f92910bd7e0165c36cd8b2fe260909b30a2f9',
        'provenance_sha256': '52cbeda62e3aa50e8cd6a34688fba7583f9fcec927cb0eb434a6907c99695f4f',
        'samples': 96000,
    },
    'corrected/blind': {
        'model_sha256': 'c3593d83b5505127a31304c8965914e01661d419dfc6c76a17a6c9bbac418806',
        'dataset_sha256': 'efd1760344b88c9eb61045454968564bb45a8cd38037e2bdcdf55b343eab3903',
        'provenance_sha256': '60edeb423d3810419e5ac971511c237c77e619a309637402b0974e658d7cb4a4',
        'samples': 96000,
    },
}
EXPECTED_RAW = {
    'demonstrations': {'episodes': 64, 'samples': 64000, 'files': 384704, 'bytes': 12638094084},
    'corrections': {'episodes': 32, 'samples': 32000, 'files': 192352, 'bytes': 6333838508},
}
RESERVE = 30 * 1024**3
MAX_CACHE_BYTES = 128 * 1024**2


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    body = path.read_bytes()
    return {'bytes': len(body), 'sha256': sha256(body).hexdigest()}


def read_pinned(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a plain pinned metadata file.')
    body = path.read_bytes()
    if sha256(body).hexdigest() != expected:
        raise ValueError('Independent input pin differs: ' + str(path))
    return json.loads(body)


def write_fresh(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def save_status(state):
    temporary = OUTPUT / 'status.pending.json'
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    temporary.replace(OUTPUT / 'status.json')


def progress(state, stage, **fields):
    row = {'at': now(), 'stage': stage, **fields}
    with (OUTPUT / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    state.update(updated_at=row['at'], last_stage=stage, free_bytes=shutil.disk_usage(OUTPUT).free)
    if state['free_bytes'] < RESERVE:
        raise ValueError('Preserve the 30 GiB reserve.')
    save_status(state)
    print(json.dumps(row, sort_keys=True), flush=True)


def preflight():
    if Path(sys.prefix) != VENV or not Path(cache.__file__).is_relative_to(VENV):
        raise ValueError('Use the existing independently verified dev5 installation.')
    if 'PYTHONPATH' in os.environ or Path.cwd() != WORKING_DIRECTORY:
        raise ValueError('Run from the supplied neutral directory outside the checkout, with PYTHONPATH unset.')
    if cache.source_digest() != SOURCE_SHA or identity(Path(cache.__file__))['sha256'] != CACHE_SHA:
        raise ValueError('Installed cache/runtime/feature source differs from reviewed cc6746a bytes.')
    if model.FEATURE_COUNT != 269 or (STUDY / 'worker.lock').exists():
        raise ValueError('Unexpected feature contract or active study worker.')
    protocol = read_pinned(STUDY / 'protocol.json', PROTOCOL_SHA)
    stages = {}
    for stage, digest in STAGE_SHA.items():
        stages[stage] = read_pinned(STUDY / 'stages' / (stage + '.json'), digest)
        seal = json.loads((STUDY / 'stages' / (stage + '.integrity.json')).read_bytes())
        if seal != {'protocol_sha256': PROTOCOL_SHA, 'sha256': digest}:
            raise ValueError('Sealed stage differs from independently pinned evidence.')
    rosters, raw = {}, {}
    for stage in EXPECTED_RAW:
        # Protocol owns training membership and order; cache build revalidates
        # every registration, stage, eligibility, source byte and sidecar binding.
        rows = [row for row in stages[stage]['rows'] if row['case']['split'] == 'train']
        registered = [row for row in protocol['rosters'][stage] if row['case']['split'] == 'train']
        if [row['id'] for row in rows] != [row['id'] for row in registered]:
            raise ValueError('Ordered training roster differs.')
        total_files = total_bytes = 0
        for row in rows:
            if (row.get('complete') is not True or row.get('initial_anchor_available') is not True
                    or row.get('initial_anchor_binding_valid') is not True
                    or row['recorded_complete_intervals'] != 1000
                    or (stage == 'demonstrations' and row.get('strict_success') is not True)):
                raise ValueError('Do not replace or drop an ineligible training episode.')
            expected_path = f'episodes/{stage}/{row["id"]}'
            if row['path'] != expected_path:
                raise ValueError('Training path differs from sealed registered identity.')
            path = STUDY / expected_path / 'integrity.json'
            inventory = read_pinned(path, row['integrity_sha256'])
            total_files += len(inventory['files']) + 1
            total_bytes += sum(item['bytes'] for item in inventory['files'].values()) + path.stat().st_size
        raw[stage] = {'episodes': len(rows), 'samples': sum(row['recorded_complete_intervals'] for row in rows),
                      'files': total_files, 'bytes': total_bytes}
        if raw[stage] != EXPECTED_RAW[stage]:
            raise ValueError('Raw training inventory differs from the prepared bounds.')
        rosters[stage] = [row['id'] for row in rows]
    training = {}
    for name, expected in GOLDEN.items():
        stage, variant = name.split('/')
        document = read_pinned(STUDY / 'models' / name / 'model.json', expected['model_sha256'])
        reference = stages['train-' + stage]['models'][variant]
        record = document['training']
        expected_episodes = rosters['demonstrations'] + (rosters['corrections'] if stage == 'corrected' else [])
        if (reference['model_sha256'] != expected['model_sha256'] or reference['training'] != record
                or reference['training_episodes'] != expected_episodes
                or record['sample_count'] != expected['samples']
                or record['dataset_sha256'] != expected['dataset_sha256']
                or record['provenance_sha256'] != expected['provenance_sha256']
                or model.digest(record['provenance']) != expected['provenance_sha256']):
            raise ValueError('Model dataset reference differs from sealed fit-stage evidence.')
        training[name] = record
    return rosters, raw, training


def match_golden(dataset, name, training):
    _, _, groups, provenance, digest = model._dataset(dataset, blind=name.endswith('/blind'))
    expected = GOLDEN[name]
    if (digest != expected['dataset_sha256'] or len(groups) != expected['samples']
            or provenance != training[name]['provenance']
            or model.digest(provenance) != expected['provenance_sha256']
            or list(dict.fromkeys(groups)) != training[name]['train_groups']):
        raise ValueError('Loaded complete dataset does not equal its independently sealed model reference: ' + name)
    return {'name': name, 'passed': True, 'dataset_sha256': digest,
            'model_sha256': expected['model_sha256'], 'samples': len(groups),
            'groups': len(set(groups)), 'episodes': len(provenance['episodes']),
            'provenance_sha256': model.digest(provenance)}


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink() or STUDY in OUTPUT.parents:
        raise ValueError('Fresh output outside study required; retain any interrupted prefix.')
    if shutil.disk_usage(PROOFS).free < RESERVE + 256 * 1024**2:
        raise ValueError('Require 30 GiB reserve plus bounded cache/log headroom.')
    OUTPUT.mkdir(mode=0o700)
    lock = OUTPUT / 'worker.lock'
    write_fresh(lock, {'pid': os.getpid(), 'started_at': now(), 'workers': 1})
    state = {'schema': 'dvidia.full-training-cache-process/v1', 'state': 'running',
             'pid': os.getpid(), 'started_at': now(), 'native_runs': 0, 'fitting_runs': 0}
    save_status(state)
    started = time.perf_counter()
    try:
        rosters, raw, training = preflight()
        retained = {'source_commit': SOURCE_COMMIT, 'source_sha256': SOURCE_SHA,
                    'cache_source_sha256': CACHE_SHA, 'source_identity': cache.source_identity(),
                    'protocol_sha256': PROTOCOL_SHA, 'stage_sha256': STAGE_SHA,
                    'model_references': GOLDEN, 'attempt_ids': rosters, 'raw_training_inventory': raw,
                    'validation_script': identity(Path(__file__))}
        write_fresh(OUTPUT / 'independent-input-pins.json', retained)
        (OUTPUT / 'independent-input-pins.json').chmod(0o444)
        progress(state, 'preflight', episodes=96, samples=96000, raw_inventory=raw)
        pins, builds = {}, []
        for stage in ('demonstrations', 'corrections'):
            began = time.perf_counter()
            pin = cache.build_cache(STUDY, OUTPUT / (stage + '-cache'),
                expected_protocol_sha256=PROTOCOL_SHA, split='train', stage=stage,
                attempt_ids=rosters[stage], blind=False)
            elapsed = time.perf_counter() - began
            if pin['sample_count'] != EXPECTED_RAW[stage]['samples'] or pin['source_sha256'] != SOURCE_SHA:
                raise ValueError('Build pin differs from registered scope.')
            write_fresh(OUTPUT / (stage + '-pins.json'), pin)
            (OUTPUT / (stage + '-pins.json')).chmod(0o444)
            pins[stage] = pin
            sizes = sum(path.stat().st_size for path in (OUTPUT / (stage + '-cache')).iterdir())
            builds.append({'stage': stage, 'seconds': elapsed, 'cache_bytes': sizes, 'pins': pin})
            progress(state, 'cache_built', cache_stage=stage, seconds=elapsed, cache_bytes=sizes, pins=pin)
            gc.collect()
        if sum(row['cache_bytes'] for row in builds) > MAX_CACHE_BYTES:
            raise ValueError('Derived caches exceed the declared 128 MiB bound.')
        rounds = []
        for index in range(3):
            round_start = time.perf_counter()
            datasets, loads = {}, {}
            for stage in ('demonstrations', 'corrections'):
                pin = pins[stage]  # retained independently, never self-read from cache
                began = time.perf_counter()
                datasets[stage] = cache.load_cache(OUTPUT / (stage + '-cache'),
                    expected_manifest_sha256=pin['manifest_sha256'], expected_protocol_sha256=PROTOCOL_SHA,
                    expected_split='train', expected_source_sha256=SOURCE_SHA)
                loads[stage] = time.perf_counter() - began
            began = time.perf_counter()
            combined = export.combine_datasets([datasets['demonstrations'], datasets['corrections']])
            combine_seconds = time.perf_counter() - began
            began = time.perf_counter()
            checks = [match_golden(datasets['demonstrations'], 'initial/full', training),
                      match_golden(combined, 'corrected/full', training)]
            for dataset, name in ((datasets['demonstrations'], 'initial/blind'), (combined, 'corrected/blind')):
                blinded = export.blind_dataset(dataset)
                checks.append(match_golden(blinded, name, training))
                del blinded
            comparison_seconds = time.perf_counter() - began
            rounds.append({'round': index + 1, 'validated_load_seconds': loads,
                           'combine_seconds': combine_seconds,
                           'golden_comparison_and_blind_projection_seconds': comparison_seconds,
                           'whole_round_seconds': time.perf_counter() - round_start, 'checks': checks})
            progress(state, 'validated_round', **rounds[-1])
            del dataset, datasets, combined
            gc.collect()
        # Recheck small source/stage/model/episode manifest pins without a second
        # raw export. A cache hit never asserts that raw frames were revalidated.
        if preflight() != (rosters, raw, training):
            raise ValueError('Independent metadata changed during measurement.')
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {'schema': 'dvidia.full-training-cache-validation/v1', 'passed': True,
                  **retained, 'builds': builds, 'rounds': rounds,
                  'raw_training_files': sum(row['files'] for row in raw.values()),
                  'raw_training_bytes': sum(row['bytes'] for row in raw.values()),
                  'derived_cache_bytes': sum(row['cache_bytes'] for row in builds),
                  'peak_process_rss_bytes': int(peak) * (1 if sys.platform == 'darwin' else 1024),
                  'hardware': {'platform': platform.platform(), 'machine': platform.machine(),
                               'python': platform.python_version(), 'numpy': np.__version__},
                  'wall_seconds': time.perf_counter() - started, 'raw_source_reverified_on_hit': False,
                  'native_runs': 0, 'fitting_runs': 0, 'extra_strict_exports': 0,
                  'development_or_confirmation_recordings_read': False, 'qualification_changed': False,
                  'comparison_scope': 'Canonical float64 numeric values, ordered groups and full provenance equal the four independently sealed model dataset identities; cache float32 storage is not original combined NPY byte identity.',
                  'timing_scope': 'Shared-host offline preparation. Imports excluded; builds include strict source, sidecar and saved-trace score validation. Each hit includes cache/source validation. Combiner and golden checks reported separately. No cold-cache flush, isolated CPU reservation or matched strict-export timing; no speedup claim.',
                  'memory_scope': 'Whole-process high-water RSS across metadata inspection, both builds, loads, combination and golden checks; not isolated per-stage memory.',
                  'source_read_scope': 'Unique raw corpus size is metadata-bound, not measured physical I/O. Strict exporter rereads some metadata, sidecars and native trace bytes; no physics is executed.'}
        write_fresh(OUTPUT / 'result.json', result)
        state.update(state='completed', completed_at=now(), elapsed_seconds=time.perf_counter()-started,
                     result=identity(OUTPUT / 'result.json'), exit_code=0)
        save_status(state)
        lock.unlink()
        print(json.dumps({'passed': True, 'result': str(OUTPUT / 'result.json'),
                          'result_identity': identity(OUTPUT / 'result.json')}, sort_keys=True), flush=True)
    except BaseException as exc:
        state.update(state='interrupted_or_failed', updated_at=now(), exception_type=type(exc).__name__,
                     reason=str(exc), elapsed_seconds=time.perf_counter()-started)
        save_status(state)
        raise  # retain all prefixes, independent pins and worker lock


if __name__ == '__main__':
    main()
