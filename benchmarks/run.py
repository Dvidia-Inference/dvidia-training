"""Reproduce a small CPU throughput and Python-process memory measurement.

Generated footage and repeated test metrics are software checks, not independent
task trials or a validated minimum hardware configuration.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
try:
    import resource
except ImportError:
    resource = None

from dvidia_training.footage_example import create_example
from dvidia_training.footage_pipeline import _fresh, inspect_run, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error('Use 1–5 repetitions.')
    output = _fresh(args.output)
    creation_start = time.perf_counter()
    create_example(output / 'source', clips=10)
    creation = time.perf_counter() - creation_start
    trials = []
    for index in range(args.repeats):
        started = time.perf_counter()
        summary = run(output / 'source', output / f'run-{index}', seed=17)
        run_wall = time.perf_counter() - started
        inspection_started = time.perf_counter()
        inspect_run(output / f'run-{index}')
        trials.append({'repeat': index, 'pipeline_wall_seconds': run_wall,
                       'pipeline_report_seconds': summary['elapsed_seconds'],
                       'integrity_inspection_seconds': time.perf_counter() - inspection_started,
                       'test_rgb_mse': summary['visual']['test']['learned_rgb_mse'],
                       'persistence_rgb_mse': summary['visual']['test']['persistence_rgb_mse'],
                       'counts': summary['counts'], 'gpu_required': False})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
    if rss is not None and platform.system() != 'Darwin':
        rss *= 1024
    result = {'schema_version': 1, 'kind': 'dvidia.small-cpu-pipeline-benchmark',
              'scope': 'synthetic-software-throughput', 'gpu_used': False,
              'hardware': {'system': platform.system(), 'architecture': platform.machine(),
                           'python': platform.python_version()},
              'dependencies': {'numpy': importlib.metadata.version('numpy'),
                               'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]},
              'synthetic_source_creation_seconds': creation,
              'python_process_lifetime_peak_rss_bytes': rss, 'trials': trials,
              'limits': ['Python RSS excludes FFmpeg/ffprobe child processes and is not total memory or a minimum requirement.',
                         'One machine, default tiny-resolution learner, ten synthetic clips; no hardware scaling claim.',
                         'The same held-out recording is re-scored for repeatability, not independent success evidence.',
                         'Wall time includes intake, fit and export; inspection is timed separately; dependency installation/imports are excluded.']}
    (output / 'benchmark.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
