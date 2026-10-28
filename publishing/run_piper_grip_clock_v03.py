"""Bind, then run four independent controller attempts per confirmation case.

This driver changes execution scheduling only. It uses spawn-isolated native
simulators, preserves the exact case/variant roster and original attempt runner,
and waits for all four controllers before the next case. Timing describes a
shared four-worker host, not isolated single-controller throughput.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from dvidia_training import piper_grip_clock_cli as lifecycle
from dvidia_training import piper_grip_clock_benchmark as runner
from dvidia_training.piper_imitation_benchmark import _atomic_create, read, write, sha


SCHEMA = 'dvidia.piper-clock-execution/v0.3'
WORKERS = 4


def declaration(root):
    return dict(schema=SCHEMA, workers=WORKERS, process_start_method='spawn',
        driver_sha256=sha(__file__), protocol_sha256=sha(Path(root)/'protocol.json'),
        schedule='Dispatch clock, blind_clock, fixed, teacher for one case; await all four before next case.',
        variants=list(lifecycle.CONFIRM_VARIANTS),
        attempts_per_controller=128, scheduled_attempts=512,
        timing_scope='Four concurrent independent simulator workers on a shared host; no isolated throughput or hard-real-time claim.',
        physical_robot_ready=False)


def bind(root):
    root = Path(root)
    with lifecycle._locked(root):
        lifecycle._study(root, mutable=True)
        _atomic_create(root/'execution-driver.py', Path(__file__).read_bytes())
        write(root/'execution-plan.json', declaration(root))


def _attempt(root, assets, case, variant):
    root = Path(root)
    model_path = root/('selected.json' if variant == 'clock' else 'blind.json') if variant in ('clock', 'blind_clock') else None
    return runner.execute(root, Path(assets), case, variant, root/'confirmation'/case['id']/variant,
                          model_path=model_path)


def confirm(root, assets):
    root, assets = Path(root).resolve(), Path(assets).resolve()
    with lifecycle._locked(root):
        protocol, frozen = lifecycle._frozen(root)
        if (read(root/'execution-plan.json') != declaration(root)
                or sha(root/'execution-driver.py') != sha(__file__)
                or 'execution-plan.json' not in frozen['files']
                or 'execution-driver.py' not in frozen['files']):
            raise ValueError('Parallel execution plan and driver must be bound before freeze.')
        errors = []
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context('spawn')) as pool:
            for case in protocol['cases']['confirmation']:
                futures = [pool.submit(_attempt, root, assets, case, variant)
                           for variant in lifecycle.CONFIRM_VARIANTS]
                # No policy result can stop the remaining paired attempts.
                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append((case['id'], exc))
        if errors:
            case_id, error = errors[0]
            raise RuntimeError(
                f'{len(errors)} attempt(s) did not complete; all confirmation cases were dispatched. '
                f'Retain evidence before reconciliation (first failure: {case_id}).'
            ) from error
        return lifecycle._report(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('bind', 'confirm'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'bind':
        bind(args.study)
    else:
        if args.assets is None:
            parser.error('confirm requires --assets')
        confirm(args.study, args.assets)


if __name__ == '__main__':
    main()
