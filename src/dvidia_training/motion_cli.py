"""Local image-motion analysis, causal forecasts, benchmark and replay."""
from __future__ import annotations

import argparse
from pathlib import Path

from .footage_data import canonical


def _box(value):
    try:
        values = [float(v) for v in value.split(',')]
        from .observations import box
        return box(values)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('Use normalized x1,y1,x2,y2, for example 0.2,0.2,0.5,0.5.') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    analyze = commands.add_parser('analyze', help='Track one manually seeded region in permitted local footage.')
    analyze.add_argument('manifest', type=Path)
    analyze.add_argument('--episode', required=True)
    analyze.add_argument('--box', required=True, type=_box)
    analyze.add_argument('--label', default='object')
    analyze.add_argument('--sample-hz', type=float, default=30.0)
    analyze.add_argument('--output', type=Path, required=True)
    benchmark = commands.add_parser('benchmark', help='Run authored synthetic tracking/forecast fixtures, not detector accuracy.')
    benchmark.add_argument('--output', type=Path, required=True)
    inspect = commands.add_parser('inspect', help='Validate hashes, source media and causal rows.')
    inspect.add_argument('directory', type=Path)
    studio = commands.add_parser('studio', help='Read-only local video replay with forecast overlays.')
    studio.add_argument('directory', type=Path)
    studio.add_argument('--port', type=int, default=8286)
    args = parser.parse_args(argv)
    try:
        if args.command == 'analyze':
            from .motion_video import analyze_video
            result = analyze_video(args.manifest, args.episode, args.output,
                                   seed_box=args.box, label=args.label, sample_hz=args.sample_hz)
            print(canonical({'status': result['status'], 'frames': len(result['rows']),
                             'claim_scope': 'manually-seeded fixed-camera image motion',
                             'output': str(args.output)}).decode())
            if result['status'] != 'completed':
                return 1
        elif args.command == 'benchmark':
            from .motion_benchmark import run_benchmark
            report = run_benchmark(args.output)
            print(canonical({'kind': report['kind'], 'cases': len(report['cases']),
                             'scope': 'authored synthetic tracking and prediction fixtures',
                             'output': str(args.output)}).decode())
            if any(c['status'] != 'completed' for c in report['cases']):
                return 1
        elif args.command == 'inspect':
            from .motion_studio import load_motion_bundle
            report, episodes = load_motion_bundle(args.directory)
            result = {'integrity': 'verified', 'independent_real_motion_accuracy': 'not_assessed',
                      'metric_speed_available': False, 'robot_control_qualified': False,
                      'episodes': [{'id': eid, 'status': e['run']['status'],
                                    'frames': len(e['run']['rows'])} for eid, e in episodes.items()],
                      'scope': 'authored synthetic fixtures' if report else 'permitted local recording'}
            print(canonical(result).decode())
        else:
            from .motion_studio import serve
            serve(args.directory, args.port)
    except (ValueError, OSError, RuntimeError) as error:
        # Provider bodies, credentials and local path traces are never receipts.
        message = str(error) if isinstance(error, (ValueError, RuntimeError)) else 'Local evidence could not be accessed.'
        parser.exit(2, f'Motion lab: {message}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
