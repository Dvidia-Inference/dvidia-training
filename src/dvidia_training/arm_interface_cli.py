"""One local interface for arm discovery, policy installation and diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


R1 = 'unitree-r1-7a'
R1_SIDES = ('left', 'right')


def _r1_description(side=None):
    description = {
        'id': R1, 'name': 'Unitree R1-7A', 'interface_id': 'urdf-kinematics',
        'arm_axes': 7, 'gripper_axes': 0, 'available_sides': list(R1_SIDES),
        'side': side, 'scope': 'Unloaded bare-arm kinematics in the vendor URDF base frame.',
        'skill_install_supported': False, 'hardware_transport': False,
        'physical_robot_ready': False, 'task_qualified': False,
    }
    if side is not None:
        from .unitree_r1_assets import source_info
        description['source'] = source_info(side)
    return description


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list', help='List simulation adapters. Add --include-references for bare-arm kinematics.')
    listing.add_argument('--include-references', action='store_true',
                         help='Also show the Unitree R1-7A bare-arm reference.')
    describe = commands.add_parser('describe')
    describe.add_argument('robot')
    describe.add_argument('--side', choices=R1_SIDES, help='Select the left or right R1-7A reference.')
    pose = commands.add_parser('pose', help='Compute the offline pose of the bare arm’s terminal link.')
    kinematics = commands.add_parser('benchmark-kinematics', help='Check bare-arm poses against MuJoCo offline.')
    for command in (pose, kinematics):
        command.add_argument('robot', choices=(R1,))
        command.add_argument('--side', required=True, choices=R1_SIDES)
    pose.add_argument('--joints', required=True, type=float, nargs=7, metavar='RAD',
                      help='Seven joint angles in URDF chain order, in radians.')
    for name in ('check', 'install'):
        command = commands.add_parser(name)
        command.add_argument('--model', type=Path, required=True)
        command.add_argument('--robot', required=True)
        command.add_argument('--interface', default='mujoco-position-v1')
        if name == 'install':
            command.add_argument('--assets', type=Path, required=True)
            command.add_argument('--output', type=Path, required=True)
    inspect = commands.add_parser('inspect')
    inspect.add_argument('installation', type=Path)
    inspect.add_argument('--assets', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('installation', type=Path)
    run.add_argument('--assets', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--start-x', type=float, default=.30)
    run.add_argument('--start-y', type=float, default=-.03)
    run.add_argument('--goal', type=int, choices=(0, 1), default=0)
    bench = commands.add_parser('benchmark')
    bench.add_argument('--assets', type=Path, required=True)
    bench.add_argument('--reference-assets', type=Path)
    bench.add_argument('--robots', nargs='+', default=['piper', 'so101', 'ur5e-2f85'])
    bench.add_argument('--duration', type=float, default=.4)
    bench.add_argument('--repeats', type=int, default=2)
    bench.add_argument('--output', type=Path, required=True)
    task = commands.add_parser('assess-task', help='Assess supplied lift conditions; no robot commands.')
    task.add_argument('document', type=Path)
    commands.add_parser('task-template', help='Print an explicitly synthetic lift-planning example.')
    args = parser.parse_args(argv)
    try:
        if args.command in ('check', 'install') and args.robot == R1:
            raise ValueError('Unitree R1-7A supports bare-arm kinematics. '
                             'The PiPER placement skill requires the PiPER arm and gripper.')
        if args.command == 'list':
            from .arm_interfaces import catalog
            result = catalog()
            if args.include_references:
                result.append(_r1_description())
        elif args.command == 'describe':
            if args.robot == R1:
                result = _r1_description(args.side)
            else:
                if args.side is not None:
                    raise ValueError('--side applies only to the Unitree R1-7A reference.')
                from .arm_interfaces import contract
                result = contract(args.robot)
        elif args.command == 'pose':
            from .unitree_r1_kinematics import pose
            result = pose(args.side, args.joints)
        elif args.command == 'benchmark-kinematics':
            from .unitree_r1_kinematics import benchmark
            result = benchmark(args.side)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result['passed'] else 1
        elif args.command == 'check':
            from .arm_skill_install import check_model
            result = check_model(args.model, args.robot, interface=args.interface)
        elif args.command == 'install':
            from .arm_skill_install import install_model
            result = install_model(args.model, args.robot, args.assets, args.output, interface=args.interface)
        elif args.command == 'inspect':
            from .arm_skill_install import inspect_installation
            result = inspect_installation(args.installation, args.assets)
        elif args.command == 'run':
            from .arm_skill_install import run_installation
            result = run_installation(args.installation, args.assets, args.output,
                                      start_x=args.start_x, start_y=args.start_y, goal=args.goal)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result['success'] else 1
        elif args.command == 'benchmark':
            from .arm_interface_benchmark import benchmark
            assets = {name: args.reference_assets if name == 'ur5e-2f85' and args.reference_assets else args.assets
                      for name in args.robots}
            result = benchmark(assets, args.output, profiles=args.robots,
                               duration=args.duration, repeats=args.repeats)
            summary = result['summary']
            print(json.dumps({'report': str(args.output/'report.json'),
                              'attempts': summary['attempts'],
                              'valid_attempts': summary['valid_attempts'],
                              'failed_attempts': summary['failed_attempts'],
                              'physical_robot_ready': False}, indent=2))
            return 0 if summary['failed_attempts'] == 0 else 1
        else:
            from .task_feasibility import assess, template
            from .reference_replay import _read, _decode
            result = template() if args.command == 'task-template' else assess(_decode(_read(args.document, 64*1024)))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError, ImportError) as exc:
        print(f'Arm request rejected: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
