"""Load reusable robot and synthetic shoe assets into local CPU simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Print the bundled asset catalog; no native engine or network required")
    fetch = commands.add_parser("fetch", help="Explicitly provision hash-pinned upstream arm models")
    fetch.add_argument("robot", choices=("ur5e", "piper", "so101"))
    fetch.add_argument("--output", type=Path, required=True)
    scene = commands.add_parser("scene", help="Run two seconds of native scene settling and export an offline 3D preview")
    scene.add_argument("--robot", choices=("ur5e", "piper", "so101"))
    scene.add_argument("--assets", type=Path, required=True)
    scene.add_argument("--pair", choices=("trainer", "ankle-boot", "slipper", "all"))
    scene.add_argument("--config", type=Path, help="A catalog scene.json descriptor; mutually exclusive with --robot/--pair")
    scene.add_argument("--output", type=Path, required=True)
    load = commands.add_parser("load", help="Compile and preview a downloaded scene descriptor offline")
    load.add_argument("config", type=Path)
    load.add_argument("--assets", type=Path, required=True)
    load.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser("benchmark", help="Run the asset coverage benchmark locally")
    benchmark.add_argument("--assets", type=Path, required=True, help="Common root containing all three fetched arm folders")
    benchmark.add_argument("--ur5e-assets", type=Path, help="Optional separate UR5e model root")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--seed", type=int, default=612047)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--source-commit", help="Frozen full source commit for the run's provenance")
    inspect = commands.add_parser("inspect", help="Verify a portable preview or benchmark without a GPU")
    inspect.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        from . import asset_library as library
        if args.command == "list":
            result = library.catalog()
        elif args.command == "fetch":
            if args.robot == "ur5e":
                from .reference_sim import fetch_assets
                result = fetch_assets(args.output)
            else:
                from .robot_profiles import fetch_assets
                result = fetch_assets(args.robot, args.output)
        elif args.command in ("scene", "load"):
            from .asset_preview import export_preview
            if args.config:
                if getattr(args, "robot", None) or getattr(args, "pair", None):
                    raise ValueError("--config cannot be combined with --robot or --pair")
                from .reference_replay import _decode, _read
                config = _decode(_read(args.config, 65536))
                expected = library.scene_config(config["robot"], config["shoe_ids"], timestep=config["timestep"])
                if config != expected or config["timestep"] != .002:
                    raise ValueError("Preview requires a pinned scene descriptor with a 0.002-second timestep")
                robot, shoes = config["robot"], config["shoe_ids"]
            else:
                if not args.robot:
                    raise ValueError("Provide --robot or --config")
                robot = args.robot
                shoes = None if args.pair in (None, "all") else [args.pair + "-left", args.pair + "-right"]
            result = export_preview(robot, args.assets, args.output, shoe_ids=shoes)
        elif args.command == "benchmark":
            from .asset_benchmark import benchmark
            roots = {name: args.assets for name in ("ur5e", "piper", "so101")}
            if args.ur5e_assets:
                roots["ur5e"] = args.ur5e_assets
            result = benchmark(roots, args.output, seed=args.seed, repeats=args.repeats, source_commit=args.source_commit)
        else:
            if (args.directory / "preview-integrity.json").is_file():
                from .asset_preview import inspect_preview
                result = inspect_preview(args.directory)
            else:
                from .asset_benchmark import inspect_benchmark
                result = inspect_benchmark(args.directory)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(2, "error: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
