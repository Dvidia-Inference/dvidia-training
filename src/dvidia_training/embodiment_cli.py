"""Offline reference mechanics and paired replay measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _write_json(path, value):
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)


def _report_html(report):
    data = json.dumps(report, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return Path(__file__).with_name("reference_report.html").read_text(encoding="utf-8").replace("__REFERENCE_DATA__", data)


def _seal(directory):
    files = {}
    for name in ("report.json", "report.html"):
        data = (directory / name).read_bytes()
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    _write_json(directory / "integrity.json", {"schema": "dvidia.reference-integrity/v1", "files": files})


def inspect_bundle(directory):
    """Bounded artifact consistency, not authenticity or metric recomputation."""
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("Duplicate receipt key")
            out[key] = value
        return out
    def bad_constant(value):
        raise ValueError("Nonfinite receipt JSON")
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Receipt directory must exist and not be a symlink")
    def read(name, limit):
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            raise ValueError("Receipt must contain bounded regular non-symlink files")
        from .reference_sim import _regular_bytes
        value = _regular_bytes(path, path.stat().st_size)
        if len(value) > limit:
            raise ValueError("Receipt file exceeds its size bound")
        return value
    def decode(data):
        try:
            value = json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=bad_constant)
            json.dumps(value, allow_nan=False)
            return value
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise ValueError("Receipt requires finite UTF-8 JSON with unique keys") from exc
    manifest = decode(read("integrity.json", 4096))
    if type(manifest) is not dict or set(manifest) != {"schema", "files"} or manifest["schema"] != "dvidia.reference-integrity/v1":
        raise ValueError("Unknown receipt integrity schema")
    if type(manifest["files"]) is not dict or set(manifest["files"]) != {"report.json", "report.html"}:
        raise ValueError("Receipt file allowlist mismatch")
    report = None
    for name, expected in manifest["files"].items():
        if type(expected) is not dict or set(expected) != {"bytes", "sha256"}:
            raise ValueError("Receipt file declaration mismatch")
        data = read(name, 16 * 1024 * 1024)
        if type(expected["bytes"]) is not int or len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("Receipt artifact digest/size mismatch")
        if name == "report.json":
            report = decode(data)
    if type(report) is not dict or report.get("schema") != "dvidia.reference-benchmark/v1" or report.get("physical_robot_ready") is not False:
        raise ValueError("Receipt does not declare a simulation-only reference benchmark")
    attempts = report.get("attempts")
    if type(attempts) is not list or any(type(x) is not dict or type(x.get("valid")) is not bool for x in attempts):
        raise ValueError("Receipt attempt declaration is invalid")
    if type(report.get("attempted")) is not int or type(report.get("valid_attempts")) is not int or report["attempted"] != len(attempts) or report.get("valid_attempts") != sum(x["valid"] for x in attempts):
        raise ValueError("Receipt attempt counts disagree")
    return {"schema": "dvidia.reference-inspection/v1", "scope": "artifact consistency only; not authenticity or metric recomputation",
            "files_verified": 2, "attempted": report["attempted"], "valid_attempts": report["valid_attempts"],
            "physical_robot_ready": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description="DVIDIA hardware reference: mechanics, CPU simulation and paired replay errors; no physical robot control")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-assets", help="Explicitly download pinned public assets and upstream licenses")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--hands", action="store_true", help="Also install left/right five-finger Shadow reference assets")
    for name in ("verify-assets", "audit", "benchmark", "view"):
        part = commands.add_parser(name)
        part.add_argument("assets", type=Path)
        if name != "view":
            part.add_argument("--hands", action="store_true")
        if name == "audit":
            part.add_argument("--output", type=Path)
        elif name == "benchmark":
            part.add_argument("--output", type=Path, required=True)
            part.add_argument("--repeats", type=int, default=2)
            part.add_argument("--seconds", type=float, default=4.)
    score = commands.add_parser("score-paired", help="Score an explicitly aligned simulator/physical trace; no automatic qualification")
    score.add_argument("trace", type=Path)
    score.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Verify bounded report artifact integrity; no native simulator needed")
    inspect.add_argument("directory", type=Path)
    replay = commands.add_parser("export-replay", help="Export actual native geometry and recorded poses to an offline interactive 3D replay")
    replay.add_argument("assets", type=Path)
    replay.add_argument("bundle", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_bundle(args.directory)
        elif args.command == "export-replay":
            from .reference_replay import export_replay
            result = export_replay(args.assets, args.bundle, args.output)
        elif args.command == "score-paired":
            from .embodiment_calibration import load_trace, score_trace
            result = score_trace(load_trace(args.trace))
            _write_json(args.output, result)
        else:
            from . import reference_sim as sim
            if args.command == "fetch-assets":
                result = sim.fetch_assets(args.output, hands=args.hands)
            elif args.command == "verify-assets":
                result = sim.verify_assets(args.assets, hands=args.hands)
            elif args.command == "audit":
                result = sim.audit_reference(args.assets, hands=args.hands)
                if args.output:
                    _write_json(args.output, result)
            elif args.command == "benchmark":
                # Check the protocol and assets before reserving a fresh output.
                if not 1 <= args.repeats <= 5 or not .1 <= args.seconds <= 10:
                    raise ValueError("Repeats must be 1–5; duration must be 0.1–10 seconds")
                sim.validate_duration(args.seconds)
                sim.verify_assets(args.assets, hands=args.hands)
                args.output.mkdir(parents=True, exist_ok=False)
                result = sim.benchmark_reference(args.assets, repeats=args.repeats, seconds=args.seconds, hands=args.hands)
                _write_json(args.output / "report.json", result)
                with (args.output / "report.html").open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(_report_html(result))
                _seal(args.output)
                print(json.dumps({"report": str(args.output / "report.html"), "attempted": result["attempted"],
                                  "valid_attempts": result["valid_attempts"], "physical_robot_ready": False}, allow_nan=False))
                return 0 if result["valid_attempts"] == result["attempted"] else 1
            elif args.command == "view":
                import mujoco
                import mujoco.viewer
                model = sim.build_model(args.assets)
                data = sim._initial(model)
                # Viewer is local simulation only. On macOS use the mjpython launcher.
                mujoco.viewer.launch(model, data)
                return 0
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        print(f"DVIDIA reference: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
