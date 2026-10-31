"""Append a small recovery-capable reader; preserve every original artifact.

Reuse the exact reviewed v0.2 CAS publisher in an isolated module instance.
Only four fixed reader files can be added to one previously absent namespace.
The original 175-file kit, dataset and GitHub release are never edited.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil

_base = Path(__file__).with_name("publish_task_environments_v02.py")
_spec = importlib.util.spec_from_file_location("dvidia_reader02_cas", _base)
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)
engine.SCHEMA = "dvidia.task-environment-reader-hub/v2"
engine.REPOS = {"space": "Dvidia/robot-simulation-lab"}
engine.NAMESPACE = "task-environments/v02-reader02/"
engine.IMPLEMENTATION = (Path(__file__).name, _base.name, "publish.py", "prepare.py")
FILES = {"index.html", "task_scene_viewer.bundle.js", "reference_replay.third-party.txt", "reader.json"}


def stage(source, output):
    source, output = engine.plain(source), engine.plain(output)
    if output.exists():
        raise engine.PublicationError("Use a fresh reader plan")
    rows = engine.files(source)
    if {item["path"] for item in rows} != FILES or sum(item["bytes"] for item in rows) > 2*1024*1024:
        raise engine.PublicationError("Expected exactly four bounded reader files")
    html = engine.read(source/"index.html")
    metadata = engine.load(source/"reader.json")
    if b'data-artifact-root="../v02/"' not in html or metadata.get("scope") != "Display recovery only; scientific evidence unchanged":
        raise engine.PublicationError("Reader must reference the unchanged original artifacts")
    expected = metadata.get("original_manifest", {})
    if expected != {"bytes": 28021, "sha256": "ce8f93cbb1bf613364fdbef92ae1a7d3bfa47eca3b2484a10f59b9ee58a5c38a"}:
        raise engine.PublicationError("Unexpected original artifact manifest")
    api = engine.public_client()
    baseline = engine.tree(api, "space")
    if any(name.startswith(engine.NAMESPACE) for name in baseline["tree"]):
        raise engine.PublicationError("Reader namespace already exists")
    manifest_path = "task-environments/v02/manifest.json"
    raw = engine.download(engine.REPOS["space"], manifest_path, "space", baseline["commit"], output.parent/"reader-anonymous-cache")
    if engine.digest(raw) != expected:
        raise engine.PublicationError("Original artifact manifest differs")
    if engine.tree(api, "space") != baseline:
        raise engine.PublicationError("Space moved during reader staging")
    output.mkdir(parents=True)
    root = output/"space"/engine.NAMESPACE
    shutil.copytree(source, root)
    plan = {"schema": engine.SCHEMA, "public": True, "namespace": engine.NAMESPACE,
        "repositories": [{"kind": "space", "repo": engine.REPOS["space"], "root": str(output/"space"),
                           "files": engine.files(output/"space"), "baseline": baseline}],
        "implementation": engine.implementation(), "qualified_robot_skill": False, "physical_robot_ready": False,
        "unchanged_artifact_manifest": expected}
    engine.atomic(output/"plan.json", plan, exclusive=True)
    return engine.checked(output/"plan.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "check", "publish", "verify"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "stage":
            result = stage(args.source, args.output)
        elif args.command == "check":
            result = engine.checked(args.plan)
        elif args.command == "publish":
            result = engine.publish(args.plan)
        else:
            result = engine.verify(args.plan)
        print(json.dumps({"schema": result["schema"], "command": args.command, "complete": True}))
    except engine.PublicationError as exc:
        parser.exit(2, str(exc)+"\n")


if __name__ == "__main__":
    main()
