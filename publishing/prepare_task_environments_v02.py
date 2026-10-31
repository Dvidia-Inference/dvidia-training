"""Audit a frozen scene study and prepare one complete, portable public kit."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
import shutil
import zipfile

from dvidia_training import asset_preview, task_scene_audit
from dvidia_training.task_scene_studio import export_viewer

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dvidia.task-environments-publication/v2"
MAX_FILE = 16*1024*1024
MAX_TOTAL = 64*1024*1024


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def identity(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Only regular artifact files may be published")
        if path.is_file():
            raw = path.read_bytes()
            if len(raw) > MAX_FILE:
                raise ValueError("Public file exceeds the bounded payload")
            name = path.relative_to(root).as_posix()
            if path.suffix in (".json", ".md", ".txt", ".html", ".js", ".py", ".usda") or path.name == "LICENSE":
                raw.decode("utf-8")
                if (any(value in raw for value in (b"/Users/", b"/private/", b"/var/folders/"))
                        or re.search(rb"(?:hf_|gh[pousr]_|apikey_)[A-Za-z0-9_]{24,}", raw)):
                    raise ValueError("Private path or credential-shaped value in public text")
            result[name] = identity(raw)
    if len(result) > 512 or sum(row["bytes"] for row in result.values()) > MAX_TOTAL:
        raise ValueError("Public inventory exceeds its bounds")
    return result


def prepare(study, assets, output):
    study, output = Path(study), Path(output)
    archive = output.with_suffix(".zip")
    if output.exists() or output.is_symlink() or archive.exists() or archive.is_symlink():
        raise ValueError("Use fresh public kit and archive destinations")
    audit = task_scene_audit.audit(study)
    export_viewer(study, output)
    _, notices = asset_preview._licenses("piper", assets)
    import mujoco
    import numpy
    from pxr import Usd
    environment = {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
        "numpy": numpy.__version__, "mujoco": mujoco.__version__, "openusd": list(Usd.GetVersion()),
        "package_preview": "0.1.3.dev2", "physics_gpu_required": False,
        "scope": "Shared-host CPU study; no hardware minimum or hard real-time qualification",
        "source_freeze_commit": "14759c5", "viewer": {"three": "0.180.0", "esbuild": "0.25.10"}}
    additions = {"audit.json": encoded(audit), "environment.json": encoded(environment),
        "LICENSE": (ROOT/"LICENSE").read_bytes(),
        "docs/task-environments-v02.md": (ROOT/"docs/task-environments-v02.md").read_bytes(),
        "tools/task_scene_audit.py": (ROOT/"src/dvidia_training/task_scene_audit.py").read_bytes(), **notices}
    report = json.loads((study/"report.json").read_bytes())
    additions["README.md"] = ("# DVIDIA task environments v0.2\n\n"
        "A reusable PiPER scene recipe with multiple rigid instances, an open bin and whole-object completion checks. "
        "The complete frozen study retains six accepted passive completion fixtures, six correctly rejected completion controls "
        "and four invalid layout inputs. These are environment/verifier checks, not a learned robot skill success rate.\n\n"
        "Serve this folder with `python -m http.server 8000 --bind 127.0.0.1` and open http://127.0.0.1:8000/. "
        "The browser edits draft recipes and displays saved native poses. Fresh physics runs require the local Python editor; "
        "see `docs/task-environments-v02.md` for installed UI and Terminal commands.\n\n"
        "`benchmark/` preserves the unchanged frozen protocol, source snapshots, every input and attempt, native geometry, "
        "all 1 ms observation traces and 100 Hz body poses. Identical robot mesh data is stored once by content hash. "
        "`audit.json` reconstructs saved observations with a separately implemented verifier, without stepping simulation. "
        "This establishes evidence consistency, not physical truth, perception accuracy or authenticity.\n\n"
        f"Recorded duration: {report['simulated_seconds']:.2f} simulation seconds in {report['wall_seconds_through_final_outputs']:.3f} "
        "wall seconds including compilation and evidence export on a shared Apple arm64 host. "
        "Preparation, camera capture, training, publication and hardware are outside that timing.\n\n"
        "The arm holds its home commands. Original shoes are rigid three-box surrogates with assumed mechanics. "
        "Arbitrary USD import, calibrated shoe deformation, learned manipulation and physical transfer remain open.\n\n"
        "`manifest.json` hashes every other file and excludes itself. Original DVIDIA code/scenes: MIT. "
        "PiPER assets: MIT, with upstream notices under licenses/. Viewer dependencies retain their notices. "
        "DVIDIA research · [@iammrriver](https://x.com/iammrriver) · hello@dvidia.org.\n").encode()
    for name, raw in additions.items():
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    files = inventory(output)
    manifest = {"schema": SCHEMA, "files": files, "excluded_self": "manifest.json",
        "audit": identity(additions["audit.json"]), "learned_skill": False, "physical_robot_ready": False}
    (output/"manifest.json").write_bytes(encoded(manifest))
    members = inventory(output)
    with archive.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for name in members:
            member = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            zipped.writestr(member, (output/name).read_bytes())
    receipt = {"schema": SCHEMA, "public_files": len(members), "public_bytes": sum(row["bytes"] for row in members.values()),
        "manifest": identity(encoded(manifest)), "archive": {"name": archive.name, "files": len(members), **identity(archive.read_bytes())},
        "audit": identity(additions["audit.json"]), "learned_skill": False, "physical_robot_ready": False}
    archive.with_suffix(".receipt.json").write_bytes(encoded(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(encoded(prepare(args.study, args.assets, args.output)).decode(), end="")
