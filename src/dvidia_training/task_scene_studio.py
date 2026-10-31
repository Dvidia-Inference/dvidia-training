"""Local, CPU-first scene editor and CLI for bounded PiPER task environments.

The browser edits a metric recipe. A single local worker records native physics;
the returned viewer plays those saved poses. No uploaded code, cloud service,
robot connection or learned policy is involved.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import shutil
import tempfile
import threading
from urllib.parse import unquote, urlsplit
import uuid

from . import task_scene

MAX_REQUEST = 32768
MAX_FILE = 32 * 1024 * 1024
_ID = re.compile(r"[a-f0-9]{24}\Z")


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Nonfinite JSON number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _regular(root, relative):
    """Resolve only a fixed artifact subtree, never following an artifact link."""
    path = root/relative
    if any(part in ("", ".", "..") for part in relative.split("/")) or "\\" in relative:
        raise ValueError("Invalid artifact path")
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError("Artifact links are not served")
    if root not in path.resolve().parents or not path.is_file() or path.stat().st_size > MAX_FILE:
        raise ValueError("Artifact unavailable")
    return path


def export_viewer(study, output):
    """Copy one recorded run and a local-only viewer to a fresh portable folder."""
    study, output = Path(study), Path(output)
    if output.exists() or output.is_symlink() or study.is_symlink() or not study.is_dir():
        raise ValueError("Use a recorded study and a fresh viewer directory")
    if not (study/"index.json").is_file():
        raise ValueError("Study has no replay index")
    paths = sorted(study.rglob("*"))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths):
        raise ValueError("Study must contain regular files only")
    files = [p for p in paths if p.is_file()]
    if len(files) > 512 or any(p.stat().st_size > MAX_FILE for p in files) or sum(p.stat().st_size for p in files) > 96*1024*1024:
        raise ValueError("Recorded viewer exceeds its portable size bounds")
    source = Path(__file__).parent
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(study, output/"benchmark")
    shutil.copyfile(source/"task_scene_viewer.html", output/"index.html")
    shutil.copyfile(source/"task_scene_viewer.bundle.js", output/"task_scene_viewer.bundle.js")
    shutil.copyfile(source/"reference_replay.third-party.txt", output/"reference_replay.third-party.txt")
    # The exported recipe is the first frozen case, never a silently changed one.
    candidates = sorted(study.glob("cases/*/input/recipe.json"))
    if not candidates:
        candidates = sorted(study.glob("*/input/recipe.json"))
    recipe = decode(candidates[0].read_bytes()) if candidates else task_scene.default_recipe()
    (output/"recipe.json").write_bytes(encoded(recipe))
    return {"files": len(files)+4, "scope": "Recorded native physics and editable draft; no browser physics or learned skill"}


class SceneServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, directory, assets, *, runner=None, robot_preview=True):
        if address[0] != "127.0.0.1":
            raise ValueError("The editor binds only to 127.0.0.1")
        directory = Path(directory).expanduser()
        if directory.is_symlink():
            raise ValueError("Choose a regular local output directory")
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory.resolve()
        self.assets = Path(assets).expanduser().resolve()
        self.runner = runner
        self.worker_lock = threading.Lock()
        self.runs = set()
        self.recipe = task_scene.default_recipe()
        self.preview = None
        self.preview_geometry = None
        if robot_preview:
            self._robot_preview()
        super().__init__(address, SceneHandler)

    def _robot_preview(self):
        from . import asset_preview, reference_sim
        with tempfile.TemporaryDirectory(prefix=".draft-", dir=self.directory) as temporary:
            draft = Path(temporary)/"scene"
            task_scene.write_scene(self.recipe, draft)
            model, receipt = task_scene.build_scene(draft/"recipe.json", self.assets)
            data = task_scene.initialize(model, receipt)
            mj, np = reference_sim._native_imports()
            mj.mj_forward(model, data)
            scene, geometry = asset_preview._scene(model, {"scope": "Native initial pose; no physics stepped"})
            self.preview_geometry = gzip.compress(geometry, mtime=0)
            scene["geometry_binary"] = {"href": "/api/robot-geometry.bin.gz", "compression": "gzip",
                "bytes": len(geometry), "compressed_bytes": len(self.preview_geometry), "sha256": sha256(geometry).hexdigest()}
            scene["initial_poses"] = np.concatenate((data.xpos, data.xquat), axis=1).tolist()
            self.preview = encoded(scene)

    def simulate(self, recipe):
        if not self.worker_lock.acquire(blocking=False):
            raise ValueError("A simulation is already running")
        try:
            recipe = task_scene.validate_recipe(recipe)
            run_id = uuid.uuid4().hex[:24]
            root = self.directory/run_id
            root.mkdir(exist_ok=False)
            self.runs.add(run_id)
            try:
                task_scene.write_scene(recipe, root/"input")
                if self.runner is None:
                    from .task_scene_benchmark import run_recipe
                    result = run_recipe(root/"input/recipe.json", self.assets, root/"recording", duration_s=1.5)
                else:
                    result = self.runner(root/"input/recipe.json", self.assets, root/"recording", duration_s=1.5)
                return {"index_url": "/runs/"+run_id+"/recording/index.json", "summary": result}
            except Exception as exc:
                (root/"error.json").write_bytes(encoded({"status": "failed", "error": str(exc)[:1000],
                    "scope": "Local preview; incomplete outputs retained"}))
                raise
        finally:
            self.worker_lock.release()


class SceneHandler(BaseHTTPRequestHandler):
    server_version = "DvidiaScene/0.2"

    def log_message(self, *_):
        pass

    def send(self, status, raw, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def local(self, mutation=False):
        host = "127.0.0.1:"+str(self.server.server_port)
        if self.headers.get("Host") != host:
            raise ValueError("Use the displayed 127.0.0.1 address")
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://"+host:
            raise ValueError("Cross-origin requests are not accepted")
        if mutation and self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            raise ValueError("Cross-site requests are not accepted")

    def do_GET(self):
        try:
            self.local()
            path = unquote(urlsplit(self.path).path)
            if path == "/api/config":
                self.send(200, encoded({"local": True, "recipe": self.server.recipe,
                    "robot_scene": "/api/robot-scene.json" if self.server.preview else None}))
            elif path == "/api/robot-scene.json" and self.server.preview:
                self.send(200, self.server.preview)
            elif path == "/api/robot-geometry.bin.gz" and self.server.preview_geometry:
                self.send(200, self.server.preview_geometry, "application/gzip")
            elif path == "/recipe.json":
                self.send(200, encoded(self.server.recipe))
            elif path in ("/", "/index.html", "/task_scene_viewer.bundle.js", "/reference_replay.third-party.txt"):
                name = "task_scene_viewer.html" if path in ("/", "/index.html") else path[1:]
                resource = Path(__file__).parent/name
                mime = "text/html; charset=utf-8" if name.endswith(".html") else "text/javascript; charset=utf-8" if name.endswith(".js") else "text/plain; charset=utf-8"
                self.send(200, resource.read_bytes(), mime)
            elif path.startswith("/runs/"):
                parts = path.split("/", 3)
                if len(parts) != 4 or not _ID.fullmatch(parts[2]) or parts[2] not in self.server.runs:
                    raise ValueError("Unknown local run")
                resource = _regular(self.server.directory, parts[2]+"/"+parts[3])
                mime = "application/gzip" if resource.suffix == ".gz" else mimetypes.guess_type(resource.name)[0] or "application/octet-stream"
                self.send(200, resource.read_bytes(), mime)
            else:
                self.send(404, encoded({"error": "Not found"}))
        except (OSError, ValueError) as exc:
            self.send(400, encoded({"error": str(exc)[:1000]}))

    def do_POST(self):
        try:
            self.local(mutation=True)
            if urlsplit(self.path).path != "/api/simulate":
                self.send(404, encoded({"error": "Not found"})); return
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                raise ValueError("Send a bounded JSON recipe")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST:
                raise ValueError("Recipe request exceeds its byte limit")
            payload = decode(self.rfile.read(length))
            if type(payload) is not dict or set(payload) != {"recipe"}:
                raise ValueError("Send exactly one recipe")
            self.send(200, encoded(self.server.simulate(payload["recipe"])))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.send(400, encoded({"error": str(exc)[:1000]}))
        except Exception:
            self.send(500, encoded({"error": "Simulation failed; the local attempt was retained. See the terminal/output directory."}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Write an editable recipe and explicit local USD")
    init.add_argument("--output", type=Path, required=True)
    author = commands.add_parser("author", help="Author a fresh immutable scene from an edited draft recipe")
    author.add_argument("--recipe", type=Path, required=True)
    author.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="Check a bounded recipe")
    validate.add_argument("--recipe", type=Path, required=True)
    run = commands.add_parser("run", help="Record one local passive physics preview")
    run.add_argument("--recipe", type=Path, required=True)
    run.add_argument("--assets", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    bench = commands.add_parser("benchmark", help="Prepare or run the frozen scene test roster")
    bench.add_argument("phase", choices=("prepare", "run"))
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--assets", type=Path)
    export = commands.add_parser("export", help="Export a portable recorded viewer")
    export.add_argument("--study", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    studio = commands.add_parser("studio", help="Open the local metric scene editor")
    studio.add_argument("--assets", type=Path, required=True)
    studio.add_argument("--output", type=Path, required=True)
    studio.add_argument("--port", type=int, default=8346)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = task_scene.write_scene(task_scene.default_recipe(), args.output)
        elif args.command == "author":
            result = task_scene.write_scene(task_scene.load_recipe(args.recipe), args.output)
        elif args.command == "validate":
            result = {"valid": True, "recipe": task_scene.load_recipe(args.recipe)}
        elif args.command == "run":
            from .task_scene_benchmark import run_recipe
            result = run_recipe(args.recipe, args.assets, args.output, duration_s=1.5)
        elif args.command == "benchmark":
            from . import task_scene_benchmark as benchmark
            if args.phase == "prepare":
                result = benchmark.prepare(args.output)
            else:
                if not args.assets:
                    parser.error("benchmark run requires --assets")
                result = benchmark.run(args.output, args.assets)
        elif args.command == "export":
            result = export_viewer(args.study, args.output)
        else:
            if not 1024 <= args.port <= 65535:
                raise ValueError("Choose a port from 1024 to 65535")
            with SceneServer(("127.0.0.1", args.port), args.output, args.assets) as server:
                print("DVIDIA environment editor: http://127.0.0.1:"+str(server.server_port)+"/", flush=True)
                print("Draft edits become saved native physics previews. Ctrl-C stops the server.", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
            return
        print(encoded(result).decode(), end="")
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc)+"\n")


if __name__ == "__main__":
    main()
