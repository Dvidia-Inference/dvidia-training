"""Portable offline 3D previews of native asset-library scenes.

The exported body poses are recorded from a two-second, held-arm settling run.
This is a geometry/contact preview, not a shoe policy or a hardware validation.
"""
from __future__ import annotations

import gzip
from hashlib import sha256
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct

from . import reference_replay as replay
from . import reference_sim as sim
from . import robot_profiles

SCHEMA = "dvidia.asset-preview/v1"
INTEGRITY_SCHEMA = "dvidia.asset-preview-integrity/v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 96 * 1024 * 1024
SCOPE = ("Recorded native MuJoCo scene settling with held arm commands. The browser displays saved "
         "poses and does not run physics. Shoes are authored rigid proxies with untested physical "
         "properties. No learned shoe skill, calibrated deformation, hardware or real-time qualification.")


def _json(value):
    return replay._json_bytes(value)


def _scene(model, source):
    """Reuse the exact compiled-mesh exporter, extending it for ellipsoids.

    Ellipsoid geometry has no mesh payload. Temporarily substituting sphere
    type for export preserves native size/transform arrays; it never steps or
    recompiles the model. Restore the type array even if validation fails.
    """
    mj, np = sim._native_imports()
    ellipsoids = np.flatnonzero(model.geom_type == int(mj.mjtGeom.mjGEOM_ELLIPSOID))
    try:
        model.geom_type[ellipsoids] = int(mj.mjtGeom.mjGEOM_SPHERE)
        scene, geometry = replay._scene(model, source)
    finally:
        model.geom_type[ellipsoids] = int(mj.mjtGeom.mjGEOM_ELLIPSOID)
    for index in ellipsoids:
        scene["geoms"][int(index)]["type"] = "ellipsoid"
    scene.update(schema="dvidia.asset-preview-model/v1", scope=SCOPE)
    return scene, geometry


def _licenses(robot, assets):
    if robot == "ur5e":
        provenance, blobs = sim._asset_payload(assets)
        folders = ("universal_robots_ur5e", "robotiq_2f85")
    else:
        item, blobs = robot_profiles._payload(robot, assets)
        provenance = robot_profiles.verify_assets(robot, assets)
        folders = (item["folder"],)
    notices = {}
    for folder in folders:
        for name in ("LICENSE", "README.md"):
            key = folder + "/" + name
            if key not in blobs:
                raise ValueError("Pinned upstream asset notice is missing: " + key)
            notices["licenses/" + key] = blobs[key]
    return provenance, notices


def export_preview(robot, assets, output, *, shoe_ids=None):
    """Compile verified assets and export one fresh, fully local 3D preview."""
    from . import asset_library as library
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Preview output must be a fresh directory")
    if shoe_ids is None:
        shoe_ids = [item["id"] for item in library.catalog()["shoes"]]
    config = library.scene_config(robot, shoe_ids, timestep=.002)
    provenance, notices = _licenses(robot, assets)
    mj, np = sim._native_imports()
    warnings, frames, rows = [], [], []
    with sim.native_warnings(warnings):
        model = library.load_scene(config, assets)
        data = library.initialize(model)
        digest = sim.model_digest(model)
        source = {"robot": robot, "assets": provenance, "mujoco": mj.__version__,
                  "compiled_model_sha256": digest, "scene_sha256": sha256(_json(config)).hexdigest()}
        scene, geometry = _scene(model, source)
        for step in range(1001):
            if step % 10 == 0:
                mj.mj_forward(model, data)
                pose = np.concatenate((data.xpos, data.xquat), axis=1)
                if not np.isfinite(pose).all() or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Preview native state became nonfinite")
                frames.append(pose)
                rows.append({"t_seconds": round(float(data.time), 9), "contact_count": int(data.ncon),
                             "joint_position": data.qpos[:model.nq - 7 * len(config["shoe_ids"])].tolist()})
            if step < 1000:
                mj.mj_step(model, data)
        if warnings or any(int(item.number) for item in data.warning):
            raise ValueError("Preview native scene emitted warnings: " + str(warnings))
    motion = np.asarray(frames, dtype="<f4").tobytes(order="C")
    packed_geometry, packed_motion = gzip.compress(geometry, mtime=0), gzip.compress(motion, mtime=0)
    scene["geometry_binary"] = {"href": "geometry.bin.gz", "compression": "gzip", "bytes": len(geometry),
                                "compressed_bytes": len(packed_geometry), "sha256": sha256(geometry).hexdigest()}
    motion_meta = {"schema": "dvidia.asset-preview-motion/v1", "source": source, "scope": SCOPE,
                   "body_count": int(model.nbody), "frame_count": len(rows), "frame_stride_floats": int(model.nbody) * 7,
                   "duration_seconds": 2., "timestep_seconds": .002, "rows": rows,
                   "motion_binary": {"href": "motion.bin.gz", "compression": "gzip", "bytes": len(motion),
                                     "compressed_bytes": len(packed_motion), "sha256": sha256(motion).hexdigest()}}
    resource = Path(__file__).parent
    files = {"scene.json": _json(config), "catalog.json": _json(library.catalog()), "model.json": _json(scene),
             "motion.json": _json(motion_meta), "geometry.bin.gz": packed_geometry, "motion.bin.gz": packed_motion,
             "index.html": replay._read(resource / "asset_library.html", 1024 * 1024),
             "asset_library.bundle.js": replay._read(resource / "asset_library.bundle.js", 4 * 1024 * 1024),
             "reference_replay.third-party.txt": replay._read(resource / "reference_replay.third-party.txt", 65536),
             "LICENSE": replay.MIT_LICENSE.encode(), **notices}
    for family in ("trainer", "ankle-boot", "slipper"):
        ids = [item for item in config["shoe_ids"] if item in (family + "-left", family + "-right")]
        if ids:
            files["scenes/" + family + ".json"] = _json(library.scene_config(robot, ids, timestep=.002))
    files["README.md"] = ("# DVIDIA asset library preview\n\n" + SCOPE + "\n\n"
        "Serve this directory with `python -m http.server 8000` and open http://localhost:8000. "
        "Everything is local; JavaScript, WebGL2 and DecompressionStream are required. "
        "The pair selector filters recorded objects visually; use a downloaded scene descriptor "
        "with dvidia-assets scene --config to compile and simulate only that pair. "
        "geometry.bin.gz and motion.bin.gz decode to bounded little-endian Float32 positions and "
        "wxyz quaternions. MIT applies to DVIDIA code and synthetic shoes. Robot model licenses "
        "and original READMEs remain under licenses/. Three.js/OrbitControls notices are included.\n").encode()
    seal = {"schema": INTEGRITY_SCHEMA, "scope": "Artifact consistency only, not authenticity or physical qualification",
            "files": {name: {"bytes": len(value), "sha256": sha256(value).hexdigest()} for name, value in files.items()}}
    files["preview-integrity.json"] = _json(seal)
    if any(len(value) > MAX_FILE_BYTES for value in files.values()) or sum(map(len, files.values())) > MAX_TOTAL_BYTES:
        raise ValueError("Preview exceeds bounded export size")
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    inspected = inspect_preview(destination)
    return {"schema": SCHEMA, "output": str(destination), "robot": robot, "scope": SCOPE,
            "physical_robot_ready": False, "learned_skill": False, "frame_count": len(rows),
            "file_count": len(files), "bytes": sum(map(len, files.values())), "files": inspected["files"]}


def _unpack(meta, payload, expected):
    if type(meta) is not dict or set(meta) != {"href", "compression", "bytes", "compressed_bytes", "sha256"}:
        raise ValueError("Preview binary framing mismatch")
    if meta["href"] != expected or meta["compression"] != "gzip":
        raise ValueError("Preview binary path or codec mismatch")
    if type(meta["bytes"]) is not int or not 0 <= meta["bytes"] <= MAX_FILE_BYTES:
        raise ValueError("Preview decoded binary size outside bounds")
    packed = payload[expected]
    if type(meta["compressed_bytes"]) is not int or meta["compressed_bytes"] != len(packed):
        raise ValueError("Preview compressed binary length mismatch")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            content = stream.read(meta["bytes"] + 1)
    except (OSError, EOFError) as exc:
        raise ValueError("Preview invalid compressed binary") from exc
    if len(content) != meta["bytes"] or sha256(content).hexdigest() != meta["sha256"]:
        raise ValueError("Preview decoded binary integrity mismatch")
    return content


def inspect_preview(directory):
    """Check bounded portable data and original notices without native imports."""
    from . import asset_library as library
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Preview must be a regular directory")
    seal = replay._decode(replay._read(root / "preview-integrity.json", 128 * 1024))
    if type(seal) is not dict or seal.get("schema") != INTEGRITY_SCHEMA or type(seal.get("files")) is not dict:
        raise ValueError("Preview integrity schema mismatch")
    if not 10 <= len(seal["files"]) <= 40:
        raise ValueError("Preview inventory outside bounds")
    payload, total = {}, 0
    for name, record in seal["files"].items():
        path = PurePosixPath(name)
        if (re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name) is None
                or str(path) != name or path.is_absolute() or any(part in (".", "..") for part in path.parts)
                or "\\" in name or any(part.startswith(".") for part in path.parts)):
            raise ValueError("Preview file path is not contained")
        target = root / name
        if any(parent.is_symlink() for parent in target.parents if parent == root or root in parent.parents):
            raise ValueError("Preview path contains a symlink")
        if type(record) is not dict or set(record) != {"bytes", "sha256"} or type(record["bytes"]) is not int:
            raise ValueError("Preview integrity file record mismatch")
        content = replay._read(target, MAX_FILE_BYTES)
        total += len(content)
        if total > MAX_TOTAL_BYTES or len(content) != record["bytes"] or sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("Preview file integrity mismatch")
        payload[name] = content
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() or path.is_symlink()}
    if actual != set(payload) | {"preview-integrity.json"}:
        raise ValueError("Preview includes missing, extra or symlink files")
    config, model, motion, catalog = [replay._decode(payload[name]) for name in ("scene.json", "model.json", "motion.json", "catalog.json")]
    if catalog != library.catalog():
        raise ValueError("Preview catalog differs from the packaged immutable catalog")
    expected = library.scene_config(config["robot"], config["shoe_ids"], timestep=config["timestep"])
    if config != expected:
        raise ValueError("Preview scene descriptor mismatch")
    descriptors = {}
    for family in ("trainer", "ankle-boot", "slipper"):
        identifiers = [item for item in config["shoe_ids"] if item in (family + "-left", family + "-right")]
        if identifiers:
            descriptors["scenes/" + family + ".json"] = library.scene_config(config["robot"], identifiers, timestep=.002)
    if {name for name in payload if name.startswith("scenes/")} != set(descriptors):
        raise ValueError("Preview downloadable descriptor inventory mismatch")
    for name, descriptor in descriptors.items():
        if replay._decode(payload[name]) != descriptor:
            raise ValueError("Preview downloadable scene descriptor mismatch")
    if model.get("schema") != "dvidia.asset-preview-model/v1" or motion.get("schema") != "dvidia.asset-preview-motion/v1":
        raise ValueError("Preview data schema mismatch")
    if model.get("scope") != SCOPE or motion.get("scope") != SCOPE or model["source"] != motion["source"]:
        raise ValueError("Preview source/scope mismatch")
    source = model["source"]
    if source.get("robot") != config["robot"] or source.get("scene_sha256") != sha256(payload["scene.json"]).hexdigest():
        raise ValueError("Preview descriptor provenance mismatch")
    robot = config["robot"]
    if robot == "ur5e":
        lock = sim.asset_lock()
        records = [record for record in lock["files"] if not record["path"].startswith("shadow_hand/")]
        if source["assets"].get("commit") != sim.COMMIT or source["assets"].get("lock_sha256") != sha256(Path(sim.__file__).with_name("reference_assets.lock.json").read_bytes()).hexdigest():
            raise ValueError("Preview robot asset provenance mismatch")
    else:
        records = robot_profiles._profile(robot)["files"]
        if source["assets"].get("commit") != robot_profiles._lock()["commit"] or source["assets"].get("lock_sha256") != sha256(robot_profiles.LOCK_PATH.read_bytes()).hexdigest():
            raise ValueError("Preview robot asset provenance mismatch")
    for record in records:
        if record["path"].endswith(("/LICENSE", "/README.md")):
            value = payload.get("licenses/" + record["path"])
            if value is None or len(value) != record["bytes"] or sha256(value).hexdigest() != record["sha256"]:
                raise ValueError("Preview original upstream license/README mismatch")
    geometry = _unpack(model["geometry_binary"], payload, "geometry.bin.gz")
    poses = _unpack(motion["motion_binary"], payload, "motion.bin.gz")
    nbody = motion["body_count"]
    if (type(nbody) is not int or not 1 <= nbody <= 128 or len(model["bodies"]) != nbody
            or motion["frame_count"] != 101 or motion["frame_stride_floats"] != nbody * 7
            or len(poses) != 101 * nbody * 28 or len(motion["rows"]) != 101):
        raise ValueError("Preview motion layout mismatch")
    for i, body in enumerate(model["bodies"]):
        if body["id"] != i or type(body["name"]) is not str:
            raise ValueError("Preview body order mismatch")
    for i, row in enumerate(motion["rows"]):
        if not math.isclose(row["t_seconds"], i * .02, abs_tol=1e-8) or type(row["contact_count"]) is not int or row["contact_count"] < 0:
            raise ValueError("Preview recorded row mismatch")
    for values in struct.iter_unpack("<7f", poses):
        if not all(math.isfinite(x) for x in values) or not math.isclose(sum(x*x for x in values[3:]), 1., abs_tol=1e-5):
            raise ValueError("Preview body pose is invalid")
    mesh_ids = set()
    for mesh in model["meshes"]:
        if mesh["id"] in mesh_ids:
            raise ValueError("Preview mesh identity duplicated")
        mesh_ids.add(mesh["id"])
        count, start, indices, offset = [mesh[key] for key in ("vertex_count", "position_byte_offset", "index_count", "index_byte_offset")]
        if (any(type(x) is not int or x < 0 for x in (count, start, indices, offset)) or count == 0 or indices % 3
                or start % 4 or offset % 4 or start + count*12 > len(geometry) or offset + indices*4 > len(geometry)):
            raise ValueError("Preview mesh binary range mismatch")
        if not all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", geometry[start:start+count*12])):
            raise ValueError("Preview mesh position is nonfinite")
        if any(v[0] >= count for v in struct.iter_unpack("<I", geometry[offset:offset+indices*4])):
            raise ValueError("Preview mesh index outside vertices")
    for geom in model["geoms"]:
        if (geom["type"] not in ("mesh", "box", "sphere", "ellipsoid", "capsule", "cylinder", "plane")
                or type(geom["body_id"]) is not int or not 0 <= geom["body_id"] < nbody
                or geom["type"] == "mesh" and geom["mesh_id"] not in mesh_ids):
            raise ValueError("Preview geom identity mismatch")
        for name, length in (("position", 3), ("quaternion_wxyz", 4), ("size", 3), ("rgba", 4)):
            replay._vector(geom[name], length, name)
    return {"schema": SCHEMA, "robot": robot, "bytes": total, "files": seal["files"],
            "scope": SCOPE, "frame_count": 101, "physical_robot_ready": False}
