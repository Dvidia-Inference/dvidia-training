"""Offline 3D replay of a sealed native reference receipt, without physics stepping.

Compiled mesh vertices and corrected geom transforms are exported together.
All body poses come from recorded qpos followed by mj_forward; a failed receipt
compatibility or position check aborts before reserving an output directory.
"""
from __future__ import annotations

from hashlib import sha256
import gzip
import json
import math
from pathlib import Path
import re
import stat

from . import reference_sim as sim


SCOPE = ("Recorded authored simulation from the frozen reference receipt. "
         "Native mesh geometry and recorded joint/linkage positions, reconstructed with mj_forward only. "
         "No live physics, robot control, task policy, physical accuracy or new speed measurement.")
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_BINARY_BYTES = 64 * 1024 * 1024
POSITION_TOLERANCE_M = 1e-9
ROTATION_TOLERANCE = 1e-10
MIT_LICENSE = """MIT License

Copyright (c) 2026 DVIDIA contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate replay input key")
        value[key] = item
    return value


def _decode(data):
    def nonfinite(_):
        raise ValueError("Nonfinite replay input")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=nonfinite)
        json.dumps(value, allow_nan=False)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Replay requires finite UTF-8 JSON with unique keys") from exc


def _read(path, limit):
    path = Path(path)
    before = path.lstat()
    size = before.st_size
    if not stat.S_ISREG(before.st_mode) or size > limit:
        raise ValueError("Replay input must be a bounded regular non-symlink file")
    content = sim._regular_bytes(path, size)
    if len(content) > limit:
        raise ValueError("Replay input exceeds its size bound")
    return content


def _receipt(bundle):
    # Preserve the original public inspector's integrity contract first.
    from .embodiment_cli import inspect_bundle
    inspect_bundle(bundle)
    root = Path(bundle)
    files = {name: _read(root / name, 4096 if name == "integrity.json" else MAX_RECEIPT_BYTES)
             for name in ("report.json", "report.html", "integrity.json")}
    manifest = _decode(files["integrity.json"])
    if (type(manifest) is not dict or set(manifest) != {"schema", "files"}
            or manifest["schema"] != "dvidia.reference-integrity/v1"
            or type(manifest["files"]) is not dict
            or set(manifest["files"]) != {"report.json", "report.html"}):
        raise ValueError("Replay receipt integrity contract changed during reading")
    for name in ("report.json", "report.html"):
        expected = manifest["files"][name]
        if (type(expected) is not dict or set(expected) != {"bytes", "sha256"}
                or type(expected["bytes"]) is not int or expected["bytes"] != len(files[name])
                or expected["sha256"] != sha256(files[name]).hexdigest()):
            raise ValueError("Replay receipt changed during reading")
    return _decode(files["report.json"]), files


def _number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Replay {description} must be finite numeric data")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"Replay {description} must be finite numeric data") from exc
    if not math.isfinite(converted):
        raise ValueError(f"Replay {description} must be finite numeric data")
    return converted


def _vector(value, length, description):
    if type(value) is not list or len(value) != length:
        raise ValueError(f"Replay {description} must have {length} values")
    return [_number(item, description) for item in value]


def _digest(value, description):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Replay {description} must be a SHA256 digest")
    return value


def _validate_report(report, assets):
    if (type(report) is not dict or report.get("schema") != "dvidia.reference-benchmark/v1"
            or report.get("profile") != sim.PROFILE or report.get("physical_robot_ready") is not False
            or report.get("task_success_evaluated") is not False):
        raise ValueError("Replay requires the simulation-only pinned reference profile")
    if type(report.get("hardware")) is not dict or report["hardware"].get("mujoco") != "3.15.0":
        raise ValueError("Replay receipt requires MuJoCo 3.15.0")
    audit = report.get("audit")
    if type(audit) is not dict or audit.get("profile") != sim.PROFILE:
        raise ValueError("Replay audit profile mismatch")
    provenance = audit.get("asset_provenance")
    if (type(provenance) is not dict or provenance.get("commit") != assets["commit"]
            or provenance.get("lock_sha256") != assets["lock_sha256"]):
        raise ValueError("Replay asset commit or lock mismatch")
    inventory = audit.get("inventory")
    if (type(inventory) is not dict or inventory.get("joint_coordinates_nq") != 14
            or inventory.get("velocity_dofs_nv") != 14 or inventory.get("actuator_count") != 7):
        raise ValueError("Replay receipt mechanical contract mismatch")
    joints = inventory.get("joints")
    if type(joints) is not list or len(joints) != 14:
        raise ValueError("Replay receipt joint inventory mismatch")
    for address, joint in enumerate(joints):
        if (type(joint) is not dict or type(joint.get("qpos_address")) is not int
                or joint["qpos_address"] != address or joint.get("type") != 3
                or type(joint.get("name")) is not str):
            raise ValueError("Replay receipt joint coordinate ordering mismatch")
    if tuple(item["name"] for item in joints[:6]) != sim.ARM_JOINTS:
        raise ValueError("Replay arm joint names mismatch")
    protocol = report.get("protocol")
    if (type(protocol) is not dict or protocol.get("cases") != list(sim.CASES)
            or protocol.get("timesteps_seconds") != [.002, .001, .0005]
            or type(protocol.get("repeats")) is not int or not 1 <= protocol["repeats"] <= 5
            or _number(protocol.get("control_dt_seconds"), "control period") != sim.CONTROL_DT):
        raise ValueError("Replay receipt protocol mismatch")
    seconds = sim.validate_duration(_number(protocol.get("seconds_per_case"), "duration"))
    attempts = report.get("attempts")
    expected_count = len(sim.CASES) * 3 * protocol["repeats"]
    if (type(attempts) is not list or len(attempts) != expected_count
            or type(report.get("attempted")) is not int or report["attempted"] != expected_count
            or type(report.get("valid_attempts")) is not int):
        raise ValueError("Replay attempt framing mismatch")
    selected, seen, valid_count = [], set(), 0
    for attempt in attempts:
        if (type(attempt) is not dict or attempt.get("case") not in sim.CASES
                or type(attempt.get("repeat")) is not int or not 0 <= attempt["repeat"] < protocol["repeats"]
                or type(attempt.get("valid")) is not bool):
            raise ValueError("Replay attempt declaration mismatch")
        timestep = sim._timestep(attempt.get("timestep_seconds"))
        identity = (attempt["case"], timestep, attempt["repeat"])
        if identity in seen:
            raise ValueError("Duplicate replay case/timestep/repeat")
        seen.add(identity)
        valid_count += attempt["valid"]
        if not attempt["valid"] or attempt["repeat"] != 0:
            continue
        if (_number(attempt.get("control_dt_seconds"), "control period") != sim.CONTROL_DT
                or _number(attempt.get("simulated_seconds"), "duration") != seconds
                or attempt.get("warnings") != [] or attempt.get("nonfinite_trajectory_omitted") is not False):
            raise ValueError("Replay valid trajectory protocol mismatch")
        _digest(attempt.get("compiled_model_sha256"), "compiled model")
        rows = attempt.get("trajectory")
        frame_count = round(seconds / sim.CONTROL_DT) + 1
        if type(rows) is not list or len(rows) != frame_count or frame_count > 501:
            raise ValueError("Replay trajectory frame count mismatch")
        for frame, row in enumerate(rows):
            if type(row) is not dict:
                raise ValueError("Replay trajectory row must be an object")
            if not math.isclose(_number(row.get("t_seconds"), "frame time"), frame * sim.CONTROL_DT,
                                abs_tol=1e-9, rel_tol=0):
                raise ValueError("Replay timestamps must match ordered 20 ms boundaries")
            for field, length in (("joint_position_rad", 6), ("joint_velocity_rad_s", 6),
                                  ("issued_joint_targets_rad", 6), ("linkage_position_rad", 8),
                                  ("tcp_position_m", 3), ("tcp_quaternion_wxyz", 4)):
                _vector(row.get(field), length, field)
            quat = row["tcp_quaternion_wxyz"]
            if not math.isclose(sum(x * x for x in quat), 1., abs_tol=1e-8, rel_tol=0):
                raise ValueError("Replay TCP quaternion must be normalized")
            for field, length in (("arm_points_m", 8), ("pad_centers_m", 2)):
                if type(row.get(field)) is not list or len(row[field]) != length:
                    raise ValueError(f"Replay {field} framing mismatch")
                for vector in row[field]:
                    _vector(vector, 3, field)
            if _number(row.get("pad_center_separation_m"), "pad separation") < 0:
                raise ValueError("Replay pad separation cannot be negative")
            if not 0 <= _number(row.get("issued_gripper_0_255"), "gripper command") <= 255:
                raise ValueError("Replay gripper command out of range")
            if type(row.get("contact_count")) is not int or not 0 <= row["contact_count"] <= 1024:
                raise ValueError("Replay contact count must be a bounded integer")
        selected.append(attempt)
    if valid_count != report["valid_attempts"] or not selected:
        raise ValueError("Replay valid attempt count mismatch or no valid repeat zero")
    return selected, joints


def _scene(model, source):
    mj, np = sim._native_imports()
    if not 1 <= model.nbody <= 128 or not 1 <= model.ngeom <= 512 or not 0 <= model.nmesh <= 128:
        raise ValueError("Replay model inventory exceeds bounds")
    if model.nmeshvert > 2_000_000 or model.nmeshface > 2_000_000:
        raise ValueError("Replay mesh inventory exceeds bounds")
    geometry = bytearray()
    meshes = []
    for index in range(model.nmesh):
        vertex_start, vertex_count = int(model.mesh_vertadr[index]), int(model.mesh_vertnum[index])
        face_start, face_count = int(model.mesh_faceadr[index]), int(model.mesh_facenum[index])
        vertices = np.asarray(model.mesh_vert[vertex_start:vertex_start + vertex_count], dtype="<f4")
        faces = np.asarray(model.mesh_face[face_start:face_start + face_count], dtype="<u4")
        if not np.isfinite(vertices).all() or faces.size and int(faces.max()) >= vertex_count:
            raise ValueError("Replay compiled mesh contains invalid coordinates or indices")
        # Merge only identical 12-byte xyz triples. This preserves every triangle
        # position exactly, including signed zero, without mesh simplification.
        triples = np.frombuffer(vertices.tobytes(order="C"), dtype="V12")
        unique, inverse = np.unique(triples, return_inverse=True)
        vertices = np.frombuffer(unique.tobytes(), dtype="<f4").reshape(-1, 3)
        faces = np.asarray(inverse[faces], dtype="<u4")
        position_offset = len(geometry)
        geometry.extend(vertices.tobytes(order="C"))
        index_offset = len(geometry)
        geometry.extend(faces.tobytes(order="C"))
        meshes.append({"id": index, "position_byte_offset": position_offset, "vertex_count": len(vertices),
                       "native_vertex_count": vertex_count,
                       "index_byte_offset": index_offset, "index_count": face_count * 3})
    types = {int(mj.mjtGeom.mjGEOM_MESH): "mesh", int(mj.mjtGeom.mjGEOM_BOX): "box",
             int(mj.mjtGeom.mjGEOM_SPHERE): "sphere", int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule",
             int(mj.mjtGeom.mjGEOM_CYLINDER): "cylinder", int(mj.mjtGeom.mjGEOM_PLANE): "plane"}
    geoms = []
    for index in range(model.ngeom):
        kind = types.get(int(model.geom_type[index]))
        if kind is None:
            raise ValueError("Replay model contains an unsupported geom type")
        rgba = model.geom_rgba[index]
        material = int(model.geom_matid[index])
        # Native rendering uses material rgba when the geom has its default rgba.
        if material >= 0 and np.array_equal(rgba, np.asarray([.5, .5, .5, 1], dtype=np.float32)):
            rgba = model.mat_rgba[material]
        body = int(model.geom_bodyid[index])
        position, quaternion = model.geom_pos[index], model.geom_quat[index]
        sameframe = int(model.geom_sameframe[index])
        # MuJoCo's compiled same-frame optimization may replace a near-identical
        # local geom transform with the exact body/inertial transform. Honor it
        # rather than rendering the unoptimized, subtly different geom arrays.
        if sameframe == int(mj.mjtSameFrame.mjSAMEFRAME_BODY):
            position, quaternion = np.zeros(3), np.asarray([1., 0., 0., 0.])
        elif sameframe == int(mj.mjtSameFrame.mjSAMEFRAME_INERTIA):
            position, quaternion = model.body_ipos[body], model.body_iquat[body]
        elif sameframe == int(mj.mjtSameFrame.mjSAMEFRAME_BODYROT):
            quaternion = np.asarray([1., 0., 0., 0.])
        elif sameframe == int(mj.mjtSameFrame.mjSAMEFRAME_INERTIAROT):
            quaternion = model.body_iquat[body]
        geoms.append({"geom_id": index, "name": sim._name(model, mj.mjtObj.mjOBJ_GEOM, index),
                      "body_id": body, "type": kind,
                      "group": int(model.geom_group[index]),
                      "size": model.geom_size[index].tolist(), "position": position.tolist(),
                      "quaternion_wxyz": quaternion.tolist(), "rgba": rgba.tolist(),
                      "mesh_id": int(model.geom_dataid[index]) if kind == "mesh" else None,
                      "collision": bool(model.geom_contype[index] or model.geom_conaffinity[index])})
    if len(geometry) > MAX_BINARY_BYTES:
        raise ValueError("Replay geometry binary exceeds bounds")
    return {"schema": "dvidia.reference-replay-scene/v1", "meshes": meshes, "geoms": geoms,
            "bodies": [{"id": i, "name": sim._name(model, mj.mjtObj.mjOBJ_BODY, i)} for i in range(model.nbody)],
            "joints": [{"name": sim._name(model, mj.mjtObj.mjOBJ_JOINT, i), "range": model.jnt_range[i].tolist()}
                       for i in range(model.njnt)], "source": source, "scope": SCOPE}, bytes(geometry)


def _poses(model, attempt, receipt_joints, residuals):
    mj, np = sim._native_imports()
    if model.nq != 14 or model.nv != 14 or model.nu != 7 or model.njnt != 14:
        raise ValueError("Replay compiled mechanical contract mismatch")
    for address, record in enumerate(receipt_joints):
        joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, record["name"])
        if joint < 0 or int(model.jnt_qposadr[joint]) != address:
            raise ValueError("Replay compiled linkage coordinate ordering mismatch")
    data = mj.MjData(model)
    frames = []
    for row in attempt["trajectory"]:
        data.qpos[:] = row["joint_position_rad"] + row["linkage_position_rad"]
        # qvel/ctrl do not change geometry; preserve them for native forward diagnostics.
        data.qvel[:6] = row["joint_velocity_rad_s"]
        data.ctrl[:] = row["issued_joint_targets_rad"] + [row["issued_gripper_0_255"]]
        mj.mj_forward(model, data)
        observed = sim._sample(model, data, row["t_seconds"], data.ctrl)
        for field in ("tcp_position_m", "arm_points_m", "pad_centers_m"):
            delta = float(np.max(np.abs(np.asarray(observed[field]) - row[field])))
            residuals[field] = max(residuals[field], delta)
            if delta > POSITION_TOLERANCE_M:
                raise ValueError(f"Replay reconstructed {field} does not match the frozen receipt")
        delta = abs(observed["pad_center_separation_m"] - row["pad_center_separation_m"])
        residuals["pad_center_separation_m"] = max(residuals["pad_center_separation_m"], delta)
        if delta > POSITION_TOLERANCE_M:
            raise ValueError("Replay reconstructed pad separation does not match the frozen receipt")
        # Quaternion sign is arbitrary; compare both equivalent representations.
        current, recorded = np.asarray(observed["tcp_quaternion_wxyz"]), np.asarray(row["tcp_quaternion_wxyz"])
        delta = float(min(np.max(np.abs(current - recorded)), np.max(np.abs(current + recorded))))
        residuals["tcp_quaternion_wxyz"] = max(residuals["tcp_quaternion_wxyz"], delta)
        if delta > ROTATION_TOLERANCE:
            raise ValueError("Replay reconstructed TCP rotation does not match the frozen receipt")
        pose = np.concatenate((data.xpos, data.xquat), axis=1)
        if not np.isfinite(pose).all():
            raise ValueError("Replay reconstructed a nonfinite body pose")
        frames.append(pose)
    return np.asarray(frames, dtype="<f4").tobytes(order="C")


def _json_bytes(value):
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def export_replay(assets, bundle, output):
    """Export an offline native-mesh replay into a fresh directory.

    This verifies artifact consistency, pinned assets, compiled models and each
    recorded pose. Neither the receipt nor these checks establish authenticity
    or physical accuracy. Native physics is never stepped.
    """
    report, receipt_files = _receipt(bundle)
    provenance, blobs = sim._asset_payload(assets)
    attempts, receipt_joints = _validate_report(report, provenance)
    source = {"profile": sim.PROFILE, "asset_commit": sim.COMMIT,
              "asset_lock_sha256": provenance["lock_sha256"], "report_sha256": sha256(receipt_files["report.json"]).hexdigest(),
              "report_href": "receipt/report.html", "receipt_href": "receipt/report.json",
              "exporter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "mujoco": "3.15.0", "compiled_model_sha256": {},
              "asset_licenses": provenance["licenses"]}
    models, scene, geometry = {}, None, None
    residuals = {name: 0. for name in ("tcp_position_m", "arm_points_m", "pad_centers_m",
                                      "pad_center_separation_m", "tcp_quaternion_wxyz")}
    motion, cases, warnings = bytearray(), [], []
    with sim.native_warnings(warnings):
        for attempt in attempts:
            timestep = attempt["timestep_seconds"]
            if timestep not in models:
                models[timestep] = sim.build_model(assets, timestep=timestep)
            model = models[timestep]
            digest = sim.model_digest(model)
            if digest != attempt["compiled_model_sha256"]:
                raise ValueError("Replay compiled model digest differs from the frozen receipt")
            source["compiled_model_sha256"][str(timestep)] = digest
            candidate_scene, candidate_geometry = _scene(model, source)
            if scene is None:
                scene, geometry = candidate_scene, candidate_geometry
            elif (candidate_geometry != geometry or candidate_scene != scene):
                raise ValueError("Replay model geometry differs across recorded timesteps")
            offset = len(motion)
            motion.extend(_poses(model, attempt, receipt_joints, residuals))
            if len(motion) > MAX_BINARY_BYTES:
                raise ValueError("Replay motion binary exceeds bounds")
            fields = ("t_seconds", "joint_position_rad", "issued_joint_targets_rad", "linkage_position_rad",
                      "issued_gripper_0_255", "contact_count", "tcp_position_m", "tcp_quaternion_wxyz",
                      "pad_center_separation_m")
            cases.append({"name": attempt["case"], "timestep_seconds": timestep, "repeat": 0,
                          "frame_count": len(attempt["trajectory"]), "pose_byte_offset": offset,
                          "rows": [{key: row[key] for key in fields} for row in attempt["trajectory"]]})
    if warnings:
        raise ValueError("Replay native forward reconstruction emitted warnings")
    body_count = len(scene["bodies"])
    compressed_geometry = gzip.compress(geometry, mtime=0)
    compressed_motion = gzip.compress(bytes(motion), mtime=0)
    geometry_meta = {"href": "geometry.bin", "compressed_href": "geometry.bin.gz", "bytes": len(geometry),
                     "compressed_bytes": len(compressed_geometry), "compression": "gzip",
                     "vertex_deduplication": "Identical Float32 xyz bytes only; every triangle coordinate preserved"}
    motion_binary_meta = {"href": "motion.bin", "compressed_href": "motion.bin.gz", "bytes": len(motion),
                          "compressed_bytes": len(compressed_motion), "compression": "gzip"}
    scene["geometry_binary"] = geometry_meta
    motion_meta = {"schema": "dvidia.reference-replay-motion/v1", "body_count": body_count,
                   "frame_stride_floats": body_count * 7, "arm_joint_names": list(sim.ARM_JOINTS),
                   "cases": cases, "source": source, "scope": SCOPE,
                   "motion_binary": motion_binary_meta,
                   "reconstruction_max_absolute_residual": residuals,
                   "binary_pose_precision": "little-endian Float32; positions meters and quaternions wxyz"}
    files = {"scene.json": _json_bytes(scene), "geometry.bin": geometry,
             "motion.json": _json_bytes(motion_meta), "motion.bin": bytes(motion),
             "geometry.bin.gz": compressed_geometry, "motion.bin.gz": compressed_motion,
             "LICENSE": MIT_LICENSE.encode("utf-8")}
    root = Path(__file__).parent
    for name in ("reference_replay.html", "reference_replay.bundle.js", "reference_replay.third-party.txt"):
        destination = "index.html" if name.endswith(".html") else name
        files[destination] = _read(root / name, 8 * 1024 * 1024)
    for folder in ("universal_robots_ur5e", "robotiq_2f85"):
        for name in ("LICENSE", "README.md"):
            key = folder + "/" + name
            if key not in blobs:
                raise ValueError("Replay upstream license or README missing from pinned assets")
            files["licenses/" + key] = blobs[key]
    for name, content in receipt_files.items():
        files["receipt/" + name] = content
    files["README.md"] = ("# DVIDIA native reference replay\n\n" + SCOPE + "\n\n"
                           "Serve this directory with a local static HTTP server and open index.html. "
                           "All geometry, motion, viewer code, receipts and license notices are local. "
                           "Receipt hashes establish consistency, not authenticity. "
                           "The viewer uses Float32 body poses; receipt reconstruction checks used native Float64.\n\n"
                           "DVIDIA exporter/viewer code: MIT. UR5e model: BSD-3-Clause. "
                           "Robotiq 2F-85 model: BSD-2-Clause. Original model notices and READMEs "
                           "are under licenses/. Viewer dependencies are listed in reference_replay.third-party.txt.\n").encode("utf-8")
    seal = {"schema": "dvidia.reference-replay-integrity/v1", "scope": "Artifact consistency only; not authenticity or physical qualification",
            "files": {name: {"bytes": len(value), "sha256": sha256(value).hexdigest()} for name, value in files.items()}}
    files["replay-integrity.json"] = _json_bytes(seal)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    return {"schema": "dvidia.reference-replay-export/v1", "output": str(destination),
            "physical_robot_ready": False, "scope": SCOPE, "cases": len(cases),
            "reconstructed_frames": sum(case["frame_count"] for case in cases), "body_count": body_count,
            "reconstruction_max_absolute_residual": residuals, "bytes": sum(len(value) for value in files.values()),
            "geometry_binary": geometry_meta, "motion_binary": motion_binary_meta,
            "files": seal["files"]}
