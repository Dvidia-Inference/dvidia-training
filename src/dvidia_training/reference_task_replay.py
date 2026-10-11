"""Replay a sealed native block task, reconstructing geometry without stepping physics.

The mechanics replay exporter and its frozen frontend stay unchanged. This
module reuses their geometry format and packages a task-specific explanatory
page with measured successes, failures, controls and the complete source tape.
"""
from __future__ import annotations

from dataclasses import asdict
import gzip
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path

from . import reference_replay as replay
from . import reference_sim as sim


PROFILE = "ur5e-2f85-rigid-block-task-v0.1"
SCOPE = ("Recorded authored, privileged-state rigid-block contact simulation. "
         "Native arm, gripper, table and free-object geometry reconstructed from full recorded qpos with mj_forward only. "
         "Success and failure labels are copied from the sealed benchmark; the viewer does not recompute them. "
         "No live physics, learned policy, shoe physics or physical robot qualification.")
MAX_FRAMES_PER_ATTEMPT = 5001
MAX_SCENES = 3
COMMAND_SCOPE = ("Targets are issued at the recorded observation boundary. At the terminal row, "
                 "the viewer carries forward the last issued target for reference; no new terminal command is invented. "
                 "Targets are requested controller commands, not measured joint torque or fingertip force.")


def _source_bundle(directory):
    from .reference_task_benchmark import load_task_bundle
    report, files = load_task_bundle(directory)
    # Reverify the immutable byte snapshot used below, never a second mutable
    # disk read. Retain the original source integrity bytes in the export.
    manifest = replay._decode(files["integrity.json"])
    if (manifest.get("schema") != "dvidia.reference-task-integrity/v1"
            or set(manifest.get("files", {})) != set(files) - {"integrity.json"}):
        raise ValueError("Task replay source integrity framing mismatch")
    for name, expected in manifest["files"].items():
        data = files[name]
        if (type(data) is not bytes or type(expected) is not dict
                or set(expected) != {"bytes", "sha256"}
                or type(expected["bytes"]) is not int or expected["bytes"] != len(data)
                or expected["sha256"] != sha256(data).hexdigest()):
            raise ValueError("Task replay source snapshot digest mismatch")
    if replay._decode(files["report.json"]) != report:
        raise ValueError("Task replay source report snapshot mismatch")
    return report, files


def _select_scenes(report, scene_ids=None):
    scenes = report["scenes"]
    known = [item["id"] for item in scenes]
    if scene_ids is not None:
        if (type(scene_ids) is not list or not 1 <= len(scene_ids) <= MAX_SCENES
                or any(type(item) is not str or item not in known for item in scene_ids)
                or len(set(scene_ids)) != len(scene_ids)):
            raise ValueError("Task replay accepts one to three distinct known scene IDs")
        return list(scene_ids)
    selected = []
    placements = [a for a in report["attempts"] if a["control"] == "placement"]
    for success in (True, False):
        candidate = next((a["scene_id"] for a in placements
                          if a["valid"] is True and a["success"] is success), None)
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    for scene_id in known:
        if len(selected) >= MAX_SCENES:
            break
        if scene_id not in selected:
            selected.append(scene_id)
    return selected


def _vector(value, length, description):
    return replay._vector(value, length, "task " + description)


def _quaternion(value, description):
    numbers = _vector(value, 4, description)
    if not math.isclose(sum(x*x for x in numbers), 1., rel_tol=0., abs_tol=1e-8):
        raise ValueError("Task replay quaternion must be normalized")
    return numbers


def _validate_episode(episode, summary, scene, profile=PROFILE):
    if (type(episode) is not dict or episode.get("schema") != "dvidia.reference-task-episode/v1"
            or episode.get("profile") != profile or episode.get("valid") is not True
            or type(episode.get("success")) is not bool
            or episode.get("id") != summary["id"] or episode.get("scene_id") != scene["id"]
            or episode.get("control") != summary["control"]
            or episode.get("config") != scene["config"] or type(episode.get("seed")) is not int
            or episode.get("seed") != scene["seed"]
            or episode.get("warnings") != [] or episode.get("error") is not None):
        raise ValueError("Task replay episode identity, scope or configuration mismatch")
    replay._digest(episode.get("compiled_model_sha256"), "task compiled model")
    rows = episode.get("trajectory")
    actions = episode.get("actions")
    if type(rows) is not list or not 2 <= len(rows) <= MAX_FRAMES_PER_ATTEMPT:
        raise ValueError("Task replay requires a bounded complete trajectory")
    if type(actions) is not list or len(actions) != len(rows) - 1:
        raise ValueError("Task replay requires one issued action per nonterminal observation")
    interval = replay._number(scene["config"].get("control_dt"), "task control period")
    if interval <= 0:
        raise ValueError("Task replay control period must be positive")
    previous = -math.inf
    for index, row in enumerate(rows):
        if type(row) is not dict:
            raise ValueError("Task replay row must be an object")
        elapsed = replay._number(row.get("t_seconds"), "task observation time")
        if (elapsed <= previous or not math.isclose(elapsed, index*interval, rel_tol=0., abs_tol=1e-9)):
            raise ValueError("Task replay times must follow complete ordered control boundaries")
        previous = elapsed
        for field, length in (("qpos", 21), ("qvel", 20), ("joint_position_rad", 6),
                              ("joint_velocity_rad_s", 6), ("linkage_position_rad", 8),
                              ("tcp_position_m", 3), ("object_position_m", 3)):
            _vector(row.get(field), length, field)
        for field in ("tcp_quaternion_wxyz", "object_quaternion_wxyz"):
            _quaternion(row.get(field), field)
        _quaternion(row["qpos"][17:21], "free object qpos orientation")
        if row["qpos"][:6] != row["joint_position_rad"] or row["qpos"][6:14] != row["linkage_position_rad"]:
            raise ValueError("Task replay named coordinates disagree with full qpos")
        if row["qvel"][:6] != row["joint_velocity_rad_s"]:
            raise ValueError("Task replay named arm velocities disagree with full qvel")
        if replay._number(row.get("pad_center_separation_m"), "task pad separation") < 0:
            raise ValueError("Task replay pad separation cannot be negative")
        if type(row.get("contact_count")) is not int or not 0 <= row["contact_count"] <= 4096:
            raise ValueError("Task replay contact count must be a bounded integer")
        if index == len(rows)-1:
            if row.get("issued_joint_targets_rad") is not None or row.get("issued_gripper_0_255") is not None:
                raise ValueError("Task replay terminal row cannot invent a new issued command")
            continue
        action = actions[index]
        if (type(action) is not dict or not math.isclose(replay._number(action.get("t_seconds"), "task action time"),
                                                       elapsed, rel_tol=0., abs_tol=1e-9)):
            raise ValueError("Task replay issued action time mismatch")
        targets = _vector(action.get("joint_targets_rad"), 6, "joint targets")
        gripper = replay._number(action.get("gripper_0_255"), "task gripper command")
        if not 0 <= gripper <= 255:
            raise ValueError("Task replay issued gripper command out of range")
        if (row.get("issued_joint_targets_rad") != targets or row.get("issued_gripper_0_255") != gripper
                or type(action.get("phase")) is not str or len(action["phase"]) > 128):
            raise ValueError("Task replay row command differs from the sealed action tape")
    return rows, actions


def _poses(model, rows, actions, residuals):
    mj, np = sim._native_imports()
    if model.nq != 21 or model.nv != 20 or model.nu != 7:
        raise ValueError("Task replay native mechanical contract mismatch")
    for address, name in enumerate(sim.ARM_JOINTS):
        index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if index < 0 or int(model.jnt_qposadr[index]) != address:
            raise ValueError("Task replay arm coordinate order mismatch")
    free = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "task_block_free")
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "task_block")
    if free < 0 or body < 0 or int(model.jnt_qposadr[free]) != 14:
        raise ValueError("Task replay free-object coordinate contract mismatch")
    data = mj.MjData(model)
    frames = []
    for index, row in enumerate(rows):
        command = actions[min(index, len(actions)-1)]
        data.qpos[:] = row["qpos"]
        data.qvel[:] = row["qvel"]
        data.ctrl[:] = command["joint_targets_rad"] + [command["gripper_0_255"]]
        mj.mj_forward(model, data)
        observed = sim._sample(model, data, row["t_seconds"], data.ctrl)
        observed["object_position_m"] = data.xpos[body].tolist()
        observed["object_quaternion_wxyz"] = data.xquat[body].tolist()
        for field in ("tcp_position_m", "object_position_m"):
            delta = float(np.max(np.abs(np.asarray(observed[field])-row[field])))
            residuals[field] = max(residuals[field], delta)
            if delta > replay.POSITION_TOLERANCE_M:
                raise ValueError(f"Task replay reconstructed {field} differs from the sealed trace")
        delta = abs(observed["pad_center_separation_m"]-row["pad_center_separation_m"])
        residuals["pad_center_separation_m"] = max(residuals["pad_center_separation_m"], delta)
        if delta > replay.POSITION_TOLERANCE_M:
            raise ValueError("Task replay reconstructed pad separation differs from the sealed trace")
        for field in ("tcp_quaternion_wxyz", "object_quaternion_wxyz"):
            current, recorded = np.asarray(observed[field]), np.asarray(row[field])
            delta = float(min(np.max(np.abs(current-recorded)), np.max(np.abs(current+recorded))))
            residuals[field] = max(residuals[field], delta)
            if delta > replay.ROTATION_TOLERANCE:
                raise ValueError(f"Task replay reconstructed {field} differs from the sealed trace")
        pose = np.concatenate((data.xpos, data.xquat), axis=1)
        if not np.isfinite(pose).all():
            raise ValueError("Task replay reconstructed a nonfinite body pose")
        frames.append(pose)
    return np.asarray(frames, dtype="<f4").tobytes(order="C")


def _display_rows(rows, actions):
    fields = ("t_seconds", "joint_position_rad", "linkage_position_rad", "contact_count",
              "tcp_position_m", "tcp_quaternion_wxyz", "pad_center_separation_m")
    display = []
    for index, row in enumerate(rows):
        action = actions[min(index, len(actions)-1)]
        copied = {key: row[key] for key in fields}
        copied.update({"issued_joint_targets_rad": action["joint_targets_rad"],
                       "issued_gripper_0_255": action["gripper_0_255"],
                       "issued_at_seconds": action["t_seconds"], "phase": action["phase"],
                       "terminal_command_carried_forward": index == len(rows)-1})
        display.append(copied)
    return display


def _task_html(template, report, included, selected_scenes, excluded):
    changes = {
        "DVIDIA · Six-axis 3D reference replay": "DVIDIA · Native block task replay",
        "Six joints. A world in three dimensions.": "A contact task in three dimensions.",
        "UR5e + Robotiq 2F-85 · Native reference meshes and recorded simulator poses.":
            "UR5e + Robotiq 2F-85 · Free rigid block, table and recorded task attempts.",
        "These are unloaded authored motions. They do not demonstrate learned skills, shoe manipulation, a real robot, or calibrated transfer accuracy. The floor, grid, axes and markers are viewing aids.":
            "These are authored-controller task attempts with a free rigid block and table. Success and failure labels come from the sealed source measurements. They do not demonstrate a learned policy, shoe manipulation or physical robot qualification. The grid, axes and joint markers are viewing aids.",
        "Numerical benchmark ↗": "Complete task benchmark ↗",
    }
    text = template.decode("utf-8")
    for old, new in changes.items():
        if text.count(old) != 1:
            raise ValueError("Task replay explanatory template changed; review its task-specific copy")
        text = text.replace(old, escape(new) if "<" in new else new)
    rows = []
    for attempt in included:
        status = "Completed" if attempt["success"] else "Failed"
        reason = attempt.get("failure_reason") or attempt.get("reason") or "—"
        rows.append("<tr><td>"+escape(attempt["scene_id"])+"</td><td>"+escape(attempt["control"])+
                    "</td><td>"+status+"</td><td>"+escape(str(reason))+"</td><td><a href=\"receipt/"+
                    escape(attempt["result_file"], quote=True)+"\">Exact attempt JSON</a></td></tr>")
    section = ("<section id=\"task-evidence\"><h2>Replay excerpt, complete denominator.</h2><p>"+
               str(len(included))+" recorded attempts from "+str(len(selected_scenes))+" scenes are shown here. "
               "The complete benchmark contains "+str(report["attempted"])+" planned attempts across "+
               str(len(report["scenes"]))+" scenes; this excerpt is not a success-rate denominator. "
               "All source attempts, including invalid attempts and negative controls, remain in the linked receipt.</p>"+
               "<p>"+escape(COMMAND_SCOPE)+"</p><div class=\"task-scroll\"><table><thead><tr><th>Scene</th>"
               "<th>Controller</th><th>Recorded outcome</th><th>Reason</th><th>Source</th></tr></thead><tbody>"+
               "".join(rows)+"</tbody></table></div><p class=\"tiny\">"+str(len(excluded))+
               " source attempts are outside the render selection or numerically invalid. "
               "Their exact IDs and exclusion reasons are listed in motion.json. The renderer never scores task success.</p></section>")
    marker = '<section class="footnotes">'
    if text.count(marker) != 1:
        raise ValueError("Task replay explanatory section anchor changed")
    text = text.replace(marker, section+marker)
    text = text.replace("</head>", "<style>#task-evidence{margin-top:24px;border-top:1px solid var(--line);padding-top:20px}"
                        "#task-evidence p{font-size:14px}.task-scroll{overflow:auto}.task-scroll table{border-collapse:collapse;"
                        "width:100%;font:12px/1.5 system-ui}.task-scroll td,.task-scroll th{text-align:left;padding:10px 8px;"
                        "border-bottom:1px solid var(--line)}</style></head>")
    return text.encode("utf-8")


def export_task_replay(assets, bundle, output, *, scene_ids=None):
    """Export selected recorded task scenes and all their valid controls.

    Original source tapes and integrity bytes are retained. Selection is an
    explicitly declared excerpt, never a new denominator or recomputed score.
    """
    from . import reference_task as task
    from .reference_task_benchmark import CONTROLS
    report, receipt_files = _source_bundle(bundle)
    if (report.get("profile") != task.PROFILE or task.PROFILE != PROFILE
            or report.get("physical_robot_ready") is not False or report.get("learned_policy") is not False
            or report.get("hardware", {}).get("mujoco") != "3.15.0"):
        raise ValueError("Task replay requires the simulation-only pinned task profile and MuJoCo 3.15.0")
    provenance, blobs = sim._asset_payload(assets)
    source_provenance = report.get("asset_provenance", {})
    if (source_provenance.get("commit") != provenance["commit"]
            or source_provenance.get("lock_sha256") != provenance["lock_sha256"]):
        raise ValueError("Task replay pinned asset provenance mismatch")
    selected_scenes = _select_scenes(report, scene_ids)
    by_scene = {scene["id"]: scene for scene in report["scenes"]}
    included = [attempt for scene_id in selected_scenes for control in CONTROLS
                for attempt in report["attempts"]
                if attempt["scene_id"] == scene_id and attempt["control"] == control and attempt["valid"]]
    included_ids = {attempt["id"] for attempt in included}
    excluded = [{"id": attempt["id"], "reason": "numerically_invalid" if not attempt["valid"] else "scene_not_selected"}
                for attempt in report["attempts"] if attempt["id"] not in included_ids]
    if not included:
        raise ValueError("Task replay selection has no numerically valid attempts")
    source = {"profile": task.PROFILE, "asset_commit": provenance["commit"],
              "asset_lock_sha256": provenance["lock_sha256"],
              "report_sha256": sha256(receipt_files["report.json"]).hexdigest(),
              "protocol_sha256": sha256(receipt_files["protocol.json"]).hexdigest(),
              "report_href": "receipt/report.html", "receipt_href": "receipt/report.json",
              "exporter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "mujoco": "3.15.0",
              "compiled_model_sha256": {}, "asset_licenses": provenance["licenses"],
              "selected_scene_ids": selected_scenes, "included_attempt_ids": [a["id"] for a in included],
              "excluded_attempts": excluded, "benchmark_attempted": report["attempted"],
              "selection_scope": "Replay excerpt only; original full benchmark denominator retained",
              "command_alignment": COMMAND_SCOPE}
    scene, geometry = None, None
    motion, cases, warnings, models, native_scenes = bytearray(), [], [], {}, {}
    residuals = {key: 0. for key in ("tcp_position_m", "tcp_quaternion_wxyz", "object_position_m",
                                     "object_quaternion_wxyz", "pad_center_separation_m")}
    with sim.native_warnings(warnings):
        for summary in included:
            config_scene = by_scene[summary["scene_id"]]
            episode = replay._decode(receipt_files[summary["result_file"]])
            rows, actions = _validate_episode(episode, summary, config_scene, task.PROFILE)
            key = json.dumps(config_scene["config"], sort_keys=True, allow_nan=False)
            if key not in models:
                try:
                    config = task.ReferenceTaskConfig(**config_scene["config"])
                except TypeError as exc:
                    raise ValueError("Task replay configuration fields differ from the native contract") from exc
                if json.loads(json.dumps(asdict(config))) != config_scene["config"]:
                    raise ValueError("Task replay native configuration differs from the source recipe")
                models[key] = task.build_task_model(assets, config)
            model = models[key]
            digest = sim.model_digest(model)
            if digest != episode["compiled_model_sha256"]:
                raise ValueError("Task replay compiled model digest differs from the sealed attempt")
            source["compiled_model_sha256"][summary["id"]] = digest
            # Matched controls share the compiled scene. Decode its native
            # triangles once; still verify each distinct scene configuration.
            if key not in native_scenes:
                candidate_scene, candidate_geometry = replay._scene(model, source)
                candidate_scene["scope"] = SCOPE
                native_scenes[key] = candidate_scene, candidate_geometry
            candidate_scene, candidate_geometry = native_scenes[key]
            if scene is None:
                scene, geometry = candidate_scene, candidate_geometry
            elif scene != candidate_scene or geometry != candidate_geometry:
                raise ValueError("Selected task scene geometry differs; export one scene ID at a time")
            offset = len(motion)
            motion.extend(_poses(model, rows, actions, residuals))
            if len(motion) > replay.MAX_BINARY_BYTES:
                raise ValueError("Task replay motion binary exceeds its byte bound")
            cases.append({"name": summary["id"]+" ["+("completed" if summary["success"] else "failed")+"]",
                          "attempt_id": summary["id"], "scene_id": summary["scene_id"],
                          "control": summary["control"], "recorded_success": summary["success"],
                          "timestep_seconds": float(model.opt.timestep), "repeat": 0,
                          "frame_count": len(rows), "pose_byte_offset": offset,
                          "rows": _display_rows(rows, actions),
                          "source_result_href": "receipt/"+summary["result_file"]})
    if warnings:
        raise ValueError("Task replay native reconstruction emitted warnings")
    body_count = len(scene["bodies"])
    packed_geometry, packed_motion = gzip.compress(geometry, mtime=0), gzip.compress(bytes(motion), mtime=0)
    geometry_meta = {"href": "geometry.bin", "compressed_href": "geometry.bin.gz", "bytes": len(geometry),
                     "compressed_bytes": len(packed_geometry), "compression": "gzip",
                     "vertex_deduplication": "Identical Float32 xyz bytes only; native triangle positions preserved"}
    motion_meta = {"href": "motion.bin", "compressed_href": "motion.bin.gz", "bytes": len(motion),
                   "compressed_bytes": len(packed_motion), "compression": "gzip"}
    scene["geometry_binary"] = geometry_meta
    motion_data = {"schema": "dvidia.reference-replay-motion/v1", "body_count": body_count,
                   "frame_stride_floats": body_count*7, "arm_joint_names": list(sim.ARM_JOINTS),
                   "cases": cases, "source": source, "scope": SCOPE, "motion_binary": motion_meta,
                   "reconstruction_max_absolute_residual": residuals,
                   "binary_pose_precision": "little-endian Float32; positions meters, quaternions wxyz"}
    scene_bytes, motion_bytes = replay._json_bytes(scene), replay._json_bytes(motion_data)
    if max(len(scene_bytes), len(motion_bytes)) > 12*1024*1024:
        raise ValueError("Task replay JSON exceeds the frozen viewer bound; export fewer scene IDs")
    root = Path(__file__).parent
    files = {"scene.json": scene_bytes, "motion.json": motion_bytes, "geometry.bin": geometry,
             "motion.bin": bytes(motion), "geometry.bin.gz": packed_geometry, "motion.bin.gz": packed_motion,
             "LICENSE": replay.MIT_LICENSE.encode("utf-8")}
    files["index.html"] = _task_html(replay._read(root/"reference_replay.html", 8*1024*1024),
                                     report, included, selected_scenes, excluded)
    for name in ("reference_replay.bundle.js", "reference_replay.third-party.txt"):
        files[name] = replay._read(root/name, 8*1024*1024)
    for folder in ("universal_robots_ur5e", "robotiq_2f85"):
        for name in ("LICENSE", "README.md"):
            key = folder+"/"+name
            if key not in blobs:
                raise ValueError("Task replay pinned model notice missing")
            files["licenses/"+key] = blobs[key]
    for name, content in receipt_files.items():
        files["receipt/"+name] = content
    files["README.md"] = ("# DVIDIA native block-task replay\n\n"+SCOPE+"\n\n"+COMMAND_SCOPE+
                          "\n\nServe this directory with a local static HTTP server. All viewer code, native geometry, "
                          "recorded motion, receipts and licensing notices are local. Replay selection is an excerpt; "
                          "the complete planned denominator and every raw attempt remain under receipt/. "
                          "Artifact hashes show consistency, not authenticity or physical qualification. "
                          "Float64 reconstruction is checked before Float32 poses are written.\n\n"
                          "DVIDIA exporter/viewer: MIT. UR5e: BSD-3-Clause. Robotiq2F-85: BSD-2-Clause. "
                          "Native model notices and READMEs are retained under licenses/. "
                          "Bundled viewer dependency notices: reference_replay.third-party.txt.\n").encode("utf-8")
    seal = {"schema": "dvidia.reference-replay-integrity/v1", "scope": "Artifact consistency only; no authenticity or physical qualification",
            "files": {name: {"bytes": len(data), "sha256": sha256(data).hexdigest()} for name, data in files.items()}}
    files["replay-integrity.json"] = replay._json_bytes(seal)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = destination/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    return {"schema": "dvidia.reference-task-replay-export/v1", "output": str(destination), "scope": SCOPE,
            "physical_robot_ready": False, "learned_policy": False, "cases": len(cases),
            "selected_scene_ids": selected_scenes, "included_attempt_ids": source["included_attempt_ids"],
            "excluded_attempts": excluded, "benchmark_attempted": report["attempted"],
            "reconstructed_frames": sum(case["frame_count"] for case in cases), "body_count": body_count,
            "reconstruction_max_absolute_residual": residuals, "geometry_binary": geometry_meta,
            "motion_binary": motion_meta, "bytes": sum(len(data) for data in files.values()), "files": seal["files"]}
