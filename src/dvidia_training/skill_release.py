"""Proposed future release contract and offline evidence-binding inspector.

This schema is NOT an existing DVIDIA export. It does not execute policies,
train from folders, validate human rights/identities, or qualify physical robots.
Draft reports distinguish locally verified bytes from declared evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


KIND = "dvidia.skill-release-proposal"
STAGES = ("source", "curated_dataset", "grounded_scenario", "executable", "evaluation", "hardware_qualification")
FAMILIES = ("rigid_pick_place", "articulated_open_close", "contact_insertion_wiping", "deformable_cable_cloth", "bimanual")
ROLES = ("source", "dataset", "review", "rights", "split", "loader_receipt", "scenario", "profile",
         "model", "kinematics", "predicates", "physics_validation", "parameter_domain", "executable",
         "training_receipt", "io_contract", "initiation", "termination", "recovery", "runtime",
         "protocol", "evaluation_receipt", "baseline", "calibration", "safety", "hardware_receipt")
MAX_MANIFEST_BYTES = 262_144
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


class SkillReleaseError(ValueError):
    pass


def _bad(path, message):
    raise SkillReleaseError(f"{path}: {message}")


def _record(value, path, fields):
    if type(value) is not dict or set(value) != set(fields):
        _bad(path, "expected exactly " + ", ".join(fields))
    return value


def _text(value, path, limit=256):
    if type(value) is not str or not value.strip() or len(value) > limit \
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        _bad(path, "expected nonblank bounded plain text")
    return value


def _id(value, path):
    if type(value) is not str or not _ID.fullmatch(value):
        _bad(path, "expected a stable identifier")
    return value


def _sha(value, path):
    if type(value) is not str or not _SHA.fullmatch(value):
        _bad(path, "expected lowercase SHA-256")
    return value


def _number(value, path, lo, hi):
    if type(value) not in (int, float) or not lo <= value <= hi or not math.isfinite(value):
        _bad(path, f"expected a finite number in [{lo}, {hi}]")
    return float(value)


def _integer(value, path, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        _bad(path, f"expected an integer in [{lo}, {hi}]")
    return value


def _list(value, path, lo=0, hi=128):
    if type(value) is not list or not lo <= len(value) <= hi:
        _bad(path, f"expected {lo}–{hi} items")
    return value


def _names(value, path, lo=1, hi=128):
    rows = [_id(v, path) for v in _list(value, path, lo, hi)]
    if len(set(rows)) != len(rows):
        _bad(path, "duplicate identifiers")
    return rows


def _json(value, limit=MAX_MANIFEST_BYTES):
    def pairs(rows):
        result = {}
        for key, item in rows:
            if key in result:
                _bad(key, "duplicate JSON key")
            result[key] = item
        return result

    def constant(token):
        _bad("JSON", f"nonfinite constant {token}")

    try:
        if type(value) is dict:
            raw = json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf8")
            result = value
        elif type(value) in (str, bytes):
            raw = value.encode("utf8") if type(value) is str else value
            if len(raw) > limit:
                _bad("JSON", "document exceeds its byte limit")
            result = json.loads(raw.decode("utf8"), object_pairs_hook=pairs, parse_constant=constant)
            json.dumps(result, allow_nan=False)
        else:
            _bad("JSON", "expected a plain dict, UTF-8 bytes or JSON text")
        if len(raw) > limit:
            _bad("JSON", "document exceeds its byte limit")
        if type(result) is not dict:
            _bad("JSON", "expected an object")
        return result, hashlib.sha256(raw).hexdigest()
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        if isinstance(exc, SkillReleaseError):
            raise
        raise SkillReleaseError("Expected finite, bounded UTF-8 JSON.") from exc


def _path(value):
    path = _text(value, "artifact.path", 512)
    if path.startswith("/") or "\\" in path or ":" in path \
            or any(part in ("", ".", "..") for part in path.split("/")):
        _bad("artifact.path", "expected a safe bundle-relative path")
    return path


def _artifact_ref(value, path, artifacts, role=None):
    _id(value, path)
    if value not in artifacts or (role is not None and artifacts[value]["role"] != role):
        _bad(path, "reference must name a declared artifact of the expected role")
    return value


def _profile(value, artifacts):
    fields = ("kind", "schema_version", "model_artifact_id", "kinematics_artifact_id", "arm_count",
              "base_frame", "tcp_frame", "tcp_transform", "joints", "control_hz", "gripper", "sensors",
              "workspace", "limitations")
    row = _record(value, "profile", fields)
    if row["kind"] != "dvidia.robot-profile-proposal" or type(row["schema_version"]) is not int or row["schema_version"] != 1:
        _bad("profile", "unsupported proposed profile kind/version")
    _artifact_ref(row["model_artifact_id"], "profile.model_artifact_id", artifacts, "model")
    _artifact_ref(row["kinematics_artifact_id"], "profile.kinematics_artifact_id", artifacts, "kinematics")
    _integer(row["arm_count"], "profile.arm_count", 1, 2)
    for name in ("base_frame", "tcp_frame"):
        _id(row[name], "profile." + name)
    transform = _list(row["tcp_transform"], "profile.tcp_transform", 7, 7)
    transform = [_number(v, "profile.tcp_transform", -10, 10) for v in transform]
    if not math.isclose(sum(v*v for v in transform[3:]), 1, rel_tol=0, abs_tol=1e-6):
        _bad("profile.tcp_transform", "quaternion [w,x,y,z] must have unit norm")
    joint_names = []
    for joint in _list(row["joints"], "profile.joints", 1, 64):
        _record(joint, "joint", ("name", "type", "unit", "range", "max_velocity", "max_effort", "effort_unit", "actuation"))
        joint_names.append(_id(joint["name"], "joint.name"))
        if joint["type"] not in ("hinge", "slide"):
            _bad("joint.type", "only explicitly ordered hinge/slide joints are supported")
        expected_units = ("rad", "Nm") if joint["type"] == "hinge" else ("m", "N")
        if (joint["unit"], joint["effort_unit"]) != expected_units:
            _bad("joint.unit", "position/effort units must match the joint type")
        limits = [_number(v, "joint.range", -100, 100) for v in _list(joint["range"], "joint.range", 2, 2)]
        if limits[0] >= limits[1]:
            _bad("joint.range", "lower limit must precede upper limit")
        _number(joint["max_velocity"], "joint.max_velocity", 1e-6, 1000)
        _number(joint["max_effort"], "joint.max_effort", 1e-6, 1e6)
        if joint["actuation"] not in ("position", "velocity", "torque"):
            _bad("joint.actuation", "unknown command semantics")
    if len(set(joint_names)) != len(joint_names):
        _bad("profile.joints", "duplicate joint names")
    _number(row["control_hz"], "profile.control_hz", .1, 10000)
    gripper = _record(row["gripper"], "gripper", ("type", "aperture_m", "force_limit_n", "geometry_sha256"))
    if gripper["type"] not in ("parallel_jaw", "suction", "dexterous", "none"):
        _bad("gripper.type", "unsupported gripper contract")
    gap = [_number(v, "gripper.aperture_m", 0, 2) for v in _list(gripper["aperture_m"], "gripper.aperture_m", 2, 2)]
    if gap[0] > gap[1]:
        _bad("gripper.aperture_m", "inverted aperture bounds")
    _number(gripper["force_limit_n"], "gripper.force_limit_n", 0, 1e5)
    _sha(gripper["geometry_sha256"], "gripper.geometry_sha256")
    sensor_names = []
    for sensor in _list(row["sensors"], "profile.sensors", 1, 64):
        _record(sensor, "sensor", ("name", "kind", "shape", "unit", "frame", "rate_hz"))
        sensor_names.append(_id(sensor["name"], "sensor.name"))
        if sensor["kind"] not in ("joint_position", "joint_velocity", "camera", "force_torque", "tactile", "privileged_state"):
            _bad("sensor.kind", "unknown observation source")
        for dimension in _list(sensor["shape"], "sensor.shape", 1, 4):
            _integer(dimension, "sensor.shape", 1, 4096)
        _text(sensor["unit"], "sensor.unit", 40)
        _id(sensor["frame"], "sensor.frame")
        _number(sensor["rate_hz"], "sensor.rate_hz", .1, 10000)
    if len(set(sensor_names)) != len(sensor_names):
        _bad("profile.sensors", "duplicate sensor names")
    workspace = _record(row["workspace"], "workspace", ("frame", "minimum", "maximum"))
    if workspace["frame"] != row["base_frame"]:
        _bad("workspace.frame", "workspace must use the declared base frame")
    low = [_number(v, "workspace.minimum", -100, 100) for v in _list(workspace["minimum"], "workspace.minimum", 3, 3)]
    high = [_number(v, "workspace.maximum", -100, 100) for v in _list(workspace["maximum"], "workspace.maximum", 3, 3)]
    if any(a >= b for a, b in zip(low, high)):
        _bad("workspace", "inverted workspace bounds")
    for limitation in _list(row["limitations"], "profile.limitations", 1, 32):
        _text(limitation, "profile.limitations", 512)
    return row


def _io(value, profile, profile_sha):
    row = _record(value, "io_contract", ("kind", "schema_version", "profile_sha256", "rate_hz",
                                       "joint_order", "observations", "actions"))
    if row["kind"] != "dvidia.policy-io-proposal" or type(row["schema_version"]) is not int or row["schema_version"] != 1:
        _bad("io_contract", "unsupported I/O kind/version")
    if row["profile_sha256"] != profile_sha or row["joint_order"] != [j["name"] for j in profile["joints"]]:
        _bad("io_contract", "profile hash and joint order must match exactly")
    if _number(row["rate_hz"], "io_contract.rate_hz", .1, 10000) != profile["control_hz"]:
        _bad("io_contract.rate_hz", "rate must match the qualified profile")
    names = []
    sensors = {s["name"]: s for s in profile["sensors"]}
    for obs in _list(row["observations"], "io_contract.observations", 1, 64):
        _record(obs, "observation", ("name", "sensor", "shape", "unit", "frame"))
        names.append(_id(obs["name"], "observation.name"))
        _id(obs["sensor"], "observation.sensor")
        if obs["sensor"] not in sensors:
            _bad("observation.sensor", "required sensor is unavailable in the profile")
        sensor = sensors[obs["sensor"]]
        if any(obs[key] != sensor[key] for key in ("shape", "unit", "frame")):
            _bad("observation", "shape, unit and frame must match the sensor contract")
    if len(set(names)) != len(names):
        _bad("io_contract.observations", "duplicate channels")
    names = []
    for action in _list(row["actions"], "io_contract.actions", 1, 16):
        _record(action, "action", ("name", "semantics", "joint_names", "units", "minimum", "maximum"))
        names.append(_id(action["name"], "action.name"))
        if action["semantics"] not in ("absolute_joint_position", "joint_velocity", "joint_torque", "gripper_aperture"):
            _bad("action.semantics", "this draft inspector supports explicit joint/aperture contracts only")
        if action["semantics"] == "gripper_aperture":
            if action["joint_names"] != [] or action["units"] != ["m"]:
                _bad("action", "aperture channel requires metre units and no joint mapping")
            limits = [profile["gripper"]["aperture_m"]]
        else:
            if action["joint_names"] != row["joint_order"]:
                _bad("action.joint_names", "joint channel must preserve the complete profile order")
            suffix = {"absolute_joint_position": "", "joint_velocity": "/s", "joint_torque": None}[action["semantics"]]
            units = [j["effort_unit"] if suffix is None else j["unit"] + suffix for j in profile["joints"]]
            if action["units"] != units:
                _bad("action.units", "command units do not match the profile joints")
            mode = {"absolute_joint_position": "position", "joint_velocity": "velocity", "joint_torque": "torque"}[action["semantics"]]
            if any(j["actuation"] != mode for j in profile["joints"]):
                _bad("action.semantics", "profile actuation does not support this command mode")
            limits = [j["range"] if mode == "position" else [-j["max_velocity"], j["max_velocity"]]
                      if mode == "velocity" else [-j["max_effort"], j["max_effort"]] for j in profile["joints"]]
        low = _list(action["minimum"], "action.minimum", len(limits), len(limits))
        high = _list(action["maximum"], "action.maximum", len(limits), len(limits))
        for a, b, bound in zip(low, high, limits):
            _number(a, "action.minimum", bound[0], bound[1]); _number(b, "action.maximum", bound[0], bound[1])
            if a > b:
                _bad("action", "inverted command bounds")
    if len(set(names)) != len(names):
        _bad("io_contract.actions", "duplicate channels")
    return row


def validate_skill_release(document, artifact_bytes: Mapping[str, bytes] | None = None):
    """Check draft structure, local content hashes, contracts and evidence bindings.

    artifact_bytes keys are declared IDs, never paths/URLs. Missing or altered
    bytes block stages. No artifact is executed or fetched. Even complete local
    evidence does not promote a draft or establish external evaluation integrity.
    """
    manifest, manifest_sha = _json(document)
    _record(manifest, "manifest", ("kind", "schema_version", "status", "release", "task", "artifacts", "dependencies", "stages"))
    if manifest["kind"] != KIND or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["status"] != "draft":
        _bad("manifest", "only the explicitly proposed version-1 draft schema is supported")
    release = _record(manifest["release"], "release", ("id", "version", "previous_manifest_sha256"))
    _id(release["id"], "release.id")
    if type(release["version"]) is not str or not re.fullmatch(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?", release["version"]):
        _bad("release.version", "expected an immutable semantic version")
    if release["previous_manifest_sha256"] is not None:
        _sha(release["previous_manifest_sha256"], "release.previous_manifest_sha256")
    task = _record(manifest["task"], "task", ("id", "family", "goal", "distribution"))
    _id(task["id"], "task.id")
    if task["family"] not in FAMILIES:
        _bad("task.family", "unknown proposed task family")
    _text(task["goal"], "task.goal", 1200); _text(task["distribution"], "task.distribution", 1200)
    artifacts = {}
    paths = set()
    for artifact in _list(manifest["artifacts"], "artifacts", 1, 128):
        _record(artifact, "artifact", ("id", "role", "path", "sha256", "bytes"))
        identity = _id(artifact["id"], "artifact.id")
        if identity in artifacts or artifact["role"] not in ROLES:
            _bad("artifacts", "duplicate ID or unsupported artifact role")
        path = _path(artifact["path"])
        if path in paths:
            _bad("artifacts", "duplicate bundle path")
        paths.add(path)
        _sha(artifact["sha256"], "artifact.sha256")
        _integer(artifact["bytes"], "artifact.bytes", 1, MAX_ARTIFACT_BYTES)
        artifacts[identity] = artifact
    dependencies = _list(manifest["dependencies"], "dependencies", 0, 64)
    dependency_names = []
    for dependency in dependencies:
        _record(dependency, "dependency", ("name", "version", "lock_artifact_id"))
        dependency_names.append(_id(dependency["name"], "dependency.name"))
        version = _text(dependency["version"], "dependency.version", 100)
        if not re.fullmatch(r"(?:\d+\.\d+(?:\.\d+)*(?:[A-Za-z0-9.+-]*)?|[a-f0-9]{40}|[a-f0-9]{64})", version):
            _bad("dependency.version", "dependencies require exact numeric versions or full commit digests")
        _artifact_ref(dependency["lock_artifact_id"], "dependency.lock_artifact_id", artifacts, "runtime")
    if len(set(dependency_names)) != len(dependency_names):
        _bad("dependencies", "duplicate dependency names")
    stages = _record(manifest["stages"], "stages", STAGES)
    supplied = {} if artifact_bytes is None else artifact_bytes
    if not isinstance(supplied, Mapping) or set(supplied) - set(artifacts):
        _bad("artifact_bytes", "supply only declared artifact IDs")
    verified, unavailable = {}, {}
    for identity, artifact in artifacts.items():
        data = supplied.get(identity)
        if data is None:
            unavailable[identity] = "bytes not supplied"
        elif type(data) is not bytes or len(data) != artifact["bytes"] or hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            unavailable[identity] = "byte length or SHA-256 mismatch"
        else:
            verified[identity] = artifact["sha256"]
    blockers = {stage: [] for stage in STAGES}
    satisfied = {stage: False for stage in STAGES}

    def refs(stage, row, fields):
        _record(row, stage, ("artifact_ids",) + tuple(fields))
        identities = _names(row["artifact_ids"], stage + ".artifact_ids")
        for identity in identities:
            _artifact_ref(identity, stage + ".artifact_ids", artifacts)
            if identity in unavailable:
                blockers[stage].append(f"{identity}: {unavailable[identity]}")
        return identities

    def require(stage, identity, role, listed):
        _artifact_ref(identity, stage, artifacts, role)
        if identity not in listed:
            _bad(stage, "all referenced evidence must be included in artifact_ids")
        return identity in verified

    def evidence(stage, identity):
        if identity not in verified:
            return None
        try:
            return _json(supplied[identity])[0]
        except SkillReleaseError as error:
            blockers[stage].append(f"{identity}: invalid evidence JSON ({error})")
            return None

    source = stages["source"]
    if source is None:
        _bad("source", "a draft must declare its source inventory")
    listed = refs("source", source, ("source_kind", "modalities"))
    if source["source_kind"] not in ("metadata", "human_observation", "robot_demonstration", "authored_simulation"):
        _bad("source.source_kind", "unknown source kind")
    modalities = _names(source["modalities"], "source.modalities", 1, 16)
    if set(modalities) - {"task_text", "image", "video", "robot_state", "robot_actions", "force", "tactile", "timestamps"}:
        _bad("source.modalities", "unknown declared modality")
    for identity in listed:
        require("source", identity, "source", listed)
    satisfied["source"] = not blockers["source"]

    dataset = stages["curated_dataset"]
    if dataset is not None:
        listed = refs("curated_dataset", dataset, ("dataset_artifact_id", "source_artifact_ids", "review_artifact_id",
                      "rights_artifact_id", "split_artifact_id", "loader_receipt_id", "robot_action_supervision", "clock_alignment"))
        require("curated_dataset", dataset["dataset_artifact_id"], "dataset", listed)
        for identity in _names(dataset["source_artifact_ids"], "dataset.source_artifact_ids"):
            _artifact_ref(identity, "dataset.source_artifact_ids", artifacts, "source")
            if identity not in source["artifact_ids"]:
                _bad("dataset.source_artifact_ids", "must derive from the declared source inventory")
        for field, role in (("review_artifact_id", "review"), ("rights_artifact_id", "rights"),
                            ("split_artifact_id", "split"), ("loader_receipt_id", "loader_receipt")):
            require("curated_dataset", dataset[field], role, listed)
        if type(dataset["robot_action_supervision"]) is not bool:
            _bad("dataset.robot_action_supervision", "expected boolean")
        if not dataset["robot_action_supervision"] or not {"robot_state", "robot_actions", "timestamps"} <= set(modalities):
            blockers["curated_dataset"].append("Robot state/action/timestamp supervision is absent; human video alone is insufficient.")
        if dataset["clock_alignment"] not in ("validated", "unverified"):
            _bad("dataset.clock_alignment", "unknown alignment state")
        if dataset["clock_alignment"] != "validated":
            blockers["curated_dataset"].append("Clock alignment needs validation.")
        rights = evidence("curated_dataset", dataset["rights_artifact_id"])
        if rights is not None and (rights.get("training") != "documented_granted" or rights.get("source_sha256s") !=
                                  sorted(artifacts[i]["sha256"] for i in dataset["source_artifact_ids"])):
            blockers["curated_dataset"].append("Training-rights evidence does not bind every current source hash.")
        review = evidence("curated_dataset", dataset["review_artifact_id"])
        if review is not None and (review.get("dataset_sha256") != artifacts[dataset["dataset_artifact_id"]]["sha256"]
                                   or review.get("review_status") != "curated"):
            blockers["curated_dataset"].append("Curator review must bind this dataset version and its reviewed state.")
        loader = evidence("curated_dataset", dataset["loader_receipt_id"])
        if loader is not None and (loader.get("dataset_sha256") != artifacts[dataset["dataset_artifact_id"]]["sha256"]
                                   or loader.get("reader_loaded_and_replayed") is not True):
            blockers["curated_dataset"].append("A loader receipt must identify and replay this exact dataset.")
        satisfied["curated_dataset"] = satisfied["source"] and not blockers["curated_dataset"]

    profile = None
    scenario = stages["grounded_scenario"]
    if scenario is not None:
        listed = refs("grounded_scenario", scenario, ("scenario_artifact_id", "profile_artifact_id", "predicates_artifact_id",
                      "physics_validation_artifact_id", "parameter_domain_artifact_id", "frame", "units", "physics_capabilities"))
        for field, role in (("scenario_artifact_id", "scenario"), ("profile_artifact_id", "profile"),
                            ("predicates_artifact_id", "predicates"), ("physics_validation_artifact_id", "physics_validation"),
                            ("parameter_domain_artifact_id", "parameter_domain")):
            require("grounded_scenario", scenario[field], role, listed)
        if scenario["frame"] != "world" or scenario["units"] != "m":
            _bad("grounded_scenario", "explicit world-frame metre coordinates required")
        capabilities = set(_names(scenario["physics_capabilities"], "scenario.physics_capabilities", 1, 16))
        required = {"rigid_pick_place": {"rigid_contact"}, "articulated_open_close": {"rigid_contact", "object_joints"},
                    "contact_insertion_wiping": {"rigid_contact", "force_feedback"},
                    "deformable_cable_cloth": {"deformable_contact", "material_calibration"},
                    "bimanual": {"rigid_contact", "inter_arm_collision"}}[task["family"]]
        if not required <= capabilities:
            blockers["grounded_scenario"].append("Task-family physics capabilities are missing: " + ", ".join(sorted(required-capabilities)))
        value = evidence("grounded_scenario", scenario["profile_artifact_id"])
        if value is not None:
            try:
                profile = _profile(value, artifacts)
                for field in ("model_artifact_id", "kinematics_artifact_id"):
                    identity = profile[field]
                    if identity not in verified:
                        blockers["grounded_scenario"].append(f"{identity}: exact model/kinematics bytes unavailable")
                if task["family"] == "bimanual" and profile["arm_count"] != 2:
                    blockers["grounded_scenario"].append("Bimanual tasks require an explicit two-arm profile, not a joint-count guess.")
                if task["family"] == "contact_insertion_wiping" and not any(s["kind"] in ("force_torque", "tactile") for s in profile["sensors"]):
                    blockers["grounded_scenario"].append("Contact-control family requires declared force/tactile feedback.")
            except SkillReleaseError as error:
                blockers["grounded_scenario"].append(str(error))
                profile = None
        predicates = evidence("grounded_scenario", scenario["predicates_artifact_id"])
        if predicates is not None and any(type(predicates.get(key)) is not list or not predicates[key]
                                          or any(type(v) is not str or not v.strip() for v in predicates[key])
                                          for key in ("initiation", "success", "failure", "recovery")):
            blockers["grounded_scenario"].append("Task predicates need initiation, observable success/failure, and bounded recovery declarations.")
        physics = evidence("grounded_scenario", scenario["physics_validation_artifact_id"])
        if physics is not None and (physics.get("scenario_sha256") != artifacts[scenario["scenario_artifact_id"]]["sha256"]
                                    or physics.get("scope") not in ("numerical_only", "physically_calibrated")):
            blockers["grounded_scenario"].append("Physics evidence must bind this scenario and state numerical versus physical scope.")
        satisfied["grounded_scenario"] = satisfied["source"] and profile is not None and not blockers["grounded_scenario"]

    executable = stages["executable"]
    if executable is not None:
        listed = refs("executable", executable, ("kind", "implementation_artifact_id", "io_contract_id", "initiation_id",
                      "termination_id", "recovery_id", "profile_artifact_id", "scenario_artifact_id",
                      "trained_dataset_artifact_id", "training_receipt_id"))
        if executable["kind"] not in ("authored_adapter", "learned_policy"):
            _bad("executable.kind", "explicit adapter versus learned-policy provenance required")
        for field, role in (("implementation_artifact_id", "executable"), ("io_contract_id", "io_contract"),
                            ("initiation_id", "initiation"), ("termination_id", "termination"), ("recovery_id", "recovery"),
                            ("profile_artifact_id", "profile"), ("scenario_artifact_id", "scenario")):
            require("executable", executable[field], role, listed)
        if scenario is None or any(executable[key] != scenario[key] for key in ("profile_artifact_id", "scenario_artifact_id")):
            blockers["executable"].append("Executable must bind the grounded scenario and exact robot profile.")
        if not dependencies:
            blockers["executable"].append("Pinned runtime dependencies and verified lock artifact are required.")
        for dependency in dependencies:
            if dependency["lock_artifact_id"] not in verified:
                blockers["executable"].append("Runtime lock bytes unavailable: " + dependency["lock_artifact_id"])
        if executable["kind"] == "learned_policy":
            if dataset is None or not satisfied["curated_dataset"]:
                blockers["executable"].append("Learned policy requires the evidence-checked curated robot dataset.")
            for field, role in (("trained_dataset_artifact_id", "dataset"), ("training_receipt_id", "training_receipt")):
                if executable[field] is None:
                    blockers["executable"].append("Missing learned-policy provenance: " + field)
                else:
                    require("executable", executable[field], role, listed)
            if dataset is not None and executable["trained_dataset_artifact_id"] != dataset["dataset_artifact_id"]:
                blockers["executable"].append("Policy training does not bind the curated dataset version.")
            if executable["training_receipt_id"] is not None:
                receipt = evidence("executable", executable["training_receipt_id"])
                if receipt is not None and (receipt.get("policy_sha256") != artifacts[executable["implementation_artifact_id"]]["sha256"]
                    or dataset is None or receipt.get("dataset_sha256") != artifacts[dataset["dataset_artifact_id"]]["sha256"]
                    or receipt.get("split_sha256") != artifacts[dataset["split_artifact_id"]]["sha256"]):
                    blockers["executable"].append("Training receipt is stale for the current policy/dataset bytes.")
        elif executable["trained_dataset_artifact_id"] is not None or executable["training_receipt_id"] is not None:
            _bad("executable", "authored adapter must not claim a learned-policy training record")
        io = evidence("executable", executable["io_contract_id"])
        if io is not None and profile is not None:
            try:
                _io(io, profile, artifacts[executable["profile_artifact_id"]]["sha256"])
            except SkillReleaseError as error:
                blockers["executable"].append(str(error))
        satisfied["executable"] = satisfied["grounded_scenario"] and not blockers["executable"]

    evaluation = stages["evaluation"]
    if evaluation is not None:
        listed = refs("evaluation", evaluation, ("protocol_id", "receipt_id", "executable_artifact_id",
                      "scenario_artifact_id", "profile_artifact_id", "split_artifact_id", "baseline_artifact_id"))
        for field, role in (("protocol_id", "protocol"), ("receipt_id", "evaluation_receipt"),
                            ("executable_artifact_id", "executable"), ("scenario_artifact_id", "scenario"),
                            ("profile_artifact_id", "profile"), ("split_artifact_id", "split"), ("baseline_artifact_id", "baseline")):
            require("evaluation", evaluation[field], role, listed)
        if executable is None or evaluation["executable_artifact_id"] != executable["implementation_artifact_id"]:
            blockers["evaluation"].append("Evaluation does not identify the current frozen executable.")
        receipt = evidence("evaluation", evaluation["receipt_id"])
        if receipt is not None:
            expected = {field: artifacts[evaluation[field]]["sha256"] for field in ("protocol_id", "executable_artifact_id",
                        "scenario_artifact_id", "profile_artifact_id", "split_artifact_id", "baseline_artifact_id")}
            expected["runtime_lock_sha256s"] = sorted(artifacts[d["lock_artifact_id"]]["sha256"] for d in dependencies)
            if receipt.get("bindings") != expected:
                blockers["evaluation"].append("Evaluation receipt bindings are stale for frozen protocol/policy/scene/profile/split/baseline.")
            if type(receipt.get("evaluator_id")) is not str or not receipt["evaluator_id"] or receipt.get("evaluator_id") == receipt.get("builder_id"):
                blockers["evaluation"].append("Independent evaluator identity must be declared separately; identity is not authenticated offline.")
            platform = receipt.get("platform")
            if type(platform) is not dict or set(platform) != {"os", "architecture", "python", "simulator_version"} \
                    or any(type(v) is not str or not v.strip() for v in platform.values()):
                blockers["evaluation"].append("Evaluation must declare the exact runtime platform and simulator version.")
            split = evidence("evaluation", evaluation["split_artifact_id"])
            if split is not None:
                try:
                    train = set(_names(split.get("train_groups"), "split.train_groups"))
                    test = set(_names(split.get("evaluation_groups"), "split.evaluation_groups"))
                    if train & test:
                        blockers["evaluation"].append("Training and evaluation source/scenario groups overlap.")
                    if split.get("grouping_basis") not in ("source_identity", "scene_identity", "object_instance"):
                        blockers["evaluation"].append("Split grouping must protect source, scene or object identity.")
                except SkillReleaseError as error:
                    blockers["evaluation"].append(str(error))
            for key in ("trials", "baseline_trials"):
                rows = receipt.get(key)
                if type(rows) is not list or not 1 <= len(rows) <= 10000 or any(type(row) is not dict
                        or type(row.get("episode_id")) is not str or not _ID.fullmatch(row["episode_id"])
                        or type(row.get("success")) is not bool or type(row.get("valid")) is not bool
                        or row["valid"] is not True for row in rows):
                    blockers["evaluation"].append("Measured valid success trials and baseline trials are required; reward is insufficient.")
            trials, baselines = receipt.get("trials"), receipt.get("baseline_trials")
            if type(trials) is list and type(baselines) is list:
                ids = [row.get("episode_id") for row in trials if type(row) is dict]
                baseline_ids = [row.get("episode_id") for row in baselines if type(row) is dict]
                if any(type(identity) is not str for identity in ids + baseline_ids) or len(set(ids)) != len(ids) \
                        or ids != baseline_ids:
                    blockers["evaluation"].append("Baseline and policy must use the same distinct episode identities.")
        if scenario is None or any(evaluation[key] != scenario[key] for key in ("profile_artifact_id", "scenario_artifact_id")):
            blockers["evaluation"].append("Evaluation must reuse the qualified scenario and profile.")
        satisfied["evaluation"] = satisfied["executable"] and not blockers["evaluation"]

    hardware = stages["hardware_qualification"]
    if hardware is not None:
        listed = refs("hardware_qualification", hardware, ("profile_artifact_id", "calibration_artifact_id", "safety_artifact_id", "receipt_id"))
        for field, role in (("profile_artifact_id", "profile"), ("calibration_artifact_id", "calibration"),
                            ("safety_artifact_id", "safety"), ("receipt_id", "hardware_receipt")):
            require("hardware_qualification", hardware[field], role, listed)
        blockers["hardware_qualification"].append("Offline draft inspection cannot establish hardware calibration, trusted evaluation or physical safety; a separate supervised qualifier is required.")
    for stage in STAGES:
        if stages[stage] is None:
            blockers[stage].append("Stage evidence has not been supplied.")
        elif stage != "source" and not satisfied[stage] and not blockers[stage]:
            blockers[stage].append("A required preceding evidence stage is incomplete.")
    candidate = "none"
    for stage in STAGES:
        if satisfied[stage]:
            candidate = stage
    return {"kind": "dvidia.skill-release-proposal-report", "schema_version": 1,
            "manifest_sha256": manifest_sha, "manifest_hash_kind": "canonical_json" if type(document) is dict else "exact_bytes",
            "release_id": release["id"], "version": release["version"],
            "status": "draft", "candidate_stage": candidate, "stage_evidence_satisfied": satisfied,
            "blockers_by_stage": blockers, "verified_artifact_hashes": verified,
            "unavailable_artifacts": unavailable, "simulation_release_ready": False, "hardware_ready": False,
            "claim_ceiling": "hash_consistent_declared", "identities_authenticated": False,
            "lifecycle_exercised": False, "semantic_leakage_checked": False, "physical_accuracy_verified": False,
            "verification_scope": "Local byte hashes and declared-contract consistency only; no execution, authenticity, rights adjudication, data-quality assessment or physical qualification.",
            "release_gate": "Drafts require a separate trusted evaluator/publisher to become immutable evaluated releases."}


def main():
    parser = argparse.ArgumentParser(description="Inspect a proposed Skillspace release offline; never execute artifacts.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--artifact-root", type=Path, help="Optional local bundle root; declared safe paths only")
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    # Validate before resolving any paths supplied by a manifest.
    report = validate_skill_release(raw)
    if args.artifact_root:
        document = _json(raw)[0]
        root = args.artifact_root.resolve(strict=True)
        supplied = {}
        for artifact in document["artifacts"]:
            path = (root / artifact["path"]).resolve()
            if not path.is_relative_to(root):
                parser.error("An artifact resolves outside the supplied bundle root.")
            if path.is_file():
                if path.stat().st_size > MAX_ARTIFACT_BYTES:
                    parser.error("An artifact exceeds the inspector's bounded byte limit.")
                supplied[artifact["id"]] = path.read_bytes()
        report = validate_skill_release(raw, supplied)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
