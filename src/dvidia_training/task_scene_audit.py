"""Independent read-only consistency audit for task-environment v0.2 evidence.

Reconstructs corner containment, native-contact labels, world-frame speed gates,
continuous dwell and all five deliberately weakened verifiers from the saved
1 ms observation trace. No USD/native import, simulator step or production goal
predicate is called. This checks saved evidence consistency, not physical truth,
sensor correctness, authenticity or hardware readiness.
"""
from __future__ import annotations

import gzip
from hashlib import sha256
import io
import itertools
import json
import math
import re
from pathlib import Path
import struct

SCHEMA = "dvidia.task-scene-audit/v2"
BENCH_SCHEMA = "dvidia.task-scene-benchmark/v2"
MAX_FILE = 64*1024*1024
MAX_TOTAL = 128*1024*1024
NAMES = ("body_origin_only", "geometry_only", "without_speed", "without_dwell", "any_instance_only")


def _identity(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def _decode(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("Duplicate audit JSON field")
            result[k] = v
        return result
    def constant(_):
        raise ValueError("Nonfinite audit JSON token")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _read(root, name):
    parts = name.split("/")
    if any(x in ("", ".", "..") for x in parts) or name.startswith("/") or "\\" in name:
        raise ValueError("Unsafe evidence path")
    path = root.joinpath(*parts)
    if any(p.is_symlink() for p in (path, *path.parents) if p == root or root in p.parents):
        raise ValueError("Evidence links are unsupported")
    if not path.is_file() or path.stat().st_size > MAX_FILE:
        raise ValueError("Missing or oversized evidence file: "+name)
    raw = path.read_bytes()
    if len(raw) > MAX_FILE:
        raise ValueError("Evidence exceeds byte bound")
    return raw


def _j(root, name):
    return _decode(_read(root, name))


def _unpack(raw, identity):
    if type(identity.get("bytes")) is not int or not 0 <= identity["bytes"] <= MAX_FILE:
        raise ValueError("Unpacked evidence exceeds byte bound")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        result = stream.read(identity["bytes"]+1)
    if _identity(result) != {"bytes": identity["bytes"], "sha256": identity["sha256"]}:
        raise ValueError("Unpacked evidence identity differs")
    return result


def _same(a, b):
    if type(a) is bool or type(b) is bool or a is None or b is None:
        return a is b
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
    if type(a) is dict and type(b) is dict:
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if type(a) is list and type(b) is list:
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _vector(value, n):
    _require(type(value) is list and len(value) == n and all(type(x) in (float, int) and math.isfinite(x) for x in value), "Malformed finite observation vector")
    return value


def _pieces(obj):
    x, y, z = obj["size_m"]
    if obj["kind"] == "box":
        return [([x, y, z], [0, 0, z/2])]
    _require(obj["kind"] == "rigid_shoe_surrogate", "Unknown audited object class")
    return [([x, y, .18*z], [0, 0, .09*z]),
            ([.85*x, .85*y, .50*z], [0, 0, .43*z]),
            ([.60*x, .70*y, .32*z], [0, 0, .84*z])]


def _rot(q, p):
    # Quaternion-vector product, independent of production rotation-matrix code.
    w, x, y, z = q
    ux, uy, uz = x, y, z
    uv = ux*p[0]+uy*p[1]+uz*p[2]
    uu = ux*ux+uy*uy+uz*uz
    cross = [uy*p[2]-uz*p[1], uz*p[0]-ux*p[2], ux*p[1]-uy*p[0]]
    return [2*uv*u+(w*w-uu)*v+2*w*c for u, v, c in zip((ux, uy, uz), p, cross)]


def _corners(obj, pose, quat):
    result = []
    _vector(pose, 3); _vector(quat, 4)
    _require(math.isclose(sum(x*x for x in quat), 1, abs_tol=1e-6), "Observation quaternion not normalized")
    for size, local in _pieces(obj):
        for signs in itertools.product((-1, 1), repeat=3):
            rotated = _rot(quat, [local[i]+signs[i]*size[i]/2 for i in range(3)])
            result.append([pose[i]+rotated[i] for i in range(3)])
    return result


def _bin_local(p, bin_):
    x, y, z = [p[i]-bin_["center_m"][i] for i in range(3)]
    yaw = -bin_["yaw_rad"]
    return [math.cos(yaw)*x-math.sin(yaw)*y, math.sin(yaw)*x+math.cos(yaw)*y, z]


def _contained(obj, item, recipe):
    bin_, margin = recipe["bin"], recipe["goal"]["containment_margin_m"]
    ix, iy, height = bin_["inner_size_m"]
    floor = bin_["wall_m"]
    for corner in _corners(obj, item["position_m"], item["quaternion_wxyz"]):
        x, y, z = _bin_local(corner, bin_)
        if (abs(x) > ix/2-margin+1e-12 or abs(y) > iy/2-margin+1e-12
                or z < floor-.001-1e-12 or z > floor+height-margin+1e-12):
            return False
    return True


def _contacts(model):
    bodies = {b["id"]: b["name"] for b in model["bodies"]}
    _require(len(bodies) == len(model["bodies"]), "Duplicate model body identity")
    geoms = {g["name"]: g for g in model["geoms"]}
    _require(len(geoms) == len(model["geoms"]), "Duplicate model geometry identity")
    robot = {name for name, g in geoms.items() if g["collision"] and 1 <= g["body_id"] <= 9}
    pads = {name for name, g in geoms.items() if g["collision"] and g["type"] == "box" and bodies.get(g["body_id"]) in ("link7", "link8")}
    _require(len(pads) == 4, "Pinned native finger-pad geometry differs")
    return geoms, robot, pads


class Recompute:
    """Pure trace consumer, intentionally independent of the live GoalMonitor."""
    def __init__(self, recipe, model):
        self.recipe, self.previous, self.since = recipe, None, None
        self.geoms, self.robot, self.pads = _contacts(model)

    def update(self, state):
        now = state["time_s"]
        _require(type(state["valid"]) is bool and type(state["warnings"]) is list, "Invalid observation validity fields")
        _require(not state["valid"] or not state["warnings"], "Native warnings contradict valid observation")
        reason = None
        if now is None or type(now) not in (int, float) or not math.isfinite(now) or now < 0 or not state["valid"]:
            reason = "invalid_native_state"
        elif self.previous is not None and now < self.previous-1e-12:
            reason = "time_reset"
        elif self.previous is not None and now-self.previous > .0015:
            reason = "unobserved_interval"
        self.previous = now
        results = {}
        if reason is None:
            _require(set(state["objects"]) == {o["id"] for o in self.recipe["objects"]}, "Observed object membership differs")
            goal = self.recipe["goal"]
            for obj in self.recipe["objects"]:
                item = state["objects"][obj["id"]]
                names = item["contact_geom_names"]
                _require(type(names) is list and len(set(names)) == len(names) and all(name in self.geoms for name in names), "Unknown native contact partner")
                names = set(names)
                derived_contacts = {"table_contact": "task_table" in names, "bin_floor_contact": "task_bin_floor" in names,
                    "robot_contact": bool(names & self.robot), "pad_contact": bool(names & self.pads)}
                _require(all(item[k] is v for k, v in derived_contacts.items()), "Contact labels differ from saved contact partners")
                linear, angular = _vector(item["linear_velocity_m_s"], 3), _vector(item["angular_velocity_rad_s"], 3)
                result = {"contained": _contained(obj, item, self.recipe),
                    "bin_floor_contact": derived_contacts["bin_floor_contact"],
                    "released": not derived_contacts["robot_contact"] and not derived_contacts["pad_contact"],
                    "linear_speed": math.hypot(*linear) <= goal["max_linear_speed_m_s"],
                    "angular_speed": math.hypot(*angular) <= goal["max_angular_speed_rad_s"]}
                result["instantaneous"] = all(result.values())
                results[obj["id"]] = result
        instantaneous = reason is None and bool(results) and all(r["instantaneous"] for r in results.values())
        if not instantaneous:
            self.since = None
        elif self.since is None:
            self.since = now
        duration = 0. if self.since is None else now-self.since
        return {"time_s": now, "objects": results, "instantaneous": instantaneous, "settled_s": duration,
            "success": instantaneous and duration+1e-12 >= self.recipe["goal"]["settle_s"],
            "unknown": reason is not None, "reason": reason}


def _weak(state, goal, recipe):
    valid, items = state["valid"] and not goal["unknown"], list(goal["objects"].values())
    def point(item):
        x, y, z = _bin_local(item["position_m"], recipe["bin"])
        ix, iy, height = recipe["bin"]["inner_size_m"]
        floor = recipe["bin"]["wall_m"]
        return abs(x) <= ix/2+1e-9 and abs(y) <= iy/2+1e-9 and floor-.001 <= z <= floor+height+1e-9
    return {"body_origin_only": valid and bool(state["objects"]) and all(point(i) for i in state["objects"].values()),
        "geometry_only": valid and bool(items) and all(i["contained"] for i in items),
        "without_speed": valid and bool(items) and all(i["contained"] and i["bin_floor_contact"] and i["released"] for i in items),
        "without_dwell": valid and goal["instantaneous"],
        "any_instance_only": valid and any(i["instantaneous"] for i in items)}


def _audit_case(root, case, entry):
    base = "cases/"+case["id"]+"/"
    recipe = _j(root, base+"input/recipe.json")
    result = _j(root, base+"result.json")
    receipt = _j(root, base+"native-receipt.json")
    model = _j(root, base+"model.json")
    interventions = _j(root, base+"interventions.json")
    geometry_meta = model["geometry_binary"]
    href = geometry_meta["href"]
    if href == "geometry.bin.gz":
        geometry_path = base+href
    elif re.fullmatch(r"\.\./\.\./geometry/[a-f0-9]{64}\.bin\.gz", href):
        geometry_path = href[len("../../"):]
        _require(geometry_meta["sha256"] == Path(geometry_path).name[:-7], "Geometry-addressed file name differs")
    else:
        raise ValueError("Unexpected bounded geometry resource path")
    _unpack(_read(root, geometry_path), geometry_meta)
    _require(result["case"] == case and result["registered_benchmark"] is True, "Case receipt differs from fixed protocol")
    _require(result["timestep_s"] == .001 and result["record_dt_s"] == .01, "Declared native/replay clock differs")
    _require(receipt["source_sha256"] == recipe["source_sha256"], "Recipe/source receipt differs")
    for name, want in recipe["source_sha256"].items():
        _require(_identity(_read(root, "source/"+name))["sha256"] == want, "Recipe source snapshot differs")

    _require(receipt["recipe"] == recipe and receipt["compiled_checks"] and all(receipt["compiled_checks"].values()), "Native receipt recipe or compiled checks differ")
    exported = {g["name"]: g for g in model["geoms"]}
    for item in (receipt["usd"]["table"], *receipt["usd"]["bin"]):
        name = "task_table" if item is receipt["usd"]["table"] else "task_bin_"+item["name"]
        geom = exported[name]
        _require(geom["type"] == "box" and geom["body_id"] == 0 and geom["collision"] is True,
                 "Exported static collision class differs")
        _require(_same(geom["size"], [v/2 for v in item["size_m"]]) and _same(geom["position"], item["center_m"])
                 and _same(geom["quaternion_wxyz"], item["quaternion_wxyz"]), "Exported bin/table collision geometry differs")
    body_by_name = {b["name"]: b["id"] for b in model["bodies"]}
    for identifier, obj in receipt["usd"]["objects"].items():
        for piece in obj["pieces"]:
            geom = exported["task_object_"+identifier+"_"+piece["name"]]
            _require(geom["type"] == "box" and geom["body_id"] == body_by_name["task_object_"+identifier]
                     and geom["collision"] is True and _same(geom["size"], [v/2 for v in piece["size_m"]])
                     and _same(geom["position"], piece["position_m"]), "Exported object collision geometry differs")
    authored = _j(root, base+"input/scene-receipt.json")
    inputs = {n: _identity(_read(root, base+"input/"+n)) for n in ("recipe.json", "workstation.usda")}
    _require(receipt["files"] == inputs == authored["files"], "Authored recipe/USD identity differs")
    _require(interventions == result["interventions"], "Intervention receipt differs")
    expected_gravity = [0., 0., 0.] if case["control"] == "gravity_disabled" else receipt["usd"]["gravity_m_s2"]
    _require(interventions["effective_gravity_m_s2"] == expected_gravity and interventions["authored_gravity_m_s2"] == receipt["usd"]["gravity_m_s2"], "Undeclared gravity intervention")
    _require(interventions["external_forces_during_stepping"] is False and interventions["state_overwrite_during_stepping"] is False, "Undeclared per-step state or force intervention")
    nobj = len(recipe["objects"])
    _require(len(interventions["initial_qpos"]) == 8+7*nobj and len(interventions["initial_qvel"]) == 8+6*nobj, "Initial native dimensions differ")
    expected_vel = [0.]*(8+6*nobj)
    if case["control"] == "initial_velocity":
        expected_vel[8] = .3
    _require(interventions["initial_qvel"] == expected_vel, "Undeclared initial velocity")
    _require(interventions["control"] == case["control"], "Intervention mode differs")
    trace_meta = result["per_step_trace"]
    _require(trace_meta["file"] == "goal-trace.jsonl.gz" and trace_meta["compression"] == "gzip", "Trace resource path differs")
    trace = _unpack(_read(root, base+"goal-trace.jsonl.gz"), trace_meta)
    lines = trace.splitlines()
    steps = round(case["duration_s"]/.001)
    _require(len(lines) == steps+1 == trace_meta["samples"], "Per-step trace count differs")
    poses = _unpack(_read(root, base+"poses.bin.gz"), result["pose_identity"])
    body_count, frame_count = result["body_count"], result["frame_count"]
    _require(body_count == 10+nobj == len(model["bodies"]) and len(poses) == frame_count*body_count*7*4, "Pose framing differs")
    _require([b["id"] for b in model["bodies"]] == list(range(body_count)), "Body ordering differs")
    body_ids = {b["name"]: b["id"] for b in model["bodies"]}
    values = struct.unpack("<"+str(len(poses)//4)+"f", poses)
    _require(all(math.isfinite(x) for x in values), "Pose payload contains nonfinite values")
    trajectory = _j(root, base+"trajectory.json")["rows"]
    _require(trajectory == entry["rows"] and len(trajectory) == frame_count, "Replay rows differ")
    _require(entry["scene"] == base+"model.json" and entry["poses"] == base+"poses.bin.gz", "Replay path differs")
    _require({"bytes": entry["bytes"], "sha256": entry["sha256"]} == result["pose_identity"], "Replay pose identity differs")
    verifier, ever, terminal = Recompute(recipe, model), {n: False for n in NAMES}, {n: False for n in NAMES}
    since = {n: None for n in ("without_speed", "any_instance_only")}
    first_success, frame, max_pose_error = None, 0, 0.
    for tick, raw in enumerate(lines):
        recorded = _decode(raw)
        _require(set(recorded) == {"state", "goal"}, "Trace row fields differ")
        state = recorded["state"]
        _require(math.isclose(state["time_s"], tick*.001, rel_tol=0, abs_tol=1e-8), "Native trace interval/time differs")
        derived = verifier.update(state)
        _require(_same(derived, recorded["goal"]), "Independent full goal reconstruction differs at "+case["id"]+":"+str(tick))
        predicates = _weak(state, derived, recipe)
        for name in NAMES:
            predicate = predicates[name]
            if name in since:
                if not predicate:
                    since[name] = None
                elif since[name] is None:
                    since[name] = state["time_s"]
                predicate = since[name] is not None and state["time_s"]-since[name]+1e-12 >= recipe["goal"]["settle_s"]
            terminal[name] = predicate
            ever[name] = ever[name] or predicate
        if derived["success"] and first_success is None:
            first_success = state["time_s"]
        if tick % 10 == 0 or tick == steps:
            row = {"t_s": state["time_s"], "all_complete": derived["success"],
                "objects": {k: {**v, **derived["objects"].get(k, {})} for k, v in state["objects"].items()},
                "settled_s": derived["settled_s"], "unknown": derived["unknown"]}
            _require(_same(row, trajectory[frame]), "100Hz replay row differs from 1ms trace")
            for obj in recipe["objects"]:
                offset = (frame*body_count+body_ids["task_object_"+obj["id"]])*7
                actual = values[offset:offset+7]
                expected = state["objects"][obj["id"]]["position_m"]+state["objects"][obj["id"]]["quaternion_wxyz"]
                error = max(abs(a-b) for a, b in zip(actual, expected))
                max_pose_error = max(max_pose_error, error)
                _require(error <= 1e-6, "Float32 replay pose differs from native trace")
            frame += 1
    _require(frame == frame_count, "Replay frame count differs")
    _require(_same(derived, result["native_goal"]) and derived["success"] is result["native_goal_success"], "Terminal goal receipt differs")
    _require(_same(first_success, result["first_goal_success_s"]), "First success time differs")
    _require(ever == result["ablations_ever_accept"] and terminal == result["ablations_at_end"], "Ablation counters differ")
    _require(result["checks_passed"] is all(result["checks"].values()), "Result check aggregate differs")
    _require(result["native_goal_success"] is case["expected_goal_success"], "Declared positive/negative outcome differs")
    _require(all(result[k] is False for k in ("physics_calibrated", "learned_skill", "physical_robot_ready")), "Unsupported qualification claim")
    return {"id": case["id"], "trace_samples": len(lines), "replay_frames": frame_count,
        "goal_success": derived["success"], "ablations_ever_accept": ever,
        "max_replay_pose_roundoff": max_pose_error, "mismatches": 0}


def _precheck_fault(recipe):
    """Independent bounded geometry counterchecks for the four registered inputs."""
    objects = recipe["objects"]
    if len({o["id"] for o in objects}) != len(objects):
        return "duplicate"
    margin = recipe["goal"]["containment_margin_m"]
    ix, iy, iz = recipe["bin"]["inner_size_m"]
    for obj in objects:
        sx, sy, sz = obj["size_m"]
        if not ((sx <= ix-2*margin and sy <= iy-2*margin) or (sy <= ix-2*margin and sx <= iy-2*margin)) or sz > iz-margin:
            return "fit"
    table = recipe["table"]
    pieces = []
    for obj in objects:
        q = [math.cos(obj["yaw_rad"]/2), 0, 0, math.sin(obj["yaw_rad"]/2)]
        points = _corners(obj, obj["position_m"], q)
        if any(abs(p[i]-table["center_m"][i]) > table["size_m"][i]/2+1e-9 for p in points for i in (0, 1)):
            return "support"
        current = [points[i:i+8] for i in range(0, len(points), 8)]
        for a, yaw_a in pieces:
            for b in current:
                if min(p[2] for p in a) >= max(p[2] for p in b)-1e-9 or min(p[2] for p in b) >= max(p[2] for p in a)-1e-9:
                    continue
                axes = [(math.cos(t), math.sin(t)) for t in (yaw_a, yaw_a+math.pi/2, obj["yaw_rad"], obj["yaw_rad"]+math.pi/2)]
                overlap = True
                for ux, uy in axes:
                    pa = [ux*p[0]+uy*p[1] for p in a]
                    pb = [ux*p[0]+uy*p[1] for p in b]
                    if max(pa) <= min(pb)+1e-9 or max(pb) <= min(pa)+1e-9:
                        overlap = False
                        break
                if overlap:
                    return "overlap"
        pieces.extend((points_, obj["yaw_rad"]) for points_ in current)
    return None


def audit(study):
    """Return verified counters; any differing saved identity/predicate raises."""
    root = Path(study)
    _require(root.is_dir() and not root.is_symlink(), "Use a regular evidence directory")
    paths = list(root.rglob("*"))
    _require(not any(p.is_symlink() for p in paths), "Evidence symlink refused")
    files = [p for p in paths if p.is_file()]
    _require(len(files) <= 512 and sum(p.stat().st_size for p in files) <= MAX_TOTAL, "Evidence package exceeds audit bounds")
    protocol, seal = _j(root, "protocol.json"), _j(root, "preparation-seal.json")
    inventory = _j(root, "input-inventory.json")
    _require(protocol["schema"] == BENCH_SCHEMA and len(protocol["cases"]) == 16, "Unsupported registered protocol")
    _require(seal == {"schema": BENCH_SCHEMA, "protocol": _identity(_read(root, "protocol.json")), "inputs": _identity(_read(root, "input-inventory.json"))}, "Preparation seal differs")
    _require(len({c["id"] for c in protocol["cases"]}) == 16, "Case identity differs")
    _require({p.name for p in (root/"cases").iterdir()} == {c["id"] for c in protocol["cases"]}, "Case inventory differs")
    actual_inputs = {str(p.relative_to(root)): _identity(p.read_bytes()) for p in sorted((root/"cases").glob("*/input/*")) if p.is_file()}
    _require(inventory == {"schema": BENCH_SCHEMA, "files": actual_inputs}, "Frozen input inventory differs")
    for name, identity in protocol["source"].items():
        _require(_identity(_read(root, "source/"+name)) == identity, "Frozen source snapshot differs")
    index, report = _j(root, "index.json"), _j(root, "report.json")
    completion = _j(root, "completion.json")
    _require(completion["report"] == _identity(_read(root, "report.json")) and completion["index"] == _identity(_read(root, "index.json")), "Completion seal differs")
    _require(report["protocol_sha256"] == seal["protocol"]["sha256"], "Report protocol differs")
    entries = {e["id"]: e for e in index["cases"]}
    results, prechecks = [], []
    for case in protocol["cases"]:
        base = "cases/"+case["id"]+"/"
        started = _j(root, base+"started.json")
        _require(started["case"] == case and started["source"] == protocol["source"], "Started intent differs")
        if case["control"] == "precheck":
            result = _j(root, base+"result.json")
            expected_fault = {"invalid-overlap": "overlap", "invalid-support": "support", "invalid-fit": "fit", "invalid-duplicate-id": "duplicate"}.get(case["name"])
            _require(expected_fault is not None and _precheck_fault(_j(root, base+"input/recipe.json")) == expected_fault, "Independent invalid-layout reconstruction differs")
            _require(result["rejected"] is True and result["native_run"] is False and result["expected_precheck_rejection"] is True and bool(result["error"]), "Invalid layout was not explicitly refused")
            _require(not (root/base/"goal-trace.jsonl.gz").exists(), "Precheck unexpectedly contains a native trace")
            prechecks.append(result)
        else:
            _require(case["id"] in entries, "Missing replay case")
            results.append(_audit_case(root, case, entries[case["id"]]))
    _require(len(entries) == len(results) == 12 and len(prechecks) == 4, "Native/precheck case counts differ")
    stored_results = [_j(root, "cases/"+c["id"]+"/result.json") for c in protocol["cases"] if c["control"] != "precheck"]
    _require(report["results"] == stored_results and report["prechecks"] == index["prechecks"] == prechecks, "Report case roster differs")
    counts = {"native_scheduled": 12, "native_valid": sum(r["valid"] for r in stored_results),
        "native_checks_passed": sum(r["checks_passed"] for r in stored_results),
        "native_goal_successes": sum(r["goal_success"] for r in results),
        "positive_scheduled": sum(c["expected_goal_success"] is True for c in protocol["cases"]),
        "positive_goal_successes": sum(r["native_goal_success"] for r in stored_results if r["expected_goal_success"] is True),
        "negative_scheduled": sum(c["expected_goal_success"] is False for c in protocol["cases"]),
        "negative_goals_correctly_rejected": sum(not r["native_goal_success"] for r in stored_results if r["expected_goal_success"] is False),
        "prechecks_scheduled": 4, "prechecks_correctly_rejected": 4}
    _require(all(report[k] == v for k, v in counts.items()), "Aggregate benchmark counters differ")
    return {"schema": SCHEMA, "protocol_sha256": seal["protocol"]["sha256"], "counts": counts,
        "trace_samples": sum(r["trace_samples"] for r in results), "replay_frames": sum(r["replay_frames"] for r in results),
        "mismatches": 0, "cases": results,
        "scope": "Read-only independent consistency reconstruction from saved native observations; no new simulation, policy, physical truth, sensor accuracy or hardware qualification.",
        "auditor_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
