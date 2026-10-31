"""Registered passive task-scene checks and bounded, unregistered local previews.

PiPER holds its native home controls. Objects are authored/dropped or explicitly
intervened on by the declared negative-control protocol; no policy is learned,
no image is used as a sensor and no manipulation success is inferred.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import time

from . import asset_preview, reference_sim as sim, task_scene as bridge

SCHEMA = "dvidia.task-scene-benchmark/v2"
REPLAY_SCHEMA = "dvidia.task-scene-replay/v2"
DT = .001
RECORD_DT = .01
MAX_JSON = 16 * 1024 * 1024
SOURCE_FILES = ("task_scene.py", "task_scene_benchmark.py", "asset_library.py", "asset_catalog.json",
    "robot_profiles.py", "robot_profiles.lock.json", "reference_sim.py", "reference_assets.lock.json",
    "asset_preview.py", "reference_replay.py")
SCOPE = ("Passive, original rigid task-scene and independent completion checks. The native PiPER holds home; "
         "no learned grasp, image-based perception, shoe deformation or physical-robot transfer is evaluated.")


def encoded(value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_JSON:
        raise ValueError("JSON output exceeds the bounded evidence size")
    return raw


def identity(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = encoded(value) if isinstance(value, (dict, list)) else value
    with path.open("xb") as stream:
        stream.write(raw)


def _unique_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def _nonfinite_constant(value):
    raise ValueError("Nonfinite JSON constant")


def _store_geometry(root, raw):
    """One immutable native mesh blob per content identity, shared by cases."""
    root = Path(root)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("Shared geometry root must be a regular local directory")
    root.mkdir(parents=True, exist_ok=True)
    digest = sha256(raw).hexdigest()
    path = root/(digest+".bin.gz")
    compressed = gzip.compress(raw, mtime=0)
    if path.exists() or path.is_symlink():
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != len(compressed)
                or path.read_bytes() != compressed):
            raise ValueError("Existing content-addressed geometry differs; evidence is never replaced")
    else:
        write(path, compressed)
    return {"href": "../../geometry/"+path.name, "compression": "gzip", **identity(raw)}


def _read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON:
        raise ValueError("Expected a bounded local JSON file")
    return json.loads(path.read_bytes(), object_pairs_hook=_unique_fields, parse_constant=_nonfinite_constant)


def sources():
    return {name: identity(Path(__file__).with_name(name).read_bytes()) for name in SOURCE_FILES}


def roster():
    """A fixed roster: six positive, six negative native, four layout prechecks."""
    definitions = [
        ("block-drop", "One block released over the open bin", True, 1.5, "normal"),
        ("shoe-drop", "Original rigid shoe surrogate released over the bin", True, 1.5, "normal"),
        ("two-blocks", "Two separate block instances", True, 1.5, "normal"),
        ("block-and-shoe", "Block and rigid shoe surrogate", True, 1.5, "normal"),
        ("rotated-bin", "Rotated bin frame and two objects", True, 1.5, "normal"),
        ("four-instances", "Four distinct block instances", True, 1.5, "normal"),
        ("outside-bin", "Stationary block on the table outside the bin", False, 1.5, "normal"),
        ("rim-overhang", "Body origin inside but collision volume overhangs the rim", False, .5, "gravity_disabled"),
        ("unsupported", "Contained and stationary without floor support", False, .5, "gravity_disabled"),
        ("moving", "Floor-supported object sliding too fast", False, .15, "initial_velocity"),
        ("short-dwell", "Supported release observed for too little time", False, .10, "normal"),
        ("one-required-outside", "One of two required instances remains outside", False, 1.5, "normal"),
        ("invalid-overlap", "Overlapping starting instances rejected", None, 0., "precheck"),
        ("invalid-support", "Table overhang rejected", None, 0., "precheck"),
        ("invalid-fit", "Object too large for the target bin rejected", None, 0., "precheck"),
        ("invalid-duplicate-id", "Duplicate instance identity rejected", None, 0., "precheck"),
    ]
    return [{"id": f"case-{i:02d}", "name": name, "label": label,
             "expected_goal_success": expected, "duration_s": duration, "control": control}
            for i, (name, label, expected, duration, control) in enumerate(definitions)]


def _world(recipe, x, y, z):
    bin_ = recipe["bin"]
    c, s = math.cos(bin_["yaw_rad"]), math.sin(bin_["yaw_rad"])
    return [bin_["center_m"][0] + c*x - s*y,
            bin_["center_m"][1] + s*x + c*y, bin_["center_m"][2] + z]


def recipe_for_case(case):
    recipe = bridge.default_recipe()
    block, shoe = deepcopy(recipe["objects"])
    def placed(obj, identifier, x=0., y=0., z=.08):
        obj = deepcopy(obj)
        obj.update(id=identifier, position_m=_world(recipe, x, y, z), yaw_rad=recipe["bin"]["yaw_rad"])
        return obj
    name = case["name"]
    if name == "rotated-bin":
        recipe["bin"]["yaw_rad"] = .4
    if name == "shoe-drop":
        objects = [placed(shoe, "shoe")]
    elif name in ("block-and-shoe", "rotated-bin"):
        objects = [placed(block, "block", 0, -.075), placed(shoe, "shoe", 0, .035)]
    elif name == "two-blocks":
        objects = [placed(block, "left", -.06), placed(block, "right", .06)]
    elif name == "four-instances":
        objects = [placed(block, "block_"+str(i), x, y) for i, (x, y) in enumerate(((-.07, -.05), (-.07, .05), (.07, -.05), (.07, .05)))]
    elif name in ("outside-bin", "one-required-outside"):
        outside = deepcopy(block)
        outside.update(id="outside", position_m=[.35, -.2, 0.])
        objects = [outside] if name == "outside-bin" else [placed(block, "inside"), outside]
    elif name == "rim-overhang":
        # Bottom-origin point is inside the bin. Full collision corners extend
        # over its side and above its top; it touches the rim with no penetration.
        objects = [placed(block, "block", recipe["bin"]["inner_size_m"][0]/2-.005,
                          0., recipe["bin"]["wall_m"]+recipe["bin"]["inner_size_m"][2])]
    elif name == "unsupported":
        objects = [placed(block, "block", z=.06)]
    elif name in ("moving", "short-dwell"):
        objects = [placed(block, "block", -.04, 0., recipe["bin"]["wall_m"])]
        if name == "moving":
            recipe["bin"]["sliding_friction"] = 0.
            objects[0]["sliding_friction"] = 0.
            recipe["goal"]["settle_s"] = .05
    else:
        objects = [placed(block, "block")]
    recipe["objects"] = objects
    if name == "invalid-overlap":
        recipe["objects"].append(placed(block, "overlap"))
    elif name == "invalid-support":
        recipe["objects"][0]["position_m"] = [1.1, 0., .05]
    elif name == "invalid-fit":
        recipe["objects"][0]["size_m"] = [.4, .4, .04]
    elif name == "invalid-duplicate-id":
        recipe["objects"] = [placed(block, "block", -.06), placed(block, "block", .06)]
    return recipe


def _input_inventory(root):
    return {str(path.relative_to(root)): identity(path.read_bytes())
            for path in sorted((root/"cases").glob("*/input/*")) if path.is_file()}


def prepare(output):
    """Freeze source, fixed protocol and exact USD inputs before native work."""
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise ValueError("Use a fresh benchmark directory; attempts are never overwritten")
    protocol = {"schema": SCHEMA, "cases": roster(), "source": sources(),
        "timestep_s": DT, "record_dt_s": RECORD_DT, "worker_count": 1, "scope": SCOPE,
        "goal_semantics": "All required collision volumes inside bin, floor-supported, released, below velocity limits continuously for recipe dwell.",
        "ablations": {
            "body_origin_only": "Naive single-point containment in the bin frame, no corner/margin/contact/speed/dwell checks. The native body origin is the bottom centre, not an estimated semantic object centre.",
            "geometry_only": "All actual collision corners contained; support, speed, release and dwell omitted.",
            "without_speed": "Full native predicates and continuous dwell, omitting linear/angular speed checks.",
            "without_dwell": "Full instantaneous native predicates; continuous dwell omitted.",
            "any_instance_only": "At least one instance meets the full instantaneous predicates; the all-required-instance check omitted."},
        "negative_controls": "Gravity disabled only for declared rim-overhang and unsupported stationary counterexamples. Moving case has a declared initial 0.3m/s linear velocity and zero authored contact friction. No pose, velocity or external-force overwrite during stepping.",
        "timing_scope": "One CPU worker. Per-case elapsed includes compile, validation, all recorded native steps, geometry, compression and case output writes. run() total includes execution and final output writes; run_benchmark() additionally records preparation plus execution in full-wall-timing.json. No camera, training, network or physical actuators.",
        "physics_calibrated": False, "learned_skill": False, "physical_robot_ready": False}
    # Resolve all validation outcomes before destination creation. Unexpected
    # authoring errors retain their preparation directory and cannot be reused.
    recipes = [recipe_for_case(case) for case in protocol["cases"]]
    root.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        write(root/"source"/name, Path(__file__).with_name(name).read_bytes())
    write(root/"protocol.json", protocol)
    for case, recipe in zip(protocol["cases"], recipes):
        folder = root/"cases"/case["id"]/"input"
        if case["control"] == "precheck":
            write(folder/"recipe.json", recipe)
        else:
            bridge.write_scene(recipe, folder)
    write(root/"input-inventory.json", {"schema": SCHEMA, "files": _input_inventory(root)})
    write(root/"preparation-seal.json", {"schema": SCHEMA,
        "protocol": identity((root/"protocol.json").read_bytes()),
        "inputs": identity((root/"input-inventory.json").read_bytes())})
    return protocol


def _verify_prepared(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Prepared benchmark must be a regular local directory")
    protocol = _read_json(root/"protocol.json")
    inventory = _read_json(root/"input-inventory.json")
    seal = _read_json(root/"preparation-seal.json")
    expected_seal = {"schema": SCHEMA, "protocol": identity((root/"protocol.json").read_bytes()),
        "inputs": identity((root/"input-inventory.json").read_bytes())}
    if seal != expected_seal or protocol.get("schema") != SCHEMA:
        raise ValueError("Frozen protocol or input inventory changed")
    if protocol["source"] != sources() or protocol["cases"] != roster():
        raise ValueError("Frozen source or case roster changed")
    snapshots = root/"source"
    if ({p.name for p in snapshots.iterdir()} != set(SOURCE_FILES)
            or any(p.is_symlink() for p in root.rglob("*"))
            or any(identity((snapshots/name).read_bytes()) != want for name, want in protocol["source"].items())):
        raise ValueError("Frozen source snapshots differ or symlinks are present")
    if {p.name for p in (root/"cases").iterdir()} != {c["id"] for c in protocol["cases"]}:
        raise ValueError("Frozen case membership changed")
    if inventory != {"schema": SCHEMA, "files": _input_inventory(root)}:
        raise ValueError("Frozen input bytes or membership changed")
    expected_names = {"recipe.json", "workstation.usda", "scene-receipt.json"}
    for case in protocol["cases"]:
        folder = root/"cases"/case["id"]
        names = {"recipe.json"} if case["control"] == "precheck" else expected_names
        if ({p.name for p in folder.iterdir()} != {"input"}
                or {p.name for p in (folder/"input").iterdir()} != names):
            raise ValueError("An attempt started already or input membership changed; automatic rerun refused")
    if (root/"execution-intent.json").exists():
        raise ValueError("A benchmark execution was already intended; automatic rerun refused")
    return protocol, seal, inventory


class _Dwell:
    def __init__(self, required):
        self.required, self.since, self.success = required, None, False

    def update(self, now, predicate):
        if not predicate:
            self.since = None
        elif self.since is None:
            self.since = now
        self.success = self.since is not None and now-self.since+1e-12 >= self.required
        return self.success


def _origin_inside(position, bin_):
    c, s = math.cos(bin_["yaw_rad"]), math.sin(bin_["yaw_rad"])
    x, y, z = [position[i]-bin_["center_m"][i] for i in range(3)]
    x, y = c*x+s*y, -s*x+c*y
    inner = bin_["inner_size_m"]
    return (abs(x) <= inner[0]/2+1e-9 and abs(y) <= inner[1]/2+1e-9
            and bin_["wall_m"]-.001 <= z <= bin_["wall_m"]+inner[2]+1e-9)


def ablation_predicates(state, goal, recipe):
    """Explicit deliberately weakened verifiers, scored against native truth."""
    valid = state["valid"] and not goal["unknown"]
    items = list(goal["objects"].values())
    return {
        "body_origin_only": valid and bool(state["objects"]) and all(_origin_inside(item["position_m"], recipe["bin"]) for item in state["objects"].values()),
        "geometry_only": valid and bool(items) and all(item["contained"] for item in items),
        "without_speed": valid and bool(items) and all(item["contained"] and item["bin_floor_contact"] and item["released"] for item in items),
        "without_dwell": valid and goal["instantaneous"],
        "any_instance_only": valid and any(item["instantaneous"] for item in items)}


def _run_native(recipe_path, assets, folder, case, registered, *, geometry_root):
    began = time.perf_counter()
    model, receipt = bridge.build_scene(recipe_path, assets)
    data = bridge.initialize(model, receipt)
    mj, np = sim._native_imports()
    if case["control"] == "gravity_disabled":
        model.opt.gravity[:] = 0
    elif case["control"] == "initial_velocity":
        joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "task_object_block_free")
        data.qvel[int(model.jnt_dofadr[joint])] = .3
    mj.mj_forward(model, data)
    interventions = {"control": case["control"], "authored_gravity_m_s2": receipt["usd"]["gravity_m_s2"],
        "effective_gravity_m_s2": model.opt.gravity.tolist(),
        "initial_qpos": data.qpos.tolist(), "initial_qvel": data.qvel.tolist(),
        "external_forces_during_stepping": False, "state_overwrite_during_stepping": False,
        "scope": "Declared verifier counterexample fixture; no learned robot control"}
    write(folder/"interventions.json", interventions)
    monitor = bridge.GoalMonitor(receipt)
    recipe = receipt["recipe"]
    dwell = {name: _Dwell(recipe["goal"]["settle_s"]) for name in ("without_speed", "any_instance_only")}
    ever = {name: False for name in ("body_origin_only", "geometry_only", "without_speed", "without_dwell", "any_instance_only")}
    terminal = dict(ever)
    first_success = None
    warnings, rows, frames, trace = [], [], [], []
    steps = round(case["duration_s"]/DT)
    if abs(case["duration_s"]-steps*DT) > 1e-10 or not 1 <= steps <= 10000:
        raise ValueError("Native preview duration must be 1ms-aligned and within10s")
    with sim.native_warnings(warnings):
        for tick in range(steps+1):
            goal = monitor.update(model, data)
            state = bridge.sample_state(model, data, receipt)
            trace.append((json.dumps({"state": state, "goal": goal}, allow_nan=False, separators=(",", ":"))+"\n").encode())
            predicates = ablation_predicates(state, goal, recipe)
            for name, predicate in predicates.items():
                terminal[name] = dwell[name].update(data.time, predicate) if name in dwell else predicate
                ever[name] = ever[name] or terminal[name]
            if goal["success"] and first_success is None:
                first_success = float(data.time)
            if tick % 10 == 0 or tick == steps:
                objects = {identifier: {**item, **goal["objects"].get(identifier, {})} for identifier, item in state["objects"].items()}
                rows.append({"t_s": float(data.time), "all_complete": goal["success"], "objects": objects,
                             "settled_s": goal["settled_s"], "unknown": goal["unknown"]})
                frames.append(np.concatenate((data.xpos, data.xquat), axis=1).astype("<f4"))
            if tick < steps:
                mj.mj_step(model, data)
                mj.mj_forward(model, data)
    finite = all(np.isfinite(frame).all() for frame in frames)
    no_warning = not warnings and not any(int(w.number) for w in data.warning)
    expected = case["expected_goal_success"]
    checks = {"complete_native_duration": abs(float(data.time)-case["duration_s"]) < 1e-8,
        "finite_body_poses": finite, "no_native_warning": no_warning,
        "goal_known": not goal["unknown"],
        "matches_declared_goal": expected is None or bool(goal["success"]) == expected}
    if registered:
        name = case["name"]
        if name == "rim-overhang":
            checks["center_only_false_completion_exposed"] = ever["body_origin_only"] and not bool(goal["success"])
        elif name == "unsupported":
            checks["geometry_only_false_completion_exposed"] = ever["geometry_only"] and not any(item["bin_floor_contact"] for item in goal["objects"].values())
        elif name == "moving":
            checks["speed_omission_false_completion_exposed"] = ever["without_speed"] and not bool(goal["success"])
        elif name == "short-dwell":
            checks["dwell_omission_false_completion_exposed"] = ever["without_dwell"] and not bool(goal["success"])
        elif name == "one-required-outside":
            checks["single_instance_false_completion_exposed"] = ever["any_instance_only"] and not bool(goal["success"])
    scene, geometry = asset_preview._scene(model, {"scope": SCOPE, "recipe_sha256": sha256(Path(recipe_path).read_bytes()).hexdigest()})
    scene["geometry_binary"] = _store_geometry(geometry_root, geometry)
    raw_poses = np.asarray(frames, dtype="<f4").tobytes()
    write(folder/"native-receipt.json", receipt)
    write(folder/"trajectory.json", {"schema": SCHEMA, "rows": rows})
    write(folder/"model.json", scene)
    write(folder/"poses.bin.gz", gzip.compress(raw_poses, mtime=0))
    raw_trace = b"".join(trace)
    if len(raw_trace) > 64*1024*1024:
        raise ValueError("Native per-step trace exceeded64MBbound")
    write(folder/"goal-trace.jsonl.gz", gzip.compress(raw_trace, mtime=0))
    result = {"schema": SCHEMA, "case": case, "registered_benchmark": registered,
        "checks": checks, "checks_passed": all(checks.values()), "native_goal_success": bool(goal["success"]),
        "expected_goal_success": expected, "first_goal_success_s": first_success, "native_goal": goal,
        "native_warnings": warnings, "valid": all(checks[key] for key in ("complete_native_duration", "finite_body_poses", "no_native_warning", "goal_known")),
        "interventions": interventions,
        "ablations_ever_accept": ever, "ablations_at_end": terminal,
        "simulated_seconds": float(data.time), "timestep_s": DT, "record_dt_s": RECORD_DT,
        "frame_count": len(frames), "body_count": int(model.nbody), "pose_identity": identity(raw_poses),
        "native_model_sha256": sim.model_digest(model), "scope": SCOPE,
        "per_step_trace": {"file": "goal-trace.jsonl.gz", "compression": "gzip", "samples": len(trace), **identity(raw_trace)},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "mujoco": mj.__version__},
        "physics_calibrated": False, "learned_skill": False, "physical_robot_ready": False}
    # Write substantial output before measuring. The immutable result receipt's
    # own serialization is separately included in the enclosing total timing.
    result["wall_seconds_through_case_output"] = time.perf_counter()-began
    result["simulated_seconds_per_wall_second"] = result["simulated_seconds"]/result["wall_seconds_through_case_output"]
    write(folder/"result.json", result)
    entry = {"id": case["id"], "label": case["label"], "scene": str(folder.name)+"/model.json",
        "poses": str(folder.name)+"/poses.bin.gz", "compression": "gzip", **identity(raw_poses),
        "frame_count": len(frames), "body_count": int(model.nbody), "dt_s": RECORD_DT,
        "duration_s": float(data.time), "metrics": {"native_goal_success": result["native_goal_success"],
            "expected_goal_success": expected, "checks_passed": result["checks_passed"],
            "wall_seconds": result["wall_seconds_through_case_output"], "simulated_seconds": float(data.time),
            "simulated_seconds_per_wall_second": result["simulated_seconds_per_wall_second"],
            "ablations_ever_accept": ever}, "rows": rows}
    return result, entry


def _summary(results, prechecks, registered):
    return {"schema": SCHEMA, "registered_benchmark": registered, "scope": SCOPE,
        "native_scheduled": len(results), "native_valid": sum(r["valid"] for r in results),
        "native_checks_passed": sum(r["checks_passed"] for r in results),
        "native_goal_successes": sum(r["native_goal_success"] for r in results),
        "positive_scheduled": sum(r["expected_goal_success"] is True for r in results),
        "positive_goal_successes": sum(r["native_goal_success"] for r in results if r["expected_goal_success"] is True),
        "negative_scheduled": sum(r["expected_goal_success"] is False for r in results),
        "negative_goals_correctly_rejected": sum(not r["native_goal_success"] for r in results if r["expected_goal_success"] is False),
        "prechecks_scheduled": len(prechecks), "prechecks_correctly_rejected": sum(r["rejected"] for r in prechecks),
        "simulated_seconds": sum(r["simulated_seconds"] for r in results),
        "case_wall_seconds": sum(r["wall_seconds_through_case_output"] for r in results),
        "results": results, "prechecks": prechecks,
        "physics_calibrated": False, "learned_skill": False, "physical_robot_ready": False}


def run(output, assets):
    """Execute each frozen case once. A started/incomplete attempt blocks reuse."""
    began = time.perf_counter()
    root = Path(output)
    protocol, seal, inventory = _verify_prepared(root)
    write(root/"execution-intent.json", {"schema": SCHEMA, "protocol": seal["protocol"],
        "inputs": seal["inputs"], "scheduled": len(protocol["cases"]), "worker_count": 1})
    results, prechecks, entries = [], [], []
    for case in protocol["cases"]:
        folder = root/"cases"/case["id"]
        write(folder/"started.json", {"case": case, "protocol": seal["protocol"], "source": protocol["source"]})
        if case["control"] == "precheck":
            error = None
            try:
                bridge.validate_recipe(_read_json(folder/"input/recipe.json"))
            except ValueError as exc:
                error = str(exc)
            result = {"id": case["id"], "label": case["label"], "rejected": error is not None,
                      "error": error, "native_run": False, "expected_precheck_rejection": True}
            write(folder/"result.json", result)
            prechecks.append(result)
        else:
            result, entry = _run_native(folder/"input/recipe.json", assets, folder, case, True, geometry_root=root/"geometry")
            entry["scene"], entry["poses"] = "cases/"+entry["scene"], "cases/"+entry["poses"]
            results.append(result)
            entries.append(entry)
        print(json.dumps({"case": case["id"], "name": case["name"],
                          "checks_passed": result.get("checks_passed", result.get("rejected")),
                          "native_goal_success": result.get("native_goal_success")}), flush=True)
    if (sources() != protocol["source"] or identity((root/"protocol.json").read_bytes()) != seal["protocol"]
            or identity((root/"input-inventory.json").read_bytes()) != seal["inputs"]
            or inventory != {"schema": SCHEMA, "files": _input_inventory(root)}):
        raise ValueError("Frozen source, protocol or input changed during execution; no summary sealed")
    summary = _summary(results, prechecks, True)
    summary["protocol_sha256"] = seal["protocol"]["sha256"]
    write(root/"index.json", {"schema": REPLAY_SCHEMA, "registered_benchmark": True, "scope": SCOPE,
        "cases": entries, "prechecks": prechecks})
    summary["wall_seconds_through_final_outputs"] = time.perf_counter()-began
    summary["simulated_seconds_per_wall_second"] = summary["simulated_seconds"]/summary["wall_seconds_through_final_outputs"]
    write(root/"report.json", summary)
    write(root/"completion.json", {"schema": SCHEMA, "report": identity((root/"report.json").read_bytes()),
        "index": identity((root/"index.json").read_bytes()), "wall_seconds_including_report_write": time.perf_counter()-began})
    return summary


def run_benchmark(output, assets):
    """Prepare and execute a new fixed roster without retrying any attempt."""
    began = time.perf_counter()
    prepare(output)
    summary = run(output, assets)
    write(Path(output)/"full-wall-timing.json", {"schema": SCHEMA,
        "wall_seconds_including_preparation_and_completion": time.perf_counter()-began,
        "scope": "Includes prepare, native execution, all report/geometry/compression/receipt writes; excludes this small timing receipt's serialization."})
    return summary


def run_recipe(recipe_path, assets, output, duration_s=1.5):
    """Bounded local preview. It is deliberately separate from registered results."""
    began = time.perf_counter()
    if type(duration_s) not in (float, int) or not math.isfinite(duration_s) or not .01 <= duration_s <= 10.:
        raise ValueError("Preview duration must be between0.01and10seconds")
    if abs(duration_s-round(duration_s/DT)*DT) > 1e-10:
        raise ValueError("Preview duration must be aligned to1ms")
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise ValueError("Use a fresh preview output; existing evidence is never overwritten")
    source = Path(recipe_path)
    # Validate/inspect the provided source through build_scene in the attempt,
    # retaining exact input bytes first, including any subsequently invalid file.
    recipe = bridge.load_recipe(source)
    root.mkdir(parents=True, exist_ok=False)
    source_id = sources()
    for name in SOURCE_FILES:
        write(root/"source"/name, Path(__file__).with_name(name).read_bytes())
    folder = root/"cases"/"preview"
    for name in ("recipe.json", "workstation.usda", "scene-receipt.json"):
        path = source.parent/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON:
            raise ValueError("Preview requires bounded regular authored scene files")
        write(folder/"input"/name, path.read_bytes())
    case = {"id": "preview", "name": "local-preview", "label": "Local passive physics preview",
            "expected_goal_success": None, "duration_s": float(duration_s), "control": "normal"}
    write(root/"execution-intent.json", {"schema": SCHEMA, "registered_benchmark": False,
        "source": source_id, "case": case, "scope": "Local preview; not a registered benchmark. "+SCOPE})
    write(folder/"started.json", {"case": case, "source": source_id})
    result, entry = _run_native(folder/"input/recipe.json", assets, folder, case, False, geometry_root=root/"geometry")
    entry["scene"], entry["poses"] = "cases/"+entry["scene"], "cases/"+entry["poses"]
    if sources() != source_id:
        raise ValueError("Preview source changed during execution; no summary sealed")
    summary = _summary([result], [], False)
    summary["scope"] = "Local preview; not a registered benchmark. "+SCOPE
    write(root/"recipe.json", recipe)
    write(root/"index.json", {"schema": REPLAY_SCHEMA, "registered_benchmark": False,
        "scope": summary["scope"], "cases": [entry], "prechecks": []})
    summary["wall_seconds_through_final_outputs"] = time.perf_counter()-began
    write(root/"report.json", summary)
    write(root/"completion.json", {"schema": SCHEMA, "registered_benchmark": False,
        "report": identity((root/"report.json").read_bytes()), "index": identity((root/"index.json").read_bytes()),
        "wall_seconds_including_report_write": time.perf_counter()-began})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "benchmark", "preview"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--assets")
    parser.add_argument("--recipe")
    parser.add_argument("--duration", type=float, default=1.5)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output)
    else:
        if not args.assets:
            parser.error("Native execution requires --assets")
        if args.command == "preview":
            if not args.recipe:
                parser.error("Preview requires --recipe")
            result = run_recipe(args.recipe, args.assets, args.output, args.duration)
        else:
            result = run(args.output, args.assets) if args.command == "run" else run_benchmark(args.output, args.assets)
    print(json.dumps({key: value for key, value in result.items() if key not in ("results", "cases", "source")}, indent=2))


if __name__ == "__main__":
    main()
