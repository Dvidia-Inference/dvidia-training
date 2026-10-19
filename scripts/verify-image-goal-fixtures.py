#!/usr/bin/env python3
"""Independently audit frozen image-goal evidence, without rendering or an API.

The numerical predicate below deliberately does not call the simulator's
``observe``/``_measure`` functions or trust their stored completion flags.
It checks recorded native evidence, not a fresh simulation or hardware run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dvidia_training.image_goal_scenes import CRITERIA, expected_labels  # noqa: E402

SCHEMA = "dvidia.image-goal-native-validation/v1"
METRICS = ("position_error_m", "yaw_error_rad", "linear_speed_m_s",
           "angular_speed_rad_s", "table_support_force_n", "robot_contact_force_n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid finite number: {label}")
    return float(value)


def same_number(actual, expected, label):
    if expected is None:
        require(actual is None, f"Expected null: {label}")
    else:
        require(math.isclose(finite(actual,label), finite(expected,label), rel_tol=1e-10, abs_tol=1e-12),
                f"Numeric mismatch: {label}")


def regular_file(root, name):
    require(isinstance(name,str) and "\\" not in name, "Manifest path must be POSIX text")
    relative=PurePosixPath(name)
    require(not relative.is_absolute() and relative.as_posix()==name and
            all(part not in ("", ".", "..") for part in relative.parts), "Unsafe manifest path")
    path=root.joinpath(*relative.parts)
    require(all(not p.is_symlink() for p in (path,*path.parents) if p==root or root in p.parents),
            f"Symlink in fixture path: {name}")
    require(path.is_file(), f"Missing regular fixture file: {name}")
    return path


def load_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value in {path.name}: {value}")
    return json.loads(path.read_text(encoding="utf-8"),parse_constant=reject)


def physical_predicate(row, criteria):
    """Recompute from numbers, ignoring instantaneous_conditions_met labels."""
    require(type(row["target_present"]) is bool, "Presence must be a boolean")
    if not row["target_present"]:
        require(all(row[key] is None for key in METRICS[:4]), "Absent target has pose/speed evidence")
        return False
    values={key:finite(row[key],key) for key in METRICS}
    require(all(value>=0 for value in values.values()), "Negative scalar physical measurement")
    return (values["position_error_m"]<=criteria["position_tolerance_m"]
            and values["yaw_error_rad"]<=criteria["yaw_tolerance_rad"]
            and values["linear_speed_m_s"]<=criteria["linear_speed_max_m_s"]
            and values["angular_speed_rad_s"]<=criteria["angular_speed_max_rad_s"]
            and values["table_support_force_n"]>criteria["support_force_min_n"]
            and values["robot_contact_force_n"]<=criteria["robot_force_max_n"])


def recompute_truth(state, criteria, dt):
    current=state["current_measurement"]
    history=state["history"]
    require(isinstance(history,list) and len(history)<=100, "Unexpected native observation window")
    current_ok=physical_predicate(current,criteria)
    count=0
    for row in reversed(history):
        if not physical_predicate(row,criteria):
            break
        count+=1
    dwell=count*dt if current_ok else 0.
    return {"present":current["target_present"],
            "complete":bool(current_ok and dwell+1e-12>=criteria["continuous_dwell_s"]),
            "physical_conditions_dwell_s":dwell,
            "position_error_m":current["position_error_m"],
            "yaw_error_rad":current["yaw_error_rad"],
            "supported":finite(current["table_support_force_n"],"support")>criteria["support_force_min_n"],
            "released":finite(current["robot_contact_force_n"],"robot contact")<=criteria["robot_force_max_n"],
            "linear_speed_m_s":current["linear_speed_m_s"],
            "angular_speed_rad_s":current["angular_speed_rad_s"]}


def compare_truth(stored, derived, label):
    require(set(stored)==set(derived), f"Truth fields differ: {label}")
    for key,value in derived.items():
        if type(value) is bool:
            require(type(stored[key]) is bool and stored[key] is value, f"Truth mismatch: {label}.{key}")
        else:
            same_number(stored[key],value,f"{label}.{key}")


def verify_snapshot(state, layout):
    """Cross-check raw native coordinates, independent of measurement labels."""
    measurement=state["current_measurement"]
    for name in ("qpos","qvel","ctrl"):
        require(isinstance(state[name],list), f"Invalid {name}")
        for value in state[name]:
            finite(value,name)
    require(len(state["ctrl"])==7, "Expected seven pinned PiPER actuator commands")
    if not measurement["target_present"]:
        return
    qpos,qvel=state["qpos"],state["qvel"]
    require(len(qpos)==15 and len(qvel)==14, "Expected PiPER plus one free shoe")
    for actual,recorded in zip(qpos[8:11],measurement["target_position_m"]):
        same_number(actual,recorded,"free-body target position")
    xy_error=math.hypot(qpos[8]-layout["xy_m"][0],qpos[9]-layout["xy_m"][1])
    same_number(xy_error,measurement["position_error_m"],"XY goal error")
    w,x,y,z=qpos[11:15]
    same_number(sum(v*v for v in (w,x,y,z)),1.,"unit free-body quaternion")
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    yaw_error=abs(math.atan2(math.sin(yaw-layout["yaw_rad"]),math.cos(yaw-layout["yaw_rad"])))
    same_number(yaw_error,measurement["yaw_error_rad"],"yaw goal error")
    same_number(math.sqrt(sum(v*v for v in qvel[8:11])),measurement["linear_speed_m_s"],"native linear speed")
    same_number(math.sqrt(sum(v*v for v in qvel[11:14])),measurement["angular_speed_rad_s"],"native angular speed")


def verify_telemetry(state, inputs, dt):
    current,history=state["current_measurement"],state["history"]
    require("complete" not in inputs and "instantaneous_conditions_met" not in inputs,
            "A verdict leaked into measured inputs")
    for key in ("target_present","target_position_m",*METRICS):
        require(inputs[key]==current[key], f"Telemetry snapshot differs: {key}")
    require(inputs["native_observation_count"]==len(history), "Native tick count differs")
    require(inputs["target_present_observation_count"]==sum(row["target_present"] for row in history),
            "Presence observation count differs")
    same_number(inputs["native_timestep_s"],dt,"native dt")
    same_number(inputs["observation_window_s"],len(history)*dt,"observation duration")
    for key in METRICS:
        values=[row[key] for row in history if row[key] is not None]
        expected={"min":min(values) if values else None,"max":max(values) if values else None}
        for statistic,value in expected.items():
            same_number(inputs["native_window_extrema"][key][statistic],value,f"{key}.{statistic}")


def verify(fixtures):
    root=Path(fixtures).absolute()
    require(root.is_dir() and not root.is_symlink(), "Fixtures must be a regular directory")
    integrity=load_json(regular_file(root,"integrity.json"))
    records=integrity["files"]
    names=[record["path"] for record in records]
    require(len(set(names))==len(names), "Duplicate integrity path")
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    required=set(names)|{"integrity.json"}
    require(required<=actual and actual-required<={"evaluation-plan.json"},
            "Integrity manifest coverage differs or unknown additional files exist")
    additional_metadata=[]
    for name in sorted(actual-required):
        data=regular_file(root,name).read_bytes()
        additional_metadata.append({"path":name,"bytes":len(data),"sha256":sha256(data).hexdigest(),
            "scope":"Runner metadata added after fixture generation; fingerprinted here, not claimed to be covered by the original fixture integrity manifest."})
    byte_count=0
    for record in records:
        path=regular_file(root,record["path"])
        data=path.read_bytes()
        require(len(data)==record["bytes"], f"Byte count differs: {record['path']}")
        require(sha256(data).hexdigest()==record["sha256"], f"Hash differs: {record['path']}")
        byte_count+=len(data)
    protocol_path=regular_file(root,"protocol.json")
    protocol=load_json(protocol_path)
    manifest=load_json(regular_file(root,"cases.json"))
    protocol_digest=sha256(protocol_path.read_bytes()).hexdigest()
    require(protocol_digest==regular_file(root,"protocol.sha256").read_text().strip()==manifest["protocol_sha256"],
            "Protocol digests disagree")
    require(protocol["criteria"]==manifest["criteria"]==CRITERIA,"Criteria differ from the frozen verifier contract")
    criteria=protocol["criteria"]
    dt=finite(protocol["physics_timestep_s"],"physics timestep")
    same_number(dt,.002,"fixed native timestep")
    layouts={row["id"]:row for row in protocol["layouts"]}
    cases=manifest["cases"]
    require(len(cases)==32 and {case["id"] for case in cases}==set(protocol["case_ids"]),
            "Frozen case coverage differs")
    category_counts=Counter(case["category"] for case in cases)
    require(set(category_counts)==set(protocol["categories"]) and set(category_counts.values())=={4},
            "Unbalanced category coverage")
    rows=[];states={}
    for case in cases:
        identifier=case["id"]
        state=load_json(regular_file(root,case["state_file"]))
        states[identifier]=state
        truth=recompute_truth(state,criteria,dt)
        compare_truth(state["physical_truth"],truth,identifier+".state")
        compare_truth(case["physical_truth"],truth,identifier+".case")
        verify_snapshot(state,layouts[case["layout_id"]])
        verify_telemetry(state,case["measured_inputs"],dt)
        require(state["solver_warnings"]==0,"Native solver warning recorded")
        require(state["object_attachment_constraints_added"]==0,"Object attachment recorded")
        require(state["model_equalities"]==1,"Unexpected equality constraint count")
        require(state["external_forces_applied"] is False,"Undeclared external force recorded")
        for row in (state["current_measurement"],*state["history"]):
            require(row["instantaneous_conditions_met"] is physical_predicate(row,criteria),
                    f"Stored instantaneous label disagrees with numbers: {identifier}")
        labels=expected_labels(case["category"],truth)
        for key in labels:
            require(case[key]==labels[key], f"Observability labels differ: {identifier}")
        visibility=case["visibility"]
        if case["category"] in ("occluded","missing_target","wrong_object"):
            require(visibility["visible_target_pixels"]==0 and visibility["target_box_xyxy_normalized"] is None,
                    "An invisible/absent target has a visible box")
        else:
            require(visibility["visible_target_pixels"]>0 and visibility["target_box_xyxy_normalized"] is not None,
                    "Visible target is missing its native segmentation box")
        if case["category"]=="unreleased":
            require(truth["supported"] and not truth["released"] and not truth["complete"],
                    "Unreleased fixture lacks supported robot contact")
            require(truth["position_error_m"]<=criteria["position_tolerance_m"] and
                    truth["yaw_error_rad"]<=criteria["yaw_tolerance_rad"],
                    "Unreleased fixture is outside the goal pose")
            require(case["expected_verdict"]=={"rgb":"unknown","rgb_telemetry":"incomplete"},
                    "Unreleased RGB overclaims physical contact force")
        rows.append({"id":identifier,"category":case["category"],"present":truth["present"],
            "complete":truth["complete"],"physical_conditions_dwell_s":truth["physical_conditions_dwell_s"],
            "supported":truth["supported"],"released":truth["released"],
            "position_error_m":truth["position_error_m"],"yaw_error_rad":truth["yaw_error_rad"],
            "table_support_force_n":state["current_measurement"]["table_support_force_n"],
            "robot_contact_force_n":state["current_measurement"]["robot_contact_force_n"],
            "linear_speed_m_s":truth["linear_speed_m_s"],"angular_speed_rad_s":truth["angular_speed_rad_s"],
            "native_history_samples":len(state["history"]),"expected_verdict":labels["expected_verdict"]})
    index={case["id"]:case for case in cases}
    aliases=[]
    require(len(manifest["aliases"])==4,"Expected four exact-pixel aliases")
    require({pair["case_id"] for pair in manifest["aliases"]}==
            {case["id"] for case in cases if case["category"]=="moving_alias"},
            "Alias manifest does not cover all four moving controls")
    for pair in manifest["aliases"]:
        moving,stationary=index[pair["case_id"]],index[pair["pixel_identical_to"]]
        require(moving["category"]=="moving_alias" and stationary["category"]=="complete" and
                moving["layout_id"]==stationary["layout_id"],"Alias identity differs")
        a=regular_file(root,moving["current_image"]).read_bytes()
        b=regular_file(root,stationary["current_image"]).read_bytes()
        require(a==b,"Current-image PNG alias bytes differ")
        require(states[moving["id"]]["qpos"]==states[stationary["id"]]["qpos"],"Alias native poses differ")
        require(states[moving["id"]]["qvel"]!=states[stationary["id"]]["qvel"],"Alias native velocities do not differ")
        require(not moving["physical_truth"]["complete"] and stationary["physical_truth"]["complete"],
                "Alias physical truth does not differ")
        require(len(states[moving["id"]]["history"])==0,"Post-intervention observation time was invented")
        aliases.append({**pair,"png_sha256":sha256(a).hexdigest(),"png_bytes":len(a),
                        "byte_identical":True,"pose_identical":True,"velocity_and_completion_differ":True})
    completed=sum(row["complete"] for row in rows)
    require(completed==8 and len(rows)-completed==24,"Expected 8 complete and 24 incomplete states")
    return {"schema":SCHEMA,"status":"passed",
            "scope":"Independent recomputation from frozen native state measurements/history and raw coordinates. No images re-rendered, no fresh physics run, no model service called and no hardware validation.",
            "fixture_generation_scope":manifest["scope"],"telemetry_scope":manifest["telemetry_scope"],
            "source_digests":{"protocol_sha256":protocol_digest,
                "cases_sha256":sha256(regular_file(root,"cases.json").read_bytes()).hexdigest(),
                "integrity_sha256":sha256(regular_file(root,"integrity.json").read_bytes()).hexdigest(),
                "verifier_sha256":sha256(Path(__file__).read_bytes()).hexdigest()},
            "files_verified":len(records),"bytes_verified":byte_count,
            "additional_runner_metadata":additional_metadata,
            "native_generation_wall_seconds":manifest["generation_wall_seconds"],
            "native_generation_timing_scope":"Recorded fixture construction, stepping and rendering; excludes Jev inference and this verification.",
            "recorded_mujoco_version":manifest["mujoco_version"],
            "verification_host":{"system":platform.system(),"release":platform.release(),
                "machine":platform.machine(),"processor":platform.processor() or None,
                "logical_cpu_count":os.cpu_count(),"python_version":platform.python_version(),
                "scope":"Host running this verifier; CPU model, RAM and GPU specifications are not inferred."},
            "criteria":criteria,"case_count":len(rows),"category_counts":dict(sorted(category_counts.items())),
            "physical_complete":completed,"physical_incomplete":len(rows)-completed,
            "zero_solver_warnings":True,"no_added_object_attachment_constraints":True,
            "no_recorded_external_forces":True,"native_equalities_per_model":1,
            "equality_scope":"Recorded upstream gripper coupling count; verified against frozen state metadata, without recompiling models.",
            "all_physical_labels_recomputed":True,"all_telemetry_extrema_recomputed":True,
            "unreleased_rgb_unknown_cases":4,"aliases":aliases,"cases":rows}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    require(args.output.absolute()!=args.fixtures.absolute() and args.fixtures.absolute() not in args.output.absolute().parents,
            "Write validation outside the frozen fixture directory")
    try:
        result=verify(args.fixtures)
        code=0
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as error:
        result={"schema":SCHEMA,"status":"failed","error":str(error)}
        code=1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("status","case_count","physical_complete","physical_incomplete","files_verified","bytes_verified","error") if key in result}))
    return code


if __name__=="__main__":
    raise SystemExit(main())
