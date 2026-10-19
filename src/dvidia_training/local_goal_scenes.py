"""Fresh seeded native states for the local physical-goal verifier benchmark.

Reuses the frozen arm, rigid proxy geometry, drives and physical rubric. Only
authored reset poses, fixed visual goal markers, the inspection cover and camera
azimuth move. These are verification fixtures, not trained policy trials.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time

from . import image_goal_scenes as frozen
from . import local_goal_verifier as verifier
from . import reference_sim as sim, robot_profiles, workstation as ws

SCHEMA = "dvidia.local-goal-native-scenes/v1"
SEED = 20261019
FRAME = "workstation_world"
SCENARIOS = frozen.CATEGORIES
_MEASUREMENTS = ("position_m","yaw_rad","linear_speed_m_s","angular_speed_rad_s",
                 "table_support_force_n","robot_contact_force_n")
NOTES = {
    "complete":"The free narrow shoe settles at the goal with table support and no robot contact.",
    "missing_target":"The target body is absent from the model; presence is a privileged native lookup.",
    "wrong_object":"An orange decoy replaces the target body; the requested target is absent.",
    "wrong_position":"The target starts 170 mm along world X from its goal.",
    "wrong_orientation":"The target starts with a 90 degree yaw error relative to the goal.",
    "unreleased":"An authored approach and close create real finger contact while the shoe remains table-supported; this is not a lift policy.",
    "moving_alias":"A declared reset intervention changes free-body linear velocity to [0.12,0,0] m/s without advancing time. It replaces the last measured sample, retaining its timestamp; no duplicate time or measured impulse is claimed.",
    "occluded":"An opaque fixed inspection cover hides the shoe from the camera; privileged native telemetry remains available.",
}


def make_layouts(seed=SEED,count=8):
    """Uniform draws in predeclared ranges, before any verifier/model scoring."""
    if type(seed) is not int or not 0<=seed<=2**63-1:
        raise ValueError("Seed must be a nonnegative bounded integer")
    if type(count) is not int or not 1<=count<=64:
        raise ValueError("Layout count must be an integer in [1,64]")
    rng=random.Random(seed)
    return [{"id":f"fresh-{i+1:02d}",
             "xy_m":[round(rng.uniform(.385,.425),6),round(rng.uniform(-.165,-.085),6)],
             "yaw_rad":round(rng.uniform(-.12,.12),6),
             "azimuth_deg":round(rng.uniform(118.,158.),3)} for i in range(count)]


def _validate_layouts(layouts):
    if type(layouts) is not list or not 1<=len(layouts)<=64:
        raise ValueError("Expected between one and 64 declared layouts")
    seen=set()
    for layout in layouts:
        if type(layout) is not dict or set(layout)!={"id","xy_m","yaw_rad","azimuth_deg"}:
            raise ValueError("Layout has missing or unexpected fields")
        identifier=layout["id"]
        if type(identifier) is not str or not identifier.startswith("fresh-") or not identifier[6:].isdigit() or identifier in seen:
            raise ValueError("Expected unique fresh-NN layout IDs")
        seen.add(identifier)
        xy=layout["xy_m"]
        if type(xy) is not list or len(xy)!=2:
            raise ValueError("Layout XY must have two components")
        for value,low,high in ((xy[0],.385,.425),(xy[1],-.165,-.085),
                               (layout["yaw_rad"],-.12,.12),(layout["azimuth_deg"],118.,158.)):
            if type(value) not in (int,float) or not low<=value<=high or not math.isfinite(value):
                raise ValueError("Layout lies outside the declared fresh ranges")


def _write(path,value):
    content=(json.dumps(value,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(content)


def build_fresh_model(assets,layout,scenario):
    """Relocate a frozen compiled scene at reset, preserving robot mechanics."""
    _validate_layouts([layout])
    if scenario not in SCENARIOS:
        raise ValueError("Unknown native scenario")
    mj,np=sim._native_imports()
    reference=frozen.LAYOUTS[0]
    model=frozen.build_model(assets,reference,scenario)
    name="orange_decoy_shoe" if scenario=="wrong_object" else ws.SHOE
    obj=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)
    if obj>=0:
        joint=int(model.body_jntadr[obj]);address=int(model.jnt_qposadr[joint])
        if model.jnt_type[joint]!=mj.mjtJoint.mjJNT_FREE:
            raise RuntimeError("Expected the unchanged free shoe body")
        xy=list(layout["xy_m"])
        if scenario=="wrong_position":xy[0]+=.17
        yaw=layout["yaw_rad"]+(math.pi/2 if scenario=="wrong_orientation" else 0.)
        model.qpos0[address:address+3]=[*xy,.012]
        model.qpos0[address+3:address+7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
    # Goal-outline geoms belong to the static world and have no collisions.
    delta=layout["yaw_rad"]-reference["yaw_rad"]
    rotation=np.asarray([[math.cos(delta),-math.sin(delta)],[math.sin(delta),math.cos(delta)]])
    for index in range(4):
        geom=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,f"goal_outline_{index}")
        if geom<0:raise RuntimeError("Frozen goal outline is missing")
        offset=model.geom_pos[geom,:2]-reference["xy_m"]
        model.geom_pos[geom,:2]=np.asarray(layout["xy_m"])+rotation@offset
        yaw=layout["yaw_rad"]
        model.geom_quat[geom]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
    cover=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,"opaque_inspection_cover")
    if cover>=0:model.geom_pos[cover,:2]=layout["xy_m"]
    # Recompute native constants after static/reset geometry changes.
    mj.mj_setConst(model,mj.MjData(model))
    return model


def _measurement(model,data,layout):
    """Augment the frozen independent oracle with raw time and yaw observations."""
    mj,_=sim._native_imports()
    row=frozen._measure(model,data,layout)
    obj=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,ws.SHOE)
    rotation=data.xmat[obj].reshape(3,3) if obj>=0 else None
    row["t_s"]=float(data.time)
    row["yaw_rad"]=None if rotation is None else math.atan2(float(rotation[1,0]),float(rotation[0,0]))
    return row


def _run(assets,layout,scenario,arm_qpos):
    mj,np=sim._native_imports()
    model=build_fresh_model(assets,layout,scenario)
    data=mj.MjData(model)
    data.qpos[:8]=arm_qpos
    data.ctrl[:6]=data.qpos[:6];data.ctrl[6]=.035
    mj.mj_forward(model,data)
    history=[]
    steps=0
    def tick():
        nonlocal steps
        mj.mj_step(model,data);mj.mj_forward(model,data);steps+=1
        if any(w.number for w in data.warning) or not all(np.isfinite(v).all() for v in (data.qpos,data.qvel,data.qacc)):
            raise RuntimeError("Native warning or non-finite state during fixture construction")
        history.append(_measurement(model,data,layout))
        del history[:-100]
    for _ in range(300):tick()
    if scenario=="unreleased":
        yaw=layout["yaw_rad"]
        target=[layout["xy_m"][0]-.085*math.cos(yaw),layout["xy_m"][1]-.085*math.sin(yaw),.055]
        for stage_target,jaw,count in (([*target[:2],.20],.035,1500),(target,.035,1000),(target,0.,600)):
            for step in range(count):
                if step%10==0:desired=ws.controller_action(model,data,stage_target,jaw)
                data.ctrl[:6]+=np.clip(desired[:6]-data.ctrl[:6],-.003,.003)
                data.ctrl[6]+=float(np.clip(desired[6]-data.ctrl[6],-.0001,.0001))
                tick()
    if scenario=="moving_alias":
        obj=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,ws.SHOE)
        dof=int(model.jnt_dofadr[int(model.body_jntadr[obj])])
        data.qvel[dof:dof+3]=[.12,0.,0.]
        mj.mj_forward(model,data)
        # The most recent observation is replaced at the same instant. Its
        # prior 99 observations remain real preceding native states.
        previous_time=history[-1]["t_s"]
        history[-1]=_measurement(model,data,layout)
        if history[-1]["t_s"]!=previous_time:
            raise RuntimeError("Velocity intervention unexpectedly advanced the native clock")
    truth,_,_=frozen.observe(model,data,layout,history)
    current=_measurement(model,data,layout)
    return model,data,truth,current,history,steps


def _packet(history):
    if len(history)!=100:
        raise RuntimeError("Fresh native fixture must retain exactly 100 samples")
    samples=[]
    for measured in history:
        row={"t_s":measured["t_s"],"object_id":ws.SHOE,"frame_id":FRAME,
             "presence":"present" if measured["target_present"] else "absent"}
        if measured["target_present"]:
            row.update(position_m=measured["target_position_m"],yaw_rad=measured["yaw_rad"],
                linear_speed_m_s=measured["linear_speed_m_s"],angular_speed_rad_s=measured["angular_speed_rad_s"],
                table_support_force_n=measured["table_support_force_n"],robot_contact_force_n=measured["robot_contact_force_n"])
        else:
            row.update({key:None for key in _MEASUREMENTS})
        samples.append(row)
    return {"schema":verifier.SCHEMA,"units":dict(verifier.UNITS),
            "uncertainty_bounds":{key:0. for key in _MEASUREMENTS},
            "sample_period_s":ws.TIMESTEP,"history_start_s":samples[0]["t_s"]-ws.TIMESTEP,
            "samples":samples}


def generate_native(assets,output,layouts):
    """Write every declared case, preserving explicit construction failures.

    No local verifier call participates in native labels. Callers score the
    observation packets later. Existing per-case files are never overwritten.
    """
    _validate_layouts(layouts)
    destination=Path(output)
    if destination.is_symlink():raise ValueError("Output cannot be a symlink")
    destination.mkdir(parents=True,exist_ok=True)
    metadata_path=destination/"native-scenes.json"
    if metadata_path.exists():raise ValueError("Native scene metadata already exists")
    for name in ("images","states","packets"):
        subdir=destination/name
        if subdir.is_symlink():raise ValueError("Output subdirectories cannot be symlinks")
        subdir.mkdir(exist_ok=True)
    # Validate output collisions before simulating anything.
    for layout in layouts:
        for scenario in SCENARIOS:
            identifier=f"native-{layout['id']}-{scenario}"
            for name,suffix in (("images","png"),("states","json"),("packets","json")):
                if (destination/name/f"{identifier}.{suffix}").exists():
                    raise ValueError("Native fixture output already exists: "+identifier)
    wall=time.perf_counter()
    asset_metadata=robot_profiles.verify_assets("piper",assets)
    arm=frozen._initial_arm(assets)
    rows=[]
    for layout in layouts:
        complete_png=None;complete_qpos=None
        goal_path=f"images/native-{layout['id']}-complete.png"
        contract=asdict(verifier.GoalContract(ws.SHOE,FRAME,tuple(layout["xy_m"]),layout["yaw_rad"]))
        contract["goal_xy_m"]=list(contract["goal_xy_m"])
        for scenario in SCENARIOS:
            identifier=f"native-{layout['id']}-{scenario}"
            start=time.perf_counter()
            row={"id":identifier,"cohort":"native","scenario":scenario,"layout_id":layout["id"],
                 "contract":contract,"now_s":None,"packet_file":None,"current_image":None,
                 "goal_image":goal_path,"state_file":f"states/{identifier}.json",
                 "physical_truth":None,"expected_verdict":None,"note":NOTES[scenario],
                 "generation_valid":False}
            try:
                model,data,truth,current,history,steps=_run(assets,layout,scenario,arm)
                png,visibility=frozen.render(model,data,layout)
                packet=_packet(history)
                problems=[]
                try:frozen._check_category(scenario,truth,current,visibility)
                except RuntimeError as error:problems.append(str(error))
                if scenario=="complete":
                    complete_png=png;complete_qpos=data.qpos.tolist()
                alias=None
                if scenario=="moving_alias":
                    alias={"paired_case_id":f"native-{layout['id']}-complete",
                           "pose_identical":data.qpos.tolist()==complete_qpos,
                           "png_byte_identical":png==complete_png}
                    if not alias["pose_identical"]:problems.append("Moving alias native pose differs from complete control")
                    if not alias["png_byte_identical"]:problems.append("Moving alias native-rendered PNG differs from complete control")
                image_path=f"images/{identifier}.png";packet_path=f"packets/{identifier}.json"
                with (destination/image_path).open("xb") as stream:stream.write(png)
                _write(destination/packet_path,packet)
                state={"schema":SCHEMA,"layout":layout,"scenario":scenario,
                    "time_s":float(data.time),"qpos":data.qpos.tolist(),"qvel":data.qvel.tolist(),
                    "ctrl":data.ctrl.tolist(),"current_measurement":current,"history":history,
                    "physical_truth":truth,"physics_timestep_s":float(model.opt.timestep),
                    "native_steps":steps,"solver_warnings":sum(int(w.number) for w in data.warning),
                    "object_attachment_constraints_added":0,"model_equalities":int(model.neq),
                    "external_forces_applied":False,"velocity_reset_intervention":scenario=="moving_alias",
                    "object_gravity_compensation_added":False,"gripper_kp_n_per_m":float(model.actuator_gainprm[6,0]),
                    "generation_errors":problems,"alias":alias}
                _write(destination/row["state_file"],state)
                row.update(now_s=float(data.time),packet_file=packet_path,current_image=image_path,
                    physical_truth=truth,expected_verdict="complete" if truth["complete"] else "incomplete",
                    generation_valid=not problems,generation_errors=problems,visibility=visibility,alias=alias)
            except Exception as error:
                # All attempted cases remain visible in the manifest. This is a
                # construction error, never counted as a model/verifier result.
                error_record={"schema":SCHEMA,"layout":layout,"scenario":scenario,
                              "construction_error":type(error).__name__+": "+str(error)}
                if not (destination/row["state_file"]).exists():
                    _write(destination/row["state_file"],error_record)
                row["generation_errors"]=[error_record["construction_error"]]
            row["generation_wall_seconds"]=time.perf_counter()-start
            rows.append(row)
    _write(metadata_path,{"schema":SCHEMA,"layouts":layouts,"scenarios":list(SCENARIOS),
        "cases":rows,"assets":asset_metadata,"mujoco_version":sim._native_imports()[0].__version__,
        "source_fingerprints":{name:sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("local_goal_scenes.py","local_goal_verifier.py","image_goal_scenes.py",
                         "workstation.py","reference_sim.py","robot_profiles.py","asset_library.py",
                         "robot_profiles.lock.json","asset_catalog.json")},
        "generation_wall_seconds":time.perf_counter()-wall,
        "scope":"Fresh seeded authored native states using frozen narrow rigid shoe/PiPER mechanics. No trained policy, hardware sensing, physical calibration or real-world transfer claim.",
        "oracle":"Frozen image_goal_scenes._measure/observe numeric predicates over all 100 native samples; independent of local_goal_verifier decisions.",
        "moving_alias_scope":"The latest native observation is replaced after an explicit instantaneous velocity reset at the same timestamp; earlier 99 samples remain unchanged.",
        "telemetry_scope":"Privileged native poses, identity and forces with explicit zero uncertainty assumptions, not real calibrated sensors.",
        "rendering":"Native MuJoCo 640x480 RGB and geom-ID segmentation; no image synthesis model."})
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--count",type=int,default=8)
    args=parser.parse_args(argv)
    rows=generate_native(args.assets,args.output,make_layouts(args.seed,args.count))
    valid=sum(row["generation_valid"] for row in rows)
    print(json.dumps({"cases":len(rows),"generation_valid":valid,"generation_invalid":len(rows)-valid,
                      "complete":sum(bool(row["physical_truth"] and row["physical_truth"]["complete"]) for row in rows),
                      "output":str(args.output)}))
    return 0 if valid==len(rows) else 1


if __name__=="__main__":raise SystemExit(main())
