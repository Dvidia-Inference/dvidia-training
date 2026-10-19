"""Frozen native PiPER image-goal verification fixtures.

These are authored *state fixtures*, not robot-policy rollouts or evidence of
sim-to-real transfer. RGB cannot reveal an instantaneous velocity or prove a
contact dwell. The matched-pixel moving control makes that limitation explicit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
import time
import zlib

from . import reference_sim as sim, robot_profiles, workstation as ws

SCHEMA = "dvidia.image-goal-scenes/v1"
TASK = ("Place the green shoe inside the outlined goal zone, matching the goal "
        "image's position and orientation. Completion requires the target shoe "
        "to have table support force greater than 0.01 N, robot contact force "
        "at most 0.01 N, XY position error at most 25 mm, yaw error at most "
        "15 degrees, linear speed at most 0.01 m/s and angular speed at most "
        "0.05 rad/s, all continuously for at least 200 ms. "
        "The goal picture is a visual reference, not proof that these physical "
        "conditions hold in the current picture. Answer unknown when the inputs "
        "cannot establish the required physical conditions. An orange shoe is "
        "a different object. The opaque inspection cover can hide the target.")
CATEGORIES = ("complete", "missing_target", "wrong_object", "wrong_position",
              "wrong_orientation", "unreleased", "moving_alias", "occluded")
LAYOUTS = (
    {"id": "layout-01", "xy_m": [.40, -.10], "yaw_rad": .0, "azimuth_deg": 130.},
    {"id": "layout-02", "xy_m": [.43, -.16], "yaw_rad": .08, "azimuth_deg": 145.},
    {"id": "layout-03", "xy_m": [.38, -.07], "yaw_rad": -.08, "azimuth_deg": 115.},
    {"id": "layout-04", "xy_m": [.42, -.13], "yaw_rad": .15, "azimuth_deg": 155.},
)
CRITERIA = {"position_tolerance_m": .025, "yaw_tolerance_rad": math.radians(15),
            "linear_speed_max_m_s": .01, "angular_speed_max_rad_s": .05,
            "support_force_min_n": .01, "robot_force_max_n": .01,
            "continuous_dwell_s": .20}
WIDTH, HEIGHT = 640, 480


def _json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def protocol():
    """Return the predeclared cases and rubric, without native imports/network."""
    return {"schema": SCHEMA, "task": TASK, "criteria": deepcopy(CRITERIA),
            "layouts": deepcopy(LAYOUTS), "categories": list(CATEGORIES),
            "case_ids": [f"{x['id']}-{c}" for x in LAYOUTS for c in CATEGORIES],
            "image_size": [WIDTH, HEIGHT], "physics_timestep_s": ws.TIMESTEP,
            "settling_s": .60, "native_observation_window_s": .20,
            "unreleased_controller": {"stages": ["approach above heel", "descend", "close"],
                "stage_duration_s": [3.,2.,1.2], "heel_offset_m": -.085,
                "tcp_height_m": [.20,.055,.055], "jaw_command_m": [.035,.035,0.],
                "native_dt_s": .002, "control_dt_s": .020,
                "scope": "Authored approach then native finger contact; not a lift or learned manipulation."},
            "moving_alias_velocity_m_s": [.12, 0., 0.],
            "synthetic_shoe": {"source": "packaged trainer-left collision primitives",
                "width_scale": .50, "mass_kg": .030,
                "visual_colors": "Green upper for target, orange upper for decoy, cream sole and toe.",
                "reason": "Declared narrow rigid proxy permits unchanged PiPER finger contact; not a full-size measured shoe."},
            "scope": "Authored native physical-state verification. Four deterministic layouts, not statistical independent real-world scenes or a trained policy benchmark.",
            "dwell_scope": "Every 2 ms native sample satisfies all thresholds for 100 consecutive ticks; no continuous-time or hardware certification claim.",
            "reset_scope": "Each fixture starts from an authored state. The moving alias changes native velocity at the snapshot without advancing time; this is a declared state intervention, not a measured physical impulse.",
            "telemetry_scope": "Privileged native state and contact history; not an implemented camera or hardware estimator.",
            "rgb_observability": "Single RGB cannot prove velocity, support force, release force or 200 ms dwell. Complete, moving, occluded and unreleased states require unknown. Even visibly closed fingers do not establish the 0.01 N contact-force threshold. Missing targets, wrong objects and visible pose violations permit incomplete.",
            "freeze": "Generate and hash this protocol before any model requests; never choose cases or thresholds from model results."}


def _validate(layout, category):
    if layout not in LAYOUTS or category not in CATEGORIES:
        raise ValueError("Expected a frozen image-goal layout and category")


def build_model(assets, layout, category="complete"):
    """Build one frozen condition with pinned arm mechanics and no attachments."""
    _validate(layout, category)
    mj, _ = sim._native_imports()
    spec = ws._spec(assets)
    shoe = spec.body(ws.SHOE)
    xy = list(layout["xy_m"])
    yaw = layout["yaw_rad"]
    if category == "missing_target":
        spec.delete(shoe)
    else:
        if category == "wrong_position":
            xy[0] += .17
        if category == "wrong_orientation":
            yaw += math.pi / 2
        if category == "wrong_object":
            shoe.name = "orange_decoy_shoe"
        shoe.pos = [*xy, .012]
        shoe.quat = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        for geom in shoe.geoms:
            geom.pos[1] *= .5
            geom.size[1] *= .5
            geom.mass *= .1
            if "upper" in geom.name:
                geom.rgba = ([.79, .26, .07, 1.] if category == "wrong_object"
                             else [.12, .43, .27, 1.])
            else:
                geom.rgba = [.84,.84,.76,1.]
    # The zone is an explicitly non-colliding visual reference on the table.
    for i, (dx, dy, sx, sy) in enumerate(((0, -.058, .16, .002),
            (0, .058, .16, .002), (-.16, 0, .002, .06), (.16, 0, .002, .06))):
        yaw = layout["yaw_rad"]
        x = layout["xy_m"][0] + dx*math.cos(yaw)-dy*math.sin(yaw)
        y = layout["xy_m"][1] + dx*math.sin(yaw)+dy*math.cos(yaw)
        spec.worldbody.add_geom(name=f"goal_outline_{i}", type=mj.mjtGeom.mjGEOM_BOX,
            pos=[x,y,.0006], size=[sx,sy,.0004],
            quat=[math.cos(yaw/2),0,0,math.sin(yaw/2)], rgba=[.10,.27,.23,1.],
            contype=0, conaffinity=0)
    if category == "occluded":
        spec.worldbody.add_geom(name="opaque_inspection_cover",
            type=mj.mjtGeom.mjGEOM_BOX, pos=[*layout["xy_m"], .14],
            size=[.24,.20,.006], rgba=[.62,.65,.62,1.], contype=1, conaffinity=1)
    spec.visual.global_.offwidth = WIDTH
    spec.visual.global_.offheight = HEIGHT
    spec.visual.headlight.ambient = [.4,.4,.4]
    spec.visual.headlight.diffuse = [.6,.6,.6]
    return spec.compile()


def _initial_arm(assets):
    model = ws.build_model(assets)
    return ws.initialize(model).qpos[:8].copy()


def _measure(model, data, layout):
    mj, np = sim._native_imports()
    obj = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ws.SHOE)
    support = robot = 0.
    contacts = []
    if obj < 0:
        return {"target_present": False, "target_position_m": None,
            "position_error_m": None, "yaw_error_rad": None,
            "linear_speed_m_s": None, "angular_speed_rad_s": None,
            "table_support_force_n": 0., "robot_contact_force_n": 0.,
            "contacts": [], "instantaneous_conditions_met": False}
    table = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "workstation_table_top")
    for i in range(data.ncon):
        c = data.contact[i]
        ga, gb = int(c.geom1), int(c.geom2)
        a, b = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
        if obj not in (a,b):
            continue
        other_geom, other_body = (gb,b) if a == obj else (ga,a)
        wrench = np.zeros(6)
        mj.mj_contactForce(model, data, i, wrench)
        normal = max(0., float(wrench[0]))
        if other_geom == table:
            support += normal
        if other_body > 0 and other_body != obj:
            robot += normal
        if normal > .0001:
            contacts.append({"other_geom": mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,other_geom),
                             "normal_force_n": normal})
    joint = int(model.body_jntadr[obj])
    dof = int(model.jnt_dofadr[joint])
    linear = float(np.linalg.norm(data.qvel[dof:dof+3]))
    angular = float(np.linalg.norm(data.qvel[dof+3:dof+6]))
    pos = data.xpos[obj].tolist()
    rotation = data.xmat[obj].reshape(3,3)
    yaw = math.atan2(float(rotation[1,0]),float(rotation[0,0]))
    yaw_error = abs(math.atan2(math.sin(yaw-layout["yaw_rad"]), math.cos(yaw-layout["yaw_rad"])))
    position_error = float(np.linalg.norm(np.asarray(pos[:2])-layout["xy_m"]))
    c = CRITERIA
    passed = (position_error <= c["position_tolerance_m"] and yaw_error <= c["yaw_tolerance_rad"]
              and linear <= c["linear_speed_max_m_s"] and angular <= c["angular_speed_max_rad_s"]
              and support > c["support_force_min_n"] and robot <= c["robot_force_max_n"])
    return {"target_present": True, "target_position_m": pos,
            "position_error_m": position_error, "yaw_error_rad": yaw_error,
            "linear_speed_m_s": linear, "angular_speed_rad_s": angular,
            "table_support_force_n": support, "robot_contact_force_n": robot,
            "contacts": contacts, "instantaneous_conditions_met": bool(passed)}


def observe(model, data, layout, history):
    """Independent physical rubric; no image/model output participates in truth."""
    measured = _measure(model,data,layout)
    count = 0
    for row in reversed(history):
        if not row["instantaneous_conditions_met"]:
            break
        count += 1
    # Current snapshot must itself satisfy the rubric, including alias velocity.
    dwell = count*float(model.opt.timestep) if measured["instantaneous_conditions_met"] else 0.
    truth = {"present": measured["target_present"],
             "complete": bool(measured["instantaneous_conditions_met"] and dwell+1e-12 >= CRITERIA["continuous_dwell_s"]),
             "physical_conditions_dwell_s": dwell,
             "position_error_m": measured["position_error_m"], "yaw_error_rad": measured["yaw_error_rad"],
             "supported": measured["table_support_force_n"] > CRITERIA["support_force_min_n"],
             "released": measured["robot_contact_force_n"] <= CRITERIA["robot_force_max_n"],
             "linear_speed_m_s": measured["linear_speed_m_s"], "angular_speed_rad_s": measured["angular_speed_rad_s"]}
    # No completion label or verdict leaks into the measured-input condition.
    inputs = {key: measured[key] for key in ("target_present", "target_position_m",
        "position_error_m", "yaw_error_rad", "linear_speed_m_s", "angular_speed_rad_s",
        "table_support_force_n", "robot_contact_force_n")}
    inputs["observation_window_s"] = len(history)*float(model.opt.timestep)
    inputs["native_timestep_s"] = float(model.opt.timestep)
    inputs["native_observation_count"] = len(history)
    inputs["target_present_observation_count"] = sum(row["target_present"] for row in history)
    inputs["native_window_extrema"] = {}
    for key in ("position_error_m", "yaw_error_rad", "linear_speed_m_s",
                "angular_speed_rad_s", "table_support_force_n", "robot_contact_force_n"):
        values = [row[key] for row in history if row[key] is not None]
        inputs["native_window_extrema"][key] = {"min": min(values) if values else None,
                                               "max": max(values) if values else None}
    return truth, inputs, measured


def _native_state(assets, layout, category, arm_qpos):
    mj, np = sim._native_imports()
    model = build_model(assets, layout, category)
    data = mj.MjData(model)
    data.qpos[:8] = arm_qpos
    data.ctrl[:6] = data.qpos[:6]
    data.ctrl[6] = .035
    mj.mj_forward(model,data)
    history = []
    def tick():
        mj.mj_step(model,data)
        mj.mj_forward(model,data)
        if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():
            raise RuntimeError("Native fixture emitted a solver warning or non-finite state")
        history.append(_measure(model,data,layout))
        del history[:-100]
    for _ in range(300):
        tick()
    if category == "unreleased":
        # A real approach and close through unchanged position actuators. The
        # shoe stays supported; this fixture tests incomplete release, not lift.
        yaw = layout["yaw_rad"]
        target = [layout["xy_m"][0]-.085*math.cos(yaw),
                  layout["xy_m"][1]-.085*math.sin(yaw), .055]
        stages = (([*target[:2], .20], .035, 1500),
                  (target, .035, 1000), (target, 0., 600))
        for stage_target, jaw, count in stages:
            for step in range(count):
                if step%10 == 0:
                    desired = ws.controller_action(model,data,stage_target,jaw)
                data.ctrl[:6] += np.clip(desired[:6]-data.ctrl[:6],-.003,.003)
                data.ctrl[6] += float(np.clip(desired[6]-data.ctrl[6],-.0001,.0001))
                tick()
    if category == "moving_alias":
        obj = mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,ws.SHOE)
        dof = int(model.jnt_dofadr[int(model.body_jntadr[obj])])
        data.qvel[dof:dof+3] = [.12,0,0]
        mj.mj_forward(model,data)
        # Instantaneous post-impulse reset fixture: no time has passed since the
        # velocity intervention. Preserve exact pose/pixels, invalidate dwell.
        history = []
    truth, inputs, measured = observe(model,data,layout,history)
    return model,data,truth,inputs,measured,history


def _png(rgb):
    """Lossless RGB PNG writer; avoids a mandatory image-library dependency."""
    h,w,_ = rgb.shape
    def chunk(kind,payload):
        return struct.pack(">I",len(payload))+kind+payload+struct.pack(">I",zlib.crc32(kind+payload)&0xffffffff)
    raw=b"".join(b"\0"+row.tobytes() for row in rgb)
    return (b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))
            +chunk(b"IDAT",zlib.compress(raw,6))+chunk(b"IEND",b""))


def render(model,data,layout):
    mj, np = sim._native_imports()
    camera = mj.MjvCamera()
    camera.lookat[:] = [.31,-.06,.08]
    camera.distance = 1.0
    camera.azimuth = layout["azimuth_deg"]
    camera.elevation = -55.
    renderer = mj.Renderer(model,HEIGHT,WIDTH)
    try:
        renderer.update_scene(data,camera=camera)
        rgb=renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data,camera=camera)
        segmentation=renderer.render().copy()
    finally:
        renderer.close()
    obj=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,ws.SHOE)
    target_geoms=np.flatnonzero(model.geom_bodyid==obj) if obj>=0 else np.asarray([],dtype=int)
    mask=np.isin(segmentation[:,:,0],target_geoms)&(segmentation[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM))
    ys,xs=np.nonzero(mask)
    bounds=None if not len(xs) else [float(xs.min()/WIDTH),float(ys.min()/HEIGHT),
                                   float((xs.max()+1)/WIDTH),float((ys.max()+1)/HEIGHT)]
    return _png(rgb), {"visible_target_pixels":int(mask.sum()), "target_box_xyxy_normalized":bounds,
                       "box_scope":"Visible target pixels only, from native geom-ID segmentation; not an amodal box."}


def _check_category(category,truth,measured,visibility):
    expected = category in ("complete","occluded")
    if truth["complete"] != expected:
        raise RuntimeError(f"Fixture {category} violates its declared completion label: {truth}")
    if category == "unreleased" and (truth["released"] or not truth["supported"]
            or truth["position_error_m"]>CRITERIA["position_tolerance_m"]):
        raise RuntimeError("Unreleased fixture must retain real robot contact at the supported goal")
    if category == "occluded" and visibility["visible_target_pixels"] != 0:
        raise RuntimeError("Occluded fixture leaks visible target pixels")
    if category not in ("missing_target","wrong_object","occluded") and not visibility["visible_target_pixels"]:
        raise RuntimeError("Expected visible target has no native segmentation pixels")


def expected_labels(category, truth):
    """Separate physical truth from conclusions supported by each input mode."""
    if category not in CATEGORIES:
        raise ValueError("Expected a frozen image-goal category")
    visible_violation = category in ("missing_target", "wrong_object",
                                    "wrong_position", "wrong_orientation")
    presence = ("unknown" if category == "occluded" else
                "present" if truth["present"] else "absent")
    return {
        "expected_verdict": {"rgb": "incomplete" if visible_violation else "unknown",
                             "rgb_telemetry": "complete" if truth["complete"] else "incomplete"},
        "expected_presence": {"rgb": presence,
                              "rgb_telemetry": "present" if truth["present"] else "absent"}}


def generate(assets,output):
    """Write fresh benchmark inputs/truth. Never contacts any model service."""
    destination=Path(output)
    if destination.exists():
        raise ValueError("Image-goal output must be a fresh directory")
    destination.mkdir(parents=True)
    frozen=_json(protocol())
    (destination/"protocol.json").write_bytes(frozen)
    (destination/"protocol.sha256").write_text(sha256(frozen).hexdigest()+"\n")
    assets_info=robot_profiles.verify_assets("piper",assets)
    arm=_initial_arm(assets)
    rows=[]; aliases=[]; start=time.perf_counter()
    (destination/"images").mkdir()
    (destination/"states").mkdir()
    for layout in LAYOUTS:
        goal_png=None
        for category in CATEGORIES:
            model,data,truth,inputs,measured,history=_native_state(assets,layout,category,arm)
            png,visibility=render(model,data,layout)
            _check_category(category,truth,measured,visibility)
            identifier=f"{layout['id']}-{category}"
            image_path=f"images/{identifier}.png"
            (destination/image_path).write_bytes(png)
            if category=="complete":
                goal_png=png
            if category=="moving_alias":
                if png!=goal_png:
                    raise RuntimeError("Matched-pixel velocity intervention changed the rendered image")
                aliases.append({"case_id":identifier,"pixel_identical_to":f"{layout['id']}-complete"})
            state={"time_s":float(data.time),"qpos":data.qpos.tolist(),"qvel":data.qvel.tolist(),
                   "ctrl":data.ctrl.tolist(),"physical_truth":truth,"current_measurement":measured,
                   "history":history,"solver_warnings":sum(int(w.number) for w in data.warning),
                   "object_attachment_constraints_added":0,"model_equalities":int(model.neq),
                   "external_forces_applied":False}
            state_path=f"states/{identifier}.json"
            (destination/state_path).write_bytes(_json(state))
            rows.append({"id":identifier,"category":category,"layout_id":layout["id"],
                "task":TASK,"current_image":image_path,"goal_image":f"images/{layout['id']}-complete.png",
                "physical_truth":truth,"measured_inputs":inputs,"visibility":visibility,
                "state_file":state_path, **expected_labels(category,truth)})
    manifest={"schema":SCHEMA,"task":TASK,"criteria":deepcopy(CRITERIA),"cases":rows,
              "aliases":aliases,"assets":assets_info,"protocol_sha256":sha256(frozen).hexdigest(),
              "mujoco_version":sim._native_imports()[0].__version__,
              "rendering":"Native MuJoCo RGB and geom segmentation; no image-generation model.",
              "generation_wall_seconds":time.perf_counter()-start,
              "scope":protocol()["scope"],"telemetry_scope":protocol()["telemetry_scope"]}
    (destination/"cases.json").write_bytes(_json(manifest))
    files=[]
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            content=path.read_bytes()
            files.append({"path":path.relative_to(destination).as_posix(),"bytes":len(content),"sha256":sha256(content).hexdigest()})
    (destination/"integrity.json").write_bytes(_json({"schema":SCHEMA+"-integrity","files":files}))
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    result=generate(args.assets,args.output)
    print(json.dumps({"schema":result["schema"],"cases":len(result["cases"]),
                      "protocol_sha256":result["protocol_sha256"],"output":args.output}))


if __name__=="__main__":
    main()
