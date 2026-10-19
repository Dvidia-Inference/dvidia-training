"""Fresh native scenes with independent RGB, drive and primary observations.

These are authored state/intervention fixtures, not learned robot policies.
Only the independent oracle reads object state and contact forces. Camera and
drive adapters receive their own native channels; no truth is passed to them.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from . import image_goal_scenes as frozen
from . import local_goal_scenes as previous
from . import local_goal_verifier as verifier
from . import sensor_goal_observers as sensors
from . import workstation as ws, robot_profiles

SCHEMA = "dvidia.sensor-native-scenes/v1"
FINAL_SEED = 2026102007
DEVELOPMENT_SEED = 2026101904
SCENARIOS = ("complete", "occluded", "wrong_position", "wrong_orientation",
             "missing_target", "wrong_object", "unreleased", "sustained_motion",
             "instant_alias", "yaw_flip180", "lookalike_wrongobject", "elevated_support")
CAMERA = sensors.CameraCalibration()
DRIVE = sensors.ActuatorCalibration()
INTERVENTIONS = {
    "complete": "None after the declared authored reset and 600 ms settling.",
    "occluded": "Fixed opaque inspection cover hides the otherwise settled target.",
    "wrong_position": "Authored initial target position is displaced 170 mm along world X.",
    "wrong_orientation": "Authored initial target yaw is rotated 90 degrees.",
    "missing_target": "The target body is absent from the model.",
    "wrong_object": "An orange decoy replaces the target body.",
    "unreleased": "Unchanged authored PiPER approach and close; then another 200 ms of native observation.",
    "sustained_motion": "At window start set target free velocity to +0.65 m/s world X and both target/table sliding friction to 0.02; advance native physics for 200 ms.",
    "instant_alias": "At the final timestamp reset target free velocity to +0.12 m/s world X without advancing time; replace the final measurement, never append a duplicate timestamp.",
    "yaw_flip180": "Authored initial target yaw is rotated 180 degrees; axial color-shape vision cannot authenticate toe direction.",
    "lookalike_wrongobject": "A decoy body has exactly the green target appearance; the semantic target body is absent.",
    "elevated_support": "A fixed colliding platform supports the shoe 120 mm above the workstation table; no object attachment or compensating force.",
}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+"\n")


def _write_image(root, rgb):
    data = frozen._png(rgb)
    name = "images/"+hashlib.sha256(data).hexdigest()+".png"
    path = root/name
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != data:
            raise RuntimeError("Content-addressed image collision")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream: stream.write(data)
    return name


def _elevated_model(assets, layout):
    """Same narrow rigid proxy on a real static collision platform."""
    mj, _ = previous.sim._native_imports()
    spec = ws._spec(assets)
    shoe = spec.body(ws.SHOE)
    yaw = layout["yaw_rad"]
    shoe.pos = [*layout["xy_m"], .132]
    shoe.quat = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    for geom in shoe.geoms:
        geom.pos[1] *= .5; geom.size[1] *= .5; geom.mass *= .1
        geom.rgba = [.12, .43, .27, 1.] if "upper" in geom.name else [.84, .84, .76, 1.]
    spec.worldbody.add_geom(name="elevated_support_platform", type=mj.mjtGeom.mjGEOM_BOX,
        pos=[*layout["xy_m"], .06], size=[.18, .07, .06],
        quat=[math.cos(yaw/2), 0, 0, math.sin(yaw/2)],
        rgba=[.50, .53, .50, 1.], friction=[.8, .005, .0001], contype=1, conaffinity=1)
    # Visual-only goal outline, identical dimensions and table plane to v0.3.
    for i, (dx, dy, sx, sy) in enumerate(((0, -.058, .16, .002), (0, .058, .16, .002),
                                         (-.16, 0, .002, .06), (.16, 0, .002, .06))):
        x = layout["xy_m"][0]+dx*math.cos(yaw)-dy*math.sin(yaw)
        y = layout["xy_m"][1]+dx*math.sin(yaw)+dy*math.cos(yaw)
        spec.worldbody.add_geom(name=f"goal_outline_{i}", type=mj.mjtGeom.mjGEOM_BOX,
            pos=[x, y, .0006], size=[sx, sy, .0004],
            quat=[math.cos(yaw/2), 0, 0, math.sin(yaw/2)],
            rgba=[.10, .27, .23, 1.], contype=0, conaffinity=0)
    spec.visual.global_.offwidth = CAMERA.width; spec.visual.global_.offheight = CAMERA.height
    spec.visual.headlight.ambient = [.4, .4, .4]; spec.visual.headlight.diffuse = [.6, .6, .6]
    return spec.compile()


def build_sensor_model(assets, layout, scenario):
    """Rebuild a declared native model for independent replay and inspection."""
    previous._validate_layouts([layout])
    if scenario not in SCENARIOS: raise ValueError("Unknown sensor scenario")
    mj, _ = previous.sim._native_imports()
    if scenario == "elevated_support": return _elevated_model(assets, layout)
    base = "wrong_object" if scenario == "lookalike_wrongobject" else scenario
    if base not in frozen.CATEGORIES: base = "complete"
    model = previous.build_fresh_model(assets, layout, base)
    if scenario == "lookalike_wrongobject":
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "orange_decoy_shoe")
        for geom in range(model.ngeom):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom) or ""
            if model.geom_bodyid[geom] == body and "upper" in name:
                model.geom_rgba[geom] = [.12, .43, .27, 1.]
    if scenario == "yaw_flip180":
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ws.SHOE)
        address = int(model.jnt_qposadr[int(model.body_jntadr[body])])
        yaw = layout["yaw_rad"]+math.pi
        model.qpos0[address+3:address+7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        mj.mj_setConst(model, mj.MjData(model))
    if scenario == "sustained_motion":
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ws.SHOE)
        table = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "workstation_table_top")
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] == body or geom == table:
                model.geom_friction[geom, 0] = .02
    return model


def _valid_step(model, data):
    mj, np = previous.sim._native_imports()
    mj.mj_step(model, data); mj.mj_forward(model, data)
    if any(w.number for w in data.warning) or not all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc)):
        raise RuntimeError("Native warning or non-finite state")


def _start(assets, layout, scenario, arm):
    mj, _ = previous.sim._native_imports()
    if scenario == "unreleased":
        model, data, *_ = previous._run(assets, layout, "unreleased", arm)
        return model, data, 3400
    model = build_sensor_model(assets, layout, scenario)
    data = mj.MjData(model); data.qpos[:8] = arm
    data.ctrl[:6] = data.qpos[:6]; data.ctrl[6] = .035
    mj.mj_forward(model, data)
    for _ in range(300): _valid_step(model, data)
    return model, data, 300


def _set_object_velocity(model, data, speed):
    mj, _ = previous.sim._native_imports()
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ws.SHOE)
    if body < 0: raise RuntimeError("Velocity intervention requires the target")
    address = int(model.jnt_dofadr[int(model.body_jntadr[body])])
    data.qvel[address:address+3] = [speed, 0., 0.]
    mj.mj_forward(model, data)


def _capture_state(data):
    return {"t_s": float(data.time), "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
            "ctrl": data.ctrl.tolist(), "actuator_force": data.actuator_force.tolist()}


def _platform_support(model, data):
    """Diagnostic oracle only: actual platform contact normal force."""
    mj, np = previous.sim._native_imports()
    platform = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "elevated_support_platform")
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ws.SHOE)
    total = 0.
    if platform < 0 or body < 0: return total
    for i in range(data.ncon):
        contact = data.contact[i]
        a, b = int(contact.geom1), int(contact.geom2)
        if platform in (a, b) and body in (int(model.geom_bodyid[a]), int(model.geom_bodyid[b])):
            force = np.zeros(6); mj.mj_contactForce(model, data, i, force)
            total += max(0., float(force[0]))
    return total


def _run_case(root, assets, layout, scenario, arm):
    model, data, steps = _start(assets, layout, scenario, arm)
    if scenario == "sustained_motion": _set_object_velocity(model, data, .65)
    start = float(data.time)
    frames = []; captures = []; history = []; drive_samples = []
    def capture():
        rgb = sensors.render_overhead_rgb(model, data, calibration=CAMERA)
        frames.append({"t_s": float(data.time), "image": _write_image(root, rgb)})
        captures.append(_capture_state(data))
    capture()
    for tick in range(1, 101):
        _valid_step(model, data); steps += 1
        history.append(previous._measurement(model, data, layout))
        drive_samples.append(sensors.read_native_actuators(model, data))
        if tick == 100 and scenario == "instant_alias":
            _set_object_velocity(model, data, .12)
            history[-1] = previous._measurement(model, data, layout)
            drive_samples[-1] = sensors.read_native_actuators(model, data)
        if tick % 10 == 0: capture()
    truth, _, current = frozen.observe(model, data, layout, history)
    current = previous._measurement(model, data, layout)
    errors = []
    expected_complete = scenario in ("complete", "occluded")
    if truth["complete"] != expected_complete:
        errors.append("Native physical rubric does not match declared scenario completion")
    if scenario == "unreleased" and (truth["released"] or not truth["supported"]):
        errors.append("Unreleased scene lost real supported robot contact")
    if scenario == "yaw_flip180" and truth["yaw_error_rad"] < 3.:
        errors.append("Half-turn scene lost the intended heading error")
    if scenario == "lookalike_wrongobject" and truth["present"]:
        errors.append("Lookalike control accidentally includes the semantic target")
    platform_force = _platform_support(model, data)
    if scenario == "elevated_support" and (platform_force <= .01 or truth["supported"] or current["target_position_m"][2] < .10):
        errors.append("Elevated scene is not supported by the declared physical platform")
    displacement = None
    if scenario == "sustained_motion":
        start_state = captures[0]
        mj, _ = previous.sim._native_imports()
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ws.SHOE)
        address = int(model.jnt_qposadr[int(model.body_jntadr[body])])
        displacement = math.dist(start_state["qpos"][address:address+2], data.qpos[address:address+2])
        if displacement < .09 or truth["linear_speed_m_s"] <= .01:
            errors.append("Motion scene did not sustain declared observable displacement and motion")
    camera = {"schema": "dvidia.sensor-camera-window/v1", "source_id": "overhead-rgb",
              "calibration_id": CAMERA.calibration_id, "frames": frames}
    drive = {"schema": "dvidia.sensor-drive-window/v1", "source_id": "piper-drive-bus-v1",
             "calibration_id": DRIVE.calibration_id, "history_start_s": start, "samples": drive_samples}
    state = {"schema": SCHEMA, "layout": layout, "scenario": scenario,
             **_capture_state(data), "current_measurement": current, "history": history,
             "camera_capture_states": captures, "physical_truth": truth,
             "physics_timestep_s": float(model.opt.timestep), "native_steps": steps,
             "prescribed_intervention": INTERVENTIONS[scenario], "solver_warnings": sum(int(w.number) for w in data.warning),
             "object_attachment_constraints_added": 0, "model_equalities": int(model.neq),
             "external_forces_applied": False, "object_gravity_compensation_added": False,
             "platform_support_force_n": platform_force, "native_window_displacement_m": displacement,
             "generation_errors": errors}
    return previous._packet(history), drive, camera, state, truth, errors


def generate_native_cases(root, assets, seed=FINAL_SEED, count=8):
    """Generate every planned native row, retaining explicit failures.

    RGB frames are eleven endpoint observations at 20 ms intervals. Primary
    and drive data are 100 interval-ending native ticks at 2 ms. The same
    elapsed 200 ms window ends at the decision timestamp for every channel.
    """
    root = Path(root); assets = Path(assets)
    if root.is_symlink(): raise ValueError("Output cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    marker = root/"sensor-native-scenes.json"
    if marker.exists(): raise ValueError("Native scene generation already exists")
    for name in ("images", "packets", "states"):
        if (root/name).is_symlink(): raise ValueError("Output child cannot be a symlink")
        (root/name).mkdir(exist_ok=True)
    layouts = previous.make_layouts(seed, count)
    for layout in layouts:
        for scenario in SCENARIOS:
            name = f"native-{layout['id']}-{scenario}"
            for relative in (f"packets/{name}-primary.json", f"packets/{name}-actuators.json", f"packets/{name}-camera.json", f"states/{name}.json"):
                if (root/relative).exists(): raise ValueError("Native output exists: "+relative)
    started = time.perf_counter()
    asset_metadata = robot_profiles.verify_assets("piper", assets)
    arm = frozen._initial_arm(assets)
    rows = []
    for layout in layouts:
        goal_image = None; complete_frames = None; complete_drives = None
        for scenario in SCENARIOS:
            identifier = f"native-{layout['id']}-{scenario}"
            contract = asdict(verifier.GoalContract(ws.SHOE, previous.FRAME, tuple(layout["xy_m"]), layout["yaw_rad"]))
            row = {"id": identifier, "cohort": "native", "scenario": scenario, "layout_id": layout["id"],
                   "contract": contract, "now_s": None, "primary_packet_file": None,
                   "actuator_packet_file": None, "camera_packet_file": None,
                   "state_file": f"states/{identifier}.json", "current_image": None, "goal_image": goal_image,
                   "physical_truth": None, "generation_valid": False, "generation_errors": [],
                   "note": INTERVENTIONS[scenario], "prescribed_intervention": INTERVENTIONS[scenario]}
            try:
                primary, drive, camera, state, truth, errors = _run_case(root, assets, layout, scenario, arm)
                if scenario == "complete":
                    goal_image = camera["frames"][-1]["image"]
                    complete_frames = camera["frames"]; complete_drives = drive
                if scenario == "instant_alias":
                    alias = {"rgb_sequence_identical": camera["frames"] == complete_frames,
                             "drive_sequence_identical": drive == complete_drives,
                             "paired_case_id": f"native-{layout['id']}-complete"}
                    state["alias"] = alias
                    if not all(alias[k] for k in ("rgb_sequence_identical", "drive_sequence_identical")):
                        errors.append("Instantaneous velocity control did not preserve its sensor aliases")
                primary_file = f"packets/{identifier}-primary.json"
                actuator_file = f"packets/{identifier}-actuators.json"
                camera_file = f"packets/{identifier}-camera.json"
                for path, value in ((primary_file, primary), (actuator_file, drive), (camera_file, camera), (row["state_file"], state)):
                    _write(root/path, value)
                row.update(now_s=state["t_s"], primary_packet_file=primary_file, actuator_packet_file=actuator_file,
                           camera_packet_file=camera_file, current_image=camera["frames"][-1]["image"],
                           goal_image=goal_image, physical_truth=truth, generation_valid=not errors,
                           generation_errors=errors)
            except Exception as error:
                row["generation_errors"] = [f"{type(error).__name__}: {error}"]
                if not (root/row["state_file"]).exists():
                    _write(root/row["state_file"], {"schema": SCHEMA, "layout": layout, "scenario": scenario,
                        "generation_valid": False, "generation_errors": row["generation_errors"]})
            rows.append(row)
    _write(marker, {"schema": SCHEMA, "seed": seed, "layouts": layouts, "scenarios": list(SCENARIOS),
                    "planned_count": len(layouts)*len(SCENARIOS), "case_count": len(rows),
                    "invalid_count": sum(not r["generation_valid"] for r in rows), "assets": asset_metadata,
                    "camera_calibration": asdict(CAMERA), "actuator_calibration": asdict(DRIVE),
                    "generation_wall_seconds": time.perf_counter()-started,
                    "scope": "Authored native verification fixtures; no learned policy, no robot hardware, no semantic identity authentication.",
                    "cases": rows})
    return rows
