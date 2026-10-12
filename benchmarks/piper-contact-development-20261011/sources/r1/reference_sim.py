"""Pinned hardware reference mechanics; no physical controller or skill policy.

Asset installation is an explicit network operation. All later work is local.
Imports of the optional native simulator are deferred until a model is requested.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import statistics
import stat
import time
from urllib.request import Request, urlopen


PROFILE = "ur5e-2f85-menagerie-reference-v0.1"
COMMIT = "0059d4335f8156206f63a35662313385f7ad6d74"
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
ARM_BODIES = ("shoulder_link", "upper_arm_link", "forearm_link",
              "wrist_1_link", "wrist_2_link", "wrist_3_link")
CONTROL_DT = .02
CASES = ("hold_open", "arm_sweep", "jaw_cycle")
SCOPE = ("Pinned reference geometry, inertias, collision proxies and coupled gripper. "
         "Authored native servos, privileged state and synthetic commands. "
         "No serial-specific calibration, measured hardware response, task policy or physical qualification.")


def asset_lock():
    return json.loads(Path(__file__).with_name("reference_assets.lock.json").read_text(encoding="utf-8"))


def _files(hands=False):
    return [x for x in asset_lock()["files"] if hands or not x["path"].startswith("shadow_hand/")]


def _regular_bytes(path, size):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Expected a regular non-symlink asset: {path.name}")
    if before.st_size != size:
        raise ValueError(f"Asset size mismatch: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != size:
            raise ValueError("Asset must remain a bounded regular file during opening")
        if before.st_ino and opened.st_ino and (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("Asset file identity changed during opening")
        content = stream.read(size + 1)
    if len(content) != size:
        raise ValueError(f"Asset size mismatch: {path.name}")
    return content


def _asset_payload(directory, *, hands=False):
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Asset root must be an existing non-symlink directory")
    records = _files(hands)
    blobs = {}
    for record in records:
        path = root / record["path"]
        # Reject redirected subdirectories before allowing the native parser access.
        if any(x.is_symlink() for x in (path.parent, path.parent.parent)):
            raise ValueError("Asset subdirectories must not be symlinks")
        content = _regular_bytes(path, record["bytes"])
        if sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Asset digest mismatch: {record['path']}")
        blobs[record["path"]] = content
    metadata = {"commit": COMMIT, "files_verified": len(records),
            "bytes_verified": sum(x["bytes"] for x in records),
            "lock_sha256": sha256(Path(__file__).with_name("reference_assets.lock.json").read_bytes()).hexdigest(),
            "licenses": {k: v for k, v in asset_lock()["licenses"].items()
                         if hands or k != "shadow_hand"}}
    return metadata, blobs


def verify_assets(directory, *, hands=False):
    return _asset_payload(directory, hands=hands)[0]


def _spec(blobs, folder, filename):
    mj, _ = _native_imports()
    prefix = folder + "/"
    local = {key[len(prefix):]: content for key, content in blobs.items() if key.startswith(prefix)}
    # The native parser receives the exact verified bytes in memory. It does not
    # reread model/mesh/include paths between verification and compilation.
    return mj.MjSpec.from_string(local[filename].decode("utf-8"),
                                 include={key: content for key, content in local.items() if key.endswith(".xml")},
                                 assets={key: content for key, content in local.items() if key.startswith("assets/")})


def fetch_assets(directory, *, hands=False):
    """Download only the packaged, pinned allowlist, retaining upstream licenses.

    A failed download leaves an incomplete fresh directory for inspection; verify
    rejects it. No credentials, account, dynamic manifest or arbitrary URL is used.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    for record in _files(hands):
        url = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/{record['path']}"
        request = Request(url, headers={"User-Agent": "DVIDIA-Reference-Simulator/0.1"})
        with urlopen(request, timeout=30) as response:
            if not response.geturl().startswith("https://raw.githubusercontent.com/"):
                raise ValueError("Asset download redirected outside the pinned source host")
            content = response.read(record["bytes"] + 1)
        if len(content) != record["bytes"] or sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Downloaded asset does not match the lock: {record['path']}")
        path = destination / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    return verify_assets(destination, hands=hands)


def _native_imports():
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install dvidia-training with the optional arm extra") from exc
    if mujoco.__version__ != "3.15.0":
        raise RuntimeError("Reference simulation requires MuJoCo 3.15.0")
    return mujoco, np


@contextmanager
def native_warnings(messages):
    from .env import _NATIVE_LOCK
    mj, _ = _native_imports()
    with _NATIVE_LOCK:
        previous = mj.get_mju_user_warning()
        mj.set_mju_user_warning(lambda value: messages.append(str(value)))
        try:
            yield
        finally:
            mj.set_mju_user_warning(previous)


def _timestep(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (.002, .001, .0005):
        raise ValueError("Reference timestep must be 0.002, 0.001 or 0.0005 seconds")
    return float(value)


def validate_duration(seconds):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not .1 <= seconds <= 10:
        raise ValueError("Reference duration must be finite, between 0.1 and 10 seconds")
    if not math.isclose(seconds / CONTROL_DT, round(seconds / CONTROL_DT), rel_tol=0, abs_tol=1e-9):
        raise ValueError("Duration must be an integer number of 20 ms boundaries")
    return float(seconds)


def build_model(directory, *, timestep=.001, probe_position=None):
    _, blobs = _asset_payload(directory)
    mj, _ = _native_imports()
    arm = _spec(blobs, "universal_robots_ur5e", "ur5e.xml")
    jaw = _spec(blobs, "robotiq_2f85", "2f85.xml")
    # Resolve child/parent option differences explicitly, preserving the jaw's
    # elliptic contact cone and impratio; do not silently accept attach warnings.
    arm.option.cone = jaw.option.cone
    arm.option.impratio = jaw.option.impratio
    arm.option.timestep = jaw.option.timestep = _timestep(timestep)
    arm.option.integrator = jaw.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    arm.attach(jaw, prefix="gripper/", site="attachment_site")
    arm.worldbody.add_geom(name="reference_floor", type=mj.mjtGeom.mjGEOM_PLANE,
                          size=[2, 2, .1], rgba=[.9, .9, .88, 1])
    if probe_position is not None:
        if len(probe_position) != 3 or not all(math.isfinite(x) for x in probe_position):
            raise ValueError("Probe requires three finite coordinates")
        arm.worldbody.add_geom(name="collision_probe", type=mj.mjtGeom.mjGEOM_SPHERE,
                              pos=probe_position, size=[.025, 0, 0], rgba=[1, .4, .2, 1])
    return arm.compile()


def model_digest(model):
    mj, np = _native_imports()
    buffer = np.empty(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, buffer=buffer)
    return sha256(buffer.tobytes()).hexdigest()


def _name(model, kind, index):
    mj, _ = _native_imports()
    return mj.mj_id2name(model, kind, int(index)) or f"unnamed_{index}"


def model_inventory(model):
    mj, np = _native_imports()
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    bodies = []
    for i in range(1, model.nbody):
        geoms = np.flatnonzero(model.geom_bodyid == i)
        bodies.append({"name": _name(model, mj.mjtObj.mjOBJ_BODY, i),
                       "mass_kg": float(model.body_mass[i]),
                       "inertia_kg_m2": model.body_inertia[i].tolist(),
                       "collision_geoms": int(np.count_nonzero(active[geoms])),
                       "visual_geoms": int(np.count_nonzero(~active[geoms]))})
    joints = [{"name": _name(model, mj.mjtObj.mjOBJ_JOINT, i),
               "type": int(model.jnt_type[i]), "limited": bool(model.jnt_limited[i]),
               "range": model.jnt_range[i].tolist(),
               "qpos_address": int(model.jnt_qposadr[i]), "velocity_address": int(model.jnt_dofadr[i])}
              for i in range(model.njnt)]
    actuators = []
    for i in range(model.nu):
        trn = int(model.actuator_trntype[i])
        kind = mj.mjtObj.mjOBJ_TENDON if trn == int(mj.mjtTrn.mjTRN_TENDON) else mj.mjtObj.mjOBJ_JOINT
        actuators.append({"name": _name(model, mj.mjtObj.mjOBJ_ACTUATOR, i),
                          "transmission_type": trn,
                          "target": _name(model, kind, model.actuator_trnid[i, 0]),
                          "control_range": model.actuator_ctrlrange[i].tolist(),
                          "control_limited": bool(model.actuator_ctrllimited[i]),
                          "native_force_range": model.actuator_forcerange[i].tolist(),
                          "force_limited": bool(model.actuator_forcelimited[i])})
    return {"joint_coordinates_nq": model.nq, "velocity_dofs_nv": model.nv,
            "joints": joints, "actuator_count": model.nu, "actuators": actuators,
            "bodies": bodies, "equality_constraints": model.neq, "tendons": model.ntendon,
            "contact_exclusions": model.nexclude, "native_sensor_definitions": model.nsensor,
            "native_sensor_channels": model.nsensordata,
            "collision_geom_count": int(np.count_nonzero(active)),
            "visual_geom_count": int(np.count_nonzero(~active)),
            "compiled_model_sha256": model_digest(model)}


def audit_reference(directory, *, hands=False):
    provenance, blobs = _asset_payload(directory, hands=hands)
    mj, _ = _native_imports()
    warnings = []
    with native_warnings(warnings):
        model = build_model(directory)
        inventory = model_inventory(model)
        if model.nq != 14 or model.nu != 7 or model.neq != 3 or model.ntendon != 1:
            raise ValueError("Pinned reference composition changed its mechanical contract")
        by_name = {x["name"]: x for x in inventory["bodies"]}
        coverage = {name: by_name[name]["collision_geoms"] for name in ARM_BODIES}
        hand_models = {}
        if hands:
            for side in ("left", "right"):
                hand = _spec(blobs, "shadow_hand", f"{side}_hand.xml").compile()
                hand_models[side] = model_inventory(hand)
    return {"schema": "dvidia.reference-audit/v1", "profile": PROFILE, "scope": SCOPE,
            "physical_robot_ready": False, "asset_provenance": provenance,
            "settings": {"timestep_seconds": .001, "integrator": "implicitfast",
                         "contact_cone": "elliptic", "impratio": 10,
                         "mount": "upstream attachment_site, gripper namespace, no added adapter mass",
                         "tcp": "upstream gripper/pinch; not a measured physical TCP"},
            "inventory": inventory, "arm_link_collision_coverage": coverage,
            "fixed_base_collision_geoms": by_name["base"]["collision_geoms"],
            "arm_command_unit": "joint target radians",
            "gripper_command_unit": "native 0–255 servo input; not meters or calibrated physical force",
            "hand_models": hand_models, "bimanual_assembly_implemented": False,
            "calibration_gates": {name: "unmeasured" for name in
                ("serial_kinematics", "base_and_tcp_frames", "actuator_response_and_latency",
                 "gripper_encoder_gap_hysteresis", "contact_and_slip", "deployed_sensor_age",
                 "physical_task_success")}, "warnings": warnings}


def _initial(model):
    mj, _ = _native_imports()
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)
    return data


def _sample(model, data, elapsed, command):
    mj, np = _native_imports()
    ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
    site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "gripper/pinch")
    pads = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"gripper/{side}_pad1") for side in ("left", "right")]
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, data.site_xmat[site])
    return {"t_seconds": elapsed, "joint_position_rad": data.qpos[model.jnt_qposadr[ids]].tolist(),
            "joint_velocity_rad_s": data.qvel[model.jnt_dofadr[ids]].tolist(),
            "linkage_position_rad": data.qpos[6:].tolist(),
            "tcp_position_m": data.site_xpos[site].tolist(), "tcp_quaternion_wxyz": quat.tolist(),
            "pad_center_separation_m": float(np.linalg.norm(data.geom_xpos[pads[0]] - data.geom_xpos[pads[1]])),
            "arm_points_m": [data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)].tolist()
                             for name in ("base",) + ARM_BODIES] + [data.site_xpos[site].tolist()],
            "pad_centers_m": [data.geom_xpos[index].tolist() for index in pads],
            "contact_count": data.ncon,
            "issued_joint_targets_rad": command[:6].tolist(), "issued_gripper_0_255": float(command[6])}


def _summary(values):
    ordered = sorted(values)
    return {"count": len(values), "median": statistics.median(ordered),
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1], "max": ordered[-1]}


def _command(case, elapsed, home):
    _, np = _native_imports()
    result = np.array(home, dtype=float, copy=True)
    if case == "arm_sweep":
        result[0] += .20 * math.sin(math.pi * elapsed)
    elif case == "jaw_cycle":
        result[6] = 127.5 * (1 - math.cos(math.pi * elapsed))
    elif case != "hold_open":
        raise ValueError("Unknown reference motion case")
    return result


def run_case(directory, case, *, timestep=.001, seconds=4.):
    mj, np = _native_imports()
    timestep = _timestep(timestep)
    seconds = validate_duration(seconds)
    if case not in CASES:
        raise ValueError("Unknown reference motion case")
    warnings = []
    ticks, chunks, trajectory = [], [], []
    with native_warnings(warnings):
        started = time.perf_counter()
        model = build_model(directory, timestep=timestep)
        compile_seconds = time.perf_counter() - started
        data = _initial(model)
        home = model.key_ctrl[0].copy()
        # Settling is outside the reported physics throughput/tick timing.
        for _ in range(round(1 / timestep)):
            data.ctrl[:] = home
            mj.mj_step(model, data)
        mj.mj_forward(model, data)
        trajectory.append(_sample(model, data, 0., home))
        physics_wall = 0.
        for boundary in range(round(seconds / CONTROL_DT)):
            elapsed = boundary * CONTROL_DT
            command = _command(case, elapsed, home)
            data.ctrl[:] = command
            start = time.perf_counter_ns()
            for _ in range(round(CONTROL_DT / timestep)):
                before = time.perf_counter_ns()
                mj.mj_step(model, data)
                ticks.append((time.perf_counter_ns() - before) / 1e6)
            mj.mj_forward(model, data)
            chunk_ms = (time.perf_counter_ns() - start) / 1e6
            physics_wall += chunk_ms / 1000
            chunks.append(chunk_ms)
            trajectory.append(_sample(model, data, (boundary + 1) * CONTROL_DT, command))
        valid = bool(not warnings and not any(x.number for x in data.warning)
                     and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                     and np.isfinite(data.qacc).all() and np.isfinite(data.site_xpos).all()
                     and np.isfinite(data.actuator_force).all()
                     and all(math.isfinite(v) for row in trajectory for k in
                             ("joint_position_rad", "joint_velocity_rad_s", "tcp_position_m", "linkage_position_rad") for v in row[k]))
    nonfinite_trajectory_omitted = False
    try:
        json.dumps(trajectory, allow_nan=False)
    except ValueError:
        trajectory = []
        valid = False
        nonfinite_trajectory_omitted = True
    return {"case": case, "valid": valid, "warnings": warnings,
            "nonfinite_trajectory_omitted": nonfinite_trajectory_omitted,
            "compiled_model_sha256": model_digest(model), "timestep_seconds": timestep,
            "control_dt_seconds": CONTROL_DT, "simulated_seconds": seconds,
            "compile_and_asset_verification_seconds": compile_seconds,
            "physics_chunks_wall_seconds": physics_wall,
            "simulation_seconds_per_wall_second": seconds / physics_wall,
            "native_tick_ms": _summary(ticks), "control_chunk_ms": _summary(chunks),
            "timing_scope": "mj_step calls with per-tick instrumentation; chunks also include boundary mj_forward. "
                            "Excludes verification/compile, settling, sampling, rendering, perception, policy, transport and real actuation.",
            "trajectory": trajectory}


def collision_probe(directory):
    """Matched overlap diagnostic, not a path planner or collision safety test."""
    mj, np = _native_imports()
    warnings = []
    attempts = []
    with native_warnings(warnings):
        model = build_model(directory)
        data = _initial(model)
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "upper_arm_link")
        active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        geom = int(np.flatnonzero((model.geom_bodyid == body) & active)[-1])
        position = data.geom_xpos[geom].copy()
        for mask in (False, True):
            model = build_model(directory, probe_position=position.tolist())
            if mask:
                bodies = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in ARM_BODIES]
                for index in np.flatnonzero(np.isin(model.geom_bodyid, bodies)):
                    model.geom_contype[index] = model.geom_conaffinity[index] = 0
            data = _initial(model)
            probe = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "collision_probe")
            pairs = []
            for contact in data.contact[:data.ncon]:
                if probe in (contact.geom1, contact.geom2):
                    other = contact.geom2 if contact.geom1 == probe else contact.geom1
                    pairs.append({"other_body": _name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[other]),
                                  "signed_distance_m": float(contact.dist)})
            attempts.append({"arm_collision_masks_disabled": mask, "probe_contact_count": len(pairs), "pairs": pairs})
    return {"scope": "Authored initial sphere overlap at an upper-arm collision proxy; no motion or real collision trial",
            "probe_position_m": position.tolist(), "attempts": attempts, "warnings": warnings}


def benchmark_reference(directory, *, repeats=2, seconds=4., hands=False):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 5:
        raise ValueError("Repeats must be an integer between 1 and 5")
    seconds = validate_duration(seconds)
    audit = audit_reference(directory, hands=hands)
    attempts = []
    for case in CASES:
        for timestep in (.002, .001, .0005):
            for repeat in range(repeats):
                try:
                    attempt = run_case(directory, case, timestep=timestep, seconds=seconds)
                except Exception as exc:
                    attempt = {"case": case, "timestep_seconds": timestep, "valid": False,
                               "error_type": type(exc).__name__,
                               "error": str(exc).replace(str(Path(directory).resolve()), "[assets]")[:500]}
                attempt["repeat"] = repeat
                attempts.append(attempt)
    convergence = []
    _, np = _native_imports()
    for case in CASES:
        for coarse in (.002, .001):
            for repeat in range(repeats):
                pair = [next(x for x in attempts if x["case"] == case and x["timestep_seconds"] == dt and x["repeat"] == repeat)
                        for dt in (coarse, coarse / 2)]
                result = {"case": case, "repeat": repeat, "timestep_pair_seconds": [coarse, coarse / 2],
                          "valid": all(x["valid"] for x in pair)}
                if result["valid"]:
                    rows = [x["trajectory"] for x in pair]
                    def differences(key):
                        return np.asarray([x[key] for x in rows[0]]) - np.asarray([x[key] for x in rows[1]])
                    dq = differences("joint_position_rad")
                    dx = differences("tcp_position_m")
                    dg = differences("pad_center_separation_m")
                    result.update({"matched_boundaries": len(rows[0]), "joint_position_rmse_rad": float(np.sqrt(np.mean(dq ** 2))),
                                   "max_joint_position_difference_rad": float(np.max(np.abs(dq))),
                                   "tcp_distance_rmse_m": float(np.sqrt(np.mean(np.sum(dx ** 2, axis=1)))),
                                   "max_tcp_distance_m": float(np.max(np.linalg.norm(dx, axis=1))),
                                   "pad_center_separation_rmse_m": float(np.sqrt(np.mean(dg ** 2))),
                                   "max_pad_center_separation_difference_m": float(np.max(np.abs(dg)))})
                convergence.append(result)
    source_files = ("reference_sim.py", "reference_assets.lock.json", "embodiment_cli.py", "reference_report.html")
    return {"schema": "dvidia.reference-benchmark/v1", "profile": PROFILE, "scope": SCOPE,
            "physical_robot_ready": False, "task_success_evaluated": False,
            "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                         "python": platform.python_version(), "mujoco": "3.15.0", "gpu_used": False},
            "protocol": {"cases": list(CASES), "timesteps_seconds": [.002, .001, .0005],
                         "repeats": repeats, "seconds_per_case": seconds, "control_dt_seconds": CONTROL_DT,
                         "same_synthetic_scene_and_commands": True,
                         "repeat_scope": "Repeated deterministic numerical/performance diagnostics, not independent physical or skill trials",
                         "pad_measurement": "Euclidean distance between upstream left_pad1/right_pad1 geom centers; not calibrated inner jaw gap",
                         "convergence_scope": "Matched 20 ms boundaries including post-settling start; adjacent timestep comparison, not comparison with reality"},
            "implementation_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_files},
            "audit": audit, "collision_probe": collision_probe(directory),
            "attempted": len(attempts), "valid_attempts": sum(x["valid"] for x in attempts),
            "attempts": attempts, "convergence": convergence}
