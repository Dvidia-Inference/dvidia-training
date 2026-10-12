"""Offline, pinned native robot scenes with original synthetic rigid shoe assets.

The packaged catalog describes assumptions, not measured shoe properties. Loading
assets supplies geometry and mechanics; it does not install a manipulation policy.
There is no arbitrary XML, executable extension, URL, or user-supplied asset parser.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from . import reference_sim as sim
from . import robot_profiles


CATALOG_PATH = Path(__file__).with_name("asset_catalog.json")
SCENE_SCHEMA = "dvidia.asset-scene/v1"
DEFAULT_SHOES = ("trainer-left", "trainer-right")
_MODEL_PREFIX = "dvidia_asset_library_"


def catalog() -> dict:
    """Read the packaged catalog without importing native or network packages.

    The digest identifies the exact installed catalog bytes. Returned dictionaries
    are independent copies, so callers cannot change later scenes by editing one.
    """
    content = CATALOG_PATH.read_bytes()
    result = json.loads(content)
    result["catalog_sha256"] = sha256(content).hexdigest()
    return result


def _selection(robot, shoe_ids, timestep):
    collection = catalog()
    if not isinstance(robot, str) or robot not in {x["id"] for x in collection["robots"]}:
        raise ValueError("robot must be ur5e, piper or so101")
    identifiers = DEFAULT_SHOES if shoe_ids is None else shoe_ids
    if not isinstance(identifiers, (list, tuple)) or not 1 <= len(identifiers) <= 6:
        raise ValueError("shoe_ids must contain between one and six packaged shoe IDs")
    allowed = {x["id"] for x in collection["shoes"]}
    if any(not isinstance(value, str) or value not in allowed for value in identifiers):
        raise ValueError("shoe_ids contains an unknown packaged shoe ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("shoe_ids must not contain duplicates")
    return collection, tuple(identifiers), sim._timestep(timestep)


def scene_config(robot: str, shoe_ids=None, timestep=.002) -> dict:
    """Create a portable, exact-catalog scene descriptor; no paths are embedded."""
    collection, identifiers, dt = _selection(robot, shoe_ids, timestep)
    return {"schema": SCENE_SCHEMA, "catalog_version": collection["version"],
            "catalog_sha256": collection["catalog_sha256"], "robot": robot,
            "shoe_ids": list(identifiers), "timestep": dt}


def load_scene(config: dict, assets):
    """Load a strict scene descriptor against locally installed pinned assets.

    Unknown fields, incompatible catalog digests and noncanonical values fail
    before reading native assets. Properties cannot be overridden by descriptors.
    """
    if not isinstance(config, dict) or set(config) != {
            "schema", "catalog_version", "catalog_sha256", "robot", "shoe_ids", "timestep"}:
        raise ValueError("Scene descriptor has missing or unknown fields")
    if not isinstance(config["shoe_ids"], list):
        raise ValueError("Scene descriptor shoe_ids must be a JSON array")
    expected = scene_config(config["robot"], config["shoe_ids"], config["timestep"])
    if config != expected:
        raise ValueError("Scene descriptor differs from the exact installed catalog contract")
    return build_scene(config["robot"], assets, shoe_ids=config["shoe_ids"], timestep=config["timestep"])


def _robot_spec(robot, assets, dt):
    """Compose only verified in-memory model bytes, preserving native mechanics."""
    mj, _ = sim._native_imports()
    if robot == "ur5e":
        _, blobs = sim._asset_payload(assets)
        spec = sim._spec(blobs, "universal_robots_ur5e", "ur5e.xml")
        jaw = sim._spec(blobs, "robotiq_2f85", "2f85.xml")
        spec.option.cone = jaw.option.cone
        spec.option.impratio = jaw.option.impratio
        spec.option.timestep = jaw.option.timestep = dt
        spec.option.integrator = jaw.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
        spec.attach(jaw, prefix="gripper/", site="attachment_site")
    else:
        item, blobs = robot_profiles._payload(robot, assets)
        spec = sim._spec(blobs, item["folder"], item["model"])
        spec.option.timestep = dt
    # Native home keyframes contain only robot coordinates. Initialize the named
    # joints explicitly after adding free objects rather than extending ambiguous
    # keyframe arrays or overwriting the objects' initial quaternion/positions.
    for key in list(spec.keys):
        spec.delete(key)
    spec.modelname = _MODEL_PREFIX + robot
    return spec


def build_scene(robot: str, assets, shoe_ids=None, timestep=.002):
    """Build a native MuJoCo model from one pinned robot and up to six shoe proxies.

    Shoes are free rigid bodies with gravity and collidable box/ellipsoid geometry.
    Their body origin is the sole bottom center. No object weld, mocap attachment,
    added force, robot controller, network request or implicit asset fetch is used.
    """
    collection, identifiers, dt = _selection(robot, shoe_ids, timestep)
    spec = _robot_spec(robot, assets, dt)
    mj, _ = sim._native_imports()
    spec.worldbody.add_geom(name="asset_table", type=mj.mjtGeom.mjGEOM_PLANE,
                            pos=[0, 0, 0], size=[2, 2, .1],
                            contype=1, conaffinity=1, condim=4,
                            friction=[.8, .005, .0001], rgba=[.88, .86, .81, 1])
    lookup = {shoe["id"]: shoe for shoe in collection["shoes"]}
    for index, identifier in enumerate(identifiers):
        shoe = lookup[identifier]
        name = "shoe_" + identifier.replace("-", "_")
        body = spec.worldbody.add_body(name=name,
                    pos=[.40 + .36 * (index // 2), .38 + .18 * (index % 2), .08], gravcomp=0)
        body.add_freejoint(name=name + "_free")
        properties = shoe["properties"]
        mass = properties["mass_kg"]["value"]
        friction = [properties[key]["value"] for key in
                    ("sliding_friction", "torsional_friction", "rolling_friction")]
        contact = properties["contact_solver"]["value"]
        for piece in shoe["geometry"]:
            body.add_geom(name=name + "_" + piece["name"],
                type={"box": mj.mjtGeom.mjGEOM_BOX, "ellipsoid": mj.mjtGeom.mjGEOM_ELLIPSOID}[piece["type"]],
                pos=piece["position_m"], size=piece["size_m"],
                mass=mass * piece["mass_fraction"], rgba=piece["rgba"],
                contype=1, conaffinity=1, friction=friction,
                priority=contact["priority"], condim=contact["condim"],
                solref=contact["solref"], solimp=contact["solimp"])
    model = spec.compile()
    expected = next(item for item in collection["robots"] if item["id"] == robot)["expected_dimensions"]
    dimensions = (expected["nq"] + 7 * len(identifiers), expected["nv"] + 6 * len(identifiers),
                  expected["nu"], expected["neq"], 0)
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != dimensions:
        raise ValueError("Composed asset scene differs from the native mechanical contract")
    return model


def initialize(model):
    """Reset a library scene and hold the robot's native home command targets.

    Free shoes retain their compiled spawn poses and can fall naturally afterwards.
    Callers own subsequent stepping; this function does not hide a settling phase.
    """
    mj, np = sim._native_imports()
    # MuJoCo stores the model name first in its packed name buffer; mj_id2name
    # does not expose mjOBJ_MODEL through the same lookup used for bodies.
    name = model.names.split(b"\0", 1)[0].decode("utf-8")
    robots = {item["id"]: item for item in catalog()["robots"]}
    if not isinstance(name, str) or not name.startswith(_MODEL_PREFIX) or name[len(_MODEL_PREFIX):] not in robots:
        raise ValueError("initialize requires a model built by the asset library")
    item = robots[name[len(_MODEL_PREFIX):]]
    data = mj.MjData(model)
    for joint_name, position in item["home_joint_positions"].items():
        joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint < 0 or int(model.jnt_type[joint]) not in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
            raise ValueError("Library robot home joint is missing or has changed type")
        data.qpos[int(model.jnt_qposadr[joint])] = position
    # All native arm actuators transmit directly to scalar joints. The reference
    # Robotiq actuator uses a tendon and its unactuated initial input remains zero.
    for index in range(model.nu):
        if int(model.actuator_trntype[index]) == int(mj.mjtTrn.mjTRN_JOINT):
            joint = int(model.actuator_trnid[index, 0])
            data.ctrl[index] = data.qpos[int(model.jnt_qposadr[joint])]
    if np.any(data.ctrl < model.actuator_ctrlrange[:, 0]) or np.any(data.ctrl > model.actuator_ctrlrange[:, 1]):
        raise ValueError("Library home command exceeds native actuator limits")
    mj.mj_forward(model, data)
    if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc, data.xpos)):
        raise ValueError("Library initialization produced nonfinite native state")
    return data
