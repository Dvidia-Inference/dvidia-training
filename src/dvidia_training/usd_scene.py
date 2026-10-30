"""A deliberately small, offline OpenUSD-to-MuJoCo workstation bridge.

This is an importer for the original v0.1 recipe, not arbitrary USD assets. The
USD SDK parses geometry and physics; the pinned PiPER remains a native MJCF
asset. No network fetch, USD plugin loading, texture, script or robot policy is
supplied by this module. Optional ``pxr`` imports happen only at stage inspection.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random

from . import asset_library, reference_sim as sim, robot_profiles


SCHEMA = "dvidia.usd-workstation-recipe/v1"
RECEIPT_SCHEMA = "dvidia.usd-native-scene/v1"
_PRIMS = {"/World": "Xform", "/World/Physics": "PhysicsScene",
          "/World/Table": "Cube", "/World/Object": "Cube",
          "/World/TableMaterial": "Material", "/World/ObjectMaterial": "Material"}
_TRANSFORM = {"size", "xformOp:translate", "xformOp:scale", "xformOp:orient", "xformOpOrder"}
_MASS = {"physics:mass", "physics:diagonalInertia", "physics:centerOfMass", "physics:principalAxes"}
_MATERIAL = {"physics:staticFriction", "physics:dynamicFriction", "physics:restitution"}
_TYPES = {"size": "double", "xformOp:translate": "double3", "xformOp:scale": "double3",
    "xformOp:orient": "quatd", "xformOpOrder": "token[]", "physics:gravityDirection": "vector3f",
    "physics:gravityMagnitude": "float", "physics:mass": "float", "physics:diagonalInertia": "float3",
    "physics:centerOfMass": "point3f", "physics:principalAxes": "quatf", "physics:collisionEnabled": "bool",
    "physics:rigidBodyEnabled": "bool", "physics:kinematicEnabled": "bool", "physics:staticFriction": "float",
    "physics:dynamicFriction": "float", "physics:restitution": "float"}
_PROPERTIES = {"/World": set(),
    "/World/Physics": {"physics:gravityDirection", "physics:gravityMagnitude"},
    "/World/Table": _TRANSFORM | {"physics:collisionEnabled", "material:binding:physics"},
    "/World/Object": _TRANSFORM | _MASS | {"physics:collisionEnabled", "physics:rigidBodyEnabled",
                                                     "physics:kinematicEnabled", "material:binding:physics"},
    "/World/TableMaterial": _MATERIAL, "/World/ObjectMaterial": _MATERIAL}
_APIS = {"/World": set(), "/World/Physics": set(),
    "/World/Table": {"PhysicsCollisionAPI", "MaterialBindingAPI"},
    "/World/Object": {"PhysicsCollisionAPI", "PhysicsRigidBodyAPI", "PhysicsMassAPI", "MaterialBindingAPI"},
    "/World/TableMaterial": {"PhysicsMaterialAPI"}, "/World/ObjectMaterial": {"PhysicsMaterialAPI"}}


def _bytes(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Expected a bounded, local regular file")
    with path.open("rb") as stream:
        content = stream.read(limit+1)
    if len(content) > limit:
        raise ValueError("Local input exceeds its byte limit")
    return content


def _number(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Invalid " + name)
    return float(value)


def _vector(value, length, low, high, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("Invalid " + name)
    return [_number(x, low, high, name) for x in value]


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(name + " has missing or unsupported fields")


def validate_recipe(recipe):
    """Validate the versioned recipe without importing USD or native packages."""
    _exact(recipe, {"schema", "stage", "stage_sha256", "robot_profile", "catalog_sha256",
                    "robot_lock_sha256", "timestep_s", "reset", "goal"}, "Recipe")
    if recipe["schema"] != SCHEMA or recipe["robot_profile"] != "piper":
        raise ValueError("Only the v1 PiPER workstation recipe is supported")
    if recipe["stage"] != "workstation.usda":
        raise ValueError("Stage must be the adjacent workstation.usda")
    if recipe["catalog_sha256"] != sha256(asset_library.CATALOG_PATH.read_bytes()).hexdigest() or recipe["robot_lock_sha256"] != sha256(robot_profiles.LOCK_PATH.read_bytes()).hexdigest():
        raise ValueError("Recipe asset catalog or robot lock has changed")
    digest = recipe["stage_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
        raise ValueError("Recipe stage digest must be lowercase SHA256")
    if recipe["timestep_s"] not in (.001, .002) or isinstance(recipe["timestep_s"], bool):
        raise ValueError("Native timestep must be 1 or 2 ms")
    reset = recipe["reset"]
    _exact(reset, {"seed", "object_xy_jitter_m", "object_yaw_jitter_rad"}, "Reset")
    if isinstance(reset["seed"], bool) or not isinstance(reset["seed"], int) or not 0 <= reset["seed"] < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    _vector(reset["object_xy_jitter_m"], 2, 0, .02, "reset XY jitter")
    _number(reset["object_yaw_jitter_rad"], 0, .2, "reset yaw jitter")
    goal = recipe["goal"]
    _exact(goal, {"object_center_m", "position_tolerance_m", "max_linear_speed_m_s",
                  "max_angular_speed_rad_s", "settle_s", "require_table_contact", "forbid_robot_contact"}, "Goal")
    _vector(goal["object_center_m"], 3, -2, 2, "goal position")
    _number(goal["position_tolerance_m"], .001, .1, "goal tolerance")
    _number(goal["max_linear_speed_m_s"], .0001, .1, "goal linear speed")
    _number(goal["max_angular_speed_rad_s"], .0001, 1, "goal angular speed")
    _number(goal["settle_s"], .05, 5, "goal settling interval")
    if goal["require_table_contact"] is not True or goal["forbid_robot_contact"] is not True:
        raise ValueError("This recipe requires table support and no robot contact")
    return json.loads(json.dumps(recipe, allow_nan=False))


def load_recipe(path):
    return validate_recipe(json.loads(_bytes(path, 16384)))


def _usd():
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError("OpenUSD inspection requires the optional usd-core package") from exc
    return Sdf, Usd, UsdGeom, UsdPhysics


def inspect_stage(path):
    """Parse one self-contained stage with the real USD SDK, rejecting other USD.

    Importing text into an anonymous layer allows inspection of composition arcs
    before opening a stage. No external layer or asset is resolved by this bridge.
    Unknown prim metadata, properties and applied APIs fail closed.
    """
    if Path(path).suffix != ".usda":
        raise ValueError("Only a local ASCII .usda stage is supported")
    raw = _bytes(path, 32768)
    Sdf, Usd, Geom, Physics = _usd()
    layer = Sdf.Layer.CreateAnonymous("dvidia-workstation.usda")
    if not layer.ImportFromString(raw.decode("utf-8")):
        raise ValueError("Invalid USD stage")
    if layer.subLayerPaths or layer.GetExternalReferences():
        raise ValueError("External references, payloads and sublayers are unsupported")
    if set(layer.pseudoRoot.ListInfoKeys()) - {"defaultPrim", "metersPerUnit", "kilogramsPerUnit", "upAxis"}:
        raise ValueError("Unsupported USD layer metadata")
    authored = {}

    def inspect_spec(spec):
        key = str(spec.path)
        if key not in _PRIMS or spec.typeName != _PRIMS[key] or spec.specifier != Sdf.SpecifierDef:
            raise ValueError("Unsupported USD prim or composition")
        if set(spec.ListInfoKeys()) - {"specifier", "typeName", "apiSchemas"}:
            raise ValueError("USD composition or prim metadata is unsupported")
        names = {prop.name for prop in spec.properties}
        if names != _PROPERTIES[key]:
            raise ValueError("USD prim has missing or unsupported properties: " + key)
        for prop in spec.properties:
            if layer.ListTimeSamplesForPath(prop.path):
                raise ValueError("Time-sampled USD properties are unsupported")
            allowed = {"custom", "variability", "targetPaths"} if isinstance(prop, Sdf.RelationshipSpec) else {"custom", "variability", "typeName", "default"}
            if set(prop.ListInfoKeys()) - allowed or prop.custom:
                raise ValueError("USD custom metadata, scripts or connections are unsupported")
            if isinstance(prop, Sdf.RelationshipSpec) != (prop.name == "material:binding:physics"):
                raise ValueError("USD property kind differs from the workstation contract")
            if isinstance(prop, Sdf.AttributeSpec) and str(prop.typeName) != _TYPES[prop.name]:
                raise ValueError("USD property type differs from the workstation contract")
            if isinstance(prop, Sdf.AttributeSpec) and prop.typeName in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                raise ValueError("USD asset dependencies are unsupported")
        authored[key] = names
        for child in spec.nameChildren:
            inspect_spec(child)

    for root in layer.rootPrims:
        inspect_spec(root)
    if set(authored) != set(_PRIMS):
        raise ValueError("Expected exactly the original six workstation prims")
    stage = Usd.Stage.Open(layer, load=Usd.Stage.LoadNone)
    if stage.GetDefaultPrim().GetPath() != Sdf.Path("/World") or not all(stage.HasAuthoredMetadata(key) for key in ("metersPerUnit", "kilogramsPerUnit", "upAxis")):
        raise ValueError("USD must explicitly declare its world and SI frame")
    if Geom.GetStageMetersPerUnit(stage) != 1. or Physics.GetStageKilogramsPerUnit(stage) != 1. or Geom.GetStageUpAxis(stage) != Geom.Tokens.z:
        raise ValueError("USD v0.1 requires metres and Z-up, with kilogram mass units")
    for path_, apis in _APIS.items():
        if set(stage.GetPrimAtPath(path_).GetAppliedSchemas()) != apis:
            raise ValueError("Unsupported or missing USD physics APIs")
    physics = Physics.Scene(stage.GetPrimAtPath("/World/Physics"))
    direction = list(physics.GetGravityDirectionAttr().Get())
    gravity = _number(physics.GetGravityMagnitudeAttr().Get(), 0, 100, "gravity")
    if direction != [0, 0, -1] or not math.isclose(gravity, 9.81, abs_tol=1e-5):
        raise ValueError("USD v0.1 requires explicitly authored standard downward gravity")
    result = {"stage_sha256": sha256(raw).hexdigest(), "units": "m-kg-s", "up_axis": "Z",
              "gravity_m_s2": [0., 0., -gravity]}
    for label, prim_path, mat_path in (("table", "/World/Table", "/World/TableMaterial"),
                                        ("object", "/World/Object", "/World/ObjectMaterial")):
        prim = stage.GetPrimAtPath(prim_path)
        cube = Geom.Cube(prim)
        size = _number(cube.GetSizeAttr().Get(), .001, 2, "cube size")
        if list(Geom.Xformable(prim).GetXformOpOrderAttr().Get()) != ["xformOp:translate", "xformOp:orient", "xformOp:scale"]:
            raise ValueError("Unsupported USD transform order")
        translation = _vector(list(prim.GetAttribute("xformOp:translate").Get()), 3, -2, 2, "translation")
        scale = _vector(list(prim.GetAttribute("xformOp:scale").Get()), 3, .001, 2, "scale")
        orient = prim.GetAttribute("xformOp:orient").Get()
        quat = [float(orient.GetReal()), *map(float, orient.GetImaginary())]
        if any(not math.isfinite(x) for x in quat) or not math.isclose(sum(x*x for x in quat), 1, abs_tol=1e-6):
            raise ValueError("USD orientation must be a unit quaternion")
        if prim.GetRelationship("material:binding:physics").GetTargets() != [Sdf.Path(mat_path)]:
            raise ValueError("Physics material must bind to the corresponding local material")
        material = Physics.MaterialAPI(stage.GetPrimAtPath(mat_path))
        dynamic = _number(material.GetDynamicFrictionAttr().Get(), 0, 2, "dynamic friction")
        static = _number(material.GetStaticFrictionAttr().Get(), 0, 2, "static friction")
        if dynamic != static or material.GetRestitutionAttr().Get() != 0:
            raise ValueError("MuJoCo v0.1 mapping requires equal static/dynamic friction and zero restitution")
        if Physics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True:
            raise ValueError("USD collision must be enabled")
        item = {"prim": prim_path, "center_m": translation, "quaternion_wxyz": quat,
                "half_size_m": [size*x/2 for x in scale], "sliding_friction": dynamic}
        if label == "object":
            rigid = Physics.RigidBodyAPI(prim)
            if rigid.GetRigidBodyEnabledAttr().Get() is not True or rigid.GetKinematicEnabledAttr().Get() is not False:
                raise ValueError("Object must be a dynamic rigid body")
            mass = Physics.MassAPI(prim)
            item["mass_kg"] = _number(mass.GetMassAttr().Get(), .001, 5, "mass")
            inertia = _vector(list(mass.GetDiagonalInertiaAttr().Get()), 3, 1e-9, 5, "diagonal inertia")
            if any(inertia[i] >= sum(inertia)-inertia[i] for i in range(3)):
                raise ValueError("Inertia must satisfy the strict triangle inequalities")
            item["diagonal_inertia_kg_m2"] = inertia
            if list(mass.GetCenterOfMassAttr().Get()) != [0, 0, 0] or mass.GetPrincipalAxesAttr().Get() != type(mass.GetPrincipalAxesAttr().Get())(1):
                raise ValueError("Only centred inertia in the cube's local axes is supported")
        result[label] = item
    # A flat horizontal support is the only task surface in this release.
    table = result["table"]
    if table["quaternion_wxyz"] != [1, 0, 0, 0] or abs(table["center_m"][2]+table["half_size_m"][2]) > 1e-7:
        raise ValueError("The horizontal tabletop must be at Z=0")
    return result


def compile_scene(recipe_path, assets):
    """Return a native MjSpec and portable receipt; USD owns object mechanics."""
    path = Path(recipe_path)
    recipe = load_recipe(path)
    stage = inspect_stage(path.parent / recipe["stage"])
    if recipe["stage_sha256"] != stage["stage_sha256"]:
        raise ValueError("USD stage differs from the recipe's pinned bytes")
    mj, _ = sim._native_imports()
    spec = asset_library._robot_spec("piper", assets, recipe["timestep_s"])
    spec.option.gravity = stage["gravity_m_s2"]
    table, obj = stage["table"], stage["object"]
    contact = {"type": mj.mjtGeom.mjGEOM_BOX, "contype": 1, "conaffinity": 1,
               "condim": 3, "solref": [.02, 1], "solimp": [.9, .95, .001, .5, 2]}
    spec.worldbody.add_geom(name="usd_table", pos=table["center_m"], quat=table["quaternion_wxyz"],
        size=table["half_size_m"], friction=[table["sliding_friction"], 0, 0], rgba=[.83, .77, .62, 1], **contact)
    body = spec.worldbody.add_body(name="usd_object", pos=obj["center_m"], quat=obj["quaternion_wxyz"],
        mass=obj["mass_kg"], inertia=obj["diagonal_inertia_kg_m2"], explicitinertial=True,
        ipos=[0, 0, 0], iquat=[1, 0, 0, 0], gravcomp=0)
    body.add_freejoint(name="usd_object_free")
    body.add_geom(name="usd_object_geom", size=obj["half_size_m"], mass=0,
        friction=[obj["sliding_friction"], 0, 0], rgba=[.29, .48, .37, 1], **contact)
    receipt = {"schema": RECEIPT_SCHEMA, "recipe": recipe, "recipe_sha256": sha256(_bytes(path, 16384)).hexdigest(),
        "usd": stage, "robot": robot_profiles.verify_assets("piper", assets),
        "mapping": {"geometry": "USD Cube dimensions and local pose to MuJoCo box",
            "mass_inertia": "explicit USD mass and centred diagonal inertia, unchanged",
            "friction": "equal USD static/dynamic coefficient to isotropic MuJoCo sliding friction; native max combination",
            "contact": "condim=3; torsional/rolling friction zero; solref=[0.02,1]; solimp=[0.9,0.95,0.001,0.5,2]",
            "restitution": "authored zero required; solver response is an assumption, not calibrated restitution",
            "robot": "pinned native MJCF; no USD articulation import"},
        "scope": {"simulation_only": True, "physics_calibrated": False, "learned_skill": False, "physical_robot_ready": False}}
    return spec, receipt


def build_scene(recipe_path, assets):
    spec, receipt = compile_scene(recipe_path, assets)
    model = spec.compile()
    mj, np = sim._native_imports()
    stage = receipt["usd"]
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "usd_object")
    geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "usd_object_geom")
    table = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "usd_table")
    checks = {"object_mass": math.isclose(float(model.body_mass[body]), stage["object"]["mass_kg"], rel_tol=1e-12),
        "object_inertia": bool(np.allclose(model.body_inertia[body], stage["object"]["diagonal_inertia_kg_m2"], rtol=1e-12, atol=1e-14)),
        "object_center_of_mass": bool(np.array_equal(model.body_ipos[body], [0, 0, 0])),
        "object_inertia_axes": bool(np.array_equal(model.body_iquat[body], [1, 0, 0, 0])),
        "object_gravity_compensation_disabled": float(model.body_gravcomp[body]) == 0,
        "object_size": bool(np.array_equal(model.geom_size[geom], stage["object"]["half_size_m"])),
        "table_size": bool(np.array_equal(model.geom_size[table], stage["table"]["half_size_m"])),
        "object_position": bool(np.array_equal(model.body_pos[body], stage["object"]["center_m"])),
        "object_orientation": bool(np.allclose(model.body_quat[body], stage["object"]["quaternion_wxyz"], rtol=0, atol=1e-12)),
        "table_position": bool(np.array_equal(model.geom_pos[table], stage["table"]["center_m"])),
        "table_orientation": bool(np.array_equal(model.geom_quat[table], stage["table"]["quaternion_wxyz"])),
        "gravity": bool(np.array_equal(model.opt.gravity, stage["gravity_m_s2"])),
        "object_friction": float(model.geom_friction[geom, 0]) == stage["object"]["sliding_friction"],
        "table_friction": float(model.geom_friction[table, 0]) == stage["table"]["sliding_friction"],
        "free_object": int(model.jnt_type[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "usd_object_free")]) == int(mj.mjtJoint.mjJNT_FREE),
        "native_dimensions": (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (15, 14, 7, 1, 0)}
    if not all(checks.values()):
        raise ValueError("Compiled native properties differ from the parsed USD contract")
    receipt["compiled_checks"] = checks
    return model, receipt


def initialize(model, recipe, seed=None):
    """Reset the native home and seeded object pose, without a hidden settling run."""
    recipe = validate_recipe(recipe)
    selected = recipe["reset"]["seed"] if seed is None else seed
    if isinstance(selected, bool) or not isinstance(selected, int) or not 0 <= selected < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    mj, _ = sim._native_imports()
    data = asset_library.initialize(model)
    joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "usd_object_free")
    if joint < 0:
        raise ValueError("A compiled USD object is required")
    q = int(model.jnt_qposadr[joint])
    rng = random.Random(selected)
    for axis, extent in enumerate(recipe["reset"]["object_xy_jitter_m"]):
        data.qpos[q+axis] += rng.uniform(-extent, extent)
    angle = rng.uniform(-recipe["reset"]["object_yaw_jitter_rad"], recipe["reset"]["object_yaw_jitter_rad"])
    # Pre-multiply the authored object quaternion by world-Z yaw.
    w, x, y, z = map(float, data.qpos[q+3:q+7])
    c, s = math.cos(angle/2), math.sin(angle/2)
    data.qpos[q+3:q+7] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
    mj.mj_forward(model, data)
    return data


def goal_predicates(model, data, recipe):
    """Independent native geometry/contact checks; no policy completion signal."""
    return _goal_predicates(model, data, validate_recipe(recipe)["goal"])


def _goal_predicates(model, data, goal):
    mj, np = sim._native_imports()
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "usd_object")
    geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "usd_object_geom")
    table = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "usd_table")
    if min(body, geom, table) < 0:
        raise ValueError("Goal requires the compiled USD object and table")
    velocity = np.zeros(6)
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body, velocity, 0)
    pairs = [(int(contact.geom1), int(contact.geom2)) for contact in data.contact[:data.ncon]]
    support = any(set(pair) == {geom, table} for pair in pairs)
    robot = any(geom in pair and table not in pair for pair in pairs)
    result = {"position": float(np.linalg.norm(data.xpos[body]-goal["object_center_m"])) <= goal["position_tolerance_m"],
              "linear_speed": float(np.linalg.norm(velocity[3:])) <= goal["max_linear_speed_m_s"],
              "angular_speed": float(np.linalg.norm(velocity[:3])) <= goal["max_angular_speed_rad_s"],
              "table_contact": support, "no_robot_contact": not robot}
    result["instantaneous"] = all(result.values())
    return result


class GoalMonitor:
    """Require consecutive native steps and independent goal predicates to settle.

    Update after every physics step. A skipped interval or time reset clears the
    dwell rather than inferring success from two distant endpoint observations.
    """
    def __init__(self, recipe):
        self.recipe = validate_recipe(recipe)
        self.since = None
        self.previous = None

    def update(self, model, data):
        now = float(data.time)
        if not math.isfinite(now) or now < 0:
            raise ValueError("Goal monitor requires finite native simulation time")
        if self.previous is not None and (now < self.previous or now-self.previous > float(model.opt.timestep)*1.5):
            self.since = None
        self.previous = now
        predicates = _goal_predicates(model, data, self.recipe["goal"])
        if not predicates["instantaneous"]:
            self.since = None
        elif self.since is None:
            self.since = now
        duration = 0. if self.since is None else now-self.since
        return {**predicates, "settled_s": duration, "success": duration >= self.recipe["goal"]["settle_s"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Parse the pinned USD geometry and physics")
    validate.add_argument("recipe", type=Path)
    build = sub.add_parser("build", help="Compile the native PiPER workstation and emit a receipt")
    build.add_argument("recipe", type=Path)
    build.add_argument("--assets", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        recipe = load_recipe(args.recipe)
        result = inspect_stage(args.recipe.parent/recipe["stage"])
        if result["stage_sha256"] != recipe["stage_sha256"]:
            raise ValueError("USD stage differs from pinned recipe")
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        _, receipt = build_scene(args.recipe, args.assets)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"schema": RECEIPT_SCHEMA, "compiled_checks": receipt["compiled_checks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
