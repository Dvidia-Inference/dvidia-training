"""Bounded multi-object OpenUSD task scenes for the pinned native PiPER.

Only original box geometry is authored/imported. Shoe surrogates are rigid,
stepped compound boxes, never measured shoes or deformable material models.
All dynamics values are explicit assumptions. No controller, arbitrary USD
composition, external asset resolution or physical-robot command is provided.
"""
from __future__ import annotations

from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import re

from . import asset_library, reference_sim as sim, robot_profiles

SCHEMA = "dvidia.task-scene-recipe/v1"
RECEIPT_SCHEMA = "dvidia.task-scene-native/v1"
AUTHOR_SCHEMA = "dvidia.task-scene-authored/v1"
EVIDENCE = "authored_assumption"
STAGE = "workstation.usda"
_SOURCE_NAMES = ("task_scene.py", "asset_library.py", "robot_profiles.py", "reference_sim.py")
_MAX_BYTES = 262144
_CONTACT = {"condim": 3, "solref": [.02, 1], "solimp": [.9, .95, .001, .5, 2]}


def _sources():
    root = Path(__file__).parent
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in _SOURCE_NAMES}


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(label+" has missing or unsupported fields")


def _number(value, low, high, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Invalid "+label)
    return float(value)


def _vector(value, n, low, high, label):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError("Invalid "+label)
    return [_number(x, low, high, label) for x in value]


def _read(path, limit=_MAX_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Expected a bounded local regular file")
    with path.open("rb") as stream:
        data = stream.read(limit+1)
    if len(data) > limit:
        raise ValueError("Input exceeds byte bound")
    return data


def default_recipe():
    """Return an editable, explicitly assumed scene; no native/USD imports."""
    return {"schema": SCHEMA, "robot_profile": "piper",
        "catalog_sha256": sha256(asset_library.CATALOG_PATH.read_bytes()).hexdigest(),
        "robot_lock_sha256": sha256(robot_profiles.LOCK_PATH.read_bytes()).hexdigest(),
        "source_sha256": _sources(), "timestep_s": .001, "control_s": .02,
        "table": {"center_m": [.4, 0., -.04], "size_m": [1.2, 1., .08],
                  "sliding_friction": .8, "property_evidence": EVIDENCE},
        "bin": {"center_m": [.65, .2, 0.], "yaw_rad": 0.,
                "inner_size_m": [.36, .25, .16], "wall_m": .015,
                "sliding_friction": .6, "property_evidence": EVIDENCE},
        "objects": [
            {"id": "block", "kind": "box", "position_m": [.35, -.2, .04], "yaw_rad": 0.,
             "size_m": [.04, .04, .04], "mass_kg": .04,
             "sliding_friction": .5, "property_evidence": EVIDENCE},
            {"id": "shoe", "kind": "rigid_shoe_surrogate", "position_m": [.57, -.2, .04], "yaw_rad": 0.,
             "size_m": [.24, .09, .07], "mass_kg": .25,
             "sliding_friction": .8, "property_evidence": EVIDENCE}],
        "goal": {"settle_s": .25, "max_linear_speed_m_s": .01,
                 "max_angular_speed_rad_s": .05, "containment_margin_m": .001,
                 "require_bin_floor_contact": True, "forbid_robot_contact": True}}


def collision_pieces(obj):
    """Pure original collision geometry; local origin is the bottom centre.

    size_m is full XYZ width, never MuJoCo half extents. The three shoe boxes
    touch at horizontal faces and have no overlapping volume. Mass is uniform
    density over the compound, not a measurement of shoe mass distribution.
    """
    sx, sy, sz = obj["size_m"]
    if obj["kind"] == "box":
        rows = [([sx, sy, sz], [0., 0., sz/2])]
    elif obj["kind"] == "rigid_shoe_surrogate":
        rows = [([sx, sy, .18*sz], [0., 0., .09*sz]),
                ([.85*sx, .85*sy, .50*sz], [0., 0., .43*sz]),
                ([.60*sx, .70*sy, .32*sz], [0., 0., .84*sz])]
    else:
        raise ValueError("Unsupported original geometry")
    volume = sum(math.prod(size) for size, _ in rows)
    return [{"name": "piece_"+str(i), "size_m": size, "position_m": pos,
             "mass_fraction": math.prod(size)/volume} for i, (size, pos) in enumerate(rows)]


def _yaw(angle):
    return [math.cos(angle/2), 0., 0., math.sin(angle/2)]


def _rotate(q, p):
    w, x, y, z = q
    vx, vy, vz = p
    return [(1-2*(y*y+z*z))*vx+2*(x*y-z*w)*vy+2*(x*z+y*w)*vz,
            2*(x*y+z*w)*vx+(1-2*(x*x+z*z))*vy+2*(y*z-x*w)*vz,
            2*(x*z-y*w)*vx+2*(y*z+x*w)*vy+(1-2*(x*x+y*y))*vz]


def _pose(position, quaternion):
    position = _vector(position, 3, -4, 4, "object position")
    quaternion = _vector(quaternion, 4, -1, 1, "object quaternion")
    if not math.isclose(sum(x*x for x in quaternion), 1., abs_tol=1e-6):
        raise ValueError("Object quaternion must be unit length")
    return position, quaternion


def object_corners(obj, position_m=None, quaternion_wxyz=None):
    """All actual compound collision corners in world coordinates, including tilt."""
    pos, quat = _pose(obj["position_m"] if position_m is None else position_m,
                      _yaw(obj["yaw_rad"]) if quaternion_wxyz is None else quaternion_wxyz)
    corners = []
    for piece in collision_pieces(obj):
        for signs in itertools.product((-1, 1), repeat=3):
            local = [piece["position_m"][i]+signs[i]*piece["size_m"][i]/2 for i in range(3)]
            shifted = _rotate(quat, local)
            corners.append([pos[i]+shifted[i] for i in range(3)])
    return corners


def containment(obj, position_m, quaternion_wxyz, bin_, margin_m=0.):
    """Pure corner test in the bin's yaw frame; no centre-only completion.

    A fixed 1 mm floor tolerance permits solver contact penetration. The supplied
    margin applies to side walls and the top rim, not the supporting floor.
    """
    margin = _number(margin_m, 0, .01, "containment margin")
    center = bin_["center_m"]
    c, s = math.cos(bin_["yaw_rad"]), math.sin(bin_["yaw_rad"])
    corners = []
    for point in object_corners(obj, position_m, quaternion_wxyz):
        x, y, z = [point[i]-center[i] for i in range(3)]
        corners.append([c*x+s*y, -s*x+c*y, z])
    inner = bin_["inner_size_m"]
    floor = bin_["wall_m"]
    axes = {"x": all(abs(p[0]) <= inner[0]/2-margin+1e-12 for p in corners),
            "y": all(abs(p[1]) <= inner[1]/2-margin+1e-12 for p in corners),
            "z": all(floor-.001-1e-12 <= p[2] <= floor+inner[2]-margin+1e-12 for p in corners)}
    return {"contained": all(axes.values()), "axes": axes,
            "corner_count": len(corners), "floor_tolerance_m": .001}


def _bin_pieces(bin_):
    ix, iy, height = bin_["inner_size_m"]
    wall = bin_["wall_m"]
    ox, oy = ix+2*wall, iy+2*wall
    pieces = [("floor", [ox, oy, wall], [0., 0., wall/2]),
        ("left", [wall, oy, height], [-(ix+wall)/2, 0., wall+height/2]),
        ("right", [wall, oy, height], [(ix+wall)/2, 0., wall+height/2]),
        ("front", [ix, wall, height], [0., -(iy+wall)/2, wall+height/2]),
        ("back", [ix, wall, height], [0., (iy+wall)/2, wall+height/2])]
    result = []
    quat = _yaw(bin_["yaw_rad"])
    for name, size, local in pieces:
        offset = _rotate(quat, local)
        result.append({"name": name, "size_m": size,
            "center_m": [bin_["center_m"][i]+offset[i] for i in range(3)], "yaw_rad": bin_["yaw_rad"]})
    return result


def _obb(center, size, yaw):
    return {"center": center, "size": size, "yaw": yaw}


def _overlap(a, b):
    """Strict-volume overlap for Z-aligned, yawed boxes (face contact allowed)."""
    if abs(a["center"][2]-b["center"][2]) >= (a["size"][2]+b["size"][2])/2-1e-9:
        return False
    delta = [b["center"][i]-a["center"][i] for i in range(2)]
    axes = lambda yaw: ((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw)))
    aa, bb = axes(a["yaw"]), axes(b["yaw"])
    for axis in (*aa, *bb):
        distance = abs(sum(delta[i]*axis[i] for i in range(2)))
        radius = sum(a["size"][j]/2*abs(sum(aa[j][i]*axis[i] for i in range(2))) for j in range(2))
        radius += sum(b["size"][j]/2*abs(sum(bb[j][i]*axis[i] for i in range(2))) for j in range(2))
        if distance >= radius-1e-9:
            return False
    return True


def validate_recipe(recipe):
    """Strict finite bounds and initial geometry checks, without optional packages."""
    _exact(recipe, {"schema", "robot_profile", "catalog_sha256", "robot_lock_sha256", "source_sha256",
        "timestep_s", "control_s", "table", "bin", "objects", "goal"}, "Recipe")
    if recipe["schema"] != SCHEMA or recipe["robot_profile"] != "piper":
        raise ValueError("Only this versioned pinned PiPER scene is supported")
    if recipe["catalog_sha256"] != sha256(asset_library.CATALOG_PATH.read_bytes()).hexdigest() or recipe["robot_lock_sha256"] != sha256(robot_profiles.LOCK_PATH.read_bytes()).hexdigest():
        raise ValueError("Catalog or robot profile lock differs")
    if recipe["source_sha256"] != _sources():
        raise ValueError("Scene source files differ from the recipe lock")
    if type(recipe["timestep_s"]) not in (int, float) or recipe["timestep_s"] != .001 or type(recipe["control_s"]) not in (int, float) or recipe["control_s"] != .02:
        raise ValueError("This scene requires 1 ms physics and 20 ms control boundaries")
    table, bin_ = recipe["table"], recipe["bin"]
    _exact(table, {"center_m", "size_m", "sliding_friction", "property_evidence"}, "Table")
    _vector(table["center_m"], 3, -1.5, 1.5, "table centre")
    _vector(table["size_m"], 3, .02, 2., "table size")
    if not .02 <= table["size_m"][2] <= .2 or abs(table["center_m"][2]+table["size_m"][2]/2) > 1e-9:
        raise ValueError("Horizontal tabletop must be at Z=0 with bounded thickness")
    _exact(bin_, {"center_m", "yaw_rad", "inner_size_m", "wall_m", "sliding_friction", "property_evidence"}, "Bin")
    _vector(bin_["center_m"], 3, -1.5, 1.5, "bin centre")
    _vector(bin_["inner_size_m"], 3, .05, .8, "bin interior")
    _number(bin_["wall_m"], .005, .05, "bin wall")
    _number(bin_["yaw_rad"], -math.pi, math.pi, "bin yaw")
    if abs(bin_["center_m"][2]) > 1e-9:
        raise ValueError("Bin outside bottom must be supported at table Z=0")
    for item in (table, bin_):
        _number(item["sliding_friction"], 0, 2, "sliding friction")
        if item["property_evidence"] != EVIDENCE:
            raise ValueError("Properties must be labelled authored assumptions")
    if not isinstance(recipe["objects"], list) or not 1 <= len(recipe["objects"]) <= 4:
        raise ValueError("One to four object instances are supported")
    goal = recipe["goal"]
    _exact(goal, {"settle_s", "max_linear_speed_m_s", "max_angular_speed_rad_s", "containment_margin_m", "require_bin_floor_contact", "forbid_robot_contact"}, "Goal")
    _number(goal["settle_s"], .05, 2., "goal dwell")
    _number(goal["max_linear_speed_m_s"], .0001, .05, "goal linear speed")
    _number(goal["max_angular_speed_rad_s"], .0001, .5, "goal angular speed")
    _number(goal["containment_margin_m"], 0, .01, "goal margin")
    if goal["require_bin_floor_contact"] is not True or goal["forbid_robot_contact"] is not True:
        raise ValueError("Goal requires bin floor support and released objects")
    ids, object_boxes = set(), []
    tx, ty = table["center_m"][:2]
    hx, hy = [x/2 for x in table["size_m"][:2]]
    def supported_xy(box):
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        for sx, sy in itertools.product((-1, 1), repeat=2):
            x = box["center"][0]+c*sx*box["size"][0]/2-s*sy*box["size"][1]/2
            y = box["center"][1]+s*sx*box["size"][0]/2+c*sy*box["size"][1]/2
            if abs(x-tx) > hx+1e-9 or abs(y-ty) > hy+1e-9:
                raise ValueError("Initial support footprint overhangs the table")
    static = [_obb(piece["center_m"], piece["size_m"], piece["yaw_rad"]) for piece in _bin_pieces(bin_)]
    for box in static:
        supported_xy(box)
    for obj in recipe["objects"]:
        _exact(obj, {"id", "kind", "position_m", "yaw_rad", "size_m", "mass_kg", "sliding_friction", "property_evidence"}, "Object")
        identifier = obj["id"]
        if not isinstance(identifier, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,23}", identifier) is None or identifier in ids:
            raise ValueError("Object IDs must be unique bounded lowercase identifiers")
        ids.add(identifier)
        if obj["kind"] not in ("box", "rigid_shoe_surrogate"):
            raise ValueError("Only original box and rigid shoe surrogate are supported")
        _vector(obj["position_m"], 3, -1.5, 1.5, "object position")
        _vector(obj["size_m"], 3, .01, .4, "object size")
        _number(obj["yaw_rad"], -math.pi, math.pi, "object yaw")
        _number(obj["mass_kg"], .005, 2., "object mass")
        _number(obj["sliding_friction"], 0, 2, "object friction")
        if obj["property_evidence"] != EVIDENCE:
            raise ValueError("Object mechanics must be labelled authored assumptions")
        if not 0 <= obj["position_m"][2] <= .3:
            raise ValueError("Initial object bottom must be above the table by at most .3 m")
        # Require at least one yaw orientation in which the full footprint fits.
        # This conservative outer-box criterion intentionally rejects nesting and
        # tilted insertion claims, which the bounded recipe cannot qualify.
        sx, sy, sz = obj["size_m"]
        ix, iy, iz = bin_["inner_size_m"]
        margin = goal["containment_margin_m"]
        if not ((sx <= ix-2*margin and sy <= iy-2*margin) or (sy <= ix-2*margin and sx <= iy-2*margin)) or sz > iz-margin:
            raise ValueError("Object cannot fit upright into the specified bin")
        boxes = []
        for piece in collision_pieces(obj):
            local = _rotate(_yaw(obj["yaw_rad"]), piece["position_m"])
            box = _obb([obj["position_m"][i]+local[i] for i in range(3)], piece["size_m"], obj["yaw_rad"])
            supported_xy(box)
            if any(_overlap(box, other) for other in static):
                raise ValueError("Object starts overlapping a bin collider")
            if any(_overlap(box, other) for other in object_boxes):
                raise ValueError("Object instances start overlapping")
            boxes.append(box)
        object_boxes.extend(boxes)
    return json.loads(json.dumps(recipe, allow_nan=False))


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Nonfinite JSON number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def load_recipe(path):
    return validate_recipe(_decode(_read(path)))


def _usd():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError("Scene authoring/import requires optional usd-core") from exc
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics


def _inertia(obj):
    pieces = collision_pieces(obj)
    mass = obj["mass_kg"]
    center = [sum(p["mass_fraction"]*p["position_m"][i] for p in pieces) for i in range(3)]
    inertia = [0., 0., 0.]
    for p in pieces:
        m = mass*p["mass_fraction"]
        d = [p["position_m"][i]-center[i] for i in range(3)]
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            inertia[axis] += m*(sum(p["size_m"][i]**2 for i in other)/12+sum(d[i]**2 for i in other))
    return center, inertia


def _expected_geometry(recipe):
    result = {"table": {"prim": "/World/Table", "center_m": recipe["table"]["center_m"],
        "size_m": recipe["table"]["size_m"], "quaternion_wxyz": [1., 0., 0., 0.],
        "sliding_friction": recipe["table"]["sliding_friction"]}, "bin": [], "objects": {}}
    for item in _bin_pieces(recipe["bin"]):
        result["bin"].append({"name": item["name"], "prim": "/World/Bin_"+item["name"],
            "center_m": item["center_m"], "size_m": item["size_m"],
            "quaternion_wxyz": _yaw(item["yaw_rad"]), "sliding_friction": recipe["bin"]["sliding_friction"]})
    for obj in recipe["objects"]:
        center, inertia = _inertia(obj)
        result["objects"][obj["id"]] = {"prim": "/World/Object_"+obj["id"],
            "position_m": obj["position_m"], "quaternion_wxyz": _yaw(obj["yaw_rad"]),
            "mass_kg": obj["mass_kg"], "center_of_mass_m": center,
            "diagonal_inertia_kg_m2": inertia, "sliding_friction": obj["sliding_friction"],
            "pieces": collision_pieces(obj)}
    return result


def _author_stage(recipe):
    Gf, Sdf, Usd, Geom, Physics = _usd()
    stage = Usd.Stage.CreateInMemory("dvidia-task-scene.usda")
    world = Geom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    Geom.SetStageUpAxis(stage, Geom.Tokens.z)
    Geom.SetStageMetersPerUnit(stage, 1.)
    Physics.SetStageKilogramsPerUnit(stage, 1.)
    physics = Physics.Scene.Define(stage, "/World/Physics")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    parsed = _expected_geometry(recipe)
    def material(path, mu):
        prim = stage.DefinePrim(path, "Material")
        api = Physics.MaterialAPI.Apply(prim)
        # USD Physics materials use schema-defined float32; importer retains
        # those actual parsed values and records the precision explicitly.
        api.CreateStaticFrictionAttr(mu)
        api.CreateDynamicFrictionAttr(mu)
        api.CreateRestitutionAttr(0.)
        return path
    def transform(prim, pos, quat, scale=None):
        xform = Geom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddOrientOp(Geom.XformOp.PrecisionDouble).Set(Gf.Quatd(quat[0], Gf.Vec3d(*quat[1:])))
        if scale is not None:
            xform.AddScaleOp(Geom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
    def cube(path, pos, quat, size, mat):
        shape = Geom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.)
        transform(shape.GetPrim(), pos, quat, size)
        Physics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        shape.GetPrim().ApplyAPI("MaterialBindingAPI")
        shape.GetPrim().CreateRelationship("material:binding:physics", custom=False).SetTargets([Sdf.Path(mat)])
    table_mat = material("/World/Material_table", recipe["table"]["sliding_friction"])
    bin_mat = material("/World/Material_bin", recipe["bin"]["sliding_friction"])
    table = parsed["table"]
    cube(table["prim"], table["center_m"], table["quaternion_wxyz"], table["size_m"], table_mat)
    for item in parsed["bin"]:
        cube(item["prim"], item["center_m"], item["quaternion_wxyz"], item["size_m"], bin_mat)
    for identifier, obj in parsed["objects"].items():
        body = Geom.Xform.Define(stage, obj["prim"]).GetPrim()
        transform(body, obj["position_m"], obj["quaternion_wxyz"])
        rigid = Physics.RigidBodyAPI.Apply(body)
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(False)
        mass = Physics.MassAPI.Apply(body)
        mass.CreateMassAttr(obj["mass_kg"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*obj["center_of_mass_m"]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*obj["diagonal_inertia_kg_m2"]))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        mat = material("/World/Material_object_"+identifier, obj["sliding_friction"])
        for piece in obj["pieces"]:
            cube(obj["prim"]+"/"+piece["name"], piece["position_m"], [1., 0., 0., 0.], piece["size_m"], mat)
    return stage.GetRootLayer().ExportToString().encode("utf-8")


def write_scene(recipe, output):
    """Author an exact self-contained USD stage; never replace an existing path."""
    recipe = validate_recipe(recipe)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Scene output already exists")
    raw = _author_stage(recipe)
    # Inspect serialized bytes before creating the destination.
    parsed = _inspect_bytes(raw, recipe)
    output.mkdir(parents=True, exist_ok=False)
    files = {"recipe.json": _json(recipe), STAGE: raw}
    receipt = {"schema": AUTHOR_SCHEMA, "files": {name: {"bytes": len(data), "sha256": sha256(data).hexdigest()} for name, data in files.items()},
        "source_sha256": _sources(), "usd": parsed, "scope": "bounded original rigid geometry; no learning or hardware execution"}
    for name, data in {**files, "scene-receipt.json": _json(receipt)}.items():
        with (output/name).open("xb") as stream:
            stream.write(data)
    return receipt


def _inspect_bytes(raw, recipe):
    """Inspect actual SDK-parsed values against the exact authored subset.

    Compare Sdf specs before opening the stage. No references, inheritance,
    variants, scripts, asset properties or plugin-defined prims are accepted.
    Importer reads mechanics from this parsed stage rather than recipe guesses.
    """
    if len(raw) > _MAX_BYTES:
        raise ValueError("USD exceeds its bound")
    Gf, Sdf, Usd, Geom, Physics = _usd()
    layer = Sdf.Layer.CreateAnonymous("dvidia-task-input.usda")
    if not layer.ImportFromString(raw.decode("utf-8")):
        raise ValueError("Invalid USD stage")
    if layer.subLayerPaths or layer.GetExternalReferences():
        raise ValueError("External USD composition is unsupported")
    if set(layer.pseudoRoot.ListInfoKeys()) != {"defaultPrim", "metersPerUnit", "kilogramsPerUnit", "upAxis"}:
        raise ValueError("USD must contain only explicit SI/Z-up layer metadata")
    expected = Sdf.Layer.CreateAnonymous("dvidia-task-expected.usda")
    expected.ImportFromString(_author_stage(recipe).decode("utf-8"))
    # Exact spec/value comparison also covers unknown metadata and wrong types.
    def inventory(target):
        result = {}
        def visit(spec):
            metadata = {str(key): str(spec.GetInfo(key)) for key in spec.ListInfoKeys()}
            props = {}
            for prop in spec.properties:
                if target.ListTimeSamplesForPath(prop.path):
                    raise ValueError("Time samples are unsupported")
                props[prop.name] = {str(key): str(prop.GetInfo(key)) for key in prop.ListInfoKeys()}
            result[str(spec.path)] = {"metadata": metadata, "properties": props}
            for child in spec.nameChildren:
                visit(child)
        for prim in target.rootPrims:
            visit(prim)
        return result
    if inventory(layer) != inventory(expected):
        raise ValueError("USD stage differs from the exact bounded recipe contract")
    stage = Usd.Stage.Open(layer, load=Usd.Stage.LoadNone)
    if (stage.GetDefaultPrim().GetPath() != Sdf.Path("/World") or Geom.GetStageMetersPerUnit(stage) != 1.
            or Physics.GetStageKilogramsPerUnit(stage) != 1. or Geom.GetStageUpAxis(stage) != Geom.Tokens.z):
        raise ValueError("USD must use explicit metre/kilogram/Z-up frame")
    result = _expected_geometry(recipe)
    result.update(stage_sha256=sha256(raw).hexdigest(), units="m-kg-s", up_axis="Z", gravity_m_s2=[0., 0., -float(Physics.Scene(stage.GetPrimAtPath("/World/Physics")).GetGravityMagnitudeAttr().Get())])
    def read_transform(path):
        prim = stage.GetPrimAtPath(path)
        quat = prim.GetAttribute("xformOp:orient").Get()
        pos = list(map(float, prim.GetAttribute("xformOp:translate").Get()))
        q = [float(quat.GetReal()), *map(float, quat.GetImaginary())]
        return prim, pos, q
    for item in (result["table"], *result["bin"]):
        prim, pos, quat = read_transform(item["prim"])
        item.update(center_m=pos, quaternion_wxyz=quat, size_m=list(map(float, prim.GetAttribute("xformOp:scale").Get())))
        target = prim.GetRelationship("material:binding:physics").GetTargets()[0]
        item["sliding_friction"] = float(Physics.MaterialAPI(stage.GetPrimAtPath(target)).GetDynamicFrictionAttr().Get())
    for identifier, item in result["objects"].items():
        prim, pos, quat = read_transform(item["prim"])
        mass = Physics.MassAPI(prim)
        item.update(position_m=pos, quaternion_wxyz=quat, mass_kg=float(mass.GetMassAttr().Get()),
            center_of_mass_m=list(map(float, mass.GetCenterOfMassAttr().Get())),
            diagonal_inertia_kg_m2=list(map(float, mass.GetDiagonalInertiaAttr().Get())))
        mat = stage.GetPrimAtPath("/World/Material_object_"+identifier)
        item["sliding_friction"] = float(Physics.MaterialAPI(mat).GetDynamicFrictionAttr().Get())
        for piece in item["pieces"]:
            child, local, _ = read_transform(item["prim"]+"/"+piece["name"])
            piece.update(position_m=local, size_m=list(map(float, child.GetAttribute("xformOp:scale").Get())))
    return result


def inspect_stage(path, recipe):
    return _inspect_bytes(_read(path), validate_recipe(recipe))


def build_scene(recipe_path, assets):
    """Read pinned authored USD mechanics and compile unchanged native PiPER."""
    path = Path(recipe_path)
    recipe = load_recipe(path)
    raw_recipe, raw_stage = _read(path), _read(path.parent/STAGE)
    authored = _decode(_read(path.parent/"scene-receipt.json"))
    _exact(authored, {"schema", "files", "source_sha256", "usd", "scope"}, "Authored receipt")
    expected_files = {name: {"bytes": len(data), "sha256": sha256(data).hexdigest()} for name, data in (("recipe.json", raw_recipe), (STAGE, raw_stage))}
    if authored["schema"] != AUTHOR_SCHEMA or authored["files"] != expected_files or authored["source_sha256"] != _sources():
        raise ValueError("Authored scene bytes or sources differ from their immutable receipt")
    parsed = _inspect_bytes(raw_stage, recipe)
    if authored["usd"] != parsed:
        raise ValueError("Authored USD parsed mechanics changed")
    mj, np = sim._native_imports()
    spec = asset_library._robot_spec("piper", assets, .001)
    spec.option.gravity = parsed["gravity_m_s2"]
    common = {"type": mj.mjtGeom.mjGEOM_BOX, "contype": 1, "conaffinity": 1, **_CONTACT}
    table = parsed["table"]
    spec.worldbody.add_geom(name="task_table", pos=table["center_m"], quat=table["quaternion_wxyz"],
        size=[x/2 for x in table["size_m"]], friction=[table["sliding_friction"], 0, 0], rgba=[.82, .76, .62, 1], **common)
    for item in parsed["bin"]:
        spec.worldbody.add_geom(name="task_bin_"+item["name"], pos=item["center_m"], quat=item["quaternion_wxyz"],
            size=[x/2 for x in item["size_m"]], friction=[item["sliding_friction"], 0, 0], rgba=[.65, .71, .66, 1], **common)
    for identifier, item in parsed["objects"].items():
        name = "task_object_"+identifier
        body = spec.worldbody.add_body(name=name, pos=item["position_m"], quat=item["quaternion_wxyz"],
            mass=item["mass_kg"], inertia=item["diagonal_inertia_kg_m2"], explicitinertial=True,
            ipos=item["center_of_mass_m"], iquat=[1, 0, 0, 0], gravcomp=0)
        body.add_freejoint(name=name+"_free")
        for piece in item["pieces"]:
            body.add_geom(name=name+"_"+piece["name"], pos=piece["position_m"], size=[x/2 for x in piece["size_m"]],
                mass=0, friction=[item["sliding_friction"], 0, 0], rgba=[.27, .43, .35, 1], **common)
    model = spec.compile()
    baseline = robot_profiles.build_model("piper", assets, timestep=.001)
    checks = {"native_dimensions": (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (8+7*len(recipe["objects"]), 8+6*len(recipe["objects"]), 7, 1, 0),
        "robot_actuators_unchanged": all(np.array_equal(getattr(model, key), getattr(baseline, key)) for key in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm", "actuator_trnid")),
        "robot_inertia_unchanged": all(np.array_equal(getattr(model, key)[:baseline.nbody], getattr(baseline, key)) for key in ("body_mass", "body_inertia", "body_gravcomp", "body_ipos", "body_iquat")),
        "robot_joints_unchanged": all(np.array_equal(getattr(model, key)[:baseline.njnt], getattr(baseline, key)) for key in ("jnt_type", "jnt_range", "jnt_axis", "jnt_pos", "jnt_limited")),
        "robot_dofs_unchanged": all(np.array_equal(getattr(model, key)[:baseline.nv], getattr(baseline, key)) for key in ("dof_damping", "dof_frictionloss", "dof_armature")),
        "timestep": model.opt.timestep == .001, "gravity": np.array_equal(model.opt.gravity, parsed["gravity_m_s2"])}
    for item in (table, *parsed["bin"]):
        name = "task_table" if item is table else "task_bin_"+item["name"]
        geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        checks[name] = bool(np.array_equal(model.geom_size[geom], [x/2 for x in item["size_m"]]) and np.array_equal(model.geom_pos[geom], item["center_m"]) and np.allclose(model.geom_quat[geom], item["quaternion_wxyz"], rtol=0, atol=1e-12) and model.geom_friction[geom, 0] == item["sliding_friction"])
    for identifier, item in parsed["objects"].items():
        name = "task_object_"+identifier
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        checks[name] = bool(model.body_mass[body] == item["mass_kg"] and np.array_equal(model.body_inertia[body], item["diagonal_inertia_kg_m2"]) and np.array_equal(model.body_ipos[body], item["center_of_mass_m"]) and model.body_gravcomp[body] == 0 and np.array_equal(model.body_pos[body], item["position_m"]) and np.allclose(model.body_quat[body], item["quaternion_wxyz"], rtol=0, atol=1e-12))
        for piece in item["pieces"]:
            geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name+"_"+piece["name"])
            checks[name+"_"+piece["name"]] = bool(np.array_equal(model.geom_size[geom], [x/2 for x in piece["size_m"]]) and np.array_equal(model.geom_pos[geom], piece["position_m"]) and model.geom_friction[geom, 0] == item["sliding_friction"])
    if not all(checks.values()):
        raise ValueError("Compiled native mechanics differ from authored USD or pinned PiPER")
    receipt = {"schema": RECEIPT_SCHEMA, "recipe": recipe, "files": expected_files, "source_sha256": _sources(),
        "usd": parsed, "robot": robot_profiles.verify_assets("piper", assets), "compiled_checks": checks,
        "mapping": {"geometry": "Parsed local USD Cube full dimensions to native half extents; open bin has five independent colliders", "mass": "USD compound uniform-density mass/COM/diagonal inertia parsed unchanged; USD physics float precision retained", "friction": "Equal USD static/dynamic friction mapped to native isotropic sliding; native maximum pair combination", "contact": _CONTACT, "gravity": "authored -Z 9.81 m/s2", "state_velocity_frame": "mj_objectVelocity global/world frame; angular rad/s and linear m/s at body COM", "object_pose_frame": "world body origin at authored bottom centre; quaternion wxyz", "robot": "pinned native MJCF unchanged; six arm axes and one coupled-jaw command"},
        "scope": {"simulation_only": True, "physics_calibrated": False, "deformation": False, "learned_skill": False, "physical_robot_ready": False}}
    # Native check adds robot clearance to the pure support/overlap validation.
    initial = initialize(model, receipt)
    if any(item["robot_contact"] for item in sample_state(model, initial, receipt)["objects"].values()):
        raise ValueError("Object initially contacts the native robot")
    return model, receipt


def _require_receipt(receipt):
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("Native task-scene receipt required")
    return validate_recipe(receipt["recipe"])


def initialize(model, receipt):
    """Reset exact authored objects and home servo targets; no hidden stepping."""
    _require_receipt(receipt)
    return asset_library.initialize(model)


def sample_state(model, data, receipt):
    """Independent native observation, including contact partners and true velocity."""
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("Native task-scene receipt required")
    recipe = receipt["recipe"]
    mj, np = sim._native_imports()
    if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc, data.xpos, data.xquat, data.ctrl)):
        return {"time_s": float(data.time) if math.isfinite(float(data.time)) else None, "objects": {}, "warnings": ["nonfinite_native_state"], "valid": False}
    contacts = [(int(c.geom1), int(c.geom2)) for c in data.contact[:data.ncon]]
    robot_bodies = set(range(1, 10))
    robot_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in robot_bodies and model.geom_contype[i]}
    pads = set()
    for name in ("link7", "link8"):
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        pads.update(i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == body and int(model.geom_type[i]) == int(mj.mjtGeom.mjGEOM_BOX) and model.geom_contype[i])
    table = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "task_table")
    floor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "task_bin_floor")
    result = {}
    if table < 0 or floor < 0 or len(pads) != 4:
        return {"time_s": float(data.time), "objects": {}, "warnings": ["missing_native_scene_channels"], "valid": False}
    for obj in recipe["objects"]:
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "task_object_"+obj["id"])
        if body < 0:
            return {"time_s": float(data.time), "objects": {}, "warnings": ["missing_native_object"], "valid": False}
        geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == body}
        partners = {b if a in geoms else a for a, b in contacts if (a in geoms) != (b in geoms)}
        velocity = np.zeros(6)
        mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body, velocity, 0)
        result[obj["id"]] = {"position_m": data.xpos[body].tolist(), "quaternion_wxyz": data.xquat[body].tolist(),
            "linear_velocity_m_s": velocity[3:].tolist(), "angular_velocity_rad_s": velocity[:3].tolist(),
            "table_contact": table in partners, "bin_floor_contact": floor in partners,
            "robot_contact": bool(partners & robot_geoms), "pad_contact": bool(partners & pads),
            "contact_geom_names": sorted(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) or ("unnamed_"+str(i)) for i in partners)}
    warnings = ["native_warning_"+str(i) for i, warning in enumerate(data.warning) if warning.number]
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        warnings.append("external_force_applied")
    return {"time_s": float(data.time), "objects": result, "warnings": warnings, "valid": not warnings}


class GoalMonitor:
    """All objects contained, released, floor-supported and slow for full dwell.

    Call after every 1 ms native step. A gap/reset/nonfinite state/solver warning
    clears dwell and reports unknown. Repeated identical timestamps add no time.
    No action/policy label is consulted. Reset explicitly between independent runs.
    """
    def __init__(self, receipt):
        self.recipe = _require_receipt(receipt)
        self.receipt = json.loads(json.dumps(receipt, allow_nan=False))
        self.previous = None
        self.since = None

    def reset(self):
        self.previous = None
        self.since = None

    def update(self, model, data):
        state = sample_state(model, data, self.receipt)
        now = state["time_s"]
        reason = None
        if now is None or not math.isfinite(now) or now < 0 or not state["valid"]:
            reason = "invalid_native_state"
        elif self.previous is not None and now < self.previous-1e-12:
            reason = "time_reset"
        elif self.previous is not None and now-self.previous > self.recipe["timestep_s"]*1.5:
            reason = "unobserved_interval"
        self.previous = now
        results = {}
        if reason is None:
            goal = self.recipe["goal"]
            for obj in self.recipe["objects"]:
                item = state["objects"][obj["id"]]
                measured = containment(obj, item["position_m"], item["quaternion_wxyz"], self.recipe["bin"], goal["containment_margin_m"])
                result = {"contained": measured["contained"], "bin_floor_contact": item["bin_floor_contact"],
                    "released": not item["robot_contact"] and not item["pad_contact"],
                    "linear_speed": math.sqrt(sum(x*x for x in item["linear_velocity_m_s"])) <= goal["max_linear_speed_m_s"],
                    "angular_speed": math.sqrt(sum(x*x for x in item["angular_velocity_rad_s"])) <= goal["max_angular_speed_rad_s"]}
                result["instantaneous"] = all(result.values())
                results[obj["id"]] = result
        instantaneous = reason is None and bool(results) and all(item["instantaneous"] for item in results.values())
        if not instantaneous:
            self.since = None
        elif self.since is None:
            self.since = now
        duration = 0. if self.since is None else now-self.since
        return {"time_s": now, "objects": results, "instantaneous": instantaneous,
            "settled_s": duration, "success": instantaneous and duration+1e-12 >= self.recipe["goal"]["settle_s"],
            "unknown": reason is not None, "reason": reason}
