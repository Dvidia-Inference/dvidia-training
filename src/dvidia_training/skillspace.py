"""Read actual DVIDIA exports and bind an explicitly authored simulation task.

Skillspace briefs and vdia:0 skill metadata do not contain robot state, actions,
or calibrated geometry.  This bridge preserves that boundary: a source alone
can be inspected, but execution requires a separate simulation grounding.
No URLs, media, imports, or source-provided executable code are evaluated here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any


MAX_DOCUMENT_BYTES = 262_144
SUPPORTED_SKILLS = ("place_cup", "place_object")
GROUNDING_VERSION = 1
ARM_PROFILE = "dvidia-authored-6dof-parallel-jaw-v0"
ADAPTER_ID = "contact-pick-place-box-v0"
TASK_ID = "ArmPickPlace-v0"
PROCEDURE = ("approach", "descend", "close", "lift", "transfer", "place", "release", "retreat", "verify")
WORKSPACE_MIN = (0.25, -0.20, 0.29)
WORKSPACE_MAX = (0.60, 0.20, 0.55)
GROUNDING_FIELDS = ("schema_version", "task_type", "provenance", "adapter_id", "arm_profile",
                    "frame", "units", "language_aliases", "workspace", "table_height", "object",
                    "target", "procedure")
_ALIASES = {
    "place cup": "place_cup", "place a cup": "place_cup",
    "place cup on the mark": "place_cup", "place a cup on the mark": "place_cup",
    "set a cup on the mark": "place_cup",
    "place object": "place_object", "place an object": "place_object",
    "place object on the mark": "place_object", "place an object on the mark": "place_object",
    "set an object on the mark": "place_object",
}


class SkillspaceError(ValueError):
    """A stable, JSON-reportable parse or unsupported-task failure."""

    def __init__(self, code: str, message: str, fields: tuple[str, ...] = ()):
        super().__init__(message)
        self.code = code
        self.fields = fields

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "fields": list(self.fields)}


def _invalid(path: str, message: str) -> None:
    raise SkillspaceError("invalid_document", f"{path}: {message}", (path,))


def _record(value: Any, path: str, *, required: tuple[str, ...] = (),
            allowed: tuple[str, ...] | None = None) -> dict[str, Any]:
    if type(value) is not dict:
        _invalid(path, "expected an object")
    if any(type(key) is not str for key in value):
        _invalid(path, "object keys must be strings")
    missing = set(required) - value.keys()
    if missing:
        _invalid(path, "missing fields: " + ", ".join(sorted(missing)))
    if allowed is not None and set(value) - set(allowed):
        _invalid(path, "unknown fields: " + ", ".join(sorted(set(value) - set(allowed))))
    return value


def _text(value: Any, path: str, maximum: int = 240, *, blank: bool = False) -> str:
    if type(value) is not str or len(value) > maximum or (not blank and not value.strip()):
        _invalid(path, f"expected {'0' if blank else '1'}–{maximum} characters")
    if any((ord(char) < 32 and char not in "\t\n\r") or 127 <= ord(char) <= 159 or 0xD800 <= ord(char) <= 0xDFFF
           for char in value):
        _invalid(path, "control characters or unpaired surrogates are not allowed")
    return value.strip()


def _number(value: Any, path: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not minimum <= value <= maximum or not math.isfinite(value):
        _invalid(path, f"expected a finite number in [{minimum}, {maximum}]")
    return float(value)


def _integer(value: Any, path: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _invalid(path, f"expected an integer in [{minimum}, {maximum}]")
    return value


def _vector(value: Any, path: str, minimum: float, maximum: float) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        _invalid(path, "expected three numeric coordinates")
    return tuple(_number(item, f"{path}[{index}]", minimum, maximum)
                 for index, item in enumerate(value))


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise SkillspaceError("invalid_document", "Source must be finite UTF-8 JSON.") from exc


def _json_input(document: dict[str, Any] | str | bytes) -> tuple[dict[str, Any], str, str]:
    if type(document) is dict:
        raw = _canonical(document)
        value = document
        hash_kind = "canonical_json"
    elif type(document) in (str, bytes):
        try:
            raw = document.encode("utf-8") if type(document) is str else document
        except UnicodeError as exc:
            raise SkillspaceError("invalid_document", "Source must be valid UTF-8.") from exc
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise SkillspaceError("document_too_large", "Source exceeds the 256 KiB JSON limit.")

        def unique_object(pairs):
            row = {}
            for key, value in pairs:
                if key in row:
                    _invalid(key, "duplicate JSON key")
                row[key] = value
            return row

        def nonfinite(token):
            _invalid("JSON", f"nonfinite constant {token} is not allowed")

        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                               parse_constant=nonfinite)
        except (ValueError, UnicodeError, RecursionError) as exc:
            if isinstance(exc, SkillspaceError):
                raise
            raise SkillspaceError("invalid_document", "Source must be valid finite UTF-8 JSON.") from exc
        _canonical(value)  # Also rejects overflowed floating-point tokens such as 1e999.
        hash_kind = "exact_bytes"
    else:
        raise SkillspaceError("invalid_document", "Expected JSON bytes, text, or a plain dict.")
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise SkillspaceError("document_too_large", "Source exceeds the 256 KiB JSON limit.")
    return _record(value, "source"), hashlib.sha256(raw).hexdigest(), hash_kind


def _skill_id(value: str) -> str:
    normalized = re.sub(r"[ _-]+", " ", value.strip().lower().rstrip("."))
    identifier = _ALIASES.get(normalized)
    if identifier is None:
        raise SkillspaceError("unsupported_task", "Only place_cup and place_object are supported; "
                              f"{value!r} has no compatible placement adapter.", ("skill",))
    return identifier


def _source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("vdia") == "0":
        _record(source, "source", required=("skill", "version", "task"))
        skill = _skill_id(_text(source["skill"], "skill", 100))
        if "rail" in source and source["rail"] != "physical":
            raise SkillspaceError("unsupported_task", "Only physical placement metadata is supported.")
        return {"source_format": "vdia:0", "skill_id": skill, "name": source["skill"],
                "task": _text(source["task"], "task", 1200),
                "version": _text(source["version"], "version", 40),
                "source_has_robot_policy": False, "permissions": "not_inferred"}
    if source.get("format") == "dvidia-contribution-brief":
        fields = ("format", "schemaVersion", "state", "skillspaceTitle", "brief",
                  "creatorIdentity", "permissions", "commercialProposal")
        _record(source, "source", required=fields, allowed=fields + ("simulation_grounding",))
        if type(source["schemaVersion"]) is not int or source["schemaVersion"] != 1 \
                or source["state"] != "local-proposal" \
                or source["creatorIdentity"] != "self-declared-local":
            _invalid("source", "unsupported contribution-brief version or identity state")
        brief_fields = ("id", "skillspaceId", "version", "taskId", "taskTitle", "goal",
                        "captureChecklist", "creatorHandle", "createdAt")
        brief = _record(source["brief"], "brief", required=brief_fields, allowed=brief_fields)
        for key, prefix in (("id", "brief"), ("skillspaceId", "space"), ("taskId", "task")):
            if type(brief[key]) is not str or not re.fullmatch(
                    prefix + r"-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", brief[key]):
                _invalid(f"brief.{key}", "expected the immutable DVIDIA ID")
        _integer(brief["version"], "brief.version", 1, 60)
        title = _text(brief["taskTitle"], "brief.taskTitle", 100)
        skill = _skill_id(title)
        _text(brief["goal"], "brief.goal", 1200)
        handle = _text(brief["creatorHandle"], "brief.creatorHandle", 32, blank=True)
        if handle and not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", handle):
            _invalid("brief.creatorHandle", "invalid local handle")
        checklist = brief["captureChecklist"]
        if type(checklist) is not list or not 1 <= len(checklist) <= 12:
            _invalid("brief.captureChecklist", "expected 1–12 items")
        for index, item in enumerate(checklist):
            _text(item, f"brief.captureChecklist[{index}]", 180)
        timestamp = _text(brief["createdAt"], "brief.createdAt", 40)
        try:
            parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as exc:
            raise SkillspaceError("invalid_document", "brief.createdAt: invalid UTC timestamp") from exc
        if parsed.strftime("%Y-%m-%dT%H:%M:%S.") + f"{parsed.microsecond // 1000:03d}Z" != timestamp:
            _invalid("brief.createdAt", "expected canonical UTC millisecond timestamp")
        permissions = _record(source["permissions"], "permissions",
                              required=("publicReuseGranted", "contributorPermissionRequired"),
                              allowed=("publicReuseGranted", "contributorPermissionRequired"))
        if permissions != {"publicReuseGranted": False, "contributorPermissionRequired": True}:
            _invalid("permissions", "unsupported local-brief permission state")
        expected_terms = {"status": "draft", "premium": "paid", "price": None,
                          "necklace": "planned-included", "hardware": "TBA",
                          "proposedFounderShareBps": 8000, "revenueDenominator": None,
                          "remainingShareRecipients": None, "contributorPermissionRequired": True,
                          "sourceAccess": "open"}
        terms = _record(source["commercialProposal"], "commercialProposal",
                        required=tuple(expected_terms), allowed=tuple(expected_terms))
        if any(type(terms[key]) is not type(value) or terms[key] != value
               for key, value in expected_terms.items()):
            _invalid("commercialProposal", "unsupported draft commercial terms")
        return {"source_format": "dvidia-contribution-brief:1", "skill_id": skill,
                "name": _text(source["skillspaceTitle"], "skillspaceTitle", 80),
                "task": title, "version": str(brief["version"]), "brief_id": brief["id"],
                "skillspace_id": brief["skillspaceId"], "task_id": brief["taskId"],
                "source_has_robot_policy": False, "permissions": "local_only_no_public_reuse"}
    raise SkillspaceError("unsupported_format", "Expected a vdia:0 skill.json or a "
                          "dvidia-contribution-brief schemaVersion:1 export.")


@dataclass(frozen=True)
class GroundedSkill:
    """A validated, simulation-only placement contract, never a learned policy."""

    skill_id: str
    source_name: str
    source_task: str
    source_format: str
    source_version: str
    source_permissions: str
    source_sha256: str
    source_hash_kind: str
    grounding_sha256: str
    source_ids: tuple[tuple[str, str], ...]
    arm_profile: str
    adapter_id: str
    task_id: str
    provenance: str
    frame: str
    units: str
    language_aliases: tuple[str, ...]
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    table_height: float
    object_kind: str
    object_position: tuple[float, float, float]
    object_size: tuple[float, float, float]
    object_mass: float
    object_friction: float
    target_position: tuple[float, float, float]
    position_tolerance: float
    dwell_seconds: float
    procedure: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        # Round-tripping gives JSON arrays rather than Python tuple assumptions.
        result = json.loads(json.dumps(asdict(self), allow_nan=False))
        result["source_ids"] = dict(self.source_ids)
        result.update({"simulation_ready": True, "robot_ready": False,
                       "source_has_robot_policy": False,
                       "object_representation": "authored_box_proxy"})
        return result

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()


def _grounding(source: dict[str, Any], sidecar: dict[str, Any] | None) -> dict[str, Any] | None:
    if sidecar is not None and "simulation_grounding" in source:
        raise SkillspaceError("ambiguous_grounding", "Supply inline grounding or a sidecar, not both.")
    value = sidecar if sidecar is not None else source.get("simulation_grounding")
    if value is None:
        return None
    _canonical(value)
    return _record(value, "simulation_grounding", required=GROUNDING_FIELDS, allowed=GROUNDING_FIELDS)


def load_skillspace(document: dict[str, Any] | str | bytes,
                   grounding: dict[str, Any] | None = None) -> GroundedSkill:
    """Validate a real source plus an explicit local grounding, with no inference.

    Bytes/text preserve the exact uploaded source SHA-256; dict inputs are hashed
    as canonical JSON, explicitly identified in source_hash_kind. A sidecar is
    authored scene/controller binding, not source-derived supervision or consent.
    """
    source, source_hash, hash_kind = _json_input(document)
    metadata = _source_metadata(source)
    row = _grounding(source, grounding)
    if row is None:
        raise SkillspaceError("grounding_required", "This source is task metadata. Bind an explicitly "
                              "authored simulation-only placement adapter and metric scene before execution.",
                              tuple("simulation_grounding." + field for field in GROUNDING_FIELDS))
    if type(row["schema_version"]) is not int or row["schema_version"] != GROUNDING_VERSION:
        _invalid("simulation_grounding.schema_version", "only version 1 is supported")
    if row["task_type"] not in SUPPORTED_SKILLS or row["task_type"] != metadata["skill_id"]:
        raise SkillspaceError("unsupported_task", "Grounding task_type must match the supported source skill.",
                              ("simulation_grounding.task_type",))
    for field, expected in (("provenance", "authored_simulation_adapter"), ("adapter_id", ADAPTER_ID),
                            ("arm_profile", ARM_PROFILE), ("frame", "world"), ("units", "m")):
        if row[field] != expected:
            _invalid(f"simulation_grounding.{field}", f"only {expected!r} is supported")
    aliases = row["language_aliases"]
    if type(aliases) is not list or not 1 <= len(aliases) <= 12:
        _invalid("simulation_grounding.language_aliases", "expected 1–12 language aliases")
    aliases = tuple(_text(alias, f"language_aliases[{index}]", 120)
                    for index, alias in enumerate(aliases))
    if len({alias.casefold() for alias in aliases}) != len(aliases):
        _invalid("simulation_grounding.language_aliases", "duplicate aliases")
    workspace = _record(row["workspace"], "workspace", required=("min", "max"), allowed=("min", "max"))
    minimum = _vector(workspace["min"], "workspace.min", -1, 1)
    maximum = _vector(workspace["max"], "workspace.max", -1, 1)
    if minimum != WORKSPACE_MIN or maximum != WORKSPACE_MAX:
        _invalid("simulation_grounding.workspace", "must match the authored arm profile workspace")
    table_height = _number(row["table_height"], "table_height", 0.29, 0.29)
    object_fields = ("kind", "position", "size", "mass", "friction")
    obj = _record(row["object"], "object", required=object_fields, allowed=object_fields)
    if obj["kind"] != "box":
        _invalid("object.kind", "only an explicitly declared box proxy is supported")
    size = _vector(obj["size"], "object.size", 0.03, 0.05)
    mass = _number(obj["mass"], "object.mass", 0.02, 0.08)
    friction = _number(obj["friction"], "object.friction", 0.4, 1.5)
    position = _vector(obj["position"], "object.position", -1, 1)
    target_fields = ("position", "position_tolerance", "dwell_seconds")
    target = _record(row["target"], "target", required=target_fields, allowed=target_fields)
    target_position = _vector(target["position"], "target.position", -1, 1)
    tolerance = _number(target["position_tolerance"], "target.position_tolerance", 0.01, 0.035)
    dwell = _number(target["dwell_seconds"], "target.dwell_seconds", 0.15, 0.6)
    supported_z = table_height + size[2] / 2
    for name, point in (("object.position", position), ("target.position", target_position)):
        if not minimum[0] <= point[0] <= maximum[0] or not minimum[1] <= point[1] <= maximum[1] \
                or not supported_z - 0.01 <= point[2] <= supported_z + 0.01:
            _invalid(name, "center must be reachable and on the declared support plane (within 10 mm)")
        if math.hypot(point[0], point[1]) > 0.56:
            _invalid(name, "center exceeds the authored raised-tool reach radius of 0.56 m")
    if math.dist(position[:2], target_position[:2]) < 0.075:
        _invalid("target.position", "placement must move the object at least 75 mm")
    if type(row["procedure"]) is not list or tuple(row["procedure"]) != PROCEDURE:
        _invalid("simulation_grounding.procedure", "must declare the supported authored placement phases")
    return GroundedSkill(
        skill_id=metadata["skill_id"], source_name=metadata["name"], source_task=metadata["task"],
        source_format=metadata["source_format"], source_version=metadata["version"],
        source_permissions=metadata["permissions"], source_sha256=source_hash,
        source_hash_kind=hash_kind, grounding_sha256=hashlib.sha256(_canonical(row)).hexdigest(),
        source_ids=tuple((key, metadata[key]) for key in ("brief_id", "skillspace_id", "task_id")
                         if key in metadata), arm_profile=ARM_PROFILE, adapter_id=ADAPTER_ID, task_id=TASK_ID,
        provenance="authored_simulation_adapter", frame="world", units="m", language_aliases=aliases,
        workspace_min=minimum, workspace_max=maximum, table_height=table_height, object_kind="box",
        object_position=position, object_size=size, object_mass=mass, object_friction=friction,
        target_position=target_position, position_tolerance=tolerance, dwell_seconds=dwell,
        procedure=PROCEDURE,
    )


def inspect_skillspace(document: dict[str, Any] | str | bytes,
                      grounding: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect without fetching files or executing source commands.

    Invalid/unsupported sources raise SkillspaceError. Supported metadata-only
    sources return grounding_required; a supplied grounding is fully validated.
    """
    source, source_hash, hash_kind = _json_input(document)
    metadata = _source_metadata(source)
    result = {**metadata, "valid": True, "source_sha256": source_hash,
              "source_hash_kind": hash_kind, "robot_ready": False,
              "simulation_ready": False, "readiness": "grounding_required",
              "missing_grounding": list(GROUNDING_FIELDS),
              "supported_adapter": ADAPTER_ID, "compatible_arm_profile": ARM_PROFILE}
    if _grounding(source, grounding) is not None:
        skill = load_skillspace(document, grounding)
        result.update({"simulation_ready": True, "readiness": "simulation_adapter_bound",
                       "missing_grounding": [], "grounding_sha256": skill.grounding_sha256,
                       "provenance": skill.provenance, "task_id": skill.task_id})
    return result
