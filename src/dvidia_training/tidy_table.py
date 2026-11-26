"""Remember and restore a bounded simulated tabletop using observed zone occupancy."""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import random
from time import perf_counter
from typing import Mapping

OBJECTS = ("A", "B", "C")
ZONES = ("left", "center", "right", "buffer")
HOME = dict(zip(OBJECTS, ZONES))
REFERENCE_SCHEMA = "dvidia.table-reference/v1"
RUN_SCHEMA = "dvidia.tidy-table-run/v1"
MOVE_LIMIT = 8


def _digest(data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_slots(observation):
    if not isinstance(observation, Mapping) or set(observation) != set(OBJECTS):
        raise ValueError("Supply one zone reading for each of A, B and C.")
    values = tuple(observation[key] for key in OBJECTS)
    if any(value is not None and value not in ZONES for value in values):
        raise ValueError("A zone reading must name left, center, right, buffer or be null.")
    return values


def _uncertainty(values):
    if None in values:
        return "At least one object could not be located in a zone."
    if len(set(values)) != len(values):
        return "Multiple objects occupy the same zone reading."
    return None


def _make_reference(observation, interchangeable):
    slots = _read_slots(observation)
    problem = _uncertainty(slots)
    if problem:
        raise ValueError(problem)
    if "buffer" in slots:
        raise ValueError("Keep the buffer free in the remembered arrangement.")
    groups = []
    used = set()
    for group in interchangeable:
        if not isinstance(group, (list, tuple)) or not 2 <= len(group) <= len(OBJECTS):
            raise ValueError("Use groups of two or three interchangeable objects.")
        if any(obj not in OBJECTS for obj in group) or len(set(group)) != len(group):
            raise ValueError("Each interchangeable group requires distinct known object labels.")
        if used.intersection(group):
            raise ValueError("An object can appear in only one interchangeable group.")
        used.update(group)
        groups.append(sorted(group))
    groups.sort()
    allowed = {obj: [zone] for obj, zone in zip(OBJECTS, slots)}
    for group in groups:
        zones = [zone for zone in ZONES if zone in {observation[obj] for obj in group}]
        for obj in group:
            allowed[obj] = zones.copy()
    return dict(schema=REFERENCE_SCHEMA, remembered=dict(zip(OBJECTS, slots)),
                interchangeable=groups, allowed_zones=allowed)


def remember_table(observation, interchangeable=()):
    """Save an unambiguous approved observation and explicit interchangeable-object rules."""
    reference = _make_reference(observation, interchangeable)
    return {**reference, "id": _digest(reference)}


def validate_reference(reference):
    if not isinstance(reference, Mapping) or reference.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("Unrecognized table reference.")
    try:
        expected = remember_table(reference["remembered"], reference["interchangeable"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete table reference.") from exc
    if reference != expected:
        raise ValueError("The reference is inconsistent with its approved arrangement or digest.")
    return expected


def plan_table(reference, observation):
    """Find the shortest single-object move sequence among four single-occupancy zones."""
    allowed = validate_reference(reference)["allowed_zones"]
    start = _read_slots(observation)
    problem = _uncertainty(start)
    if problem:
        return dict(status="unknown", moves=[], reason=problem)
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        state, moves = queue.popleft()
        if all(zone in allowed[obj] for obj, zone in zip(OBJECTS, state)):
            return dict(status="moves" if moves else "tidy", moves=moves,
                        reason="A shortest zone sequence is available." if moves else
                               "Every object occupies an approved zone.")
        free = next(zone for zone in ZONES if zone not in state)
        for i, obj in enumerate(OBJECTS):
            new = state[:i] + (free,) + state[i+1:]
            if new not in seen:
                seen.add(new)
                step = {"object": obj, "from": state[i], "to": free}
                queue.append((new, moves + [step]))
    raise RuntimeError("No path in the four-zone tabletop.")


def _initial_slots(initial):
    slots = _read_slots(initial)
    problem = _uncertainty(slots)
    if problem:
        raise ValueError(problem)
    return dict(zip(OBJECTS, slots))


def _offsets(seed, supplied):
    if supplied is None:
        rng = random.Random(seed)
        return {obj: [round(rng.uniform(-.004, .004), 6) for _ in range(2)]
                for obj in OBJECTS}
    if set(supplied) != set(OBJECTS):
        raise ValueError("Each object requires its own XY offset.")
    for coords in supplied.values():
        if (not isinstance(coords, (list, tuple)) or len(coords) != 2
                or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                       or abs(v) > .004 for v in coords)):
            raise ValueError("Tabletop offsets must be finite XY metres within ±0.004 m.")
    return {obj: list(xy) for obj, xy in supplied.items()}


def _native_factory():
    from .tidy_table_native import NativeTable
    return NativeTable


def capture_table(initial, *, assets, seed=0, interchangeable=(), offsets_xy=None, frame_sink=None):
    """Fit a reference to a rendered observation, refusing incomplete or ambiguous layouts."""
    layout = _initial_slots(initial)
    offsets = _offsets(seed, offsets_xy)
    started = perf_counter()
    with _native_factory()(assets, layout, offsets_xy=offsets) as sim:
        observation = sim.observe()
        if frame_sink is not None:
            frame_sink(sim.last_frame)
    return dict(reference=remember_table(observation, interchangeable),
                observation=observation, seed=seed, offsets_xy=offsets,
                capture_wall_ms=(perf_counter() - started)*1000)


def run_table_case(reference, initial, *, assets, seed=0, offsets_xy=None, frame_sink=None):
    """Execute one no-retry episode in one persistent native scene, reobserving after moves."""
    approved = validate_reference(reference)
    layout = _initial_slots(initial)
    offsets = _offsets(seed, offsets_xy)
    started = perf_counter()

    def elapsed():
        return (perf_counter()-started)*1000

    result = dict(schema=RUN_SCHEMA, reference_id=approved["id"], seed=seed,
                  initial=layout, offsets_xy=offsets, status="failed", success=False,
                  moves=[], decisions=[], first_decision_ms=None, time_to_first_move_ms=None,
                  elapsed_wall_ms=None, native_score=None, camera_claimed_tidy=False,
                  timing_scope="Wall clock from local run after input validation through scene setup, RGB observation, planning and native actuation. Excludes historical reference capture, process startup and UI transport.")
    seen = set()
    try:
        with _native_factory()(assets, layout, offsets_xy=offsets) as sim:
            result["setup_wall_ms"] = elapsed()
            while True:
                observation = sim.observe()
                if frame_sink is not None:
                    frame_sink(sim.last_frame)
                began_plan = perf_counter()
                plan = plan_table(approved, observation)
                decision = dict(observation=dict(observation), plan=plan,
                                planner_wall_ms=(perf_counter()-began_plan)*1000,
                                elapsed_wall_ms=elapsed())
                result["decisions"].append(decision)
                if result["first_decision_ms"] is None:
                    result["first_decision_ms"] = decision["elapsed_wall_ms"]
                if plan["status"] == "unknown":
                    result.update(status="unknown", reason=plan["reason"])
                    break
                if plan["status"] == "tidy":
                    result["camera_claimed_tidy"] = True
                    break
                signature = tuple(observation[obj] for obj in OBJECTS)
                if signature in seen or len(result["moves"]) >= MOVE_LIMIT:
                    result["reason"] = "The arrangement repeated or exhausted its move budget."
                    break
                seen.add(signature)
                step = plan["moves"][0]
                move_start = elapsed()
                # A native exception can occur after commands were issued. Preserve
                # that attempted move even when its first-motion timestamp is lost.
                pending = dict(requested=step, call_start_elapsed_ms=move_start,
                               verified=False, first_motion_ms=None, motion_observation="unavailable")
                result["moves"].append(pending)
                steps_before = getattr(sim, "control_steps", None)
                try:
                    receipt = sim.move(step["object"], step["to"])
                except Exception as exc:
                    pending["error"] = f"{type(exc).__name__}: {exc}"
                    steps_after = getattr(sim, "control_steps", None)
                    if steps_before is not None and steps_after is not None:
                        pending["control_steps"] = steps_after - steps_before
                    result.update(status="error", reason=pending["error"])
                    break
                pending.update(receipt)
                pending["motion_observation"] = ("measured" if receipt["first_motion_ms"] is not None
                                                 else "threshold_not_reached")
                relative = receipt["first_motion_ms"]
                if result["time_to_first_move_ms"] is None and relative is not None:
                    result["time_to_first_move_ms"] = move_start + relative
                if not receipt["verified"]:
                    result["reason"] = receipt.get("reason") or "The native move failed verification."
                    break
            result["native_score"] = sim.score(approved["allowed_zones"])
            if result["camera_claimed_tidy"]:
                if not result["moves"] and result["native_score"]["issued_control_steps"] != 0:
                    result["reason"] = "The camera reported tidy after unexpected native control steps."
                elif result["native_score"]["success"]:
                    result.update(status="completed" if result["moves"] else "tidy",
                                  success=True, reason="The final table passed native verification.")
                else:
                    result["reason"] = "The camera goal disagreed with the independent native scorer."
    except Exception as exc:
        result.update(status="error", reason=f"{type(exc).__name__}: {exc}", success=False)
    if result["time_to_first_move_ms"] is not None:
        result["motion_observation"] = "measured"
    elif result["moves"]:
        result["motion_observation"] = result["moves"][-1]["motion_observation"]
    else:
        result["motion_observation"] = "no_move_requested"
    result["elapsed_wall_ms"] = elapsed()
    return result
