"""CableEndReach-v0: ideal endpoint-force control of an anchored elastic cable.

Native MuJoCo CPU dynamics, privileged state, SI units, no rendering/networking.
The endpoint attachment is ideal: this task does not model jaws, grasp slip,
camera observations, tactile sensing, axial stretch or physical robot transfer.
Material/contact parameters are authored simulation settings, not calibration.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import threading
from typing import Any

import mujoco
import numpy as np

PIN = "3.15.0"
ENV_ID = "CableEndReach-v0"
STATE_SIGNATURE = int(mujoco.mjtState.mjSTATE_INTEGRATION)
_NATIVE_LOCK = threading.RLock()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    timestep: float = 0.001
    control_dt: float = 0.02
    horizon: float = 6.0
    dwell: float = 0.2
    tolerance: float = 0.025
    force_limit: float = 2.0
    rope_length: float = 0.5
    rope_radius: float = 0.004
    rope_nodes: int = 21
    density: float = 1000.0
    bend: float = 1e5
    twist: float = 1e5
    joint_damping: float = 0.03
    gravity: float = 9.81
    variation_fraction: float = 0.08
    settle_time: float = 1.0
    anchor_height: float = 0.7
    target_offset: tuple[float, float, float] = (0.06, 0.05, 0.04)
    target_jitter: float = 0.005
    contact_friction: float = 0.5
    workspace_radius_factor: float = 1.1

    def __post_init__(self):
        for name in ("timestep", "control_dt", "horizon", "dwell", "tolerance",
                     "force_limit", "rope_length", "rope_radius", "density",
                     "bend", "twist", "joint_damping", "settle_time",
                     "anchor_height", "workspace_radius_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("gravity", "target_jitter", "contact_friction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.rope_nodes, int) or isinstance(self.rope_nodes, bool) or not 5 <= self.rope_nodes <= 65:
            raise ValueError("rope_nodes must be an integer in [5, 65]")
        if not math.isfinite(self.variation_fraction) or not 0 <= self.variation_fraction <= 0.2:
            raise ValueError("variation_fraction must lie in [0, 0.2]")
        for numerator, denominator, label in ((self.control_dt, self.timestep, "control_dt/timestep"),
                                               (self.horizon, self.control_dt, "horizon/control_dt"),
                                               (self.settle_time, self.timestep, "settle_time/timestep")):
            ratio = numerator / denominator
            if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"{label} must be a positive integer")
        if self.control_dt < self.timestep or self.horizon < self.control_dt or self.dwell > self.horizon:
            raise ValueError("invalid timing range")
        if self.rope_radius * 4 >= self.rope_length / (self.rope_nodes - 1):
            raise ValueError("rope radius must be smaller than one quarter of segment length")
        offset = np.asarray(self.target_offset, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("target_offset must be three finite SI coordinates")
        object.__setattr__(self, "target_offset", tuple(float(x) for x in offset))


class CableEndReachEnv:
    """Seeded feedback task with actual endpoint dwell and explicit failures.

    ``reset(seed) -> (observation, info)``
    ``step(force) -> (observation, reward, terminated, truncated, info)``

    Force is a world-frame 3-vector in newtons. Invalid actions raise ValueError
    before changing state. Stepping after termination/truncation raises RuntimeError.
    ``snapshot`` and ``restore`` preserve the complete integration state plus
    episode counters; snapshots are JSON-compatible and restricted to this config.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        if not isinstance(self.config, Config):
            raise TypeError("config must be Config")
        if mujoco.mj_versionString() != PIN or mujoco.__version__ != PIN:
            raise RuntimeError(f"This task requires native MuJoCo {PIN}")
        self.config_hash = _digest(asdict(self.config))
        self.model = None
        self.data = None
        self._ready = False
        self._done = False

    @contextmanager
    def _warnings(self, collected: list[str]):
        # MuJoCo callbacks are process-global. Serialize our native sections and
        # restore any preexisting callback without modifying global configuration.
        with _NATIVE_LOCK:
            previous = mujoco.get_mju_user_warning()
            mujoco.set_mju_user_warning(lambda message: collected.append(str(message)))
            try:
                yield
            finally:
                mujoco.set_mju_user_warning(previous)

    def _make_scene(self, parameters: dict[str, float]):
        c = self.config
        xml = f'''<mujoco model="{ENV_ID}">
        <compiler angle="radian"/>
        <option timestep="{c.timestep:.17g}" gravity="0 0 {-c.gravity:.17g}"
          solver="Newton" iterations="100" tolerance="1e-10"
          integrator="implicitfast" cone="elliptic"/>
        <size memory="64M"/>
        <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
        <worldbody>
          <composite type="cable" curve="s" count="{c.rope_nodes} 1 1"
            size="{parameters['rope_length']:.17g}" offset="0 0 {c.anchor_height:.17g}" initial="none">
            <plugin plugin="mujoco.elasticity.cable">
              <config key="bend" value="{parameters['bend']:.17g}"/>
              <config key="twist" value="{parameters['twist']:.17g}"/>
            </plugin>
            <joint kind="main" damping="{c.joint_damping:.17g}"/>
            <geom type="capsule" size="{parameters['rope_radius']:.17g}"
              density="{parameters['density']:.17g}" friction="{c.contact_friction:.17g} 0 0"
              solref="0.02 1" solimp="0.9 0.95 0.001" condim="3"/>
          </composite>
        </worldbody></mujoco>'''
        spec = mujoco.MjSpec.from_string(xml)
        last = spec.body("B_last")
        if last is None or len(last.geoms) != 1:
            raise RuntimeError("Installed cable generator has an unsupported endpoint layout")
        # The generated final capsule has an explicit local from/to centerline.
        # Place the force site at its actual far endpoint, not at body COM.
        tip_local = np.asarray(last.geoms[0].fromto[3:], dtype=float)
        if not np.isfinite(tip_local).all() or np.linalg.norm(tip_local) <= 0:
            raise RuntimeError("Cable endpoint centerline could not be identified")
        last.add_site(name="rope_tip", pos=tip_local, size=[parameters["rope_radius"]], rgba=[0.2, 0.6, 0.9, 1])
        self.scene_xml = spec.to_xml()
        model = spec.compile()
        return model

    def reset(self, seed: int = 0):
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self._ready = False
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        c = self.config
        self.parameters = {name: float(getattr(c, name) * rng.uniform(1 - c.variation_fraction, 1 + c.variation_fraction))
                           for name in ("rope_length", "rope_radius", "density", "bend", "twist")}
        if 4 * self.parameters["rope_radius"] >= self.parameters["rope_length"] / (c.rope_nodes - 1):
            raise ValueError("Seeded radius/length violates the segment-clearance profile")
        warnings = []
        with self._warnings(warnings):
            self.model = self._make_scene(self.parameters)
            self.data = mujoco.MjData(self.model)
            self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "rope_tip")
            self.tip_body_id = int(self.model.site_bodyid[self.tip_site_id])
            self._rope_body_ids = np.arange(1, self.model.nbody, dtype=int)
            self.anchor = self.model.body_pos[1].copy()
            self._last_action = np.zeros(3)
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(round(c.settle_time / c.timestep)):
                before = float(self.data.time)
                mujoco.mj_step(self.model, self.data)
                if not self._finite() or not math.isclose(float(self.data.time) - before, c.timestep, rel_tol=0, abs_tol=1e-10):
                    raise RuntimeError("Cable settling diverged or silently reset")
            mujoco.mj_forward(self.model, self.data)
        if warnings or self._warning_counts():
            raise RuntimeError(f"Cable settling produced warnings: {warnings or self._warning_counts()}")
        jitter = rng.uniform(-c.target_jitter, c.target_jitter, 3)
        self.target = self.data.site_xpos[self.tip_site_id].copy() + np.asarray(c.target_offset) + jitter
        # First segment is fixed. The remaining rope cannot leave the sphere
        # centered at its first movable joint. This is a conservative reach check,
        # not proof that the controller can reach every accepted target.
        first_movable = self.data.xpos[2].copy()
        fixed_segment = float(np.linalg.norm(first_movable - self.anchor))
        free_length = self.parameters["rope_length"] - fixed_segment
        if np.linalg.norm(self.target - first_movable) > free_length * 0.98:
            raise ValueError("target is outside the conservative anchored-cable reach envelope")
        if np.linalg.norm(self.target - self.data.site_xpos[self.tip_site_id]) <= c.tolerance:
            raise ValueError("target must require nontrivial endpoint movement")
        self.scene_hash = _digest({"mjcf": self.scene_xml, "target": self.target.tolist(), "parameters": self.parameters})
        self._elapsed = 0.0
        self._steps = 0
        self._dwell_elapsed = 0.0
        self._done = False
        self._terminated = False
        self._truncated = False
        self._reason = "running"
        self._native_start_time = float(self.data.time)
        self._all_warnings = []
        self._contact_samples = 0
        self._max_contacts = 0
        self._ready = True
        return self._observation(), self._info()

    def _finite(self):
        return all(np.isfinite(values).all() for values in (self.data.qpos, self.data.qvel, self.data.qacc,
                                                          self.data.plugin_state, self.data.site_xpos))

    def _warning_counts(self):
        return {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(self.data.warning) if w.number}

    def _observation(self):
        if not self._observations_available():
            return {"endpoint_position": [], "endpoint_velocity": [],
                    "target": self.target.tolist(), "rope_points": [],
                    "last_action": self._last_action.tolist()}
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.tip_site_id, velocity, 0)
        points = np.vstack((self.data.xpos[self._rope_body_ids], self.data.site_xpos[self.tip_site_id]))
        return {"endpoint_position": self.data.site_xpos[self.tip_site_id].tolist(),
                "endpoint_velocity": velocity[3:].tolist(), "target": self.target.tolist(),
                "rope_points": points.tolist(), "last_action": self._last_action.tolist()}

    def _observations_available(self):
        return self._reason not in ("invalid_state", "engine_exception") and self._finite()

    def _info(self):
        native_time = float(self.data.time)
        distance = (float(np.linalg.norm(self.data.site_xpos[self.tip_site_id] - self.target))
                    if self._observations_available() else None)
        if distance is not None and not math.isfinite(distance):
            distance = None
        return {"env_id": ENV_ID, "seed": self.seed, "reason": self._reason,
                "valid": self._reason not in ("invalid_state", "solver_warning", "workspace_violation", "engine_exception"),
                "success": self._terminated and self._reason == "success",
                "simulation_time": self._elapsed,
                "native_time": native_time if math.isfinite(native_time) else None,
                "steps": self._steps, "dwell_elapsed": self._dwell_elapsed,
                "observations_available": self._observations_available(),
                "distance": distance,
                "contact_count": int(self.data.ncon), "contact_samples": self._contact_samples,
                "max_contact_count": self._max_contacts, "warnings": list(self._all_warnings),
                "warning_counts": self._warning_counts(), "config_hash": self.config_hash,
                "scene_hash": self.scene_hash, "config": asdict(self.config), "parameters": dict(self.parameters),
                "engine": PIN, "numpy": np.__version__, "qualification": "simulation-only",
                "observation_profile": "privileged-simulator-state", "actuation_profile": "ideal-endpoint-cartesian-force-v0",
                "anchor": self.anchor.tolist(), "force_limit": self.config.force_limit,
                "workspace_radius": self.config.workspace_radius_factor * self.parameters["rope_length"],
                "physics_substeps": round(self.config.control_dt / self.config.timestep)}

    def step(self, force):
        if not self._ready:
            raise RuntimeError("reset is required before step")
        if self._done:
            raise RuntimeError("episode is finished; reset before another step")
        try:
            action = np.asarray(force, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("force must be three finite world-frame newton values") from error
        if action.shape != (3,) or not np.isfinite(action).all() or np.linalg.norm(action) > self.config.force_limit + 1e-12:
            raise ValueError("force must be finite, shape (3,), and within the configured norm limit")
        previous_distance = float(np.linalg.norm(self.data.site_xpos[self.tip_site_id] - self.target))
        self._last_action = action.copy()
        warnings = []
        failure = None
        with self._warnings(warnings):
            try:
                for _ in range(round(self.config.control_dt / self.config.timestep)):
                    self.data.qfrc_applied[:] = 0
                    mujoco.mj_applyFT(self.model, self.data, action, np.zeros(3),
                                     self.data.site_xpos[self.tip_site_id], self.tip_body_id, self.data.qfrc_applied)
                    before = float(self.data.time)
                    mujoco.mj_step(self.model, self.data)
                    if not self._finite() or not math.isclose(float(self.data.time) - before, self.config.timestep, rel_tol=0, abs_tol=1e-10):
                        failure = "invalid_state"
                        break
                    mujoco.mj_forward(self.model, self.data)
                    if not self._finite():
                        failure = "invalid_state"
                        break
                    self._elapsed += self.config.timestep
                    self._contact_samples += int(self.data.ncon)
                    self._max_contacts = max(self._max_contacts, int(self.data.ncon))
                    if warnings or self._warning_counts():
                        failure = "solver_warning"
                        break
                    if np.max(np.linalg.norm(self.data.xpos[self._rope_body_ids] - self.anchor, axis=1)) > self.config.workspace_radius_factor * self.parameters["rope_length"]:
                        failure = "workspace_violation"
                        break
                    distance = float(np.linalg.norm(self.data.site_xpos[self.tip_site_id] - self.target))
                    self._dwell_elapsed = self._dwell_elapsed + self.config.timestep if distance <= self.config.tolerance else 0.0
            except (mujoco.FatalError, mujoco.UnexpectedError) as error:
                warnings.append(str(error))
                failure = "engine_exception"
        self._all_warnings.extend(warnings)
        self._steps += 1
        if failure:
            self._reason = failure
            self._terminated = True
        elif self._dwell_elapsed + 1e-12 >= self.config.dwell:
            self._reason = "success"
            self._terminated = True
        elif self._steps >= round(self.config.horizon / self.config.control_dt):
            self._reason = "horizon"
            self._truncated = True
        self._done = self._terminated or self._truncated
        observation = self._observation()
        distance = (float(np.linalg.norm(np.asarray(observation["endpoint_position"]) - self.target))
                    if self._observations_available() else previous_distance)
        reward = previous_distance - distance - 0.001 * float(np.dot(action, action)) * self.config.control_dt
        if self._reason == "success":
            reward += 1.0
        if failure:
            reward = -1.0
        return observation, float(reward), self._terminated, self._truncated, self._info()

    def snapshot(self):
        if not self._ready:
            raise RuntimeError("reset is required before snapshot")
        if not self._info()["valid"] or not self._observations_available():
            raise RuntimeError("invalid numerical/workspace episodes have no resumable snapshot")
        state = np.empty(mujoco.mj_stateSize(self.model, STATE_SIGNATURE))
        mujoco.mj_getState(self.model, self.data, state, STATE_SIGNATURE)
        return {"schema_version": 1, "env_id": ENV_ID, "engine": PIN,
                "config_hash": self.config_hash, "scene_hash": self.scene_hash,
                "seed": self.seed, "state_signature": STATE_SIGNATURE, "integration_state": state.tolist(),
                "plugin_state": self.data.plugin_state.tolist(), "target": self.target.tolist(),
                "warning_state": [{"number": int(w.number), "lastinfo": int(w.lastinfo)} for w in self.data.warning],
                "episode": {"elapsed": self._elapsed, "steps": self._steps, "dwell_elapsed": self._dwell_elapsed,
                            "done": self._done, "terminated": self._terminated, "truncated": self._truncated,
                            "reason": self._reason, "native_start_time": self._native_start_time,
                            "last_action": self._last_action.tolist(), "warnings": list(self._all_warnings),
                            "contact_samples": self._contact_samples, "max_contacts": self._max_contacts}}

    def restore(self, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1 or snapshot.get("env_id") != ENV_ID or snapshot.get("engine") != PIN:
            raise ValueError("unsupported snapshot schema/environment/engine")
        if snapshot.get("config_hash") != self.config_hash or snapshot.get("state_signature") != STATE_SIGNATURE:
            raise ValueError("snapshot configuration/state signature mismatch")
        # Validate all episode metadata before resetting or mutating this live
        # environment. A separate candidate also protects against native forward
        # errors in an externally supplied but superficially finite state.
        try:
            seed = snapshot["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("invalid seed")
            state = np.asarray(snapshot["integration_state"], dtype=float)
            target = np.asarray(snapshot["target"], dtype=float)
            plugin = np.asarray(snapshot["plugin_state"], dtype=float)
            episode = snapshot["episode"]
            if not isinstance(episode, dict):
                raise ValueError("invalid episode")
            steps = episode["steps"]
            if not isinstance(steps, int) or isinstance(steps, bool) or not 0 <= steps <= round(self.config.horizon / self.config.control_dt):
                raise ValueError("invalid step count")
            elapsed, dwell, native_start = (float(episode[name]) for name in ("elapsed", "dwell_elapsed", "native_start_time"))
            if not all(math.isfinite(x) and x >= 0 for x in (elapsed, dwell, native_start)):
                raise ValueError("invalid times")
            if not math.isclose(elapsed, steps * self.config.control_dt, rel_tol=0, abs_tol=1e-9) or dwell > elapsed + 1e-9:
                raise ValueError("inconsistent elapsed/dwell")
            done, terminated, truncated = (episode[name] for name in ("done", "terminated", "truncated"))
            if any(type(flag) is not bool for flag in (done, terminated, truncated)):
                raise ValueError("invalid termination flags")
            reason = episode["reason"]
            expected = {"running": (False, False, False), "success": (True, True, False), "horizon": (True, False, True)}
            if reason not in expected or (done, terminated, truncated) != expected[reason]:
                raise ValueError("inconsistent termination reason")
            if reason == "horizon" and steps != round(self.config.horizon / self.config.control_dt):
                raise ValueError("premature horizon")
            if reason == "running" and steps >= round(self.config.horizon / self.config.control_dt):
                raise ValueError("unfinished horizon")
            if (reason == "success") != (dwell + 1e-12 >= self.config.dwell):
                raise ValueError("inconsistent success/dwell")
            last_action = np.asarray(episode["last_action"], dtype=float)
            if last_action.shape != (3,) or not np.isfinite(last_action).all() or np.linalg.norm(last_action) > self.config.force_limit + 1e-12:
                raise ValueError("invalid last action")
            warnings = episode["warnings"]
            if not isinstance(warnings, list) or warnings:
                raise ValueError("warning-bearing episodes are not resumable")
            counts = [episode["contact_samples"], episode["max_contacts"]]
            if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in counts):
                raise ValueError("invalid contact counts")
            warning_state = snapshot["warning_state"]
            if not isinstance(warning_state, list) or len(warning_state) != int(mujoco.mjtWarning.mjNWARNING):
                raise ValueError("invalid warning state")
            for warning in warning_state:
                if not isinstance(warning, dict) or set(warning) != {"number", "lastinfo"} or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in warning.values()) or warning["number"]:
                    raise ValueError("invalid warning diagnostics")
            if state.ndim != 1 or not np.isfinite(state).all() or target.shape != (3,) or not np.isfinite(target).all() or plugin.ndim != 1 or not np.isfinite(plugin).all():
                raise ValueError("invalid physics state shape/values")
            if not len(state) or not math.isclose(float(state[0]), native_start + elapsed, rel_tol=0, abs_tol=1e-9):
                raise ValueError("inconsistent native clock")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid snapshot metadata: {error}") from error
        candidate = CableEndReachEnv(self.config)
        candidate.reset(seed)
        if snapshot.get("scene_hash") != candidate.scene_hash or target.tolist() != candidate.target.tolist():
            raise ValueError("snapshot scene/target mismatch")
        if state.shape != (mujoco.mj_stateSize(candidate.model, STATE_SIGNATURE),) or plugin.shape != candidate.data.plugin_state.shape:
            raise ValueError("snapshot integration/plugin state has invalid shape")
        if not math.isclose(native_start, candidate._native_start_time, rel_tol=0, abs_tol=1e-10):
            raise ValueError("snapshot settling clock mismatch")
        forward_warnings = []
        try:
            with candidate._warnings(forward_warnings):
                mujoco.mj_setState(candidate.model, candidate.data, state, STATE_SIGNATURE)
                # Populate derived poses/velocities, then restore the complete saved
                # integration values so mj_forward cannot change the warm start.
                mujoco.mj_forward(candidate.model, candidate.data)
                mujoco.mj_setState(candidate.model, candidate.data, state, STATE_SIGNATURE)
        except (mujoco.FatalError, mujoco.UnexpectedError) as error:
            raise ValueError(f"snapshot physics could not be restored: {error}") from error
        if forward_warnings or not candidate._finite() or candidate._warning_counts():
            raise ValueError("snapshot physics produced invalid state/warnings")
        if candidate.data.plugin_state.tolist() != plugin.tolist():
            raise ValueError("snapshot plugin state mismatch")
        if reason == "success" and np.linalg.norm(candidate.data.site_xpos[candidate.tip_site_id] - target) > self.config.tolerance:
            raise ValueError("snapshot success lacks an endpoint inside the target")
        for i, warning in enumerate(warning_state):
            candidate.data.warning[i].number = warning["number"]
            candidate.data.warning[i].lastinfo = warning["lastinfo"]
        candidate._elapsed, candidate._steps, candidate._dwell_elapsed = elapsed, steps, dwell
        candidate._done, candidate._terminated, candidate._truncated = done, terminated, truncated
        candidate._reason, candidate._native_start_time = reason, native_start
        candidate._last_action = last_action.copy()
        candidate._all_warnings = list(warnings)
        candidate._contact_samples, candidate._max_contacts = counts
        self.__dict__.update(candidate.__dict__)
        return self._observation()
