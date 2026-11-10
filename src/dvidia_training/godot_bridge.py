"""Local desktop rendering bridge; MuJoCo alone advances the pinned workstation.

This is an authored target-tracking research interface, not a learned policy or
a hardware interface. Requests advance a fixed amount of simulation time; wall
clock delays, rendering frames and disconnected clients never advance physics.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import socket
import time

from . import asset_preview, reference_replay, reference_sim, workstation

SCHEMA = "dvidia.godot-bridge/v1"
INTEGRITY_SCHEMA = "dvidia.godot-scene-integrity/v1"
HOST = "127.0.0.1"
DEFAULT_PORT = 8430
MAX_REQUEST_BYTES = 16 * 1024
IDLE_TIMEOUT_SECONDS = 30.
MAX_SEQUENCE = 2**53 - 1
SUBSTEPS = 10
SCOPE = ("Desktop Godot rendering of the pinned native PiPER workstation. "
         "MuJoCo alone advances 10 x 2 ms physics ticks per accepted step. "
         "Authored IK and synthetic rigid shoe; no learned policy, calibrated "
         "contact, deformation, hardware control or real-time qualification.")
TIMING_SCOPE = "Offline request computation only; excludes network, rendering and client scheduling."


class ProtocolError(ValueError):
    def __init__(self, message, code="invalid_request"):
        super().__init__(message)
        self.code = code


def _json(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def decode_request(payload):
    """Accept one bounded, finite, unique-key UTF-8 JSON object."""
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_REQUEST_BYTES:
        raise ProtocolError("Request must contain 1 to 16384 bytes", "request_size")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("Nonfinite value")

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)
        json.dumps(value, allow_nan=False)
    except (ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ProtocolError("Request requires finite UTF-8 JSON with unique keys", "invalid_json") from exc
    if type(value) is not dict:
        raise ProtocolError("Request must be a JSON object")
    return value


def _fields(request, fields):
    if set(request) != set(fields):
        raise ProtocolError("Request fields do not match the command schema")


def _number(value, low, high, label):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ProtocolError(label + " must be a bounded finite number")
    return float(value)


class Session:
    """One client, one initialized scene, strict command authority and sequence.

    ``backend`` is injectable for transport-only tests. Production always uses
    the unchanged workstation module. Invalid inputs are checked before any
    dynamics; a native failure faults the session because it may have partly
    advanced a control interval.
    """

    def __init__(self, model, model_sha256, *, backend=workstation):
        if type(model_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
            raise ValueError("Expected native model SHA256 digest")
        self.model, self.model_sha256, self.backend = model, model_sha256, backend
        self.data = backend.initialize(model)
        self.controller_state = {}
        self.target = list(workstation.HOME_TARGET)
        self.jaw = .035
        self.expected_sequence = 1
        self.ready = False
        self.faulted = False

    def error(self, exc):
        return {"type": "error", "code": exc.code, "message": str(exc),
                "expected_sequence": self.expected_sequence}

    def process(self, payload):
        try:
            return self.handle(decode_request(payload))
        except ProtocolError as exc:
            return self.error(exc)

    def handle(self, request):
        if type(request) is not dict:
            raise ProtocolError("Request must be a JSON object")
        if self.faulted:
            raise ProtocolError("Native simulation faulted; reconnect to initialize a new scene", "simulation_fault")
        kind = request.get("type")
        if not self.ready:
            _fields(request, ("type", "schema", "model_sha256"))
            if kind != "hello" or request["schema"] != SCHEMA:
                raise ProtocolError("A matching bridge hello is required before commands", "handshake_required")
            if request["model_sha256"] != self.model_sha256:
                raise ProtocolError("Prepared scene and running native model differ", "model_mismatch")
            self.ready = True
            return {"type": "ready", "schema": SCHEMA, "model_sha256": self.model_sha256,
                    "state": self.backend.state(self.model, self.data, self.target),
                    "physics_dt": workstation.TIMESTEP, "control_dt": workstation.CONTROL_DT,
                    "scope": SCOPE, "timing_scope": TIMING_SCOPE}
        if kind == "ping":
            _fields(request, ("type",))
            return {"type": "pong", "model_sha256": self.model_sha256,
                    "expected_sequence": self.expected_sequence}
        if kind not in ("step", "reset"):
            raise ProtocolError("Only step, reset and ping are supported after hello")
        _fields(request, ("type", "sequence", "target", "jaw") if kind == "step" else ("type", "sequence"))
        sequence = request["sequence"]
        if type(sequence) is not int or not 1 <= sequence <= MAX_SEQUENCE or sequence != self.expected_sequence:
            raise ProtocolError("Sequence must equal the next accepted command number", "sequence_mismatch")
        if kind == "step":
            target = request["target"]
            if type(target) is not list or len(target) != 3:
                raise ProtocolError("Target requires three coordinates")
            target = [_number(value, *bounds, "Target") for value, bounds in zip(target, workstation.BOUNDS)]
            jaw = _number(request["jaw"], 0., .035, "Jaw")
        else:
            target, jaw = list(workstation.HOME_TARGET), .035
        started = time.perf_counter()
        try:
            if kind == "step":
                state = self.backend.run_control(self.model, self.data, self.controller_state, target, jaw, SUBSTEPS)
            else:
                self.data = self.backend.initialize(self.model)
                self.controller_state = {}
                state = self.backend.state(self.model, self.data, target)
        except Exception as exc:
            self.faulted = True
            raise ProtocolError("Native simulation failed; this interval may have partially advanced. Reconnect to reset.",
                                "simulation_fault") from exc
        compute_ms = (time.perf_counter() - started) * 1000
        self.target, self.jaw = target, jaw
        self.expected_sequence += 1
        return {"type": "state", "sequence": sequence, "model_sha256": self.model_sha256,
                "state": state, "compute_ms": compute_ms, "timing_scope": TIMING_SCOPE}


def handle_connection(connection, session):
    """Read bounded newline frames; a disconnect discards any unissued prefix."""
    buffer = bytearray()
    try:
        while True:
            # At most one request plus a small receive chunk is held at a time.
            chunk = connection.recv(min(4096, MAX_REQUEST_BYTES + 1 - len(buffer)))
            if not chunk:
                return
            buffer.extend(chunk)
            while b"\n" in buffer:
                line, _, rest = buffer.partition(b"\n")
                buffer = bytearray(rest)
                response = session.process(bytes(line))
                connection.sendall(_json(response))
                if session.faulted:
                    return
            if len(buffer) > MAX_REQUEST_BYTES:
                connection.sendall(_json(session.error(ProtocolError("Request exceeds 16384 bytes", "request_size"))))
                return
    except (ConnectionError, OSError):
        # No timer or free-running stepping thread exists after a client leaves.
        return


def prepare(assets, output):
    """Export a fresh local renderer scene without running physics."""
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Godot scene output must be a fresh directory")
    if any(parent.is_symlink() for parent in destination.absolute().parents):
        raise ValueError("Godot scene output cannot traverse a symlink")
    provenance, notices = asset_preview._licenses("piper", assets)
    model = workstation.build_model(assets)
    digest = reference_sim.model_digest(model)
    mj, _ = reference_sim._native_imports()
    source = {"robot": "piper", "assets": provenance, "mujoco": mj.__version__,
              "compiled_model_sha256": digest}
    scene, geometry = asset_preview._scene(model, source)
    if reference_sim.model_digest(model) != digest:
        raise ValueError("Geometry export changed the native model")
    scene["scope"] = SCOPE
    scene["bridge"] = {"schema": SCHEMA, "model_sha256": digest,
                       "home_target": list(workstation.HOME_TARGET),
                       "bounds": [list(pair) for pair in workstation.BOUNDS],
                       "gripper_initial": .035, "physics_dt": workstation.TIMESTEP,
                       "control_dt": workstation.CONTROL_DT, "scope": SCOPE}
    scene["geometry_binary"] = {"href": "geometry.bin", "compression": "none", "bytes": len(geometry),
                                "sha256": sha256(geometry).hexdigest()}
    files = {"scene.json": _json(scene), "geometry.bin": geometry,
             "LICENSE": reference_replay.MIT_LICENSE.encode(), **notices}
    files["README.md"] = ("# DVIDIA Godot workstation scene\n\n" + SCOPE + "\n\n"
        "Prepared geometry only: no physics ticks or benchmark have run. Native positions are metres "
        "in Z-up coordinates; quaternions are wxyz. Binary positions are little-endian Float32 and "
        "indices are little-endian UInt32. The Godot renderer maps [x,y,z] to [x,z,-y]. "
        "Run the bridge separately against the same pinned local robot assets. The hello digest "
        "must match this compiled model. Upstream robot notices are retained under licenses/. "
        "DVIDIA code and synthetic shoes use MIT; MuJoCo is an external Apache-2.0 runtime. "
        "Hashes establish artifact consistency, not authenticity or physical qualification.\n").encode()
    seal = {"schema": INTEGRITY_SCHEMA, "model_sha256": digest,
            "scope": "Artifact consistency only; not authenticity or physical qualification",
            "files": {name: {"bytes": len(value), "sha256": sha256(value).hexdigest()}
                      for name, value in files.items()}}
    files["scene-integrity.json"] = _json(seal)
    if any(len(value) > asset_preview.MAX_FILE_BYTES for value in files.values()) or sum(map(len, files.values())) > asset_preview.MAX_TOTAL_BYTES:
        raise ValueError("Godot scene exceeds the bounded export size")
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(value)
    return {"schema": SCHEMA, "output": str(destination), "model_sha256": digest,
            "file_count": len(files), "bytes": sum(map(len, files.values())),
            "physics_steps": 0, "scope": SCOPE}


def serve(assets, port=DEFAULT_PORT, *, once=False):
    """Serve one local TCP client at a time, with a new scene per connection."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Port must be an integer in [1,65535]")
    model = workstation.build_model(assets)
    digest = reference_sim.model_digest(model)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((HOST, port))
        listener.listen(1)
        print(json.dumps({"schema": SCHEMA, "status": "listening", "host": HOST, "port": port,
                          "model_sha256": digest, "scope": SCOPE}), flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(IDLE_TIMEOUT_SECONDS)
                handle_connection(connection, Session(model, digest))
            if once:
                return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("prepare", help="Export a fresh local Godot scene without physics steps")
    export.add_argument("--assets", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    server = sub.add_parser("serve", help="Serve fixed-step MuJoCo physics to one loopback client")
    server.add_argument("--assets", required=True, type=Path)
    server.add_argument("--port", type=int, default=DEFAULT_PORT)
    server.add_argument("--once", action="store_true", help="Exit after the first client disconnects")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            print(json.dumps(prepare(args.assets, args.output), indent=2))
        else:
            serve(args.assets, args.port, once=args.once)
    except KeyboardInterrupt:
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, str(exc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
