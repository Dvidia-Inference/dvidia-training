"""Transport and authority checks; fake backend tests never assert physical accuracy."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from dvidia_training import godot_bridge as bridge


DIGEST = "a" * 64


class FakeBackend:
    def __init__(self):
        self.actions = []
        self.resets = 0

    def initialize(self, model):
        self.resets += 1
        return {"time": 0., "qpos": [0.] * 15}

    def state(self, model, data, target=None):
        return deepcopy(data)

    def run_control(self, model, data, controller_state, target, jaw, substeps):
        self.actions.append((id(controller_state), deepcopy(target), jaw, substeps))
        controller_state["native_steps"] = controller_state.get("native_steps", 0) + substeps
        data["time"] = controller_state["native_steps"] * .002
        data["qpos"][0] += .001
        return self.state(model, data, target)


def hello():
    return {"type": "hello", "schema": bridge.SCHEMA, "model_sha256": DIGEST}


def step(sequence=1):
    return {"type": "step", "sequence": sequence, "target": [.32, -.145, .18], "jaw": .035}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.session = bridge.Session(object(), DIGEST, backend=self.backend)

    def request(self, value):
        return self.session.process(json.dumps(value).encode())

    def test_hello_requires_matching_schema_and_model_and_never_steps(self):
        for value in (step(), {**hello(), "schema": "other"}, {**hello(), "model_sha256": "b" * 64},
                      {**hello(), "render_fps": 60}):
            self.assertEqual(self.request(value)["type"], "error")
            self.assertFalse(self.session.ready)
        reply = self.request(hello())
        self.assertEqual(reply["type"], "ready")
        self.assertEqual(reply["state"]["time"], 0.)
        self.assertEqual(reply["model_sha256"], DIGEST)
        self.assertEqual(self.backend.actions, [])

    def test_each_step_is_exactly_ten_ticks_using_persistent_controller_state(self):
        self.request(hello())
        first, second = self.request(step()), self.request(step(2))
        self.assertEqual(first["state"]["time"], .02)
        self.assertEqual(second["state"]["time"], .04)
        self.assertEqual(self.session.controller_state["native_steps"], 20)
        self.assertEqual([row[3] for row in self.backend.actions], [10, 10])
        self.assertEqual(self.backend.actions[0][0], self.backend.actions[1][0])
        self.assertEqual(second["sequence"], 2)
        self.assertGreaterEqual(second["compute_ms"], 0.)
        self.assertIn("excludes network", second["timing_scope"])

    def test_bad_target_jaw_sequence_or_unknown_control_is_rejected_before_ticks(self):
        self.request(hello())
        requests = [{**step(), "target": value} for value in
                    (None, [True, 0, .2], [.6, 0, .2], [.3, -.25, .2], [.3, 0, .01], [.3, 0],
                     [10**1000, 0, .2], [float("nan"), 0, .2], [.3, 0, float("inf")])]
        requests += [{**step(), "jaw": value} for value in (None, True, -.001, .036, float("nan"), 10**1000)]
        requests += [{**step(), "sequence": value} for value in (None, True, 0, -1, 2, 1., 2**53)]
        requests += [{**step(), "substeps": 100}, {**step(), "impulse": [0, 0, .1]},
                     {**step(), "render_fps": 144}, {"type": "teleport", "qpos": [0] * 15},
                     {"type": "camera", "position": [1, 2, 3]}, {"type": "hello", "schema": bridge.SCHEMA}]
        for value in requests:
            with self.subTest(value=value):
                self.assertEqual(self.request(value)["type"], "error")
                self.assertEqual(self.session.expected_sequence, 1)
                self.assertEqual(self.session.data["time"], 0.)
                self.assertEqual(self.backend.actions, [])
        self.assertEqual(self.request(step())["type"], "state")

    def test_duplicate_and_skipped_sequences_do_not_replay_commands(self):
        self.request(hello())
        self.request(step())
        for sequence in (1, 3):
            self.assertEqual(self.request(step(sequence))["code"], "sequence_mismatch")
        self.assertEqual(self.session.data["time"], .02)
        self.assertEqual(len(self.backend.actions), 1)
        self.assertEqual(self.request(step(2))["state"]["time"], .04)

    def test_reset_reinitializes_without_ticks_and_sequence_remains_monotonic(self):
        self.request(hello())
        self.request(step())
        reset = self.request({"type": "reset", "sequence": 2})
        self.assertEqual(reset["state"]["time"], 0.)
        self.assertEqual(self.session.controller_state, {})
        self.assertEqual(self.backend.resets, 2)
        self.assertEqual(len(self.backend.actions), 1)
        self.assertEqual(self.request(step(3))["state"]["time"], .02)

    def test_ping_has_no_control_authority(self):
        self.request(hello())
        for _ in range(20):
            self.assertEqual(self.request({"type": "ping"})["type"], "pong")
        self.assertEqual(self.backend.actions, [])
        self.assertEqual(self.session.expected_sequence, 1)

    def test_invalid_json_and_duplicate_keys_are_rejected(self):
        self.request(hello())
        for value in (b"", b"{", b"[]", b"null", b"\xff", b'{"type":"ping","type":"step"}',
                      b'{"type":"step","sequence":1,"target":[0.3,0,1e400],"jaw":0}',
                      b" " * (bridge.MAX_REQUEST_BYTES + 1), b"[" * 1200):
            with self.subTest(value=value[:60]):
                self.assertEqual(self.session.process(value)["type"], "error")
        self.assertEqual(self.backend.actions, [])
        self.assertEqual(self.session.expected_sequence, 1)

    def test_native_exception_faults_session_and_does_not_claim_rejected_before_dynamics(self):
        self.request(hello())

        def broken(*args):
            self.session.data["time"] = .006
            raise RuntimeError("solver failure")

        self.backend.run_control = broken
        response = self.request(step())
        self.assertEqual(response["code"], "simulation_fault")
        self.assertIn("partially advanced", response["message"])
        self.assertTrue(self.session.faulted)
        self.assertEqual(self.request({"type": "reset", "sequence": 1})["code"], "simulation_fault")
        self.assertEqual(self.session.data["time"], .006)


class SocketTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.session = bridge.Session(object(), DIGEST, backend=self.backend)
        self.server, self.client = socket.socketpair()
        self.client.settimeout(2)
        self.thread = threading.Thread(target=bridge.handle_connection, args=(self.server, self.session), daemon=True)
        self.thread.start()
        self.reader = self.client.makefile("rb")

    def tearDown(self):
        self.reader.close()
        self.client.close()
        self.server.close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def read(self):
        return json.loads(self.reader.readline())

    def test_chunked_and_combined_frames_preserve_one_command_one_interval(self):
        wire = bridge._json(hello())
        for piece in (wire[:2], wire[2:17], wire[17:]):
            self.client.sendall(piece)
        self.assertEqual(self.read()["type"], "ready")
        self.client.sendall(bridge._json(step()) + bridge._json(step(2)))
        self.assertEqual(self.read()["state"]["time"], .02)
        self.assertEqual(self.read()["state"]["time"], .04)
        self.assertEqual(len(self.backend.actions), 2)

    def test_incomplete_command_disconnected_before_newline_never_ticks(self):
        self.client.sendall(bridge._json(hello()))
        self.read()
        self.client.sendall(bridge._json(step())[:-1])
        self.client.shutdown(socket.SHUT_WR)
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.backend.actions, [])

    def test_oversize_frame_is_bounded_and_closes_without_ticks(self):
        self.client.sendall(b"x" * (bridge.MAX_REQUEST_BYTES + 1))
        self.assertEqual(self.read()["code"], "request_size")
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.backend.actions, [])

    def test_malformed_line_can_be_followed_by_valid_handshake(self):
        self.client.sendall(b"{\n" + bridge._json(hello()))
        self.assertEqual(self.read()["code"], "invalid_json")
        self.assertEqual(self.read()["type"], "ready")
        self.assertEqual(self.backend.actions, [])


class ExportTests(unittest.TestCase):
    def test_prepare_exports_geometry_hash_and_notices_without_initialize_or_dynamics(self):
        fake_model = object()
        geometry = b"little endian geometry fixture"
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp).resolve() / "scene"
            with patch.object(bridge.asset_preview, "_licenses", return_value=({"commit": "pinned"}, {"licenses/piper/LICENSE": b"upstream"})), \
                 patch.object(bridge.workstation, "build_model", return_value=fake_model), \
                 patch.object(bridge.reference_sim, "model_digest", return_value=DIGEST), \
                 patch.object(bridge.reference_sim, "_native_imports", return_value=(SimpleNamespace(__version__="fixture"), None)), \
                 patch.object(bridge.asset_preview, "_scene", return_value=({"geoms": [], "bodies": [], "meshes": []}, geometry)), \
                 patch.object(bridge.workstation, "initialize") as initialize, \
                 patch.object(bridge.workstation, "run_control") as dynamics:
                result = bridge.prepare(Path(tmp) / "assets", destination)
            self.assertEqual(result["physics_steps"], 0)
            initialize.assert_not_called()
            dynamics.assert_not_called()
            scene = json.loads((destination / "scene.json").read_text())
            self.assertEqual(scene["bridge"]["model_sha256"], DIGEST)
            self.assertEqual(scene["bridge"]["physics_dt"], .002)
            self.assertEqual(scene["bridge"]["control_dt"], .02)
            self.assertEqual(scene["geometry_binary"]["sha256"], sha256(geometry).hexdigest())
            self.assertEqual((destination / "geometry.bin").read_bytes(), geometry)
            self.assertEqual((destination / "licenses/piper/LICENSE").read_bytes(), b"upstream")
            seal = json.loads((destination / "scene-integrity.json").read_text())
            for name, record in seal["files"].items():
                payload = (destination / name).read_bytes()
                self.assertEqual(len(payload), record["bytes"])
                self.assertEqual(sha256(payload).hexdigest(), record["sha256"])
            with self.assertRaisesRegex(ValueError, "fresh directory"):
                bridge.prepare("unused", destination)

    def test_output_refuses_symlink_ancestor_and_existing_directory_before_native_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "existing").mkdir()
            (root / "linked").symlink_to(root / "existing", target_is_directory=True)
            with patch.object(bridge.workstation, "build_model") as build:
                for path in (root / "existing", root / "linked", root / "linked/child"):
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        bridge.prepare("unused", path)
                build.assert_not_called()

    def test_server_port_validation_and_fixed_loopback_host(self):
        self.assertEqual(bridge.HOST, "127.0.0.1")
        for port in (0, -1, 65536, True, "8430"):
            with self.assertRaises(ValueError):
                bridge.serve("unused", port)

    def test_server_binds_loopback_sets_idle_timeout_and_once_exits(self):
        listener, connection = MagicMock(), MagicMock()
        listener.__enter__.return_value = listener
        connection.__enter__.return_value = connection
        listener.accept.return_value = (connection, ("127.0.0.1", 1234))
        connection.recv.return_value = b""
        backend = FakeBackend()
        session = bridge.Session(object(), DIGEST, backend=backend)
        with patch.object(bridge.socket, "socket", return_value=listener), \
             patch.object(bridge.workstation, "build_model"), \
             patch.object(bridge.reference_sim, "model_digest", return_value=DIGEST), \
             patch.object(bridge, "Session", return_value=session), patch("builtins.print"):
            bridge.serve("unused", 8431, once=True)
        listener.bind.assert_called_once_with(("127.0.0.1", 8431))
        connection.settimeout.assert_called_once_with(30.)
        listener.accept.assert_called_once()
        self.assertEqual(backend.actions, [])


if __name__ == "__main__":
    unittest.main()
