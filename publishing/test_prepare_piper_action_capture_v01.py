"""Read-only capture packaging tests; no renderer, simulator or network calls."""
import builtins
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import prepare_piper_action_capture_v01 as release


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs" / "piper-action-capture-validation-20261030"


class InventoryAndSourceTests(unittest.TestCase):
    def test_synthetic_private_path_exception_does_not_allow_real_prefix_siblings(self):
        synthetic = b"/" + b"private/example"
        declared = (ROOT / "tests/test_piper_action_capture.py").read_bytes()
        release.public_text_check("tests/test_piper_action_capture.py", declared)
        for filename, raw in (
            ("README.md", b"path = '" + synthetic + b"'\n"),
            ("tests/test_piper_action_capture.py", b"path = '" + synthetic + b"-secret/real-file'\n"),
            ("tests/test_piper_action_capture.py", b"path = '" + synthetic + b"/assets-user/real-file'\n"),
        ):
            with self.subTest(filename=filename, raw_kind=raw[:8]), self.assertRaises(ValueError):
                release.public_text_check(filename, raw)

    def test_inventory_binds_every_relative_file_and_refuses_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            (root / "frames/rgb.png").write_bytes(b"retained frame bytes")
            (root / "report.json").write_bytes(b'{"qualified":false}\n')
            expected = {name: release.identity((root / name).read_bytes())
                for name in ("frames/rgb.png", "report.json")}
            self.assertEqual(release.inventory(root), expected)
            (root / "linked.json").symlink_to(root / "report.json")
            with self.assertRaises(ValueError):
                release.inventory(root)

    def source_fixture(self, project):
        source = project / "src" / "dvidia_training"
        source.mkdir(parents=True)
        expected = {}
        for name in release.SOURCES:
            raw = (name + " immutable capture source\n").encode()
            (source / name).write_bytes(raw)
            expected[name] = sha256(raw).hexdigest()
        return source, expected

    def test_all_source_pins_are_required_and_tampering_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            source, expected = self.source_fixture(project)
            found = release.source_files(project, expected)
            self.assertEqual(set(found), set(release.SOURCES))
            name = release.SOURCES[0]
            original = (source / name).read_bytes()
            (source / name).write_bytes(original + b"changed\n")
            with self.assertRaises(ValueError):
                release.source_files(project, expected)
            (source / name).unlink()
            with self.assertRaises((ValueError, OSError)):
                release.source_files(project, expected)

    def test_missing_or_extra_source_pin_cannot_change_the_source_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            source, expected = self.source_fixture(project)
            missing = dict(expected)
            del missing[release.SOURCES[0]]
            extra = {**expected, "unrelated.py": "a" * 64}
            for changed in (missing, extra):
                with self.subTest(pin_count=len(changed)), self.assertRaises(ValueError):
                    release.source_files(project, changed)


class PortablePublicKitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "frozen-inputs"
        self.inputs.mkdir()
        (self.inputs / "a.json").write_bytes(b'{"case":"fixture","physical_robot_ready":false}\n')
        (self.inputs / "LICENSE").write_bytes(b"MIT license fixture retained exactly.\n")
        self.summary = {
            "schema": release.SCHEMA,
            "scope": "Capture validation only",
            "cases": [],
            "repeat": {"snapshot_pairs": 0, "identical_rgb_and_depth_frame_pairs": 0,
                "max_difference": 0, "same_commands": True, "same_controls": True},
            "missing_tracker_channels": 4,
            "depth_warning": "Unverified synthetic depth",
            "timing_scope": "Paused simulation clock",
            "physical_robot_ready": False,
            "trained_skill": False,
        }
        self.input_result = {"summary": self.summary, "episodes": [],
            "copy_files": {"evidence/a.json": self.inputs / "a.json", "LICENSE": self.inputs / "LICENSE"}}
        self.audit_patch = patch.object(release, "audit_inputs", return_value=self.input_result)
        self.assets_patch = patch.object(release, "report_assets", return_value={
            "index.html": b"<!doctype html><title>Capture fixture</title>\n",
            "report.js": b"// Fixture viewer asset.\n",
        })
        self.audit_patch.start()
        self.assets_patch.start()

    def tearDown(self):
        self.audit_patch.stop()
        self.assets_patch.stop()
        self.temporary.cleanup()

    def prepare(self, output):
        return release.prepare(self.inputs, output, project=ROOT)

    def test_full_kit_manifest_and_adjacent_zip_are_deterministic_and_inputs_unchanged(self):
        before = release.inventory(self.inputs)
        outputs = [self.root / "kit-one", self.root / "kit-two"]
        for output in outputs:
            self.prepare(output)
            manifest = json.loads((output / "manifest.json").read_bytes())
            actual = release.inventory(output)
            self.assertEqual(manifest["files"], {name: value for name, value in actual.items() if name != "manifest.json"})
            self.assertNotIn(str(self.root).encode(), (output / "manifest.json").read_bytes())
            archive_path = output.with_suffix(".zip")
            self.assertTrue(archive_path.is_file())
            self.assertNotIn(archive_path.name, actual)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(archive.namelist(), sorted(actual))
                for name in archive.namelist():
                    self.assertFalse(Path(name).is_absolute())
                    self.assertNotIn("..", Path(name).parts)
                    self.assertEqual(archive.read(name), (output / name).read_bytes())
            self.assertEqual(release.inventory(self.inputs), before)
            self.assertEqual((output / "evidence/a.json").read_bytes(), (self.inputs / "a.json").read_bytes())
            self.assertEqual((output / "LICENSE").read_bytes(), (self.inputs / "LICENSE").read_bytes())
        self.assertEqual(outputs[0].with_suffix(".zip").read_bytes(), outputs[1].with_suffix(".zip").read_bytes())

    def test_existing_output_cannot_be_overwritten(self):
        output = self.root / "kit"
        self.prepare(output)
        sealed = release.inventory(output)
        archive = output.with_suffix(".zip").read_bytes()
        with self.assertRaises((ValueError, FileExistsError)):
            self.prepare(output)
        self.assertEqual(release.inventory(output), sealed)
        self.assertEqual(output.with_suffix(".zip").read_bytes(), archive)

    def test_existing_adjacent_receipt_cannot_be_overwritten(self):
        output = self.root / "kit"
        receipt = output.with_suffix(".receipt.json")
        receipt.write_bytes(b"Retained prior evidence receipt.\n")
        with self.assertRaises((ValueError, FileExistsError)):
            self.prepare(output)
        self.assertEqual(receipt.read_bytes(), b"Retained prior evidence receipt.\n")
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".zip").exists())

    def test_archive_cannot_include_itself_inside_payload(self):
        target = self.inputs / "inside.zip"
        with self.assertRaises(ValueError):
            release.archive(self.inputs, target)
        self.assertFalse(target.exists())

    def test_archive_cannot_silently_omit_a_file_removed_after_inventory(self):
        target = self.root / "removed-input.zip"
        original_inventory = release.inventory
        first = True

        def remove_after_first_inventory(folder, *args, **kwargs):
            nonlocal first
            result = original_inventory(folder, *args, **kwargs)
            if first:
                first = False
                (self.inputs / "a.json").unlink()
            return result

        with patch.object(release, "inventory", side_effect=remove_after_first_inventory), self.assertRaises((ValueError, OSError)):
            release.archive(self.inputs, target)

    def test_input_change_between_audit_and_copy_is_refused(self):
        self.assets_patch.stop()

        def mutate_after_inventory():
            path = self.inputs / "a.json"
            path.write_bytes(path.read_bytes() + b" ")
            return {"index.html": b"<!doctype html><title>Fixture</title>\n"}

        output = self.root / "kit"
        with patch.object(release, "report_assets", side_effect=mutate_after_inventory), self.assertRaises(ValueError):
            self.prepare(output)
        self.assertFalse(output.with_suffix(".zip").exists())

    def test_local_absolute_path_text_is_refused_before_public_archive(self):
        forbidden = b"/" + b"Users/person/private-recording.json"
        (self.inputs / "a.json").write_bytes(b'{"path":"' + forbidden + b'"}\n')
        output = self.root / "kit"
        with self.assertRaises(ValueError):
            self.prepare(output)
        self.assertFalse(output.with_suffix(".zip").exists())


class CompactDerivativeGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        # Retain the real canonical IDs with one tiny frame per fixture case.
        self.cases = {name: (0, status) for name, (_, status) in release.CASES.items()}
        self.cases_patch = patch.object(release, "CASES", self.cases)
        self.cases_patch.start()

    def tearDown(self):
        self.cases_patch.stop()
        self.temporary.cleanup()

    def seal(self, kit):
        manifest = {"schema": release.SCHEMA, "files": release.inventory(kit)}
        manifest["files"].pop("manifest.json", None)
        raw = release.encoded(manifest)
        (kit / "manifest.json").write_bytes(raw)
        kit.with_suffix(".receipt.json").write_bytes(release.encoded({
            "manifest": release.identity(raw), "archive": {"name": "fixture.zip", "bytes": 0, "sha256": "0" * 64}}))

    def make_kit(self, name):
        kit = self.root / name
        kit.mkdir()
        episodes = []
        for case in self.cases:
            relative = f"evidence/registered/{case}/frames/rgb-0000.png"
            path = kit / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"Fixture PNG bytes; never decoded or encoded by tests.")
            episodes.append({"id": case, "rows": [{"rgb": relative, "t": 0.0}]})
            audit = kit / "audits" / (case + "-audit.json")
            audit.parent.mkdir(exist_ok=True)
            audit.write_bytes(release.encoded({"image_shape_hwc": [120, 160, 3]}))
        (kit / "viewer-data.json").write_bytes(release.encoded({"schema": release.SCHEMA, "episodes": episodes}))
        self.seal(kit)
        return kit

    def test_source_kit_tamper_or_missing_episode_refuses_before_encoder(self):
        for failure in ("tampered-frame", "missing-episode"):
            kit = self.make_kit(failure)
            if failure == "tampered-frame":
                path = next((kit / "evidence").rglob("*.png"))
                path.write_bytes(path.read_bytes() + b"changed")
            else:
                data_path = kit / "viewer-data.json"
                data = json.loads(data_path.read_bytes())
                data["episodes"].pop()
                data_path.write_bytes(release.encoded(data))
                self.seal(kit)
            output = self.root / (failure + "-web")
            with self.subTest(failure=failure), patch.object(release.subprocess, "run") as encoder, self.assertRaises(ValueError):
                release.prepare_web(kit, output)
            encoder.assert_not_called()
            self.assertFalse(output.exists())

    def test_existing_compact_output_refuses_without_encoder_or_overwrite(self):
        output = self.root / "web"
        output.mkdir()
        (output / "retained.txt").write_bytes(b"Retained prior derivative.\n")
        with patch.object(release.subprocess, "run") as encoder, self.assertRaises(ValueError):
            release.prepare_web(self.root / "not-read", output)
        encoder.assert_not_called()
        self.assertEqual((output / "retained.txt").read_bytes(), b"Retained prior derivative.\n")

    def test_wrong_ffprobe_frame_count_cannot_produce_success_receipt(self):
        kit = self.make_kit("source")
        output = self.root / "web"

        def fake_encoder(command, **kwargs):
            if "-version" in command:
                return SimpleNamespace(stdout="ffmpeg fixture version\n")
            if Path(command[0]).name == "ffprobe":
                stream = {"nb_read_frames": "2", "r_frame_rate": "50/1", "width": 160, "height": 120,
                    "codec_name": "h264", "pix_fmt": "yuv420p", "duration": "0.04"}
                return SimpleNamespace(stdout=json.dumps({"streams": [stream]}).encode())
            Path(command[-1]).write_bytes(b"Fixture encoded bytes; no actual encoder was run.")
            return SimpleNamespace(stdout=b"")

        with patch.object(release.subprocess, "run", side_effect=fake_encoder) as encoder, self.assertRaises(ValueError):
            release.prepare_web(kit, output, ffmpeg="/fixture-tools/ffmpeg")
        self.assertEqual(encoder.call_count, 3)
        self.assertFalse((output / "derivation-receipt.json").exists())
        self.assertFalse((output / "manifest.json").exists())


@unittest.skipUnless((STUDY / "interrupted-prefix").is_dir(), "Requires only the retained three-frame interrupted prefix")
class InterruptedCaptureReadbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.case = Path(self.temporary.name) / "interrupted-prefix"
        shutil.copytree(STUDY / "interrupted-prefix", self.case)
        shutil.copyfile(STUDY / "interrupted-prefix-audit.json", self.case.with_name(self.case.name + "-audit.json"))

    def tearDown(self):
        self.temporary.cleanup()

    def test_interrupted_prefix_is_retained_with_unknown_outcome_and_refused_targets(self):
        real_import = builtins.__import__

        def no_native_import(name, *args, **kwargs):
            if name.split(".")[0] in {"mujoco", "numpy", "pxr", "PIL", "huggingface_hub"}:
                raise AssertionError("Read-only byte audit must not import a native, decoder or publication runtime")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_native_import):
            result = release.verify_case(self.case, 2, "interrupted")
        self.assertEqual(result["recording"]["outcome"]["task_result"], "unknown")
        self.assertTrue(result["audit"]["training_pair_export_refused"])
        self.assertFalse(result["report"]["strict_learning_targets_available"])

    def test_missing_frame_is_refused_instead_of_silently_dropped(self):
        (self.case / "frames/rgb-0001.png").unlink()
        with self.assertRaises((ValueError, OSError)):
            release.verify_case(self.case, 2, "interrupted")

    def test_modified_native_artifact_is_refused(self):
        path = self.case / "native/0001.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            release.verify_case(self.case, 2, "interrupted")

    def test_stored_audit_must_bind_this_recording_and_keep_refusal(self):
        path = self.case.with_name(self.case.name + "-audit.json")
        original = json.loads(path.read_bytes())
        for field in ("recording", "training_pair_export_refused", "frame_bytes"):
            changed = json.loads(json.dumps(original))
            if field == "recording":
                changed[field]["sha256"] = "0" * 64
            elif field == "training_pair_export_refused":
                changed[field] = False
            else:
                changed["frame_artifact_check"]["bytes"] += 1
            path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                release.verify_case(self.case, 2, "interrupted")

    def test_repeat_comparison_requires_the_same_initial_ramped_targets(self):
        left = release.verify_case(self.case, 2, "interrupted")
        right = json.loads(json.dumps(left))
        right["recording"]["initial_applied_targets"]["joint_targets_rad"][0] += 0.001
        fields = ("joint_position_rad", "joint_velocity_rad_s", "gripper_joint_position_m",
            "gripper_joint_velocity_m_s", "tcp_position_estimate_m", "tcp_quaternion_estimate_wxyz")
        stored = {"schema": "dvidia.native-action-capture-repeat-comparison/v0.1",
            "snapshot_pairs": 3, "control_steps": 2, "identical_rgb_and_depth_frame_pairs": 3,
            "maximum_absolute_difference_by_named_channel": {name: 0.0 for name in fields},
            "same_command_schedule": True, "same_applied_controls": True, "physical_robot_ready": False}
        with self.assertRaises(ValueError):
            release.compare_repeat(left, right, stored)


if __name__ == "__main__":
    unittest.main()
