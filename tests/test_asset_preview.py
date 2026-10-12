"""Portable asset previews must retain native shape and bounded evidence."""
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import asset_cli, asset_library as library, asset_preview as preview


class PreviewPortableTests(unittest.TestCase):
    def test_compressed_payload_is_bounded_and_hash_checked(self):
        original = b"bounded native geometry"
        packed = gzip.compress(original, mtime=0)
        metadata = {"href": "geometry.bin.gz", "compression": "gzip", "bytes": len(original),
                    "compressed_bytes": len(packed), "sha256": sha256(original).hexdigest()}
        self.assertEqual(preview._unpack(metadata, {"geometry.bin.gz": packed}, "geometry.bin.gz"), original)
        for field, value in (("bytes", len(original)-1), ("bytes", True), ("sha256", "0"*64),
                             ("href", "../geometry.bin.gz"), ("compression", "zip")):
            changed = dict(metadata, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                preview._unpack(changed, {"geometry.bin.gz": packed}, "geometry.bin.gz")

    def test_existing_output_fails_before_assets_or_native_loading(self):
        with tempfile.TemporaryDirectory() as root, patch.object(library, "load_scene") as load:
            with self.assertRaisesRegex(ValueError, "fresh"):
                preview.export_preview("piper", "/nonexistent", root)
            load.assert_not_called()

    def test_inspector_rejects_escaping_and_windows_drive_paths(self):
        for bad_path in ("../outside.json", "C:/outside.json", "C:outside.json", "folder\\outside.json"):
            with self.subTest(path=bad_path), tempfile.TemporaryDirectory() as root:
                files = {bad_path: {"bytes": 0, "sha256": "0"*64}}
                files.update({str(i)+".json": {"bytes": 0, "sha256": "0"*64} for i in range(9)})
                (Path(root)/"preview-integrity.json").write_text(json.dumps({"schema": preview.INTEGRITY_SCHEMA, "files": files}))
                with self.assertRaisesRegex(ValueError, "not contained"):
                    preview.inspect_preview(root)

    def test_cli_load_rejects_changed_descriptor_before_export(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "scene.json"
            config = library.scene_config("piper")
            config["catalog_sha256"] = "0"*64
            path.write_text(json.dumps(config))
            with patch.object(preview, "export_preview") as export, self.assertRaises(SystemExit) as caught:
                asset_cli.main(["load", str(path), "--assets", "/missing", "--output", str(Path(root)/"out")])
            self.assertEqual(caught.exception.code, 2)
            export.assert_not_called()


@unittest.skipUnless(os.environ.get("DVIDIA_ASSET_PREVIEW_TEST_ASSETS"), "Set DVIDIA_ASSET_PREVIEW_TEST_ASSETS for pinned native PiPER assets")
class PreviewNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.output = cls.root / "preview"
        cls.result = preview.export_preview("piper", os.environ["DVIDIA_ASSET_PREVIEW_TEST_ASSETS"], cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_native_export_has_all_shoes_ellipsoids_and_settling(self):
        result = preview.inspect_preview(self.output)
        self.assertEqual(result["frame_count"], 101)
        config = json.loads((self.output/"scene.json").read_text())
        self.assertEqual(len(config["shoe_ids"]), 6)
        model = json.loads((self.output/"model.json").read_text())
        self.assertTrue(any(g["type"] == "mesh" for g in model["geoms"]))
        self.assertTrue(any(g["type"] == "ellipsoid" for g in model["geoms"]))
        motion = json.loads((self.output/"motion.json").read_text())
        self.assertEqual(motion["rows"][0]["contact_count"], 0)
        self.assertGreater(motion["rows"][-1]["contact_count"], 0)
        self.assertFalse(result["physical_robot_ready"])

    def test_integrity_rejects_modified_payload_and_extra_files(self):
        for mutate in ("payload", "extra", "license", "descriptor"):
            output = self.root / mutate
            shutil.copytree(self.output, output)
            if mutate == "payload":
                (output/"motion.bin.gz").write_bytes(b"invalid")
            elif mutate == "extra":
                (output/"unexpected.json").write_text("{}")
            else:
                path = next((output/"licenses").rglob("LICENSE")) if mutate == "license" else output/"scenes/trainer.json"
                content = b"replacement" if mutate == "license" else json.dumps(library.scene_config("ur5e")).encode()
                path.write_bytes(content)
                seal_path = output/"preview-integrity.json"
                seal = json.loads(seal_path.read_text())
                seal["files"][path.relative_to(output).as_posix()] = {"bytes": len(content), "sha256": sha256(content).hexdigest()}
                seal_path.write_text(json.dumps(seal))
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                preview.inspect_preview(output)

    def test_scene_descriptor_reloads_selected_pair(self):
        config = json.loads((self.output/"scenes/slipper.json").read_text())
        self.assertEqual(config["shoe_ids"], ["slipper-left", "slipper-right"])
        from dvidia_training import reference_sim as sim
        model = library.load_scene(config, os.environ["DVIDIA_ASSET_PREVIEW_TEST_ASSETS"])
        mj, _ = sim._native_imports()
        self.assertGreater(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "shoe_slipper_left"), 0)
        self.assertEqual(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "shoe_trainer_left"), -1)


if __name__ == "__main__":
    unittest.main()
