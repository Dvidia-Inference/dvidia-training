"""Pinned offline NVIDIA Fabric appearance adapter contract tests."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

from dvidia_training import material_appearance as appearance


def _fixture_appearance(folder: Path, size: int = 64) -> Path:
    folder.mkdir()
    rgb = bytes((index * 13) % 256 for index in range(size * size * 3))
    png = appearance._png_rgb(rgb, size)
    (folder / "texture.rgb").write_bytes(rgb)
    (folder / "texture.png").write_bytes(png)
    manifest = {
        "schema": appearance.SCHEMA,
        "material_id": "Fabric_Test",
        "source": {
            "repository": appearance.REPOSITORY, "release": appearance.RELEASE,
            "release_url": appearance.RELEASE_URL, "source_sha": appearance.SOURCE_SHA,
            "license": appearance.LICENSE, "archive_sha256": appearance.ARCHIVE_SHA256,
            "material_path": "Materials/Fabric/Fabric_Test.usda",
            "material_sha256": "2" * 64,
            "diffuse_texture_path": "Materials/Fabric/textures/Fabric_Test_diffuse.jpg",
            "diffuse_texture_sha256": "3" * 64, "diffuse_texture_bytes": 123,
        },
        "texture": {
            "width": size, "height": size, "channels": 3, "pixel_format": "RGB24",
            "resize_method": "FFmpeg bilinear exact-square resize", "rgb_path": "texture.rgb",
            "rgb_bytes": len(rgb), "rgb_sha256": appearance._sha(rgb), "png_path": "texture.png",
            "png_bytes": len(png), "png_sha256": appearance._sha(png),
        },
        "appearance_kind": "diffuse_albedo_approximation",
        "limitations": list(appearance._LIMITATIONS),
    }
    (folder / "appearance.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


class MaterialAppearanceTests(unittest.TestCase):
    def test_archive_inventory_rejects_traversal_duplicates_and_symlinks(self):
        cases = [
            [("../escape", b"x")],
            [("Materials//Fabric/a.usda", b"x")],
            [("Materials/Fabric/a.usda", b"a"), ("Materials/Fabric/a.usda", b"b")],
        ]
        for entries in cases:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "bad.zip"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(path, "w") as archive:
                        for name, content in entries:
                            archive.writestr(name, content)
                with zipfile.ZipFile(path) as archive, self.assertRaises(ValueError):
                    appearance._safe_zip_inventory(archive)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "link.zip"
            info = zipfile.ZipInfo("Materials/Fabric/link.usda")
            info.create_system = 3
            info.external_attr = (0o120777 << 16)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(info, "target")
            with zipfile.ZipFile(path) as archive, self.assertRaisesRegex(ValueError, "symlinks"):
                appearance._safe_zip_inventory(archive)

    def test_real_archive_is_optional_but_pinned_when_supplied(self):
        path = os.environ.get("DVIDIA_FABRIC_ARCHIVE")
        if not path:
            self.skipTest("DVIDIA_FABRIC_ARCHIVE is not set")
        result = appearance.catalog_archive(path)
        self.assertEqual(result["archive_sha256"], appearance.ARCHIVE_SHA256)
        self.assertEqual(len(result["materials"]), 13)
        self.assertIn("Fabric_Canvas", {row["id"] for row in result["materials"]})

    def test_prepare_rejects_unknown_material_size_and_existing_output(self):
        # These checks happen before archive access or output creation.
        for material, size in [("../Fabric_Canvas", 64), ("Fabric_Canvas", 63),
                               ("Fabric_Canvas", 513), ("Fabric_Canvas", True)]:
            with self.subTest(material=material, size=size), self.assertRaises(ValueError):
                appearance.prepare_appearance("missing.zip", material, "not-created", size)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "present"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "new directory"):
                appearance.prepare_appearance("missing.zip", "Fabric_Canvas", output, 64)

    def test_load_appearance_rejects_tampered_manifest_and_texture(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = _fixture_appearance(Path(temp) / "appearance")
            metadata, rgb = appearance.load_appearance(folder)
            self.assertEqual(metadata["material_id"], "Fabric_Test")
            self.assertEqual(rgb.shape, (64, 64, 3))
            original = (folder / "appearance.json").read_text()
            data = json.loads(original)
            data["source"]["release"] = "latest"
            (folder / "appearance.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                appearance.load_appearance(folder)
            (folder / "appearance.json").write_text(original)
            (folder / "texture.rgb").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                appearance.load_appearance(folder)

    def test_apply_changes_only_visual_material_fields_and_compiled_mechanics(self):
        import mujoco as mj

        with tempfile.TemporaryDirectory() as temp:
            folder = _fixture_appearance(Path(temp) / "appearance")

            def make_spec():
                spec = mj.MjSpec()
                spec.option.gravity = [0, 0, -9.81]
                body = spec.worldbody.add_body(name="payload", pos=[0, 0, 0.3])
                body.add_joint(name="hinge", type=mj.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
                body.add_geom(name="shoe_upper", type=mj.mjtGeom.mjGEOM_BOX,
                              size=[.1, .05, .03], mass=.4, friction=[.8, .02, .003],
                              condim=4, contype=1, conaffinity=1)
                body.add_geom(name="sole", type=mj.mjtGeom.mjGEOM_BOX,
                              pos=[0, 0, -.04], size=[.12, .06, .01], mass=.2,
                              friction=[.9, .03, .004], condim=4, contype=1, conaffinity=1)
                spec.add_actuator(name="motor", target="hinge", trntype=mj.mjtTrn.mjTRN_JOINT,
                                  ctrlrange=[-1, 1],
                                  ctrllimited=True, gear=[0, 0, 0, 0, 2, 0])
                return spec

            plain = make_spec().compile()
            spec = make_spec()
            receipt = appearance.apply_to_spec(spec, folder, ["shoe_upper"], prefix="sample")
            textured = spec.compile()
            self.assertEqual(receipt["geom_names"], ["shoe_upper"])
            self.assertEqual(textured.ntex, 1)
            self.assertEqual(textured.nmat, 1)
            self.assertEqual(int(textured.geom_matid[mj.mj_name2id(textured, mj.mjtObj.mjOBJ_GEOM, "shoe_upper")]), 0)
            for attr in ("body_mass", "body_inertia", "geom_friction", "geom_contype",
                         "geom_conaffinity", "geom_condim", "geom_size", "geom_pos",
                         "jnt_type", "jnt_axis", "actuator_trnid", "actuator_gear",
                         "actuator_ctrlrange", "actuator_ctrllimited"):
                with self.subTest(attribute=attr):
                    np.testing.assert_array_equal(getattr(plain, attr), getattr(textured, attr))
            np.testing.assert_array_equal(plain.opt.gravity, textured.opt.gravity)
            self.assertEqual((plain.nq, plain.nv, plain.nu, plain.nbody, plain.ngeom),
                             (textured.nq, textured.nv, textured.nu, textured.nbody, textured.ngeom))

    def test_apply_selection_validation(self):
        import mujoco as mj

        with tempfile.TemporaryDirectory() as temp:
            folder = _fixture_appearance(Path(temp) / "appearance")
            spec = mj.MjSpec()
            spec.worldbody.add_geom(name="target", type=mj.mjtGeom.mjGEOM_BOX, size=[1, 1, 1])
            for names in ([], ["missing"], ["target", "target"], "target", [True]):
                with self.subTest(names=names), self.assertRaises(ValueError):
                    appearance.apply_to_spec(spec, folder, names)


if __name__ == "__main__":
    unittest.main()
