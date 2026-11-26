from hashlib import sha256
from importlib.resources import files
from unittest import TestCase
from unittest.mock import patch
from xml.etree import ElementTree

from dvidia_training.unitree_r1_assets import source_info, urdf_bytes


class UnitreeAssetsTests(TestCase):
    def test_both_canonical_descriptions_are_distinct_and_include_seven_axes(self):
        for side, wrist_range in (("left", "0.524"), ("right", "0.785")):
            with self.subTest(side=side):
                info = source_info(side)
                data = urdf_bytes(side)
                self.assertEqual(sha256(data).hexdigest(), info["urdf_sha256"])
                self.assertIn("/urdf/", info["archive_member"])
                joints = ElementTree.fromstring(data).findall("joint")
                self.assertEqual([joint.get("name") for joint in joints], [f"J{i}" for i in range(1, 8)])
                self.assertEqual({joint.get("type") for joint in joints}, {"revolute"})
                self.assertEqual(joints[5].find("limit").get("upper"), wrist_range)
        self.assertNotEqual(source_info("left")["urdf_sha256"], source_info("right")["urdf_sha256"])

    def test_license_matches_original_unitree_notice(self):
        notice = files("dvidia_training").joinpath("unitree_r1_7a", "LICENSE.txt").read_bytes()
        self.assertEqual(sha256(notice).hexdigest(), source_info("right")["license_sha256"])
        self.assertIn(b"Unitree Robotics", notice)

    def test_only_an_explicit_supported_side_is_accepted(self):
        for invalid in ("", "R", "../right", None):
            with self.subTest(side=invalid), self.assertRaises(ValueError):
                urdf_bytes(invalid)

    def test_corrupted_resource_is_rejected(self):
        with patch("dvidia_training.unitree_r1_assets.files") as resources:
            resources.return_value.joinpath.return_value.read_bytes.return_value = b"wrong model"
            with self.assertRaisesRegex(ValueError, "differs from the pinned"):
                urdf_bytes("right")
