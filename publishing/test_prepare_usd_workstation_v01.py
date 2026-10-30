"""Read-only public evidence checks; no native rollouts or credentials."""
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import prepare_usd_workstation_v01 as release


class PublicArchiveTests(unittest.TestCase):
    def test_zip_is_deterministic_preserves_all_bytes_and_stays_outside_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root/"public"
            (payload/"nested").mkdir(parents=True)
            (payload/"nested/data.json").write_bytes(b'{"x":1}\n')
            (payload/"LICENSE").write_bytes(b"original license\n")
            one = release.deterministic_zip(payload, root/"one.zip")
            two = release.deterministic_zip(payload, root/"two.zip")
            self.assertEqual(one, two)
            self.assertEqual((root/"one.zip").read_bytes(), (root/"two.zip").read_bytes())
            with zipfile.ZipFile(root/"one.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(archive.namelist(), ["LICENSE", "nested/data.json"])
                for name in archive.namelist():
                    self.assertEqual(archive.read(name), (payload/name).read_bytes())
            with self.assertRaises(ValueError):
                release.deterministic_zip(payload, payload/"archive.zip")
            with self.assertRaises(ValueError):
                release.deterministic_zip(payload, root/"one.zip")

    def test_public_text_scanner_rejects_local_paths_and_credential_shapes(self):
        for raw in (b"/"+b"Users/person/private.json", b"api"+b"key_"+b"a"*30, b"hf_"+b"z"*35):
            with self.subTest(raw_kind=raw[:4]), self.assertRaises(ValueError):
                release.public_text_check("README.md", raw)
        release.public_text_check("README.md", b"https://research.dvidia.org/ original MIT scene")

    def test_json_and_inventory_reject_ambiguous_or_linked_evidence(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                release.parse(raw)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"real").write_text("retained")
            (root/"link").symlink_to(root/"real")
            with self.assertRaises(ValueError):
                release.inventory(root)


STUDY = release.ROOT/"runs/usd-workstation-v01-20261030"
HAS_USD = importlib.util.find_spec("pxr") is not None


@unittest.skipUnless(STUDY.is_dir() and HAS_USD, "Requires the retained 45-case study and optional USD SDK; no physics is rerun")
class RecordedStudyAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)/"study"
        shutil.copytree(STUDY, self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_45_recorded_cases_rescore_and_all_pose_frames_match(self):
        audited = release.audit_study(self.root)
        self.assertEqual(audited["counts"], {"scheduled": 45, "valid": 45, "passed": 45,
            "normal_scheduled": 36, "normal_passed": 36, "negative_scheduled": 9, "negative_passed": 9})
        self.assertEqual(audited["mismatches"], 0)
        self.assertEqual(len(audited["cases"]), 45)
        self.assertEqual(audited["recorded_frames"], 33*201+12*101)
        self.assertLessEqual(audited["max_object_xyz_error_m"], 1e-6)
        self.assertTrue(any("1 ms" in text for text in audited["limitations"]))

    def test_missing_failed_or_control_attempt_cannot_disappear(self):
        shutil.rmtree(self.root/"cases/case-44")
        with self.assertRaisesRegex(ValueError, "membership"):
            release.audit_study(self.root)

    def test_pose_hash_relabel_cannot_hide_incorrect_object_geometry(self):
        import numpy as np
        path = self.root/"cases/case-00/poses.bin.gz"
        raw = gzip.decompress(path.read_bytes())
        poses = np.frombuffer(raw, dtype="<f4").copy().reshape((201, 11, 7))
        poses[100, 10, 0] += .01
        replacement = poses.tobytes()
        path.write_bytes(gzip.compress(replacement, mtime=0))
        motion_path = self.root/"motion.json"
        motion = json.loads(motion_path.read_bytes())
        motion["cases"][0]["pose_identity"] = release.identity(replacement)
        motion_path.write_bytes(release.encoded(motion))
        with self.assertRaisesRegex(ValueError, "XYZ differs"):
            release.audit_study(self.root)

    def test_result_score_claim_is_reconstructed_from_its_saved_trace(self):
        path = self.root/"cases/case-00/result.json"
        result = json.loads(path.read_bytes())
        result["checks"]["supported"] = False
        path.write_bytes(release.encoded(result))
        with self.assertRaisesRegex(ValueError, "trajectory score"):
            release.audit_study(self.root)


if __name__ == "__main__":
    unittest.main()
