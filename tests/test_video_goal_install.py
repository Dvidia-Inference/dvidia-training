import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from dvidia_training import video_goal_install as install
from dvidia_training.video_goal_bridge import PlanarVideoCalibration


def _valid_proposal(video, calibration):
    digest = hashlib.sha256(Path(video).read_bytes()).hexdigest()
    return {
        "schema": install.video_goal_bridge.SCHEMA,
        "status": "proposed",
        "reason": "stable_visible_terminal_xy",
        "goal_xy_m": [0.40, 0.08],
        "frame_count": 3,
        "observations": [{"t_s": t, "status": "observed", "position_xy_m": [0.40, 0.08],
                          "reason": "single_green_candidate"} for t in (0.0, 0.2, 0.4)],
        "timing_ms": {"decode": 1.0, "observe": 1.0, "total": 2.0},
        "video_sha256": digest,
        "video_duration_s": 0.6,
        "last_sample_time_s": 0.4,
        "unobserved_tail_s": 0.2,
        "calibration": asdict(calibration),
        "robot_policy_learned": False,
        "scope": install.video_goal_bridge.SCOPE,
        "limitations": list(install.video_goal_bridge.LIMITATIONS),
        "timing_scope": ("Local stat, probe, decode, observation and source hashing wall time; "
                         "excludes acquisition, terminal dwell before upload, network and actuation."),
    }


class _Config:
    def __init__(self, *, object_position_m, target_position_m):
        assert len(object_position_m) == len(target_position_m) == 3
        assert .25 <= target_position_m[0] <= .48
        assert -.15 <= target_position_m[1] <= .15
        assert target_position_m[2] == .08
        self.object_position_m = object_position_m
        self.target_position_m = target_position_m


def _fake_piper_module(run_episode=None):
    module = ModuleType("dvidia_training.piper_task")
    module.PiperTaskConfig = _Config
    if run_episode is not None:
        module.run_episode = run_episode
    return module


class VideoGoalInstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.video = self.tmp_path / "source.mp4"
        self.video.write_bytes(b"synthetic-source-video")
        self.calibration = PlanarVideoCalibration()
        self.proposal = _valid_proposal(self.video, self.calibration)
        self.fake_runtime = {"profile": "piper", "source_sha256": {"test": "a" * 64},
                             "python": "3.test", "numpy": "test", "mujoco": "test"}
        self._install_patches()
        self.bundle = self.tmp_path / "candidate"
        self.manifest = install.create_candidate(self.video, self.calibration, self.bundle)

    def _install_patches(self):
        for context in (
            patch.object(install, "infer_video_goal", return_value=self.proposal),
            patch.object(install, "_runtime", return_value=self.fake_runtime),
            patch.dict(sys.modules, {"dvidia_training.piper_task": _fake_piper_module()}),
        ):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def test_create_and_load_round_trip_is_data_only(self):
        self.assertEqual({path.name for path in self.bundle.iterdir()},
                         {"manifest.json", "proposal.json", "calibration.json"})
        self.assertFalse(any(path.read_bytes() == self.video.read_bytes() for path in self.bundle.iterdir()))
        loaded = install.load_candidate(self.bundle)
        self.assertEqual(loaded["manifest"], self.manifest)
        self.assertEqual(loaded["proposal"]["goal_xy_m"], [0.40, 0.08])
        self.assertEqual(loaded["manifest"]["video"], {
            "sha256": hashlib.sha256(self.video.read_bytes()).hexdigest(),
            "bytes": self.video.stat().st_size,
            "included": False,
        })

    def test_tampered_bundle_and_stale_runtime_are_rejected(self):
        (self.bundle / "proposal.json").write_text("{}", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
            install.load_candidate(self.bundle)
        pristine = self.tmp_path / "pristine"
        with patch.object(install, "infer_video_goal", return_value=self.proposal), \
                patch.object(install, "_runtime", return_value=self.fake_runtime), \
                patch.dict(sys.modules, {"dvidia_training.piper_task": _fake_piper_module()}):
            install.create_candidate(self.video, self.calibration, pristine)
        with patch.object(install, "_runtime", return_value={**self.fake_runtime, "numpy": "changed"}), \
                patch.dict(sys.modules, {"dvidia_training.piper_task": _fake_piper_module()}):
            with self.assertRaisesRegex(ValueError, "runtime fingerprint is stale"):
                install.load_candidate(pristine)

    def test_unknown_proposal_and_unsupported_profile_are_rejected(self):
        unknown = _valid_proposal(self.video, self.calibration)
        unknown.update(status="unknown", goal_xy_m=None)
        output = self.tmp_path / "should-not-exist"
        with patch.object(install, "infer_video_goal", return_value=unknown), \
                patch.dict(sys.modules, {"dvidia_training.piper_task": _fake_piper_module()}):
            with self.assertRaisesRegex(ValueError, "Only a proposed"):
                install.create_candidate(self.video, self.calibration, output)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "Only the piper"):
            install.load_candidate(self.bundle, profile="other-arm")

    def test_unknown_candidate_cannot_reach_episode_runner(self):
        proposal = json.loads((self.bundle / "proposal.json").read_text())
        proposal.update(status="unknown", goal_xy_m=None)
        raw = json.dumps(proposal, indent=2, ensure_ascii=False, allow_nan=False).encode("utf8") + b"\n"
        (self.bundle / "proposal.json").write_bytes(raw)
        manifest = json.loads((self.bundle / "manifest.json").read_text())
        manifest["files"]["proposal.json"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        video = manifest["video"]
        binding = {"profile": "piper", "video_sha256": video["sha256"], "video_bytes": video["bytes"],
                   "goal_xy_m": manifest["goal_xy_m"], "files": manifest["files"], "runtime": self.fake_runtime}
        manifest["candidate_id"] = hashlib.sha256(install._canonical(binding)).hexdigest()[:24]
        (self.bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        calls = []
        fake_piper = _fake_piper_module(run_episode=lambda *args, **kwargs: calls.append((args, kwargs)))
        with patch.dict(sys.modules, {"dvidia_training.piper_task": fake_piper}):
            with self.assertRaisesRegex(ValueError, "Only a proposed"):
                install.execute_candidate(self.bundle, self.tmp_path, (.30, -.03), self.tmp_path / "run")
        self.assertEqual(calls, [])
        self.assertFalse((self.tmp_path / "run").exists())

    def test_execution_receipt_binds_goal_bundle_and_compressed_episode(self):
        episode = {"valid": True, "success": False, "reason": "test", "metrics": {"steps": 2}}
        calls = []
        fake_piper = _fake_piper_module(run_episode=lambda assets, config, **kwargs: (
            calls.append((assets, config, kwargs)) or episode))
        profiles = ModuleType("dvidia_training.robot_profiles")
        profiles.verify_assets = lambda profile, assets: {"profile": profile, "verified": True}
        output = self.tmp_path / "run"
        with patch.dict(sys.modules, {"dvidia_training.piper_task": fake_piper,
                                     "dvidia_training.robot_profiles": profiles}):
            receipt = install.execute_candidate(self.bundle, self.tmp_path / "assets", (.30, -.03), output, seed=7)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], {"seed": 7, "control": "placement"})
        self.assertEqual(calls[0][1].target_position_m, (.40, .08, .08))
        self.assertEqual(receipt["goal_xy_m"], [.40, .08])
        self.assertEqual(receipt["bundle_manifest_sha256"],
                         hashlib.sha256((self.bundle / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(receipt["episode_file"], "episode.json.gz")
        self.assertEqual(receipt["runtime_fingerprint"], install._sha_bytes(install._canonical(self.fake_runtime)))
        with gzip.open(output / receipt["episode_file"], "rt", encoding="utf8") as stream:
            self.assertEqual(json.load(stream), episode)
        self.assertTrue((output / "receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
