"""Spatial-only video goal proposal contract and bounded media decode tests."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dvidia_training.video_goal_bridge import (
    PlanarVideoCalibration,
    infer_frames,
    infer_video_goal,
    _read_json,
)


def rgb_frame(center=(160, 120), *, width=320, height=240, size=20, color=(20, 180, 30)):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = center
    half = size // 2
    image[max(0, cy-half):min(height, cy+half), max(0, cx-half):min(width, cx+half)] = color
    return image


def frames(centers, times=None):
    if times is None:
        times = [index * .2 for index in range(len(centers))]
    return [{"t_s": stamp, "rgb": rgb_frame(center)} for stamp, center in zip(times, centers)]


class PlanarVideoCalibrationTests(unittest.TestCase):
    def test_calibration_rejects_bool_and_out_of_range_values(self):
        with self.assertRaises(ValueError):
            PlanarVideoCalibration(width=True)
        with self.assertRaises(ValueError):
            PlanarVideoCalibration(center_xy_m=(.36, False))
        with self.assertRaises(ValueError):
            PlanarVideoCalibration(vertical_span_m=float("nan"))
        with self.assertRaises(ValueError):
            PlanarVideoCalibration(terminal_dwell_s=.49)


class InferFramesTests(unittest.TestCase):
    def test_calibrated_projection_scales_x_and_flips_image_y(self):
        calibration = PlanarVideoCalibration(center_xy_m=(.36, .1), vertical_span_m=.5)
        result = infer_frames(frames([(200, 80)] * 5), calibration)
        self.assertEqual(result["status"], "proposed")
        scale = .5 / 240
        self.assertAlmostEqual(result["goal_xy_m"][0], .36 + 40 * scale, places=6)
        self.assertAlmostEqual(result["goal_xy_m"][1], .1 + 40 * scale, places=6)
        self.assertFalse(result["robot_policy_learned"])

    def test_missing_multiple_and_clipped_green_regions_are_unknown(self):
        calibration = PlanarVideoCalibration()
        no_target = np.zeros((240, 320, 3), dtype=np.uint8)
        absent = infer_frames([{"t_s": i * .2, "rgb": no_target.copy()} for i in range(5)], calibration)
        self.assertEqual(absent["status"], "unknown")
        self.assertEqual(absent["observations"][-1]["reason"], "no_green_candidate")

        two = rgb_frame()
        two[40:60, 40:60] = (20, 180, 30)
        ambiguous = infer_frames([{"t_s": i * .2, "rgb": two.copy()} for i in range(5)], calibration)
        self.assertEqual(ambiguous["status"], "unknown")
        self.assertEqual(ambiguous["observations"][-1]["reason"], "ambiguous_green_candidates")

        clipped = np.zeros((240, 320, 3), dtype=np.uint8)
        clipped[50:70, 0:20] = (20, 180, 30)
        edge = infer_frames([{"t_s": i * .2, "rgb": clipped.copy()} for i in range(5)], calibration)
        self.assertEqual(edge["status"], "unknown")
        self.assertEqual(edge["observations"][-1]["reason"], "candidate_clipped_by_image_edge")

    def test_moving_goal_fails_stability_and_stable_terminal_window_proposes(self):
        moving = infer_frames(frames([(80, 120), (110, 120), (140, 120), (170, 120), (200, 120)]))
        self.assertEqual(moving["status"], "unknown")
        self.assertEqual(moving["reason"], "terminal_position_spread_exceeded")

        stable = infer_frames(frames([(100, 120), (120, 120), (150, 120), (150, 120), (150, 120), (150, 120)]))
        self.assertEqual(stable["status"], "proposed")
        self.assertAlmostEqual(stable["goal_xy_m"][0], .36 - 10 * (.5 / 240), places=6)

    def test_terminal_occlusion_does_not_fall_back_to_an_older_visible_goal(self):
        rows = frames([(150, 120)] * 4)
        rows.append({"t_s": .8, "rgb": np.zeros((240, 320, 3), dtype=np.uint8)})
        result = infer_frames(rows)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "final_decoded_frame_unobservable")
        self.assertIsNone(result["goal_xy_m"])

    def test_clip_tail_past_last_selected_sample_forces_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delayed-tail.mp4"
            path.write_bytes(b"bounded fixture")
            decoded = frames([(150, 120)] * 5)
            with patch("dvidia_training.video_goal_bridge.probe_video", return_value={
                    "width": 320, "height": 240, "start_seconds": 0., "duration_seconds": 1.1}), \
                 patch("dvidia_training.video_goal_bridge._decode_video", return_value=(decoded, 4.)):
                result = infer_video_goal(path)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "unobserved_clip_tail_exceeded")
        self.assertIsNone(result["goal_xy_m"])
        self.assertAlmostEqual(result["video_duration_s"], 1.1)
        self.assertAlmostEqual(result["last_sample_time_s"], .8)
        self.assertAlmostEqual(result["unobserved_tail_s"], .3)
        self.assertIn("source hashing", result["timing_scope"])

    def test_bad_frame_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            infer_frames([{"t_s": 0, "rgb": rgb_frame()}] * 2)
        broken = frames([(150, 120)] * 3)
        broken[2]["t_s"] = broken[1]["t_s"]
        with self.assertRaises(ValueError):
            infer_frames(broken)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are required")
class VideoDecodeTests(unittest.TestCase):
    def test_encoded_local_mp4_uses_real_pts_and_hashes_original_video(self):
        calibration = PlanarVideoCalibration()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "green-target.mp4"
            source = np.stack([rgb_frame((150, 120)) for _ in range(10)])
            encode = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", "320x240", "-r", "10", "-i", "pipe:0",
                 "-frames:v", "10", "-an", "-c:v", "mpeg4", "-q:v", "2", str(path)],
                input=source.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(encode.returncode, 0, encode.stderr.decode("utf8", "replace"))
            result = infer_video_goal(path, calibration)
            self.assertEqual(result["status"], "proposed")
            self.assertEqual(result["frame_count"], len(result["observations"]))
            self.assertEqual(result["observations"][0]["t_s"], 0.)
            self.assertTrue(all(b["t_s"] > a["t_s"] for a, b in zip(result["observations"], result["observations"][1:])))
            self.assertEqual(len(result["video_sha256"]), 64)
            self.assertGreaterEqual(result["timing_ms"]["decode"], 0)
            self.assertFalse(result["robot_policy_learned"])

    def test_rejects_bad_media_path_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "bad.mp4"
            fake.write_bytes(b"not video")
            with self.assertRaises(ValueError):
                infer_video_goal(fake)
            non_video = root / "frame.png"
            non_video.write_bytes(b"x")
            with self.assertRaises(ValueError):
                infer_video_goal(non_video)

    def test_calibration_file_is_bounded_and_unknown_fields_are_value_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({"width": 320, "made_up": 1}), encoding="utf8")
            with self.assertRaises(ValueError):
                _read_json(path)
            path.write_bytes(b" " * (64 * 1024 + 1))
            with self.assertRaises(ValueError):
                _read_json(path)


if __name__ == "__main__":
    unittest.main()
