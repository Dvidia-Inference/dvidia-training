"""Actual local decoding, causal feature tracking and visible failure checks."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import motion_video as mv
from dvidia_training.motion import validate_motion_rows
from dvidia_training.observations import digest, file_hash

try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:
    HAS_CV = False


def manifest_for(directory, video):
    manifest = {'kind': 'dvidia.observation-source.v1', 'task': 'Authored motion fixture',
                'episodes': [{'id': 'fixture', 'path': video.name,
                    'title': 'Generated neutral textured rectangle', 'session_id': 'generated-session',
                    'source_recording_id': 'generated-source', 'kind': 'synthetic',
                    'credit': 'DVIDIA authored test fixture', 'license': 'MIT',
                    'source_url': 'https://example.invalid/authored-test-fixture',
                    'permission': 'local_analysis', 'sha256': file_hash(video)}]}
    target = directory / 'source.json'
    target.write_text(json.dumps(manifest))
    return target


@unittest.skipUnless(HAS_CV and shutil.which('ffmpeg') and shutil.which('ffprobe'),
                     'Requires observations extra and local FFmpeg/ffprobe.')
class MotionVideoTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def video(self, *, count=12, fps=30, moving=True, textured=True,
              pts_filter=None, offset=None):
        width, height = 160, 120
        rng = np.random.default_rng(31415)
        texture = rng.integers(30, 230, (40, 48, 3), dtype='uint8')
        images = []
        for index in range(count):
            image = np.full((height, width, 3), 20, dtype='uint8')
            x = 20 + (index * 2 if moving else 0)
            if textured:
                image[30:70, x:x+48] = texture
            images.append(image.tobytes())
        video = self.root / 'fixture.mp4'
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
                   '-r', str(fps), '-i', 'pipe:0', '-an']
        if pts_filter:
            command.extend(['-vf', pts_filter])
        command.extend(['-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv420p',
                        '-fps_mode', 'vfr'])
        if offset is not None:
            command.extend(['-output_ts_offset', str(offset)])
        command.append(str(video))
        result = subprocess.run(command, input=b''.join(images), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode()[-1000:])
        return video

    def analyze(self, video, name='run', **kwargs):
        return mv.analyze_video(manifest_for(self.root, video), 'fixture', self.root/name,
            seed_box=[24/160, 34/120, 64/160, 66/120], **kwargs)

    def test_actual_decode_tracks_translation_and_preserves_data_hashes(self):
        video = self.video()
        receipt = self.analyze(video)
        self.assertEqual(receipt['status'], 'completed', receipt['errors'])
        self.assertEqual(receipt['frame_count'], 12)
        self.assertEqual(receipt['tracked_frame_count'], 12)
        self.assertFalse(receipt['robot_skill_qualified'])
        self.assertEqual(receipt['accuracy_status'], 'not_assessed')
        self.assertEqual(receipt['seed']['kind'], 'human_authored_image_region')
        self.assertEqual(receipt['protocol']['analysis_image_size_pixels'], [160, 120])
        self.assertEqual(receipt['protocol']['original_image_size_pixels'], [160, 120])
        self.assertEqual(receipt['protocol']['display_geometry']['geometry_status'], 'identity_display_geometry')
        self.assertEqual(receipt['source']['display_geometry']['effective_sample_aspect_ratio'], [1, 1])
        self.assertEqual(file_hash(self.root/'run'/'original.mp4'), file_hash(video))
        self.assertEqual(receipt['media']['bytes'], video.stat().st_size)
        self.assertEqual(receipt['protocol_sha256'], digest(receipt['protocol']))
        self.assertEqual(receipt['rows_sha256'], digest(receipt['rows']))
        self.assertEqual(receipt['frames_sha256'], digest(receipt['frames']))
        self.assertEqual(receipt['seed_sha256'], digest(receipt['seed']))
        for name, key in [('protocol', 'protocol'), ('rows', 'rows'), ('frames', 'frames')]:
            self.assertEqual(json.loads((self.root/'run'/f'{name}.json').read_text()), receipt[key])
        validate_motion_rows(receipt['rows'], 160, 120)
        for frame, row in zip(receipt['frames'], receipt['rows'], strict=True):
            self.assertEqual(frame['source_pts_seconds'], row['acquired_time_seconds'])
            self.assertEqual(frame['source_pts_seconds'], float(frame['source_pts_integer'] *
                             Fraction(*frame['source_time_base'])))
        self.assertAlmostEqual(receipt['rows'][-1]['acquired_time_seconds'], 11/30, places=12)
        for row in receipt['rows'][1:]:
            self.assertAlmostEqual(row['velocity_xy_pixels_per_second'][0], 60, delta=2)
            self.assertAlmostEqual(row['velocity_xy_pixels_per_second'][1], 0, delta=2)
        self.assertAlmostEqual(receipt['rows'][-1]['observed_anchor_xy_pixels'][0], 66, delta=0.5)
        self.assertEqual(set(p.name for p in (self.root/'run').iterdir()),
                         {'original.mp4', 'seed.png', 'protocol.json', 'rows.json', 'frames.json', 'run.json'})

    def test_actual_irregular_pts_are_not_reconstructed_from_nominal_rate(self):
        video = self.video(count=4, fps=10,
                           pts_filter="setpts='if(eq(N,0),0,if(eq(N,1),1,if(eq(N,2),5,6)))'")
        receipt = self.analyze(video)
        self.assertEqual(receipt['status'], 'completed', receipt['errors'])
        stamps = [row['acquired_time_seconds'] for row in receipt['rows']]
        self.assertEqual(len(stamps), 4)
        for actual, expected in zip(stamps, [0, 0.1, 0.5, 0.6], strict=True):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(receipt['rows'][2]['status'], 'lost')
        self.assertEqual(receipt['frames'][2]['tracking']['reason'],
                         'timestamp_gap_exceeds_tracking_envelope')
        self.assertEqual(receipt['rows'][3]['status'], 'lost')
        self.assertEqual(receipt['rows'][3]['forecasts'], [])

    def test_low_texture_loss_is_visible_and_never_reacquires(self):
        video = self.video(count=6, textured=False)
        receipt = self.analyze(video)
        self.assertEqual(receipt['status'], 'completed', receipt['errors'])
        self.assertEqual(receipt['tracked_frame_count'], 0)
        self.assertTrue(all(row['status'] == 'lost' for row in receipt['rows']))
        self.assertTrue(all(row['observed_anchor_xy_pixels'] is None and not row['forecasts']
                            for row in receipt['rows']))
        self.assertEqual(receipt['frames'][0]['tracking']['reason'], 'insufficient_initial_features')
        validate_motion_rows(receipt['rows'], 160, 120)

    def test_overwrite_rejected_and_original_receipt_preserved(self):
        video = self.video(count=3)
        self.analyze(video)
        original = (self.root/'run'/'run.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            self.analyze(video)
        self.assertEqual((self.root/'run'/'run.json').read_bytes(), original)

    def test_changed_original_produces_failed_intake_receipt_without_paths(self):
        video = self.video(count=3)
        manifest = manifest_for(self.root, video)
        data = json.loads(manifest.read_text())
        data['episodes'][0]['sha256'] = '0'*64
        manifest.write_text(json.dumps(data))
        receipt = mv.analyze_video(manifest, 'fixture', self.root/'run',
                                  seed_box=[0.1, 0.1, 0.5, 0.5])
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['errors'][0]['stage'], 'intake')
        self.assertEqual(receipt['rows'], [])
        self.assertIsNone(receipt['media'])
        self.assertNotIn(str(self.root), json.dumps(receipt))
        self.assertTrue((self.root/'run'/'run.json').is_file())

    def test_nonzero_original_timeline_is_rejected(self):
        video = self.video(count=3, offset=1)
        receipt = self.analyze(video)
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['errors'][0]['stage'], 'intake')
        self.assertEqual(receipt['rows'], [])

    def rotated_video(self, source, angle):
        """Create real track rotation, accounting for FFmpeg metadata changes."""
        target = self.root / f'rotation-{angle}.mp4'
        commands = [
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
             '-i', str(source), '-map', '0', '-c', 'copy',
             '-metadata:s:v:0', f'rotate={angle}', str(target)],
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
             '-display_rotation', str(angle), '-i', str(source),
             '-map', '0', '-c', 'copy', str(target)],
        ]
        for command in commands:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=20, check=False)
            if result.returncode:
                continue
            proof = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream_tags=rotate:stream_side_data=side_data_type,displaymatrix,rotation',
                '-of', 'json', str(target)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=10, check=False)
            if proof.returncode or len(proof.stdout) > 65536:
                continue
            stream = json.loads(proof.stdout)['streams'][0]
            angles = [entry.get('rotation', 0) for entry in stream.get('side_data_list', [])]
            if 'rotate' in stream.get('tags', {}):
                angles.append(stream['tags']['rotate'])
            if any(float(value) != 0 for value in angles):
                return target
        self.skipTest(f'Installed FFmpeg cannot encode inspectable {angle}-degree stream-copy rotation metadata.')

    def assert_geometry_intake_rejected(self, video, message):
        receipt = self.analyze(video)
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['errors'][0]['stage'], 'intake')
        self.assertIn(message, receipt['errors'][0]['error'])
        self.assertEqual(receipt['rows'], [])
        self.assertEqual(receipt['frames'], [])
        self.assertIsNone(receipt['media'])
        self.assertFalse((self.root/'run'/'seed.png').exists())
        self.assertFalse(any((self.root/'run').glob('original.*')))
        self.assertTrue((self.root/'run'/'run.json').is_file())

    def test_actual_90_degree_metadata_rotation_is_rejected_before_decode(self):
        source = self.video(count=3)
        self.assert_geometry_intake_rejected(self.rotated_video(source, 90), 'Rotated video')

    def test_actual_180_degree_metadata_rotation_is_rejected_before_decode(self):
        source = self.video(count=3)
        self.assert_geometry_intake_rejected(self.rotated_video(source, 180), 'Rotated video')

    def test_actual_anamorphic_sample_aspect_ratio_is_rejected_before_decode(self):
        video = self.video(count=3, pts_filter='setsar=2/1')
        self.assert_geometry_intake_rejected(video, 'Non-square sample aspect ratio')

    def test_display_matrix_requires_exact_identity_even_if_rotation_is_zero(self):
        identity = '\n00000000: 65536 0 0\n00000001: 0 65536 0\n00000002: 0 0 1073741824\n'
        matrix = {'side_data_type': 'Display Matrix', 'displaymatrix': identity, 'rotation': 0}
        metadata = {'streams': [{'sample_aspect_ratio': '1:1', 'side_data_list': [matrix]}]}
        self.assertEqual(mv._display_geometry_from_metadata(metadata)['display_matrix'], 'explicit-identity')
        for value in (identity.replace('65536 0 0', '-65536 0 0', 1),
                      identity.replace('65536 0 0', '65536 5 0', 1),
                      identity.replace('0 0 1073741824', '100 0 1073741824'),
                      'unknown display matrix'):
            with self.subTest(value=value), self.assertRaises(mv.MotionVideoError):
                mv._display_geometry_from_metadata({'streams': [{'sample_aspect_ratio': '1:1',
                    'side_data_list': [{**matrix, 'displaymatrix': value}]}]})

    def test_rotation_tags_and_unknown_display_metadata_cannot_claim_identity(self):
        for tags in ({'rotate': '90'}, {'rotation': '180'}, {'ROTATE': 'nan'}, {'rotate': True}):
            with self.subTest(tags=tags), self.assertRaises(mv.MotionVideoError):
                mv._display_geometry_from_metadata({'streams': [{'sample_aspect_ratio': '1:1', 'tags': tags}]})
        with self.assertRaises(mv.MotionVideoError):
            mv._display_geometry_from_metadata({'streams': [{'sample_aspect_ratio': '1:1',
                'side_data_list': [{'side_data_type': 'Display Matrix', 'rotation': 0}]}]})
        for ratio in ('16:15', '4:3', '1:0', 'invalid', True):
            with self.subTest(ratio=ratio), self.assertRaises(mv.MotionVideoError):
                mv._display_geometry_from_metadata({'streams': [{'sample_aspect_ratio': ratio}]})
        for ratio in (None, 'N/A', '0:1'):
            source = {} if ratio is None else {'sample_aspect_ratio': ratio}
            self.assertEqual(mv._display_geometry_from_metadata({'streams': [source]})['sample_aspect_ratio_basis'],
                             'unspecified-raster-default')

    def test_invalid_limits_and_unknown_episode_create_no_output(self):
        video = self.video(count=3)
        manifest = manifest_for(self.root, video)
        for extra in ({'sample_hz': 31}, {'sample_hz': True}, {'max_duration_seconds': 61},
                      {'max_frames': 1801}, {'max_frames': True},
                      {'seed_box': [0.5, 0.1, 0.1, 0.5]}):
            args = {'seed_box': [0.1, 0.1, 0.5, 0.5], **extra}
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                mv.analyze_video(manifest, 'fixture', self.root/'invalid', **args)
        with self.assertRaises(ValueError):
            mv.analyze_video(manifest, 'other', self.root/'invalid', seed_box=[0, 0, 1, 1])
        self.assertFalse((self.root/'invalid').exists())

    def test_frame_cap_does_not_create_false_framerate_or_extra_frames(self):
        video = self.video(count=12)
        receipt = self.analyze(video, max_frames=4)
        self.assertEqual(receipt['status'], 'completed', receipt['errors'])
        self.assertEqual(receipt['frame_count'], 4)
        self.assertEqual(receipt['protocol']['max_frames'], 4)
        self.assertEqual(receipt['frames'][-1]['processed_index'], 3)

    def test_decoder_failure_retains_partial_rows(self):
        video = self.video(count=3)
        original = mv._decoded_frames

        def interrupted(*args, **kwargs):
            stream = original(*args, **kwargs)
            try:
                yield next(stream)
                raise mv.MotionVideoError('Authored decoder failure fixture.')
            finally:
                stream.close()

        with patch.object(mv, '_decoded_frames', interrupted):
            receipt = self.analyze(video)
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['frame_count'], 1)
        self.assertEqual(receipt['rows_sha256'], digest(receipt['rows']))
        self.assertEqual(receipt['errors'][0]['stage'], 'decode_and_track')
        self.assertTrue((self.root/'run'/'seed.png').is_file())

    def test_no_reacquisition_even_if_later_texture_appears(self):
        tracker = mv._FeatureAnchor(cv2, np, 160, 120, [0.1, 0.1, 0.8, 0.8])
        blank = np.full((120, 160, 3), 30, dtype='uint8')
        center, _, _, _ = tracker.update(0.0, blank)
        self.assertIsNone(center)
        texture = np.random.default_rng(41).integers(0, 256, (120, 160, 3), dtype='uint8')
        center, _, _, state = tracker.update(0.03, texture)
        self.assertIsNone(center)
        self.assertEqual(state['reason'], 'insufficient_initial_features')

    def test_ambiguous_flow_clusters_produce_terminal_loss(self):
        tracker = mv._FeatureAnchor(cv2, np, 160, 120, [0.1, 0.1, 0.8, 0.8])
        image = np.random.default_rng(42).integers(0, 256, (120, 160, 3), dtype='uint8')
        center, _, _, _ = tracker.update(0.0, image)
        self.assertIsNotNone(center)
        # Authored optical-flow evidence: equally supported opposed motions.
        tracker.points = np.array([[40 + i, 50] for i in range(12)],
                                  dtype='float32').reshape(-1, 1, 2)
        old = tracker.points.copy()
        forward = old.copy()
        forward[:6, :, 0] += 8
        forward[6:, :, 0] -= 8
        ok = np.ones((12, 1), dtype='uint8')
        with patch.object(cv2, 'calcOpticalFlowPyrLK',
                          side_effect=[(forward, ok, None), (old, ok, None)]):
            center, support, _, state = tracker.update(0.03, image)
        self.assertIsNone(center)
        self.assertEqual(support, 0)
        self.assertEqual(state['reason'], 'ambiguous_feature_motion')

    def test_decoder_watchdog_stops_blocked_owned_process(self):
        original_popen = subprocess.Popen
        processes = []

        def blocked(*args, **kwargs):
            child = original_popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                  **kwargs)
            processes.append(child)
            return child

        with patch.object(mv.subprocess, 'Popen', side_effect=blocked):
            with self.assertRaisesRegex(mv.MotionVideoError, 'watchdog'):
                list(mv._decoded_frames(self.root/'never-read.mp4', 16, 16,
                    sample_hz=30, max_frames=2, duration=1, timeout_seconds=0.05))
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())

    def test_excess_decoder_diagnostics_are_bounded_and_child_stopped(self):
        original_popen = subprocess.Popen
        processes = []

        def excessive(*args, **kwargs):
            child = original_popen([sys.executable, '-c',
                'import sys,time; sys.stderr.write("x"*9000); sys.stderr.flush(); time.sleep(30)'],
                **kwargs)
            processes.append(child)
            return child

        with patch.object(mv.subprocess, 'Popen', side_effect=excessive):
            with self.assertRaisesRegex(mv.MotionVideoError, 'bounded metadata budget'):
                list(mv._decoded_frames(self.root/'never-read.mp4', 16, 16,
                    sample_hz=30, max_frames=2, duration=1, timeout_seconds=2))
        self.assertIsNotNone(processes[0].poll())

    def test_display_geometry_probe_watchdog_stops_blocked_owned_child(self):
        original_popen = subprocess.Popen
        processes = []

        def blocked(*args, **kwargs):
            child = original_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
            processes.append(child)
            return child

        with patch.object(mv.subprocess, 'Popen', side_effect=blocked):
            with self.assertRaisesRegex(mv.MotionVideoError, '10-second watchdog'):
                mv._probe_display_geometry(self.root/'never-read.mp4', timeout_seconds=0.05)
        self.assertIsNotNone(processes[0].poll())

    def test_display_geometry_probe_stdout_is_bounded_and_child_stopped(self):
        original_popen = subprocess.Popen
        processes = []

        def oversized(*args, **kwargs):
            child = original_popen([sys.executable, '-c',
                'import sys,time; sys.stdout.write("x"*70000); sys.stdout.flush(); time.sleep(30)'], **kwargs)
            processes.append(child)
            return child

        with patch.object(mv.subprocess, 'Popen', side_effect=oversized):
            with self.assertRaisesRegex(mv.MotionVideoError, '64 KiB limit'):
                mv._probe_display_geometry(self.root/'never-read.mp4', timeout_seconds=2)
        self.assertIsNotNone(processes[0].poll())

    def test_public_display_guard_rejects_missing_files_and_symlinks(self):
        with self.assertRaisesRegex(mv.MotionVideoError, 'regular local video'):
            mv.validate_display_geometry(self.root/'missing.mp4')
        video = self.video(count=3)
        link = self.root/'linked.mp4'
        try:
            link.symlink_to(video)
        except (OSError, NotImplementedError):
            self.skipTest('This platform does not permit creating a symlink fixture.')
        with self.assertRaisesRegex(mv.MotionVideoError, 'regular local video'):
            mv.validate_display_geometry(link)


if __name__ == '__main__':
    unittest.main()
