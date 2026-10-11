"""Real generated motion evidence, read-only replay and tamper rejection."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest

from dvidia_training.footage_data import canonical
from dvidia_training.motion_benchmark import score_motion
from dvidia_training.motion_studio import MotionServer, load_motion_bundle, load_motion_run
from dvidia_training.motion_video import analyze_video
from dvidia_training.observations import digest, file_hash

try:
    import numpy as np
    import cv2
    HAS_CV = True
except ImportError:
    HAS_CV = False


def write_document(path, value):
    Path(path).write_bytes(canonical(value))


def write_run_parts(directory, run):
    for filename, key in (('protocol.json', 'protocol'), ('rows.json', 'rows'),
                          ('frames.json', 'frames')):
        write_document(directory/filename, run[key])
        run[key+'_sha256'] = digest(run[key])
    write_document(directory/'run.json', run)


@unittest.skipUnless(HAS_CV and shutil.which('ffmpeg') and shutil.which('ffprobe'),
                     'Requires observations extra and local FFmpeg/ffprobe.')
class MotionStudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.base_temp.name)
        width, height, count = 160, 120, 12
        texture = np.random.default_rng(4711).integers(30, 230, (40, 48, 3), dtype='uint8')
        frames = []
        for index in range(count):
            frame = np.full((height, width, 3), 20, dtype='uint8')
            frame[30:70, 20+2*index:68+2*index] = texture
            frames.append(frame.tobytes())
        cls.video = cls.base/'authored.mp4'
        result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '160x120', '-r', '30', '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv420p', '-threads', '1',
            '-fps_mode', 'vfr', str(cls.video)], input=b''.join(frames), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=20, check=False)
        if result.returncode:
            raise RuntimeError('Could not generate own motion-studio fixture.')
        cls.manifest = {'kind': 'dvidia.observation-source.v1', 'task': 'Authored replay fixture',
            'episodes': [{'id': 'fixture', 'path': cls.video.name,
                'title': 'Own neutral textured rectangle', 'session_id': 'studio-test-session',
                'source_recording_id': 'studio-test-recording', 'kind': 'synthetic',
                'credit': 'DVIDIA authored engineering fixture', 'license': 'MIT',
                'source_url': 'https://example.invalid/dvidia-authored-fixture',
                'permission': 'local_analysis', 'sha256': file_hash(cls.video)}]}
        write_document(cls.base/'source.json', cls.manifest)
        cls.seed = [24/width, 34/height, 64/width, 66/height]
        cls.receipt = analyze_video(cls.base/'source.json', 'fixture', cls.base/'complete',
                                   seed_box=cls.seed)
        if cls.receipt['status'] != 'completed' or len(cls.receipt['rows']) != count:
            raise RuntimeError('Own fixture did not produce a complete motion run.')
        wrong = deepcopy(cls.manifest)
        wrong['episodes'][0]['sha256'] = '0'*64
        write_document(cls.base/'wrong-source.json', wrong)
        cls.failed_receipt = analyze_video(cls.base/'wrong-source.json', 'fixture', cls.base/'failed',
                                          seed_box=cls.seed)
        if cls.failed_receipt['status'] != 'failed' or cls.failed_receipt['rows']:
            raise RuntimeError('Own wrong-hash fixture did not preserve its failed intake.')
        # One-case benchmark envelope around the genuine generated run. The
        # authored trajectory is evaluator-only and never enters analyze_video.
        cls.benchmark = cls.base/'benchmark'
        (cls.benchmark/'inputs').mkdir(parents=True)
        shutil.copyfile(cls.video, cls.benchmark/'inputs'/cls.video.name)
        shutil.copyfile(cls.base/'source.json', cls.benchmark/'inputs/source.json')
        shutil.copytree(cls.base/'complete', cls.benchmark/'cases/fixture')
        references = [{'frame_index': index, 'source_pts_seconds': row['acquired_time_seconds'],
            'anchor_xy_pixels': [44+2*index, 50], 'visible': True,
            'velocity_scoreable': index > 0,
            'velocity_xy_pixels_per_second': [60, 0] if index else None}
            for index, row in enumerate(cls.receipt['rows'])]
        reference_path = cls.benchmark/'inputs/fixture.reference.json'
        write_document(reference_path, {'kind': 'dvidia.authored-motion-reference.v1',
            'role': 'evaluator_only_never_supplied_to_tracker', 'rows': references})
        scores = score_motion(cls.receipt['rows'], references)
        write_document(cls.benchmark/'cases/fixture/scores.json', scores)
        benchmark_protocol = {'kind': 'dvidia.motion-benchmark-protocol.v1',
            'fixture_recipe': {'fps': 30}, 'horizons_seconds': [.1, .25, .5],
            'sources_manifest_sha256': file_hash(cls.benchmark/'inputs/source.json'),
            'fixtures': [{'id': 'fixture', 'video_path': cls.video.name,
                'video_sha256': file_hash(cls.video), 'reference_path': reference_path.name,
                'reference_sha256': file_hash(reference_path),
                'seed_box_xyxy_normalized': cls.seed}]}
        write_document(cls.benchmark/'protocol.json', benchmark_protocol)
        write_document(cls.benchmark/'report.json', {'kind': 'dvidia.motion-benchmark.v1',
            'protocol_sha256': file_hash(cls.benchmark/'protocol.json'),
            'attempted': 1, 'completed': 1, 'failed': 0,
            'cases': [{'id': 'fixture', 'status': 'completed', 'run_path': 'cases/fixture/run.json',
                'run_sha256': file_hash(cls.benchmark/'cases/fixture/run.json'),
                'scores_path': 'cases/fixture/scores.json',
                'scores_sha256': file_hash(cls.benchmark/'cases/fixture/scores.json'),
                'scores': scores}]})

    @classmethod
    def tearDownClass(cls):
        cls.base_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root/'run'
        shutil.copytree(self.base/'complete', self.run_dir)

    def read_run(self):
        return json.loads((self.run_dir/'run.json').read_bytes())

    def copy_benchmark(self):
        destination = self.root/'benchmark'
        shutil.copytree(self.benchmark, destination)
        return destination

    def read_report(self, directory):
        return json.loads((directory/'report.json').read_bytes())

    def update_benchmark_protocol(self, directory, mutate):
        protocol = json.loads((directory/'protocol.json').read_bytes())
        mutate(protocol)
        write_document(directory/'protocol.json', protocol)
        report = self.read_report(directory)
        report['protocol_sha256'] = file_hash(directory/'protocol.json')
        write_document(directory/'report.json', report)

    def test_actual_generated_run_and_bundle_load_preserve_scopes(self):
        run = load_motion_run(self.run_dir/'run.json')
        self.assertEqual(run, self.receipt)
        self.assertEqual(run['protocol']['analysis_image_size_pixels'], [160, 120])
        self.assertEqual(run['frame_count'], 12)
        self.assertEqual(run['accuracy_status'], 'not_assessed')
        self.assertFalse(run['robot_skill_qualified'])
        report, episodes = load_motion_bundle(self.run_dir)
        self.assertIsNone(report)
        self.assertEqual(set(episodes), {'episode'})
        self.assertEqual(episodes['episode']['run'], run)
        self.assertIsNone(episodes['episode']['scores'])

    def test_changed_sidecar_protocol_rows_and_frames_are_rejected(self):
        for name, field, value in [('protocol', 'sample_hz_requested', 20),
                                   ('rows', 'support_count', 999),
                                   ('frames', 'source_pts_seconds', .01)]:
            path = self.run_dir/(name+'.json')
            original = path.read_bytes()
            changed = json.loads(original)
            if name == 'protocol':
                changed[field] = value
            else:
                changed[0][field] = value
            write_document(path, changed)
            with self.subTest(sidecar=name), self.assertRaises(ValueError):
                load_motion_run(self.run_dir/'run.json')
            path.write_bytes(original)

    def test_changed_receipt_hash_is_rejected(self):
        run = self.read_run()
        run['rows_sha256'] = '0'*64
        write_document(self.run_dir/'run.json', run)
        with self.assertRaises(ValueError):
            load_motion_run(self.run_dir/'run.json')

    def test_forged_forecast_is_rejected_even_with_recomputed_hashes(self):
        run = self.read_run()
        run['rows'][1]['forecasts'][0]['center_xy_pixels'][0] += 10
        write_run_parts(self.run_dir, run)
        with self.assertRaisesRegex(ValueError, 'causal|schema|evidence'):
            load_motion_run(self.run_dir/'run.json')

    def test_source_timestamp_mismatch_rejected_even_when_frames_rehashed(self):
        run = self.read_run()
        run['frames'][1]['source_pts_seconds'] += .001
        write_run_parts(self.run_dir, run)
        with self.assertRaisesRegex(ValueError, 'presentation|timestamp|source'):
            load_motion_run(self.run_dir/'run.json')

    def test_seed_receipt_hash_and_protocol_binding_are_checked(self):
        run = self.read_run()
        run['seed']['label'] = 'Unrelated seed'
        write_document(self.run_dir/'run.json', run)
        with self.assertRaises(ValueError):
            load_motion_run(self.run_dir/'run.json')
        run['seed_sha256'] = digest(run['seed'])
        write_document(self.run_dir/'run.json', run)
        with self.assertRaises(ValueError):
            load_motion_run(self.run_dir/'run.json')

    def test_same_size_seed_evidence_change_is_detected(self):
        run = self.read_run()
        image = self.run_dir/run['seed_evidence']['path']
        original = image.read_bytes()
        image.write_bytes(bytes([original[0] ^ 1])+original[1:])
        self.assertEqual(image.stat().st_size, len(original))
        with self.assertRaises(ValueError):
            load_motion_run(self.run_dir/'run.json')

    def test_same_size_source_media_change_is_detected(self):
        run = self.read_run()
        video = self.run_dir/run['media']['path']
        original = video.read_bytes()
        video.write_bytes(bytes([original[0] ^ 1])+original[1:])
        self.assertEqual(video.stat().st_size, run['media']['bytes'])
        with self.assertRaisesRegex(ValueError, 'media|source'):
            load_motion_run(self.run_dir/'run.json')

    def test_display_geometry_descriptor_cannot_disagree_with_retained_media(self):
        for fields in (('source',), ('protocol',), ('source', 'protocol')):
            run = deepcopy(self.receipt)
            for field in fields:
                run[field]['display_geometry']['display_rotation_degrees'] = 180
            write_run_parts(self.run_dir, run)
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, 'display geometry'):
                load_motion_run(self.run_dir/'run.json')

    def test_older_identity_receipt_without_geometry_descriptors_remains_replayable(self):
        run = self.read_run()
        run['protocol'].pop('display_geometry', None)
        run['source'].pop('display_geometry', None)
        write_run_parts(self.run_dir, run)
        loaded = load_motion_run(self.run_dir/'run.json')
        self.assertEqual(loaded['rows'], self.receipt['rows'])
        self.assertNotIn('display_geometry', loaded['protocol'])
        self.assertNotIn('display_geometry', loaded['source'])
        report, episodes = load_motion_bundle(self.run_dir)
        self.assertIsNone(report)
        self.assertEqual(episodes['episode']['run']['media']['sha256'], self.receipt['media']['sha256'])

    def rotated_video(self, source, angle):
        """Require real nonidentity metadata; support pre/post-FFmpeg 9 recipes."""
        target = self.root/f'rotated-{angle}.mp4'
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
            metadata = json.loads(proof.stdout)
            streams = metadata.get('streams', [])
            if len(streams) != 1:
                continue
            stream = streams[0]
            angles = [entry.get('rotation', 0) for entry in stream.get('side_data_list', [])]
            if 'rotate' in stream.get('tags', {}):
                angles.append(stream['tags']['rotate'])
            if any(float(value) % 360 != 0 for value in angles):
                return target
        self.skipTest('Installed FFmpeg cannot create inspectable stream-copy rotation metadata.')

    def test_older_rotated_receipt_cannot_bypass_startup_geometry_check_by_rebinding_hashes(self):
        run = self.read_run()
        media_path = self.run_dir/run['media']['path']
        rotated = self.rotated_video(media_path, 90)
        shutil.copyfile(rotated, media_path)
        changed_hash, changed_size = file_hash(media_path), media_path.stat().st_size
        run['media'].update(sha256=changed_hash, bytes=changed_size)
        run['source'].update(sha256=changed_hash, bytes=changed_size)
        run['protocol']['source_sha256'] = changed_hash
        run['protocol'].pop('display_geometry', None)
        run['source'].pop('display_geometry', None)
        write_run_parts(self.run_dir, run)
        # Every retained hash is coherent. The media's actual display transform,
        # rather than a stale/missing descriptor, must reject replay startup.
        self.assertEqual(run['media']['sha256'], file_hash(media_path))
        self.assertEqual(run['protocol_sha256'], digest(run['protocol']))
        with self.assertRaisesRegex(ValueError, 'Rotated video|display geometry'):
            load_motion_run(self.run_dir/'run.json')
        with self.assertRaisesRegex(ValueError, 'Rotated video|display geometry'):
            MotionServer(('127.0.0.1', 0), self.run_dir)

    def test_missing_evidence_file_is_rejected(self):
        (self.run_dir/'frames.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            load_motion_run(self.run_dir/'run.json')

    def test_failed_preprobe_receipt_with_no_dimensions_or_rows_is_replayable(self):
        failed_dir = self.root/'failed'
        shutil.copytree(self.base/'failed', failed_dir)
        failed = load_motion_run(failed_dir/'run.json')
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['rows'], [])
        self.assertEqual(failed['frames'], [])
        self.assertIsNone(failed['media'])
        self.assertNotIn('analysis_image_size_pixels', failed['protocol'])
        report, episodes = load_motion_bundle(failed_dir)
        self.assertIsNone(report)
        self.assertEqual(episodes['episode']['run']['errors'][0]['stage'], 'intake')

    def test_completed_receipt_cannot_silently_drop_decoded_evidence(self):
        run = self.read_run()
        run['rows'], run['frames'] = [], []
        write_run_parts(self.run_dir, run)
        with self.assertRaises(ValueError):
            load_motion_run(self.run_dir/'run.json')

    def test_benchmark_bundle_recomputes_real_run_scores_from_retained_references(self):
        directory = self.copy_benchmark()
        report, episodes = load_motion_bundle(directory)
        self.assertEqual((report['attempted'], report['completed'], report['failed']), (1, 1, 0))
        self.assertEqual(set(episodes), {'fixture'})
        self.assertEqual(episodes['fixture']['scores'], report['cases'][0]['scores'])
        self.assertEqual(episodes['fixture']['run']['source']['sha256'], file_hash(self.video))

    def test_benchmark_score_tamper_rejected_even_with_matching_score_hash_and_report(self):
        directory = self.copy_benchmark()
        report = self.read_report(directory)
        report['cases'][0]['scores']['forecasts']['0.1']['constant_velocity']['mean'] = 999
        path = directory/report['cases'][0]['scores_path']
        write_document(path, report['cases'][0]['scores'])
        report['cases'][0]['scores_sha256'] = file_hash(path)
        write_document(directory/'report.json', report)
        with self.assertRaisesRegex(ValueError, 'scores|evaluator'):
            load_motion_bundle(directory)

    def test_changed_benchmark_reference_rejected_before_replay(self):
        directory = self.copy_benchmark()
        path = directory/'inputs/fixture.reference.json'
        reference = json.loads(path.read_bytes())
        reference['rows'][5]['anchor_xy_pixels'][0] += 20
        write_document(path, reference)
        with self.assertRaisesRegex(ValueError, 'hash|evidence'):
            load_motion_bundle(directory)
        # Updating the reference's hash alone cannot retain stale evaluator scores.
        self.update_benchmark_protocol(directory, lambda p: p['fixtures'][0].update(
            reference_sha256=file_hash(path)))
        with self.assertRaisesRegex(ValueError, 'scores|evaluator'):
            load_motion_bundle(directory)

    def test_benchmark_case_run_hash_and_completion_counts_are_checked(self):
        directory = self.copy_benchmark()
        original = (directory/'report.json').read_bytes()
        report = self.read_report(directory)
        report['cases'][0]['run_sha256'] = '0'*64
        write_document(directory/'report.json', report)
        with self.assertRaises(ValueError):
            load_motion_bundle(directory)
        (directory/'report.json').write_bytes(original)
        report = self.read_report(directory)
        report['attempted'] = 2
        write_document(directory/'report.json', report)
        with self.assertRaisesRegex(ValueError, 'counts|attempts'):
            load_motion_bundle(directory)

    def test_benchmark_outer_case_identity_matches_its_actual_run(self):
        directory = self.copy_benchmark()
        self.update_benchmark_protocol(directory, lambda p: p['fixtures'][0].update(id='other'))
        report = self.read_report(directory)
        report['cases'][0]['id'] = 'other'
        write_document(directory/'report.json', report)
        with self.assertRaises(ValueError):
            load_motion_bundle(directory)

    def test_benchmark_outer_fixture_source_matches_the_source_actually_analyzed(self):
        directory = self.copy_benchmark()
        input_video = directory/'inputs'/self.video.name
        original = input_video.read_bytes()
        input_video.write_bytes(bytes([original[0] ^ 1])+original[1:])
        self.update_benchmark_protocol(directory, lambda p: p['fixtures'][0].update(
            video_sha256=file_hash(input_video)))
        with self.assertRaises(ValueError):
            load_motion_bundle(directory)

    def start_server(self, directory=None):
        server = MotionServer(('127.0.0.1', 0), directory or self.run_dir)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        def stop():
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)
        self.addCleanup(stop)
        return server

    def request(self, server, method, path, *, headers=None, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_loopback_get_head_and_exact_media_ranges(self):
        server = self.start_server()
        status, headers, body = self.request(server, 'GET', '/')
        self.assertEqual(status, 200)
        self.assertIn('text/html', headers['Content-Type'])
        self.assertTrue(body)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        status, _, body = self.request(server, 'GET', '/api/episodes')
        self.assertEqual(status, 200)
        listing = json.loads(body)
        self.assertEqual(listing['episodes'][0]['id'], 'episode')
        status, _, body = self.request(server, 'GET', '/api/episodes/episode')
        self.assertEqual(status, 200)
        episode = json.loads(body)
        self.assertEqual(episode['run']['rows'], self.receipt['rows'])
        self.assertEqual(episode['video_url'], '/api/episodes/episode/video')
        original = (self.run_dir/self.receipt['media']['path']).read_bytes()
        status, headers, body = self.request(server, 'GET', episode['video_url'])
        self.assertEqual(status, 200)
        self.assertEqual(body, original)
        self.assertEqual(headers['Accept-Ranges'], 'bytes')
        status, headers, body = self.request(server, 'GET', episode['video_url'],
                                            headers={'Range': 'bytes=4-19'})
        self.assertEqual(status, 206)
        self.assertEqual(body, original[4:20])
        self.assertEqual(headers['Content-Range'], f'bytes 4-19/{len(original)}')
        self.assertEqual(self.request(server, 'GET', episode['video_url'],
                                     headers={'Range': 'bytes=-8'})[2], original[-8:])
        status, headers, body = self.request(server, 'HEAD', episode['video_url'])
        self.assertEqual(status, 200)
        self.assertEqual(body, b'')
        self.assertEqual(int(headers['Content-Length']), len(original))
        for requested in ('bytes=999999-', 'bytes=0-2,5-9', 'bytes=-0', 'bytes=20-4'):
            with self.subTest(requested=requested):
                self.assertEqual(self.request(server, 'GET', episode['video_url'],
                                             headers={'Range': requested})[0], 416)
        self.assertEqual(self.request(server, 'GET', '/api/episodes/missing')[0], 404)
        self.assertEqual(self.request(server, 'GET', '/unknown')[0], 404)

    def test_foreign_host_origin_and_cross_site_are_rejected(self):
        server = self.start_server()
        for headers in ({'Host': 'evil.invalid'}, {'Origin': 'https://evil.invalid'},
                        {'Sec-Fetch-Site': 'cross-site'}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request(server, 'GET', '/api/episodes', headers=headers)[0], 403)
                self.assertEqual(self.request(server, 'GET', '/api/episodes/episode/video', headers=headers)[0], 403)
        expected = f'http://127.0.0.1:{server.server_port}'
        self.assertEqual(self.request(server, 'GET', '/api/episodes', headers={'Origin': expected})[0], 200)
        with self.assertRaises(ValueError):
            MotionServer(('0.0.0.0', 0), self.run_dir)

    def test_post_is_read_only_and_does_not_change_bundle(self):
        server = self.start_server()
        original = {file.name: file.read_bytes() for file in self.run_dir.iterdir()}
        status, headers, body = self.request(server, 'POST', '/api/episodes/episode',
            headers={'Content-Type': 'application/json'}, body=b'{"rows":[]}')
        self.assertEqual(status, 405)
        self.assertEqual(headers['Allow'], 'GET, HEAD')
        self.assertIn('read-only', json.loads(body)['error'])
        self.assertEqual({file.name: file.read_bytes() for file in self.run_dir.iterdir()}, original)

    def test_server_detects_same_size_media_change_after_first_read(self):
        server = self.start_server()
        path = '/api/episodes/episode/video'
        self.assertEqual(self.request(server, 'GET', path)[0], 200)
        video = self.run_dir/self.receipt['media']['path']
        original = video.read_bytes()
        video.write_bytes(bytes([original[0] ^ 1])+original[1:])
        self.assertEqual(video.stat().st_size, len(original))
        status, _, body = self.request(server, 'GET', path)
        self.assertEqual(status, 409)
        self.assertIn('inconsistent', json.loads(body)['error'])

    def test_failed_intake_is_visible_without_media(self):
        failed_dir = self.root/'failed'
        shutil.copytree(self.base/'failed', failed_dir)
        server = self.start_server(failed_dir)
        status, _, body = self.request(server, 'GET', '/api/episodes')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['episodes'][0]['status'], 'failed')
        status, _, body = self.request(server, 'GET', '/api/episodes/episode')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['run']['errors'][0]['stage'], 'intake')
        self.assertEqual(self.request(server, 'GET', '/api/episodes/episode/video')[0], 404)


if __name__ == '__main__':
    unittest.main()
