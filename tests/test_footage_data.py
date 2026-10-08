"""Real synthetic media intake and data/path/grouping boundary checks."""
from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from dvidia_training.footage_data import (DVIDIA_HOSTS, FootageDataError, _fetch, _url,
                                prepare, probe_video)


class FootageDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = tempfile.TemporaryDirectory()
        cls.media = Path(cls.fixtures.name)
        for index, color in enumerate(('red', 'blue', 'green', 'yellow', 'purple', 'white')):
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            f'color=c={color}:s=64x48:r=5', '-t', '0.6', '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-y', str(cls.media/f'{index}.mp4')],
                           stdin=subprocess.DEVNULL, check=True, timeout=15)

    @classmethod
    def tearDownClass(cls):
        cls.fixtures.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'source'; self.source.mkdir()
        self.output = self.root/'prepared'

    def videos(self, count=3):
        for index in range(count):
            shutil.copyfile(self.media/f'{index}.mp4', self.source/f'{index}.mp4')

    def manifest(self, rows=None):
        if rows is None:
            rows = [{'id': f'episode-{index}', 'path': f'{index}.mp4',
                     'source_recording_id': f'original-{index}', 'complete': True}
                    for index in range(3)]
        value = {'schema_version': 1, 'kind': 'dvidia.skillspace-training',
                 'task': 'synthetic color-fixture placement', 'media': rows}
        (self.source/'skillspace.training.json').write_text(json.dumps(value))
        return value

    def test_actual_probe_and_exact_media_bytes_are_frozen(self):
        self.videos(); original = self.manifest()
        raw = (self.source/'skillspace.training.json').read_bytes()
        with patch('socket.create_connection', side_effect=AssertionError('Offline intake attempted network')):
            dataset = prepare(self.source, self.output)
        self.assertEqual(dataset['kind'], 'dvidia.footage-dataset')
        self.assertEqual(dataset['source_manifest_sha256'], sha256(raw).hexdigest())
        self.assertEqual((self.output/'source_manifest.json').read_bytes(), raw)
        self.assertEqual(dataset['actual_media_episodes'], 3)
        self.assertEqual(dataset['independent_groups'], 3)
        self.assertTrue(dataset['capability']['visual_ready'])
        self.assertFalse(dataset['capability']['movement_ready'])
        for episode in dataset['episodes']:
            media = self.output/episode['media_path']
            row = next(row for row in original['media'] if row['id'] == episode['id'])
            self.assertEqual(media.read_bytes(), (self.source/row['path']).read_bytes())
            self.assertEqual(episode['media_sha256'], sha256(media.read_bytes()).hexdigest())
            self.assertAlmostEqual(episode['duration_seconds'], .6)
            self.assertEqual((episode['width'], episode['height'], episode['fps']), (64, 48, 5.))
            self.assertNotIn('actions_path', episode)
            self.assertNotIn('actions_sha256', episode)
        self.assertEqual({row['split'] for row in dataset['episodes']}, {'train', 'dev', 'test'})

    def test_folder_discovery_marks_unknown_lineage_without_claiming_novelty(self):
        self.videos()
        dataset = prepare(self.source, self.output)
        self.assertEqual(dataset['task'], 'unspecified footage task')
        self.assertEqual(sum(row['kind'] == 'unknown_lineage' for row in dataset['lineage_warnings']), 3)
        self.assertEqual(sum(row['complete'] for row in dataset['episodes']), 0)
        self.assertNotIn(str(self.source), json.dumps(dataset))

    def test_transitive_lineage_and_exact_duplicate_media_cannot_cross_splits(self):
        self.videos(6); shutil.copyfile(self.source/'0.mp4', self.source/'duplicate.mp4')
        rows = [
            {'id': 'a', 'path': '0.mp4', 'source_recording_id': 'recording-a'},
            {'id': 'b', 'path': '1.mp4', 'source_recording_id': 'recording-a', 'session_id': 'session-b'},
            {'id': 'c', 'path': '2.mp4', 'session_id': 'session-b', 'shoe_pair_id': 'pair-c'},
            {'id': 'd', 'path': '3.mp4', 'shoe_pair_id': 'pair-c'},
            {'id': 'e', 'path': '4.mp4', 'source_recording_id': 'recording-e'},
            {'id': 'f', 'path': '5.mp4', 'source_recording_id': 'recording-f'},
            {'id': 'duplicate-a', 'path': 'duplicate.mp4'}]
        self.manifest(rows); dataset = prepare(self.source, self.output)
        joined = [row for row in dataset['episodes'] if row['id'] in ('a', 'b', 'c', 'd', 'duplicate-a')]
        self.assertEqual(len({row['group_id'] for row in joined}), 1)
        self.assertEqual(len({row['split'] for row in joined}), 1)
        self.assertEqual(dataset['independent_groups'], 3)
        self.assertEqual(dataset['duplicate_media_groups'], [['a', 'duplicate-a']])

    def test_too_few_independent_groups_fail_instead_of_splitting_frames(self):
        self.videos()
        self.manifest([{'id': str(i), 'path': f'{i}.mp4', 'session_id': 'one-session'} for i in range(3)])
        with self.assertRaisesRegex(FootageDataError, 'three independent'):
            prepare(self.source, self.output)
        self.assertFalse(self.output.exists())
        for index in (1, 2):
            shutil.copyfile(self.source/'0.mp4', self.source/f'{index}.mp4')
        self.manifest()
        with self.assertRaisesRegex(FootageDataError, 'three independent'):
            prepare(self.source, self.output)

    def test_splits_are_deterministic_for_manifest_reordering(self):
        self.videos(6)
        rows = [{'id': f'episode-{i}', 'path': f'{i}.mp4', 'source_recording_id': f'original-{i}'} for i in range(6)]
        self.manifest(rows); first = prepare(self.source, self.output, seed=17)
        self.manifest(list(reversed(rows))); second = prepare(self.source, self.root/'other', seed=17)
        self.assertEqual({row['id']: row['split'] for row in first['episodes']},
                         {row['id']: row['split'] for row in second['episodes']})

    def test_sidecars_are_hash_bound_finite_json_without_movement_readiness(self):
        self.videos()
        value = {'kind': 'dvidia.aligned-movement-demonstration', 'schema_version': 1,
                 'samples': [], 'provenance': {'source_kind': 'synthetic-test'}}
        raw = json.dumps(value).encode(); (self.source/'actions.json').write_bytes(raw)
        manifest = self.manifest(); manifest['media'][0]['actions_path'] = 'actions.json'
        self.manifest(manifest['media'])
        dataset = prepare(self.source, self.output)
        episode = dataset['episodes'][0]
        self.assertEqual((self.output/episode['actions_path']).read_bytes(), raw)
        self.assertEqual(episode['actions_sha256'], sha256(raw).hexdigest())
        self.assertEqual(dataset['capability']['action_sidecars_present'], 1)
        self.assertFalse(dataset['capability']['movement_ready'])

    def test_images_and_reference_metadata_do_not_become_demonstrations(self):
        (self.source/'shoe.jpg').write_bytes(b'illustrative placeholder')
        with self.assertRaisesRegex(FootageDataError, 'images only'):
            prepare(self.source, self.output)
        (self.source/'shoe.jpg').unlink()
        (self.source/'skill.json').write_text('{"skill":"tie_lace","demos":100}')
        with self.assertRaisesRegex(FootageDataError, 'counts and starter metadata'):
            prepare(self.source, self.output)

    def test_manifest_unknown_entry_fields_duplicate_ids_and_nonfinite_values_reject(self):
        self.videos()
        for change in (lambda rows: rows[0].update(code='import os'),
                       lambda rows: rows[1].update(id=rows[0]['id']),
                       lambda rows: rows[0].update(complete=1),
                       lambda rows: rows[0].update(session_id=True)):
            value = self.manifest(); change(value['media']); self.manifest(value['media'])
            with self.assertRaises(FootageDataError):
                prepare(self.source, self.output)
        for raw in (b'{"media":[],"media":[]}', b'{"value":NaN}', b'{"value":1e999}'):
            (self.source/'skillspace.training.json').write_bytes(raw)
            with self.assertRaises(FootageDataError):
                prepare(self.source, self.output)

    def test_local_traversal_absolute_paths_symlinks_and_missing_artifacts_reject(self):
        self.videos()
        outside = self.root/'outside.mp4'; shutil.copyfile(self.media/'0.mp4', outside)
        (self.source/'link.mp4').symlink_to(outside)
        for invalid in ('../outside.mp4', str(outside), 'link.mp4', 'missing.mp4', 'C:\\outside.mp4'):
            value = self.manifest(); value['media'][0]['path'] = invalid; self.manifest(value['media'])
            with self.subTest(path=invalid), self.assertRaises(FootageDataError):
                prepare(self.source, self.output)
        (self.source/'skillspace.training.json').unlink()
        (self.source/'skillspace.training.json').symlink_to(self.root/'nonexistent.json')
        with self.assertRaises(FootageDataError):
            prepare(self.source, self.output)

    def test_zip_preserves_media_and_rejects_traversal_symlinks_code_and_expansion(self):
        self.videos(); self.manifest()
        valid = self.root/'export.zip'
        with zipfile.ZipFile(valid, 'w') as archive:
            for file in self.source.iterdir():
                archive.write(file, 'export/'+file.name)
        self.assertEqual(prepare(valid, self.output)['actual_media_episodes'], 3)
        for index, entry in enumerate(('../bad.mp4', '/absolute.mp4', 'file.py', 'a\\bad.mp4')):
            path = self.root/f'bad-{index}.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr(entry, b'bad')
            with self.assertRaises(FootageDataError):
                prepare(path, self.root/f'badout-{index}')
        symlink = self.root/'symlink.zip'
        with zipfile.ZipFile(symlink, 'w') as archive:
            info = zipfile.ZipInfo('link.mp4'); info.create_system = 3; info.external_attr = 0o120777 << 16
            archive.writestr(info, '../other.mp4')
        with self.assertRaises(FootageDataError):
            prepare(symlink, self.root/'symlinkout')
        with patch('dvidia_training.footage_data.MAX_TOTAL_BYTES', 10):
            with self.assertRaises(FootageDataError):
                prepare(valid, self.root/'smallout')

    def test_invalid_media_probe_and_timeout_leave_no_dataset(self):
        self.videos(); self.manifest()
        (self.source/'0.mp4').write_bytes(b'not a video')
        with self.assertRaisesRegex(FootageDataError, 'decoded'):
            prepare(self.source, self.output)
        self.assertFalse(self.output.exists())
        with patch('dvidia_training.footage_data.subprocess.run', side_effect=subprocess.TimeoutExpired('ffprobe', 10)):
            with self.assertRaisesRegex(FootageDataError, '10 seconds'):
                probe_video(self.media/'0.mp4')

    def test_disguised_playlist_cannot_probe_unlisted_media(self):
        self.videos()
        disguised = self.source/'playlist.mp4'
        disguised.write_text("ffconcat version 1.0\nfile '0.mp4'\n")
        baseline = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file',
                                   '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                                   '-of', 'json', str(disguised)], capture_output=True, timeout=10)
        self.assertEqual(baseline.returncode, 0)
        self.assertEqual(json.loads(baseline.stdout)['streams'][0]['width'], 64)
        with self.assertRaisesRegex(FootageDataError, 'decoded'):
            probe_video(disguised)

    def test_probe_requires_measured_timing_rate_and_dimensions(self):
        stream = {'duration': '0.6', 'start_time': '0', 'avg_frame_rate': '5/1', 'width': 64, 'height': 48}
        variants = [({'streams': [stream], 'format': []}),
                    ({'streams': [{**stream, 'avg_frame_rate': '0/0'}]}),
                    ({'streams': [{key: value for key, value in stream.items() if key != 'duration'}]}),
                    ({'streams': [{**stream, 'width': True}]}),
                    ({'streams': [{**stream, 'height': 20000}]})]
        for variant in variants:
            result = subprocess.CompletedProcess('ffprobe', 0, stdout=json.dumps(variant).encode(), stderr=b'')
            with self.subTest(variant=variant), patch('dvidia_training.footage_data.subprocess.run', return_value=result):
                with self.assertRaises(FootageDataError):
                    probe_video(self.media/'0.mp4')

    def test_local_discovery_and_manifest_sizes_are_bounded(self):
        self.videos()
        with patch('dvidia_training.footage_data.MAX_FILES', 2):
            with self.assertRaisesRegex(FootageDataError, 'too many filesystem'):
                prepare(self.source, self.output)
        self.manifest()
        with patch('dvidia_training.footage_data.MAX_MANIFEST_BYTES', 10):
            with self.assertRaisesRegex(FootageDataError, 'size limit'):
                prepare(self.source, self.output)
        with patch('dvidia_training.footage_data.MAX_VIDEO_BYTES', 10):
            with self.assertRaisesRegex(FootageDataError, 'size limit'):
                prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_offline_mode_never_fetches_and_existing_output_is_preserved(self):
        with patch('dvidia_training.footage_data._fetch', side_effect=AssertionError('Network fetch attempted')):
            with self.assertRaisesRegex(FootageDataError, 'Offline'):
                prepare('https://dvidia.org/store/dvidia/place_cup', self.output)
        self.output.mkdir(); (self.output/'keep.txt').write_text('keep')
        with self.assertRaisesRegex(FootageDataError, 'never overwritten'):
            prepare(self.source, self.output)
        self.assertEqual((self.output/'keep.txt').read_text(), 'keep')

    def test_public_url_rejects_private_dns_credentials_and_token_queries(self):
        private = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))]
        with patch('dvidia_training.footage_data.socket.getaddrinfo', return_value=private):
            with self.assertRaisesRegex(FootageDataError, 'public IP'):
                _url('https://dvidia.org/file.json', DVIDIA_HOSTS)
        for url in ('http://dvidia.org/file.json', 'https://user:password@dvidia.org/file.json',
                    'https://dvidia.org/file.json?token=private', 'https://localhost/file.json',
                    'https://dvidia.org:444/file.json'):
            with self.assertRaises(FootageDataError):
                _url(url, DVIDIA_HOSTS)

    def test_online_requires_explicit_media_and_accepts_public_research_manifest(self):
        metadata = b'{"skill":"shoe-organizing","demos":100}'
        with patch('dvidia_training.footage_data._fetch', return_value=(metadata, 'https://dvidia.org/packs/team/shoes/skill.json')) as fetch:
            with self.assertRaisesRegex(FootageDataError, 'metadata only'):
                prepare('https://dvidia.org/store/team/shoes', self.output, offline=False)
            self.assertEqual(fetch.call_count, 1)
        manifest = self.manifest(); url = 'https://research.dvidia.org/demo/skillspace.training.json'
        received = []
        def fetch(url_value, maximum, hosts):
            received.append(url_value)
            self.assertIn('research.dvidia.org', hosts)
            if url_value == url:
                return json.dumps(manifest).encode(), url_value
            return (self.media/Path(url_value).name).read_bytes(), url_value
        with patch('dvidia_training.footage_data._fetch', side_effect=fetch):
            dataset = prepare(url, self.output, offline=False)
        self.assertEqual(dataset['source_url'], url)
        self.assertEqual(dataset['source_kind'], 'public_dvidia_manifest')
        self.assertEqual(received, [url, 'https://research.dvidia.org/demo/0.mp4',
                                   'https://research.dvidia.org/demo/1.mp4', 'https://research.dvidia.org/demo/2.mp4'])

    def test_https_connection_pins_validated_public_ip_and_retains_tls_hostname(self):
        class FakeSocket:
            def sendall(self, data):
                pass
            def makefile(self, mode):
                return io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}')
            def close(self):
                pass
        public = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))]
        context = MagicMock(); fake = FakeSocket(); context.wrap_socket.return_value = fake
        with patch('dvidia_training.footage_data.socket.getaddrinfo', return_value=public), \
             patch('dvidia_training.footage_data.socket.create_connection', return_value=fake) as connect, \
             patch('dvidia_training.footage_data.ssl.create_default_context', return_value=context):
            raw, _ = _fetch('https://dvidia.org/file.json', 10, DVIDIA_HOSTS)
        self.assertEqual(raw, b'{}')
        connect.assert_called_once_with(('8.8.8.8', 443), timeout=15)
        context.wrap_socket.assert_called_once_with(fake, server_hostname='dvidia.org')

    def test_redirects_revalidate_hosts_and_download_limits(self):
        class FakeSocket:
            def __init__(self, response):
                self.response = response
            def sendall(self, data):
                pass
            def makefile(self, mode):
                return io.BytesIO(self.response)
            def close(self):
                pass
        public = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))]
        responses = [b'HTTP/1.1 302 Found\r\nLocation: https://localhost/private.json\r\nContent-Length: 0\r\n\r\n',
                     b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{}']
        for response in responses:
            context = MagicMock(); fake = FakeSocket(response); context.wrap_socket.return_value = fake
            with patch('dvidia_training.footage_data.socket.getaddrinfo', return_value=public), \
                 patch('dvidia_training.footage_data.socket.create_connection', return_value=fake) as connect, \
                 patch('dvidia_training.footage_data.ssl.create_default_context', return_value=context):
                with self.assertRaises(FootageDataError):
                    _fetch('https://dvidia.org/file.json', 10, DVIDIA_HOSTS)
                self.assertEqual(connect.call_count, 1)


if __name__ == '__main__':
    unittest.main()
