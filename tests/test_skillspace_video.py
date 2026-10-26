"""Offline package and saved-source integrity checks for Skillspace videos."""
from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stderr
from hashlib import sha256
import io
import json
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch
import uuid
import zipfile

from dvidia_training.footage_data import FootageDataError
from dvidia_training.skillspace_video import (SOURCE_PACKAGE_LIMIT,
    import_skillspace_video, inspect_skillspace_video, main)


class SkillspaceVideoImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video = Path(__file__).parents[1] / 'src/dvidia_training/video_goal_example.mp4'
        cls.video_bytes = cls.video.read_bytes()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Resolve macOS /var and /tmp aliases so these test paths themselves do
        # not cross a symlink ancestor that the importer correctly rejects.
        self.root = Path(self.temp.name).resolve()
        self.identity = str(uuid.uuid4())
        self.output = self.root / 'imported'
        self.manifest = {
            'schema': 'dvidia.skillspace-video.v1',
            'platform_origin': 'https://dvidia.org',
            'exported_at': '2026-10-25T12:34:56.000Z',
            'contribution': {
                'id': self.identity,
                'revision': '2026-10-24T08:10:11+00:00',
                'skillspace_id': 'personal-' + self.identity,
                'task_title': 'Synthetic example video',
            },
            'video': {
                'filename': 'video.mp4', 'original_name': 'placement.mp4',
                'content_type': 'video/mp4', 'bytes': len(self.video_bytes),
                'sha256': sha256(self.video_bytes).hexdigest(),
            },
            'processing': {'scope': 'local-only', 'accepted': True},
            'calibration': None,
        }

    def package(self, *, manifest=None, video=None, entries=None, compression=zipfile.ZIP_STORED,
                package_name=None):
        package = self.root / (package_name or f'dvidia-video-{self.identity}.zip')
        rows = entries
        if rows is None:
            raw = json.dumps(self.manifest if manifest is None else manifest,
                             separators=(',', ':')).encode()
            rows = [('manifest.json', raw),
                    ((self.manifest if manifest is None else manifest)['video']['filename'],
                     self.video_bytes if video is None else video)]
        with zipfile.ZipFile(package, 'w', compression=compression) as archive:
            for name, content in rows:
                archive.writestr(name, content, compress_type=compression)
        return package

    def test_import_and_reinspect_real_packaged_example_video(self):
        package = self.package()
        returned = import_skillspace_video(package, self.output)
        self.assertEqual(returned, self.manifest)
        self.assertEqual((self.output / 'video.mp4').read_bytes(), self.video_bytes)
        self.assertEqual(inspect_skillspace_video(self.output), self.manifest)

    def test_colon_in_original_filename_metadata_round_trips_safely(self):
        manifest = deepcopy(self.manifest)
        manifest['video']['original_name'] = 'take:001.mp4'
        package = self.package(manifest=manifest)
        self.assertEqual(import_skillspace_video(package, self.output), manifest)
        self.assertEqual(inspect_skillspace_video(self.output), manifest)
        self.assertEqual((self.output / 'video.mp4').read_bytes(), self.video_bytes)

    def test_payload_integrity_and_declared_size_are_required(self):
        wrong_digest = deepcopy(self.manifest)
        wrong_digest['video']['sha256'] = '0' * 64
        wrong_length = deepcopy(self.manifest)
        wrong_length['video']['bytes'] += 1
        for manifest in (wrong_digest, wrong_length):
            with self.subTest(manifest=manifest['video']), self.assertRaises(FootageDataError):
                import_skillspace_video(self.package(manifest=manifest), self.output)
            self.assertFalse(self.output.exists())
        valid = self.package()
        imported = self.root / 'saved'
        import_skillspace_video(valid, imported)
        (imported / 'video.mp4').write_bytes(b'changed')
        with self.assertRaises(FootageDataError):
            inspect_skillspace_video(imported)

    def test_manifest_schema_types_exact_keys_controls_and_duplicate_json_reject(self):
        variants = []
        for mutate in (
                lambda row: row.update(schema='dvidia.skillspace-video.v2'),
                lambda row: row.update(extra='not allowed'),
                lambda row: row.update(platform_origin='https://user:pass@dvidia.org'),
                lambda row: row['contribution'].update(task_title='bad\x7ftext'),
                lambda row: row['processing'].update(accepted=1),
                lambda row: row.update(calibration={}),
                lambda row: row['video'].update(original_name='../escape.mp4')):
            variant = deepcopy(self.manifest)
            mutate(variant)
            variants.append(variant)
        for manifest in variants:
            with self.subTest(manifest=manifest), self.assertRaises(FootageDataError):
                import_skillspace_video(self.package(manifest=manifest), self.output)
            self.assertFalse(self.output.exists())
        raw = b'{"schema":"dvidia.skillspace-video.v1","schema":"dvidia.skillspace-video.v1"}'
        package = self.package(entries=[('manifest.json', raw), ('video.mp4', self.video_bytes)])
        with self.assertRaises(FootageDataError):
            import_skillspace_video(package, self.output)

    def test_zip_paths_duplicates_symlinks_compression_and_extra_entries_reject(self):
        attacks = [
            [('manifest.json', json.dumps(self.manifest).encode()), ('../video.mp4', self.video_bytes)],
            [('manifest.json', json.dumps(self.manifest).encode()), ('video.mp4', self.video_bytes),
             ('video.mp4', self.video_bytes)],
            [('manifest.json', json.dumps(self.manifest).encode()), ('video.mp4', self.video_bytes),
             ('other.txt', b'extra')],
        ]
        for rows in attacks:
            with self.subTest(rows=[name for name, _ in rows]), self.assertRaises(FootageDataError):
                import_skillspace_video(self.package(entries=rows), self.output)
            self.assertFalse(self.output.exists())

        package = self.root / f'dvidia-video-{self.identity}.zip'
        with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('manifest.json', json.dumps(self.manifest).encode())
            info = zipfile.ZipInfo('video.mp4')
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b'video.mp4')
        with self.assertRaises(FootageDataError):
            import_skillspace_video(package, self.output)

        compressed = self.package(compression=zipfile.ZIP_DEFLATED)
        with self.assertRaises(FootageDataError):
            import_skillspace_video(compressed, self.output)

    def test_archive_size_bomb_partial_package_and_renamed_upload_are_handled(self):
        package = self.package()
        with patch('dvidia_training.skillspace_video.SOURCE_PACKAGE_LIMIT', 10):
            with self.assertRaises(FootageDataError):
                import_skillspace_video(package, self.output)
        malformed = self.root / f'dvidia-video-{self.identity}.zip'
        malformed.write_bytes(b'PK\x03\x04partial')
        with self.assertRaises(FootageDataError):
            import_skillspace_video(malformed, self.output)
        # Browsers may rename a valid export, and the outer filename is not an
        # authenticity boundary or a source of contribution identity.
        renamed_upload = self.package(package_name=f'dvidia-video-{uuid.uuid4()}.zip')
        manifest = import_skillspace_video(renamed_upload, self.output)
        self.assertEqual(manifest['contribution']['id'], self.identity)
        self.assertEqual(inspect_skillspace_video(self.output), self.manifest)
        self.assertFalse(list(self.root.glob('.skillspace-video-*')))
        self.assertLessEqual(SOURCE_PACKAGE_LIMIT, 100 * 1024 * 1024 + 64 * 1024)

    def test_central_directory_preflight_rejects_before_zipfile_opens(self):
        cases = ((2, 4097), (3, 0))
        for index, (entry_count, central_size) in enumerate(cases):
            package = self.root / f'preflight-{index}.zip'
            # EOCD with either a central directory above the 4 KiB cap or an
            # entry count beyond this format's exact two-file contract.
            package.write_bytes(struct.pack('<4s4H2LH', b'PK\x05\x06', 0, 0,
                                             entry_count, entry_count,
                                             central_size, 0, 0))
            with patch('dvidia_training.skillspace_video.zipfile.ZipFile',
                       side_effect=AssertionError('zipfile opened before EOCD preflight')) as opener:
                with self.subTest(entry_count=entry_count, central_size=central_size), \
                     self.assertRaises(FootageDataError):
                    import_skillspace_video(package, self.output)
                opener.assert_not_called()

    def test_cli_bad_local_path_exits_cleanly_without_traceback(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            main(['import-video', '--package', str(self.root / 'missing.zip'),
                  '--output', str(self.output)])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn('dvidia-skillspace: error:', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())

    def test_preexisting_output_and_symlinked_ancestors_are_preserved_and_rejected(self):
        package = self.package()
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaises(FootageDataError):
            import_skillspace_video(package, self.output)
        self.assertEqual(sentinel.read_text(), 'keep')

        target = self.root / 'real-parent' / 'out'
        target.parent.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(target.parent, target_is_directory=True)
        with self.assertRaises(FootageDataError):
            import_skillspace_video(package, alias / 'out')
        self.assertFalse(target.exists())

    def test_staging_permission_failure_leaves_no_target(self):
        package = self.package()
        with patch('dvidia_training.skillspace_video.tempfile.mkdtemp',
                   side_effect=PermissionError('denied')):
            with self.assertRaises(FootageDataError):
                import_skillspace_video(package, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
