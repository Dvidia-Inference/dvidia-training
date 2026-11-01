from copy import deepcopy
import json
from pathlib import Path
import shutil
import stat
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import episode_archive_cache as cache
import prepare_camera_action_v01 as evidence


class EpisodeArchiveCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.study = self.root / 'study'; self.study.mkdir()
        self.cache = self.root / 'cache'
        self.prefix = 'episodes/diagnostic/case-teacher'
        self.episode = self.study / self.prefix
        self.make_episode(self.episode)
        self.inventory = evidence.episode_inventory(self.episode)
        self.row = dict(id='case-teacher', case={'id': 'case', 'split': 'development'},
            controller='teacher', negative_control=None, path=self.prefix, status='complete',
            complete=True, recorded_complete_intervals=1000, strict_success=False,
            receipt_sha256=evidence.identity(self.episode / 'rollout.json')['sha256'],
            integrity_sha256=evidence.identity(self.episode / 'integrity.json')['sha256'])
        self.protocol = dict(rosters={name: [] for name in cache.STAGES},
            storage={'maximum_study_bytes': evidence.MAX_TOTAL})
        self.protocol['rosters']['diagnostic'] = [{k: self.row[k] for k in ('id', 'case', 'controller', 'negative_control')}]
        evidence.write(self.study / 'protocol.json', self.protocol)
        self.protocol_hash = evidence.identity(self.study / 'protocol.json')['sha256']
        evidence.write(self.study / 'registration.json', {'protocol_sha256': self.protocol_hash})
        self.seal_stage()

    def make_episode(self, root, status='complete'):
        evidence.write(root / 'started.json', {'synthetic_test_fixture': True})
        evidence.write(root / 'rollout.json', dict(status=status,
            recorded_complete_intervals=1000 if status == 'complete' else 0))
        evidence.write(root / 'integrity.json', dict(schema='dvidia.camera-action-rollout/v2/integrity',
            files={p.name: evidence.identity(p) for p in root.iterdir()}))

    def seal_stage(self):
        path = self.study / 'stages/diagnostic.json'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(evidence.canonical(dict(schema='dvidia.camera-action-study/v1/stage',
            stage='diagnostic', status='complete', scheduled=1,
            protocol_sha256=self.protocol_hash, rows=[self.row])))
        path.with_name('diagnostic.integrity.json').write_bytes(evidence.canonical(dict(
            sha256=evidence.identity(path)['sha256'], protocol_sha256=self.protocol_hash)))

    def seed_cache(self):
        directory = self.cache / cache.key(self.inventory, self.prefix)
        directory.mkdir(parents=True)
        evidence.archive(self.episode, self.inventory, directory / 'episode.zip', prefix=self.prefix)
        return directory

    @staticmethod
    def private_copy(source, destination, record):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as target:
            target.write(source.read_bytes())

    def test_hit_checks_source_and_every_member_without_trusting_receipt(self):
        directory = self.seed_cache()
        (directory / 'receipt.json').write_text('invalid, explicitly not trusted')
        original = cache.validate_archive(directory / 'episode.zip', self.inventory, self.prefix)
        destination = self.root / 'output/episode.zip'
        with patch.object(cache, '_private_copy', side_effect=self.private_copy):
            result = cache.reuse(self.episode, self.inventory, destination, prefix=self.prefix, cache=self.cache)
        self.assertEqual(result, original)
        self.assertEqual(destination.read_bytes(), (directory / 'episode.zip').read_bytes())
        self.assertNotEqual(destination.stat().st_ino, (directory / 'episode.zip').stat().st_ino)
        (directory / 'episode.zip').write_bytes(b'changed source')
        self.assertEqual(evidence.identity(destination)['sha256'], original['sha256'])

    def test_miss_does_not_create_destination_or_copy(self):
        with patch.object(cache, '_private_copy') as copy:
            self.assertIsNone(cache.reuse(self.episode, self.inventory, self.root / 'missing.zip',
                prefix=self.prefix, cache=self.cache))
            copy.assert_not_called()
        self.assertFalse((self.root / 'missing.zip').exists())

    def test_original_mutation_refuses_even_when_cached_members_are_valid(self):
        self.seed_cache()
        (self.episode / 'started.json').write_text('{}')
        with patch.object(cache, '_private_copy') as copy, self.assertRaisesRegex(ValueError, 'Original episode'):
            cache.reuse(self.episode, self.inventory, self.root / 'wrong.zip', prefix=self.prefix, cache=self.cache)
        copy.assert_not_called()

    def test_zip_member_tampering_roster_types_and_path_escape_refuse(self):
        source = self.seed_cache() / 'episode.zip'
        with zipfile.ZipFile(source) as archive:
            members = [(item, archive.read(item)) for item in archive.infolist()]
        for kind in ('bytes', 'extra', 'duplicate', 'symlink', 'path', 'timestamp', 'stored'):
            with self.subTest(kind=kind):
                target = self.root / (kind + '.zip')
                with zipfile.ZipFile(target, 'x') as out:
                    for index, (item, raw) in enumerate(members):
                        item = deepcopy(item)
                        if index == 0:
                            if kind == 'bytes': raw = b'X' * len(raw)
                            if kind == 'symlink': item.external_attr = (stat.S_IFLNK | 0o777) << 16
                            if kind == 'path': item.filename = '../escape'
                            if kind == 'timestamp': item.date_time = (2026, 1, 1, 0, 0, 0)
                            if kind == 'stored': item.compress_type = zipfile.ZIP_STORED
                        out.writestr(item, raw)
                    if kind == 'extra': out.writestr('extra', b'no')
                    if kind == 'duplicate':
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', UserWarning)
                            out.writestr(members[0][0], members[0][1])
                with self.assertRaises(ValueError):
                    cache.validate_archive(target, self.inventory, self.prefix)

    def test_valid_zip_comments_extra_fields_order_prefix_and_trailing_bytes_refuse(self):
        source = self.seed_cache() / 'episode.zip'
        original = source.read_bytes()
        with zipfile.ZipFile(source) as archive:
            members = [(item, archive.read(item)) for item in archive.infolist()]
        for kind in ('archive_comment', 'member_comment', 'extra', 'reverse', 'prefix', 'trailing', 'local_extra'):
            with self.subTest(kind=kind):
                target = self.root / (kind + '.zip')
                if kind in ('prefix', 'trailing', 'local_extra'):
                    raw = bytearray(original)
                    if kind == 'prefix': raw = b'unreviewed prefix' + raw
                    if kind == 'trailing': raw += b'unreviewed trailing payload'
                    if kind == 'local_extra':
                        offset = 30 + len(members[0][0].filename.encode())
                        raw[offset:offset+2] = struct.pack('<H', 0x9999)
                    target.write_bytes(raw)
                else:
                    ordered = list(reversed(members)) if kind == 'reverse' else members
                    with zipfile.ZipFile(target, 'x') as out:
                        if kind == 'archive_comment': out.comment = b'unreviewed archive comment'
                        for index, (item, raw) in enumerate(ordered):
                            item = deepcopy(item)
                            item._compresslevel = 1
                            if index == 0 and kind == 'member_comment': item.comment = b'unreviewed member comment'
                            if index == 0 and kind == 'extra': item.extra = struct.pack('<HH', 0x9999, 4) + b'data'
                            with out.open(item, 'w', force_zip64=True) as stream: stream.write(raw)
                with zipfile.ZipFile(target) as recognized:
                    self.assertEqual(len(recognized.infolist()), len(members))  # structurally valid ZIP
                with self.assertRaises(ValueError):
                    cache.validate_archive(target, self.inventory, self.prefix)

    def test_trailing_data_inside_valid_deflate_member_refuses(self):
        source = self.seed_cache() / 'episode.zip'
        raw = bytearray(source.read_bytes())
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist(); central = archive.start_dir
        first = members[0]
        extra_offset = 30 + len(first.filename.encode())
        compressed_end = extra_offset + 20 + first.compress_size
        raw[compressed_end:compressed_end] = b'junk'
        struct.pack_into('<Q', raw, extra_offset + 12, first.compress_size + 4)
        cursor = central + 4
        for index, item in enumerate(members):
            if index == 0: struct.pack_into('<L', raw, cursor + 20, item.compress_size + 4)
            else: struct.pack_into('<L', raw, cursor + 42, item.header_offset + 4)
            cursor += 46 + len(item.filename.encode())
        struct.pack_into('<L', raw, len(raw) - 6, central + 4)
        target = self.root / 'compressed-trailing.zip'; target.write_bytes(raw)
        with zipfile.ZipFile(target) as archive:
            self.assertIsNone(archive.testzip())  # stdlib accepts unused compressed suffix
        with self.assertRaisesRegex(ValueError, 'trailing compressed data'):
            cache.validate_archive(target, self.inventory, self.prefix)

    def test_bounded_multichunk_decompression_and_unverifiable_compression_level(self):
        directory = self.root / 'large'; directory.mkdir()
        (directory / 'a').write_bytes(b'\x00' * (3 * 1024 * 1024 + 17))
        (directory / 'b').write_bytes(bytes(range(256)) * 300)
        inventory = {p.name: evidence.identity(p) for p in directory.iterdir()}
        archive = self.root / 'level-six.zip'
        evidence.archive(directory, inventory, archive, prefix=self.prefix, compresslevel=6)
        result = cache.validate_archive(archive, inventory, self.prefix)
        self.assertIsNone(result['compression_level'])
        self.assertEqual(result['compression_method'], 'DEFLATE')
        self.assertEqual(result['unpacked_bytes'], sum(r['bytes'] for r in inventory.values()))

    def test_collision_symlink_overlap_and_incomplete_cache_fail_closed(self):
        directory = self.seed_cache()
        collision = self.root / 'existing.zip'; collision.write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            cache.reuse(self.episode, self.inventory, collision, prefix=self.prefix, cache=self.cache)
        self.assertEqual(collision.read_text(), 'preserve')
        (directory / 'episode.zip').unlink()
        with self.assertRaises(OSError):
            cache.reuse(self.episode, self.inventory, self.root / 'new.zip', prefix=self.prefix, cache=self.cache)
        (directory / 'episode.zip').symlink_to(self.episode / 'rollout.json')
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            cache.reuse(self.episode, self.inventory, self.root / 'new.zip', prefix=self.prefix, cache=self.cache)
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            cache.reuse(self.episode, self.inventory, self.cache / 'new.zip', prefix=self.prefix, cache=self.cache)

    def test_only_sealed_completed_roster_is_eligible_including_task_failures(self):
        (self.study / 'worker.lock').write_text('live native worker is allowed')
        result, remaining = cache.candidates(self.study)
        self.assertEqual([row['prefix'] for row in result], [self.prefix])
        self.assertFalse(self.row['strict_success'])  # no success-based cherry-picking
        self.assertEqual(remaining, evidence.MAX_TOTAL - sum(r['bytes'] for r in self.inventory.values()))
        (self.study / 'stages/diagnostic.integrity.json').unlink()
        with patch.object(evidence, 'episode_inventory', side_effect=AssertionError('read active episode')):
            self.assertEqual(cache.candidates(self.study), ([], evidence.MAX_TOTAL))

    def test_refused_episode_is_left_for_final_fresh_archiving(self):
        shutil.rmtree(self.episode)
        self.make_episode(self.episode, status='refused')
        self.row.update(status='refused', complete=False, recorded_complete_intervals=0,
            receipt_sha256=evidence.identity(self.episode / 'rollout.json')['sha256'],
            integrity_sha256=evidence.identity(self.episode / 'integrity.json')['sha256'])
        self.seal_stage()
        eligible, remaining = cache.candidates(self.study)
        self.assertEqual(eligible, [])
        self.assertLess(remaining, evidence.MAX_TOTAL)

    def test_registered_roster_stage_and_episode_bindings_are_checked(self):
        original = deepcopy(self.row)
        for field, value in (('case', {'id': 'substituted'}), ('path', '../escape'),
                ('receipt_sha256', '0' * 64)):
            with self.subTest(field=field):
                self.row = {**original, field: value}; self.seal_stage()
                with self.assertRaises(ValueError): cache.candidates(self.study)
        self.row = original; self.seal_stage()
        (self.study / 'stages/diagnostic.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'identity changed'): cache.candidates(self.study)

    def test_precompress_is_bounded_and_reserves_remaining_raw_capacity(self):
        raw = sum(r['bytes'] for r in self.inventory.values())
        remaining = evidence.MAX_TOTAL - raw
        minimum_free = evidence.DISK_RESERVE + remaining + cache.archive_headroom(self.inventory, self.prefix)
        with patch.object(cache.shutil, 'disk_usage', return_value=SimpleNamespace(free=minimum_free - 1)), \
                patch.object(evidence, 'archive') as archive:
            with self.assertRaisesRegex(ValueError, 'future registered recording'):
                cache.precompress(self.study, self.cache)
            archive.assert_not_called()
        self.assertFalse((self.cache / 'worker.lock').exists())
        with patch.object(cache.shutil, 'disk_usage', return_value=SimpleNamespace(free=minimum_free)):
            result = cache.precompress(self.study, self.cache, limit=1)
            repeated = cache.precompress(self.study, self.cache, limit=1)
        self.assertEqual(len(result['created']), 1)
        self.assertEqual(result['remaining_registered_raw_bytes_reserved'], remaining)
        self.assertFalse(result['physics_executed']); self.assertFalse(result['qualification_changed'])
        self.assertEqual(repeated['created'], [])
        self.assertEqual(repeated['existing_entries_skipped'], 1)
        for limit in (0, 453, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'finite episode limit'):
                cache.precompress(self.study, self.cache, limit=limit)

    def test_exclusive_worker_and_distinct_tree(self):
        self.cache.mkdir(); (self.cache / 'worker.lock').write_text('other worker')
        with self.assertRaises(FileExistsError): cache.precompress(self.study, self.cache)
        self.assertEqual((self.cache / 'worker.lock').read_text(), 'other worker')
        for directory in (self.study, self.study / 'cache', self.root):
            with self.assertRaisesRegex(ValueError, 'separate cache'): cache.precompress(self.study, directory)

    def test_private_copy_uses_isolated_reviewed_freeze_only(self):
        record = {'bytes': 42, 'sha256': 'a' * 64}
        with patch.object(cache.subprocess, 'run', return_value=SimpleNamespace(returncode=0,
                stdout=json.dumps({'method': 'apfs_clone'}))) as run:
            cache._private_copy(self.root / 'source.zip', self.root / 'target.zip', record)
        args, kwargs = run.call_args
        command = args[0]
        self.assertEqual(command[1:3], ['-I', '-c'])
        self.assertIn('from publish_camera_action_v01 import freeze_file', command[3])
        self.assertNotIn('client(', command[3]); self.assertNotIn('publish(', command[3])
        self.assertEqual(command[-2:], ['42', 'a' * 64]); self.assertEqual(kwargs['timeout'], 300)
        with patch.object(cache.subprocess, 'run', return_value=SimpleNamespace(returncode=1,
                stdout='', stderr='private failure details')):
            with self.assertRaisesRegex(ValueError, 'source evidence remains unchanged'):
                cache._private_copy(self.root / 'source.zip', self.root / 'target.zip', record)


if __name__ == '__main__':
    unittest.main()
