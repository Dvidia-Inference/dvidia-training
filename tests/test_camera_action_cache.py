from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import camera_action_cache as cache
from dvidia_training import camera_action_export as export
from dvidia_training import camera_action_model as model
from dvidia_training import camera_action_study as study
from tests.test_camera_action_export import fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.canonical(value) + b'\n')
    return sha256(path.read_bytes()).hexdigest()


class CacheTests(unittest.TestCase):
    def setUp(self):
        # CI fixture tests must not require 30 GiB of actual host free space.
        # The explicit low-disk test applies its narrower nested patch below.
        ample = shutil._ntuple_diskusage(200 * 1024**3, 50 * 1024**3, 150 * 1024**3)
        disk_usage = patch.object(cache.shutil, 'disk_usage', return_value=ample)
        disk_usage.start()
        self.addCleanup(disk_usage.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.study = self.root/'study'; self.study.mkdir()
        self.out = self.root/'cache'
        self.recordings = {}
        self.rows, cases = [], {'train': [], 'development': [], 'confirmation': []}
        for group, split in [('train-a', 'train'), ('train-b', 'train'), ('dev-a', 'development')]:
            case = dict(id=group+'-goal0', group_id=group, split=split, goal_id=0,
                        goal_position_m=[.35, .08, .08], object_position_m=[.3, -.03, .082], seed=17)
            cases[split].append(case)
            attempt = case['id']+'-teacher'; relative = 'episodes/demonstrations/'+attempt
            episode = self.study/relative; episode.parent.mkdir(parents=True, exist_ok=True)
            recording = fixture(episode, group=group)
            self.recordings[str(episode)] = recording
            receipt = write(episode/'rollout.json', {'status': 'complete'})
            self.rows.append(dict(id=attempt, case=case, controller='teacher', negative_control=None,
                path=relative, complete=True, initial_anchor_available=True, initial_anchor_binding_valid=True,
                strict_success=True, recorded_complete_intervals=3, receipt_sha256=receipt,
                integrity_sha256=sha256((episode/'integrity.json').read_bytes()).hexdigest()))
        self.protocol = dict(schema=study.SCHEMA, source_sha256=study.runtime_hashes(), cases=cases,
                             rosters={'demonstrations': [{k: r[k] for k in ('id','case','controller','negative_control')} for r in self.rows]})
        self.save_protocol()
        self.save_stage()
        self.capture = patch('dvidia_training.camera_action_capture.load_rollout', side_effect=lambda p: self.recordings[str(p)])
        self.capture.start()

    def tearDown(self):
        self.capture.stop()
        for path in self.root.rglob('*'):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)
        self.temp.cleanup()

    def save_protocol(self):
        self.protocol_sha = write(self.study/'protocol.json', self.protocol)
        write(self.study/'registration.json', dict(schema=study.SCHEMA+'/registration',
              source_sha256=self.protocol['source_sha256'], protocol_sha256=self.protocol_sha))

    def save_stage(self):
        h = write(self.study/'stages/demonstrations.json', dict(schema=study.SCHEMA+'/stage', stage='demonstrations',
                  protocol_sha256=self.protocol_sha, status='complete', scheduled=len(self.rows), rows=self.rows))
        write(self.study/'stages/demonstrations.integrity.json', dict(sha256=h, protocol_sha256=self.protocol_sha))

    def build(self, **kw):
        self.pins = cache.build_cache(self.study, self.out, expected_protocol_sha256=self.protocol_sha, **kw)
        return self.pins

    def load(self, **kw):
        args = dict(expected_manifest_sha256=self.pins['manifest_sha256'], expected_protocol_sha256=self.protocol_sha,
                    expected_split=self.pins['split'], expected_source_sha256=self.pins['source_sha256'])
        args.update(kw)
        return cache.load_cache(self.out, **args)

    def amend(self, *, transform=None, manifest_change=None):
        self.out.chmod(0o755)
        manifest = json.loads((self.out/'manifest.json').read_text())
        if transform:
            p = self.out/'features.npy'; value = np.load(p, allow_pickle=False); value = transform(value)
            stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
            p.chmod(0o644); p.write_bytes(stream.getvalue())
            manifest['arrays']['features.npy']['sha256'] = sha256(p.read_bytes()).hexdigest()
            manifest['arrays']['features.npy']['bytes'] = p.stat().st_size
        if manifest_change:
            manifest_change(manifest)
        (self.out/'manifest.json').chmod(0o644)
        return write(self.out/'manifest.json', manifest)

    def test_atomic_write_removes_writable_temporary_before_sealing_target(self):
        target = self.root/'portable-write.bin'
        unlinked = []
        original_unlink = Path.unlink
        def writable_unlink(path, *args, **kwargs):
            if path.name.startswith('.pending-'):
                self.assertTrue(path.stat().st_mode & stat.S_IWUSR)
                unlinked.append(path.name)
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', writable_unlink):
            cache._write(target, b'complete cached artifact')
        self.assertEqual(len(unlinked), 1)
        self.assertEqual(target.read_bytes(), b'complete cached artifact')
        self.assertFalse(target.stat().st_mode & stat.S_IWUSR)
        self.assertEqual(list(self.root.glob('.pending-*')), [])
        # An existing target is never replaced, and its failed writer leaves
        # no temporary file behind on either POSIX or Windows semantics.
        with self.assertRaises(FileExistsError):
            cache._write(target, b'replacement must not happen')
        self.assertEqual(target.read_bytes(), b'complete cached artifact')
        self.assertEqual(list(self.root.glob('.pending-*')), [])

    def test_equality_to_strict_export_and_no_raw_reads_on_hit(self):
        strict = export.combine_datasets([export.export_episode(self.study/r['path']) for r in self.rows[:2]])
        self.build()
        with patch.object(export, 'export_episode', side_effect=AssertionError('must not export on hit')):
            loaded = self.load()
        np.testing.assert_array_equal(strict['features'], loaded['features'])
        np.testing.assert_array_equal(strict['targets'], loaded['targets'])
        self.assertEqual(strict['provenance'], loaded['provenance'])
        self.assertEqual(strict['features'].dtype, np.dtype('float64'))
        self.assertEqual(loaded['features'].dtype, np.dtype('float32'))
        self.assertEqual(model._dataset(strict, blind=False)[-1], self.pins['dataset_sha256'])
        self.assertFalse(loaded['features'].flags.writeable)
        # Cache remains a snapshot of validated source bytes even if raw data is
        # later removed; this explicitly does NOT reverify raw bytes on a hit.
        shutil.rmtree(self.study)
        np.testing.assert_array_equal(loaded['features'], self.load()['features'])

    def test_single_episode_blind_is_identical(self):
        first = self.rows[0]
        strict = export.export_episode(self.study/first['path'], blind=True)
        self.build(attempt_ids=[first['id']], blind=True)
        loaded = self.load()
        np.testing.assert_array_equal(strict['features'], loaded['features'])
        self.assertEqual(strict['provenance'], loaded['provenance'])
        self.assertTrue(np.all(loaded['features'][:, model.LIVE_FEATURE_START:] == 0.))

    def test_missing_masks_retained(self):
        self.build(attempt_ids=[self.rows[0]['id']])
        x = self.load()['features']; i = model.FEATURE_NAMES.index('lag0:visual_available')
        self.assertEqual(x[:, i].tolist(), [1., 0., 1.])
        self.assertTrue(np.all(x[1, i-4:i] == 0.))

    def test_expected_pins_are_required_and_self_edited_manifest_is_not_authority(self):
        self.build()
        for kw in [dict(expected_manifest_sha256='0'*64), dict(expected_protocol_sha256='0'*64),
                   dict(expected_source_sha256='0'*64), dict(expected_split='development')]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.load(**kw)
        self.amend(manifest_change=lambda m: m.update(dataset_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'independently retained'):
            self.load()

    def test_unknown_files_and_unsealed_directories_fail(self):
        self.build()
        self.out.chmod(0o755)
        (self.out/'extra').write_text('not allowed')
        with self.assertRaises(ValueError): self.load()
        (self.out/'extra').unlink()
        (self.out/'manifest.json').chmod(0o644)
        (self.out/'manifest.json').unlink()
        with self.assertRaises(ValueError): self.load()

    def test_symlinked_cache_file_fails_when_platform_permits_fixture(self):
        self.build()
        self.out.chmod(0o755)
        original = self.out/'features.npy'; moved = self.root/'moved.npy'; original.rename(moved)
        try:
            original.symlink_to(moved)
        except (OSError, NotImplementedError):
            moved.rename(original)
            self.skipTest('This host does not permit symlink creation')
        with self.assertRaisesRegex(ValueError, 'symlink'): self.load()

    def test_hardlinked_cache_file_fails_when_platform_permits_fixture(self):
        self.build()
        self.out.chmod(0o755)
        original = self.out/'features.npy'; moved = self.root/'moved.npy'; original.rename(moved)
        try:
            original.hardlink_to(moved)
        except (OSError, NotImplementedError):
            moved.rename(original)
            self.skipTest('This host does not permit hard-link creation')
        with self.assertRaisesRegex(ValueError, 'ordinary'): self.load()

    def test_changed_reordered_truncated_arrays_fail(self):
        self.build()
        p = self.out/'features.npy'; p.chmod(0o644); original = p.read_bytes()
        for body in (original[:-1], original[:140]+bytes([original[140]^1])+original[141:]):
            p.write_bytes(body)
            with self.assertRaises(ValueError): self.load()
        p.write_bytes(original)
        x = np.load(p, allow_pickle=False); np.save(p, x[::-1], allow_pickle=False)
        with self.assertRaisesRegex(ValueError, 'array bytes'): self.load()

    def test_rebound_bad_masks_nan_order_and_dtype_still_fail(self):
        self.build(attempt_ids=[self.rows[0]['id']])
        manifest_body=(self.out/'manifest.json').read_bytes(); array_body=(self.out/'features.npy').read_bytes()
        def mask(x):
            x[1, model.FEATURE_NAMES.index('lag0:visual_available')] = .5; return x
        def filled_missing(x):
            x[1, model.FEATURE_NAMES.index('lag0:visual_x_m/.5')] = 1.; return x
        def nan(x): x[1, 0]=float('nan'); return x
        for transform in (mask, filled_missing, nan, lambda x: x[::-1], lambda x: x.astype(np.float64)):
            self.out.chmod(0o755)
            for name, body in [('manifest.json',manifest_body),('features.npy',array_body)]:
                p=self.out/name;p.chmod(0o644);p.write_bytes(body)
            amended = self.amend(transform=transform)
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                self.load(expected_manifest_sha256=amended)

    def test_split_and_protocol_order_prevent_leakage(self):
        for kw in (dict(split='confirmation'), dict(attempt_ids=[self.rows[2]['id']]),
                   dict(attempt_ids=[self.rows[1]['id'], self.rows[0]['id']]),
                   dict(attempt_ids=[self.rows[0]['id'], self.rows[0]['id']]),
                   dict(stage='corrections', split='development')):
            with self.subTest(kw=kw), self.assertRaises(ValueError): self.build(**kw)
        self.protocol['cases']['development'][0]['group_id'] = 'train-a'
        self.save_protocol(); self.save_stage()
        with self.assertRaisesRegex(ValueError, 'cross registered splits'): self.build()

    def test_source_and_episode_seals_are_checked_before_build(self):
        episode=self.study/self.rows[0]['path']; (episode/'integrity.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'sealed stage'): self.build()
        self.assertFalse(self.out.exists())

    def test_correction_uses_teacher_queries_and_preserves_executed_authority(self):
        row = dict(self.rows[0]); row['controller'] = 'mixed'; row['id'] = 'correction-' + row['case']['id'] + '-mixed'
        row['path'] = 'episodes/corrections/' + row['id']; row['strict_success'] = False
        episode = self.study / row['path']; episode.parent.mkdir(parents=True)
        self.recordings[str(episode)] = fixture(episode, mode='mixed', group=row['case']['group_id'])
        row['receipt_sha256'] = write(episode/'rollout.json', {'status':'complete'})
        row['integrity_sha256'] = sha256((episode/'integrity.json').read_bytes()).hexdigest()
        self.protocol['rosters']['corrections'] = [{k: row[k] for k in ('id','case','controller','negative_control')}]
        self.save_protocol()
        h = write(self.study/'stages/corrections.json', dict(schema=study.SCHEMA+'/stage', stage='corrections',
            protocol_sha256=self.protocol_sha, status='complete', scheduled=1, rows=[row]))
        write(self.study/'stages/corrections.integrity.json', dict(sha256=h, protocol_sha256=self.protocol_sha))
        strict = export.export_episode(episode, label_source='teacher_query')
        self.build(stage='corrections')
        loaded = self.load()
        np.testing.assert_array_equal(strict['targets'], loaded['targets'])
        self.assertEqual(loaded['targets'][:,0].tolist(), [.8]*3)
        self.assertEqual(loaded['provenance'], strict['provenance'])
        self.assertEqual(self.recordings[str(episode)]['steps'][0]['command']['authority'], 'student')

    def test_active_study_low_disk_and_existing_destination_refused(self):
        (self.study/'worker.lock').write_text('active')
        with self.assertRaisesRegex(ValueError, 'actively'): self.build()
        (self.study/'worker.lock').unlink()
        with patch.object(cache.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(1,1,1)):
            with self.assertRaisesRegex(ValueError, '30 GiB'): self.build()
        self.assertFalse(self.out.exists())
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh'): self.build()
        with self.assertRaisesRegex(ValueError, 'outside'):
            cache.build_cache(self.study, self.study/'cache', expected_protocol_sha256=self.protocol_sha)


if __name__ == '__main__':
    unittest.main()
