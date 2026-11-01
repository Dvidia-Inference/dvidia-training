from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import prepare_camera_action_v01 as package


class EvidencePackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.episode = self.root / 'episode'
        self.episode.mkdir()
        package.write(self.episode / 'started.json', {'status': 'started'})
        package.write(self.episode / 'rollout.json', {'status': 'complete', 'synthetic': True})
        inventory = {p.name: package.identity(p) for p in self.episode.iterdir()}
        package.write(self.episode / 'integrity.json',
            {'schema': 'dvidia.camera-action-rollout/v2/integrity', 'files': inventory})

    def test_deterministic_archives_restore_exact_original_integrity_inventory(self):
        inventory = package.episode_inventory(self.episode)
        first = package.archive(self.episode, inventory, self.root / 'first.zip', prefix='episodes/diagnostic/example')
        second = package.archive(self.episode, inventory, self.root / 'second.zip', prefix='episodes/diagnostic/example')
        self.assertEqual(first, second)
        self.assertEqual(first['compression_level'],1)
        self.assertEqual((self.root / 'first.zip').read_bytes(), (self.root / 'second.zip').read_bytes())
        with zipfile.ZipFile(self.root / 'first.zip') as archive:
            for name, expected in inventory.items():
                raw = archive.read('episodes/diagnostic/example/' + name)
                self.assertEqual({'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}, expected)
                self.assertEqual(archive.getinfo('episodes/diagnostic/example/' + name).date_time, (1980, 1, 1, 0, 0, 0))

    def test_explicit_reviewed_compression_level_preserves_identical_raw_inventory(self):
        inventory=package.episode_inventory(self.episode)
        first=package.archive(self.episode,inventory,self.root/'fast.zip',compresslevel=1)
        second=package.archive(self.episode,inventory,self.root/'compact.zip',compresslevel=6)
        self.assertEqual(first['inventory_sha256'],second['inventory_sha256'])
        self.assertEqual(first['unpacked_bytes'],second['unpacked_bytes'])
        self.assertEqual(second['compression_level'],6)
        with self.assertRaisesRegex(ValueError,'reviewed ZIP'):
            package.archive(self.episode,inventory,self.root/'unreviewed.zip',compresslevel=9)

    def test_changed_sealed_source_refuses_during_streaming_copy(self):
        inventory = package.episode_inventory(self.episode)
        (self.episode / 'rollout.json').write_text('{"status":"different"}\n')
        with self.assertRaisesRegex(ValueError, 'identity changed|bytes changed'):
            package.archive(self.episode, inventory, self.root / 'changed.zip')

    def test_missing_extra_symlink_and_unsealed_attempt_are_refused(self):
        (self.episode / 'extra.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'unsealed, missing or extra'):
            package.episode_inventory(self.episode)
        (self.episode / 'extra.json').unlink()
        (self.episode / 'rollout.json').unlink()
        (self.episode / 'rollout.json').symlink_to(self.episode / 'started.json')
        with self.assertRaisesRegex(ValueError, 'symbolic links'):
            package.episode_inventory(self.episode)
        (self.episode / 'rollout.json').unlink()
        (self.episode / 'integrity.json').unlink()
        with self.assertRaises(OSError):
            package.episode_inventory(self.episode)

    def test_path_escape_and_secret_shaped_json_are_refused(self):
        for name in ('../secret', '/tmp/secret', 'frames/../../secret', './frame', 'frame\\secret'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                package.relative(name)
        (self.episode / 'rollout.json').write_text('"' + 'hf_' + 'A' * 32 + '"')
        inventory = {p.name: package.identity(p) for p in self.episode.iterdir()}
        with self.assertRaisesRegex(ValueError, 'Credential-shaped'):
            package.archive(self.episode, inventory, self.root / 'secret.zip')

    def test_source_wheel_and_tar_bind_registered_runtime_bytes(self):
        raw = b'"""Synthetic source fixture."""\n'
        sources = {'camera_action_study.py': sha256(raw).hexdigest()}
        wheel = self.root / 'fixture.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('dvidia_training/camera_action_study.py', raw)
        source = self.root / 'fixture.tar.gz'
        with tarfile.open(source, 'w:gz') as archive:
            entry = tarfile.TarInfo('fixture/src/dvidia_training/camera_action_study.py')
            entry.size = len(raw)
            archive.addfile(entry, io.BytesIO(raw))
            entry = tarfile.TarInfo('fixture/.gitignore')
            entry.size = 3
            archive.addfile(entry, io.BytesIO(b'foo'))
        self.assertEqual(package._runtime_members(wheel, sources, wheel=True), sources)
        self.assertEqual(package._runtime_members(source, sources, wheel=False), sources)
        with self.assertRaisesRegex(ValueError, 'exact registered runtime'):
            package._runtime_members(wheel, {'camera_action_study.py': '0' * 64}, wheel=True)

    def test_source_archive_cannot_include_private_environment_or_links(self):
        source = self.root / 'private.tar.gz'
        with tarfile.open(source, 'w:gz') as archive:
            entry = tarfile.TarInfo('fixture/.env')
            entry.size = 1
            archive.addfile(entry, io.BytesIO(b'x'))
        with self.assertRaisesRegex(ValueError, 'private source'):
            package._runtime_members(source, {}, wheel=False)
        source = self.root / 'linked.tar.gz'
        with tarfile.open(source, 'w:gz') as archive:
            entry = tarfile.TarInfo('fixture/redirect')
            entry.type = tarfile.SYMTYPE
            entry.linkname = '/tmp/other'
            archive.addfile(entry)
        with self.assertRaisesRegex(ValueError, 'links or special'):
            package._runtime_members(source, {}, wheel=False)

    def test_source_archive_must_include_exact_audit_implementation(self):
        source=self.root/'audit-source.tar.gz'; proof={}
        with tarfile.open(source,'w:gz') as archive:
            for name,field in (('study_audit.py','audit_source_sha256'),('independent_audit.py','diagnostic_helper_source_sha256')):
                raw=('synthetic '+name).encode(); proof[field]=sha256(raw).hexdigest()
                info=tarfile.TarInfo('fixture/benchmarks/camera-action-v01/'+name); info.size=len(raw)
                archive.addfile(info,io.BytesIO(raw))
        package.verify_audit_source_archive(source,proof)
        proof['audit_source_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'differs from the independent'):
            package.verify_audit_source_archive(source,proof)

    def test_oversized_source_refuses_before_creating_destination(self):
        source = self.root / 'oversized-source.tar.gz'
        with source.open('wb') as stream:
            stream.truncate(package.MAX_SOURCE + 1)
        destination = self.root / 'public' / source.name
        with self.assertRaisesRegex(ValueError, 'oversized'):
            package._copy(source, destination, maximum=package.MAX_SOURCE)
        self.assertFalse(destination.parent.exists())

    def test_running_worker_is_refused_before_inspection_or_archiving(self):
        study = self.root / 'study'
        study.mkdir()
        (study / 'worker.lock').write_text('{}')
        with patch('dvidia_training.camera_action_study.inspect') as inspect:
            with self.assertRaisesRegex(ValueError, 'worker'):
                package.prepare(study, self.root / 'payload', audit='unused', card='unused',
                    wheel='unused', source_archive='unused', source_commit='a' * 40)
            inspect.assert_not_called()
        self.assertFalse((self.root / 'payload').exists())

    def test_full_study_audit_requires_every_current_stage_and_unchanged_evidence(self):
        from copy import deepcopy
        study = self.root / 'study'; study.mkdir()
        package.write(study/'protocol.json', dict(source_sha256={'piper_feedback_score.py': 'e'*64}))
        for name in ('registration.json', 'fixed-model.json', 'report.json'):
            package.write(study/name, {'fixture': name})
        row = dict(id='unattempted', case={'id': 'scene'}, controller='teacher', negative_control=None,
            status='unattempted', complete=False, strict_success=False)
        receipt = dict(rows=[row])
        for suffix in ('.json', '.integrity.json', '.started.json'):
            package.write(study/('stages/diagnostic'+suffix), receipt)
        inventory = {p.relative_to(study).as_posix(): package.identity(p) for p in study.rglob('*') if p.is_file()}
        helpers = Path(package.__file__).resolve().parents[1]/'benchmarks/camera-action-v01'
        proof = dict(schema='dvidia.camera-action-study-audit/v1', passed=True, errors=[],
            protocol_sha256=package.identity(study/'protocol.json')['sha256'],
            physics_executed=False, estimator_executed=False, model_executed=False, source_authenticated=False, physical_robot_ready=False,
            evidence_inventory=inventory, evidence_inventory_sha256=sha256(package.canonical(inventory)).hexdigest(),
            stages_verified=['diagnostic'], attempts_verified=1, results=[dict(**row, stage='diagnostic')], qualification=None,
            audit_source_sha256=package.identity(helpers/'study_audit.py')['sha256'],
            diagnostic_helper_source_sha256=package.identity(helpers/'independent_audit.py')['sha256'], scorer_source_sha256='e'*64)
        proof_path=self.root/'full-audit.json'; package.write(proof_path, proof)
        state={'qualified': False}
        package.verify_study_audit(study, proof_path, {'diagnostic': receipt}, state)
        with self.assertRaisesRegex(ValueError, 'omits or invents'):
            package.verify_study_audit(study, proof_path, {'diagnostic': receipt, 'demonstrations': receipt}, state)
        with self.assertRaisesRegex(ValueError, 'requires independently'):
            package.verify_study_audit(study, proof_path, {'diagnostic': receipt}, {'qualified': True})
        changed = deepcopy(proof); changed['results'][0]['strict_success']=True
        proof_path.write_bytes(package.canonical(changed))
        with self.assertRaisesRegex(ValueError, 'outcome or registered scene'):
            package.verify_study_audit(study, proof_path, {'diagnostic': receipt}, state)
        proof_path.write_bytes(package.canonical(proof)); (study/'fixed-model.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed after auditing'):
            package.verify_study_audit(study, proof_path, {'diagnostic': receipt}, state)

    def test_qualified_publication_requires_full_study_audit_before_reading_payload(self):
        study=self.root/'study'; study.mkdir()
        with patch('dvidia_training.camera_action_study.inspect', return_value={'status':'complete', 'qualified':True}):
            with self.assertRaisesRegex(ValueError, 'requires a full-study'):
                package.prepare(study,self.root/'payload',audit='unused',card='unused',wheel='unused',
                    source_archive='unused',source_commit='a'*40)
        self.assertFalse((self.root/'payload').exists())

    def test_blocked_study_packages_all_sealed_attempts_without_model_artifacts(self):
        import shutil
        from test_camera_action_hardware import fixture
        from dvidia_training import camera_action_study as workflow
        study = self.root / 'study'
        study.mkdir()
        source_bytes = b'"""Synthetic runtime fixture."""\n'
        sources = {'camera_action_study.py': sha256(source_bytes).hexdigest()}
        package.write(study / 'protocol.json', {'source_sha256': sources})
        package.write(study / 'registration.json', {'fixture': 'registration'})
        package.write(study / 'fixed-model.json', {'fixture': 'comparator'})
        rows = []
        for index in range(4):
            identity = f'diagnostic-development-{index:02}-goal0-teacher'
            directory = study / 'episodes/diagnostic' / identity
            shutil.copytree(self.episode, directory)
            rows.append(dict(id=identity, case=dict(id=identity, group_id=f'development-{index:02}',
                split='development'), path=directory.relative_to(study).as_posix(),
                controller='teacher', negative_control=None, status='complete', strict_success=False,
                receipt_sha256=package.identity(directory / 'rollout.json')['sha256'],
                integrity_sha256=package.identity(directory / 'integrity.json')['sha256']))
        diagnostic = dict(status='blocked', rows=rows)
        package.write(study / 'stages/diagnostic.json', diagnostic)
        proof = dict(schema='dvidia.camera-action-diagnostic-audit/v1', passed=True,
            errors=[], attempts_verified=4, physics_executed=False, estimator_executed=False,
            results=[dict(case_id=r['case']['id'], strict_success=False) for r in rows])
        audit = self.root / 'audit.json'
        package.write(audit, proof)
        card = self.root / 'README.md'
        card.write_text('Synthetic blocked-study fixture; no trained policy.\n')
        wheel = self.root / 'fixture.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('dvidia_training/camera_action_study.py', source_bytes)
        source = self.root / 'fixture.tar.gz'
        commit = 'a' * 40
        with tarfile.open(source, 'w:gz', format=tarfile.PAX_FORMAT, pax_headers={'comment': commit}) as archive:
            entry = tarfile.TarInfo('fixture/src/dvidia_training/camera_action_study.py')
            entry.size = len(source_bytes)
            archive.addfile(entry, io.BytesIO(source_bytes))
        state = dict(status='blocked', qualification='research_candidate', qualified=False,
            report=dict(status='blocked', qualification='research_candidate', qualified=False, physical_robot_ready=False))
        output = self.root / 'payload'
        hardware=self.root/'hardware.json';package.write(hardware,fixture())
        import episode_archive_cache as cache
        prefix=rows[0]['path']; episode_inventory=package.episode_inventory(study/prefix)
        cache_root=self.root/'episode-cache'
        cached=cache_root/cache.key(episode_inventory,prefix)/'episode.zip'
        package.archive(study/prefix,episode_inventory,cached,prefix=prefix)
        def private_copy(source,destination,record):
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as target:
                target.write(source.read_bytes())
        with patch.object(workflow, 'inspect', return_value=state), \
                patch.object(workflow, '_stage', side_effect=lambda root, name: diagnostic if name == 'diagnostic' else None), \
                patch.object(cache,'_private_copy',side_effect=private_copy) as copy, \
                patch.object(package,'archive',wraps=package.archive) as archive:
            result = package.prepare(study, output, audit=audit, card=card, wheel=wheel,
                source_archive=source, source_commit=commit, hardware=hardware,episode_cache=cache_root)
        copy.assert_called_once()
        self.assertEqual(archive.call_count,4)  # one metadata ZIP and three cache misses
        self.assertEqual((output/(prefix+'.zip')).read_bytes(),cached.read_bytes())
        self.assertEqual(result['episode_count'], 4)
        self.assertEqual(result['trained_models'], 0)
        self.assertFalse((output / 'models').exists())
        manifest = package.read(output / 'manifest.json')
        self.assertIn('Four diagnostic episodes only', manifest['independent_audit_scope'])
        self.assertFalse(manifest['qualified'])
        self.assertEqual(manifest['hardware_context_file'],'hardware.json')
        self.assertEqual((output/'hardware.json').read_bytes(),hardware.read_bytes())
        self.assertEqual(len(list((output / 'episodes').rglob('*.zip'))), 4)
        import publish_camera_action_v01
        inventory = publish_camera_action_v01.files(output)
        self.assertEqual(len(inventory), len(manifest['files']) + 2)


if __name__ == '__main__':
    unittest.main()
