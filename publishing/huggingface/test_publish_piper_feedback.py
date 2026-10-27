"""Offline publication tests; FakeHub never reads a credential or uses a network."""
from hashlib import sha1, sha256
import base64
import csv
import gzip
import io
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_piper_feedback as publication


def write(root, name, value):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else publication._encoded(value))


def npz():
    stream = io.BytesIO()
    header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }\n"
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('values.npy', b'\x93NUMPY\x01\x00'+len(header).to_bytes(2, 'little')+header+b'\0'*8)
    return stream.getvalue()


def zipped(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    return stream.getvalue()


def wheel(source=b"const scheme = 'file://';\n"):
    members = {'dvidia_training/worker.js': source,
        publication.DIST_ROOT+'WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
        publication.DIST_ROOT+'METADATA': b'Metadata-Version: 2.4\nName: dvidia-training\nVersion: 0.1.0\n'}
    records = io.StringIO()
    writer = csv.writer(records)
    for name, raw in members.items():
        writer.writerow([name, 'sha256='+base64.urlsafe_b64encode(sha256(raw).digest()).rstrip(b'=').decode(), str(len(raw))])
    writer.writerow([publication.DIST_ROOT+'RECORD', '', ''])
    members[publication.DIST_ROOT+'RECORD'] = records.getvalue().encode()
    return members


def smoke(*, native=None, episode=None):
    members = {'base-model.json': b'{}', 'feedback-model.json': b'{}', 'started.json': b'{}',
        'native.json': b'{}', 'score.json': b'{"valid":true,"success":true}',
        'native.npz': native or npz(), 'episode.json.gz': episode or gzip.compress(b'{"fixture":true}', mtime=0)}
    install = dict(physical_robot_ready=False, base_model_raw_sha256=sha256(members['base-model.json']).hexdigest(),
        feedback_model_raw_sha256=sha256(members['feedback-model.json']).hexdigest(),
        files={n: sha256(raw).hexdigest() for n, raw in members.items()})
    members['install.json'] = publication._encoded(install)
    members['receipt.json'] = publication._encoded(dict(files={n: sha256(raw).hexdigest() for n, raw in members.items()}))
    return members


class ContractArchiveTests(unittest.TestCase):
    def inspect_reviewed_fixture(self, name, raw):
        with patch.dict(publication.CONTRACT_ARCHIVES, {name: publication._receipt(raw)}):
            return publication._inspect(name, raw)

    def test_only_two_contract_paths_and_exact_archive_digests_are_accepted(self):
        for name in ('contracts/arbitrary.zip', 'docs/installation-smoke.zip', 'contracts/arbitrary.whl'):
            with self.assertRaises(publication.PublicationError):
                publication._relative(name)
        name = 'contracts/dvidia_training-0.1.0-py3-none-any.whl'
        self.assertEqual(publication._relative(name).as_posix(), name)
        with self.assertRaisesRegex(publication.PublicationError, 'exact reviewed'):
            publication._inspect(name, zipped(wheel()))

    def test_reviewed_wheel_validates_record_and_allows_only_bare_quoted_file_scheme(self):
        name = 'contracts/dvidia_training-0.1.0-py3-none-any.whl'
        self.assertGreater(self.inspect_reviewed_fixture(name, zipped(wheel())), 0)
        for source in (b"const secret='hf_"+b'Z'*35+b"';", b"const path='file:///Users/private/data';"):
            with self.assertRaisesRegex(publication.PublicationError, 'credential or private path'):
                self.inspect_reviewed_fixture(name, zipped(wheel(source)))

    def test_rehashed_archive_with_forged_wheel_record_or_unapproved_footage_is_rejected(self):
        name = 'contracts/dvidia_training-0.1.0-py3-none-any.whl'
        members = wheel()
        members['dvidia_training/worker.js'] += b'// changed after RECORD\n'
        with self.assertRaisesRegex(publication.PublicationError, 'RECORD digest'):
            self.inspect_reviewed_fixture(name, zipped(members))
        members = wheel()
        members['dvidia_training/raw-user-video.mp4'] = b'not an approved example asset'
        with self.assertRaisesRegex(publication.PublicationError, 'reviewed simulator/example'):
            self.inspect_reviewed_fixture(name, zipped(members))

    def test_archive_traversal_symlink_encryption_duplicates_and_expansion_are_rejected(self):
        with self.assertRaisesRegex(publication.PublicationError, 'unsafe'):
            publication._archive_members(zipped({'../escape.py': b'bad'}))
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            info = zipfile.ZipInfo('dvidia_training/link.py')
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b'/outside')
        with self.assertRaisesRegex(publication.PublicationError, 'linked'):
            publication._archive_members(stream.getvalue())
        stream = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(stream, 'w') as archive:
                archive.writestr('same.py', b'one')
                archive.writestr('same.py', b'two')
        with self.assertRaisesRegex(publication.PublicationError, 'duplicate'):
            publication._archive_members(stream.getvalue())
        encrypted = bytearray(zipped({'data.py': b'code'}))
        local, central = encrypted.index(b'PK\x03\x04'), encrypted.index(b'PK\x01\x02')
        encrypted[local+6] |= 1
        encrypted[central+8] |= 1
        with self.assertRaisesRegex(publication.PublicationError, 'encrypted'):
            publication._archive_members(bytes(encrypted))
        with patch.object(publication, 'ARCHIVE_DECODED_LIMIT', 1), self.assertRaisesRegex(publication.PublicationError, 'decoded bound'):
            publication._archive_members(zipped({'data.py': b'oversized'}))

    def test_installation_archive_scans_nested_gzip_npz_and_receipt_hashes(self):
        name = 'contracts/installation-smoke.zip'
        self.assertGreater(self.inspect_reviewed_fixture(name, zipped(smoke())), 0)
        token_episode = gzip.compress(publication._encoded({'note': 'hf_'+'Z'*35}), mtime=0)
        with self.assertRaisesRegex(publication.PublicationError, 'credential'):
            self.inspect_reviewed_fixture(name, zipped(smoke(episode=token_episode)))
        stream = io.BytesIO()
        header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1,), }\n"
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('unsafe.npy', b'\x93NUMPY\x01\x00'+len(header).to_bytes(2,'little')+header+b'pickle')
        with self.assertRaisesRegex(publication.PublicationError, 'Object or structured'):
            self.inspect_reviewed_fixture(name, zipped(smoke(native=stream.getvalue())))
        members = smoke()
        members['started.json'] = b'{"changed":true}'
        with self.assertRaisesRegex(publication.PublicationError, 'smoke hashes'):
            self.inspect_reviewed_fixture(name, zipped(members))


def reseal(root):
    files = {p.relative_to(root).as_posix(): publication._receipt(p.read_bytes())
             for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'integrity.json'}
    write(root, 'integrity.json', dict(schema=publication.INTEGRITY_SCHEMA, files=files))


def fixture(base):
    run, viewer, stage = base/'run', base/'viewer', base/'stage'
    run.mkdir()
    viewer.mkdir()
    source = b'# frozen fixture implementation\n'
    runtime = {'piper_feedback_model.py': sha256(source).hexdigest()}
    write(run, 'sources/piper_feedback_model.py', source)
    # Force more than one operation batch, without large or slow fixture data.
    for index in range(28):
        write(run, f'sources/extra{index:02}.py', f'# fixture {index}\n'.encode())
    write(run, 'base-model.json', {'schema': 'fixture-base'})
    cases = {}
    for split, count in (('train', 64), ('development', 16), ('confirmation', 24)):
        cases[split] = [dict(id=f'{split}-{i:02}', condition='nominal' if i%2 == 0 else 'jaw_interruption',
                            start_xy=[.3, -.03], goal_xy=[.35, .08], seed=i, pulses=[])
                        for i in range(count)]
    protocol = dict(schema=publication.STUDY_SCHEMA, physical_robot_ready=False,
                    variants=list(publication.VARIANTS), cases=cases,
                    base_model_raw_sha256=sha256((run/'base-model.json').read_bytes()).hexdigest())
    write(run, 'protocol.json', protocol)
    protocol_sha = sha256((run/'protocol.json').read_bytes()).hexdigest()
    for name, blind in (('full', False), ('blind', True)):
        for filename, value in (('model.json', {'schema': 'fixture-feedback', 'blind': blind}),
                                ('receipt.json', {'fixture': 'train'}), ('development.json', {'fixture': 'development'})):
            write(run, f'candidates/{name}/{filename}', value)
    frozen = dict(schema=publication.STUDY_SCHEMA, protocol_sha256=protocol_sha,
        base_model_sha256=protocol['base_model_raw_sha256'], runtime_sha256=runtime, selected='full', blind='blind')
    for name, prefix in (('full', 'selected'), ('blind', 'blind')):
        for filename, field in (('model.json', prefix+'_sha256'), ('receipt.json', prefix+'_receipt_sha256'),
                                ('development.json', prefix+'_development_sha256')):
            frozen[field] = sha256((run/f'candidates/{name}/{filename}').read_bytes()).hexdigest()
    write(run, 'frozen.json', frozen)

    def attempt(folder, case, variant, collect=False, success=False):
        binding = dict(case=case, variant=variant, collect_labels=collect, protocol_sha256=protocol_sha, runtime_sha256=runtime)
        score = dict(valid=True, success=success)
        write(run, folder+'/started.json', binding)
        write(run, folder+'/episode.json.gz', gzip.compress(publication._encoded(dict(case_id=case['id'], valid=True)), mtime=0))
        write(run, folder+'/native.npz', npz())
        write(run, folder+'/native.json', dict(schema='fixture-native'))
        write(run, folder+'/score.json', score)
        if collect:
            write(run, folder+'/samples.npz', npz())
        files = {p.name: sha256(p.read_bytes()).hexdigest() for p in (run/folder).iterdir() if p.is_file()}
        write(run, folder+'/receipt.json', dict(binding=binding, files=files))
        return score

    for case in cases['train']:
        attempt('demonstrations/'+case['id'], case, 'teacher', collect=True, success=True)
    rows = []
    for case in cases['confirmation']:
        for variant in publication.VARIANTS:
            # Deliberately unsuccessful learned model: staging must not invent improvement.
            success = variant == 'teacher'
            score = attempt(f"confirmation/{case['id']}/{variant}", case, variant, success=success)
            rows.append(dict(case_id=case['id'], condition=case['condition'], variant=variant, score=score))
    report = dict(schema=publication.STUDY_SCHEMA, physical_robot_ready=False,
        protocol_sha256=protocol_sha, frozen_sha256=sha256((run/'frozen.json').read_bytes()).hexdigest(),
        rows=rows, limitations=['Synthetic simulation fixture; no hardware qualification.'],
        variants={v: dict(scheduled=24, valid=24, successes=24 if v == 'teacher' else 0) for v in publication.VARIANTS})
    write(run, 'report.json', report)
    bindings = {key: sha256((run/name).read_bytes()).hexdigest() for key, name in
                (('protocol_sha256', 'protocol.json'), ('frozen_sha256', 'frozen.json'), ('report_sha256', 'report.json'))}
    write(run, 'audit.json', dict(schema=publication.AUDIT_SCHEMA, status='passed', **bindings))
    reseal(run)
    for name in ('report.json', 'protocol.json', 'frozen.json'):
        write(viewer, name, (run/name).read_bytes())
    for name, value in (('index.html', b'<!doctype html><title>Recorded fixture</title>'),
                        ('motion.json', {}), ('scene.json', {}), ('geometry.bin', b'1234'), ('geometry.bin.gz', gzip.compress(b'1234', mtime=0))):
        write(viewer, name, value)
    write(viewer, 'viewer-integrity.json', dict(schema=publication.VIEWER_SCHEMA,
        files={p.name: publication._receipt(p.read_bytes()) for p in sorted(viewer.iterdir())}))
    return run, viewer, stage


def staged(base):
    run, viewer, stage = fixture(base)
    plan = publication.stage(run, viewer, 'f'*40, 'a'*40, stage)
    return stage/'publication-plan.json', plan


def lineage_fixture(base):
    run, viewer, stage = fixture(base)
    old = run/'prior-studies/study01'
    old.mkdir(parents=True)
    for name in ('protocol.json', 'base-model.json'):
        shutil.copyfile(run/name, old/name)
    shutil.copytree(run/'sources', old/'sources')
    shutil.copytree(run/'demonstrations', old/'demonstrations')
    protocol = json.loads((old/'protocol.json').read_bytes())
    protocol_sha = sha256((old/'protocol.json').read_bytes()).hexdigest()
    runtime = json.loads((run/'frozen.json').read_bytes())['runtime_sha256']

    def old_attempt(folder, case, variant, collect):
        binding = dict(case=case, variant=variant, collect_labels=collect, protocol_sha256=protocol_sha, runtime_sha256=runtime)
        values = {'started.json': binding, 'episode.json.gz': gzip.compress(publication._encoded({'case_id': case['id']}), mtime=0),
                  'native.npz': npz(), 'native.json': {'fixture': True}, 'score.json': dict(valid=True, success=False)}
        if collect:
            values['samples.npz'] = npz()
        for name, value in values.items():
            write(old, folder+'/'+name, value)
        write(old, folder+'/receipt.json', dict(binding=binding,
            files={n: sha256((old/folder/n).read_bytes()).hexdigest() for n in values}))
    for case in protocol['cases']['train']:
        if case['condition'] == 'jaw_interruption':
            old_attempt('dagger/'+case['id'], case, 'mixture', True)
    for index in range(5):
        candidate = f'negative{index}'
        prefix = 'candidates/'+candidate
        write(old, prefix+'/model.json', {'fixture': candidate})
        model_sha = sha256((old/prefix/'model.json').read_bytes()).hexdigest()
        write(old, prefix+'/receipt.json', dict(model_sha256=model_sha, protocol_sha256=protocol_sha, source_sha256=runtime))
        rows = []
        for case in protocol['cases']['development']:
            folder = prefix+'/development/'+case['id']
            old_attempt(folder, case, 'feedback', False)
            rows.append(dict(case_id=case['id'], condition=case['condition'], score=json.loads((old/folder/'score.json').read_bytes())))
        write(old, prefix+'/development.json', dict(model_sha256=model_sha, protocol_sha256=protocol_sha,
            runtime_sha256=runtime, scheduled=16, successes=0, rows=rows))
    history = {p.relative_to(old).as_posix(): publication._receipt(p.read_bytes()) for p in sorted(old.rglob('*')) if p.is_file()}
    parent = dict(schema='dvidia.piper-feedback-training-parent/v1', protocol_sha256=protocol_sha,
        runtime_sha256=runtime, files=history,
        teacher_sample_files={f"demonstrations/{c['id']}/samples.npz": history[f"demonstrations/{c['id']}/samples.npz"]['sha256'] for c in protocol['cases']['train']})
    write(run, 'training-parent.json', parent)
    parent_sha = sha256((run/'training-parent.json').read_bytes()).hexdigest()
    shutil.rmtree(run/'demonstrations')
    active_protocol = {**protocol, 'training_parent_manifest_sha256': parent_sha}
    write(run, 'protocol.json', active_protocol)
    active_sha = sha256((run/'protocol.json').read_bytes()).hexdigest()
    frozen = json.loads((run/'frozen.json').read_bytes())
    frozen['protocol_sha256'] = active_sha
    for candidate, prefix in (('full', 'selected'), ('blind', 'blind')):
        write(run, f'candidates/{candidate}/receipt.json', dict(training_parent_sha256=parent_sha))
        frozen[prefix+'_receipt_sha256'] = sha256((run/f'candidates/{candidate}/receipt.json').read_bytes()).hexdigest()
    write(run, 'frozen.json', frozen)
    report = json.loads((run/'report.json').read_bytes())
    report.update(protocol_sha256=active_sha, frozen_sha256=sha256((run/'frozen.json').read_bytes()).hexdigest())
    write(run, 'report.json', report)
    for path in (run/'confirmation').glob('*/*/started.json'):
        bound = json.loads(path.read_bytes())
        bound['protocol_sha256'] = active_sha
        path.write_bytes(publication._encoded(bound))
        receipt_path = path.parent/'receipt.json'
        receipt = json.loads(receipt_path.read_bytes())
        receipt['binding'] = bound
        receipt['files']['started.json'] = sha256(path.read_bytes()).hexdigest()
        receipt_path.write_bytes(publication._encoded(receipt))
    bindings = {key: sha256((run/name).read_bytes()).hexdigest() for key, name in
                (('protocol_sha256', 'protocol.json'), ('frozen_sha256', 'frozen.json'), ('report_sha256', 'report.json'))}
    write(run, 'audit.json', dict(schema=publication.AUDIT_SCHEMA, status='passed', training_parent_sha256=parent_sha, **bindings))
    reseal(run)
    for name in ('report.json', 'protocol.json', 'frozen.json'):
        write(viewer, name, (run/name).read_bytes())
    write(viewer, 'viewer-integrity.json', dict(schema=publication.VIEWER_SCHEMA,
        files={p.name: publication._receipt(p.read_bytes()) for p in sorted(viewer.iterdir()) if p.name != 'viewer-integrity.json'}))
    return run, viewer, stage


class Add:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo, self.path_or_fileobj = path_in_repo, path_or_fileobj
        self._upload_mode = self._should_ignore = None


class RepositoryNotFoundError(Exception):
    pass


class FakeHub:
    def __init__(self, unknown=False):
        attrs = (b'*.bin filter=lfs diff=lfs merge=lfs -text\n'
                 b'*.gz filter=lfs diff=lfs merge=lfs -text\n'
                 b'*.npz filter=lfs diff=lfs merge=lfs -text\n')
        self.repos = {'space': dict(head='a'*40, files={'.gitattributes': attrs, 'old/index.html': b'existing'}, history={}, lfs=set())}
        self.repos['space']['history']['a'*40] = dict(self.repos['space']['files'])
        self.events, self.commits, self.unknown = [], [], unknown

    def repo_info(self, repo_id, repo_type):
        self.events.append(('read', repo_type))
        if repo_type not in self.repos:
            raise RepositoryNotFoundError()
        repo = self.repos[repo_type]
        return SimpleNamespace(private=False, sha=repo['head'], siblings=[SimpleNamespace(rfilename=n) for n in repo['files']],
                               sdk='static' if repo_type == 'space' else None)

    def list_repo_tree(self, repo_id, repo_type, revision, recursive=True):
        repo = self.repos[repo_type]
        for name, raw in repo['history'][revision].items():
            blob = sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            yield SimpleNamespace(path=name, blob_id=blob, size=len(raw),
                lfs=SimpleNamespace(sha256=sha256(raw).hexdigest()) if name in repo['lfs'] else None)

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        assert private is False and exist_ok is False and repo_type not in self.repos
        self.events.append(('create', repo_type))
        files, head = {'.gitattributes': b'*.npz filter=lfs diff=lfs merge=lfs -text\n'}, '1'*40
        self.repos[repo_type] = dict(files=files, head=head, history={head: dict(files)}, lfs=set())

    def create_commit(self, repo_id, repo_type, operations, parent_commit, commit_message):
        repo = self.repos[repo_type]
        assert repo['head'] == parent_commit
        assert len(operations) <= 25 and sum(len(op.path_or_fileobj) for op in operations) <= 8*1024*1024
        names = [op.path_in_repo for op in operations]
        assert not set(names)&set(repo['files'])
        self.events.append(('commit', repo_type))
        for op in operations:
            repo['files'][op.path_in_repo] = op.path_or_fileobj
            if repo_type == 'space':
                assert op._should_ignore is False
                assert (op._upload_mode == 'lfs') == op.path_in_repo.endswith(publication.LFS_SUFFIXES)
            if op._upload_mode == 'lfs':
                repo['lfs'].add(op.path_in_repo)
        head = sha1((parent_commit+'\0'+'\0'.join(names)).encode()).hexdigest()
        repo.update(head=head)
        repo['history'][head] = dict(repo['files'])
        self.commits.append((repo_type, head, names, parent_commit))
        if self.unknown:
            self.unknown = False
            raise RuntimeError('simulated lost response after commit')
        return SimpleNamespace(oid=head)

    def download(self, repo_id, name, kind, revision):
        self.events.append(('download', kind, revision, name))
        return self.repos[kind]['history'][revision][name]

    def external(self, kind, name, raw):
        repo = self.repos[kind]
        repo['files'][name] = raw
        repo['head'] = sha1((repo['head']+name).encode()).hexdigest()
        repo['history'][repo['head']] = dict(repo['files'])


class FeedbackPublicationTests(unittest.TestCase):
    def test_lineage_preserves_original_negative_study_without_relabeling_or_duplicate_demos(self):
        with tempfile.TemporaryDirectory() as directory:
            run, viewer, stage = lineage_fixture(Path(directory))
            old_binding = (run/'prior-studies/study01/demonstrations/train-00/started.json').read_bytes()
            plan = publication.stage(run, viewer, 'f'*40, 'a'*40, stage)
            self.assertFalse((stage/'dataset/demonstrations').exists())
            self.assertEqual((stage/'dataset/prior-studies/study01/demonstrations/train-00/started.json').read_bytes(), old_binding)
            self.assertEqual(len(list((stage/'dataset/prior-studies/study01').glob('dagger/*/native.npz'))), 32)
            self.assertEqual(len(list((stage/'dataset/prior-studies/study01').glob('candidates/*/development/*/native.npz'))), 80)
            self.assertFalse((stage/'model/prior-studies').exists())
            self.assertTrue((stage/'model/training-parent.json').exists())
            self.assertEqual(publication.checked_plan(stage/'publication-plan.json'), plan)

    def test_lineage_rejects_missing_original_failure_even_with_a_new_outer_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            run, viewer, stage = lineage_fixture(Path(directory))
            (run/'prior-studies/study01/candidates/negative0/development/development-00/native.npz').unlink()
            reseal(run)
            with self.assertRaisesRegex(publication.PublicationError, 'Imported history differs'):
                publication.stage(run, viewer, 'f'*40, 'a'*40, stage)

    def test_lineage_rejects_old_protocol_relabeling_and_unbound_active_training(self):
        with tempfile.TemporaryDirectory() as directory:
            run, viewer, stage = lineage_fixture(Path(directory))
            path = run/'prior-studies/study01/demonstrations/train-00/started.json'
            original = path.read_bytes()
            bound = json.loads(original)
            bound['protocol_sha256'] = sha256((run/'protocol.json').read_bytes()).hexdigest()
            path.write_bytes(publication._encoded(bound))
            reseal(run)
            with self.assertRaisesRegex(publication.PublicationError, 'Imported history differs'):
                publication.stage(run, viewer, 'f'*40, 'a'*40, stage)
            path.write_bytes(original)
            write(run, 'candidates/full/receipt.json', {'training_parent_sha256': '0'*64})
            frozen = json.loads((run/'frozen.json').read_bytes())
            frozen['selected_receipt_sha256'] = sha256((run/'candidates/full/receipt.json').read_bytes()).hexdigest()
            write(run, 'frozen.json', frozen)
            report = json.loads((run/'report.json').read_bytes())
            report['frozen_sha256'] = sha256((run/'frozen.json').read_bytes()).hexdigest()
            write(run, 'report.json', report)
            audit = json.loads((run/'audit.json').read_bytes())
            audit.update(frozen_sha256=report['frozen_sha256'], report_sha256=sha256((run/'report.json').read_bytes()).hexdigest())
            write(run, 'audit.json', audit)
            reseal(run)
            with self.assertRaisesRegex(publication.PublicationError, 'candidate training receipt'):
                publication.stage(run, viewer, 'f'*40, 'a'*40, stage)
    def test_stage_preserves_all_evidence_and_small_space_without_claiming_benefit(self):
        with tempfile.TemporaryDirectory() as directory:
            path, plan = staged(Path(directory))
            self.assertEqual([r['repo_id'] for r in plan['repositories']], list(publication.REPOSITORIES.values()))
            dataset = path.parent/'dataset'
            self.assertEqual(len(list(dataset.glob('confirmation/*/*/native.npz'))), 120)
            self.assertEqual(len(list(dataset.glob('demonstrations/*/native.npz'))), 64)
            self.assertFalse(list((path.parent/'space').rglob('*.npz')))
            self.assertIn('| feedback | 0 / 24 |', (path.parent/'model/README.md').read_text())
            self.assertIn('viewer/index.html', (path.parent/'space'/publication.SPACE_NAMESPACE/'index.html').read_text())
            self.assertEqual(publication.checked_plan(path), plan)

    def test_incomplete_report_or_unbound_audit_fails_before_stage_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            run, viewer, stage = fixture(Path(directory))
            report = json.loads((run/'report.json').read_bytes())
            report['rows'].pop()
            write(run, 'report.json', report)
            reseal(run)
            with self.assertRaises(publication.PublicationError):
                publication.stage(run, viewer, 'f'*40, 'a'*40, stage)
            self.assertFalse(stage.exists())

    def test_missing_native_or_modified_failed_attempt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, viewer, stage = fixture(Path(directory))
            (run/'confirmation/confirmation-00/feedback/native.npz').unlink()
            reseal(run)
            with self.assertRaisesRegex(publication.PublicationError, 'native attempt'):
                publication.stage(run, viewer, 'f'*40, 'a'*40, stage)

    def test_symlink_and_credential_inside_compressed_data_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, viewer, stage = fixture(base)
            (run/'sources/link.py').symlink_to(run/'sources/piper_feedback_model.py')
            with self.assertRaisesRegex(publication.PublicationError, 'symlink'):
                publication.stage(run, viewer, 'f'*40, 'a'*40, stage)
            (run/'sources/link.py').unlink()
            write(run, 'demonstrations/train-00/episode.json.gz', gzip.compress(json.dumps({'note': 'hf_'+'Z'*35}).encode()))
            with self.assertRaisesRegex(publication.PublicationError, 'credential'):
                publication.stage(run, viewer, 'f'*40, 'a'*40, stage)

    def test_tampered_plan_destination_and_added_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, plan = staged(Path(directory))
            plan['repositories'][0]['repo_id'] = 'Other/model'
            write(path.parent, path.name, plan)
            with self.assertRaisesRegex(publication.PublicationError, 'destination'):
                publication.checked_plan(path)
            plan['repositories'][0]['repo_id'] = publication.MODEL_REPO
            write(path.parent, path.name, plan)
            write(path.parent/'space', 'root.html', b'unreviewed')
            with self.assertRaisesRegex(publication.PublicationError, 'inventory'):
                publication.checked_plan(path)

    def test_publish_preflights_all_repos_journals_bounds_and_preserves_space(self):
        with tempfile.TemporaryDirectory() as directory:
            path, plan = staged(Path(directory))
            hub = FakeHub()
            receipt = publication.publish(path, api=hub, downloader=hub.download, operation_cls=Add)
            self.assertTrue(receipt['anonymous_readback'])
            self.assertEqual(hub.repos['space']['files']['old/index.html'], b'existing')
            self.assertTrue((path.parent/publication.JOURNAL_NAME).exists())
            first_write = next(i for i,e in enumerate(hub.events) if e[0] in ('create', 'commit'))
            self.assertTrue({('read', k) for k in publication.REPOSITORIES} <= set(hub.events[:first_write]))
            self.assertGreater(len(hub.commits), 3)
            for kind, revision, names, parent in hub.commits:
                self.assertTrue(all(('download', kind, revision, n) in hub.events for n in names))
            with self.assertRaisesRegex(publication.PublicationError, 'receipt already'):
                publication.publish(path, api=hub, downloader=hub.download, operation_cls=Add)

    def test_unknown_outcome_reconciles_only_exact_pending_batch_without_reupload(self):
        with tempfile.TemporaryDirectory() as directory:
            path, plan = staged(Path(directory))
            hub = FakeHub(unknown=True)
            with self.assertRaisesRegex(publication.PublicationError, 'uncertain'):
                publication.publish(path, api=hub, downloader=hub.download, operation_cls=Add)
            first = hub.commits[0]
            journal = json.loads((path.parent/publication.JOURNAL_NAME).read_bytes())
            self.assertEqual(journal['repositories']['model']['pending']['paths'], first[2])
            result = publication.publish(path, api=hub, downloader=hub.download, operation_cls=Add)
            self.assertEqual(len(result['repositories']), 3)
            self.assertEqual(sum(c[1] == first[1] for c in hub.commits), 1)
            journal = json.loads((path.parent/publication.JOURNAL_NAME).read_bytes())
            self.assertTrue(journal['repositories']['model']['batches'][0]['reconciled'])

    def test_unknown_outcome_with_outside_change_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path, plan = staged(Path(directory))
            hub = FakeHub(unknown=True)
            with self.assertRaises(publication.PublicationError):
                publication.publish(path, api=hub, downloader=hub.download, operation_cls=Add)
            hub.external('space', 'old/index.html', b'modified by somebody else')
            count = len(hub.commits)
            with self.assertRaisesRegex(publication.PublicationError, 'existing repository file'):
                publication.publish(path, api=hub, downloader=hub.download, operation_cls=Add)
            self.assertEqual(len(hub.commits), count)

    def test_existing_new_repo_or_moved_space_refuses_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, plan = staged(Path(directory))
            hub = FakeHub()
            hub.create_repo(publication.MODEL_REPO, 'model', False, False)
            hub.events.clear()
            with self.assertRaisesRegex(publication.PublicationError, 'must not already exist'):
                publication.publish(path, api=hub, downloader=hub.download, operation_cls=Add)
            self.assertFalse(hub.commits)
            del hub.repos['model']
            hub.external('space', 'old/another.html', b'new externally')
            with self.assertRaisesRegex(publication.PublicationError, 'baseline commit'):
                publication.publish(path, api=hub, downloader=hub.download, operation_cls=Add)
            self.assertFalse(hub.commits)

    def test_anonymous_corruption_stops_after_one_batch_with_pending_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path, plan = staged(Path(directory))
            hub = FakeHub()
            def corrupt(repo, name, kind, revision):
                value = hub.download(repo, name, kind, revision)
                return b'?'*len(value) if kind == 'model' else value
            with self.assertRaisesRegex(publication.PublicationError, 'public bytes differ'):
                publication.publish(path, api=hub, downloader=corrupt, operation_cls=Add)
            self.assertEqual(len(hub.commits), 1)
            journal = json.loads((path.parent/publication.JOURNAL_NAME).read_bytes())
            self.assertIsNotNone(journal['repositories']['model']['pending'])

    def test_no_object_or_traversal_arrays_and_byte_batches_are_bounded(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('../secret.npy', b'bad')
        with self.assertRaisesRegex(publication.PublicationError, 'unsafe members'):
            publication._inspect('native.npz', stream.getvalue())
        row = dict(files=[dict(path=f'{i}.npz', bytes=5*1024*1024) for i in range(3)])
        self.assertEqual(publication._batches(row, {i['path'] for i in row['files']}), [['0.npz'], ['1.npz'], ['2.npz']])

    def test_public_sdk_cache_symlink_is_hashed_but_artifact_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b'expected public bytes'
            write(root, 'blob.bin', raw)
            (root/'cached.bin').symlink_to(root/'blob.bin')
            item = dict(path='public.bin', **publication._receipt(raw))
            row = dict(repo_id=publication.DATASET_REPO, repo_type='dataset')
            self.assertEqual(publication._download(row, 'a'*40, item,
                lambda *args: root/'cached.bin'), raw)
            with self.assertRaisesRegex(publication.PublicationError, 'symlink'):
                publication._read(root/'cached.bin')


if __name__ == '__main__':
    unittest.main()
