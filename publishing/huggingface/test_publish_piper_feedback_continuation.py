"""Separate continuation, using real scientific staging and a bounded FakeHub."""
from copy import deepcopy
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import publish_piper_feedback_continuation as continuation
from test_publish_piper_feedback import staged, Add


original = continuation.original


def entry(raw, lfs=False):
    return dict(blob=sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(), size=len(raw),
                lfs_sha256=sha256(raw).hexdigest() if lfs else None)


class FakeHub:
    def __init__(self):
        attributes = b''.join(b'*'+suffix.encode()+b' filter=lfs diff=lfs merge=lfs -text\n' for suffix in original.LFS_SUFFIXES)
        self.repos, self.commits, self.reads, self.pending_path = {}, [], [], None
        self.ambiguous_after = False; self.ambiguous_before = False
        self.bad_download = None; self.mutate_after = False; self.private = False; self.wrong_space_mode = False
        self.auto_dataset_wheel_rule = False
        self.dataset_rule_override = None
        self.append_rule_on_every_commit = False
        for kind, head in [('model', '1'*40), ('dataset', '2'*40), ('space', 'a'*40)]:
            files = {'.gitattributes': attributes}
            if kind == 'space': files['old/index.html'] = b'old shared Space content'
            self.repos[kind] = dict(head=head, files=files, lfs=set(), history={head: dict(files)},
                                    parents={head: None}, titles={head: 'Original repository'})

    def repo_info(self, repo_id, *, repo_type):
        assert repo_id == original.REPOSITORIES[repo_type]
        repo = self.repos[repo_type]
        return SimpleNamespace(sha=repo['head'], private=self.private, siblings=[], sdk='static')

    def list_repo_tree(self, repo_id, *, repo_type, revision, recursive):
        assert repo_id == original.REPOSITORIES[repo_type] and recursive
        repo = self.repos[repo_type]
        for name, raw in repo['history'][revision].items():
            value = entry(raw, name in repo['lfs'])
            yield SimpleNamespace(path=name, blob_id=value['blob'], size=len(raw),
                                  lfs={'sha256': value['lfs_sha256']} if value['lfs_sha256'] else None)

    def tree(self, kind):
        repo = self.repos[kind]
        return {name: entry(raw, name in repo['lfs']) for name, raw in repo['files'].items()}

    def create_commit(self, repo_id, *, repo_type, operations, parent_commit, commit_message):
        assert repo_id == original.REPOSITORIES[repo_type]
        repo = self.repos[repo_type]
        assert repo['head'] == parent_commit
        assert len(operations) <= continuation.MAX_FILES
        assert sum(len(op.path_or_fileobj) for op in operations) <= continuation.MAX_BYTES
        names = [op.path_in_repo for op in operations]
        assert not set(names)&set(repo['files'])
        if self.pending_path is not None:
            pending = json.loads(self.pending_path.read_bytes())['pending']
            assert pending == dict(repo_type=repo_type, parent_commit=parent_commit,
                paths=names, group_index=pending['group_index'], metadata_transition=pending['metadata_transition'])
        if self.ambiguous_before:
            self.ambiguous_before = False
            raise RuntimeError('Connection lost before the commit arrived')
        for operation in operations:
            assert len(operation.path_or_fileobj) <= original.MAX_FILE
            repo['files'][operation.path_in_repo] = operation.path_or_fileobj
            lfs = operation.path_in_repo.endswith(original.LFS_SUFFIXES)
            if repo_type == 'space':
                assert operation._should_ignore is False
                assert (operation._upload_mode == 'lfs') == lfs
            if lfs and not (repo_type == 'space' and self.wrong_space_mode): repo['lfs'].add(operation.path_in_repo)
        if repo_type == 'dataset' and (self.append_rule_on_every_commit or self.auto_dataset_wheel_rule and continuation.WHEEL in names):
            rule = self.dataset_rule_override if self.dataset_rule_override is not None else continuation.WHEEL_RULE
            repo['files'][continuation.ATTRIBUTES] += rule
        head = sha1((parent_commit+'\0'+commit_message+'\0'+'\0'.join(names)).encode()).hexdigest()
        repo['head'] = head; repo['history'][head] = dict(repo['files'])
        repo['parents'][head] = parent_commit; repo['titles'][head] = commit_message
        self.commits.append(dict(kind=repo_type, parent=parent_commit, head=head, names=names))
        if self.mutate_after:
            self.mutate_after = False
            self.external(repo_type, 'unexpected.json', b'{"foreign":true}')
        if self.ambiguous_after:
            self.ambiguous_after = False
            raise RuntimeError('Reply lost after the commit landed')
        return SimpleNamespace(oid=head)

    def list_repo_commits(self, repo_id, *, repo_type, revision):
        repo = self.repos[repo_type]
        while revision is not None:
            yield SimpleNamespace(commit_id=revision, title=repo['titles'][revision])
            revision = repo['parents'][revision]

    def download(self, repo_id, name, kind, revision):
        self.reads.append((kind, revision, name))
        raw = self.repos[kind]['history'][revision][name]
        return bytes([raw[0]^1])+raw[1:] if name == self.bad_download else raw

    def external(self, kind, name, raw):
        repo = self.repos[kind]; parent = repo['head']
        repo['files'][name] = raw
        head = sha1((parent+name+str(len(raw))).encode()).hexdigest()
        repo['head'] = head; repo['history'][head] = dict(repo['files'])
        repo['parents'][head] = parent; repo['titles'][head] = 'Foreign change'


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.plan_path, self.original_plan = staged(self.home)
        self.journal_path = self.plan_path.parent/original.JOURNAL_NAME
        self.pid = 123456789
        self.api = FakeHub()
        states = {}
        for row in self.original_plan['repositories']:
            kind = row['repo_type']; repo = self.api.repos[kind]
            state = dict(baseline_sha=repo['head'] if kind == 'space' else None,
                created_sha=repo['head'] if kind != 'space' else None,
                baseline_tree=self.api.tree(kind), batches=[], pending=None)
            if kind == 'space': state['lfs_suffixes'] = list(original.LFS_SUFFIXES)
            batches = original._batches(row, {item['path'] for item in row['files']})
            prefix = batches if kind == 'model' else batches[:1] if kind == 'dataset' else []
            for batch in prefix:
                parent = repo['head']
                operations = [Add(name, (self.plan_path.parent/kind/name).read_bytes()) for name in batch]
                result = self.api.create_commit(row['repo_id'], repo_type=kind, operations=operations,
                    parent_commit=parent, commit_message='Original verified evidence prefix')
                state['batches'].append(dict(parent_commit=parent, commit=result.oid, paths=batch, reconciled=False))
            states[kind] = state
        self.original_journal = dict(schema=original.SCHEMA+'/journal',
            plan_sha256=sha256(self.plan_path.read_bytes()).hexdigest(), repositories=states)
        self.journal_path.write_bytes(original._encoded(self.original_journal))
        self.saved_plan = self.plan_path.read_bytes(); self.saved_journal = self.journal_path.read_bytes()
        self.boundary = self.home/'boundary.json'
        self.write_boundary()
        self.output = self.home/'continuation'
        self.dead = patch.object(continuation, '_dead', lambda pid: None)
        self.dead.start(); self.addCleanup(self.dead.stop)

    def write_boundary(self):
        self.boundary.write_bytes(original._encoded(dict(schema=continuation.SCHEMA+'/boundary',
            original_plan=original._receipt(self.plan_path.read_bytes()), original_journal=original._receipt(self.journal_path.read_bytes()),
            stopped_writer_pid=self.pid, anonymous_public_trees=True,
            metadata_evidence=None,
            repositories={kind: dict(repo_id=repo_id, repo_type=kind, head=self.api.repos[kind]['head'], tree=self.api.tree(kind))
                          for kind, repo_id in original.REPOSITORIES.items()})))

    def stage(self):
        result = continuation.stage(self.plan_path, self.journal_path, self.boundary, self.pid, self.output)
        self.api.pending_path = self.output/continuation.JOURNAL
        return result

    def publish(self):
        return continuation.publish(self.output/continuation.PLAN,
            api=self.api, downloader=self.api.download, operation_cls=Add)

    def test_offline_stage_preserves_original_documents_and_exact_logical_batch_groups(self):
        with patch.object(continuation, 'client', side_effect=AssertionError('credentials forbidden')):
            plan = self.stage()
            continuation.checked_plan(self.output/continuation.PLAN)
        self.assertEqual(self.plan_path.read_bytes(), self.saved_plan)
        self.assertEqual(self.journal_path.read_bytes(), self.saved_journal)
        for row in plan['repositories']:
            source_row = next(r for r in self.original_plan['repositories'] if r['repo_type'] == row['repo_type'])
            old = self.original_journal['repositories'][row['repo_type']]
            logical = original._batches(source_row, {item['path'] for item in source_row['files']})
            flattened = [b['paths'] for group in row['groups'] for b in group['logical_batches']]
            self.assertEqual(flattened, logical[len(old['batches']):])
            self.assertTrue(all(group['bytes'] <= 64*1024*1024 and len(group['paths']) <= 100 for group in row['groups']))
        self.assertTrue(any(len(g['logical_batches']) > 1 for row in plan['repositories'] for g in row['groups']))

    def test_pending_boundary_wrong_tree_or_changed_source_is_rejected_offline(self):
        dataset_row = next(r for r in self.original_plan['repositories'] if r['repo_type'] == 'dataset')
        pending_paths = original._batches(dataset_row, {i['path'] for i in dataset_row['files']})[1]
        journal = deepcopy(self.original_journal)
        state = journal['repositories']['dataset']
        state['pending'] = dict(parent_commit=state['batches'][-1]['commit'], paths=pending_paths)
        self.journal_path.write_bytes(original._encoded(journal)); self.write_boundary()
        with self.assertRaisesRegex(continuation.PublicationError, 'pending outcome'): self.stage()
        self.journal_path.write_bytes(self.saved_journal); self.write_boundary()
        value = json.loads(self.boundary.read_bytes()); value['repositories']['dataset']['tree']['unexpected.json'] = entry(b'{}')
        self.boundary.write_bytes(original._encoded(value))
        with self.assertRaisesRegex(continuation.PublicationError, 'outside'): self.stage()
        self.write_boundary()
        (self.plan_path.parent/'dataset'/'README.md').write_bytes(b'changed scientific bytes')
        with self.assertRaises(continuation.PublicationError): self.stage()
        self.assertFalse(self.output.exists())

    def test_source_or_original_journal_changed_after_stage_is_rejected_before_writes(self):
        self.stage(); count = len(self.api.commits)
        self.journal_path.write_bytes(self.saved_journal+b'\n')
        with self.assertRaisesRegex(continuation.PublicationError, 'immutable original'): self.publish()
        self.assertEqual(len(self.api.commits), count)
        self.journal_path.write_bytes(self.saved_journal)
        (self.plan_path.parent/'dataset'/'sources/extra00.py').write_bytes(b'# changed source\n')
        with self.assertRaises(continuation.PublicationError): self.publish()
        self.assertEqual(len(self.api.commits), count)

    def test_capture_boundary_is_public_read_only_and_checks_exact_heads(self):
        before = len(self.api.commits)
        value = continuation.capture_boundary(self.plan_path, self.journal_path, self.pid, self.home/'captured', api=self.api)
        self.assertTrue(value['anonymous_public_trees'])
        self.assertEqual(len(self.api.commits), before)
        self.api.external('dataset', 'unexpected.json', b'{}')
        with self.assertRaisesRegex(continuation.PublicationError, 'head moved'):
            continuation.capture_boundary(self.plan_path, self.journal_path, self.pid, self.home/'bad-capture', api=self.api)

    def test_complete_publication_checks_every_new_byte_and_preserves_old_objects_and_receipt_schema(self):
        plan = self.stage(); before = {kind: self.api.tree(kind) for kind in original.REPOSITORIES}
        old_count = len(self.api.commits)
        receipt = self.publish()
        self.assertEqual(receipt['schema'], original.SCHEMA+'/receipt')
        self.assertEqual(receipt['plan_sha256'], sha256(self.saved_plan).hexdigest())
        self.assertTrue(receipt['anonymous_readback'])
        self.assertEqual(self.plan_path.read_bytes(), self.saved_plan)
        self.assertEqual(self.journal_path.read_bytes(), self.saved_journal)
        self.assertEqual(len(self.api.commits)-old_count, sum(len(row['groups']) for row in plan['repositories']))
        for kind, tree in before.items():
            self.assertTrue(all(self.api.tree(kind)[name] == value for name, value in tree.items()))
        committed = {(c['kind'], c['head'], name) for c in self.api.commits[old_count:] for name in c['names']}
        self.assertEqual(set(self.api.reads), committed)
        self.assertEqual(receipt['continuation']['immutable_original_journal'], original._receipt(self.saved_journal))
        self.assertEqual(receipt['continuation']['implementation'], plan['implementation'])
        journal = json.loads((self.output/continuation.JOURNAL).read_bytes())
        self.assertIsNone(journal['pending'])
        self.assertTrue(all(g['anonymous_readback'] and g['commit_round_trip_s'] >= 0 and g['verification_s'] >= 0
                            for state in journal['repositories'].values() for g in state['groups']))
        # Downstream metadata/capture helpers accept the final schema and bindings.
        import publish_piper_feedback_dataset_metadata as metadata
        parent = next(row['commit'] for row in receipt['repositories'] if row['repo_type'] == 'dataset')
        metadata._completed_receipt(original._encoded(receipt), parent)
        with self.assertRaisesRegex(continuation.PublicationError, 'complete'): self.publish()

    def test_unexpected_remote_change_or_private_head_is_rejected_before_new_commits(self):
        self.stage(); before = len(self.api.commits)
        self.api.private = True
        with self.assertRaises(continuation.PublicationError): self.publish()
        self.assertEqual(len(self.api.commits), before)
        self.api.private = False
        self.api.external('dataset', 'unexpected.json', b'{}')
        with self.assertRaisesRegex(continuation.PublicationError, 'outside'): self.publish()
        self.assertEqual(len(self.api.commits), before)

    def test_partial_anonymous_readback_keeps_pending_and_resume_does_not_duplicate_commit(self):
        plan = self.stage(); first_row = next(row for row in plan['repositories'] if row['groups'])
        first_name = first_row['groups'][0]['paths'][0]
        self.api.bad_download = first_name
        before = len(self.api.commits)
        with self.assertRaisesRegex(continuation.PublicationError, 'Anonymous public bytes'): self.publish()
        progress = json.loads((self.output/continuation.JOURNAL).read_bytes())
        self.assertIsNotNone(progress['pending'])
        self.assertFalse((self.output/'publication-receipt.json').exists())
        first_commit = self.api.commits[-1]['head']
        self.api.bad_download = None
        self.publish()
        self.assertEqual(sum(c['head'] == first_commit for c in self.api.commits[before:]), 1)
        progress = json.loads((self.output/continuation.JOURNAL).read_bytes())
        group = progress['repositories'][first_row['repo_type']]['groups'][0]
        self.assertTrue(group['reconciled']); self.assertIsNone(group['commit_round_trip_s'])

    def test_ambiguous_commit_after_landing_is_reconciled_exactly_without_repeat(self):
        self.stage(); before = len(self.api.commits); self.api.ambiguous_after = True
        with self.assertRaisesRegex(continuation.PublicationError, 'uncertain'): self.publish()
        landed = self.api.commits[-1]['head']
        self.publish()
        self.assertEqual(sum(c['head'] == landed for c in self.api.commits[before:]), 1)

    def test_ambiguous_request_that_did_not_land_retries_only_after_exact_parent_tree_reconciliation(self):
        self.stage(); before = len(self.api.commits); self.api.ambiguous_before = True
        with self.assertRaisesRegex(continuation.PublicationError, 'uncertain'): self.publish()
        self.assertEqual(len(self.api.commits), before)
        self.publish()
        self.assertGreater(len(self.api.commits), before)

    def test_foreign_commit_above_ambiguous_group_cannot_be_adopted(self):
        self.stage(); self.api.ambiguous_after = True
        with self.assertRaises(continuation.PublicationError): self.publish()
        kind = self.api.commits[-1]['kind']; self.api.external(kind, 'foreign.json', b'{}')
        before = len(self.api.commits)
        with self.assertRaisesRegex(continuation.PublicationError, 'immediate continuation'): self.publish()
        self.assertEqual(len(self.api.commits), before)

    def test_mutation_during_commit_is_detected(self):
        self.stage(); self.api.mutate_after = True
        with self.assertRaisesRegex(continuation.PublicationError, 'outside'): self.publish()
        self.assertIsNotNone(json.loads((self.output/continuation.JOURNAL).read_bytes())['pending'])
        self.assertFalse((self.output/'publication-receipt.json').exists())

    def test_wrong_space_representation_is_detected_before_receipt(self):
        self.stage(); self.api.wrong_space_mode = True
        with self.assertRaisesRegex(continuation.PublicationError, 'Space representation'): self.publish()
        progress = json.loads((self.output/continuation.JOURNAL).read_bytes())
        self.assertEqual(progress['pending']['repo_type'], 'space')
        self.assertFalse((self.output/'publication-receipt.json').exists())

    def test_existing_stage_output_and_concurrent_writer_are_refused(self):
        self.stage()
        with self.assertRaisesRegex(continuation.PublicationError, 'fresh'): self.stage()
        with continuation._writer_lock(self.output), self.assertRaisesRegex(continuation.PublicationError, 'exclusive local lock'):
            self.publish()

    def test_original_journal_changed_during_final_readback_prevents_completion_receipt(self):
        self.stage()
        download = self.api.download

        def changed_original(repo_id, name, kind, revision):
            raw = download(repo_id, name, kind, revision)
            if kind == 'space':
                self.journal_path.write_bytes(self.saved_journal+b'\n')
            return raw
        with self.assertRaisesRegex(continuation.PublicationError, 'review documents changed'):
            continuation.publish(self.output/continuation.PLAN,
                api=self.api, downloader=changed_original, operation_cls=Add)
        self.assertFalse((self.output/'publication-receipt.json').exists())

    def test_runtime_source_mutation_after_stage_is_rejected_before_credentials_or_network_writes(self):
        runtime = self.home/'publisher-runtime'; runtime.mkdir()
        for name in continuation.IMPLEMENTATION_FILES:
            shutil.copyfile(continuation.IMPLEMENTATION_ROOT/name, runtime/name)
        with patch.object(continuation, 'IMPLEMENTATION_ROOT', runtime):
            plan = self.stage()
            bound = plan['implementation']['files']
            self.assertEqual(set(bound), set(continuation.IMPLEMENTATION_FILES))
            self.assertEqual((self.output/'implementation'/'publish_piper_feedback_continuation.py').read_bytes(),
                             (runtime/'publish_piper_feedback_continuation.py').read_bytes())
            changed = runtime/'publish_piper_feedback_continuation.py'
            changed.write_bytes(changed.read_bytes()+b'\n# changed after review\n')
            before = len(self.api.commits)
            with patch.object(continuation, 'client', side_effect=AssertionError('credentials must not be accessed')), \
                    self.assertRaisesRegex(continuation.PublicationError, 'runtime implementation changed'):
                continuation.publish(self.output/continuation.PLAN)
            self.assertEqual(len(self.api.commits), before)


class ExitedWriterTests(unittest.TestCase):
    def test_physical_groups_preserve_logical_membership_at_both_caps(self):
        for size, count, expected_max in [(8*1024*1024, 17, 8), (1, 226, 100)]:
            row = dict(files=[dict(path=f'sources/item{i:04}.py', bytes=size, sha256='a'*64) for i in range(count)])
            state = dict(created_sha='a'*40, baseline_sha=None,
                baseline_tree={'.gitattributes': entry(b'original attributes')}, batches=[], pending=None)
            groups = continuation._groups(row, state)
            self.assertEqual([n for g in groups for n in g['paths']], sorted(item['path'] for item in row['files']))
            self.assertEqual(max(len(g['paths']) for g in groups), expected_max)
            self.assertTrue(all(g['bytes'] <= continuation.MAX_BYTES for g in groups))

    def test_alive_pid_and_ambiguous_permission_fail_closed(self):
        with self.assertRaisesRegex(continuation.PublicationError, 'still alive'): continuation._dead(os.getpid())
        with patch.object(os, 'kill', side_effect=PermissionError()), self.assertRaisesRegex(continuation.PublicationError, 'cannot be established'):
            continuation._dead(12345)
        with patch.object(os, 'kill', side_effect=ProcessLookupError()): continuation._dead(12345)


class WheelMetadataEvidenceTests(unittest.TestCase):
    def fixture(self):
        before = b'*.zip filter=lfs diff=lfs merge=lfs -text\n'
        after = before+continuation.WHEEL_RULE
        wheel = dict(path=continuation.WHEEL, **original.CONTRACT_ARCHIVES[continuation.WHEEL])
        plan = dict(repositories=[dict(repo_type=k, files=[wheel]) for k in ('model', 'dataset')])
        commits = ['1'*40, continuation.MODEL_WHEEL_COMMIT, '3'*40, '4'*40]
        parent = 'a'*40
        batches = []
        for index, commit in enumerate(commits):
            batches.append(dict(commit=commit, parent_commit=parent,
                paths=[continuation.WHEEL] if index == 1 else [f'sources/item{index}.py']))
            parent = commit
        model = dict(created_sha='a'*40, baseline_sha=None, baseline_tree={continuation.ATTRIBUTES: entry(before)}, batches=batches)
        dataset = dict(created_sha='b'*40, baseline_sha=None, baseline_tree={continuation.ATTRIBUTES: entry(before)}, batches=[])
        journal = dict(repositories=dict(model=model, dataset=dataset))
        plan_raw, journal_raw = original._encoded(plan), original._encoded(journal)
        value = dict(schema=continuation.SCHEMA+'/wheel-attributes-evidence',
            original_plan=original._receipt(plan_raw), original_journal=original._receipt(journal_raw),
            model=dict(repo_id=original.MODEL_REPO, path=continuation.ATTRIBUTES, wheel=wheel,
                added_line_utf8=continuation.WHEEL_RULE.decode(), anonymous_readbacks=True,
                before=continuation._attribute_readback('a'*40, before),
                after=continuation._attribute_readback('4'*40, after),
                provenance=[dict(index=i, commit=b['commit'], parent_commit=b['parent_commit'],
                    attributes_entry=entry(before if i == 0 else after)) for i,b in enumerate(batches)]),
            dataset_before=continuation._attribute_readback('b'*40, before))
        return plan, journal, plan_raw, journal_raw, value

    def test_exact_historical_model_append_and_dataset_before_are_verified_offline(self):
        plan, journal, plan_raw, journal_raw, value = self.fixture()
        self.assertEqual(continuation._metadata_evidence(value, plan, journal, plan_raw, journal_raw), value)
        self.assertEqual(len(continuation.WHEEL_RULE), 85)

    def test_changed_line_extra_bytes_wrong_manifest_or_wrong_commit_provenance_fail(self):
        plan, journal, plan_raw, journal_raw, value = self.fixture()
        for raw in [value['model']['before']['utf8'].encode()+b'*.whl filter=lfs diff=lfs merge=lfs -text\n',
                    value['model']['after']['utf8'].encode()+b'# extra edit\n']:
            changed = deepcopy(value)
            changed['model']['after'] = continuation._attribute_readback('4'*40, raw)
            with self.assertRaisesRegex(continuation.PublicationError, 'single appended'):
                continuation._metadata_evidence(changed, plan, journal, plan_raw, journal_raw)
        changed = deepcopy(value); changed['model']['wheel']['sha256'] = 'f'*64
        with self.assertRaises(continuation.PublicationError):
            continuation._metadata_evidence(changed, plan, journal, plan_raw, journal_raw)
        changed = deepcopy(value); changed['model']['provenance'][0]['attributes_entry'] = changed['model']['after']['entry']
        with self.assertRaisesRegex(continuation.PublicationError, 'provenance'):
            continuation._metadata_evidence(changed, plan, journal, plan_raw, journal_raw)

    def test_dataset_transition_reserves_one_path_and_full_after_bytes_in_the_wheel_group(self):
        _, _, _, _, evidence = self.fixture()
        wheel = evidence['model']['wheel']
        state = dict(created_sha='b'*40, baseline_sha=None, baseline_tree={'.gitattributes': entry(b'x\n')}, batches=[], pending=None)
        row = dict(repo_type='dataset', files=[dict(path=f'sources/a{i:04}.py', bytes=1, sha256='0'*64) for i in range(200)]+[wheel])
        reserve = len(evidence['dataset_before']['utf8'].encode()+continuation.WHEEL_RULE)
        groups = continuation._groups(row, state, reserve)
        transition = continuation._dataset_transition([dict(repo_type='dataset', groups=groups)], evidence)
        self.assertEqual(sum(g['metadata_reserved_bytes'] > 0 for g in groups), 1)
        self.assertEqual(groups[transition['group_index']]['metadata_reserved_bytes'], reserve)
        self.assertTrue(all(len(g['paths'])+int(g['metadata_reserved_bytes'] > 0) <= 100 for g in groups))
        self.assertTrue(all(g['bytes'] <= continuation.MAX_BYTES for g in groups))
        self.assertEqual(transition['after']['utf8'], evidence['dataset_before']['utf8']+continuation.WHEEL_RULE.decode())


class WheelMetadataRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.source = self.home/'source'; self.output = self.home/'continuation'; self.output.mkdir()
        self.api = FakeHub(); self.api.auto_dataset_wheel_rule = True
        self.raw = b'exact finite wheel transport fixture bytes'
        path = self.source/'dataset'/continuation.WHEEL; path.parent.mkdir(parents=True); path.write_bytes(self.raw)
        before = self.api.repos['dataset']['files'][continuation.ATTRIBUTES]
        after = before+continuation.WHEEL_RULE
        self.group = dict(index=0, logical_batches=[dict(index=120, paths=[continuation.WHEEL])],
            paths=[continuation.WHEEL], bytes=len(self.raw)+len(after), metadata_reserved_bytes=len(after))
        self.row = dict(repo_id=original.DATASET_REPO, repo_type='dataset', boundary_head=self.api.repos['dataset']['head'],
            original_manifest=[dict(path=continuation.WHEEL, **original._receipt(self.raw))], groups=[self.group])
        self.transition = dict(repo_id=original.DATASET_REPO, repo_type='dataset', path=continuation.ATTRIBUTES,
            group_index=0, wheel=self.row['original_manifest'][0], added_line_utf8=continuation.WHEEL_RULE.decode(),
            before=continuation._attribute_readback(self.row['boundary_head'], before),
            after=dict(entry=entry(after), receipt=original._receipt(after), utf8=after.decode()))
        self.rows = [self.row if kind == 'dataset' else dict(repo_id=repo, repo_type=kind,
            boundary_head=self.api.repos[kind]['head'], original_manifest=[], groups=[])
            for kind,repo in original.REPOSITORIES.items()]
        self.plan = dict(repositories=self.rows, dataset_metadata_transition=self.transition)
        self.plan_sha = 'd'*64
        self.boundary = dict(repositories={kind: dict(head=self.api.repos[kind]['head'], tree=self.api.tree(kind))
                                          for kind in original.REPOSITORIES})
        self.old_journal = dict(repositories={kind: dict(lfs_suffixes=list(original.LFS_SUFFIXES)) for kind in original.REPOSITORIES})
        self.journal_path = self.output/continuation.JOURNAL
        self.progress,self.expected = continuation._load_progress(self.journal_path,self.plan,self.plan_sha,self.boundary,self.source,self.old_journal)
        self.progress['pending'] = dict(repo_type='dataset', group_index=0, parent_commit=self.row['boundary_head'],
            paths=[continuation.WHEEL], metadata_transition=self.transition)
        original._atomic(self.journal_path,self.progress); self.api.pending_path=self.journal_path

    def commit(self):
        return self.api.create_commit(original.DATASET_REPO,repo_type='dataset',
            operations=[Add(continuation.WHEEL,self.raw)],parent_commit=self.row['boundary_head'],
            commit_message=continuation._message(self.plan_sha,'dataset',self.group))

    def finish(self, revision, *, transition=True):
        return continuation._finish_group(self.api,self.api.download,self.row,self.group,{continuation.WHEEL:self.raw},
            revision,self.expected['dataset'],self.progress,self.journal_path,self.old_journal['repositories']['dataset'],
            reconciled=False,duration=.1,transition=self.transition if transition else None)

    def test_correct_automatic_dataset_delta_is_read_back_and_sealed_in_journal(self):
        result=self.commit(); self.finish(result.oid)
        self.assertIn(('dataset',result.oid,continuation.ATTRIBUTES),self.api.reads)
        self.assertIsNone(self.progress['pending'])
        self.assertEqual(self.progress['repositories']['dataset']['groups'][0]['entries'][continuation.ATTRIBUTES],self.transition['after']['entry'])
        continuation._load_progress(self.journal_path,self.plan,self.plan_sha,self.boundary,self.source,self.old_journal)

    def test_wrong_line_extra_line_and_missing_or_early_delta_fail_without_acknowledgment(self):
        for rule in [b'*.whl filter=lfs diff=lfs merge=lfs -text\n', continuation.WHEEL_RULE+b'# extra edit\n', b'']:
            with self.subTest(rule=rule):
                repo=self.api.repos['dataset']; repo['head']=self.row['boundary_head']
                repo['files']=dict(repo['history'][repo['head']]);self.api.dataset_rule_override=rule
                result=self.commit()
                with self.assertRaisesRegex(continuation.PublicationError,'outside'): self.finish(result.oid)
                self.assertIsNotNone(self.progress['pending'])
        repo=self.api.repos['dataset'];repo['head']=self.row['boundary_head'];repo['files']=dict(repo['history'][repo['head']])
        self.api.dataset_rule_override=None
        result=self.commit()
        with self.assertRaisesRegex(continuation.PublicationError,'outside'): self.finish(result.oid,transition=False)

    def test_unknown_commit_reply_reconciles_the_exact_dataset_metadata_delta_without_recommit(self):
        self.api.ambiguous_after=True
        with self.assertRaises(RuntimeError):self.commit()
        before=len(self.api.commits)
        progress,expected=continuation._load_progress(self.journal_path,self.plan,self.plan_sha,self.boundary,self.source,self.old_journal)
        continuation._reconcile(self.api,self.api.download,self.plan,self.plan_sha,progress,expected,
            self.source,self.old_journal,self.journal_path)
        self.assertEqual(len(self.api.commits),before)
        self.assertIsNone(progress['pending'])
        self.assertTrue(progress['repositories']['dataset']['groups'][0]['reconciled'])
        self.assertIn(continuation.ATTRIBUTES,progress['repositories']['dataset']['groups'][0]['entries'])


if __name__ == '__main__':
    unittest.main()
