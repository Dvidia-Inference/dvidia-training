"""Publication boundaries preserve historical evidence and reject stale releases."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import publish_prerequisites as publisher


class PrerequisitesPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.parent, self.export = self.root / 'parent', self.root / 'export'
        self.parent.mkdir()
        self.export.mkdir()
        self.original = {'index.html': b'<main><section aria-labelledby="replay-title">old evidence</section></main>',
            'README.md': b'---\nsdk: static\n---\nOld release\n',
            'provenance.json': b'{"old_evidence":{"preserve":true}}\n',
            '.gitattributes': b'*.bin filter=lfs diff=lfs merge=lfs -text\n'}
        self.tree = {name: {'blob': publisher.git_blob(raw), 'size': len(raw), 'lfs_sha256': None}
                     for name, raw in self.original.items()}
        self.tree['workstation/v01/geometry.bin'] = {'blob': 'd' * 40, 'size': 2000000, 'lfs_sha256': 'e' * 64}
        for name, raw in self.original.items():
            (self.parent / name).write_bytes(raw)
        self.save_snapshot()
        case = {'id': 'hinge-one', 'task': 'hinged-box', 'policy': 'baseline', 'case': {'id': 'one'}}
        measured = {**case, 'success': False, 'valid': True}
        protocol = publisher.encoded({'schema': 'dvidia.skill-prerequisites/v1', 'source_commit': 'b' * 40,
            'split': 'heldout', 'attempts': [case]})
        benchmark = publisher.encoded({'schema': 'dvidia.skill-prerequisites/v1',
            'source': {'commit': 'b' * 40, 'protocol_sha256': publisher.record(protocol)['sha256']},
            'split': 'heldout', 'learned_policy': False, 'hardware_validated': False, 'attempts': [measured]})
        self.release = {'index.html': b'<h1>Recorded native attempts</h1>',
            'benchmark.json': benchmark, 'protocol.json': protocol, 'trials/hinge-one.json': publisher.encoded(measured),
            'motion.bin.gz': gzip.compress(b'native motion', mtime=0), 'LICENSE': b'MIT test fixture'}
        self.save_release()

    def save_snapshot(self):
        (self.parent / 'snapshot.json').write_bytes(publisher.encoded({'schema': 'dvidia.prerequisites-parent/v1',
            'repo_id': publisher.REPO_ID, 'parent_commit': 'a' * 40, 'tree': self.tree}))

    def save_release(self):
        for name, raw in self.release.items():
            (self.export / name).parent.mkdir(parents=True, exist_ok=True)
            (self.export / name).write_bytes(raw)
        (self.export / 'integrity.json').write_bytes(publisher.encoded({'files': {
            name: publisher.record(raw) for name, raw in self.release.items()}}))

    def stage(self):
        return publisher.stage(self.export, self.parent, self.root / 'staged', 'b' * 40)

    def test_delta_retains_history_and_appends_honest_root_pointers(self):
        result = self.stage()
        plan, payload = publisher.check(result['plan'])
        self.assertEqual(plan['parent_tree'], self.tree)
        self.assertEqual(set(payload) & set(self.tree), publisher.ROOT_FILES)
        self.assertEqual(len(payload), len(self.release) + 4)
        self.assertIn(self.original['README.md'], payload['README.md'])
        self.assertIn(b'Recorded native MuJoCo', payload['index.html'])
        self.assertIn(b'rake is mounted', payload['README.md'])
        provenance = json.loads(payload['provenance.json'])
        self.assertEqual(provenance['old_evidence'], {'preserve': True})
        self.assertFalse(provenance['prerequisites']['learned_policy'])

    def test_payload_tampering_after_stage_is_rejected(self):
        result = self.stage()
        (self.root / 'staged/payload/prerequisites/v01/benchmark.json').write_bytes(b'{"success":true}')
        with self.assertRaisesRegex(publisher.PublicationError, 'Payload changed'):
            publisher.check(result['plan'])

    def test_resealed_omitted_failure_or_changed_source_is_rejected(self):
        benchmark = json.loads(self.release['benchmark.json'])
        benchmark['attempts'] = []
        self.release['benchmark.json'] = publisher.encoded(benchmark)
        self.save_release()
        with self.assertRaisesRegex(publisher.PublicationError, 'Every planned attempt'):
            self.stage()
        benchmark['source']['commit'] = 'c' * 40
        self.release['benchmark.json'] = publisher.encoded(benchmark)
        self.save_release()
        with self.assertRaisesRegex(publisher.PublicationError, 'Source, protocol'):
            self.stage()

    def test_only_dense_native_trace_can_be_omitted_from_summary_index(self):
        receipt = json.loads(self.release['trials/hinge-one.json'])
        receipt['native_trace'] = [[0., .1], [.002, .2]]
        self.release['trials/hinge-one.json'] = publisher.encoded(receipt)
        self.save_release()
        publisher.check_export(self.export, 'b' * 40)
        receipt['hidden_events'] = ['unreported failure']
        self.release['trials/hinge-one.json'] = publisher.encoded(receipt)
        self.save_release()
        with self.assertRaisesRegex(publisher.PublicationError, 'Complete attempt receipts'):
            publisher.check_export(self.export, 'b' * 40)

    def test_unsealed_added_file_is_rejected(self):
        (self.export / 'extra.json').write_bytes(b'{}')
        with self.assertRaisesRegex(publisher.PublicationError, 'integrity seal'):
            self.stage()

    def test_symlink_file_and_symlink_directory_are_rejected(self):
        (self.export / 'shortcut.json').symlink_to(self.export / 'benchmark.json')
        with self.assertRaisesRegex(publisher.PublicationError, 'symlinks'):
            self.stage()
        (self.export / 'shortcut.json').unlink()
        (self.export / 'nested').symlink_to(self.parent, target_is_directory=True)
        with self.assertRaisesRegex(publisher.PublicationError, 'symlinks'):
            self.stage()

    def test_private_path_in_escaped_compressed_json_is_rejected(self):
        raw = gzip.compress(b'{"path":"\\u002fUsers/roguedev/private"}', mtime=0)
        with self.assertRaisesRegex(publisher.PublicationError, 'private path'):
            publisher.inspect_bytes('attempts/one.json.gz', raw)

    def test_bounded_decompression_and_trailing_members(self):
        with patch.object(publisher, 'MAX_DECODED', 20):
            with self.assertRaisesRegex(publisher.PublicationError, 'exceeds its bound'):
                publisher.inspect_bytes('motion.bin.gz', gzip.compress(b'x' * 21))
        with self.assertRaisesRegex(publisher.PublicationError, 'trailing data'):
            publisher.inspect_bytes('motion.bin.gz', gzip.compress(b'x') + gzip.compress(b'y'))

    def test_invalid_paths_and_nonfinite_json(self):
        for name in ('../secret.json', '/file.json', 'a/../b.json', 'a//b.json', '.secret', 'a\\b.json',
                     'a\nb.json', 'secret.py', '%2e%2e/secret.json', 'index.html?x.json'):
            with self.subTest(name=name), self.assertRaises(publisher.PublicationError):
                publisher.path_name(name)
        with self.assertRaisesRegex(publisher.PublicationError, 'finite JSON'):
            publisher.inspect_bytes('benchmark.json', b'{"value":NaN}')

    def test_parent_root_tampering_and_existing_namespace_are_rejected(self):
        (self.parent / 'README.md').write_bytes(b'changed')
        with self.assertRaisesRegex(publisher.PublicationError, 'root bytes changed'):
            self.stage()
        (self.parent / 'README.md').write_bytes(self.original['README.md'])
        self.tree['prerequisites/v01/old.json'] = {'blob': 'c' * 40, 'size': 2, 'lfs_sha256': None}
        self.save_snapshot()
        with self.assertRaisesRegex(publisher.PublicationError, 'must be new'):
            self.stage()

    def test_gate_binds_source_benchmark_integrity_and_all_checks(self):
        plan, _ = publisher.check(self.stage()['plan'])
        gate = {'source_commit': plan['source_commit'], 'tests_passed': True, 'browser_qa_passed': True,
            'benchmark_reviewed': True,
            'benchmark_sha256': plan['files']['prerequisites/v01/benchmark.json']['sha256'],
            'integrity_sha256': plan['files']['prerequisites/v01/integrity.json']['sha256']}
        publisher.check_gate(gate, plan)
        for name, value in [('source_commit', 'c' * 40), ('browser_qa_passed', False), ('tests_passed', 1),
                            ('benchmark_sha256', '0' * 64), ('integrity_sha256', '0' * 64)]:
            with self.subTest(name=name), self.assertRaisesRegex(publisher.PublicationError, 'source-bound'):
                publisher.check_gate({**gate, name: value}, plan)

    def test_remote_tree_keeps_all_historical_git_and_lfs_entries(self):
        payload = {'prerequisites/v01/new.bin.gz': b'new'}
        old_attributes = self.original['.gitattributes']
        attributes = old_attributes + b'prerequisites/v01/new.bin.gz filter=lfs diff=lfs merge=lfs -text\n'
        after = {**self.tree, 'prerequisites/v01/new.bin.gz': {'blob': 'f' * 40, 'size': 3, 'lfs_sha256': None},
            '.gitattributes': {'blob': publisher.git_blob(attributes), 'size': len(attributes), 'lfs_sha256': None}}
        self.assertEqual(publisher.verify_tree(self.tree, after, payload, old_attributes, attributes), len(self.tree) - 1)
        changed = {**after, 'workstation/v01/geometry.bin': {**after['workstation/v01/geometry.bin'], 'lfs_sha256': '0' * 64}}
        with self.assertRaisesRegex(publisher.PublicationError, 'Historical Git/LFS'):
            publisher.verify_tree(self.tree, changed, payload, old_attributes, attributes)
        with self.assertRaisesRegex(publisher.PublicationError, 'unexpected paths'):
            publisher.verify_tree(self.tree, {**after, 'extra.json': {}}, payload, old_attributes, attributes)

    def test_hub_metadata_cannot_replace_prior_rules_or_add_unrelated_rule(self):
        payload = {'prerequisites/v01/one.png': b'image'}
        original = self.original['.gitattributes']
        for attributes in (b'replaced', original + b'old.png filter=lfs diff=lfs merge=lfs -text\n'):
            after = {**self.tree, 'prerequisites/v01/one.png': {},
                '.gitattributes': {'blob': publisher.git_blob(attributes), 'size': len(attributes), 'lfs_sha256': None}}
            with self.assertRaises(publisher.PublicationError):
                publisher.verify_tree(self.tree, after, payload, original, attributes)

    def prepare_v02(self):
        first = publisher.stage(self.export, self.parent, self.root / 'v01-release', 'b' * 40)
        _, prior_payload = publisher.check(first['plan'])
        self.tree.update({name: {'blob': publisher.git_blob(raw), 'size': len(raw), 'lfs_sha256': None}
                          for name, raw in prior_payload.items()})
        for name in publisher.ROOT_FILES:
            self.original[name] = prior_payload[name]
            (self.parent / name).write_bytes(prior_payload[name])
        self.save_snapshot()
        snapshot = json.loads((self.parent / 'snapshot.json').read_bytes())
        snapshot['namespace'] = 'prerequisites/v02'
        (self.parent / 'snapshot.json').write_bytes(publisher.encoded(snapshot))
        protocol = json.loads(self.release['protocol.json'])
        protocol['schema'] = 'dvidia.continuous-shoe-grip/v2'
        self.release['protocol.json'] = publisher.encoded(protocol)
        benchmark = json.loads(self.release['benchmark.json'])
        benchmark['schema'] = protocol['schema']
        benchmark['source']['protocol_sha256'] = publisher.record(self.release['protocol.json'])['sha256']
        measured = benchmark['attempts'][0]
        measured['native_trace_columns'] = ['time_s', 'normal_force_n']
        measured['independent_score'] = {'schema': 'dvidia.continuous-shoe-grip-score/v2',
            'task_success': False, 'valid': True, 'compiled_model_sha256': 'c' * 64}
        self.release['benchmark.json'] = publisher.encoded(benchmark)
        self.release['trials/hinge-one.json'] = publisher.encoded({**measured, 'native_trace': [[0., .1], [.002, .2]]})
        self.save_release()
        return prior_payload

    def test_v02_appends_new_version_without_changing_v01_provenance_or_files(self):
        old = self.prepare_v02()
        result = publisher.stage(self.export, self.parent, self.root / 'staged', 'b' * 40, 'prerequisites/v02')
        plan, payload = publisher.check(result['plan'])
        self.assertEqual(plan['namespace'], 'prerequisites/v02')
        self.assertEqual(set(payload) & set(self.tree), publisher.ROOT_FILES)
        before = json.loads(old['provenance.json'])
        after = json.loads(payload['provenance.json'])
        self.assertEqual(after['prerequisites'], before['prerequisites'])
        self.assertEqual(after['continuous_shoe_grip']['path'], 'prerequisites/v02')
        self.assertIn(old['README.md'], payload['README.md'])
        self.assertIn(b'prerequisites/v01/index.html', payload['index.html'])
        self.assertIn(b'prerequisites/v02/index.html', payload['index.html'])
        gate = {'source_commit': 'b' * 40, 'tests_passed': True, 'browser_qa_passed': True,
            'benchmark_reviewed': True,
            'benchmark_sha256': plan['files']['prerequisites/v02/benchmark.json']['sha256'],
            'integrity_sha256': plan['files']['prerequisites/v02/integrity.json']['sha256']}
        publisher.check_gate(gate, plan)

    def test_v02_requires_its_schema_and_matching_parent_target(self):
        self.prepare_v02()
        with self.assertRaisesRegex(publisher.PublicationError, 'Unexpected parent snapshot'):
            self.stage()
        with self.assertRaisesRegex(publisher.PublicationError, 'Source, protocol'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v01')
        with self.assertRaisesRegex(publisher.PublicationError, 'Unsupported explicit'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v99')

    def test_v02_rejects_missing_independent_score_or_trace_on_valid_attempt(self):
        self.prepare_v02()
        receipt = json.loads(self.release['trials/hinge-one.json'])
        for trace in (None, [], [[0., .1, .2]]):
            with self.subTest(trace=trace):
                self.release['trials/hinge-one.json'] = publisher.encoded({**receipt, 'native_trace': trace})
                self.save_release()
                with self.assertRaisesRegex(publisher.PublicationError, 'independent score'):
                    publisher.check_export(self.export, 'b' * 40, 'prerequisites/v02')
        benchmark = json.loads(self.release['benchmark.json'])
        del benchmark['attempts'][0]['independent_score']
        del receipt['independent_score']
        self.release['benchmark.json'] = publisher.encoded(benchmark)
        self.release['trials/hinge-one.json'] = publisher.encoded(receipt)
        self.save_release()
        with self.assertRaisesRegex(publisher.PublicationError, 'independent score'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v02')

    def test_v02_cannot_reseal_a_replacement_of_a_v01_artifact(self):
        self.prepare_v02()
        result = publisher.stage(self.export, self.parent, self.root / 'staged', 'b' * 40, 'prerequisites/v02')
        plan_path = Path(result['plan'])
        plan = json.loads(plan_path.read_bytes())
        replacement = plan_path.parent / 'payload/prerequisites/v01/benchmark.json'
        replacement.parent.mkdir(parents=True)
        replacement.write_bytes(b'{}')
        plan['files']['prerequisites/v01/benchmark.json'] = publisher.record(b'{}')
        plan_path.write_bytes(publisher.encoded(plan))
        with self.assertRaisesRegex(publisher.PublicationError, 'only append'):
            publisher.check(plan_path)

    def test_v02_cannot_publish_success_when_its_independent_score_fails(self):
        self.prepare_v02()
        benchmark = json.loads(self.release['benchmark.json'])
        receipt = json.loads(self.release['trials/hinge-one.json'])
        benchmark['attempts'][0]['success'] = receipt['success'] = True
        self.release['benchmark.json'] = publisher.encoded(benchmark)
        self.release['trials/hinge-one.json'] = publisher.encoded(receipt)
        self.save_release()
        with self.assertRaisesRegex(publisher.PublicationError, 'independent score'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v02')

    def test_v02_refuses_an_existing_version_even_when_export_is_valid(self):
        self.prepare_v02()
        snapshot_path = self.parent / 'snapshot.json'
        snapshot = json.loads(snapshot_path.read_bytes())
        snapshot['tree']['prerequisites/v02/old.json'] = {'blob': 'c' * 40, 'size': 2, 'lfs_sha256': None}
        snapshot_path.write_bytes(publisher.encoded(snapshot))
        with self.assertRaisesRegex(publisher.PublicationError, 'must be new'):
            publisher.stage(self.export, self.parent, self.root / 'staged', 'b' * 40, 'prerequisites/v02')

    def prepare_v03(self):
        self.prepare_v02()
        release = publisher.stage(self.export, self.parent, self.root / 'v02-release', 'b' * 40, 'prerequisites/v02')
        _, prior_payload = publisher.check(release['plan'])
        self.tree.update({name: {'blob': publisher.git_blob(raw), 'size': len(raw), 'lfs_sha256': None}
                          for name, raw in prior_payload.items()})
        for name in publisher.ROOT_FILES:
            self.original[name] = prior_payload[name]
            (self.parent / name).write_bytes(prior_payload[name])
        self.save_snapshot()
        snapshot = json.loads((self.parent / 'snapshot.json').read_bytes())
        snapshot['namespace'] = 'prerequisites/v03'
        (self.parent / 'snapshot.json').write_bytes(publisher.encoded(snapshot))
        protocol = json.loads(self.release['protocol.json'])
        protocol['schema'] = 'dvidia.graspability/v3'
        expected = protocol['attempts'][0]
        expected.update(task='in-reach-graspability', policy='shared-v03',
            case={'id': 'one', 'object_shape': 'easy-block', 'tool_condition': 'mounted'})
        self.release['protocol.json'] = publisher.encoded(protocol)
        measured = {**expected, 'valid': True, 'success': False, 'robot_actuator_sha256': 'e' * 64,
            'metrics': {'native_steps': 450, 'simulation_seconds': .9},
            'events': [{'type': 'settled_reference', 't': .4}],
            'native_trace_columns': ['t_s', 'tool_object_force_n', 'object_robot_force_n',
                'left_finger_force_n', 'right_finger_force_n', 'object_x_m', 'object_y_m',
                'object_z_m', 'world_object_force_n']}
        trace = [[i * .002, 0., 0., 0., 0., .4, -.1, .001, .3] for i in range(451)]
        measured['independent_score'] = {**publisher.graspability_scorer()({**measured, 'native_trace': trace}),
            'compiled_model_sha256': 'd' * 64}
        benchmark = json.loads(self.release['benchmark.json'])
        benchmark.update(schema=protocol['schema'], attempts=[measured])
        benchmark['source']['protocol_sha256'] = publisher.record(self.release['protocol.json'])['sha256']
        self.release['benchmark.json'] = publisher.encoded(benchmark)
        self.release['trials/hinge-one.json'] = publisher.encoded({**measured, 'native_trace': trace})
        self.save_release()
        return prior_payload

    def update_v03_row(self, change):
        benchmark = json.loads(self.release['benchmark.json'])
        receipt = json.loads(self.release['trials/hinge-one.json'])
        change(benchmark['attempts'][0])
        trace = receipt['native_trace']
        self.release['benchmark.json'] = publisher.encoded(benchmark)
        self.release['trials/hinge-one.json'] = publisher.encoded({**benchmark['attempts'][0], 'native_trace': trace})
        self.save_release()

    def test_v03_is_additive_and_does_not_describe_tool_removal_or_retrieval_as_completed(self):
        old = self.prepare_v03()
        result = publisher.stage(self.export, self.parent, self.root / 'staged', 'b' * 40, 'prerequisites/v03')
        plan, payload = publisher.check(result['plan'])
        self.assertEqual(set(payload) & set(self.tree), publisher.ROOT_FILES)
        before, after = json.loads(old['provenance.json']), json.loads(payload['provenance.json'])
        for key, value in before.items():
            self.assertEqual(after[key], value)
        self.assertEqual(after['shoe_graspability']['path'], 'prerequisites/v03')
        for version in ('v01', 'v02', 'v03'):
            self.assertIn(f'prerequisites/{version}/index.html'.encode(), payload['index.html'])
        self.assertIn(old['README.md'], payload['README.md'])
        self.assertIn(b'does not demonstrate autonomous tool removal', payload['README.md'])
        self.assertEqual(plan['namespace'], 'prerequisites/v03')

    def test_v03_rejects_missing_force_columns_and_non_dense_native_trace(self):
        self.prepare_v03()
        receipt = json.loads(self.release['trials/hinge-one.json'])
        receipt['native_trace'][1][0] = .003
        self.release['trials/hinge-one.json'] = publisher.encoded(receipt)
        self.save_release()
        with self.assertRaisesRegex(publisher.PublicationError, 'every native 2 ms'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v03')
        receipt['native_trace'][1][0] = .002
        self.release['trials/hinge-one.json'] = publisher.encoded(receipt)
        self.update_v03_row(lambda r: r['native_trace_columns'].__setitem__(8, 'unmeasured_world_force'))
        with self.assertRaisesRegex(publisher.PublicationError, 'direct-lift evidence'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v03')

    def test_v03_success_cannot_omit_initial_reference_or_retention(self):
        self.prepare_v03()
        self.update_v03_row(lambda r: (r.update(success=True), r['independent_score'].update(task_success=True)))
        with self.assertRaisesRegex(publisher.PublicationError, 'initial settled reference'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v03')

    def test_v03_rejects_v02_score_and_resealed_tool_condition_change(self):
        self.prepare_v03()
        self.update_v03_row(lambda r: r['independent_score'].update(schema='dvidia.continuous-shoe-grip-score/v2'))
        with self.assertRaisesRegex(publisher.PublicationError, 'independent score'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v03')
        self.update_v03_row(lambda r: (r['independent_score'].update(schema='dvidia.graspability-score/v3'),
            r['case'].update(tool_condition='absent')))
        with self.assertRaisesRegex(publisher.PublicationError, 'differ from the frozen protocol'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v03')

    def test_v03_rejects_changed_actuator_fingerprint_across_matched_scenes(self):
        self.prepare_v03()
        protocol = json.loads(self.release['protocol.json'])
        benchmark = json.loads(self.release['benchmark.json'])
        second = json.loads(self.release['trials/hinge-one.json'])
        second['id'] = 'hinge-two'
        second['robot_actuator_sha256'] = 'f' * 64
        second['case']['tool_condition'] = 'absent'
        protocol['attempts'].append({k: second[k] for k in ('id', 'task', 'policy', 'case')})
        benchmark['attempts'].append({k: v for k, v in second.items() if k != 'native_trace'})
        self.release['protocol.json'] = publisher.encoded(protocol)
        benchmark['source']['protocol_sha256'] = publisher.record(self.release['protocol.json'])['sha256']
        self.release['benchmark.json'] = publisher.encoded(benchmark)
        self.release['trials/hinge-two.json'] = publisher.encoded(second)
        self.save_release()
        with self.assertRaisesRegex(publisher.PublicationError, 'shared robot actuator fingerprint'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v03')

    def test_v03_rejects_resealed_truncated_tail_and_force_or_reference_forgery(self):
        self.prepare_v03()
        original = self.release['trials/hinge-one.json']
        for mutation in ('tail', 'force', 'reference'):
            with self.subTest(mutation=mutation):
                receipt = json.loads(original)
                if mutation == 'tail':
                    receipt['native_trace'].pop()
                elif mutation == 'force':
                    receipt['native_trace'][-1][8] = 0.
                else:
                    receipt['native_trace'][200][7] += .002
                self.release['trials/hinge-one.json'] = publisher.encoded(receipt)
                self.save_release()
                with self.assertRaisesRegex(publisher.PublicationError, 'independent'):
                    publisher.check_export(self.export, 'b' * 40, 'prerequisites/v03')

    def test_v03_rejects_forged_success_even_with_plausible_reference_and_window(self):
        self.prepare_v03()
        self.update_v03_row(lambda r: (r.update(success=True), r['independent_score'].update(
            task_success=True, strict_retention_pass=True, strict_passing_ticks=250)))
        with self.assertRaisesRegex(publisher.PublicationError, 'independently recomputed'):
            publisher.check_export(self.export, 'b' * 40, 'prerequisites/v03')


if __name__ == '__main__':
    unittest.main()
