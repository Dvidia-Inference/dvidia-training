from copy import deepcopy
from hashlib import sha256
import ast
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import camera_action_cache as cache
from dvidia_training import camera_action_model as model
from dvidia_training import camera_action_reference_export as ref
from dvidia_training import camera_action_study as study
from tests.test_camera_action_export import fixture


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = model.canonical(document)+b'\n'
    path.write_bytes(body)
    return sha256(body).hexdigest()


def action(vector):
    return dict(joint_targets_rad=list(vector[:6]), gripper_position_m=float(vector[6]))


def native_trace(previous, target):
    """Independent oracle: execute ONLY the two original pinned reference lines."""
    source = Path(model.__file__).with_name('piper_task.py').read_text()
    tree = ast.parse(source)
    env = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PiperTaskEnv')
    step = next(n for n in env.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
    loop = next(n for n in ast.walk(step) if isinstance(n, ast.For) and n.body
        and isinstance(n.body[0], ast.AugAssign) and isinstance(n.body[0].target, ast.Attribute)
        and n.body[0].target.attr == 'command_targets')
    module = ast.Module(body=loop.body[:2], type_ignores=[])
    if sha256(ast.dump(module, include_attributes=False).encode()).hexdigest() != ref.CONTRACT['limiter_ast_sha256']:
        raise AssertionError('Native numerical oracle changed')
    code = compile(ast.fix_missing_locations(module), '<two-reference-statements-only>', 'exec')
    state = SimpleNamespace(command_targets=np.asarray(previous[:6], dtype=np.float64).copy(),
        command_gripper=float(previous[6]), config=SimpleNamespace(
            joint_target_rate_rad_s=1.5, jaw_target_rate_m_s=.05, timestep=.001))
    scope = dict(self=state, targets=np.asarray(target[:6], dtype=np.float64),
                 jaw=float(target[6]), np=np, float=float)
    rows = []
    for _ in range(20):
        exec(code, {'__builtins__': {}}, scope)
        rows.append([*state.command_targets.tolist(), state.command_gripper])
    return np.asarray(rows, dtype='<f8')


class ReferenceExportTests(unittest.TestCase):
    def setUp(self):
        ample = shutil._ntuple_diskusage(200*1024**3, 50*1024**3, 150*1024**3)
        disk = patch.object(cache.shutil, 'disk_usage', return_value=ample)
        disk.start(); self.addCleanup(disk.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.counter = 0

    def tearDown(self):
        for path in self.root.rglob('*'):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)
        self.temp.cleanup()

    def parent(self, *, mixed=False, blind=False, query=.8, mutation=None):
        """Only authored tiny metadata fixtures; no real frames/native/models.

        The existing strict export's native recording reader is replaced at the
        fixture boundary. Parent cache/export validation is otherwise unchanged.
        Reference-export construction itself receives no mocked loader.
        """
        self.counter += 1
        root = self.root/f'fixture-{self.counter}'; root.mkdir()
        parent = self.root/f'parent-{self.counter}'
        stage = 'corrections' if mixed else 'demonstrations'
        mode = 'mixed' if mixed else 'teacher'
        recordings, rows, cases = {}, [], {'train': [], 'development': [], 'confirmation': []}
        for index, group in enumerate(('train-a', 'train-b')):
            case = dict(id=group+'-goal0', group_id=group, split='train', goal_id=0,
                goal_position_m=[.35, .08, .08], object_position_m=[.3, -.03, .082], seed=17)
            cases['train'].append(case)
            attempt = case['id']+'-'+mode
            path = f'episodes/{stage}/{attempt}'; directory = root/path
            directory.parent.mkdir(parents=True, exist_ok=True)
            recording = fixture(directory, mode=mode, group=group)
            recording['schema'] = 'dvidia.camera-action-rollout/v2'
            # Two distinct high precision reset values with the SAME float32
            # representation. Neither can be recovered exactly from features.
            first = np.asarray([.10000000000000002 + index*1e-10, 1.4, -1.3, 0., 0., 0., .035])
            recording['initial_applied_targets'] = action(first)
            previous = first
            for i, step in enumerate(recording['steps']):
                issued = model.action_vector(step['command']['issued'])
                if mixed:
                    issued[0] = -.3  # executed action opposes its query label
                step['command']['issued'] = action(issued)
                step['command']['issued_at_seconds'] = i*.02
                after = deepcopy(recording['steps'][i+1]['before'] if i < 2 else step['before'])
                after['timestamp_seconds'] = (i+1)*.02
                step['after'] = after
                end = native_trace(previous, issued)[-1]
                step['applied'] = dict(interval_start_seconds=i*.02, interval_end_seconds=(i+1)*.02,
                    ramped_targets=action(end), native_position_controls=[*end[:6].tolist(), .035],
                    rate_limited=True, scope='final_native_tick_only')
                previous = end
                if mixed:
                    target = dict(joint_targets_rad=[query, 1.4, -1.3, 0., 0., 0.], gripper_position_m=.01)
                    pathq = directory/step['teacher_query']['path']
                    body = model.canonical({'action': target, 'origin': 'privileged_simulator_teacher'})+b'\n'
                    pathq.write_bytes(body)
                    step['teacher_query'].update(bytes=len(body), sha256=sha256(body).hexdigest())
            recording['initial'] = deepcopy(recording['steps'][0]['before'])
            if mutation:
                mutation(recording)
            write(directory/'recording.json', recording)
            receipt = write(directory/'rollout.json', {'status': 'complete'})
            recordings[str(directory)] = recording
            rows.append(dict(id=attempt, case=case, controller=mode, negative_control=None,
                path=path, complete=True, initial_anchor_available=True, initial_anchor_binding_valid=True,
                strict_success=not mixed, recorded_complete_intervals=3, receipt_sha256=receipt,
                integrity_sha256=sha256((directory/'integrity.json').read_bytes()).hexdigest()))
        protocol = dict(schema=study.SCHEMA, source_sha256=study.runtime_hashes(), cases=cases,
            rosters={stage: [{k: row[k] for k in ('id', 'case', 'controller', 'negative_control')} for row in rows]})
        protocol_sha = write(root/'protocol.json', protocol)
        write(root/'registration.json', dict(schema=study.SCHEMA+'/registration',
            protocol_sha256=protocol_sha, source_sha256=protocol['source_sha256']))
        stage_sha = write(root/f'stages/{stage}.json', dict(schema=study.SCHEMA+'/stage', stage=stage,
            protocol_sha256=protocol_sha, status='complete', scheduled=len(rows), rows=rows))
        write(root/f'stages/{stage}.integrity.json', dict(protocol_sha256=protocol_sha, sha256=stage_sha))
        with patch('dvidia_training.camera_action_capture.load_rollout', side_effect=lambda p: recordings[str(p)]):
            pins = cache.build_cache(root, parent, expected_protocol_sha256=protocol_sha, stage=stage, blind=blind)
        return dict(root=root, parent=parent, recordings=recordings, rows=rows, pins=pins,
                    args=dict(expected_parent_manifest_sha256=pins['manifest_sha256'],
                        expected_parent_source_sha256=pins['source_sha256'], expected_protocol_sha256=protocol_sha,
                        expected_stage=stage, expected_split='train', expected_source_sha256=ref.source_digest()))

    def build(self, fixture, output=None, **overrides):
        output = output or self.root/f'references-{self.counter}'
        args = dict(fixture['args']); args.update(overrides)
        pins = ref.build_reference_export(fixture['root'], fixture['parent'], output, **args)
        return output, pins

    def load(self, fixture, built, **overrides):
        output, pins = built
        args = dict(fixture['args'], expected_manifest_sha256=pins['manifest_sha256']); args.update(overrides)
        return ref.load_reference_export(output, fixture['parent'], **args)

    def test_exact_reference_precision_and_episode_resets(self):
        f = self.parent(); built = self.build(f); loaded = self.load(f, built)
        expected = []
        for recording in f['recordings'].values():
            expected.append(model.action_vector(recording['initial_applied_targets']))
            expected.extend(model.action_vector(s['applied']['ramped_targets']) for s in recording['steps'][:-1])
        np.testing.assert_array_equal(loaded['previous_references'], np.asarray(expected))
        self.assertNotEqual(loaded['previous_references'][0, 0], loaded['previous_references'][3, 0])
        feature_index = model.FEATURE_NAMES.index('lag0:previous_applied_joint_1_rad/3')
        x = loaded['parent_dataset']['features']
        self.assertEqual(x[0, feature_index], x[3, feature_index])
        self.assertNotEqual(float(x[0, feature_index])*3., loaded['previous_references'][0, 0])
        np.testing.assert_array_equal(loaded['feasible_targets'], ref.feasible_targets(
            loaded['previous_references'], loaded['parent_dataset']['targets']))
        for name in ('previous_references', 'feasible_targets'):
            self.assertEqual(loaded[name].dtype.str, '<f8')
            self.assertFalse(loaded[name].flags.writeable)

    def test_all_twenty_steps_and_endpoint_reissue_match_original_numeric_oracle(self):
        rng = np.random.default_rng(171709)
        limits = np.asarray(ref.CONTRACT['limits_native'])
        previous = rng.uniform(limits[:, 0], limits[:, 1], size=(24, 7))
        targets = rng.uniform(limits[:, 0], limits[:, 1], size=(24, 7))
        previous[0] = limits[:, 0]; targets[0] = limits[:, 1]
        previous[1] = limits[:, 1]; targets[1] = limits[:, 0]
        targets[2] = previous[2]
        original = np.stack([native_trace(r, a) for r, a in zip(previous, targets)])
        observed = np.stack(list(ref._reference_updates(previous, targets)), axis=1)
        np.testing.assert_array_equal(observed, original)
        endpoint = ref.feasible_targets(previous, targets)
        reissued = np.stack(list(ref._reference_updates(previous, endpoint)), axis=1)
        np.testing.assert_array_equal(reissued, original)
        delta = (endpoint-previous)/np.asarray([.03]*6+[.001])
        decoded = previous+delta*np.asarray([.03]*6+[.001])
        decoded_trace = np.stack(list(ref._reference_updates(previous, decoded)), axis=1)
        np.testing.assert_allclose(decoded_trace, original, rtol=0., atol=1e-12)

    def test_queries_change_labels_not_inputs_or_reference_authority(self):
        left = self.parent(mixed=True, query=.8); l = self.load(left, self.build(left))
        right = self.parent(mixed=True, query=-.8); r = self.load(right, self.build(right))
        np.testing.assert_array_equal(l['previous_references'], r['previous_references'])
        np.testing.assert_array_equal(l['parent_dataset']['features'], r['parent_dataset']['features'])
        self.assertFalse(np.array_equal(l['feasible_targets'], r['feasible_targets']))
        self.assertEqual(l['parent_dataset']['targets'][0, 0], .8)
        self.assertLess(l['previous_references'][1, 0], l['previous_references'][0, 0])
        self.assertGreater(l['feasible_targets'][0, 0], l['previous_references'][0, 0])

    def test_native_truth_reference_action_and_future_values_do_not_enter_export(self):
        baseline = self.parent(); a = self.load(baseline, self.build(baseline))
        def mutate(recording):
            for row in recording['steps']:
                row['native_truth'] = {'pose': [12345]*9}
                row['reference'] = {'action': [99]*7}
            recording['steps'][-1]['after']['ignored_future_truth'] = [888]*7
        changed = self.parent(mutation=mutate); b = self.load(changed, self.build(changed))
        for field in ('previous_references', 'feasible_targets'):
            np.testing.assert_array_equal(a[field], b[field])
        np.testing.assert_array_equal(a['parent_dataset']['features'], b['parent_dataset']['features'])

    def test_metadata_only_build_and_load_need_no_native_or_raw_reader(self):
        f = self.parent(mixed=True)
        with (patch('dvidia_training.camera_action_capture.load_rollout', side_effect=AssertionError('no replay')),
              patch('dvidia_training.camera_action_export.export_episode', side_effect=AssertionError('no raw export'))):
            built = self.build(f)
            shutil.rmtree(f['root'])
            loaded = self.load(f, built)
        self.assertFalse(loaded['raw_source_reverified'])

    def test_immutable_parent_pins_stage_split_feedback_and_source_required(self):
        f = self.parent()
        for kwargs in (dict(expected_parent_manifest_sha256='0'*64), dict(expected_protocol_sha256='0'*64),
                       dict(expected_parent_source_sha256='0'*64), dict(expected_source_sha256='0'*64),
                       dict(expected_stage='corrections'), dict(expected_split='development'),
                       dict(expected_split='confirmation')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.build(f, **kwargs)
        blind = self.parent(blind=True)
        with self.assertRaisesRegex(ValueError, 'full-feedback'): self.build(blind)

    def test_wrong_index_time_ancestry_and_authority_fail_even_with_resealed_parent(self):
        mutations = [lambda r: r['steps'][1].update(index=0),
            lambda r: r['steps'][0]['applied'].update(interval_start_seconds=.02),
            lambda r: r['steps'][0]['applied'].update(interval_end_seconds=.04),
            lambda r: r['steps'][0]['command'].update(issued_at_seconds=.02),
            lambda r: r['steps'][0]['applied'].update(ramped_targets=deepcopy(r['initial_applied_targets'])),
            lambda r: r['steps'][0]['applied'].update(ramped_targets=deepcopy(r['steps'][1]['applied']['ramped_targets'])),
            lambda r: r['steps'][0]['applied'].update(ramped_targets=deepcopy(r['steps'][0]['command']['issued']))]
        for mutate in mutations:
            f = self.parent(mutation=mutate)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError): self.build(f)

    def test_metadata_or_source_mutation_during_build_refuses_before_seal(self):
        for kind in ('recording', 'stage', 'source'):
            f = self.parent()
            original = ref._metadata
            calls = 0
            def mutate_after_read(root, episode):
                nonlocal calls
                recording = original(root, episode)
                calls += 1
                if calls == 1:
                    if kind == 'stage':
                        (root/'stages/demonstrations.json').write_text('{}')
                    elif kind == 'recording':
                        p = root/episode['path']/'recording.json'
                        p.write_bytes(p.read_bytes()+b' ')
                    else:
                        # Source-byte binding changed between the two checks;
                        # no shared production source file is edited by a test.
                        ref.SOURCE_PINS['piper_task.py'] = '0'*64
                return recording
            with patch.dict(ref.SOURCE_PINS), patch.object(ref, '_metadata', side_effect=mutate_after_read):
                with self.subTest(kind=kind), self.assertRaises(ValueError): self.build(f)
            self.assertFalse((self.root/f'references-{self.counter}').exists())

    def test_changed_recording_and_stage_are_rejected_before_output(self):
        f = self.parent(); p = Path(next(iter(f['recordings'])))/'recording.json'
        recording = json.loads(p.read_text()); recording['initial_applied_targets']['joint_targets_rad'][0] += 1e-8
        write(p, recording)
        with self.assertRaisesRegex(ValueError, 'metadata differs'): self.build(f)
        self.assertFalse((self.root/f'references-{self.counter}').exists())
        f = self.parent(); (f['root']/'stages/demonstrations.json').write_text('{}')
        with self.assertRaises(ValueError): self.build(f)

    def test_tampering_and_self_edited_manifest_do_not_replace_caller_authority(self):
        f = self.parent(); built = self.build(f); out, _ = built
        p = out/'previous-references.npy'; original = p.read_bytes(); p.chmod(0o644)
        for body in (original[:-1], original[:140]+bytes([original[140]^1])+original[141:]):
            p.write_bytes(body)
            with self.assertRaises(ValueError): self.load(f, built)
        p.write_bytes(original)
        value = np.load(p, allow_pickle=False); np.save(p, value[::-1], allow_pickle=False)
        manifest = json.loads((out/'manifest.json').read_text())
        manifest['arrays']['previous-references.npy']['sha256'] = sha256(p.read_bytes()).hexdigest()
        (out/'manifest.json').chmod(0o644); write(out/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'caller digest'): self.load(f, built)

    def test_rebound_bad_dtype_nonfinite_and_endpoint_are_rejected(self):
        f = self.parent(); built = self.build(f); out, _ = built
        arrays = {name: (out/name).read_bytes() for name in ('previous-references.npy', 'feasible-targets.npy')}
        original = (out/'manifest.json').read_bytes()
        for kind in ('dtype', 'nonfinite', 'endpoint', 'row_order', 'trailing'):
            manifest = json.loads(original)
            for name, body in arrays.items():
                p = out/name; p.chmod(0o644); p.write_bytes(body)
            name = 'feasible-targets.npy' if kind == 'endpoint' else 'previous-references.npy'
            p = out/name; value = np.load(p, allow_pickle=False)
            if kind == 'dtype': value = value.astype('<f4')
            elif kind == 'nonfinite': value[0, 0] = np.nan
            elif kind == 'row_order': value = value[::-1].copy()
            else: value[0, 0] += .0001
            stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
            body = stream.getvalue() + (b'extra' if kind == 'trailing' else b'')
            p.write_bytes(body); manifest['arrays'][name].update(bytes=len(body), sha256=sha256(body).hexdigest())
            (out/'manifest.json').chmod(0o644); new_hash = write(out/'manifest.json', manifest)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.load(f, built, expected_manifest_sha256=new_hash)

    def test_unknown_files_missing_seal_links_and_fresh_output(self):
        f = self.parent(); built = self.build(f); out, _ = built; out.chmod(0o755)
        with self.assertRaises(ValueError): self.build(f, output=out)
        with self.assertRaises(ValueError): self.build(f, output=f['root']/'derived')
        with self.assertRaises(ValueError): self.build(f, output=f['parent']/'derived')
        (out/'unknown').write_text('extra')
        with self.assertRaises(ValueError): self.load(f, built)
        (out/'unknown').unlink()
        path = out/'previous-references.npy'; moved = self.root/'moved.npy'; path.rename(moved)
        try:
            path.symlink_to(moved)
        except (OSError, NotImplementedError):
            moved.rename(path)
        else:
            with self.assertRaisesRegex(ValueError, 'symlink'): self.load(f, built)
            path.unlink(); moved.rename(path)
        (out/'manifest.json').chmod(0o644); (out/'manifest.json').unlink()
        with self.assertRaises(ValueError): self.load(f, built)

    def test_source_changes_low_disk_active_worker_and_interrupted_prefix_refuse(self):
        f = self.parent()
        with patch.dict(ref.SOURCE_PINS, {'piper_task.py': '0'*64}):
            with self.assertRaisesRegex(ValueError, 'frozen source'): self.build(f)
        with patch.object(cache.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(1, 1, 1)):
            with self.assertRaisesRegex(ValueError, '30 GiB'): self.build(f)
        (f['root']/'worker.lock').write_text('active')
        with self.assertRaisesRegex(ValueError, 'actively'): self.build(f)
        (f['root']/'worker.lock').unlink()
        original = cache._write
        def interrupt(path, body):
            if path.name == 'feasible-targets.npy': raise KeyboardInterrupt('synthetic interruption')
            original(path, body)
        with patch.object(cache, '_write', side_effect=interrupt), self.assertRaises(KeyboardInterrupt): self.build(f)
        out = self.root/f'references-{self.counter}'
        self.assertTrue((out/'previous-references.npy').is_file())
        self.assertFalse((out/'manifest.json').exists())
        with self.assertRaises(ValueError): self.build(f)


if __name__ == '__main__':
    unittest.main()
