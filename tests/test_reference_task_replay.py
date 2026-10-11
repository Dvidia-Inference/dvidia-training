"""Task replay evidence stays aligned to native state and the full denominator."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import reference_replay as generic
from dvidia_training import reference_sim as sim
from dvidia_training import reference_task_replay as replay
from dvidia_training import reference_task_benchmark as bench


def small_episode():
    command = {'t_seconds': 0., 'joint_targets_rad': [0.]*6, 'gripper_0_255': 0., 'phase': 'hold'}
    row = {'t_seconds': 0., 'qpos': [0.]*17+[1., 0., 0., 0.], 'qvel': [0.]*20,
           'joint_position_rad': [0.]*6, 'joint_velocity_rad_s': [0.]*6, 'linkage_position_rad': [0.]*8,
           'tcp_position_m': [0.]*3, 'tcp_quaternion_wxyz': [1., 0., 0., 0.],
           'object_position_m': [0.]*3, 'object_quaternion_wxyz': [1., 0., 0., 0.],
           'pad_center_separation_m': .08, 'contact_count': 0,
           'issued_joint_targets_rad': [0.]*6, 'issued_gripper_0_255': 0.}
    final = {**deepcopy(row), 't_seconds': .02, 'issued_joint_targets_rad': None,
             'issued_gripper_0_255': None}
    scene = {'id': 'nominal-00', 'seed': 1, 'config': {'control_dt': .02}}
    episode = {'schema': 'dvidia.reference-task-episode/v1', 'profile': replay.PROFILE,
               'id': 'nominal-00--hold', 'scene_id': scene['id'], 'seed': 1, 'control': 'hold',
               'config': scene['config'], 'valid': True, 'success': False,
               'compiled_model_sha256': 'a'*64, 'warnings': [], 'error': None,
               'trajectory': [row, final], 'actions': [command]}
    summary = {'id': episode['id'], 'control': 'hold'}
    return episode, summary, scene


def write_bundle(root, protocol, results, provenance):
    root.mkdir()
    (root/'attempts').mkdir()
    protocol = json.loads(json.dumps(protocol))
    files = {'protocol.json': generic._json_bytes(protocol), 'report.html': b'<p>Native test fixture</p>'}
    summaries = []
    for scene, attempts in zip(protocol['scenes'], results):
        for control, result in zip(bench.CONTROLS, attempts):
            identity = scene['id']+'--'+control
            result = deepcopy(result)
            result.update(id=identity, scene_id=scene['id'], control=control,
                          physical_robot_ready=False, config=scene['config'], seed=scene['seed'])
            name = 'attempts/'+identity+'.json'
            files[name] = generic._json_bytes(result)
            summaries.append({**bench._summary(result), 'result_file': name})
    report = {'schema': bench.SCHEMA, 'profile': replay.PROFILE, 'physical_robot_ready': False,
              'learned_policy': False, 'cohort': protocol['cohort'], 'protocol': protocol,
              'protocol_sha256': sha256(files['protocol.json']).hexdigest(),
              'implementation_sha256': protocol['implementation_sha256'],
              'scenes': protocol['scenes'], 'attempted': len(summaries),
              'valid_attempts': sum(x['valid'] for x in summaries), 'attempts': summaries,
              'summary': bench.aggregate(summaries, protocol['scenes']),
              'asset_provenance': provenance, 'hardware': {'mujoco': '3.15.0'}}
    files['report.json'] = generic._json_bytes(report)
    # Deliberately noncanonical whitespace: exporter must retain exact bytes.
    files['integrity.json'] = json.dumps({'schema': 'dvidia.reference-task-integrity/v1', 'files': {
        name: {'bytes': len(data), 'sha256': sha256(data).hexdigest()} for name, data in files.items()
    }}, indent=3).encode()+b'\n'
    for name, data in files.items():
        (root/name).write_bytes(data)
    return report, files


class TaskReplayContractTests(unittest.TestCase):
    def test_selection_uses_placement_success_and_failure_then_retains_scene_order(self):
        report = {'scenes': [{'id': f'nominal-{i:02}'} for i in range(4)], 'attempts': [
            {'scene_id': 'nominal-03', 'control': 'hold', 'valid': True, 'success': True},
            {'scene_id': 'nominal-00', 'control': 'placement', 'valid': False, 'success': False},
            {'scene_id': 'nominal-02', 'control': 'placement', 'valid': True, 'success': True},
            {'scene_id': 'nominal-01', 'control': 'placement', 'valid': True, 'success': False}]}
        self.assertEqual(replay._select_scenes(report), ['nominal-02', 'nominal-01', 'nominal-00'])
        self.assertEqual(replay._select_scenes(report, ['nominal-03']), ['nominal-03'])
        for selection in ([], ['unknown'], ['nominal-00']*2, ['nominal-00','nominal-01','nominal-02','nominal-03']):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                replay._select_scenes(report, selection)

    def test_complete_command_boundaries_and_measured_state_are_required(self):
        episode, summary, scene = small_episode()
        replay._validate_episode(episode, summary, scene)
        mutations = (
            lambda e: e.update(seed=True), lambda e: e.update(warnings=['native warning']),
            lambda e: e.update(compiled_model_sha256='bad'), lambda e: e['trajectory'].pop(),
            lambda e: e['actions'].append(deepcopy(e['actions'][0])),
            lambda e: e['trajectory'][1].update(t_seconds=.021),
            lambda e: e['trajectory'][0]['qpos'].__setitem__(0, .1),
            lambda e: e['trajectory'][0]['qvel'].__setitem__(0, .1),
            lambda e: e['trajectory'][0]['qpos'].__setitem__(17, 2.),
            lambda e: e['trajectory'][0].update(tcp_position_m=[float('nan'),0,0]),
            lambda e: e['trajectory'][0].update(contact_count=True),
            lambda e: e['trajectory'][1].update(issued_joint_targets_rad=[0.]*6),
            lambda e: e['actions'][0].update(t_seconds=.02),
            lambda e: e['actions'][0].update(gripper_0_255=256),
            lambda e: e['actions'][0].update(joint_targets_rad=[.1]*6),
            lambda e: e['trajectory'][0].update(pad_center_separation_m=-.1))
        for mutate in mutations:
            changed = deepcopy(episode)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                replay._validate_episode(changed, summary, scene)

    def test_terminal_display_holds_previous_command_without_mutating_raw_tape(self):
        episode, _, _ = small_episode()
        saved = deepcopy(episode)
        rows = replay._display_rows(episode['trajectory'], episode['actions'])
        self.assertFalse(rows[0]['terminal_command_carried_forward'])
        self.assertTrue(rows[-1]['terminal_command_carried_forward'])
        self.assertEqual(rows[-1]['issued_at_seconds'], 0.)
        self.assertEqual(rows[-1]['issued_joint_targets_rad'], [0.]*6)
        self.assertEqual(episode, saved)
        self.assertIsNone(episode['trajectory'][-1]['issued_joint_targets_rad'])

    def test_task_copy_preserves_scope_denominator_and_escapes_raw_descriptions(self):
        template = (Path(generic.__file__).parent/'reference_replay.html').read_bytes()
        report = {'attempted': 12, 'scenes': [1,2,3,4]}
        included = [{'scene_id': 'nominal-00', 'control': 'placement', 'success': False,
                     'reason': '<script>fake</script>', 'result_file': 'attempts/nominal-00--placement.json'}]
        html = replay._task_html(template, report, included, ['nominal-00'], [{'id': 'other'}]).decode()
        self.assertIn('1 recorded attempts from 1 scenes', html)
        self.assertIn('complete benchmark contains 12 planned attempts', html)
        self.assertIn('this excerpt is not a success-rate denominator', html)
        self.assertIn('&lt;script&gt;fake&lt;/script&gt;', html)
        self.assertNotIn('<script>fake</script>', html)
        self.assertNotIn('unloaded authored motions', html)
        self.assertIn('does not rerun physics', html)
        self.assertIn('not measured joint torque', html)
        with self.assertRaisesRegex(ValueError, 'template changed'):
            replay._task_html(b'changed template', report, included, ['nominal-00'], [])

    def test_source_snapshot_keeps_original_manifest_and_rechecks_immutable_bytes(self):
        report = {'source': 'fixture'}
        files = {'report.json': generic._json_bytes(report), 'report.html': b'fixture', 'protocol.json': b'{}'}
        manifest = {'schema': 'dvidia.reference-task-integrity/v1', 'files': {
            name: {'bytes': len(data), 'sha256': sha256(data).hexdigest()} for name, data in files.items()}}
        files['integrity.json'] = json.dumps(manifest, indent=5).encode()+b'\n'
        with patch.object(bench, 'load_task_bundle', return_value=(report, files)):
            loaded, preserved = replay._source_bundle('unused')
            self.assertEqual(loaded, report)
            self.assertEqual(preserved['integrity.json'], files['integrity.json'])
            files['report.html'] += b'changed'
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                replay._source_bundle('unused')


@unittest.skipUnless(os.environ.get('DVIDIA_REFERENCE_ASSETS'), 'Set DVIDIA_REFERENCE_ASSETS to the pinned local native assets')
class NativeTaskReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dvidia_training import reference_task as task
        cls.assets = Path(os.environ['DVIDIA_REFERENCE_ASSETS'])
        cls.mj, cls.np = sim._native_imports()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.protocol = bench.make_protocol(nominal=1, stress=1)
        results = []
        for scene in cls.protocol['scenes']:
            config = replace(task.ReferenceTaskConfig(**scene['config']), timeout_s=.1)
            scene['config'] = asdict(config)
            episode = task.run_episode(cls.assets, config, seed=scene['seed'], control='hold')
            if episode['valid'] is not True:
                raise AssertionError('Native hold fixture invalid: '+str(episode['error']))
            # Tests use one actual hold state tape for all three identity slots.
            # They exercise replay identity/geometry, not benchmark controller outcomes.
            results.append([deepcopy(episode) for _ in bench.CONTROLS])
        cls.bundle = Path(cls.temporary.name)/'receipt'
        cls.report, cls.files = write_bundle(cls.bundle, cls.protocol, results, sim.verify_assets(cls.assets))
        cls.output = Path(cls.temporary.name)/'replay'
        with patch.object(cls.mj, 'mj_step', side_effect=AssertionError('Replay must not advance physics')):
            cls.summary = replay.export_task_replay(cls.assets, cls.bundle, cls.output)
        cls.scene = json.loads((cls.output/'scene.json').read_bytes())
        cls.motion = json.loads((cls.output/'motion.json').read_bytes())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_native_block_table_target_and_all_valid_matched_controls_are_present(self):
        self.assertEqual(self.summary['cases'], 6)
        self.assertEqual(self.summary['benchmark_attempted'], 6)
        self.assertEqual(self.summary['reconstructed_frames'], 36)
        self.assertEqual(self.summary['excluded_attempts'], [])
        self.assertFalse(self.summary['learned_policy'])
        self.assertFalse(self.summary['physical_robot_ready'])
        geoms = {g['name']:g for g in self.scene['geoms']}
        for name in ('task_block_geom','task_table','task_target'):
            self.assertIn(name, geoms)
            self.assertEqual(geoms[name]['type'], 'box')
        self.assertTrue(geoms['task_block_geom']['collision'])
        self.assertFalse(geoms['task_target']['collision'])
        self.assertTrue(all(value <= 1e-10 for value in self.summary['reconstruction_max_absolute_residual'].values()))
        self.assertTrue(all(case['recorded_success'] is False for case in self.motion['cases']))

    def test_pose_binary_matches_each_native_body_and_varied_static_target(self):
        from dvidia_training import reference_task as task
        np, mj = self.np, self.mj
        content = (self.output/'motion.bin').read_bytes()
        targets = []
        for case in self.motion['cases']:
            raw = json.loads(self.files['attempts/'+case['attempt_id']+'.json'])
            model = task.build_task_model(self.assets, task.ReferenceTaskConfig(**raw['config']))
            data = mj.MjData(model)
            poses = np.frombuffer(content, dtype='<f4', count=case['frame_count']*model.nbody*7,
                                  offset=case['pose_byte_offset']).reshape(case['frame_count'],model.nbody,7)
            for index, row in enumerate(raw['trajectory']):
                data.qpos[:] = row['qpos']
                mj.mj_forward(model,data)
                expected = np.concatenate((data.xpos,data.xquat),axis=1).astype('<f4')
                self.assertEqual(poses[index].tobytes(), expected.tobytes())
            target = mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'task_target_frame')
            targets.append(poses[0,target,:3].tolist())
        self.assertNotEqual(targets[0],targets[3])

    def test_source_receipt_and_gzip_are_exact_and_existing_output_is_preserved(self):
        self.assertEqual((self.output/'receipt/integrity.json').read_bytes(), self.files['integrity.json'])
        for name, original in self.files.items():
            self.assertEqual((self.output/'receipt'/name).read_bytes(), original)
        for name in ('geometry.bin','motion.bin'):
            self.assertEqual(gzip.decompress((self.output/(name+'.gz')).read_bytes()), (self.output/name).read_bytes())
        seal = json.loads((self.output/'replay-integrity.json').read_bytes())
        for name, record in seal['files'].items():
            data = (self.output/name).read_bytes()
            self.assertEqual(record, {'bytes':len(data),'sha256':sha256(data).hexdigest()})
        with self.assertRaises(FileExistsError):
            replay.export_task_replay(self.assets,self.bundle,self.output)

    def test_valid_subset_excludes_other_scene_without_changing_source_denominator(self):
        output = Path(self.temporary.name)/'subset'
        chosen = self.report['scenes'][0]['id']
        result = replay.export_task_replay(self.assets,self.bundle,output,scene_ids=[chosen])
        self.assertEqual(result['cases'],3)
        self.assertEqual(result['benchmark_attempted'],6)
        self.assertEqual(len(result['excluded_attempts']),3)
        self.assertTrue(all(x['reason']=='scene_not_selected' for x in result['excluded_attempts']))
        self.assertEqual((output/'receipt/report.json').read_bytes(),self.files['report.json'])

    def test_mismatched_measured_state_or_compiled_model_fails_before_output_exists(self):
        from dvidia_training import reference_task as task
        raw = json.loads(self.files[self.report['attempts'][0]['result_file']])
        model = task.build_task_model(self.assets, task.ReferenceTaskConfig(**raw['config']))
        for field in ('tcp_position_m','object_position_m','pad_center_separation_m'):
            changed = deepcopy(raw)
            if field=='pad_center_separation_m':
                changed['trajectory'][0][field] += .001
            else:
                changed['trajectory'][0][field][0] += .001
            residuals = dict.fromkeys(self.summary['reconstruction_max_absolute_residual'],0.)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'differs from the sealed trace'):
                replay._poses(model,changed['trajectory'],changed['actions'],residuals)
        files = dict(self.files)
        raw['compiled_model_sha256']='0'*64
        files[self.report['attempts'][0]['result_file']] = generic._json_bytes(raw)
        output = Path(self.temporary.name)/'mismatch'
        with patch.object(replay,'_source_bundle',return_value=(self.report,files)), \
                self.assertRaisesRegex(ValueError,'model digest differs'):
            replay.export_task_replay(self.assets,self.bundle,output)
        self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
