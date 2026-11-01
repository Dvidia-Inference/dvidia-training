"""Independent audit tests: no simulator, renderer, tracker or policy execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import os
import subprocess
import sys
import tempfile
import unittest

HELPERS = Path(__file__).resolve().parents[1] / 'benchmarks/camera-action-v01'
_spec = importlib.util.spec_from_file_location('dvidia_camera_action_study_audit_tests', HELPERS / 'study_audit.py')
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def protocol():
    value = dict(cases={}, rosters={}, goals=[[.35, .08, .08], [.40, -.08, .08]],
        start_domain=dict(x_m=[.287, .311], y_m=[-.042, -.018]), historical_start_exclusion=[],
        data_quality=dict(minimum_successful_episodes_per_goal=28),
        acceptance=dict(student_successes_minimum=58, student_attempts=64, confidence=.975,
            bootstrap_resamples=10000, bootstrap_seed=311402, minimum_common_admitted_groups=24))
    offset = 0
    for split, count in (('train', 32), ('development', 8), ('confirmation', 32)):
        value['cases'][split] = []
        for i in range(count):
            group = f'{split}-{i:02}'
            for goal in (0, 1):
                value['cases'][split].append(dict(id=f'{group}-goal{goal}', group_id=group, split=split,
                    goal_id=goal, seed=311401+2*(offset+i)+goal,
                    object_position_m=[.288+(offset+i)*.0003, -.03, .082], goal_position_m=value['goals'][goal]))
        offset += count
    def attempt(case, controller, negative=None):
        return dict(id=case['id']+'-'+controller+('-'+negative if negative else ''), case=deepcopy(case),
            controller=controller, negative_control=negative)
    cases = value['cases']; roster = value['rosters']
    roster['diagnostic'] = [attempt(c, 'teacher') for c in cases['development'][:4]]
    roster['demonstrations'] = [attempt(c, 'teacher') for c in cases['train']+cases['development']]
    roster['corrections'] = [attempt(cases['train'][2*i+i%2], 'mixed') for i in range(32)]
    for stage in ('initial', 'corrected'):
        roster['development-'+stage] = [attempt(c, control) for c in cases['development'] for control in ('student', 'blind')]
    roster['confirmation'] = [attempt(c, control) for c in cases['confirmation'] for control in ('student', 'blind', 'fixed', 'teacher')]
    roster['negative'] = [attempt(c, 'student', fault) for c in cases['confirmation'][:4]
        for fault in ('initial_missing', 'initial_stale', 'altered_calibration', 'open_gripper')]
    return value


def row(attempt, success=False):
    return dict(**deepcopy(attempt), status='complete', complete=True, strict_success=success,
        initial_anchor_available=True, initial_anchor_binding_valid=True, recorded_complete_intervals=1000,
        critical_contact_ticks=0, provenance_ok=attempt['controller'] != 'teacher',
        score=dict(valid=True, success=success, strict_v01=dict(trajectory_complete=True)))


class StudyAuditTests(unittest.TestCase):
    def test_roster_rejects_split_leak_controller_substitution_and_missing_fault(self):
        p = protocol(); audit.validate_roster(p)
        for modify in (
            lambda x: x['rosters']['confirmation'][0].update(controller='teacher'),
            lambda x: x['rosters']['negative'][0].update(negative_control='initial_stale'),
            lambda x: x['rosters']['corrections'][0].update(case=x['cases']['confirmation'][0]),
            lambda x: x['cases']['confirmation'][0].update(group_id='train-00'),
            lambda x: x['cases']['train'][0].update(goal_position_m=[0, 0, 0]),
        ):
            changed = deepcopy(p); modify(changed)
            with self.assertRaises(ValueError): audit.validate_roster(changed)

    def test_data_quality_requires_group_coverage_and_per_destination_success(self):
        p = protocol(); rows = [row(a, True) for a in p['rosters']['demonstrations']]
        self.assertTrue(audit.data_quality(rows, p)['passed'])
        rows[0]['strict_success'] = rows[1]['strict_success'] = False
        self.assertEqual(audit.data_quality(rows, p)['missing_training_groups'], ['train-00'])
        rows = [row(a, True) for a in p['rosters']['demonstrations']]
        for r in rows[:10:2]: r['initial_anchor_available'] = False
        self.assertEqual(audit.data_quality(rows, p)['successful_episodes_per_goal']['0'], 27)
        self.assertFalse(audit.data_quality(rows, p)['passed'])

    def test_mixed_authority_is_seeded_and_teacher_query_is_not_executed_student_command(self):
        limits = dict(joint_limits_rad=[[-3, 3]]*6, gripper_position_limits_m=[0., .035])
        teacher = dict(joint_targets_rad=[.1]*6, gripper_position_m=.02)
        student = dict(joint_targets_rad=[.2]*6, gripper_position_m=.01)
        truth, checked = random.Random(311401), random.Random(311401)
        for _ in range(20):
            authority = 'teacher' if truth.random() < .5 else 'student'
            requested = deepcopy(teacher if authority == 'teacher' else student)
            step = dict(command=dict(operator_correction=False, authority=authority, requested=requested,
                issued=deepcopy(requested), issue_reason='unchanged'))
            query = dict(origin='privileged_simulator_teacher', action=deepcopy(teacher))
            audit.command_check(step, query, 'mixed', None, checked, limits)
        step['command']['authority'] = 'teacher'
        step['command']['requested'] = student
        step['command']['issued'] = student
        with self.assertRaisesRegex(ValueError, 'separately sealed query'):
            audit.command_check(step, query, 'teacher', None, None, limits)
        step['command']['authority'] = 'student'
        with self.assertRaisesRegex(ValueError, 'teacher intervention'):
            audit.command_check(step, query, 'student', None, None, limits)

    def test_open_gripper_intervention_and_nested_numeric_negative_score(self):
        limits = dict(joint_limits_rad=[[-3, 3]]*6, gripper_position_limits_m=[0., .035])
        command = dict(joint_targets_rad=[0.]*6, gripper_position_m=.01)
        step = dict(command=dict(operator_correction=False, authority='student', requested=command,
            issued={**command, 'gripper_position_m': .035}, issue_reason='open_gripper_negative_control'))
        audit.command_check(step, None, 'student', 'open_gripper', None, limits)
        with self.assertRaisesRegex(ValueError, 'registered intervention'):
            audit.command_check(step, None, 'student', None, None, limits)
        r = row(protocol()['rosters']['negative'][3]); self.assertTrue(audit.negative_correct(r))
        r['score'] = dict(valid=True, trajectory_complete=True)
        self.assertFalse(audit.negative_correct(r))
        r['score'] = dict(valid=False, strict_v01=dict(trajectory_complete=True))
        self.assertFalse(audit.negative_correct(r))
        r['negative_control'] = 'initial_missing'; r['status'] = 'refused'
        self.assertFalse(audit.negative_correct(r))
        r['recorded_complete_intervals'] = 0; self.assertTrue(audit.negative_correct(r))

    def test_perception_missing_masks_and_declared_initial_calibration_fault(self):
        frame = {'index': 0, 'rgb': {'path': 'frames/rgb-0000.png', 'bytes': 1, 'sha256': 'a'*64},
            'depth': {'path': 'frames/depth-0000.npy', 'bytes': 1, 'sha256': 'b'*64}}
        snapshot = dict(timestamp_seconds=.02, camera=dict(value=frame, captured_at_seconds=0., available_at_seconds=0., age_seconds=.02))
        calibration = dict(base_from_camera=[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])
        recording = dict(calibration=calibration, binding=dict(calibration_sha256=audit.digest(calibration)), tracker_sha256='f'*64)
        def document(fault=None):
            actual=deepcopy(calibration)
            if fault=='altered_calibration': actual['base_from_camera'][0]*=1.1
            captured=-.2 if fault=='initial_stale' else 0.
            estimate=dict(status='missing', frame_index=0, captured_at_seconds=captured, available_at_seconds=captured,
                decision_at_seconds=.02, age_seconds=.02-captured, signals={})
            for key in audit.VISION:
                estimate[key]=None; estimate['signals'][key]=dict(status='missing', value=None)
            return dict(index=1, snapshot_sha256=audit.digest(snapshot), decision_at_seconds=.02,
                frame=frame, calibration_sha256=recording['binding']['calibration_sha256'], tracker_sha256='f'*64,
                input_calibration_sha256=audit.digest(actual), negative_control=fault, estimate=estimate)
        for fault in (None, 'initial_missing', 'initial_stale', 'altered_calibration', 'open_gripper'):
            audit._perception(document(fault), snapshot, recording, fault, 1)
        changed=document(); changed['estimate']['signals']['object_position_estimate_m']['value']=[0, 0, 0]
        with self.assertRaisesRegex(ValueError, 'Unavailable perception'):
            audit._perception(changed, snapshot, recording, None, 1)
        changed=document('altered_calibration'); changed['input_calibration_sha256']=audit.digest(calibration)
        with self.assertRaisesRegex(ValueError, 'input calibration'):
            audit._perception(changed, snapshot, recording, 'altered_calibration', 1)
        changed=document('initial_stale'); changed['estimate']['age_seconds']=0.
        with self.assertRaisesRegex(ValueError, 'age differs'):
            audit._perception(changed, snapshot, recording, 'initial_stale', 1)

    def test_snapshot_cadence_frame_reuse_and_raw_truth_exclusion(self):
        missing=dict(status='missing', value=None)
        signals={name:deepcopy(missing) for name in audit.VISION}
        for name in audit.ENCODERS:
            signals[name]=dict(status='available', value=[0., 0.], captured_at_seconds=.02, available_at_seconds=.02, age_seconds=0.)
        frame=dict(index=0, rgb=dict(path='frames/rgb.png', bytes=2, sha256='a'*64), depth=dict(path='frames/depth.npy', bytes=3, sha256='b'*64))
        snap=dict(timestamp_seconds=.02, signals=signals, vision_frame_index=0,
            camera=dict(value=frame, captured_at_seconds=0., available_at_seconds=0., age_seconds=.02))
        files={ref['path']:{k:ref[k] for k in ('bytes','sha256')} for ref in (frame['rgb'],frame['depth'])}
        audit._snapshot(snap, 1, files, {})
        changed=deepcopy(snap); changed['signals']['object_position_estimate_m']={'status':'available', 'value':[1, 2, 3]}
        with self.assertRaisesRegex(ValueError, 'native object truth'): audit._snapshot(changed, 1, files, {})
        changed=deepcopy(snap); changed['camera']['captured_at_seconds']=.02
        with self.assertRaisesRegex(ValueError, 'capture time'): audit._snapshot(changed, 1, files, {})
        with self.assertRaisesRegex(ValueError, 'Reused camera'): audit._snapshot(snap, 1, files, {0:dict(index=0)})
        files['frames/rgb.png']['sha256']='c'*64
        with self.assertRaisesRegex(ValueError, 'byte-bound'): audit._snapshot(snap, 1, files, {})

    def test_bootstrap_groups_both_goals_and_excludes_both_goals_if_one_not_admitted(self):
        p = protocol(); rows = [row(a, a['controller']=='student') for a in p['rosters']['confirmation']]
        result = audit.bootstrap(rows, 'fixed', p['acceptance'])
        self.assertEqual((result['difference'], result['lower'], result['upper'], result['groups']), (1., 1., 1., 32))
        rows[2]['initial_anchor_available'] = False
        result = audit.bootstrap(rows, 'fixed', p['acceptance'], admitted_only=True)
        self.assertEqual((result['groups'], result['excluded_groups']), (31, 1))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit.bootstrap(rows+[rows[0]], 'fixed', p['acceptance'])

    def test_final_gates_fail_when_advantage_exists_only_through_refusals_or_contacts(self):
        p = protocol(); rows = [row(a, a['controller']=='student') for a in p['rosters']['confirmation']]
        negatives = [row(a) for a in p['rosters']['negative']]
        for r in negatives:
            if r['negative_control'] != 'open_gripper':
                r.update(status='refused', recorded_complete_intervals=0, complete=False)
        self.assertTrue(audit.final_gates(rows, negatives, p, stages_complete=True)['qualified'])
        for r in rows:
            if r['controller'] == 'fixed' and int(r['case']['group_id'][-2:]) < 9:
                r.update(initial_anchor_available=False, recorded_complete_intervals=0)
        result = audit.final_gates(rows, negatives, p, stages_complete=True)
        self.assertFalse(result['gates']['positive_common_admitted_comparisons'])
        rows[0]['critical_contact_ticks'] = 1
        self.assertFalse(audit.final_gates(rows, negatives, p, stages_complete=True)['gates']['no_critical_contact_increase'])
        self.assertFalse(audit.final_gates(rows, negatives, p, stages_complete=False)['gates']['all_registered_attempts_accounted'])

    def test_stage_binding_and_model_replacement_are_detected_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'stages').mkdir()
            def save(name, obj):
                p=root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(audit.canonical(obj))
            receipt = dict(stage='diagnostic', protocol_sha256='a'*64, model_bindings={'teacher': 'b'*64})
            save('stages/diagnostic.json', receipt)
            save('stages/diagnostic.started.json', receipt)
            save('stages/diagnostic.integrity.json', dict(protocol_sha256='a'*64, sha256=audit.identity(root/'stages/diagnostic.json')['sha256']))
            audit._stage(root, 'diagnostic', 'a'*64, {})
            save('stages/diagnostic.started.json', {**receipt, 'model_bindings': {'teacher': 'c'*64}})
            with self.assertRaisesRegex(ValueError, 'Started stage'):
                audit._stage(root, 'diagnostic', 'a'*64, {})
            save('models/initial/full/model.json', {'weights': 'original'})
            stages = {'train-initial': dict(models=dict(full=dict(model_sha256=audit.identity(root/'models/initial/full/model.json')['sha256'])))}
            audit._model_hash(root, 'initial', 'student', stages, {})
            save('models/initial/full/model.json', {'weights': 'replacement'})
            with self.assertRaisesRegex(ValueError, 'sealed training receipt'):
                audit._model_hash(root, 'initial', 'student', stages, {})

    def test_active_worker_is_not_read_or_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'worker.lock').write_text('running')
            result = audit.audit(directory)
            self.assertFalse(result['passed']); self.assertEqual(result['attempts_verified'], 0)
            self.assertIn('worker', result['errors'][0]['reason'])

    def test_training_audit_rejects_matched_full_blind_subset_and_validation_roster_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); p=protocol()
            p['fitting']=dict(epochs=50,batch_size=512,learning_rate=.003,seed=17,hidden_units=[64,64])
            p['source_sha256']={'camera_action_model.py':'a'*64,'camera_action_export.py':'b'*64}
            def save(name,value):
                target=root/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(audit.canonical(value))
            save('protocol.json',p)
            rows=[]
            for case in p['cases']['train'][:2]+p['cases']['development'][:1]:
                attempt=dict(id=case['id']+'-teacher',case=case,controller='teacher',negative_control=None)
                r=row(attempt,True); r.update(recorded_complete_intervals=2,path='episodes/demonstrations/'+attempt['id'])
                recording=dict(control_mode='teacher',provenance=dict(source_group=case['group_id'],case_id=case['id']),
                    binding=dict(calibration_sha256='c'*64),tracker_sha256='d'*64,
                    steps=[dict(perception=dict(sha256='e'*64),teacher_query=None,command=dict(authority='teacher')) for _ in range(2)])
                save(r['path']+'/recording.json',recording); save(r['path']+'/integrity.json',{'fixture':case['id']})
                rows.append(r)
            stages={'demonstrations':dict(data_quality={'passed':True},rows=rows),'train-initial':dict(models={})}
            original=[audit._export_source(root,r,'issued') for r in rows[:2]]
            def models(sources, validation_ids=None):
                for variant in ('full','blind'):
                    provenance=dict(episodes=sources,source_identity_sha256=audit.digest(sources),blind=variant=='blind',
                        export_source_sha256='b'*64,feature_contract_sha256='f'*64)
                    training=dict(stage='initial',model_source_sha256='a'*64,epochs=50,batch_size=512,learning_rate=.003,seed=17,
                        train_groups=['train-00'],sample_count=4,validation=dict(groups=['development-00'],sample_count=2),
                        provenance=provenance,provenance_sha256=audit.digest(provenance))
                    model=dict(schema='dvidia.camera-action-mlp/v1',blind=variant=='blind',
                        observation_origin='rendered_rgbd_and_simulated_encoders',feature_contract='camera-encoder-absolute-causal160/v1',
                        action_representation='absolute_native_targets',hidden_units=[64,64],feature_contract_sha256='f'*64,training=training)
                    path=f'models/initial/{variant}/model.json';save(path,model); identity=audit.identity(root/path)
                    receipt=dict(path=path,model_bytes=identity['bytes'],model_sha256=identity['sha256'],
                        training_episodes=[r['id'] for r in rows[:2]],development_episodes=validation_ids or [rows[2]['id']],
                        training=training,dataset_provenance=provenance)
                    stages['train-initial']['models'][variant]=receipt
                    save(f'models/initial/{variant}/receipt.json',receipt)
                    save(f'models/initial/{variant}/started.json',dict(stage='initial',variant=variant,
                        protocol_sha256=audit.identity(root/'protocol.json')['sha256']))
            models(original);audit._training(root,stages,p,{})
            models(original[:1])
            with self.assertRaisesRegex(ValueError,'omits, duplicates or changes'):
                audit._training(root,stages,p,{})
            models([original[0],original[0]])
            with self.assertRaisesRegex(ValueError,'omits, duplicates or changes'):
                audit._training(root,stages,p,{})
            models(original,validation_ids=[rows[0]['id']])
            with self.assertRaisesRegex(ValueError,'episode roster differs'):
                audit._training(root,stages,p,{})

    def test_numeric_comparison_does_not_accept_booleans_as_numeric_gates(self):
        self.assertTrue(audit._equivalent({'low': .1}, {'low': .10000000000000002}))
        self.assertFalse(audit._equivalent({'passed': True}, {'passed': 1}))

    def test_imports_do_not_load_environment_tracker_controller_or_study_runtime(self):
        env=dict(os.environ, PYTHONPATH=str(HELPERS)+os.pathsep+str(HELPERS.parents[1]/'src'))
        completed=subprocess.run([sys.executable, '-c',
            "import study_audit, sys; forbidden=('camera_action_study','camera_action_capture','camera_action_model','piper_feedback_env','piper_rgbd_tracker','piper_motor_primitive','mujoco'); assert not any(m.rsplit('.',1)[-1] in forbidden for m in sys.modules)"],
            env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_encoder_tool_pose_and_velocity_are_reconstructed_without_kinematic_runtime(self):
        import numpy as np
        transform=np.eye(4); transform[:3, 3]=[1., 2., 3.]
        state=dict(qvel=[0.]*6+[.01, -.01], tcp_position_m=[1.2, 2.3, 3.52003], tcp_quaternion_wxyz=[1., 0., 0., 0.])
        snapshot=dict(signals=dict(gripper_joint_velocity_m_s={'value':[.01, -.01]},
            tcp_position_estimate_m={'value':[.2, .3, .4]}, tcp_quaternion_estimate_wxyz={'value':[-1., 0., 0., 0.]}))
        audit._encoder_pose(snapshot,state,transform)
        snapshot['signals']['tcp_position_estimate_m']['value']=[1.2, 2.3, 3.4]
        with self.assertRaisesRegex(ValueError, 'tool position'): audit._encoder_pose(snapshot,state,transform)


if __name__ == '__main__': unittest.main()
