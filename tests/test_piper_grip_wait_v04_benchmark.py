"""Independent lifecycle contracts; synthetic fixtures make no physics claims."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_grip_clock as clock
from dvidia_training import piper_grip_clock_model as clock_model
from dvidia_training import piper_grip_wait_v04 as runtime
from dvidia_training import piper_grip_wait_v04_benchmark as bench
from dvidia_training import piper_imitation as core
from dvidia_training import piper_feedback_score
from tests.test_piper_feedback_model import observation
from tests.test_piper_grip_clock_runtime import changing_path_model
from tests.test_piper_grip_clock_benchmark import FakeEnv, FakePolicy
from tests.test_piper_feedback_score import fixture as score_fixture, GOAL


def replace(path, value):
    Path(path).write_bytes(core.canonical(value))


SCORE = dict(valid=True, success=False, reason='fixture_failed',
             native_recomputed=dict(grasp_seen=False, retention_lost=False,
                                    unexpected_contact_native_ticks=0),
             post_grip_analysis=dict(valid=True, eligible=False, activated=False,
                                     effective=False, scheduled=False))


class GripWaitBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = changing_path_model()
        cls.jaw = clock.reference_jaw_manifest(cls.base)
        cls.full = clock_model.train(dict(features=np.zeros((8, clock_model.FEATURE_COUNT)),
            targets=np.zeros(8), groups=['fixture-a']*4+['fixture-b']*4),
            base_sha256=core.digest(cls.base), jaw_sha256=core.digest(cls.jaw),
            jaw_mode='coordinated_reference', min_leaf=2)

    def setUp(self):
        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.parent = Path(temp.name)
        self.prior = self.parent/'prior'; self.prior.mkdir()
        self.root = self.parent/'study'
        bench.write(self.prior/'protocol.json', dict(schema='fixture-previous-protocol'))
        documents = {'base-model.json': self.base, 'jaw-model.json': self.jaw, 'selected.json': self.full}
        for name, document in documents.items(): bench.write(self.prior/name, document)
        frozen = dict(protocol_sha256=bench.sha(self.prior/'protocol.json'),
            runtime_sha256=bench.old.runtime_hashes(),
            base_model_raw_sha256=bench.sha(self.prior/'base-model.json'),
            jaw_model_raw_sha256=bench.sha(self.prior/'jaw-model.json'),
            selected_sha256=bench.sha(self.prior/'selected.json'),
            selected='full4', blind='blind4', jaw_mode='coordinated_reference',
            files={name: dict(bytes=(self.prior/name).stat().st_size, sha256=bench.sha(self.prior/name))
                   for name in documents})
        bench.write(self.prior/'frozen.json', frozen)
        bench.write(self.prior/'report.json', dict(protocol_sha256=bench.sha(self.prior/'protocol.json'),
            frozen_sha256=bench.sha(self.prior/'frozen.json')))
        # Synthetic documents have the exact same model contracts, with their
        # own declared raw-byte identity rather than the unavailable real study.
        handle = patch.object(bench.old, 'BASE_SHA', bench.sha(self.prior/'base-model.json'))
        handle.start(); self.addCleanup(handle.stop)

    def prepare(self):
        return bench.prepare(self.root, self.prior)

    def fake_episode(self, selected, variant, config, *, aborted=False):
        teacher, intervention = bench.old._options()
        return dict(schema=bench.old.EPISODE_SCHEMA, study_schema=bench.SCHEMA,
            config=bench.old._canonical(asdict(bench.old.config(selected))),
            case_id=selected['id'], seed=selected['seed'], condition=selected['condition'], policy_kind=variant,
            base_model_sha256=core.digest(self.base), jaw_model_sha256=core.digest(self.jaw),
            model_sha256=core.digest(self.full) if variant in ('guarded','clock') else None,
            jaw_mode='coordinated_reference', teacher_config=asdict(teacher), post_grip_parameters=intervention,
            supervisor_config=config if variant=='guarded' else None,
            implementation_sha256=bench.sources(), policy_diagnostics=dict(aborted=aborted, wait_ticks=0),
            actions=[], trajectory=[], valid=False, success=False,
            timing=dict(inference_samples_s=[], control_step_samples_s=[],wall_seconds=0.))

    def fake_rollout(self, assets, selected, base, jaw, full, variant, config):
        teacher, intervention = bench.old._options()
        metadata = dict(post_grip_intervention=dict(parameters=dict(
            interruption=selected['condition']=='post_grip', **intervention)))
        return self.fake_episode(selected,variant,config), {'fixture':np.asarray([1.])}, metadata

    def test_roster_is_deterministic_fresh_grouped_and_condition_seed_paired(self):
        a, b = bench.make_cases(), bench.make_cases()
        self.assertEqual(a,b)
        prior = [c['start_xy'] for cases in bench.old_cli.make_cases().values() for c in cases]
        old_points = [*bench.old_cli.historical_starts(), *prior]
        seen=[]
        for phase, count in [('development',4),('confirmation',32)]:
            rows=a[phase]
            self.assertEqual(len(rows),count*4)
            groups={tuple(row['start_xy']) for row in rows}
            self.assertEqual(len(groups),count)
            for point in groups:
                own=[row for row in rows if tuple(row['start_xy'])==point]
                self.assertEqual({(tuple(r['goal_xy']),r['condition']) for r in own},
                                 {(tuple(g),c) for g in bench.primitive.GOALS for c in ('nominal','post_grip')})
                for goal in bench.primitive.GOALS:
                    paired=[r for r in own if r['goal_xy']==list(goal)]
                    self.assertEqual(len({r['seed'] for r in paired}),1)
                for other in old_points+seen:
                    self.assertGreaterEqual(np.linalg.norm(np.asarray(point)-other),.00075-1e-12)
                seen.append(point)
        self.assertEqual(len(seen),36)

    def test_prepare_binds_prior_exact_policy_bytes_and_rejects_modified_selected_input(self):
        # A valid JSON tree can still be the wrong frozen candidate.
        changed=deepcopy(self.full); changed['nodes'][0]['advance_probability']=.75
        replace(self.prior/'selected.json', changed)
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.root.exists())

    def test_protocol_candidate_and_selection_changes_are_detected_before_execution(self):
        self.prepare()
        for field in ('candidate','selection','variant'):
            protocol=bench.read(self.root/'protocol.json')
            changed=deepcopy(protocol)
            if field=='candidate': changed['candidates']['wait200']['maximum_wait_s']=.50
            elif field=='selection': changed['selection']='Select the best confirmation result instead.'
            else: changed['variants']=['guarded']
            replace(self.root/'protocol.json',changed)
            with self.subTest(field=field), self.assertRaises(ValueError): bench.checked(self.root)
            replace(self.root/'protocol.json',protocol)

    def test_frozen_input_and_source_snapshot_changes_are_detected(self):
        self.prepare()
        for path in (self.root/'selected.json', self.root/'source'/'piper_grip_wait_v04.py'):
            original=path.read_bytes(); path.write_bytes(original+b'\n')
            with self.subTest(path=path.name), self.assertRaises(ValueError): bench.checked(self.root)
            path.write_bytes(original)
        self.assertEqual(bench.checked(self.root)['schema'],bench.SCHEMA)

    def test_unlisted_case_wrong_config_and_wrong_folder_cannot_start_physics(self):
        protocol=self.prepare(); case=protocol['cases']['development'][0]
        cfg=protocol['candidates']['wait200']
        badcase=deepcopy(case);badcase['seed']+=1
        failures=[(badcase,cfg,self.root/'development'/'wait200'/case['id']),
                  (case,{**cfg,'maximum_wait_s':.22},self.root/'development'/'wait200'/case['id']),
                  (case,cfg,self.root/'unlisted'/case['id'])]
        with patch.object(bench,'rollout',side_effect=AssertionError('Unbound physics must not run')):
            for selected,config,folder in failures:
                with self.subTest(folder=folder),self.assertRaises(ValueError):
                    bench.execute(self.root,None,selected,'guarded',folder,config)

    def test_started_incomplete_attempt_is_retained_without_automatic_rerun(self):
        protocol=self.prepare(); case=protocol['cases']['development'][0]
        folder=self.root/'development'/'wait200'/case['id']
        bench.write(folder/'started.json',dict(interrupted=True))
        before=(folder/'started.json').read_bytes()
        with patch.object(bench,'rollout',side_effect=AssertionError('Never rerun interrupted physics')):
            with self.assertRaisesRegex(ValueError,'Interrupted'):
                bench.execute(self.root,None,case,'guarded',folder,protocol['candidates']['wait200'])
        self.assertEqual((folder/'started.json').read_bytes(),before)

    def test_failed_complete_attempt_reuses_identical_evidence_without_physics(self):
        protocol=self.prepare(); case=protocol['cases']['development'][0];cfg=protocol['candidates']['wait200']
        folder=self.root/'development'/'wait200'/case['id']
        with patch.object(bench,'rollout',side_effect=self.fake_rollout),patch.object(
                bench.old,'_score',return_value=(deepcopy(SCORE),deepcopy(SCORE['post_grip_analysis']))):
            row=bench.execute(self.root,None,case,'guarded',folder,cfg)
        self.assertTrue(row['valid']);self.assertFalse(row['success'])
        before={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
        with patch.object(bench,'rollout',side_effect=AssertionError('Never repeat failed physics')),patch.object(
                bench.old,'_score',return_value=(deepcopy(SCORE),deepcopy(SCORE['post_grip_analysis']))):
            again=bench.execute(self.root,None,case,'guarded',folder,cfg)
        self.assertEqual(row,again)
        self.assertEqual(before,{p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()})

    def test_raw_strict_success_after_supervisor_abort_stays_primary_failure(self):
        protocol=self.prepare();case=protocol['cases']['development'][0];cfg=protocol['candidates']['wait200']
        episode=self.fake_episode(case,'guarded',cfg,aborted=True)
        score={**deepcopy(SCORE),'success':True}
        row=bench.summary(case,'guarded',score,episode,'a'*64)
        self.assertTrue(row['valid']);self.assertTrue(row['raw_strict_success'])
        self.assertTrue(row['aborted']);self.assertFalse(row['success'])

    def test_unmodified_native_strict_scorer_accepts_guarded_policy_kind(self):
        episode, arrays, metadata=score_fixture()
        episode['policy_kind']='guarded'
        result=piper_feedback_score.score_feedback_episode(episode,arrays,metadata,GOAL)
        self.assertTrue(result['valid'],result)
        self.assertTrue(result['success'],result)
        # Cadence/control completeness is numeric, independent of the new name.
        episode['actions'].pop()
        result=piper_feedback_score.score_feedback_episode(episode,arrays,metadata,GOAL)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'],'native_control_trace_mismatch')

    def test_rollout_keeps_1000_controls_after_early_success_and_after_abort(self):
        case=bench.make_cases()['development'][0]
        cfg=asdict(runtime.GripWaitConfig(maximum_wait_s=.20))
        class AbortedPolicy(FakePolicy):
            def diagnostics(self):return dict(aborted=True,wait_ticks=10)
        with patch('dvidia_training.piper_grip_clock_env.PostGripPiperTaskEnv',FakeEnv),patch(
                'dvidia_training.robot_profiles.verify_assets',return_value={'fixture':True}),patch(
                'dvidia_training.reference_sim._native_imports',return_value=(SimpleNamespace(__version__='fixture'),None)),patch(
                'dvidia_training.reference_sim.model_digest',return_value='a'*64),patch.object(
                runtime,'GuardedLearnedClock',AbortedPolicy):
            episode,arrays,metadata=bench.rollout('unused',case,self.base,self.jaw,self.full,'guarded',cfg)
        self.assertEqual(len(episode['actions']),1000)
        self.assertEqual(len(episode['trajectory']),1001)
        self.assertEqual(episode['simulated_seconds'],20.)
        self.assertTrue(episode['policy_diagnostics']['aborted'])
        self.assertEqual(FakeEnv.instances[-1].steps,1000)

    def test_retained_attempt_exact_roster_and_assigned_intervention_are_bound(self):
        protocol=self.prepare();case=protocol['cases']['development'][0];cfg=protocol['candidates']['wait200']
        folder=self.root/'development'/'wait200'/case['id']
        with patch.object(bench,'rollout',side_effect=self.fake_rollout),patch.object(
                bench.old,'_score',return_value=(deepcopy(SCORE),deepcopy(SCORE['post_grip_analysis']))):
            bench.execute(self.root,None,case,'guarded',folder,cfg)
        original={p.name:p.read_bytes() for p in folder.iterdir()}
        for mode in ('extra-file','missing-file','wrong-started','episode-teacher','native-condition'):
            for path in folder.iterdir():path.unlink()
            for name,data in original.items():(folder/name).write_bytes(data)
            receipt=bench.read(folder/'receipt.json')
            if mode=='extra-file':(folder/'extra.json').write_text('{}')
            elif mode=='missing-file':(folder/'post-grip.json').unlink()
            elif mode=='wrong-started':
                replace(folder/'started.json',{'wrong':True})
                receipt['files']['started.json']=bench.sha(folder/'started.json')
                replace(folder/'receipt.json',receipt)
            elif mode=='episode-teacher':
                episode=bench.load_episode(folder/'episode.json.gz')
                episode['teacher_config']['gate_offset_s']=.32
                (folder/'episode.json.gz').unlink();bench.save_episode(folder/'episode.json.gz',episode)
                receipt['files']['episode.json.gz']=bench.sha(folder/'episode.json.gz')
                replace(folder/'receipt.json',receipt)
            else:
                meta=bench.read(folder/'native.json')
                meta['post_grip_intervention']['parameters']['interruption']=True
                replace(folder/'native.json',meta)
                receipt['files']['native.json']=bench.sha(folder/'native.json')
                replace(folder/'receipt.json',receipt)
            with self.subTest(mode=mode),patch.object(bench,'rollout',side_effect=AssertionError('Never rerun')),patch.object(
                    bench.old,'_score',return_value=(deepcopy(SCORE),deepcopy(SCORE['post_grip_analysis']))):
                with self.assertRaises(ValueError):bench.execute(self.root,None,case,'guarded',folder,cfg)

    def test_confirmation_requires_frozen_matching_protocol_source_and_candidate(self):
        protocol=self.prepare();case=protocol['cases']['confirmation'][0]
        folder=self.root/'confirmation'/case['id']/'guarded';cfg=protocol['candidates']['wait200']
        frozen=dict(protocol_sha256=bench.sha(self.root/'protocol.json'),source_sha256=bench.sources(),selected='wait200')
        bench.write(self.root/'frozen.json',frozen)
        with patch.object(bench,'rollout',side_effect=AssertionError('No unbound confirmation physics')):
            with self.assertRaisesRegex(ValueError,'selection'):
                bench.execute(self.root,None,case,'guarded',folder,protocol['candidates']['wait500'])
            for field in ('protocol_sha256','source_sha256'):
                changed=deepcopy(frozen);changed[field]='0'*64 if field.endswith('sha256') else {}
                replace(self.root/'frozen.json',changed)
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'selection'):
                    bench.execute(self.root,None,case,'guarded',folder,cfg)
                replace(self.root/'frozen.json',frozen)

    def test_native_scorer_requires_all_20000_rows_and_1000_commands_for_20_seconds(self):
        episode, arrays, metadata=score_fixture()
        count=20000
        for name,value in list(arrays.items()):
            arrays[name]=np.concatenate((value,np.repeat(value[-1:],count-len(value),axis=0)))
        arrays['native_tick']=np.arange(1,count+1,dtype=float)
        arrays['t_seconds']=arrays['native_tick']*.001
        metadata['rows']=count;episode['config']['timeout_s']=20.
        template=episode['trajectory'][-1];first=episode['trajectory'][0]
        names=('object_position_m','object_quaternion_wxyz','object_linear_velocity_m_s',
               'object_angular_velocity_rad_s','tcp_position_m','tcp_quaternion_wxyz',
               'gripper_joint_position_m','joint_position_rad')
        episode['trajectory']=[];episode['actions']=[]
        for k in range(1001):
            state=deepcopy(first if k==0 else template);state['t_seconds']=k*.02
            if k:
                for name in names:state[name]=arrays[name][k*20-1].tolist()
                for name in ('grasp_contact_counts','pad_normal_forces_n'):
                    state[name]={side:int(v) if name=='grasp_contact_counts' else float(v)
                        for side,v in zip(('left','right'),arrays[name][k*20-1])}
                for name in ('table_object_contact_count','unexpected_contact_count'):
                    state[name]=int(arrays[name][k*20-1])
            if k<1000:
                action=dict(t_seconds=k*.02,joint_targets_rad=[0.]*6,gripper_position_m=.035)
                episode['actions'].append(action)
                state.update(issued_joint_targets_rad=[0.]*6,issued_gripper_position_m=.035,command=action)
            episode['trajectory'].append(state)
        episode['metrics']['dwell_seconds']=19.520
        episode['policy_kind']='guarded'
        result=piper_feedback_score.score_feedback_episode(episode,arrays,metadata,GOAL)
        self.assertTrue(result['valid'],result)
        self.assertTrue(result['success'],result)
        metadata['rows']=19999
        truncated=piper_feedback_score.score_feedback_episode(episode,arrays,metadata,GOAL)
        self.assertFalse(truncated['valid'])
        self.assertEqual(truncated['reason'],'incomplete_native_trajectory')
        metadata['rows']=20000;episode['actions'].pop()
        truncated=piper_feedback_score.score_feedback_episode(episode,arrays,metadata,GOAL)
        self.assertFalse(truncated['valid'])
        self.assertEqual(truncated['reason'],'native_control_trace_mismatch')

    def test_predeclared_selection_prioritizes_success_then_aborts_contacts_and_total_wait(self):
        # Each tuple is (successes, aborts, critical attempts, per-attempt wait).
        scenarios=[
            ({'wait200':(1,0,0,0),'wait360':(0,0,0,0),'wait500':(2,1,1,20)},'wait500'),
            ({'wait200':(0,1,0,0),'wait360':(0,0,1,10),'wait500':(0,2,0,0)},'wait360'),
            ({'wait200':(0,0,2,0),'wait360':(0,0,1,0),'wait500':(0,0,0,20)},'wait500'),
            ({'wait200':(0,0,0,4),'wait360':(0,0,0,2),'wait500':(0,0,0,3)},'wait360')]
        for index,(settings,winner) in enumerate(scenarios):
            root=self.parent/f'priority-{index}';protocol=bench.prepare(root,self.prior)
            def row(case,name):
                i=protocol['cases']['development'].index(case);success,aborts,critical,wait=settings[name]
                return dict(case_id=case['id'],candidate=name,valid=True,success=i<success,
                    aborted=success<=i<success+aborts,critical_contact=i<critical,
                    policy_diagnostics=dict(wait_ticks=wait))
            rows=[]
            for name in protocol['candidates']:
                for case in protocol['cases']['development']:
                    bench.write(root/'development'/name/case['id']/'receipt.json',{'fixture':True})
                    rows.append(row(case,name))
            bench.write(root/'development-results.json',dict(schema=bench.SCHEMA,phase='development',
                protocol_sha256=bench.sha(root/'protocol.json'),rows=rows))
            def execute(root,assets,case,variant,folder,config):return row(case,folder.parent.name)
            with self.subTest(priority=index),patch.object(bench,'execute',side_effect=execute):
                self.assertEqual(bench.freeze(root)['selected'],winner)

    def test_development_aggregate_cannot_replace_independently_retained_results(self):
        protocol=self.prepare()
        for name in protocol['candidates']:
            for case in protocol['cases']['development']:
                bench.write(self.root/'development'/name/case['id']/'receipt.json',{'fixture':True})
        bench.write(self.root/'development-results.json',dict(schema=bench.SCHEMA,phase='development',
            protocol_sha256=bench.sha(self.root/'protocol.json'),rows=[]))
        with patch.object(bench,'execute',return_value=dict(valid=True,success=False,aborted=False,
                critical_contact=False,policy_diagnostics=dict(wait_ticks=0))):
            with self.assertRaisesRegex(ValueError,'aggregate'):bench.freeze(self.root)
        self.assertFalse((self.root/'frozen.json').exists())

    def test_selection_reconstructs_all_development_evidence_and_tie_breaks_shortest_budget(self):
        protocol=self.prepare();rows=[]
        for name,config in protocol['candidates'].items():
            for case in protocol['cases']['development']:
                folder=self.root/'development'/name/case['id']
                bench.write(folder/'receipt.json',{'fixture':True})
                rows.append(dict(case_id=case['id'],candidate=name,valid=True,success=False,aborted=False,
                    critical_contact=False,policy_diagnostics=dict(wait_ticks=10)))
        bench.write(self.root/'development-results.json',dict(schema=bench.SCHEMA,phase='development',
            protocol_sha256=bench.sha(self.root/'protocol.json'),rows=rows))
        def selected_row(root,assets,case,variant,folder,config):
            return dict(case_id=case['id'],candidate=folder.parent.name,valid=True,success=False,
                        aborted=False,critical_contact=False,policy_diagnostics=dict(wait_ticks=10))
        with patch.object(bench,'execute',side_effect=selected_row) as execute:
            frozen=bench.freeze(self.root)
        self.assertEqual(execute.call_count,48)
        self.assertEqual(frozen['selected'],'wait200')
        self.assertEqual([r['name'] for r in frozen['ranking']],['wait200','wait360','wait500'])
        with self.assertRaisesRegex(ValueError,'frozen'):
            bench.freeze(self.root)
        with self.assertRaisesRegex(ValueError,'closed'):
            bench.run(self.root,None,'development')


if __name__=='__main__':unittest.main()
