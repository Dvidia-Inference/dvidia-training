"""Numeric reconstruction and deliberate tampering; no native physics."""
from copy import deepcopy
from dataclasses import asdict
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from publishing import audit_piper_grip_wait_v04 as audit
from dvidia_training import piper_grip_clock as frozen
from dvidia_training import piper_grip_clock_model as clock_model
from dvidia_training import piper_grip_wait_v04 as runtime
from tests.test_piper_grip_clock_runtime import changing_path_model
from tests.test_piper_feedback_model import observation


class IndependentGripWaitAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=changing_path_model();cls.jaw=frozen.reference_jaw_manifest(cls.base)
        cls.model=clock_model.train(dict(features=np.zeros((8,58)),targets=np.zeros(8),
            groups=['fixture-a']*4+['fixture-b']*4),base_sha256=audit.previous._canonical_hash(cls.base),
            jaw_sha256=audit.previous._canonical_hash(cls.jaw),jaw_mode='coordinated_reference',min_leaf=2)

    def recorded(self,mode='secure',budget=.20):
        config=asdict(runtime.GripWaitConfig(maximum_wait_s=budget))
        policy=runtime.GuardedLearnedClock(self.base,self.jaw,self.model,config=runtime.GripWaitConfig(**config))
        episode=dict(policy_kind='guarded',trajectory=[],actions=[])
        for i in range(61):
            state=observation(i*.02,contacts=1 if i>=25 else 0)
            if i>=25:
                force=.1 if mode in ('timeout','support-loss') else .2
                state['pad_normal_forces_n']=dict(left=force,right=force)
            if mode=='support-loss' and i>=28:state['table_object_contact_count']=0
            if mode=='never-eligible' and i>=25:
                state['grasp_contact_counts']['right']=0
                state['table_object_contact_count']=0 if i>=28 else 1
            if i==60:
                state.update(issued_joint_targets_rad=None,issued_gripper_position_m=None,command=None)
                episode['trajectory'].append(state);break
            action=policy(state)
            record=dict(t_seconds=state['t_seconds'],**action,clock_decision=deepcopy(policy.last_decision))
            state.update(issued_joint_targets_rad=deepcopy(action['joint_targets_rad']),
                         issued_gripper_position_m=action['gripper_position_m'],command=deepcopy(record))
            episode['trajectory'].append(state);episode['actions'].append(record)
        episode['policy_diagnostics']=policy.diagnostics()
        return episode,config

    def inspect(self,episode,config):return audit.inspect_guarded_trace(episode,self.base,self.model,config)

    def mismatch_count(self,trace):return sum(v for k,v in trace.items() if k.endswith('_mismatches'))

    def test_valid_sampled_ready_latch_reconstructs_with_zero_mismatches(self):
        episode,config=self.recorded()
        trace=self.inspect(episode,config)
        self.assertEqual(self.mismatch_count(trace),0,trace)
        self.assertTrue(trace['reconstructed_acquisition_released'])
        self.assertFalse(trace['reconstructed_aborted'])
        self.assertEqual(trace['reconstructed_wait_ticks'],5)

    def test_timeout_and_support_loss_reconstruct_independently(self):
        for mode,reason in [('timeout','wait_budget_exhausted'),('support-loss','support_lost_before_secure'),
                            ('never-eligible','support_lost_before_secure')]:
            with self.subTest(mode=mode):
                episode,config=self.recorded(mode)
                trace=self.inspect(episode,config)
                self.assertEqual(self.mismatch_count(trace),0,trace)
                self.assertTrue(trace['reconstructed_aborted'])
                self.assertEqual(trace['reconstructed_abort_reason'],reason)
        episode,config=self.recorded('timeout',budget=.50)
        trace=self.inspect(episode,config)
        self.assertEqual(trace['reconstructed_wait_ticks'],25)
        self.assertEqual(self.mismatch_count(trace),0,trace)

    def test_forged_readiness_or_abort_fields_do_not_drive_reconstruction(self):
        for mode,key,value in [('secure','secure_grip',True),('timeout','aborted',False),
                               ('timeout','reason','secure_grip_latched')]:
            with self.subTest(key=key):
                episode,config=self.recorded(mode)
                index=25 if key=='secure_grip' else 40
                episode['actions'][index]['clock_decision'][key]=value
                trace=self.inspect(episode,config)
                self.assertGreater(trace['supervisor_metadata_mismatches'],0)
                self.assertEqual(trace['reconstructed_aborted'],mode=='timeout')

    def test_forged_final_abort_diagnostic_cannot_turn_timeout_into_success(self):
        episode,config=self.recorded('timeout')
        episode['policy_diagnostics'].update(aborted=False,abort_reason=None,wait_ticks=0)
        trace=self.inspect(episode,config)
        self.assertTrue(trace['reconstructed_aborted'])
        self.assertGreaterEqual(trace['diagnostic_mismatches'],3)

    def test_post_abort_arm_motion_or_jaw_opening_is_detected(self):
        for channel in ('arm','jaw'):
            with self.subTest(channel=channel):
                episode,config=self.recorded('timeout')
                if channel=='arm':episode['actions'][40]['joint_targets_rad'][0]+=.02
                else:episode['actions'][40]['gripper_position_m']+=.001
                trace=self.inspect(episode,config)
                self.assertEqual(trace['command_mismatches'],1)
                self.assertGreater(trace['state_action_binding_mismatches'],0)

    def test_nonbinary_clock_or_forged_causal_features_are_detected(self):
        for mode in ('increment','feature'):
            with self.subTest(mode=mode):
                episode,config=self.recorded()
                decision=episode['actions'][27]['clock_decision']
                if mode=='increment':decision['increment_s']=.01
                else:decision['features'][9]+=.1
                trace=self.inspect(episode,config)
                self.assertGreater(trace['progress_mismatches'] if mode=='increment' else trace['feature_mismatches'],0)

    def test_different_budget_cannot_authenticate_a_recorded_timeout(self):
        episode,config=self.recorded('timeout')
        config['maximum_wait_s']=.36
        trace=self.inspect(episode,config)
        self.assertGreater(trace['supervisor_metadata_mismatches'],0)
        self.assertGreater(trace['diagnostic_mismatches'],0)
        self.assertEqual(trace['reconstructed_wait_ticks'],18)

    def test_prefix_trace_is_preserved_but_not_labeled_complete(self):
        episode,config=self.recorded()
        trace=self.inspect(episode,config)
        self.assertEqual(trace['action_count'],60)
        self.assertFalse(trace['complete_20s_action_trace'])
        self.assertEqual(self.mismatch_count(trace),0)

    def test_auditor_imports_no_v04_runtime_or_lifecycle(self):
        source=Path(audit.__file__).read_text()
        parsed=ast.parse(source)
        modules=[]
        for node in ast.walk(parsed):
            if isinstance(node,ast.ImportFrom):modules.append(node.module or '')
            elif isinstance(node,ast.Import):modules.extend(x.name for x in node.names)
        self.assertFalse(any('piper_grip_wait_v04' in x for x in modules),modules)

    def test_duplicate_json_keys_and_symlink_inputs_are_rejected(self):
        with self.assertRaises(ValueError):audit.previous._json(b'{"aborted":true,"aborted":false}')
        with TemporaryDirectory() as folder:
            path=Path(folder);(path/'data.json').write_text('{}');(path/'linked.json').symlink_to(path/'data.json')
            with self.assertRaises(ValueError):audit._read(path/'linked.json')


if __name__=='__main__':unittest.main()
