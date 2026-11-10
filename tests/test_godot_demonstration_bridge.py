"""Transport/authority tests with test doubles; no native execution."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from dvidia_training import godot_demonstration_bridge as bridge

DIGEST='a'*64
HOME=[.3,0.,.25]


class Capture:
    def __init__(self,index):
        self.root=Path('/not-written')/str(index)
        self.config=SimpleNamespace(duration_s=.1)
        self.source_kind='simulation_scripted_input'
        self.session=SimpleNamespace(env=SimpleNamespace(elapsed=0.))
        self.events=[];self.actions=[];self.reasons=[];self.fail=False;self.finish_error=False
    def open(self):
        return dict(header=dict(binding=dict(model_sha256=DIGEST)))
    def context(self):
        return dict(snapshot=dict(signals=dict(joint_position_rad=dict(value=[0.]*6))),
                    native_truth='must not reach controller')
    def step(self,action,*,input_event=None):
        self.session.env.elapsed+=.02
        if self.fail:raise RuntimeError('partial native advance')
        self.actions.append((deepcopy(action),deepcopy(input_event)))
        return dict(done=len(self.actions)>=5)
    def record_event(self,name,detail=None):self.events.append(name)
    def finish(self,reason='operator_stopped'):
        self.reasons.append(reason)
        return dict(reason=reason,sealed=True,native_evidence_saved=not self.finish_error,
                    error='disk failure' if self.finish_error else None)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.captures=[];self.calls=[]
        def factory(index):
            capture=Capture(index);self.captures.append(capture);return capture
        def controller(joints,target,jaw):
            self.calls.append((joints,target,jaw))
            return dict(joint_targets_rad=joints,gripper_position_m=jaw)
        self.session=bridge.Session(factory,controller,model_sha256=DIGEST,home_target=HOME,
            max_episodes=2,state_reader=lambda native,target:dict(time=native.env.elapsed,target=target))
    def call(self,kind,**kw):
        if kind not in ('hello','ping'):kw.setdefault('sequence',self.session.expected_sequence)
        return self.session.process(json.dumps(dict(type=kind,**kw)).encode())
    def hello(self):return self.call('hello',schema=bridge.SCHEMA,model_sha256=DIGEST)
    def step(self,**kw):return self.call('step',target=kw.pop('target',HOME),jaw=kw.pop('jaw',.035),**kw)
    def test_explicit_record_and_causal_controller_inputs(self):
        self.assertEqual(self.step()['type'],'error')
        self.assertEqual(self.hello()['recording']['status'],'ready')
        self.assertEqual(self.step()['code'],'recording_status')
        self.call('begin')
        response=self.step()
        self.assertEqual(response['state']['time'],.02)
        self.assertEqual(self.calls,[([0.]*6,HOME,.035)])
        action,event=self.captures[0].actions[0]
        self.assertEqual(event['target_world_m'],HOME)
        self.assertEqual(action['gripper_position_m'],.035)
    def test_bad_requests_never_advance_or_consume_sequence(self):
        self.hello();self.call('begin');sequence=self.session.expected_sequence
        cases=[dict(target=[.1,.2]),dict(target=[100,0,.2]),dict(jaw=True),dict(jaw=.1),dict(sequence=0)]
        for case in cases:
            self.assertEqual(self.step(**case)['type'],'error')
            self.assertEqual(self.session.expected_sequence,sequence)
        for raw in (b'{"type":"step","type":"ping"}',b'[]',b'{"type":NaN}'):
            self.assertEqual(self.session.process(raw)['type'],'error')
        self.assertEqual(self.captures[0].session.env.elapsed,0.)
        self.assertEqual(self.calls,[])
    def test_pause_ping_resume_and_finish_do_not_advance(self):
        self.hello();self.call('begin');self.step()
        self.call('pause');sequence=self.session.expected_sequence
        self.assertEqual(self.step()['code'],'recording_status')
        self.assertEqual(self.call('ping')['expected_sequence'],sequence)
        self.assertEqual(self.session.expected_sequence,sequence)
        self.call('resume');self.call('finish')
        self.assertEqual(self.captures[0].session.env.elapsed,.02)
        self.assertEqual(self.captures[0].events,['begin','pause','resume'])
        self.assertEqual(self.session.status,'finished')
        self.call('finish')
        self.assertEqual(self.captures[0].reasons,['operator_stopped'])
    def test_reset_seals_old_episode_and_budget_refuses_without_mutation(self):
        self.hello();self.call('begin');self.step();response=self.call('reset')
        self.assertEqual(response['state']['time'],0.)
        self.assertEqual(response['recording']['status'],'ready')
        self.assertEqual(self.captures[0].reasons,['operator_reset'])
        self.assertNotEqual(self.captures[0].root,self.captures[1].root)
        self.assertEqual(self.call('reset')['code'],'episode_budget')
        self.assertEqual(self.captures[1].reasons,[])
    def test_partial_native_interval_faults_and_never_retries(self):
        self.hello();self.call('begin');self.captures[0].fail=True
        self.assertEqual(self.step()['code'],'recording_fault')
        self.assertEqual(self.step()['code'],'session_fault')
        self.assertEqual(self.captures[0].session.env.elapsed,.02)
        self.assertEqual(self.captures[0].reasons,['session_fault'])
    def test_budget_finish_and_disconnect_have_truthful_authority(self):
        self.hello();self.call('begin')
        for _ in range(5):self.step()
        self.assertEqual(self.session.status,'finished')
        self.assertEqual(self.captures[0].reasons,['capture_budget_complete'])
        self.session.disconnect()
        self.assertEqual(len(self.captures[0].reasons),1)
        self.call('reset');self.session.disconnect()
        self.assertEqual(self.captures[1].events,['disconnect'])
        self.assertEqual(self.captures[1].reasons,['client_disconnected'])
    def test_wrong_model_and_protocol_are_rejected(self):
        for values in (dict(schema='old',model_sha256=DIGEST),dict(schema=bridge.SCHEMA,model_sha256='b'*64)):
            self.assertEqual(self.call('hello',**values)['type'],'error')
        self.assertFalse(self.session.ready)
        self.assertEqual(self.captures[0].actions,[])

    def test_failed_sealing_is_not_reported_as_saved_and_reset_does_not_hide_it(self):
        self.hello();self.call('begin');self.step()
        self.captures[0].finish_error=True
        self.assertEqual(self.call('reset')['code'],'recording_fault')
        self.assertTrue(self.session.faulted)
        self.assertEqual(len(self.captures),1)
        self.assertEqual(self.step()['code'],'session_fault')


if __name__=='__main__':unittest.main()
