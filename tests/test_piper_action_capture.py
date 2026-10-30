"""Data-only producer failure tests; synthetic doubles never claim native runs."""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from dvidia_training import action_demonstration as tape, piper_action_capture as capture
from tests.test_action_demonstration import fixture


class FakeSession:
    """Authored test double: no renderer, physical robot or native stepping."""
    failure=None

    def __init__(self, assets, config):
        self.config=config;self.position=0;self.closed=False;self.physics_attempted=False
        self.recording,self.assets=fixture(steps=round(config.duration_s/tape.DT),kind='simulation_authored_controller')
        self.recording['calibration']['valid_until_seconds']=config.duration_s
        self.recording['binding']['calibration_sha256']=tape.digest(self.recording['calibration'])

    def open(self):
        if self.failure=='open':raise RuntimeError('Authored setup failure; no native run.')
        return ({key:deepcopy(self.recording[key]) for key in tape.HEADER},deepcopy(self.recording['initial_applied_targets']),dict(runtime='authored_test_double'))

    def sample(self,index):
        if self.failure=='sample' and index==3:raise RuntimeError('Authored camera failure.')
        snapshot=deepcopy(self.recording['initial'] if index==0 else self.recording['steps'][index-1]['after'])
        paths=[snapshot['camera']['value'][channel]['path'] for channel in ('rgb','depth')]
        return snapshot,{name:self.assets[name] for name in paths},dict(authored_test_double=True),dict(render_encode_wall_seconds=.001)

    def decide(self):
        return deepcopy(self.recording['steps'][self.position]['command']['issued']),dict(authored_test_double=True)

    def advance(self,action):
        self.position+=1
        if self.failure=='advance' and self.position==3:raise RuntimeError('Authored partial native interval failure.')
        return deepcopy(self.recording['steps'][self.position-1]['applied'])

    def status(self):
        return dict(physics_attempted=False,native_initialized=False,actual_native_elapsed_seconds=self.position*tape.DT,
            actual_native_data_time_seconds=None,native_info=None,test_double=True)

    def close(self):self.closed=True


def camera():
    return SimpleNamespace(forward=[0,0,-1],up=[0,1,0],pos=[0,0,1],frustum_near=.01,
        frustum_top=.005,frustum_bottom=-.005,frustum_center=0.,orthographic=False)


class PiperActionCaptureTests(unittest.TestCase):
    def setUp(self):FakeSession.failure=None

    def test_config_budget_units_and_engineering_interruption_bounds(self):
        self.assertEqual(capture.CaptureConfig(duration_s=20.).width,160)
        for change in ({'duration_s':True},{'duration_s':20.02},{'duration_s':.11},{'duration_s':float('nan')},
                {'duration_s':10**1000},
                {'mode':'operator'},{'width':True},{'height':4096},{'seed':True},
                {'interrupt_after':-1},{'interrupt_after':10},{'interrupt_after':True},
                {'duration_s':20.,'width':320,'height':240}):
            with self.subTest(change=change),self.assertRaises(ValueError):capture.CaptureConfig(**change)

    def test_optical_axes_pixel_centres_and_nonidentity_robot_base(self):
        calibration=capture.optical_calibration(camera(),base_position=[.2,.3,.4],
            base_rotation=[[0,-1,0],[1,0,0],[0,0,1]],width=160,height=120,duration_s=.2)
        transform=np.array(calibration['base_from_camera']).reshape(4,4)
        np.testing.assert_allclose(transform[:3,:3],[[0,-1,0],[-1,0,0],[0,0,-1]],atol=1e-12)
        np.testing.assert_allclose(transform[:3,3],[-.3,.2,.6],atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(transform[:3,:3]),1.)
        self.assertEqual(calibration['intrinsics'],dict(width=160,height=120,fx=120.,fy=120.,cx=80.,cy=60.))
        for change in ('orthographic','offcentre','degenerate'):
            bad=camera()
            if change=='orthographic':bad.orthographic=True
            elif change=='offcentre':bad.frustum_center=.01
            else:bad.up=[0,0,-1]
            with self.subTest(change=change),self.assertRaises(ValueError):capture.optical_calibration(bad,base_position=[0,0,0],base_rotation=np.eye(3),width=160,height=120,duration_s=.2)

    def test_completed_flow_preserves_boundaries_artifacts_and_issued_vs_applied(self):
        with TemporaryDirectory() as directory,patch.object(capture,'_NativeSession',FakeSession):
            root=Path(directory)/'completed';report=capture.capture('unused',root,config=capture.CaptureConfig(duration_s=.1))
            self.assertEqual(report['status'],'complete');self.assertEqual(report['started_intervals'],5)
            self.assertEqual(report['recorded_complete_intervals'],5);self.assertEqual(report['uncommitted_native_intervals'],0)
            recording=tape.load_demonstration(root/'recording.json')
            self.assertEqual(recording['steps'][0]['command']['issued'],FakeSession('unused',capture.CaptureConfig(duration_s=.1)).recording['steps'][0]['command']['issued'])
            self.assertNotEqual(recording['steps'][0]['command']['issued']['joint_targets_rad'][0],recording['steps'][0]['applied']['native_position_controls'][0])
            for index,step in enumerate(recording['steps']):
                self.assertEqual(step['before']['timestamp_seconds'],index*.02)
                self.assertEqual(step['command']['issued_at_seconds'],step['before']['timestamp_seconds'])
                self.assertEqual(step['after']['camera']['captured_at_seconds'],step['after']['timestamp_seconds'])
                self.assertTrue((root/f'decisions/{index:04}.json').is_file())
                self.assertTrue((root/f'transitions/{index:04}.json').is_file())
            proof=capture.inspect_capture(root)
            self.assertTrue(proof['frames']['byte_hashes_verified']);self.assertFalse(proof['physics_executed'])
            self.assertFalse(report['task_success_verified']);self.assertFalse(report['strict_learning_targets_available'])
            before={str(path.relative_to(root)):sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file()}
            with self.assertRaises(FileExistsError):capture.capture('unused',root)
            self.assertEqual(before,{str(path.relative_to(root)):sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file()})

    def test_declared_interruption_before_third_action_preserves_two_unknown_steps(self):
        with TemporaryDirectory() as directory,patch.object(capture,'_NativeSession',FakeSession):
            root=Path(directory)/'interrupted';report=capture.capture('unused',root,config=capture.CaptureConfig(duration_s=.1,interrupt_after=2))
            self.assertEqual(report['status'],'interrupted');self.assertEqual(report['started_intervals'],2)
            self.assertEqual(report['recorded_complete_intervals'],2)
            self.assertFalse((root/'decisions/0002.json').exists())
            recording=tape.load_demonstration(root/'recording.json')
            self.assertEqual(recording['outcome']['task_result'],'unknown')
            self.assertEqual(recording['outcome']['end_seconds'],.04)
            with self.assertRaisesRegex(ValueError,'Interrupted/partial'):tape.validate_demonstration(recording,require_targets=True)
            self.assertEqual(capture.inspect_capture(root)['status'],'interrupted')

    def test_failed_step_or_camera_retains_only_complete_prefix_not_fabricated_after(self):
        for failure in ('advance','sample'):
            with self.subTest(failure=failure),TemporaryDirectory() as directory,patch.object(capture,'_NativeSession',FakeSession):
                FakeSession.failure=failure;root=Path(directory)/failure
                report=capture.capture('unused',root,config=capture.CaptureConfig(duration_s=.1))
                self.assertEqual(report['status'],'interrupted');self.assertEqual(report['started_intervals'],3)
                self.assertEqual(report['recorded_complete_intervals'],2);self.assertEqual(report['uncommitted_native_intervals'],1)
                self.assertTrue((root/'decisions/0002.json').exists());self.assertFalse((root/'transitions/0002.json').exists())
                recording=tape.load_demonstration(root/'recording.json')
                self.assertEqual(recording['outcome']['end_seconds'],.04);self.assertEqual(recording['outcome']['task_result'],'unknown')
                self.assertGreater(report['native_status']['actual_native_elapsed_seconds'],.04)
                capture.inspect_capture(root)

    def test_setup_refusal_and_unsealed_hard_stop_cannot_claim_complete(self):
        with TemporaryDirectory() as directory,patch.object(capture,'_NativeSession',FakeSession):
            FakeSession.failure='open';root=Path(directory)/'setup-refused';report=capture.capture('unused',root)
            self.assertEqual(report['status'],'refused');self.assertEqual(report['recorded_complete_intervals'],0)
            self.assertFalse((root/'recording.json').exists());self.assertEqual(capture.inspect_capture(root)['status'],'refused')
            hard=Path(directory)/'hard-stop';hard.mkdir();(hard/'started.json').write_text('{}')
            with self.assertRaises(OSError):capture.inspect_capture(hard)

    def test_setup_error_does_not_embed_private_asset_paths(self):
        with TemporaryDirectory() as directory,patch.object(capture,'_NativeSession',FakeSession),patch.object(FakeSession,'open',side_effect=FileNotFoundError(2,'No such file or directory','/private/example/assets/model.xml')):
            root=Path(directory)/'refused';report=capture.capture('/private/example/assets',root)
            self.assertEqual(report['error']['message'],'No such file or directory')
            self.assertNotIn('/private/example',json.dumps(report))

    def test_inspector_refuses_missing_changed_extra_or_symlinked_evidence(self):
        for change in ('changed','missing','extra','nested-integrity','symlink'):
            with self.subTest(change=change),TemporaryDirectory() as directory,patch.object(capture,'_NativeSession',FakeSession):
                root=Path(directory)/'capture';capture.capture('unused',root,config=capture.CaptureConfig(duration_s=.1))
                path=next((root/'frames').iterdir())
                if change=='changed':path.write_bytes(b'changed')
                elif change=='missing':path.unlink()
                elif change=='extra':(root/'unexpected.txt').write_text('extra')
                elif change=='nested-integrity':(root/'native/integrity.json').write_text('{}')
                else:
                    target=Path(directory)/'outside';path.rename(target);path.symlink_to(target)
                with self.assertRaises((ValueError,OSError)):capture.inspect_capture(root)

    def test_real_sample_adapter_uses_actual_encoder_jaw_velocity_and_keeps_tracker_missing(self):
        config=capture.CaptureConfig(width=64,height=64)
        class Renderer:
            scene=SimpleNamespace(camera=[camera()]);depth=False
            def disable_depth_rendering(self):self.depth=False
            def enable_depth_rendering(self):self.depth=True
            def update_scene(self,*args,**kwargs):pass
            def render(self):return np.full((64,64),.75,dtype=np.float32) if self.depth else np.full((64,64,3),42,dtype=np.uint8)
        session=capture._NativeSession('unused',config);session.np=np;session.renderer=Renderer();session.camera='fake-camera'
        def warning_context(warnings):
            if not warnings:warnings.append('Authored renderer capability warning, not a physics warning.')
            return nullcontext()
        session.native_origin=.5;session.sim=SimpleNamespace(native_warnings=warning_context)
        session.base=0;session.tool=1
        data=SimpleNamespace(time=.5,qpos=np.array([0.,.5,-.5,0.,0.,0.,.025,-.025]+[0.]*7),
            qvel=np.array([0.]*6+[-.003,.004]+[0.]*6),xpos=np.array([[0.,0.,0.],[.1,.2,.3]]),xmat=np.array([np.eye(3).ravel(),np.eye(3).ravel()]))
        session.env=SimpleNamespace(data=data,elapsed=0.)
        session.mj=SimpleNamespace(mju_mat2Quat=lambda target,rotation:target.__setitem__(slice(None),[1.,0.,0.,0.]))
        session.calibration=session._calibration();session.info=dict(valid=True);session.observation=dict(object_position_m=[9.,9.,9.],scope='privileged test truth')
        snapshot,frames,diagnostic,timing=session.sample(0)
        self.assertEqual(snapshot['signals']['gripper_joint_velocity_m_s']['value'],[-.003,.004])
        self.assertEqual(snapshot['signals']['gripper_joint_position_m']['value'],[.025,-.025])
        self.assertEqual(snapshot['signals']['tcp_position_estimate_m']['value'],[.1,.2,.3])
        for name,contract in tape.SIGNALS.items():
            if contract['origin']=='declared_rgbd_tracker':self.assertEqual(snapshot['signals'][name],capture._missing())
        self.assertEqual(diagnostic['policy_observation']['object_position_m'],[9.,9.,9.])
        self.assertTrue(frames['frames/rgb-0000.png'].startswith(b'\x89PNG\r\n\x1a\n'))
        depth=np.load(io.BytesIO(frames['frames/depth-0000.npy']),allow_pickle=False)
        self.assertEqual(depth.shape,(64,64));self.assertEqual(depth.dtype,np.dtype('<f4'));self.assertTrue(np.all(depth==.75))
        self.assertEqual(timing['renderer_warnings'],session.renderer_warnings)
        self.assertFalse(session.status()['depth_accuracy_verified'])

    def test_actual_advance_refuses_partial_intervals_and_records_native_controls_separately(self):
        session=capture._NativeSession('unused',capture.CaptureConfig());session.env=SimpleNamespace(elapsed=0.,warnings=[],command_targets=np.array([.003,.5,-.5,0.,0.,0.]),command_gripper=.0248,data=SimpleNamespace(ctrl=np.array([.004,.5,-.5,0.,0.,0.,.0248])))
        def step(action):session.env.elapsed+=.02;return {},dict(valid=True)
        session.env.step=step;action=dict(joint_targets_rad=[.01,.5,-.5,0.,0.,0.],gripper_position_m=.0245)
        result=session.advance(action)
        self.assertEqual(result['ramped_targets']['joint_targets_rad'][0],.003);self.assertEqual(result['native_position_controls'][0],.004)
        session.env.step=lambda action:({},dict(valid=True))
        with self.assertRaisesRegex(RuntimeError,'incomplete'):session.advance(action)


if __name__=='__main__':unittest.main()
