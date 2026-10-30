"""Authored data-only fixtures; no physics, collected video or robot commands."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from dvidia_training import action_demonstration as tape


def fixture(steps=2,kind='authored_fixture'):
    profile,lock_sha,record_sha=tape.profile_contract()
    limits=dict(joint_limits_rad=[[-2.618,2.618],[0.,3.14],[-2.697,0.],[-1.832,1.832],[-1.22,1.22],[-3.14,3.14]],
        gripper_position_limits_m=[0.,.035],joint_target_rate_rad_s=1.5,jaw_target_rate_m_s=.05)
    calibration=dict(id='authored-fixture-camera',base_frame='robot_base',tool_frame='link6',camera_frame='rgbd_optical',
        intrinsics=dict(width=640,height=480,fx=500.,fy=500.,cx=320.,cy=240.),
        base_from_camera=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],depth_units='m',valid_from_seconds=0.,valid_until_seconds=20.)
    binding={name:sha256(('AUTHORED FIXTURE ONLY '+name).encode()).hexdigest() for name in tape.SHA_FIELDS}
    binding.update(profile_lock_sha256=lock_sha,profile_record_sha256=record_sha,
        limits_sha256=tape.digest(limits),calibration_sha256=tape.digest(calibration))
    header=dict(schema=tape.SCHEMA,scope='simulation-only',physical_robot_ready=False,
        provenance=dict(episode_id='authored-example-00',source_recording_id='authored-example-source',session_id='authored-example-session',
            source_group='authored-example-group',kind=kind,producer_revision='authored-data-only-v1',replay_of_sha256=None),
        binding=binding,profile=profile,limits=limits,calibration=calibration,
        clock=dict(kind='shared_simulation_elapsed_seconds',origin='producer_zero',control_dt_seconds=.02,maximum_sensor_age_seconds=.1),
        signal_contract=deepcopy(tape.SIGNALS),goal_position_m=[.35,.08,.08])
    assets={}
    def snapshot(index):
        time=index*.02
        values=dict(joint_position_rad=[index*.001,.5,-.5,0.,0.,0.],joint_velocity_rad_s=[.05,0.,0.,0.,0.,0.],
            gripper_joint_position_m=[.025-min(index,10)*.0001,-.025+min(index,10)*.0001],gripper_joint_velocity_m_s=[-.005,.005],
            object_position_estimate_m=[.30,-.03,.082],object_quaternion_estimate_wxyz=[1.,0.,0.,0.],
            object_position_uncertainty_m=[.001],object_orientation_uncertainty_rad=[.01],
            tcp_position_estimate_m=[.30,-.03,.15],tcp_quaternion_estimate_wxyz=[1.,0.,0.,0.])
        signals={name:dict(status='available',captured_at_seconds=time,available_at_seconds=time,age_seconds=0.,value=value) for name,value in values.items()}
        artifacts={}
        for channel in ('rgb','depth'):
            # These byte identities refer to authored placeholder JSON, never
            # to actual RGB-D images. The public example omits these blobs.
            path=f'frames/{channel}-{index:04}.fixture.json'
            body=(json.dumps(dict(authored_placeholder=True,not_image_pixels=True,index=index,channel=channel))+'\n').encode()
            assets[path]=body;artifacts[channel]=dict(path=path,bytes=len(body),sha256=sha256(body).hexdigest())
        frame=dict(index=index,exposure_start_seconds=time,exposure_end_seconds=time,**artifacts)
        return dict(timestamp_seconds=time,signals=signals,vision_frame_index=index,camera=dict(status='available',captured_at_seconds=time,
            available_at_seconds=time,age_seconds=0.,value=frame))
    initial=snapshot(0);initial_action=dict(joint_targets_rad=[0.,.5,-.5,0.,0.,0.],gripper_position_m=.025)
    recorder=tape.Recorder(header,initial,initial_action)
    previous=initial
    authority={'authored_fixture':'authored_fixture','simulation_teleoperation':'operator','simulation_authored_controller':'authored_controller'}[kind]
    for index in range(steps):
        after=snapshot(index+1);time=index*.02
        bounded_index=min(index,10)
        requested=dict(joint_targets_rad=[.02+bounded_index*.005,.5,-.5,0.,0.,0.],gripper_position_m=.024)
        issued=dict(joint_targets_rad=[.01+bounded_index*.005,.5,-.5,0.,0.,0.],gripper_position_m=.0245)
        reference=dict(joint_targets_rad=[.006+bounded_index*.005,.5,-.5,0.,0.,0.],gripper_position_m=.025)
        ramped=dict(joint_targets_rad=[.003+bounded_index*.002,.5,-.5,0.,0.,0.],gripper_position_m=.0248)
        step=dict(index=index,before=deepcopy(previous),reference=dict(computed_at_seconds=time,progress_seconds=time,action=reference),
            command=dict(issued_at_seconds=time,requested=requested,issued=issued,authority=authority,
                operator_correction=kind=='simulation_teleoperation' and index==1,issue_reason='producer_limiter'),
            applied=dict(interval_start_seconds=time,interval_end_seconds=(index+1)*.02,ramped_targets=ramped,
                native_position_controls=[.004+bounded_index*.002,.5,-.5,0.,0.,0.,.0248],rate_limited=True,scope='final_native_tick_only'),after=after)
        recorder.append(step);previous=after
    return recorder.finish(),assets


def refresh_before(t):
    for index,step in enumerate(t['steps']):step['before']=deepcopy(t['initial'] if index==0 else t['steps'][index-1]['after'])


def write_assets(root,assets):
    for name,body in assets.items():
        path=Path(root)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)


class ActionDemonstrationTests(unittest.TestCase):
    def test_data_only_roundtrip_is_detached_and_never_overwrites(self):
        original,assets=fixture();result=tape.validate_demonstration(original)
        result['initial']['signals']['joint_position_rad']['value'][0]=9.
        self.assertEqual(original['initial']['signals']['joint_position_rad']['value'][0],0.)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'tape.json';receipt=tape.write_demonstration(path,original)
            self.assertEqual(tape.load_demonstration(path),original)
            self.assertEqual(receipt['sha256'],sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):tape.write_demonstration(path,original)

    def test_fixture_and_video_only_cannot_claim_collected_learning_targets(self):
        recording,_=fixture();summary=tape.inspect_demonstration(recording)
        self.assertTrue(summary['motor_command_targets_present']);self.assertFalse(summary['causal_learning_pairs_declared_available'])
        self.assertFalse(summary['corrective_demonstration_quality_verified'])
        with self.assertRaisesRegex(ValueError,'Authored fixture'):
            tape.validate_demonstration(recording,require_targets=True)
        recording['provenance']['kind']='video_only'
        with self.assertRaisesRegex(ValueError,'Video-only'):tape.validate_demonstration(recording)

    def test_target_export_uses_causal_before_and_issued_not_applied_or_after(self):
        recording,assets=fixture(kind='simulation_teleoperation')
        with TemporaryDirectory() as directory:
            write_assets(directory,assets);export=tape.learning_pairs(recording,artifacts=directory)
        row=export['rows'][0];step=recording['steps'][0]
        self.assertEqual(row['before'],step['before'])
        self.assertEqual(row['issued_action_target'],step['command']['issued'])
        self.assertAlmostEqual(row['residual_action_target'][0],.004)
        self.assertNotEqual(row['issued_action_target']['joint_targets_rad'][0],step['applied']['native_position_controls'][0])
        self.assertNotIn('after',row);self.assertNotIn('outcome',row);self.assertNotIn('operator_correction',row)
        self.assertTrue(export['artifact_readback']['byte_hashes_verified']);self.assertFalse(export['artifact_readback']['images_decoded'])
        self.assertFalse(export['source_authenticated']);self.assertFalse(export['physical_robot_ready'])
        self.assertEqual(export['residual_action_units'],['rad']*6+['m'])
        self.assertEqual(export['residual_action_order'],recording['profile']['actuator_names'])
        self.assertEqual(export['source_context']['binding'],recording['binding'])
        self.assertEqual(export['selection_metadata']['outcome']['task_result'],'unverified')
        self.assertNotIn('selection_metadata',export['input_fields'])
        self.assertIn('Not selected successful corrections',export['selection_scope'])

    def test_missing_sensor_is_preserved_without_zero_fill_and_refuses_export(self):
        recording,_=fixture(kind='simulation_teleoperation')
        missing=dict(status='missing',captured_at_seconds=None,available_at_seconds=None,age_seconds=None,value=None)
        recording['steps'][-1]['after']['signals']['gripper_joint_velocity_m_s']=missing
        validated=tape.validate_demonstration(recording)
        self.assertIsNone(validated['steps'][-1]['after']['signals']['gripper_joint_velocity_m_s']['value'])
        with self.assertRaisesRegex(ValueError,'Missing or stale'):tape.validate_demonstration(recording,require_targets=True)
        missing['value']=[0.,0.]
        with self.assertRaisesRegex(ValueError,'null values'):tape.validate_demonstration(recording)

    def test_failed_motor_labels_are_retained_as_non_feature_selection_metadata(self):
        recording,assets=fixture(kind='simulation_authored_controller')
        recording['outcome'].update(task_result='failure',scorer_sha256='1'*64,evidence_sha256='2'*64)
        with TemporaryDirectory() as directory:
            write_assets(directory,assets);result=tape.learning_pairs(recording,artifacts=directory)
        self.assertEqual(result['selection_metadata']['outcome'],recording['outcome'])
        self.assertEqual(result['selection_metadata']['provenance']['kind'],'simulation_authored_controller')
        self.assertEqual(result['selection_metadata']['operator_correction_steps'],0)
        self.assertFalse(result['corrective_demonstration_quality_verified'])
        self.assertEqual(result['input_fields'],['before','reference','goal_position_m'])
        self.assertEqual(len(result['rows']),len(recording['steps']))

    def test_stale_required_measurement_refuses_pairs_without_dropping_recording(self):
        recording,_=fixture(steps=8,kind='simulation_teleoperation')
        initial=deepcopy(recording['initial']['signals']['joint_velocity_rad_s'])
        for step in recording['steps']:
            time=step['after']['timestamp_seconds'];sample=deepcopy(initial)
            sample.update(age_seconds=time,status='available' if time<=.1 else 'stale')
            step['after']['signals']['joint_velocity_rad_s']=sample
        refresh_before(recording)
        self.assertEqual(tape.validate_demonstration(recording)['steps'][-1]['after']['signals']['joint_velocity_rad_s']['status'],'stale')
        with self.assertRaisesRegex(ValueError,'Missing or stale'):tape.validate_demonstration(recording,require_targets=True)

    def test_captured_but_not_yet_arrived_camera_or_sensor_cannot_enter_decision(self):
        for sensor in ('camera','joint_position_rad'):
            recording,_=fixture();observation=recording['steps'][0]['after']
            sample=observation['camera'] if sensor=='camera' else observation['signals'][sensor]
            sample['available_at_seconds']=.03;refresh_before(recording)
            with self.subTest(sensor=sensor),self.assertRaisesRegex(ValueError,'causally precede'):
                tape.validate_demonstration(recording)

    def test_future_exposure_or_changed_reused_frame_identity_is_refused(self):
        recording,_=fixture();recording['initial']['camera']['value']['exposure_end_seconds']=.001;refresh_before(recording)
        with self.assertRaisesRegex(ValueError,'exposure end'):tape.validate_demonstration(recording)
        recording,_=fixture();last=recording['steps'][-1]['after']['camera'];previous=recording['steps'][-2]['after']['camera']
        last.update(captured_at_seconds=previous['captured_at_seconds'],available_at_seconds=previous['available_at_seconds'],
            age_seconds=.02,value=deepcopy(previous['value']))
        last['value']['rgb']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'reused camera frame'):tape.validate_demonstration(recording)

    def test_out_of_order_discontinuous_partial_steps_and_future_reference_refused(self):
        for change in ('index','continuity','interval','reference','applied'):
            recording,_=fixture();step=recording['steps'][1]
            if change=='index':step['index']=0
            elif change=='continuity':step['before']['signals']['joint_position_rad']['value'][0]+=.1
            elif change=='interval':step['after']['timestamp_seconds']=.03
            elif change=='reference':step['reference']['computed_at_seconds']=.021
            else:step['applied']['interval_end_seconds']=.039
            with self.subTest(change=change),self.assertRaises(ValueError):tape.validate_demonstration(recording)

    def test_profile_units_signed_measured_jaws_and_seven_control_count(self):
        recording,_=fixture();validated=tape.validate_demonstration(recording)
        self.assertLess(validated['initial']['signals']['gripper_joint_position_m']['value'][1],0)
        for change in ('legacy-gap','control-count','jaw-range','units'):
            bad=deepcopy(recording)
            if change=='legacy-gap':bad['steps'][0]['command']['issued']['gripper_position_m']=.08
            elif change=='control-count':bad['steps'][0]['applied']['native_position_controls'].append(.0248)
            elif change=='jaw-range':bad['limits']['gripper_position_limits_m']=[0.,.08];bad['binding']['limits_sha256']=tape.digest(bad['limits'])
            else:bad['profile']['native_command_units'][-1]='rad'
            with self.subTest(change=change),self.assertRaises(ValueError):tape.validate_demonstration(bad)

    def test_scene_calibration_and_limits_bindings_are_checked(self):
        for change in ('calibration','rotation','frame','limits','profile'):
            recording,_=fixture()
            if change=='calibration':recording['calibration']['intrinsics']['fx']=501.
            elif change=='rotation':recording['calibration']['base_from_camera'][0]=2.;recording['binding']['calibration_sha256']=tape.digest(recording['calibration'])
            elif change=='frame':recording['calibration']['base_frame']='world';recording['binding']['calibration_sha256']=tape.digest(recording['calibration'])
            elif change=='limits':recording['limits']['jaw_target_rate_m_s']=.06
            else:recording['binding']['profile_lock_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):tape.validate_demonstration(recording)

    def test_rehashing_cannot_widen_known_native_ranges_but_tightening_is_permitted(self):
        recording,_=fixture(kind='simulation_teleoperation')
        recording['limits']['joint_limits_rad'][1]=[-10.,10.]
        recording['binding']['limits_sha256']=tape.digest(recording['limits'])
        recording['steps'][0]['command']['issued']['joint_targets_rad'][1]=-5.
        with self.assertRaisesRegex(ValueError,'never widen'):tape.validate_demonstration(recording,require_targets=True)
        recording,_=fixture();recording['limits']['joint_limits_rad'][1]=[.1,3.]
        recording['binding']['limits_sha256']=tape.digest(recording['limits'])
        self.assertEqual(tape.validate_demonstration(recording)['limits']['joint_limits_rad'][1],[.1,3.])

    def test_ramped_endpoints_can_falsify_declared_caps_without_equating_native_controls(self):
        for dimension in ('arm','jaw'):
            recording,_=fixture()
            if dimension=='arm':recording['steps'][0]['applied']['ramped_targets']['joint_targets_rad'][0]=2.
            else:recording['steps'][0]['applied']['ramped_targets']['gripper_position_m']=.01
            with self.subTest(dimension=dimension),self.assertRaisesRegex(ValueError,'rate bound'):tape.validate_demonstration(recording)
        recording,_=fixture();recording['steps'][0]['applied']['native_position_controls'][0]=.5
        self.assertEqual(tape.validate_demonstration(recording)['steps'][0]['applied']['native_position_controls'][0],.5)
        self.assertFalse(tape.inspect_demonstration(recording)['intermediate_rate_bounds_verified'])

    def test_acquisition_chronology_survives_missing_gaps_without_forward_fill(self):
        for channel in ('joint_velocity_rad_s','camera'):
            recording,_=fixture(steps=4)
            missing=dict(status='missing',captured_at_seconds=None,available_at_seconds=None,age_seconds=None,value=None)
            observation=recording['steps'][1]['after']
            if channel=='camera':
                observation['camera']=deepcopy(missing);observation['vision_frame_index']=None
                for name,contract in tape.SIGNALS.items():
                    if contract['origin']=='declared_rgbd_tracker':observation['signals'][name]=deepcopy(missing)
                later=recording['steps'][2]['after'];old=deepcopy(recording['initial']['camera'])
                old['age_seconds']=later['timestamp_seconds'];later['camera']=old;later['vision_frame_index']=0
                for name,contract in tape.SIGNALS.items():
                    if contract['origin']=='declared_rgbd_tracker':
                        sample=deepcopy(recording['initial']['signals'][name]);sample['age_seconds']=later['timestamp_seconds'];later['signals'][name]=sample
            else:
                observation['signals'][channel]=deepcopy(missing)
                later=recording['steps'][2]['after']['signals'][channel]
                later.update(captured_at_seconds=.01,available_at_seconds=.01,age_seconds=.05)
            refresh_before(recording)
            with self.subTest(channel=channel),self.assertRaisesRegex(ValueError,'missing gaps'):tape.validate_demonstration(recording)

    def test_camera_and_tracker_frame_association_is_explicit(self):
        for change in ('index','capture','arrival'):
            recording,_=fixture();snapshot=recording['steps'][-1]['after']
            if change=='index':snapshot['vision_frame_index']=55
            elif change=='capture':snapshot['signals']['object_position_estimate_m'].update(captured_at_seconds=.03,available_at_seconds=.03,age_seconds=.01)
            else:
                snapshot['camera'].update(captured_at_seconds=.03,available_at_seconds=.039,age_seconds=.01)
                snapshot['camera']['value'].update(exposure_start_seconds=.03,exposure_end_seconds=.03)
                for name,contract in tape.SIGNALS.items():
                    if contract['origin']=='declared_rgbd_tracker':snapshot['signals'][name].update(captured_at_seconds=.03,age_seconds=.01)
                snapshot['signals']['object_position_estimate_m']['available_at_seconds']=.038
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'Tracker'):tape.validate_demonstration(recording)

    def test_native_clock_roundoff_is_preserved_but_real_overshoot_refused(self):
        recording,_=fixture(steps=1000);end=20.00000000000146;last=recording['steps'][-1]
        last['after']['timestamp_seconds']=end
        for sample in list(last['after']['signals'].values())+[last['after']['camera']]:
            sample.update(captured_at_seconds=end,available_at_seconds=end)
        frame=last['after']['camera']['value'];frame.update(exposure_start_seconds=end,exposure_end_seconds=end)
        last['applied']['interval_end_seconds']=end;recording['outcome']['end_seconds']=end
        self.assertEqual(tape.validate_demonstration(recording)['outcome']['end_seconds'],end)
        recording['steps'][-1]['after']['timestamp_seconds']=20.001
        with self.assertRaisesRegex(ValueError,'bounds'):tape.validate_demonstration(recording)

    def test_interrupted_prefix_and_zero_step_attempts_remain_unknown_ineligible(self):
        recording,_=fixture(kind='simulation_teleoperation')
        recording['outcome'].update(status='interrupted',task_result='unknown',reason='producer_interrupted')
        self.assertEqual(tape.validate_demonstration(recording)['outcome']['task_result'],'unknown')
        with self.assertRaisesRegex(ValueError,'Interrupted/partial'):tape.validate_demonstration(recording,require_targets=True)
        recording['steps']=[];recording['outcome']['end_seconds']=0.
        self.assertFalse(tape.inspect_demonstration(recording)['motor_command_targets_present'])
        recording['outcome']['task_result']='failure'
        with self.assertRaisesRegex(ValueError,'unknown'):tape.validate_demonstration(recording)

    def test_task_labels_need_independent_evidence_and_are_not_features(self):
        recording,_=fixture();recording['outcome']['task_result']='success'
        with self.assertRaisesRegex(ValueError,'scorer and evidence'):tape.validate_demonstration(recording)
        recording['outcome'].update(scorer_sha256='1'*64,evidence_sha256='2'*64)
        self.assertEqual(tape.validate_demonstration(recording)['outcome']['task_result'],'success')
        self.assertFalse(tape.inspect_demonstration(recording)['source_authenticated'])

    def test_duplicate_unknown_boolean_nonfinite_values_refused(self):
        for bad in (b'{"schema":1,"schema":2}',b'{"value":NaN}',b'\xff'):
            with self.subTest(raw=bad),self.assertRaises(ValueError):tape.validate_demonstration(bad)
        for change in ('unknown','boolean','infinite'):
            recording,_=fixture()
            if change=='unknown':recording['steps'][0]['command']['extra']='not validated'
            elif change=='boolean':recording['steps'][0]['applied']['interval_start_seconds']=False
            else:recording['steps'][0]['command']['issued']['joint_targets_rad'][0]=float('inf')
            with self.subTest(change=change),self.assertRaises(ValueError):tape.validate_demonstration(recording)

    def test_frame_readback_refuses_changed_missing_or_escaping_files(self):
        recording,assets=fixture(kind='simulation_teleoperation')
        with TemporaryDirectory() as directory:
            with self.assertRaises(OSError):tape.learning_pairs(recording,artifacts=directory)
            write_assets(directory,assets);path=Path(directory)/next(iter(assets));path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'byte identity'):tape.learning_pairs(recording,artifacts=directory)
        recording['initial']['camera']['value']['rgb']['path']='../escape';refresh_before(recording)
        with self.assertRaisesRegex(ValueError,'relative path'):tape.validate_demonstration(recording)

    def test_read_only_replay_check_has_explicit_source_and_separate_units(self):
        reference,_=fixture();replayed=deepcopy(reference)
        replayed['provenance'].update(episode_id='replayed-fixture',source_recording_id='replayed-source',replay_of_sha256=tape.digest(reference))
        replayed['steps'][-1]['after']['signals']['joint_position_rad']['value'][0]+=.001
        result=tape.replay_check(reference,replayed)
        self.assertTrue(result['command_schedule_matches']);self.assertFalse(result['physics_executed']);self.assertFalse(result['robot_commands_sent'])
        self.assertGreater(result['joint_error_rad']['max_abs'],0)
        self.assertIn('applied_joint_reference_error_rad',result);self.assertIn('applied_finger_reference_error_m',result)
        self.assertNotIn('applied_native_error_mixed_units',result)
        replayed['binding']['scene_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'schedule bindings'):tape.replay_check(reference,replayed)

    def test_replay_cannot_compare_different_commands_or_missing_origin(self):
        reference,_=fixture();replayed=deepcopy(reference)
        with self.assertRaisesRegex(ValueError,'schedule bindings'):tape.replay_check(reference,replayed)
        replayed['provenance']['replay_of_sha256']=tape.digest(reference)
        replayed['steps'][0]['command']['issued']['joint_targets_rad'][0]+=.001
        with self.assertRaisesRegex(ValueError,'issued commands'):tape.replay_check(reference,replayed)
        replayed=deepcopy(reference);replayed['provenance']['replay_of_sha256']=tape.digest(reference)
        replayed['initial_applied_targets']['joint_targets_rad'][0]=.001
        with self.assertRaisesRegex(ValueError,'schedule bindings'):tape.replay_check(reference,replayed)

    def test_recorder_seals_once_and_never_records_rejected_transition(self):
        recording,_=fixture();header={key:recording[key] for key in tape.HEADER}
        recorder=tape.Recorder(header,recording['initial'],recording['initial_applied_targets'])
        bad=deepcopy(recording['steps'][0]);bad['index']=4
        with self.assertRaises(ValueError):recorder.append(bad)
        self.assertEqual(recorder.steps,[])
        recorder.append(recording['steps'][0]);result=recorder.finish(status='interrupted',task_result='unknown',reason='producer_stopped')
        self.assertEqual(len(result['steps']),1)
        with self.assertRaises(ValueError):recorder.append(recording['steps'][1])
        with self.assertRaises(ValueError):recorder.finish()

    def test_modified_issue_reason_requires_honest_authority(self):
        for reason in ('unchanged','operator_override'):
            recording,_=fixture();recording['steps'][0]['command']['issue_reason']=reason
            with self.subTest(reason=reason),self.assertRaises(ValueError):tape.validate_demonstration(recording)

    def test_cli_example_is_honest_and_targets_refuse_authored_fixture(self):
        example=Path(__file__).parents[1]/'examples/action-demonstration-v01/authored-example.json'
        self.assertEqual(tape.load_demonstration(example),fixture()[0])
        for command in ('validate','inspect'):
            result=subprocess.run([sys.executable,'-m','dvidia_training.action_demonstration',command,str(example)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);report=json.loads(result.stdout)
            self.assertFalse(report['physical_robot_ready']);self.assertFalse(report['source_authenticated'])
            self.assertFalse(report['causal_learning_pairs_declared_available']);self.assertFalse(report['frame_artifacts_verified'])
        with TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,'-m','dvidia_training.action_demonstration','targets',str(example),'--artifacts',directory],capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr);report=json.loads(result.stdout)
            self.assertEqual(report['status'],'refused');self.assertIn('Authored fixture',report['reason'])
            self.assertFalse(report['physics_executed']);self.assertFalse(report['robot_commands_sent'])

    def test_cli_replay_compares_two_files_without_executing_controls(self):
        reference,_=fixture();replayed=deepcopy(reference)
        replayed['provenance'].update(episode_id='authored-replay',replay_of_sha256=tape.digest(reference))
        with TemporaryDirectory() as directory:
            a=Path(directory)/'reference.json';b=Path(directory)/'replayed.json'
            tape.write_demonstration(a,reference);tape.write_demonstration(b,replayed)
            result=subprocess.run([sys.executable,'-m','dvidia_training.action_demonstration','replay-check',str(a),'--replayed',str(b)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);report=json.loads(result.stdout)
        self.assertTrue(report['initial_ramped_targets_match']);self.assertTrue(report['command_schedule_matches'])
        self.assertFalse(report['physics_executed']);self.assertFalse(report['robot_commands_sent'])

    def test_file_symlinks_and_optional_native_imports_are_refused(self):
        recording,_=fixture()
        with TemporaryDirectory() as directory:
            path=Path(directory)/'tape.json';tape.write_demonstration(path,recording);link=Path(directory)/'link.json';link.symlink_to(path)
            with self.assertRaises(ValueError):tape.load_demonstration(link)
        code='''import builtins
original=builtins.__import__
def guarded(name,*a,**kw):
 if name.split('.')[0] in ('numpy','mujoco'):raise RuntimeError('Native import forbidden')
 return original(name,*a,**kw)
builtins.__import__=guarded
from dvidia_training.action_demonstration import inspect_demonstration
from tests.test_action_demonstration import fixture
assert inspect_demonstration(fixture()[0])['physical_robot_ready'] is False
'''
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
