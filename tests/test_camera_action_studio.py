"""Real loopback API tests with data-only injected workers; no native physics."""
from hashlib import sha256
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from dvidia_training import camera_action_studio as robot
from dvidia_training.training_studio import configure
from tests.test_training_studio import artifact_run, artifact_inspect, prepare_data


def fake_check(study, assets):
    root = Path(study)
    state = (root/'state.txt').read_text()
    return dict(schema=robot.SCHEMA,
        binding=dict(study=str(root.resolve()), assets=str(Path(assets).resolve()),
            protocol_sha256=sha256((root/'protocol.json').read_bytes()).hexdigest(), assets_sha256='a'*64),
        inspection=dict(status=state, next_actions=[] if state!='ready' else [dict(command='collect',stage='diagnostic',requires_assets=True)],
            stage_summaries={}, report=dict(qualified=False, qualification='research_candidate'),
            blocked_stages=['diagnostic'] if state=='blocked' else []),
        install_available=state=='complete', selected_model=str(root/'model.json') if state=='complete' else None,
        physical_robot_ready=False)


def fake_worker(kind, study, assets, output, expected_binding=None, installation=None):
    result = fake_check(study, assets)
    if expected_binding is not None and result['binding'] != expected_binding:
        raise ValueError('Inputs changed')
    if kind == 'run':
        (Path(study)/'state.txt').write_text('complete')
        result = fake_check(study, assets)
    status = 'checked' if kind=='check' else 'complete' if kind=='run' else 'installed'
    if result['inspection']['status']=='blocked': status='blocked'
    result['status'] = status
    if kind=='install':
        destination = Path(installation); destination.mkdir(parents=True)
        (destination/'test-only.txt').write_text('No real model installation')
        result['installation'] = dict(path=str(destination),record={'test_only':True})
    return result


class CameraStudioHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.study=self.root/'study'; self.study.mkdir(); (self.study/'protocol.json').write_text('{"test":true}')
        (self.study/'state.txt').write_text('ready')
        self.assets=self.root/'assets'; self.assets.mkdir()
        self.events=[]; self.servers=[]
        self.server=self.make_server()

    def make_server(self, worker=fake_worker, runner=artifact_run):
        server=configure(self.root/('server'+str(len(self.servers))),port=0,
            runner=runner,preparer=prepare_data,inspector=artifact_inspect,
            robot_worker=worker,robot_checker=fake_check)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
        thread.start();self.servers.append((server,thread));return server

    def tearDown(self):
        for event in self.events:event.set()
        for server,thread in self.servers:
            server.shutdown();server.server_close();thread.join(timeout=2)
        self.temp.cleanup()

    def request(self,path,body=None,*,method='POST',headers=None,server=None):
        server=server or self.server;port=server.server_address[1]
        connection=http.client.HTTPConnection('127.0.0.1',port,timeout=3)
        h={'Content-Type':'application/json','Origin':f'http://127.0.0.1:{port}'};h.update(headers or {})
        connection.request(method,path,body=None if body is None else json.dumps(body),headers=h)
        response=connection.getresponse();raw=response.read();code=response.status;mime=response.getheader('Content-Type','');connection.close()
        return code,json.loads(raw) if mime.startswith('application/json') else raw

    def payload(self):return dict(study=str(self.study),assets=str(self.assets))

    def done(self,server=None):
        for _ in range(200):
            status,state=self.request('/api/robot/state',method='GET',server=server)
            self.assertEqual(status,200)
            if state['status']!='running':return state
            time.sleep(.01)
        self.fail('Injected worker did not finish')

    def checked(self,server=None):
        status,body=self.request('/api/robot/check',self.payload(),server=server)
        self.assertEqual(status,202,body);return self.done(server)

    def test_simple_ui_and_empty_state_do_not_claim_robot_readiness(self):
        code,html=self.request('/robot',method='GET')
        self.assertEqual(code,200);self.assertIn(b'Teach an arm, then test it.',html)
        self.assertIn(b'id="install" disabled',html);self.assertIn(b'Plain videos alone',html)
        self.assertNotIn(b'innerHTML',html)
        code,footage=self.request('/',method='GET');self.assertIn(b'href="/robot"',footage)
        state=self.done();self.assertFalse(state['can_install']);self.assertFalse(state['can_run'])
        self.assertFalse(state['physical_robot_ready'])

    def test_existing_host_origin_and_cross_site_boundary_applies_to_robot_routes(self):
        for route,method,body in (('/robot','GET',None),('/api/robot/state','GET',None),('/api/robot/check','POST',self.payload())):
            for headers in ({'Host':'evil.example'},{'Origin':'https://evil.example'},{'Sec-Fetch-Site':'cross-site'}):
                code,_=self.request(route,body,method=method,headers=headers)
                self.assertEqual(code,400)
        self.assertIsNone(self.server.robot.job)

    def test_check_run_install_uses_same_worker_and_installs_outside_study(self):
        state=self.checked();self.assertTrue(state['checked']);self.assertTrue(state['can_run']);self.assertFalse(state['can_install'])
        code,_=self.request('/api/robot/run',self.payload());self.assertEqual(code,202)
        state=self.done();self.assertEqual(state['status'],'complete');self.assertTrue(state['can_install'])
        code,_=self.request('/api/robot/install',self.payload());self.assertEqual(code,202)
        state=self.done();self.assertEqual(state['status'],'installed')
        installation=Path(state['summary']['installation']['path'])
        self.assertTrue(installation.is_relative_to(self.server.directory/'installations'))
        self.assertFalse(installation.is_relative_to(self.study));self.assertTrue(installation.is_dir())

    def test_train_before_check_and_plain_video_input_are_refused(self):
        self.assertEqual(self.request('/api/robot/run',self.payload())[0],400)
        video=self.root/'video.mp4';video.write_bytes(b'not a study')
        self.assertEqual(self.request('/api/robot/check',{**self.payload(),'study':str(video)})[0],400)
        self.assertEqual(self.request('/api/robot/check',{**self.payload(),'study':'https://example.com/video'})[0],400)

    def test_one_shared_job_rejects_footage_during_robot_and_robot_during_footage(self):
        event=threading.Event();self.events.append(event)
        def slow_robot(*args):event.wait(3);return fake_worker(*args)
        server=self.make_server(worker=slow_robot)
        self.assertEqual(self.request('/api/robot/check',self.payload(),server=server)[0],202)
        self.assertEqual(self.request('/api/train',{'source':'/fake'},server=server)[0],409)
        code,state=self.request('/api/state',method='GET',server=server)
        self.assertTrue(state['robot_job_running'])
        self.assertEqual(self.request('/api/robot/check',self.payload(),server=server)[0],409)
        event.set();self.done(server)
        event2=threading.Event();self.events.append(event2)
        def slow_footage(*args,**kwargs):event2.wait(3);return artifact_run(*args,**kwargs)
        server=self.make_server(runner=slow_footage)
        self.assertEqual(self.request('/api/train',{'source':'/fake'},server=server)[0],202)
        self.assertEqual(self.request('/api/robot/check',self.payload(),server=server)[0],409)
        self.assertTrue(self.request('/api/robot/state',method='GET',server=server)[1]['busy'])
        event2.set()

    def test_blocker_clears_stale_install_capability_and_never_retries(self):
        (self.study/'state.txt').write_text('complete')
        self.assertTrue(self.checked()['can_install'])
        (self.study/'state.txt').write_text('blocked')
        state=self.checked();self.assertEqual(state['status'],'blocked')
        self.assertFalse(state['checked']);self.assertFalse(state['can_install']);self.assertFalse(state['can_run'])
        self.assertEqual(self.request('/api/robot/install',self.payload())[0],400)

    def test_protocol_change_invalidates_check_and_result(self):
        (self.study/'state.txt').write_text('complete')
        self.assertTrue(self.checked()['can_install'])
        (self.study/'protocol.json').write_text('{"test":"changed"}')
        state=self.done();self.assertEqual(state['status'],'failed');self.assertFalse(state['can_install'])
        self.assertIsNone(state['summary']);self.assertIn('protocol changed',state['error'])
        self.assertEqual(self.request('/api/robot/install',self.payload())[0],400)

    def test_terminal_worker_lock_allows_check_but_requires_recheck_before_execution(self):
        lock=self.study/'worker.lock'
        lock.write_text(json.dumps(dict(pid=999999,operation='collect')))
        state=self.checked()
        self.assertTrue(state['checked']);self.assertTrue(state['study_locked'])
        self.assertEqual(state['study_worker_lock']['operation'],'collect')
        self.assertFalse(state['can_run']);self.assertFalse(state['can_install'])
        self.assertTrue(lock.exists())
        # The test owns this fake lock; the implementation never removes it.
        lock.unlink()
        state=self.done();self.assertFalse(state['study_locked'])
        self.assertTrue(state['recheck_required']);self.assertFalse(state['can_run'])
        self.assertTrue(self.checked()['can_run'])

    def test_new_terminal_lock_after_check_blocks_direct_run_request(self):
        self.assertTrue(self.checked()['can_run'])
        lock=self.study/'worker.lock';lock.write_text('{"operation":"collect"}')
        code,result=self.request('/api/robot/run',self.payload())
        self.assertEqual(code,400);self.assertIn('Terminal',result['error']);self.assertTrue(lock.exists())

    def test_worker_error_preserves_error_without_stale_success(self):
        def fail(*args):raise ValueError('Fake renderer unavailable')
        server=self.make_server(worker=fail)
        state=self.checked(server)
        self.assertEqual(state['status'],'failed');self.assertIn('renderer unavailable',state['error'])
        self.assertFalse(state['can_install']);self.assertIsNone(state['summary'])


class CameraWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'job';self.output.mkdir()
        self.binding=dict(study='test-study',assets='test-assets',protocol_sha256='a'*64,assets_sha256='b'*64)

    def value(self,status,action=None):
        return dict(binding=self.binding,inspection=dict(status=status,next_actions=[] if action is None else [action]),install_available=False)

    def test_worker_dispatches_registered_terminal_stages_without_shell(self):
        from dvidia_training import camera_action_study as study
        checks=[self.value('ready',dict(command='collect',stage='diagnostic',requires_assets=True)),
            self.value('ready',dict(command='train',stage='initial',requires_assets=False)),
            self.value('ready',dict(command='evaluate',stage='development-initial',requires_assets=True)),self.value('complete')]
        with patch.object(robot,'check_study',side_effect=checks),patch.object(study,'collect') as collect,patch.object(study,'train') as train,patch.object(study,'evaluate') as evaluate:
            result=robot.run_worker('run','test-study','test-assets',self.output,self.binding)
        collect.assert_called_once_with('test-study','test-assets',stage='diagnostic')
        train.assert_called_once_with('test-study',stage='initial')
        evaluate.assert_called_once_with('test-study','test-assets',stage='development-initial')
        self.assertEqual(result['status'],'complete');self.assertEqual(len(result['executed_stages']),3)
        self.assertTrue((self.output/'robot-receipt.json').is_file())

    def test_worker_stops_at_blocker_without_retry(self):
        from dvidia_training import camera_action_study as study
        with patch.object(robot,'check_study',side_effect=[self.value('ready',dict(command='collect',stage='diagnostic',requires_assets=True)),self.value('blocked')]),patch.object(study,'collect') as collect:
            result=robot.run_worker('run','test-study','test-assets',self.output,self.binding)
        self.assertEqual(result['status'],'blocked');collect.assert_called_once()

    def test_worker_rejects_changed_binding_before_stage_and_preserves_failure(self):
        with patch.object(robot,'check_study',return_value=self.value('ready')):
            with self.assertRaisesRegex(ValueError,'changed'):
                robot.run_worker('run','test-study','test-assets',self.output,{**self.binding,'protocol_sha256':'c'*64})
        receipt=json.loads((self.output/'robot-receipt.json').read_text())
        self.assertEqual(receipt['status'],'failed');self.assertEqual(receipt['executed_stages'],[])


if __name__=='__main__':unittest.main()
