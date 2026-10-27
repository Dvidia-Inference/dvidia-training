from dataclasses import asdict
from hashlib import sha256
import http.client
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

from dvidia_training import video_goal_studio as studio
from dvidia_training.video_goal_bridge import PlanarVideoCalibration


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Requires FFmpeg/ffprobe')
class StudioHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = studio.VideoGoalStudioServer(('127.0.0.1', 0), Path(self.temp.name)/'workspace', runner=studio._worker)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path, value=None, *, raw=None, method='POST', headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        body = raw if raw is not None else json.dumps(value).encode() if value is not None else None
        extra = {'Origin':self.origin, 'Content-Type':'application/json'}
        extra.update(headers or {})
        try:
            connection.request(method, path, body=body, headers=extra)
            response = connection.getresponse(); data = response.read()
            if response.getheader('Content-Type','').startswith('application/json'): data=json.loads(data)
            return response.status, dict(response.getheaders()), data
        finally: connection.close()

    def wait(self):
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            status, _, state = self.request('/api/state', method='GET')
            self.assertEqual(status,200)
            if state['job']['status'] != 'running': return state
            time.sleep(.02)
        self.fail('Worker did not complete')

    def create(self):
        status, _, state=self.request('/api/example', {})
        self.assertEqual(status,200)
        source=state['source']
        status, _, state=self.request('/api/create',dict(source_id=source['id'],calibration=source['calibration']))
        self.assertEqual(status,200)
        state=self.wait()
        self.assertEqual(state['job']['status'],'complete',state['job']['error'])
        return state

    def source_package(self, *, corrupt=False):
        video=Path(studio.__file__).with_name('video_goal_example.mp4').read_bytes()
        manifest={
            'schema':'dvidia.skillspace-video.v1','platform_origin':'https://dvidia.org',
            'exported_at':'2026-10-25T20:00:00.000Z',
            'contribution':{'id':'55155155-1551-4551-8551-551551551551',
                'revision':'2026-10-25T19:00:00.000Z', 'skillspace_id':'clothing-folding',
                'task_title':'Synthetic block placement'},
            'video':{'filename':'video.mp4','original_name':'block.mp4',
                'content_type':'video/mp4','bytes':len(video),'sha256':sha256(video).hexdigest()},
            'processing':{'scope':'local-only','accepted':True},'calibration':None}
        raw=io.BytesIO()
        with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('manifest.json',json.dumps(manifest))
            archive.writestr('video.mp4',video if not corrupt else b'x'*len(video))
        return raw.getvalue(),manifest

    def test_same_origin_controls_and_no_filesystem_route(self):
        self.assertEqual(self.request('/api/state',method='GET',headers={'Host':'attacker.invalid'})[0],403)
        self.assertEqual(self.request('/api/example',{},headers={'Origin':'https://dvidia.org'})[0],403)
        self.assertEqual(self.request('/api/state',method='GET',headers={'Sec-Fetch-Site':'cross-site'})[0],403)
        self.assertEqual(self.request('/../../pyproject.toml',method='GET')[0],404)
        _, headers, page=self.request('/',method='GET')
        self.assertIn('frame-ancestors',headers['Content-Security-Policy'])
        self.assertIn(b'video-player',page)
        for name in ('studio-mapping.js', 'studio-ui.js'):
            status, headers, script = self.request('/'+name, method='GET')
            self.assertEqual(status, 200)
            self.assertTrue(headers['Content-Type'].startswith('text/javascript'))
            self.assertGreater(len(script), 100)
        self.assertEqual(self.request('/../studio_ui.js', method='GET')[0], 404)

    def test_real_create_export_import_and_media_ranges(self):
        state=self.create(); source=state['source']; candidate=state['candidate']
        status, headers, data=self.request(source['media_url'],method='GET',headers={'Range':'bytes=0-15'})
        self.assertEqual((status,len(data)),(206,16))
        self.assertTrue(headers['Content-Range'].startswith('bytes 0-15/'))
        self.assertEqual(self.request(source['media_url'],method='GET',headers={'Range':'bytes=-0'})[0],416)
        status, _, archive=self.request(candidate['download_url'],method='GET')
        self.assertEqual(status,200)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle: self.assertEqual(set(bundle.namelist()),set(studio.FILES))
        status, _, imported=self.request('/api/import',raw=archive,headers={'Content-Type':'application/zip'})
        self.assertEqual(status,200)
        self.assertIsNone(imported['source'])
        self.assertEqual(imported['candidate']['id'],candidate['id'])
        self.assertFalse(imported['candidate']['policy_learned'])

    def test_new_source_invalidates_candidate_and_failed_job_stays_failed(self):
        self.create()
        status, _, state=self.request('/api/upload?name=invalid.mp4',raw=b'not a video',headers={'Content-Type':'application/octet-stream'})
        self.assertEqual(status,200)
        self.assertIsNone(state['candidate'])
        self.assertIsNone(state['job'])
        self.assertIsNone(state['source']['calibration'])
        self.assertFalse(state['source']['sample'])
        self.request('/api/create',dict(source_id=state['source']['id'],calibration=asdict(PlanarVideoCalibration())))
        failed=self.wait()
        self.assertEqual(failed['job']['status'],'failed')
        self.assertIsNone(failed['candidate'])
        self.assertIsNone(failed['job']['result'])

    def test_no_default_calibration_or_unready_robot_execution(self):
        status, _, state=self.request('/api/example',{})
        self.assertEqual(self.request('/api/create',dict(source_id=state['source']['id'],calibration={}))[0],400)
        state=self.create()
        status, _, error=self.request('/api/run',dict(candidate_id=state['candidate']['id'],object_xy=[.294,-.025],seed=1))
        self.assertEqual(status,400)
        self.assertIn('assets',error['error'])

    def test_only_successful_user_calibration_is_restored_for_its_upload(self):
        video=Path(studio.__file__).with_name('video_goal_example.mp4').read_bytes()
        status, _, state=self.request('/api/upload?name=clip.mp4',raw=video,headers={'Content-Type':'application/octet-stream'})
        self.assertEqual(status,200)
        self.assertIsNone(state['source']['calibration'])
        calibration=asdict(PlanarVideoCalibration())
        self.request('/api/create',dict(source_id=state['source']['id'],calibration=calibration))
        state=self.wait()
        self.assertEqual(state['job']['status'],'complete')
        expected=json.loads(json.dumps(calibration))
        self.assertEqual(state['source']['calibration'],expected)
        _, _, changed=self.request('/api/upload?name=other.mp4',raw=video,headers={'Content-Type':'application/octet-stream'})
        self.assertIsNone(changed['source']['calibration'])
        self.assertIsNone(changed['candidate'])

    def test_malformed_oversized_and_code_uploads_are_rejected(self):
        self.assertEqual(self.request('/api/upload?name=payload.py',raw=b'code',headers={'Content-Type':'application/octet-stream'})[0],400)
        self.assertEqual(self.request('/api/create',raw=b'{"source_id":"a","source_id":"b"}')[0],400)
        self.assertEqual(self.request('/api/upload?name=a.mp4',raw=b'x',headers={'Content-Type':'application/octet-stream','Content-Length':str(studio.MAX_VIDEO_BYTES+1)})[0],400)

    def test_only_one_worker_job_can_run(self):
        event=threading.Event()
        original=self.server.runner
        def blocked(*args):
            event.wait(4)
            return original(*args)
        self.server.runner=blocked
        _, _, state=self.request('/api/example',{})
        payload=dict(source_id=state['source']['id'],calibration=state['source']['calibration'])
        try:
            self.assertEqual(self.request('/api/create',payload)[0],200)
            self.assertEqual(self.request('/api/example',{})[0],400)
            self.assertEqual(self.request('/api/create',payload)[0],400)
        finally: event.set()
        self.assertEqual(self.wait()['job']['status'],'complete')

    def test_skillspace_package_opens_source_and_requires_its_own_calibration(self):
        self.create()
        raw,manifest=self.source_package()
        status, _, state=self.request('/api/import-video',raw=raw,headers={'Content-Type':'application/zip'})
        self.assertEqual(status,200,state)
        self.assertIsNone(state['candidate'])
        self.assertIsNone(state['job'])
        source=state['source']
        self.assertEqual(source['name'],'block.mp4')
        self.assertFalse(source['sample'])
        self.assertIsNone(source['calibration'])
        self.assertEqual(source['skillspace']['id'],manifest['contribution']['id'])
        self.assertEqual(source['skillspace']['video_sha256'],manifest['video']['sha256'])
        status,_,video=self.request(source['media_url'],method='GET')
        self.assertEqual(status,200)
        self.assertEqual(sha256(video).hexdigest(),manifest['video']['sha256'])
        self.assertEqual(self.request('/api/create',dict(source_id=source['id'],calibration={}))[0],400)
        self.request('/api/create',dict(source_id=source['id'],calibration=asdict(PlanarVideoCalibration())))
        self.assertEqual(self.wait()['job']['status'],'complete')

    def test_bad_package_preserves_existing_source_and_cross_origin_is_rejected(self):
        original=self.create()
        raw,_=self.source_package(corrupt=True)
        self.assertEqual(self.request('/api/import-video',raw=raw,headers={'Content-Type':'application/zip'})[0],400)
        _,_,state=self.request('/api/state',method='GET')
        self.assertEqual(state['source'],original['source'])
        self.assertEqual(state['candidate'],original['candidate'])
        self.assertEqual(self.request('/api/import-video',raw=b'x',
            headers={'Content-Type':'application/zip','Origin':'https://dvidia.org'})[0],403)
        self.assertFalse(list(Path(self.temp.name).rglob('*.zip')))

    def test_package_import_startup_and_size_boundary(self):
        raw,manifest=self.source_package()
        package=Path(self.temp.name).resolve()/'selected.zip'; package.write_bytes(raw)
        self.server.import_video(package)
        self.assertEqual(self.server.state()['source']['skillspace']['revision'],manifest['contribution']['revision'])
        self.assertEqual(self.request('/api/import-video',raw=b'x',headers={
            'Content-Type':'application/zip','Content-Length':str(studio.SOURCE_PACKAGE_LIMIT+1)})[0],400)

    def test_changed_imported_video_or_provenance_cannot_create_a_candidate(self):
        raw,_=self.source_package()
        _,_,state=self.request('/api/import-video',raw=raw,headers={'Content-Type':'application/zip'})
        source=self.server.source
        video=source['path']; original=video.read_bytes()
        video.write_bytes(b'x'*len(original))
        self.assertEqual(self.request(state['source']['media_url'],method='GET')[0],400)
        payload=dict(source_id=source['id'],calibration=asdict(PlanarVideoCalibration()))
        status,_,error=self.request('/api/create',payload)
        self.assertEqual(status,400)
        self.assertIn('changed',error['error'])
        self.assertIsNone(self.server.candidate)

        _,_,state=self.request('/api/import-video',raw=raw,headers={'Content-Type':'application/zip'})
        source=self.server.source
        manifest_path=source['path'].parent/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        manifest['contribution']['task_title']='Changed after import'
        manifest_path.write_text(json.dumps(manifest))
        self.assertEqual(self.request('/api/create',dict(
            source_id=source['id'],calibration=asdict(PlanarVideoCalibration())))[0],400)
        self.assertIsNone(self.server.candidate)


class ImportBoundaryTests(unittest.TestCase):
    def test_archive_paths_duplicates_and_expansion_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            for names in (['../manifest.json','proposal.json','calibration.json'],
                          ['manifest.json','manifest.json','calibration.json'],
                          ['manifest.json','proposal.json','calibration.json','execute.py']):
                raw=io.BytesIO()
                import warnings
                with warnings.catch_warnings(), zipfile.ZipFile(raw,'w') as archive:
                    warnings.simplefilter('ignore',UserWarning)
                    for name in names: archive.writestr(name,b'{}')
                with self.subTest(names=names), self.assertRaises(ValueError):
                    studio.import_bundle(raw.getvalue(),Path(temp)/'candidate')
            raw=io.BytesIO()
            with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json',b'{}')
                archive.writestr('proposal.json',b' '*(studio.MAX_PROPOSAL_BYTES+1))
                archive.writestr('calibration.json',b'{}')
            with self.assertRaises(ValueError): studio.import_bundle(raw.getvalue(),Path(temp)/'candidate')
            self.assertFalse((Path(temp)/'candidate').exists())

    def test_non_loopback_bind_rejected_before_creating_a_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'not-created'
            with self.assertRaises(ValueError): studio.VideoGoalStudioServer(('0.0.0.0',0),root)
            self.assertFalse(root.exists())


class AssetSetupTests(unittest.TestCase):
    def test_interrupted_download_does_not_block_retry_or_expose_partial_assets(self):
        from dvidia_training import robot_profiles
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'assets'
            def verify(profile,path):
                if not (Path(path)/'agilex_piper'/'complete').is_file(): raise ValueError('missing')
                return {'profile':'piper'}
            def interrupted(profile,path):
                (Path(path)/'agilex_piper').mkdir()
                raise OSError('interrupted')
            def complete(profile,path):
                (Path(path)/'agilex_piper').mkdir()
                (Path(path)/'agilex_piper'/'complete').write_text('verified')
            with patch.object(robot_profiles,'verify_assets',side_effect=verify), patch.object(robot_profiles,'fetch_assets',side_effect=interrupted):
                with self.assertRaises(OSError): studio.provision_assets(root)
            self.assertEqual(list(root.iterdir()),[])
            with patch.object(robot_profiles,'verify_assets',side_effect=verify), patch.object(robot_profiles,'fetch_assets',side_effect=complete) as fetch:
                self.assertEqual(studio.provision_assets(root),{'profile':'piper'})
                studio.provision_assets(root)
                self.assertEqual(fetch.call_count,1)

    def test_existing_invalid_assets_are_preserved(self):
        from dvidia_training import robot_profiles
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'assets'; target=root/'agilex_piper'; target.mkdir(parents=True)
            (target/'existing').write_text('preserve')
            with patch.object(robot_profiles,'verify_assets',side_effect=ValueError('invalid')), patch.object(robot_profiles,'fetch_assets') as fetch:
                with self.assertRaisesRegex(ValueError,'preserved'): studio.provision_assets(root)
                fetch.assert_not_called()
            self.assertEqual((target/'existing').read_text(),'preserve')


if __name__ == '__main__': unittest.main()
