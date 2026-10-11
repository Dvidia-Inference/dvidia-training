"""Contract, real FFmpeg intake, and loopback review acceptance checks."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest

from dvidia_training import observations as o
from dvidia_training.observation_pipeline import run, load_episodes, validate_manifest
from dvidia_training.observation_studio import ObservationServer


def detection(label='cup', bounds=None):
    return {'label': label, 'score': .7, 'xyxy': bounds or [.2, .2, .5, .5]}


def episode():
    frames = [{'time_seconds': t, 'detections': [detection()]} for t in [0, .5, 1]]
    return {'id': 'fixture', 'source': {'sha256': 'a'*64}, 'duration_seconds': 2,
            'observations': o.associate(frames, 2)}


def payload(e, base=0):
    return {'base_revision': base, 'events': o.initial_review(e)['events'],
            'outcome': 'unknown', 'status': 'in_review'}


class ObservationContractTest(unittest.TestCase):
    def test_association_preserves_evidence_and_splits_occlusion(self):
        frames = [{'time_seconds': t, 'detections': [detection()]} for t in [0, .5, 2]]
        tracks = o.associate(frames, 3)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(tracks[0]['boxes']), 2)
        self.assertEqual(tracks[0]['end_seconds'], 1)
        self.assertEqual(tracks[1]['start_seconds'], 2)

    def test_same_frame_cannot_merge_two_objects(self):
        tracks = o.associate([{'time_seconds': 0, 'detections': [detection(), detection()]}], 1)
        self.assertEqual(len(tracks), 2)

    def test_bad_scores_boxes_and_timestamps_rejected(self):
        for row in [detection(bounds=[0, 0, float('nan'), 1]), detection(bounds=[.5,.2,.2,.5]),
                    {**detection(), 'score': True}, {**detection(), 'score': 2}]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                o.associate([{'time_seconds': 0, 'detections': [row]}], 1)
        with self.assertRaises(ValueError):
            o.associate([{'time_seconds': 0, 'detections': []}]*2, 1)

    def test_proximity_does_not_claim_grasp_or_success(self):
        proposals = o.propose_relations([{'time_seconds': 0, 'detections': [detection(), detection('hand')]}], 1)
        self.assertEqual(len(proposals), 1)
        self.assertIn('image overlap', proposals[0]['label'])
        self.assertEqual(o.initial_review(episode())['outcome'], 'unknown')

    def test_review_keeps_raw_separate_and_binds_source(self):
        e = episode(); original = deepcopy(e)
        p = payload(e); p['events'][0]['label'] = 'Cup held, needs checking'
        result = o.validate_review(p, e, o.initial_review(e))
        self.assertEqual(e, original)
        self.assertEqual(result['source_sha256'], e['source']['sha256'])
        self.assertEqual(result['revision'], 1)

    def test_stale_review_rejected(self):
        e = episode()
        with self.assertRaises(RuntimeError):
            o.validate_review(payload(e, 1), e, o.initial_review(e))

    def test_invalid_observation_ids_are_controlled_errors(self):
        e = episode()
        for value in ([], {}, 3, True):
            p = payload(e); p['events'][0]['observation_id'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                o.validate_review(p, e, o.initial_review(e))

    def test_cannot_drop_or_duplicate_model_proposals(self):
        e = episode()
        for events in [[], payload(e)['events']*2]:
            with self.assertRaises(ValueError):
                o.validate_review({**payload(e), 'events': events}, e, o.initial_review(e))

    def test_reviewed_requires_all_decisions(self):
        e = episode(); p = payload(e); p['status'] = 'reviewed'
        with self.assertRaises(ValueError):
            o.validate_review(p, e, o.initial_review(e))
        p['events'][0]['decision'] = 'rejected'
        self.assertEqual(o.validate_review(p, e, o.initial_review(e))['outcome'], 'unknown')

    def test_manual_event_accepted_only_with_valid_interval(self):
        e = episode(); p = payload(e)
        manual = dict(id='manual-new', observation_id=None, label='Reach', start_seconds=0,
                      end_seconds=.5, decision='approved')
        p['events'].append(manual)
        self.assertEqual(len(o.validate_review(p,e,o.initial_review(e))['events']), 2)
        for end in [0, 3, float('inf')]:
            manual['end_seconds'] = end
            with self.assertRaises(ValueError):
                o.validate_review(p,e,o.initial_review(e))

    def test_unknown_accuracy_without_independent_references(self):
        self.assertIsNone(o.evaluate_reviews([episode()])['accuracy'])

    def test_changed_review_chain_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            e = episode(); first = o.validate_review(payload(e),e,o.initial_review(e))
            target = Path(directory)/'review-000001.json'; target.write_text(json.dumps(first))
            self.assertEqual(o.latest_review(directory,e)['revision'], 1)
            first['parent_sha256'] = 'b'*64; target.write_text(json.dumps(first))
            with self.assertRaises(ValueError):
                o.latest_review(directory,e)
            target.write_text('[]')
            with self.assertRaises(ValueError):
                o.latest_review(directory,e)


class FixtureDetector:
    metadata = {'kind': 'synthetic-test-only', 'device': 'cpu'}
    def detect(self, _):
        return [detection()]


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class ObservationAcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        cls.video = cls.root/'synthetic.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=160x120:r=10:d=1.2',
                        '-c:v','libx264','-pix_fmt','yuv420p',str(cls.video)], check=True)
        cls.row = dict(id='fixture',path='synthetic.mp4',title='Synthetic fixture',session_id='session1',
            source_recording_id='synthetic1',kind='synthetic',credit='DVIDIA software fixture',
            license='MIT',source_url='local-synthetic-fixture',permission='local_analysis',sha256=o.file_hash(cls.video))
        cls.manifest = cls.root/'source.json'
        cls.manifest.write_text(json.dumps(dict(kind='dvidia.observation-source.v1',task='Test fixture',episodes=[cls.row])))
        cls.output = cls.root/'run'
        cls.result = run(cls.manifest, cls.output, FixtureDetector())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_decoder_hash_timestamps_and_offline_bundle(self):
        receipt, episodes = load_episodes(self.output)
        self.assertEqual(receipt['completed'], 1)
        e = episodes['fixture']
        self.assertEqual([f['time_seconds'] for f in e['frames']], [0,.5,1])
        self.assertTrue(all(len(f['sha256']) == 64 for f in e['frames']))
        self.assertEqual(e['source']['sha256'], self.row['sha256'])
        self.assertNotIn(str(self.root), json.dumps(e))

    def test_existing_run_not_overwritten(self):
        with self.assertRaises(ValueError):
            run(self.manifest, self.output, FixtureDetector())

    def test_nonzero_video_start_keeps_failed_attempt(self):
        video = self.root/'offset.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(self.video), '-c', 'copy',
                        '-output_ts_offset', '2', str(video)], check=True)
        manifest = self.root/'offset.json'
        manifest.write_text(json.dumps(dict(kind='dvidia.observation-source.v1', task='Timing fixture',
            episodes=[{**self.row, 'path':video.name, 'sha256':o.file_hash(video)}])))
        result = run(manifest, self.root/'offset-run', FixtureDetector())
        self.assertEqual(result['completed'], 0)
        self.assertIn('start time', result['failures'][0]['error'])

    def test_run_metadata_is_bounded_before_reading(self):
        copy=self.root/'oversized'; shutil.copytree(self.output,copy)
        (copy/'run.json').write_bytes(b' '* (2*1024*1024+1))
        with self.assertRaises(ValueError): load_episodes(copy)

    def test_hash_failure_preserved(self):
        manifest = self.root/'bad.json'
        manifest.write_text(json.dumps(dict(kind='dvidia.observation-source.v1',task='Test',episodes=[{**self.row,'sha256':'a'*64}])))
        result = run(manifest, self.root/'bad-run', FixtureDetector())
        self.assertEqual(result['completed'],0)
        self.assertEqual(len(result['failures']),1)

    def test_unsafe_or_unpermitted_sources_rejected(self):
        for index, changes in enumerate([{'permission':'upload'}, {'path':'../synthetic.mp4'}, {'sha256':'bad'}]):
            path=self.root/f'invalid{index}.json'
            path.write_text(json.dumps(dict(kind='dvidia.observation-source.v1',task='Test',episodes=[{**self.row,**changes}])))
            with self.assertRaises(ValueError): validate_manifest(path)

    def test_tampered_episode_rejected(self):
        copy=self.root/'tampered'; shutil.copytree(self.output,copy)
        path=copy/'fixture/episode.json'; data=json.loads(path.read_text()); data['title']='Changed'; path.write_text(json.dumps(data))
        with self.assertRaises(ValueError): load_episodes(copy)

    def test_http_review_range_origin_and_reopen(self):
        run_copy=self.root/'http'; shutil.copytree(self.output,run_copy)
        server=ObservationServer(('127.0.0.1',0),run_copy)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        def request(method,path,body=None,headers=None):
            conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
            conn.request(method,path,body=body,headers=headers or {})
            response=conn.getresponse(); data=response.read(); result=(response.status,dict(response.getheaders()),data); conn.close(); return result
        try:
            status,_,raw=request('GET','/api/episodes/fixture'); self.assertEqual(status,200)
            e=json.loads(raw); p=payload(e)
            headers={'Content-Type':'application/json','Origin':f'http://127.0.0.1:{server.server_port}'}
            self.assertEqual(request('POST','/api/episodes/fixture/review',json.dumps(p),{'Content-Type':'application/json'})[0],403)
            self.assertEqual(request('POST','/api/episodes/fixture/review',json.dumps(p),headers)[0],200)
            self.assertEqual(request('POST','/api/episodes/fixture/review',json.dumps(p),headers)[0],409)
            malformed=payload(e, 1); malformed['events'][0]['observation_id']=[]
            self.assertEqual(request('POST','/api/episodes/fixture/review',json.dumps(malformed),headers)[0],400)
            self.assertEqual(json.loads(request('GET','/api/episodes/fixture')[2])['review']['revision'],1)
            self.assertEqual(o.latest_review(run_copy/'fixture',e)['revision'],1)
            self.assertEqual(request('GET','/api/episodes',headers={'Host':'evil.test'})[0],403)
            self.assertEqual(request('GET','/api/episodes',headers={'Origin':'https://evil.test'})[0],403)
            video=self.video.read_bytes()
            status,headers,data=request('GET','/api/episodes/fixture/video',headers={'Range':'bytes=4-19'})
            self.assertEqual((status,data),(206,video[4:20])); self.assertEqual(headers['Content-Length'],'16')
            self.assertEqual(request('GET','/api/episodes/fixture/video',headers={'Range':'bytes=-8'})[2],video[-8:])
            self.assertEqual(request('GET','/api/episodes/fixture/video',headers={'Range':'bytes=999999-'})[0],416)
            self.assertEqual(request('GET','/api/episodes/fixture/video',headers={'Range':'bytes=0-2,5-9'})[0],416)
            self.assertEqual(request('HEAD','/api/episodes/fixture/video')[2],b'')
            self.assertEqual(request('GET','/../../source.json')[0],404)
            changed=run_copy/'fixture'/e['source']['video_path']
            original=changed.read_bytes(); changed.write_bytes(b'x'+original[1:])
            self.assertEqual(request('GET','/api/episodes/fixture/video')[0],409)
            self.assertEqual(request('POST','/api/episodes/fixture/review',json.dumps(payload(e,1)),
                {'Content-Type':'application/json','Origin':f'http://127.0.0.1:{server.server_port}'})[0],400)
        finally:
            server.shutdown(); thread.join(); server.server_close()


if __name__ == '__main__':
    unittest.main()
