"""Publication-boundary and actual synthetic media tests; no remote publication."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('observation_space_prepare', HERE / 'prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg/ffprobe required')
class SpacePreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        cls.plan = prepare.stage('Dvidia', cls.root / 'publication')
        cls.space = cls.root / 'publication' / 'space'

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_public_inventory_and_integrity(self):
        expected = {'README.md', 'LICENSE', 'THIRD_PARTY.md', 'demo-adapter.js',
            'generate_demo.py', 'index.html', 'provenance.json',
            'source/observation_studio.html', 'licenses/object_detection_yolox-LICENSE.txt',
            'licenses/palm_detection_mediapipe-LICENSE.txt', 'demo/visible.mp4',
            'demo/occluded.mp4', 'demo/visible.json', 'demo/occluded.json',
            'demo/episodes.json', 'demo/provenance.json'}
        row = self.plan['repository']
        self.assertEqual(row['repo_id'], 'Dvidia/observation-lab')
        self.assertEqual(row['space_sdk'], 'static')
        self.assertFalse(self.plan['private'])
        self.assertEqual({p['path'] for p in row['files']}, expected)
        self.assertEqual({p.relative_to(self.space).as_posix() for p in self.space.rglob('*') if p.is_file()}, expected)
        self.assertLess(self.plan['total_bytes'], 2 * 1024 * 1024)
        for item in row['files']:
            self.assertEqual(prepare.record(self.space / item['path']),
                             {key: item[key] for key in ('bytes', 'sha256')})

    def test_no_models_private_paths_or_runtime_uploads(self):
        for name in ('provenance.json', 'demo/provenance.json', 'demo/visible.json', 'demo/occluded.json'):
            raw = (self.space / name).read_text()
            self.assertNotIn('/Users/', raw)
            self.assertNotIn('/private/', raw)
            self.assertNotIn('Bearer ', raw)
        adapter = (self.space / 'demo-adapter.js').read_text()
        self.assertNotIn("method:'POST'", adapter)
        self.assertNotIn('http://', adapter)
        self.assertNotIn('https://', adapter)
        index = (self.space / 'index.html').read_text()
        self.assertIn('No detector, VLM or robot policy runs here.', index)
        self.assertIn('Authored observations', index)
        self.assertNotIn('Original model output.', index)
        self.assertNotIn('    save(id,body) { return this.request', index)
        self.assertIn('observation-lab-v0.1.0/docs/observations.md', index)
        self.assertNotIn('type="file"', index)

    def test_original_ui_and_licenses_preserved(self):
        original = prepare.ROOT / 'src' / 'dvidia_training' / 'observation_studio.html'
        self.assertEqual((self.space / 'source/observation_studio.html').read_bytes(), original.read_bytes())
        self.assertEqual(sha256(original.read_bytes()).hexdigest(), self.plan['ui_sha256'])
        self.assertEqual((self.space / 'LICENSE').read_bytes(), (prepare.ROOT / 'LICENSE').read_bytes())
        for name in ('object_detection_yolox-LICENSE.txt', 'palm_detection_mediapipe-LICENSE.txt'):
            self.assertEqual((self.space / 'licenses' / name).read_bytes(),
                             (prepare.ROOT / 'third_party' / 'observation-models' / name).read_bytes())

    def test_actual_video_hashes_and_authored_boxes(self):
        for name in ('visible', 'occluded'):
            episode = json.loads((self.space / 'demo' / (name + '.json')).read_text())
            video = self.space / 'demo' / (name + '.mp4')
            self.assertEqual(episode['source']['sha256'], sha256(video.read_bytes()).hexdigest())
            self.assertEqual(episode['source']['kind'], 'synthetic')
            self.assertEqual(episode['review']['outcome'], 'unknown')
            self.assertEqual(episode['recipe']['model']['models'], [])
            metadata = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_streams',
                '-show_format', '-of', 'json', str(video)], check=True, capture_output=True).stdout)
            self.assertEqual(len(metadata['streams']), 1)
            stream = metadata['streams'][0]
            self.assertEqual((stream['width'], stream['height']), (640, 360))
            self.assertEqual(stream['codec_name'], 'h264')
            self.assertAlmostEqual(float(metadata['format']['duration']), 8)
            for event in episode['observations']:
                for box in event['boxes']:
                    self.assertLess(box['time_seconds'], 8)
                    self.assertTrue(all(0 <= n <= 1 for n in box['xyxy']))

    def test_refuses_overwrite_and_unsafe_outputs(self):
        with self.assertRaisesRegex(ValueError, 'fresh'):
            prepare.stage('Dvidia', self.root / 'publication')
        target = self.root / 'linked'
        target.symlink_to(self.root / 'publication', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            prepare.stage('Dvidia', target / 'new')
        with self.assertRaises(ValueError):
            prepare.stage('../elsewhere', self.root / 'invalid')

    def test_detects_ui_contract_drift(self):
        with self.assertRaisesRegex(ValueError, 'UI changed'):
            prepare.replace_once('changed upstream interface', 'required marker', 'new')


if __name__ == '__main__':
    unittest.main()
