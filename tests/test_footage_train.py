"""Real CPU learning, leakage guards and strict data-only model tests."""
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import numpy as np
from dvidia_training import footage_train as ft

def write_json(path, document):
    path.write_bytes(ft.canonical(document) + b'\n')

def video(path, offset=0, count=32):
    pixels = np.empty((count, 16, 16, 3), dtype=np.uint8)
    for frame in range(count):
        pixels[frame] = 40 + offset + frame * 2 + np.indices((16, 16))[0][..., None]
    result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '16x16', '-framerate', '16', '-i', 'pipe:0', '-c:v', 'ffv1', str(path)], input=pixels.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stderr.decode()

def make_dataset(tmp_path):
    if not shutil.which('ffmpeg'):
        raise unittest.SkipTest('Local CPU ffmpeg is required for actual footage learning.')
    episodes = []
    for index in range(6):
        path = tmp_path / f'clip-{index}.mkv'
        video(path, index)
        episodes.append({'id': f'clip-{index}', 'media_path': path.name, 'media_sha256': sha256(path.read_bytes()).hexdigest(), 'duration_seconds': 2.0, 'start_seconds': 0.0, 'width': 16, 'height': 16, 'fps': 16.0, 'source_recording_id': f'recording-{index}', 'session_id': f'session-{index}', 'shoe_pair_id': f'pair-{index}', 'complete': True, 'task': 'move an object', 'group_id': f'group-{index}', 'split': ('train', 'dev', 'test')[index // 2], 'actions_path': None, 'actions_sha256': None})
    document = {'kind': 'dvidia.footage-dataset', 'schema_version': 1, 'task': 'move an object', 'source_manifest_sha256': 'a' * 64, 'capability': {'visual_ready': True}, 'episodes': episodes}
    path = tmp_path / 'dataset.json'
    write_json(path, document)
    return (path, document)

def sidecar(episode, count=40):
    rows = []
    for index in range(count):
        error = [0.15 + 0.003 * index, 0.1, 0.03, 0.02, 0.0, 0.0]
        rows.append({'timestamp_seconds': (index + 0.5) * episode['duration_seconds'] / count, 'context': [0.01 * index + int(episode['id'][-1]) * .001, -0.5, 0.6, 0.1, -0.2, 0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1], 'error': error, 'delta': [error[0] * 0.01, error[1] * 0.02, 0.0, 0.001, 0.0, 0.0]})
    record = {'kind': 'dvidia.movement-alignment-validation', 'schema_version': 1, 'status': 'validated', 'scope': 'simulation-only', 'media_sha256': episode['media_sha256'], 'samples_sha256': ft.digest(rows), 'feature_contract': ft.FEATURE_CONTRACT, 'method': 'Test numerical fixture, not a physical validation', 'evidence_sha256': 'b' * 64}
    return {'kind': 'dvidia.aligned-movement-demonstration', 'schema_version': 1, 'feature_contract': ft.FEATURE_CONTRACT, 'media_sha256': episode['media_sha256'], 'samples': rows, 'provenance': {'source_kind': 'native-simulation-telemetry', 'validation': record}}

def add_sidecars(path, document):
    for episode in document['episodes']:
        action = path.parent / (episode['id'] + '.actions.json')
        write_json(action, sidecar(episode))
        episode['actions_path'] = action.name
        episode['actions_sha256'] = sha256(action.read_bytes()).hexdigest()
    write_json(path, document)
import unittest
import importlib.util
import tempfile
from unittest.mock import patch

class _PatchAdapter:

    def __init__(self, test):
        self.test = test

    def setattr(self, obj, name, value):
        active = patch.object(obj, name, value)
        active.start()
        self.test.addCleanup(active.stop)

class FootageTrainTests(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = make_dataset(self.root)

    def test_real_cpu_learning_and_safe_artifact(self):
        dataset = self.dataset
        tmp_path = self.root
        monkeypatch = _PatchAdapter(self)
        path, _ = dataset
        output = tmp_path / 'trained'
        report = ft.train(path, output, frames_per_clip=8, resolution=8, latent_dim=2)
        assert report['test']['learned_rgb_mse'] < report['test']['persistence_rgb_mse'] * 0.1
        assert report['test']['episodes'] == 2
        assert report['test']['transitions'] == 14
        assert report['physical_robot_ready'] is False
        assert report['robot_policy_trained'] is False
        assert report['capability']['movement_candidate_trained'] is False
        metadata, arrays = ft.load_visual_model(output / 'model.json')
        assert set(arrays) == {'mean', 'basis', 'transition'}
        assert arrays['basis'].shape == (2, 192)
        assert all((array.dtype == np.float32 for array in arrays.values()))
        assert ft.validate_saved_model(output) == metadata
        assert json.loads((output / 'report.json').read_text()) == report
        with self.assertRaisesRegex(ValueError, 'frozen reports'):
            ft.train(path, output)

    def test_test_pixels_do_not_change_weights(self):
        dataset = self.dataset
        tmp_path = self.root
        monkeypatch = _PatchAdapter(self)
        path, document = dataset
        first, second = (tmp_path / 'first', tmp_path / 'second')
        ft.train(path, first, frames_per_clip=8, resolution=8, latent_dim=2)
        for episode in document['episodes'][4:]:
            media = path.parent / episode['media_path']
            media.unlink()
            video(media, 75)
            episode['media_sha256'] = sha256(media.read_bytes()).hexdigest()
        write_json(path, document)
        ft.train(path, second, frames_per_clip=8, resolution=8, latent_dim=2)
        model1, arrays1 = ft.load_visual_model(first / 'model.json')
        model2, arrays2 = ft.load_visual_model(second / 'model.json')
        assert model1['training'] == model2['training']
        for key in arrays1:
            np.testing.assert_array_equal(arrays1[key], arrays2[key])

    def test_split_integrity_rejected_before_decode(self):
        dataset = self.dataset
        tmp_path = self.root
        monkeypatch = _PatchAdapter(self)
        for kind in ['group', 'recording', 'shoe', 'session', 'hash', 'missing', 'path', 'nonfinite']:
            with self.subTest(kind=kind):
                path, document = dataset
                train, test = (document['episodes'][0], document['episodes'][4])
                if kind in ('group', 'recording', 'shoe', 'session'):
                    field = {'group': 'group_id', 'recording': 'source_recording_id', 'shoe': 'shoe_pair_id', 'session': 'session_id'}[kind]
                    test[field] = train[field]
                elif kind == 'hash':
                    test['media_sha256'] = train['media_sha256']
                elif kind == 'missing':
                    for episode in document['episodes']:
                        episode['split'] = 'train'
                elif kind == 'path':
                    train['media_path'] = '../outside.mkv'
                else:
                    train['duration_seconds'] = float('inf')
                if kind == 'nonfinite':
                    path.write_text(json.dumps(document))
                else:
                    write_json(path, document)

                def must_not_decode(*args, **kwargs):
                    self.fail('Invalid intake reached decoder')
                monkeypatch.setattr(ft, '_extract_frames', must_not_decode)
                output = tmp_path / 'rejected'
                with self.assertRaises(ValueError):
                    ft.train(path, output)
                assert not output.exists()

    def test_tampered_hash_and_symlink(self):
        dataset = self.dataset
        tmp_path = self.root
        monkeypatch = _PatchAdapter(self)
        path, document = dataset
        media = path.parent / document['episodes'][0]['media_path']
        original = media.read_bytes()
        media.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            ft.train(path, tmp_path / 'hash-rejected')
        target = tmp_path / 'original.mkv'
        target.write_bytes(original)
        media.unlink()
        media.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            ft.train(path, tmp_path / 'link-rejected')

    def test_insufficient_actual_frames(self):
        dataset = self.dataset
        tmp_path = self.root
        monkeypatch = _PatchAdapter(self)
        path, document = dataset
        media = path.parent / document['episodes'][0]['media_path']
        media.unlink()
        video(media, count=1)
        document['episodes'][0]['media_sha256'] = sha256(media.read_bytes()).hexdigest()
        write_json(path, document)
        with self.assertRaisesRegex(ValueError, 'required number'):
            ft.train(path, tmp_path / 'too-short')

    def test_real_playlist_disguised_as_mp4_cannot_decode_unlisted_media(self):
        path, document = self.dataset
        with tempfile.TemporaryDirectory() as external_name:
            external = Path(external_name)/'unlisted.ts'
            conversion = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
                '-i', str(self.root/'clip-0.mkv'), '-c:v', 'mpeg2video', '-f', 'mpegts', str(external)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(conversion.returncode, 0, conversion.stderr.decode())
            shutil.copyfile(external, self.root/'unlisted.ts')
            playlist = self.root/'disguised.mp4'
            playlist.write_text("ffconcat version 1.0\nfile 'unlisted.ts'\n")
            # The synthetic control proves filename and file/pipe protocols alone
            # allow this playlist to decode bytes absent from the manifest.
            control = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error',
                '-protocol_whitelist', 'file,pipe', '-i', str(playlist), '-frames:v', '1',
                '-vf', 'scale=8:8', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(control.returncode, 0, control.stderr.decode())
            self.assertEqual(len(control.stdout), 8*8*3)
            episode = document['episodes'][0]
            episode['media_path'] = playlist.name
            episode['media_sha256'] = sha256(playlist.read_bytes()).hexdigest()
            write_json(path, document)
            captured = []
            real_run = subprocess.run
            def capture(*args, **kwargs):
                result = real_run(*args, **kwargs)
                captured.append(result)
                return result
            with patch.object(ft.subprocess, 'run', side_effect=capture):
                with self.assertRaisesRegex(ValueError, 'required number'):
                    ft.train(path, self.root/'playlist-rejected')
            self.assertEqual(len(captured), 1)
            self.assertNotEqual(captured[0].returncode, 0)
            self.assertIn('whitelist', captured[0].stderr.decode().lower())
            self.assertEqual(captured[0].stdout, b'')
            self.assertFalse((self.root/'playlist-rejected').exists())
            ordinary_name = self.root/'plain.m3u8'
            playlist.rename(ordinary_name)
            episode['media_path'] = ordinary_name.name
            write_json(path, document)
            with self.assertRaisesRegex(ValueError, 'video-container extension'):
                ft.validate_dataset(path)

    def test_malformed_advertised_actions_abort_visual(self):
        dataset = self.dataset
        tmp_path = self.root
        monkeypatch = _PatchAdapter(self)
        path, document = dataset
        add_sidecars(path, document)
        episode = document['episodes'][0]
        action = path.parent / episode['actions_path']
        value = json.loads(action.read_text())
        value['provenance']['validation']['samples_sha256'] = '0' * 64
        write_json(action, value)
        episode['actions_sha256'] = sha256(action.read_bytes()).hexdigest()
        write_json(path, document)
        with self.assertRaisesRegex(ValueError, 'bindings'):
            ft.train(path, tmp_path / 'bad-actions')

    @unittest.skipUnless(importlib.util.find_spec('mujoco') is not None, 'Install the optional arm extra for movement fitting.')
    def test_motor_candidate_learns_only_aligned_train_labels(self):
        dataset = self.dataset
        tmp_path = self.root
        monkeypatch = _PatchAdapter(self)
        path, document = dataset
        add_sidecars(path, document)
        output = tmp_path / 'movement'
        report = ft.train_movement(path, output)
        assert report['scope'] == 'simulation-only'
        assert report['capability'] == {'movement_candidate_trained': True, 'closed_loop_qualified': False}
        assert report['physical_robot_ready'] is False
        assert report['test']['learned_joint_delta_mse'] < report['test']['zero_delta_mse'] * 0.01
        model = json.loads((output / 'movement_head.json').read_text())
        assert model['train_cases'] == ['clip-0', 'clip-1']
        assert model['training']['sample_count'] == 80
        assert report['movement_head_sha256'] == ft.digest(model)

    def test_duplicate_motor_trajectory_cannot_cross_splits(self):
        dataset = self.dataset
        path, document = dataset
        add_sidecars(path, document)
        source = sidecar(document['episodes'][0])
        target = document['episodes'][4]
        duplicate = sidecar(target)
        duplicate['samples'] = source['samples']
        # Timestamp changes must not make the same action sequence independent.
        for row in duplicate['samples']:
            row['timestamp_seconds'] += .001
        duplicate['provenance']['validation']['samples_sha256'] = ft.digest(duplicate['samples'])
        action = path.parent / target['actions_path']
        write_json(action, duplicate)
        target['actions_sha256'] = sha256(action.read_bytes()).hexdigest()
        write_json(path, document)
        with self.assertRaisesRegex(ValueError, 'Duplicate movement trajectory'):
            ft.validate_dataset(path)
        with self.assertRaisesRegex(ValueError, 'Duplicate movement trajectory'):
            ft.train(path, self.root/'bad-split')
        self.assertFalse((self.root/'bad-split').exists())

    def test_alignment_contract_numeric_and_lineage_bounds(self):
        dataset = self.dataset
        tmp_path = self.root
        monkeypatch = _PatchAdapter(self)
        for kind in ['timestamp', 'context', 'delta', 'source', 'scope']:
            with self.subTest(kind=kind):
                _, document = dataset
                episode = document['episodes'][0]
                value = sidecar(episode)
                if kind == 'timestamp':
                    value['samples'][1]['timestamp_seconds'] = value['samples'][0]['timestamp_seconds']
                elif kind == 'context':
                    value['samples'][0]['context'][0] = 1e+300
                elif kind == 'delta':
                    value['samples'][0]['delta'][0] = 0.2
                elif kind == 'source':
                    value['provenance']['source_kind'] = 'human-video-unvalidated'
                else:
                    value['provenance']['validation']['scope'] = 'physical-ready'
                with self.assertRaises(ValueError):
                    ft.validate_movement_sidecar(value, episode)

    def test_object_npz_header_rejected_before_numpy_allocation(self):
        dataset = self.dataset
        tmp_path = self.root
        monkeypatch = _PatchAdapter(self)
        path, _ = dataset
        output = tmp_path / 'safe'
        ft.train(path, output, frames_per_clip=8, resolution=8, latent_dim=2)
        _, arrays = ft.load_visual_model(output / 'model.json')
        arrays['mean'] = np.array([object()], dtype=object)
        np.savez_compressed(output / 'visual_model.npz', **arrays)
        model = json.loads((output / 'model.json').read_text())
        model['arrays_sha256'] = sha256((output / 'visual_model.npz').read_bytes()).hexdigest()
        write_json(output / 'model.json', model)
        with self.assertRaisesRegex(ValueError, 'header'):
            ft.load_visual_model(output / 'model.json')

    def test_model_array_tamper_rejected(self):
        dataset = self.dataset
        tmp_path = self.root
        monkeypatch = _PatchAdapter(self)
        path, _ = dataset
        output = tmp_path / 'safe'
        ft.train(path, output, frames_per_clip=8, resolution=8, latent_dim=2)
        artifact = output / 'visual_model.npz'
        artifact.write_bytes(artifact.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            ft.load_visual_model(output / 'model.json')
if __name__ == '__main__':
    unittest.main()
