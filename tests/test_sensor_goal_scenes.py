"""Declared native camera/drive windows, interventions and retained failures."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import sensor_goal_scenes as scenes


class SceneProtocolTests(unittest.TestCase):
    def test_twelve_declared_scenarios_and_invalid_scene_rejected(self):
        self.assertEqual(len(scenes.SCENARIOS), 12)
        self.assertEqual(len(set(scenes.SCENARIOS)), 12)
        self.assertEqual(set(scenes.SCENARIOS), set(scenes.INTERVENTIONS))
        layout = scenes.previous.make_layouts(scenes.DEVELOPMENT_SEED, 1)[0]
        with self.assertRaises(ValueError): scenes.build_sensor_model('/missing', layout, 'invented')

    def test_construction_failures_remain_in_the_full_planned_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(scenes.robot_profiles, 'verify_assets', return_value={'test': True}), \
                 patch.object(scenes.frozen, '_initial_arm', return_value=[]), \
                 patch.object(scenes, '_run_case', side_effect=RuntimeError('declared test failure')):
                rows = scenes.generate_native_cases(Path(directory), Path('/missing'), seed=scenes.DEVELOPMENT_SEED, count=1)
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(not row['generation_valid'] for row in rows))
            self.assertTrue(all(row['physical_truth'] is None for row in rows))
            for row in rows:
                self.assertTrue((Path(directory)/row['state_file']).is_file())
                self.assertIn('declared test failure', row['generation_errors'][0])
            metadata = json.loads((Path(directory)/'sensor-native-scenes.json').read_text())
            self.assertEqual(metadata['planned_count'], 12)
            self.assertEqual(metadata['invalid_count'], 12)
            with self.assertRaises(ValueError):
                scenes.generate_native_cases(Path(directory), Path('/missing'), seed=scenes.DEVELOPMENT_SEED, count=1)


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set pinned native PiPER assets')
class NativeWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.rows = scenes.generate_native_cases(cls.root, Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS']),
                                                seed=scenes.DEVELOPMENT_SEED, count=1)
        cls.by_scenario = {row['scenario']: row for row in cls.rows}

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def read(self, scenario, key):
        return json.loads((self.root/self.by_scenario[scenario][key]).read_text())

    def test_every_case_has_aligned_native_windows_and_content_addressed_rgb(self):
        self.assertEqual(len(self.rows), 12)
        self.assertTrue(all(row['generation_valid'] for row in self.rows),
                        [(r['scenario'], r['generation_errors']) for r in self.rows if not r['generation_valid']])
        self.assertEqual(sum(row['physical_truth']['complete'] for row in self.rows), 2)
        for row in self.rows:
            name = row['scenario']
            primary = self.read(name, 'primary_packet_file')
            drive = self.read(name, 'actuator_packet_file')
            camera = self.read(name, 'camera_packet_file')
            state = self.read(name, 'state_file')
            self.assertEqual(set(camera), {'schema', 'source_id', 'calibration_id', 'frames'})
            self.assertEqual(camera['schema'], 'dvidia.sensor-camera-window/v1')
            self.assertEqual(camera['source_id'], 'overhead-rgb')
            self.assertEqual(drive['schema'], 'dvidia.sensor-drive-window/v1')
            self.assertEqual(drive['source_id'], 'piper-drive-bus-v1')
            self.assertEqual(len(primary['samples']), 100)
            self.assertEqual(len(drive['samples']), 100)
            self.assertEqual(len(camera['frames']), 11)
            self.assertEqual(len(state['camera_capture_states']), 11)
            self.assertAlmostEqual(camera['frames'][0]['t_s'], drive['history_start_s'])
            self.assertAlmostEqual(primary['history_start_s'], drive['history_start_s'])
            self.assertAlmostEqual(row['now_s']-drive['history_start_s'], .2)
            self.assertEqual(row['now_s'], camera['frames'][-1]['t_s'])
            self.assertEqual(row['now_s'], primary['samples'][-1]['t_s'])
            self.assertEqual(row['now_s'], drive['samples'][-1]['t_s'])
            for a, b in zip(camera['frames'], camera['frames'][1:]): self.assertAlmostEqual(b['t_s']-a['t_s'], .02)
            self.assertEqual(len(set(x['t_s'] for x in primary['samples'])), 100)
            for frame in camera['frames']:
                raw = (self.root/frame['image']).read_bytes()
                self.assertTrue(raw.startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertEqual(Path(frame['image']).stem, hashlib.sha256(raw).hexdigest())
            self.assertEqual(state['solver_warnings'], 0)
            self.assertEqual(state['object_attachment_constraints_added'], 0)
            self.assertFalse(state['external_forces_applied'])

    def test_real_motion_grip_platform_and_declared_aliases(self):
        complete = self.read('complete', 'primary_packet_file')
        alias = self.read('instant_alias', 'primary_packet_file')
        self.assertEqual(complete['samples'][:-1], alias['samples'][:-1])
        self.assertNotEqual(complete['samples'][-1]['linear_speed_m_s'], alias['samples'][-1]['linear_speed_m_s'])
        self.assertEqual(self.read('complete', 'camera_packet_file'), self.read('instant_alias', 'camera_packet_file'))
        self.assertEqual(self.read('complete', 'actuator_packet_file'), self.read('instant_alias', 'actuator_packet_file'))
        moving = self.read('sustained_motion', 'state_file')
        self.assertGreater(moving['native_window_displacement_m'], .09)
        self.assertGreater(moving['physical_truth']['linear_speed_m_s'], .01)
        held = self.by_scenario['unreleased']['physical_truth']
        self.assertTrue(held['supported']); self.assertFalse(held['released'])
        raised = self.read('elevated_support', 'state_file')
        self.assertGreater(raised['platform_support_force_n'], .01)
        self.assertGreater(raised['current_measurement']['target_position_m'][2], .10)
        self.assertFalse(raised['physical_truth']['supported'])
        self.assertFalse(self.by_scenario['lookalike_wrongobject']['physical_truth']['present'])
        self.assertEqual(self.read('complete', 'camera_packet_file'), self.read('lookalike_wrongobject', 'camera_packet_file'))


if __name__ == '__main__': unittest.main()
