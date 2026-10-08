"""Scene inputs must reach native physics without altering installed source evidence."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from dvidia_training.arm_runner import authored_grounding, config_for, validate_scene
from dvidia_training.skillspace import load_skillspace
from dvidia_training.studio import preview_worker, run_worker
from dvidia_training.arm_pack import export_arm_pack, verify_arm_pack


class SceneInputTests(unittest.TestCase):
    def test_dimensions_material_and_actuator_caps_reach_native_preview(self):
        scene = {'object_size': [.03, .05, .05], 'object_mass': .08, 'object_friction': .4,
                 'jaw_force_limit': .2, 'joint_torque_limit': 20.,
                 'object_position': [.42, -.04, .315], 'target_position': [.54, .1, .315]}
        result = preview_worker(scene)
        self.assertEqual(result['observation']['object_size'], scene['object_size'])
        for name in ('object_mass', 'object_friction', 'jaw_force_limit', 'joint_torque_limit'):
            self.assertEqual(result['info']['config'][name], scene[name])
        self.assertAlmostEqual(result['observation']['target_position'][2], .315)

    def test_scene_overrides_do_not_modify_installed_grounding(self):
        raw = (Path(__file__).resolve().parents[1]/'src/dvidia_training/place_cup.skill.json').read_bytes()
        skill = load_skillspace(raw, authored_grounding(raw))
        configuration = config_for(skill, {'object_size': [.03, .05, .04], 'object_mass': .08})
        self.assertEqual(configuration.object_half_size, (.015, .025, .02))
        self.assertEqual(configuration.object_mass, .08)
        self.assertEqual(skill.object_size, (.04, .04, .04))
        self.assertEqual(skill.object_mass, .04)

    def test_nonfinite_wrong_units_and_out_of_profile_inputs_are_rejected(self):
        for scene in ({'object_size': [30, 40, 50]}, {'object_size': [True, .04, .04]},
                      {'object_mass': float('inf')}, {'jaw_force_limit': 0},
                      {'joint_torque_limit': 41}, {'object_friction': .1},
                      {'object_orientation': [1, 0, 0, 0]}):
            with self.subTest(scene=scene), self.assertRaises(ValueError):
                validate_scene(scene)

    def test_falsy_invalid_scenes_cannot_silently_select_defaults(self):
        raw = (Path(__file__).resolve().parents[1]/'src/dvidia_training/place_cup.skill.json').read_bytes()
        skill = load_skillspace(raw, authored_grounding(raw))
        for scene in (False, [], 0, ''):
            with self.subTest(scene=scene), self.assertRaises(ValueError):
                config_for(skill, scene)

    def test_native_run_preserves_exportable_inputs_and_contact_feedback(self):
        raw = (Path(__file__).resolve().parents[1]/'src/dvidia_training/place_cup.skill.json').read_bytes()
        grounding = authored_grounding(raw)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = run_worker(raw, grounding, None, 'placement', 'a'*24, None, output)
            self.assertTrue(result['episodes'][0]['success'])
            self.assertIn('pad_normal_forces', result['episodes'][0]['trace'][-1])
            self.assertEqual((output/'source.json').read_bytes(), raw)
            self.assertTrue((output/'grounding.json').is_file())
            manifest = export_arm_pack(output)
            self.assertEqual(verify_arm_pack(output/'skill-adapter.zip'), manifest)
            self.assertEqual(manifest['runtime'], result['runtime'])


if __name__ == '__main__':
    unittest.main()
