"""Candidate persistence and exact-scene proof binding; synthetic JSON fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

from dvidia_training.capsule_installer import (capsule_runtime, install_capsule, read_capsule,
    read_capsule_installation, record_qualification, scene_digest)
from dvidia_training.skill_capsule import build_capsule, encode_capsule
from tests.test_skill_capsule import model, grounding, native_fixture, SCENE


class CapsuleInstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = (Path(__file__).resolve().parents[1]/'src/dvidia_training/place_cup.skill.json').read_bytes()
        self.capsule = build_capsule(source, grounding(), model(), runtime=capsule_runtime())
        self.record = install_capsule(encode_capsule(self.capsule), self.root)
        self.identity = self.record['installation_id']

    def result(self, *, control=False):
        result = native_fixture(self.capsule, control=control)
        result['runtime'] = self.capsule['runtime']
        return result

    def test_import_remains_candidate_and_cannot_run_without_local_scene_proof(self):
        self.assertEqual(self.record['status'], 'candidate')
        self.assertFalse(self.record['physical_robot_ready'])
        with self.assertRaisesRegex(ValueError, 'Validate'):
            read_capsule_installation(self.root, self.identity, SCENE, require_qualified=True)

    def test_qualification_is_specific_to_material_and_actuator_inputs(self):
        student = self.result()
        control = self.result(control=True)
        record_qualification(self.root, self.identity, SCENE, student, control)
        validated = read_capsule_installation(self.root, self.identity, SCENE, require_qualified=True)
        self.assertEqual(validated['status'], 'validated_simulation_scene')
        for key, value in [('object_mass', .06), ('jaw_force_limit', .3), ('seed', 1)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Validate'):
                read_capsule_installation(self.root, self.identity, {**SCENE, key:value}, require_qualified=True)

    def test_tampered_native_control_invalidates_persisted_scene_proof(self):
        record_qualification(self.root, self.identity, SCENE,
                             self.result(), self.result(control=True))
        path = self.root/self.identity/'qualifications'/scene_digest(SCENE)/'controls.json'
        result = json.loads(path.read_text()); result['episodes'][0]['success'] = True
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'Validate'):
            read_capsule_installation(self.root, self.identity, SCENE, require_qualified=True)

    def test_changed_capsule_cannot_reuse_an_installation_identity(self):
        path = self.root/self.identity/'capsule.json'
        raw = json.loads(path.read_text()); raw['source']['text'] += ' '
        path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError): read_capsule(self.root, self.identity)

    def test_scene_hash_normalizes_numeric_representation(self):
        self.assertEqual(scene_digest(SCENE), scene_digest({**SCENE, 'jaw_force_limit':15, 'joint_torque_limit':40}))


if __name__ == '__main__':
    unittest.main()
