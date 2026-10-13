"""Native motion and portable browser-physics artifact contracts."""
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest

from dvidia_training import workstation as ws, reference_sim as sim, robot_profiles


class WorkstationInputs(unittest.TestCase):
    def test_vectors_reject_nonfinite_and_ambiguous_values(self):
        for value in (None, '1,2,3', [True, 0, 0], [float('nan'), 0, 0], [0, 0], [100, 0, 0], [10**1000, 0, 0]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ws._vec(value, 3, 'target')

    def test_protocol_is_bounded_authored_tracking_with_no_grasp_claim(self):
        plan = ws.protocol()
        self.assertEqual(len(plan['actions']), 6)
        self.assertEqual(plan['timestep'], .002)
        for action in plan['actions']:
            self.assertEqual(action['control_steps'], 100)
            self.assertTrue(all(low <= item <= high for item, (low, high) in zip(action['target'], ws.BOUNDS)))
        self.assertIn('not reported outcomes', ws.protocol.__doc__)


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set DVIDIA_ROBOT_PROFILE_ASSETS to the pinned asset root')
class NativeWorkstationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.mj, cls.np = sim._native_imports()
        cls.model = ws.build_model(cls.assets)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.runtime = cls.root/'runtime'
        cls.runtime.mkdir()
        for source in ws.RUNTIME_FILES.values():
            (cls.runtime/source).write_bytes(b'\0asm\x01\0\0\0' if source.endswith('.wasm') else b'fixture runtime\n')
        cls.output = cls.root/'export'
        cls.exported = ws.export(cls.assets, cls.output, runtime_directory=cls.runtime)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_upstream_actuators_mass_inertia_and_joint_contracts_are_unchanged(self):
        baseline = robot_profiles.build_model('piper', self.assets, timestep=.002)
        model = self.model
        self.assertEqual((model.nq, model.nv, model.nu, model.neq, model.nmocap), (15, 14, 7, 1, 0))
        for key in ('actuator_ctrlrange', 'actuator_forcerange', 'actuator_gainprm', 'actuator_biasprm', 'actuator_trnid'):
            self.np.testing.assert_array_equal(getattr(model, key), getattr(baseline, key))
        for key in ('body_mass', 'body_inertia', 'body_gravcomp'):
            self.np.testing.assert_array_equal(getattr(model, key)[:baseline.nbody], getattr(baseline, key))
        self.np.testing.assert_array_equal(model.jnt_range[:8], baseline.jnt_range)
        _, shoe = ws._ids(model)
        self.assertAlmostEqual(model.body_mass[shoe], .3)
        self.assertEqual(int(model.jnt_type[model.body_jntadr[shoe]]), int(self.mj.mjtJoint.mjJNT_FREE))
        self.assertEqual(model.body_gravcomp[shoe], 0)

    def test_reset_is_deterministic_and_has_no_hidden_settling_or_attachment(self):
        first, second = ws.initialize(self.model), ws.initialize(self.model)
        self.np.testing.assert_array_equal(first.qpos, second.qpos)
        self.assertEqual(first.time, 0)
        self.assertFalse(self.np.any(first.qvel))
        self.assertFalse(self.np.any(first.xfrc_applied))
        self.assertFalse(self.np.any(first.qfrc_applied))
        self.assertEqual(first.ncon, 0)
        self.assertLess(ws.state(self.model, first, ws.HOME_TARGET)['position_error_m'], 1e-7)

    def test_target_uses_rate_limited_actuators_and_real_dynamics(self):
        model, data = self.model, ws.initialize(self.model)
        previous_ctrl, previous_qpos = data.ctrl.copy(), data.qpos.copy()
        desired = ws.controller_action(model, data, [.32, -.145, .18], .035)
        self.np.testing.assert_array_equal(data.qpos, previous_qpos)
        self.np.testing.assert_array_equal(data.ctrl, previous_ctrl)
        control = {}
        ws.run_control(model, data, control, [.32, -.145, .18], .035, 1)
        self.assertLessEqual(float(self.np.max(self.np.abs(data.ctrl[:6]-previous_ctrl[:6]))), .0030000001)
        self.assertGreater(float(self.np.linalg.norm(data.qpos[:6]-previous_qpos[:6])), 0)
        self.assertGreater(float(self.np.linalg.norm(data.qpos[:6]-desired[:6])), .01)
        for _ in range(199):
            ws.run_control(model, data, control, [.32, -.145, .18], .035, 10)
        self.assertLess(ws.state(model, data, [.32, -.145, .18])['position_error_m'], .005)
        self.assertTrue(all(item.number == 0 for item in data.warning))

    def test_control_chunking_keeps_the_physics_clock(self):
        model = self.model
        first, second = ws.initialize(model), ws.initialize(model)
        a, b = {}, {}
        for _ in range(20):
            ws.run_control(model, first, a, [.32, -.145, .18], .02, 10)
            for chunk in (3, 2, 5):
                ws.run_control(model, second, b, [.32, -.145, .18], .02, chunk)
        self.np.testing.assert_array_equal(first.qpos, second.qpos)
        self.np.testing.assert_array_equal(first.ctrl, second.ctrl)
        self.assertEqual(a, b)

    def test_shoe_settles_with_gravity_and_true_support_force(self):
        model, data = self.model, ws.initialize(self.model)
        _, shoe = ws._ids(model)
        self.assertAlmostEqual(data.xpos[shoe, 2], .012)
        for _ in range(50):
            ws.step(model, data, ws.HOME_TARGET, .035)
        self.assertLess(abs(data.xpos[shoe, 2]), .0005)
        force = 0.
        table = self.mj.mj_name2id(model, self.mj.mjtObj.mjOBJ_GEOM, 'workstation_table_top')
        for index in range(data.ncon):
            contact = data.contact[index]
            if table in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == table else contact.geom1
                if model.geom_bodyid[other] == shoe:
                    wrench = self.np.zeros(6)
                    self.mj.mj_contactForce(model, data, index, wrench)
                    force += wrench[0]
        self.assertAlmostEqual(force, .3*9.81, delta=.02)
        self.assertFalse(self.np.any(data.xfrc_applied))
        free = ws.build_model(self.assets)
        free.geom_contype[table] = free.geom_conaffinity[table] = 0
        falling = ws.initialize(free)
        for _ in range(10):
            ws.step(free, falling, ws.HOME_TARGET, .035)
        self.assertLess(falling.xpos[shoe, 2], -.10)

    def test_impulse_is_a_bounded_single_tick_and_clears_afterward(self):
        model, data = self.model, ws.initialize(self.model)
        for _ in range(50):
            ws.step(model, data, ws.HOME_TARGET, .035)
        before = ws.state(model, data)['shoe_position']
        ws.step(model, data, ws.HOME_TARGET, .035, impulse_xyz=[0, .12, 0])
        for _ in range(20):
            ws.step(model, data, ws.HOME_TARGET, .035)
        after = ws.state(model, data)['shoe_position']
        self.assertGreater(float(self.np.linalg.norm(self.np.asarray(after)-before)), .001)
        self.assertFalse(self.np.any(data.xfrc_applied))
        for impulse in ([.1, .1, 0], [0, .121, 0], [0, float('nan'), 0]):
            before = data.qpos.copy()
            with self.assertRaises(ValueError):
                ws.step(model, data, ws.HOME_TARGET, .035, impulse_xyz=impulse)
            self.np.testing.assert_array_equal(data.qpos, before)

    def test_invalid_targets_jaws_and_step_counts_fail_before_motion(self):
        data = ws.initialize(self.model)
        before = data.qpos.copy()
        for args in [([.6, 0, .2], .035, 10), (list(ws.HOME_TARGET), .04, 10),
                     (list(ws.HOME_TARGET), .035, True), (list(ws.HOME_TARGET), .035, 0),
                     (list(ws.HOME_TARGET), .035, 101)]:
            with self.assertRaises(ValueError):
                ws.run_control(self.model, data, {}, *args)
        self.np.testing.assert_array_equal(data.qpos, before)

    def test_export_xml_recompiles_to_identical_model_and_home(self):
        result = ws.inspect_export(self.output)
        manifest = json.loads((self.output/'workstation.json').read_text())
        self.assertEqual(result['compiled_model_sha256'], sim.model_digest(self.model))
        self.assertEqual(manifest['interaction_limits']['push_impulse_norm_max_ns'], .12)
        self.assertEqual(manifest['interaction_limits']['push_duration_s'], .002)
        blobs = {item['path']:(self.output/item['path']).read_bytes() for item in manifest['assets']}
        native = self.mj.MjModel.from_xml_string((self.output/'scene.xml').read_text(), assets=blobs)
        self.assertEqual(sim.model_digest(native), sim.model_digest(self.model))
        self.np.testing.assert_array_equal(ws.initialize(native).qpos, manifest['home_qpos'])
        self.assertEqual(json.loads((self.output/'model.json').read_text())['scope'], ws.SCOPE)
        self.assertTrue((self.output/'index.html').exists())
        self.assertFalse((self.output/'workstation.html').exists())
        with self.assertRaises(ValueError):
            ws.export(self.assets, self.output, runtime_directory=self.runtime)

    def _reseal(self, path, name, raw):
        (path/name).write_bytes(raw)
        sealpath = path/'workstation-integrity.json'
        seal = json.loads(sealpath.read_text())
        seal['files'][name] = {'bytes':len(raw), 'sha256':sha256(raw).hexdigest()}
        sealpath.write_bytes(ws._json(seal))

    def test_export_inspection_rejects_resealed_geometry_provenance_and_xml_imports(self):
        import shutil
        cases = ['metadata', 'geometry', 'provenance', 'xml', 'license', 'asset']
        for case in cases:
            directory = self.root/case
            shutil.copytree(self.output, directory)
            if case in ('metadata', 'provenance'):
                value = json.loads((directory/'workstation.json').read_text())
                if case == 'metadata':value['home_qpos'][0] += .1
                else:value['provenance']['commit'] = '0'*40
                self._reseal(directory, 'workstation.json', ws._json(value))
            elif case == 'geometry':
                value = json.loads((directory/'model.json').read_text())
                value['geoms'][0]['size'][0] += .1
                self._reseal(directory, 'model.json', ws._json(value))
            elif case == 'xml':
                raw = (directory/'scene.xml').read_bytes().replace(b'</mujoco>', b'<include file="/tmp/external.xml"/></mujoco>')
                self._reseal(directory, 'scene.xml', raw)
                value = json.loads((directory/'workstation.json').read_text())
                value['native_scene'] = {'path':'scene.xml', 'bytes':len(raw), 'sha256':sha256(raw).hexdigest()}
                self._reseal(directory, 'workstation.json', ws._json(value))
            else:
                name = 'licenses/agilex_piper/LICENSE' if case == 'license' else 'assets/base_link.stl'
                self._reseal(directory, name, b'changed')
            with self.subTest(case=case), self.assertRaises(ValueError):
                ws.inspect_export(directory)

    def test_export_rejects_symlinks_unknown_files_and_missing_runtime_without_partial_output(self):
        import shutil
        directory = self.root/'symlink'
        shutil.copytree(self.output, directory)
        (directory/'extra').symlink_to(self.runtime)
        with self.assertRaises(ValueError):ws.inspect_export(directory)
        missing_runtime = self.root/'missing-runtime'
        missing_runtime.mkdir()
        destination = self.root/'must-not-exist'
        with self.assertRaises((ValueError, OSError)):
            ws.export(self.assets, destination, runtime_directory=missing_runtime)
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob('.dvidia-workstation-*')))


if __name__ == '__main__':
    unittest.main()
