"""Native geometry and sealed pose reconstruction, without physics stepping."""
from copy import deepcopy
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import reference_replay as replay
from dvidia_training import reference_sim as sim


BUNDLE = Path(__file__).resolve().parent.parent / 'benchmarks/hardware-reference-20261011'


def small_report(source=None):
    report = (deepcopy(source) if source is not None
              else json.loads((BUNDLE / 'report.json').read_text()))
    report['protocol']['repeats'] = 1
    report['protocol']['seconds_per_case'] = .1
    report['attempts'] = [attempt for attempt in report['attempts'] if attempt['repeat'] == 0]
    for attempt in report['attempts']:
        attempt['simulated_seconds'] = .1
        attempt['trajectory'] = attempt['trajectory'][:6]
    report['attempted'] = report['valid_attempts'] = 9
    return report


def write_bundle(root, report):
    root.mkdir()
    (root / 'report.json').write_text(json.dumps(report, allow_nan=False))
    (root / 'report.html').write_text('<html>Frozen receipt fixture</html>')
    files = {name: {'bytes': (root / name).stat().st_size,
                    'sha256': sha256((root / name).read_bytes()).hexdigest()}
             for name in ('report.json', 'report.html')}
    (root / 'integrity.json').write_text(json.dumps({'schema': 'dvidia.reference-integrity/v1', 'files': files}))


class ReplayInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = small_report()
        cls.provenance = cls.report['audit']['asset_provenance']

    def test_receipt_snapshot_is_bounded_and_rechecks_original_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'receipt'
            write_bundle(root, self.report)
            parsed, files = replay._receipt(root)
            self.assertEqual(parsed, self.report)
            self.assertEqual(set(files), {'report.json', 'report.html', 'integrity.json'})
            with patch('dvidia_training.embodiment_cli.inspect_bundle', return_value={}):
                (root / 'report.json').write_bytes((root / 'report.json').read_bytes() + b' ')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    replay._receipt(root)

    def test_duplicate_nonfinite_and_overlarge_inputs_are_rejected(self):
        for content in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'\xff'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                replay._decode(content)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'large'
            path.write_bytes(b'x' * 17)
            with self.assertRaisesRegex(ValueError, 'bounded'):
                replay._read(path, 16)
            alias = Path(directory) / 'alias'
            try:
                alias.symlink_to(path)
            except OSError:
                return
            with self.assertRaisesRegex(ValueError, 'non-symlink'):
                replay._read(alias, 32)

    def test_growth_after_size_check_is_rejected_with_original_bounded_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'growing'
            path.write_bytes(b'x' * 16)
            original_read = sim._regular_bytes
            def grow_then_read(file, size):
                self.assertEqual(size, 16)
                file.write_bytes(b'x' * 17)
                return original_read(file, size)
            with patch.object(sim, '_regular_bytes', side_effect=grow_then_read) as opened:
                with self.assertRaisesRegex(ValueError, 'size mismatch'):
                    replay._read(path, 16)
                opened.assert_called_once_with(path, 16)

    def test_valid_receipt_selects_only_first_valid_repeat(self):
        attempts, joints = replay._validate_report(self.report, self.provenance)
        self.assertEqual(len(attempts), 9)
        self.assertEqual(len(joints), 14)
        self.assertEqual(sum(len(item['trajectory']) for item in attempts), 54)
        changed = deepcopy(self.report)
        changed['attempts'][0]['valid'] = False
        changed['valid_attempts'] -= 1
        attempts, _ = replay._validate_report(changed, self.provenance)
        self.assertEqual(len(attempts), 8)

    def test_asset_profile_engine_and_coordinate_compatibility_are_required(self):
        mutations = (
            lambda r: r.update(profile='other-arm'),
            lambda r: r.update(physical_robot_ready=True),
            lambda r: r.update(task_success_evaluated=True),
            lambda r: r['hardware'].update(mujoco='3.14.0'),
            lambda r: r['audit']['asset_provenance'].update(commit='0' * 40),
            lambda r: r['audit']['asset_provenance'].update(lock_sha256='0' * 64),
            lambda r: r['audit']['inventory'].update(joint_coordinates_nq=6),
            lambda r: r['audit']['inventory']['joints'][7].update(qpos_address=8),
            lambda r: r['audit']['inventory']['joints'][0].update(name='unexpected'),
        )
        for mutate in mutations:
            changed = deepcopy(self.report)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                replay._validate_report(changed, self.provenance)

    def test_protocol_and_binary_frame_inputs_are_strict(self):
        mutations = (
            lambda r: r['protocol'].update(repeats=True),
            lambda r: r['protocol'].update(control_dt_seconds=.03),
            lambda r: r['protocol'].update(seconds_per_case=.11),
            lambda r: r['protocol'].update(seconds_per_case=10 ** 1000),
            lambda r: r['attempts'].pop(),
            lambda r: r['attempts'].__setitem__(1, deepcopy(r['attempts'][0])),
            lambda r: r['attempts'][0].update(repeat=False),
            lambda r: r['attempts'][0].update(timestep_seconds=True),
            lambda r: r['attempts'][0].update(compiled_model_sha256='bad'),
            lambda r: r['attempts'][0]['trajectory'].pop(),
            lambda r: r['attempts'][0]['trajectory'][0].update(t_seconds=.02),
            lambda r: r['attempts'][0]['trajectory'][1].update(t_seconds=0),
            lambda r: r['attempts'][0]['trajectory'][0].update(joint_position_rad=[0] * 5),
            lambda r: r['attempts'][0]['trajectory'][0].update(linkage_position_rad=[0] * 9),
            lambda r: r['attempts'][0]['trajectory'][0].update(tcp_position_m=[float('nan'), 0, 0]),
            lambda r: r['attempts'][0]['trajectory'][0].update(tcp_position_m=[10 ** 1000, 0, 0]),
            lambda r: r['attempts'][0]['trajectory'][0].update(tcp_quaternion_wxyz=[2, 0, 0, 0]),
            lambda r: r['attempts'][0]['trajectory'][0].update(pad_centers_m=[[0, 0, 0]]),
            lambda r: r['attempts'][0]['trajectory'][0].update(contact_count=True),
            lambda r: r['attempts'][0]['trajectory'][0].update(issued_gripper_0_255=256),
        )
        for mutate in mutations:
            changed = deepcopy(self.report)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                replay._validate_report(changed, self.provenance)


@unittest.skipUnless(os.environ.get('DVIDIA_REFERENCE_ASSETS'),
                     'Set DVIDIA_REFERENCE_ASSETS to the pinned local reference assets')
class NativeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(os.environ['DVIDIA_REFERENCE_ASSETS'])
        cls.mj, cls.np = sim._native_imports()
        cls.temporary = tempfile.TemporaryDirectory()
        # A native model's serialized digest can vary across architectures.
        # Keep the published macOS receipt frozen; exercise reconstruction with
        # a newly sealed local-platform fixture and retain strict digest checks.
        cls.report = sim.benchmark_reference(cls.assets, repeats=1, seconds=4.)
        if cls.report['valid_attempts'] != 9:
            raise AssertionError('Local native replay fixture must contain nine valid attempts')
        cls.bundle = Path(cls.temporary.name) / 'receipt'
        write_bundle(cls.bundle, cls.report)
        cls.output = Path(cls.temporary.name) / 'replay'
        original_read = replay._read
        def fixture_frontend(path, limit):
            if Path(path).parent == Path(replay.__file__).parent and Path(path).name.startswith('reference_replay.'):
                return b'<!-- offline frontend fixture -->'
            return original_read(path, limit)
        with patch.object(replay, '_read', side_effect=fixture_frontend), \
                patch.object(cls.mj, 'mj_step', side_effect=AssertionError('Replay must not step physics')):
            cls.summary = replay.export_replay(cls.assets, cls.bundle, cls.output)
        cls.scene = json.loads((cls.output / 'scene.json').read_text())
        cls.motion = json.loads((cls.output / 'motion.json').read_text())
        cls.geometry = (cls.output / 'geometry.bin').read_bytes()
        cls.poses = cls.np.frombuffer((cls.output / 'motion.bin').read_bytes(), dtype='<f4')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_export_has_native_meshes_and_reconstructs_every_recorded_first_repeat_frame(self):
        self.assertEqual(self.summary['cases'], 9)
        self.assertEqual(self.summary['reconstructed_frames'], 1809)
        self.assertEqual(self.summary['body_count'], 22)
        self.assertEqual(len(self.scene['meshes']), 28)
        self.assertEqual(len(self.scene['geoms']), 58)
        self.assertEqual(sum(x['collision'] for x in self.scene['geoms']), 24)
        self.assertEqual(self.motion['arm_joint_names'], list(sim.ARM_JOINTS))
        self.assertTrue(all(value <= 1e-10 for value in self.summary['reconstruction_max_absolute_residual'].values()))
        self.assertEqual(self.motion['frame_stride_floats'], 22 * 7)
        self.assertFalse(self.summary['physical_robot_ready'])

    def test_exact_vertex_deduplication_preserves_every_native_triangle_coordinate(self):
        model = sim.build_model(self.assets, timestep=.002)
        np = self.np
        for mesh in self.scene['meshes']:
            index = mesh['id']
            vertices = np.frombuffer(self.geometry, dtype='<f4', count=mesh['vertex_count'] * 3,
                                     offset=mesh['position_byte_offset']).reshape(-1, 3)
            faces = np.frombuffer(self.geometry, dtype='<u4', count=mesh['index_count'],
                                  offset=mesh['index_byte_offset']).reshape(-1, 3)
            start, count = model.mesh_vertadr[index], model.mesh_vertnum[index]
            native_vertices = np.asarray(model.mesh_vert[start:start + count], dtype='<f4')
            start, count = model.mesh_faceadr[index], model.mesh_facenum[index]
            native_faces = model.mesh_face[start:start + count]
            self.assertEqual(vertices[faces].tobytes(), native_vertices[native_faces].tobytes())
            self.assertEqual(mesh['position_byte_offset'] % 4, 0)
            self.assertEqual(mesh['index_byte_offset'] % 4, 0)
            self.assertLessEqual(mesh['vertex_count'], mesh['native_vertex_count'])
        self.assertLess(sum(x['vertex_count'] for x in self.scene['meshes']), model.nmeshvert)

    def test_every_binary_frame_and_local_geom_transform_agrees_with_native_world_geometry(self):
        mj, np = self.mj, self.np
        for case in self.motion['cases']:
            model = sim.build_model(self.assets, timestep=case['timestep_seconds'])
            data = mj.MjData(model)
            poses = self.poses[case['pose_byte_offset'] // 4:
                               case['pose_byte_offset'] // 4 + case['frame_count'] * self.motion['frame_stride_floats']]
            poses = poses.reshape(case['frame_count'], model.nbody, 7)
            geom_body = np.asarray([x['body_id'] for x in self.scene['geoms']])
            local_position = np.asarray([x['position'] for x in self.scene['geoms']])
            local_matrix = np.asarray([self._rotation(x['quaternion_wxyz']) for x in self.scene['geoms']])
            for frame, row in enumerate(case['rows']):
                data.qpos[:] = row['joint_position_rad'] + row['linkage_position_rad']
                mj.mj_forward(model, data)
                expected = np.concatenate((data.xpos, data.xquat), axis=1).astype('<f4')
                self.assertEqual(poses[frame].tobytes(), expected.tobytes())
                body_rotation = np.asarray([self._rotation(x[3:]) for x in poses[frame]])
                world_position = poses[frame, geom_body, :3] + np.einsum('gij,gj->gi', body_rotation[geom_body], local_position)
                world_rotation = np.einsum('gij,gjk->gik', body_rotation[geom_body], local_matrix)
                np.testing.assert_allclose(world_position, data.geom_xpos, atol=2e-7, rtol=0)
                np.testing.assert_allclose(world_rotation.reshape(-1, 9), data.geom_xmat, atol=3e-7, rtol=0)

    def _rotation(self, quaternion):
        matrix = self.np.empty(9)
        self.mj.mju_quat2Mat(matrix, self.np.asarray(quaternion, dtype=float))
        return matrix.reshape(3, 3)

    def test_exported_colors_agree_with_native_visual_materials(self):
        mj, np = self.mj, self.np
        model = sim.build_model(self.assets, timestep=.002)
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        option = mj.MjvOption()
        option.geomgroup[:] = 1
        scene = mj.MjvScene(model, maxgeom=256)
        mj.mjv_updateScene(model, data, option, None, mj.MjvCamera(), mj.mjtCatBit.mjCAT_ALL, scene)
        observed = {geom.objid: geom.rgba for geom in scene.geoms[:scene.ngeom]
                    if geom.objtype == mj.mjtObj.mjOBJ_GEOM}
        for geom in self.scene['geoms']:
            self.assertEqual(geom['group'], int(model.geom_group[geom['geom_id']]))
            np.testing.assert_array_equal(geom['rgba'], observed[geom['geom_id']])

    def test_original_receipts_licenses_and_compressed_data_are_complete_and_sealed(self):
        for name in ('report.json', 'report.html', 'integrity.json'):
            self.assertEqual((self.output / 'receipt' / name).read_bytes(), (self.bundle / name).read_bytes())
        for folder in ('universal_robots_ur5e', 'robotiq_2f85'):
            for name in ('LICENSE', 'README.md'):
                self.assertEqual((self.output / 'licenses' / folder / name).read_bytes(), (self.assets / folder / name).read_bytes())
        self.assertIn('DVIDIA contributors', (self.output / 'LICENSE').read_text())
        self.assertEqual(gzip.decompress((self.output / 'geometry.bin.gz').read_bytes()), self.geometry)
        self.assertEqual(gzip.decompress((self.output / 'motion.bin.gz').read_bytes()), (self.output / 'motion.bin').read_bytes())
        self.assertEqual((self.output / 'geometry.bin.gz').read_bytes(), gzip.compress(self.geometry, mtime=0))
        seal = json.loads((self.output / 'replay-integrity.json').read_text())
        self.assertEqual(seal['schema'], 'dvidia.reference-replay-integrity/v1')
        for name, expected in seal['files'].items():
            data = (self.output / name).read_bytes()
            self.assertEqual(len(data), expected['bytes'])
            self.assertEqual(sha256(data).hexdigest(), expected['sha256'])
        self.assertEqual(self.motion['source']['report_sha256'], sha256((self.bundle / 'report.json').read_bytes()).hexdigest())

    def test_incompatible_compiled_model_is_explicitly_rejected_before_reconstruction(self):
        changed = small_report(self.report)
        changed['attempts'][0]['compiled_model_sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_bundle(root / 'receipt', changed)
            with patch.object(replay, '_poses', side_effect=AssertionError('Incompatible model must not reconstruct')):
                with self.assertRaisesRegex(ValueError, 'compiled model digest differs'):
                    replay.export_replay(self.assets, root / 'receipt', root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_incompatible_or_modified_receipts_fail_before_creating_output(self):
        mutations = (
            lambda r: r['attempts'][0].update(compiled_model_sha256='0' * 64),
            lambda r: r['audit']['inventory']['joints'][7].update(name='gripper/wrong'),
            lambda r: r['attempts'][0]['trajectory'][0]['tcp_position_m'].__setitem__(0, 10.),
            lambda r: r['attempts'][0]['trajectory'][0]['arm_points_m'][0].__setitem__(0, 10.),
            lambda r: r['attempts'][0]['trajectory'][0]['pad_centers_m'][0].__setitem__(0, 10.),
            lambda r: r['attempts'][0]['trajectory'][0].update(pad_center_separation_m=10.),
            lambda r: r['attempts'][0]['trajectory'][0].update(tcp_quaternion_wxyz=[1, 0, 0, 0]),
        )
        for mutate in mutations:
            changed = small_report(self.report)
            mutate(changed)
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_bundle(root / 'receipt', changed)
                with self.assertRaises(ValueError):
                    replay.export_replay(self.assets, root / 'receipt', root / 'output')
                self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
