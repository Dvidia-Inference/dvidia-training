"""Pinned mechanics, asset integrity and explicitly synthetic diagnostics."""
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dvidia_training import reference_sim as sim


class DownloadResponse(io.BytesIO):
    def __init__(self, content, url):
        super().__init__(content)
        self.url = url

    def geturl(self):
        return self.url


class ReferenceAssetTests(unittest.TestCase):
    def setUp(self):
        self.contents = {
            'universal_robots_ur5e/LICENSE': b'arm notice\n',
            'robotiq_2f85/assets/pad.stl': b'original pad mesh',
            'shadow_hand/left_hand.xml': b'<mujoco model="left"/>',
        }
        self.lock = {
            'schema': 'dvidia.reference-assets/v1',
            'repository': 'https://github.com/google-deepmind/mujoco_menagerie',
            'commit': sim.COMMIT,
            'licenses': {'universal_robots_ur5e': 'BSD-3-Clause',
                         'robotiq_2f85': 'BSD-2-Clause', 'shadow_hand': 'Apache-2.0'},
            'files': [{'path': name, 'bytes': len(content), 'sha256': sha256(content).hexdigest()}
                      for name, content in self.contents.items()],
        }

    def write_assets(self, root, *, hands=False):
        for name, content in self.contents.items():
            if name.startswith('shadow_hand/') and not hands:
                continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def downloader(self, request, timeout):
        self.assertEqual(timeout, 30)
        expected = f'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{sim.COMMIT}/'
        self.assertTrue(request.full_url.startswith(expected))
        self.assertIn('DVIDIA', request.get_header('User-agent'))
        return DownloadResponse(self.contents[request.full_url.removeprefix(expected)], request.full_url)

    def test_packaged_lock_pins_primary_sources_and_licenses(self):
        lock = sim.asset_lock()
        self.assertEqual(lock['commit'], sim.COMMIT)
        self.assertEqual(lock['repository'], 'https://github.com/google-deepmind/mujoco_menagerie')
        self.assertEqual(lock['licenses'], {'universal_robots_ur5e': 'BSD-3-Clause',
                                          'robotiq_2f85': 'BSD-2-Clause', 'shadow_hand': 'Apache-2.0'})
        paths = [item['path'] for item in lock['files']]
        self.assertEqual(len(paths), len(set(paths)))
        for item in lock['files']:
            path = Path(item['path'])
            self.assertFalse(path.is_absolute())
            self.assertNotIn('..', path.parts)
            self.assertIn(path.parts[0], lock['licenses'])
            self.assertIs(type(item['bytes']), int)
            self.assertGreater(item['bytes'], 0)
            self.assertRegex(item['sha256'], r'^[0-9a-f]{64}$')
        by_name = {item['path']: item for item in lock['files']}
        self.assertEqual(by_name['universal_robots_ur5e/ur5e.xml']['sha256'],
                         'ffd0a7b336d3c2d27c35a15cc0d4aaa95196769aca190ac8d6522d3318c345a0')
        self.assertEqual(by_name['robotiq_2f85/2f85.xml']['sha256'],
                         'd48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de')
        for prefix in lock['licenses']:
            self.assertIn(f'{prefix}/LICENSE', by_name)

    def test_default_install_uses_only_allowlisted_arm_and_gripper_bytes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sim, 'asset_lock', return_value=self.lock), \
                patch.object(sim, 'urlopen', side_effect=self.downloader) as opened:
            output = Path(directory) / 'fresh'
            result = sim.fetch_assets(output)
            self.assertEqual(opened.call_count, 2)
            self.assertEqual(result['files_verified'], 2)
            self.assertEqual(result['bytes_verified'], sum(len(v) for k, v in self.contents.items()
                                                         if not k.startswith('shadow_hand/')))
            self.assertNotIn('shadow_hand', result['licenses'])
            self.assertFalse((output / 'shadow_hand').exists())
            self.assertEqual((output / 'universal_robots_ur5e/LICENSE').read_bytes(), b'arm notice\n')

    def test_hands_install_includes_optional_allowlist_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sim, 'asset_lock', return_value=self.lock), \
                patch.object(sim, 'urlopen', side_effect=self.downloader) as opened:
            output = Path(directory) / 'fresh'
            result = sim.fetch_assets(output, hands=True)
            self.assertEqual(result['files_verified'], 3)
            self.assertEqual(result['licenses']['shadow_hand'], 'Apache-2.0')
            original = {name: (output / name).read_bytes() for name in self.contents}
            with self.assertRaises(FileExistsError):
                sim.fetch_assets(output, hands=True)
            self.assertEqual(opened.call_count, 3)
            self.assertEqual(original, {name: (output / name).read_bytes() for name in self.contents})

    def test_wrong_size_or_digest_download_is_not_written(self):
        name = self.lock['files'][0]['path']
        variants = (self.contents[name][:-1], self.contents[name] + b'x', b'x' * len(self.contents[name]))
        with patch.object(sim, 'asset_lock', return_value=self.lock):
            for content in variants:
                with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / 'failed'
                    def broken(request, timeout):
                        return DownloadResponse(content, request.full_url)
                    with patch.object(sim, 'urlopen', side_effect=broken), self.assertRaises(ValueError):
                        sim.fetch_assets(output)
                    self.assertFalse((output / name).exists())
                    with self.assertRaises((ValueError, OSError)):
                        sim.verify_assets(output)

    def test_external_redirect_is_rejected_before_any_asset_write(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sim, 'asset_lock', return_value=self.lock):
            output = Path(directory) / 'redirect'
            with patch.object(sim, 'urlopen', return_value=DownloadResponse(b'arm notice\n',
                                                                          'https://raw.githubusercontent.com.evil.invalid/file')), \
                    self.assertRaisesRegex(ValueError, 'redirected'):
                sim.fetch_assets(output)
            self.assertEqual(list(output.iterdir()), [])

    def test_partial_network_failure_preserves_receipts_and_cannot_verify(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sim, 'asset_lock', return_value=self.lock):
            output = Path(directory) / 'partial'
            first = DownloadResponse(self.contents[self.lock['files'][0]['path']],
                                     f'https://raw.githubusercontent.com/{sim.COMMIT}/first')
            with patch.object(sim, 'urlopen', side_effect=[first, OSError('connection closed')]), \
                    self.assertRaises(OSError):
                sim.fetch_assets(output)
            self.assertEqual((output / self.lock['files'][0]['path']).read_bytes(), b'arm notice\n')
            with self.assertRaises((ValueError, OSError)):
                sim.verify_assets(output)
            with self.assertRaises(FileExistsError):
                sim.fetch_assets(output)

    def test_verify_detects_missing_size_digest_and_directory_replacements(self):
        with patch.object(sim, 'asset_lock', return_value=self.lock):
            for failure in ('missing', 'size', 'digest', 'directory'):
                with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.write_assets(root)
                    path = root / self.lock['files'][0]['path']
                    if failure == 'missing':
                        path.unlink()
                    elif failure == 'size':
                        path.write_bytes(path.read_bytes() + b'!')
                    elif failure == 'digest':
                        path.write_bytes(b'x' * path.stat().st_size)
                    else:
                        path.unlink()
                        path.mkdir()
                    with self.assertRaises((ValueError, OSError)):
                        sim.verify_assets(root)

    def test_verify_checks_optional_hand_files_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sim, 'asset_lock', return_value=self.lock):
            root = Path(directory)
            self.write_assets(root)
            self.assertEqual(sim.verify_assets(root)['files_verified'], 2)
            with self.assertRaises((ValueError, OSError)):
                sim.verify_assets(root, hands=True)
            self.write_assets(root, hands=True)
            self.assertEqual(sim.verify_assets(root, hands=True)['files_verified'], 3)

    def test_verify_rejects_root_file_and_parent_symlinks(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sim, 'asset_lock', return_value=self.lock):
            parent = Path(directory)
            root = parent / 'assets'
            root.mkdir()
            self.write_assets(root)
            root_link = parent / 'redirected-root'
            try:
                root_link.symlink_to(root, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest('This host does not permit symlink creation')
            with self.assertRaises(ValueError):
                sim.verify_assets(root_link)
            asset = root / self.lock['files'][0]['path']
            original = parent / 'original-license'
            original.write_bytes(asset.read_bytes())
            asset.unlink()
            asset.symlink_to(original)
            with self.assertRaises(ValueError):
                sim.verify_assets(root)
            asset.unlink()
            asset.write_bytes(original.read_bytes())
            mesh_dir = root / 'robotiq_2f85/assets'
            real_dir = parent / 'actual-meshes'
            mesh_dir.rename(real_dir)
            mesh_dir.symlink_to(real_dir, target_is_directory=True)
            with self.assertRaises(ValueError):
                sim.verify_assets(root)

    def test_binary_verification_is_portable_without_posix_only_open_flags(self):
        portable = SimpleNamespace(O_RDONLY=os.O_RDONLY, open=os.open, fdopen=os.fdopen, fstat=os.fstat)
        with tempfile.TemporaryDirectory() as directory, patch.object(sim, 'asset_lock', return_value=self.lock):
            root = Path(directory)
            self.write_assets(root)
            with patch.object(sim, 'os', portable):
                self.assertEqual(sim.verify_assets(root)['files_verified'], 2)

    def test_after_open_identity_regular_type_and_size_are_checked(self):
        actual_fstat = os.fstat
        with tempfile.TemporaryDirectory() as directory, patch.object(sim, 'asset_lock', return_value=self.lock):
            root = Path(directory)
            self.write_assets(root)
            source = root / self.lock['files'][0]['path']
            for changed in ('mode', 'size', 'identity'):
                if changed == 'identity' and not source.stat().st_ino:
                    continue
                def fstat(descriptor):
                    original = actual_fstat(descriptor)
                    fields = {name: getattr(original, name) for name in ('st_mode', 'st_size', 'st_ino', 'st_dev')}
                    if changed == 'mode':
                        fields['st_mode'] = 0
                    elif changed == 'size':
                        fields['st_size'] += 1
                    else:
                        fields['st_ino'] += 1
                    return SimpleNamespace(**fields)
                portable = SimpleNamespace(O_RDONLY=os.O_RDONLY, open=os.open, fdopen=os.fdopen, fstat=fstat)
                with self.subTest(changed=changed), patch.object(sim, 'os', portable), self.assertRaises(ValueError):
                    sim.verify_assets(root)


@unittest.skipUnless(os.environ.get('DVIDIA_REFERENCE_ASSETS'),
                     'Set DVIDIA_REFERENCE_ASSETS to the pinned local reference assets')
class NativeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(os.environ['DVIDIA_REFERENCE_ASSETS'])
        cls.mj, cls.np = sim._native_imports()
        cls.audit = sim.audit_reference(cls.assets)
        cls.model = sim.build_model(cls.assets)

    def test_mechanical_contract_distinguishes_linkage_coordinates_from_commands(self):
        inventory = self.audit['inventory']
        self.assertEqual((inventory['joint_coordinates_nq'], inventory['velocity_dofs_nv'],
                          len(inventory['joints']), inventory['actuator_count']), (14, 14, 14, 7))
        self.assertEqual((inventory['equality_constraints'], inventory['tendons']), (3, 1))
        self.assertEqual(inventory['native_sensor_channels'], 0)
        self.assertEqual(inventory['native_sensor_definitions'], 0)
        self.assertEqual(self.model.nmocap, 0)
        self.assertTrue(all(item['type'] == int(self.mj.mjtJoint.mjJNT_HINGE)
                            for item in inventory['joints']))
        self.assertEqual([item['name'] for item in inventory['joints'][:6]], list(sim.ARM_JOINTS))
        self.assertTrue(all(item['name'].startswith('gripper/') for item in inventory['joints'][6:]))
        jaw = inventory['actuators'][-1]
        self.assertEqual(jaw['name'], 'gripper/fingers_actuator')
        self.assertEqual(jaw['target'], 'gripper/split')
        self.assertEqual(jaw['transmission_type'], int(self.mj.mjtTrn.mjTRN_TENDON))
        self.assertEqual(jaw['control_range'], [0., 255.])

    def test_native_compile_uses_verified_snapshot_in_memory_without_path_rereads(self):
        metadata, blobs = sim._asset_payload(self.assets)
        # The caller path does not exist. Every XML include and mesh must come
        # from the verified snapshot rather than a second filesystem read.
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'assets-do-not-exist'
            with patch.object(sim, '_asset_payload', return_value=(metadata, blobs)):
                model = sim.build_model(missing)
            self.assertFalse(missing.exists())
            self.assertEqual((model.nq, model.nv, model.nu, model.neq, model.ntendon), (14, 14, 7, 3, 1))
            self.assertEqual(sim.model_digest(model), sim.model_digest(self.model))

    def test_moving_arm_links_have_collision_coverage_and_base_gap_is_reported(self):
        self.assertEqual(self.audit['arm_link_collision_coverage'],
                         dict(zip(sim.ARM_BODIES, (1, 2, 2, 1, 2, 1))))
        self.assertEqual(self.audit['fixed_base_collision_geoms'], 0)
        self.assertGreater(self.audit['inventory']['visual_geom_count'], 0)
        self.assertGreater(self.audit['inventory']['collision_geom_count'], 0)
        self.assertFalse(self.audit['physical_robot_ready'])
        self.assertFalse(self.audit['bimanual_assembly_implemented'])
        self.assertTrue(all(value == 'unmeasured' for value in self.audit['calibration_gates'].values()))
        self.assertEqual(self.audit['warnings'], [])

    def test_namespace_and_tool_sites_preserve_upstream_attachment(self):
        sites = [self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_SITE, index)
                 for index in range(self.model.nsite)]
        self.assertEqual(sites, ['attachment_site', 'gripper/pinch'])
        data = sim._initial(self.model)
        attachment = self.model.site('attachment_site').id
        pinch = self.model.site('gripper/pinch').id
        self.assertTrue(self.np.isfinite(data.site_xpos).all())
        # A tool mount has a finite offset; pinch is not silently the bare flange.
        self.assertGreater(float(self.np.linalg.norm(data.site_xpos[attachment] - data.site_xpos[pinch])), .1)

    def test_native_servo_control_and_force_ranges_are_bounded(self):
        self.assertTrue(self.model.actuator_ctrllimited.all())
        self.assertTrue(self.model.actuator_forcelimited.all())
        self.np.testing.assert_array_equal(self.model.actuator_forcerange,
                                          [[-150, 150]] * 3 + [[-28, 28]] * 3 + [[-5, 5]])
        for case in sim.CASES:
            for elapsed in (0., .1, 1., 2., 3., 4., 10.):
                command = sim._command(case, elapsed, self.model.key_ctrl[0])
                self.assertTrue(self.np.all(command >= self.model.actuator_ctrlrange[:, 0]))
                self.assertTrue(self.np.all(command <= self.model.actuator_ctrlrange[:, 1]))
        # A command beyond the range cannot bypass native force saturation.
        data = sim._initial(self.model)
        data.ctrl[:] = 1000.
        self.mj.mj_forward(self.model, data)
        self.mj.mj_step(self.model, data)
        self.assertTrue(self.np.isfinite(data.qpos).all())
        self.assertTrue(self.np.all(data.actuator_force <= self.model.actuator_forcerange[:, 1]))
        self.assertTrue(self.np.all(data.actuator_force >= self.model.actuator_forcerange[:, 0]))

    def test_overlap_probe_contact_disappears_with_matched_arm_collision_masks(self):
        result = sim.collision_probe(self.assets)
        enabled, masked = result['attempts']
        self.assertFalse(enabled['arm_collision_masks_disabled'])
        self.assertTrue(masked['arm_collision_masks_disabled'])
        self.assertGreater(enabled['probe_contact_count'], 0)
        self.assertTrue(any(item['other_body'] == 'upper_arm_link' and item['signed_distance_m'] < 0
                            for item in enabled['pairs']))
        self.assertEqual(masked['probe_contact_count'], 0)
        self.assertIn('no motion or real collision trial', result['scope'])
        self.assertEqual(result['warnings'], [])

    def test_short_native_trajectories_share_boundaries_are_finite_and_deterministic(self):
        for case in sim.CASES:
            with self.subTest(case=case):
                coarse = sim.run_case(self.assets, case, timestep=.002, seconds=.1)
                fine = sim.run_case(self.assets, case, timestep=.001, seconds=.1)
                repeat = sim.run_case(self.assets, case, timestep=.002, seconds=.1)
                self.assertTrue(coarse['valid'] and fine['valid'] and repeat['valid'])
                self.assertEqual(coarse['trajectory'], repeat['trajectory'])
                times = [item['t_seconds'] for item in coarse['trajectory']]
                self.assertEqual(times, [item['t_seconds'] for item in fine['trajectory']])
                self.np.testing.assert_allclose(times, self.np.arange(6) * .02, atol=1e-15, rtol=0)
                self.assertEqual(coarse['native_tick_ms']['count'], 50)
                self.assertEqual(fine['native_tick_ms']['count'], 100)
                self.assertEqual(coarse['control_chunk_ms']['count'], 5)
                for run in (coarse, fine):
                    self.assertEqual(run['warnings'], [])
                    self.assertGreater(run['simulation_seconds_per_wall_second'], 0)
                    for row in run['trajectory']:
                        for name in ('joint_position_rad', 'joint_velocity_rad_s', 'linkage_position_rad',
                                     'tcp_position_m', 'tcp_quaternion_wxyz', 'pad_centers_m'):
                            self.assertTrue(self.np.isfinite(self.np.asarray(row[name])).all())
                    self.assertIn('Excludes', run['timing_scope'])
                    self.assertIn('real actuation', run['timing_scope'])

    def test_diagnostic_benchmark_has_no_task_success_or_physical_readiness_claim(self):
        report = sim.benchmark_reference(self.assets, repeats=1, seconds=.1)
        self.assertEqual((report['attempted'], report['valid_attempts']), (9, 9))
        self.assertEqual(len(report['convergence']), 6)
        self.assertTrue(all(item['valid'] and item['matched_boundaries'] == 6
                            for item in report['convergence']))
        self.assertFalse(report['physical_robot_ready'])
        self.assertFalse(report['task_success_evaluated'])
        self.assertFalse(report['hardware']['gpu_used'])
        self.assertIn('not independent physical or skill trials', report['protocol']['repeat_scope'])
        self.assertIn('not calibrated inner jaw gap', report['protocol']['pad_measurement'])
        self.assertIn('not comparison with reality', report['protocol']['convergence_scope'])
        for item in report['convergence']:
            for key, value in item.items():
                if key.endswith(('_rad', '_m')):
                    self.assertTrue(math.isfinite(value) and value >= 0)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False))['schema'],
                         'dvidia.reference-benchmark/v1')

    def test_input_protocol_errors_are_rejected_before_native_compilation(self):
        with patch.object(sim, 'build_model', side_effect=AssertionError('should not compile')):
            for seconds in (True, .11, 0., 11., float('nan'), float('inf')):
                with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                    sim.run_case(self.assets, 'hold_open', seconds=seconds)
            with self.assertRaises(ValueError):
                sim.run_case(self.assets, 'unknown', seconds=.1)
            for timestep in (True, .003, float('nan'), '.001'):
                with self.subTest(timestep=timestep), self.assertRaises(ValueError):
                    sim.run_case(self.assets, 'hold_open', timestep=timestep, seconds=.1)

    def test_left_and_right_hands_are_separate_five_finger_reference_models(self):
        hand_files = [self.assets / 'shadow_hand' / f'{side}_hand.xml' for side in ('left', 'right')]
        if not all(path.is_file() for path in hand_files):
            self.skipTest('Optional Shadow hand assets are not installed')
        audit = sim.audit_reference(self.assets, hands=True)
        self.assertFalse(audit['bimanual_assembly_implemented'])
        self.assertEqual(set(audit['hand_models']), {'left', 'right'})
        for side, model in audit['hand_models'].items():
            self.assertEqual((model['joint_coordinates_nq'], model['velocity_dofs_nv'],
                              model['actuator_count'], model['tendons']), (24, 24, 20, 4))
            names = [item['name'] for item in model['joints']]
            prefix = 'lh_' if side == 'left' else 'rh_'
            for finger in ('FF', 'MF', 'RF', 'LF', 'TH'):
                self.assertTrue(any(name.startswith(prefix + finger) for name in names))
        self.assertFalse(audit['physical_robot_ready'])


if __name__ == '__main__':
    unittest.main()
