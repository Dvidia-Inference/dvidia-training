"""Dependency gates and read-only hardware estimates without engine imports."""
from contextlib import redirect_stdout
import builtins
import ctypes
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from dvidia_training import doctor

DEMUXERS = 'Demuxers:\n D  avi AVI\n D  matroska,webm Matroska / WebM\n D  mov,mp4,m4a,3gp,3g2,mj2 MOV\n'
ENCODERS = 'Encoders:\n V....D libx264 H.264\n V..... rawvideo raw video\n'


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.versions = {'numpy': '2.5.3', 'mujoco': '3.15.0'}
        self.addCleanup(patch.stopall)
        patch.object(doctor.metadata, 'version', side_effect=lambda name: self.versions[name]).start()
        patch.object(doctor.shutil, 'which', side_effect=lambda name: '/tools/' + name).start()
        patch.object(doctor, '_query', side_effect=self.query).start()
        patch.object(doctor, '_ram_bytes', return_value=4 * doctor.GIB).start()
        patch.object(doctor, '_disk', return_value={'checked_directory': '/scratch', 'free_bytes': 4 * doctor.GIB}).start()
        patch.object(doctor.platform, 'system', return_value='Linux').start()
        patch.object(doctor.platform, 'machine', return_value='x86_64').start()
        patch.object(doctor.os, 'cpu_count', return_value=2).start()
        patch.object(doctor.sys, 'version_info', (3, 12, 9)).start()

    def query(self, path, *arguments):
        name = Path(path).name
        if arguments == ('-version',):
            return name + ' version test\n'
        if arguments[-1] == '-demuxers':
            return DEMUXERS
        if arguments[-1] == '-encoders':
            return ENCODERS
        raise AssertionError('Unexpected system query.')

    def test_core_check_has_no_numpy_or_mujoco_import_or_gpu_gate(self):
        original = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'mujoco'):
                raise AssertionError('Doctor imported a numerical engine.')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded_import):
            report = doctor.diagnose()
        self.assertTrue(report['ok'])
        self.assertEqual(report['status'], 'dependencies-present')
        self.assertFalse(report['hardware']['gpu_required'])
        self.assertEqual(report['hardware']['max_active_jobs'], 1)
        self.assertFalse(report['sizing_estimates']['verified_minimum'])

    def test_mujoco_is_required_only_for_explicit_arm_mode(self):
        def version(name):
            if name == 'mujoco':
                raise doctor.metadata.PackageNotFoundError(name)
            return self.versions[name]
        with patch.object(doctor.metadata, 'version', side_effect=version):
            self.assertTrue(doctor.diagnose()['ok'])
            report = doctor.diagnose(arm=True)
        self.assertFalse(report['ok'])
        self.assertEqual(report['failed_requirements'], ['mujoco'])

    def test_python_and_numpy_version_mismatches_are_dependency_failures(self):
        self.versions['numpy'] = '2.5.2'
        with patch.object(doctor.sys, 'version_info', (3, 11, 9)):
            report = doctor.diagnose()
        self.assertEqual(report['failed_requirements'], ['python', 'numpy'])

    def test_missing_ffprobe_and_missing_demuxer_block_training_dependencies(self):
        with patch.object(doctor.shutil, 'which', side_effect=lambda name: None if name == 'ffprobe' else '/tools/' + name):
            self.assertEqual(doctor.diagnose()['failed_requirements'], ['ffprobe'])
        def restricted(path, *arguments):
            if Path(path).name == 'ffprobe' and arguments[-1] == '-demuxers':
                return ' D  mov,mp4 MOV\n'
            return self.query(path, *arguments)
        with patch.object(doctor, '_query', side_effect=restricted):
            report = doctor.diagnose()
        self.assertEqual(report['failed_requirements'], ['ffprobe'])
        check = next(row for row in report['checks'] if row['name'] == 'ffprobe')
        self.assertEqual(check['missing_demuxers'], ['avi', 'matroska', 'webm'])

    def test_fixture_encoder_is_optional_and_unknown_or_small_resources_do_not_gate(self):
        def no_encoder(path, *arguments):
            return ' V..... rawvideo raw video\n' if arguments[-1] == '-encoders' else self.query(path, *arguments)
        with patch.object(doctor, '_query', side_effect=no_encoder), \
             patch.object(doctor, '_ram_bytes', return_value=None), \
             patch.object(doctor, '_disk', return_value={'checked_directory': None, 'free_bytes': None}):
            report = doctor.diagnose()
        self.assertTrue(report['ok'])
        self.assertEqual(len(report['warnings']), 3)
        with patch.object(doctor, '_ram_bytes', return_value=doctor.GIB), \
             patch.object(doctor, '_disk', return_value={'checked_directory': '/scratch', 'free_bytes': 1}):
            report = doctor.diagnose()
        self.assertTrue(report['ok'])
        self.assertTrue(any('unverified' in warning for warning in report['warnings']))

    def test_cli_json_exit_code_depends_on_required_checks(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = doctor.main(['--json', '--directory', '/scratch'])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stream.getvalue())['ok'])
        self.versions['numpy'] = 'other'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(doctor.main([]), 1)

    def test_disk_query_does_not_create_requested_directory(self):
        # Restore this helper only; all dependency queries remain mocked.
        patch.stopall()
        with tempfile.TemporaryDirectory() as temporary:
            requested = Path(temporary) / 'not-created' / 'runs'
            result = doctor._disk(requested)
            self.assertEqual(result['checked_directory'], str(Path(temporary)))
            self.assertIsInstance(result['free_bytes'], int)
            self.assertFalse(requested.exists())

    def test_ram_queries_cover_linux_macos_windows_and_unknown(self):
        patch.stopall()
        with patch.object(doctor.os, 'sysconf', side_effect=lambda key: {'SC_PHYS_PAGES': 1048576, 'SC_PAGE_SIZE': 4096}[key]):
            self.assertEqual(doctor._ram_bytes('Linux'), 4 * doctor.GIB)
        with patch.object(doctor, '_query', return_value=str(8 * doctor.GIB)) as query:
            self.assertEqual(doctor._ram_bytes('Darwin'), 8 * doctor.GIB)
            query.assert_called_once_with('/usr/sbin/sysctl', '-n', 'hw.memsize')
        def memory_status(pointer):
            pointer._obj.ullTotalPhys = 16 * doctor.GIB
            return 1
        function = MagicMock(side_effect=memory_status)
        windows = SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=function))
        with patch.object(ctypes, 'windll', windows, create=True):
            self.assertEqual(doctor._ram_bytes('Windows'), 16 * doctor.GIB)
        self.assertIsNone(doctor._ram_bytes('UnqualifiedOS'))

    def test_queries_are_bounded_and_cannot_turn_errors_into_success(self):
        patch.stopall()
        with patch.object(doctor.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ffprobe', 5)):
            self.assertIsNone(doctor._query('ffprobe', '-version'))
        result = subprocess.CompletedProcess('ffprobe', 0, stdout=b'x' * (doctor.QUERY_OUTPUT_LIMIT + 1), stderr=b'')
        with patch.object(doctor.subprocess, 'run', return_value=result):
            self.assertIsNone(doctor._query('ffprobe', '-version'))


if __name__ == '__main__':
    unittest.main()
