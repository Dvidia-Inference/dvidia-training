"""Historical audit suites must coexist without a shared module-name cache."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULES = ('test_camera_action_study_audit', 'test_incremental_independent_audit',
           'test_piper_retention_independent_audit')


class IndependentAuditImportTests(unittest.TestCase):
    def test_all_audit_suites_collect_together_in_either_order(self):
        script = """
import importlib, sys, types, unittest
sentinel = types.ModuleType('independent_audit')
sys.modules['independent_audit'] = sentinel
modules = [importlib.import_module(name) for name in sys.argv[1:]]
assert sys.modules['independent_audit'] is sentinel
assert len({id(module.audit) for module in modules}) == len(modules)
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
result = unittest.TextTestRunner(verbosity=0).run(suite)
forbidden = {'mujoco', 'piper_feedback_env', 'camera_action_capture', 'camera_action_model',
             'piper_rgbd_tracker', 'piper_motor_primitive'}
assert not any(name.rsplit('.', 1)[-1] in forbidden for name in sys.modules)
raise SystemExit(not result.wasSuccessful())
"""
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT/'src'), str(ROOT/'tests'))))
        for names in (MODULES, tuple(reversed(MODULES))):
            with self.subTest(order=names):
                result = subprocess.run([sys.executable, '-c', script, *names], cwd=ROOT,
                    env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
