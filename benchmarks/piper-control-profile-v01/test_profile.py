"""Pure correctness checks for the development diagnostic, no native stepping."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('piper_control_profile', Path(__file__).with_name('profile.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProfileProtocolTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_nonfinite_and_overflow(self):
        for raw in (b'{"x":0,"x":1}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.assertRaises(ValueError): module.decode(raw)

    def test_comparison_catches_contacts_commands_and_outcomes_but_not_timing(self):
        original = {key: None for key in module.COMPARISON_FIELDS}
        original.update(trajectory=[{'grasp_contact_counts': {'left': 1, 'right': 1}}],
                        actions=[{'gripper_position_m': .01}], metrics={'contact_pair_native_ticks': {'1:2': 3}})
        changed = deepcopy(original)
        changed['timing'] = {'wall_seconds': 999.}
        self.assertTrue(module.compare_episode(original, changed)['exact_match'])
        for key, value in (('trajectory', []), ('actions', []), ('metrics', {}), ('success', True)):
            altered = deepcopy(changed); altered[key] = value
            self.assertIn(key, module.compare_episode(original, altered)['different_fields'])
        altered = deepcopy(original); del altered['error']
        self.assertFalse(module.compare_episode(original, altered)['exact_match'])

    def test_comparison_distinguishes_boolean_and_numeric_json(self):
        original = {key: None for key in module.COMPARISON_FIELDS}
        original['valid'] = False
        altered = dict(original, valid=0)
        self.assertFalse(module.compare_episode(original, altered)['exact_match'])

    def test_latency_summary_is_fixed_linear_quantile_and_strict_budget(self):
        summary = module.summary([.0, .01, .02, .03])
        self.assertEqual(summary['count'], 4)
        self.assertEqual(summary['over_nominal_20ms'], 1)
        self.assertAlmostEqual(summary['p50_s'], .015)
        self.assertAlmostEqual(summary['p95_s'], .0285)
        self.assertEqual(module.summary([]), {'count': 0})


if __name__ == '__main__': unittest.main()
