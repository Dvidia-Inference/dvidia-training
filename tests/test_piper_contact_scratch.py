"""Contact semantics at force thresholds and between mutable native reads."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_task


class ContactReadTests(unittest.TestCase):
    def fixture(self, rows):
        env = piper_task.PiperTaskEnv('unused')
        env.model = object()
        env.pad_geoms = {'left': [3, 4], 'right': [5, 6]}
        env.robot_geoms = set(range(1, 7))
        env.moving_geoms = set(range(2, 7))
        env.arm_geoms = {2}
        env.block_geom, env.table_geom, env.floor_geom = 7, 8, 9
        env.data = SimpleNamespace(
            contact=[SimpleNamespace(geom1=a, geom2=b) for a, b, _ in rows],
            ncon=len(rows),
        )

        def read_force(model, data, index, destination):
            # A fresh force is provided for every query, as for native contact.
            destination[:] = [rows[index][2], 0., 0., 0., 0., 0.]

        native = SimpleNamespace(mj_contactForce=read_force)
        return env, native

    def test_strict_threshold_keeps_order_and_all_contact_classes(self):
        rows = [(7, 3, .009), (7, 4, .01), (7, 3, .011),
                (5, 7, .02), (7, 8, .03), (7, 2, .04),
                (7, 9, .05), (2, 8, .06), (2, 3, .07),
                (1, 9, .08), (3, 5, .09)]
        env, native = self.fixture(rows)
        with patch.object(piper_task.sim, '_native_imports', return_value=(native, np)):
            counts, forces, table, critical, pairs = env._contacts()
        self.assertEqual(counts, {'left': 1, 'right': 1})
        self.assertEqual(forces, {'left': .011, 'right': .02})
        self.assertEqual(table, 1)
        self.assertEqual(critical, 4)
        # Logs include active pairs even when they are neither grasp nor critical.
        self.assertEqual(pairs, [{'geom_ids': sorted([a, b]), 'normal_force_n': f}
                                 for a, b, f in rows[2:]])

    def test_another_read_sees_changed_contacts_without_mutating_old_result(self):
        rows = [(3, 7, .02), (5, 7, .03)]
        env, native = self.fixture(rows)
        with patch.object(piper_task.sim, '_native_imports', return_value=(native, np)):
            first = env._contacts()
            rows[0] = (3, 7, 0.)
            rows[1] = (5, 7, .06)
            second = env._contacts()
            env.data.ncon = 0
            third = env._contacts()
        self.assertEqual(first[:2], ({'left': 1, 'right': 1}, {'left': .02, 'right': .03}))
        self.assertEqual(second[:2], ({'left': 0, 'right': 1}, {'left': 0., 'right': .06}))
        self.assertEqual(third, ({'left': 0, 'right': 0}, {'left': 0., 'right': 0.}, 0, 0, []))

    def test_geometry_membership_change_is_visible_on_next_call(self):
        env, native = self.fixture([(7, 3, .02)])
        with patch.object(piper_task.sim, '_native_imports', return_value=(native, np)):
            self.assertEqual(env._contacts()[3], 0)
            env.pad_geoms['left'] = [4]
            result = env._contacts()
        self.assertEqual(result[0], {'left': 0, 'right': 0})
        self.assertEqual(result[3], 1)


if __name__ == '__main__':
    unittest.main()
