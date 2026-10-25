import unittest

import numpy as np

from dvidia_training.material_appearance_benchmark import mechanics, mechanics_comparison


class MechanicsInvariantTests(unittest.TestCase):
    def test_visual_change_matches_but_friction_and_actuator_changes_do_not(self):
        import mujoco as mj

        model = mj.MjModel.from_xml_string('''<mujoco><worldbody><body>
          <joint name="hinge"/><geom type="box" size=".1 .1 .1" mass="1"/>
          </body></worldbody><actuator><position joint="hinge"/></actuator></mujoco>''')
        baseline = mechanics(model)
        model.geom_rgba[:] = [.1, .8, .2, 1.]
        self.assertTrue(mechanics_comparison(baseline, mechanics(model))['equal'])
        model.geom_friction[0, 0] += .1
        changed = mechanics_comparison(baseline, mechanics(model))
        self.assertFalse(changed['equal'])
        self.assertIn('geom_friction', changed['changed_arrays'])
        model.actuator_gainprm[0, 0] += 1.
        self.assertIn('actuator_gainprm', mechanics_comparison(baseline, mechanics(model))['changed_arrays'])

    def test_snapshot_does_not_alias_native_arrays(self):
        import mujoco as mj

        model = mj.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
        before = mechanics(model)
        gravity = before['gravity'].copy()
        model.opt.gravity[:] = 0
        np.testing.assert_array_equal(before['gravity'], gravity)
        self.assertFalse(mechanics_comparison(before, mechanics(model))['equal'])


if __name__ == '__main__':
    unittest.main()
