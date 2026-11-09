import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy

import numpy as np

MODULE = Path(__file__).with_name('preflight.py')
spec = importlib.util.spec_from_file_location('head_preflight', MODULE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
ROOT = MODULE.resolve().parents[2]


class HeadPreflightTests(unittest.TestCase):
    def test_source_reset_is_bound(self):
        self.assertEqual(p.bind_sources(ROOT)['reset_reference'], p.R0)

    def test_changed_source_refused_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in p.SOURCES:
                target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, target)
            target = root/'src/dvidia_training/piper_task.py'
            target.write_bytes(target.read_bytes().replace(b'1.57, -1.3485', b'1.58, -1.3485'))
            output = root/'evidence'
            with self.assertRaisesRegex(ValueError, 'Source binding changed'):
                p.run(root, output)
            self.assertFalse(output.exists())

    def test_deterministic_random_initialization(self):
        a, b = p.parameters(), p.parameters()
        for left, right in zip(a[0]+a[1], b[0]+b[1]):
            self.assertEqual(left.dtype, np.dtype('float32'))
            self.assertEqual(left.tobytes(), right.tobytes())
        self.assertEqual([x.shape for x in a[0]], [(269,64),(64,64),(64,7)])

    def test_explicit_case_roster(self):
        rows = p.cases()
        self.assertEqual(rows, p.cases())
        self.assertEqual(len(rows), 128)
        self.assertEqual(len(set(r['id'] for r in rows)), 128)
        self.assertEqual(sum(r['kind']=='reset' for r in rows), 64)
        for i in range(0, len(rows), 2):
            self.assertEqual(rows[i]['features'], rows[i+1]['features'])
            self.assertEqual(rows[i]['previous_reference']['value'], p.R0)

    def test_preprojection_equivalence_even_outside_bounds(self):
        head = [100.]*7
        a = p.decode(head, p.reference(p.R0), 'absolute')
        d = p.decode(head, p.reference(p.R0), 'delta')
        self.assertEqual(a.tobytes(), d.tobytes())
        self.assertGreater(a[-1], .035)  # Saturation cannot manufacture a pass.

    def test_later_reference_offset(self):
        ref = p.reference([.1,1.6,-1.3,.2,0.,-.1,.02])
        a = p.decode([.3]*7, ref, 'absolute')
        d = p.decode([.3]*7, ref, 'delta')
        np.testing.assert_allclose(d-a, np.asarray(ref['value'])-p.R0, atol=1e-12, rtol=0.)
        self.assertFalse(np.array_equal(a,d))

    def test_native_endpoint_loss_and_derivative(self):
        decoded = [1.,1.,-1.,0.,0.,0.,.02]
        target = [.9,1.1,-1.1,.1,-.1,0.,.019]
        loss, grad = p.endpoint_loss_gradient(decoded, target)
        error = np.asarray(decoded)-target
        s = np.asarray([3.]*6+[.035]); d=np.asarray([.03]*6+[.001])
        self.assertEqual(loss, float(np.mean((error/s)**2)))
        np.testing.assert_array_equal(grad, 2.*error*d/(7.*s**2))

    def test_reset_gradient_and_central_difference(self):
        result = p.measure(p.cases()[2], p.parameters())
        self.assertTrue(result['passed'])
        self.assertTrue(result['reset_bitwise_decoded_equal'])
        self.assertEqual(result['d_loss_d_head_max_abs_difference'], 0.)
        self.assertLessEqual(result['gradient_finite_difference_max_abs_error'], 1e-8)

    def test_mislabeled_later_reference_cannot_pass_reset(self):
        row = p.cases()[1]; row['kind']='reset'
        self.assertFalse(p.measure(row, p.parameters())['passed'])

    def test_negative_contract_and_authority_checks(self):
        checks = p.negative_checks()
        self.assertEqual(len(checks), 9)
        self.assertTrue(all(c['rejected'] for c in checks))

    def test_loss_unit_change_refused(self):
        c=deepcopy(p.CONTRACT); c['endpoint_loss_scale'][-1]=.07
        with self.assertRaises(ValueError):
            p.endpoint_loss_gradient(p.R0,p.R0,c)

    def test_reference_validation(self):
        for bad in (float('nan'),float('inf'),True,'1',None):
            r=p.reference(p.R0); r['value'][0]=bad
            with self.assertRaises(ValueError):p.decode([0.]*7,r,'delta')
        r=p.reference(p.R0);r['value'][-1]=.036
        with self.assertRaises(ValueError):p.decode([0.]*7,r,'delta')
        r=p.reference(p.R0);r['order']=list(reversed(r['order']))
        with self.assertRaises(ValueError):p.decode([0.]*7,r,'delta')

    def test_unknown_decoder_refused(self):
        with self.assertRaises(ValueError):p.decode([0.]*7,p.reference(p.R0),'measured-relative')

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);(output/'marker').write_text('keep')
            with self.assertRaises(ValueError):p.run(ROOT,output)
            self.assertEqual((output/'marker').read_text(),'keep')

    def test_no_native_package_imported(self):
        self.assertFalse(any(n=='mujoco' or n.startswith('dvidia_training') for n in sys.modules))


if __name__ == '__main__':
    unittest.main()
