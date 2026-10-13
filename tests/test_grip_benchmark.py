"""Release evidence must retain failures and enforce paired comparison semantics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dvidia_training import grip_benchmark as bench


class GripBenchmarkTests(unittest.TestCase):
    def test_pair_gains_losses_are_not_inferred_from_invalid_attempts(self):
        a={'task':'tool-reach','case':{'id':'a'},'policy':'baseline','valid':True,'success':False,
           'independent_score':{'compiled_model_sha256':'abc'}}
        b={**a,'policy':'recovery','success':True}
        self.assertTrue(bench.paired_comparisons([a,b])[0]['gain'])
        self.assertTrue(bench.paired_comparisons([a,b])[0]['same_compiled_model'])
        b['independent_score']={'compiled_model_sha256':'different'}
        self.assertFalse(bench.paired_comparisons([a,b])[0]['gain'])
        self.assertFalse(bench.paired_comparisons([a,b])[0]['same_compiled_model'])
        b['valid']=False
        self.assertFalse(bench.paired_comparisons([a,b])[0]['gain'])

    def test_audit_failure_retains_attempt_and_continues(self):
        from dvidia_training import continuous_grip, grip_protocol
        case={'id':'fixture'}
        plan={'attempts':[{'id':'fixture-'+p,'task':'tool-reach','case':case,'policy':p} for p in bench.POLICIES]}
        result={'valid':True,'success':True,'metrics':{'outcome':'completed'},'events':[],
                'model':None,'frames':[], 'native_trace':[[1.,0]],'native_trace_columns':['t','force']}
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'fresh'
            with patch.object(bench,'protocol',return_value=plan), \
                 patch.object(bench,'implementation_hashes',return_value={}), \
                 patch.object(bench.asset_preview,'_licenses',return_value=({},{})), \
                 patch.object(continuous_grip,'run_case',return_value=result), \
                 patch.object(grip_protocol,'audit_result',side_effect=ValueError('/Users/private/path')):
                report=bench.benchmark('assets',output,split='development')
            self.assertEqual(len(report['attempts']),2)
            self.assertTrue(all(not r['valid'] and not r['success'] for r in report['attempts']))
            self.assertNotIn('/Users/',(output/'benchmark.json').read_text())
            receipt=json.loads((output/'trials/fixture-recovery.json').read_text())
            self.assertEqual(receipt['native_trace'],[[1.,0]])
            self.assertEqual(receipt['metrics']['outcome'],'completed')
            self.assertEqual(report['groups'][0]['outcomes'],{'invalid_attempt':1})

    def test_existing_output_and_invalid_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):bench.benchmark('assets',directory)
            with self.assertRaises(ValueError):bench.benchmark('assets',Path(directory)/'new',source_commit='HEAD')


if __name__=='__main__':unittest.main()
