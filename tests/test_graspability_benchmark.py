"""The diagnostic export retains attempts and never substitutes controller claims."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import graspability_benchmark as bench


def plan():
    return {'attempts':[dict(id='fixture-'+tool,task='in-reach-graspability',policy='shared-v03',
        case=dict(id='fixture',object_shape='easy-block',mass=.03,friction=.6,
                  object_x=.40,object_y=-.10,tool_condition=tool)) for tool in ('mounted','absent')]}


def rows():
    return [dict(a,valid=True,success=False,metrics={},wall_seconds=.1,
                 robot_actuator_sha256='a'*64,independent_score={'compiled_model_sha256':str(i)*64})
            for i,a in enumerate(plan()['attempts'])]


class GraspabilityBenchmarkTests(unittest.TestCase):
    def test_pairing_requires_same_drives_but_allows_declared_morphology_change(self):
        data=rows();data[1]['success']=True
        pair=bench.paired_comparisons(data)[0]
        self.assertTrue(pair['valid_pair']);self.assertTrue(pair['absent_gain'])
        data[1]['robot_actuator_sha256']='b'*64
        self.assertFalse(bench.paired_comparisons(data)[0]['valid_pair'])
        self.assertFalse(bench.paired_comparisons(data)[0]['absent_gain'])
        data[1]['robot_actuator_sha256']='a'*64;data[1]['valid']=False
        self.assertFalse(bench.paired_comparisons(data)[0]['absent_gain'])
        with self.assertRaises(ValueError):bench.paired_comparisons(data[:1])
        data[1]['case']['mass']=.30
        with self.assertRaises(ValueError):bench.paired_comparisons(data)

    def test_independent_failure_overrides_claim_without_erasing_native_evidence(self):
        from dvidia_training import graspability,graspability_protocol
        result=dict(valid=True,success=True,metrics={'outcome':'completed','simulation_seconds':.002},events=[],
                    model=None,frames=[],native_trace=[[.002,1]],native_trace_columns=['t','value'])
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'fresh'
            with patch.object(bench,'protocol',return_value=plan()), \
                 patch.object(bench,'implementation_hashes',return_value={}), \
                 patch.object(bench,'runtime_files',return_value={'index.html':b'fixture'}), \
                 patch.object(bench.asset_preview,'_licenses',return_value=({},{})), \
                 patch.object(graspability,'run_case',return_value=result), \
                 patch.object(bench,'robot_actuator_fingerprint',return_value='a'*64), \
                 patch.object(bench.sim,'model_digest',return_value='b'*64), \
                 patch.object(bench.evidence,'export_trial',return_value={}), \
                 patch.object(graspability_protocol,'audit_result',return_value={'task_success':False,'controller_claim_agrees':False}):
                report=bench.benchmark('assets',output)
            self.assertEqual(len(report['attempts']),2)
            self.assertTrue(all(r['valid'] and not r['success'] for r in report['attempts']))
            self.assertTrue(all(r['scored_outcome']=='retention_rejected' for r in report['attempts']))
            self.assertTrue(all('native_trace' not in r for r in report['attempts']))
            receipt=json.loads((output/'trials/fixture-mounted.json').read_text())
            self.assertEqual(receipt['native_trace'],result['native_trace'])
            self.assertTrue(receipt['controller_reported_success'])
            self.assertTrue((output/'integrity.json').exists())

    def test_corrupt_audit_retains_invalid_attempt_and_continues_without_private_path(self):
        from dvidia_training import graspability,graspability_protocol
        result=dict(valid=True,success=False,metrics={'outcome':'lift_not_retained'},events=[],
                    model=None,frames=[],native_trace=[[.002,1]],native_trace_columns=['t','value'])
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'fresh'
            with patch.object(bench,'protocol',return_value=plan()), \
                 patch.object(bench,'implementation_hashes',return_value={}), \
                 patch.object(bench,'runtime_files',return_value={'index.html':b'fixture'}), \
                 patch.object(bench.asset_preview,'_licenses',return_value=({},{})), \
                 patch.object(graspability,'run_case',return_value=result), \
                 patch.object(bench,'robot_actuator_fingerprint',return_value='a'*64), \
                 patch.object(graspability_protocol,'audit_result',side_effect=ValueError('/Users/private/path')):
                report=bench.benchmark('assets',output)
            self.assertTrue(all(not r['valid'] and not r['success'] for r in report['attempts']))
            self.assertNotIn('/Users/',(output/'benchmark.json').read_text())
            receipt=json.loads((output/'trials/fixture-absent.json').read_text())
            self.assertEqual(receipt['native_trace'],result['native_trace'])
            self.assertEqual(receipt['metrics']['outcome'],'lift_not_retained')

    def test_existing_output_invalid_source_and_unsafe_trial_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):bench.benchmark('assets',directory)
            with self.assertRaises(ValueError):bench.benchmark('assets',Path(directory)/'new',source_commit='HEAD')
            bad=deepcopy(plan());bad['attempts'][0]['id']='../escape'
            with patch.object(bench,'protocol',return_value=bad),self.assertRaises(ValueError):
                bench.benchmark('assets',Path(directory)/'new')


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'),'Set pinned robot assets')
class FingerprintTests(unittest.TestCase):
    def test_drive_change_is_detected_while_shape_load_tool_changes_are_permitted(self):
        from dvidia_training import graspability
        assets=Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS']);base=graspability.cases()[0]
        a=graspability.build_model(assets,base,'mounted')
        b=graspability.build_model(assets,dict(base,object_shape='current-shoe',mass=.30),'absent')
        self.assertEqual(bench.robot_actuator_fingerprint(a),bench.robot_actuator_fingerprint(b))
        self.assertNotEqual(bench.sim.model_digest(a),bench.sim.model_digest(b))
        b.actuator_gainprm[6,0]*=2
        self.assertNotEqual(bench.robot_actuator_fingerprint(a),bench.robot_actuator_fingerprint(b))


if __name__=='__main__':unittest.main()
