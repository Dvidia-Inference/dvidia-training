"""Failure retention and comparison semantics for prerequisite experiments."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from dvidia_training import prerequisite_benchmark as bench

class PrerequisiteBenchmarkTests(unittest.TestCase):
    def test_frozen_protocol_has_every_matched_pair(self):
        p=bench.protocol('heldout')
        self.assertEqual(len(p['attempts']),16)
        for task in bench.TASKS:
            rows=[r for r in p['attempts'] if r['task']==task]
            self.assertEqual(len(rows),8)
            for case in {r['case']['id'] for r in rows}:
                self.assertEqual({r['policy'] for r in rows if r['case']['id']==case},set(bench.POLICIES))
        for split in ('all',None,'training'):
            with self.assertRaises(ValueError):bench.protocol(split)

    def test_safe_abort_is_not_task_success_and_nuisance_is_paired(self):
        case={'id':'fixture'}
        a={'task':'hinged-box','policy':'baseline','case':case,'valid':True,'success':True,
           'metrics':{'lid_robot_collision':False,'intervention_count':0,'completion_time_s':1.6},'wall_seconds':1}
        b={**a,'policy':'recovery','metrics':{'lid_robot_collision':False,'intervention_count':1,'completion_time_s':2.32}}
        pair=bench.paired_comparisons([a,b])[0]
        self.assertTrue(pair['nuisance_interruption'])
        self.assertFalse(pair['lid_contact_avoided'])
        self.assertAlmostEqual(pair['completion_delay_s'],.72)
        b.update(success=False,metrics={'outcome':'safe_abort'})
        group=bench.summarize([b])[0]
        self.assertEqual(group['successes'],0)
        self.assertEqual(group['outcomes'],{'safe_abort':1})

    def test_bad_metric_does_not_drop_later_attempts_or_leak_private_path(self):
        class Fake:
            @staticmethod
            def cases(split):return [{'id':'bad'},{'id':'exception'}]
            @staticmethod
            def run_case(assets,case,policy,record):
                if case['id']=='exception':raise ValueError('/Users/private/secret')
                return {'success':False,'valid':True,'metrics':{'peak':float('nan')},'events':[],
                        'model':None,'frames':[], 'native_trace':[[0,float('nan')]]}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'evidence'
            with patch.object(bench,'modules',return_value={'hinged-box':Fake}), \
                 patch.object(bench,'export_trial',return_value={}), \
                 patch.object(bench,'implementation_hashes',return_value={}), \
                 patch.object(bench.asset_preview,'_licenses',return_value=({},{})):
                result=bench.benchmark('/private/assets',out,split='development')
            self.assertEqual(len(result['attempts']),4)
            self.assertTrue(all(not r['valid'] and not r['success'] for r in result['attempts']))
            self.assertTrue(all(r['errors'] for r in result['attempts']))
            self.assertNotIn('/Users/',(out/'benchmark.json').read_text())
            self.assertEqual(len(list((out/'trials').glob('*.json'))),4)
            seal=json.loads((out/'integrity.json').read_text())
            self.assertIn('benchmark.json',seal['files'])

    def test_export_failure_counts_invalid_and_preserves_reported_outcome(self):
        class Fake:
            @staticmethod
            def cases(split):return [{'id':'completed-before-export'}]
            @staticmethod
            def run_case(assets,case,policy,record):
                return {'success':True,'valid':True,'metrics':{'outcome':'completed','peak_force_n':2.5},
                        'events':[{'t':1.,'type':'completion','message':'Waypoint reached'}],
                        'native_trace':[[1.,2.5]],'native_trace_columns':['t','force_n'],
                        'model':None,'frames':[]}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'evidence'
            with patch.object(bench,'modules',return_value={'hinged-box':Fake}), \
                 patch.object(bench,'export_trial',side_effect=ValueError('Invalid replay')), \
                 patch.object(bench,'implementation_hashes',return_value={}), \
                 patch.object(bench.asset_preview,'_licenses',return_value=({},{})):
                result=bench.benchmark('/private/assets',out,split='development')
            self.assertEqual(len(result['attempts']),2)
            for group in result['groups']:
                self.assertEqual(group['valid'],0)
                self.assertEqual(group['successes'],0)
                self.assertEqual(group['outcomes'],{'invalid_attempt':1})
            for row in result['attempts']:
                self.assertFalse(row['valid']);self.assertFalse(row['success'])
                self.assertEqual(row['metrics'],{'outcome':'completed','peak_force_n':2.5})
                self.assertEqual(row['events'][0]['message'],'Waypoint reached')
                receipt=json.loads((out/'trials'/(row['id']+'.json')).read_text())
                self.assertEqual(receipt['native_trace'],[[1.,2.5]])
                self.assertTrue(receipt['errors'])

    def test_output_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):bench.benchmark('/missing',directory)

if __name__=='__main__':unittest.main()
