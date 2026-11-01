import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

PATH=Path(__file__).with_name('independent_audit.py')
spec=importlib.util.spec_from_file_location('retention_independent',PATH)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class IndependentAuditTests(unittest.TestCase):
    def fixture(self):
        n=5200
        arrays={name:np.zeros((n,) if width==1 else(n,width),dtype=np.float64) for name,width in audit.FIELDS.items()}
        arrays['t_seconds']=np.arange(1,n+1)*.001
        arrays['object_position_m'][:]=[.3,0,.08]
        arrays['object_position_m'][20:,2]+=.006
        arrays['object_minus_tcp_local_m'][:]=[0,0,-.1]
        arrays['grasp_contact_counts'][:]=1;arrays['pad_normal_forces_n'][:]=.02
        return arrays,{'start_time_s':.02,'object_origin_m':[.3,0,.08],'local_offset_origin_m':[0,0,-.1]}
    def test_native_100ms_and_hidden_gap(self):
        self.assertTrue(audit.self_tests()['passed'])
    def test_force_ceiling_in_admission_window(self):
        a,p=self.fixture();a['pad_normal_forces_n'][100,0]=10.001
        condition,streak,qualified,*_=audit.native_predicate(a,p)
        self.assertFalse(condition[100]);self.assertEqual(streak[100],0);self.assertFalse(qualified[119])
        a['pad_normal_forces_n'][100,0]=10.
        self.assertTrue(audit.native_predicate(a,p)[2][119])
    def test_exact_five_second_native_interval_and_gap(self):
        a,p=self.fixture();condition,*_=audit.native_predicate(a,p)
        start=audit.row_at(.12,a['t_seconds']);end=audit.row_at(5.12,a['t_seconds'])
        self.assertEqual(end-start,5000);self.assertTrue(condition[start:end+1].all())
        a['grasp_contact_counts'][end-1,1]=0
        self.assertFalse(audit.native_predicate(a,p)[0][start:end+1].all())
    def test_optimized_python_refused_before_import(self):
        result=subprocess.run([sys.executable,'-O',str(PATH),'--help'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('refuses optimized Python',result.stderr)

if __name__=='__main__':unittest.main()
