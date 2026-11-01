"""Run the Studio result renderer in a minimal DOM; no server or physics."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


HTML = Path(__file__).resolve().parents[1] / 'src/dvidia_training/camera_action_studio.html'
NODE = shutil.which('node')
HARNESS = r"""
const fs=require('fs'),vm=require('vm'),input=JSON.parse(fs.readFileSync(0,'utf8'));
const script=input.html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Also reject syntax errors anywhere in the shipped script.
const elements={};
class Element {
 constructor(){this.children=[];this.textContent='';this.hidden=false;this.dataset={};}
 replaceChildren(...children){this.children=children;}
 append(...children){this.children.push(...children);}
}
const get=id=>elements[id]||(elements[id]=new Element());
const context=vm.createContext({$:get,text:(id,value)=>{get(id).textContent=value;},
 document:{createElement:()=>new Element()}});
const start=script.indexOf('function showBenchmark(report)'),end=script.indexOf('function show(summary)',start);
vm.runInContext(script.slice(start,end),context);
for(const report of input.reports){context.report=report;vm.runInContext('showBenchmark(report)',context);}
process.stdout.write(JSON.stringify(elements));
"""


@unittest.skipUnless(NODE, 'Node is required for the shipped JavaScript renderer test.')
class CameraStudioRenderTests(unittest.TestCase):
    def render(self, *reports):
        result = subprocess.run([NODE, '-e', HARNESS],
            input=json.dumps(dict(html=HTML.read_text(), reports=reports)),
            capture_output=True, text=True, check=True, timeout=10)
        return json.loads(result.stdout)

    @staticmethod
    def cells(element):
        return [[cell['textContent'] for cell in row['children']] for row in element['children']]

    def test_each_controller_keeps_its_own_success_and_contact_denominators(self):
        report=dict(selected_stage='corrected',controllers={key:dict(scheduled=64,
            strict_successes=success,critical_contact_episodes=contacts,contact_counts_complete=True)
            for key,success,contacts in [('student',58,1),('blind',40,2),('fixed',60,0),('teacher',64,0)]},
            gates=dict(at_least_58_of_64=True,no_critical_contact_increase=False))
        result=self.render(report)
        self.assertEqual(self.cells(result['controllers']),[
            ['Learned policy','58 / 64','1 / 64'],
            ['Matched blind learner','40 / 64','2 / 64'],
            ['Fixed primitive','60 / 64','0 / 64'],
            ['Authored teacher','64 / 64','0 / 64']])
        self.assertFalse(result['benchmark-results']['hidden'])
        self.assertTrue(result['benchmark-pending']['hidden'])
        self.assertEqual(result['selected-model']['textContent'],'Selected model: candidate after one correction round.')
        gates=self.cells(result['gates'])
        self.assertEqual(gates[1][1],'Passed');self.assertEqual(gates[4][1],'Failed')
        self.assertEqual(gates[0][1],'Pending')

    def test_missing_or_incomplete_fields_are_not_zero_filled(self):
        result=self.render(dict(selected_stage='initial',controllers=dict(student=dict(
            strict_successes=12,critical_contact_episodes=2,contact_counts_complete=False))))
        rows=self.cells(result['controllers'])
        self.assertEqual(rows[0],['Learned policy','Not reported','2 reported · incomplete'])
        self.assertEqual(rows[1],['Matched blind learner','Not reported','Not reported'])
        self.assertEqual(result['selected-model']['textContent'],'Selected model: initial candidate.')

    def test_pending_or_blocked_report_clears_old_controller_counts_and_gates(self):
        ready=dict(selected_stage='corrected',controllers=dict(student=dict(scheduled=64,strict_successes=64)),
            gates=dict(at_least_58_of_64=True))
        result=self.render(ready,dict(status='blocked',controllers={},gates={}))
        self.assertTrue(result['benchmark-results']['hidden'])
        self.assertFalse(result['benchmark-pending']['hidden'])
        self.assertEqual(result['controllers']['children'],[])
        self.assertEqual(result['selected-model']['textContent'],'Model selection pending.')
        self.assertTrue(all(row[1]=='Pending' for row in self.cells(result['gates'])))

    def test_gate_status_requires_boolean_evidence_and_counts_require_integers(self):
        result=self.render(dict(controllers=dict(student=dict(scheduled=64,strict_successes=True,
            critical_contact_episodes=-1)),gates=dict(at_least_58_of_64='true',
            no_critical_contact_increase=0,all_negative_controls_correct=False)))
        self.assertEqual(self.cells(result['controllers'])[0][1:],['Not reported','Not reported'])
        gates=self.cells(result['gates'])
        self.assertEqual(gates[1][1],'Pending');self.assertEqual(gates[4][1],'Pending')
        self.assertEqual(gates[5][1],'Failed')


if __name__ == '__main__':
    unittest.main()
