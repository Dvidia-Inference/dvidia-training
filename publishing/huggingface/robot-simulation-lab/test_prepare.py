"""Data-only staging, exact model notices and complete task receipt boundaries."""
from copy import deepcopy
from hashlib import sha256
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('robot_sim_prepare',HERE/'prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
publisher = prepare.publisher
from dvidia_training import reference_task_benchmark as bench


def record(raw):
    return {'bytes':len(raw),'sha256':sha256(raw).hexdigest()}


def fixture(root):
    """Small invented data-only fixture, never a physical or native accuracy test."""
    root.mkdir()
    protocol = json.loads(json.dumps(bench.make_protocol(nominal=1,stress=0)))
    scene = protocol['scenes'][0]
    payload = {'receipt/protocol.json':bench._json_bytes(protocol),'receipt/report.html':b'test fixture'}
    attempts,cases = [],[]
    for control in bench.CONTROLS:
        identity = scene['id']+'--'+control
        raw = {'id':identity,'scene_id':scene['id'],'control':control,'valid':True,'success':False,
               'physical_robot_ready':False,'profile':protocol['profile'],'config':scene['config'],'seed':scene['seed'],
               'trajectory':[],'actions':[],'arm_tape':[]}
        filename = 'attempts/'+identity+'.json'
        payload['receipt/'+filename] = bench._json_bytes(raw)
        attempts.append({**bench._summary(raw),'result_file':filename})
        cases.append({'attempt_id':identity,'scene_id':scene['id'],'control':control,'recorded_success':False})
    report = {'schema':bench.SCHEMA,'profile':protocol['profile'],'cohort':protocol['cohort'],
              'physical_robot_ready':False,'learned_policy':False,'implementation_sha256':protocol['implementation_sha256'],
              'protocol':protocol,'protocol_sha256':sha256(payload['receipt/protocol.json']).hexdigest(),
              'scenes':protocol['scenes'],'attempted':3,'valid_attempts':3,'attempts':attempts,
              'summary':bench.aggregate(attempts,protocol['scenes'])}
    payload['receipt/report.json'] = bench._json_bytes(report)
    payload['receipt/integrity.json'] = json.dumps({'schema':'dvidia.reference-task-integrity/v1',
        'files':{name.removeprefix('receipt/'):record(raw) for name,raw in payload.items()}},indent=5).encode()+b'\n'
    payload['scene.json'] = bench._json_bytes({'schema':'dvidia.reference-replay-scene/v1'})
    payload['motion.json'] = bench._json_bytes({'schema':'dvidia.reference-replay-motion/v1','cases':cases,
        'source':{'profile':protocol['profile'],'benchmark_attempted':3,
                  'report_sha256':sha256(payload['receipt/report.json']).hexdigest()}})
    for name,raw in {'index.html':b'Native replay fixture','geometry.bin':b'geometry','geometry.bin.gz':b'compressed geometry',
                     'motion.bin':b'motion','motion.bin.gz':b'compressed motion','reference_replay.bundle.js':b'/* fixture */',
                     'reference_replay.third-party.txt':b'Three.js MIT notice','LICENSE':b'MIT','README.md':b'Fixture',
                     'licenses/universal_robots_ur5e/LICENSE':b'BSD-3-Clause exact notice\n',
                     'licenses/universal_robots_ur5e/README.md':b'UR5e upstream notice\n',
                     'licenses/robotiq_2f85/LICENSE':b'BSD-2-Clause exact notice\n',
                     'licenses/robotiq_2f85/README.md':b'Robotiq upstream notice\n'}.items():
        payload[name] = raw
    reseal(payload)
    for name,raw in payload.items():
        path = root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
    return payload


def reseal(payload):
    payload['replay-integrity.json'] = json.dumps({'schema':'dvidia.reference-replay-integrity/v1',
        'files':{name:record(raw) for name,raw in payload.items() if name!='replay-integrity.json'}},indent=3).encode()+b'\n'


def profile_fixture(root,identity):
    root.mkdir()
    report = {'schema':'dvidia.robot-profile-benchmark/v1','task_qualified':False,'physical_robot_ready':False,
        'profile':{'id':identity,'name':'PiPER' if identity=='piper' else 'SO-101',
                   'arm_axes':6 if identity=='piper' else 5,'gripper_axes':1,'simulation_asset_license':'BSD-2-Clause'},
        'summary':{'attempt_count':3,'valid_attempts':3}}
    payload = {'report.json':bench._json_bytes(report),'report.html':b'Unloaded mechanics fixture',
               'licenses/'+identity+'/LICENSE':b'Original profile license fixture\n'}
    records = [{'path':name,**record(raw)} for name,raw in sorted(payload.items())]
    payload['integrity.json'] = json.dumps({'schema':'dvidia.robot-profile-integrity/v1','files':records},indent=4).encode()+b'\n'
    for name,raw in payload.items():
        file = root/name
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_bytes(raw)
    return payload,{'report':report,'files':records,'valid_attempts':3,'attempts':3}


def piper_task_fixture(root):
    """Compact invented failed/invalid source tapes; loader is patched in staging tests."""
    from dvidia_training import piper_task_benchmark as benchmark
    root.mkdir()
    scenes = [{'id':stratum+'-00','stratum':stratum,'seed':index,'config':{}}
              for index,stratum in enumerate(('nominal','low_friction'))]
    protocol = {'schema':benchmark.PROTOCOL_SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.SCOPE,
        'physical_robot_ready':False,'learned_policy':False,'cohort':'development','scenes':scenes,
        'controls':list(benchmark.CONTROLS),'planned_attempts':6,
        'native_command_units':['rad']*6+['m'],'simulation_asset_license':'MIT'}
    payload = {'protocol.json':bench._json_bytes(protocol),'report.html':b'PiPER contact fixture, every attempt',
        'licenses/agilex_piper/LICENSE':b'Original MIT PiPER license fixture\n',
        'licenses/agilex_piper/README.md':b'Original model notice fixture\n'}
    attempts = []
    for scene in scenes:
        for control in benchmark.CONTROLS:
            identity = scene['id']+'--'+control
            raw = {'schema':benchmark.task.SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.task.SCOPE,
                'id':identity,'scene_id':scene['id'],'control':control,'physical_robot_ready':False,
                'valid':not (scene['stratum']=='low_friction' and control=='hold'),'success':False,
                'reason':'fixture failure','trajectory':[{'fixture_native_state':0}],
                'actions':[{'fixture_command':0}],'arm_tape':[[0]*6]}
            name = 'attempts/'+identity+'.json'
            payload[name] = bench._json_bytes(raw)
            attempts.append({**benchmark._summary(raw),'result_file':name})
    report = {'schema':benchmark.SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.SCOPE,
        'physical_robot_ready':False,'learned_policy':False,'simulation_asset_license':'MIT',
        'cohort':'development','protocol':protocol,'protocol_sha256':sha256(payload['protocol.json']).hexdigest(),
        'scenes':scenes,'attempted':6,'valid_attempts':5,'attempts':attempts,
        'summary':benchmark.aggregate(attempts,scenes)}
    payload['report.json'] = bench._json_bytes(report)
    piper_task_reseal(root,report,payload)
    return report,payload


def piper_task_reseal(root,report,payload):
    payload['report.json'] = bench._json_bytes(report)
    payload['integrity.json'] = bench._json_bytes({'schema':'dvidia.piper-task-integrity/v1',
        'files':{name:record(raw) for name,raw in payload.items() if name!='integrity.json'}})
    for name,raw in payload.items():
        path = root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)


def piper_grip_fixture(root):
    """Small invented gzip tapes; native semantic loader is patched in these tests."""
    from dvidia_training import piper_grip_benchmark as benchmark
    root.mkdir()
    scenes = [{'id':f'friction{round(friction*100):03}-layout{index:02}',
        'friction':friction,'stratum':stratum,'layout_index':index,'config':{'friction':friction}}
        for friction,stratum in zip(benchmark.FRICTIONS,benchmark.STRATA) for index in (0,1)]
    protocol = {'schema':benchmark.PROTOCOL_SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.SCOPE,
        'physical_robot_ready':False,'learned_policy':False,'simulation_asset_license':'MIT',
        'cohort':'development','scenes':scenes,'variants':list(benchmark.VARIANTS),'planned_attempts':30,
        'native_command_units':['rad']*6+['m']}
    payload = {'protocol.json':bench._json_bytes(protocol),'report.html':b'Every compressed attempt is linked',
        'licenses/agilex_piper/LICENSE':b'Original MIT model fixture\n',
        'licenses/agilex_piper/README.md':b'Original model notice fixture\n'}
    attempts = []
    for scene in scenes:
        for variant in benchmark.VARIANTS:
            identity=scene['id']+'--'+variant
            valid=not (scene['friction']==.15 and scene['layout_index']==1 and variant=='hold')
            raw={'schema':benchmark.task.SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.task.SCOPE,
                'id':identity,'scene_id':scene['id'],'variant':variant,'physical_robot_ready':False,'learned_policy':False,
                'valid':valid,'success':False,'config':scene['config'],'control':'hold' if variant=='hold' else
                    'replay_open_jaw' if variant.endswith('open_jaw') else 'placement',
                'policy_mode':'guarded' if variant.startswith('guarded') else 'baseline',
                'trajectory':[{'fixture':0}],'actions':[{'fixture':0}],'arm_tape':[[0]*6],
                'grip_evaluation':{'qualification_decisions':[],'alarms':[],'following_opportunity':False if valid else None,
                    'sampled_drop_events':0,'aborted':False,'retry_count':0}}
            name='attempts/'+identity+'.json.gz'
            payload[name]=gzip.compress(bench._json_bytes(raw),mtime=0)
            attempts.append({**benchmark._summary(raw),'result_file':name})
    report={'schema':benchmark.SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.SCOPE,
        'physical_robot_ready':False,'learned_policy':False,'simulation_asset_license':'MIT','cohort':'development',
        'protocol':protocol,'protocol_sha256':sha256(payload['protocol.json']).hexdigest(),
        'scenes':scenes,'attempted':30,'valid_attempts':29,'attempts':attempts,
        'summary':benchmark.aggregate(attempts,scenes)}
    piper_grip_reseal(root,report,payload)
    return report,payload


def piper_grip_reseal(root,report,payload):
    from dvidia_training import piper_grip_benchmark as benchmark
    payload['report.json']=bench._json_bytes(report)
    records={}
    for name,data in payload.items():
        if name=='integrity.json': continue
        records[name]=record(data)
        if name.endswith('.json.gz'):
            decoded=benchmark.decode_attempt(data)
            records[name].update(encoding='gzip',decoded_bytes=len(decoded),decoded_sha256=sha256(decoded).hexdigest())
    payload['integrity.json']=bench._json_bytes({'schema':benchmark.INTEGRITY_SCHEMA,'files':records})
    for name,data in payload.items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)


def piper_contact_fixture(root):
    """Invented data-only full inventory; semantic/native loader is patched."""
    from dvidia_training import piper_contact_benchmark as benchmark
    root.mkdir()
    scenes = [{'id':f'friction{round(f*100):03}-mass{round(m*1000):03}-layout{i:02}',
        'friction':f,'mass_kg':m,'stratum':stratum,'layout_index':i,'config':{'friction':f,'mass_kg':m}}
        for f,stratum in zip(benchmark.FRICTIONS,benchmark.STRATA) for m in benchmark.MASSES for i in (0,1)]
    protocol = {'schema':benchmark.PROTOCOL_SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.SCOPE,
        'physical_robot_ready':False,'learned_policy':False,'simulation_asset_license':'MIT',
        'cohort':'development','scenes':scenes,'variants':list(benchmark.VARIANTS),'planned_attempts':84,
        'native_command_units':['rad']*6+['m']}
    payload = {'protocol.json':bench._json_bytes(protocol),'report.html':b'All eighty-four canonical attempts are linked',
        'licenses/agilex_piper/LICENSE':b'Original MIT model fixture\n',
        'licenses/agilex_piper/README.md':b'Original model notice fixture\n'}
    attempts = []
    for scene in scenes:
        for variant in benchmark.VARIANTS:
            identity=scene['id']+'--'+variant
            control,mode=benchmark._variant(variant)
            valid=not (scene['friction']==.15 and scene['mass_kg']==.06 and scene['layout_index']==1 and variant=='hold')
            following={'qualification_decisions':[],'alarms':[],'following_opportunity':False if valid else None,
                       'sampled_drop_events':0,'aborted':False,'retry_count':0}
            raw={'schema':benchmark.task.SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.task.SCOPE,
                'id':identity,'scene_id':scene['id'],'variant':variant,'physical_robot_ready':False,'learned_policy':False,
                'valid':valid,'success':False,'config':scene['config'],'control':control,'policy_mode':mode,
                'trajectory':[{'fixture':0}],'actions':[{'fixture':0}],'arm_tape':[[0]*6],
                'grip_guard':{'schema':'dvidia.piper-grip-guard/v1' if mode=='baseline' else 'dvidia.piper-contact-memory/v0.3',
                              'mode':mode,'active':control=='placement' and mode!='baseline'},
                'contact_evaluation':{'following':following,'interventions':[],
                    'resume_decisions':0,'loaded_settling_observed_at_resume':0,
                    'low_tcp_motion_observed_at_resume':0,'unresolved_resume_windows':0,
                    'resumed_completed':False,'resumed_not_completed':False}}
            name='attempts/'+identity+'.json.gz'
            payload[name]=gzip.compress(bench._json_bytes(raw),mtime=0)
            attempts.append({**benchmark._summary(raw),'result_file':name})
    report={'schema':benchmark.SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.SCOPE,
        'physical_robot_ready':False,'learned_policy':False,'simulation_asset_license':'MIT','cohort':'development',
        'protocol':protocol,'protocol_sha256':sha256(payload['protocol.json']).hexdigest(),
        'scenes':scenes,'attempted':84,'valid_attempts':83,'attempts':attempts,
        'summary':benchmark.aggregate(attempts,scenes)}
    piper_contact_reseal(root,report,payload)
    return report,payload


def piper_contact_reseal(root,report,payload):
    from dvidia_training import piper_contact_benchmark as benchmark
    payload['report.json']=bench._json_bytes(report)
    records={}
    for name,data in payload.items():
        if name=='integrity.json': continue
        records[name]=record(data)
        if name.endswith('.json.gz'):
            decoded=benchmark.decode_attempt(data)
            records[name].update(encoding='gzip',decoded_bytes=len(decoded),decoded_sha256=sha256(decoded).hexdigest())
    payload['integrity.json']=bench._json_bytes({'schema':benchmark.INTEGRITY_SCHEMA,'files':records})
    for name,data in payload.items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)

def piper_recovery_fixture(root):
    """Invented data-only full inventory; semantic/native loader is patched."""
    from dvidia_training import piper_recovery_benchmark as benchmark
    root.mkdir()
    scenes = [{'id':f'friction{round(f*100):03}-mass{round(m*1000):03}-layout{i:02}',
        'friction':f,'mass_kg':m,'stratum':stratum,'layout_index':i,'config':{'friction':f,'mass_kg':m}}
        for f,stratum in zip(benchmark.FRICTIONS,benchmark.STRATA) for m in benchmark.MASSES for i in (0,1)]
    protocol = {'schema':benchmark.PROTOCOL_SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.SCOPE,
        'physical_robot_ready':False,'learned_policy':False,'simulation_asset_license':'MIT',
        'cohort':'development','scenes':scenes,'variants':list(benchmark.VARIANTS),'planned_attempts':84,
        'native_command_units':['rad']*6+['m']}
    payload = {'protocol.json':bench._json_bytes(protocol),'report.html':b'All eighty-four canonical attempts are linked',
        'licenses/agilex_piper/LICENSE':b'Original MIT model fixture\n',
        'licenses/agilex_piper/README.md':b'Original model notice fixture\n'}
    attempts = []
    for scene in scenes:
        for variant in benchmark.VARIANTS:
            identity=scene['id']+'--'+variant
            control,mode=benchmark._variant(variant)
            valid=not (scene['friction']==.15 and scene['mass_kg']==.06 and scene['layout_index']==1 and variant=='hold')
            following={'qualification_decisions':[],'alarms':[],'following_opportunity':False if valid else None,
                       'sampled_drop_events':0,'aborted':False,'retry_count':0}
            raw={'schema':benchmark.task.SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.task.SCOPE,
                'id':identity,'scene_id':scene['id'],'variant':variant,'physical_robot_ready':False,'learned_policy':False,
                'valid':valid,'success':False,'config':scene['config'],'control':control,'policy_mode':mode,
                'trajectory':[{'fixture':0}],'actions':[{'fixture':0}],'arm_tape':[[0]*6],
                'grip_guard':{'schema':'dvidia.piper-grip-guard/v1' if mode=='baseline' else 'dvidia.piper-contact-memory/v0.3',
                              'mode':mode,'active':control=='placement' and mode!='baseline','retry_count':0,'aborted':False},
                'contact_evaluation':{'following':following,'interventions':[],
                    'resume_decisions':0,'loaded_settling_observed_at_resume':0,
                    'low_tcp_motion_observed_at_resume':0,'unresolved_resume_windows':0,
                    'resumed_completed':False,'resumed_not_completed':False}}
            raw['recovery_evaluation']={'contact':raw['contact_evaluation'],'holds':[],
                'restoration_started':0,'full_speed_restored':0,'completion_time_s':None,'abort_time_s':None}
            name='attempts/'+identity+'.json.gz'
            payload[name]=gzip.compress(bench._json_bytes(raw),mtime=0)
            attempts.append({**benchmark._summary(raw),'result_file':name})
    report={'schema':benchmark.SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.SCOPE,
        'physical_robot_ready':False,'learned_policy':False,'simulation_asset_license':'MIT','cohort':'development',
        'protocol':protocol,'protocol_sha256':sha256(payload['protocol.json']).hexdigest(),
        'scenes':scenes,'attempted':84,'valid_attempts':83,'attempts':attempts,
        'summary':benchmark.aggregate(attempts,scenes)}
    piper_recovery_reseal(root,report,payload)
    return report,payload


def piper_recovery_reseal(root,report,payload):
    from dvidia_training import piper_recovery_benchmark as benchmark
    payload['report.json']=bench._json_bytes(report)
    records={}
    for name,data in payload.items():
        if name=='integrity.json': continue
        records[name]=record(data)
        if name.endswith('.json.gz'):
            decoded=benchmark.decode_attempt(data)
            records[name].update(encoding='gzip',decoded_bytes=len(decoded),decoded_sha256=sha256(decoded).hexdigest())
    payload['integrity.json']=bench._json_bytes({'schema':benchmark.INTEGRITY_SCHEMA,'files':records})
    for name,data in payload.items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)


def piper_reliability_fixture(root,seed=950031):
    """Invented data-only full inventory; semantic/native loader is patched."""
    from dvidia_training import piper_reliability_benchmark as benchmark
    root.mkdir()
    scenes = [{'id':f'friction{round(f*100):03}-mass{round(m*1000):03}-layout{i:02}',
        'friction':f,'mass_kg':m,'stratum':stratum,'layout_index':i,'config':{'friction':f,'mass_kg':m}}
        for f,stratum in zip(benchmark.FRICTIONS,benchmark.STRATA) for m in benchmark.MASSES for i in (0,1)]
    protocol = {'schema':benchmark.PROTOCOL_SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.SCOPE,
        'physical_robot_ready':False,'learned_policy':False,'simulation_asset_license':'MIT',
        'cohort':'development','seed':seed,'scenes':scenes,'variants':list(benchmark.VARIANTS),'planned_attempts':12*len(benchmark.VARIANTS),
        'native_command_units':['rad']*6+['m']}
    payload = {'protocol.json':bench._json_bytes(protocol),'report.html':b'All eighty-four canonical attempts are linked',
        'licenses/agilex_piper/LICENSE':b'Original MIT model fixture\n',
        'licenses/agilex_piper/README.md':b'Original model notice fixture\n'}
    attempts = []
    for scene in scenes:
        for variant in benchmark.VARIANTS:
            identity=scene['id']+'--'+variant
            control,mode=benchmark._variant(variant)
            valid=not (scene['friction']==.15 and scene['mass_kg']==.06 and scene['layout_index']==1 and variant=='hold')
            following={'qualification_decisions':[],'alarms':[],'following_opportunity':False if valid else None,
                       'sampled_drop_events':0,'aborted':False,'retry_count':0}
            raw={'schema':benchmark.task.SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.task.SCOPE,
                'id':identity,'scene_id':scene['id'],'variant':variant,'physical_robot_ready':False,'learned_policy':False,
                'valid':valid,'success':False,'config':scene['config'],'control':control,'policy_mode':mode,
                'trajectory':[{'fixture':0}],'actions':[{'fixture':0}],'arm_tape':[[0]*6],
                'grip_guard':{'schema':'dvidia.piper-grip-guard/v1' if mode=='baseline' else 'dvidia.piper-contact-memory/v0.3',
                              'mode':mode,'active':control=='placement' and mode!='baseline','retry_count':0,'aborted':False},
                'contact_evaluation':{'following':following,'interventions':[],
                    'resume_decisions':0,'loaded_settling_observed_at_resume':0,
                    'low_tcp_motion_observed_at_resume':0,'unresolved_resume_windows':0,
                    'resumed_completed':False,'resumed_not_completed':False}}
            raw['recovery_evaluation']={'contact':raw['contact_evaluation'],'holds':[],
                'restoration_started':0,'full_speed_restored':0,'second_probe_started':0,'second_probe_qualified':0,'completion_time_s':None,'abort_time_s':None}
            name='attempts/'+identity+'.json.gz'
            payload[name]=gzip.compress(bench._json_bytes(raw),mtime=0)
            attempts.append({**benchmark._summary(raw),'result_file':name})
    report={'schema':benchmark.SCHEMA,'profile':benchmark.task.PROFILE,'scope':benchmark.SCOPE,
        'physical_robot_ready':False,'learned_policy':False,'simulation_asset_license':'MIT','cohort':'development',
        'protocol':protocol,'protocol_sha256':sha256(payload['protocol.json']).hexdigest(),
        'scenes':scenes,'attempted':12*len(benchmark.VARIANTS),'valid_attempts':12*len(benchmark.VARIANTS)-1,'attempts':attempts,
        'summary':benchmark.aggregate(attempts,scenes),'paired_comparisons':benchmark.paired_comparisons(attempts,scenes),
        'quality_gates':{'complete_planned_denominator':True,'all_native_attempts_valid':False,'all_controls_fail':True}}
    piper_reliability_reseal(root,report,payload)
    return report,payload


def piper_reliability_reseal(root,report,payload):
    from dvidia_training import piper_reliability_benchmark as benchmark
    payload['report.json']=bench._json_bytes(report)
    records={}
    for name,data in payload.items():
        if name=='integrity.json': continue
        records[name]=record(data)
        if name.endswith('.json.gz'):
            decoded=benchmark.decode_attempt(data)
            records[name].update(encoding='gzip',decoded_bytes=len(decoded),decoded_sha256=sha256(decoded).hexdigest())
    payload['integrity.json']=bench._json_bytes({'schema':benchmark.INTEGRITY_SCHEMA,'files':records})
    for name,data in payload.items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)


class RobotSimulationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.replay = self.root/'replay'
        self.original = fixture(self.replay)
        self.output = self.root/'publication'
        self.guard = patch.object(publisher,'client',side_effect=AssertionError('Staging must not read credentials'))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_v05_three_cohorts_keep_every_original_byte_and_separate_denominators(self):
        from dvidia_training import piper_reliability_benchmark as reliability,piper_recovery_benchmark as recovery
        cohorts=[]
        for seed in (950031,950131,950231):
            source=self.root/str(seed)
            report,payload=piper_reliability_fixture(source,seed)
            cohorts.append((source,report,payload))
        prior_source=self.root/'prior'
        prior_report,prior_files=piper_recovery_fixture(prior_source)
        lookup={source:(report,payload) for source,report,payload in cohorts}
        with patch.object(reliability,'load_reliability_bundle',side_effect=lambda path:lookup[path]), \
                patch.object(recovery,'load_recovery_bundle',return_value=(prior_report,prior_files)):
            plan=prepare.stage(self.replay,self.output,source_commit='a'*40,piper_recovery_benchmark=prior_source,
                               piper_reliability_benchmarks=[source for source,_,_ in cohorts])
        self.assertEqual(publisher.checked_plan(self.output/'publication-plan.json'),plan)
        for source,report,payload in cohorts:
            prefix='tasks/piper-reliability-v05/development-'+str(report['protocol']['seed'])
            for name,raw in payload.items():
                self.assertEqual((self.output/'space'/prefix/name).read_bytes(),raw)
        for prefix,payload in (('tasks/piper-recovery-v04',prior_files),('replay',self.original)):
            for name,raw in payload.items():
                self.assertEqual((self.output/'space'/prefix/name).read_bytes(),raw)
        landing=(self.output/'space/index.html').read_text()
        self.assertEqual(landing.count('PiPER reliability v0.5 · synthetic'),3)
        self.assertIn('Every failed or invalid attempt remains in the denominator.',landing)
        self.assertIn('cohort label alone does not establish independence',landing)
        self.assertIn('0 gained, 0 lost, 0 shared successful scenes, 12 neither completed',landing)
        self.assertIn('No shared-success completion-time comparison is available.',landing)
        self.assertIn('all native attempts valid: not met',landing)
        provenance=json.loads((self.output/'space/provenance.json').read_bytes())
        entries=[v for k,v in provenance['contact_task_benchmarks'].items() if k.startswith('piper-reliability-v05/')]
        self.assertEqual(len(entries),3)
        self.assertTrue(all(r['attempted']==12*len(reliability.VARIANTS) for r in entries))
        self.assertEqual(provenance['publication_byte_bounds']['stored_payload_bytes'],512*1024*1024)
        self.assertEqual(provenance['publication_byte_bounds']['per_file_bytes'],32*1024*1024)
        self.assertIn('PiPER reliability v0.5',(self.output/'space/README.md').read_text())

    def test_v05_rejects_unbounded_duplicate_or_malformed_cohorts_before_writes(self):
        from dvidia_training import piper_reliability_benchmark as reliability
        source=self.root/'reliability';report,payload=piper_reliability_fixture(source)
        for supplied in ('not-a-list',[source]*4,[source,source]):
            with self.subTest(supplied=supplied),patch.object(reliability,'load_reliability_bundle',return_value=(report,payload)),self.assertRaises(ValueError):
                prepare.stage(self.replay,self.output,source_commit='a'*40,piper_reliability_benchmarks=supplied)
            self.assertFalse(self.output.exists())
        for invalid in (True,-1,999989,'../escape'):
            changed=deepcopy(report);changed['protocol']['seed']=invalid
            with self.subTest(invalid=invalid),patch.object(reliability,'load_reliability_bundle',return_value=(changed,payload)),self.assertRaises(ValueError):
                prepare.stage(self.replay,self.output,source_commit='a'*40,piper_reliability_benchmarks=[source])
            self.assertFalse(self.output.exists())

    def test_v05_rechecks_complete_seals_scope_raw_bytes_and_limits(self):
        from dvidia_training import piper_reliability_benchmark as reliability
        for variant in ('scope','count','summary','drop','decoded_hash','changed','private','unlisted','stored','decoded','file'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as folder:
                root=Path(folder).resolve();source=root/'reliability';report,payload=piper_reliability_fixture(source)
                if variant=='scope': report['learned_policy']=True
                elif variant=='count': report['attempted']-=1
                elif variant=='summary': report['summary'][0]['success']=1
                elif variant=='drop':
                    name=report['attempts'].pop()['result_file'];payload.pop(name);(source/name).unlink()
                elif variant=='private':
                    name=report['attempts'][0]['result_file'];raw=json.loads(reliability.decode_attempt(payload[name]))
                    raw['note']='/Users/roguedev/private';payload[name]=gzip.compress(bench._json_bytes(raw),mtime=0)
                    report['attempts'][0]={**reliability._summary(raw),'result_file':name}
                piper_reliability_reseal(source,report,payload)
                if variant=='decoded_hash':
                    seal=json.loads(payload['integrity.json']);name=report['attempts'][0]['result_file']
                    seal['files'][name]['decoded_sha256']='0'*64
                    payload['integrity.json']=bench._json_bytes(seal);(source/'integrity.json').write_bytes(payload['integrity.json'])
                elif variant=='changed': (source/'report.html').write_bytes(b'changed after portable inspection')
                elif variant=='unlisted': (source/'extra.json').write_bytes(b'{}')
                with patch.object(reliability,'load_reliability_bundle',return_value=(report,payload)), \
                        patch.object(publisher,'MAX_TOTAL_BYTES',1 if variant=='stored' else publisher.MAX_TOTAL_BYTES), \
                        patch.object(publisher,'MAX_FILE_BYTES',1 if variant=='file' else publisher.MAX_FILE_BYTES), \
                        patch.object(reliability,'MAX_DECODED_BYTES',1 if variant=='decoded' else reliability.MAX_DECODED_BYTES),self.assertRaises(ValueError):
                    prepare.stage(self.replay,root/'output',source_commit='a'*40,piper_reliability_benchmarks=[source])
                self.assertFalse((root/'output').exists())

    def test_v05_summary_exposes_gains_losses_shared_timing_without_pooled_success_claim(self):
        report,_=piper_reliability_fixture(self.root/'summary')
        paired=report['paired_comparisons'][0]
        paired.update(gained_successes=2,lost_successes=1,common_successes=3,neither_succeeded=6,common_success_mean_delta_s=.7)
        summary=prepare._reliability_summary(report)
        self.assertIn('2 gained, 1 lost, 3 shared successful scenes, 6 neither completed',summary)
        self.assertIn('+0.700 simulated seconds (candidate minus reference)',summary)
        self.assertNotIn('improved',summary)

    def test_staging_retains_every_exact_receipt_binary_and_component_notice(self):
        plan = prepare.stage(self.replay,self.output,source_commit='a'*40)
        self.assertEqual(plan['repository']['repo_id'],publisher.REPO_ID)
        self.assertFalse(plan['private'])
        self.assertEqual(publisher.checked_plan(self.output/'publication-plan.json'),plan)
        for name,raw in self.original.items():
            self.assertEqual((self.output/'space/replay'/name).read_bytes(),raw)
        card = (self.output/'space/README.md').read_text()
        self.assertIn('sdk: static',card)
        self.assertIn('BSD-3-Clause',card)
        self.assertIn('BSD-2-Clause',card)
        self.assertIn('no minimum RAM',card)
        self.assertNotIn('{{',card)
        landing = (self.output/'space/index.html').read_text()
        self.assertIn('3 planned attempts',landing)
        self.assertIn('Upcoming · evidence pending',landing)
        self.assertIn('replay/index.html',landing)

    def test_unsealed_edits_unlisted_files_and_symlinks_fail_before_output_is_reserved(self):
        for variant in ('changed','extra','symlink'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                source = root/'replay'
                fixture(source)
                if variant=='changed':
                    (source/'receipt/report.html').write_bytes(b'changed')
                elif variant=='extra':
                    (source/'unlisted.json').write_bytes(b'{}')
                else:
                    (source/'linked.json').symlink_to(source/'scene.json')
                output = root/'publication'
                with self.assertRaises(ValueError): prepare.stage(source,output,source_commit='a'*40)
                self.assertFalse(output.exists())

    def test_resealed_false_source_labels_or_missing_component_licenses_are_rejected(self):
        for variant in ('wrong_label','missing_license'):
            payload = deepcopy(self.original)
            if variant=='wrong_label':
                motion = json.loads(payload['motion.json'])
                motion['cases'][0]['recorded_success'] = True
                payload['motion.json'] = bench._json_bytes(motion)
            else:
                payload.pop('licenses/robotiq_2f85/LICENSE')
                (self.replay/'licenses/robotiq_2f85/LICENSE').unlink()
            reseal(payload)
            for name,raw in payload.items():
                (self.replay/name).write_bytes(raw)
            with self.subTest(variant=variant),self.assertRaises(ValueError):
                prepare.stage(self.replay,self.output,source_commit='a'*40)
            self.assertFalse(self.output.exists())

    def test_staging_private_path_or_binary_policy_is_rejected(self):
        for name,raw in [('private.txt',b'/Users/roguedev/private'),('policy.bin',b'weights')]:
            payload = deepcopy(self.original)
            payload[name] = raw
            reseal(payload)
            path = self.replay/name
            path.write_bytes(raw)
            (self.replay/'replay-integrity.json').write_bytes(payload['replay-integrity.json'])
            with self.subTest(name=name),self.assertRaises(ValueError):
                prepare.stage(self.replay,self.output,source_commit='a'*40)
            self.assertFalse(self.output.exists())
            path.unlink()

    def test_fresh_directory_and_full_revision_are_required(self):
        with self.assertRaises(ValueError): prepare.stage(self.replay,self.output,source_commit='short')
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        with self.assertRaises(FileExistsError): prepare.stage(self.replay,self.output,source_commit='a'*40)

    def test_reviewed_starter_removal_records_only_exact_css_hash_and_requires_parent(self):
        style = self.root/'style.css'
        raw = b'body { color: black; }\n'
        style.write_bytes(raw)
        with self.assertRaises(publisher.PublicationError):
            prepare.stage(self.replay,self.output,source_commit='a'*40,remove_starter_stylesheet=style)
        self.assertFalse(self.output.exists())
        plan = prepare.stage(self.replay,self.output,source_commit='a'*40,expected_remote_commit='b'*40,
                             remove_starter_stylesheet=style)
        self.assertEqual(plan['remove_files'],[{'path':'style.css',**record(raw)}])
        self.assertFalse((self.output/'space/style.css').exists())
        self.assertEqual(publisher.checked_plan(self.output/'publication-plan.json'),plan)

    def test_two_profile_bundles_retain_exact_traces_and_licenses_with_correct_axes_and_scope(self):
        from dvidia_training import robot_profiles
        roots = [self.root/'piper',self.root/'so101']
        fixtures = [profile_fixture(root,name) for root,name in zip(roots,('piper','so101'))]
        inspected = {root:fixture[1] for root,fixture in zip(roots,fixtures)}
        with patch.object(robot_profiles,'inspect_benchmark',side_effect=lambda root:inspected[root]):
            prepare.stage(self.replay,self.output,source_commit='a'*40,profile_benchmarks=roots)
        for identity,(payload,_) in zip(('piper','so101'),fixtures):
            for name,raw in payload.items():
                self.assertEqual((self.output/'space/profiles'/identity/name).read_bytes(),raw)
        landing = (self.output/'space/index.html').read_text()
        self.assertIn('6 arm axes + 1 gripper axis',landing)
        self.assertIn('5 arm axes + 1 gripper axis',landing)
        self.assertIn('Recorded native mechanics · unloaded',landing)
        self.assertIn('profiles/piper/report.json',landing)
        self.assertIn('profiles/so101/integrity.json',landing)
        provenance = json.loads((self.output/'space/provenance.json').read_bytes())
        self.assertEqual(set(provenance['profile_benchmarks']),{'piper','so101'})
        self.assertFalse(provenance['profile_benchmarks']['piper']['task_qualified'])
        self.assertNotIn('{{',(self.output/'space/README.md').read_text())

    def test_profile_snapshot_identity_scope_and_integrity_are_rechecked(self):
        from dvidia_training import robot_profiles
        source = self.root/'piper'
        payload,inspected = profile_fixture(source,'piper')
        for variant in ('bytes','scope','axes','manifest','duplicate'):
            for name,raw in payload.items(): (source/name).write_bytes(raw)
            checked = deepcopy(inspected)
            if variant=='bytes':
                (source/'report.html').write_bytes(b'changed after inspection')
            elif variant in ('scope','axes'):
                if variant=='scope': checked['report']['task_qualified']=True
                else: checked['report']['profile']['arm_axes']=5
                changed = bench._json_bytes(checked['report'])
                (source/'report.json').write_bytes(changed)
                for row in checked['files']:
                    if row['path']=='report.json': row.update(record(changed))
                (source/'integrity.json').write_bytes(bench._json_bytes({'schema':'dvidia.robot-profile-integrity/v1','files':checked['files']}))
            elif variant=='manifest':
                (source/'integrity.json').write_bytes(bench._json_bytes({'schema':'wrong','files':checked['files']}))
            sources = [source,source] if variant=='duplicate' else [source]
            with self.subTest(variant=variant),patch.object(robot_profiles,'inspect_benchmark',return_value=checked),self.assertRaises(ValueError):
                prepare.stage(self.replay,self.output,source_commit='a'*40,profile_benchmarks=sources)
            self.assertFalse(self.output.exists())

    def test_piper_contact_pilot_preserves_all_failed_and_invalid_attempts_and_separate_mechanics(self):
        from dvidia_training import piper_task_benchmark as benchmark,robot_profiles
        source = self.root/'piper-task'
        report,payload = piper_task_fixture(source)
        profile = self.root/'piper'
        _,inspected = profile_fixture(profile,'piper')
        with patch.object(benchmark,'load_task_bundle',return_value=(report,payload)), \
                patch.object(robot_profiles,'inspect_benchmark',return_value=inspected):
            plan = prepare.stage(self.replay,self.output,source_commit='a'*40,
                piper_task_benchmark=source,profile_benchmarks=[profile])
        self.assertEqual(publisher.checked_plan(self.output/'publication-plan.json'),plan)
        for name,raw in payload.items():
            self.assertEqual((self.output/'space/tasks/piper'/name).read_bytes(),raw)
        for name,raw in self.original.items():
            self.assertEqual((self.output/'space/replay'/name).read_bytes(),raw)
        landing = (self.output/'space/index.html').read_text()
        self.assertIn('Recorded PiPER contact pilot · synthetic',landing)
        self.assertIn('Recorded native mechanics · unloaded',landing)
        self.assertIn('6 planned attempts across 2 synthetic scenes',landing)
        self.assertIn('low_friction / hold: 0/1 completed, 0 numerically valid',landing)
        self.assertIn('tasks/piper/report.html',landing)
        self.assertIn('tasks/piper/protocol.json',landing)
        self.assertIn('tasks/piper/licenses/agilex_piper/LICENSE',landing)
        self.assertIn('No PiPER 3D viewer is included',landing)
        self.assertIn('/blob/'+'a'*40+'/docs/piper-contact-pilot.md',landing)
        provenance = json.loads((self.output/'space/provenance.json').read_bytes())
        pilot = provenance['contact_task_benchmarks']['piper']
        self.assertEqual((pilot['attempted'],pilot['valid_attempts']),(6,5))
        self.assertTrue(all(row['success']==0 for row in pilot['summary']))
        self.assertFalse(pilot['physical_robot_ready'])
        self.assertFalse(pilot['learned_policy'])
        self.assertEqual(pilot['source_integrity'],record(payload['integrity.json']))
        self.assertIn('## PiPER contact pilot',(self.output/'space/README.md').read_text())

    def test_piper_contact_profile_scope_denominator_counts_and_full_tape_must_agree(self):
        from dvidia_training import piper_task_benchmark as benchmark
        for variant in ('profile','scope','physical','learned','count','summary','drop','raw_outcome','extra','license','private'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                source = root/'piper-task'
                report,payload = piper_task_fixture(source)
                if variant=='profile': report['profile']='other-task'
                elif variant=='scope': report['scope']='Physical learned skills'
                elif variant=='physical': report['physical_robot_ready']=True
                elif variant=='learned': report['learned_policy']=True
                elif variant=='count': report['attempted']=5
                elif variant=='summary': report['summary'][0]['success']=1
                elif variant=='drop':
                    missing = report['attempts'].pop()['result_file']
                    payload.pop(missing)
                    (source/missing).unlink()
                    report['attempted']=5
                elif variant=='raw_outcome':
                    name = report['attempts'][0]['result_file']
                    raw = json.loads(payload[name]);raw['success']=True
                    payload[name]=bench._json_bytes(raw)
                elif variant=='extra': payload['extra.json']=b'{}'
                elif variant=='license':
                    name='licenses/agilex_piper/LICENSE'
                    payload.pop(name);(source/name).unlink()
                else: payload['private.txt']=b'/Users/roguedev/private'
                piper_task_reseal(source,report,payload)
                output = root/'publication'
                with patch.object(benchmark,'load_task_bundle',return_value=(report,payload)),self.assertRaises(ValueError):
                    prepare.stage(self.replay,output,source_commit='a'*40,piper_task_benchmark=source)
                self.assertFalse(output.exists())

    def test_piper_task_snapshot_seal_paths_and_total_budget_are_rechecked_before_writes(self):
        from dvidia_training import piper_task_benchmark as benchmark
        for variant in ('changed','manifest','unlisted','symlink','total'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                source = root/'piper-task'
                report,payload = piper_task_fixture(source)
                if variant=='changed': (source/'report.html').write_bytes(b'changed after inspection')
                elif variant=='manifest':
                    manifest=json.loads(payload['integrity.json']);manifest['files']['report.html']['sha256']='0'*64
                    payload['integrity.json']=bench._json_bytes(manifest)
                    (source/'integrity.json').write_bytes(payload['integrity.json'])
                elif variant=='unlisted': (source/'unlisted.json').write_bytes(b'{}')
                elif variant=='symlink': (source/'linked.json').symlink_to(source/'report.json')
                output = root/'publication'
                cap = sum(len(raw) for raw in self.original.values())+sum(len(raw) for raw in payload.values())-1
                with patch.object(benchmark,'load_task_bundle',return_value=(report,payload)), \
                        patch.object(publisher,'MAX_TOTAL_BYTES',cap if variant=='total' else publisher.MAX_TOTAL_BYTES), \
                        self.assertRaises(ValueError):
                    prepare.stage(self.replay,output,source_commit='a'*40,piper_task_benchmark=source)
                self.assertFalse(output.exists())

    def test_grip_comparison_retains_all_thirty_canonical_tapes_and_original_pilot(self):
        from dvidia_training import piper_grip_benchmark as grip,piper_task_benchmark as pilot
        source=self.root/'grip';report,payload=piper_grip_fixture(source)
        baseline=self.root/'pilot';pilot_report,pilot_files=piper_task_fixture(baseline)
        with patch.object(grip,'load_grip_bundle',return_value=(report,payload)), \
                patch.object(pilot,'load_task_bundle',return_value=(pilot_report,pilot_files)):
            plan=prepare.stage(self.replay,self.output,source_commit='a'*40,
                piper_task_benchmark=baseline,piper_grip_benchmark=source)
        self.assertEqual(publisher.checked_plan(self.output/'publication-plan.json'),plan)
        for name,data in payload.items():
            self.assertEqual((self.output/'space/tasks/piper-grip'/name).read_bytes(),data)
        for name,data in pilot_files.items():
            self.assertEqual((self.output/'space/tasks/piper'/name).read_bytes(),data)
        for name,data in self.original.items():
            self.assertEqual((self.output/'space/replay'/name).read_bytes(),data)
        landing=(self.output/'space/index.html').read_text()
        self.assertIn('30 planned attempts across six synthetic scenes',landing)
        self.assertIn('μ=0.55 / guarded_placement: 0/2 completed',landing)
        self.assertIn('μ=0.15 / hold: 0/2 completed, 1 numerically valid',landing)
        self.assertIn('tasks/piper-grip/report.html',landing)
        self.assertIn('/blob/'+'a'*40+'/docs/piper-grip-recovery.md',landing)
        provenance=json.loads((self.output/'space/provenance.json').read_bytes())
        compared=provenance['contact_task_benchmarks']['piper-grip']
        self.assertEqual((compared['attempted'],compared['valid_attempts']),(30,29))
        self.assertEqual(len(compared['attempt_ids']),30)
        self.assertTrue(all(row['success']==0 for row in compared['summary']))
        self.assertEqual(compared['source_integrity'],record(payload['integrity.json']))
        self.assertGreater(compared['decoded_bytes'],compared['stored_bytes'])
        self.assertFalse(compared['learned_policy'])
        card=(self.output/'space/README.md').read_text()
        self.assertIn('canonical deterministic gzip',card)
        self.assertIn('dvidia-piper-grip inspect',card)

    def test_grip_leading_completion_comparison_is_derived_and_exposes_regression(self):
        report,_=piper_grip_fixture(self.root/'grip-summary')
        outcomes = (
            ((2,2,1),(1,0,0),'baseline 5/6 versus guarded 1/6',True),
            ((0,0,0),(0,0,0),'baseline 0/6 versus guarded 0/6',False),
            ((1,0,0),(1,2,0),'baseline 1/6 versus guarded 3/6',False),
        )
        for baseline,guarded,expected,regressed in outcomes:
            with self.subTest(expected=expected):
                candidate=deepcopy(report)
                for variant,counts in (('baseline_placement',baseline),('guarded_placement',guarded)):
                    rows=[row for row in candidate['summary'] if row['variant']==variant]
                    for row,count in zip(rows,counts):
                        row['success']=count
                summary=prepare._grip_summary(candidate)
                self.assertTrue(summary.startswith('Placement completion: '+expected+'.'))
                self.assertEqual('This cohort does not show improved placement success.' in summary,regressed)
                self.assertIn('Retries and qualification decisions do not imply successful recovery.',summary)
                card=prepare._piper_grip_card(candidate,'a'*40)
                self.assertIn('<strong>Placement completion: '+expected+'.</strong>',card)
                self.assertLess(card.index('Placement completion:'),card.index('privileged native state'))
                self.assertEqual('This cohort does not show improved placement success.' in card,regressed)

    def test_grip_source_scope_pairing_denominator_decoded_seals_and_bytes_are_rechecked(self):
        from dvidia_training import piper_grip_benchmark as grip
        for variant in ('scope','count','summary','drop','decoded_hash','changed','private','unlisted','symlink'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as folder:
                root=Path(folder).resolve();source=root/'grip';report,payload=piper_grip_fixture(source)
                if variant=='scope': report['learned_policy']=True
                elif variant=='count': report['attempted']=29
                elif variant=='summary': report['summary'][0]['success']=1
                elif variant=='drop':
                    name=report['attempts'].pop()['result_file'];payload.pop(name);(source/name).unlink()
                elif variant=='private':
                    name=report['attempts'][0]['result_file'];raw=json.loads(grip.decode_attempt(payload[name]))
                    raw['note']='/Users/roguedev/private'
                    payload[name]=gzip.compress(bench._json_bytes(raw),mtime=0)
                    report['attempts'][0]={**grip._summary(raw),'result_file':name}
                piper_grip_reseal(source,report,payload)
                if variant=='decoded_hash':
                    seal=json.loads(payload['integrity.json']);name=report['attempts'][0]['result_file']
                    seal['files'][name]['decoded_sha256']='0'*64
                    payload['integrity.json']=bench._json_bytes(seal);(source/'integrity.json').write_bytes(payload['integrity.json'])
                elif variant=='changed': (source/'report.html').write_bytes(b'changed after inspection')
                elif variant=='unlisted': (source/'unlisted.json').write_bytes(b'{}')
                elif variant=='symlink': (source/'linked.json').symlink_to(source/'report.json')
                output=root/'publication'
                with patch.object(grip,'load_grip_bundle',return_value=(report,payload)),self.assertRaises(ValueError):
                    prepare.stage(self.replay,output,source_commit='a'*40,piper_grip_benchmark=source)
                self.assertFalse(output.exists())

    def test_grip_staging_keeps_existing_publication_cap_and_bound_decoding(self):
        from dvidia_training import piper_grip_benchmark as grip
        source=self.root/'grip';report,payload=piper_grip_fixture(source)
        cap=sum(len(raw) for raw in self.original.values())+sum(len(raw) for raw in payload.values())-1
        for variant in ('stored','decoded'):
            with self.subTest(variant=variant),patch.object(grip,'load_grip_bundle',return_value=(report,payload)), \
                    patch.object(publisher,'MAX_TOTAL_BYTES',cap if variant=='stored' else publisher.MAX_TOTAL_BYTES), \
                    patch.object(grip,'MAX_DECODED_BYTES',1 if variant=='decoded' else grip.MAX_DECODED_BYTES), \
                    self.assertRaises(ValueError):
                prepare.stage(self.replay,self.output,source_commit='a'*40,piper_grip_benchmark=source)
            self.assertFalse(self.output.exists())
        plan=prepare.stage(self.replay,self.output,source_commit='a'*40)
        self.assertFalse(any(row['path'].startswith('tasks/piper-grip/') for row in plan['repository']['files']))
        self.assertNotIn('PiPER guarded grip comparison',(self.output/'space/index.html').read_text())

    def test_v03_retains_all_eighty_four_attempts_and_exact_prior_evidence(self):
        from dvidia_training import piper_contact_benchmark as contact,piper_grip_benchmark as grip,piper_task_benchmark as pilot
        source=self.root/'contact';report,payload=piper_contact_fixture(source)
        old_grip=self.root/'grip';grip_report,grip_files=piper_grip_fixture(old_grip)
        old_pilot=self.root/'pilot';pilot_report,pilot_files=piper_task_fixture(old_pilot)
        with patch.object(contact,'load_contact_bundle',return_value=(report,payload)), \
                patch.object(grip,'load_grip_bundle',return_value=(grip_report,grip_files)), \
                patch.object(pilot,'load_task_bundle',return_value=(pilot_report,pilot_files)):
            plan=prepare.stage(self.replay,self.output,source_commit='a'*40,piper_task_benchmark=old_pilot,
                piper_grip_benchmark=old_grip,piper_contact_benchmark=source)
        self.assertEqual(publisher.checked_plan(self.output/'publication-plan.json'),plan)
        for prefix,files in (('tasks/piper-contact-v03',payload),('tasks/piper-grip',grip_files),
                             ('tasks/piper',pilot_files),('replay',self.original)):
            for name,data in files.items():
                self.assertEqual((self.output/'space'/prefix/name).read_bytes(),data)
        self.assertEqual(len(payload),90)
        landing=(self.output/'space/index.html').read_text()
        comparison='Placement completion: baseline 0/12, probe-only 0/12, history 0/12.'
        self.assertIn('<strong>'+comparison+'</strong>',landing)
        self.assertIn('84 planned attempts across twelve synthetic scenes',landing)
        self.assertIn('μ=0.15, mass=0.06 kg / hold: 0/2 completed, 1 numerically valid',landing)
        self.assertIn('/blob/'+'a'*40+'/docs/piper-contact-memory-v03.md',landing)
        self.assertIn('Retries, resumes and release refusals do not imply successful placement or recovery.',landing)
        self.assertLess(landing.index(comparison),landing.index('Three authored controllers'))
        provenance=json.loads((self.output/'space/provenance.json').read_bytes())
        compared=provenance['contact_task_benchmarks']['piper-contact-v03']
        self.assertEqual((compared['attempted'],compared['valid_attempts']),(84,83))
        self.assertEqual(len(compared['attempt_ids']),84)
        self.assertEqual(compared['source_integrity'],record(payload['integrity.json']))
        self.assertGreater(compared['decoded_bytes'],compared['stored_bytes'])
        self.assertFalse(compared['learned_policy'])
        self.assertEqual(provenance['publication_byte_bounds']['stored_payload_bytes'],512*1024*1024)
        self.assertEqual(provenance['publication_byte_bounds']['per_file_bytes'],32*1024*1024)
        readme=(self.output/'space/README.md').read_text()
        self.assertIn(comparison,readme)
        self.assertIn('512 MiB of stored files',readme)

    def test_v03_summary_derives_all_three_completion_totals_without_claiming_improvement(self):
        report,_=piper_contact_fixture(self.root/'summary')
        for outcome in ((0,0,0),(5,1,2),(1,4,8)):
            with self.subTest(outcome=outcome):
                candidate=deepcopy(report)
                for variant,count in zip(('baseline_placement','probe_only_placement','history_placement'),outcome):
                    for row in candidate['summary']:
                        if row['variant']==variant:
                            row['success']=min(count,row['attempted']);count-=row['success']
                    self.assertEqual(count,0)
                expected=f'Placement completion: baseline {outcome[0]}/12, probe-only {outcome[1]}/12, history {outcome[2]}/12.'
                self.assertTrue(prepare._contact_summary(candidate).startswith(expected))
                card=prepare._piper_contact_card(candidate,'a'*40)
                self.assertIn('<strong>'+expected+'</strong>',card)
                self.assertNotIn('improved',card)
                self.assertIn('do not imply successful placement or recovery',card)

    def test_v03_scope_full_grid_denominator_decoded_hashes_and_snapshots_are_rechecked(self):
        from dvidia_training import piper_contact_benchmark as contact
        for variant in ('scope','count','mass','summary','missing_summary','drop','decoded_hash','changed','private','header_private','unlisted','symlink'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as folder:
                root=Path(folder).resolve();source=root/'contact';report,payload=piper_contact_fixture(source)
                if variant=='scope': report['physical_robot_ready']=True
                elif variant=='count': report['attempted']=83
                elif variant=='mass': report['scenes'][0]['mass_kg']=.06
                elif variant=='summary': report['summary'][0]['success']=1
                elif variant=='missing_summary':
                    report['summary']=[row for row in report['summary'] if row['variant']!='probe_only_placement']
                elif variant=='drop':
                    name=report['attempts'].pop()['result_file'];payload.pop(name);(source/name).unlink()
                elif variant=='private':
                    name=report['attempts'][0]['result_file'];raw=json.loads(contact.decode_attempt(payload[name]))
                    raw['note']='/Users/roguedev/private';payload[name]=gzip.compress(bench._json_bytes(raw),mtime=0)
                    report['attempts'][0]={**contact._summary(raw),'result_file':name}
                elif variant=='header_private':
                    name=report['attempts'][0]['result_file'];data=payload[name]
                    header=bytearray(data[:10]);header[3]|=8
                    payload[name]=bytes(header)+b'/Users/roguedev/private\0'+data[10:]
                piper_contact_reseal(source,report,payload)
                if variant=='decoded_hash':
                    seal=json.loads(payload['integrity.json']);name=report['attempts'][0]['result_file']
                    seal['files'][name]['decoded_sha256']='0'*64
                    payload['integrity.json']=bench._json_bytes(seal);(source/'integrity.json').write_bytes(payload['integrity.json'])
                elif variant=='changed': (source/'report.html').write_bytes(b'changed after inspection')
                elif variant=='unlisted': (source/'unlisted.json').write_bytes(b'{}')
                elif variant=='symlink': (source/'linked.json').symlink_to(source/'report.json')
                output=root/'publication'
                with patch.object(contact,'load_contact_bundle',return_value=(report,payload)),self.assertRaises(ValueError):
                    prepare.stage(self.replay,output,source_commit='a'*40,piper_contact_benchmark=source)
                self.assertFalse(output.exists())

    def test_v03_optional_flag_preserves_byte_caps_and_decoding_bounds(self):
        from dvidia_training import piper_contact_benchmark as contact
        source=self.root/'contact';report,payload=piper_contact_fixture(source)
        cap=sum(len(raw) for raw in self.original.values())+sum(len(raw) for raw in payload.values())-1
        for variant in ('stored','decoded','file'):
            with self.subTest(variant=variant),patch.object(contact,'load_contact_bundle',return_value=(report,payload)), \
                    patch.object(publisher,'MAX_TOTAL_BYTES',cap if variant=='stored' else publisher.MAX_TOTAL_BYTES), \
                    patch.object(publisher,'MAX_FILE_BYTES',1 if variant=='file' else publisher.MAX_FILE_BYTES), \
                    patch.object(contact,'MAX_DECODED_BYTES',1 if variant=='decoded' else contact.MAX_DECODED_BYTES), \
                    self.assertRaises(ValueError):
                prepare.stage(self.replay,self.output,source_commit='a'*40,piper_contact_benchmark=source)
            self.assertFalse(self.output.exists())
        plan=prepare.stage(self.replay,self.output,source_commit='a'*40)
        self.assertFalse(any(row['path'].startswith('tasks/piper-contact-v03/') for row in plan['repository']['files']))
        self.assertNotIn('PiPER contact memory v0.3',(self.output/'space/index.html').read_text())

    def test_v04_retains_all_eighty_four_attempts_and_exact_prior_evidence(self):
        from dvidia_training import piper_recovery_benchmark as contact,piper_grip_benchmark as grip,piper_task_benchmark as pilot
        source=self.root/'contact';report,payload=piper_recovery_fixture(source)
        old_grip=self.root/'grip';grip_report,grip_files=piper_grip_fixture(old_grip)
        old_pilot=self.root/'pilot';pilot_report,pilot_files=piper_task_fixture(old_pilot)
        with patch.object(contact,'load_recovery_bundle',return_value=(report,payload)), \
                patch.object(grip,'load_grip_bundle',return_value=(grip_report,grip_files)), \
                patch.object(pilot,'load_task_bundle',return_value=(pilot_report,pilot_files)):
            plan=prepare.stage(self.replay,self.output,source_commit='a'*40,piper_task_benchmark=old_pilot,
                piper_grip_benchmark=old_grip,piper_recovery_benchmark=source)
        self.assertEqual(publisher.checked_plan(self.output/'publication-plan.json'),plan)
        for prefix,files in (('tasks/piper-recovery-v04',payload),('tasks/piper-grip',grip_files),
                             ('tasks/piper',pilot_files),('replay',self.original)):
            for name,data in files.items():
                self.assertEqual((self.output/'space'/prefix/name).read_bytes(),data)
        self.assertEqual(len(payload),90)
        landing=(self.output/'space/index.html').read_text()
        comparison='Placement completion: v0.3 0/12, v0.4 carry-only 0/12, v0.4 full 0/12.'
        self.assertIn('<strong>'+comparison+'</strong>',landing)
        self.assertIn('84 planned attempts across twelve synthetic scenes',landing)
        self.assertIn('μ=0.15, mass=0.06 kg / hold: 0/2 completed, 1 numerically valid',landing)
        self.assertIn('/blob/'+'a'*40+'/docs/piper-contact-recovery-v04.md',landing)
        self.assertIn('Retries, resumes and grip refusals do not imply successful placement or recovery.',landing)
        self.assertLess(landing.index(comparison),landing.index('Frozen v0.3, v0.4 carry-only and v0.4 full'))
        provenance=json.loads((self.output/'space/provenance.json').read_bytes())
        compared=provenance['contact_task_benchmarks']['piper-recovery-v04']
        self.assertEqual((compared['attempted'],compared['valid_attempts']),(84,83))
        self.assertEqual(len(compared['attempt_ids']),84)
        self.assertEqual(compared['source_integrity'],record(payload['integrity.json']))
        self.assertGreater(compared['decoded_bytes'],compared['stored_bytes'])
        self.assertFalse(compared['learned_policy'])
        self.assertEqual(provenance['publication_byte_bounds']['stored_payload_bytes'],512*1024*1024)
        self.assertEqual(provenance['publication_byte_bounds']['per_file_bytes'],32*1024*1024)
        readme=(self.output/'space/README.md').read_text()
        self.assertIn(comparison,readme)
        self.assertIn('512 MiB of stored files',readme)

    def test_v04_summary_derives_all_three_completion_totals_without_claiming_improvement(self):
        report,_=piper_recovery_fixture(self.root/'summary')
        for outcome in ((0,0,0),(5,1,2),(1,4,8)):
            with self.subTest(outcome=outcome):
                candidate=deepcopy(report)
                for variant,count in zip(('history_placement','recovery_carry_placement','recovery_placement'),outcome):
                    for row in candidate['summary']:
                        if row['variant']==variant:
                            row['success']=min(count,row['attempted']);count-=row['success']
                    self.assertEqual(count,0)
                expected=f'Placement completion: v0.3 {outcome[0]}/12, v0.4 carry-only {outcome[1]}/12, v0.4 full {outcome[2]}/12.'
                self.assertTrue(prepare._recovery_summary(candidate).startswith(expected))
                card=prepare._piper_recovery_card(candidate,'a'*40)
                self.assertIn('<strong>'+expected+'</strong>',card)
                self.assertNotIn('improved',card)
                self.assertIn('do not imply successful placement or recovery',card)

    def test_v04_speed_summary_keeps_regressions_and_common_success_boundary(self):
        report,_=piper_recovery_fixture(self.root/'timing')
        report['paired_comparisons']=[{'reference':'history','candidate':'recovery',
            'common_successes':3,'gained_successes':0,'lost_successes':2,
            'common_success_mean_delta_s':1.234}]
        summary=prepare._recovery_summary(report)
        self.assertIn('lost placements: 2',summary)
        self.assertIn('+1.234 simulated seconds (v0.4 minus v0.3)',summary)
        self.assertNotIn('faster',summary)
        report['paired_comparisons'][0].update(common_successes=0,common_success_mean_delta_s=None)
        self.assertNotIn('Mean completion-time change',prepare._recovery_summary(report))

    def test_v04_scope_full_grid_denominator_decoded_hashes_and_snapshots_are_rechecked(self):
        from dvidia_training import piper_recovery_benchmark as contact
        for variant in ('scope','count','mass','summary','missing_summary','drop','decoded_hash','changed','private','header_private','unlisted','symlink'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as folder:
                root=Path(folder).resolve();source=root/'contact';report,payload=piper_recovery_fixture(source)
                if variant=='scope': report['physical_robot_ready']=True
                elif variant=='count': report['attempted']=83
                elif variant=='mass': report['scenes'][0]['mass_kg']=.06
                elif variant=='summary': report['summary'][0]['success']=1
                elif variant=='missing_summary':
                    report['summary']=[row for row in report['summary'] if row['variant']!='recovery_carry_placement']
                elif variant=='drop':
                    name=report['attempts'].pop()['result_file'];payload.pop(name);(source/name).unlink()
                elif variant=='private':
                    name=report['attempts'][0]['result_file'];raw=json.loads(contact.decode_attempt(payload[name]))
                    raw['note']='/Users/roguedev/private';payload[name]=gzip.compress(bench._json_bytes(raw),mtime=0)
                    report['attempts'][0]={**contact._summary(raw),'result_file':name}
                elif variant=='header_private':
                    name=report['attempts'][0]['result_file'];data=payload[name]
                    header=bytearray(data[:10]);header[3]|=8
                    payload[name]=bytes(header)+b'/Users/roguedev/private\0'+data[10:]
                piper_recovery_reseal(source,report,payload)
                if variant=='decoded_hash':
                    seal=json.loads(payload['integrity.json']);name=report['attempts'][0]['result_file']
                    seal['files'][name]['decoded_sha256']='0'*64
                    payload['integrity.json']=bench._json_bytes(seal);(source/'integrity.json').write_bytes(payload['integrity.json'])
                elif variant=='changed': (source/'report.html').write_bytes(b'changed after inspection')
                elif variant=='unlisted': (source/'unlisted.json').write_bytes(b'{}')
                elif variant=='symlink': (source/'linked.json').symlink_to(source/'report.json')
                output=root/'publication'
                with patch.object(contact,'load_recovery_bundle',return_value=(report,payload)),self.assertRaises(ValueError):
                    prepare.stage(self.replay,output,source_commit='a'*40,piper_recovery_benchmark=source)
                self.assertFalse(output.exists())

    def test_v04_optional_flag_preserves_byte_caps_and_decoding_bounds(self):
        from dvidia_training import piper_recovery_benchmark as contact
        source=self.root/'contact';report,payload=piper_recovery_fixture(source)
        cap=sum(len(raw) for raw in self.original.values())+sum(len(raw) for raw in payload.values())-1
        for variant in ('stored','decoded','file'):
            with self.subTest(variant=variant),patch.object(contact,'load_recovery_bundle',return_value=(report,payload)), \
                    patch.object(publisher,'MAX_TOTAL_BYTES',cap if variant=='stored' else publisher.MAX_TOTAL_BYTES), \
                    patch.object(publisher,'MAX_FILE_BYTES',1 if variant=='file' else publisher.MAX_FILE_BYTES), \
                    patch.object(contact,'MAX_DECODED_BYTES',1 if variant=='decoded' else contact.MAX_DECODED_BYTES), \
                    self.assertRaises(ValueError):
                prepare.stage(self.replay,self.output,source_commit='a'*40,piper_recovery_benchmark=source)
            self.assertFalse(self.output.exists())
        plan=prepare.stage(self.replay,self.output,source_commit='a'*40)
        self.assertFalse(any(row['path'].startswith('tasks/piper-recovery-v04/') for row in plan['repository']['files']))
        self.assertNotIn('PiPER grip recovery v0.4',(self.output/'space/index.html').read_text())


if __name__=='__main__':
    unittest.main()
