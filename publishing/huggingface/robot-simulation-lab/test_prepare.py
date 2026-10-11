"""Data-only staging, exact model notices and complete task receipt boundaries."""
from copy import deepcopy
from hashlib import sha256
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


if __name__=='__main__':
    unittest.main()
