"""Lean recorded-export contracts without stepping physics or running policies."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import base64
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_grip_clock_viewer as viewer
from dvidia_training import piper_grip_clock_model as clock
from dvidia_training import piper_imitation as core
from tests.test_piper_feedback_model import observation


def put(root, name, value):
    raw = core.canonical(value)
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


class ClockViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.root = self.path/'study'
        self.root.mkdir()
        self.revision = 'a'*40

    def documents(self):
        cases = viewer.lifecycle.make_cases()['confirmation']
        sources, runtime = viewer.lifecycle.source_hashes(), viewer.runner.runtime_hashes()
        for name in ('base-model.json', 'jaw-model.json', 'selected.json', 'blind.json'):
            put(self.root, name, {'synthetic_unit_test': name})
        for name in sources:
            path = self.root/'sources'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(viewer.lifecycle.__file__).with_name(name).read_bytes())
        protocol = dict(schema=viewer.runner.SCHEMA, cases={'confirmation': cases}, variants=list(viewer.VARIANTS),
                        source_sha256=sources, physical_robot_ready=False, jaw_mode='coordinated_reference')
        p = put(self.root, 'protocol.json', protocol)
        frozen = dict(schema=viewer.runner.SCHEMA, protocol_sha256=sha256(p).hexdigest(),
                      runtime_sha256=runtime, lifecycle_source_sha256=sources, physical_robot_ready=False,
                      jaw_mode='coordinated_reference', **{key: sha256((self.root/name).read_bytes()).hexdigest()
                      for name, key in (('base-model.json', 'base_model_raw_sha256'), ('jaw-model.json', 'jaw_model_raw_sha256'),
                                        ('selected.json', 'selected_sha256'), ('blind.json', 'blind_sha256'))})
        f = put(self.root, 'frozen.json', frozen)
        rows = [dict(case_id=case['id'], condition=case['condition'], variant=variant,
                     score=dict(valid=True, success=variant == 'fixed', post_grip_analysis={}), receipt_sha256='e'*64)
                for case in cases for variant in viewer.VARIANTS]
        report = dict(schema=viewer.runner.SCHEMA, protocol_sha256=sha256(p).hexdigest(),
                      frozen_sha256=sha256(f).hexdigest(), jaw_mode='coordinated_reference',
                      physical_robot_ready=False, qualified_robot_skill=False, rows=rows,
                      variants={variant: dict(scheduled=128, valid=128, strict_successes=128 if variant == 'fixed' else 0)
                                for variant in viewer.VARIANTS}, contrasts=[], gates={},
                      advance_to_followup_simulation=False, limitations=['Synthetic unit-test data; no physics.'])
        put(self.root, 'report.json', report)
        return protocol, frozen, report

    def test_index_space_three_digits_empty_pose_and_exact_payload_reconstruction(self):
        for index in (0, 99, 100, 257, 319):
            value = viewer.payload(index, b'[]', b'')
            raw = viewer.script(value)
            self.assertTrue(raw.startswith(b'globalThis["__dvidiaPiperClockResourcesV1"]('))
            self.assertIn(f'"replay-{index:03}"'.encode(), raw)
            for receipt, expected in zip(value['resources'], (b'[]', b'')):
                packed = base64.b64decode(receipt['data'], validate=True)
                self.assertEqual(gzip.decompress(packed), expected)
                self.assertEqual(sha256(packed).hexdigest(), receipt['compressed_sha256'])
                self.assertEqual(sha256(expected).hexdigest(), receipt['sha256'])
        for index in (-1, 320, True, 1.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                viewer.payload(index, b'[]', b'')
        for name in ('../poses-000.bin', 'poses-320.bin', 'poses-999.bin', 'frames-0010.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                viewer.resource(name, b'[]')
        value = viewer.payload(100, b'[]', b'')
        for change in ('sha256', 'data', 'href'):
            altered = deepcopy(value)
            altered['resources'][0][change] = '0'*64 if change == 'sha256' else 'AAAA' if change == 'data' else 'frames-099.json'
            with self.subTest(change=change), self.assertRaises(ValueError):
                viewer.script(altered)

    def test_pinned_evidence_links_reject_mutable_foreign_wrong_case_and_wrong_file(self):
        case_id = 'confirmation-xy00-g0-nominal'
        href = viewer._native_href(self.revision, case_id, 'clock', 'native.npz')
        self.assertEqual(viewer.validate_evidence_href(href, self.revision, case_id, 'clock', 'native.npz'), href)
        for bad in (href.replace(self.revision, 'main'), href.replace('huggingface.co', 'example.com'),
                    href+'?download=1', href.replace('/clock/', '/blind_clock/'), href.replace('native.npz', 'score.json')):
            with self.subTest(href=bad), self.assertRaises(ValueError):
                viewer.validate_evidence_href(bad, self.revision, case_id, 'clock', 'native.npz')
        for revision in ('main', 'A'*40, 'a'*39, True):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                viewer._native_href(revision, case_id, 'clock', 'native.npz')

    def test_verified_study_requires_all512_outcomes_frozen_sources_and_numeric_report_agreement(self):
        _, _, report = self.documents()
        original = {name: (self.root/name).read_bytes() for name in viewer.INPUT_DOCUMENTS}
        with patch.object(viewer.lifecycle, '_report', return_value=report) as verify:
            raw, _, _, actual = viewer._verified_study(self.root)
        verify.assert_called_once_with(self.root.resolve())
        self.assertEqual(raw, original)
        self.assertEqual(actual, report)
        self.assertEqual({name: (self.root/name).read_bytes() for name in viewer.INPUT_DOCUMENTS}, original)
        for field, value in (('rows', report['rows'][:-1]), ('qualified_robot_skill', True),
                             ('frozen_sha256', '0'*64), ('jaw_mode', 'physical_adapter')):
            bad = dict(report, **{field: value})
            put(self.root, 'report.json', bad)
            with self.subTest(field=field), patch.object(viewer.lifecycle, '_report') as verify, self.assertRaises(ValueError):
                viewer._verified_study(self.root)
            verify.assert_not_called()
        put(self.root, 'report.json', report)
        with patch.object(viewer.lifecycle, '_report', return_value={**report, 'gates': {'changed': True}}), self.assertRaises(ValueError):
            viewer._verified_study(self.root)
        source = next(iter(viewer.lifecycle.source_hashes()))
        (self.root/'sources'/source).write_bytes(b'changed')
        with patch.object(viewer.lifecycle, '_report') as verify, self.assertRaises(ValueError):
            viewer._verified_study(self.root)
        verify.assert_not_called()

    def test_readonly_export_does_not_accept_a_mock_fixture_as_a_real_completed_study(self):
        self.documents()
        # No mocked verifier here: missing real lifecycle/pre-confirm evidence
        # must stop, rather than passing only superficial document hashes.
        with self.assertRaises((ValueError, OSError)):
            viewer._verified_study(self.root)

    def test_selection_keeps_failed_learned_attempts_and_first_scheduled_controls(self):
        _, _, report = self.documents()
        selected = viewer.selected_replays(report['rows'])
        self.assertEqual(len(selected), 258)
        self.assertEqual(sum(row['variant'] == 'clock' for row in selected), 128)
        self.assertEqual(sum(row['variant'] == 'blind_clock' for row in selected), 128)
        self.assertTrue(all(not row['score']['success'] for row in selected if row['variant'] in ('clock', 'blind_clock')))
        for variant in ('fixed', 'teacher'):
            self.assertEqual([row for row in selected if row['variant'] == variant],
                             [next(row for row in report['rows'] if row['variant'] == variant)])

    def test_attempt_receipts_bind_every_original_file_and_exact_model_request(self):
        protocol, frozen, report = self.documents()
        case, row = protocol['cases']['confirmation'][0], report['rows'][0]
        prefix = 'confirmation/'+case['id']+'/clock'
        binding = dict(schema=viewer.runner.SCHEMA, case=case, split='confirmation', variant='clock',
            collect_labels=False, jaw_mode='coordinated_reference', protocol_sha256=frozen['protocol_sha256'],
            runtime_sha256=frozen['runtime_sha256'], model_raw_sha256=frozen['selected_sha256'],
            base_model_raw_sha256=frozen['base_model_raw_sha256'], jaw_model_raw_sha256=frozen['jaw_model_raw_sha256'])
        files = {}
        for name, value in (('started.json', binding), ('score.json', row['score']), ('post-grip.json', {}), ('native.json', {})):
            files[name] = put(self.root, prefix+'/'+name, value)
        folder = self.root/prefix
        files['native.npz'] = b'numeric-archive-identity-is-checked-by-full-report-verifier'
        files['episode.json.gz'] = gzip.compress(b'{}', mtime=0)
        for name in ('native.npz', 'episode.json.gz'):
            (folder/name).write_bytes(files[name])
        raw = put(self.root, prefix+'/receipt.json', dict(binding=binding, case_id=case['id'], variant='clock',
            files={name: sha256(value).hexdigest() for name, value in files.items()}))
        row = {**row, 'receipt_sha256': sha256(raw).hexdigest()}
        evidence, inputs, episode = viewer._attempt(self.root, row, case, frozen, self.revision, decode_episode=False)
        self.assertIsNone(episode)
        self.assertEqual(len(inputs), 7)
        self.assertEqual(evidence['native.npz']['sha256'], sha256(files['native.npz']).hexdigest())
        (folder/'native.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'receipt'):
            viewer._attempt(self.root, row, case, frozen, self.revision, decode_episode=False)
        (folder/'native.npz').write_bytes(files['native.npz'])
        with self.assertRaisesRegex(ValueError, 'identity'):
            viewer._attempt(self.root, row, case, {**frozen, 'selected_sha256': '0'*64}, self.revision, decode_episode=False)
        (folder/'unexpected.py').write_bytes(b'pass')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            viewer._attempt(self.root, row, case, frozen, self.revision, decode_episode=False)

    def episode(self, variant):
        initial, terminal = observation(), observation(.02)
        increment = .02 if variant in ('fixed', 'blind_clock') else 0.
        feature = clock.ClockFeatures(blind=variant == 'blind_clock')(initial, {'gripper_position_m': .035}, 0.)
        decision = dict(physical_time_s=0., progress_before_s=0., progress_after_s=increment, increment_s=increment,
                        features=feature.tolist(), nominal_physical_jaw_m=.035, frozen_jaw_target_m=.035,
                        jaw_mode='coordinated_reference', decision_origin=viewer.ORIGINS[variant])
        if variant in ('clock', 'blind_clock'):
            decision['advance_probability'] = 1. if increment else 0.
        elif variant == 'teacher':
            decision['secure_grip'] = False
        action = dict(t_seconds=0., joint_targets_rad=initial['joint_position_rad'], gripper_position_m=.035,
                      clock_decision=decision)
        initial.update(command=deepcopy(action), issued_joint_targets_rad=initial['joint_position_rad'], issued_gripper_position_m=.035)
        terminal.update(command=None, issued_joint_targets_rad=None, issued_gripper_position_m=None)
        return dict(config={}, valid=False, actions=[action], trajectory=[initial, terminal], compiled_model_sha256='f'*64)

    @staticmethod
    def frame_rows(episode):
        return [dict(t_seconds=state['t_seconds'], joint_position_rad=state['joint_position_rad'],
                     gripper_joint_position_m=state['gripper_joint_position_m'],
                     object_position_m=state['object_position_m'], tcp_position_m=state['tcp_position_m'],
                     issued_gripper_position_m=state['issued_gripper_position_m']) for state in episode['trajectory']]

    def test_frame_clocks_preserve_hold_and_terminal_state_without_fabricating_actions(self):
        episode = self.episode('clock')
        rows = viewer._clock_rows(episode, self.frame_rows(episode), 'clock')
        self.assertEqual(rows[0]['virtual_progress_s'], 0.)
        self.assertEqual(rows[0]['next_clock_decision']['increment_s'], 0.)
        self.assertEqual(rows[-1]['clock_debt_s'], .02)
        self.assertIsNone(rows[-1]['next_clock_decision'])
        self.assertEqual(rows[0]['next_clock_decision']['origin'], 'learned_grip_clock')
        for field, value in (('progress_before_s', .02), ('increment_s', .01),
                             ('decision_origin', 'authored_causal_grip_teacher'), ('advance_probability', 1.)):
            bad = deepcopy(episode); bad['actions'][0]['clock_decision'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                viewer._clock_rows(bad, self.frame_rows(bad), 'clock')
        bad = deepcopy(episode); bad['trajectory'][-1]['command'] = bad['actions'][0]
        with self.assertRaises(ValueError):
            viewer._clock_rows(bad, self.frame_rows(bad), 'clock')
        interrupted = self.episode('clock'); interrupted['trajectory'].pop()
        self.assertEqual(len(viewer._clock_rows(interrupted, self.frame_rows(interrupted), 'clock')), 1)

    def test_full_mocked_export_is_lean_sealed_pinned_and_preserves_source_document_bytes(self):
        protocol, frozen, report = self.documents()
        raw = {name: (self.root/name).read_bytes() for name in viewer.INPUT_DOCUMENTS}
        native_scene = dict(bodies=[{'id': 0, 'name': 'synthetic'}], meshes=[], geoms=[])
        def attempt(root, row, case, bound, revision, *, decode_episode):
            evidence = {name: dict(href=viewer._native_href(revision, case['id'], row['variant'], name), bytes=2, sha256='d'*64)
                        for name in viewer.ATTEMPT_FILES|{'receipt.json'}}
            return evidence, {}, self.episode(row['variant']) if decode_episode else None
        def poses(model, episode, residuals):
            return np.zeros(len(episode['trajectory'])*7, dtype='<f4').tobytes(), self.frame_rows(episode)
        def build(command, check):
            self.assertTrue(check)
            Path(command[-1]).write_bytes(b'/* mocked native script build; no execution */')
        with ExitStack() as stack:
            stack.enter_context(patch.object(viewer, '_verified_study', return_value=(raw, protocol, frozen, report)))
            stack.enter_context(patch.object(viewer, '_attempt', side_effect=attempt))
            stack.enter_context(patch.object(viewer, '_poses', side_effect=poses))
            stack.enter_context(patch.object(viewer, '_scene', return_value=(native_scene, b'geometry-bytes')))
            stack.enter_context(patch.object(viewer.sim, 'native_warnings', side_effect=lambda warnings: nullcontext()))
            stack.enter_context(patch.object(viewer.sim, 'model_digest', return_value='f'*64))
            stack.enter_context(patch('dvidia_training.piper_task.build_task_model', return_value=object()))
            stack.enter_context(patch('dvidia_training.piper_task._asset_root', return_value=self.path))
            stack.enter_context(patch('dvidia_training.robot_profiles.verify_assets', return_value={'synthetic_unit_test': True}))
            stack.enter_context(patch('dvidia_training.robot_profiles._payload', return_value=({}, {'agilex_piper/LICENSE': b'MIT unit test'})))
            stack.enter_context(patch.object(viewer.subprocess, 'run', side_effect=build))
            output = self.path/'viewer'
            result = viewer.export_viewer(self.root, output, self.path, self.revision)
        self.assertEqual(result['replay_cases'], 258)
        self.assertEqual(result['report_attempts'], 512)
        seal = json.loads((output/'viewer-integrity.json').read_bytes())
        files = {path.relative_to(output).as_posix() for path in output.rglob('*') if path.is_file()}
        self.assertEqual(files, set(seal['files'])|{'viewer-integrity.json'})
        self.assertFalse(any(name.endswith(('.npz', '.bin', '.gz')) or name.startswith('confirmation/') for name in files))
        self.assertEqual(len([name for name in files if name.startswith('replay-data-')]), 257)
        for name, receipt in seal['files'].items():
            self.assertEqual(viewer._receipt((output/name).read_bytes()), receipt)
        for name in ('report.json', 'protocol.json', 'frozen.json'):
            self.assertEqual((output/name).read_bytes(), raw[name])
            self.assertEqual((self.root/name).read_bytes(), raw[name])
        motion = json.loads((output/'motion.json').read_bytes())
        manifest = json.loads((output/'replay-resources.json').read_bytes())
        self.assertEqual(len(motion['all_attempts']), 512)
        self.assertEqual(manifest['first']['id'], 'replay-000')
        self.assertFalse(motion['cases'][0]['score']['success'])
        for entry in motion['all_attempts']:
            for name, receipt in entry['evidence'].items():
                viewer.validate_evidence_href(receipt['href'], self.revision, entry['case_id'], entry['variant'], name)
        self.assertIn('?sha256=', (output/'index.html').read_text())
        self.assertIn('integrity="sha384-', (output/'index.html').read_text())
        self.assertLess(result['bytes'], viewer.MAX_NAMESPACE)
        # Exercise the actual browser validators against the Python export;
        # isolated VM functions only, without DOM/WebGL or native simulation.
        compatibility = r'''
const fs=require('fs'),vm=require('vm'),zlib=require('zlib'),crypto=require('crypto');
const root=process.argv[1],read=name=>fs.readFileSync(root+'/'+name);
const manifest=JSON.parse(read('replay-resources.json'));
let source=read('piper_grip_clock_viewer.js').toString().replace(/^import .*;\n/gm,'').replace('__DVIDIA_CLOCK_RESOURCE_MANIFEST__',JSON.stringify(manifest));
source=source.slice(0,source.lastIndexOf('\nmain().catch('));
source+='\nmodel=sceneDoc;meta=motionDoc;validateManifest();validateScene(geometryBytes);validateAttempts();globalThis.check={decode,payload,validateRows,replayKey};';
const geometry=zlib.gunzipSync(Buffer.from(manifest.embedded['geometry.bin'].data,'base64'));
const context=vm.createContext({crypto:crypto.webcrypto,DecompressionStream,Blob,TextDecoder,URL,AbortController,setTimeout,clearTimeout,atob,btoa,
sceneDoc:JSON.parse(read('scene.json')),motionDoc:JSON.parse(read('motion.json')),geometryBytes:new Uint8Array(geometry).buffer});
vm.runInContext(source,context);
(async()=>{for(const index of [0,1,99,100,257]){const d=manifest.cases[index];let p;
if(index===0)p=manifest.first;else{const literal=read(d.script.href).toString();const prefix='globalThis["__dvidiaPiperClockResourcesV1"]('+JSON.stringify(d.id)+',';if(!literal.startsWith(prefix)||!literal.endsWith(');\n'))throw Error('wrapper grammar');p=JSON.parse(literal.slice(prefix.length,-3));}
context.check.payload(p,d);const values=await Promise.all(p.resources.map((r,i)=>context.check.decode(r,d.resources[i])));
context.check.validateRows(context.motionDoc.cases[index],JSON.parse(new TextDecoder().decode(values[0])),values[1]);
}if(context.check.replayKey(context.motionDoc.cases[0])===context.check.replayKey(context.motionDoc.cases[1]))throw Error('cache identity');
process.stdout.write('Python/browser compatibility passed\n');})().catch(e=>{console.error(e);process.exitCode=1});
'''
        checked = subprocess.run(['node', '-e', compatibility, str(output)], check=True, capture_output=True, text=True)
        self.assertIn('compatibility passed', checked.stdout)
        with self.assertRaises(ValueError):
            viewer.export_viewer(self.root, output, self.path, self.revision)

    def test_existing_destination_symlink_and_mutable_revision_stop_before_any_science_access(self):
        output = self.path/'existing'; output.mkdir()
        with patch.object(viewer, '_verified_study') as verify, self.assertRaises(ValueError):
            viewer.export_viewer(self.root, output, self.path, self.revision)
        verify.assert_not_called()
        link = self.path/'link'; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            viewer.export_viewer(link, self.path/'new-viewer', self.path, self.revision)
        with patch.object(viewer, '_verified_study') as verify, self.assertRaises(ValueError):
            viewer.export_viewer(self.root, self.path/'new-viewer', self.path, 'main')
        verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
