"""Card bytes and publication bindings, with synthetic facts and no native work."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_camera_action_v01 as package
import render_camera_action_card as renderer
import publish_camera_action_v01 as publisher
from huggingface.camera_action_model_cards import validate_receipt, RECEIPT_FILE


class ModelCardBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.study = self.root/'study'
        self.cards = self.root/'cards'; self.commit = 'a'*40
        self.summary = dict(controllers=[], training=[], hardware=[])
        self.report = dict(status='blocked', qualification='research_candidate', qualified=False,
            physical_robot_ready=False, selected_stage=None, gates={})
        self.state = dict(report=self.report, status='blocked', qualified=False, qualification='research_candidate')
        self.protocol = {'fixture': 'synthetic pure publication test'}
        for name, value in [('protocol.json', self.protocol), ('registration.json', {}), ('report.json', self.report)]:
            package.write(self.study/name, value)
        self.receipts = {}; self.models = []; self.records = []
        fitted = {}
        for variant in ('full', 'blind'):
            name = f'models/initial/{variant}/model.json'
            model = dict(blind=variant == 'blind', training=dict(stage='initial', sample_count=1))
            package.write(self.study/name, model)
            identity = package.identity(self.study/name)
            record = dict(path=name, model_sha256=identity['sha256'], model_bytes=identity['bytes'],
                training=model['training'], training_episodes=['synthetic'], development_episodes=[])
            package.write(self.study/f'models/initial/{variant}/receipt.json', record)
            package.write(self.study/f'models/initial/{variant}/started.json', {'synthetic': True})
            fitted[variant] = record; self.models.append(name)
            self.records.append(dict(stage='initial', variant=variant, path=name,
                identity=identity, selected=False, training=record, model=model))
        self.receipts['train-initial'] = dict(status='complete', models=fitted)
        for suffix in ('.json', '.integrity.json', '.started.json'):
            package.write(self.study/('stages/train-initial'+suffix), self.receipts['train-initial'])
        self.metadata = {p.relative_to(self.study).as_posix(): package.identity(p) for p in package.files(self.study)}
        self.bindings = dict(self.metadata)
        self.data = dict(study=self.study, source_commit=self.commit, state=self.state,
            protocol=self.protocol, report=self.report, receipts=self.receipts, summary=self.summary,
            models=self.records, hardware=None, audit=None, phase=None, study_bindings=self.bindings)
        self.make_cards()

    def make_cards(self):
        texts = {'README.md': renderer.dataset_card(self.data)}
        texts.update({f"model-cards/{r['stage']}/{r['variant']}/README.md": renderer.model_card(self.data, r)
                      for r in self.records})
        self.cards.mkdir(exist_ok=True)
        (self.cards/'summary.json').write_bytes(package.canonical(self.summary))
        for name, text in texts.items():
            path = self.cards/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
        self.receipt = dict(schema=renderer.SCHEMA, source_commit=self.commit,
            protocol_sha256=self.metadata['protocol.json']['sha256'], report_sha256=self.metadata['report.json']['sha256'],
            status=self.report['status'], qualification=self.report['qualification'], qualified=False,
            full_study_audited=False, physical_robot_ready=False, summary_reconstructed=True,
            physics_executed=False, model_executed=False,
            source_inputs={'summary.json': package.identity(self.cards/'summary.json'),
                'generator.py': package.identity(Path(renderer.__file__)),
                'summarize_results.py': package.identity(renderer.SUMMARY_SOURCE)},
            study_bindings=self.bindings, files={name: package.identity(self.cards/name) for name in texts},
            model_cards=[dict(card=f"model-cards/{r['stage']}/{r['variant']}/README.md", model=r['path'],
                model_sha256=r['identity']['sha256'], selected=r['selected']) for r in self.records])
        self.save_receipt()

    def save_receipt(self):
        (self.cards/'card-receipt.json').write_bytes(package.canonical(self.receipt)+b'\n')

    def check(self, *, audit='unused'):
        # Model schema is independently tested by the learner suite. Here the
        # mock admits visibly synthetic weights to isolate publication binding.
        with patch('dvidia_training.camera_action_model.validate_model', side_effect=lambda value:value), \
                patch.object(renderer, 'recompute_summary', return_value=self.summary):
            return package.checked_model_cards(self.cards, study=self.study, card=self.cards/'README.md',
                source_commit=self.commit, state=self.state, protocol=self.protocol,
                receipts=self.receipts, metadata=self.metadata, models=list(reversed(self.models)), audit=audit)

    def test_exact_generated_pair_and_correct_relative_download_links(self):
        result = self.check()
        self.assertEqual(len(result['cards']), 2)
        self.assertEqual(set(result['files']), set(result['cards'])|{RECEIPT_FILE})
        for variant in ('full', 'blind'):
            text = (self.cards/f'model-cards/initial/{variant}/README.md').read_text()
            self.assertIn(f'[model.json](../../../models/initial/{variant}/model.json)', text)
        self.assertIn('[Read card](model-cards/initial/full/README.md)', (self.cards/'README.md').read_text())

    def test_text_tamper_is_refused_even_after_receipt_rehash(self):
        name = 'model-cards/initial/full/README.md'
        (self.cards/name).write_text('Fabricated perfect hardware success.')
        self.receipt['files'][name] = package.identity(self.cards/name); self.save_receipt()
        with self.assertRaisesRegex(ValueError, 'fresh sealed-evidence rendering'): self.check()

    def test_selected_model_hash_report_protocol_commit_and_roster_tampering_refuse(self):
        original = deepcopy(self.receipt)
        for field in ('selected', 'model_sha256', 'report_sha256', 'protocol_sha256', 'source_commit', 'duplicate'):
            with self.subTest(field=field):
                self.receipt = deepcopy(original)
                if field == 'selected': self.receipt['model_cards'][0][field] = True
                elif field == 'model_sha256': self.receipt['model_cards'][0][field] = '0'*64
                elif field == 'duplicate': self.receipt['model_cards'].append(deepcopy(self.receipt['model_cards'][0]))
                else: self.receipt[field] = '0'*(40 if field=='source_commit' else 64)
                self.save_receipt()
                with self.assertRaises(ValueError): self.check()

    def test_extra_file_empty_directory_symlink_and_secret_are_refused(self):
        extra = self.cards/'extra.md'; extra.write_text('arbitrary')
        with self.assertRaisesRegex(ValueError, 'arbitrary files'): self.check()
        extra.unlink(); extra.mkdir()
        with self.assertRaisesRegex(ValueError, 'arbitrary empty'): self.check()
        extra.rmdir(); extra.symlink_to(self.cards/'README.md')
        with self.assertRaisesRegex(ValueError, 'symbolic'): self.check()
        extra.unlink()
        path=self.cards/'model-cards/initial/full/README.md'; path.write_text('hf_'+'A'*32)
        with self.assertRaisesRegex(ValueError, 'Credential-shaped'): self.check()

    def test_changed_generator_binding_cannot_reuse_stale_cards(self):
        self.receipt['source_inputs']['generator.py']['sha256'] = '0'*64; self.save_receipt()
        with self.assertRaisesRegex(ValueError, 'generator or auxiliary'): self.check()

    def test_summary_input_identity_and_reconstructed_values_are_both_bound(self):
        original = deepcopy(self.receipt)
        self.receipt['source_inputs']['summary.json']['sha256'] = '0'*64; self.save_receipt()
        with self.assertRaisesRegex(ValueError, 'generator or auxiliary'): self.check()
        self.receipt = original
        (self.cards/'summary.json').write_bytes(package.canonical({**self.summary, 'invented_latency': 0.00001}))
        self.receipt['source_inputs']['summary.json'] = package.identity(self.cards/'summary.json'); self.save_receipt()
        with self.assertRaisesRegex(ValueError, 'Retained card summary'): self.check()

    def test_standalone_diagnostic_audit_is_supported_and_bound_to_actual_input(self):
        audit = self.root/'audit.json'; package.write(audit, {'fixture': 'diagnostic audit'})
        self.receipt['source_inputs']['independent-audit.json'] = package.identity(audit); self.save_receipt()
        self.check(audit=audit)
        audit.write_bytes(package.canonical({'fixture': 'different audit'}))
        with self.assertRaisesRegex(ValueError, 'generator or auxiliary'): self.check(audit=audit)

    def test_phase_helper_cannot_be_claimed_without_coverage_and_audit(self):
        self.receipt['source_inputs']['diagnostic_phase_coverage.py'] = package.identity(renderer.PHASE_SOURCE)
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, 'phase provenance'): self.check()

    def test_source_archive_must_contain_exact_bound_card_helpers(self):
        import io
        import tarfile
        source = self.root/'source.tar.gz'
        helpers = {'publishing/render_camera_action_card.py': Path(renderer.__file__).read_bytes(),
            'benchmarks/camera-action-v01/summarize_results.py': renderer.SUMMARY_SOURCE.read_bytes()}
        def archive(values):
            with tarfile.open(source, 'w:gz') as output:
                for name, value in values.items():
                    info = tarfile.TarInfo('fixture/'+name); info.size = len(value)
                    output.addfile(info, io.BytesIO(value))
        archive(helpers); package.verify_card_source_archive(source, self.receipt)
        name = 'publishing/render_camera_action_card.py'
        archive({**helpers, name: b'x'*len(helpers[name])})
        with self.assertRaisesRegex(ValueError, 'differs from the bound card'): package.verify_card_source_archive(source, self.receipt)
        archive({name: helpers[name]})
        with self.assertRaisesRegex(ValueError, 'each card implementation'): package.verify_card_source_archive(source, self.receipt)

    def test_preparation_copies_generated_pair_receipt_and_models_into_public_payload(self):
        import io
        import tarfile
        import zipfile
        from dvidia_training import camera_action_study as workflow
        runtime=b'"""Synthetic runtime fixture; never executed."""\n'
        self.protocol['source_sha256']={'camera_action_study.py':sha256(runtime).hexdigest()}
        (self.study/'protocol.json').write_bytes(package.canonical(self.protocol))
        rows=[]
        for index in range(4):
            name=f'diagnostic-{index}-teacher';prefix='episodes/diagnostic/'+name
            package.write(self.study/prefix/'started.json',{'fixture':True})
            package.write(self.study/prefix/'rollout.json',{'status':'complete','fixture':True})
            package.write(self.study/prefix/'integrity.json',dict(schema='dvidia.camera-action-rollout/v2/integrity',
                files={n:package.identity(self.study/prefix/n) for n in ('started.json','rollout.json')}))
            rows.append(dict(id=name,case=dict(id=name,group_id=f'dev-{index}',split='development'),
                path=prefix,controller='teacher',negative_control=None,status='complete',strict_success=False,
                receipt_sha256=package.identity(self.study/prefix/'rollout.json')['sha256'],
                integrity_sha256=package.identity(self.study/prefix/'integrity.json')['sha256']))
        self.receipts['diagnostic']=dict(status='blocked',rows=rows)
        for suffix in ('.json','.integrity.json','.started.json'):
            package.write(self.study/('stages/diagnostic'+suffix),self.receipts['diagnostic'])
        self.metadata={p.relative_to(self.study).as_posix():package.identity(p) for p in package.files(self.study)
                       if not p.is_relative_to(self.study/'episodes')}
        self.bindings=dict(self.metadata);self.data['study_bindings']=self.bindings
        self.make_cards()
        audit=self.root/'audit.json'
        package.write(audit,dict(schema='dvidia.camera-action-diagnostic-audit/v1',passed=True,errors=[],
            attempts_verified=4,physics_executed=False,estimator_executed=False,
            results=[dict(case_id=r['case']['id'],strict_success=False) for r in rows]))
        wheel=self.root/'fixture.whl';source=self.root/'source.tar.gz'
        with zipfile.ZipFile(wheel,'w') as archive:
            archive.writestr('dvidia_training/camera_action_study.py',runtime)
        with tarfile.open(source,'w:gz',format=tarfile.PAX_FORMAT,pax_headers={'comment':self.commit}) as archive:
            info=tarfile.TarInfo('fixture/src/dvidia_training/camera_action_study.py');info.size=len(runtime)
            archive.addfile(info,io.BytesIO(runtime))
            for name, path in [('publishing/render_camera_action_card.py', Path(renderer.__file__)),
                               ('benchmarks/camera-action-v01/summarize_results.py', renderer.SUMMARY_SOURCE)]:
                value=path.read_bytes();info=tarfile.TarInfo('fixture/'+name);info.size=len(value)
                archive.addfile(info,io.BytesIO(value))
        output=self.root/'prepared'
        with patch.object(workflow,'inspect',return_value=self.state), \
                patch.object(workflow,'_stage',side_effect=lambda _root,name:self.receipts.get(name)), \
                patch.object(workflow,'_require',side_effect=lambda _root,name:self.receipts[name]), \
                patch.object(renderer,'recompute_summary',return_value=self.summary), \
                patch('dvidia_training.camera_action_model.validate_model',side_effect=lambda value:value):
            result=package.prepare(self.study,output,audit=audit,card=self.cards/'README.md',wheel=wheel,
                source_archive=source,source_commit=self.commit,model_cards=self.cards)
        self.assertEqual(result['trained_models'],2)
        manifest=package.read(output/'manifest.json')
        self.assertEqual(len(manifest['model_card_files']),2)
        self.assertEqual(manifest['model_card_receipt_file'],RECEIPT_FILE)
        for name in manifest['model_card_files']:
            self.assertEqual((output/name).read_bytes(),(self.cards/name).read_bytes())
        self.assertEqual((output/RECEIPT_FILE).read_bytes(),(self.cards/'card-receipt.json').read_bytes())
        self.assertEqual(len(publisher.files(output)),len(manifest['files'])+2)

    def test_blocked_without_models_has_empty_cards_and_a_bound_receipt(self):
        import shutil
        shutil.rmtree(self.study/'models'); shutil.rmtree(self.study/'stages'); shutil.rmtree(self.cards)
        self.models=[]; self.records=[]; self.receipts={}
        self.metadata={p.name:package.identity(p) for p in package.files(self.study)}; self.bindings=dict(self.metadata)
        self.data.update(models=[],receipts={},study_bindings=self.bindings)
        self.make_cards(); result=self.check()
        self.assertEqual(result['cards'],[]);self.assertEqual(set(result['files']),{RECEIPT_FILE})

    def make_publication(self):
        root=self.root/'publication';root.mkdir()
        for name in publisher.REQUIRED-{'manifest.json','SHA256SUMS'}:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'Synthetic publication fixture.')
        (root/'README.md').write_bytes((self.cards/'README.md').read_bytes())
        for name in ('protocol.json','registration.json','report.json',*self.models):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((self.study/name).read_bytes())
        for name in self.receipt['files']:
            if name=='README.md':continue
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((self.cards/name).read_bytes())
        (root/RECEIPT_FILE).write_bytes((self.cards/'card-receipt.json').read_bytes())
        for name in ('source/fixture.whl','source/fixture.tar.gz'):
            path=root/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'fixture')
        self.public=root
        self.manifest=dict(schema=publisher.EVIDENCE_SCHEMA,repo_id=publisher.REPO,repo_type='dataset',
            status='blocked',qualification='research_candidate',qualified=False,original_footage_included=False,
            physical_robot_ready=False,source_commit=self.commit,episodes=[],episode_count=0,
            trained_model_files=self.models,model_card_receipt_file=RECEIPT_FILE,
            model_card_files=sorted(r['card'] for r in self.receipt['model_cards']))
        self.seal_publication();return root

    def seal_publication(self):
        inventory={p.relative_to(self.public).as_posix():publisher.identity(p) for p in self.public.rglob('*')
                   if p.is_file() and p.name not in ('manifest.json','SHA256SUMS')}
        self.manifest['files']=inventory
        (self.public/'manifest.json').write_bytes(package.canonical(self.manifest))
        inventory=dict(inventory,**{'manifest.json':publisher.identity(self.public/'manifest.json')})
        (self.public/'SHA256SUMS').write_text(''.join(f'{r["sha256"]}  {n}\n' for n,r in sorted(inventory.items())))

    def test_publisher_accepts_only_exact_receipted_cards_without_credentials(self):
        public=self.make_publication()
        with patch.object(publisher,'client',side_effect=AssertionError('credential access')):
            result=publisher.files(public)
        self.assertIn(RECEIPT_FILE,{r['path'] for r in result})
        self.manifest['model_card_files']=[];self.seal_publication()
        with self.assertRaisesRegex(publisher.PublicationError,'files or provenance'):publisher.files(public)

    def test_publisher_rejects_changed_card_even_with_resealed_payload_manifest(self):
        public=self.make_publication()
        (public/'model-cards/initial/full/README.md').write_text('Changed after review.')
        self.seal_publication()
        with self.assertRaisesRegex(publisher.PublicationError,'files or provenance'):publisher.files(public)

    def test_publisher_refuses_missing_receipt_selection_and_arbitrary_paths(self):
        public=self.make_publication()
        self.manifest['model_card_receipt_file']=None;self.seal_publication()
        with self.assertRaisesRegex(publisher.PublicationError,'explicit receipt'):publisher.files(public)
        self.manifest['model_card_receipt_file']=RECEIPT_FILE
        value=deepcopy(self.receipt);value['model_cards'][0]['selected']=True
        (public/RECEIPT_FILE).write_bytes(package.canonical(value));self.seal_publication()
        with self.assertRaisesRegex(publisher.PublicationError,'receipt differs'):publisher.files(public)
        self.assertFalse(publisher.allowed('model-cards/initial/full/extra.py'))
        self.assertFalse(publisher.allowed('model-cards/foreign/full/README.md'))


if __name__=='__main__':unittest.main()
