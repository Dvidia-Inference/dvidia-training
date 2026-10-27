"""No-network publication contracts for the fixed native-resource Space overlay."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import publish_piper_feedback_native_space as overlay
from test_derive_piper_feedback_native_resources import fixture


DATASET,SPACE,MODEL,FINAL='a'*40,'d'*40,'e'*40,'c'*40


def entry(raw,lfs=False):
    return dict(blob='f'*40,size=len(raw),lfs_sha256=sha256(raw).hexdigest()) if lfs else overlay._entry(raw)


class Operation:
    def __init__(self,**kwargs):self.__dict__.update(kwargs)


class API:
    def __init__(self,tree,contents,native):
        self.head=SPACE;self.trees={SPACE:deepcopy(tree)};self.contents={SPACE:dict(contents)}
        self.histories={SPACE:[SimpleNamespace(commit_id=SPACE,title='completed original publication')]}
        self.native=native;self.calls=[];self.drop_after=False;self.drop_before=False;self.force_lfs=False
    def repo_info(self,repo,repo_type,revision=None):
        return SimpleNamespace(private=False,siblings=[],sdk='static',sha=(revision or FINAL) if repo_type=='dataset' else self.head)
    def list_repo_tree(self,repo,repo_type,revision,recursive):
        rows={n:dict(blob='b'*40,size=r['bytes'],lfs_sha256=r['sha256']) for n,r in self.native.items()} if repo_type=='dataset' else self.trees[revision]
        return [SimpleNamespace(path=n,blob_id=r['blob'],size=r['size'],lfs={'sha256':r['lfs_sha256']} if r['lfs_sha256'] else None) for n,r in rows.items()]
    def list_repo_commits(self,repo,repo_type,revision):return self.histories[revision]
    def create_commit(self,repo,repo_type,parent_commit,operations,commit_message):
        assert repo==overlay.study.SPACE_REPO and repo_type=='space'
        self.calls.append(dict(parent=parent_commit,paths=[o.path_in_repo for o in operations],message=commit_message))
        if parent_commit!=self.head:raise RuntimeError('parent guard')
        if self.drop_before:self.drop_before=False;raise RuntimeError('uncertain before server acceptance')
        tree=deepcopy(self.trees[self.head]);content=dict(self.contents[self.head])
        for op in operations:
            assert op._upload_mode=='regular' and op._should_ignore is False
            content[op.path_in_repo]=op.path_or_fileobj;tree[op.path_in_repo]=entry(op.path_or_fileobj,self.force_lfs)
        revision=f'{len(self.trees):040x}'
        self.trees[revision]=tree;self.contents[revision]=content
        self.histories[revision]=[SimpleNamespace(commit_id=revision,title=commit_message)]+self.histories[self.head]
        self.head=revision
        if self.drop_after:self.drop_after=False;raise RuntimeError('lost accepted reply')
        return SimpleNamespace(oid=revision)
    def download(self,repo,name,kind,revision):return self.contents[revision][name]
    def foreign(self):
        old=self.head;revision='9'*40;self.trees[revision]=dict(self.trees[old]);self.trees[revision]['unrelated.txt']=entry(b'foreign')
        self.contents[revision]=dict(self.contents[old],**{'unrelated.txt':b'foreign'})
        self.histories[revision]=[SimpleNamespace(commit_id=revision,title='foreign change')]+self.histories[old];self.head=revision


class NativeSpaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.original=self.root/'original';self.original.mkdir();files,_=fixture(self.original)
        report=dict(variants={v:dict(scheduled=24,successes=n) for v,n in [('feedback',6),('blind',8),('feedforward',5),('teacher',24),('open_jaw',0)]})
        files.update({'protocol.json':b'{"protocol":"frozen"}','frozen.json':b'{"model":"frozen"}','report.json':overlay.study._encoded(report)})
        for name,raw in files.items():(self.original/name).write_bytes(raw)
        seal=dict(schema=overlay.study.VIEWER_SCHEMA,files={n:overlay.study._receipt(b) for n,b in files.items()})
        seal_raw=overlay.study._encoded(seal);(self.original/'viewer-integrity.json').write_bytes(seal_raw)
        self.original_sha=sha256(seal_raw).hexdigest();self.original_files=dict(files,**{'viewer-integrity.json':seal_raw})
        self.patcher=patch.object(overlay,'ORIGINAL_VIEWER_SHA',self.original_sha);self.patcher.start();self.addCleanup(self.patcher.stop)
        def build(args,check):Path(args[-1]).write_bytes(b'compiled native fixture '+sha256(Path(args[-2]).read_bytes()).digest())
        self.build=patch.object(overlay.native.subprocess,'run',side_effect=build);self.build.start();self.addCleanup(self.build.stop)
        self.viewer=self.root/'viewer';result=overlay.native.derive(self.original,self.viewer,expected_viewer_sha256=self.original_sha,dataset_revision=FINAL)
        self.derived_sha=result['derived_viewer_integrity_sha256']
        roots={'index.html':b'<html><main><h1>DVIDIA</h1></main></html>','README.md':b'---\nsdk: static\n---\n\nDVIDIA research\n',
               '.gitattributes':b'*.bin filter=lfs diff=lfs merge=lfs -text\n*.gz filter=lfs diff=lfs merge=lfs -text\n'}
        self.contents={overlay.PREFIX+n:b for n,b in self.original_files.items()};self.contents.update(roots);self.contents['unrelated.txt']=b'existing'
        self.tree={n:entry(b,n.endswith(('.bin','.gz'))) for n,b in self.contents.items()}
        self.baseline=self.root/'baseline';self.baseline.mkdir()
        for n,b in roots.items():(self.baseline/n).write_bytes(b)
        (self.baseline/'tree.json').write_bytes(overlay.study._encoded(dict(schema=overlay.hosting.BASELINE_SCHEMA,repo_id=overlay.study.SPACE_REPO,repo_type='space',commit=SPACE,tree=self.tree)))
        proof=dict(bytes=1,sha256='1'*64)
        historical=dict(schema=overlay.continuation.SCHEMA,original_plan=dict(bytes=1,sha256='2'*64),original_journal=proof,boundary=proof,
                        implementation=overlay.continuation._implementation())
        self.historical=self.root/'historical';self.historical.mkdir();(self.historical/'implementation').mkdir()
        for n in overlay.continuation.IMPLEMENTATION_FILES:(self.historical/'implementation'/n).write_bytes((overlay.ROOT/n).read_bytes())
        self.continuation_plan=self.historical/'continuation-plan.json';self.continuation_plan.write_bytes(overlay.study._encoded(historical))
        receipt=dict(schema=overlay.study.SCHEMA+'/receipt',anonymous_readback=True,physical_robot_ready=False,source_commit='1'*40,plan_sha256='2'*64,
            source_integrity_sha256='3'*64,bindings={k+'_sha256':sha256(files[k+'.json']).hexdigest() for k in ('protocol','report','frozen')},
            repositories=[dict(repo_type=k,repo_id=overlay.study.REPOSITORIES[k],commit=c,verified_files=1) for k,c in [('model',MODEL),('dataset',DATASET),('space',SPACE)]],
            continuation=dict(schema=overlay.continuation.SCHEMA,plan=overlay.study._receipt(self.continuation_plan.read_bytes()),immutable_original_journal=proof,boundary=proof,journal=proof,
                implementation=overlay.continuation._implementation(),history={k:dict(original_prefix_commits=[c],continuation_commits=[]) for k,c in [('model',MODEL),('dataset',DATASET),('space',SPACE)]}))
        self.receipt=self.root/'continuation.json';self.receipt.write_bytes(overlay.study._encoded(receipt))
        self.card=self.root/'metadata.json';self.card.write_bytes(overlay.study._encoded(dict(schema=overlay.metadata.SCHEMA+'/receipt',repo_id=overlay.study.DATASET_REPO,
            parent_commit=DATASET,commit=FINAL,anonymous_before_and_after_readback=True,scientific_seal_unchanged=True,physical_robot_ready=False,
            original_scientific_integrity_sha256='3'*64,changed_paths=sorted(overlay.metadata.ALLOWED))))
        self.output=self.root/'staged'
    def stage(self):
        return overlay.stage(self.original,self.viewer,self.receipt,FINAL,SPACE,self.baseline,self.output,
            expected_viewer_sha256=self.derived_sha,continuation_plan=self.continuation_plan,dataset_metadata_receipt=self.card)
    def api(self,plan):return API(self.tree,self.contents,plan['native_files'])
    def publish(self,api):return overlay.publish(self.output/overlay.PLAN,api=api,downloader=api.download,operation_cls=Operation)
    def test_offline_exact_projection_motion_pin_and_bounded_atomic_activation(self):
        before={p.relative_to(self.original).as_posix():p.read_bytes() for p in self.original.rglob('*') if p.is_file()}
        plan=self.stage();checked,tree,payload,_=overlay.checked_plan(self.output/overlay.PLAN)
        self.assertEqual(plan,checked);self.assertEqual(tree,self.tree)
        self.assertEqual(set(payload)&set(tree),overlay.ACTIVATION)
        self.assertEqual(plan['groups'][0]['kind'],'regular-git-preflight');self.assertEqual(plan['groups'][-1]['kind'],'activation')
        self.assertEqual(set(plan['groups'][-1]['paths']),overlay.ACTIVATION)
        for group in plan['groups']:self.assertLessEqual(group['bytes'],group['cap']);self.assertLessEqual(len(group['paths']),25)
        self.assertEqual(plan['caps']['activation_commit_bytes'],16*1024*1024)
        for p in self.viewer.rglob('*'):
            if p.is_file():
                name=p.relative_to(self.viewer).as_posix()
                self.assertEqual(payload.get(overlay.PREFIX+name,before.get(name)),p.read_bytes())
        self.assertEqual(before,{p.relative_to(self.original).as_posix():p.read_bytes() for p in self.original.rglob('*') if p.is_file()})
    def test_publish_preserves_full_baseline_and_activates_last_after_regular_git_probe(self):
        plan=self.stage();api=self.api(plan);receipt=self.publish(api)
        self.assertEqual(len(api.calls),len(plan['groups']));self.assertEqual(set(api.calls[-1]['paths']),overlay.ACTIVATION)
        self.assertEqual(api.calls[0]['paths'],[plan['preflight_bundle_path']])
        for n,r in self.tree.items():
            if n not in overlay.ACTIVATION:self.assertEqual(api.trees[api.head][n],r)
        self.assertTrue(receipt['all_overlay_objects_regular_git']);self.assertTrue(receipt['anonymous_full_overlay_readback'])
        self.assertEqual(api.contents[api.head][overlay.PREFIX+'viewer-integrity.json'],(self.viewer/'viewer-integrity.json').read_bytes())
        self.assertIn(b'learner without live feedback 8/24',api.contents[api.head]['README.md'])
        with self.assertRaises(overlay.PublicationError):self.publish(api)
    def test_lost_successful_reply_reconciles_exact_child_without_duplicate_commit(self):
        plan=self.stage();api=self.api(plan);api.drop_after=True
        with self.assertRaises(overlay.PublicationError):self.publish(api)
        self.assertIsNotNone(json.loads((self.output/overlay.JOURNAL).read_bytes())['pending'])
        receipt=self.publish(api);self.assertEqual(len(api.calls),len(plan['groups']))
        self.assertTrue(receipt['verified_commits'][0]['reconciled'])
    def test_lost_unaccepted_reply_retries_only_identical_parent_guarded_group(self):
        plan=self.stage();api=self.api(plan);api.drop_before=True
        with self.assertRaises(overlay.PublicationError):self.publish(api)
        self.publish(api);self.assertEqual(api.calls[0],api.calls[1]);self.assertEqual(len(api.calls),len(plan['groups'])+1)
    def test_foreign_move_after_unknown_reply_is_not_reconciled_or_retried(self):
        plan=self.stage();api=self.api(plan);api.drop_after=True
        with self.assertRaises(overlay.PublicationError):self.publish(api)
        api.foreign()
        with self.assertRaises(overlay.PublicationError):self.publish(api)
        self.assertEqual(len(api.calls),1)
    def test_forced_lfs_preflight_stops_before_any_activation(self):
        plan=self.stage();api=self.api(plan);api.force_lfs=True
        with self.assertRaises(overlay.PublicationError):self.publish(api)
        self.assertEqual(len(api.calls),1);self.assertTrue(set(api.calls[0]['paths']).isdisjoint(overlay.ACTIVATION))
    def test_anonymous_readback_failure_keeps_pending_and_resumes_exact_child(self):
        plan=self.stage();api=self.api(plan);download=api.download;failed=False
        def flaky(repo,name,kind,revision):
            nonlocal failed
            if name==plan['preflight_bundle_path'] and not failed:
                failed=True;raise RuntimeError('anonymous download failed')
            return download(repo,name,kind,revision)
        with self.assertRaises(RuntimeError):overlay.publish(self.output/overlay.PLAN,api=api,downloader=flaky,operation_cls=Operation)
        self.assertEqual(len(api.calls),1);self.assertIsNotNone(json.loads((self.output/overlay.JOURNAL).read_bytes())['pending'])
        receipt=self.publish(api);self.assertEqual(len(api.calls),len(plan['groups']));self.assertTrue(receipt['verified_commits'][0]['reconciled'])
    def test_final_readback_failure_can_finish_without_repeating_any_commit(self):
        plan=self.stage();api=self.api(plan);download=api.download;probe_reads=0
        def flaky(repo,name,kind,revision):
            nonlocal probe_reads
            if name==plan['preflight_bundle_path']:
                probe_reads+=1
                if probe_reads==2:raise RuntimeError('final readback failed')
            return download(repo,name,kind,revision)
        with self.assertRaises(RuntimeError):overlay.publish(self.output/overlay.PLAN,api=api,downloader=flaky,operation_cls=Operation)
        self.assertEqual(len(api.calls),len(plan['groups']));self.assertFalse((self.output/overlay.RECEIPT).exists())
        self.publish(api);self.assertEqual(len(api.calls),len(plan['groups']))
    def test_head_change_during_readback_prevents_group_acknowledgment(self):
        plan=self.stage();api=self.api(plan);download=api.download
        def moving(repo,name,kind,revision):
            raw=download(repo,name,kind,revision)
            if name==plan['preflight_bundle_path']:api.foreign()
            return raw
        with self.assertRaises(overlay.PublicationError):overlay.publish(self.output/overlay.PLAN,api=api,downloader=moving,operation_cls=Operation)
        self.assertEqual(len(api.calls),1);journal=json.loads((self.output/overlay.JOURNAL).read_bytes())
        self.assertEqual(journal['groups'],[]);self.assertIsNotNone(journal['pending'])
    def test_unrelated_head_tree_or_native_dataset_mutation_prevents_writes(self):
        for kind in ('head','tree','dataset'):
            with self.subTest(kind=kind):
                if self.output.exists():
                    import shutil;shutil.rmtree(self.output)
                plan=self.stage();api=self.api(plan)
                if kind=='head':api.foreign()
                elif kind=='tree':api.trees[SPACE]['unrelated.txt']=entry(b'changed')
                else:api.native=deepcopy(api.native);next(iter(api.native.values()))['sha256']='0'*64
                with self.assertRaises(overlay.PublicationError):self.publish(api)
                self.assertEqual(api.calls,[])
    def test_rehashed_payload_extra_path_and_plan_group_tampering_are_rejected(self):
        self.stage();path=self.output/overlay.PLAN;original=path.read_bytes();plan=json.loads(original)
        target=self.output/'payload'/overlay.PREFIX/'motion.json';before=target.read_bytes();target.write_bytes(before+b' ')
        row=next(r for r in plan['files'] if r['path']==overlay.PREFIX+'motion.json');row['after']=overlay.study._receipt(target.read_bytes());row['git_entry']=overlay._entry(target.read_bytes())
        path.write_bytes(overlay.study._encoded(plan))
        with self.assertRaises(overlay.PublicationError):overlay.checked_plan(path)
        target.write_bytes(before);path.write_bytes(original);(self.output/'payload'/'extra.txt').write_bytes(b'extra')
        with self.assertRaises(overlay.PublicationError):overlay.checked_plan(path)
        (self.output/'payload'/'extra.txt').unlink();plan=json.loads(original);plan['groups'][-1]['paths'].append('unrelated.txt');path.write_bytes(overlay.study._encoded(plan))
        with self.assertRaises(overlay.PublicationError):overlay.checked_plan(path)
    def test_attributes_and_final_receipt_bindings_fail_closed(self):
        with self.assertRaises(overlay.PublicationError):overlay._attributes(b'*.js filter=lfs\n',dict(self.tree,**{'.gitattributes':entry(b'*.js filter=lfs\n')}),{overlay.PREFIX+'x.js':b'x'})
        with self.assertRaises(overlay.PublicationError):overlay._attributes(self.contents['.gitattributes'],dict(self.tree,**{overlay.PREFIX+'.gitattributes':entry(b'')}),{overlay.PREFIX+'x.js':b'x'})
        card=json.loads(self.card.read_bytes());card['parent_commit']='0'*40;self.card.write_bytes(overlay.study._encoded(card))
        with self.assertRaises(overlay.PublicationError):self.stage()
    def test_pending_journal_and_implementation_snapshot_tampering_stop_before_retry(self):
        plan=self.stage();api=self.api(plan);api.drop_after=True
        with self.assertRaises(overlay.PublicationError):self.publish(api)
        journal_path=self.output/overlay.JOURNAL;raw=journal_path.read_bytes();journal=json.loads(raw);journal['pending']['message']='different request'
        journal_path.write_bytes(overlay.study._encoded(journal))
        with self.assertRaises(overlay.PublicationError):self.publish(api)
        self.assertEqual(len(api.calls),1);journal_path.write_bytes(raw)
        snapshot=self.output/'implementation'/'publish_piper_feedback_native_space.py';snapshot.write_bytes(snapshot.read_bytes()+b'\n')
        with self.assertRaises(overlay.PublicationError):self.publish(api)
        self.assertEqual(len(api.calls),1)
    def test_historical_continuation_stays_valid_after_legitimate_current_metadata_source_change(self):
        current=self.root/'current-runtime';current.mkdir()
        for n in overlay.IMPLEMENTATION:(current/n).write_bytes((overlay.ROOT/n).read_bytes())
        changed=current/'publish_piper_feedback_dataset_metadata.py';changed.write_bytes(changed.read_bytes()+b'\n# reviewed post-publication metadata inventory repair\n')
        with patch.object(overlay,'ROOT',current):
            plan=self.stage();overlay.checked_plan(self.output/overlay.PLAN)
        old=json.loads((self.output/'continuation-plan.json').read_bytes())['implementation']['files']
        self.assertNotEqual(old['publish_piper_feedback_dataset_metadata.py'],plan['implementation']['publish_piper_feedback_dataset_metadata.py'])
        self.assertEqual((self.output/'historical-implementation'/'publish_piper_feedback_dataset_metadata.py').read_bytes(),
                         (self.historical/'implementation'/'publish_piper_feedback_dataset_metadata.py').read_bytes())
    def test_historical_plan_and_snapshot_tampering_rejected_even_after_staging(self):
        self.stage();plan=self.output/overlay.PLAN
        historical=self.output/'continuation-plan.json';raw=historical.read_bytes();historical.write_bytes(raw+b' ')
        with self.assertRaises(overlay.PublicationError):overlay.checked_plan(plan)
        historical.write_bytes(raw);snapshot=self.output/'historical-implementation'/'publish_piper_feedback_dataset_metadata.py'
        snapshot.write_bytes(snapshot.read_bytes()+b'\n')
        with self.assertRaises(overlay.PublicationError):overlay.checked_plan(plan)
    def test_activation_has_new_fixed_16mib_cap_without_relaxing_asset_or_file_counts(self):
        probe='probe.js';payload={n:b'x' for n in overlay.ACTIVATION};payload[probe]=b'x'
        names=sorted(overlay.ACTIVATION);payload[names[0]]=b'x'*(5*1024*1024);payload[names[1]]=b'x'*(5*1024*1024)
        groups=overlay._groups(payload,probe);self.assertGreater(groups[-1]['bytes'],8*1024*1024);self.assertLessEqual(groups[-1]['bytes'],16*1024*1024)
        payload[names[2]]=b'x'*(7*1024*1024)
        with self.assertRaises(overlay.PublicationError):overlay._groups(payload,probe)


if __name__=='__main__':unittest.main()
