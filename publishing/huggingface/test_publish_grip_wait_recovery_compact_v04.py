"""No-network compact publication guards; synthetic transport fixtures only."""
from contextlib import redirect_stderr
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import publish_grip_wait_recovery_compact_v04 as pub


class NotFound(Exception):pass
class Operation:
    def __init__(self,**kwargs):self.__dict__.update(kwargs)


class FakeAPI:
    def __init__(self):
        self.head=None;self.contents={};self.histories={};self.number=100
        self.calls=[];self.created=0;self.member=True
        self.fail_before=self.fail_after=self.fail_creation=False
        self.token='mock-only-publisher-credential-value'
        self.downloads=[]

    def add(self,files):
        self.number+=1;self.head=f'{self.number:040x}'
        self.contents[self.head]=deepcopy(files)
        self.histories[self.head]=[SimpleNamespace(commit_id=self.head,title='baseline')]

    def whoami(self):return {'orgs':[{'name':'Dvidia'}] if self.member else []}

    def repo_info(self,repo,*,repo_type,revision=None,files_metadata,token):
        assert repo==pub.REPO and repo_type=='dataset' and token is False and files_metadata is True
        if self.head is None:raise NotFound()
        oid=revision or self.head
        siblings=[SimpleNamespace(rfilename=name,size=len(body),blob_id=pub.git_identity(body)['blob'],lfs=None)
                  for name,body in self.contents[oid].items()]
        return SimpleNamespace(sha=oid,private=False,siblings=siblings)

    def create_repo(self,repo,*,repo_type,private,exist_ok):
        assert repo==pub.REPO and repo_type=='dataset' and private is False and exist_ok is False and self.head is None
        self.created+=1;self.add({'.gitattributes':b'*.zip filter=lfs diff=lfs merge=lfs -text\n'})
        if self.fail_creation:
            self.fail_creation=False;raise RuntimeError(self.token+' uncertain creation')

    def create_commit(self,repo,*,repo_type,parent_commit,operations,commit_message):
        assert repo==pub.REPO and repo_type=='dataset' and parent_commit==self.head
        self.calls.append(dict(parent=parent_commit,operations=operations))
        if self.fail_before:
            self.fail_before=False;raise RuntimeError(self.token+' uncertain unaccepted write')
        files=deepcopy(self.contents[self.head]);parent=self.head
        for operation in operations:
            assert operation._upload_mode=='regular' and operation._should_ignore is False
            assert isinstance(operation.path_or_fileobj,bytes)
            assert operation.path_in_repo not in files
            files[operation.path_in_repo]=operation.path_or_fileobj
        self.number+=1;self.head=f'{self.number:040x}';self.contents[self.head]=files
        self.histories[self.head]=[SimpleNamespace(commit_id=self.head,title=commit_message),*self.histories[parent]]
        if self.fail_after:
            self.fail_after=False;raise RuntimeError(self.token+' uncertain accepted write')
        return SimpleNamespace(oid=self.head)

    def list_repo_commits(self,repo,*,repo_type,revision,token):
        assert repo==pub.REPO and repo_type=='dataset' and token is False
        return self.histories[revision]

    def move(self):
        parent=self.head;self.number+=1;self.head=f'{self.number:040x}'
        self.contents[self.head]=deepcopy(self.contents[parent])
        self.histories[self.head]=[SimpleNamespace(commit_id=self.head,title='foreign commit'),*self.histories[parent]]

    def download(self,repo,name,*,repo_type,revision,token,endpoint,cache_dir,force_download):
        assert repo==pub.REPO and repo_type=='dataset' and token is False
        assert endpoint=='https://huggingface.co' and force_download is True
        self.downloads.append((revision,name))
        path=Path(cache_dir)/revision/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(self.contents[revision][name]);return str(path)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class CompactPublicationTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.home=Path(temp.name);self.payload=self.home/'payload';self.output=self.home/'plan';self.path=self.output/'plan.json'
        self.api=FakeAPI();self.payload.mkdir()
        source_body=b'# Synthetic source snapshot. No real benchmark is simulated here.\n'
        source_digest=sha256(source_body).hexdigest()
        sources={}
        for name in ('piper_grip_wait_v04.py','piper_grip_wait_v04_benchmark.py','recover_piper_grip_wait_v04.py',
                     'audit_piper_grip_wait_v04.py','audit_piper_grip_clock_v03.py'):
            destination='source/'+name;(self.payload/destination).parent.mkdir(exist_ok=True)
            (self.payload/destination).write_bytes(source_body)
            sources[destination]=dict(source_path='synthetic/'+name,**pub.digest(source_body))
        rows=[]
        for variant,count in pub.SUCCESSES.items():
            successful=set([i for i in range(128) if i!=18][:count])
            for i in range(128):
                unknown=i==18
                row=dict(case_id='confirmation-xy04-g1-nominal' if unknown else f'case-{i}',variant=variant,
                    interrupted_unevaluable=unknown,valid=not unknown,success=i in successful,
                    raw_strict_success=None if unknown else i in successful,native_summary=None if unknown else {},
                    post_grip=None if unknown else {},success_is_failure_imputation=unknown)
                rows.append(row)
        report=dict(rows=rows,valid_scored_attempts=508,all_512_valid=False,qualified_robot_skill=False,physical_robot_ready=False,
                    driver_sha256=source_digest,independent_auditor_sha256=source_digest)
        fixtures={'README.md':b'Synthetic fixture, not a result.\n',
            'report.json':json.dumps(report).encode(),
            'audit-summary.json':json.dumps(dict(dependency_auditor_sha256=source_digest)).encode(),
            'interruptions.json':b'{"synthetic":true}\n',
            'source-lineage.json':json.dumps(dict(frozen_runtime_sha256={
                'piper_grip_wait_v04.py':source_digest,'piper_grip_wait_v04_benchmark.py':source_digest})).encode()}
        for name,body in fixtures.items():
            destination=self.payload/pub.PREFIX/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(body)
        handle=patch.object(pub,'COMPACT',{name:sha256(body).hexdigest() for name,body in fixtures.items()})
        handle.start();self.addCleanup(handle.stop)
        paper=b'# Synthetic paper fixture\n';(self.payload/pub.PAPER).parent.mkdir(parents=True);(self.payload/pub.PAPER).write_bytes(paper)
        handle=patch.object(pub,'PAPER_SHA',sha256(paper).hexdigest());handle.start();self.addCleanup(handle.stop)
        for name,body in {'README.md':b'---\nlicense: mit\n---\nCompact simulation summaries; no raw archive.\n',
                          'LICENSE':b'MIT License\n','NOTICE.txt':b'No weights. Full raw native evidence not uploaded.\n',
                          'source/README.md':b'Source subset, not a standalone runtime.\n'}.items():
            (self.payload/name).write_bytes(body)
        scope=dict(schema='dvidia.piper-grip-wait-compact-release/v1',repo_id=pub.REPO,repo_type='dataset',
            scheduled_attempts=512,completed_scored_attempts=508,interrupted_unevaluable_attempts=4,
            full_raw_native_archive_uploaded=False,full_raw_report_uploaded=False,full_raw_native_evidence_retained_locally=True,
            training_weights_included=False,newly_trained_model=False,authored_acquisition_supervisor=True,
            learned_prediction_advisory=True,physical_robot_ready=False,qualified_robot_skill=False,
            successes=pub.SUCCESSES,full_raw_report_sha256=pub.RAW_SHA,sources=sources)
        write_json(self.payload/'scope.json',scope);self.integrity()
        sdk=SimpleNamespace(CommitOperationAdd=Operation,HfApi=lambda **kwargs:self.api,hf_hub_download=self.api.download)
        errors=SimpleNamespace(RepositoryNotFoundError=NotFound)
        handle=patch.dict(sys.modules,{'huggingface_hub':sdk,'huggingface_hub.errors':errors});handle.start();self.addCleanup(handle.stop)
        self.client=patch.object(pub,'client',return_value=(self.api,self.api.token)).start();self.addCleanup(patch.stopall)

    def integrity(self):
        entries={p.relative_to(self.payload).as_posix():pub.digest(p.read_bytes())
                 for p in self.payload.rglob('*') if p.is_file() and p.name!='integrity.json'}
        write_json(self.payload/'integrity.json',dict(schema='fixture-integrity',files=entries))

    def stage(self):return pub.stage(self.payload,self.output,api=self.api)

    def test_stage_and_check_are_anonymous_no_keychain_no_publish(self):
        with patch.object(pub,'client',side_effect=AssertionError('Stage/check cannot read Keychain')):
            plan=self.stage();self.assertEqual(pub.checked(self.path),plan)
        self.assertIsNone(plan['expected_parent']);self.assertEqual(plan['repo_id'],pub.REPO)
        review=pub.load(self.output/'review.json')
        self.assertFalse(review['publication_executed']);self.assertFalse(review['credentials_accessed'])
        self.assertEqual(self.api.calls,[]);self.assertEqual(self.api.created,0)

    def test_unknown_outcomes_and_raw_archive_claims_cannot_be_relabelled(self):
        scope=pub.load(self.payload/'scope.json')
        for key,value in (('interrupted_unevaluable_attempts',0),('full_raw_native_archive_uploaded',True),('newly_trained_model',True)):
            with self.subTest(key=key):
                write_json(self.payload/'scope.json',{**scope,key:value});self.integrity()
                with self.assertRaises(pub.PublicationError):self.stage()
        write_json(self.payload/'scope.json',scope)
        report_path=self.payload/pub.PREFIX/'report.json';report=pub.load(report_path)
        report['rows'][18]['native_summary']={'fake_contact':True};write_json(report_path,report);self.integrity()
        # Even when a synthetic fixture digest is declared, the semantic guard
        # still rejects inventing an observation for an unknown attempt.
        with patch.dict(pub.COMPACT,{'report.json':pub.digest(report_path.read_bytes())['sha256']}):
            with self.assertRaisesRegex(pub.PublicationError,'Interrupted outcomes'):self.stage()

    def test_local_changes_or_destination_tampering_block_before_keychain(self):
        self.stage();notice=self.payload/'NOTICE.txt';notice.write_bytes(notice.read_bytes()+b'changed')
        with patch.object(pub,'client',side_effect=AssertionError('Invalid bytes cannot access Keychain')):
            with self.assertRaises(pub.PublicationError):pub.publish(self.path)
        self.assertEqual(self.api.created,0)
        self.integrity();plan=pub.load(self.path);plan['repo_id']='Other/target';pub.write(self.path,plan)
        with self.assertRaises(pub.PublicationError):pub.checked(self.path)

    def test_existing_unrelated_files_and_identical_paths_are_preserved_without_rewrite(self):
        readme=(self.payload/'README.md').read_bytes()
        self.api.add({'foreign.md':b'preserve me','README.md':readme})
        self.stage();baseline=self.api.head;receipt=pub.publish(self.path)
        self.assertEqual(receipt['parent'],baseline)
        self.assertEqual(self.api.contents[receipt['commit']]['foreign.md'],b'preserve me')
        self.assertFalse(any(op.path_in_repo=='README.md' for op in self.api.calls[0]['operations']))
        self.assertEqual(pub.publish(self.path),receipt);self.assertEqual(len(self.api.calls),1)

    def test_existing_conflicting_path_or_moved_expected_parent_cannot_be_overwritten(self):
        self.api.add({'README.md':b'foreign content'})
        with self.assertRaisesRegex(pub.PublicationError,'pre-existing'):self.stage()
        self.api=FakeAPI();self.api.add({'foreign.md':b'keep'})
        with patch.object(pub,'client',return_value=(self.api,self.api.token)):
            self.stage();self.api.move()
            with self.assertRaisesRegex(pub.PublicationError,'Expected parent'):pub.publish(self.path)
        self.assertEqual(self.api.calls,[])

    def test_lost_accepted_reply_reconciles_exact_child_without_duplicate_commit(self):
        self.stage();self.api.fail_after=True
        with self.assertRaisesRegex(pub.PublicationError,'uncertain'):pub.publish(self.path)
        self.assertEqual(pub.load(self.output/'journal.json')['status'],'pending')
        receipt=pub.publish(self.path)
        self.assertEqual(receipt['commit'],self.api.head);self.assertEqual(len(self.api.calls),1)

    def test_lost_unaccepted_reply_retries_only_same_parent(self):
        self.stage();self.api.fail_before=True
        with self.assertRaises(pub.PublicationError):pub.publish(self.path)
        pub.publish(self.path)
        self.assertEqual(len(self.api.calls),2);self.assertEqual(self.api.calls[0]['parent'],self.api.calls[1]['parent'])

    def test_foreign_child_after_uncertain_reply_is_not_misidentified(self):
        self.stage();self.api.fail_after=True
        with self.assertRaises(pub.PublicationError):pub.publish(self.path)
        self.api.move()
        with self.assertRaisesRegex(pub.PublicationError,'immediate child'):pub.publish(self.path)
        self.assertEqual(len(self.api.calls),1)

    def test_uncertain_creation_retains_intent_and_never_creates_twice(self):
        self.stage();self.api.fail_creation=True
        with self.assertRaisesRegex(pub.PublicationError,'creation reply uncertain'):pub.publish(self.path)
        self.assertEqual(pub.load(self.output/'journal.json')['status'],'creating')
        pub.publish(self.path);self.assertEqual(self.api.created,1)

    def test_literal_publisher_credential_is_rejected_without_remote_writes_or_serialization(self):
        (self.payload/'NOTICE.txt').write_text('Synthetic credential literal: '+self.api.token);self.integrity();self.stage()
        with self.assertRaisesRegex(pub.PublicationError,'credential occurs'):pub.publish(self.path)
        self.assertEqual(self.api.created,0);self.assertEqual(self.api.calls,[])
        self.assertFalse((self.output/'journal.json').exists())
        self.assertTrue(all(self.api.token.encode() not in p.read_bytes() for p in self.output.iterdir() if p.is_file()))
        self.client.assert_called_once_with(self.output.resolve()/'publisher-cache')

    def test_credential_shaped_data_symlinks_and_binary_weights_are_rejected(self):
        for name,body in (('weights.bin',b'weights'),('NOTICE.txt',b'hf_'+b'x'*40)):
            path=self.payload/name;previous=path.read_bytes() if path.exists() else None
            path.write_bytes(body);self.integrity()
            with self.subTest(name=name),self.assertRaises(pub.PublicationError):self.stage()
            if previous is None:path.unlink()
            else:path.write_bytes(previous)
        (self.payload/'linked.md').symlink_to(self.payload/'README.md')
        with self.assertRaises(pub.PublicationError):self.stage()

    def test_only_public_anonymous_pinned_bytes_can_complete_verification(self):
        self.stage();pub.publish(self.path)
        with patch.object(pub,'client',side_effect=AssertionError('Verification cannot read Keychain')):
            proof=pub.verify(self.path,api=self.api,downloader=self.api.download)
        self.assertTrue(proof['anonymous_full_readback'])
        self.assertEqual(len(self.api.downloads),len(pub.checked(self.path)['files']))
        self.assertTrue(all(oid==self.api.head for oid,name in self.api.downloads))
        original=self.api.download
        def corrupt(*args,**kwargs):
            path=Path(original(*args,**kwargs));path.write_bytes(b'corrupted public bytes');return str(path)
        with self.assertRaisesRegex(pub.PublicationError,'anonymous file bytes'):
            pub.verify(self.path,api=self.api,downloader=corrupt)

    def test_api_exception_body_is_never_printed(self):
        stream=io.StringIO()
        with patch.object(pub,'checked',side_effect=RuntimeError(self.api.token)),redirect_stderr(stream):
            result=pub.main(['check','--plan',str(self.path)])
        self.assertEqual(result,2);self.assertIn('RuntimeError',stream.getvalue());self.assertNotIn(self.api.token,stream.getvalue())


if __name__=='__main__':unittest.main()
