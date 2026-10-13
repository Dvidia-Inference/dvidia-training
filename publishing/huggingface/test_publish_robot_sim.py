"""Offline fixed-destination publication checks. No real token or network use."""
from hashlib import sha256
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import publish_robot_sim as publisher
from publish import PublicationError


class RobotSimulationPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.space = self.root/'space'
        self.space.mkdir()
        for name,raw in {
            'README.md':b'---\ntitle: Test\nsdk: static\napp_file: index.html\nlicense: mit\n---\nNative recorded evidence.\n',
            'index.html':b'<!doctype html><p>Native recorded simulation</p>',
            'LICENSE':b'MIT','provenance.json':b'{}',
            'replay/geometry.bin':b'native geometry','replay/motion.bin.gz':b'recorded poses'}.items():
            path = self.space/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
        self.plan = {'kind':publisher.KIND,'schema_version':1,'namespace':'Dvidia','private':False,
                     'expected_remote_commit':None,'repository':{'repo_id':publisher.REPO_ID,'repo_type':'space',
                     'space_sdk':'static','directory':'space','files':[]},'total_bytes':0}
        self.path = self.root/'publication-plan.json'
        self.refresh()
        self.credential_guard = patch.object(publisher,'client',side_effect=AssertionError('Real credentials are forbidden in these tests'))
        self.credential_guard.start()
        self.addCleanup(self.credential_guard.stop)

    def refresh(self):
        self.plan['repository']['files'] = [{'path':path.relative_to(self.space).as_posix(),
            'bytes':path.stat().st_size,'sha256':sha256(path.read_bytes()).hexdigest()}
            for path in sorted(self.space.rglob('*')) if path.is_file()]
        self.plan['total_bytes'] = sum(f['bytes'] for f in self.plan['repository']['files'])
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.plan,indent=2)+'\n')

    def api(self,existing=()):
        api = Mock()
        api.whoami.return_value = {'orgs':[{'name':'Dvidia'}]}
        api.repo_info.return_value = SimpleNamespace(private=False,sdk='static',sha='a'*40,
            siblings=[SimpleNamespace(rfilename=name) for name in existing])
        api.create_commit.return_value = SimpleNamespace(oid='b'*40)
        return api

    def test_accepts_complete_named_static_space_and_native_binaries(self):
        self.assertEqual(publisher.checked_plan(self.path),self.plan)

    def test_live_workstation_native_files_are_scoped(self):
        for name in ['workstation/v01/mujoco.wasm', 'workstation/v01/scene.xml',
                     'workstation/v01/assets/link1.stl', 'workstation/v01/assets/link2.obj']:
            self.assertEqual(publisher._artifact_name(name).as_posix(), name)
        for name in ['mujoco.wasm', 'workstation/v02/mujoco.wasm', 'workstation/v01/plugin.wasm',
                     'workstation/v01/assets/../../secret.xml', 'workstation/v01/assets/model.xml']:
            with self.subTest(name=name), self.assertRaises(PublicationError):
                publisher._artifact_name(name)

    def test_rejects_unrelated_destination_private_sdk_and_path_escape(self):
        for field,value in [('repo_id','Dvidia/observation-lab'),('repo_type','model'),
                            ('space_sdk','docker'),('directory','../outside')]:
            changed = json.loads(json.dumps(self.plan))
            changed['repository'][field] = value
            self.path.write_text(json.dumps(changed))
            with self.subTest(field=field),self.assertRaises(PublicationError):
                publisher.checked_plan(self.path)
        self.plan['private'] = True
        self.save()
        with self.assertRaises(PublicationError): publisher.checked_plan(self.path)

    def test_rejects_unlisted_tampered_weight_hidden_secret_and_private_path(self):
        extra = self.space/'extra.json'
        extra.write_bytes(b'{}')
        with self.assertRaises(PublicationError): publisher.checked_plan(self.path)
        extra.unlink()
        (self.space/'index.html').write_bytes(b'changed')
        with self.assertRaises(PublicationError): publisher.checked_plan(self.path)
        for name,raw in [('policy.bin',b'weights'),('.env',b'secret'),('weights.onnx',b'weights'),
                         ('secret.txt',b'hf_'+b'a'*32),('private.txt',b'/Users/roguedev/private')]:
            file = self.space/name
            file.write_bytes(raw)
            self.refresh()
            with self.subTest(name=name),self.assertRaises(PublicationError): publisher.checked_plan(self.path)
            file.unlink()
        self.refresh()

    def test_rejects_symlink_and_malformed_totals_before_credentials(self):
        alias = self.space/'alias.json'
        alias.symlink_to(self.space/'provenance.json')
        with self.assertRaises(PublicationError): publisher.checked_plan(self.path)
        alias.unlink()
        self.plan['total_bytes'] += 1
        self.save()
        with self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)

    def test_frontmatter_cannot_be_spoofed_by_body_or_duplicate_keys(self):
        cards = [
            '---\nsdk: docker\napp_file: app.py\nlicense: apache-2.0\n---\nsdk: static\napp_file: index.html\nlicense: mit\n',
            '---\nsdk: docker\nsdk: static\napp_file: index.html\nlicense: mit\n---\n',
            '---\nsdk: static\n"sdk": gradio\napp_file: index.html\nlicense: mit\n---\n',
            '---\nsdk: static\n  sdk: gradio\napp_file: index.html\nlicense: mit\n---\n',
            'sdk: static\napp_file: index.html\nlicense: mit\n',
            '---\nsdk: static\napp_file: index.html\nlicense: mit\n']
        for card in cards:
            (self.space/'README.md').write_text(card)
            self.refresh()
            with self.subTest(card=card),self.assertRaises(PublicationError): publisher.checked_plan(self.path)

    def test_byte_bounds_are_checked_without_credential_access(self):
        self.assertEqual(publisher.MAX_TOTAL_BYTES,768*1024*1024)
        self.assertEqual(publisher.MAX_FILE_BYTES,32*1024*1024)
        with patch.object(publisher,'MAX_TOTAL_BYTES',self.plan['total_bytes']):
            self.assertEqual(publisher.checked_plan(self.path),self.plan)
        with patch.object(publisher,'MAX_TOTAL_BYTES',self.plan['total_bytes']-1),self.assertRaises(PublicationError):
            publisher.checked_plan(self.path)
        with patch.object(publisher,'MAX_FILE_BYTES',4),self.assertRaises(PublicationError):
            publisher.checked_plan(self.path)
        with patch.object(publisher,'MAX_TOTAL_BYTES',16),self.assertRaises(PublicationError):
            publisher.checked_plan(self.path)

    def test_receipt_hash_and_upload_bytes_use_original_snapshot_despite_local_edits(self):
        api = self.api()
        original_plan = self.path.read_bytes()
        original_index = (self.space/'index.html').read_bytes()
        def change_local():
            self.path.write_text('{"changed_after_review":true}')
            (self.space/'index.html').write_bytes(b'changed after payload freeze')
            return {'orgs':[{'name':'Dvidia'}]}
        api.whoami.side_effect = change_local
        with patch.object(publisher,'client',return_value=(api,'fixture-token')), \
                patch.object(publisher,'verified_downloads') as verify:
            result = publisher.publish_robot_sim(self.path)
        self.assertEqual(result['plan_sha256'],sha256(original_plan).hexdigest())
        operations = api.create_commit.call_args.kwargs['operations']
        uploaded = {operation.path_in_repo:operation.path_or_fileobj for operation in operations}
        self.assertEqual(uploaded['index.html'],original_index)
        self.assertEqual(api.create_commit.call_args.kwargs['parent_commit'],'a'*40)
        self.assertTrue(all(call.args[2] is False for call in verify.call_args_list))

    def test_same_plan_retry_is_idempotent_but_changed_parent_and_unrelated_files_fail(self):
        all_files = [f['path'] for f in self.plan['repository']['files']]
        api = self.api(all_files)
        with patch.object(publisher,'client',return_value=(api,'fixture-token')),patch.object(publisher,'verified_downloads'):
            self.assertEqual(publisher.publish_robot_sim(self.path)['commit'],'a'*40)
        api.create_commit.assert_not_called()
        for existing,parent in [(['unrelated.txt'],None),(all_files,'c'*40)]:
            self.plan['expected_remote_commit'] = parent
            self.save()
            api = self.api(existing)
            with patch.object(publisher,'client',return_value=(api,'fixture-token')),patch.object(publisher,'verified_downloads'),self.assertRaises(PublicationError):
                publisher.publish_robot_sim(self.path)
            api.create_commit.assert_not_called()

    def test_reviewed_exact_parent_allows_an_explicit_static_space_update(self):
        self.plan['expected_remote_commit'] = 'a'*40
        self.save()
        api = self.api([f['path'] for f in self.plan['repository']['files']])
        with patch.object(publisher,'client',return_value=(api,'fixture-token')),patch.object(publisher,'verified_downloads'):
            self.assertEqual(publisher.publish_robot_sim(self.path)['commit'],'b'*40)
        self.assertEqual(api.create_commit.call_args.kwargs['parent_commit'],'a'*40)
        api.create_repo.assert_not_called()

    def test_starter_stylesheet_removal_needs_exact_parent_bounded_hash_and_one_allowed_path(self):
        valid = {'path':'style.css','bytes':20,'sha256':sha256(b'fixture starter css!').hexdigest()}
        variants = [([valid],None),([valid,valid],'a'*40),([None],'a'*40),
            ([{**valid,'path':'unrelated.txt'}],'a'*40),
            ([{**valid,'bytes':True}],'a'*40),([{**valid,'bytes':-1}],'a'*40),
            ([{**valid,'bytes':publisher.MAX_STARTER_STYLESHEET_BYTES+1}],'a'*40),
            ([{**valid,'sha256':'short'}],'a'*40),([{**valid,'extra':'field'}],'a'*40),
            ({'style.css':valid},'a'*40)]
        for removals,parent in variants:
            self.plan['expected_remote_commit'] = parent
            self.plan['remove_files'] = removals
            self.save()
            with self.subTest(removals=removals,parent=parent),self.assertRaises(PublicationError):
                publisher.publish_robot_sim(self.path)

    def test_reviewed_bootstrap_verifies_and_deletes_only_style_at_exact_parent_in_one_commit(self):
        raw = b'body { color: black; }\n'
        removal = {'path':'style.css','bytes':len(raw),'sha256':sha256(raw).hexdigest()}
        self.plan.update(expected_remote_commit='a'*40,remove_files=[removal])
        self.save()
        api = self.api(['.gitattributes','README.md','index.html','style.css'])
        with patch.object(publisher,'client',return_value=(api,'fixture-token')), \
                patch.object(publisher,'verified_downloads') as verify:
            result = publisher.publish_robot_sim(self.path)
        self.assertEqual(result['removed_files'],[removal])
        self.assertEqual(verify.call_args_list[0].args,({**self.plan['repository'],'files':[removal]},'a'*40,False))
        self.assertEqual(verify.call_args_list[1].args,(self.plan['repository'],'b'*40,False))
        operations = api.create_commit.call_args.kwargs['operations']
        deletes = [operation for operation in operations if type(operation).__name__=='CommitOperationDelete']
        adds = [operation for operation in operations if type(operation).__name__=='CommitOperationAdd']
        self.assertEqual([operation.path_in_repo for operation in deletes],['style.css'])
        self.assertFalse(deletes[0].is_folder)
        self.assertEqual({operation.path_in_repo for operation in adds},set(f['path'] for f in self.plan['repository']['files']))
        self.assertEqual(api.create_commit.call_args.kwargs['parent_commit'],'a'*40)
        api.create_commit.assert_called_once()
        api.create_repo.assert_not_called()

    def test_bootstrap_changed_remote_bytes_parent_missing_stylesheet_and_unrelated_files_block_commit(self):
        removal = {'path':'style.css','bytes':1,'sha256':sha256(b'x').hexdigest()}
        self.plan.update(expected_remote_commit='a'*40,remove_files=[removal])
        self.save()
        for variant in ('bytes','parent','missing','unrelated'):
            existing = ['README.md','index.html','style.css']
            if variant=='missing': existing.remove('style.css')
            if variant=='unrelated': existing.append('unrelated.txt')
            api = self.api(existing)
            if variant=='parent': api.repo_info.return_value.sha = 'c'*40
            with self.subTest(variant=variant),patch.object(publisher,'client',return_value=(api,'fixture-token')), \
                    patch.object(publisher,'verified_downloads',side_effect=PublicationError('Remote hash mismatch')) as verify, \
                    self.assertRaises(PublicationError):
                publisher.publish_robot_sim(self.path)
            api.create_commit.assert_not_called()
            api.create_repo.assert_not_called()
            if variant!='bytes': verify.assert_not_called()

    def compressed_episode(self,extra=None,comment=None,folder='piper-grip'):
        episode = {'schema':'dvidia.piper-task-episode/v1','profile':'piper-rigid-block-task-v0.1',
                   'physical_robot_ready':False,'learned_policy':False,**(extra or {})}
        data = gzip.compress(json.dumps(episode).encode(),mtime=0)
        if comment is not None:
            header=bytearray(data[:10]);header[3]|=8
            data=bytes(header)+comment+b'\0'+data[10:]
        target=self.space/('tasks/'+folder+'/attempts/friction080-layout00--guarded_placement.json.gz')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        self.refresh()
        return target,data

    def test_v03_compressed_namespace_keeps_raw_and_decoded_privacy_scope_and_bounds(self):
        target,valid=self.compressed_episode(folder='piper-contact-v03')
        publisher.checked_plan(self.path)
        self.assertEqual(json.loads(publisher._artifact_plaintext(target.relative_to(self.space).as_posix(),valid))['profile'],
                         'piper-rigid-block-task-v0.1')
        for name in ('tasks/piper-contact-v03/attempt.json.gz','tasks/piper-contact-v04/attempts/attempt.json.gz',
                     'tasks/piper-contact-v03/attempts/.hidden.json.gz','tasks/piper-contact-v03/attempts/weights.bin.gz'):
            with self.subTest(name=name),self.assertRaises(PublicationError): publisher._artifact_name(name)
        for extra,comment in [({'note':'/Users/roguedev/private'},None),({},b'hf_'+b'a'*32),
                              ({'physical_robot_ready':True},None),({'learned_policy':True},None)]:
            self.compressed_episode(extra,comment,folder='piper-contact-v03')
            with self.subTest(extra=extra,comment=comment),self.assertRaises(PublicationError):
                publisher.publish_robot_sim(self.path)
        for data in (b'not gzip',valid+valid,gzip.compress(b'not JSON',mtime=0)):
            target.write_bytes(data);self.refresh()
            with self.subTest(data=data[:10]),self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)
        self.compressed_episode({'large':'x'*2000},folder='piper-contact-v03')
        from dvidia_training import piper_grip_benchmark as benchmark
        with patch.object(benchmark,'MAX_FILE_BYTES',512),self.assertRaises(PublicationError):
            publisher.publish_robot_sim(self.path)

    def test_v03_exact_credential_is_checked_in_gzip_header_and_decoded_payload_before_remote_calls(self):
        for extra,comment in [({'note':'fixture-token'},None),({},b'fixture-token')]:
            self.compressed_episode(extra,comment,folder='piper-contact-v03')
            api=self.api()
            with self.subTest(extra=extra,comment=comment),patch.object(publisher,'client',return_value=(api,'fixture-token')), \
                    self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)
            api.whoami.assert_not_called()
            api.create_commit.assert_not_called()

    def test_v04_compressed_namespace_keeps_raw_and_decoded_privacy_scope_and_bounds(self):
        target,valid=self.compressed_episode(folder='piper-recovery-v04')
        publisher.checked_plan(self.path)
        self.assertEqual(json.loads(publisher._artifact_plaintext(target.relative_to(self.space).as_posix(),valid))['profile'],
                         'piper-rigid-block-task-v0.1')
        for name in ('tasks/piper-recovery-v04/attempt.json.gz','tasks/piper-recovery-v05/attempts/attempt.json.gz',
                     'tasks/piper-recovery-v04/attempts/.hidden.json.gz','tasks/piper-recovery-v04/attempts/weights.bin.gz'):
            with self.subTest(name=name),self.assertRaises(PublicationError): publisher._artifact_name(name)
        for extra,comment in [({'note':'/Users/roguedev/private'},None),({},b'hf_'+b'a'*32),
                              ({'physical_robot_ready':True},None),({'learned_policy':True},None)]:
            self.compressed_episode(extra,comment,folder='piper-recovery-v04')
            with self.subTest(extra=extra,comment=comment),self.assertRaises(PublicationError):
                publisher.publish_robot_sim(self.path)
        for data in (b'not gzip',valid+valid,gzip.compress(b'not JSON',mtime=0)):
            target.write_bytes(data);self.refresh()
            with self.subTest(data=data[:10]),self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)
        self.compressed_episode({'large':'x'*2000},folder='piper-recovery-v04')
        from dvidia_training import piper_grip_benchmark as benchmark
        with patch.object(benchmark,'MAX_FILE_BYTES',512),self.assertRaises(PublicationError):
            publisher.publish_robot_sim(self.path)

    def test_v04_exact_credential_is_checked_in_gzip_header_and_decoded_payload_before_remote_calls(self):
        for extra,comment in [({'note':'fixture-token'},None),({},b'fixture-token')]:
            self.compressed_episode(extra,comment,folder='piper-recovery-v04')
            api=self.api()
            with self.subTest(extra=extra,comment=comment),patch.object(publisher,'client',return_value=(api,'fixture-token')), \
                    self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)
            api.whoami.assert_not_called()
            api.create_commit.assert_not_called()

    def test_v05_scoped_cohorts_keep_gzip_privacy_corruption_and_decoding_limits(self):
        folder='piper-reliability-v05/confirmation-950031'
        target,valid=self.compressed_episode(folder=folder)
        publisher.checked_plan(self.path)
        for name in ('tasks/piper-reliability-v05/attempts/a.json.gz',
                     'tasks/piper-reliability-v05/arbitrary/attempts/a.json.gz',
                     'tasks/piper-reliability-v05/confirmation-999989/attempts/a.json.gz',
                     'tasks/piper-reliability-v05/development-01/attempts/a.json.gz',
                     'tasks/piper-reliability-v05/development-1/attempts/weights.bin.gz'):
            with self.subTest(name=name),self.assertRaises(PublicationError):
                publisher._artifact_name(name)
        for extra,comment in [({'note':'/Users/roguedev/private'},None),({},b'hf_'+b'a'*32),
                              ({'physical_robot_ready':True},None),({'learned_policy':True},None)]:
            self.compressed_episode(extra,comment,folder=folder)
            with self.subTest(extra=extra,comment=comment),self.assertRaises(PublicationError):
                publisher.publish_robot_sim(self.path)
        for data in (b'not gzip',valid+valid,gzip.compress(b'not JSON',mtime=0)):
            target.write_bytes(data);self.refresh()
            with self.subTest(data=data[:10]),self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)
        self.compressed_episode({'large':'x'*2000},folder=folder)
        from dvidia_training import piper_grip_benchmark as benchmark
        with patch.object(benchmark,'MAX_FILE_BYTES',512),self.assertRaises(PublicationError):
            publisher.publish_robot_sim(self.path)

    def test_v05_cohort_limit_is_enforced_again_in_publication_plan(self):
        for seed in (1,2,3):
            self.compressed_episode(folder='piper-reliability-v05/confirmation-'+str(seed))
        publisher.checked_plan(self.path)
        self.compressed_episode(folder='piper-reliability-v05/confirmation-4')
        with self.assertRaisesRegex(PublicationError,'At most three'):
            publisher.publish_robot_sim(self.path)

    def escaped_episode(self,folder,value):
        target,_=self.compressed_episode({'note':'escape-fixture'},folder=folder)
        raw=gzip.decompress(target.read_bytes()).replace(b'escape-fixture',
            ''.join('\\u'+format(ord(char),'04x') for char in value).encode())
        self.assertNotIn(value.encode(),raw)
        data=gzip.compress(raw,mtime=0)
        target.write_bytes(data);self.refresh()
        return target,data

    def test_escaped_json_private_paths_and_token_patterns_are_rejected_without_rewriting_evidence(self):
        for folder in ('piper-grip','piper-contact-v03','piper-recovery-v04','piper-reliability-v05/confirmation-950031'):
            for value in ('/Users/roguedev/private','hf_'+'a'*32):
                with self.subTest(folder=folder,value=value):
                    target,data=self.escaped_episode(folder,value)
                    with self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)
                    self.assertEqual(target.read_bytes(),data)
                    target.unlink();self.refresh()

    def test_escaped_exact_in_memory_credential_is_rejected_before_remote_identity(self):
        for folder in ('piper-grip','piper-contact-v03','piper-recovery-v04','piper-reliability-v05/confirmation-950031'):
            target,data=self.escaped_episode(folder,'fixture-token')
            api=self.api()
            with self.subTest(folder=folder),patch.object(publisher,'client',return_value=(api,'fixture-token')), \
                    self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)
            api.whoami.assert_not_called()
            api.create_commit.assert_not_called()
            self.assertEqual(target.read_bytes(),data)
            target.unlink();self.refresh()

    def test_normalized_json_privacy_parsing_keeps_bounds_and_rejects_invalid_strings(self):
        target,_=self.compressed_episode(folder='piper-contact-v03')
        name=target.relative_to(self.space).as_posix()
        nested=gzip.decompress(target.read_bytes()).removesuffix(b'}')+b',"note":'+b'['*1500+b'0'+b']'*1500+b'}'
        for data in (gzip.compress(nested,mtime=0),
                     gzip.compress(b'{"schema":"dvidia.piper-task-episode/v1","schema":"duplicate"}',mtime=0)):
            with self.subTest(data=data[:10]),self.assertRaises(PublicationError):
                publisher._artifact_json_texts(name,data)
        _,data=self.compressed_episode({'note':'\ud800'},folder='piper-contact-v03')
        with self.assertRaises(PublicationError): publisher._artifact_json_texts(name,data)
        _,data=self.compressed_episode({'large':'x'*2000},folder='piper-contact-v03')
        with patch.object(publisher,'MAX_FILE_BYTES',512),self.assertRaises(PublicationError):
            publisher._artifact_json_texts(name,data)

    def test_only_scoped_compressed_grip_json_is_allowed_and_private_decoded_or_header_bytes_fail(self):
        self.compressed_episode()
        publisher.checked_plan(self.path)
        for name in ('attempt.json.gz','tasks/piper/attempts/attempt.json.gz','tasks/other/attempts/attempt.json.gz',
                     'tasks/piper-grip/attempt.json.gz','tasks/piper-grip/attempts/weights.bin.gz',
                     'tasks/piper-grip/attempts/.hidden.json.gz'):
            with self.subTest(name=name),self.assertRaises(PublicationError): publisher._artifact_name(name)
        for extra,comment in [({'note':'/Users/roguedev/private'},None),({'note':'hf_'+'a'*32},None),
                              ({},b'hf_'+b'a'*32),({},b'/Users/roguedev/private')]:
            self.compressed_episode(extra,comment)
            with self.subTest(extra=extra,comment=comment),self.assertRaises(PublicationError):
                publisher.publish_robot_sim(self.path)

    def test_compressed_grip_corruption_scope_concatenation_and_decoded_bounds_fail_before_credentials(self):
        target,valid=self.compressed_episode()
        publisher._artifact_plaintext(target.relative_to(self.space).as_posix(),valid)
        from dvidia_training import piper_grip_benchmark as benchmark
        for data in (b'not gzip',valid+valid,gzip.compress(b'not JSON',mtime=0),
                     gzip.compress(b'{"physical_robot_ready":true}',mtime=0)):
            target.write_bytes(data);self.refresh()
            with self.subTest(data=data[:10]),self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)
        self.compressed_episode({'large':'x'*2000})
        with patch.object(benchmark,'MAX_FILE_BYTES',512),self.assertRaises(PublicationError):
            publisher.publish_robot_sim(self.path)

    def test_exact_credential_in_gzip_header_or_decoded_json_is_rejected_after_freeze(self):
        for extra,comment in [({'note':'fixture-token'},None),({},b'fixture-token')]:
            self.compressed_episode(extra,comment)
            api=self.api()
            with self.subTest(extra=extra,comment=comment),patch.object(publisher,'client',return_value=(api,'fixture-token')), \
                    self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)
            api.whoami.assert_not_called()
            api.create_commit.assert_not_called()


if __name__=='__main__':
    unittest.main()
