"""No-network scope/concurrency tests, plus unchanged-publisher integration."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Lock
import time
import io
import json
from unittest import TestCase, main, mock

import grip_clock_prefetch_http_v03 as prefetch
import publish_grip_clock_v03 as pub
from publish import PublicationError
import test_publish_grip_clock_v03 as fixture


class ReadAheadTests(TestCase):
    def setUp(self):
        self.fixture = fixture.V03PublicationTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        for index in range(24):
            (self.fixture.inputs['space']/pub.NAMESPACE/f'{index:02}.json').write_bytes(pub.encoded(dict(index=index)))
        self.plan = self.fixture.stage()
        self.path = self.fixture.output/pub.PLAN
        self.api = self.fixture.api

    def begin(self, kind='space'):
        state = pub._state(self.plan, pub.decoded(pub.read(self.fixture.baseline,pub.MAX_PLAN)))
        row = next(item for item in self.plan['repositories'] if item['repo_type']==kind)
        value = state['repositories'][kind]
        if kind!='space':
            self.api.create_repo(pub.REPOS[kind],repo_type=kind,private=False,exist_ok=False)
            head,tree=pub._remote(self.api,kind)
            value.update(head=head,baseline_head=head,baseline_tree=tree)
        value['pending']=dict(kind='commit',index=0,parent=value['head'],message=pub._message(state['plan_sha256'],kind,0))
        revision='b'*40
        pub.atomic(self.path.parent/pub.JOURNAL,state)
        raw={item['path']:pub.read(self.path.parent/kind/item['path']) for item in row['files']}
        return state,row,revision,raw

    def finish(self,state,row,revision,raw):
        value=state['repositories'][row['repo_type']]
        index=len(value['commits']);paths=row['batches'][index]
        value['commits'].append(dict(index=index,parent=value['head'],commit=revision,paths=paths,
            entries={name:pub._entry(raw[name],lfs=next(x for x in row['files'] if x['path']==name)['representation']=='lfs') for name in paths},
            message=value['pending']['message'],anonymous_readback=True,reconciled=False))
        value.update(head=revision,pending=None)
        pub.atomic(self.path.parent/pub.JOURNAL,state)

    def test_unchanged_publisher_integration_and_final_fresh_reads(self):
        calls=[]
        def fetch(*args):
            calls.append(args)
            return self.api.download(*args)
        with mock.patch.object(prefetch.direct,'download',side_effect=fetch), prefetch.ReadAhead(self.path,workers=4,window=8) as reader:
            result=pub.publish(self.path,api=self.api,downloader=reader.download,operation_cls=fixture.Operation)
        self.assertTrue(result['anonymous_full_readback'])
        self.assertEqual([scope['phase'] for scope in reader.scope_receipts].count('final'),3)
        for row in self.plan['repositories']:
            final=self.api.repos[row['repo_type']]['head']
            for item in row['files']:
                matches=sum(args==(pub.REPOS[row['repo_type']],item['path'],row['repo_type'],final) for args in calls)
                self.assertEqual(matches,2 if item['path'] in row['batches'][-1] else 1)

    def test_same_revision_batch_then_final_must_get_all_bytes_again(self):
        state,row,revision,raw=self.begin()
        calls=[]
        with mock.patch.object(prefetch.direct,'download',side_effect=lambda repo,name,kind,rev:(calls.append((name,rev)) or raw[name])), prefetch.ReadAhead(self.path) as reader:
            for name in row['batches'][0]: reader(pub.REPOS['space'],name,'space',revision)
            self.finish(state,row,revision,raw)
            for item in row['files']: reader(pub.REPOS['space'],item['path'],'space',revision)
            self.assertEqual([scope['phase'] for scope in reader.scope_receipts],['batch','final'])
        self.assertEqual(len(calls),2*len(raw))

    def test_out_of_order_parallel_callers_respect_worker_and_window_bounds(self):
        _,row,revision,raw=self.begin()
        lock=Lock(); active=peak=0
        def fetch(repo,name,kind,rev):
            nonlocal active,peak
            with lock: active+=1;peak=max(peak,active)
            time.sleep(.002*(1+int(name.rsplit('/',1)[-1][:2])%3) if name.endswith('.json') else .004)
            with lock: active-=1
            return raw[name]
        names=list(reversed(row['batches'][0]))
        with mock.patch.object(prefetch.direct,'download',side_effect=fetch), prefetch.ReadAhead(self.path,workers=3,window=4) as reader:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results=list(pool.map(lambda name:reader(pub.REPOS['space'],name,'space',revision),names))
            self.assertEqual(results,[raw[name] for name in names])
            self.assertLessEqual(reader.peak_futures,4)
            self.assertLessEqual(peak,3)
            self.assertGreaterEqual(peak,2)

    def test_earlier_batch_bytes_are_fetched_again_at_final_revision(self):
        state,row,first,raw=self.begin('model');calls=[]
        with mock.patch.object(prefetch.direct,'download',side_effect=lambda repo,name,kind,rev:(calls.append((name,rev)) or raw[name])),prefetch.ReadAhead(self.path) as reader:
            for name in row['batches'][0]:reader(pub.REPOS['model'],name,'model',first)
            self.finish(state,row,first,raw)
            value=state['repositories']['model']
            value['pending']=dict(kind='commit',index=1,parent=first,message=pub._message(state['plan_sha256'],'model',1))
            pub.atomic(self.path.parent/pub.JOURNAL,state)
            final='c'*40
            for name in row['batches'][1]:reader(pub.REPOS['model'],name,'model',final)
            self.finish(state,row,final,raw)
            for item in row['files']:reader(pub.REPOS['model'],item['path'],'model',final)
        self.assertEqual([revision for name,revision in calls if name=='.gitattributes'],[first,final])

    def test_wrong_repo_revision_path_and_duplicate_request_are_rejected(self):
        _,row,revision,raw=self.begin(); name=row['batches'][0][0]
        with mock.patch.object(prefetch.direct,'download',side_effect=lambda repo,name,kind,rev:raw[name]), prefetch.ReadAhead(self.path) as reader:
            for args in [('other/repo',name,'space',revision),(pub.REPOS['space'],name,'space','main'),
                         (pub.REPOS['space'],'unknown.json','space',revision),
                         (pub.REPOS['space'],name,'space','a'*40)]:
                with self.subTest(args=args),self.assertRaises(PublicationError): reader(*args)
            reader(pub.REPOS['space'],name,'space',revision)
            with self.assertRaises(PublicationError):reader(pub.REPOS['space'],name,'space',revision)
            with self.assertRaises(PublicationError):reader(pub.REPOS['space'],row['batches'][0][1],'space','c'*40)

    def test_scope_cannot_switch_while_previous_scope_is_not_consumed(self):
        _,row,revision,raw=self.begin();name=row['batches'][0][0]
        with mock.patch.object(prefetch.direct,'download',side_effect=lambda repo,name,kind,rev:raw[name]), prefetch.ReadAhead(self.path) as reader:
            reader(pub.REPOS['space'],name,'space',revision)
            with self.assertRaises(PublicationError):reader(pub.REPOS['model'],'README.md','model','c'*40)

    def test_wrong_journal_plan_identity_fails_before_get(self):
        state,row,revision,_=self.begin();state['plan_sha256']='f'*64
        pub.atomic(self.path.parent/pub.JOURNAL,state)
        with mock.patch.object(prefetch.direct,'download') as get,prefetch.ReadAhead(self.path) as reader:
            with self.assertRaises(PublicationError):reader(pub.REPOS['space'],row['batches'][0][0],'space',revision)
            get.assert_not_called()

    def test_source_plan_changed_between_scopes_is_rejected(self):
        state,row,revision,raw=self.begin()
        with mock.patch.object(prefetch.direct,'download',side_effect=lambda repo,name,kind,rev:raw[name]),prefetch.ReadAhead(self.path) as reader:
            for name in row['batches'][0]: reader(pub.REPOS['space'],name,'space',revision)
            self.finish(state,row,revision,raw)
            self.path.write_bytes(self.path.read_bytes()+b' ')
            with self.assertRaises(PublicationError):reader(pub.REPOS['space'],row['batches'][0][0],'space',revision)

    def test_bad_bytes_fail_before_return_and_failed_prefetch_cannot_be_hidden(self):
        _,row,revision,_=self.begin();name=row['batches'][0][0]
        with mock.patch.object(prefetch.direct,'download',return_value=b'wrong'),prefetch.ReadAhead(self.path,workers=2,window=2) as reader:
            with self.assertRaises(PublicationError):reader(pub.REPOS['space'],name,'space',revision)

    def test_safe_failure_diagnostic_retains_planned_path_status_and_class_only(self):
        _,row,revision,_=self.begin();name=row['batches'][0][0]
        error=prefetch.direct.PublicReadbackError(name,'HTTPError',403,1,'unexpected_http_status')
        diagnostic=io.StringIO()
        with mock.patch.object(prefetch.direct,'download',side_effect=error),mock.patch.object(prefetch.sys,'stderr',diagnostic),prefetch.ReadAhead(self.path,workers=1,window=1) as reader:
            with self.assertRaises(PublicationError):reader(pub.REPOS['space'],name,'space',revision)
        rows=[json.loads(line) for line in diagnostic.getvalue().splitlines()]
        self.assertEqual(rows[0]['path'],name)
        self.assertEqual(rows[0]['http_status'],403)
        self.assertEqual(rows[0]['failure_class'],'HTTPError')
        self.assertNotIn('url',rows[0])

    def test_completed_final_scope_cannot_be_replayed_and_close_rejects_calls(self):
        state,row,revision,raw=self.begin();self.finish(state,row,revision,raw)
        with mock.patch.object(prefetch.direct,'download',side_effect=lambda repo,name,kind,rev:raw[name]),prefetch.ReadAhead(self.path) as reader:
            for item in row['files']:reader(pub.REPOS['space'],item['path'],'space',revision)
            with self.assertRaises(PublicationError):reader(pub.REPOS['space'],row['files'][0]['path'],'space',revision)
        with self.assertRaises(PublicationError):reader(pub.REPOS['space'],row['files'][0]['path'],'space',revision)

    def test_constructor_bounds_and_declared_file_identity_validation(self):
        for workers,window in [(0,4),(4,3),(4,17),(True,4)]:
            with self.subTest(workers=workers,window=window),self.assertRaises(PublicationError):
                prefetch.ReadAhead(self.path,workers=workers,window=window)
        original=self.path.read_bytes(); bad=deepcopy(self.plan)
        bad['repositories'][0]['files'][0]['bytes']=8*1024*1024+1
        self.path.write_bytes(pub.encoded(bad))
        with self.assertRaises(PublicationError):prefetch.ReadAhead(self.path)
        self.path.write_bytes(original)


if __name__=='__main__':main()
