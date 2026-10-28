"""Bounded byte read-ahead; the original publisher owns every remote guard/write.

Only an active journal-declared batch, or a fresh final all-file scope, can be
read. Batch revision acceptance remains the unchanged publisher's remote-tree
guard: a pending journal intentionally does not yet contain that new commit.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import sys
from threading import Condition

import grip_clock_public_http as direct
import publish_grip_clock_v03 as pub
from publish import PublicationError


class ReadAhead:
    def __init__(self, plan_path, workers=12, window=16):
        if (type(workers) is not int or type(window) is not int
                or not 1 <= workers <= window <= 16):
            raise PublicationError('Read-ahead requires 1 <= workers <= window <= 16.')
        self.path = pub.plain(plan_path)
        self.raw = pub.read(self.path, pub.MAX_PLAN)
        self.plan = pub.decoded(self.raw)
        plan = self.plan
        if (type(plan) is not dict or plan.get('schema') != pub.SCHEMA
                or plan.get('public') is not True or plan.get('physical_robot_ready') is not False
                or plan.get('space_namespace') != pub.NAMESPACE
                or plan.get('implementation') != pub._implementation()
                or self.raw != pub.encoded(plan)):
            raise PublicationError('Read-ahead requires the exact canonical publication plan.')
        baseline_raw = pub.read(self.path.parent/'inputs'/'space-baseline.json', pub.MAX_PLAN)
        if pub.receipt(baseline_raw) != plan.get('baseline'):
            raise PublicationError('Read-ahead baseline differs from the bound publication plan.')
        self.baseline = pub._baseline(pub.decoded(baseline_raw))
        rows = plan.get('repositories')
        kinds = [row.get('repo_type') for row in rows] if type(rows) is list else []
        if not kinds or kinds != [kind for kind in pub.REPOS if kind in kinds]:
            raise PublicationError('Read-ahead requires an ordered subset of the fixed repositories.')
        self.rows, self.items = {}, {}
        for row in rows:
            kind = row['repo_type']
            if (row.get('repo_id') != pub.REPOS[kind] or row.get('directory') != kind
                    or row.get('expected_parent') != (self.baseline['commit'] if kind == 'space' else None)
                    or type(row.get('files')) is not list or not 1 <= len(row['files']) <= pub.MAX_FILES):
                raise PublicationError('Invalid read-ahead repository declaration.')
            items = {}
            for item in row['files']:
                if (type(item) is not dict or set(item) != {'path','bytes','sha256','representation'}
                        or type(item['bytes']) is not int or not 0 <= item['bytes'] <= direct.MAX_BYTES
                        or not isinstance(item['sha256'], str) or pub.SHA.fullmatch(item['sha256']) is None
                        or item['representation'] not in ('git','lfs')):
                    raise PublicationError('Invalid read-ahead expected byte identity.')
                pub.relative(item['path'], attributes=True)
                if item['path'] in items:
                    raise PublicationError('Duplicate read-ahead expected path.')
                items[item['path']] = item
            if row.get('batches') != pub._batches(row['files']):
                raise PublicationError('Read-ahead batches differ from the deterministic publication plan.')
            self.rows[kind], self.items[kind] = row, items
        self.workers, self.window = workers, window
        self._condition = Condition()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='dvidia-public-get')
        self._scope = None
        self._completed = set()
        self._futures, self._claimed, self._consumed = {}, set(), set()
        self._remaining = deque()
        self._fatal = None
        self._closed = False
        self.peak_futures = 0
        self.scope_receipts = []

    def _new_scope(self, kind, name, revision):
        if pub.read(self.path, pub.MAX_PLAN) != self.raw or pub._implementation() != self.plan['implementation']:
            raise PublicationError('Read-ahead publication plan or original publisher changed.')
        journal = self.path.parent/pub.JOURNAL
        if not journal.is_file():
            raise PublicationError('Read-ahead requires the publisher-owned recovery journal.')
        try:
            state, _ = pub._journal(journal, self.plan, self.baseline)
            value, row = state['repositories'][kind], self.rows[kind]
            pending = value['pending']
            if pending is not None and pending.get('kind') == 'commit':
                if revision == pending['parent']:
                    raise PublicationError('Read-ahead batch requires the new publisher-verified commit.')
                phase, index = 'batch', pending['index']
                names = row['batches'][index]
            elif pending is None and len(value['commits']) == len(row['batches']) and value['head'] == revision:
                phase, index = 'final', None
                names = [item['path'] for item in row['files']]
            else:
                raise PublicationError('Read-ahead request is outside a pending batch or completed final revision.')
        except (KeyError, TypeError, IndexError, ValueError):
            raise PublicationError('Malformed read-ahead journal state.') from None
        scope = (kind, revision, phase, index)
        if name not in names or scope in self._completed:
            raise PublicationError('Read-ahead request is outside its fresh declared scope.')
        self._scope, self._names = scope, set(names)
        self._futures, self._claimed, self._consumed = {}, set(), set()
        # First caller may arrive out of order; give its bytes priority.
        self._remaining = deque([name] + [item for item in names if item != name])
        self.scope_receipts.append(dict(kind=kind,revision=revision,phase=phase,index=index,
                                       journal_canonical_sha256=sha256(pub.encoded(state)).hexdigest(),
                                       plan_sha256=sha256(self.raw).hexdigest(),files=len(names)))
        self._fill()

    def _fetch(self, name):
        kind, revision, _, _ = self._scope
        try:
            raw = direct.download(pub.REPOS[kind], name, kind, revision)
        except Exception as error:
            diagnostic = dict(phase='anonymous-readback-failure', repo_type=kind, path=name,
                exception_class=type(error).__name__, failure_class=getattr(error,'failure_class',type(error).__name__),
                http_status=getattr(error,'http_status',None), attempts=getattr(error,'attempts',None),
                check_code=getattr(error,'check_code',None))
            print(json.dumps(diagnostic,sort_keys=True),file=sys.stderr,flush=True)
            raise PublicationError('Read-ahead anonymous GET failed; inspect the safe diagnostic.') from None
        item = self.items[kind][name]
        if (type(raw) is not bytes or len(raw) != item['bytes']
                or sha256(raw).hexdigest() != item['sha256']):
            print(json.dumps(dict(phase='anonymous-readback-failure',repo_type=kind,path=name,
                                 failure_class='ByteIdentityMismatch',http_status=200),sort_keys=True),
                  file=sys.stderr,flush=True)
            raise PublicationError('Read-ahead public bytes differ from the exact plan SHA/size.')
        return raw

    def _done(self, future):
        with self._condition:
            if not future.cancelled() and future.exception() is not None:
                self._fatal = PublicationError('Read-ahead anonymous GET or byte verification failed.')
            self._condition.notify_all()

    def _submit(self, name):
        future = self._pool.submit(self._fetch, name)
        self._futures[name] = future
        future.add_done_callback(self._done)
        self.peak_futures = max(self.peak_futures, len(self._futures))

    def _fill(self):
        while (self._remaining and len(self._futures) < self.window
               and self._fatal is None and not self._closed):
            self._submit(self._remaining.popleft())

    def download(self, repo, name, kind, revision):
        if (kind not in self.rows or repo != pub.REPOS[kind] or name not in self.items[kind]
                or not isinstance(revision,str) or pub.OID.fullmatch(revision) is None):
            raise PublicationError('Read-ahead requires a declared fixed repository/path/exact revision.')
        with self._condition:
            if self._closed or self._fatal is not None:
                raise PublicationError('Read-ahead is closed or failed.')
            if self._scope is None or len(self._consumed) == len(self._names):
                if self._scope is not None:
                    self._completed.add(self._scope)
                self._new_scope(kind,name,revision)
            if self._scope[:2] != (kind,revision) or name not in self._names:
                raise PublicationError('Read-ahead cannot switch scope before all previous bytes are consumed.')
            if name in self._consumed or name in self._claimed:
                raise PublicationError('Read-ahead requests cannot consume a scoped path twice.')
            # Four publisher callers may arrive out of order. Replace an
            # unclaimed completed/queued prefetch rather than exceed the window.
            while name not in self._futures:
                if self._closed:
                    raise PublicationError('Read-ahead is closed.')
                if self._fatal is not None:
                    raise self._fatal
                if len(self._futures) >= self.window:
                    for other, future in list(self._futures.items()):
                        if other not in self._claimed and (future.done() or future.cancel()):
                            del self._futures[other]
                            self._remaining.append(other)
                            break
                    else:
                        self._condition.wait()
                        continue
                self._remaining.remove(name)
                self._submit(name)
            self._claimed.add(name)
            future = self._futures[name]
        try:
            raw = future.result()
        except Exception:
            raise PublicationError('Read-ahead anonymous GET or byte verification failed.') from None
        with self._condition:
            if self._closed:
                raise PublicationError('Read-ahead is closed.')
            if self._fatal is not None:
                raise self._fatal
            self._claimed.remove(name)
            self._consumed.add(name)
            del self._futures[name]
            self._fill()
            self._condition.notify_all()
        return raw

    __call__ = download

    def close(self):
        with self._condition:
            self._closed = True
            for future in self._futures.values():
                future.cancel()
            self._condition.notify_all()
        self._pool.shutdown(wait=True,cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *ignored):
        self.close()
