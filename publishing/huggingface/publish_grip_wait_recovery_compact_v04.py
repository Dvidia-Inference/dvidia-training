"""Publish only reviewed compact v0.4 recovery summaries to one public dataset.

stage/check/verify are anonymous. publish obtains its credential exclusively
through the existing named macOS Keychain client, never login or environment
tokens. Exact parent commits, preservation of existing files, durable intents,
and anonymous pinned byte readback protect this small additive publication.
The full raw study/native archive and model weights are out of scope.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha1,sha256
import json
import logging
import os
from pathlib import Path,PurePosixPath
import re
import sys
import tempfile

from publish import client,PublicationError

REPO='Dvidia/piper-grip-wait-recovery-v04'
KIND='dataset'
SCHEMA='dvidia.grip-wait-compact-publication/v1'
OID=re.compile(r'[a-f0-9]{40}\Z')
HEX256=re.compile(r'[a-f0-9]{64}\Z')
PREFIX='benchmarks/piper-grip-wait-v04/recovery-20261030/'
COMPACT={
    'README.md':'9bd0b69e4e19c5e5fdaa61bc419edeb5df905e6f50a11803db536f3244792c3a',
    'report.json':'acded121e3593e944b34c1cc5efde2b0d6bda511cb5ebdfffa16741243c65fe6',
    'audit-summary.json':'a1cb4dd0a35228fd738a2881c50f6791d5fc799ac5b8f3b3cb27dd2797ad7541',
    'interruptions.json':'5e7724978a532d802bcefa63144dab9d180ebc41f9a7c3bbc64af895bea2e403',
    'source-lineage.json':'30a6c8f4fde7fb9d006d906f67ee1d66798a2a1cd059a187f3b7a74c3a6d6d9f'}
PAPER='reports/Bounded grip waiting v04.md'
PAPER_SHA='2557ac42b4ff2da720700185e62620a75ca835cb34fd000f4b470e587048dc50'
RAW_SHA='87d252d2ffe2aea5a1bca6b50d926abbd2181b4f51695471dfb5747af91e4e56'
SUCCESSES=dict(guarded=42,clock=32,fixed=44,teacher=41)
MAX_FILE=2*1024*1024
MAX_TOTAL=4*1024*1024
SECRET=re.compile(rb'(?:hf_|gh[pousr]_|github_pat_|apikey_)[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
PRIVATE_PATH=re.compile(rb'/(?:Users|private|var|tmp)/')


def plain(path):
    path=Path(path).absolute()
    if any(p.is_symlink() and str(p) not in ('/var','/tmp') for p in (path,*path.parents)):
        raise PublicationError('Publication paths cannot traverse artifact symlinks.')
    return path.resolve()


def relative(name):
    if not isinstance(name,str):raise PublicationError('Expected a plain artifact-relative path.')
    p=PurePosixPath(name)
    if (not name or p.is_absolute() or str(p)!=name or '\\' in name or ':' in name
            or any(part in ('','..','.') or part.startswith('.') for part in p.parts)
            or any(ord(c)<32 for c in name)):
        raise PublicationError('Expected a plain artifact-relative path.')
    return p


def raw(path):
    path=plain(path)
    if not path.is_file() or path.stat().st_size>MAX_FILE:
        raise PublicationError('Missing or oversized compact artifact.')
    return path.read_bytes()


def digest(body):return dict(bytes=len(body),sha256=sha256(body).hexdigest())


def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise PublicationError('Duplicate JSON keys are not allowed.')
        result[key]=value
    return result


def load(path):
    return json.loads(raw(path),object_pairs_hook=_pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(PublicationError('Nonfinite JSON is not allowed.')))


def write(path,value):
    path=plain(path);path.parent.mkdir(parents=True,exist_ok=True)
    body=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.compact-writing-',delete=False) as stream:
            temporary=Path(stream.name);stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def implementation():
    return {name:digest(raw(Path(__file__).with_name(name))) for name in (Path(__file__).name,'publish.py','prepare.py')}


def semantic(root):
    scope=load(root/'scope.json');report=load(root/PREFIX/'report.json')
    expected=dict(schema='dvidia.piper-grip-wait-compact-release/v1',repo_id=REPO,repo_type=KIND,
        scheduled_attempts=512,completed_scored_attempts=508,interrupted_unevaluable_attempts=4,
        full_raw_native_archive_uploaded=False,full_raw_report_uploaded=False,
        full_raw_native_evidence_retained_locally=True,training_weights_included=False,newly_trained_model=False,
        authored_acquisition_supervisor=True,learned_prediction_advisory=True,
        physical_robot_ready=False,qualified_robot_skill=False,successes=SUCCESSES,full_raw_report_sha256=RAW_SHA)
    if any(scope.get(key)!=value for key,value in expected.items()):
        raise PublicationError('Compact release scope or unknown-outcome declarations changed.')
    for name,want in COMPACT.items():
        if digest(raw(root/PREFIX/name))['sha256']!=want:
            raise PublicationError('An original compact recovery artifact changed.')
    if digest(raw(root/PAPER))['sha256']!=PAPER_SHA:
        raise PublicationError('The reviewed research note changed.')
    rows=report.get('rows',[])
    if (len(rows)!=512 or len({(r['case_id'],r['variant']) for r in rows})!=512
            or report.get('qualified_robot_skill') is not False or report.get('physical_robot_ready') is not False
            or report.get('valid_scored_attempts')!=508 or report.get('all_512_valid') is not False):
        raise PublicationError('Scheduled, scored and unknown outcomes must remain explicit.')
    unknown=[r for r in rows if r.get('interrupted_unevaluable')]
    if (len(unknown)!=4 or {r['variant'] for r in unknown}!=set(SUCCESSES)
            or any(r['case_id']!='confirmation-xy04-g1-nominal' or r.get('valid') is not False
                or r.get('success') is not False
                or r.get('raw_strict_success') is not None or r.get('native_summary') is not None
                or r.get('post_grip') is not None or r.get('success_is_failure_imputation') is not True for r in unknown)):
        raise PublicationError('Interrupted outcomes cannot be relabeled as observed failures or successes.')
    for variant,count in SUCCESSES.items():
        own=[r for r in rows if r['variant']==variant]
        if len(own)!=128 or sum(bool(r['valid']) for r in own)!=127 or sum(bool(r['success']) for r in own)!=count:
            raise PublicationError('Frozen controller denominators or successes changed.')
    lineage=load(root/PREFIX/'source-lineage.json');audit=load(root/PREFIX/'audit-summary.json')
    required_sources={
        'source/piper_grip_wait_v04.py':lineage['frozen_runtime_sha256']['piper_grip_wait_v04.py'],
        'source/piper_grip_wait_v04_benchmark.py':lineage['frozen_runtime_sha256']['piper_grip_wait_v04_benchmark.py'],
        'source/recover_piper_grip_wait_v04.py':report['driver_sha256'],
        'source/audit_piper_grip_wait_v04.py':report['independent_auditor_sha256'],
        'source/audit_piper_grip_clock_v03.py':audit['dependency_auditor_sha256']}
    snapshots=scope.get('sources',{})
    if not set(required_sources)<=set(snapshots):raise PublicationError('Required scoped source snapshots are missing.')
    for name,identity in snapshots.items():
        if not name.startswith('source/') or not name.endswith('.py'):
            raise PublicationError('Source snapshots must be explicitly scoped Python files.')
        value=digest(raw(root/relative(name)))
        if value!={key:identity.get(key) for key in ('bytes','sha256')}:
            raise PublicationError('Source snapshot identity differs from its scope manifest.')
        if name in required_sources and value['sha256']!=required_sources[name]:
            raise PublicationError('Source snapshot differs from the audited frozen source.')
    return scope


def files(root):
    root=plain(root)
    if not root.is_dir():raise PublicationError('Expected a compact payload directory.')
    result=[]
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise PublicationError('Compact payloads cannot contain symlinks.')
        if not path.is_file():continue
        name=path.relative_to(root).as_posix();relative(name)
        if path.suffix not in ('.md','.json','.py','.txt') and name!='LICENSE':
            raise PublicationError('This compact release publishes text only, without arrays, archives or weights.')
        body=raw(path)
        if SECRET.search(body) or PRIVATE_PATH.search(body):
            raise PublicationError('Public artifact contains a credential-shaped literal or private absolute path.')
        try:body.decode('utf-8')
        except UnicodeDecodeError:raise PublicationError('Compact artifacts must be UTF-8 text.') from None
        result.append(dict(path=name,**digest(body)))
    if not 10<=len(result)<=48 or sum(x['bytes'] for x in result)>MAX_TOTAL:
        raise PublicationError('Compact publication exceeds the reviewed file or size limit.')
    required={'README.md','LICENSE','NOTICE.txt','scope.json','integrity.json','source/README.md',PAPER}|{PREFIX+name for name in COMPACT}
    if not required<={x['path'] for x in result}:raise PublicationError('Required compact scope, licensing and evidence are missing.')
    roster=load(root/'integrity.json').get('files')
    expected={x['path']:{k:x[k] for k in ('bytes','sha256')} for x in result if x['path']!='integrity.json'}
    if roster!=expected:raise PublicationError('Compact integrity manifest differs from the exact payload inventory.')
    semantic(root)
    return result


def anonymous_api():
    # Explicit anonymous requests plus SDK configuration prevent implicit local
    # login/environment tokens. The named Keychain is never read here.
    for name in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):os.environ.pop(name,None)
    os.environ.update(HF_DEBUG='0',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1',HF_ENDPOINT='https://huggingface.co')
    logging.disable(logging.CRITICAL)
    from huggingface_hub import HfApi
    return HfApi(endpoint='https://huggingface.co',token=False)


def remote(api,revision=None,*,missing=False):
    from huggingface_hub.errors import RepositoryNotFoundError
    try:info=api.repo_info(REPO,repo_type=KIND,revision=revision,files_metadata=True,token=False)
    except RepositoryNotFoundError:
        if missing:return None,{}
        raise PublicationError('The planned public compact dataset is unavailable.') from None
    if info.private is not False or not OID.fullmatch(info.sha or ''):
        raise PublicationError('Expected a public fixed dataset at a pinned commit.')
    result={}
    for entry in info.siblings:
        name=entry.rfilename
        # A pre-existing .gitattributes file is preserved, never uploaded or edited.
        if name!='.gitattributes':relative(name)
        if name in result or type(entry.size) is not int or entry.size<0:
            raise PublicationError('Remote inventory is incomplete or ambiguous.')
        lfs=entry.lfs.sha256 if entry.lfs else None
        if not OID.fullmatch(entry.blob_id or '') or lfs is not None and not HEX256.fullmatch(lfs):
            raise PublicationError('Remote Git/LFS identities are incomplete.')
        result[name]=dict(bytes=entry.size,blob=entry.blob_id,lfs_sha256=lfs)
    return info.sha,result


def git_identity(body):
    return dict(bytes=len(body),blob=sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest(),lfs_sha256=None)


def expected_tree(root,items,baseline):
    expected=dict(baseline)
    for item in items:
        body=raw(root/relative(item['path']))
        if digest(body)!={key:item[key] for key in ('bytes','sha256')}:
            raise PublicationError('Reviewed compact bytes changed.')
        identity=git_identity(body)
        if item['path'] in baseline and baseline[item['path']]!=identity:
            raise PublicationError('Refusing to replace a pre-existing path with different bytes.')
        expected[item['path']]=identity
    return expected


def stage(payload,output,*,api=None):
    root,output=plain(payload),plain(output)
    if output.exists():raise PublicationError('Use a fresh compact publication plan directory.')
    items=files(root);api=api or anonymous_api();parent,baseline=remote(api,missing=True)
    expected_tree(root,items,baseline)
    plan=dict(schema=SCHEMA,repo_id=REPO,repo_type=KIND,public=True,payload_root=str(root),
        expected_parent=parent,baseline=baseline,files=items,implementation=implementation(),
        full_raw_native_archive_uploaded=False,training_weights_included=False,
        qualified_robot_skill=False,physical_robot_ready=False)
    output.mkdir(parents=True);write(output/'plan.json',plan)
    changes=dict(schema=SCHEMA+'/review',repo_id=REPO,expected_parent=parent,
        repository_creation_required=parent is None,add=[item['path'] for item in items if item['path'] not in baseline],
        identical_existing=[item['path'] for item in items if item['path'] in baseline],
        preserve_existing=sorted(set(baseline)-{x['path'] for x in items}),overwrite=[],delete=[],
        file_count=len(items),total_bytes=sum(x['bytes'] for x in items),
        publication_executed=False,credentials_accessed=False,scope=semantic(root))
    write(output/'review.json',changes)
    return plan


def checked(path):
    path=plain(path);plan=load(path)
    if (plan.get('schema')!=SCHEMA or plan.get('repo_id')!=REPO or plan.get('repo_type')!=KIND
            or plan.get('public') is not True or plan.get('implementation')!=implementation()
            or plan.get('full_raw_native_archive_uploaded') is not False or plan.get('training_weights_included') is not False
            or plan.get('qualified_robot_skill') is not False or plan.get('physical_robot_ready') is not False
            or plan.get('expected_parent') is not None and not OID.fullmatch(plan['expected_parent'])):
        raise PublicationError('Plan differs from the fixed reviewed compact dataset contract.')
    root=plain(plan['payload_root'])
    if files(root)!=plan['files']:raise PublicationError('Staged compact payload bytes changed.')
    expected_tree(root,plan['files'],plan['baseline'])
    return plan


@contextmanager
def locked(folder):
    import fcntl
    path=plain(folder)/'.compact-publisher.lock'
    if path.is_symlink():raise PublicationError('Publisher lock cannot be a symlink.')
    descriptor=os.open(path,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
    finally:os.close(descriptor)


def publish(path):
    path=plain(path)
    with locked(path.parent):
        plan=checked(path);plan_sha=digest(raw(path))['sha256'];root=plain(plan['payload_root'])
        journal=path.parent/'journal.json'
        state=load(journal) if journal.exists() else dict(schema=SCHEMA,plan_sha256=plan_sha,status='ready',
            parent=plan['expected_parent'],baseline=plan['baseline'],commit=None)
        if state.get('schema')!=SCHEMA or state.get('plan_sha256')!=plan_sha:
            raise PublicationError('Existing durable intent binds a different compact plan.')
        # The only credential path is the existing fixed named Keychain client.
        api,token=client(path.parent/'publisher-cache')
        bodies={item['path']:raw(root/item['path']) for item in plan['files']}
        if any(token.encode() in body for body in bodies.values()):
            raise PublicationError('Publishing credential occurs in staged bytes; no write was attempted.')
        if 'dvidia' not in {str(org.get('name','')).lower() for org in api.whoami().get('orgs',[])}:
            raise PublicationError('The named publisher is not a member of Dvidia.')
        head,current=remote(api,missing=True)
        if state['parent'] is None:
            if state['status']!='creating' and head is not None:
                raise PublicationError('Dataset appeared after staging; review a fresh expected-parent plan.')
            if head is None:
                state['status']='creating';write(journal,state)
                try:api.create_repo(REPO,repo_type=KIND,private=False,exist_ok=False)
                except Exception:raise PublicationError('Repository creation reply uncertain; preserve the durable intent.') from None
                head,current=remote(api)
            if set(current)-{'.gitattributes'}:
                raise PublicationError('Uncertain dataset creation contains unexpected files.')
            state.update(parent=head,baseline=current,status='ready');write(journal,state)
        expected=expected_tree(root,plan['files'],state['baseline'])
        title='DVIDIA compact grip-wait recovery v0.4 '+plan_sha
        if state['commit'] is not None:
            if head!=state['commit'] or current!=expected:
                raise PublicationError('Previously published compact dataset moved or changed.')
        elif state['status']=='pending' and head!=state['parent']:
            history=api.list_repo_commits(REPO,repo_type=KIND,revision=head,token=False)
            if (len(history)<2 or history[0].commit_id!=head or history[0].title!=title
                    or history[1].commit_id!=state['parent'] or current!=expected):
                raise PublicationError('Uncertain write is not the exact reviewed immediate child.')
            state.update(commit=head,status='committed');write(journal,state)
        else:
            if head!=state['parent'] or current!=state['baseline']:
                raise PublicationError('Expected parent or preserved destination files changed.')
            checked(path)
            from huggingface_hub import CommitOperationAdd
            operations=[]
            for item in plan['files']:
                body=bodies[item['path']]
                if digest(body)!={key:item[key] for key in ('bytes','sha256')}:
                    raise PublicationError('Frozen upload bytes differ from the reviewed inventory.')
                if item['path'] in state['baseline']:continue  # Identical files are preserved, not rewritten.
                operation=CommitOperationAdd(path_in_repo=item['path'],path_or_fileobj=body)
                operation._upload_mode='regular';operation._should_ignore=False;operations.append(operation)
            if not operations:
                state.update(commit=head,status='committed');write(journal,state)
            else:
                state['status']='pending';write(journal,state)
                try:reply=api.create_commit(REPO,repo_type=KIND,parent_commit=head,operations=operations,commit_message=title)
                except Exception:raise PublicationError('Commit reply uncertain; preserve the durable parent-bound intent.') from None
                head,current=remote(api)
                if head!=reply.oid or current!=expected:
                    raise PublicationError('Public child or full preserved Git tree differs from the reviewed change.')
                state.update(commit=head,status='committed');write(journal,state)
        checked(path)
        receipt=dict(schema=SCHEMA+'/receipt',plan_sha256=plan_sha,repo_id=REPO,repo_type=KIND,
            parent=state['parent'],commit=state['commit'],baseline=state['baseline'],files=plan['files'],
            all_public_git_identities_verified=True,anonymous_full_readback=False,
            full_raw_native_archive_uploaded=False,training_weights_included=False,
            qualified_robot_skill=False,physical_robot_ready=False)
        write(path.parent/'publication-receipt.json',receipt)
        return receipt


def verify(path,*,api=None,downloader=None):
    path=plain(path);plan=checked(path);receipt=load(path.parent/'publication-receipt.json')
    if receipt.get('plan_sha256')!=digest(raw(path))['sha256'] or receipt.get('repo_id')!=REPO:
        raise PublicationError('Publication receipt binds a different compact plan.')
    revision=receipt.get('commit')
    if not OID.fullmatch(revision or ''):raise PublicationError('Expected a pinned public compact revision.')
    api=api or anonymous_api()
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader=hf_hub_download
    head,current=remote(api)
    expected=expected_tree(plain(plan['payload_root']),plan['files'],receipt['baseline'])
    if head!=revision or current!=expected:raise PublicationError('Dataset moved before anonymous readback.')
    def one(item):
        result=downloader(REPO,item['path'],repo_type=KIND,revision=revision,token=False,
            endpoint='https://huggingface.co',cache_dir=str(path.parent/'anonymous-cache'),force_download=True)
        body=Path(result).read_bytes()
        if digest(body)!={key:item[key] for key in ('bytes','sha256')}:
            raise PublicationError('Pinned anonymous file bytes differ from the reviewed compact payload.')
        return dict(path=item['path'],**digest(body))
    with ThreadPoolExecutor(max_workers=4) as pool:proof=list(pool.map(one,plan['files']))
    head,current=remote(api)
    if head!=revision or current!=expected:raise PublicationError('Dataset moved during anonymous readback.')
    checked(path)
    result=dict(schema=SCHEMA+'/anonymous-readback',plan_sha256=receipt['plan_sha256'],repo_id=REPO,
        repo_type=KIND,commit=revision,files=proof,anonymous_full_readback=True,
        preserved_existing_files_unchanged=True,full_raw_native_archive_uploaded=False,
        training_weights_included=False,qualified_robot_skill=False,physical_robot_ready=False)
    write(path.parent/'anonymous-readback.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('stage','check','publish','verify'))
    parser.add_argument('--payload');parser.add_argument('--output');parser.add_argument('--plan')
    args=parser.parse_args(argv)
    try:
        if args.command=='stage':result=stage(args.payload,args.output)
        elif args.command=='check':result=checked(args.plan)
        elif args.command=='publish':result=publish(args.plan)
        else:result=verify(args.plan)
        print(json.dumps(dict(schema=result['schema'],command=args.command,completed=True,repo_id=REPO)))
        return 0
    except Exception as error:
        # API exceptions can contain authentication/request details. Never echo them.
        print('Compact publication stopped: '+(str(error) if isinstance(error,PublicationError) else type(error).__name__),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
