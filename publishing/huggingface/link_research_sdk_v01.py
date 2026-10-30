"""Two-file discovery overlay; anonymous staging, one named-Keychain CAS write.

This changes only the shared Space's README and index. Scientific evidence,
all prior cards, branding, metadata and every other Git/LFS object stay bound
to an exact parent tree. Unknown write replies require exact-child recovery.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import sys

import publish_usd_workstation_v01 as p
from publish import PublicationError, client

SCHEMA = 'dvidia.research-sdk-space-discovery/v1'
REPO = 'Dvidia/robot-simulation-lab'
ALLOWED = ('README.md', 'index.html')
PLAN, JOURNAL, RECEIPT = 'plan.json', 'pending.json', 'publication-receipt.json'
MARKER = b'<!-- dvidia-research-sdk-20261030 -->'
ANCHOR = b'<section aria-labelledby="workstation-title">'
IMPLEMENTATION = (Path(__file__).name, 'publish_usd_workstation_v01.py', 'publish.py', 'prepare.py')
VIEWER = 'usd-workstation/v01/index.html'
PUBLIC_VIEWER = 'https://dvidia-robot-simulation-lab.static.hf.space/'+VIEWER


def source_contract():
    return {
        'usd': dict(repo='Dvidia/usd-workstation-v01', revision='7bec8497aee2f198798ee61ce31e4e8aaaaeae80',
            path='audit.json', sha256='23a5a83bf3e5ef1b3a4ce684e301661145382a817ec1775eb512b62339d0c35d'),
        'v04': dict(repo='Dvidia/piper-grip-wait-recovery-v04', revision='4e515300adf98027560ecdba9715e420695ffdcb',
            path='benchmarks/piper-grip-wait-v04/recovery-20261030/report.json',
            sha256='acded121e3593e944b34c1cc5efde2b0d6bda511cb5ebdfffa16741243c65fe6'),
        'sdk': dict(version='0.1.2', source_commit='00f0a64b2a6e2a2bf7ee3f6deb09bacfc7b50691',
            release_url='https://github.com/Dvidia-Inference/dvidia-training/releases/tag/training-v0.1.2',
            wheel_sha256='f5b914290604e8ba553e9e26533a2467ea0703be8ec20bfd0063a9df496cf783',
            recorder_tests=27, default_branch_tests=203)}


def _implementation():
    return {name:p.digest(p.read(Path(__file__).with_name(name))) for name in IMPLEMENTATION}


def _atomic(path,value,exclusive=False):
    p.atomic(path,value,exclusive=exclusive)
    # The intent must survive a completed rename before the one remote write.
    descriptor=os.open(p.plain(path).parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def _entry(raw):
    return dict(bytes=len(raw), blob=sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(), lfs_sha256=None)


def _source_summary(sources):
    contract=source_contract()
    if set(sources)!={'usd','v04'}:raise PublicationError('Expected both exact published evidence snapshots.')
    for name,raw in sources.items():
        if sha256(raw).hexdigest()!=contract[name]['sha256']:raise PublicationError('Published evidence differs from its fixed source pin.')
    usd,v04=(json.loads(sources[name]) for name in ('usd','v04'))
    counts=dict(scheduled=45,valid=45,passed=45,normal_scheduled=36,normal_passed=36,negative_scheduled=9,negative_passed=9)
    if usd.get('mismatches')!=0 or usd.get('counts')!=counts:raise PublicationError('USD link requires all 45 audited passive physics checks.')
    wanted={'guarded':42,'clock':32,'fixed':44,'teacher':41}
    if (v04.get('schema')!='dvidia.piper-grip-wait-compact-recovery/v1' or v04.get('scheduled_attempts')!=512
            or v04.get('completed_scored_attempts')!=508 or v04.get('interrupted_unevaluable_attempts')!=4
            or v04.get('qualified_robot_skill') is not False or v04.get('physical_robot_ready') is not False
            or set(v04.get('variants',{}))!=set(wanted)):
        raise PublicationError('Grip-wait link requires the complete compact recovery scope and four unknown outcomes.')
    for name,number in wanted.items():
        row=v04['variants'][name]
        if row.get('scheduled')!=128 or row.get('success')!=number or row.get('interrupted_unevaluable')!=1:
            raise PublicationError('Grip-wait displayed counts differ from the published recovery evidence.')
    contrasts=v04.get('contrasts',[])
    if (len(contrasts)!=2 or {r.get('comparator') for r in contrasts}!={'clock','fixed'}
            or any(len(r.get('interval',[]))!=2 or not r['interval'][0]<0<r['interval'][1] for r in contrasts)):
        raise PublicationError('Both published primary grip-wait comparisons must remain inconclusive.')
    return dict(usd_passive_checks=counts,v04_scheduled_successes=wanted,v04_scheduled_per_variant=128,
        v04_unknown_outcomes=4,v04_primary_comparisons='inconclusive',sdk=contract['sdk'],
        qualified_robot_skill=False,physical_robot_ready=False)


def derive(before):
    if set(before)!=set(ALLOWED) or any(type(raw) is not bytes for raw in before.values()):
        raise PublicationError('Only the two existing root documents may be changed.')
    if (any(MARKER in raw for raw in before.values()) or before['index.html'].count(b'</main>')!=1
            or before['index.html'].count(ANCHOR)!=1):
        raise PublicationError('Discovery marker must be absent and the existing main/card boundaries must occur once.')
    sdk=source_contract()['sdk'];v04=source_contract()['v04']
    dataset=f"https://huggingface.co/datasets/{v04['repo']}/tree/{v04['revision']}"
    research='https://research.dvidia.org/papers/bounded-grip-waiting-v04/'
    body=(f'{MARKER.decode()}\n<section aria-labelledby="research-sdk-update-20261030">'
        '<h2 id="research-sdk-update-20261030">New tools and research.</h2>'
        '<h3>A workstation built from OpenUSD.</h3>'
        '<p>45 passive physics checks: 36 drop/slide cases and 9 disabled-physics controls. '
        'The browser replays recorded native poses; this is not a learned manipulation success rate.</p>'
        f'<div class="links"><a href="{VIEWER}">Inspect the recorded workstation →</a>'
        '<a href="https://huggingface.co/datasets/Dvidia/usd-workstation-v01">Download the workstation kit ↗</a></div>'
        '<h3>Does bounded grip waiting help?</h3>'
        '<p>Scheduled successes: bounded waiting 42/128, learned timing 32/128, fixed progression 44/128, '
        'authored teacher 41/128. Both primary comparisons are inconclusive. Four interrupted outcomes '
        'remain unknown; the scheduled analysis imputes them as failures.</p>'
        f'<div class="links"><a href="{dataset}">Compact evidence and interruption record ↗</a>'
        f'<a href="{research}">Read the study ↗</a></div>'
        '<h3>Training SDK 0.1.2: preserve the action evidence.</h3>'
        '<p>A data-only PiPER demonstration recorder, validator and Terminal tools. '
        '27 recorder tests are included in the 203 passing default-branch tests. '
        'No live capture, trained skill or physical robot qualification is provided.</p>'
        f'<div class="links"><a href="{sdk["release_url"]}">Download the SDK and local tools ↗</a></div>'
        '</section>\n').encode()
    readme=(f'\n\n{MARKER.decode()}\n## New tools and research\n\n'
        f'- **OpenUSD workstation:** [Recorded workstation]({PUBLIC_VIEWER}) · '
        '[Download kit](https://huggingface.co/datasets/Dvidia/usd-workstation-v01). '
        '45 passive checks: 36 normal drop/slide cases and 9 disabled-physics controls. Recorded poses, not learned manipulation success.\n'
        f'- **Bounded grip waiting v0.4:** [Compact evidence]({dataset}) · [Research]({research}). '
        'Scheduled successes: bounded waiting 42/128, learned timing 32/128, fixed progression 44/128, authored teacher 41/128. '
        'Both primary comparisons are inconclusive; four interrupted outcomes remain unknown and are failure-imputed only for the scheduled analysis.\n'
        f'- **Training SDK 0.1.2:** [GitHub release]({sdk["release_url"]}). '
        'Data-only PiPER recorder, validator and Terminal tools. The 27 recorder tests are included in 203 passing default-branch tests. '
        'No live capture, trained skill or physical robot qualification.\n').encode()
    payload={'README.md':before['README.md']+readme,'index.html':before['index.html'].replace(ANCHOR,body+ANCHOR,1)}
    for raw in payload.values():
        p.scan(raw)
        if len(raw)>1024*1024:raise PublicationError('Root documents exceed the bounded overlay size.')
    return payload


def stage(parent,output,*,api=None,downloader=None):
    if not isinstance(parent,str) or not p.OID.fullmatch(parent):raise PublicationError('An exact reviewed parent commit is required.')
    output=p.plain(output)
    if output.exists():raise PublicationError('Use a fresh discovery staging directory.')
    api=api or p.public_client()
    downloader=downloader or (lambda repo,name,kind,rev:p.download(repo,name,kind,rev,output.parent/'discovery-anonymous-cache'))
    baseline=p.tree(api,'space')
    if baseline['commit']!=parent or VIEWER not in baseline['tree']:raise PublicationError('Expected the current parent with the published USD viewer.')
    before={name:downloader(REPO,name,'space',parent) for name in ALLOWED}
    p.compare({name:baseline['tree'][name] for name in ALLOWED},{name:_entry(raw) for name,raw in before.items()})
    sources={name:downloader(row['repo'],row['path'],'dataset',row['revision']) for name,row in source_contract().items() if name!='sdk'}
    summary=_source_summary(sources);payload=derive(before)
    if p.tree(api,'space')!=baseline:raise PublicationError('Space moved during anonymous discovery staging.')
    output.mkdir(parents=True)
    for directory,items in [('before',before),('payload',payload),('sources',{name+'.json':raw for name,raw in sources.items()}),
                            ('implementation',{name:p.read(Path(__file__).with_name(name)) for name in IMPLEMENTATION})]:
        (output/directory).mkdir()
        for name,raw in items.items():(output/directory/name).write_bytes(raw)
    _atomic(output/'baseline.json',baseline,exclusive=True)
    plan=dict(schema=SCHEMA,repo_id=REPO,repo_type='space',parent=parent,allowed_paths=list(ALLOWED),
        baseline=p.digest(p.read(output/'baseline.json',p.MAX_PLAN)),source_contract=source_contract(),summary=summary,
        before={name:p.digest(raw) for name,raw in before.items()},payload={name:p.digest(raw) for name,raw in payload.items()},
        implementation=_implementation(),qualified_robot_skill=False,physical_robot_ready=False)
    _atomic(output/PLAN,plan,exclusive=True);checked(output/PLAN);return plan


def checked(path):
    path=p.plain(path);plan=p.load(path)
    if (plan.get('schema')!=SCHEMA or plan.get('repo_id')!=REPO or plan.get('repo_type')!='space'
            or plan.get('allowed_paths')!=list(ALLOWED) or not p.OID.fullmatch(plan.get('parent',''))
            or plan.get('source_contract')!=source_contract() or plan.get('implementation')!=_implementation()
            or plan.get('qualified_robot_skill') is not False or plan.get('physical_robot_ready') is not False):
        raise PublicationError('Discovery plan destination, fixed sources or implementation changed.')
    baseline_raw=p.read(path.parent/'baseline.json',p.MAX_PLAN);baseline=json.loads(baseline_raw)
    if p.digest(baseline_raw)!=plan.get('baseline') or baseline.get('commit')!=plan['parent']:
        raise PublicationError('Discovery baseline differs from its exact parent seal.')
    if not isinstance(baseline.get('tree'),dict) or VIEWER not in baseline['tree']:
        raise PublicationError('Discovery baseline lacks the published USD viewer.')
    collections={}
    for directory,names in [('before',ALLOWED),('payload',ALLOWED),('sources',('usd.json','v04.json')),('implementation',IMPLEMENTATION)]:
        folder=p.plain(path.parent/directory)
        if {x.name for x in folder.iterdir()}!=set(names):raise PublicationError('Staged discovery file roster changed.')
        collections[directory]={name:p.read(folder/name) for name in names}
    for name,raw in collections['implementation'].items():
        if p.digest(raw)!=plan['implementation'][name]:raise PublicationError('Staged implementation snapshot changed.')
    before,payload=collections['before'],collections['payload']
    for name in ALLOWED:
        if p.digest(before[name])!=plan['before'].get(name) or p.digest(payload[name])!=plan['payload'].get(name):
            raise PublicationError('Staged root document bytes changed.')
    p.compare({name:baseline['tree'][name] for name in ALLOWED},{name:_entry(raw) for name,raw in before.items()})
    if payload!=derive(before) or plan['summary']!=_source_summary({name:collections['sources'][name+'.json'] for name in ('usd','v04')}):
        raise PublicationError('Discovery text or evidence summary no longer derives exactly.')
    return plan,baseline,payload


@contextmanager
def _lock(folder):
    path=p.plain(folder)/'.discovery-publisher.lock';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:yield
    finally:os.close(fd);path.unlink()


def _wanted(baseline,payload):return dict(baseline['tree'],**{name:_entry(raw) for name,raw in payload.items()})


def _readback(api,downloader,plan,baseline,payload,revision):
    current=p.tree(api,'space')
    if current['commit']!=revision:raise PublicationError('Space moved before anonymous overlay readback.')
    p.compare(current['tree'],_wanted(baseline,payload))
    for name,raw in payload.items():
        if downloader(REPO,name,'space',revision)!=raw:raise PublicationError('Anonymous pinned root bytes differ from the reviewed overlay.')
    if p.tree(api,'space')!=current:raise PublicationError('Space moved during anonymous overlay readback.')
    return current


def publish(path,*,api=None,downloader=None,operation_cls=None,resume_unchanged_parent=False):
    path=p.plain(path)
    with _lock(path.parent):
        plan,baseline,payload=checked(path);plan_id=p.digest(p.read(path,p.MAX_PLAN));pending=path.parent/JOURNAL
        message='DVIDIA research and SDK discovery '+plan_id['sha256']
        intent=dict(schema=SCHEMA+'/intent',plan=plan_id,parent=plan['parent'],message=message)
        if pending.exists() and p.load(pending)!=intent:raise PublicationError('Existing pending intent binds another reviewed overlay.')
        if (path.parent/RECEIPT).exists():return verify(path,api=api,downloader=downloader)
        token=None
        if api is None:
            api,token=client(path.parent/'publisher-cache')
            if 'dvidia' not in {str(row.get('name','')).lower() for row in api.whoami().get('orgs',[])}:
                raise PublicationError('Named publishing identity is outside Dvidia.')
        downloader=downloader or (lambda repo,name,kind,rev:p.download(repo,name,kind,rev,path.parent/'anonymous-cache'))
        if token and any(token.encode() in raw for raw in payload.values()):raise PublicationError('Publishing credential appeared in overlay bytes.')
        current=p.tree(api,'space');reconciled=False
        if pending.exists() and current['commit']==plan['parent'] and not resume_unchanged_parent:
            raise PublicationError('Pending reply still has the unchanged parent; review before an explicit guarded resume.')
        if pending.exists() and current['commit']!=plan['parent']:
            history=api.list_repo_commits(REPO,repo_type='space',revision=current['commit'],token=False)
            if (len(history)<2 or history[0].commit_id!=current['commit'] or history[0].title!=message or history[1].commit_id!=plan['parent']):
                raise PublicationError('Unknown reply is not the exact intended immediate child; no new write attempted.')
            revision=current['commit'];reconciled=True
        else:
            if current!=baseline or p.tree(api,'space')!=baseline:raise PublicationError('Space parent moved outside the exact reviewed tree.')
            if operation_cls is None:
                from huggingface_hub import CommitOperationAdd
                operation_cls=CommitOperationAdd
            operations=[]
            for name in ALLOWED:
                operation=operation_cls(path_in_repo=name,path_or_fileobj=payload[name]);operation._upload_mode='regular';operation._should_ignore=False
                operations.append(operation)
            checked(path)
            if p.digest(p.read(path,p.MAX_PLAN))!=plan_id:raise PublicationError('Discovery plan changed before upload.')
            if not pending.exists():_atomic(pending,intent,exclusive=True)
            try:
                reply=api.create_commit(REPO,repo_type='space',parent_commit=plan['parent'],operations=operations,commit_message=message)
            except Exception:raise PublicationError('Commit reply uncertain; preserve pending intent and review before resuming.') from None
            revision=reply.oid
        if not isinstance(revision,str) or not p.OID.fullmatch(revision) or revision==plan['parent']:
            raise PublicationError('Commit lacks a new exact revision; retain pending intent.')
        _readback(api,downloader,plan,baseline,payload,revision)
        receipt=dict(schema=SCHEMA+'/receipt',plan=plan_id,repo_id=REPO,repo_type='space',parent=plan['parent'],commit=revision,
            files={name:p.digest(raw) for name,raw in payload.items()},unchanged_git_lfs_files=len(baseline['tree'])-2,
            anonymous_readback=True,reconciled=reconciled,qualified_robot_skill=False,physical_robot_ready=False)
        _atomic(path.parent/RECEIPT,receipt,exclusive=True);return receipt


def verify(path,*,api=None,downloader=None):
    path=p.plain(path);plan,baseline,payload=checked(path);receipt=p.load(path.parent/RECEIPT)
    if (receipt.get('schema')!=SCHEMA+'/receipt' or receipt.get('plan')!=p.digest(p.read(path,p.MAX_PLAN))
            or receipt.get('repo_id')!=REPO or receipt.get('repo_type')!='space' or receipt.get('parent')!=plan['parent']
            or not p.OID.fullmatch(receipt.get('commit','')) or receipt.get('commit')==plan['parent']
            or receipt.get('files')!={name:p.digest(raw) for name,raw in payload.items()}
            or receipt.get('unchanged_git_lfs_files')!=len(baseline['tree'])-2 or receipt.get('anonymous_readback') is not True
            or receipt.get('qualified_robot_skill') is not False or receipt.get('physical_robot_ready') is not False):
        raise PublicationError('Completed overlay receipt differs from its reviewed plan.')
    api=api or p.public_client()
    downloader=downloader or (lambda repo,name,kind,rev:p.download(repo,name,kind,rev,path.parent/'anonymous-cache'))
    _readback(api,downloader,plan,baseline,payload,receipt['commit']);return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('stage','check','publish','verify'))
    parser.add_argument('--parent');parser.add_argument('--output');parser.add_argument('--plan')
    parser.add_argument('--resume-unchanged-parent',action='store_true',help='Explicitly reviewed resume; retains the same intent and parent CAS.')
    args=parser.parse_args(argv)
    try:
        if args.command=='stage':result=stage(args.parent,args.output)
        elif args.command=='check':result=checked(args.plan)[0]
        elif args.command=='publish':result=publish(args.plan,resume_unchanged_parent=args.resume_unchanged_parent)
        else:result=verify(args.plan)
        print(json.dumps(dict(schema=result['schema'],completed=True,commit=result.get('commit'),parent=result.get('parent'))));return 0
    except Exception as error:
        print('Discovery overlay stopped: '+(str(error) if isinstance(error,PublicationError) else type(error).__name__),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
