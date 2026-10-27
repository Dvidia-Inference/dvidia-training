"""Strict, resumable post-publication hosting overlay for one fixed public Space.

Stage/check are offline. Only publish uses the named Keychain publisher client.
The original continuation and all of its implementation-bound modules are read
without modification. Native resources are preflighted as regular Git content,
then the exact derived viewer and root links activate in one guarded commit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fnmatch
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import derive_piper_feedback_native_resources as native
import publish_piper_feedback as study
import publish_piper_feedback_continuation as continuation
import publish_piper_feedback_dataset_metadata as metadata
import publish_piper_feedback_hosting as hosting
from publish import PublicationError, checked_remote, client
from publish_piper_imitation_batches import _matches
import capture_piper_feedback_baselines as public


SCHEMA = 'dvidia.piper-feedback-native-space-overlay/v1'
ORIGINAL_VIEWER_SHA = '2731ec8df376fde096ed2ff30540fd4de1fabd698fb1a1be650345d5bdf73d04'
PREFIX = study.SPACE_NAMESPACE+'/viewer/'
ACTIVATION = {PREFIX+name for name in ('index.html','piper_feedback_viewer.js','piper_feedback_viewer.bundle.js','motion.json','viewer-integrity.json')} | {'index.html','README.md'}
ASSET_CAP, ACTIVATION_CAP, FILE_CAP = 8*1024*1024, 16*1024*1024, 25
PLAN, JOURNAL, RECEIPT = 'native-space-plan.json', 'native-space-journal.json', 'native-space-publication-receipt.json'
ROOT = Path(__file__).resolve().parent
IMPLEMENTATION = tuple(continuation.IMPLEMENTATION_FILES)+(
    'derive_piper_feedback_geometry_chunks.py','derive_piper_feedback_native_resources.py',
    'piper_feedback_native_resources.js','publish_piper_feedback_native_space.py')


def _entry(raw):
    return dict(blob=sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),size=len(raw),lfs_sha256=None)


def _implementation():
    return {name:study._receipt(study._read(ROOT/name)) for name in IMPLEMENTATION}


def _implementation_check(expected, snapshot):
    if expected != _implementation():
        raise PublicationError('Reviewed overlay implementation changed.')
    names={p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()}
    if names != set(IMPLEMENTATION):
        raise PublicationError('Overlay implementation snapshot inventory changed.')
    for name,receipt in expected.items():
        if study._receipt(study._read(snapshot/name)) != receipt:
            raise PublicationError('Overlay implementation snapshot bytes changed.')


def _receipt_shape(value):
    return (isinstance(value,dict) and set(value)=={'bytes','sha256'} and type(value['bytes']) is int and value['bytes']>0
            and isinstance(value['sha256'],str) and study.SHA.fullmatch(value['sha256']) is not None)


def _completed(raw, historical_raw, historical_files, space_commit, dataset_commit, card_raw, original):
    value=study._json(raw)
    rows=value.get('repositories',[]) if isinstance(value,dict) else []
    datasets=[r for r in rows if isinstance(r,dict) and r.get('repo_type')=='dataset']
    if len(datasets)!=1:
        raise PublicationError('A final continuation publication receipt is required.')
    receipt,_=metadata._completed_receipt(raw,datasets[0]['commit'])
    c=receipt.get('continuation')
    if (not isinstance(c,dict) or c.get('schema')!=continuation.SCHEMA
            or not all(_receipt_shape(c.get(k)) for k in ('plan','immutable_original_journal','boundary','journal'))
            or not isinstance(c.get('history'),dict) or set(c['history'])!=set(study.REPOSITORIES)):
        raise PublicationError('Receipt does not establish a completed continuation history.')
    historical=study._json(historical_raw)
    if (not isinstance(historical,dict) or historical.get('schema')!=continuation.SCHEMA
            or c['plan']!=study._receipt(historical_raw) or c.get('implementation')!=historical.get('implementation')
            or c['immutable_original_journal']!=historical.get('original_journal')
            or c['boundary']!=historical.get('boundary')
            or historical.get('original_plan',{}).get('sha256')!=receipt['plan_sha256']):
        raise PublicationError('Historical continuation plan differs from its exact final receipt.')
    implementation=historical.get('implementation')
    if (not isinstance(implementation,dict) or set(implementation)!={'schema','files'}
            or implementation['schema']!=continuation.SCHEMA+'/implementation'
            or not isinstance(implementation['files'],dict) or set(implementation['files'])!=set(continuation.IMPLEMENTATION_FILES)
            or set(historical_files)!=set(continuation.IMPLEMENTATION_FILES)):
        raise PublicationError('Historical continuation implementation inventory changed.')
    for name,entry in implementation['files'].items():
        if not _receipt_shape(entry) or study._receipt(historical_files[name])!=entry:
            raise PublicationError('Historical continuation implementation snapshot changed.')
    for row in receipt['repositories']:
        history=c['history'][row['repo_type']]
        if not isinstance(history,dict) or set(history)!={'original_prefix_commits','continuation_commits'}:
            raise PublicationError('Malformed completed continuation history.')
        commits=[]
        for key in ('original_prefix_commits','continuation_commits'):
            if not isinstance(history[key],list):raise PublicationError('Malformed continuation commit list.')
            commits.extend(history[key])
        if (not commits or len(set(commits))!=len(commits) or commits[-1]!=row['commit']
                or any(not isinstance(v,str) or study.COMMIT.fullmatch(v) is None for v in commits)):
            raise PublicationError('Continuation final revision differs from its complete history.')
    space=next(r for r in receipt['repositories'] if r['repo_type']=='space')
    if space['commit']!=space_commit:
        raise PublicationError('Space baseline must be the exact final continuation commit.')
    for name in ('protocol','report','frozen'):
        if sha256(original[name+'.json']).hexdigest()!=receipt['bindings'][name+'_sha256']:
            raise PublicationError('Original viewer science differs from the continuation receipt.')
    initial_dataset=datasets[0]['commit']
    if dataset_commit==initial_dataset:
        if card_raw is not None:raise PublicationError('Unexpected dataset metadata receipt for an unchanged dataset commit.')
    else:
        card=study._json(card_raw) if card_raw is not None else None
        if (not isinstance(card,dict) or card.get('schema')!=metadata.SCHEMA+'/receipt'
                or card.get('repo_id')!=study.DATASET_REPO or card.get('parent_commit')!=initial_dataset
                or card.get('commit')!=dataset_commit or card.get('anonymous_before_and_after_readback') is not True
                or card.get('scientific_seal_unchanged') is not True or card.get('physical_robot_ready') is not False
                or card.get('original_scientific_integrity_sha256')!=receipt['source_integrity_sha256']
                or card.get('changed_paths')!=sorted(metadata.ALLOWED)):
            raise PublicationError('Final dataset commit must bind the exact verified metadata-overlay receipt.')
    return receipt


def _attributes(raw, tree, payload):
    if '.gitattributes' not in tree or not _matches(raw,tree['.gitattributes']) or tree['.gitattributes']['lfs_sha256'] is not None:
        raise PublicationError('Root attributes require exact regular-Git baseline bytes.')
    for name in tree:
        if name.endswith('/.gitattributes') and any(path.startswith(name.rsplit('/',1)[0]+'/') for path in payload):
            raise PublicationError('A nested attributes override affects reviewed hosting files.')
    for line in raw.decode('utf8').splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        parts=line.split()
        if not parts or parts[0].startswith(('[','/','!')) or any(c in parts[0] for c in ('"','\\')):
            raise PublicationError('Unsupported attributes grammar requires explicit review.')
        if 'filter=lfs' not in parts[1:]:continue
        pattern=parts[0]
        if any(fnmatch.fnmatchcase(path if '/' in pattern else path.rsplit('/',1)[-1],pattern) for path in payload):
            raise PublicationError('Attributes would assign a hosting text/script file to LFS.')


def _groups(payload, probe):
    groups=[dict(kind='regular-git-preflight',paths=[probe])]
    assets=sorted(set(payload)-ACTIVATION-{probe})
    batch=[];size=0
    for name in assets:
        amount=len(payload[name])
        if batch and (len(batch)>=FILE_CAP or size+amount>ASSET_CAP):
            groups.append(dict(kind='assets',paths=batch));batch=[];size=0
        batch.append(name);size+=amount
    if batch:groups.append(dict(kind='assets',paths=batch))
    groups.append(dict(kind='activation',paths=sorted(ACTIVATION)))
    for index,group in enumerate(groups):
        group.update(index=index,bytes=sum(len(payload[n]) for n in group['paths']),cap=ACTIVATION_CAP if group['kind']=='activation' else ASSET_CAP)
        if len(group['paths'])>FILE_CAP or group['bytes']>group['cap']:
            raise PublicationError('Overlay commit exceeds its declared activation/asset bound.')
    return groups


def _compose(original_dir, derived_dir, expected_derived, baseline_raw, roots, receipt_raw, historical_raw, historical_files, card_raw, dataset_commit, space_commit, output):
    hosting._commit(dataset_commit,'Final dataset');hosting._commit(space_commit,'Final Space baseline')
    old,oldseal=hosting._source_viewer(original_dir,ORIGINAL_VIEWER_SHA)
    derived,sealed=hosting._source_viewer(derived_dir,expected_derived)
    receipt=_completed(receipt_raw,historical_raw,historical_files,space_commit,dataset_commit,card_raw,old)
    tree=hosting._tree_document(study._json(baseline_raw),space_commit)
    viewer_tree={n.removeprefix(PREFIX):r for n,r in tree.items() if n.startswith(PREFIX)}
    if set(viewer_tree)!=set(old) or any(not _matches(raw,viewer_tree[name]) for name,raw in old.items()):
        raise PublicationError('Initial Space viewer differs from the exact frozen original projection.')
    if set(roots)!={'index.html','README.md','.gitattributes'} or any(n not in tree or not _matches(raw,tree[n]) for n,raw in roots.items()):
        raise PublicationError('Root baseline bytes differ from the exact full Space tree.')
    with TemporaryDirectory(prefix='piper-native-overlay-') as temporary:
        target=Path(temporary)/'derived'
        native.derive(original_dir,target,expected_viewer_sha256=ORIGINAL_VIEWER_SHA,dataset_revision=dataset_commit)
        rebuilt,_=hosting._source_viewer(target,expected_derived)
        if rebuilt!=derived:raise PublicationError('Viewer differs from the exact native-resource/pin derivation.')
    expected_motion,native_files=hosting._pin_motion(old['motion.json'],dataset_commit)
    if derived['motion.json']!=expected_motion:
        raise PublicationError('Derived external motion differs from the exact evidence pin.')
    permitted={'index.html','piper_feedback_viewer.js','piper_feedback_viewer.bundle.js','motion.json','viewer-integrity.json'}
    additions=set(derived)-set(old)
    if (additions!={'BROWSER-MANAGED-RESOURCES.json'}|{f'replay-data-{i:02}.js' for i in range(1,28)}
            or any(derived[n]!=raw for n,raw in old.items() if n not in permitted)):
        raise PublicationError('Only the fixed native-resource frontend and dataset links may differ.')
    payload={PREFIX+n:derived[n] for n in permitted|additions}
    index,readme=hosting._root_links(roots['index.html'],roots['README.md'],study._json(old['report.json']))
    payload.update({'index.html':index,'README.md':readme})
    probe=study.SPACE_NAMESPACE+'/hosting-assets/browser-bundle-'+sha256(derived['piper_feedback_viewer.bundle.js']).hexdigest()+'.js'
    if probe in tree:raise PublicationError('The content-addressed preflight asset already exists; review the existing overlay.')
    payload[probe]=derived['piper_feedback_viewer.bundle.js']
    if any(n in tree for n in payload if n not in ACTIVATION):
        raise PublicationError('An append-only hosting asset collides with the exact baseline.')
    _attributes(roots['.gitattributes'],tree,payload)
    for name,raw in payload.items():
        study._relative(name)
        if len(raw)>study.MAX_FILE:raise PublicationError('A hosting file exceeds 8 MiB.')
        study._scan(raw,json_value=name.endswith('.json'),source_text=name.endswith('.js'))
    if sum(len(v) for v in derived.values())>study.MAX_SPACE:
        raise PublicationError('Derived viewer exceeds the fixed Space projection budget.')
    before={n:(roots[n] if n in roots else old[n.removeprefix(PREFIX)]) for n in ACTIVATION}
    plan=dict(schema=SCHEMA,repo_id=study.SPACE_REPO,repo_type='space',parent_commit=space_commit,
        dataset_repo=study.DATASET_REPO,dataset_commit=dataset_commit,
        original_viewer_directory=os.path.relpath(original_dir,output),derived_viewer_directory=os.path.relpath(derived_dir,output),
        original_viewer_integrity_sha256=ORIGINAL_VIEWER_SHA,derived_viewer_integrity_sha256=expected_derived,
        inputs=dict(baseline_tree=study._receipt(baseline_raw),continuation_receipt=study._receipt(receipt_raw),
                    continuation_plan=study._receipt(historical_raw),historical_implementation={n:study._receipt(b) for n,b in historical_files.items()},
                    dataset_metadata_receipt=study._receipt(card_raw) if card_raw is not None else None,
                    roots={n:study._receipt(raw) for n,raw in roots.items()}),
        implementation=_implementation(),
        files=[dict(path=n,before=study._receipt(before[n]) if n in before else None,after=study._receipt(raw),git_entry=_entry(raw)) for n,raw in sorted(payload.items())],
        groups=_groups(payload,probe),preflight_bundle_path=probe,native_files=native_files,
        caps=dict(asset_commit_bytes=ASSET_CAP,activation_commit_bytes=ACTIVATION_CAP,individual_file_bytes=study.MAX_FILE,files_per_commit=FILE_CAP),
        source_publication_plan_sha256=receipt['plan_sha256'],physical_robot_ready=False,
        scope='Post-publication hosting only. Exact native-resource derived viewer retained on both hosts; original science and all unrelated Space paths preserved. One extra content-addressed namespace-level bundle preflight asset is declared outside viewer/. Assets publish first; seven frontend/root paths activate atomically only after regular-Git preflight verification.')
    return plan,tree,payload,before


def stage(original_viewer,viewer,continuation_receipt,dataset_commit,space_commit,baseline,output,*,expected_viewer_sha256,continuation_plan,dataset_metadata_receipt=None):
    original_viewer,viewer,baseline,output=map(study._plain,(original_viewer,viewer,baseline,output))
    if output.exists() or any(p==output or p in output.parents or output in p.parents for p in (original_viewer,viewer,baseline)):
        raise PublicationError('Overlay staging must be fresh and separate from all inputs.')
    raw=study._read(continuation_receipt);card=study._read(dataset_metadata_receipt) if dataset_metadata_receipt is not None else None
    historical_path=study._plain(continuation_plan);historical_raw=study._read(historical_path)
    historical_files={n:study._read(historical_path.parent/'implementation'/n) for n in continuation.IMPLEMENTATION_FILES}
    historical_names={p.relative_to(historical_path.parent/'implementation').as_posix() for p in (historical_path.parent/'implementation').rglob('*') if p.is_file()}
    if historical_names!=set(continuation.IMPLEMENTATION_FILES):raise PublicationError('Historical implementation snapshot inventory changed.')
    baseline_raw=study._read(baseline/'tree.json');roots={n:study._read(baseline/n) for n in ('index.html','README.md','.gitattributes')}
    plan,_,payload,_=_compose(original_viewer,viewer,expected_viewer_sha256,baseline_raw,roots,raw,historical_raw,historical_files,card,dataset_commit,space_commit,output)
    output.mkdir(parents=True,exist_ok=False)
    for name,data in [('continuation-receipt.json',raw),('continuation-plan.json',historical_raw),('baseline-tree.json',baseline_raw)]+([('dataset-metadata-receipt.json',card)] if card is not None else []):
        (output/name).write_bytes(data)
    for folder,files in [('baseline-files',roots),('payload',payload),('historical-implementation',historical_files),('implementation',{n:study._read(ROOT/n) for n in IMPLEMENTATION})]:
        for name,data in files.items():
            target=output/folder/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (output/PLAN).write_bytes(study._encoded(plan))
    return plan


def _reference(root,name):
    if not isinstance(name,str) or not name or Path(name).is_absolute() or '\\' in name:
        raise PublicationError('Source references must be explicit relative local paths.')
    return study._plain(root/name)


def checked_plan(path):
    path=study._plain(path);plan=study._json(study._read(path))
    if not isinstance(plan,dict) or plan.get('schema')!=SCHEMA:raise PublicationError('Expected the fixed native Space overlay plan.')
    _implementation_check(plan.get('implementation'),path.parent/'implementation')
    historical_directory=path.parent/'historical-implementation'
    historical_files={n:study._read(historical_directory/n) for n in continuation.IMPLEMENTATION_FILES}
    if {p.relative_to(historical_directory).as_posix() for p in historical_directory.rglob('*') if p.is_file()}!=set(continuation.IMPLEMENTATION_FILES):
        raise PublicationError('Staged historical continuation implementation inventory changed.')
    card_path=path.parent/'dataset-metadata-receipt.json'
    args=(_reference(path.parent,plan.get('original_viewer_directory')),_reference(path.parent,plan.get('derived_viewer_directory')),
        plan.get('derived_viewer_integrity_sha256'),study._read(path.parent/'baseline-tree.json'),
        {n:study._read(path.parent/'baseline-files'/n) for n in ('index.html','README.md','.gitattributes')},
        study._read(path.parent/'continuation-receipt.json'),study._read(path.parent/'continuation-plan.json'),historical_files,study._read(card_path) if card_path.exists() else None,
        plan.get('dataset_commit'),plan.get('parent_commit'),path.parent)
    expected,tree,payload,before=_compose(*args)
    if plan!=expected:raise PublicationError('Overlay plan changed beyond the exact reviewed derivation.')
    actual={}
    for p in (path.parent/'payload').rglob('*'):
        study._plain(p)
        if p.is_file():actual[p.relative_to(path.parent/'payload').as_posix()]=study._read(p)
    if actual!=payload:raise PublicationError('Overlay payload differs from the exact reviewed projection.')
    return plan,tree,payload,before


def _remote(api):
    info=api.repo_info(study.SPACE_REPO,repo_type='space');checked_remote(info,'space');hosting._commit(info.sha,'Space head')
    return info.sha,public._tree(api,'space',info.sha)


def _history(api,commit):
    result={}
    for row in api.list_repo_commits(study.SPACE_REPO,repo_type='space',revision=commit):
        name,title=getattr(row,'commit_id',None),getattr(row,'title',None)
        if not isinstance(name,str) or study.COMMIT.fullmatch(name) is None or not isinstance(title,str) or name in result or len(result)>=20000:
            raise PublicationError('Malformed or excessive pinned Space history.')
        result[name]=title
    if commit not in result:raise PublicationError('Space history omits its pinned head.')
    return result


def _message(plan_sha,group):
    return f'DVIDIA native replay overlay {plan_sha} group {group["index"]} {group["kind"]}'


def _load_progress(path,plan,plan_sha,baseline,payload):
    progress=study._json(study._read(path)) if path.exists() else dict(schema=SCHEMA+'/journal',plan_sha256=plan_sha,head=plan['parent_commit'],groups=[],pending=None)
    if (not isinstance(progress,dict) or set(progress)!={'schema','plan_sha256','head','groups','pending'}
            or progress['schema']!=SCHEMA+'/journal' or progress['plan_sha256']!=plan_sha or not isinstance(progress['groups'],list)
            or len(progress['groups'])>len(plan['groups'])):
        raise PublicationError('Overlay journal differs from the exact plan.')
    tree=dict(baseline);head=plan['parent_commit']
    for index,record in enumerate(progress['groups']):
        group=plan['groups'][index];entries={n:_entry(payload[n]) for n in group['paths']}
        if (not isinstance(record,dict) or set(record)!={'index','parent_commit','commit','entries','anonymous_readback','reconciled'}
                or record['index']!=index or record['parent_commit']!=head or record['entries']!=entries
                or record['anonymous_readback'] is not True or type(record['reconciled']) is not bool):
            raise PublicationError('Completed overlay group changed.')
        hosting._commit(record['commit'],'Journal commit');head=record['commit'];tree.update(entries)
    if progress['head']!=head:raise PublicationError('Overlay journal head differs from its verified groups.')
    pending=progress['pending']
    if pending is not None:
        if len(progress['groups'])>=len(plan['groups']):raise PublicationError('Pending commit appears after completed activation.')
        group=plan['groups'][len(progress['groups'])]
        if pending!=dict(index=group['index'],parent_commit=head,paths=group['paths'],message=_message(plan_sha,group)):
            raise PublicationError('Pending request differs from the exact next group.')
    return progress,tree


def _finish(api,downloader,plan,progress,expected,payload,group,revision,journal,*,reconciled):
    hosting._commit(revision,'Published overlay revision')
    head,tree=_remote(api);wanted=dict(expected);wanted.update({n:_entry(payload[n]) for n in group['paths']})
    if head!=revision or tree!=wanted:raise PublicationError('Overlay commit changed unexpected paths or produced an LFS pointer.')
    for name in group['paths']:
        raw=study._downloaded_bytes(downloader(study.SPACE_REPO,name,'space',revision),study.MAX_FILE)
        if raw!=payload[name] or not _matches(raw,tree[name]):raise PublicationError('Anonymous regular-Git readback differs from reviewed bytes.')
    if _remote(api)!=(head,tree):raise PublicationError('Space moved during anonymous readback.')
    progress['groups'].append(dict(index=group['index'],parent_commit=progress['head'],commit=revision,
        entries={n:tree[n] for n in group['paths']},anonymous_readback=True,reconciled=reconciled))
    progress['head']=revision;progress['pending']=None;study._atomic(journal,progress);expected.clear();expected.update(tree)


def _reconcile(api,downloader,plan,plan_sha,progress,expected,payload,journal):
    head,tree=_remote(api);pending=progress['pending']
    if pending is None:
        if head!=progress['head'] or tree!=expected:raise PublicationError('Space moved outside the exact overlay journal.')
        return
    group=plan['groups'][pending['index']]
    if head==progress['head']:
        if tree!=expected or _remote(api)!=(head,tree):raise PublicationError('Pending outcome is still changing; do not retry.')
        # Same parent and exact body: the server's atomic parent guard permits
        # only one child if the original unknown request arrives later.
        progress['pending']=None;study._atomic(journal,progress);return
    prior,current=_history(api,progress['head']),_history(api,head)
    if set(current)!=set(prior)|{head} or current.get(head)!=pending['message']:
        raise PublicationError('Unknown outcome is not the exact intended immediate child.')
    _finish(api,downloader,plan,progress,expected,payload,group,head,journal,reconciled=True)


@contextmanager
def _lock(directory):
    import fcntl
    name=study._plain(directory/'native-space-writer.lock')
    descriptor=os.open(name,os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise PublicationError('Another native Space overlay writer is active.') from None
        yield
    finally:os.close(descriptor)


def _download(repo,name,kind,revision):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo,name,repo_type=kind,revision=revision,token=False,endpoint='https://huggingface.co')


def publish(path,*,api=None,downloader=None,operation_cls=None):
    path=study._plain(path)
    with _lock(path.parent):
        if (path.parent/RECEIPT).exists():raise PublicationError('Overlay is complete; its receipt cannot be overwritten.')
        plan,baseline,payload,before=checked_plan(path);plan_raw=study._read(path);plan_sha=sha256(plan_raw).hexdigest()
        journal=path.parent/JOURNAL;progress,expected=_load_progress(journal,plan,plan_sha,baseline,payload)
        token=None
        if api is None:api,token=client(path.parent/'publisher-cache')
        if operation_cls is None:
            from huggingface_hub import CommitOperationAdd
            operation_cls=CommitOperationAdd
        downloader=downloader or _download
        if token and any(token.encode() in raw for raw in payload.values()):raise PublicationError('A credential appeared in a frozen payload.')
        dataset=api.repo_info(study.DATASET_REPO,repo_type='dataset',revision=plan['dataset_commit']);checked_remote(dataset,'dataset')
        if dataset.sha!=plan['dataset_commit']:raise PublicationError('Pinned dataset did not resolve exactly.')
        dataset_tree=public._tree(api,'dataset',dataset.sha)
        for name,entry in plan['native_files'].items():
            if dataset_tree.get(name,{}).get('size')!=entry['bytes'] or dataset_tree.get(name,{}).get('lfs_sha256')!=entry['sha256']:
                raise PublicationError('Pinned native dataset LFS identity changed.')
        _reconcile(api,downloader,plan,plan_sha,progress,expected,payload,journal)
        if not progress['groups'] and progress['pending'] is None:
            for name,raw in before.items():
                read=study._downloaded_bytes(downloader(study.SPACE_REPO,name,'space',plan['parent_commit']),study.MAX_FILE)
                if read!=raw:raise PublicationError('Anonymous activation baseline differs from the original publication.')
        attributes=study._downloaded_bytes(downloader(study.SPACE_REPO,'.gitattributes','space',plan['parent_commit']),study.MAX_FILE)
        _attributes(attributes,baseline,payload)
        if not journal.exists():study._atomic(journal,progress)
        for group in plan['groups'][len(progress['groups']):]:
            _implementation_check(plan['implementation'],path.parent/'implementation')
            if study._read(path)!=plan_raw:raise PublicationError('Reviewed overlay plan changed during publication.')
            for name in group['paths']:
                if study._read(path.parent/'payload'/name)!=payload[name]:raise PublicationError('Frozen upload payload changed.')
            head,tree=_remote(api)
            if head!=progress['head'] or tree!=expected:raise PublicationError('Space moved before the next guarded commit.')
            if group['kind']=='activation':
                probe=plan['preflight_bundle_path']
                if expected.get(probe)!=_entry(payload[PREFIX+'piper_feedback_viewer.bundle.js']):
                    raise PublicationError('Large native bundle has not passed regular-Git preflight.')
            operations=[]
            for name in group['paths']:
                op=operation_cls(path_in_repo=name,path_or_fileobj=payload[name]);op._upload_mode='regular';op._should_ignore=False;operations.append(op)
            progress['pending']=dict(index=group['index'],parent_commit=head,paths=group['paths'],message=_message(plan_sha,group));study._atomic(journal,progress)
            try:result=api.create_commit(study.SPACE_REPO,repo_type='space',parent_commit=head,operations=operations,commit_message=_message(plan_sha,group))
            except Exception:raise PublicationError('Commit reply is uncertain; retain this exact pending journal for reconciliation.') from None
            revision=getattr(result,'oid',None)
            if revision==head:raise PublicationError('Commit did not return a new revision; reconcile the retained pending request.')
            _finish(api,downloader,plan,progress,expected,payload,group,revision,journal,reconciled=False)
        head,tree=_remote(api)
        if head!=progress['head'] or tree!=expected:raise PublicationError('Final Space differs from the complete overlay journal.')
        for name,raw in payload.items():
            if study._downloaded_bytes(downloader(study.SPACE_REPO,name,'space',head),study.MAX_FILE)!=raw:
                raise PublicationError('Final anonymous full overlay readback differs from the reviewed bytes.')
        if _remote(api)!=(head,tree):raise PublicationError('Space moved during final anonymous readback.')
        receipt=dict(schema=SCHEMA+'/receipt',plan_sha256=plan_sha,parent_commit=plan['parent_commit'],commit=head,
            dataset_commit=plan['dataset_commit'],original_viewer_integrity_sha256=ORIGINAL_VIEWER_SHA,
            derived_viewer_integrity_sha256=plan['derived_viewer_integrity_sha256'],changed_paths=sorted(payload),
            content_addressed_preflight_asset=plan['preflight_bundle_path'],all_overlay_objects_regular_git=True,
            native_lfs_receipts_checked=120,anonymous_full_overlay_readback=True,unchanged_baseline_paths=len(baseline)-len(ACTIVATION),
            viewer_url=study.VIEWER_URL,physical_robot_ready=False,
            continuation_receipt=plan['inputs']['continuation_receipt'],dataset_metadata_receipt=plan['inputs']['dataset_metadata_receipt'],
            verified_commits=progress['groups'])
        study._atomic(path.parent/RECEIPT,receipt);return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('stage')
    for name in ('original-viewer','viewer','continuation-receipt','continuation-plan','dataset-commit','space-commit','baseline','output','expected-viewer-sha256'):
        prep.add_argument('--'+name,required=True)
    prep.add_argument('--dataset-metadata-receipt')
    for name in ('check','publish'):sub.add_parser(name).add_argument('plan',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='stage':result=stage(args.original_viewer,args.viewer,args.continuation_receipt,args.dataset_commit,args.space_commit,args.baseline,args.output,expected_viewer_sha256=args.expected_viewer_sha256,continuation_plan=args.continuation_plan,dataset_metadata_receipt=args.dataset_metadata_receipt)
        elif args.command=='check':result=checked_plan(args.plan)[0]
        else:result=publish(args.plan)
        print(json.dumps(dict(schema=result['schema'],command=args.command,complete=True),indent=2));return 0
    except Exception as error:
        print('Native Space overlay stopped ('+type(error).__name__+'); preserve the exact plan and pending journal.',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
