"""Prepare one reviewed static robot Space from explicitly supplied sealed evidence.

Local staging only. No credentials, authenticated requests, footage discovery,
weights, recursive source checkout copies or hardware commands are used.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(ROOT/'src'))
import publish_robot_sim as publisher
from dvidia_training.reference_replay import _decode
from dvidia_training.reference_task_benchmark import load_task_bundle


def _record(raw):
    return {'bytes':len(raw),'sha256':sha256(raw).hexdigest()}


def _sealed_replay(directory):
    root = Path(directory).absolute()
    publisher._plain(root)
    if not root.is_dir():
        raise ValueError('Expected an explicitly supplied sealed native task replay directory.')
    seal_bytes = publisher._bytes(root/'replay-integrity.json',1024*1024)
    seal = _decode(seal_bytes)
    declarations = seal.get('files') if type(seal) is dict else None
    if (type(seal) is not dict or seal.get('schema') != 'dvidia.reference-replay-integrity/v1'
            or type(declarations) is not dict or not 15 <= len(declarations) <= 400):
        raise ValueError('Expected a bounded sealed native replay inventory.')
    payload = {}
    total = 0
    for name,record in declarations.items():
        raw = publisher._bytes(root/publisher._artifact_name(name),publisher.MAX_FILE_BYTES)
        if record != _record(raw):
            raise ValueError('Native replay artifact differs from its original seal.')
        publisher._no_private_bytes(raw)
        total += len(raw)
        if total > publisher.MAX_TOTAL_BYTES:
            raise ValueError('Native replay exceeds the Space publication budget.')
        payload[name] = raw
    inventory = set()
    for file in root.rglob('*'):
        publisher._plain(file)
        if file.is_file():
            inventory.add(file.relative_to(root).as_posix())
    if inventory != set(payload)|{'replay-integrity.json'}:
        raise ValueError('Native replay contains files outside its sealed inventory.')
    required = {'index.html','scene.json','motion.json','geometry.bin','geometry.bin.gz','motion.bin','motion.bin.gz',
                'reference_replay.bundle.js','reference_replay.third-party.txt','LICENSE','README.md',
                'receipt/report.json','receipt/report.html','receipt/protocol.json','receipt/integrity.json',
                'licenses/universal_robots_ur5e/LICENSE','licenses/universal_robots_ur5e/README.md',
                'licenses/robotiq_2f85/LICENSE','licenses/robotiq_2f85/README.md'}
    if not required <= payload.keys():
        raise ValueError('Native task replay is missing complete evidence or model notices.')
    report,receipt_files = load_task_bundle(root/'receipt')
    for name,raw in receipt_files.items():
        if payload.get('receipt/'+name) != raw:
            raise ValueError('Native task receipt changed while taking the publication snapshot.')
    scene,motion = _decode(payload['scene.json']),_decode(payload['motion.json'])
    if (scene.get('schema') != 'dvidia.reference-replay-scene/v1'
            or motion.get('schema') != 'dvidia.reference-replay-motion/v1'
            or motion.get('source',{}).get('profile') != 'ur5e-2f85-rigid-block-task-v0.1'
            or report.get('profile') != 'ur5e-2f85-rigid-block-task-v0.1'
            or report.get('physical_robot_ready') is not False or report.get('learned_policy') is not False
            or motion['source'].get('report_sha256') != sha256(payload['receipt/report.json']).hexdigest()
            or motion['source'].get('benchmark_attempted') != report['attempted']):
        raise ValueError('Only the pinned simulation-only native task replay may be staged.')
    selected = motion.get('cases')
    if type(selected) is not list or not 1 <= len(selected) <= 9:
        raise ValueError('Native task replay selection is outside the reviewed excerpt bound.')
    source_ids = {a['id']:a for a in report['attempts']}
    for case in selected:
        source = source_ids.get(case.get('attempt_id'))
        if (source is None or source['valid'] is not True or source['success'] != case.get('recorded_success')
                or source['scene_id'] != case.get('scene_id') or source['control'] != case.get('control')):
            raise ValueError('Replay selection does not match the complete sealed task report.')
    payload['replay-integrity.json'] = seal_bytes
    return payload,report,motion


def _summary(report,*,html=False):
    groups = []
    for row in report['summary']:
        groups.append(f"{row['stratum']} / {row['control']}: {row['success']}/{row['attempted']} completed, {row['valid']} numerically valid")
    note = (f"Frozen {report['cohort']} cohort: {report['attempted']} planned attempts across {len(report['scenes'])} synthetic scenes. "
            'Counts are descriptive, with no population or physical reliability claim.')
    if html:
        return '<strong>'+escape(note)+'</strong><br>'+'<br>'.join(escape(line) for line in groups)
    return note+'\n\n'+'\n'.join('- '+line for line in groups)


def _sealed_profile(directory):
    from dvidia_training.robot_profiles import inspect_benchmark
    root = Path(directory).absolute()
    publisher._plain(root)
    if not root.is_dir():
        raise ValueError('Profile evidence must be an explicitly supplied plain benchmark directory.')
    # Inspector verifies the profile report and its declared hashes. Snapshot
    # those exact reviewed bytes, then recheck against the same declarations.
    inspected = inspect_benchmark(root)
    records = inspected.get('files')
    if (type(records) is not list or not 3 <= len(records) <= 12
            or any(type(item) is not dict or set(item)!={'path','bytes','sha256'} for item in records)
            or any(type(item['path']) is not str for item in records)
            or len({item['path'] for item in records}) != len(records)):
        raise ValueError('Invalid bounded native profile evidence inventory.')
    declarations = {item['path']:{key:item[key] for key in ('bytes','sha256')} for item in records}
    payload = {}
    total = 0
    for name,record in declarations.items():
        raw = publisher._bytes(root/publisher._artifact_name(name),publisher.MAX_FILE_BYTES)
        publisher._no_private_bytes(raw)
        if record != _record(raw):
            raise ValueError('Native profile evidence changed after inspection.')
        total += len(raw)
        if total>publisher.MAX_TOTAL_BYTES:
            raise ValueError('Native profile evidence exceeds the Space publication budget.')
        payload[name] = raw
    manifest_bytes = publisher._bytes(root/'integrity.json',1024*1024)
    manifest = _decode(manifest_bytes)
    if (type(manifest) is not dict or manifest.get('schema') != 'dvidia.robot-profile-integrity/v1'
            or manifest.get('files') != records):
        raise ValueError('Native profile integrity changed after inspection.')
    actual = set()
    for file in root.rglob('*'):
        publisher._plain(file)
        if file.is_file():
            actual.add(file.relative_to(root).as_posix())
    if actual != set(payload)|{'integrity.json'}:
        raise ValueError('Native profile evidence contains files outside its sealed inventory.')
    report = _decode(payload.get('report.json',b'null'))
    if type(report) is not dict or report != inspected.get('report'):
        raise ValueError('Native profile report snapshot differs from the validated report.')
    profile = report.get('profile',{})
    if type(profile) is not dict:
        raise ValueError('Invalid native profile metadata.')
    expected_axes = {'piper':6,'so101':5}
    identity = profile.get('id')
    if (report.get('schema') != 'dvidia.robot-profile-benchmark/v1'
            or identity not in expected_axes or type(profile.get('arm_axes')) is not int
            or profile['arm_axes'] != expected_axes[identity] or type(profile.get('gripper_axes')) is not int
            or profile['gripper_axes'] != 1
            or report.get('task_qualified') is not False or report.get('physical_robot_ready') is not False
            or type(profile.get('name')) is not str or type(profile.get('simulation_asset_license')) is not str
            or not {'report.json','report.html','licenses/'+identity+'/LICENSE'} <= payload.keys()):
        raise ValueError('Profile bundle differs from the reviewed unloaded PiPER/SO-101 contract.')
    payload['integrity.json'] = manifest_bytes
    return identity,payload,report


def _sealed_piper_task(directory):
    from dvidia_training import piper_task_benchmark as benchmark
    root = Path(directory).absolute()
    publisher._plain(root)
    if not root.is_dir():
        raise ValueError('PiPER task evidence must be an explicitly supplied plain benchmark directory.')
    report,files = benchmark.load_task_bundle(root)
    if (type(report) is not dict or type(files) is not dict or not 9<=len(files)<=42
            or 'integrity.json' not in files or any(type(raw) is not bytes for raw in files.values())):
        raise ValueError('Invalid bounded PiPER task evidence snapshot.')
    manifest = _decode(files['integrity.json'])
    if (type(manifest) is not dict or set(manifest)!={'schema','files'}
            or manifest['schema']!='dvidia.piper-task-integrity/v1' or type(manifest['files']) is not dict
            or set(manifest['files'])!=set(files)-{'integrity.json'}):
        raise ValueError('PiPER task snapshot differs from its original integrity inventory.')
    total = 0
    for name,raw in files.items():
        source = root/publisher._artifact_name(name)
        if len(raw)>publisher.MAX_FILE_BYTES or publisher._bytes(source,publisher.MAX_FILE_BYTES)!=raw:
            raise ValueError('PiPER task evidence changed after validated inspection.')
        publisher._no_private_bytes(raw)
        total += len(raw)
        if total>publisher.MAX_TOTAL_BYTES:
            raise ValueError('PiPER task evidence exceeds the Space publication budget.')
        if name!='integrity.json':
            record = manifest['files'][name]
            if (type(record) is not dict or set(record)!={'bytes','sha256'}
                    or type(record['bytes']) is not int or record!=_record(raw)):
                raise ValueError('PiPER task snapshot digest differs from its original seal.')
    actual = set()
    for file in root.rglob('*'):
        publisher._plain(file)
        if file.is_file():
            actual.add(file.relative_to(root).as_posix())
    if actual!=set(files):
        raise ValueError('PiPER task evidence contains files outside its sealed inventory.')
    required = {'report.json','report.html','protocol.json','integrity.json',
                'licenses/agilex_piper/LICENSE','licenses/agilex_piper/README.md'}
    if not required<=set(files) or _decode(files['report.json'])!=report:
        raise ValueError('PiPER task report or complete model notices are missing from the snapshot.')
    protocol = _decode(files['protocol.json'])
    if (report.get('schema')!='dvidia.piper-task-benchmark/v1'
            or report.get('profile')!='piper-rigid-block-task-v0.1'
            or report.get('physical_robot_ready') is not False or report.get('learned_policy') is not False
            or report.get('simulation_asset_license')!='MIT' or report.get('scope')!=benchmark.SCOPE
            or type(protocol) is not dict or protocol.get('schema')!='dvidia.piper-task-protocol/v1'
            or protocol.get('profile')!=report['profile'] or protocol.get('scope')!=benchmark.SCOPE
            or protocol.get('physical_robot_ready') is not False or protocol.get('learned_policy') is not False
            or protocol.get('native_command_units')!=['rad']*6+['m']
            or protocol.get('simulation_asset_license')!='MIT'
            or report.get('cohort') not in ('development','confirmation') or protocol.get('cohort')!=report['cohort']
            or report.get('protocol')!=protocol
            or report.get('protocol_sha256')!=sha256(files['protocol.json']).hexdigest()):
        raise ValueError('Only the reviewed authored, synthetic, simulation-only PiPER contact pilot may be staged.')
    scenes,attempts = report.get('scenes'),report.get('attempts')
    if (type(scenes) is not list or not 1<=len(scenes)<=12 or scenes!=protocol.get('scenes')
            or any(type(scene) is not dict or type(scene.get('id')) is not str
                   or scene.get('stratum') not in ('nominal','low_friction') for scene in scenes)
            or len({scene['id'] for scene in scenes})!=len(scenes)
            or type(attempts) is not list or len(attempts)!=len(scenes)*3
            or type(report.get('attempted')) is not int or report['attempted']!=len(attempts)
            or protocol.get('controls')!=list(benchmark.CONTROLS)
            or type(protocol.get('planned_attempts')) is not int or protocol['planned_attempts']!=len(attempts)):
        raise ValueError('PiPER contact pilot must retain its complete planned denominator.')
    identities = {scene['id']+'--'+control for scene in scenes for control in benchmark.CONTROLS}
    seen,result_files = set(),set()
    for attempt in attempts:
        if (type(attempt) is not dict or type(attempt.get('id')) is not str
                or type(attempt.get('scene_id')) is not str
                or attempt['id'] not in identities or attempt['id'] in seen
                or attempt.get('control') not in benchmark.CONTROLS
                or attempt.get('scene_id')+'--'+attempt['control']!=attempt['id']
                or type(attempt.get('valid')) is not bool or type(attempt.get('success')) is not bool
                or (attempt['success'] and not attempt['valid'])):
            raise ValueError('PiPER contact pilot attempt identity or outcome differs from the complete plan.')
        name = 'attempts/'+attempt['id']+'.json'
        if attempt.get('result_file')!=name or name not in files:
            raise ValueError('PiPER contact pilot is missing a complete source attempt.')
        raw = _decode(files[name])
        if type(raw) is not dict or benchmark._summary(raw)!={key:value for key,value in attempt.items() if key!='result_file'}:
            raise ValueError('PiPER contact pilot outcome differs from its complete source attempt.')
        seen.add(attempt['id']);result_files.add(name)
    if (set(files)!=required|result_files or type(report.get('valid_attempts')) is not int
            or report['valid_attempts']!=sum(row['valid'] for row in attempts)
            or report.get('summary')!=benchmark.aggregate(attempts,scenes)):
        raise ValueError('PiPER contact pilot inventory or stratified counts differ from all source attempts.')
    return files,report


def _piper_task_card(report,source_commit):
    if report is None:
        return ''
    return ('<article class="profile"><span class="status">Recorded PiPER contact pilot · synthetic</span>'
            '<h2>PiPER: authored grasp, carry and release</h2><p>Six arm commands in radians and one jaw command in metres. '
            'This pilot uses an authored controller and privileged native simulator state. Contact is uncalibrated; '
            'no learned skill, shoe competence or physical robot is qualified. No PiPER 3D viewer is included.</p>'
            '<div class="summary">'+_summary(report,html=True)+'</div><div class="links">'
            '<a href="tasks/piper/report.html">Every contact attempt ↗</a>'
            '<a href="tasks/piper/report.json">Complete task report ↗</a>'
            '<a href="tasks/piper/protocol.json">Frozen protocol ↗</a>'
            '<a href="tasks/piper/integrity.json">All task file hashes ↗</a>'
            '<a href="tasks/piper/licenses/agilex_piper/LICENSE">Original MIT model license ↗</a>'
            '<a href="https://github.com/Dvidia-Inference/dvidia-training/blob/'+source_commit+
            '/docs/piper-contact-pilot.md">Pinned PiPER contact guide ↗</a></div></article>')


def _sealed_piper_grip(directory):
    from dvidia_training import piper_grip_benchmark as benchmark
    root = Path(directory).absolute()
    publisher._plain(root)
    if not root.is_dir():
        raise ValueError('PiPER grip evidence must be an explicitly supplied plain benchmark directory.')
    report,files = benchmark.load_grip_bundle(root)
    required = {'report.json','report.html','protocol.json','integrity.json',
                'licenses/agilex_piper/LICENSE','licenses/agilex_piper/README.md'}
    if (type(report) is not dict or type(files) is not dict or len(files)!=36
            or not required<=set(files) or any(type(raw) is not bytes for raw in files.values())):
        raise ValueError('PiPER grip comparison must retain every planned compressed attempt and model notice.')
    manifest = _decode(files['integrity.json'])
    if (type(manifest) is not dict or set(manifest)!={'schema','files'}
            or manifest['schema']!='dvidia.piper-grip-integrity/v1' or type(manifest['files']) is not dict
            or set(manifest['files'])!=set(files)-{'integrity.json'} or _decode(files['report.json'])!=report):
        raise ValueError('PiPER grip snapshot differs from its original complete seal/report.')
    protocol = _decode(files['protocol.json'])
    if (report.get('schema')!='dvidia.piper-grip-benchmark/v1' or type(protocol) is not dict
            or protocol.get('schema')!='dvidia.piper-grip-protocol/v1'
            or report.get('profile')!='piper-rigid-block-task-v0.1' or protocol.get('profile')!=report['profile']
            or any(doc.get('physical_robot_ready') is not False or doc.get('learned_policy') is not False
                   or doc.get('scope')!=benchmark.SCOPE or doc.get('simulation_asset_license')!='MIT'
                   for doc in (report,protocol))
            or report.get('protocol')!=protocol or report.get('cohort') not in ('development','confirmation')
            or protocol.get('cohort')!=report['cohort']
            or protocol.get('native_command_units')!=['rad']*6+['m']
            or protocol.get('variants')!=list(benchmark.VARIANTS)
            or type(protocol.get('planned_attempts')) is not int or protocol['planned_attempts']!=30
            or report.get('protocol_sha256')!=sha256(files['protocol.json']).hexdigest()):
        raise ValueError('Only the pinned authored, synthetic PiPER grip comparison may be staged.')
    scenes,attempts = report.get('scenes'),report.get('attempts')
    if (type(scenes) is not list or len(scenes)!=6 or scenes!=protocol.get('scenes')
            or any(type(scene) is not dict or type(scene.get('id')) is not str
                   or scene.get('friction') not in benchmark.FRICTIONS or type(scene.get('layout_index')) is not int
                   or scene['layout_index'] not in (0,1) for scene in scenes)
            or len({scene['id'] for scene in scenes})!=6
            or {(scene['friction'],scene['layout_index']) for scene in scenes}!={(f,i) for f in benchmark.FRICTIONS for i in (0,1)}
            or type(attempts) is not list or len(attempts)!=30
            or type(report.get('attempted')) is not int or report['attempted']!=30):
        raise ValueError('PiPER grip comparison must retain all six scenes and thirty paired attempts.')
    expected = {scene['id']+'--'+variant for scene in scenes for variant in benchmark.VARIANTS}
    lookup = {}
    for row in attempts:
        if (type(row) is not dict or type(row.get('id')) is not str or type(row.get('scene_id')) is not str
                or type(row.get('variant')) is not str or row['id'] not in expected or row['id'] in lookup
                or row['id']!=row['scene_id']+'--'+row['variant']
                or row.get('result_file')!='attempts/'+row['id']+'.json.gz'
                or type(row.get('valid')) is not bool or type(row.get('success')) is not bool
                or (row['success'] and not row['valid'])):
            raise ValueError('PiPER grip attempt differs from its complete paired plan.')
        lookup[row['id']]=row
    if (set(files)!=required|{row['result_file'] for row in attempts}
            or report.get('summary')!=benchmark.aggregate(attempts,scenes)
            or type(report.get('valid_attempts')) is not int or report['valid_attempts']!=sum(row['valid'] for row in attempts)):
        raise ValueError('PiPER grip inventory or friction-stratified counts differ from all attempts.')
    total,decoded_total = 0,0
    for name,data in files.items():
        publisher._artifact_name('tasks/piper-grip/'+name)
        if len(data)>publisher.MAX_FILE_BYTES or publisher._bytes(root/name,publisher.MAX_FILE_BYTES)!=data:
            raise ValueError('PiPER grip evidence changed after validated inspection.')
        decoded = data
        if name!='integrity.json':
            record = manifest['files'][name]
            compressed = name.startswith('attempts/')
            keys = {'bytes','sha256','encoding','decoded_bytes','decoded_sha256'} if compressed else {'bytes','sha256'}
            if (type(record) is not dict or set(record)!=keys or type(record['bytes']) is not int
                    or {key:record[key] for key in ('bytes','sha256')}!=_record(data)):
                raise ValueError('PiPER grip stored bytes differ from their original seal.')
            if compressed:
                decoded = benchmark.decode_attempt(data,record)
                raw = _decode(decoded)
                identity = name.removeprefix('attempts/').removesuffix('.json.gz')
                if (type(raw) is not dict or benchmark._summary(raw)!=
                        {key:value for key,value in lookup[identity].items() if key!='result_file'}):
                    raise ValueError('PiPER grip outcome differs from its complete decoded source attempt.')
        publisher._no_private_bytes(data)
        publisher._no_private_bytes(decoded)
        total+=len(data);decoded_total+=len(decoded)
        if total>publisher.MAX_TOTAL_BYTES or decoded_total>benchmark.MAX_DECODED_BYTES:
            raise ValueError('PiPER grip evidence exceeds its stored or decoded byte budget.')
    actual = set()
    for path in root.rglob('*'):
        publisher._plain(path)
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual!=set(files):
        raise ValueError('PiPER grip comparison contains files outside its original sealed inventory.')
    return files,report,{'stored_bytes':total,'decoded_bytes':decoded_total}


def _grip_summary(report,*,html=False):
    note = (f"Frozen {report['cohort']} cohort: {report['attempted']} planned attempts across six synthetic scenes. "
            'Two shared layouts at each friction; five variants per scene. Counts are descriptive, not physical reliability or detector accuracy.')
    rows = [f"μ={row['friction']:.2f} / {row['variant']}: {row['success']}/{row['attempted']} completed, {row['valid']} numerically valid"
            for row in report['summary']]
    if html:
        return '<strong>'+escape(note)+'</strong><br>'+'<br>'.join(escape(line) for line in rows)
    return note+'\n\n'+'\n'.join('- '+line for line in rows)


def _piper_grip_card(report,source_commit):
    if report is None:
        return ''
    return ('<article class="profile"><span class="status">PiPER guarded grip comparison · synthetic</span>'
            '<h2>Test the grip. Inspect recovery.</h2><p>Baseline and guarded controllers use privileged native state. '
            'Each has its own matched open-jaw replay; a shared hold control remains. Qualification decisions, retries, '
            'alarms, unresolved opportunities and aborts are retained. These are authored policies with uncalibrated '
            'contact, not learned skills or a physically qualified detector.</p><div class="summary">'+_grip_summary(report,html=True)+
            '</div><div class="links"><a href="tasks/piper-grip/report.html">Full paired comparison ↗</a>'
            '<a href="tasks/piper-grip/report.json">All outcomes and diagnostics ↗</a>'
            '<a href="tasks/piper-grip/protocol.json">Frozen guard and evaluation contract ↗</a>'
            '<a href="tasks/piper-grip/integrity.json">Stored and decoded attempt hashes ↗</a>'
            '<a href="https://github.com/Dvidia-Inference/dvidia-training/blob/'+source_commit+
            '/docs/piper-grip-recovery.md">Pinned offline inspection guide ↗</a></div>'
            '<p class="tiny">Every full attempt is retained as deterministic gzip JSON. The CLI checks compressed and '
            'decoded hashes with bounded decompression; no archive extraction is needed. This comparison adds numerical '
            'evidence to the existing contact pilot; it adds no PiPER 3D viewer.</p></article>')


def _profile_cards(profile_reports=(),source_commit='main',piper_task_report=None,piper_grip_report=None):
    profiles = [
        ('UR5e + Robotiq 2F-85','Recorded native contact task',
         'Six arm joints, eight articulated gripper coordinates and a free rigid block. Full source outcomes and matched controls are available.',
         'replay/index.html','Inspect the task replay ↗'),
        ('Additional native arm profiles','Upcoming · evidence pending',
         'More open-source arm models are being evaluated. A listing becomes supported only after pinned asset checks, native mechanics receipts and a reviewed replay.',
         'https://github.com/Dvidia-Inference/dvidia-training/blob/main/docs/hardware-roadmap.md','Follow the hardware roadmap ↗'),
        ('Two arms and dexterous hands','Upcoming · no task evidence',
         'The roadmap extends to bilateral manipulation and five-finger hands. Contact tasks, sensing adapters and physical qualification remain to be built.',
         'https://research.dvidia.org/','Follow the research ↗')]
    recorded = []
    for report in profile_reports:
        profile = report['profile']
        identity = profile['id']
        recorded.append('<article class="profile"><span class="status">Recorded native mechanics · unloaded</span><h2>'+escape(profile['name'])+
            '</h2><p>'+str(profile['arm_axes'])+' arm axes + '+str(profile['gripper_axes'])+' gripper axis. '
            'This separate bundle measures native model structure and authored unloaded motion. Its numerical receipts do not qualify a contact task, learned policy or physical robot. '
            'Simulation asset license: '+escape(profile['simulation_asset_license'])+'.</p>'
            '<div class="links"><a href="profiles/'+identity+'/report.html">Numerical report ↗</a>'
            '<a href="profiles/'+identity+'/report.json">Complete recorded states ↗</a>'
            '<a href="profiles/'+identity+'/integrity.json">File hashes ↗</a>'
            '<a href="profiles/'+identity+'/licenses/'+identity+'/LICENSE">Model license ↗</a></div>'
            '<a href="https://github.com/Dvidia-Inference/dvidia-training/blob/'+source_commit+'/docs/robot-simulation-library.md">Pinned local simulator guide ↗</a></article>')
    if len(profile_reports)==2:
        profiles.pop(1)
    authored = ['<article class="profile"><span class="status">'+escape(status)+'</span><h2>'+escape(title)+
                   '</h2><p>'+escape(description)+'</p><a href="'+escape(href,quote=True)+'">'+escape(label)+
                   '</a></article>' for title,status,description,href,label in profiles]
    return authored[0]+_piper_task_card(piper_task_report,source_commit)+_piper_grip_card(piper_grip_report,source_commit)+''.join(recorded)+''.join(authored[1:])


def stage(replay,output,*,source_commit=None,expected_remote_commit=None,profile_benchmarks=None,
          remove_starter_stylesheet=None,piper_task_benchmark=None,piper_grip_benchmark=None):
    replay_files,report,motion = _sealed_replay(replay)
    if profile_benchmarks is not None and (type(profile_benchmarks) is not list or len(profile_benchmarks)>2):
        raise ValueError('Supply at most two explicit native profile benchmark directories.')
    if source_commit is None:
        source_commit = subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if type(source_commit) is not str or publisher.SHA.fullmatch(source_commit) is None:
        raise ValueError('Source must be pinned to a full committed Git revision.')
    if expected_remote_commit is not None and (type(expected_remote_commit) is not str or publisher.SHA.fullmatch(expected_remote_commit) is None):
        raise ValueError('Reviewed updates must name the exact remote parent commit.')
    removals = []
    if remove_starter_stylesheet is not None:
        raw = publisher._bytes(Path(remove_starter_stylesheet).absolute(),publisher.MAX_STARTER_STYLESHEET_BYTES)
        removals = [{'path':'style.css',**_record(raw)}]
    publisher._reviewed_removals({'expected_remote_commit':expected_remote_commit,'remove_files':removals})
    payload = {'replay/'+name:raw for name,raw in replay_files.items()}
    profile_records,profile_reports = {},[]
    for directory in profile_benchmarks or []:
        identity,profile_files,profile_report = _sealed_profile(directory)
        if identity in profile_records:
            raise ValueError('A robot profile may appear only once in the reviewed Space payload.')
        payload.update({'profiles/'+identity+'/'+name:raw for name,raw in profile_files.items()})
        profile_records[identity] = {'source_integrity':_record(profile_files['integrity.json']),
                                    'source_report':_record(profile_files['report.json']),
                                    'profile':profile_report['profile'],'task_qualified':False,'physical_robot_ready':False}
        profile_reports.append(profile_report)
    piper_task_report = None
    task_records = {}
    if piper_task_benchmark is not None:
        task_files,piper_task_report = _sealed_piper_task(piper_task_benchmark)
        payload.update({'tasks/piper/'+name:raw for name,raw in task_files.items()})
        task_records['piper'] = {'profile':piper_task_report['profile'],'source_integrity':_record(task_files['integrity.json']),
            'source_report':_record(task_files['report.json']),'source_protocol':_record(task_files['protocol.json']),
            'cohort':piper_task_report['cohort'],'attempt_ids':[row['id'] for row in piper_task_report['attempts']],
            'native_command_units':['rad']*6+['m'],'attempted':piper_task_report['attempted'],
            'valid_attempts':piper_task_report['valid_attempts'],'summary':piper_task_report['summary'],
            'physical_robot_ready':False,'learned_policy':False,'scope':piper_task_report['scope'],
            'byte_preservation':'Complete original protocol, report, every raw attempt, integrity and MIT model notices retained under tasks/piper/'}
    piper_grip_report = None
    if piper_grip_benchmark is not None:
        grip_files,piper_grip_report,sizes = _sealed_piper_grip(piper_grip_benchmark)
        payload.update({'tasks/piper-grip/'+name:raw for name,raw in grip_files.items()})
        task_records['piper-grip'] = {'profile':piper_grip_report['profile'],'source_integrity':_record(grip_files['integrity.json']),
            'source_report':_record(grip_files['report.json']),'source_protocol':_record(grip_files['protocol.json']),
            'cohort':piper_grip_report['cohort'],'attempt_ids':[row['id'] for row in piper_grip_report['attempts']],
            'attempted':30,'valid_attempts':piper_grip_report['valid_attempts'],'summary':piper_grip_report['summary'],
            'native_command_units':['rad']*6+['m'],'physical_robot_ready':False,'learned_policy':False,'scope':piper_grip_report['scope'],
            'canonical_attempt_storage':'Deterministic single-member gzip JSON; original stored and decoded SHA-256/byte sizes retained in source seal',
            'byte_preservation':'Complete original protocol, report, thirty canonical attempt files, integrity and MIT notices retained under tasks/piper-grip/',
            **sizes}
    readme = publisher._bytes(HERE/'templates/README.md',1024*1024).decode()
    index = publisher._bytes(HERE/'templates/index.html',1024*1024).decode()
    profile_summary = ('No separate unloaded numerical profile bundles are included in this payload.'
                       if not profile_reports else '\n'.join('- '+p['profile']['name']+': '+str(p['profile']['arm_axes'])+
                           ' arm axes + 1 gripper axis; recorded unloaded native mechanics, not task-qualified. '
                           'Complete report, traces, hashes and original '+p['profile']['simulation_asset_license']+
                           ' notice: `profiles/'+p['profile']['id']+'/`.' for p in profile_reports))
    piper_summary = ('' if piper_task_report is None else '## PiPER contact pilot\n\n'+_summary(piper_task_report)+
        '\n\nAuthored controller, synthetic scenes and privileged native simulator state. Contact is uncalibrated; '
        'no learned skill or physical robot is qualified. This is separate from the unloaded PiPER mechanics bundle; '
        'no PiPER 3D viewer is included. Every attempted episode, including failed and invalid attempts, remains under '
        '`tasks/piper/attempts/`. Inspect [the complete task report](tasks/piper/report.html), '
        '[frozen protocol](tasks/piper/protocol.json), [file hashes](tasks/piper/integrity.json), and '
        '[original MIT model license](tasks/piper/licenses/agilex_piper/LICENSE). '
        '[Local PiPER contact guide](https://github.com/Dvidia-Inference/dvidia-training/blob/'+source_commit+
        '/docs/piper-contact-pilot.md).\n')
    grip_summary = ('' if piper_grip_report is None else '## PiPER guarded grip comparison\n\n'+_grip_summary(piper_grip_report)+
        '\n\nBaseline and guarded authored controllers use privileged native state and their own matched open-jaw '
        'replays plus a common hold control. Full qualification decisions, unresolved outcomes, absent opportunities, '
        'alarms, retries and aborts remain in the source evidence. These synthetic trials are not detector accuracy, '
        'learned competence or physical qualification.\n\nEvery full attempt is stored as canonical deterministic gzip '
        'JSON under `tasks/piper-grip/attempts/`; the original seal records both compressed and decoded bytes and '
        'SHA-256. Download all files in `tasks/piper-grip/` together and run the pinned `dvidia-piper-grip inspect` '
        'command from [the local guide](https://github.com/Dvidia-Inference/dvidia-training/blob/'+source_commit+
        '/docs/piper-grip-recovery.md). No archive extraction is needed. '
        '[Full comparison](tasks/piper-grip/report.html) · [Original protocol](tasks/piper-grip/protocol.json) · '
        '[Stored and decoded hashes](tasks/piper-grip/integrity.json).\n')
    readme = readme.replace('{{BENCHMARK_SUMMARY}}',_summary(report)).replace('{{PROFILE_SUMMARY}}',profile_summary).replace('{{PIPER_TASK_SUMMARY}}',piper_summary).replace('{{PIPER_GRIP_SUMMARY}}',grip_summary).replace('{{SOURCE_COMMIT}}',source_commit)
    index = index.replace('{{BENCHMARK_SUMMARY}}',_summary(report,html=True)).replace('{{PROFILE_CARDS}}',_profile_cards(profile_reports,source_commit,piper_task_report,piper_grip_report)).replace('{{SOURCE_COMMIT}}',source_commit)
    if '{{' in readme or '{{' in index:
        raise ValueError('Space template contains an unresolved publication placeholder.')
    payload.update({'README.md':readme.encode(),'index.html':index.encode(),'LICENSE':publisher._bytes(ROOT/'LICENSE',1024*1024)})
    provenance = {'schema':'dvidia.robot-simulation-space-provenance/v1','scope':'Recorded native simulation evidence; authored controller, no learned policy or physical qualification',
                  'source_commit':source_commit,'physical_robot_ready':False,'learned_policy':False,'live_browser_physics':False,
                  'task_profile':report['profile'],'benchmark_attempted':report['attempted'],'replay_attempts':len(motion['cases']),
                  'replay_excerpt_note':'Selected attempts are not the full success-rate denominator',
                  'source_replay_integrity':_record(replay_files['replay-integrity.json']),
                  'source_report':_record(replay_files['receipt/report.json']),
                  'source_task_integrity':_record(replay_files['receipt/integrity.json']),
                  'byte_preservation':'Every original replay file, complete raw task attempt, integrity file and model notice is retained byte-for-byte under replay/',
                  'software_license':'MIT','native_model_licenses':{'UR5e':'BSD-3-Clause','Robotiq 2F-85':'BSD-2-Clause'},
                  'profile_benchmarks':profile_records,'contact_task_benchmarks':task_records,
                  'profile_gallery':'UR5e replay and explicitly supplied PiPER pilot have authored contact-task evidence; separate PiPER/SO-101 numerical bundles remain unloaded mechanics evidence; all remaining profiles stay upcoming'}
    provenance['native_model_licenses'].update({p['profile']['name']:p['profile']['simulation_asset_license'] for p in profile_reports})
    if piper_task_report is not None:
        provenance['native_model_licenses']['PiPER contact pilot']='MIT'
    if piper_grip_report is not None:
        provenance['native_model_licenses']['PiPER guarded grip comparison']='MIT'
    payload['provenance.json'] = (json.dumps(provenance,indent=2)+'\n').encode()
    total = sum(len(raw) for raw in payload.values())
    if total > publisher.MAX_TOTAL_BYTES:
        raise ValueError('Space payload exceeds its publication budget.')
    for name,raw in payload.items():
        publisher._artifact_name(name)
        publisher._no_private_artifact_bytes(name,raw)
        if len(raw)>publisher.MAX_FILE_BYTES:
            raise ValueError('Space artifact exceeds its publication byte bound.')
    destination = Path(output).absolute()
    publisher._plain(destination)
    if destination.exists():
        raise FileExistsError('Choose a fresh publication directory.')
    (destination/'space').mkdir(parents=True,exist_ok=False)
    for name,raw in payload.items():
        path = destination/'space'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    plan = {'kind':publisher.KIND,'schema_version':1,'namespace':'Dvidia','private':False,
            'source_commit':source_commit,'expected_remote_commit':expected_remote_commit,
            'remove_files':removals,
            'repository':{'repo_id':publisher.REPO_ID,'repo_type':'space','space_sdk':'static','directory':'space',
                          'files':[{'path':name,**_record(raw)} for name,raw in sorted(payload.items())]},
            'total_bytes':total,'source_replay_integrity_sha256':provenance['source_replay_integrity']['sha256'],
            'scope':provenance['scope']}
    path = destination/'publication-plan.json'
    path.write_text(json.dumps(plan,indent=2)+'\n')
    publisher.checked_plan(path)
    return plan


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source-commit')
    parser.add_argument('--expected-remote-commit')
    parser.add_argument('--remove-starter-stylesheet',type=Path,
                        help='Reviewed local copy of the exact remote style.css; requires --expected-remote-commit')
    parser.add_argument('--profile-benchmark',type=Path,action='append',default=[])
    parser.add_argument('--piper-task-benchmark',type=Path)
    parser.add_argument('--piper-grip-benchmark',type=Path)
    args = parser.parse_args()
    result = stage(args.replay,args.output,source_commit=args.source_commit,expected_remote_commit=args.expected_remote_commit,
                   profile_benchmarks=args.profile_benchmark,remove_starter_stylesheet=args.remove_starter_stylesheet,
                   piper_task_benchmark=args.piper_task_benchmark,piper_grip_benchmark=args.piper_grip_benchmark)
    print(json.dumps({'repo_id':result['repository']['repo_id'],'files':len(result['repository']['files']),
                      'bytes':result['total_bytes'],'sdk':'static'},indent=2))
