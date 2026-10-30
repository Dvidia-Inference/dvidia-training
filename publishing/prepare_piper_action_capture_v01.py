"""Seal recorded PiPER capture evidence and an offline, data-only report.

No simulator, renderer, model training, network or publisher is executed here.
Original captures and independent audit records are copied byte for byte.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

from dvidia_training import action_demonstration as tape, piper_action_capture as capture

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='dvidia.piper-action-capture-public-kit/v0.1'
MAX_FILE=16*1024*1024
MAX_TOTAL=600*1024*1024
CASES={'nominal-full':(1000,'complete'),'nominal-repeat':(1000,'complete'),'interrupted-prefix':(2,'interrupted')}
SOURCES=capture.SOURCES
TRACKER={name for name,contract in tape.SIGNALS.items() if contract['origin']=='declared_rgbd_tracker'}


def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def identity(raw):return dict(bytes=len(raw),sha256=sha256(raw).hexdigest())


def read(path,limit=MAX_FILE):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Expected a regular immutable input file.')
    with path.open('rb') as stream:body=stream.read(limit+1)
    if len(body)>limit:raise ValueError('Input exceeds the public file byte limit.')
    return body


def relative(name):
    p=PurePosixPath(name)
    if (not name or p.is_absolute() or str(p)!=name or '\\' in name or ':' in name
            or any(x in ('','..','.') or x.startswith('.') for x in p.parts)):
        raise ValueError('Expected an ordinary relative public path.')
    return p


def inventory(folder,max_total=MAX_TOTAL):
    folder=Path(folder)
    if folder.is_symlink() or not folder.is_dir():raise ValueError('Expected an ordinary evidence directory.')
    files={}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink() or not(path.is_dir() or path.is_file()):raise ValueError('Evidence cannot contain links or special files.')
        if path.is_file():
            name=path.relative_to(folder).as_posix();relative(name);files[name]=identity(read(path))
    if sum(row['bytes'] for row in files.values())>max_total:raise ValueError('Evidence exceeds the public total byte limit.')
    return files


def public_text_check(name,body):
    if Path(name).suffix not in ('.json','.py','.md','.txt','.log','.html','.js') and Path(name).name!='LICENSE':return
    body.decode('utf-8')
    private=(b'/'+b'Users/',b'/'+b'private/',b'/'+b'var/folders/')
    scan=body
    if (name=='tests/test_piper_action_capture.py'
            and sha256(body).hexdigest()=='4c3d51c785a04c014ed11a39a288038e55b1a8be6c6b4653a340c349eb4eef49'):
        # This captured regression deliberately names a synthetic path, never
        # a user directory. Keep its original source bytes intact.
        scan=scan.replace(b'/'+b'private/example',b'<declared-synthetic-fixture>')
    if any(value in scan for value in private) or re.search(rb'(?:hf_|gh[pousr]_|apikey_)[A-Za-z0-9_]{24,}',scan):
        raise ValueError('Public text contains an absolute user path or credential-shaped bytes.')


def source_files(project,expected_mapping):
    if set(expected_mapping)!=set(SOURCES):raise ValueError('Expected all seven captured source identities.')
    result={}
    for name,wanted in expected_mapping.items():
        path=Path(project)/'src/dvidia_training'/name
        if sha256(read(path)).hexdigest()!=wanted:raise ValueError('Captured source bytes changed.')
        result[name]=path
    return result


def verify_case(folder,expected_steps,expected_status):
    folder=Path(folder);inspection=capture.inspect_capture(folder)
    report=tape._decode(read(folder/'capture.json'))
    raw=read(folder/'recording.json');recording=tape.validate_demonstration(raw)
    audit=tape._decode(read(folder.parent/(folder.name+'-audit.json')))
    snapshots=[recording['initial']]+[s['after'] for s in recording['steps']]
    if (report['status']!=expected_status or len(recording['steps'])!=expected_steps
            or report['recorded_complete_intervals']!=expected_steps or report['frame_files']!=2*len(snapshots)
            or report['strict_learning_targets_available'] is not False or report['tracker_implemented'] is not False
            or report['operator_corrections_collected'] is not False or report['task_success_verified'] is not False
            or report['physical_robot_ready'] is not False):raise ValueError('Capture status, count or unqualified scope differs.')
    expected_outcome='unverified' if expected_status=='complete' else 'unknown'
    if recording['outcome']['task_result']!=expected_outcome:raise ValueError('Native outcomes cannot qualify the recording task result.')
    for index,state in enumerate(snapshots):
        missing={name for name,sample in state['signals'].items() if sample['status']=='missing'}
        if missing!=TRACKER or any(state['signals'][name]['value'] is not None for name in TRACKER):
            raise ValueError('Exactly four object-tracker channels must remain missing/null.')
        if state['camera']['value']['index']!=index:raise ValueError('Capture must retain every distinct ordered boundary frame.')
    try:tape.validate_demonstration(recording,require_targets=True)
    except ValueError:pass
    else:raise ValueError('Unimplemented perception cannot qualify training-target export.')
    if (audit.get('schema')!='dvidia.native-action-capture-independent-readback/v0.1'
            or audit.get('recording')!=identity(raw) or audit.get('steps')!=expected_steps
            or audit.get('snapshots')!=len(snapshots) or audit.get('decoded_rgb_images')!=len(snapshots)
            or audit.get('decoded_depth_arrays')!=len(snapshots) or audit.get('outcome')!=recording['outcome']
            or audit.get('tracker_missing_channels_per_snapshot')!=[4]
            or audit.get('causal_camera_boundary_check') is not True
            or audit.get('training_pair_export_refused') is not True or audit.get('physical_robot_ready') is not False
            or audit.get('frame_artifact_check')!=inspection['frames']):
        raise ValueError('Stored independent audit does not bind the complete capture or refusal scope.')
    return dict(report=report,recording=recording,audit=audit)


def quantile(values,p):
    ordered=sorted(values)
    if not ordered:return None
    index=(len(ordered)-1)*p;low=math.floor(index);high=math.ceil(index)
    return ordered[low]+(ordered[high]-ordered[low])*(index-low)


def timing(report):
    frames=report['timing']['frames'];render=[x['render_encode_wall_seconds'] for x in frames]
    boundary=[x['full_boundary_wall_seconds'] for x in frames if 'full_boundary_wall_seconds' in x]
    return dict(capture_wall_seconds=report['timing']['wall_seconds'],render_samples=len(render),boundary_samples=len(boundary),
        render_p50_seconds=quantile(render,.5),render_p95_seconds=quantile(render,.95),
        boundary_p95_seconds=quantile(boundary,.95),boundary_max_seconds=max(boundary) if boundary else None,
        boundaries_over_20ms=sum(value>.02 for value in boundary),
        percentile_method='Linear interpolation at (n-1)*p in sorted observations.',
        scope='Shared host; simulation pauses for rendering. Capture wall excludes final capture.json/integrity sealing. Not a hard real-time or minimum-hardware result.')


def episode(case_id,result):
    recording=result['recording'];report=result['report'];steps=recording['steps']
    snapshots=[recording['initial']]+[s['after'] for s in steps];rows=[]
    vector=lambda action:list(action['joint_targets_rad'])+[action['gripper_position_m']]
    for index,state in enumerate(snapshots):
        signals=state['signals'];prior=steps[index-1] if index else None;camera=state['camera']['value']
        rows.append(dict(t=state['timestamp_seconds'],rgb=f'evidence/registered/{case_id}/'+camera['rgb']['path'],
            depth=f'evidence/registered/{case_id}/'+camera['depth']['path'],
            measured=signals['joint_position_rad']['value']+[signals['gripper_joint_position_m']['value'][0]],
            jaw_pair=signals['gripper_joint_position_m']['value'],
            velocity=signals['joint_velocity_rad_s']['value']+signals['gripper_joint_velocity_m_s']['value'],
            issued=vector(prior['command']['issued']) if prior else None,
            ramped=vector(prior['applied']['ramped_targets'] if prior else recording['initial_applied_targets']),
            native=prior['applied']['native_position_controls'] if prior else None))
    return dict(id=case_id,label={'nominal-full':'Full capture','nominal-repeat':'Same-source repeat','interrupted-prefix':'Interrupted prefix'}[case_id],
        status=report['status'],steps=len(steps),snapshots=len(snapshots),native_success=report['native_status']['native_info']['success'],
        recorded_task_result=recording['outcome']['task_result'],wall_seconds=report['timing']['wall_seconds'],
        warnings=report['native_status'].get('renderer_warnings',[]),rows=rows)


def compare_repeat(left,right,stored):
    a,b=left['recording'],right['recording'];snapshots=lambda r:[r['initial']]+[s['after'] for s in r['steps']]
    fields=('joint_position_rad','joint_velocity_rad_s','gripper_joint_position_m','gripper_joint_velocity_m_s','tcp_position_estimate_m','tcp_quaternion_estimate_wxyz')
    maxima={name:0. for name in fields};matching=0
    if (a['binding']!=b['binding'] or len(a['steps'])!=len(b['steps']) or a['goal_position_m']!=b['goal_position_m']
            or a['initial_applied_targets']!=b['initial_applied_targets']):
        raise ValueError('Repeat must bind the same configuration, initial ramped targets and step schedule.')
    for x,y in zip(snapshots(a),snapshots(b)):
        if x['timestamp_seconds']!=y['timestamp_seconds']:raise ValueError('Repeat timestamps differ.')
        for name in fields:maxima[name]=max(maxima[name],max(abs(v-w) for v,w in zip(x['signals'][name]['value'],y['signals'][name]['value'])))
        matching+=all(x['camera']['value'][name]['sha256']==y['camera']['value'][name]['sha256'] for name in ('rgb','depth'))
    same_commands=all(x['command']==y['command'] and x['reference']==y['reference'] for x,y in zip(a['steps'],b['steps']))
    same_controls=all(x['applied']==y['applied'] for x,y in zip(a['steps'],b['steps']))
    count=len(a['steps'])+1
    if (stored.get('schema')!='dvidia.native-action-capture-repeat-comparison/v0.1' or stored.get('snapshot_pairs')!=count
            or stored.get('control_steps')!=len(a['steps']) or stored.get('identical_rgb_and_depth_frame_pairs')!=matching
            or stored.get('maximum_absolute_difference_by_named_channel')!=maxima
            or stored.get('same_command_schedule')!=same_commands or stored.get('same_applied_controls')!=same_controls
            or stored.get('physical_robot_ready') is not False):raise ValueError('Stored repeat comparison differs from the sealed recordings.')
    return dict(snapshot_pairs=count,identical_rgb_and_depth_frame_pairs=matching,max_difference=max(maxima.values()),
                maximum_absolute_difference_by_named_channel=maxima,same_commands=same_commands,same_controls=same_controls)


def audit_inputs(study,*,project=ROOT,development_root=None):
    study=Path(study);project=Path(project);development_root=Path(development_root) if development_root else project/'runs'
    study_files=inventory(study);protocol=tape._decode(read(study/'protocol.json'))
    if (protocol.get('schema')!='dvidia.piper-action-capture-validation/v0.1' or protocol.get('physical_robot_ready') is not False
            or protocol.get('trained_skill') is not False or [c['id'] for c in protocol['cases']]!=list(CASES)):
        raise ValueError('Expected the complete three-case registered capture protocol.')
    results={case:verify_case(study/case,*expect) for case,expect in CASES.items()}
    source=results['nominal-full']['report']['metadata']['configuration']['implementation_sha256']
    if any(value['report']['metadata']['configuration']['implementation_sha256']!=source for value in results.values()):
        raise ValueError('Registered captures used different frozen source revisions.')
    selected=source_files(project,source);copy_files={}
    allowed_roots={'protocol.json','audit_recording.py','repeat-comparison.json'}|{f'{case}-audit.json' for case in CASES}|{f'{case}-console.log' for case in CASES}
    private_build_logs={'preview-wheel-build.log','preview-wheel-build02.log','preview-dependency-download.log',
                        'preview-installed-native-smoke.log','preview-installed-test.log','local-preview.png'}
    actual_roots={name for name in study_files if '/' not in name}
    if (not allowed_roots<=actual_roots or actual_roots-allowed_roots-private_build_logs
            or any(name.split('/')[0] not in set(CASES)|{'__pycache__'} for name in study_files if '/' in name)):
        raise ValueError('Registered study file roster differs from the retained three captures and audits.')
    for name in study_files:
        if name in private_build_logs or name.startswith('__pycache__/'):continue
        if '/' in name:destination='evidence/registered/'+name
        elif name=='protocol.json':destination=name
        elif name=='audit_recording.py':destination='tools/'+name
        elif name.endswith('-console.log'):destination='logs/registered/'+name
        else:destination='audits/'+name
        copy_files[destination]=study/name
    for name,path in selected.items():copy_files['source/'+name]=path
    for name in ('test_action_demonstration.py','test_piper_action_capture.py'):copy_files['tests/'+name]=project/'tests'/name
    for name in ('prepare_piper_action_capture_v01.py','test_prepare_piper_action_capture_v01.py'):
        copy_files['tools/'+name]=project/'publishing'/name
    copy_files['tools/test_piper_action_capture_report.js']=project/'publishing/test_piper_action_capture_report.js'
    for name in ('action-demonstrations.md','piper-action-capture.md'):copy_files['docs/'+name]=project/'docs'/name
    for name in inventory(project/'benchmarks/piper-action-capture-v01'):
        copy_files['compact-evidence/'+name]=project/'benchmarks/piper-action-capture-v01'/name
    copy_files['paper.md']=project/'reports/Native PiPER action capture v01.md'
    copy_files.update({'LICENSE':project/'LICENSE',
        'licenses/agilex_piper-MIT.txt':project/'third_party/robot-models/agilex_piper-LICENSE.txt',
        'licenses/mujoco-Apache-2.0.txt':project/'third_party/mujoco/LICENSE.txt',
        'licenses/mujoco-third-party.txt':project/'third_party/mujoco/THIRD_PARTY_NOTICES.txt'})
    for input_name,destination in [('piper-native-capture-smoke-01-20261030','sandbox-refusal'),('piper-native-capture-smoke-02-20261030','privileged-smoke')]:
        folder=development_root/input_name;capture.inspect_capture(folder)
        for name in inventory(folder):copy_files[f'evidence/development/{destination}/{name}']=folder/name
        copy_files[f'development/{destination}-console.log']=development_root/(input_name+'.log')
    copy_files['development/smoke02-readback.json']=development_root/'piper-native-capture-smoke-02-readback-20261030.json'
    copy_files['development/smoke02-warning.log']=development_root/'piper-native-capture-smoke-02-20261030.log'
    preview=development_root/'piper-action-capture-preview-package-20261030'
    for name in ('installed-smoke.json','wheel-manifest.json','requirements-installed-test.txt'):
        copy_files['preview/'+name]=preview/name
    wheel=preview/'wheel/dvidia_training-0.1.3.dev1-py3-none-any.whl'
    if identity(read(wheel))!=dict(bytes=5494221,sha256='4ff5542d104c9b35f486fff80d4d2eced529be7a4295391535bccf459cff07bb'):
        raise ValueError('Preview wheel differs from the independently installed source preview.')
    copy_files['preview/'+wheel.name]=wheel
    repeat=compare_repeat(results['nominal-full'],results['nominal-repeat'],tape._decode(read(study/'repeat-comparison.json')))
    summary=dict(schema=SCHEMA,scope='Engineering action/RGB-D capture validation, not a manipulation improvement or learned-skill benchmark.',
        cases=[dict(id=case,status=r['report']['status'],steps=len(r['recording']['steps']),snapshots=len(r['recording']['steps'])+1,
            frame_files=r['report']['frame_files'],frame_bytes=r['audit']['frame_artifact_check']['bytes'],
            native_success=r['report']['native_status']['native_info']['success'],recorded_task_result=r['recording']['outcome']['task_result'],
            timing=timing(r['report']),renderer_warnings=r['report']['native_status'].get('renderer_warnings',[])) for case,r in results.items()],
        repeat=repeat,missing_tracker_channels=4,
        depth_warning='CGL reported ARB_clip_control unavailable while mjDEPTH_ZEROFAR was requested. Depth precision is limited; exact repeated arrays do not establish depth accuracy.',
        timing_scope='Shared host and paused simulation clock; final capture.json/integrity sealing excluded from capture wall. No hard real-time or minimum-hardware claim.',
        preview=dict(version='0.1.3.dev1',status='source preview, distinct from the stable data-only SDK 0.1.2',wheel=identity(read(wheel))),
        source_sha256=source,protocol=identity(read(study/'protocol.json')),
        excluded_build_artifacts=dict(names=sorted(name for name in study_files if name in private_build_logs or name.startswith('__pycache__/')),
            reason='Local build/install consoles, UI preview screenshot and Python bytecode are not registered capture evidence; portable preview proof, requirements and wheel manifest are included separately.'),
        physical_robot_ready=False,trained_skill=False)
    return dict(summary=summary,episodes=[episode(case,r) for case,r in results.items()],copy_files=copy_files)


def report_assets():
    return {'index.html':read(ROOT/'publishing/piper_action_capture_report.html'),
            'report.js':read(ROOT/'publishing/piper_action_capture_report.js')}


README='''# DVIDIA · Native PiPER action capture v0.1

This kit preserves three registered native captures, all image/depth bytes,
motor evidence, independent readback audits and their failure/precision limits.
It is an engineering capture milestone, not a learned skill or improvement benchmark.

Extract this entire archive, then run `python -m http.server 8000` in this
directory and open http://127.0.0.1:8000. The browser reads recorded frames and
actuator traces; it executes no physics and connects to no robot or remote API.

Two 20-second runs repeat one fixed start, source controller and seed. Their
native task-success flags remain separate from the recording's unverified task
result. The interrupted prefix retains an unknown result. Four object-tracker
channels are missing/null throughout, so strict training-target export refuses.
CGL depth precision warnings remain explicit; byte equality is not accuracy.

The `preview/` wheel is source preview **0.1.3.dev1**, distinct from stable
data-only SDK 0.1.2. See its installation smoke proof and platform-specific
requirements. No dependency wheels are bundled. Native capture requires the
optional native dependencies, pinned PiPER assets and a functioning OpenGL
context; no minimum hardware or hard real-time performance is established.
The seven source extracts preserve captured revisions; they are not a full
standalone source checkout. Full project: https://github.com/Dvidia-Inference/dvidia-training

Reproducing `tools/audit_recording.py` additionally needs Pillow 12.0.0 and
NumPy 2.5.3, matching the archived independent audits. Pillow is not a native
capture/inspection product dependency; dependency wheels are not bundled.

Original DVIDIA code/evidence: MIT. PiPER model asset notices: MIT; MuJoCo:
Apache-2.0 and retained notices. Dependencies keep their own licenses.
No learned weights, private footage, credentials or physical-arm readiness claim.

`manifest.json` binds every other file. The adjacent deterministic ZIP and
receipt are outside this directory. Hashes prove consistency, not authenticity.
'''


def archive(folder,target):
    folder=Path(folder).absolute();target=Path(target).absolute()
    if target==folder or folder in target.parents:raise ValueError('Archive must remain outside the sealed folder.')
    expected=inventory(folder)
    with Path(target).open('xb') as stream,zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for name in sorted(expected):
            path=folder/relative(name)
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            entry.create_system=3;entry.external_attr=0o100644<<16;entry.compress_type=zipfile.ZIP_DEFLATED
            body=read(path)
            if identity(body)!=expected[name]:raise ValueError('Sealed public file changed during archiving.')
            zipped.writestr(entry,body,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    if inventory(folder)!=expected:raise ValueError('Sealed public inventory changed during archiving.')
    return identity(read(target,MAX_TOTAL))


def prepare(study,output,*,project=ROOT,development_root=None):
    output=Path(output)
    if output.exists() or output.with_suffix('.zip').exists() or output.with_suffix('.receipt.json').exists():
        raise ValueError('Use a fresh public kit destination, adjacent archive and receipt.')
    audited=audit_inputs(study,project=project,development_root=development_root)
    inputs={}
    for name,path in audited['copy_files'].items():
        relative(name);body=read(path);public_text_check(name,body);inputs[name]=(Path(path),identity(body))
    generated=report_assets()
    generated.update({'README.md':README.encode(),'summary.json':encoded(audited['summary']),
        'viewer-data.json':encoded(dict(schema=SCHEMA,summary=audited['summary'],episodes=audited['episodes'])),
        'development/README.md':('Development evidence only: the sandbox CoreGraphics refusal and earlier privileged 0.1-second smoke predate the registered protocol and frozen producer revision. They are retained separately, not included in the three registered case denominator. The earlier smoke did not capture its depth warning into structured metadata; the original warning log is retained separately.\n').encode(),
        'kit-audit.json':encoded(dict(schema=SCHEMA+'/read-only-audit',input_files=len(inputs),
            source_authentication=False,native_physics_rerun=False,training_executed=False,
            scope='Complete original capture-integrity and referenced-frame hashes rechecked; stored independent decoded-image audits bound to recording bytes. Repeat comparison rechecked from recordings. This packager does not re-decode images or validate depth accuracy.'))})
    if set(generated)&set(inputs):raise ValueError('Generated report may not overwrite original evidence.')
    output.mkdir(parents=True)
    for name,(path,wanted) in inputs.items():
        relative(name);body=read(path)
        if identity(body)!=wanted:raise ValueError('Input changed during public copying.')
        public_text_check(name,body);destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(body)
    for name,body in generated.items():
        relative(name);public_text_check(name,body);destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(body)
    # Three original posters support a compact research article without copying
    # the complete frame collection into the research site's source tree.
    for item in audited['episodes']:
        if not item['rows']:continue
        row=item['rows'][min(len(item['rows'])-1,500)];body=read(output/row['rgb'])
        destination=output/'posters'/(item['id']+'.png');destination.parent.mkdir(exist_ok=True);destination.write_bytes(body)
    files=inventory(output)
    manifest=dict(schema=SCHEMA,files=files,excluded_self='manifest.json',physical_robot_ready=False,trained_skill=False)
    (output/'manifest.json').write_bytes(encoded(manifest))
    zipped=output.with_suffix('.zip');zip_id=archive(output,zipped)
    receipt=dict(schema=SCHEMA,files=len(files)+1,bytes=sum(v['bytes'] for v in files.values())+len(encoded(manifest)),
        manifest=identity(encoded(manifest)),archive=dict(name=zipped.name,**zip_id),physical_robot_ready=False,trained_skill=False)
    output.with_suffix('.receipt.json').write_bytes(encoded(receipt));return receipt


def prepare_web(kit,output,*,ffmpeg='/opt/homebrew/bin/ffmpeg'):
    """Small recorded-video derivative; originals remain solely in sealed kit.

    FFmpeg encodes actual RGB PNGs, never synthesizes frames or runs physics.
    Every input-frame hash and probed output count remains in the receipt.
    """
    kit,output=Path(kit),Path(output)
    if output.exists():raise ValueError('Use a fresh compact web derivative directory.')
    manifest_raw=read(kit/'manifest.json');manifest=tape._decode(manifest_raw);observed=inventory(kit)
    if observed!={**manifest['files'],'manifest.json':identity(manifest_raw)}:
        raise ValueError('Source kit differs from its sealed complete manifest.')
    archive_receipt=tape._decode(read(kit.with_suffix('.receipt.json')))
    if archive_receipt['manifest']!=identity(manifest_raw):raise ValueError('Source archive receipt differs from the kit seal.')
    data=tape._decode(read(kit/'viewer-data.json'))
    if [item['id'] for item in data['episodes']]!=list(CASES):raise ValueError('Compact derivative requires the exact complete registered episode roster.')
    data['download_archive_url']='https://huggingface.co/datasets/Dvidia/piper-action-capture-v01/resolve/main/piper-action-capture-v01.zip?download=true'
    output.mkdir(parents=True);(output/'media').mkdir()
    command_version=subprocess.run([ffmpeg,'-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    derived=[];ffprobe=str(Path(ffmpeg).with_name('ffprobe'))
    for item in data['episodes']:
        rows=item['rows'];case=item['id'];target=output/'media'/(case+'.mp4')
        if case not in CASES or len(rows)!=CASES[case][0]+1:raise ValueError('Compact derivative requires the complete registered episode roster.')
        sources=[dict(path=row['rgb'],**identity(read(kit/relative(row['rgb'])))) for row in rows]
        for index,row in enumerate(rows):
            if row['rgb']!=f'evidence/registered/{case}/frames/rgb-{index:04}.png':raise ValueError('Derived video requires the exact contiguous saved RGB frame paths.')
        subprocess.run([ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-framerate','50','-start_number','0',
            '-i',str(kit/f'evidence/registered/{case}/frames/rgb-%04d.png'),'-frames:v',str(len(rows)),
            '-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-an','-map_metadata','-1',
            '-fflags','+bitexact','-flags:v','+bitexact','-movflags','+faststart',str(target)],check=True,capture_output=True)
        probe=subprocess.run([ffprobe,'-v','error','-count_frames','-select_streams','v:0','-show_entries',
            'stream=nb_read_frames,width,height,r_frame_rate,duration,codec_name,pix_fmt','-of','json',str(target)],check=True,capture_output=True)
        streams=tape._decode(probe.stdout)['streams']
        dimensions=tape._decode(read(kit/f'audits/{case}-audit.json'))['image_shape_hwc']
        if (len(streams)!=1 or int(streams[0]['nb_read_frames'])!=len(rows) or streams[0]['r_frame_rate']!='50/1'
                or streams[0]['codec_name']!='h264' or streams[0]['pix_fmt']!='yuv420p'
                or streams[0]['width']!=dimensions[1] or streams[0]['height']!=dimensions[0]):
            raise ValueError('Derived video frame count or encoding differs from the declared RGB display.')
        item['video']='media/'+case+'.mp4';item['raw_evidence_available']=False
        derived.append(dict(case_id=case,input_rgb_frames=sources,input_frame_count=len(rows),source_time_start_s=rows[0]['t'],
            source_time_end_s=rows[-1]['t'],constant_display_fps=50,ffprobe=streams[0],output=dict(path=item['video'],**identity(read(target))),
            scope='Lossy H.264 display derived only from saved RGB PNGs. No depth rendering, generated motion, native replay or camera accuracy claim. The final displayed frame adds one 20 ms container interval beyond the last recorded timestamp.'))
    copies=['summary.json','protocol.json','LICENSE','paper.md','development/README.md','development/smoke02-warning.log',
        'evidence/development/sandbox-refusal/capture.json','evidence/development/privileged-smoke/capture.json']
    copies += ['audits/'+case+'-audit.json' for case in CASES]+['audits/repeat-comparison.json']
    copies += ['posters/'+case+'.png' for case in CASES]
    copies += [name for name in observed if name.startswith('licenses/')]
    for name in copies:
        body=read(kit/relative(name));destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(body)
    for name,body in report_assets().items():(output/name).write_bytes(body)
    (output/'README.md').write_text('# DVIDIA · Recorded PiPER capture viewer\n\nThis small web derivative displays lossy RGB video and all saved boundary telemetry. It runs no physics and supplies no learned skill. Download the complete original capture kit, source preview wheel, PNG/NPY frames and audits from https://huggingface.co/datasets/Dvidia/piper-action-capture-v01 . The derivation receipt lists every original RGB input hash and probed video frame count. Depth precision is limited; object tracking is missing and strict training export remains refused.\n')
    (output/'viewer-data.json').write_bytes(json.dumps(data,separators=(',',':'),allow_nan=False).encode()+b'\n')
    receipt=dict(schema=SCHEMA+'/web-derivative',source_kit_manifest=identity(manifest_raw),source_archive=archive_receipt['archive'],
        ffmpeg_version=command_version,episodes=derived,physical_robot_ready=False,trained_skill=False,
        scope='Recorded RGB display derivative with all boundary telemetry; complete original PNG/NPY and motor evidence remain in the source archive. No physical/simulation execution.')
    (output/'derivation-receipt.json').write_bytes(encoded(receipt))
    if inventory(kit)!=observed:raise ValueError('Original source kit changed during video derivation.')
    files=inventory(output,max_total=32*1024*1024)
    for name in files:public_text_check(name,read(output/name))
    (output/'manifest.json').write_bytes(encoded(dict(schema=SCHEMA+'/web',files=files,excluded_self='manifest.json',physical_robot_ready=False)))
    return dict(files=len(files)+1,bytes=sum(row['bytes'] for row in files.values())+(output/'manifest.json').stat().st_size,
                manifest=identity(read(output/'manifest.json')),source_kit_manifest=identity(manifest_raw))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--project',default=ROOT);parser.add_argument('--development-root');args=parser.parse_args(argv)
    try:print(json.dumps(prepare(args.study,args.output,project=args.project,development_root=args.development_root),indent=2));return 0
    except (ValueError,OSError,KeyError,TypeError) as error:print(json.dumps(dict(status='refused',reason=str(error))));return 1


if __name__=='__main__':raise SystemExit(main())
