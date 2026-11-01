"""Registered static RGB-D fixtures, isolated estimates, then offline scoring.

This is a perception experiment, not a manipulation rollout. Native poses are
rendering/evaluation fixtures only. Every scheduled ID is journalled before
rendering; incomplete attempts cannot silently become a smaller denominator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from .piper_action_capture import CaptureConfig, _NativeSession, _json, _png, _record

SCHEMA = 'dvidia.piper-rgbd-benchmark/v1'
VARIANTS = ('clear', 'partial_occlusion', 'tilt15', 'tilt30', 'absent',
            'occluded', 'ambiguous', 'invalid_depth', 'stale', 'future')
SOURCES = ('piper_rgbd_tracker.py', 'piper_rgbd_benchmark.py',
           'piper_action_capture.py', 'action_demonstration.py', 'piper_task.py',
           'reference_task.py', 'reference_sim.py', 'robot_profiles.py', 'robot_profiles.lock.json')
INPUTS = ('rgb.npy', 'depth.npy', 'input.json')
SIGNALS = ('object_position_estimate_m', 'object_quaternion_estimate_wxyz',
           'object_position_uncertainty_m', 'object_orientation_uncertainty_rad')
TIMING_SCOPE = ('Authored timestamp-ordering fixtures at a paused synthetic scene; '
                'wall/CPU timers measure offline estimator computation, not physical '
                'exposure, asynchronous delivery or robot control latency.')


def _load(path):
    def reject(value):
        raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _identity(path):
    path = Path(path)
    return dict(bytes=path.stat().st_size, sha256=_digest(path))


def hashes():
    root = Path(__file__).parent
    return {name: _digest(root / name) for name in SOURCES}


def runtime():
    # Distribution metadata does not initialize MuJoCo or a graphics context.
    distribution = importlib.metadata.distribution('mujoco')
    renderer = distribution.locate_file('mujoco/rendering/classic/renderer.py')
    return dict(python=platform.python_version(), numpy=np.__version__,
                mujoco=distribution.version, renderer_sha256=_digest(renderer),
                system=platform.system(), machine=platform.machine())


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(311301)
    groups = []
    for index in range(24):
        groups.append(dict(id=f'pose{index:02d}', split='development' if index < 8 else 'confirmation',
                           position_m=[float(rng.uniform(.27, .36)), float(rng.uniform(-.075, .065)),
                                       float(rng.choice([.08, .105, .13]))],
                           yaw_rad=float(rng.uniform(0, math.pi/2))))
    protocol = dict(schema=SCHEMA, seed=311301, groups=groups,
                    resolutions=[[160,120], [320,240]], variants=list(VARIANTS),
                    scope='Authored static rendered fixtures in the native PiPER scene; no task execution or new training actions.',
                    estimator_inputs='RGB, optical-Z depth, declared camera calibration and causal frame timestamps only.',
                    expected_available=['clear'], challenge_positive=['partial_occlusion'],
                    expected_abstain=[v for v in VARIANTS if v not in ('clear','partial_occlusion')],
                    provisional_gate=dict(clear_coverage_min=.9, clear_position_p95_m_max=.01,
                                          clear_yaw_p95_rad_max=math.radians(10), negative_false_available_max=0),
                    symmetry='Upright yaw modulo pi/2; tilted cases are unsupported challenges.',
                    uncertainty='Heuristic scales, not probabilistic coverage guarantees.',
                    split_unit='All variants and resolutions of a physical pose group stay in one split.',
                    depth_accuracy_verified=False)
    _json(output/'protocol.json', protocol)
    return protocol


def _schedule(protocol, split):
    if split not in ('development', 'confirmation') or protocol.get('schema') != SCHEMA:
        raise ValueError('Unknown split or protocol schema')
    groups = protocol['groups']; variants = protocol['variants']; resolutions = protocol['resolutions']
    if (len({g['id'] for g in groups}) != len(groups)
            or any(g['split'] not in ('development', 'confirmation') for g in groups)
            or len(set(variants)) != len(variants) or set(variants) != set(VARIANTS)
            or resolutions != [[160,120], [320,240]]):
        raise ValueError('Protocol has duplicate groups, unsupported variants or resolutions')
    rows = []
    for width, height in resolutions:
        for group in groups:
            if group['split'] != split:
                continue
            if not isinstance(group['id'], str) or not group['id'].isalnum():
                raise ValueError('Unsafe pose ID')
            for variant in variants:
                rows.append(dict(id=f'{group["id"]}-{width}-{variant}', group_id=group['id'],
                                 variant=variant, width=width, height=height, frame_index=len(rows)))
    if not rows:
        raise ValueError('No scheduled fixtures for split')
    return rows


def _npy(path, value):
    buf=io.BytesIO(); np.save(buf,value,allow_pickle=False); return _record(path,buf.getvalue())


def _error(error):
    # Paths and unknown library text are unnecessary for the public receipt.
    return dict(type=type(error).__name__, reason='fixture_or_pipeline_operation_failed')


def _visual_box(session, position, size, rotation, rgba):
    """Render-only geometry, with a segmentation ID so it also exists in depth."""
    scene = session.renderer.scene
    if scene.ngeom >= scene.maxgeom:
        raise ValueError('Scene geometry capacity exhausted')
    geom = scene.geoms[scene.ngeom]
    session.mj.mjv_initGeom(geom, session.mj.mjtGeom.mjGEOM_BOX,
                          np.asarray(size, dtype=np.float64), np.asarray(position, dtype=np.float64),
                          np.asarray(rotation, dtype=np.float64).reshape(9), np.asarray(rgba, dtype=np.float32))
    # Classic depth rendering enables segmentation/IDCOLOR. A default segid -1
    # would make an RGB-only occluder and invalidate the aligned-depth fixture.
    geom.segid = max([g.segid for g in scene.geoms[:scene.ngeom]] + [-1]) + 1
    scene.ngeom += 1
    return geom.segid


def _renderer_binding(session, header, metadata):
    base_rotation = session.env.data.xmat[session.base].reshape(3,3)
    base_position = session.env.data.xpos[session.base]
    if (not np.allclose(base_rotation, np.eye(3), atol=1e-9, rtol=0)
            or not np.allclose(base_position, 0., atol=1e-9, rtol=0)):
        raise ValueError('Authored world-coordinate fixture requires identity fixed robot base')
    geom = session.env.block_geom
    return dict(header_binding=header['binding'], assets=metadata['assets'],
                runtime=metadata['runtime'], camera=metadata['camera'],
                calibration=session._calibration(),
                base_position_m=base_position.tolist(), base_rotation=base_rotation.tolist(),
                object_half_size_m=session.env.model.geom_size[geom].tolist(),
                object_rgba=session.env.model.geom_rgba[geom].tolist(),
                orientation_equivalence='upright yaw modulo pi/2')


def _check_freeze(output):
    from .piper_rgbd_tracker import TrackerConfig
    frozen = _load(output/'freeze.json')
    if (frozen['sources'] != hashes() or frozen['protocol_sha256'] != _digest(output/'protocol.json')
            or frozen['config'] != asdict(TrackerConfig()) or frozen['runtime'] != runtime()):
        raise ValueError('Confirmation source, protocol, configuration or runtime differs from freeze')
    for name, identity in frozen['development_evidence'].items():
        if _identity(output/'development'/name) != identity:
            raise ValueError('Development evidence changed after freeze')
    return frozen


def _render_fixture(session, item, group, folder, baseline, address, original_rgba, binding):
    session.env.data.qpos[:] = baseline
    session.env.data.qvel[:] = 0.
    variant = item['variant']; yaw = group['yaw_rad']
    tilt = math.radians(15 if variant=='tilt15' else 30 if variant=='tilt30' else 0)
    q = np.array([math.cos(yaw/2)*math.cos(tilt/2), math.cos(yaw/2)*math.sin(tilt/2),
                  math.sin(yaw/2)*math.sin(tilt/2), math.sin(yaw/2)*math.cos(tilt/2)])
    session.env.data.qpos[address:address+3] = group['position_m']
    session.env.data.qpos[address+3:address+7] = q
    geom = session.env.block_geom
    session.env.model.geom_rgba[geom] = original_rgba
    if variant == 'absent': session.env.model.geom_rgba[geom,3] = 0.
    session.mj.mj_forward(session.env.model, session.env.data)
    native_time = float(session.env.data.time)
    with session.sim.native_warnings(session.renderer_warnings):
        session.renderer.disable_depth_rendering()
        session.renderer.update_scene(session.env.data, camera=session.camera)
        calibration = session._calibration()
        if calibration != binding['calibration']:
            raise ValueError('Camera calibration changed inside the fixture set')
        base_rotation = session.env.data.xmat[session.base].reshape(3,3)
        base_position = session.env.data.xpos[session.base].copy()
        world_position = session.env.data.xpos[session.env.block].copy()
        object_rotation = base_rotation.T @ session.env.data.xmat[session.env.block].reshape(3,3)
        base_q = np.empty(4)
        session.mj.mju_mat2Quat(base_q, object_rotation.reshape(9))
        base_yaw = math.atan2(object_rotation[1,0], object_rotation[0,0])
        base_tilt = math.acos(float(np.clip(object_rotation[2,2], -1., 1.)))
        optical_world = base_rotation @ np.asarray(calibration['base_from_camera']).reshape(4,4)[:3,:3]
        custom_segids = []
        if variant in ('occluded', 'partial_occlusion'):
            centre = world_position-.06*optical_world[:,2]
            if variant == 'partial_occlusion': centre = centre+.026*optical_world[:,0]
            custom_segids.append(_visual_box(session, centre,
                [.06 if variant=='occluded' else .025,.06,.002], optical_world, [.4,.43,.46,1]))
        if variant == 'ambiguous':
            custom_segids.append(_visual_box(session, world_position+np.array([0.,.09,0.]),
                                           [.02]*3, np.eye(3), original_rgba))
        rgb = session.renderer.render().copy()
        session.renderer.enable_depth_rendering()
        try: depth = session.renderer.render().copy().astype('<f4')
        finally: session.renderer.disable_depth_rendering()
    if (rgb.shape != (item['height'], item['width'], 3) or rgb.dtype != np.uint8
            or depth.shape != (item['height'], item['width']) or not np.isfinite(depth).all()
            or np.any(depth <= 0) or float(session.env.data.time) != native_time):
        raise ValueError('Renderer did not supply finite aligned RGB-D at one static boundary')
    if variant == 'invalid_depth': depth[:] = np.nan
    _record(folder/'rgb.png', _png(rgb)); _npy(folder/'rgb.npy', rgb); _npy(folder/'depth.npy', depth)
    timing = dict(frame_index=item['frame_index'], captured_at_seconds=.2,
                  available_at_seconds=.201 if variant=='future' else .2,
                  decision_at_seconds=.301 if variant=='stale' else .2)
    _json(folder/'input.json', dict(calibration=calibration, timing=timing))
    # The estimator process never opens this file; truth is joined only offline.
    _json(folder/'truth.json', dict(position_m=(base_rotation.T@(world_position-base_position)).tolist(),
        quaternion_wxyz=base_q.tolist(), yaw_rad=base_yaw, tilt_rad=base_tilt,
        present=variant!='absent', base_frame='robot_base', custom_visual_segids=custom_segids))


def render(output, assets, split):
    output=Path(output); protocol=_load(output/'protocol.json'); scheduled=_schedule(protocol,split)
    frozen = _check_freeze(output) if split=='confirmation' else None
    target=output/split; target.mkdir(exist_ok=False)
    source_hashes=hashes(); runtime_identity=runtime()
    _json(target/'scheduled.json',dict(schema=SCHEMA,split=split,records=scheduled,
                                      protocol_sha256=_digest(output/'protocol.json')))
    groups={g['id']:g for g in protocol['groups']}; records=[]; bindings={}; session_errors=[]; interrupted=False
    for width,height in protocol['resolutions']:
        session=None; setup_error=None; binding=None; baseline=None; address=None; original_rgba=None
        items=[r for r in scheduled if r['width']==width]
        try:
            if interrupted: raise InterruptedError('Previous fixture was interrupted')
            session=_NativeSession(assets,CaptureConfig(duration_s=1.,width=width,height=height))
            header,_,metadata=session.open()
            binding=_renderer_binding(session,header,metadata)
            if frozen is not None and binding != frozen['renderer_bindings'].get(str(width)):
                raise ValueError('Renderer, assets, model, calibration or object differs from freeze')
            bindings[str(width)]=binding
            _json(target/f'renderer-{width}.json',dict(metadata=metadata,binding=binding))
            baseline=session.env.data.qpos.copy()
            jid=session.mj.mj_name2id(session.env.model,session.mj.mjtObj.mjOBJ_JOINT,'task_block_free')
            if jid < 0: raise ValueError('Missing free object joint')
            address=int(session.env.model.jnt_qposadr[jid])
            original_rgba=session.env.model.geom_rgba[session.env.block_geom].copy()
        except (Exception,KeyboardInterrupt) as exc:
            setup_error=_error(exc)
            if isinstance(exc,KeyboardInterrupt): interrupted=True
        for item in items:
            folder=target/item['id']; folder.mkdir()
            started=time.perf_counter(); error=None; status='rendered'
            if interrupted or setup_error:
                status='unattempted'; error=setup_error or dict(type='Interrupted',reason='preceding_interruption')
            else:
                _json(folder/'started.json',dict(**item,status='started'))
                try:
                    _render_fixture(session,item,groups[item['group_id']],folder,baseline,address,original_rgba,binding)
                except (Exception,KeyboardInterrupt) as exc:
                    status='failed'; error=_error(exc)
                    if isinstance(exc,KeyboardInterrupt): interrupted=True
            files={p.name:_identity(p) for p in sorted(folder.iterdir()) if p.is_file()}
            row=dict(**item,render_status=status,error=error,files=files,
                     render_wall_seconds=time.perf_counter()-started,
                     renderer_warnings=list(session.renderer_warnings) if session is not None else [])
            _json(folder/'receipt.json',row)
            records.append(dict(**row,receipt_sha256=_digest(folder/'receipt.json')))
        if session is not None:
            try: session.close()
            except Exception as exc: session_errors.append(dict(width=width,error=_error(exc)))
    _json(target/'manifest.json',dict(schema=SCHEMA,split=split,records=records,source_hashes=source_hashes,
        source_stable=source_hashes==hashes(),runtime=runtime_identity,renderer_bindings=bindings,
        protocol_sha256=_digest(output/'protocol.json'),scheduled_sha256=_digest(target/'scheduled.json'),
        session_errors=session_errors,timing_scope=TIMING_SCOPE,depth_accuracy_verified=False))
    return dict(scheduled=len(scheduled),rendered=sum(r['render_status']=='rendered' for r in records),split=split,
                pipeline_complete=not session_errors and source_hashes==hashes()
                and all(r['render_status']=='rendered' for r in records))


def _manifest(output, split):
    """Validate complete schedule and journals without reading truth or image bytes."""
    folder=output/split; protocol=_load(output/'protocol.json'); manifest=_load(folder/'manifest.json')
    scheduled=_schedule(protocol,split)
    if (manifest['schema']!=SCHEMA or manifest['split']!=split
            or manifest['protocol_sha256']!=_digest(output/'protocol.json')
            or manifest['scheduled_sha256']!=_digest(folder/'scheduled.json')
            or _load(folder/'scheduled.json')['records']!=scheduled or not manifest['source_stable']):
        raise ValueError('Manifest protocol, schedule or source changed')
    rows=manifest['records']
    if len(rows)!=len(scheduled) or [r['id'] for r in rows]!=[r['id'] for r in scheduled]:
        raise ValueError('Manifest must contain each scheduled ID exactly once in order')
    for item, registered in zip(rows,scheduled):
        if any(item.get(k)!=v for k,v in registered.items()) or item['render_status'] not in ('rendered','failed','unattempted'):
            raise ValueError('Fixture differs from registered schedule')
        receipt=folder/item['id']/'receipt.json'
        if _digest(receipt)!=item['receipt_sha256'] or _load(receipt)!={k:v for k,v in item.items() if k!='receipt_sha256'}:
            raise ValueError('Fixture receipt changed')
        if item['render_status']=='rendered' and not set((*INPUTS,'rgb.png','truth.json')).issubset(item['files']):
            raise ValueError('Rendered fixture has missing artifacts')
        for name in item['files']:
            if Path(name).name!=name or name in ('','.', '..'):
                raise ValueError('Invalid artifact filename')
    return protocol,manifest


def _verify_files(folder, item, names):
    for name in names:
        if _identity(folder/item['id']/name)!=item['files'][name]:
            raise ValueError('Sealed fixture artifact changed: '+name)


def freeze(output):
    from .piper_rgbd_tracker import TrackerConfig
    output=Path(output)
    if (output/'confirmation').exists(): raise ValueError('Cannot freeze after confirmation has started')
    _,manifest=_manifest(output,'development')
    if set(manifest['renderer_bindings'])!={'160','320'} or manifest['runtime']!=runtime():
        raise ValueError('Both development renderer bindings and current runtime are required')
    for item in manifest['records']:
        _verify_files(output/'development',item,item['files'])
    predictions=_load(output/'development/predictions.json')
    report=_load(output/'development/report.json')
    current_sources=hashes()
    expected_ids=[item['id'] for item in manifest['records']]
    if (predictions['schema']!=SCHEMA or predictions['split']!='development'
            or predictions['source_hashes']!=current_sources or predictions['runtime']!=runtime()
            or predictions['manifest_sha256']!=_digest(output/'development/manifest.json')
            or predictions['protocol_sha256']!=_digest(output/'protocol.json')
            or [item['id'] for item in predictions['predictions']]!=expected_ids):
        raise ValueError('Development predictions do not bind the current candidate and exact rendered fixtures')
    if (report['schema']!=SCHEMA or report['split']!='development'
            or report.get('scorer_source_hashes')!=current_sources
            or report['predictions_sha256']!=_digest(output/'development/predictions.json')
            or report['manifest_sha256']!=_digest(output/'development/manifest.json')
            or report['scheduled']!=len(expected_ids) or [item['id'] for item in report['rows']]!=expected_ids):
        raise ValueError('Development report does not bind the current scorer and exact predictions')
    # Preserve rendering lineage separately: development may intentionally reuse
    # earlier immutable images, but its predictions and scorer must be current.
    rendering_lineage=dict(source_hashes=manifest['source_hashes'],
        manifest_sha256=_digest(output/'development/manifest.json'),
        reused_earlier_rendering_sources=manifest['source_hashes']!=current_sources,
        scope='Exact rendered files and receipts verified; reuse never substitutes earlier estimator predictions.')
    # Preserve the exact development evidence used to choose this frozen candidate.
    evidence={name:_identity(output/'development'/name) for name in ('manifest.json','predictions.json','report.json')}
    _json(output/'freeze.json',dict(schema=SCHEMA,sources=hashes(),config=asdict(TrackerConfig()),
        protocol_sha256=_digest(output/'protocol.json'),runtime=runtime(),
        renderer_bindings=manifest['renderer_bindings'],development_evidence=evidence,
        development_rendering_lineage=rendering_lineage,
        development_only=True,created_unix_seconds=time.time()))


def _valid_result(result, timing):
    if not isinstance(result,dict) or result.get('status') not in ('available','missing','stale','refused'):
        raise ValueError('Malformed estimator status')
    if not isinstance(result.get('reason'),str): raise ValueError('Missing estimator reason')
    # Ensure no nonfinite diagnostic can abort later evidence serialization.
    json.dumps(result,allow_nan=False)
    if any(result.get(k)!=v for k,v in timing.items()): raise ValueError('Estimator source frame/timestamps changed')
    if result['status']=='available':
        if not timing['captured_at_seconds']<=timing['available_at_seconds']<=timing['decision_at_seconds']:
            raise ValueError('Available estimate is noncausal')
        if timing['decision_at_seconds']-timing['captured_at_seconds']>.1+1e-12:
            raise ValueError('Available estimate is stale')
        if result.get('base_frame')!='robot_base': raise ValueError('Wrong estimator frame')
        for name,size in zip(SIGNALS[:2],(3,4)):
            value=result.get(name)
            if not isinstance(value,list) or len(value)!=size or not all(type(x) in (int,float) and math.isfinite(x) for x in value):
                raise ValueError('Invalid pose output')
        q=result[SIGNALS[1]]
        if not math.isclose(sum(x*x for x in q),1.,abs_tol=1e-6,rel_tol=0): raise ValueError('Quaternion is not normalized')
        # Only upright canonical yaw is in this estimator's declared contract.
        if abs(q[1])>1e-8 or abs(q[2])>1e-8: raise ValueError('Output exceeds upright orientation contract')
        for name in SIGNALS[2:]:
            value=result.get(name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0: raise ValueError('Invalid uncertainty scale')
    elif any(result.get(name) is not None for name in SIGNALS):
        raise ValueError('Abstention carries a fabricated pose or scale')
    nested=result.get('signals')
    if not isinstance(nested,dict) or set(nested)!=set(SIGNALS):
        raise ValueError('Estimator must supply the exact four nested observation signals')
    if result['status']=='available':
        age=timing['decision_at_seconds']-timing['captured_at_seconds']
        if type(result.get('age_seconds')) not in (int,float) or result['age_seconds']!=age:
            raise ValueError('Top-level estimator age differs from source frame')
        for name in SIGNALS:
            value=result[name]
            expected=dict(status='available',captured_at_seconds=timing['captured_at_seconds'],
                available_at_seconds=timing['available_at_seconds'],age_seconds=age,
                value=value if isinstance(value,list) else [value])
            actual=nested[name]
            numeric_fields=('captured_at_seconds','available_at_seconds','age_seconds')
            if (not isinstance(actual,dict) or actual!=expected
                    or any(type(actual.get(key)) not in (int,float) for key in numeric_fields)
                    or not isinstance(actual.get('value'),list)
                    or any(type(value) not in (int,float) for value in actual['value'])):
                raise ValueError('Nested observation signal differs from top-level pose, scale or source timing')
    else:
        expected=dict(status='missing',captured_at_seconds=None,available_at_seconds=None,age_seconds=None,value=None)
        if any(nested[name]!=expected for name in SIGNALS):
            raise ValueError('Abstention nested signals must be entirely missing/null')


def estimate(output,split):
    """Only named image, depth and calibration files cross the estimator boundary."""
    from .piper_rgbd_tracker import estimate_pose
    output=Path(output);folder=output/split
    frozen=_check_freeze(output) if split=='confirmation' else None
    _,manifest=_manifest(output,split)
    if frozen is not None and manifest['renderer_bindings']!=frozen['renderer_bindings']:
        raise ValueError('Confirmation rendering differs from frozen bindings')
    destination=folder/'predictions.json';journal=folder/'estimates'
    if destination.exists() or journal.exists(): raise FileExistsError('Estimation already started; never retry or overwrite an evaluation')
    journal.mkdir();predictions=[];interrupted=False
    _json(journal/'started.json',dict(schema=SCHEMA,manifest_sha256=_digest(folder/'manifest.json'),sources=hashes()))
    for item in manifest['records']:
        before=time.perf_counter();result=None;error=None;status='complete';input_ids={}
        estimate_wall=None;estimate_cpu=None;call_started=None;cpu_started=None
        if item['render_status']!='rendered' or interrupted:
            status='unattempted';error=dict(type='UnavailableFixture',reason='render_failed_or_preceding_interruption')
        else:
            try:
                _verify_files(folder,item,INPUTS)
                data=folder/item['id'];inputs=_load(data/'input.json')
                input_ids={name:item['files'][name] for name in INPUTS}
                rgb=np.load(data/'rgb.npy',allow_pickle=False);depth=np.load(data/'depth.npy',allow_pickle=False)
                _json(journal/(item['id']+'-started.json'),dict(id=item['id'],inputs=input_ids))
                call_started=time.perf_counter();cpu_started=time.process_time()
                result=estimate_pose(rgb,depth,inputs['calibration'],**inputs['timing'])
                estimate_wall=time.perf_counter()-call_started;estimate_cpu=time.process_time()-cpu_started
                _valid_result(result,inputs['timing'])
            except (Exception,KeyboardInterrupt) as exc:
                status='failed';error=_error(exc);result=None
                if call_started is not None and estimate_wall is None:
                    estimate_wall=time.perf_counter()-call_started;estimate_cpu=time.process_time()-cpu_started
                if isinstance(exc,KeyboardInterrupt): interrupted=True
        row=dict(id=item['id'],estimate_status=status,result=result,error=error,
                 wall_seconds=estimate_wall,cpu_seconds=estimate_cpu,pipeline_wall_seconds=time.perf_counter()-before,inputs=input_ids)
        _json(journal/(item['id']+'.json'),row)
        predictions.append(dict(**row,receipt_sha256=_digest(journal/(item['id']+'.json'))))
    _json(destination,dict(schema=SCHEMA,split=split,source_hashes=hashes(),runtime=runtime(),
        manifest_sha256=_digest(folder/'manifest.json'),protocol_sha256=_digest(output/'protocol.json'),
        freeze_sha256=_digest(output/'freeze.json') if frozen is not None else None,predictions=predictions,
        timing_scope=TIMING_SCOPE))
    return dict(scheduled=len(predictions),estimated=sum(r['estimate_status']=='complete' for r in predictions),split=split,
                pipeline_complete=all(r['estimate_status']=='complete' for r in predictions))


def score(output,split):
    """Identity-checked offline truth join; never call the estimator or simulator."""
    output=Path(output);folder=output/split;protocol,manifest=_manifest(output,split)
    scorer_sources=hashes()
    prediction_path=folder/'predictions.json';predictions=_load(prediction_path)
    if (predictions['schema']!=SCHEMA or predictions['split']!=split
            or predictions['manifest_sha256']!=_digest(folder/'manifest.json')
            or predictions['protocol_sha256']!=_digest(output/'protocol.json')):
        raise ValueError('Prediction binding differs from sealed fixtures')
    if split=='confirmation':
        frozen=_check_freeze(output)
        if (predictions['freeze_sha256']!=_digest(output/'freeze.json')
                or predictions['source_hashes']!=frozen['sources'] or predictions['runtime']!=frozen['runtime']
                or manifest['source_hashes']!=frozen['sources'] or manifest['runtime']!=frozen['runtime']
                or manifest['renderer_bindings']!=frozen['renderer_bindings']):
            raise ValueError('Confirmation evidence differs from frozen candidate')
    if [p['id'] for p in predictions['predictions']]!=[r['id'] for r in manifest['records']]:
        raise ValueError('Predictions must cover every scheduled ID exactly once in order')
    rows=[]
    for pred,item in zip(predictions['predictions'],manifest['records']):
        receipt=folder/'estimates'/(pred['id']+'.json')
        if _digest(receipt)!=pred['receipt_sha256'] or _load(receipt)!={k:v for k,v in pred.items() if k!='receipt_sha256'}:
            raise ValueError('Estimate journal changed')
        _verify_files(folder,item,item['files'])
        if pred['estimate_status'] not in ('complete','failed','unattempted'): raise ValueError('Unknown estimate status')
        result=pred['result'];available=False;error=None;yaw_error=None;pos_scale=None;angle_scale=None
        if pred['estimate_status']=='complete':
            if item['render_status']!='rendered' or pred['inputs']!={name:item['files'][name] for name in INPUTS}:
                raise ValueError('Completed estimate lacks exact rendered input identities')
            inputs=_load(folder/item['id']/'input.json');_valid_result(result,inputs['timing'])
            truth=_load(folder/item['id']/'truth.json');available=result['status']=='available'
            if available and truth['present']:
                error=float(np.linalg.norm(np.asarray(result['object_position_estimate_m'])-truth['position_m']))
                q=result['object_quaternion_estimate_wxyz'];yaw=2*math.atan2(q[3],q[0]);delta=yaw-truth['yaw_rad']
                # Native acos(R[2,2]) can report ~1e-8 rad for an upright
                # matrix because of roundoff. Classify only declared upright
                # fixtures, with a one-microradian numerical tolerance.
                upright=item['variant'] not in ('tilt15','tilt30') and abs(truth['tilt_rad'])<=1e-6
                yaw_error=abs((delta+math.pi/4)%(math.pi/2)-math.pi/4) if upright else None
                pos_scale=result['object_position_uncertainty_m'];angle_scale=result['object_orientation_uncertainty_rad']
        elif result is not None: raise ValueError('Failed estimate contains a result')
        rows.append(dict(**item,estimate_status=pred['estimate_status'],estimate_error=pred['error'],
            status=result['status'] if result else 'pipeline_failure',reason=result['reason'] if result else pred['error']['reason'],
            available=available,position_error_m=error,yaw_error_rad=yaw_error,
            position_heuristic_scale_m=pos_scale,orientation_heuristic_scale_rad=angle_scale,
            position_error_within_heuristic_scale=(error<=pos_scale) if error is not None else None,
            yaw_error_within_heuristic_scale=(yaw_error<=angle_scale) if yaw_error is not None else None,
            wall_seconds=pred['wall_seconds'],cpu_seconds=pred['cpu_seconds']))
    summaries=[];gates=[]
    def percentile(group,key,p):
        values=[r[key] for r in group if r[key] is not None]
        return float(np.percentile(values,p)) if values else None
    for width,_ in protocol['resolutions']:
        for variant in protocol['variants']:
            group=[r for r in rows if r['width']==width and r['variant']==variant]
            summaries.append(dict(width=width,variant=variant,scheduled=len(group),available=sum(r['available'] for r in group),
                pipeline_failures=sum(r['estimate_status']!='complete' for r in group),
                position_evaluated=sum(r['position_error_m'] is not None for r in group),
                yaw_evaluated=sum(r['yaw_error_rad'] is not None for r in group),
                position_p50_m=percentile(group,'position_error_m',50),position_p95_m=percentile(group,'position_error_m',95),
                yaw_p95_rad=percentile(group,'yaw_error_rad',95),wall_p50_s=percentile(group,'wall_seconds',50),
                wall_p95_s=percentile(group,'wall_seconds',95),cpu_p95_s=percentile(group,'cpu_seconds',95),
                heuristic_comparisons=sum(r['position_error_m'] is not None for r in group),
                yaw_heuristic_comparisons=sum(r['yaw_error_rad'] is not None for r in group),
                position_error_within_heuristic_scale=sum(r['position_error_within_heuristic_scale'] is True for r in group),
                yaw_error_within_heuristic_scale=sum(r['yaw_error_within_heuristic_scale'] is True for r in group)))
        clear=[r for r in rows if r['width']==width and r['variant']=='clear']
        negative=[r for r in rows if r['width']==width and r['variant'] in protocol['expected_abstain']]
        limits=protocol['provisional_gate'];coverage=sum(r['available'] for r in clear)/len(clear)
        position=percentile(clear,'position_error_m',95);yaw=percentile(clear,'yaw_error_rad',95)
        false_available=sum(r['available'] for r in negative)
        complete=not manifest['session_errors'] and all(r['estimate_status']=='complete' for r in rows if r['width']==width)
        checks=dict(all_scheduled_fixtures_complete=complete,
            clear_pose_metrics_complete=all(r['position_error_m'] is not None and r['yaw_error_rad'] is not None
                                            for r in clear if r['available']),
            clear_coverage=coverage>=limits['clear_coverage_min'],
            clear_position=position is not None and position<=limits['clear_position_p95_m_max'],
            clear_yaw=yaw is not None and yaw<=limits['clear_yaw_p95_rad_max'],
            no_negative_false_available=false_available<=limits['negative_false_available_max'])
        gates.append(dict(width=width,passed=all(checks.values()),checks=checks,clear_scheduled=len(clear),
                          clear_coverage=coverage,negative_scheduled=len(negative),negative_false_available=false_available))
    report=dict(schema=SCHEMA,split=split,scorer_source_hashes=scorer_sources,
        predictions_sha256=_digest(prediction_path),manifest_sha256=_digest(folder/'manifest.json'),
        rows=rows,summary=summaries,gates=gates,scheduled=len(rows),timing_scope=TIMING_SCOPE,
        uncertainty_scope='Descriptive error <= heuristic scale counts only; not calibrated probability or guaranteed bounds. Yaw comparison applies only to upright present objects.',
        scope='Static simulated known-cube perception; no policy, action-learning or physical-camera/robot qualification.',
        physical_robot_ready=False,depth_accuracy_verified=False)
    if hashes()!=scorer_sources: raise ValueError('Scorer sources changed during offline scoring')
    _json(folder/'report.json',report)
    return report['summary']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','render','freeze','estimate','score'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--assets',type=Path)
    parser.add_argument('--split',choices=['development','confirmation'],default='development');args=parser.parse_args(argv)
    if args.command=='prepare':result=prepare(args.output)
    elif args.command=='render':result=render(args.output,args.assets,args.split)
    elif args.command=='freeze':result=freeze(args.output)
    elif args.command=='estimate':result=estimate(args.output,args.split)
    else:result=score(args.output,args.split)
    print(json.dumps(result,allow_nan=False))
    # Missing/refused estimates are legitimate benchmark outcomes. A pipeline
    # exception or unattempted fixture is different: retain its receipt and
    # denominator, but make shell automation stop instead of reporting success.
    return 1 if args.command in ('render','estimate') and not result['pipeline_complete'] else 0


if __name__=='__main__':raise SystemExit(main())
