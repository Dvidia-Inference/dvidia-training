"""Direct graspability controls with unchanged pinned PiPER mechanics.

Three declared rigid object geometries, explicit mass loads, and a mounted-rake
versus tool-free morphology ablation. This does not detach a tool, discover a
grasp, learn from footage, or calibrate hardware. Every object stays a free body.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

from . import workstation as ws, reference_sim as sim

TASK = 'in-reach-graspability'
SHAPES = ('easy-block', 'simplified-shoe', 'current-shoe')
TOOLS = ('mounted', 'absent')
TRACE_COLUMNS = ['t_s', 'tool_object_force_n', 'object_robot_force_n',
    'left_finger_force_n', 'right_finger_force_n', 'object_x_m', 'object_y_m',
    'object_z_m', 'world_object_force_n', 'tcp_x_m', 'tcp_y_m', 'tcp_z_m',
    'object_lowest_z_m', 'gripper_actuator_force_n', 'jaw_opening_m']
# Dimensions are intentionally different controls, not replicas of one shoe.
SHAPE_DIMENSIONS = {'easy-block': [.04, .04, .04],
    'simplified-shoe': [.14, .05, .06], 'current-shoe': [.28, .10, .087]}
SHAPE_OFFSETS = {'easy-block': 0., 'simplified-shoe': 0., 'current-shoe': -.085}
TIMING = {'settle': .4, 'approach': 2., 'close': 1.2, 'lift': 2., 'retention_hold': 1.}


def cases(split='development'):
    """Known development controls only; fresh evaluation is owned independently."""
    if split != 'development':
        raise ValueError('Only development cases are owned by the simulator')
    return [dict(id=f'dev-{shape}-{int(mass*1000)}g', split=split,
        object_shape=shape, mass=mass, friction=.6, object_x=.40, object_y=-.10,
        yaw_rad=0., load_band='light-control' if mass==.03 else 'shoe-load')
        for shape in SHAPES for mass in (.03, .30)]


def validate_case(case, tool='mounted'):
    if tool not in TOOLS:
        raise ValueError('Tool condition must be mounted or absent')
    if not isinstance(case, dict) or not isinstance(case.get('id'), str) or not case['id']:
        raise ValueError('Expected a named graspability case')
    if case.get('object_shape') not in SHAPES:
        raise ValueError('Unknown rigid object shape')
    limits = {'mass': (.01, .6), 'friction': (.1, 1.5),
              'object_x': (.36, .46), 'object_y': (-.16, -.04), 'yaw_rad': (-.15, .15)}
    for key, (low, high) in limits.items():
        value=case.get(key, 0. if key=='yaw_rad' else None)
        if type(value) not in (int, float) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Invalid graspability case parameter: '+key)


def _add_rake(spec, mj):
    # Identical geometry/material/masses to tool_reach.build_model's fixed rake.
    quat=list(ws.ORIENTATION);quat[1:]=[-q for q in quat[1:]]
    tool=spec.body('link6').add_body(name='reach_rake', pos=[0,0,.12003], quat=quat)
    tool.add_geom(name='reach_rake_mount',type=mj.mjtGeom.mjGEOM_BOX,
        pos=[-.03,0,.06],size=[.03,.009,.009],mass=.01,
        rgba=[.18,.27,.27,1],contype=1,conaffinity=1)
    tool.add_geom(name='reach_rake_shaft',type=mj.mjtGeom.mjGEOM_BOX,
        pos=[.175,0,.060],size=[.175,.009,.009],mass=.08,
        rgba=[.72,.46,.20,1],contype=1,conaffinity=1,condim=4,friction=[.7,.005,.0001])
    tool.add_geom(name='reach_rake_paddle',type=mj.mjtGeom.mjGEOM_BOX,
        pos=[.35,0,.005],size=[.010,.035,.055],mass=.035,
        rgba=[.18,.27,.27,1],contype=1,conaffinity=1,condim=4,friction=[.7,.005,.0001])


def build_model(assets, case, tool='mounted'):
    """Compile a declared condition; no geometry or physics edits during flight."""
    validate_case(case, tool)
    mj,_=sim._native_imports()
    spec=ws._spec(assets)
    body=spec.body(ws.SHOE)
    body.pos=[case['object_x'],case['object_y'],.012]
    yaw=case.get('yaw_rad',0.)
    body.quat=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
    if case['object_shape']!='current-shoe':
        for geom in list(body.geoms):
            spec.delete(geom)
        size=[v/2 for v in SHAPE_DIMENSIONS[case['object_shape']]]
        body.add_geom(name=ws.SHOE+'_proxy', type=mj.mjtGeom.mjGEOM_BOX,
            pos=[0,0,size[2]],size=size,mass=case['mass'],
            rgba=[.22,.40,.50,1],contype=1,conaffinity=1,condim=4,priority=1,
            friction=[case['friction'],.005,.0001],solref=[.02,1],solimp=[.9,.95,.001,.5,2])
    else:
        for geom in body.geoms:
            geom.mass*=case['mass']/.30
            geom.friction=[case['friction'],.005,.0001]
    if tool=='mounted':
        _add_rake(spec,mj)
    return mj.MjModel.from_xml_string(spec.to_xml(),assets=spec.assets)


def object_lowest_z(model, data, object_id):
    """Exact lowest world-z of the collidable box/ellipsoid primitives."""
    mj,np=sim._native_imports()
    values=[]
    start=int(model.body_geomadr[object_id]);count=int(model.body_geomnum[object_id])
    for geom in range(start,start+count):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        z_axis=data.geom_xmat[geom].reshape(3,3)[2]
        size=model.geom_size[geom]
        if model.geom_type[geom]==mj.mjtGeom.mjGEOM_BOX:
            radius=float(np.abs(z_axis)@size)
        elif model.geom_type[geom]==mj.mjtGeom.mjGEOM_ELLIPSOID:
            radius=float(np.linalg.norm(z_axis*size))
        else:
            raise ValueError('Unsupported object collision geometry for clearance')
        values.append(float(data.geom_xpos[geom,2])-radius)
    if not values:
        raise ValueError('Object has no collision geometry')
    return min(values)


def run_case(assets, case, tool='mounted', record=True, *, close_gripper=True):
    """Run one fixed direct-lift trajectory, logging all 2 ms native samples."""
    validate_case(case,tool)
    if type(close_gripper) is not bool:
        raise ValueError('close_gripper must be boolean')
    mj,np=sim._native_imports();wall=time.perf_counter()
    model=build_model(assets,case,tool);data=ws.initialize(model)
    tcp,obj=ws._ids(model)
    rake=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'reach_rake')
    fingers=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,n) for n in ('link7','link8')]
    phase='settle';target=np.asarray(ws.HOME_TARGET);jaw=.035;desired=data.ctrl.copy()
    steps=0;trace=[];frames=[];events=[];compute=[];max_delta=0.;closed=None
    bounds_lo=[v[0] for v in ws.BOUNDS];bounds_hi=[v[1] for v in ws.BOUNDS]
    def event(kind,**values):
        events.append(dict(t=round(float(data.time),6),type=kind,
            message=str(values.get('reason',values.get('name',kind))).replace('_',' '),**values))
    def tick():
        nonlocal steps,desired,max_delta
        if steps%10==0:
            start=time.perf_counter();desired=ws.controller_action(model,data,target.tolist(),jaw)
            compute.append((time.perf_counter()-start)*1000)
        old=data.ctrl.copy()
        data.ctrl[:6]+=np.clip(desired[:6]-data.ctrl[:6],-.003,.003)
        data.ctrl[6]+=float(np.clip(desired[6]-data.ctrl[6],-.0001,.0001))
        max_delta=max(max_delta,float(np.max(np.abs(data.ctrl[:6]-old[:6]))))
        mj.mj_step(model,data);mj.mj_forward(model,data);steps+=1
        if not all(np.isfinite(v).all() for v in (data.qpos,data.qvel,data.qacc)) or any(w.number for w in data.warning):
            raise RuntimeError('Invalid native graspability state')
        forces=[0.,0.,0.,0.,0.]  # rake, all robot, left, right, world
        for i in range(data.ncon):
            contact=data.contact[i];a=int(model.geom_bodyid[contact.geom1]);b=int(model.geom_bodyid[contact.geom2])
            if obj not in (a,b):continue
            other=b if a==obj else a;wrench=np.zeros(6)
            mj.mj_contactForce(model,data,i,wrench);normal=max(0.,float(wrench[0]))
            forces[4 if other==0 else 1]+=normal
            if other==rake and rake>=0:forces[0]+=normal
            if other in fingers:forces[2+fingers.index(other)]+=normal
        t=round(float(data.time),6);position=data.xpos[obj].tolist();actual_tcp=data.site_xpos[tcp].tolist()
        lowest=object_lowest_z(model,data,obj)
        trace.append([t,*forces[:4],*position,forces[4],*actual_tcp,lowest,
            float(data.actuator_force[6]),float(data.qpos[6]-data.qpos[7])])
        if record and steps%10==0:
            frames.append(dict(t=t,body_poses=np.column_stack((data.xpos,data.xquat)).tolist(),
                phase=phase,target=target.tolist(),object_position=position,shoe_position=position,
                tcp_position=actual_tcp,qpos_rad_m=data.qpos.tolist(),qvel_rad_m_s=data.qvel.tolist(),
                issued_ctrl=data.ctrl.tolist(),requested_joint_target=desired.tolist(),
                finger_normal_forces_n=forces[2:4],tool_object_force_n=forces[0],
                world_object_force_n=forces[4],other_robot_force_n=max(0.,forces[1]-sum(forces[2:4])-forces[0]),
                object_lowest_z_m=lowest,gripper_actuator_force_n=float(data.actuator_force[6]),
                jaw_opening_m=float(data.qpos[6]-data.qpos[7]),jaw_qpos_m=data.qpos[6:8].tolist()))
    def motion(name,end,seconds,opening=.035):
        nonlocal phase,target,jaw
        phase=name;jaw=opening;start=np.clip(data.site_xpos[tcp],bounds_lo,bounds_hi)
        end=np.asarray(end);n=round(seconds/.002)
        event('phase',name=name,target=end.tolist())
        for i in range(n):
            u=(i+1)/n;u=u*u*(3-2*u)
            target=np.clip(start*(1-u)+end*u,bounds_lo,bounds_hi);tick()
    event('assumptions',tool_condition=tool,tool_changes_during_episode=False,
        privileged_observations=True,object_attachment=False,gripper_gains_unchanged=True,
        direct_pickup=True,close_gripper=close_gripper,shape_dimensions_m=SHAPE_DIMENSIONS[case['object_shape']])
    motion('settle',ws.HOME_TARGET,TIMING['settle'])
    tail=np.asarray(trace[-50:])
    settled=bool(len(tail)==50 and np.all(tail[:,8]>.01) and np.all(tail[:,2]<.01)
                 and np.all(np.ptp(tail[:,5:8],axis=0)<=.001))
    reference=float(data.xpos[obj,2])
    if settled:
        event('settled_reference',object_z_m=reference,object_lowest_z_m=trace[-1][12])
        aim=[float(data.xpos[obj,0])+SHAPE_OFFSETS[case['object_shape']],float(data.xpos[obj,1]),.055]
        event('grasp_target',target=aim,shape_specific_x_offset_m=SHAPE_OFFSETS[case['object_shape']],
            scope='authored geometry-specific pose; identical across tool conditions')
        motion('approach',aim,TIMING['approach'])
        closure_target=0. if close_gripper else .035
        motion('close',aim,TIMING['close'],closure_target)
        closed=trace[-1]
        event('closure_check',finger_normal_forces_n=closed[3:5],world_object_force_n=closed[8],
            gripper_actuator_force_n=closed[13],jaw_opening_m=closed[14])
        motion('lift',[aim[0],aim[1],.28],TIMING['lift'],closure_target)
        motion('retention_hold',[aim[0],aim[1],.28],TIMING['retention_hold'],closure_target)
    tail=trace[-250:]
    success=bool(settled and len(tail)==250 and all(r[7]-reference>=.035 and min(r[3:5])>.01
        and r[1]<.01 and r[8]<.01 and r[2]-sum(r[3:5])-r[1]<.01 for r in tail))
    event('lift_check',retained=success,settled_reference_available=settled)
    duration=float(data.time);native_wall=time.perf_counter()-wall
    metrics=dict(outcome='completed' if success else 'lift_not_retained' if settled else 'settling_failed',
        simulation_seconds=duration,wall_seconds=native_wall,real_time_factor=duration/native_wall,
        native_steps=steps,physics_timestep_s=.002,policy_interval_s=.02,
        solver_warnings=sum(int(w.number) for w in data.warning),max_joint_command_delta_rad=max_delta,
        object_external_force_used=bool(np.any(data.xfrc_applied) or np.any(data.qfrc_applied)),
        object_equality_constraints=int(sum(obj in (int(model.eq_obj1id[i]),int(model.eq_obj2id[i])) for i in range(model.neq) if int(model.eq_type[i]) in (0,1))),
        retained_lift_success=success,grasp_attempted=settled,settled_reference_available=settled,
        object_mass_kg=float(model.body_mass[obj]),object_inertia_kg_m2=model.body_inertia[obj].tolist(),object_weight_n=float(model.body_mass[obj]*abs(model.opt.gravity[2])),
        object_dimensions_m=SHAPE_DIMENSIONS[case['object_shape']],tool_condition=tool,
        gripper_kp_n_per_m=float(model.actuator_gainprm[6,0]),
        gripper_kv_n_s_per_m=float(-model.actuator_biasprm[6,2]),
        gripper_actuator_forcerange_n=model.actuator_forcerange[6].tolist(),
        gripper_control_range_m=model.actuator_ctrlrange[6].tolist(),
        nominal_jaw_travel_m=.070,jaw_span_scope='joint7 minus joint8 slidecoordinates; nominal parallelfinger span, not a hardware measured aperture',close_gripper=close_gripper,peak_contact_force_n=max(r[2] for r in trace),
        peak_object_origin_rise_m=max(r[7]-reference for r in trace if r[0]>=TIMING['settle']),
        peak_lowest_surface_z_m=max(r[12] for r in trace if r[0]>=TIMING['settle']),
        final_object_position_m=data.xpos[obj].tolist(),
        final_lowest_surface_z_m=trace[-1][12],
        closure_observation=None if closed is None else dict(finger_normal_forces_n=closed[3:5],
            world_support_n=closed[8],actuator_force_n=closed[13],jaw_opening_m=closed[14]),
        policy_compute_p50_ms=float(np.percentile(compute,50)),policy_compute_p95_ms=float(np.percentile(compute,95)),
        policy_compute_max_ms=max(compute),
        timing_scope='CPU native model initialization, control, stepping, trace collection; excludes browser rendering and export',
        controller_scope='fixed authored shape-specific pickup with pinned150degree orientation; same trajectory and drive settings across tool conditions',
        comparison_scope='direct pickup morphology and load controls; tool absent from episode start, no tool dropping or retrieval',
        retention_criterion='final500ms:origin rises at least35mm above settledreference,both finger normalforces>.01N,world/rake/nonfinger robot forceeach<.01N',
        clearance_scope='analytic minimumworldz of collisionprimitives; reported separately, not substitute for allworldsupport check',
        force_scope='MuJoCo native normalforces and actuatorforce; simulation parameters are not hardware calibration')
    return dict(task=TASK,case=deepcopy(case),policy='shared-v03',tool=tool,tool_condition=tool,success=success,valid=True,
        metrics=metrics,events=events,frames=frames,model=model,native_trace=trace,
        native_trace_columns=TRACE_COLUMNS.copy(),control_compute_ms=compute)
