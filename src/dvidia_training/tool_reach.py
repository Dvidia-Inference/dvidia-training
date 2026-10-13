"""Authored prerequisite experiment: pull an out-of-workspace shoe with a rake.

The lightweight rake is a declared pre-equipped rigid fixture, not a discovered
or grasped tool. Shoes are free bodies throughout. Only native joint actuators
move the arm; no shoe weld, teleport, applied wrench, or hidden carrying force
is used. All observations are privileged simulator truth. The 0.54 m limit is
this controller's operating envelope, not a proof of global IK infeasibility.
"""
from __future__ import annotations
from copy import deepcopy
import math
import time
from . import workstation as ws, reference_sim as sim

TASK = 'out-of-reach-shoe'
LIFT_HEIGHT = .28  # Uniform development-selected waypoint; within workstation bounds.
# Frozen before controller tuning. Heldout configurations must not be used to
# choose thresholds or waypoints; they vary several factors simultaneously.
_CASES = {
 'development': [
  dict(id='reach-dev-standard', shoe_x=.72, shoe_y=-.10, mass=.30, friction=.60, tool_length=.35, lateral_error=.00),
  dict(id='reach-dev-offset', shoe_x=.72, shoe_y=-.12, mass=.30, friction=.70, tool_length=.35, lateral_error=.065),
  dict(id='reach-dev-heavy', shoe_x=.73, shoe_y=-.08, mass=.50, friction=.95, tool_length=.36, lateral_error=.00),
  dict(id='reach-dev-miss-retry', shoe_x=.72, shoe_y=-.10, mass=.30, friction=.60, tool_length=.35, lateral_error=.12, purpose='development recovery branch coverage added after first pilot'),
  dict(id='reach-dev-short', shoe_x=.73, shoe_y=-.10, mass=.30, friction=.60, tool_length=.16, lateral_error=.00),
 ],
 'heldout': [
  dict(id='reach-heldout-light', shoe_x=.705, shoe_y=-.09, mass=.22, friction=.45, tool_length=.34, lateral_error=.025),
  dict(id='reach-heldout-offset', shoe_x=.725, shoe_y=-.115, mass=.38, friction=.80, tool_length=.36, lateral_error=-.075),
  dict(id='reach-heldout-heavy', shoe_x=.735, shoe_y=-.075, mass=.55, friction=1.05, tool_length=.37, lateral_error=.035),
  dict(id='reach-heldout-short', shoe_x=.72, shoe_y=-.105, mass=.27, friction=.55, tool_length=.18, lateral_error=.015),
 ]}


def cases(split='development'):
    if split not in _CASES: raise ValueError('split must be development or heldout')
    return [dict(row, split=split) for row in deepcopy(_CASES[split])]


def _validate_case(case):
    limits = {'shoe_x':(.69,.75), 'shoe_y':(-.15,-.05), 'mass':(.1,1.),
              'friction':(.1,1.5), 'tool_length':(.1,.4), 'lateral_error':(-.12,.12)}
    if not isinstance(case,dict) or not isinstance(case.get('id'),str): raise ValueError('Expected a named case')
    for key,(low,high) in limits.items():
        value=case.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Invalid case parameter: '+key)


def build_model(assets, case, policy='recovery'):
    """Build one native model. Baseline is explicitly the no-tool ablation."""
    if policy not in ('baseline', 'recovery'): raise ValueError('Unknown policy')
    _validate_case(case)
    mj, np = sim._native_imports()
    spec=ws._spec(assets)
    shoe=spec.body(ws.SHOE)
    shoe.pos=[case['shoe_x'],case['shoe_y'],.012]
    for geom in shoe.geoms:
        geom.mass *= case['mass']/.30
        geom.friction=[case['friction'],.005,.0001]
    if policy=='recovery':
        # At the nominal TCP orientation this frame is aligned with the world.
        # Its fixed mount is explicit. There is no tool/shoe equality constraint.
        quat=list(ws.ORIENTATION);quat[1:]=[-q for q in quat[1:]]
        tool=spec.body('link6').add_body(name='reach_rake',pos=[0,0,.12003],quat=quat)
        length=case['tool_length']
        tool.add_geom(name='reach_rake_mount',type=mj.mjtGeom.mjGEOM_BOX,
            pos=[-.03,0,.06],size=[.03,.009,.009],mass=.01,rgba=[.18,.27,.27,1],contype=1,conaffinity=1)
        tool.add_geom(name='reach_rake_shaft',type=mj.mjtGeom.mjGEOM_BOX,
            pos=[length/2,0,.060],size=[length/2,.009,.009],mass=.08,
            rgba=[.72,.46,.20,1],contype=1,conaffinity=1,condim=4,friction=[.7,.005,.0001])
        tool.add_geom(name='reach_rake_paddle',type=mj.mjtGeom.mjGEOM_BOX,
            pos=[length,0,.005],size=[.010,.035,.055],mass=.035,
            rgba=[.18,.27,.27,1],contype=1,conaffinity=1,condim=4,friction=[.7,.005,.0001])
    return mj.MjModel.from_xml_string(spec.to_xml(),assets=spec.assets)


def run_case(assets, case, policy='baseline', record=True):
    """Run a bounded authored policy with 2 ms evidence and 20 ms commands."""
    if policy not in ('baseline','recovery'): raise ValueError('Unknown policy')
    mj,np=sim._native_imports()
    wall=time.perf_counter()
    model=build_model(assets,case,policy);data=ws.initialize(model)
    tcp,shoe=ws._ids(model)
    tool=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'reach_rake')
    fingers=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name) for name in ('link7','link8')]
    events=[];frames=[];steps=0;phase='settle';target=np.array(ws.HOME_TARGET);jaw=.035
    desired=data.ctrl.copy();peak=0.;impulse=0.;contact_steps=0;peak_tick=0.
    policy_ms=[];all_peak=0.;finger_forces=[0.,0.];force_peak_by_phase={};last_force=0.;native_trace=[];world_force=0.
    max_delta=0.;max_tracking=0.;lift_peak=0.;recovery_attempts=0;help_requested=False
    initial=np.array([case['shoe_x'],case['shoe_y'],0.])
    retrieved=False;grasp_attempted=False;lift_retained=False;valid=True
    def event(kind,**values):
        message=str(values.get('reason',values.get('name',kind))).replace('_',' ')
        events.append({'t':round(float(data.time),6),'type':kind,'message':message,**values})
    def tick():
        nonlocal steps,desired,peak,impulse,contact_steps,peak_tick,max_delta,max_tracking,lift_peak,valid,all_peak,finger_forces,last_force,world_force
        if steps%10==0:
            started=time.perf_counter();desired=ws.controller_action(model,data,target.tolist(),jaw)
            policy_ms.append((time.perf_counter()-started)*1000)
        old=data.ctrl.copy()
        data.ctrl[:6]+=np.clip(desired[:6]-data.ctrl[:6],-1.5*.002,1.5*.002)
        data.ctrl[6]+=np.clip(desired[6]-data.ctrl[6],-.05*.002,.05*.002)
        max_delta=max(max_delta,float(np.max(np.abs(data.ctrl[:6]-old[:6]))))
        mj.mj_step(model,data);mj.mj_forward(model,data);steps+=1
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):
            valid=False;raise RuntimeError('Invalid native state')
        force=0.;all_force=0.;finger_forces=[0.,0.];world_force=0.
        for i in range(data.ncon):
            c=data.contact[i]; b1=int(model.geom_bodyid[c.geom1]);b2=int(model.geom_bodyid[c.geom2])
            if shoe in (b1,b2):
                other=b2 if b1==shoe else b1
                wrench=np.zeros(6);mj.mj_contactForce(model,data,i,wrench);normal=max(0.,float(wrench[0]))
                if other!=0:all_force+=normal
                else:world_force+=normal
                if other==tool and tool>=0:force+=normal
                if other in fingers:finger_forces[fingers.index(other)]+=normal
        last_force=force;all_peak=max(all_peak,all_force)
        force_peak_by_phase[phase]=max(force_peak_by_phase.get(phase,0.),all_force)
        native_trace.append([round(float(data.time),6),force,all_force,*finger_forces,*data.xpos[shoe].tolist(),world_force])
        if force>0:
            contact_steps+=1;impulse+=force*.002
            if force>peak:peak=force;peak_tick=float(data.time)
        max_tracking=max(max_tracking,float(np.linalg.norm(data.site_xpos[tcp]-target)))
        if phase in ('lift','direct_lift'):lift_peak=max(lift_peak,float(data.xpos[shoe,2]))
        if record and steps%10==0:
            frames.append({'t':round(float(data.time),6),'body_poses':np.column_stack((data.xpos,data.xquat)).tolist(),
               'phase':phase,'target':target.tolist(),'shoe_position':data.xpos[shoe].tolist(),
               'qpos_rad_m':data.qpos.tolist(),'qvel_rad_m_s':data.qvel.tolist(),
               'issued_ctrl':data.ctrl.tolist(),'requested_joint_target':desired.tolist(),
               'tcp_position':data.site_xpos[tcp].tolist(),'tool_shoe_force_n':force,
               'finger_normal_forces_n':list(finger_forces),'world_shoe_force_n':world_force})
    def go(name,point,seconds=1.5,opening=.035):
        nonlocal phase,target,jaw
        phase=name;target=np.array(point,dtype=float);jaw=opening;event('phase',name=name,target=target.tolist())
        for _ in range(round(seconds/.002)): tick()
    def travel(name,end,seconds=2.,opening=.035):
        nonlocal phase,target,jaw
        phase=name;jaw=opening;start=np.clip(data.site_xpos[tcp], [b[0] for b in ws.BOUNDS], [b[1] for b in ws.BOUNDS]);end=np.asarray(end)
        event('phase',name=name,target=end.tolist())
        for i in range(round(seconds/.002)):
            f=(i+1)/round(seconds/.002);f=f*f*(3-2*f);target=start*(1-f)+end*f;tick()
    event('assumptions',pre_equipped_rigid_tool=policy=='recovery',privileged_observations=True,
          shoe_attachment=False,controller_reach_limit_x_m=.54,tool_acquisition_tested=False)
    go('settle',ws.HOME_TARGET,.4)
    if policy=='baseline':
        grasp_attempted=True
        event('direct_reach_attempt',desired_x_m=float(data.xpos[shoe,0]),clipped_x_m=.54)
        travel('direct_reach',[.54,case['shoe_y'],.075],2.5)
        go('direct_close',[.54,case['shoe_y'],.075],1.,0.)
        go('direct_lift',[.50,case['shoe_y'],.20],1.5,0.)
        event('direct_reach_failed',shoe_x_m=float(data.xpos[shoe,0]))
    else:
        length=case['tool_length']
        # Need room for the paddle behind the entire shoe, not merely its centre.
        requested=float(data.xpos[shoe,0])+.14+.025-length
        if requested>.54:
            help_requested=True;event('help_requested',reason='tool_cannot_reach_far_edge',required_tcp_x_m=requested)
            go('hold_for_help',ws.HOME_TARGET,.6)
        else:
            for attempt in range(2):
                if attempt:recovery_attempts+=1
                y=float(data.xpos[shoe,1])+(case['lateral_error'] if attempt==0 else 0.)
                far=float(data.xpos[shoe,0])+.14+.025-length
                far=float(np.clip(far,.18,.54))
                travel('approach_tool',[far,y,.25],1.8)
                travel('lower_tool',[far,y,.075],1.0)
                travel('pull_shoe',[.20,y,.075],3.0)
                go('verify_pull',[.20,y,.075],.4)
                displacement=initial[0]-float(data.xpos[shoe,0])
                retrieved=bool(float(data.xpos[shoe,0])<=.53 and abs(float(data.xpos[shoe,1]))<=.21 and abs(float(data.xpos[shoe,2]))<.035)
                event('retrieval_check',attempt=attempt+1,retrieved=retrieved,shoe_x_m=float(data.xpos[shoe,0]),displacement_m=displacement)
                if retrieved: break
                event('recovery',reason='insufficient_retrieval',action='recenter_and_retry' if attempt==0 else 'stop_and_ask')
                go('release_tool',[.20,y,.22],1.)
            if retrieved:
                # With the rake lifted clear, try the actual jaw around the upper.
                # This fixture reduces dexterity. Retained lift is measured, never
                # inferred from retrieval or closure commands.
                go('clear_rake',[.24,case['shoe_y'],.25],1.2)
                sx,sy=float(data.xpos[shoe,0]),float(data.xpos[shoe,1])
                grasp_attempted=True
                travel('grasp_approach',[float(np.clip(sx-.045,.18,.54)),sy,.060],1.8)
                go('close_jaws',[float(np.clip(sx-.045,.18,.54)),sy,.060],1.,0.)
                travel('lift',[float(np.clip(sx-.045,.18,.54)),sy,LIFT_HEIGHT],1.5,0.)
                phase='lift';target=np.array([float(np.clip(sx-.045,.18,.54)),sy,LIFT_HEIGHT]);jaw=0.
                retained=[]
                for _ in range(250):tick();retained.append(float(data.xpos[shoe,2])>.035 and min(finger_forces)>.01 and last_force<.01 and world_force<.01)
                lift_retained=all(retained)
                event('lift_check',retained=lift_retained,shoe_bottom_z_m=float(data.xpos[shoe,2]))
                if not lift_retained:
                    help_requested=True;event('help_requested',reason='retrieved_but_jaw_lift_not_retained')
            else:
                help_requested=True;event('help_requested',reason='bounded_recovery_exhausted')
    final=data.xpos[shoe].copy()
    shoe_dof=int(model.jnt_dofadr[int(model.body_jntadr[shoe])])
    final_speed=float(np.linalg.norm(data.qvel[shoe_dof:shoe_dof+3]))
    success=retrieved and lift_retained
    metrics={'retrieval_success':retrieved,'grasp_attempted':grasp_attempted,'retained_lift_success':lift_retained,
      'final_shoe_speed_m_s':final_speed,
      'shoe_displacement_x_m':float(initial[0]-final[0]),'final_shoe_position_m':final.tolist(),
      'peak_lift_m':lift_peak,'recovery_attempts':recovery_attempts,'help_requested':help_requested,
      'peak_tool_shoe_force_n':peak,'peak_tool_shoe_force_time_s':peak_tick,'tool_shoe_impulse_ns':impulse,
      'tool_shoe_contact_ticks':contact_steps,'physics_timestep_s':.002,'policy_interval_s':.02,
      'max_joint_command_delta_rad':max_delta,'max_tracking_error_m':max_tracking,
      'simulation_seconds':float(data.time),'wall_seconds':time.perf_counter()-wall,
      'solver_warnings':sum(int(w.number) for w in data.warning),'native_steps':steps,
      'shoe_external_force_used':bool(np.any(data.xfrc_applied) or np.any(data.qfrc_applied)),
      'shoe_equality_constraints':int(sum(shoe in (int(model.eq_obj1id[i]),int(model.eq_obj2id[i])) for i in range(model.neq) if int(model.eq_type[i]) in (0,1))),
      'recovery_count':recovery_attempts,'intervention_count':int(help_requested),
      'peak_contact_force_n':all_peak,'contact_force_scope':'sum of normal shoe-to-robot/tool forces, excluding table',
      'force_peak_by_phase_n':force_peak_by_phase,
      'policy_compute_p50_ms':float(np.percentile(policy_ms,50)),
      'policy_compute_p95_ms':float(np.percentile(policy_ms,95)),
      'policy_compute_max_ms':max(policy_ms),'wall_includes_model_compile':True,
      'recorded_state_scope':'privileged native state; qpos has 6 arm radians, 2 jaw metres, shoe XYZ metres and quaternion wxyz; qvel has 6 arm rad/s, 2 jaw m/s, shoe linear m/s and angular rad/s; controls are 6 joint radians plus jaw metres; requested targets update every20ms and issued controls are rate limited every2ms',
      'retention_criterion':'shoe bottom >35 mm with both fingers contacting, tool and world force each <0.01 N for final 500 ms',
      'no_tool_baseline':policy=='baseline','tool_pre_equipped':policy=='recovery'}
    metrics['outcome']=('completed' if success else 'failed_direct_reach' if policy=='baseline'
        else 'retrieved_lift_failed' if retrieved
        else 'deferred_insufficient_reach' if any(e.get('reason')=='tool_cannot_reach_far_edge' for e in events)
        else 'recovery_exhausted')
    metrics['real_time_factor']=metrics['simulation_seconds']/metrics['wall_seconds']
    return {'task':TASK,'case':deepcopy(case),'policy':policy,'success':success,'valid':valid,
            'metrics':metrics,'events':events,'frames':frames,'model':model,
            'native_trace':native_trace,'native_trace_columns':['t_s','tool_shoe_force_n','shoe_robot_force_n',
                'left_finger_force_n','right_finger_force_n','shoe_x_m','shoe_y_m','shoe_z_m','world_shoe_force_n'],
            'control_compute_ms':policy_ms}
