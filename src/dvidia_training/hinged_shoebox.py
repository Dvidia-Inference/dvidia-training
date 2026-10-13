"""Passive shoe-box lid / PiPER access pilot, using native MuJoCo contact.

Policies are authored, use privileged joint/TCP state and do not learn from video.
The task is reaching an inspection waypoint inside an open box, then returning;
it is not shoe retrieval. A closed box requires assistance. Reset angle/velocity
are declared initial conditions. During rollout only PiPER actuators are driven:
no lid motor, mocap body, object attachment, external force or pose rewrite.
"""
from __future__ import annotations

import math
import time
from copy import deepcopy

from . import reference_sim as sim, workstation as ws

TASK = 'hinged-shoebox'
TIMESTEP = .002
CONTROL_DT = .02
DURATION = 6.
START = (.36, -.235, .17)
ACCESS = (.36, -.125, .13)
WITHDRAW = (.36, -.27, .22)
LID = 'shoebox_lid'
HINGE = 'shoebox_hinge'
CONTACT_THRESHOLD_N = .5
SCOPE = ('Authored PiPER shoe-box inspection-waypoint access with privileged native '
         'hinge and robot states. Synthetic rigid shoe/lid, uncalibrated contact. '
         'No learned policy, shoe retrieval, damage model or physical robot qualification.')

# Frozen before development-policy tuning. These are different exact reset states,
# not an IID statistical generalization study. Never tune a policy on heldout runs.
_CASES = {
    'development': [
        {'id':'open-light','lid_angle_deg':112.,'lid_velocity_rad_s':0.,'lid_mass_kg':.12,'hinge_damping':.025},
        {'id':'falling-light','lid_angle_deg':72.,'lid_velocity_rad_s':-.10,'lid_mass_kg':.12,'hinge_damping':.025},
        {'id':'falling-heavy','lid_angle_deg':65.,'lid_velocity_rad_s':-.30,'lid_mass_kg':.24,'hinge_damping':.06},
        {'id':'transient-closing','lid_angle_deg':108.,'lid_velocity_rad_s':-.40,'lid_mass_kg':.12,'hinge_damping':.025},
    ],
    'heldout': [
        {'id':'open-medium','lid_angle_deg':105.,'lid_velocity_rad_s':0.,'lid_mass_kg':.18,'hinge_damping':.04},
        {'id':'falling-medium','lid_angle_deg':78.,'lid_velocity_rad_s':-.15,'lid_mass_kg':.18,'hinge_damping':.04},
        {'id':'falling-low','lid_angle_deg':58.,'lid_velocity_rad_s':-.25,'lid_mass_kg':.10,'hinge_damping':.02},
        {'id':'transient-damped','lid_angle_deg':110.,'lid_velocity_rad_s':-.35,'lid_mass_kg':.20,'hinge_damping':.06},
    ],
}


def cases(split='development'):
    if split not in _CASES:
        raise ValueError('split must be development or heldout')
    return [dict(deepcopy(case), split=split) for case in _CASES[split]]


def _validate_case(case):
    if not isinstance(case,dict) or case not in cases('development')+cases('heldout'):
        raise ValueError('Expected an exact frozen hinged-shoebox case')
    return deepcopy(case)


def build_model(assets, case):
    """Return a compiled physical scene; nothing has been stepped or settled."""
    case = _validate_case(case)
    mj, _ = sim._native_imports()
    spec = ws._spec(assets)
    # Retain the exact PiPER mechanics/actuators and free authored trainer.
    spec.body(ws.SHOE).pos = [.36,-.11,.018]
    for geom in list(spec.geoms):
        if geom.name.startswith('workstation_rack_'):
            spec.delete(geom)
    world = spec.worldbody
    material = [.57,.43,.29,1]
    for name,pos,size in [
        ('bottom',[.36,-.105,.006],[.175,.135,.006]),
        ('front',[.36,-.238,.057],[.175,.007,.045]),
        ('back',[.36,.028,.057],[.175,.007,.045]),
        ('left',[.188,-.105,.057],[.007,.135,.045]),
        ('right',[.532,-.105,.057],[.007,.135,.045]),
    ]:
        world.add_geom(name='shoebox_'+name,type=mj.mjtGeom.mjGEOM_BOX,
                       pos=pos,size=size,rgba=material,friction=[.6,.005,.0001],
                       solref=[.01,1])
    lid = world.add_body(name=LID,pos=[.36,.035,.108])
    lid.add_joint(name=HINGE,type=mj.mjtJoint.mjJNT_HINGE,axis=[-1,0,0],
                  limited=True,range=[0,math.radians(120)],damping=case['hinge_damping'],
                  armature=0,solref_limit=[.015,1])
    lid.add_geom(name='shoebox_lid_panel',type=mj.mjtGeom.mjGEOM_BOX,
                 pos=[0,-.141,0],size=[.181,.135,.006],
                 mass=case['lid_mass_kg'],rgba=[.68,.53,.37,1],
                 friction=[.6,.005,.0001],solref=[.01,1],condim=4)
    # A small seam makes the hinge's axis legible in the replay, but contributes
    # no collision or inertial mass. The physical hinge is the revolute joint.
    world.add_geom(name='shoebox_hinge_visual',type=mj.mjtGeom.mjGEOM_CAPSULE,
                   fromto=[.18,.035,.108,.54,.035,.108],size=[.003,0,0],
                   contype=0,conaffinity=0,rgba=[.23,.24,.24,1])
    model = spec.compile()
    if (model.nq,model.nv,model.nu,model.neq,model.nmocap)!=(16,15,7,1,0):
        raise ValueError('Unexpected PiPER / free shoe / passive hinge dimensions')
    return model


def initialize(model, case):
    """Declared reset-only pose solve; no object mutations occur during rollout."""
    mj,np = sim._native_imports()
    case = _validate_case(case)
    data = ws.initialize(model)
    joint = model.joint(HINGE)
    data.qpos[joint.qposadr[0]] = math.radians(case['lid_angle_deg'])
    for _ in range(240):
        mj.mj_forward(model,data)
        data.qpos[:6] = ws.controller_action(model,data,START,.025,validate=False)[:6]
    data.qvel[:] = 0
    data.qvel[joint.dofadr[0]] = case['lid_velocity_rad_s']
    data.ctrl[:6] = data.qpos[:6]
    data.ctrl[6] = .025
    mj.mj_forward(model,data)
    if not np.isfinite(data.qpos).all():
        raise RuntimeError('Invalid reset')
    return data


def risk_observation(angle, velocity, tcp):
    """Causal conservative swept-lid guard, deliberately independent of case IDs.

    The aperture gate remains conservative: avoid entry into any descending lid
    below 115 degrees; entering only resumes after 200 ms of a stable/open state.
    This is a hand-authored geometric guard, not an uncertainty-calibrated model.
    """
    return bool(velocity < -.03 and angle < math.radians(115)
                and .14 < float(tcp[0]) < .58
                and -.28 < float(tcp[1]) < .06 and float(tcp[2]) < .35)


def _contact(model,data,lid_id,robot_ids):
    mj,np = sim._native_imports()
    peak = 0.; count = 0; other_peak = 0.
    for index in range(data.ncon):
        contact = data.contact[index]
        b1,b2 = int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])
        if (b1==lid_id and b2 in robot_ids) or (b2==lid_id and b1 in robot_ids):
            force=np.zeros(6);mj.mj_contactForce(model,data,index,force)
            peak=max(peak,float(abs(force[0])));count+=1
        elif ((b1 in robot_ids-{1}) != (b2 in robot_ids-{1})) and not (b1 in robot_ids and b2 in robot_ids):
            force=np.zeros(6);mj.mj_contactForce(model,data,index,force)
            other_peak=max(other_peak,float(abs(force[0])))
    return peak,count,other_peak


def run_case(assets,case,policy='baseline',record=True):
    """Run a frozen case with 2 ms native observation and 20 ms control decisions.

    ``native_trace`` retains every step's causal hinge/risk/contact observations.
    Frame poses at 50 Hz are replay data, not physics executed in the viewer.
    ``success`` is task success only; safe_abort/help is never counted as success.
    """
    case = _validate_case(case)
    if policy not in ('baseline','recovery'):
        raise ValueError('policy must be baseline or recovery')
    if type(record) is not bool:
        raise ValueError('record must be boolean')
    mj,np = sim._native_imports()
    model=build_model(assets,case);data=initialize(model,case)
    tcp_id=model.site(ws.TCP).id;lid_id=model.body(LID).id
    robot_ids=set(range(1,model.body(ws.SHOE).id))
    hinge=model.joint(HINGE);qindex=int(hinge.qposadr[0]);vindex=int(hinge.dofadr[0])
    events=[];frames=[];trace=[]
    phase='access';target=ACCESS;desired=data.ctrl.copy()
    intervention_count=0;resumed_count=0;false_interventions=0
    risk_steps=0;contact_steps=0;peak_force=0.;other_peak_force=0.;window_peak=0.;impulse=0.
    first_risk=None;first_intervention=None;first_contact=None
    contact_active=False;risk_active=False;safe_dwell=0.;access_dwell=0.
    reached=False;returned=False;safe_abort=False;valid=True;failure_reason=None
    decision_ms=[];wall_started=time.perf_counter()
    def event(kind,**values):
        messages={'withdrawal_started':'Lid descending. Withdraw from the access path.',
                  'withdrawal_complete':'Arm has reached the withdrawal waypoint.',
                  'access_resumed':'Lid aperture stable. Resume the inspection waypoint.',
                  'help_requested':'Box closed. Stop here and request help reopening it.',
                  'hazardous_contact':'Lid contacted the robot above the synthetic benchmark threshold.',
                  'inspection_waypoint_reached':'Inspection waypoint held for 200 ms; no shoe grasp attempted.',
                  'return_complete':'Arm returned from the box.'}
        events.append({'t':round(float(data.time),6),'type':kind,'message':messages.get(kind,kind.replace('_',' ')),**values})
    for step in range(round(DURATION/TIMESTEP)+1):
        angle=float(data.qpos[qindex]);velocity=float(data.qvel[vindex]);tcp=data.site_xpos[tcp_id].copy()
        risk=risk_observation(angle,velocity,tcp)
        contact_force,contact_count,other_force=_contact(model,data,lid_id,robot_ids)
        other_peak_force=max(other_peak_force,other_force)
        peak_force=max(peak_force,contact_force);window_peak=max(window_peak,contact_force)
        if step:
            impulse+=contact_force*TIMESTEP;contact_steps+=int(contact_force>CONTACT_THRESHOLD_N);risk_steps+=int(risk)
        if risk and first_risk is None:first_risk=float(data.time)
        if risk!=risk_active:
            event('hazard_started' if risk else 'hazard_cleared',lid_angle_rad=angle,lid_velocity_rad_s=velocity)
            risk_active=risk
        harmful=contact_force>CONTACT_THRESHOLD_N
        if harmful and first_contact is None:first_contact=float(data.time)
        if harmful!=contact_active:
            event('hazardous_contact' if harmful else 'lid_robot_contact_cleared',normal_force_n=contact_force)
            contact_active=harmful
        if record:
            trace.append([round(float(data.time),6),angle,velocity,contact_force,int(risk),other_force])
        if step%10==0:
            decision_started=time.perf_counter()
            if policy=='recovery' and risk and phase in ('access','return'):
                phase='withdraw';target=WITHDRAW;intervention_count+=1
                false_interventions+=int(not risk)
                if first_intervention is None:first_intervention=float(data.time)
                event('withdrawal_started',reason='descending_lid_in_access_volume')
            if phase=='withdraw':
                target=WITHDRAW
                if float(np.linalg.norm(tcp-np.asarray(WITHDRAW))) < .018:
                    phase='wait';safe_dwell=0.;event('withdrawal_complete')
            elif phase=='wait':
                target=WITHDRAW
                aperture_open=angle>math.radians(95) and velocity>=-.015
                safe_dwell=safe_dwell+CONTROL_DT if aperture_open else 0.
                if safe_dwell>=.2:
                    phase='access';target=ACCESS;resumed_count+=1;event('access_resumed')
                elif angle<math.radians(12) and abs(velocity)<.15:
                    safe_abort=True;phase='help';event('help_requested',reason='closed_box_requires_reopening')
            elif phase=='access':
                target=ACCESS
                at_access=float(np.linalg.norm(tcp-np.asarray(ACCESS)))<.012
                clear=angle>math.radians(45)
                access_dwell=access_dwell+CONTROL_DT if at_access and clear else 0.
                if access_dwell>=.2:
                    reached=True;phase='return';target=WITHDRAW;event('inspection_waypoint_reached')
            elif phase=='return':
                target=WITHDRAW
                if float(np.linalg.norm(tcp-np.asarray(WITHDRAW)))<.018:
                    returned=True;phase='done';event('return_complete')
            elif phase in ('done','help'):
                target=WITHDRAW
            desired=ws.controller_action(model,data,target,.025,validate=False)
            decision_ms.append((time.perf_counter()-decision_started)*1000)
            if record:
                poses=np.concatenate((data.xpos,data.xquat),axis=1)
                frames.append({'t':round(float(data.time),6),'body_poses':poses.tolist(),'phase':phase,
                               'lid_angle_rad':angle,'lid_velocity_rad_s':velocity,
                               'tcp_position_m':tcp.tolist(),'target_position_m':list(target),
                               'qpos_rad_m':data.qpos.tolist(),'qvel_rad_m_s':data.qvel.tolist(),
                               'issued_ctrl':data.ctrl.tolist(),'requested_joint_target':desired.tolist(),
                               'risk':risk,'lid_robot_force_n':contact_force,
                               'native_window_peak_force_n':window_peak,'other_environment_robot_force_n':other_force})
            window_peak=0.
        if step==round(DURATION/TIMESTEP):break
        data.ctrl[:6]+=np.clip(desired[:6]-data.ctrl[:6],-ws.CONTROLLER['joint_rate_rad_s']*TIMESTEP,ws.CONTROLLER['joint_rate_rad_s']*TIMESTEP)
        data.ctrl[6]+=float(np.clip(desired[6]-data.ctrl[6],-ws.CONTROLLER['gripper_rate_m_s']*TIMESTEP,ws.CONTROLLER['gripper_rate_m_s']*TIMESTEP))
        # No qpos/qvel/force writes during rollout: only native joint actuators.
        mj.mj_step(model,data);mj.mj_forward(model,data)
        if not all(np.isfinite(value).all() for value in (data.qpos,data.qvel,data.qacc)) or any(w.number for w in data.warning):
            valid=False;failure_reason='nonfinite_state_or_solver_warning';event('invalid_native_state');break
    wall_seconds=time.perf_counter()-wall_started
    collision=peak_force>CONTACT_THRESHOLD_N
    other_collision=other_peak_force>CONTACT_THRESHOLD_N
    success=bool(valid and reached and returned and not collision and not other_collision)
    safe_abort=bool(valid and safe_abort and not collision and not other_collision)
    if valid and not success and not safe_abort:
        failure_reason=('lid_robot_contact' if collision else 'other_environment_robot_contact' if other_collision
                        else 'inspection_waypoint_not_reached' if not reached else 'return_timeout')
    completion_time=next((e['t'] for e in events if e['type']=='return_complete'),None)
    withdrawal_time=next((e['t'] for e in events if e['type']=='withdrawal_complete'),None)
    outcome='recovered' if success and resumed_count else 'completed' if success else 'safe_abort' if safe_abort else 'failed'
    metrics={'outcome':outcome,'task_success':success,'inspection_waypoint_reached':reached,'return_completed':returned,
             'safe_abort':safe_abort,'help_requested':any(e['type']=='help_requested' for e in events),
             'resumed_count':resumed_count,'recovery_count':resumed_count,'intervention_count':intervention_count,
             'missed_interventions':int(first_risk is not None and first_intervention is None),
             'false_interventions':None,'predicate_mismatch_interventions':false_interventions,
             'false_intervention_definition':'Not identifiable from the treated rollout alone. Compare this case against baseline: a withdrawal when baseline completes without contact is a counterfactual nuisance interruption, not proven necessary avoidance.',
             'missed_intervention_definition':'At least one observed descending-lid predicate with no withdrawal command; this is guard coverage, not calibrated danger classification.',
             'lid_robot_collision':collision,'other_environment_robot_collision':other_collision,
             'peak_contact_force_n':max(peak_force,other_peak_force),
             'peak_lid_robot_normal_force_n':peak_force,'peak_other_environment_robot_force_n':other_peak_force,
             'lid_robot_peak_force_time_integral_ns':impulse,'contact_duration_s':contact_steps*TIMESTEP,
             'hazard_duration_s':risk_steps*TIMESTEP,'first_hazard_s':first_risk,
             'first_intervention_s':first_intervention,'first_command_s':first_intervention,'first_contact_s':first_contact,
             'detection_time_s':first_risk,'command_time_s':first_intervention,
             'reaction_measurement_scope':'Privileged native predicate sampled at 2 ms; authored command at 20 ms. No sensor, transport or actuator-response latency included.',
             'first_response_latency_s':None if first_intervention is None else max(0.,first_intervention-(first_risk or 0.)),
             'completion_time_s':completion_time,
             'withdrawal_duration_s':None if withdrawal_time is None or first_intervention is None else withdrawal_time-first_intervention,
             'simulated_seconds':float(data.time),'wall_seconds':wall_seconds,
             'physics_realtime_factor':float(data.time)/wall_seconds,
             'decision_latency_p50_ms':float(np.percentile(decision_ms,50)),
             'decision_latency_p95_ms':float(np.percentile(decision_ms,95)),
             'decision_latency_max_ms':max(decision_ms),'native_steps':int(round(data.time/TIMESTEP)),
             'state_action_frame_timing':'Frame state and issued_ctrl are observed at t before the next native step; requested_joint_target is the authored 20 ms decision subsequently rate-limited into issued_ctrl.',
             'state_action_units':{'qpos_rad_m':{'0:6':'arm rad','6:8':'finger m','8:11':'shoe position m','11:15':'shoe quaternion wxyz dimensionless','15':'hinge rad'},
                                  'qvel_rad_m_s':{'0:6':'arm rad/s','6:8':'finger m/s','8:11':'shoe linear m/s','11:14':'shoe angular rad/s','14':'hinge rad/s'},
                                  'issued_ctrl':{'0:6':'arm rad','6':'jaw m'},'requested_joint_target':{'0:6':'arm rad','6':'jaw m'}},
             'terminal_phase':phase,'failure_reason':failure_reason,'solver_warnings':int(sum(w.number for w in data.warning))}
    return {'task':TASK,'case':case,'policy':policy,'success':success,'valid':valid,
            'metrics':metrics,'events':events,'frames':frames,'model':model,
            'native_trace_columns':['t','lid_angle_rad','lid_velocity_rad_s','lid_robot_normal_force_n','risk','other_environment_robot_force_n'],
            'native_trace':trace,'scope':SCOPE}
