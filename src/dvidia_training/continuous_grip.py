"""Continuous authored grip verification on the unchanged v0.1 rake scene.

Baseline means the v0.1 mounted-rake controller, not its no-tool capability
control. New actions use only the same bounded joint/jaw position actuators.
Every observation is privileged simulator state; no learned perception,
calibrated force limits, tool acquisition, or hardware safety claim is made.
"""
from __future__ import annotations
from copy import deepcopy
import math
import time
from . import tool_reach as v01, workstation as ws, reference_sim as sim

TASK='tool-reach'
MAX_SECONDS=30.
GRASP_CANDIDATES=((- .045,.060,150,False,0.),(-.045,.060,150,False,0.))
ATTEMPT_RESERVE_SECONDS=7.5  # max1.8approach+1close+1.5lift+.5hold+2.1cleanup, plus margin


def cases(split='development'):
    if split=='development': return v01.cases('development')
    if split=='regression': return [dict(c,split='regression',previous_split=c['split']) for c in v01.cases('development')+v01.cases('heldout')]
    raise ValueError('New heldout cases are owned by the independent benchmark protocol')


def _quaternion(yaw,pitch=150,roll=0.):
    """World-z yaw times the original downward wrist quaternion."""
    a,b=math.cos(math.radians(pitch)/2),math.sin(math.radians(pitch)/2)
    cr,sr=math.cos(math.radians(roll)/2),math.sin(math.radians(roll)/2)
    w,x,y,z=cr*a,sr*a,cr*b,sr*b;c,s=math.cos(yaw/2),math.sin(yaw/2)
    return [c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]


def controller_action(model,data,target,jaw,orientation):
    """Bounded IK with explicit orientation, without modifying workstation globals."""
    mj,np=sim._native_imports();tcp,_=ws._ids(model)
    ws._vec(target,3,'target');ws._vec([jaw],1,'jaw',0,.035)
    if any(not low-1e-12<=v<=high+1e-12 for v,(low,high) in zip(target,ws.BOUNDS)):
        raise ValueError('Target exceeds original workstation envelope')
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv));mj.mj_jacSite(model,data,jp,jr,tcp)
    current,rot=np.empty(4),np.empty(3);mj.mju_mat2Quat(current,data.site_xmat[tcp]);mj.mju_subQuat(rot,np.asarray(orientation),current)
    rot=data.site_xmat[tcp].reshape(3,3)@rot;delta=np.asarray(target)-data.site_xpos[tcp]
    for vector,cap in ((delta,.025),(rot,.12)):
        norm=float(np.linalg.norm(vector))
        if norm>cap:vector*=cap/norm
    jac=np.vstack((jp[:,:6],.35*jr[:,:6]));error=np.concatenate((delta,.35*rot))
    dq=jac.T@np.linalg.solve(jac@jac.T+.012**2*np.eye(6),error)
    angles=np.clip(data.qpos[:6]+np.clip(dq,-.18,.18),model.actuator_ctrlrange[:6,0]+.02,model.actuator_ctrlrange[:6,1]-.02)
    return np.concatenate((angles,[jaw]))


class Episode:
    def __init__(self,assets,case,record):
        self.mj,self.np=sim._native_imports();self.case=deepcopy(case);self.record=record;self.started=time.perf_counter()
        self.model=v01.build_model(assets,case,'recovery');self.data=ws.initialize(self.model)
        self.tcp,self.shoe=ws._ids(self.model);self.tool=self.mj.mj_name2id(self.model,self.mj.mjtObj.mjOBJ_BODY,'reach_rake')
        self.fingers=[self.mj.mj_name2id(self.model,self.mj.mjtObj.mjOBJ_BODY,n) for n in ('link7','link8')]
        self.phase='settle';self.target=self.np.asarray(ws.HOME_TARGET);self.jaw=.035;self.orientation=list(ws.ORIENTATION);self.custom=False
        self.desired=self.data.ctrl.copy();self.steps=0;self.frames=[];self.events=[];self.trace=[];self.policy_ms=[]
        self.force=[0.,0.,0.,0.,0.];self.max_delta=0.;self.first_loss=None;self.loss_since=None;self.monitor=False
        self.recoveries=0;self.retrieval_retries=0;self.grasp_attempts=0;self.retrieved=False;self.retained=False;self.help=False;self.reason=None
        self.peak_lift=0.;self.guard_mode=None;self.test_initial=None;self.qualified_since=None;self.pending_recovery=None;self.slip_since=None
        self.event('assumptions',pre_equipped_rake=True,physical_model_unchanged=True,privileged_observations=True)
    def event(self,kind,**fields):
        self.events.append(dict(t=round(float(self.data.time),6),type=kind,message=str(fields.get('reason',fields.get('name',kind))).replace('_',' '),**fields))
    def point(self):return self.data.xpos[self.shoe].copy()
    def yaw(self):
        w,x,y,z=self.data.xquat[self.shoe]
        return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    def clip(self,p):return self.np.clip(p,[x[0] for x in ws.BOUNDS],[x[1] for x in ws.BOUNDS])
    def bilateral(self):return min(self.force[2:4])>.01
    def supported(self):return self.force[4]>.01
    def qualified(self):return self.point()[2]>.035 and self.bilateral() and self.force[0]<.01 and self.force[4]<.01 and self.force[1]-sum(self.force[2:4])-self.force[0]<.01
    def tick(self):
        if self.steps>=round(MAX_SECONDS/.002):raise TimeoutError('Episode exceeded declared duration')
        if self.steps%10==0:
            t=time.perf_counter()
            self.desired=(controller_action(self.model,self.data,self.target.tolist(),self.jaw,self.orientation) if self.custom
                          else ws.controller_action(self.model,self.data,self.target.tolist(),self.jaw))
            self.policy_ms.append((time.perf_counter()-t)*1000)
        old=self.data.ctrl.copy();self.data.ctrl[:6]+=self.np.clip(self.desired[:6]-self.data.ctrl[:6],-.003,.003)
        self.data.ctrl[6]+=float(self.np.clip(self.desired[6]-self.data.ctrl[6],-.0001,.0001))
        self.max_delta=max(self.max_delta,float(max(abs(self.data.ctrl[:6]-old[:6]))))
        if self.pending_recovery is not None:
            self.event('recovery_control_issued',requested_at_s=self.pending_recovery,issued_ctrl=self.data.ctrl.tolist(),requested_joint_target=self.desired.tolist(),time_convention='pre-step control packet; physics advances during the following 2 ms')
            self.pending_recovery=None
        self.mj.mj_step(self.model,self.data);self.mj.mj_forward(self.model,self.data);self.steps+=1
        if not all(self.np.isfinite(v).all() for v in (self.data.qpos,self.data.qvel,self.data.qacc)) or any(w.number for w in self.data.warning):raise RuntimeError('Invalid native state')
        tool,total,left,right,world=0.,0.,0.,0.,0.
        for i in range(self.data.ncon):
            contact=self.data.contact[i];a=int(self.model.geom_bodyid[contact.geom1]);b=int(self.model.geom_bodyid[contact.geom2])
            if self.shoe not in (a,b):continue
            other=b if a==self.shoe else a;wrench=self.np.zeros(6);self.mj.mj_contactForce(self.model,self.data,i,wrench);n=max(0.,float(wrench[0]))
            if other==0:world+=n
            else:total+=n
            if other==self.tool:tool+=n
            if other==self.fingers[0]:left+=n
            if other==self.fingers[1]:right+=n
        self.force=[tool,total,left,right,world];t=round(float(self.data.time),6);p=self.point();tcp=self.data.site_xpos[self.tcp]
        self.trace.append([t,tool,total,left,right,*p.tolist(),world,*tcp.tolist()])
        if self.phase in ('test_lift','lift','retention_hold'):self.peak_lift=max(self.peak_lift,float(p[2]))
        if self.monitor and not self.bilateral():
            if self.first_loss is None:self.first_loss=t
        elif self.monitor:self.first_loss=None
        if self.record and self.steps%10==0:
            self.frames.append({'t':t,'body_poses':self.np.column_stack((self.data.xpos,self.data.xquat)).tolist(),'phase':self.phase,
                'target':self.target.tolist(),'target_orientation_wxyz':self.orientation,'shoe_position':p.tolist(),'tcp_position':tcp.tolist(),
                'qpos_rad_m':self.data.qpos.tolist(),'qvel_rad_m_s':self.data.qvel.tolist(),'issued_ctrl':self.data.ctrl.tolist(),
                'requested_joint_target':self.desired.tolist(),'tool_shoe_force_n':tool,'finger_normal_forces_n':[left,right],'world_shoe_force_n':world})
    def guard(self):
        """Called only on the 20 ms observation clock; native loss origin is logged separately."""
        if not self.monitor:return True
        if not self.bilateral():
            if self.loss_since is None:self.loss_since=float(self.data.time)
            if float(self.data.time)-self.loss_since>=.039:
                self.reason='bilateral_contact_lost'
                self.event('grip_loss_detected',reason=self.reason,first_native_loss_s=self.first_loss)
                return False
        else:self.loss_since=None
        # A falling shoe can keep touching both jaws. Compare a short physical
        # motion window, requiring actual downward object motion as well as
        # object/TCP separation; benign upward load transfer does not trigger.
        window=self.trace[-101:]
        slipping=(len(window)==101 and window[-1][7]-window[0][7]<-.010
                  and (window[-1][7]-window[-1][11])-(window[0][7]-window[0][11])<-.035)
        if slipping:
            if self.slip_since is None:self.slip_since=float(self.data.time)
            if float(self.data.time)-self.slip_since>=.039:
                self.reason='relative_downward_slip'
                self.event('grip_loss_detected',reason=self.reason,first_native_loss_s=None,
                           relative_drop_m=(window[0][7]-window[0][11])-(window[-1][7]-window[-1][11]))
                return False
        else:self.slip_since=None
        return True
    def motion(self,name,end,seconds,opening=.035,monitor=False):
        self.phase=name;self.jaw=opening;self.monitor=monitor;self.loss_since=None;self.first_loss=None;self.slip_since=None
        start=self.clip(self.data.site_xpos[self.tcp]);end=self.clip(end);n=round(seconds/.002)
        self.event('phase',name=name,target=end.tolist())
        for i in range(n):
            if self.steps%10==0 and not self.guard():self.monitor=False;return False
            f=(i+1)/n;f=f*f*(3-2*f);self.target=start*(1-f)+end*f;self.tick()
        self.monitor=False;return True
    def hold(self,name,seconds,opening=None,monitor=False):
        self.phase=name
        if opening is not None:self.jaw=opening
        if not (self.monitor and monitor):self.loss_since=None;self.first_loss=None;self.slip_since=None
        self.monitor=monitor
        self.event('phase',name=name,target=self.target.tolist())
        n=round(seconds/.002)
        for _ in range(n):
            if self.steps%10==0 and not self.guard():self.monitor=False;return False
            self.tick()
        self.monitor=False;return True
    def controlled_lower(self,reason):
        self.recoveries+=1;self.pending_recovery=round(float(self.data.time),6);self.event('grip_recovery_started',reason=reason,requested_target=[*self.data.site_xpos[self.tcp,:2],.060])
        self.motion('controlled_lower',[*self.data.site_xpos[self.tcp,:2],.060],.75,0.)
        self.hold('support_check',.20,0.)
        if not self.supported():
            self.help=True;self.reason='support_not_reestablished';return False
        self.hold('open_on_support',.50,.035)
        self.motion('regrasp_clearance',[*self.data.site_xpos[self.tcp,:2],.20],.65,.035)
        return True
    def retrieve(self):
        self.hold('settle',.4)
        requested=float(self.point()[0])+.165-self.case['tool_length']
        if requested>.54:
            self.help=True;self.reason='tool_cannot_reach_far_edge';self.event('help_requested',reason=self.reason);self.hold('hold_for_help',.6);return False
        for attempt in range(2):
            if attempt:self.retrieval_retries+=1
            y=float(self.point()[1])+(self.case['lateral_error'] if attempt==0 else 0.)
            far=float(self.np.clip(float(self.point()[0])+.165-self.case['tool_length'],.18,.54))
            self.motion('approach_tool',[far,y,.25],1.8)
            self.motion('lower_tool',[far,y,.075],1.)
            self.motion('pull_shoe',[.20,y,.075],3.)
            self.hold('verify_pull',.4)
            self.retrieved=bool(self.point()[0]<=.53 and abs(self.point()[1])<=.21 and abs(self.point()[2])<.035)
            self.event('retrieval_check',retrieved=self.retrieved,shoe_x_m=float(self.point()[0]))
            if self.retrieved:break
            self.event('recovery',reason='recenter_rake')
            self.target=self.np.array([.20,y,.22]);self.hold('release_tool',1.)
        if not self.retrieved:self.help=True;self.reason='retrieval_exhausted';return False
        self.target=self.np.array([.24,self.case['shoe_y'],.25]);self.hold('clear_rake',1.2)
        self.reference_support=self.force[4]
        self.event('post_clear_pose',position_m=self.point().tolist(),yaw_rad=self.yaw(),reference_support_n=self.reference_support)
        return True
    def monitored_lift(self,end):
        """An early observed 40 mm checkpoint on the original continuous path.

        Continuing the same bounded trajectory avoids introducing an artificial
        stop. The checkpoint verifies incipient transfer, not unsupported lift.
        """
        self.phase='test_lift';self.jaw=0.;self.monitor=True
        self.loss_since=None;self.first_loss=None;self.slip_since=None
        start=self.clip(self.data.site_xpos[self.tcp]);shoe_start=self.point()
        end=self.np.asarray(end);n=round(1.5/.002);checked=False
        self.event('phase',name='test_lift',target=end.tolist())
        for i in range(n):
            if self.steps%10==0:
                if not self.guard():self.monitor=False;return False
                if not checked and self.data.site_xpos[self.tcp,2]-start[2]>=.040:
                    checked=True;rise=float(self.point()[2]-shoe_start[2])
                    unloaded=self.force[4]<.7*self.reference_support and self.force[0]<.01
                    passed=rise>.003 and unloaded and self.bilateral()
                    self.event('test_lift_check',object_rise_m=rise,tcp_rise_m=float(self.data.site_xpos[self.tcp,2]-start[2]),
                               bilateral=self.bilateral(),unloaded=unloaded,passed=passed,
                               support_force_n=self.force[4],reference_support_n=self.reference_support,
                               scope='incipient load transfer during continuous lift; strict unsupported retention scored separately')
                    if not passed:
                        self.reason='test_lift_not_following_or_supported';self.event('grip_loss_detected',reason=self.reason,first_native_loss_s=None)
                        self.monitor=False;return False
                    self.phase='lift';self.event('phase',name='lift',target=end.tolist())
            f=(i+1)/n;f=f*f*(3-2*f);self.target=start*(1-f)+end*f;self.tick()
        if not checked:
            self.monitor=False;self.reason='test_lift_target_not_reached'
            self.event('test_lift_check',passed=False,reason=self.reason)
            return False
        # Keep debounce history into the retention hold; do not reset a loss
        # merely because the trajectory switches from motion to a hold.
        return True
    def grasp(self):
        self.custom=True
        for index,(offset,z,pitch,align_yaw,roll) in enumerate(GRASP_CANDIDATES):
            # Final attempt retains sufficient budget for controlled lowering.
            if self.data.time+ATTEMPT_RESERVE_SECONDS>MAX_SECONDS:self.reason='bounded_episode_budget';break
            self.grasp_attempts+=1;p=self.point();yaw=self.yaw() if align_yaw else 0.;self.orientation=_quaternion(yaw,pitch,roll);self.custom=align_yaw
            aim=self.np.array([p[0]+offset*math.cos(yaw),p[1]+offset*math.sin(yaw),z])
            if any(not low+.008<=v<=high-.008 for v,(low,high) in zip(aim[:2],ws.BOUNDS[:2])):
                self.reason='grasp_pose_outside_margin';self.event('grasp_rejected',reason=self.reason);continue
            self.event('grasp_target',attempt=index+1,object_position_m=p.tolist(),object_yaw_rad=self.yaw(),commanded_yaw_rad=yaw,target=aim.tolist(),orientation_wxyz=list(self.orientation))
            if align_yaw:self.motion('align_above',[aim[0],aim[1],.18],.7)
            self.motion('grasp_approach',aim,1.8 if not align_yaw else 1.0)
            self.hold('close_jaws',1.,0.)
            if not self.bilateral():
                self.reason='bilateral_acquisition_failed';self.event('grip_loss_detected',reason=self.reason,first_native_loss_s=None)
                if not self.controlled_lower(self.reason):break
                continue
            self.event('bilateral_acquired',finger_forces_n=self.force[2:4])
            lifted=self.monitored_lift([aim[0],aim[1],.28])
            if lifted:lifted=self.hold('retention_hold',.5,0.,True)
            tail=self.trace[-250:]
            self.retained=bool(lifted and len(tail)==250 and all(r[7]>.035 and min(r[3:5])>.01 and r[1]<.01 and r[8]<.01 and r[2]-sum(r[3:5])-r[1]<.01 for r in tail))
            self.event('lift_check',retained=self.retained)
            if self.retained:return True
            if lifted:self.reason='terminal_retention_failed';self.event('grip_loss_detected',reason=self.reason,first_native_loss_s=None)
            if not self.controlled_lower(self.reason):break
        self.help=True;self.event('help_requested',reason=self.reason or 'bounded_regrasp_exhausted');return False
    def result(self,valid=True):
        p=self.point();wall=time.perf_counter()-self.started
        metrics={'outcome':('recovered' if self.recoveries else 'completed') if self.retained else self.reason or 'incomplete','retrieval_success':self.retrieved,
            'retained_lift_success':self.retained,'grasp_attempted':bool(self.grasp_attempts),'grasp_attempts':self.grasp_attempts,
            'recovery_count':self.recoveries,'retrieval_retries':self.retrieval_retries,'intervention_count':self.recoveries+int(self.help),
            'help_requested':self.help,'episode_budget_s':MAX_SECONDS,'max_grasp_attempts':len(GRASP_CANDIDATES),'attempt_and_cleanup_reserve_s':ATTEMPT_RESERVE_SECONDS,'peak_contact_force_n':max(r[2] for r in self.trace),'peak_lift_m':self.peak_lift,
            'final_shoe_position_m':p.tolist(),'simulation_seconds':float(self.data.time),'native_steps':self.steps,'wall_seconds':wall,
            'real_time_factor':float(self.data.time)/wall,'physics_timestep_s':.002,'policy_interval_s':.02,
            'max_joint_command_delta_rad':self.max_delta,'solver_warnings':sum(int(w.number) for w in self.data.warning),
            'shoe_external_force_used':bool(self.np.any(self.data.xfrc_applied) or self.np.any(self.data.qfrc_applied)),
            'policy_compute_p50_ms':float(self.np.percentile(self.policy_ms,50)),'policy_compute_p95_ms':float(self.np.percentile(self.policy_ms,95)),
            'policy_compute_max_ms':max(self.policy_ms),'retention_claim_scope':'strict','regrasp_scope':'one supported lowering, reopening and recentered retry using the same original grasp geometry; no alternate learned grasp',
            'comparison_scope':'unchanged v0.1 mounted-rake controller versus continuous-grip controller on identical model',
            'observation_scope':'privileged native state; first contact loss recorded at2ms, decisions at20ms; loss debounce requires three failed samples spanning40ms, not continuous native loss; motion-slip window200ms plus sampled debounce; command issue is not hardware response',
            'recorded_state_scope':'qpos:6armrad,2jawm,shoefreeXYZm+quatwxyz; qvel:6armrad/s,2jawm/s,shoelinearm/s+angularrad/s; ctrl:6rad+1jawm',
            'test_lift_scope':'40 mm observed TCP checkpoint during a continuous trajectory; incipient object rise and partial unloading, not unsupported success',
            'retention_criterion':'final500ms continuous originz>35mm, bothfingernormalforces>0.01N, world/tool/otherrobot support each<0.01N'}
        return dict(task=TASK,case=self.case,policy='recovery',valid=valid,success=self.retained and valid,metrics=metrics,events=self.events,
                    frames=self.frames,model=self.model,native_trace=self.trace,native_trace_columns=['t_s','tool_shoe_force_n','shoe_robot_force_n','left_finger_force_n','right_finger_force_n','shoe_x_m','shoe_y_m','shoe_z_m','world_shoe_force_n','tcp_x_m','tcp_y_m','tcp_z_m'],control_compute_ms=self.policy_ms)


def run_case(assets,case,policy='baseline',record=True):
    if policy=='baseline':
        r=v01.run_case(assets,case,policy='recovery',record=record)
        r['policy']='baseline';r['task']=TASK
        r['metrics']['comparison_scope']='v0.1 mounted-rake controller, not original no-tool baseline'
        return r
    if policy!='recovery':raise ValueError('Unknown policy')
    ep=Episode(assets,case,record)
    try:
        if ep.retrieve():ep.grasp()
    except TimeoutError:
        ep.reason='bounded_episode_budget';ep.help=True;ep.event('help_requested',reason=ep.reason)
    return ep.result()
