"""Frozen development/held-out benchmark for the authored native CPU arm.

Source snapshots are isolated packages, including their own environment module.
The CLI exposes development results before an explicit final-source freeze;
held-out baseline/final evaluation is a single sealed comparison thereafter.
"""
from __future__ import annotations

import argparse
import ast
import copy
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import types

import mujoco
import numpy as np

DEFAULT_ROOT = Path('runs/arm-improvement-v1')
SOURCE_FILES = ('arm_env.py', 'arm_policy.py', 'env.py')
TIMING = ('Single-world instrumented CPU episode: compilation, reset/settling, '
          'controller/IK, native stepping, observations and control-cadence diagnostics; '
          'excludes interpreter startup, imports, source copying, JSON I/O, installation, '
          'rendering and web UI. No GPU measurement or real-robot transfer claim.')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def _label(value):
    if not isinstance(value, str) or not re.fullmatch('[a-z][a-z0-9_-]{0,31}', value):
        raise ValueError('Snapshot label must be a short lowercase identifier.')
    return value


def snapshot_source(root, label, source=Path('simlab')):
    """Copy the complete arm source dependency closure; never overwrite a label."""
    destination = Path(root)/'snapshots'/_label(label)
    manifest = destination/'snapshot.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        for name, expected in record['files'].items():
            if sha256((destination/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Frozen source integrity failure: {label}/{name}')
        return record
    contents = {name: (Path(source)/name).read_bytes() for name in SOURCE_FILES}
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in contents.items():
        (destination/name).write_bytes(data)
    (destination/'__init__.py').write_text('"""Isolated arm benchmark snapshot."""\n')
    record = {'label': label, 'files': {name: sha256(data).hexdigest() for name, data in contents.items()},
              'exact_original_sources': True}
    write_json(manifest, record)
    return record


def load_snapshot(root, label):
    record = snapshot_source(root, label)
    directory = (Path(root)/'snapshots'/label).resolve()
    package = '_frozen_arm_'+digest({'source':record,'directory':str(directory)})[:16]
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        module.__package__ = package
        sys.modules[package] = module
    # Relative .env resolves inside this package, never to live simlab.env.
    env = importlib.import_module(package+'.arm_env')
    policy = importlib.import_module(package+'.arm_policy')
    return env, policy, record


def _case(identifier, xy, size, mass, friction, jaw=15., torque=40., group='nominal', seed=0):
    sx, sy, tx, ty = xy
    z = .29+size[2]/2
    return {'id': identifier, 'group': group, 'seed': seed,
            'config': {'object_position': [sx, sy, z], 'target_position': [tx, ty, z],
                       'object_half_size': [x/2 for x in size], 'object_mass': mass,
                       'object_friction': friction, 'jaw_force_limit': jaw,
                       'joint_torque_limit': torque, 'scene_jitter': 0.}}


def make_protocol():
    rows = [
        ((.42,-.04,.54,.10),(.04,.04,.04),.04,.8),
        ((.25,-.18,.48,.18),(.03,.03,.03),.02,.4),
        ((.26,.18,.54,-.07),(.05,.05,.05),.08,1.5),
        ((.54,-.09,.28,.10),(.04,.03,.05),.06,.8),
        ((.46,.20,.31,-.18),(.03,.05,.04),.06,1.2),
        ((.53,.15,.32,.14),(.05,.04,.03),.02,.4),
        ((.32,-.20,.54,.10),(.035,.045,.05),.08,1.),
        ((.55,.06,.35,-.18),(.045,.03,.035),.04,1.5),
        ((.38,.16,.50,-.16),(.04,.05,.03),.08,.4),
        ((.48,-.18,.26,-.17),(.05,.03,.05),.06,1.2),
        ((.31,.05,.52,.18),(.03,.04,.03),.08,.4),
        ((.51,.12,.30,-.10),(.05,.05,.05),.02,.8),
        ((.27,-.04,.54,-.11),(.04,.04,.04),.04,.8),
        ((.33,.19,.52,-.15),(.03,.035,.04),.04,1.5),
        ((.42,-.10,.50,.10),(.04,.04,.04),.08,.4,.15,40.),
        ((.47,.08,.31,-.14),(.05,.035,.04),.08,1.5,.3,40.),
        ((.35,-.15,.50,.12),(.03,.05,.05),.08,.8,.8,40.),
        ((.40,.00,.52,.15),(.04,.04,.04),.04,.8,15.,.5),
        ((.52,-.15,.30,.15),(.05,.05,.05),.08,1.2,15.,2.),
        ((.27,.18,.50,-.12),(.04,.03,.045),.06,1.2,2.,5.),
    ]
    dev = []
    for index, row in enumerate(rows):
        xy, size, mass, friction, *caps = row
        dev.append(_case(f'dev-{index:02}', xy, size, mass, friction,
                         *(caps or [15.,40.]), group='nominal' if index<14 else 'actuator_stress', seed=100+index))
    rng = np.random.default_rng(2026100717)
    held = []
    sizes = [.03,.035,.04,.045,.05]
    masses = [.02,.03,.04,.06,.08]
    frictions = [.4,.7,.8,1.,1.3,1.5]
    for index in range(32):
        while True:
            xy = np.round(rng.uniform([.251,-.198,.251,-.198],[.551,.198,.551,.198]),6)
            if (np.linalg.norm(xy[:2])<=.555 and np.linalg.norm(xy[2:])<=.555
                    and np.linalg.norm(xy[:2]-xy[2:])>=.09):
                break
        size = [sizes[index%5],sizes[(2*index+1)%5],sizes[(3*index+2)%5]]
        stress = index>=24
        held.append(_case(f'held-{index:02}', xy.tolist(), size, masses[(index*3)%5],
                          frictions[(index*5)%6], [.12,.25,.5,1.][index%4] if stress else 15.,
                          [.75,2.,5.,10.][index%4] if stress else 40.,
                          'actuator_stress' if stress else 'nominal', seed=10001+index))
    probes = [
        {'id':'reject-radial','override':{'object_position':[.60,.20,.31]},'exception':'ValueError'},
        {'id':'reject-small-object','override':{'object_half_size':[.01,.02,.02]},'exception':'ValueError'},
        {'id':'reject-heavy-object','override':{'object_mass':.10},'exception':'ValueError'},
        {'id':'reject-zero-jaw-force','override':{'jaw_force_limit':0.},'exception':'ValueError'},
        {'id':'unsupported-orientation','override':{'object_quaternion':[1.,0.,0.,0.]},'exception':'TypeError'},
        {'id':'unsupported-obstacles','override':{'obstacles':[]},'exception':'TypeError'},
        {'id':'unsupported-deformable','override':{'deformable':True},'exception':'TypeError'},
    ]
    return {'schema_version':1, 'name':'authored-arm-engineering-v1', 'engine':'MuJoCo 3.15.0',
            'dev':dev, 'heldout':held, 'rejection_probes':probes,
            'counts':{'dev':20,'heldout':32,'rejection_probes':7},
            'selection':'Fixed dev layouts and independently generated deterministic heldout layouts; no scene jitter. Different seed numbers alone are not coverage.',
            'tuning_rule':'Only development outcomes may guide two engineering rounds. Freeze final exact source before one baseline-versus-final heldout comparison. No heldout feedback tuning.',
            'success_rule':'Environment gate unchanged: loaded bilateral/table-free >=55mm lift retained100ms; released jaws; TCP>=100mm above object; table support, target tolerance, linear/angular settling and250ms default dwell.',
            'physics_scope':'Rigid axis-aligned boxes at reset, upright tool goal; no obstacle/selfcollision, cup handle, perception, deformable, orientation-generalization or physical-transfer qualification.',
            'contact_law':'Equal-priority geom max mixing: jaw mu=max(1,object_friction), table mu=max(.8,object_friction). Raw friction below floors is not low-friction grasp coverage.',
            'input':'Six joint position targets in radians plus one jaw opening width in metres; two physical jaw joints receive width/2. Actuator values are not an inferred demonstration.',
            'controls':'Idle holds initial six joint targets with jaws open; replay-open uses identical recorded joint-target tape while forcing jaws open. Both retain native bounded servos.',
            'timing_scope':TIMING, 'metric_sampling':'Native joint/force/contact diagnostic maxima sampled after each20ms control step; env max pad force tracks1ms substeps. Sampled maxima are not continuous physical force peaks.'}


def initialize(root=DEFAULT_ROOT, source=Path('simlab')):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_source(root,'v0',source)
    path = root/'protocol.json'
    if not path.exists():
        write_json(path,make_protocol())
        (root/'protocol.sha256').write_text(sha256(path.read_bytes()).hexdigest()+'\n')
    return read_protocol(root)


def read_protocol(root):
    root = Path(root)
    path = root/'protocol.json'
    if sha256(path.read_bytes()).hexdigest() != (root/'protocol.sha256').read_text().strip():
        raise ValueError('Frozen protocol bytes changed.')
    return json.loads(path.read_text())


def system_info():
    def sysctl(key):
        try:
            return subprocess.check_output(['/usr/sbin/sysctl','-n',key],stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            return None
    return {'utc':datetime.now(timezone.utc).isoformat(),'system':platform.system(),
            'macos':platform.mac_ver()[0],'kernel_release':platform.release(),
            'architecture':platform.machine(),'processor':sysctl('machdep.cpu.brand_string'),
            'logical_cpus':os.cpu_count(),'memory_bytes':sysctl('hw.memsize'),
            'python':platform.python_version(),'mujoco':mujoco.__version__,'numpy':np.__version__,
            'installed_packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}


def _failure_category(info, obs):
    if info.get('success'):return 'success'
    if not info.get('valid',False):return 'invalid_physics'
    if not info.get('grasp_seen'):return 'failed_grasp'
    if not info.get('lift_seen'):return 'failed_retained_lift'
    if (not any(obs.get('grasp_contacts',{}).values()) and obs.get('gripper_width',0.)<=.055
            and info.get('target_distance',1.)>info.get('config',{}).get('position_tolerance',.025)):
        return 'lost_after_lift'
    if obs.get('gripper_width',0.)<=.055 or any(obs.get('grasp_contacts',{}).values()):return 'failed_release'
    if info.get('target_distance',1.)>info.get('config',{}).get('position_tolerance',.025):return 'failed_target'
    return 'failed_settle_or_retreat'


def run_case(case, env_module, policy_module, *, control='placement', action_tape=None, trace=False):
    """Run one real engine episode; keep compact metrics unless explicitly tracing."""
    started = time.perf_counter()
    record = {'case_id':case['id'],'group':case['group'],'case_sha256':digest(case),
              'seed':case['seed'],'control':control,'success':False,'trace_enabled':trace}
    env = None
    stage = 'configure'
    tape, frames, latencies = [], [], []
    try:
        config = env_module.ArmConfig(**case['config'])
        env = env_module.ArmEnv(config)
        stage = 'reset'
        obs, info = env.reset(seed=case['seed'])
        setup = time.perf_counter()-started
        policy = policy_module.ArmPickPlacePolicy(env)
        initial_joint = obs['joint_position']
        initial_object = np.asarray(obs['object_position'])
        stats = {'samples':0,'max_abs_joint_velocity_rad_s':[0.]*6,
                 'max_abs_actuator_force':[0.]*8,'max_abs_native_ctrl':[0.]*8,
                 'max_abs_commanded_joint_target_rad':[0.]*6,
                 'commanded_gripper_width_min_max_m':[.08,0.],
                 'actual_gripper_width_min_max_m':[.08,0.],
                 'joint_position_min_rad':[1e9]*6,'joint_position_max_rad':[-1e9]*6,
                 'actuator_saturation_samples':[0]*8,'min_joint_limit_margin_rad':1e9,
                 'max_contact_count_sampled':0,'max_contact_penetration_m_sampled':0.,
                 'retained_lift_control_samples':0,'max_object_displacement_m':0.,
                 'max_object_linear_speed_m_s':0.,'max_object_angular_speed_rad_s':0.,
                 'jaw_contact_sliding_mu':[],'table_contact_sliding_mu':[]}
        stage_ticks = Counter()
        joint_ranges = env.model.jnt_range[env.joint_ids]
        force_caps = np.max(np.abs(env.model.actuator_forcerange),axis=1)
        relative_min, relative_max = np.full(3,np.inf), np.full(3,-np.inf)
        for tick in range(round(config.horizon/config.control_dt)):
            began = time.perf_counter()
            if control=='idle':
                action = {'joint_targets':initial_joint,'gripper_width':.08};stage='idle_hold'
            elif control=='replay_open_jaw':
                if not action_tape:raise ValueError('Open-jaw replay needs an action tape.')
                saved = action_tape[min(tick,len(action_tape)-1)]
                action = {**saved['action'],'gripper_width':.08};stage=saved['stage']
            elif control=='placement':
                action = policy(obs);stage=policy.stage
            else:raise ValueError('Unknown control.')
            obs, _, terminated, truncated, info = env.step(action)
            tape.append({'stage':stage,'action':action})
            latencies.append(time.perf_counter()-began)
            stage_ticks[stage]+=1
            if not info['valid']:break
            stats['samples']+=1
            stats['max_abs_commanded_joint_target_rad']=np.maximum(stats['max_abs_commanded_joint_target_rad'],np.abs(action['joint_targets'])).tolist()
            for key,value in (('commanded_gripper_width_min_max_m',action['gripper_width']),('actual_gripper_width_min_max_m',obs['gripper_width'])):
                stats[key]=[min(stats[key][0],value),max(stats[key][1],value)]
            actual_force=np.abs(env.data.actuator_force)
            stats['max_abs_actuator_force']=np.maximum(stats['max_abs_actuator_force'],actual_force).tolist()
            stats['max_abs_native_ctrl']=np.maximum(stats['max_abs_native_ctrl'],np.abs(env.data.ctrl)).tolist()
            stats['actuator_saturation_samples']=(np.asarray(stats['actuator_saturation_samples'])+(actual_force>=.95*force_caps)).tolist()
            stats['max_abs_joint_velocity_rad_s']=np.maximum(stats['max_abs_joint_velocity_rad_s'],np.abs(env.data.qvel[env.joint_dofs])).tolist()
            q=env.data.qpos[env.joint_qpos]
            stats['joint_position_min_rad']=np.minimum(stats['joint_position_min_rad'],q).tolist()
            stats['joint_position_max_rad']=np.maximum(stats['joint_position_max_rad'],q).tolist()
            stats['min_joint_limit_margin_rad']=min(stats['min_joint_limit_margin_rad'],float(np.min(np.minimum(q-joint_ranges[:,0],joint_ranges[:,1]-q))))
            stats['max_contact_count_sampled']=max(stats['max_contact_count_sampled'],int(env.data.ncon))
            stats['max_contact_penetration_m_sampled']=max(stats['max_contact_penetration_m_sampled'],max([max(0.,-float(c.dist)) for c in env.data.contact[:env.data.ncon]],default=0.))
            for contact in env.data.contact[:env.data.ncon]:
                pair={int(contact.geom1),int(contact.geom2)}
                if env.geom_ids['object_geom'] not in pair:continue
                name='table_contact_sliding_mu' if env.geom_ids['table'] in pair else 'jaw_contact_sliding_mu'
                mu=float(contact.friction[0])
                if mu not in stats[name]:stats[name].append(mu)
            pos=np.asarray(obs['object_position'])
            stats['max_object_displacement_m']=max(stats['max_object_displacement_m'],float(np.linalg.norm(pos-initial_object)))
            stats['max_object_linear_speed_m_s']=max(stats['max_object_linear_speed_m_s'],float(np.linalg.norm(obs['object_velocity'])))
            stats['max_object_angular_speed_rad_s']=max(stats['max_object_angular_speed_rad_s'],float(np.linalg.norm(obs['object_angular_velocity'])))
            if all(obs['grasp_contacts'][s]>0 for s in ('left','right')) and obs['table_contacts']==0 and pos[2]-initial_object[2]>=.055:
                stats['retained_lift_control_samples']+=1
                relative=pos-np.asarray(obs['end_effector_position'])
                relative_min=np.minimum(relative_min,relative);relative_max=np.maximum(relative_max,relative)
            if trace:frames.append({'time_s':info['simulation_time'],'stage':stage,'object':obs['object_position'],'tcp':obs['end_effector_position']})
            if terminated or truncated:break
        stats['retained_object_minus_tcp_range_world_m']=(relative_max-relative_min).tolist() if stats['retained_lift_control_samples'] else None
        stats['actuator_force_caps']=force_caps.tolist()
        stats['actuator_units']=['N*m']*6+['N']*2
        record.update(success=info['success'],status='completed',reason=info['reason'],
                      failure_category=_failure_category(info,obs),terminal_stage=stage,
                      stage_control_ticks=dict(stage_ticks),simulated_seconds=info['simulation_time'],
                      control_steps=len(tape),setup_seconds=setup,final_info=info,
                      final_object_position=obs['object_position'],metrics=stats,
                      control_latency_p50_seconds=float(np.median(latencies)),
                      control_latency_p95_seconds=float(np.percentile(latencies,95)),
                      model={'nq':int(env.model.nq),'nv':int(env.model.nv),'nu':int(env.model.nu),
                             'nmocap':int(env.model.nmocap),'neq':int(env.model.neq)},
                      warning_counts={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(env.data.warning) if w.number})
        if hasattr(policy,'diagnostics'):
            record['policy_diagnostics']=policy.diagnostics()
        if trace:record['trace']=frames
    except (ValueError,TypeError,RuntimeError,np.linalg.LinAlgError) as error:
        record.update(status='rejected' if stage=='configure' else 'exception',reason=type(error).__name__,
                      error=str(error),terminal_stage=stage,failure_category='configuration_rejected' if stage=='configure' else 'invalid_execution')
        if env is not None and hasattr(env,'data'):
            record['warning_counts']={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(env.data.warning) if w.number}
            record['engine_warnings']=list(getattr(env,'_warnings',[]))
    record['episode_wall_seconds']=time.perf_counter()-started
    record['simulated_seconds_per_wall_second']=record.get('simulated_seconds',0.)/record['episode_wall_seconds']
    return record,tape


def rejection_checks(probes, env_module):
    results=[]
    for probe in probes:
        try:
            env_module.ArmConfig(**probe['override'])
            result={'id':probe['id'],'rejected':False,'passed':False}
        except (ValueError,TypeError) as error:
            result={'id':probe['id'],'rejected':True,'exception':type(error).__name__,
                    'message':str(error),'passed':type(error).__name__==probe['exception']}
        results.append(result)
    return results


def summarize(records):
    groups={}
    for group in sorted({row['group'] for row in records}):
        rows=[r for r in records if r['group']==group]
        groups[group]={'successes':sum(r['success'] for r in rows),'cases':len(rows),
                       'failure_categories':dict(Counter(r['failure_category'] for r in rows))}
    return {'successes':sum(r['success'] for r in records),'cases':len(records),'groups':groups,
            'reasons':dict(Counter(r['reason'] for r in records)),
            'episode_wall_seconds':sum(r['episode_wall_seconds'] for r in records),
            'simulated_seconds':sum(r.get('simulated_seconds',0.) for r in records)}


def contract_signatures(root, label):
    """Allow instrumentation/control changes, reject changed task/physical laws."""
    snapshot_source(root,label)
    return _source_contract(Path(root)/'snapshots'/label/'arm_env.py')


def _source_contract(path):
    tree=ast.parse(Path(path).read_text())
    classes={node.name:node for node in tree.body if isinstance(node,ast.ClassDef)}
    methods={node.name:node for node in classes['ArmEnv'].body if isinstance(node,ast.FunctionDef)}
    names={'_grasp_seen','lift','retained_lift','_lift_dwell','_lift_seen',
           'released','supported','settled','hand_clear','qualifies','_dwell'}
    gate=[]
    for node in ast.walk(methods['step']):
        if isinstance(node,(ast.Assign,ast.AugAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            targets={target.id if isinstance(target,ast.Name) else target.attr if isinstance(target,ast.Attribute) else '' for target in targets}
            if targets&names:gate.append(ast.dump(node,include_attributes=False))
        if isinstance(node,ast.If) and 'dwell_seconds' in ast.dump(node.test):
            gate.append(ast.dump(node,include_attributes=False))
    constants=[ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id in ('JOINT_LIMITS','JOINT_KP','HOME','ARM_PROFILE','TASK_ID','ADAPTER_ID') for t in node.targets)]
    # These two explicitly authorized extensions are controller target ramps,
    # not changes to the original physical fixture or success thresholds.
    class ControllerExtensions(ast.NodeTransformer):
        fields={'joint_target_rate_limit','jaw_width_rate_limit'}
        def visit_AnnAssign(self,node):
            return None if isinstance(node.target,ast.Name) and node.target.id in self.fields else self.generic_visit(node)
        def visit_Tuple(self,node):
            node.elts=[item for item in node.elts if not (isinstance(item,ast.Constant) and item.value in self.fields)]
            return self.generic_visit(node)
    configuration=ControllerExtensions().visit(copy.deepcopy(classes['ArmConfig']))
    return {'configuration':digest(ast.dump(configuration,include_attributes=False)),
            'physics_xml':digest(ast.dump(methods['_xml'],include_attributes=False)),
            'loaded_contact_gate':digest(ast.dump(methods['_contact_state'],include_attributes=False)),
            'success_gate':digest(gate),'embodiment_constants':digest(constants)}


def wilson(successes, cases):
    if cases==0:return None
    if not 0<=successes<=cases:raise ValueError('Invalid binomial counts.')
    z=1.959963984540054
    rate=successes/cases;scale=1+z*z/cases
    centre=(rate+z*z/(2*cases))/scale
    width=z*math.sqrt(rate*(1-rate)/cases+z*z/(4*cases*cases))/scale
    return [max(0.,centre-width),min(1.,centre+width)]


def rates(records):
    def one(rows):
        n=len(rows);x=sum(r['success'] for r in rows)
        return {'successes':x,'cases':n,'rate':x/n if n else None,'wilson95':wilson(x,n)}
    result=one(records)
    result['groups']={group:one([r for r in records if r['group']==group]) for group in sorted({r['group'] for r in records})}
    return result


def paired_outcomes(before, after):
    old={r['case_id']:r for r in before};new={r['case_id']:r for r in after}
    if len(old)!=len(before) or len(new)!=len(after) or set(old)!=set(new):raise ValueError('Paired case IDs differ or repeat.')
    counts={'improved':0,'regressed':0,'both_success':0,'both_failure':0,'cases':len(old)}
    groups={}
    for ident,a in old.items():
        b=new[ident]
        if a['case_sha256']!=b['case_sha256'] or a['group']!=b['group']:raise ValueError('Paired case configuration differs.')
        category='both_success' if a['success'] and b['success'] else 'both_failure' if not a['success'] and not b['success'] else 'improved' if b['success'] else 'regressed'
        counts[category]+=1
        group=groups.setdefault(a['group'],{'improved':0,'regressed':0,'both_success':0,'both_failure':0,'cases':0})
        group[category]+=1;group['cases']+=1
    counts['groups']=groups
    return counts


def evaluate(label, *, root=DEFAULT_ROOT, source=Path('simlab'), trace=False, split='dev'):
    if split!='dev':raise ValueError('Heldout evaluation requires freeze-final and heldout commands.')
    root=Path(root);protocol=read_protocol(root)
    path=root/_label(label)/'dev.json'
    if path.exists():raise ValueError('This labelled development result already exists; use a new round label.')
    snapshot_source(root,label,source)
    env,policy,snapshot=load_snapshot(root,label)
    records=[run_case(case,env,policy,trace=trace)[0] for case in protocol['dev']]
    result={'schema_version':1,'label':label,'split':'dev','protocol_sha256':sha256((root/'protocol.json').read_bytes()).hexdigest(),
            'snapshot':snapshot,'system':system_info(),'timing_scope':TIMING,
            'harness_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'contract_signatures':contract_signatures(root,label),
            'metric_sampling':protocol['metric_sampling'],'records':records,
            'rejection_checks':rejection_checks(protocol['rejection_probes'],env),'summary':summarize(records)}
    write_json(path,result)
    return result


def freeze_final(*,root=DEFAULT_ROOT,source=Path('simlab')):
    root=Path(root);read_protocol(root)
    if (root/'final-freeze.json').exists():raise ValueError('Final source was already frozen.')
    original_contract=contract_signatures(root,'v0')
    final_contract=_source_contract(Path(source)/'arm_env.py')
    if original_contract!=final_contract:
        raise ValueError('Final source changed frozen physics/config/contact/success semantics.')
    snapshot=snapshot_source(root,'final',source)
    if final_contract!=contract_signatures(root,'final'):
        raise ValueError('Final snapshot differs from the source contract checked before copying.')
    record={'schema_version':1,'protocol_sha256':sha256((root/'protocol.json').read_bytes()).hexdigest(),
            'snapshot':snapshot,'unchanged_contract_signatures':final_contract,
            'rule':'No further policy tuning after heldout outcomes become visible.'}
    write_json(root/'final-freeze.json',record)
    return record


def heldout(*,root=DEFAULT_ROOT):
    root=Path(root);protocol=read_protocol(root)
    frozen=json.loads((root/'final-freeze.json').read_text())
    if frozen['protocol_sha256']!=sha256((root/'protocol.json').read_bytes()).hexdigest():raise ValueError('Freeze protocol mismatch.')
    if frozen['snapshot']!=snapshot_source(root,'final'):raise ValueError('Final source mismatch.')
    if (root/'heldout-started.json').exists():raise ValueError('Heldout comparison already started; do not tune or repeat it.')
    write_json(root/'heldout-started.json',frozen)
    comparison={}
    for label in ('v0','final'):
        env,policy,snapshot=load_snapshot(root,label)
        records=[];controls=[]
        for index,case in enumerate(protocol['heldout']):
            row,tape=run_case(case,env,policy)
            records.append(row)
            if index in (0,7,25):
                controls.extend([run_case(case,env,policy,control='idle')[0],
                                 run_case(case,env,policy,control='replay_open_jaw',action_tape=tape)[0]])
        result={'schema_version':1,'label':label,'split':'heldout','protocol_sha256':frozen['protocol_sha256'],
                'snapshot':snapshot,'system':system_info(),'timing_scope':TIMING,
                'harness_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
                'contract_signatures':contract_signatures(root,label),
                'metric_sampling':protocol['metric_sampling'],'records':records,
                'control_checks':controls,'summary':summarize(records)}
        write_json(root/label/'heldout.json',result)
        comparison[label]=result['summary']
    write_json(root/'heldout-comparison.json',comparison)
    make_assessment(root)
    return comparison


def make_assessment(root=DEFAULT_ROOT):
    """Compact publication payload: counts, paired results, scope and provenance."""
    root=Path(root);protocol=read_protocol(root)
    baseline=json.loads((root/'v0/heldout.json').read_text())
    final=json.loads((root/'final/heldout.json').read_text())
    labels=[label for label in ('v0','round1','round2','final') if (root/'snapshots'/label/'snapshot.json').exists()]
    sources={label:snapshot_source(root,label)['files'] for label in labels}
    rounds=[]
    for label in ('v0','round1','round2'):
        path=root/label/'dev.json'
        if path.exists():
            data=json.loads(path.read_text())
            rounds.append({'label':label,**rates(data['records'])})
    def timing(data):
        records=data['records'];wall=sum(r['episode_wall_seconds'] for r in records);sim=sum(r.get('simulated_seconds',0.) for r in records)
        return {'episode_wall_seconds':wall,'simulated_seconds':sim,
                'simulated_seconds_per_wall_second':sim/wall,
                'episode_real_time_factor_p50':float(np.median([r['simulated_seconds_per_wall_second'] for r in records])),
                'control_latency_p50_seconds_median':float(np.median([r['control_latency_p50_seconds'] for r in records if 'control_latency_p50_seconds' in r])),
                'control_latency_p95_seconds_median':float(np.median([r['control_latency_p95_seconds'] for r in records if 'control_latency_p95_seconds' in r]))}
    control={}
    for label,data in (('baseline',baseline),('final',final)):
        rows=data['control_checks']
        control[label]={'episodes':len(rows),'unexpected_successes':sum(r['success'] for r in rows),
                        'engine_warning_episodes':sum(bool(r.get('warning_counts')) or bool(r.get('final_info',{}).get('warnings')) for r in rows),
                        'checks':[{'case_id':r['case_id'],'control':r['control'],'success':r['success'],
                                   'reason':r['reason'],'max_object_displacement_m':r.get('metrics',{}).get('max_object_displacement_m')} for r in rows]}
    signatures=contract_signatures(root,'v0')
    result={'schema_version':1,'scope':'simulation-only','protocol_sha256':baseline['protocol_sha256'],
            'criteria':{'unchanged':signatures==contract_signatures(root,'final'),
                        'sha256':signatures['success_gate'],'text':protocol['success_rule']},
            'source_hashes':sources,'dev_rounds':rounds,
            'heldout':{'baseline':rates(baseline['records']),'final':rates(final['records']),
                       'paired':paired_outcomes(baseline['records'],final['records'])},
            'controls':control,'timing':{'baseline':timing(baseline),'final':timing(final)},
            'timing_scope':TIMING,'metric_sampling':protocol['metric_sampling'],
            'uncertainty_note':'Wilson 95% binomial reference intervals for a small fixed heterogeneous engineering suite; this is not an iid population sample or a real-world reliability interval.',
            'rejection_probe_count':len(protocol['rejection_probes']),
            'limits':[protocol['physics_scope'],protocol['contact_law'],
                      'One execution per controller per frozen heldout case; no repeated controlled timing experiment or GPU measurement.'],
            'measurement_harness_hashes':{'baseline':baseline['harness_sha256'],'final':final['harness_sha256']}}
    result['controller_extensions']={'joint_target_rate_limit_rad_s':3.,'jaw_width_rate_limit_m_s':.16,
                                     'scope':'Round2 target trajectory ramps before gravity feedforward; actual joint/jaw velocities are measured, not clamped.'}
    write_json(root/'assessment.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('init','dev','freeze-final','heldout'))
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--source',type=Path,default=Path('simlab'))
    parser.add_argument('--label',default='v0')
    args=parser.parse_args()
    if args.command=='init':result={'protocol_counts':initialize(args.root,args.source)['counts']}
    elif args.command=='dev':result=evaluate(args.label,root=args.root,source=args.source)['summary']
    elif args.command=='freeze-final':result=freeze_final(root=args.root,source=args.source)
    else:result=heldout(root=args.root)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
