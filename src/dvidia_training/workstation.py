"""Live native PiPER workstation: authored pose tracking, synthetic free shoes.

Target tracking uses native position actuators and damped differential IK. The
scene contains no shoe weld, grasp attachment, hidden carrying force or policy
trained from footage. Rich visual detail is separate from collision evidence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import gzip
import io
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
import tempfile
import shutil
import xml.etree.ElementTree as ET
import zlib

from . import asset_library, asset_preview, reference_sim as sim, robot_profiles

SCHEMA = 'dvidia.live-workstation/v1'
INTEGRITY_SCHEMA = 'dvidia.live-workstation-integrity/v1'
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
RUNTIME_FILES = {'index.html':'workstation.html',
                 'workstation.bundle.js':'workstation.bundle.js',
                 'workstation_worker.bundle.js':'workstation_worker.bundle.js',
                 'mujoco.wasm':'mujoco.wasm',
                 'workstation.third-party.txt':'workstation.third-party.txt'}
TIMESTEP = .002
CONTROL_DT = .02
TCP = 'workstation_tcp'
SHOE = 'shoe_trainer_left'
HOME_TARGET = (.36, -.06, .27)
ORIENTATION = (math.cos(5 * math.pi / 12), 0., math.sin(5 * math.pi / 12), 0.)
BOUNDS = ((.18, .54), (-.24, .24), (.055, .34))
CONTROLLER = {'orientation_weight': .25, 'translation_error_cap_m': .045,
              'rotation_error_cap_rad': .15, 'damping': .012,
              'max_joint_delta_rad': .18, 'joint_margin_rad': .02,
              'joint_rate_rad_s': 1.5, 'gripper_rate_m_s': .05,
              'gripper_range_m': [0., .035]}
SCOPE = ('Live native PiPER position tracking with authored IK, pinned upstream '
         'mechanics and a free synthetic rigid shoe. No learned shoe manipulation '
         'policy, calibrated shoe contact, deformation or hardware qualification.')


def _json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def _vec(value, count, label, low=-10., high=10.):
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(label + ' requires ' + str(count) + ' values')
    if any(type(x) not in (int, float) or not low <= x <= high or not math.isfinite(x) for x in value):
        raise ValueError(label + ' must contain bounded finite numbers')
    return list(map(float, value))


def _spec(assets):
    spec = asset_library._robot_spec('piper', assets, TIMESTEP)
    mj, _ = sim._native_imports()
    spec.modelname = 'dvidia_live_piper_workstation_v01'
    spec.body('link6').add_site(name=TCP, pos=[0, 0, .12003], size=[.004, 0, 0], rgba=[.18,.65,.49,1])
    world = spec.worldbody
    world.add_geom(name='workstation_floor', type=mj.mjtGeom.mjGEOM_PLANE,
                   pos=[0,0,-.8], size=[2,2,.1], rgba=[.82,.84,.79,1])
    world.add_geom(name='workstation_table_top', type=mj.mjtGeom.mjGEOM_BOX,
                   pos=[.22,0,-.04], size=[.65,.46,.04], rgba=[.62,.43,.26,1])
    for index,(x,y) in enumerate(((-.34,-.37),(-.34,.37),(.77,-.37),(.77,.37))):
        world.add_geom(name='workstation_table_leg_'+str(index),type=mj.mjtGeom.mjGEOM_BOX,
                       pos=[x,y,-.44],size=[.028,.028,.36],rgba=[.16,.19,.20,1])
    world.add_geom(name='workstation_rack_shelf',type=mj.mjtGeom.mjGEOM_BOX,
                   pos=[.37,.33,.10],size=[.20,.10,.012],rgba=[.49,.35,.23,1])
    for index,x in enumerate((.165,.575)):
        world.add_geom(name='workstation_rack_side_'+str(index),type=mj.mjtGeom.mjGEOM_BOX,
                       pos=[x,.33,.065],size=[.012,.11,.065],rgba=[.23,.27,.26,1])
    world.add_geom(name='workstation_rack_back',type=mj.mjtGeom.mjGEOM_BOX,
                   pos=[.37,.435,.065],size=[.20,.009,.065],rgba=[.23,.27,.26,1])
    shoe = next(item for item in asset_library.catalog()['shoes'] if item['id']=='trainer-left')
    body = world.add_body(name=SHOE,pos=[.32,-.145,.012],gravcomp=0)
    body.add_freejoint(name=SHOE+'_free')
    properties=shoe['properties']
    for piece in shoe['geometry']:
        body.add_geom(name=SHOE+'_'+piece['name'],
            type={'box':mj.mjtGeom.mjGEOM_BOX,'ellipsoid':mj.mjtGeom.mjGEOM_ELLIPSOID}[piece['type']],
            pos=piece['position_m'],size=piece['size_m'],mass=properties['mass_kg']['value']*piece['mass_fraction'],
            rgba=piece['rgba'],contype=1,conaffinity=1,condim=4,priority=1,
            friction=[properties[name]['value'] for name in ('sliding_friction','torsional_friction','rolling_friction')],
            solref=[.02,1],solimp=[.9,.95,.001,.5,2])
    return spec


def build_model(assets):
    spec=_spec(assets)
    mj,_=sim._native_imports()
    # Canonical XML is the identical input used by the browser engine.
    model=mj.MjModel.from_xml_string(spec.to_xml(),assets=spec.assets)
    if (model.nq,model.nv,model.nu,model.neq,model.nmocap)!=(15,14,7,1,0):
        raise ValueError('Workstation mechanical dimensions differ from the pinned PiPER contract')
    return model


def _ids(model):
    mj,_=sim._native_imports()
    if model.names.split(b'\0',1)[0] != b'dvidia_live_piper_workstation_v01':
        raise ValueError('Expected a native workstation model')
    tcp=mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,TCP)
    shoe=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,SHOE)
    if tcp<0 or shoe<0:
        raise ValueError('Workstation TCP or shoe is missing')
    return tcp,shoe


def controller_action(model,data,target_position_m,gripper_position_m,*,validate=True):
    """One 20 ms differential-IK target; does not mutate physics or teleport bodies."""
    target=_vec(target_position_m,3,'target')
    if validate and any(not bounds[0]<=value<=bounds[1] for value,bounds in zip(target,BOUNDS)):
        raise ValueError('Target is outside the bounded workstation volume')
    jaw=_vec([gripper_position_m],1,'gripper',0,.035)[0]
    mj,np=sim._native_imports()
    tcp,_=_ids(model)
    jacp,jacr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mj.mj_jacSite(model,data,jacp,jacr,tcp)
    current,rotation=np.empty(4),np.empty(3)
    mj.mju_mat2Quat(current,data.site_xmat[tcp])
    mj.mju_subQuat(rotation,np.asarray(ORIENTATION),current)
    rotation=data.site_xmat[tcp].reshape(3,3)@rotation
    translation=np.asarray(target)-data.site_xpos[tcp]
    for vector,limit in ((translation,CONTROLLER['translation_error_cap_m']),(rotation,CONTROLLER['rotation_error_cap_rad'])):
        norm=float(np.linalg.norm(vector))
        if norm>limit:vector*=limit/norm
    weight=CONTROLLER['orientation_weight']
    jac=np.vstack((jacp[:,:6],weight*jacr[:,:6]))
    error=np.concatenate((translation,weight*rotation))
    delta=jac.T@np.linalg.solve(jac@jac.T+CONTROLLER['damping']**2*np.eye(6),error)
    targets=np.clip(data.qpos[:6]+np.clip(delta,-CONTROLLER['max_joint_delta_rad'],CONTROLLER['max_joint_delta_rad']),
                    model.actuator_ctrlrange[:6,0]+CONTROLLER['joint_margin_rad'],
                    model.actuator_ctrlrange[:6,1]-CONTROLLER['joint_margin_rad'])
    return np.concatenate((targets,[jaw]))


def initialize(model):
    """Initialize a declared home pose; no simulation time or shoe settling hidden."""
    mj,np=sim._native_imports();_ids(model)
    data=mj.MjData(model)
    data.qpos[:8]=[0,1.57,-1.3485,0,0,0,.035,-.035]
    # Solve only the robot's initial pose. This reset-only kinematic solve is not
    # used when responding to target changes or interacting with the shoe.
    for _ in range(180):
        mj.mj_forward(model,data)
        command=controller_action(model,data,HOME_TARGET,.035)
        data.qpos[:6]=command[:6]
    data.qvel[:]=0
    data.ctrl[:6]=data.qpos[:6]
    data.ctrl[6]=.035
    mj.mj_forward(model,data)
    if not np.isfinite(data.qpos).all():raise ValueError('Invalid workstation home state')
    return data


def run_control(model,data,controller_state,target_position_m,gripper_position_m,substeps=10,*,impulse_xyz=None):
    """Advance bounded native ticks, retaining the 20 ms controller clock.

    Pass the same initially empty dictionary on every call. Target packets can
    arrive at rendering frequency; desired joint commands update every ten native
    steps, independently of rendering. The returned state is privileged truth.
    """
    if type(substeps) is not int or not 1<=substeps<=100:
        raise ValueError('substeps must be an integer in [1,100]')
    if type(controller_state) is not dict or set(controller_state)-{'native_steps','desired_ctrl'}:
        raise ValueError('Controller state requires its own empty or initialized dictionary')
    if type(controller_state.get('native_steps',0)) is not int or controller_state.get('native_steps',0)<0:
        raise ValueError('Controller native step count is invalid')
    mj,np=sim._native_imports();_,shoe=_ids(model)
    target=_vec(target_position_m,3,'target')
    if any(not bounds[0]<=value<=bounds[1] for value,bounds in zip(target,BOUNDS)):
        raise ValueError('Target is outside the bounded workstation volume')
    jaw=_vec([gripper_position_m],1,'gripper',0,.035)[0]
    impulse=None if impulse_xyz is None else _vec(impulse_xyz,3,'shoe impulse',-.12,.12)
    if impulse is not None and float(np.linalg.norm(impulse)) > .12 + 1e-12:
        raise ValueError('Shoe impulse norm exceeds 0.12 N s')
    for index in range(substeps):
        native_steps=controller_state.get('native_steps',0)
        if native_steps%10==0 or 'desired_ctrl' not in controller_state:
            controller_state['desired_ctrl']=controller_action(model,data,target,jaw).tolist()
        targets=np.asarray(_vec(controller_state['desired_ctrl'],7,'desired controls'))
        data.ctrl[:6]+=np.clip(targets[:6]-data.ctrl[:6],-CONTROLLER['joint_rate_rad_s']*TIMESTEP,CONTROLLER['joint_rate_rad_s']*TIMESTEP)
        data.ctrl[6]+=float(np.clip(targets[6]-data.ctrl[6],-CONTROLLER['gripper_rate_m_s']*TIMESTEP,CONTROLLER['gripper_rate_m_s']*TIMESTEP))
        data.xfrc_applied[:]=0
        if impulse is not None and index==0:data.xfrc_applied[shoe,:3]=np.asarray(impulse)/TIMESTEP
        mj.mj_step(model,data)
        data.xfrc_applied[:]=0
        mj.mj_forward(model,data)
        if not all(np.isfinite(x).all() for x in (data.qpos,data.qvel,data.qacc)) or any(w.number for w in data.warning):
            raise RuntimeError('Invalid workstation native state or solver warning')
        controller_state['native_steps']=native_steps+1
    return state(model,data,target_position_m)


def step(model,data,target_position_m,gripper_position_m,*,impulse_xyz=None):
    """Convenience wrapper for exactly one complete 20 ms control interval."""
    return run_control(model,data,{},target_position_m,gripper_position_m,10,impulse_xyz=impulse_xyz)


def protocol():
    """Deterministic initial state and authored tracking sequence for parity tests.

    Numeric tolerances below are proposed test criteria, not reported outcomes.
    This is target tracking; none of these targets constitutes a grasp policy.
    """
    return {'schema':'dvidia.live-workstation-protocol/v1','scope':SCOPE,
            'timestep':TIMESTEP,'control_dt':CONTROL_DT,'orientation_wxyz':list(ORIENTATION),
            'actions':[{'name':name,'target':list(target),'jaw':jaw,'control_steps':100}
                       for name,target,jaw in [('home',HOME_TARGET,.035),
                           ('above_shoe',(.32,-.145,.18),.035),
                           ('inspect_front',(.42,-.02,.20),.035),
                           ('inspect_side',(.32,.12,.18),.035),
                           ('close_clear',HOME_TARGET,0.),('open_clear',HOME_TARGET,.035)]],
            'acceptance':{'settled_position_error_m':.005,'settled_orientation_error_rad':math.radians(3),
                          'solver_warnings':0,'unintended_moving_arm_contacts':0,
                          'native_browser_position_tolerance_m':.001,
                          'scope':'Declared target values require measured native and browser results; no automatic qualification.'}}


def state(model,data,target_position_m=None):
    mj,np=sim._native_imports();tcp,shoe=_ids(model)
    contacts=[]
    for i in range(data.ncon):
        c=data.contact[i];force=np.zeros(6);mj.mj_contactForce(model,data,i,force)
        if force[0]>.001:contacts.append({'geom1':int(c.geom1),'geom2':int(c.geom2),'normal_force_n':float(force[0])})
    quaternion=np.empty(4);mj.mju_mat2Quat(quaternion,data.site_xmat[tcp])
    return {'time':float(data.time),'qpos':data.qpos.tolist(),'ctrl':data.ctrl.tolist(),
            'body_positions':data.xpos.tolist(),'body_quaternions':data.xquat.tolist(),
            'tcp_position':data.site_xpos[tcp].tolist(),'tcp_quaternion':quaternion.tolist(),
            'shoe_position':data.xpos[shoe].tolist(),'contacts':contacts,
            'position_error_m':None if target_position_m is None else float(np.linalg.norm(data.site_xpos[tcp]-target_position_m)),
            'observation_origin':'privileged native simulator state'}


def metadata(model,data):
    mj,_=sim._native_imports();tcp,shoe=_ids(model)
    joints=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'joint'+str(i)) for i in range(1,7)]
    return {'schema':SCHEMA,'scope':SCOPE,'physical_robot_ready':False,'learned_policy':False,
            'robot':'piper','timestep':TIMESTEP,'control_dt':CONTROL_DT,'scene_url':'scene.xml',
            'model_url':'model.json','geometry_url':'geometry.bin.gz','tcp_site_name':TCP,'tcp_site_id':tcp,
            'joint_names':['joint'+str(i) for i in range(1,7)],'joint_ids':joints,
            'arm_qpos_addresses':[int(model.jnt_qposadr[j]) for j in joints],
            'arm_dof_addresses':[int(model.jnt_dofadr[j]) for j in joints],
            'actuator_names':['joint'+str(i) for i in range(1,7)],'actuator_ids':list(range(6)),
            'gripper_actuator_names':['gripper'],'gripper_actuator_ids':[6],
            'actuator_ctrlrange':model.actuator_ctrlrange.tolist(),
            'home_qpos':data.qpos.tolist(),'home_ctrl':data.ctrl.tolist(),
            'initial_target_xyz':list(HOME_TARGET),'bounds_xyz':[list(row) for row in BOUNDS],
            'orientation_wxyz':list(ORIENTATION),'controller':dict(CONTROLLER),
            'shoe_body_names':[SHOE],'shoe_body_ids':[shoe],'shoe_asset_ids':['trainer-left'],
            'native_version':mj.__version__,'compiled_model_sha256':sim.model_digest(model),
            'camera':{'target':[.24,0,.10],'position':[1.02,-1.45,1.02]},
            'environment_geoms':{'table':'workstation_table_top','rack':'workstation_rack_shelf','floor':'workstation_floor'},
            'interaction_limits':{'push_impulse_norm_max_ns':.12, 'push_duration_s':TIMESTEP,
                'grip_scope':'Jaw motion and native contacts are available; no validated retained shoe lift or autonomous placement.'}}


def export(assets,output,*,runtime_directory=None):
    """Export native XML/verified meshes plus renderer geometry and metadata."""
    destination=Path(output)
    if destination.exists() or destination.is_symlink():raise ValueError('Workstation output must be fresh')
    item,blobs=robot_profiles._payload('piper',assets)
    spec=_spec(assets);native_xml=spec.to_xml().encode()
    mj,_=sim._native_imports()
    model=mj.MjModel.from_xml_string(native_xml.decode(),assets=spec.assets);data=initialize(model)
    manifest=metadata(model,data)
    manifest['provenance']=robot_profiles.verify_assets('piper',assets)
    scene,geometry=asset_preview._scene(model,{'robot':'piper','native_version':manifest['native_version'],
                    'compiled_model_sha256':manifest['compiled_model_sha256'],'scope':SCOPE})
    scene.update(schema='dvidia.live-workstation-model/v1',scope=SCOPE)
    packed=gzip.compress(geometry,mtime=0)
    scene['geometry_binary']={'href':'geometry.bin.gz','compression':'gzip','bytes':len(geometry),'compressed_bytes':len(packed),'sha256':sha256(geometry).hexdigest()}
    files={'scene.xml':native_xml,'model.json':_json(scene),'geometry.bin.gz':packed,
           'catalog.json':_json(asset_library.catalog()),'LICENSE':asset_preview.replay.MIT_LICENSE.encode(),
           'licenses/agilex_piper/LICENSE':blobs[item['folder']+'/LICENSE'],
           'licenses/agilex_piper/README.md':blobs[item['folder']+'/README.md']}
    for path,raw in blobs.items():
        local=path.removeprefix(item['folder']+'/')
        if local.startswith('assets/'):files[local]=raw
    manifest['assets']=[{'path':name,'bytes':len(raw),'sha256':sha256(raw).hexdigest()} for name,raw in sorted(files.items()) if name.startswith('assets/')]
    native_assets=b''.join(files[asset['path']] for asset in manifest['assets'])
    files['native-assets.bin.gz']=gzip.compress(native_assets,mtime=0)
    manifest['native_asset_bundle']={'path':'native-assets.bin.gz','compression':'gzip',
        'bytes':len(files['native-assets.bin.gz']),'sha256':sha256(files['native-assets.bin.gz']).hexdigest(),
        'uncompressed_bytes':len(native_assets)}
    manifest['native_scene']={'path':'scene.xml','bytes':len(files['scene.xml']),'sha256':sha256(files['scene.xml']).hexdigest()}
    runtime=Path(runtime_directory) if runtime_directory is not None else Path(__file__).parent
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError('Runtime directory must be a plain directory')
    for name,source in RUNTIME_FILES.items():
        files[name]=asset_preview.replay._read(runtime/source,MAX_FILE_BYTES)
    manifest['native_wasm']={'path':'mujoco.wasm','bytes':len(files['mujoco.wasm']),
                             'sha256':sha256(files['mujoco.wasm']).hexdigest()}
    files['mujoco.wasm.gz']=gzip.compress(files['mujoco.wasm'],mtime=0)
    manifest['wasm_bundle']={'path':'mujoco.wasm.gz','compression':'gzip',
        'bytes':len(files['mujoco.wasm.gz']),'sha256':sha256(files['mujoco.wasm.gz']).hexdigest(),
        'uncompressed_bytes':len(files['mujoco.wasm'])}
    files['README.md']=("# DVIDIA live shoe workstation v0.1\n\n"+SCOPE+"\n\n"
        "Serve this folder with `python -m http.server 8000` and open http://localhost:8000. "
        "It runs MuJoCo physics in a browser worker. JavaScript, WebAssembly, WebGL2 and "
        "DecompressionStream are required. No model download, remote inference or GPU training "
        "is needed; physics runs locally on the CPU and rendering uses WebGL. Browser performance "
        "depends on the machine. All assets and runtime files are included.\n\n"
        "For faster startup, native-assets.bin.gz delivers the original mesh bytes in one "
        "compressed request. Each unpacked mesh is checked against its original SHA-256. "
        "The raw assets remain included for independent reproduction.\n\n"
        "The physics engine also downloads compressed, with its decoded SHA-256 verified. "
        "Interrupted downloads make up to three attempts with a 30-second idle timeout and "
        "a 120-second total deadline per resource. Progress and a reload button expose recovery.\n\n"
        "Native PiPER meshes and MIT notices are pinned to the upstream commit in workstation.json. "
        "The rigid procedural shoe is MIT, with assumed untested physical properties in catalog.json. "
        "The controller is authored differential IK, not a learned policy. The jaw can make physical "
        "contacts; retained shoe lifting is not qualified. MuJoCo and Three.js notices are in "
        "workstation.third-party.txt. Integrity hashes check file consistency, not authenticity.\n").encode()
    files['workstation.json']=_json(manifest)
    seal={'schema':INTEGRITY_SCHEMA,'scope':'Artifact consistency, not authenticity or physical qualification',
         'files':{name:{'bytes':len(raw),'sha256':sha256(raw).hexdigest()} for name,raw in sorted(files.items())}}
    files['workstation-integrity.json']=_json(seal)
    if any(len(raw)>MAX_FILE_BYTES for raw in files.values()) or sum(map(len,files.values()))>MAX_TOTAL_BYTES:
        raise ValueError('Workstation exceeds bounded export size')
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.dvidia-workstation-',dir=destination.parent))
    try:
        for name,raw in files.items():
            path=staging/name;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(raw)
        inspected=inspect_export(staging)
        if destination.exists() or destination.is_symlink():
            raise ValueError('Workstation output must remain fresh')
        staging.rename(destination)
    finally:
        if staging.exists():shutil.rmtree(staging)
    return {**inspected,'initial':state(model,data,HOME_TARGET)}


def _unpack_native_assets(bundle,assets,payload):
    """Validate optional delivery compression without changing native inputs."""
    expected_keys={'path','compression','bytes','sha256','uncompressed_bytes'}
    total=sum(asset['bytes'] for asset in assets)
    if (type(bundle) is not dict or set(bundle)!=expected_keys
            or bundle['path']!='native-assets.bin.gz' or bundle['compression']!='gzip'
            or type(bundle['bytes']) is not int or not 0<=bundle['bytes']<=MAX_FILE_BYTES
            or type(bundle['sha256']) is not str or re.fullmatch('[a-f0-9]{64}',bundle['sha256']) is None
            or type(bundle['uncompressed_bytes']) is not int or bundle['uncompressed_bytes']!=total
            or not 0<=total<=MAX_TOTAL_BYTES):
        raise ValueError('Workstation native asset bundle descriptor mismatch')
    packed=payload.get(bundle['path'])
    if packed is None or len(packed)!=bundle['bytes'] or sha256(packed).hexdigest()!=bundle['sha256']:
        raise ValueError('Workstation native asset bundle checksum mismatch')
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(packed),mode='rb') as stream:
            unpacked=stream.read(total+1)
    except (OSError,EOFError,zlib.error) as exc:
        raise ValueError('Invalid compressed native asset bundle') from exc
    if len(unpacked)!=total:raise ValueError('Workstation native asset bundle size mismatch')
    offset=0
    for asset in assets:
        raw=unpacked[offset:offset+asset['bytes']];offset+=asset['bytes']
        if raw!=payload[asset['path']] or sha256(raw).hexdigest()!=asset['sha256']:
            raise ValueError('Workstation bundled mesh differs from its pinned original')
    return unpacked


def _unpack_wasm(manifest,payload):
    """Require compressed delivery to reproduce the original native engine exactly."""
    native=manifest.get('native_wasm')
    raw=payload['mujoco.wasm']
    if native!={'path':'mujoco.wasm','bytes':len(raw),'sha256':sha256(raw).hexdigest()}:
        raise ValueError('Workstation native WASM descriptor mismatch')
    bundle=manifest.get('wasm_bundle')
    packed=payload.get('mujoco.wasm.gz')
    if (packed is None or bundle!={'path':'mujoco.wasm.gz','compression':'gzip',
            'bytes':len(packed),'sha256':sha256(packed).hexdigest(),'uncompressed_bytes':len(raw)}):
        raise ValueError('Workstation compressed WASM descriptor mismatch')
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(packed),mode='rb') as stream:
            unpacked=stream.read(len(raw)+1)
    except (OSError,EOFError,zlib.error) as exc:
        raise ValueError('Invalid compressed WASM') from exc
    if unpacked!=raw:raise ValueError('Workstation compressed WASM differs from the original engine')
    return unpacked


def inspect_export(directory):
    """Verify a bounded, self-contained native scene and its renderer geometry.

    Inspection reads data, compiles only the contained XML with the fixed pinned
    mesh allowlist, and never executes the packaged JavaScript. The manifest seal
    proves consistency, not the identity or trustworthiness of its publisher.
    """
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():raise ValueError('Workstation must be a plain directory')
    read,decode=asset_preview.replay._read,asset_preview.replay._decode
    seal=decode(read(root/'workstation-integrity.json',128*1024))
    if type(seal) is not dict or seal.get('schema')!=INTEGRITY_SCHEMA or type(seal.get('files')) is not dict:
        raise ValueError('Workstation integrity schema mismatch')
    records=seal['files']
    if not 10<=len(records)<=128:raise ValueError('Workstation inventory outside bounds')
    payload={};total=0
    for name,record in records.items():
        if (type(name) is not str or re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*',name) is None
                or str(PurePosixPath(name))!=name or any(part.startswith('.') for part in PurePosixPath(name).parts)):
            raise ValueError('Workstation file path is not contained')
        path=root/name
        if any(parent.is_symlink() for parent in path.parents if parent==root or root in parent.parents):
            raise ValueError('Workstation path contains a symlink')
        if (type(record) is not dict or set(record)!={'bytes','sha256'} or type(record['bytes']) is not int
                or not 0<=record['bytes']<=MAX_FILE_BYTES or type(record['sha256']) is not str
                or re.fullmatch('[a-f0-9]{64}',record['sha256']) is None):
            raise ValueError('Workstation file record mismatch')
        raw=read(path,MAX_FILE_BYTES);total+=len(raw)
        if len(raw)!=record['bytes'] or sha256(raw).hexdigest()!=record['sha256'] or total>MAX_TOTAL_BYTES:
            raise ValueError('Workstation file integrity mismatch')
        payload[name]=raw
    paths=list(root.rglob('*'))
    if any(path.is_symlink() for path in paths):raise ValueError('Workstation contains a symlink')
    actual={path.relative_to(root).as_posix() for path in paths if path.is_file()}
    if actual!=set(payload)|{'workstation-integrity.json'}:raise ValueError('Workstation file inventory mismatch')
    profile=robot_profiles._profile('piper')
    asset_names=set();notice_names=set()
    for record in profile['files']:
        original=record['path'];local=original.removeprefix(profile['folder']+'/')
        if local.startswith('assets/'):
            name=local;asset_names.add(name)
        elif local in ('LICENSE','README.md'):
            name='licenses/'+original;notice_names.add(name)
        else:continue
        raw=payload.get(name)
        if raw is None or len(raw)!=record['bytes'] or sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('Workstation differs from pinned upstream mesh or notice: '+name)
    expected=set(RUNTIME_FILES)|asset_names|notice_names|{'scene.xml','model.json','geometry.bin.gz','catalog.json','LICENSE','README.md','workstation.json'}
    manifest=decode(payload['workstation.json'])
    if type(manifest) is not dict or manifest.get('schema')!=SCHEMA:raise ValueError('Workstation schema mismatch')
    if 'native_asset_bundle' in manifest:expected.add('native-assets.bin.gz')
    if 'wasm_bundle' in manifest:expected.add('mujoco.wasm.gz')
    if set(payload)!=expected:raise ValueError('Workstation portable file allowlist mismatch')
    if payload['LICENSE']!=asset_preview.replay.MIT_LICENSE.encode():raise ValueError('Workstation MIT notice mismatch')
    if not payload['mujoco.wasm'].startswith(b'\0asm\x01\0\0\0'):raise ValueError('Invalid WebAssembly header')
    scene=decode(payload['model.json'])
    if decode(payload['catalog.json'])!=asset_library.catalog():raise ValueError('Workstation catalog differs from the immutable catalog')
    expected_assets=[{'path':name,**records[name]} for name in sorted(asset_names)]
    if manifest.get('assets')!=expected_assets or manifest.get('native_scene')!={'path':'scene.xml',**records['scene.xml']}:
        raise ValueError('Workstation native asset manifest mismatch')
    if 'native_asset_bundle' in manifest:
        _unpack_native_assets(manifest['native_asset_bundle'],expected_assets,payload)
    if 'wasm_bundle' in manifest or 'native_wasm' in manifest:
        _unpack_wasm(manifest,payload)
    lock=robot_profiles._lock()
    provenance={'profile':'piper','repository':lock['repository'],'commit':lock['commit'],
                'files_verified':len(profile['files']),'bytes_verified':sum(item['bytes'] for item in profile['files']),
                'lock_sha256':sha256(robot_profiles.LOCK_PATH.read_bytes()).hexdigest(),
                'simulation_asset_license':profile['simulation_asset_license']}
    if manifest.get('provenance')!=provenance:raise ValueError('Workstation source provenance mismatch')
    native=payload['scene.xml']
    if b'<!' in native:raise ValueError('Workstation XML declarations are unsupported')
    try:xml=ET.fromstring(native)
    except ET.ParseError as exc:raise ValueError('Invalid workstation XML') from exc
    if xml.tag!='mujoco' or len(list(xml.iter()))>4096:raise ValueError('Invalid workstation XML root or element count')
    for element in xml.iter():
        if element.tag in ('include','plugin','extension'):
            raise ValueError('Workstation external XML or plugins are unsupported')
        if 'file' in element.attrib:
            name=element.attrib['file']
            if element.tag!='mesh' or '/' in name or '\\' in name or 'assets/'+name not in asset_names:
                raise ValueError('Workstation XML file reference is not a pinned local mesh')
    compilers=xml.findall('compiler')
    if len(compilers)!=1 or compilers[0].get('meshdir')!='assets/' or compilers[0].get('assetdir') or compilers[0].get('texturedir'):
        raise ValueError('Workstation XML asset directory mismatch')
    mj,np=sim._native_imports()
    if manifest.get('native_version')!=mj.__version__:raise ValueError('Inspection requires the recorded native MuJoCo version')
    warnings=[]
    with sim.native_warnings(warnings):
        model=mj.MjModel.from_xml_string(native.decode(),assets={name:payload[name] for name in asset_names})
        if (model.nq,model.nv,model.nu,model.neq,model.nmocap)!=(15,14,7,1,0):
            raise ValueError('Workstation native mechanical contract mismatch')
        data=initialize(model)
    if warnings or any(item.number for item in data.warning):raise ValueError('Workstation native load emitted a warning')
    native_meta=metadata(model,data)
    expected_metadata=set(native_meta)|{'assets','native_scene','provenance'}
    if 'native_asset_bundle' in manifest:expected_metadata.add('native_asset_bundle')
    if 'wasm_bundle' in manifest:expected_metadata.update(('native_wasm','wasm_bundle'))
    if set(manifest)!=expected_metadata:
        raise ValueError('Workstation metadata inventory mismatch')
    if any(manifest.get(key)!=value for key,value in native_meta.items()):
        raise ValueError('Workstation metadata differs from the compiled native scene')
    source={'robot':'piper','native_version':manifest['native_version'],
            'compiled_model_sha256':manifest['compiled_model_sha256'],'scope':SCOPE}
    expected_scene,geometry=asset_preview._scene(model,source)
    expected_scene.update(schema='dvidia.live-workstation-model/v1',scope=SCOPE)
    unpacked=asset_preview._unpack(scene.get('geometry_binary'),payload,'geometry.bin.gz')
    if unpacked!=geometry or {key:value for key,value in scene.items() if key!='geometry_binary'}!=expected_scene:
        raise ValueError('Workstation renderer differs from its compiled native geometry')
    return {'schema':SCHEMA,'robot':'piper','files':records,'file_count':len(payload)+1,
            'bytes':total+len(read(root/'workstation-integrity.json',128*1024)),
            'scope':SCOPE,'physical_robot_ready':False,'learned_policy':False,
            'compiled_model_sha256':manifest['compiled_model_sha256']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-directory',type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(export(args.assets,args.output,runtime_directory=args.runtime_directory),indent=2))


if __name__=='__main__':main()
