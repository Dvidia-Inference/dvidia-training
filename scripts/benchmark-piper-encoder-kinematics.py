"""Bounded static geometry timing, with no learned model or dynamics execution."""
import argparse, inspect, json, os, platform, sys, time
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

try:
    import resource
except ImportError:
    resource = None

def peak_rss_bytes():
    if resource is None or sys.platform not in ('darwin', 'linux'):
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--assets', required=True, help='Existing pinned PiPER asset root')
parser.add_argument('--output', required=True, help='Fresh JSON receipt path')
args = parser.parse_args()
out = Path(args.output)
if out.exists():
    raise FileExistsError('Timing output must be fresh.')
import numpy as np
from dvidia_training.piper_encoder_kinematics import PiperEncoderKinematics, ARM_LIMITS_RAD
from dvidia_training import reference_sim
source = Path(inspect.getfile(PiperEncoderKinematics))
expected = "6606bb8aad04e5abb1e9b6d1a8ec628a4114acd167862167184fd4eef1d21c34"
if sha256(source.read_bytes()).hexdigest() != expected:
    raise ValueError("This v0.1 probe requires the frozen encoder-kinematics source.")
mj, _ = reference_sim._native_imports()
with ExitStack() as guard:
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_collision','mj_fwdPosition'):
        guard.enter_context(patch.object(mj,name,side_effect=AssertionError('No dynamics in static timing.')))
    start = time.perf_counter_ns()
    robot = PiperEncoderKinematics(args.assets)
    constructor_ms = (time.perf_counter_ns()-start)/1e6
    lower, upper = np.array(ARM_LIMITS_RAD).T
    qs = np.random.default_rng(20261109).uniform(lower+.1, upper-.1, (64,6))
    targets = []
    for q in qs:
        pose = robot.fk(q)
        targets.append((np.array(pose['tcp_position_world_m'])+[.005,-.003,.004],pose['tcp_quaternion_world_wxyz']))
    for q,(position,quaternion) in zip(qs,targets):
        robot.fk(q); robot.pose_action(q,position,quaternion,.02)
    samples = {'fk': [],'pose_action': []}
    for i in range(1024):
        q=qs[i%64]; position,quaternion=targets[i%64]
        start=time.perf_counter_ns();robot.fk(q);samples['fk'].append((time.perf_counter_ns()-start)/1e6)
        start=time.perf_counter_ns();action=robot.pose_action(q,position,quaternion,.02);samples['pose_action'].append((time.perf_counter_ns()-start)/1e6)
        assert len(action['joint_targets_rad'])==6 and np.isfinite(action['joint_targets_rad']).all()
    assert robot._data.time == 0.
receipt = {'schema':'dvidia.piper-encoder-kinematics-static-timing/v1','source_sha256':expected,
    'probe_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Shared-host static CPU API timing only. Targets were prepared before timing. No dynamics, rendering, learned inference, training, IO loop or hardware commands; no real-time deadline claim.',
    'configurations':64,'warmup_calls_per_api':64,'timed_calls_per_api':1024,'seed':20261109,
    'constructor_wall_ms_including_asset_validation_and_compile':constructor_ms,
    'api_wall_ms':{name:{'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'max':float(max(values))} for name,values in samples.items()},
    'host':{'machine':platform.machine(),'platform':sys.platform,'macos':platform.mac_ver()[0],'python':platform.python_version(),'numpy':np.__version__,'mujoco':mj.__version__,'logical_cpus':os.cpu_count(),'declared_blas_threads':{key:os.getenv(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},'whole_process_peak_rss_bytes':peak_rss_bytes()},
    'native_time_seconds':0.,'physical_robot_ready':False,'task_qualified':False}
assert sha256(source.read_bytes()).hexdigest()==expected
with out.open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(receipt,sort_keys=True))
