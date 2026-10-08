"""Installed-arm runtime proof under macOS process-level network denial."""
import ctypes
import errno
import json
from pathlib import Path
import platform
import socket
import struct
import sys
from .arm_runner import authored_grounding, run_skill, write_run
from .arm_qualify import SCENES


def main():
    if platform.system()!='Darwin':
        raise RuntimeError('This verification harness is macOS-specific.')
    libc=ctypes.CDLL(None,use_errno=True)
    libc.socket.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int]
    libc.connect.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32]
    libc.close.argtypes=[ctypes.c_int]
    descriptor=libc.socket(socket.AF_INET,socket.SOCK_STREAM,0)
    observed_errno=ctypes.get_errno() if descriptor<0 else 0
    if descriptor>=0:
        address=ctypes.create_string_buffer(struct.pack('BBH4s8s',16,socket.AF_INET,socket.htons(9),socket.inet_aton('127.0.0.1'),bytes(8)))
        result=libc.connect(descriptor,address,16)
        observed_errno=ctypes.get_errno() if result<0 else 0
        libc.close(descriptor)
    if observed_errno not in (errno.EPERM,errno.EACCES):
        raise RuntimeError('Native network self-test was not denied. Launch under the documented sandbox policy.')
    source=Path(__file__).with_name('place_cup.skill.json').read_bytes()
    grounding=authored_grounding(source)
    episodes=[]
    payload=None
    for index,(name,start,target) in enumerate(SCENES[:3]):
        scene={'object_position':start,'target_position':target,'seed':500+index}
        payload=run_skill(source,grounding,scene,offline=True)
        payload['episodes'][0]['scene_id']=name
        episodes.extend(payload['episodes'])
    if not all(e['success'] and not e['final_info']['warnings'] for e in episodes):
        raise RuntimeError('Offline arm execution failed.')
    payload['episodes']=episodes
    payload['summary']={'successes':sum(e['success'] for e in episodes),'episodes':len(episodes),
                        'simulated_seconds':sum(e['simulated_seconds'] for e in episodes),
                        'episode_wall_seconds':sum(e['episode_wall_seconds'] for e in episodes)}
    payload['native_network_policy']={'platform':'macOS sandbox-exec',
        'policy':'(version 1) (allow default) (deny network*)',
        'native_connect_self_test':'libc connect to 127.0.0.1:9 denied before rollouts',
        'errno':observed_errno,'passed':True,
        'scope':'Installed runtime only. Repeats three qualified scenes under process-level denial; no air-gapped dependency installation or other OS qualification.'}
    output=Path(sys.argv[1]) if len(sys.argv)>1 else Path('runs/arm-native-offline')
    write_run(output,payload,source=source,grounding=grounding)
    (output/'native-policy.json').write_text(json.dumps(payload['native_network_policy'],indent=2)+'\n')


if __name__=='__main__':
    main()
