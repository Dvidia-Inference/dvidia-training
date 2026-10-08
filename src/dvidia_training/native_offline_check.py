"""macOS-only system network-denial check; run under sandbox-exec, never bare."""
import ctypes
import errno
import json
from pathlib import Path
import platform
import socket
import struct
import sys
from .cli import NetworkGuard, payload_for, rollout, write_run
from .env import Config
from .pack import encode
from .policies import FeedbackPolicy


def main():
    if platform.system() != 'Darwin':
        raise RuntimeError('This verification harness is macOS-specific.')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.socket.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int]
    libc.connect.argtypes = [ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32]
    libc.close.argtypes = [ctypes.c_int]
    descriptor = libc.socket(socket.AF_INET,socket.SOCK_STREAM,0)
    observed_errno = ctypes.get_errno() if descriptor < 0 else 0
    if descriptor >= 0:
        # BSD sockaddr_in: length, address family, network-order port, loopback.
        address = ctypes.create_string_buffer(struct.pack('BBH4s8s',16,socket.AF_INET,socket.htons(9),socket.inet_aton('127.0.0.1'),bytes(8)))
        result = libc.connect(descriptor,address,16)
        observed_errno = ctypes.get_errno() if result < 0 else 0
        libc.close(descriptor)
    if observed_errno not in (errno.EPERM,errno.EACCES):
        raise RuntimeError(f'Native policy self-test was not denied: errno={observed_errno}. Launch under the documented sandbox policy.')
    config = Config()
    parameters = FeedbackPolicy().parameters()
    with NetworkGuard(True) as guard:
        episodes = [rollout(config,seed,parameters) for seed in [300,301,302]]
        payload = payload_for(config,episodes,parameters,guard)
        if guard.attempts or not all(e['success'] for e in episodes):
            raise RuntimeError('Offline runtime verification failed.')
    payload['native_network_policy'] = {
        'platform':'macOS sandbox-exec', 'policy':'(version 1) (allow default) (deny network*)',
        'native_connect_self_test':'127.0.0.1:9 via libc, denied before rollouts',
        'errno':observed_errno, 'passed':True,
        'scope':'Installed runtime and child process network denied; does not exercise air-gapped installation or another operating system.'}
    output = Path(sys.argv[1]) if len(sys.argv)>1 else Path('runs/native-offline')
    write_run(output,payload,parameters)
    (output/'native-policy.json').write_bytes(encode(payload['native_network_policy']))
    print(json.dumps(payload['native_network_policy']))


if __name__ == '__main__':
    main()
