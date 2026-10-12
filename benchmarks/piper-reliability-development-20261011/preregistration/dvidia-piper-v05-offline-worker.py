import errno,json,socket,sys
try:
 s=socket.socket();s.connect(('127.0.0.1',9))
except OSError as e:
 if e.errno!=errno.EPERM:raise RuntimeError('OS network-denial self-check did not report EPERM') from e
 print(json.dumps({'network_denial_self_check':'EPERM','scope':'OS sandbox prevents socket creation or connect'}),file=sys.stderr,flush=True)
else:raise RuntimeError('OS network-denial self-check unexpectedly connected')
from dvidia_training.piper_reliability_benchmark import main
main()
