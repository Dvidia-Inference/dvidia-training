"""Run the immutable original probe audit with assertions explicitly enabled.

The original recorded audit uses assertions. This wrapper refuses optimized
Python before importing or executing it, then pins its exact reviewed bytes.
"""
if not __debug__:
    raise RuntimeError('Probe audit refuses optimized Python; run without -O/-OO.')

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

ORIGINAL_SHA256 = '491ac28f8e9026eb0673b15356dbc7fa164402e873a5da3c5e2e9303fd3733d7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-script', type=Path, required=True)
    parser.add_argument('--guard-receipt', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if sys.flags.optimize or args.audit_script.is_symlink() or not args.audit_script.is_file():
        raise RuntimeError('Expected normal Python and a plain original audit file.')
    raw = args.audit_script.read_bytes()
    if sha256(raw).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError('Original audit source identity changed.')
    if args.guard_receipt.exists() or args.guard_receipt.is_symlink():
        raise RuntimeError('Guard receipt must be new.')
    output = Path(remaining[remaining.index('--output')+1])
    prior = sys.argv
    try:
        sys.argv = [str(args.audit_script), *remaining]
        runpy.run_path(str(args.audit_script), run_name='__main__')
    finally:
        sys.argv = prior
    result = output.read_bytes()
    receipt = {'schema': 'dvidia.probe-audit-normal-mode/v1', 'optimization_level': sys.flags.optimize,
               'assertions_enabled': __debug__, 'original_audit_sha256': ORIGINAL_SHA256,
               'guard_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
               'readback_output': {'bytes': len(result), 'sha256': sha256(result).hexdigest()},
               'scope': 'A guarded readback of the same frozen records; no native physics rerun.'}
    with args.guard_receipt.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
