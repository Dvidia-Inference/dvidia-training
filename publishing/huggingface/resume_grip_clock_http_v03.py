"""Run the unchanged reviewed publisher with bounded anonymous HTTP readback.

The original publisher retains credential loading, membership checks, full input
validation, parent guards, commit journals and every byte/Git/LFS comparison.
This driver supplies only its supported downloader argument and records the
transport and previously verified commit prefix. It never constructs API calls.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from grip_clock_prefetch_http_v03 import ReadAhead
import publish_grip_clock_v03 as pub
from publish import PublicationError


def run(plan_path, *, record_name='http-transport-attempt.json'):
    path = pub.plain(plan_path)
    if record_name not in {'http-transport-attempt.json', 'http-transport-attempt-02.json',
                           'http-transport-attempt-03.json', 'http-transport-attempt-04.json'}:
        raise PublicationError('Use a declared fresh transport-attempt record name.')
    record_path = path.parent/record_name
    if record_path.exists():
        raise PublicationError('Transport-attempt records must be fresh.')
    plan_raw = pub.read(path, pub.MAX_PLAN)
    journal_path = path.parent/pub.JOURNAL
    journal_raw = pub.read(journal_path, pub.MAX_PLAN) if journal_path.exists() else None
    prefix = {}
    if journal_raw is not None:
        journal = pub.decoded(journal_raw)
        if journal.get('plan_sha256') != sha256(plan_raw).hexdigest():
            raise PublicationError('Existing journal does not bind this exact plan.')
        prefix = {kind: len(value['commits']) for kind, value in journal['repositories'].items()}
    implementation = {name: pub.receipt(pub.read(Path(__file__).with_name(name))) for name in (
        'resume_grip_clock_http_v03.py', 'grip_clock_public_http.py',
        'grip_clock_prefetch_http_v03.py', 'publish_grip_clock_v03.py')}
    proof = dict(schema='dvidia.grip-clock-public-http-transport/v0.3',
        plan=pub.receipt(plan_raw), prior_journal=pub.receipt(journal_raw) if journal_raw is not None else None,
        previously_verified_commit_prefix=prefix, implementation=implementation,
        transport='Anonymous bounded HTTPS GET of exact fixed-repository resolve revisions.',
        publisher_unchanged=True, original_credential_and_parent_guards=True,
        original_four_verification_threads=True,
        anonymous_get_workers=12, prefetch_window_files=16,
        fresh_final_readback=True, all_expected_byte_comparisons=True,
        camera_or_physics_execution=False, physical_robot_ready=False)
    pub.atomic(record_path, proof, exclusive=True)
    # Do not pass api= or operation_cls=: the original authenticated publisher
    # owns all writes and its in-memory exact-credential checks remain enabled.
    with ReadAhead(path, workers=12, window=16) as reader:
        result = pub.publish(path, downloader=reader.download)
        transport_scopes = reader.scope_receipts
        peak_futures = reader.peak_futures
    receipt_raw = pub.read(path.parent/pub.RECEIPT, pub.MAX_PLAN)
    if pub.decoded(receipt_raw) != result:
        raise PublicationError('Completed publication receipt differs from returned result.')
    complete = dict(proof, publication_receipt=pub.receipt(receipt_raw),
                    transport_scopes=transport_scopes, peak_futures=peak_futures, complete=True)
    pub.atomic(record_path.with_name(record_path.stem+'-complete.json'), complete, exclusive=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--record-name', default='http-transport-attempt.json')
    args = parser.parse_args(argv)
    try:
        value = run(args.plan, record_name=args.record_name)
        print(json.dumps(dict(schema=value['schema'], complete=True)), flush=True)
        return 0
    except Exception as error:
        message = str(error) if isinstance(error, PublicationError) else type(error).__name__
        print('HTTP publication stopped: '+message, file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
