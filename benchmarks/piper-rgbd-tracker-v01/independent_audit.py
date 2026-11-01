"""Read-only audit of saved RGB-D confirmation evidence (stdlib + NumPy).

No simulator, tracker or benchmark implementation is imported or executed.
Truth is used only to audit sealed estimates and rendered fixture geometry.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np


VARIANTS = ('clear', 'partial_occlusion', 'tilt15', 'tilt30', 'absent',
            'occluded', 'ambiguous', 'invalid_depth', 'stale', 'future')
SIGNALS = ('object_position_estimate_m', 'object_quaternion_estimate_wxyz',
           'object_position_uncertainty_m', 'object_orientation_uncertainty_rad')
INPUTS = ('rgb.npy', 'depth.npy', 'input.json')
SCOPE = ('Independent readback of static synthetic known-cube perception only. '
         'No simulation, estimator, action learning or robot qualification was run. '
         'Depth accuracy and heuristic uncertainty scales remain uncalibrated.')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class Audit:
    def __init__(self, study):
        self.study = Path(study); self.errors = []; self.identities = {}

    def check(self, condition, label):
        if not condition:
            self.errors.append(label)
        return bool(condition)

    def identity(self, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing/nonregular evidence file: '+path.name)
        body = path.read_bytes()
        result = dict(bytes=len(body), sha256=sha256(body).hexdigest())
        try: name = str(path.relative_to(self.study))
        except ValueError: name = 'external-source/'+path.name
        self.identities[name] = result
        return result

    def load(self, path):
        self.identity(path)
        def bad(value): raise ValueError('Nonfinite JSON evidence')
        return json.loads(Path(path).read_text(), parse_constant=bad)

    def equal(self, actual, expected, label):
        if isinstance(expected, bool) or expected is None or isinstance(expected, str):
            return self.check(type(actual) is type(expected) and actual == expected, label)
        if isinstance(expected, int):
            return self.check(type(actual) is int and actual == expected, label)
        if isinstance(expected, float):
            return self.check(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-9), label)
        if isinstance(expected, list):
            if not self.check(isinstance(actual, list) and len(actual) == len(expected), label): return False
            return all([self.equal(a, b, label+f'[{i}]') for i, (a, b) in enumerate(zip(actual, expected))])
        if isinstance(expected, dict):
            if not self.check(isinstance(actual, dict) and set(actual) == set(expected), label): return False
            return all([self.equal(actual[k], value, label+'.'+k) for k, value in expected.items()])
        raise TypeError('Unsupported audit comparison')


def percentile(rows, key, percent):
    values = [r[key] for r in rows if r[key] is not None]
    # Explicit linear interpolation rather than calling the runner's reducer.
    if not values: return None
    values.sort(); location = (len(values)-1)*percent/100
    lower = int(math.floor(location)); upper = int(math.ceil(location))
    return float(values[lower]+(values[upper]-values[lower])*(location-lower))


def _signal_contract(audit, result, timing, label):
    audit.check(isinstance(result, dict), label+'.result_mapping')
    audit.check(result.get('status') in ('available', 'missing', 'stale', 'refused'), label+'.status')
    for key, value in timing.items(): audit.equal(result.get(key), value, label+'.'+key)
    causal = timing['captured_at_seconds'] <= timing['available_at_seconds'] <= timing['decision_at_seconds']
    age = timing['decision_at_seconds']-timing['captured_at_seconds'] if causal else None
    audit.equal(result.get('age_seconds'), age, label+'.age')
    audit.check(result.get('base_frame') == 'robot_base', label+'.base_frame')
    audit.check(result.get('depth_accuracy_verified') is False, label+'.depth_accuracy_flag')
    signals = result.get('signals', {})
    audit.check(set(signals) == set(SIGNALS), label+'.signal_names')
    if result['status'] == 'available':
        audit.check(causal and age <= .1+1e-12, label+'.fresh_causal_available')
        for name, size in zip(SIGNALS[:2], (3, 4)):
            value = result.get(name)
            if not (isinstance(value, list) and len(value) == size
                    and all(type(v) in (int, float) and math.isfinite(v) for v in value)):
                raise ValueError('Malformed available pose')
        quaternion = result[SIGNALS[1]]
        audit.check(math.isclose(sum(x*x for x in quaternion), 1., abs_tol=1e-6, rel_tol=0), label+'.quaternion_norm')
        audit.check(abs(quaternion[1]) <= 1e-8 and abs(quaternion[2]) <= 1e-8, label+'.upright_quaternion')
        for name in SIGNALS[2:]:
            value = result.get(name)
            if not (type(value) in (int, float) and math.isfinite(value) and value >= 0):
                raise ValueError('Malformed available uncertainty')
        for name in SIGNALS:
            value = result[name] if isinstance(result[name], list) else [result[name]]
            audit.equal(signals.get(name), dict(status='available', value=value,
                captured_at_seconds=timing['captured_at_seconds'], available_at_seconds=timing['available_at_seconds'],
                age_seconds=age), label+'.signals.'+name)
    else:
        for name in SIGNALS:
            audit.check(result.get(name) is None, label+'.abstain.'+name)
            audit.equal(signals.get(name), dict(status='missing', value=None,
                captured_at_seconds=None, available_at_seconds=None, age_seconds=None), label+'.signals.'+name)


def _occluder_check(audit, folder, group_id, width):
    """Independently project the cube centre and inspect saved depth, not RGB labels."""
    clear = folder/f'{group_id}-{width}-clear'; hidden = folder/f'{group_id}-{width}-occluded'
    calibration = audit.load(clear/'input.json')['calibration']
    other = audit.load(hidden/'input.json')['calibration']
    audit.equal(other, calibration, group_id+f'-{width}.occluder_calibration')
    truth = audit.load(clear/'truth.json'); hidden_truth = audit.load(hidden/'truth.json')
    audit.equal(hidden_truth['position_m'], truth['position_m'], group_id+'.occluder_same_object')
    transform = np.asarray(calibration['base_from_camera'], dtype=float).reshape(4, 4)
    optical = transform[:3, :3].T @ (np.asarray(truth['position_m'])-transform[:3, 3])
    intrinsics = calibration['intrinsics']; height = intrinsics['height']
    if optical[2] <= .1: raise ValueError('Occluder fixture projects behind/inside camera')
    column = int(math.floor(intrinsics['fx']*optical[0]/optical[2]+intrinsics['cx']))
    row = int(math.floor(intrinsics['fy']*optical[1]/optical[2]+intrinsics['cy']))
    if not 0 <= column < width or not 0 <= row < height: raise ValueError('Object centre lies outside saved frame')
    original = np.load(clear/'depth.npy', allow_pickle=False)
    covered = np.load(hidden/'depth.npy', allow_pickle=False)
    if original.shape != (height, width) or covered.shape != (height, width): raise ValueError('Occluder depth alignment failed')
    before = float(original[row, column]); after = float(covered[row, column])
    if not math.isfinite(before) or not math.isfinite(after):
        raise ValueError('Nonfinite occluder comparison depth')
    # Fixture card centre is 60mm toward the camera, with 2mm half-thickness.
    # The front face is constant optical Z, even away from the pixel principal ray.
    expected = float(optical[2]-.062)
    okay = math.isfinite(before) and math.isfinite(after) and abs(after-expected) <= .002 and before-after >= .015
    audit.check(okay, group_id+f'-{width}.occluder_depth_present')
    return dict(group_id=group_id, width=width, pixel=[column, row],
        clear_depth_m=before, occluded_depth_m=after, expected_card_depth_m=expected,
        card_depth_absolute_error_m=abs(after-expected), foreground_separation_m=before-after, passed=okay)


def audit_study(study, source_root=None):
    audit = Audit(study); folder = audit.study/'confirmation'; summaries = []; gates = []; occluders = []
    processed = 0
    try:
        protocol = audit.load(audit.study/'protocol.json'); frozen = audit.load(audit.study/'freeze.json')
        manifest = audit.load(folder/'manifest.json'); scheduled = audit.load(folder/'scheduled.json')
        predictions = audit.load(folder/'predictions.json'); report = audit.load(folder/'report.json')
        audit.equal(frozen['protocol_sha256'], audit.identity(audit.study/'protocol.json')['sha256'], 'freeze.protocol')
        for name, identity in frozen['development_evidence'].items():
            if Path(name).name != name: raise ValueError('Unsafe development evidence name')
            audit.equal(audit.identity(audit.study/'development'/name), identity, 'development.'+name)
        if source_root is not None:
            for name, digest in frozen['sources'].items():
                if Path(name).name != name: raise ValueError('Unsafe source filename')
                audit.equal(audit.identity(Path(source_root)/name)['sha256'], digest, 'source.'+name)
        groups = protocol['groups']; confirmation = [g for g in groups if g['split'] == 'confirmation']
        audit.check(len(groups) == 24 and len(confirmation) == 16 and len({g['id'] for g in groups}) == 24,
                    'exact_24_groups_16_confirmation')
        if any(not isinstance(g['id'], str) or not g['id'].isalnum() for g in groups): raise ValueError('Unsafe group id')
        audit.equal(protocol['resolutions'], [[160, 120], [320, 240]], 'resolutions')
        audit.equal(protocol['variants'], list(VARIANTS), 'variants')
        audit.equal(protocol['expected_available'], ['clear'], 'available_protocol')
        audit.equal(protocol['challenge_positive'], ['partial_occlusion'], 'partial_protocol')
        negatives = [v for v in VARIANTS if v not in ('clear', 'partial_occlusion')]
        audit.equal(protocol['expected_abstain'], negatives, 'negative_protocol')
        expected = []
        for width, height in ((160, 120), (320, 240)):
            for group in confirmation:
                for variant in VARIANTS:
                    expected.append(dict(id=f'{group["id"]}-{width}-{variant}', group_id=group['id'],
                        variant=variant, width=width, height=height, frame_index=len(expected)))
        audit.equal(scheduled['records'], expected, 'scheduled_records')
        audit.check(len(expected) == 320, 'exact_320_expected_ids')
        for document, name in ((manifest, 'manifest'), (predictions, 'predictions'), (scheduled, 'scheduled')):
            audit.equal(document['protocol_sha256'], frozen['protocol_sha256'], name+'.protocol')
            audit.equal(document['split'], 'confirmation', name+'.split')
        audit.equal(manifest['scheduled_sha256'], audit.identity(folder/'scheduled.json')['sha256'], 'manifest.schedule')
        audit.equal(predictions['manifest_sha256'], audit.identity(folder/'manifest.json')['sha256'], 'predictions.manifest')
        audit.equal(report['manifest_sha256'], predictions['manifest_sha256'], 'report.manifest')
        audit.equal(report['predictions_sha256'], audit.identity(folder/'predictions.json')['sha256'], 'report.predictions')
        audit.equal(predictions['freeze_sha256'], audit.identity(audit.study/'freeze.json')['sha256'], 'predictions.freeze')
        audit.check(manifest['source_stable'] is True, 'render_source_stable')
        for document, name in ((manifest, 'manifest'), (predictions, 'predictions')):
            audit.equal(document['source_hashes'], frozen['sources'], name+'.frozen_sources')
            audit.equal(document['runtime'], frozen['runtime'], name+'.frozen_runtime')
        audit.equal(manifest['renderer_bindings'], frozen['renderer_bindings'], 'renderer_bindings')
        items = manifest['records']; estimates = predictions['predictions']; published_rows = report['rows']
        ids = [row['id'] for row in expected]
        for rows, name in ((items, 'manifest'), (estimates, 'predictions'), (published_rows, 'report')):
            if [row['id'] for row in rows] != ids: raise ValueError(name+' missing/duplicate/unknown/reordered scheduled ids')
        computed = []
        for registered, item, pred, published in zip(expected, items, estimates, published_rows):
            label = item['id']; fixture = folder/label
            for name, value in registered.items(): audit.equal(item.get(name), value, label+'.registration.'+name)
            receipt = audit.load(fixture/'receipt.json')
            audit.equal(audit.identity(fixture/'receipt.json')['sha256'], item['receipt_sha256'], label+'.render_receipt_hash')
            audit.equal(receipt, {k: v for k, v in item.items() if k != 'receipt_sha256'}, label+'.render_receipt')
            for name, identity in item['files'].items():
                if Path(name).name != name or name in ('', '.', '..'): raise ValueError('Unsafe artifact name')
                audit.equal(audit.identity(fixture/name), identity, label+'.artifact.'+name)
            sealed = audit.load(folder/'estimates'/(label+'.json'))
            audit.equal(audit.identity(folder/'estimates'/(label+'.json'))['sha256'], pred['receipt_sha256'], label+'.estimate_receipt_hash')
            audit.equal(sealed, {k: v for k, v in pred.items() if k != 'receipt_sha256'}, label+'.estimate_receipt')
            status = pred['estimate_status']; available = False; error = None; yaw_error = None; pos_scale = None; angle_scale = None
            audit.check(status in ('complete', 'failed', 'unattempted'), label+'.estimate_status')
            if status == 'complete':
                audit.check(item['render_status'] == 'rendered', label+'.completed_fixture')
                audit.equal(pred['inputs'], {name: item['files'][name] for name in INPUTS}, label+'.input_ids')
                inputs = audit.load(fixture/'input.json'); result = pred['result']
                _signal_contract(audit, result, inputs['timing'], label)
                truth = audit.load(fixture/'truth.json'); available = result['status'] == 'available'
                if available and truth['present']:
                    error = math.sqrt(sum((a-b)**2 for a, b in zip(result[SIGNALS[0]], truth['position_m'])))
                    q = result[SIGNALS[1]]; angle = 2*math.atan2(q[3], q[0])
                    if item['variant'] not in ('tilt15', 'tilt30') and abs(truth['tilt_rad']) <= 1e-6:
                        difference = angle-truth['yaw_rad']
                        yaw_error = abs((difference+math.pi/4) % (math.pi/2)-math.pi/4)
                    pos_scale, angle_scale = result[SIGNALS[2]], result[SIGNALS[3]]
            else:
                audit.check(pred['result'] is None, label+'.failed_result_null')
            metrics = dict(available=available, position_error_m=error, yaw_error_rad=yaw_error,
                position_heuristic_scale_m=pos_scale, orientation_heuristic_scale_rad=angle_scale,
                position_error_within_heuristic_scale=error <= pos_scale if error is not None else None,
                yaw_error_within_heuristic_scale=yaw_error <= angle_scale if yaw_error is not None else None,
                wall_seconds=pred['wall_seconds'], cpu_seconds=pred['cpu_seconds'])
            for name, value in metrics.items(): audit.equal(published.get(name), value, label+'.report.'+name)
            for name in ('wall_seconds', 'cpu_seconds'):
                audit.check(metrics[name] is None or (type(metrics[name]) in (int, float) and math.isfinite(metrics[name]) and metrics[name] >= 0), label+'.'+name)
            computed.append(dict(**registered, **metrics, estimate_status=status)); processed += 1
        for width, _ in ((160, 120), (320, 240)):
            for variant in VARIANTS:
                rows = [r for r in computed if r['width'] == width and r['variant'] == variant]
                summary = dict(width=width, variant=variant, scheduled=len(rows), available=sum(r['available'] for r in rows),
                    pipeline_failures=sum(r['estimate_status'] != 'complete' for r in rows),
                    position_evaluated=sum(r['position_error_m'] is not None for r in rows),
                    yaw_evaluated=sum(r['yaw_error_rad'] is not None for r in rows),
                    position_p50_m=percentile(rows,'position_error_m',50), position_p95_m=percentile(rows,'position_error_m',95),
                    yaw_p95_rad=percentile(rows,'yaw_error_rad',95), wall_p50_s=percentile(rows,'wall_seconds',50),
                    wall_p95_s=percentile(rows,'wall_seconds',95), cpu_p95_s=percentile(rows,'cpu_seconds',95),
                    heuristic_comparisons=sum(r['position_error_m'] is not None for r in rows),
                    yaw_heuristic_comparisons=sum(r['yaw_error_rad'] is not None for r in rows),
                    position_error_within_heuristic_scale=sum(r['position_error_within_heuristic_scale'] is True for r in rows),
                    yaw_error_within_heuristic_scale=sum(r['yaw_error_within_heuristic_scale'] is True for r in rows))
                summaries.append(summary)
            rows = [r for r in computed if r['width'] == width]; clear = [r for r in rows if r['variant'] == 'clear']
            negative = [r for r in rows if r['variant'] in negatives]; limits = protocol['provisional_gate']
            coverage = sum(r['available'] for r in clear)/len(clear)
            position = percentile(clear, 'position_error_m', 95); yaw = percentile(clear, 'yaw_error_rad', 95)
            false = sum(r['available'] for r in negative)
            checks = dict(all_scheduled_fixtures_complete=not manifest['session_errors'] and all(r['estimate_status'] == 'complete' for r in rows),
                clear_pose_metrics_complete=all(r['position_error_m'] is not None and r['yaw_error_rad'] is not None for r in clear if r['available']),
                clear_coverage=coverage >= limits['clear_coverage_min'],
                clear_position=position is not None and position <= limits['clear_position_p95_m_max'],
                clear_yaw=yaw is not None and yaw <= limits['clear_yaw_p95_rad_max'], no_negative_false_available=false <= limits['negative_false_available_max'])
            gates.append(dict(width=width, passed=all(checks.values()), checks=checks, clear_scheduled=len(clear),
                clear_coverage=coverage, negative_scheduled=len(negative), negative_false_available=false))
            for group in confirmation: occluders.append(_occluder_check(audit, folder, group['id'], width))
        audit.equal(report['summary'], summaries, 'report.summary')
        audit.equal(report['gates'], gates, 'report.gates')
        audit.equal(report['scheduled'], 320, 'report.scheduled')
        audit.check(report['physical_robot_ready'] is False and report['depth_accuracy_verified'] is False, 'report.scope_flags')
    except Exception as error:
        message = str(error).replace(str(audit.study), '<study>')
        if source_root is not None: message = message.replace(str(source_root), '<source>')
        audit.errors.append('fatal:'+type(error).__name__+':'+message[:180])
    inventory = dict(sorted(audit.identities.items()))
    return dict(schema='dvidia.rgbd-independent-audit/v1', passed=not audit.errors,
        mismatch_count=len(audit.errors), mismatches=audit.errors, processed_confirmation_ids=processed,
        expected_confirmation_ids=320, source_bytes_checked=source_root is not None,
        source_bytes_verified=source_root is not None and not audit.errors,
        evidence_files_read=len(inventory), evidence_inventory_sha256=sha256(canonical(inventory)).hexdigest(),
        evidence_inventory=inventory, summaries=summaries, gates=gates, occluder_depth_checks=occluders, scope=SCOPE,
        simulator_executed=False, estimator_executed=False, physical_robot_ready=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-root', type=Path); parser.add_argument('--repo', type=Path)
    args = parser.parse_args(argv)
    if args.source_root is not None and args.repo is not None: parser.error('Choose --source-root or --repo')
    if args.output.exists(): raise FileExistsError('Audit output already exists; refusing overwrite')
    source = args.source_root or (args.repo/'src/dvidia_training' if args.repo else None)
    report = audit_study(args.study, source)
    body = canonical(report)+b'\n'
    with args.output.open('xb') as stream: stream.write(body)
    print(json.dumps(dict(passed=report['passed'], mismatch_count=report['mismatch_count'],
        processed_confirmation_ids=report['processed_confirmation_ids'], audit_sha256=sha256(body).hexdigest())))
    return 0 if report['passed'] else 1


if __name__ == '__main__': sys.exit(main())
