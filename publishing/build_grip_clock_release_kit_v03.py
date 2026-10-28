"""Build a compact, portable v0.3 installer kit and benchmark digest offline.

Copies existing model/wheel/smoke bytes; never trains, runs physics, changes a
sealed study or publishes remotely. Outputs must be fresh. The supplied smoke
receipts explicitly distinguish inherited dependencies from a fresh resolver.
"""
from __future__ import annotations

import argparse
import base64
import configparser
import csv
from email.parser import BytesParser
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


SCHEMA = 'dvidia.piper-grip-clock-release-kit/v0.3'
WHEEL = 'dvidia_training-0.1.1-py3-none-any.whl'
WHEEL_SHA = '01e61f5e0e536f3c0b446110d108a5c056fa6c8f001fc18e6d55d92bdd076c47'
ARCHIVE = 'dvidia-piper-grip-clock-v0.3-kit.zip'
DIST = 'dvidia_training-0.1.1.dist-info/'
RELEASE = 'https://github.com/Dvidia-Inference/dvidia-training/releases/tag/piper-grip-clock-v0.3'
MODEL = 'https://huggingface.co/Dvidia/piper-grip-clock-v03'
DATASET = 'https://huggingface.co/datasets/Dvidia/piper-grip-clock-demonstrations-v03'
PAPER = 'https://research.dvidia.org/papers/learned-piper-grip-clock-v03/'
PRIVATE = re.compile(rb'/Users/|/private/tmp/|file:///(?![\s\"\'])')
TOKEN = re.compile(rb'(?:hf_[A-Za-z0-9]{30,}|apikey_[a-f0-9]{20,}_[a-f0-9]{20,}|sk-[A-Za-z0-9_-]{30,}|Bearer\s+[A-Za-z0-9_-]{20,})')


def digest(raw):
    return sha256(raw).hexdigest()


def receipt(raw):
    return dict(bytes=len(raw), sha256=digest(raw))


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def scan(raw):
    if PRIVATE.search(raw) or TOKEN.search(raw):
        raise ValueError('Possible private path or credential in a public release artifact.')


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Release inputs must be ordinary files.')
    raw = path.read_bytes()
    if len(raw) > 16*1024**2:
        raise ValueError('Release input exceeds its bounded size.')
    return raw


def decode(raw):
    return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))


def verify_wheel(raw, parity, model_root):
    if digest(raw) != WHEEL_SHA or parity['wheel'] != dict(name=WHEEL, **receipt(raw)):
        raise ValueError('Wheel differs from the existing reviewed 0.1.1 artifact.')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        rows = archive.infolist()
        if len({row.filename for row in rows}) != len(rows) or sum(row.file_size for row in rows) > 64*1024**2:
            raise ValueError('Wheel contains duplicate files or excessive expansion.')
        members = {}
        for row in rows:
            path = PurePosixPath(row.filename)
            if path.is_absolute() or '..' in path.parts or row.is_dir() or row.file_size > 16*1024**2:
                raise ValueError('Wheel has an unsafe archive member.')
            members[row.filename] = archive.read(row)
        records = list(csv.reader(io.StringIO(members[DIST+'RECORD'].decode())))
        if {row[0] for row in records} != set(members) or len(records) != len(members):
            raise ValueError('Wheel RECORD does not cover exactly its members.')
        for name, hashed, size in records:
            if name == DIST+'RECORD':
                if hashed or size:
                    raise ValueError('Wheel RECORD must leave its own hash empty.')
            else:
                expected = base64.urlsafe_b64encode(sha256(members[name]).digest()).rstrip(b'=').decode()
                if hashed != 'sha256='+expected or size != str(len(members[name])):
                    raise ValueError('Wheel member differs from its RECORD hash.')
        actual = {name.removeprefix('dvidia_training/'): receipt(body)
                  for name, body in members.items() if name.startswith('dvidia_training/')}
        if actual != parity['all_packaged_sources'] or len(actual) != parity['all_packaged_source_count']:
            raise ValueError('Wheel/source parity inventory differs from the complete packaged source map.')
        for name, expected in parity['prepared_source_parity_sha256'].items():
            body = read(model_root/'runtime'/name)
            if digest(body) != expected or members['dvidia_training/'+name] != body:
                raise ValueError('A frozen main-study runtime module differs from the wheel.')
        metadata = BytesParser().parsebytes(members[DIST+'METADATA'])
        if (metadata['Version'] != '0.1.1' or metadata['Requires-Python'] != '>=3.12'
                or 'arm' not in metadata.get_all('Provides-Extra', [])
                or 'mujoco==3.15.0; extra == "arm"' not in metadata.get_all('Requires-Dist', [])
                or metadata['License-Expression'] != 'MIT AND Apache-2.0'):
            raise ValueError('Wheel version, Python, native extra or license metadata changed.')
        entrypoints = configparser.ConfigParser(); entrypoints.read_string(members[DIST+'entry_points.txt'].decode())
        if entrypoints['console_scripts'].get('dvidia-piper-clock-skill') != 'dvidia_training.piper_grip_clock_skill:main':
            raise ValueError('The installed native skill command is missing.')
        licenses = {name: members[DIST+'licenses/'+name] for name in metadata.get_all('License-File', [])}
    return licenses


def summaries(model_root, report_raw, audit_raw, wheel_raw):
    report, audit = decode(report_raw), decode(audit_raw)
    if (report['physical_robot_ready'] is not False or report['qualified_robot_skill'] is not False
            or report['advance_to_followup_simulation'] is not False
            or audit['audited_attempts'] != 768 or audit['audit_mismatches'] != 0):
        raise ValueError('The release requires the exact reviewed unqualified study and clean 768-attempt audit.')
    counts = {}
    for variant in ('clock', 'blind_clock', 'fixed', 'teacher'):
        summary = report['variants'][variant]
        rows = [row for row in report['rows'] if row['variant'] == variant]
        if (len(rows) != 128 or summary['scheduled'] != 128
                or sum(row['score']['valid'] is True and row['score']['success'] is True for row in rows) != summary['strict_successes']):
            raise ValueError('Compact counts differ from the complete confirmation report.')
        counts[variant] = summary
    protocol_raw, frozen_raw = read(model_root/'protocol.json'), read(model_root/'frozen.json')
    frozen = decode(frozen_raw)
    for name, key in [('base-model.json', 'base_model_raw_sha256'), ('jaw-model.json', 'jaw_model_raw_sha256'),
                      ('selected.json', 'selected_sha256'), ('blind.json', 'blind_sha256')]:
        if digest(read(model_root/name)) != frozen[key]:
            raise ValueError('Release model bytes differ from the confirmation freeze.')
    if frozen['protocol_sha256'] != digest(protocol_raw) or report['frozen_sha256'] != digest(frozen_raw):
        raise ValueError('Report/protocol/freeze binding changed.')
    selected, blind = decode(read(model_root/'selected.json')), decode(read(model_root/'blind.json'))
    if selected['blind'] is not False or blind['blind'] is not True or selected['training']['sample_count'] != 64000:
        raise ValueError('Expected the exact full and matched blind training scope.')
    result = dict(schema=SCHEMA+'/report-summary', source_report=receipt(report_raw),
        protocol_sha256=digest(protocol_raw), frozen_sha256=digest(frozen_raw),
        confirmation_attempts=512, independent_xy_groups=32, variants=counts, gates=report['gates'],
        qualified_robot_skill=False, physical_robot_ready=False,
        advance_to_followup_simulation=False, limitations=report['limitations'],
        conclusion='Full sensing has more completions than the blind clock, but its group interval includes zero; the fixed-timing reference completes more tasks. The qualification gates are not met.')
    contrasts = dict(schema=SCHEMA+'/paired-comparisons', source_report=receipt(report_raw),
        unit='Whole initial-XY group; both goals and conditions stay paired.', independent_xy_groups=32,
        intervals=report['contrasts'], interpretation='The two marginal 97.5% t intervals give Bonferroni simultaneous 95% coverage under the stated model; exploratory 32-group evidence, not a power qualification.')
    audited = {key: value for key, value in audit.items() if key != 'rows'}
    audited.update(source_audit=receipt(audit_raw), raw_attempt_rows_omitted=True)
    fit = dict(schema=SCHEMA+'/model-fit-scope', physical_robot_ready=False,
        architecture='Small NumPy binary hold/advance tree over 58 causal privileged simulator features; six arm references and original jaw reference share virtual progress.',
        object='40 g rigid block', arm='PiPER reference model', goals_xy_m=[[.35, .08], [.40, -.08]],
        start_bounds_xy_m=[[.284, .314], [-.045, -.015]],
        training=dict(episodes=64, xy_groups=16, labels=64000, hold_weight=20, advance_weight=1,
                      failed_demonstration_rows_retained=True, label_source='Authored acquisition-only causal grip teacher; no human footage.'),
        selected={key: selected['training'][key] for key in ('max_depth', 'min_leaf', 'max_bins', 'sample_count', 'weight_sum')},
        blind={key: blind['training'][key] for key in ('max_depth', 'min_leaf', 'max_bins', 'sample_count', 'weight_sum')},
        masks='Blind clock zeros live/history slots 8:56; both clocks retain context and clock debt.',
        runtime_teacher_available=False, new_grasp_geometry_learned=False, continuous_slip_recovery_established=False,
        source_report=receipt(report_raw), source_wheel=dict(name=WHEEL, **receipt(wheel_raw)),
        models={name: receipt(read(model_root/name)) for name in ('base-model.json', 'jaw-model.json', 'selected.json', 'blind.json')})
    return result, contrasts, audited, fit


def readme(report, wheel_receipt, *, benchmark=False):
    variants = report['variants']
    table = '\n'.join(f"| {name} | {variants[name]['strict_successes']}/128 | {variants[name]['nominal_successes']}/64 | {variants[name]['post_grip_successes']}/64 |" for name in ('clock', 'blind_clock', 'fixed', 'teacher'))
    intro = f"""# DVIDIA PiPER grip clock v0.3

Simulation-only research: learn when to hold or advance a frozen manipulation reference. This release does not qualify a physical robot skill.

| Controller | Strict completions | Nominal | Assigned post-grip condition |
| --- | ---: | ---: | ---: |
{table}

All 512 scheduled attempts remain, across 32 paired XY groups. Full sensing versus blind is +14.06 percentage points, with group interval -8.52 to +36.64 points; this does not establish a positive benefit. Versus fixed timing it is -21.88 points, interval -36.77 to -6.98 points. These are marginal 97.5% t intervals with Bonferroni simultaneous 95% coverage for the two contrasts. The model fails the declared qualification gates.

The clock uses 58 causal privileged simulator features and 64,000 synthetic hold/advance labels from all 64 teacher attempts, including unsuccessful demonstrations. The same virtual progress drives the six arm references and original jaw reference. The blind clock masks its live/history features. No new grasp geometry, human-video training, continuous learned slip recovery or physical transfer is established.

[Research and replay]({PAPER}) · [Model]({MODEL}) · [Complete evidence]({DATASET}) · [Release]({RELEASE}). Exact public revisions are recorded in the release publication receipt. `report-summary.json`, `paired-comparisons.json` and `audit-summary.json` preserve the measured results and audit scope without full native arrays.
"""
    if benchmark:
        return (intro+"""
The independent audit reconstructed 768 attempts with zero mismatches. Sampled readiness uses six 20 ms observations spanning 100 ms; 1 ms interval diagnostics are reported separately. A sampled readiness signal does not certify a continuously secure physical grip.

The known installed smoke is one nominal start (0.312, -0.037) m, goal 0; both installed models completed it. It was selected as an installation check and is outside the 128-case confirmation denominator. Its two successes are not an accuracy estimate. The portable parity and smoke receipts explicitly record inherited pinned dependencies rather than a fresh package resolver or Linux qualification.

Measured study host: Apple M5, ten logical CPUs, 24 GiB memory, no GPU for training/native simulation. Full-clock inference p95 was 1.916 ms; complete simulator-step p95 9.594 ms, maximum 265.909 ms, with 728 of 128,000 steps exceeding 20 ms. These exclude camera, communication, rendering and real actuators. The host is a reference, not a minimum requirement or hard real-time certification.

Use the compact release kit for wheel installation and the complete dataset for reproduction. DVIDIA code/models/evidence use MIT; bundled components retain their original license notices. This directory adds no new experiments.
""").encode()
    return (intro+f"""
## Install from Terminal

Use Python 3.12 or newer. Unzip this kit into a directory and enter it. The native extra is `arm`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install './{WHEEL}[arm]'
dvidia-robot-sim fetch piper --output ./robot-assets
dvidia-robot-sim audit piper ./robot-assets
```

On Windows activate with `.venv\\Scripts\\Activate.ps1` in PowerShell. Installation resolves NumPy 2.5.3 and MuJoCo 3.15.0. Dependency installation and the initial pinned-asset fetch need network access; subsequent native skill runs use no hosted model, account or GPU. Platform-compatible dependency wheels must be available. Minimum CPU/RAM requirements and Linux deployment have not been qualified.

Run the full-sensing clock on the known installation-smoke scenario with a fresh output directory:

```sh
dvidia-piper-clock-skill \\
  --base-model ./models/base-model.json \\
  --jaw-model ./models/jaw-model.json \\
  --model ./models/selected.json \\
  --assets ./robot-assets \\
  --start-x 0.312 --start-y -0.037 --goal 0 \\
  --output ./my-clock-attempt
```

Run the matched blind clock by changing `--model` to `./models/blind.json` and `--output` to a new directory, such as `./my-blind-attempt`. The model document selects the correct input mode. The CLI saves a full 20-second native attempt, 1 ms telemetry, independent strict score and receipt; it refuses existing/interrupted output directories. Exit status 0 means strict success, 1 a numerically valid failed task, and 2 invalid input/evidence. It collects no teacher labels and injects no artificial disturbance.

This known successful nominal smoke is an installation check, not an unseen task, the confirmation cohort or evidence that arbitrary starting states succeed. Both smoke successes are separate from the 44/128 and 26/128 confirmation counts. `installation-smoke.json` records two installed processes; `wheel-source-parity.json` verifies all 186 packaged files and the 19 frozen study modules. These checks inherited existing pinned dependencies through a documented local environment; a fresh dependency resolver was not tested.

## Scope and verification

The artifact is a 40 g rigid-block PiPER timing candidate. Starting X is 0.284–0.314 m, Y -0.045–-0.015 m; goals are (0.35, 0.08) and (0.40, -0.08) m. All six joint targets use radians; the coupled jaw command is a single-finger slide target of 0–0.035 m, not a calibrated full aperture. Camera sensing, other arms/objects/goals and hardware require separate evaluation.

Measured study host: Apple M5, ten logical CPUs, 24 GiB memory. This is a measured CPU reference, not a minimum hardware requirement. Full-clock inference p95 was 1.916 ms and full simulator-step p95 9.594 ms, but maximum step latency was 265.909 ms; 728 of 128,000 steps exceeded 20 ms. Timing includes dense native logging and excludes camera, communication, rendering and physical actuators. Hard real-time performance is not qualified.

Wheel version: 0.1.1; bytes: {wheel_receipt['bytes']}; SHA256:

```text
{wheel_receipt['sha256']}
```

`FILES.sha256` checks every kit payload file; `kit-integrity.json` records the same payload inventory. The release's outer `SHA256SUMS` also identifies the ZIP and wheel. The wheel declares `MIT AND Apache-2.0`; DVIDIA code, models and authored numeric evidence use MIT. `licenses/` preserves the wheel's original third-party notices, including MuJoCo, optional observation components and PiPER asset licensing. Native robot meshes are fetched from the packaged pinned allowlist and are not bundled here.

`protocol.json` and `frozen.json` identify the original benchmark, whose full attempts live in the evidence dataset. Running the single installed scenario does not rerun or extend that frozen confirmation study.
""").encode()


def write_tree(directory, files):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Output directories must be fresh.')
    directory.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or 'private-local-smoke' in name or name.endswith('.py'):
            raise ValueError('Private scripts and unsafe paths cannot enter the release payload.')
        if name != WHEEL:
            scan(raw)
        target = directory/path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)


def build(repo, output, benchmarks):
    repo, output, benchmarks = Path(repo).resolve(), Path(output), Path(benchmarks)
    if output.exists() or benchmarks.exists():
        raise ValueError('Release-kit and benchmark outputs must both be fresh.')
    model_root = repo/'runs/piper-grip-clock-v03-payloads-20261027/model'
    wheel_root = repo/'runs/piper-grip-clock-v03-wheel-20261027'
    wheel_raw = read(wheel_root/WHEEL)
    parity_raw = read(wheel_root/'wheel-source-parity.json')
    smoke_raw = read(repo/'runs/piper-grip-clock-v03-installed-smoke-20261027/installation-smoke.json')
    parity, smoke = decode(parity_raw), decode(smoke_raw)
    if smoke['wheel_sha256'] != WHEEL_SHA or smoke['physical_robot_ready'] is not False or smoke['fresh_processes'] != 2:
        raise ValueError('Installed smoke differs from the reviewed wheel or scope.')
    licenses = verify_wheel(wheel_raw, parity, model_root)
    report_raw, audit_raw = read(model_root/'report.json'), read(model_root/'audit.json')
    report, contrasts, audit, fit = summaries(model_root, report_raw, audit_raw, wheel_raw)
    compact = {'report-summary.json': encode(report), 'paired-comparisons.json': encode(contrasts),
        'audit-summary.json': encode(audit), 'model-fit-scope.json': encode(fit),
        'wheel-source-parity.json': parity_raw, 'installation-smoke.json': smoke_raw,
        'host.json': read(model_root/'docs/host.json'), 'LICENSE': read(model_root/'LICENSE')}
    provenance = dict(schema=SCHEMA+'/provenance', physical_robot_ready=False, new_training=False, new_physics=False,
        inputs={name: receipt(raw) for name, raw in [('report.json', report_raw), ('audit.json', audit_raw),
            ('wheel-source-parity.json', parity_raw), ('installation-smoke.json', smoke_raw), (WHEEL, wheel_raw)]},
        installed_smoke_case=dict(start_xy_m=[.312, -.037], goal=0, condition='nominal',
                                  use='Known installation smoke only; excluded from confirmation denominator.'),
        dependency_resolution_tested=False, source_helper_sha256=digest(read(__file__)))
    compact['provenance.json'] = encode(provenance)
    kit = dict(compact, **{WHEEL: wheel_raw, 'README.md': readme(report, receipt(wheel_raw))})
    for name in ('base-model.json', 'jaw-model.json', 'selected.json', 'blind.json'):
        kit['models/'+name] = read(model_root/name)
    for name in ('protocol.json', 'frozen.json'):
        kit[name] = read(model_root/name)
    for name, raw in licenses.items():
        kit['licenses/'+name] = raw
    entries = {name: receipt(raw) for name, raw in sorted(kit.items())}
    kit['kit-integrity.json'] = encode(dict(schema=SCHEMA, physical_robot_ready=False, files=entries))
    kit['FILES.sha256'] = ''.join(f'{value["sha256"]}  {name}\n' for name, value in entries.items()).encode()
    write_tree(output, kit)
    archive_path = output/ARCHIVE
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(kit.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    archive_raw = read(archive_path)
    if len(archive_raw) >= 16*1024**2:
        raise ValueError('Compact release archive must stay below 16 MiB.')
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(kit):
            raise ValueError('Release ZIP inventory or CRC verification failed.')
        for name, raw in kit.items():
            if archive.read(name) != raw:
                raise ValueError('Release ZIP differs from the exact staged member bytes.')
    (output/'SHA256SUMS').write_text(f'{digest(archive_raw)}  {ARCHIVE}\n{WHEEL_SHA}  {WHEEL}\n')
    benchmark_files = dict(compact, **{'README.md': readme(report, receipt(wheel_raw), benchmark=True)})
    benchmark_files['integrity.json'] = encode(dict(schema=SCHEMA+'/benchmark-digest',
        files={name: receipt(raw) for name, raw in sorted(benchmark_files.items())}, physical_robot_ready=False))
    write_tree(benchmarks, benchmark_files)
    result = dict(schema=SCHEMA+'/build-receipt', archive=dict(name=ARCHIVE, **receipt(archive_raw)),
        wheel=dict(name=WHEEL, **receipt(wheel_raw)), kit_files=len(kit),
        benchmark_files=len(benchmark_files), wheel_record_reverified=True,
        all_packaged_files_reverified=len(parity['all_packaged_sources']),
        frozen_runtime_files_reverified=len(parity['prepared_source_parity_sha256']),
        archive_bytes_reverified=True, private_paths_or_credentials_found=False,
        physical_robot_ready=False, new_training=False, new_physics=False,
        inputs=provenance['inputs'])
    (output/'build-receipt.json').write_bytes(encode(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmarks', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.output, args.benchmarks), indent=2))


if __name__ == '__main__':
    main()
