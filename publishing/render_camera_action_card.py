"""Render publication cards from a stopped study, never from hand-entered metrics.

No simulation, training, credentials or network calls. A supplied posthoc summary
is recomputed from sealed numeric receipts; a completed study additionally needs
the independent full-study audit. Blocked studies remain explicit candidates.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

import prepare_camera_action_v01 as evidence
from huggingface.camera_action_hardware import MAX_BYTES as MAX_HARDWARE_BYTES, validate_hardware

SCHEMA = 'dvidia.camera-action-cards/v1'
REPO = Path(__file__).resolve().parents[1]
SUMMARY_SOURCE = REPO / 'benchmarks/camera-action-v01/summarize_results.py'
PHASE_SOURCE = REPO / 'benchmarks/camera-action-v01/diagnostic_phase_coverage.py'
SOURCE_URL = 'https://github.com/Dvidia-Inference/dvidia-training'
STAGES = ('diagnostic', 'demonstrations', 'train-initial', 'development-initial',
          'corrections', 'train-corrected', 'development-corrected', 'confirmation', 'negative')
BUDGETS = {'diagnostic': 4, 'demonstrations': 80, 'development-initial': 32,
           'corrections': 32, 'development-corrected': 32, 'confirmation': 256, 'negative': 16}


def recompute_summary(study):
    spec = importlib.util.spec_from_file_location('camera_action_card_posthoc', SUMMARY_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize(study)


def recompute_phase(study, diagnostic_audit):
    spec = importlib.util.spec_from_file_location('camera_action_card_phase', PHASE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze(study, diagnostic_audit)


def _cell(value):
    if value is None:
        return 'Not reported'
    return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '\\|').replace('`', '&#96;').replace('\n', ' ').replace('\r', ' ')


def _number(value, digits=3):
    return 'Not reported' if value is None else f'{value:.{digits}f}'


def _table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|',
            *('| ' + ' | '.join(_cell(cell) for cell in row) + ' |' for row in rows), '']


def _inputs(study, *, summary, source_commit, hardware=None, study_audit=None, phase_coverage=None, diagnostic_audit=None):
    from dvidia_training import camera_action_study as workflow
    study = evidence.plain(study)
    if (study / 'worker.lock').exists():
        raise ValueError('Resolve the active or interrupted study worker before rendering publication cards.')
    if not re.fullmatch('[a-f0-9]{40}', source_commit):
        raise ValueError('An exact reviewed source commit is required.')
    state = workflow.inspect(study)
    if state['status'] not in ('complete', 'blocked'):
        raise ValueError('Cards require a completed study or an explicitly blocked study.')
    protocol = evidence.read(study / 'protocol.json')
    if {name: len(rows) for name, rows in protocol['rosters'].items()} != BUDGETS:
        raise ValueError('This card template requires the exact registered v0.1 stage budgets.')
    if (protocol.get('capture') != dict(duration_s=20., width=320, height=240, camera_hz=25,
            control_hz=50, physics_timestep_s=.001, camera_pairs_per_complete_episode=501)
            or protocol.get('fitting') != dict(hidden_units=[64, 64], epochs=50, batch_size=512,
                learning_rate=.003, seed=17, normalization='training-only', corrections_rounds=1,
                teacher_mix_probability=.5)
            or protocol.get('start_domain', {}).get('x_m') != [.287, .311]
            or protocol.get('start_domain', {}).get('y_m') != [-.042, -.018]
            or {key: len(value) for key, value in protocol.get('cases', {}).items()} !=
                dict(train=64, development=16, confirmation=64)):
        raise ValueError('Capture, fitting or start-domain registration differs from this v0.1 methodology.')
    report = evidence.read(study / 'report.json')
    if report != state['report'] or report.get('physical_robot_ready') is not False:
        raise ValueError('Saved report differs from sealed study inspection.')
    if state['status'] == 'complete' and report['status'] != 'complete':
        raise ValueError('Completed study lacks a complete result report.')
    if state['status'] == 'blocked' and (report.get('status') != 'blocked' or report.get('qualified') is not False):
        raise ValueError('Blocked studies must remain unqualified research candidates.')
    if study_audit is None and state['status'] != 'blocked':
        raise ValueError('Completed cards require the independent full-study audit.')
    receipts = {name: receipt for name in STAGES if (receipt := workflow._stage(study, name)) is not None}
    proof = evidence.verify_study_audit(study, study_audit, receipts, state) if study_audit is not None else None
    document = evidence.read(summary)
    if document != recompute_summary(study):
        raise ValueError('Posthoc summary differs from fresh read-only reconstruction of sealed receipts.')
    if document.get('schema') != 'dvidia.camera-action-posthoc-summary/v1' or document.get('physical_robot_ready') is not False:
        raise ValueError('Expected the simulation-only posthoc summary format.')
    expected_qualification = dict(status=report['status'], qualified=report['qualified'],
        qualification=report['qualification'], report_sha256=evidence.identity(study / 'report.json')['sha256'],
        source='existing study report; not newly awarded by this posthoc tool')
    if document.get('qualification') != expected_qualification:
        raise ValueError('Posthoc qualification is not bound to the final report.')
    bindings = {'protocol.json': evidence.identity(study / 'protocol.json'),
        'registration.json': evidence.identity(study / 'registration.json'),
        'report.json': evidence.identity(study / 'report.json')}
    for name, receipt in receipts.items():
        for suffix in ('.json', '.integrity.json', '.started.json'):
            path = 'stages/' + name + suffix
            bindings[path] = evidence.identity(study / path)
        if document['source_receipts'].get('stages/' + name + '.json') != bindings['stages/' + name + '.json']['sha256']:
            raise ValueError('Posthoc stage identity differs from the sealed receipt.')
    if set(document['source_receipts']) != {'stages/' + name + '.json' for name in receipts}:
        raise ValueError('Posthoc stage roster differs from current sealed evidence.')
    selected = report.get('selected_stage')
    if selected is not None:
        frozen = evidence.read(study / 'freeze.json')
        if selected not in ('initial', 'corrected') or frozen.get('selected_stage') != selected:
            raise ValueError('Selected model stage differs from frozen selection.')
        bindings['freeze.json'] = evidence.identity(study / 'freeze.json')
    models = []
    for stage in ('initial', 'corrected'):
        receipt = receipts.get('train-' + stage)
        if receipt is None:
            continue
        if set(receipt['models']) != {'full', 'blind'}:
            raise ValueError('Each sealed fit must retain the matched full and blind models.')
        for variant in ('full', 'blind'):
            record = receipt['models'][variant]
            path = f'models/{stage}/{variant}/model.json'
            if record['path'] != path:
                raise ValueError('Model receipt path differs from its fitting stage.')
            actual = evidence.identity(study / path)
            if actual != dict(bytes=record['model_bytes'], sha256=record['model_sha256']):
                raise ValueError('Model bytes differ from the sealed fitting receipt.')
            from dvidia_training.camera_action_model import validate_model
            model = validate_model(evidence.read(study / path))
            if (model.get('schema') != 'dvidia.camera-action-mlp/v1' or model.get('blind') is not (variant == 'blind')
                    or model.get('physical_robot_ready') is not False or model.get('hidden_units') != [64, 64]
                    or model['training'] != record.get('training') or model['training']['stage'] != stage):
                raise ValueError('Model schema or architecture differs from this card template.')
            for filename in ('model.json', 'receipt.json', 'started.json'):
                name = f'models/{stage}/{variant}/{filename}'
                bindings[name] = evidence.identity(study / name)
            if evidence.read(study / f'models/{stage}/{variant}/receipt.json') != record:
                raise ValueError('Individual model receipt differs from sealed fitting stage.')
            models.append(dict(stage=stage, variant=variant, path=path, identity=actual,
                selected=selected == stage, training=record, model=model))
    if selected and {row['variant'] for row in models if row['selected']} != {'full', 'blind'}:
        raise ValueError('Selected full/blind pair is missing.')
    expected_model_files = {name for name in bindings if name.startswith('models/')}
    model_root = study / 'models'
    if model_root.exists():
        actual_model_files = {path.relative_to(study).as_posix() for path in evidence.files(model_root)}
        expected_directories = {str(Path(name).parent) for name in expected_model_files}
        expected_directories |= {str(Path(name).parent) for name in expected_directories}
        actual_directories = {path.relative_to(study).as_posix() for path in model_root.rglob('*') if path.is_dir()}
        if actual_model_files != expected_model_files or actual_directories != expected_directories:
            raise ValueError('Unsealed or orphan model prefixes cannot disappear from a publication card.')
    hardware_doc = None
    if hardware is not None:
        evidence.identity(hardware, maximum=MAX_HARDWARE_BYTES, secret_scan=True)
        hardware_doc = validate_hardware(evidence.read(hardware))
        declared_runtime = {name: hardware_doc[name] for name in ('system', 'machine', 'python')}
        declared_runtime.update(hardware_doc['versions'])
        if any(any(row['recorded_runtime'].get(name) != value for name, value in declared_runtime.items())
               for row in document['hardware']):
            raise ValueError('Public hardware/software context differs from the recorded episode runtime.')
    phase = None
    if phase_coverage is not None:
        if diagnostic_audit is None:
            raise ValueError('Phase coverage requires its independently audited diagnostic source.')
        phase = evidence.read(phase_coverage)
        if (phase.get('schema') != 'dvidia.camera-action-phase-coverage/v1'
                or phase.get('protocol_sha256') != bindings['protocol.json']['sha256']
                or phase.get('diagnostic_audit_sha256') != evidence.identity(diagnostic_audit)['sha256']
                or phase != recompute_phase(study, diagnostic_audit)):
            raise ValueError('Diagnostic phase coverage differs from this protocol.')
    elif diagnostic_audit is not None:
        raise ValueError('A diagnostic audit is used only with its supplied phase coverage.')
    return dict(study=study, source_commit=source_commit, state=state, protocol=protocol,
        report=report, receipts=receipts, summary=document, models=models, hardware=hardware_doc,
        audit=proof, phase=phase, study_bindings=bindings)


def _status(data):
    report = data['report']
    if report['qualified']:
        return 'Bounded simulation skill: all registered gates passed. Physical robot ready: **no**.'
    if report['status'] == 'blocked':
        return '**Research candidate — blocked.** The study stopped at a declared blocker; missing stages are unrun, not successful. Physical robot ready: **no**.'
    return '**Research candidate — qualification gates failed.** A completed benchmark does not establish an installable qualified skill. Physical robot ready: **no**.'


def _results(data):
    report, summary = data['report'], data['summary']
    lines = ['## Registered experiment and results', '',
        '72 fresh starting-position groups are split 32 training / 8 development / 32 confirmation. '
        'Both destinations and all related recordings stay within their group. '
        'The tiny interpolation domain is x = 0.287–0.311 m, y = −0.042–−0.018 m, z = 0.082 m. '
        'Demonstrations comprise 64 training and 16 development episodes. No failed scene is replaced.', '']
    rows = []
    for name in STAGES:
        receipt = data['receipts'].get(name)
        budget = BUDGETS.get(name, '2 models; 50 epochs each')
        rows.append((name, budget, receipt['status'] if receipt else 'Unrun',
            receipt.get('strict_successes') if receipt and name in BUDGETS else None))
    lines += _table(['Stage', 'Registered budget', 'Status', 'Strict successful attempts'], rows)
    lines += ['All registered rollout budgets total **452 attempts**. Fitting does not add native attempts. '
        'The 32 correction episodes use training groups only and seeded 50% teacher control. '
        'Development evaluates two stages × two model variants × 16 cases = 64 rollouts. '
        'Frozen confirmation is 64 cases × four controllers = 256 attempts. Negatives add 16 attempts.', '']
    lines += ['Before fitting, all four diagnostics must strictly succeed with a valid initial visual anchor '
        'within 10 mm of native diagnostic truth. Teacher data must cover every training start-group '
        'at least once and provide at least 28 successful usable episodes per destination.', '']
    quality = data['receipts'].get('demonstrations', {}).get('data_quality')
    if quality is not None:
        lines += ['Recorded teacher-data gate:', '', '```json', json.dumps(quality, indent=2, sort_keys=True), '```', '']
    rows = [(row['stage'] + ' / ' + row['controller'], f"{row['strict_successes']} / {row['scheduled']}",
             row['complete'], row['native_assessable'], json.dumps(row['statuses'], sort_keys=True))
            for row in summary['controllers']]
    lines += _table(['Stage / controller', 'Strict success / scheduled', 'Complete', 'Native assessable', 'Retained statuses'], rows)
    lines += ['Contact-affected counts use the independent 1 ms native scorer; unassessed attempts stay explicit.', '']
    lines += _table(['Stage / controller', 'Contact-affected / assessed', 'Unassessed', 'Critical native ticks'],
        [(r['stage'] + ' / ' + r['controller'],
          f"{r['critical_contacts']['attempts_with_contacts']} / {r['critical_contacts']['assessed_attempts']}",
          r['critical_contacts']['unassessed_attempts'], r['critical_contacts']['native_ticks'])
         for r in summary['controllers']])
    lines += ['Component cells are **failed / conditionally assessed; unassessed overall / eligible / unknown prerequisite**. '
        'Lift requires observed grasp, carry/retention require observed lift, and release/settling/retreat '
        'require supported carry. These conditional diagnostics do not replace all-attempt task success.', '']
    component_rows = []
    for r in summary['controllers']:
        cells = [r['stage'] + ' / ' + r['controller']]
        for name in ('grasp', 'lift', 'carry', 'retention', 'release', 'settling', 'retreat'):
            c = r['components'][name]
            cells.append(f"{c['conditional_failed']} / {c['conditional_assessed']}; "
                f"{c['unassessed']} / {c['conditional_unassessed']} / {c['prerequisite_unknown']}")
        component_rows.append(cells)
    lines += _table(['Stage / controller', 'Grasp', 'Lift', 'Carry', 'Retention', 'Release', 'Settling', 'Retreat'], component_rows)
    lines += ['These posthoc tables retain unknown prerequisites and all-attempt unassessed counts; '
        'an absent prerequisite does not become an observed component failure.', '']
    lines += ['### Qualification gates', '',
        'Required: ≥58/64 strict learned placements; positive 97.5% paired bootstrap intervals versus '
        'both fixed and blind controllers, grouping both destinations by starting position; positive intervals '
        'also on at least 24 common admitted groups; no increase in critical-contact episode frequency; '
        'all 16 negative controls correct; no privileged learner inputs or teacher evaluation fallback.', '']
    lines += _table(['Gate', 'Result'], [(name, 'PASS' if passed else 'FAIL') for name, passed in report['gates'].items()]) if report['gates'] else ['Gates are **unassessed** because the study is blocked.', '']
    for label, key in (('All scheduled starting groups', 'comparisons'), ('Common admitted starting groups', 'admitted_comparisons')):
        if report.get(key):
            lines += [label + ' (verbatim structured result):', '', '```json',
                      json.dumps(report[key], indent=2, sort_keys=True, allow_nan=False), '```', '']
    for name, receipt in data['receipts'].items():
        if receipt.get('status') == 'blocked':
            lines += [f"Blocked at **{name}**: {_cell(receipt.get('reason'))}.", '']
    return lines


def _observations(data):
    lines = ['## Perception, speed and hardware', '',
        'RGB-D is captured at 320×240 and 25 Hz; encoders/commands at 50 Hz; independent native checks '
        'at 1 ms. A complete 20-second episode has 501 camera pairs and 1,000 motor intervals. '
        'Missing estimates remain missing with masks and frame age; the initial anchor is explicitly historical. '
        'The known-upright-block tracker can abstain when the gripper occludes or tilts the block.', '']
    rows = []
    for row in data['summary']['controllers']:
        p, t = row['perception'], row['timing']
        inference = t['policy_inference_wall_seconds']
        loop = t['full_boundary_wall_seconds']
        rows.append((row['stage'] + ' / ' + row['controller'],
            f"{p['available_decisions']} / {p['assessed_decisions']}", p['episodes_unknown'],
            _number(p['maximum_missing_seconds']),
            _number(inference['p50'] * 1000 if inference['p50'] is not None else None) if inference['applicable'] else 'Not applicable',
            _number(inference['p95'] * 1000 if inference['p95'] is not None else None) if inference['applicable'] else 'Not applicable',
            _number(loop['p95'] * 1000 if loop['p95'] is not None else None)))
    lines += _table(['Stage / controller', 'Available / assessed visual decisions', 'Episodes unreported',
        'Longest missing (s)', 'Inference p50 (ms)', 'Inference p95 (ms)', 'Full boundary p95 (ms)'], rows)
    lines += ['Latencies pool individual recorded boundaries. A teacher zero is not a learned-policy inference measurement. '
        'Synthetic camera time, source-frame age, simulation time and wall-clock computation are distinct. '
        'The simulator pauses for computation on a shared host: these numbers do **not** establish physical '
        'sensor-to-actuator latency, a hard deadline or hardware control frequency.', '']
    warning_rows = []
    for row in data['summary']['controllers']:
        warnings = row.get('renderer_warnings')
        if warnings:
            warning_rows.append((row['stage']+' / '+row['controller'],
                f"{warnings['episodes_with_warnings']} / {warnings['episodes_reported']}",
                warnings['episodes_unreported'], json.dumps(warnings['warning_boundaries_by_message'], sort_keys=True)))
    if warning_rows:
        lines += ['Recorded graphics warnings are retained below. Counts refer to render boundaries '
            'carrying cumulative warning history, not new warning emissions or independent depth-error trials. '
            'An absent field remains unreported; no physical '
            'depth accuracy is inferred from a warning-free frame.', '']
        lines += _table(['Stage / controller', 'Affected / reported episodes', 'Unreported episodes',
            'Warning messages and boundary counts'], warning_rows)
    phase = data['phase']
    if phase:
        lines += [f"**Diagnostic manipulation blind spot:** {phase['manipulation_available']} / "
            f"{phase['manipulation_decisions']} visual estimates were available during "
            + ', '.join(_cell(name) for name in phase['manipulation_phases'])
            + f" across {phase['cases']} teacher diagnostic episodes. "
            'This is a diagnostic subset, not student confirmation. Teacher phase is used only in this '
            'offline readback and is excluded from learner features; repeated 50 Hz decisions share 25 Hz camera frames.', '']
    rows = [(row['stage'] + ' / ' + row['variant'], row['model_bytes'], _number(row['training_wall_seconds']),
        _number(row['shared_export_wall_seconds']), row['process_peak_rss_bytes'], row['training_episodes'])
        for row in data['summary']['training']]
    lines += _table(['Fit', 'Model bytes', 'Fit/save wall seconds', 'Shared export seconds', 'Process peak RSS bytes', 'Training episodes'], rows)
    lines += ['Process peak RSS includes shared dataset export and any preceding fit; it is not isolated model memory. '
        'Training uses CPU NumPy. Native rendering needs a working graphics driver/OpenGL context; no CUDA requirement '
        'or minimum-hardware qualification is inferred from this host.', '']
    if data['hardware']:
        lines += ['Public host context supplied with this release (`hardware.json`); software identifiers '
            'are cross-checked against bound episode runtime receipts when present:', '', '```json',
            json.dumps(data['hardware'], indent=2, sort_keys=True), '```', '']
    else:
        lines += ['CPU model, memory capacity and GPU identity were not supplied in a bound public hardware context; '
            'those details remain **not reported**.', '']
    for row in data['summary']['hardware']:
        lines += [f"Recorded runtime in {row['episode_receipts']} episode receipts:", '', '```json',
            json.dumps(row['recorded_runtime'], indent=2, sort_keys=True), '```', '']
    return lines


def dataset_card(data):
    protocol_hash = data['study_bindings']['protocol.json']['sha256']
    lines = ['---', 'license: mit', 'task_categories:', '- robotics', 'language:', '- en',
        'tags:', '- robotics', '- simulation', '- behavior-cloning', '- synthetic-data', '- piper', '---', '',
        '# DVIDIA · Camera-to-action v0.1', '', _status(data), '',
        'A local, reproducible PiPER block-placement experiment: learn seven absolute motor targets '
        'from synchronized synthetic RGB-D, encoders and demonstrated actions; evaluate the complete '
        'approach, grasp, lift, carry, release, settling and retreat task on unseen starting positions.', '',
        f"[Reviewed source]({SOURCE_URL}/tree/{data['source_commit']}) · "
        '[Research](https://research.dvidia.org) · [DVIDIA](https://dvidia.org)', '',
        f"Protocol SHA-256: `{protocol_hash}`. Source commit: `{data['source_commit']}`.", '',
        '## Dataset origin and intended use', '',
        'All footage is rendered in MuJoCo. Motor labels come from an authored teacher with privileged '
        'simulator state. No human video, human teleoperation, private footage or physical robot measurements '
        'are claimed. Initial demonstrations label issued targets; the correction round labels separately '
        'recorded teacher suggestions, while the tape also preserves the commands actually executed.', '',
        'Use this evidence to study bounded camera/encoder imitation and its failures. '
        'It does not demonstrate arbitrary-video action recovery, general world modeling, shoes, '
        'arbitrary-arm transfer or physical hardware qualification.', '',
        '## Model contract and selection', '',
        'A CPU NumPy MLP with two 64-unit tanh hidden layers predicts six joint targets and one gripper target. '
        'Each fitting stage uses 50 epochs, batch size 512, Adam learning rate 0.003 and seed 17. '
        'Training-only normalization includes explicit missing fills and masks. The model uses 160 ms '
        'of causal history, current encoders, encoder-derived link6 pose, previous applied servo references, '
        'goal, elapsed time and a historical initial visual position. It never receives teacher phase, '
        '`reference.action`, the current teacher target, native object/contact truth, future frames or '
        'native object orientation.', '',
        'The matched blind model masks all live/historical sensor features, retaining the same initial '
        'visual context, goal, time and causal prior-command history. It sees the same training episodes. '
        'The legacy tcp_* feature fields are link6 body-origin estimates, not a calibrated tool point; '
        'the native scorer uses a distal site at link6 local [0, 0, 0.12003] m.', '',
        'Selection uses full-learner development success, then critical-contact ticks, then common '
        'physical-action-scale prediction loss, then initial before corrected. The blind comparator '
        'comes from that same selected stage. Confirmation permits no teacher intervention.', '']
    models = data['models']
    lines += _table(['Stage', 'Variant', 'Selected', 'Model artifact', 'Model card', 'Bytes', 'SHA-256'],
        [(r['stage'], r['variant'], r['selected'], r['path'],
          f"[Read card](model-cards/{r['stage']}/{r['variant']}/README.md)",
          r['identity']['bytes'], r['identity']['sha256']) for r in models])
    if not models:
        lines += ['No completed model fit is sealed before the blocker. No downloadable skill is claimed.', '']
    lines += _results(data) + _observations(data)
    lines += ['## Limitations and independent verification', '',
        'Negative controls are deliberately narrow: initial_missing blanks the initial RGB while leaving '
        'the object in the simulator; initial_stale sets capture/availability to −0.2 s; altered_calibration '
        'scales one rotation element by 1.1; open_gripper forces the jaw open. Passing these checks does '
        'not establish detection of physical absence, all stale-frame cases or small valid rigid calibration errors.', '',
        'Depth, contact, friction, actuator fidelity, sensor delivery and the sim-to-real gap remain '
        'unqualified. A passing integrity audit certifies consistency of saved evidence, not task success. '
        'The posthoc summary does not tune or award qualification.', '',
        ('The passed full-study audit covers every currently sealed stage, roster, action authority, clocks '
         'and independent native score reconstruction. No physics, tracker or policy was re-executed.' if data['audit'] else
         '**No full-study independent audit was supplied.** This blocked publication is explicitly unaudited beyond '
         'the sealed receipts and read-only posthoc reconstruction; it cannot award a qualified-skill label.'), '',
        'See `report.json`, `protocol.json`, `full-study-audit.json` when supplied, and the raw per-episode '
        'integrity inventories. The table generator rechecks a local posthoc summary against sealed receipts; '
        'When packaged with the model cards, `model-cards-receipt.json` binds their generated bytes to the same models and report. '
        'All failed, refused, interrupted and unattempted outcomes remain in '
        'their registered denominators. Unassessed component outcomes are not guessed from overall failure.', '']
    selected = next((record for record in models if record['stage'] == data['report'].get('selected_stage')
                     and record['variant'] == 'full'), None)
    if selected is not None:
        lines += ['## Try the selected simulation candidate', '',
            f"Download the [selected full-feedback model]({selected['path']}) as `model.json` and the "
            '[released wheel](https://huggingface.co/datasets/Dvidia/piper-camera-action-v01/tree/main/source). Only these two files plus the separately provisioned PiPER assets '
            'are needed to try the candidate. The raw episode corpus and source archive are not needed '
            'for this installation. The [reviewed source](' + SOURCE_URL + '/tree/' + data['source_commit'] +
            ') remains available for inspection.', '',
            'Installation does not change the model\'s qualification. This simulation candidate cannot '
            'command physical hardware. Use Python 3.12 and a working OpenGL renderer; the tested '
            'renderer is macOS CGL. Other platforms remain unqualified. Replace the wheel path below '
            'with the actual downloaded filename, retaining the `[arm]` extra.', '',
            '```sh', 'python3.12 -m venv .venv', 'source .venv/bin/activate',
            "python -m pip install '/path/to/downloaded-wheel.whl[arm]'",
            f"printf '%s  model.json\\n' '{selected['identity']['sha256']}' | shasum -a256 -c -",
            'dvidia-robot-sim fetch piper --output /path/to/robot-assets',
            'dvidia-arm check --model model.json --robot piper',
            'dvidia-arm install --model model.json --robot piper --assets /path/to/robot-assets --output /path/to/new-installed-skill',
            'dvidia-arm run /path/to/new-installed-skill --assets /path/to/robot-assets --output /path/to/new-attempt --start-x .30 --start-y -.03 --goal 0',
            '```', '', 'Use fresh output directories; the saved attempt preserves its independent task outcome. '
            'A successful install is not a successful placement.', '']
    lines += ['## Reproduce the complete study', '',
        'The release contains source/wheel artifacts, `study-metadata.zip`, and one deterministic ZIP '
        'per recorded attempt under `episodes/`. Verify the manifest and checksums before extracting. '
        'Archives preserve each original recording integrity file; the compact diagnostic viewer is '
        'not the full raw archive.', '',
        '```sh', 'shasum -a256 -c SHA256SUMS',
        'python -m zipfile -e study-metadata.zip restored-study', '```', '',
        'Extract every episode archive listed in `manifest.json` into that same fresh `restored-study` '
        'directory; each archive already contains its `episodes/<stage>/<attempt>/` prefix. '
        'Install the released wheel with its `arm` extra, or extract the reviewed source archive and run:', '',
        '```sh', 'python3.12 -m venv .venv', 'source .venv/bin/activate',
        "python -m pip install '.[arm]'", 'dvidia-train robot inspect --study /path/to/restored-study',
        'dvidia-training-studio --directory /path/to/new-local-runs --port 8363', '```', '',
        'The UI at `http://127.0.0.1:8363/robot` and Terminal share the same run record. '
        'Reproduction uses a **fresh** study, the published fixed comparator and separately provisioned '
        'verified PiPER assets; original study attempts are immutable. Provision the pinned assets separately:', '',
        '```sh', 'dvidia-robot-sim fetch piper --output /path/to/robot-assets', '```', '',
        '```sh', 'dvidia-train robot prepare --study /path/to/new-study --fixed-model /path/to/restored-study/fixed-model.json',
        'dvidia-train robot collect --study /path/to/new-study --assets /path/to/robot-assets --stage diagnostic',
        'dvidia-train robot inspect --study /path/to/new-study', '```', '',
        f"[Complete collect/train/evaluate/package commands]({SOURCE_URL}/blob/{data['source_commit']}/docs/camera-action-v01.md). "
        'After dependencies/assets are installed, inference and training are local. The exact PiPER '
        'simulation adapter is the only accepted embodiment. Native stepping is CPU work; rendering '
        'needs OpenGL. The tested renderer is macOS CGL. Preserve 30 GiB free disk; raw evidence is '
        'bounded at 120 GiB. Other platforms and hardware minimums remain unqualified.', '']
    lines += ['## License and attribution', '',
        'Original DVIDIA code, generated dataset evidence and learned parameter JSON are MIT. '
        'Upstream robot models, MuJoCo and other third-party components retain their own licenses '
        'and notices; assets are provisioned separately. See `LICENSE` and the source third-party notices. '
        'Individual contributor attribution is intentionally unset.', '']
    return '\n'.join(lines)


def model_card(data, record):
    variant = record['variant']
    selected = record['selected']
    lines = ['---', 'license: mit', 'tags:', '- robotics', '- simulation', '- behavior-cloning', '- piper', '---', '',
        f"# DVIDIA · PiPER camera-to-action · {record['stage']} / {variant}", '',
        (_status(data) if selected and variant == 'full' else
         '**Research candidate — comparison artifact.** This model has no separately awarded skill qualification. '
         'The study gates concern the selected full-feedback model. Physical robot ready: **no**.'), '',
        ('This is the selected full-feedback model.' if variant == 'full' else
         'This is the matched blind baseline, not the deployable full-feedback skill.') if selected else
        'This development candidate was not selected for frozen confirmation.', '',
        f"Model file: [model.json](../../../{record['path']}) ({record['identity']['bytes']} bytes). SHA-256: `{record['identity']['sha256']}`.", '',
        f"Reviewed source: [{data['source_commit']}]({SOURCE_URL}/tree/{data['source_commit']}). "
        f"Protocol: `{data['study_bindings']['protocol.json']['sha256']}`.", '',
        '## Inputs, output and training', '',
        'Data-only JSON, schema `dvidia.camera-action-mlp/v1`; no pickle or remote code. '
        'CPU NumPy, two 64-unit tanh hidden layers, seven absolute targets (six joints and jaw). '
        'Native joint/rate limits and independent task checks remain in the simulation runtime.', '',
        ('Full feedback uses historical initial visual position, goal, time, current encoder/link6 measurements, '
         'previous applied commands and 160 ms of causal sensor history with explicit masks and age.' if variant == 'full' else
         'All live and historical sensor-feedback tensor channels are masked. Initial visual context, goal, time '
         'and causal previous-applied-command history remain, exactly matching the full model context.'), '',
        'Labels are authored privileged-simulator teacher actions; they are not recovered human actions. '
        'The correction pass uses separately recorded teacher queries at mixed-controller visited states. '
        'Executed commands and suggestions retain separate authority. No current teacher command, '
        'reference.action, teacher phase, native object/contact truth or future frame is a learner input.', '',
        'Fitting budget: seed 17, 50 epochs, batch 512, Adam 0.003, training-only normalization including '
        'missing zeros/masks. The full and blind models use identical source recordings.', '',
        '## Measured fitting cost', '']
    training = record['training']
    lines += _table(['Measurement', 'Value'], [(key, training.get(key)) for key in
        ('training_wall_seconds', 'shared_export_wall_seconds', 'process_peak_rss_bytes', 'memory_scope')])
    lines += [f"Training recordings: {len(training['training_episodes'])}; development recordings: {len(training['development_episodes'])}.", '',
        'Model provenance:', '', '```json', json.dumps({key: record['model']['training'].get(key) for key in
            ('stage', 'sample_count', 'train_groups', 'dataset_sha256', 'provenance_sha256', 'model_source_sha256')},
            indent=2, sort_keys=True), '```', '',
        '## Evaluation and use', '',
        'See the [dataset card](../../../README.md) and `report.json` at the release root '
        'for all four confirmation controllers, failed gates, grouped intervals, negative controls and '
        'perception/latency denominators. A model file or successful installation is not evidence of skill qualification.', '']
    matching = [r for r in data['summary']['controllers'] if r['controller'] == ('student' if variant == 'full' else 'blind')
        and (r['stage'] == 'development-' + record['stage'] or selected and r['stage'] == 'confirmation')]
    lines += _table(['Evaluation', 'Strict success / scheduled', 'Complete', 'Native assessable'],
        [(r['stage'], f"{r['strict_successes']} / {r['scheduled']}", r['complete'], r['native_assessable']) for r in matching])
    lines += ['Only one simulated PiPER, known rigid upright block and two known destinations were studied. '
        'Object estimates can disappear throughout manipulation. Link6 pose is not the calibrated distal tool point. '
        'No shoe skill, human-video motor recovery, arbitrary-arm transfer, physical robot readiness or real-time guarantee. '
        'Teacher latency zeroes are inapplicable; process RSS includes export/preceding fits.', '',
        'Original learned parameters and DVIDIA code are MIT; upstream dependencies retain their notices. '
        'This artifact cannot command physical hardware through the provided installer.', '']
    return '\n'.join(lines)


def render(study, output, *, summary, source_commit, hardware=None, study_audit=None, phase_coverage=None, diagnostic_audit=None):
    study, output = evidence.plain(study), evidence.plain(output)
    if output.exists() or output == study or output.is_relative_to(study):
        raise ValueError('Use a fresh cards directory outside the study.')
    data = _inputs(study, summary=summary, source_commit=source_commit, hardware=hardware,
                   study_audit=study_audit, phase_coverage=phase_coverage, diagnostic_audit=diagnostic_audit)
    documents = {'README.md': dataset_card(data)}
    for record in data['models']:
        documents[f"model-cards/{record['stage']}/{record['variant']}/README.md"] = model_card(data, record)
    for text in documents.values():
        evidence.scan(text.encode())
    sources = {'summary.json': evidence.identity(summary), 'generator.py': evidence.identity(Path(__file__)),
        'summarize_results.py': evidence.identity(SUMMARY_SOURCE)}
    for name, path in (('hardware.json', hardware), ('full-study-audit.json', study_audit),
                      ('diagnostic-phase-coverage.json', phase_coverage), ('independent-audit.json', diagnostic_audit)):
        if path is not None:
            sources[name] = evidence.identity(path)
    if phase_coverage is not None:
        sources['diagnostic_phase_coverage.py'] = evidence.identity(PHASE_SOURCE)
    files = {}
    output.mkdir(parents=True, exist_ok=False)
    # Retain the exact input bytes so packaging can verify this claimed source
    # identity as well as independently reconstructing its numeric content.
    with Path(summary).open('rb') as source, (output / 'summary.json').open('xb') as target:
        copied = 0
        for chunk in iter(lambda: source.read(1024*1024), b''):
            if copied + len(chunk) > sources['summary.json']['bytes']:
                raise ValueError('Summary grew while retaining the generated card source.')
            target.write(chunk)
            copied += len(chunk)
        if copied != sources['summary.json']['bytes']:
            raise ValueError('Summary shrank while retaining the generated card source.')
    if evidence.identity(output / 'summary.json') != sources['summary.json']:
        raise ValueError('Summary changed while retaining the generated card source.')
    for name, text in documents.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        files[name] = evidence.identity(path)
    receipt = dict(schema=SCHEMA, source_commit=source_commit,
        protocol_sha256=data['study_bindings']['protocol.json']['sha256'],
        report_sha256=data['study_bindings']['report.json']['sha256'], status=data['report']['status'],
        qualification=data['report']['qualification'], qualified=data['report']['qualified'],
        full_study_audited=data['audit'] is not None, physical_robot_ready=False,
        summary_reconstructed=True, physics_executed=False, model_executed=False,
        source_inputs=sources, study_bindings=data['study_bindings'], files=files,
        model_cards=[dict(card=f"model-cards/{r['stage']}/{r['variant']}/README.md", model=r['path'],
            model_sha256=r['identity']['sha256'], selected=r['selected']) for r in data['models']])
    evidence.write(output / 'card-receipt.json', receipt)
    return dict(status='complete', output=str(output), cards=len(files), qualification=data['report']['qualification'],
        receipt_sha256=evidence.identity(output / 'card-receipt.json')['sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'output', 'summary', 'source-commit'):
        parser.add_argument('--' + name, required=True)
    for name in ('hardware', 'study-audit', 'phase-coverage', 'diagnostic-audit'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(render(args.study, args.output, summary=args.summary,
            source_commit=args.source_commit, hardware=args.hardware, study_audit=args.study_audit,
            phase_coverage=args.phase_coverage, diagnostic_audit=args.diagnostic_audit), indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status='refused', reason=str(exc))))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
