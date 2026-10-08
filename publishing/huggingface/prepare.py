"""Stage only the already-public synthetic release; never scan local user runs."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELEASE = '2026-10-08-footage-pipeline-v1'
REPOS = {'dataset': 'dvidia-training-examples', 'model': 'dvidia-training-pilot', 'space': 'dvidia-training'}


def safe_relative(name):
    if (type(name) is not str or not name or '\\' in name or ':' in name
            or any(p in ('', '.', '..') for p in name.split('/'))
            or PurePosixPath(name).is_absolute()):
        raise ValueError('Expected a safe relative publication path.')
    return Path(name)


def receipt(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def fresh(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Publication directories must not contain symlinks.')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Use a new or empty publication directory.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_public_archive(archive, output):
    """Extract bounded, data-only, hash-checked release archives."""
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if len(entries) > 200 or len({x.filename for x in entries}) != len(entries):
            raise ValueError('Unexpected public archive inventory.')
        if sum(x.file_size for x in entries) > 64 * 1024 * 1024:
            raise ValueError('Public archive exceeds the staging limit.')
        for entry in entries:
            relative = safe_relative(entry.filename)
            if (entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16)
                    or entry.flag_bits & 1 or relative.suffix not in {'.json', '.mp4', '.npz'}):
                raise ValueError('Only plain release data files may be staged.')
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(entry))


def stage(namespace, evidence, output):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,95}', namespace):
        raise ValueError('Expected a Hugging Face namespace.')
    evidence = Path(evidence).absolute()
    if any(p.is_symlink() for p in (evidence, *evidence.parents)):
        raise ValueError('Public evidence may not use symlinks.')
    manifest = json.loads((evidence / 'manifest.json').read_text())
    if manifest.get('kind') != 'dvidia.footage-pipeline-release' or manifest.get('schema_version') != 1:
        raise ValueError('Expected the frozen public footage release manifest.')
    rows = {row['path']: row for row in manifest['files']}
    if len(rows) != len(manifest['files']):
        raise ValueError('Duplicate public release artifacts.')
    for name, row in rows.items():
        path = evidence / safe_relative(name)
        if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
            raise ValueError('Missing plain public evidence file.')
        if receipt(path) != {'bytes': row['bytes'], 'sha256': row['sha256']}:
            raise ValueError('Public release evidence differs from its manifest.')
    output = fresh(output)
    for directory in REPOS:
        (output / directory).mkdir()

    def copy_evidence(name, destination):
        if name not in rows:
            raise ValueError('Attempt to publish an unlisted evidence artifact.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(evidence / safe_relative(name), destination)

    for lane in ('visual', 'native'):
        archive = f'{lane}-skillspace.zip'
        target = output / 'dataset' / 'archives' / archive
        copy_evidence(archive, target)
        extract_public_archive(target, output / 'dataset' / lane / 'source')
        prefix = f'{lane}-training/dataset/'
        for name in sorted(rows):
            if name.startswith(prefix):
                copy_evidence(name, output / 'dataset' / lane / 'prepared' / name[len(prefix):])
        target = output / 'model' / lane / 'training-result.zip'
        copy_evidence(f'{lane}-training/training-result.zip', target)
        extract_public_archive(target, target.parent)

    for name in sorted(rows):
        if name.startswith('qualification/'):
            copy_evidence(name, output / 'model' / 'native' / name)
    for directory in ('dataset', 'model'):
        copy_evidence('release-ledger.json', output / directory / 'provenance' / 'release-ledger.json')
        shutil.copyfile(evidence / 'manifest.json', output / directory / 'provenance' / 'source-manifest.json')
    copy_evidence('cpu-benchmark.json', output / 'model' / 'benchmarks' / 'macos-m5.json')
    copy_evidence('qualification/replay.html', output / 'space' / 'replay.html')
    replay = output / 'space' / 'replay.html'
    replay_text = replay.read_text()
    replacements = {
        '<title>DVIDIA · Skillspace lab</title>': '<title>DVIDIA · Recorded native qualification</title>',
        'Give the arm a task.': 'Replay one measured scene.',
        'A DVIDIA skill release, installed with a local articulated-arm controller and executed in different grounded scenes.':
            'A learned joint-movement candidate, replayed from one exact held-out native simulation scene.',
        'This first adapter uses a box proxy for the placement task, an authored controller and exact simulator observations. Native jaw contacts move the free object. Qualification for a physical robot and training from demonstrations come next.':
            'Recorded native simulation: learned joint-movement mapping with authored task phases, gripper and recovery, plus privileged simulator observations. Jaw contacts move the free box. One exact scene; no hardware qualification or human-video action learning.',
        'DVIDIA research · offline skill execution prototype': 'DVIDIA research · saved simulation evidence · no live physics',
    }
    for old, new in replacements.items():
        if replay_text.count(old) != 1:
            raise ValueError('Published replay labels changed; review the visualization before staging.')
        replay_text = replay_text.replace(old, new)
    # Add the saved open-jaw control from the same published exact scene, using
    # the replay's 100 ms sampling. No simulator is run and no states are changed.
    pattern = r'(<script id="embedded-run" type="application/json">)(.*?)(</script>)'
    embedded_match = re.search(pattern, replay_text, re.S)
    if embedded_match is None:
        raise ValueError('The measured replay is missing its embedded data.')
    embedded = json.loads(embedded_match.group(2))
    controls = json.loads((evidence / 'qualification' / 'controls.json').read_text())
    if len(embedded['episodes']) != 1 or len(controls['episodes']) != 1:
        raise ValueError('Expected one candidate and one saved control in the exact scene.')
    student = embedded['episodes'][0]
    original_control = controls['episodes'][0]
    if (original_control['policy'] != 'replay_open_jaw' or original_control['seed'] != student['seed']
            or original_control['success'] is not False or student['success'] is not True
            or controls['environment_config'] != embedded['environment_config']):
        raise ValueError('The saved control is not the published one-scene comparison.')
    control = {k: v for k, v in original_control.items() if k not in ('actions', 'trace')}
    control['trace'] = original_control['trace'][::5]
    if control['trace'][-1] != original_control['trace'][-1]:
        control['trace'].append(original_control['trace'][-1])
    embedded['episodes'].append(control)
    embedded['summary'] = {'successes': 1, 'episodes': 2,
        'simulated_seconds': sum(e['simulated_seconds'] for e in embedded['episodes']),
        'episode_wall_seconds': sum(e['episode_wall_seconds'] for e in embedded['episodes'])}
    embedded['replay_sampling'] = '100 ms recorded observations, including each final state; full candidate and open-jaw control traces are in the model repository.'
    serialized = json.dumps(embedded, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    replay_text = replay_text[:embedded_match.start(2)] + serialized + replay_text[embedded_match.end(2):]
    replay.write_text(replay_text)
    (output / 'space' / 'replay-provenance.json').write_text(json.dumps({
        'source_release': RELEASE, 'source_path': 'qualification/replay.html',
        'source_sha256': rows['qualification/replay.html']['sha256'],
        'control_source_path': 'qualification/controls.json',
        'control_source_sha256': rows['qualification/controls.json']['sha256'],
        'published_sha256': receipt(replay)['sha256'],
        'change': 'Display labels corrected and the saved same-scene open-jaw control added at 100 ms sampling. Candidate states and all recorded physics values are unchanged; no new experiment.'
    }, indent=2) + '\n')
    copy_evidence('qualification/qualification.json', output / 'space' / 'qualification.json')
    copy_evidence('cpu-benchmark.json', output / 'space' / 'cpu-benchmark.json')

    for directory in REPOS:
        target = output / directory
        shutil.copyfile(ROOT / 'LICENSE', target / 'LICENSE')
        image = target / 'assets' / 'branding' / 'github-header.png'
        image.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'assets' / 'branding' / 'github-header.png', image)
        if directory == 'space':
            for source in sorted((HERE / 'templates' / 'space').iterdir()):
                if not source.is_file() or source.is_symlink() or source.suffix not in ('.html', '.md'):
                    raise ValueError('Unexpected static Space template.')
                (target / source.name).write_text(source.read_text().replace('{{HF_ORG}}', namespace))
        else:
            source = HERE / 'templates' / f'{directory}-README.md'
            (target / 'README.md').write_text(source.read_text().replace('{{HF_ORG}}', namespace))
    shutil.copyfile(HERE / 'templates' / 'loader.py', output / 'model' / 'loader.py')

    plan = {'schema_version': 1, 'kind': 'dvidia.huggingface-publication',
            'namespace': namespace, 'private': False, 'evidence_release': RELEASE,
            'repositories': []}
    for directory, name in REPOS.items():
        base = output / directory
        files = [{'path': p.relative_to(base).as_posix(), **receipt(p)}
                 for p in sorted(base.rglob('*')) if p.is_file()]
        row = {'repo_id': namespace + '/' + name, 'repo_type': directory,
               'directory': directory, 'files': files}
        if directory == 'space':
            row['space_sdk'] = 'static'
        plan['repositories'].append(row)
    (output / 'publication-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--org', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, default=ROOT.parent / 'dvidia-research' / 'topics' / 'physical-grounding' / 'runs' / RELEASE)
    args = parser.parse_args()
    plan = stage(args.org, args.evidence, args.output)
    print(json.dumps({'prepared': [{'repo_id': r['repo_id'], 'files': len(r['files']),
          'bytes': sum(f['bytes'] for f in r['files'])} for r in plan['repositories']]}, indent=2))
