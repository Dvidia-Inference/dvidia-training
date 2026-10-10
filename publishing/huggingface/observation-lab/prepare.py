"""Prepare an explicit synthetic-only static Observation Lab Space; never upload."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAX_TOTAL_BYTES = 2 * 1024 * 1024


def record(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def plain(path):
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Preparation accepts only the named plain source files.')
    return path


def fresh(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Publication output may not contain symlinks.')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Choose a fresh or empty publication directory.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Observation UI changed; review the demo adapter before preparing.')
    return text.replace(old, new)


def stage(namespace, output):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,95}', namespace):
        raise ValueError('Expected a Hugging Face organization name.')
    output = fresh(output)
    space = output / 'space'
    space.mkdir()
    # Explicit inventory only: no private runs, clips, models, credentials or
    # recursive source/package copies can enter the public Space.
    sources = {
        'README.md': HERE / 'templates' / 'README.md',
        'demo-adapter.js': HERE / 'templates' / 'demo-adapter.js',
        'generate_demo.py': HERE / 'generate_demo.py',
        'LICENSE': ROOT / 'LICENSE',
        'THIRD_PARTY.md': HERE / 'templates' / 'THIRD_PARTY.md',
        'licenses/object_detection_yolox-LICENSE.txt': ROOT / 'third_party' / 'observation-models' / 'object_detection_yolox-LICENSE.txt',
        'licenses/palm_detection_mediapipe-LICENSE.txt': ROOT / 'third_party' / 'observation-models' / 'palm_detection_mediapipe-LICENSE.txt',
        'source/observation_studio.html': ROOT / 'src' / 'dvidia_training' / 'observation_studio.html',
    }
    source_records = []
    for name, source in sources.items():
        plain(source)
        target = space / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_records.append({'source_path': source.relative_to(ROOT).as_posix(),
                               'published_path': name, **record(source)})
    readme = space / 'README.md'
    readme.write_text(readme.read_text().replace('{{HF_ORG}}', namespace))
    ui = (space / 'source' / 'observation_studio.html').read_text()
    start = ui.index('  const api = {\n')
    end = ui.index('  function error(message)', start)
    if '    save(id,body)' not in ui[start:end]:
        raise ValueError('Expected the documented Observation API.')
    ui = ui[:start] + '  const api = window.DvidiaObservationDemo;\n' + ui[end:]
    ui = replace_once(ui, '<script>\n(() => {', '<script src="demo-adapter.js"></script>\n<script>\n(() => {')
    ui = replace_once(ui, '<title>Observation lab · DVIDIA</title>', '<title>Observation Lab · Synthetic demo · DVIDIA</title>')
    ui = replace_once(ui, 'Local review · Footage stays here', 'Synthetic demo · Edits stay in your browser')
    ui = replace_once(ui, '<div class="workspace" id="workspace" aria-busy="true">', '''<section class="demo-intro" aria-label="Synthetic demonstration scope">
      <h2>Try the review workflow</h2>
      <p>These two schematic clips and their sample boxes were authored for this demo. No detector, VLM or robot policy runs here. Play, edit, approve or reject an event, then save and download your review.</p>
      <p id="demo-storage-status" role="status">Saved demo revisions stay in this browser. No footage or edits are uploaded.</p>
      <nav aria-label="Observation Lab resources"><a href="https://github.com/Dvidia-Inference/dvidia-training/blob/main/docs/observations.md" target="_blank" rel="noopener noreferrer">Install the local tool</a><a href="https://github.com/Dvidia-Inference/dvidia-training/blob/main/benchmarks/observation-20261010/README.md" target="_blank" rel="noopener noreferrer">Read the actual pilot</a><a href="https://github.com/Dvidia-Inference/dvidia-training/blob/main/docs/observation-models.md" target="_blank" rel="noopener noreferrer">Models and licenses</a><button id="reset-demo" type="button">Reset this browser’s demo</button></nav>
    </section>
    <div class="workspace" id="workspace" aria-busy="true">''')
    ui = replace_once(ui, '    :root { color-scheme:', '''    .demo-intro { margin:0 0 24px; padding:20px; border:1px solid #dddeda; border-radius:16px; background:#fffefa; }
    .demo-intro p { margin-top:10px; max-width:940px; color:#656961; }
    .demo-intro nav { display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; margin-top:14px; }
    .demo-intro a { display:inline-flex; align-items:center; min-height:44px; }
    :root { color-scheme:''')
    replacements = {
        'Connecting to your local episodes…': 'Loading the synthetic demo…',
        'Your footage, ready for a closer look.': 'A schematic clip, ready to review.',
        'Loading the local observation record.': 'Loading authored demonstration annotations.',
        'Raw model': 'Authored',
        'Raw observations': 'Authored observations',
        'Original model output. Select an event to review.': 'Authored sample proposals. Select an event to review.',
        'Select an observation or add an event the model missed.': 'Select a sample annotation or add an event you can verify.',
        'Hand boxes mark detected palms; overlap does not prove a grasp.': 'Demo boxes were authored; no object detector or hand model ran.',
        'Boxes are sampled detections.': 'Boxes are sampled image annotations.',
        'Model observation': 'Authored sample',
        ', raw model output': ', authored sample',
        'Detector score ': 'Authored example score ',
        'Synthetic fixture · Sampled detections': 'Synthetic fixture · Authored boxes',
        'Sampled model detections': 'Authored sample annotations',
        'This local review only plays footage served by the local episode server.': 'The demo only plays its included same-origin synthetic clips.',
        'The local observation server is unavailable.': 'The synthetic demo could not be loaded.',
        'Check the local server, then try again.': 'Refresh the demo, then try again.',
        'could not play the local video': 'could not play the synthetic video',
        'Check that the local episode contains a browser-compatible video.': 'Try a browser with MP4/H.264 playback; the sample annotation remains available.',
        'DVIDIA · Observation pilot': 'DVIDIA · Synthetic workflow demo',
        'Saved revision ${episode.review?.revision ?? 0}': 'Saved in this browser · Revision ${episode.review?.revision ?? 0}',
    }
    for old, new in replacements.items():
        if old not in ui:
            raise ValueError('Observation UI display contract changed: ' + old)
        ui = ui.replace(old, new)
    ui = ui.replace('https://github.com/Dvidia-Inference/dvidia-training/blob/main/docs/observations.md',
                    'https://github.com/Dvidia-Inference/dvidia-training/blob/observation-lab-v0.1.0/docs/observations.md')
    ui = ui.replace('https://github.com/Dvidia-Inference/dvidia-training/blob/main/docs/observation-models.md',
                    'https://github.com/Dvidia-Inference/dvidia-training/blob/observation-lab-v0.1.0/docs/observation-models.md')
    ui = ui.replace('https://github.com/Dvidia-Inference/dvidia-training/blob/main/benchmarks/observation-20261010/README.md',
                    'https://github.com/Dvidia-Inference/dvidia-training/blob/observation-lab-v0.1.0/benchmarks/observation-20261010/README.md')
    ui = replace_once(ui, '  initialize();', '''  $('reset-demo').addEventListener('click',async () => {
    if (!confirm('Reset only this browser’s demo reviews? Download saved reviews first if you want to keep them.')) return;
    await api.reset(); location.reload();
  });
  initialize();''')
    (space / 'index.html').write_text(ui)
    subprocess.run([shutil.which('python3') or 'python3', str(HERE / 'generate_demo.py'),
                    '--output', str(space / 'demo')], check=True)
    provenance = {'kind': 'dvidia.observation-space-provenance.v1',
        'scope': 'authored_synthetic_review_workflow', 'live_inference': False,
        'private_footage_included': False, 'model_weights_included': False,
        'dvidia_trained_weights': False, 'source_files': source_records,
        'ui_derivative': 'Observation UI copied exactly to source/, with client-only API and explicit authored-demo labels in index.html.',
        'review_storage': 'browser storage only; memory fallback if blocked; no server writes',
        'fixture_provenance': 'demo/provenance.json'}
    (space / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    inventory = [{'path': p.relative_to(space).as_posix(), **record(p)}
                 for p in sorted(space.rglob('*')) if p.is_file()]
    if sum(row['bytes'] for row in inventory) > MAX_TOTAL_BYTES:
        raise ValueError('Synthetic demo exceeds its 2 MiB publication budget.')
    source_commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                   check=True, capture_output=True, text=True).stdout.strip()
    plan = {'kind': 'dvidia.observation-lab-publication', 'schema_version': 1,
            'namespace': namespace, 'private': False, 'source_commit': source_commit,
            'ui_sha256': record(ROOT / 'src' / 'dvidia_training' / 'observation_studio.html')['sha256'],
            'repository': {'repo_id': namespace + '/observation-lab', 'repo_type': 'space',
                           'space_sdk': 'static', 'directory': 'space', 'files': inventory},
            'total_bytes': sum(row['bytes'] for row in inventory)}
    (output / 'publication-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--org', default='Dvidia')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    plan = stage(args.org, args.output)
    print(json.dumps({'repo_id': plan['repository']['repo_id'], 'files': len(plan['repository']['files']),
                      'bytes': plan['total_bytes'], 'sdk': plan['repository']['space_sdk']}, indent=2))
