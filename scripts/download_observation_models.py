"""Explicitly download only the two pinned, official observation model assets.

Run with --download and --output PATH. Ordinary detector construction never
downloads. The complete ONNX payload is under 40 MB; every byte count and SHA
is checked before an atomic rename. No fetched code is executed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dvidia_training.observation_models import MODEL_ASSETS, model_metadata


def fetch(url, path, maximum, expected_sha=None):
    if path.is_symlink() or path.exists():
        raise ValueError(f'Refusing to replace an existing artifact: {path.name}')
    part = path.with_name(path.name + '.partial')
    digest, count, created = sha256(), 0, False
    try:
        with part.open('xb') as output:
            created = True
            with urllib.request.urlopen(url, timeout=30) as response:
                while chunk := response.read(min(1024 * 1024, maximum + 1 - count)):
                    count += len(chunk)
                    if count > maximum:
                        raise ValueError('Official artifact exceeded its size cap.')
                    digest.update(chunk)
                    output.write(chunk)
        if expected_sha and (count != maximum or digest.hexdigest() != expected_sha):
            raise ValueError('Official model size or SHA-256 did not match its reviewed Git LFS pointer.')
        part.rename(path)
    except BaseException:
        if created:
            part.unlink(missing_ok=True)
        raise
    return {'name': path.name, 'bytes': count, 'sha256': digest.hexdigest(), 'source': url}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Explicitly authorize the bounded network downloads.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.download:
        parser.error('No files downloaded. Pass --download to provision the reviewed models.')
    root = args.output.absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('Output path may not contain symlinks.')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a new or empty directory; existing model assets are never overwritten.')
    root.mkdir(parents=True, exist_ok=True)
    receipt = {'kind': 'dvidia.observation-model-assets', 'models': [], 'licenses': []}
    for asset in MODEL_ASSETS:
        record = model_metadata(asset)
        downloaded = fetch(record['download_url'], root / asset['name'], asset['bytes'], asset['sha256'])
        receipt['models'].append({**record, 'downloaded_sha256': downloaded['sha256']})
        receipt['licenses'].append(fetch(record['license_url'], root / (asset['directory'] + '-LICENSE.txt'),
                                         asset['license_bytes'], asset['license_sha256']))
        print(f"Verified {asset['name']}: {asset['bytes']} bytes", flush=True)
    (root / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Model provenance saved to {root / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
