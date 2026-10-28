"""Copy retained development evidence; declare local-path-only caller edits.

No physics, fitting or mutation of original evidence. Native arrays, episodes,
commands and controller snapshots retain their exact bytes. Only analysis.py
execution wrappers may replace maintainer filesystem roots with portable names.
"""
from pathlib import Path
from hashlib import sha256
import argparse,json,sys

sys.path.insert(0,str(Path(__file__).with_name('huggingface')))
from publish_grip_clock_v03 import scan,plain,read,encoded

RUNS=(('piper-clock-v03-feasibility-20261027',16),
      ('piper-clock-v03-feasibility02-20261027',16),
      ('piper-clock-v03-feasibility03-20261027',16),
      ('piper-clock-v03-coordinated-feasibility-20261027',32),
      ('piper-grip-clock-v03-runner-smoke-20261027',1),
      ('piper-grip-clock-v03-runner-smoke02-20261027',2),
      ('piper-grip-clock-v03-parallel-smoke-20261027',4))


def prepare(runs,output):
    runs,output=plain(runs),plain(output)
    if (output.exists() or output == runs
            or any(output.is_relative_to(runs/name) or (runs/name).is_relative_to(output)
                   for name, _ in RUNS)):
        raise ValueError('Use a fresh output separate from the original evidence roots.')
    project=runs.parent
    asset_root=project.parent/'dvidia-training'/'runs'/'robot-profile-upstream'/'assets'
    output.mkdir(parents=True)
    manifest=dict(schema='dvidia.piper-clock-development-lineage/v0.3',native_attempts=87,
        main_study_attempts=768,pooled_with_main_study=False,files={},caller_derivations=[],
        warning='The first two architecture-A pilots contain original invalid command metadata plus explicitly derived metadata-only projections. Derived projections are not additional physics attempts.',
        scope='Development and technical evidence on already-known starts, excluded from fitting and fresh confirmation.')
    for name,expected in RUNS:
        source=runs/name
        paths=sorted(path for path in source.rglob('*') if path.is_file())
        if sum(path.name=='native.npz' for path in paths)!=expected:
            raise ValueError('Native attempt count differs for '+name)
        for path in paths:
            raw=read(path); public=raw
            if path.suffix=='.py':
                public=raw.replace(str(asset_root).encode(),b'./robot-assets').replace(str(project).encode(),b'.')
            if public!=raw:
                if path.name!='analysis.py':
                    raise ValueError('Only analysis.py caller paths may be redacted.')
                manifest['caller_derivations'].append(dict(path=(Path(name)/path.relative_to(source)).as_posix(),
                    original_sha256=sha256(raw).hexdigest(),public_sha256=sha256(public).hexdigest(),
                    change='Replace maintainer-specific asset and working-tree root literals with ./robot-assets and .; original retained locally under the recorded digest. Controller and data bytes unchanged.'))
            relative=(Path(name)/path.relative_to(source)).as_posix()
            scan(public,path.name)
            target=output/relative; target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write(public)
            manifest['files'][relative]=dict(bytes=len(public),sha256=sha256(public).hexdigest(),
                original_sha256=sha256(raw).hexdigest(),exact_original_bytes=public==raw)
            if sha256(path.read_bytes()).hexdigest()!=sha256(raw).hexdigest():
                raise ValueError('Original evidence changed during copy.')
    (output/'lineage-manifest.json').write_bytes(encoded(manifest))
    (output/'README.md').write_text('''# Retained development lineage

These 87 native attempts are separate from the frozen 768-attempt main study.
All were performed on already-known starting positions. They do not contribute
teacher rows, candidate selection outcomes or fresh confirmation counts.

Architecture A retains three 16-attempt pilots. Its first two contain original
issued-command metadata errors. Their saved metadata-only projections rebuild
command mirrors from actual saved actions; they do not rerun dynamics, alter
native observations or count as additional attempts. Architecture B retains its
32-attempt gate-offset pilot. Three runner attempts include an unsupported-goal
invalid prefix followed by two valid corrected technical checks. Four further
spawn-worker checks validate execution infrastructure on a known nominal scene.
The unsupported-goal script stopped after its first attempt; its second declared
case was not executed and is not counted as a native attempt.

Native NPZ arrays, original/derived episodes and controller source snapshots
retain exact original bytes. Some `analysis.py` wrappers contain maintainer path
literals. Only those path literals are replaced in public copies; each original
and public SHA-256 and edit scope is declared in `lineage-manifest.json`.
Original wrappers remain locally preserved under their original digests.
These wrappers record debugging history; the main study CLI is the supported
reproduction path. Synthetic simulation evidence only; no hardware qualification.
''')
    return dict(native_attempts=87,files=len(manifest['files']),caller_derivations=len(manifest['caller_derivations']),manifest_sha256=sha256((output/'lineage-manifest.json').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.runs,args.output),sort_keys=True))
