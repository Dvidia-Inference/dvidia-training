"""Local compact actuator learning; no language service or physical arm commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def write_report_page(directory):
    """Embed the exact retained receipt in a self-contained offline report."""
    directory = Path(directory)
    receipt = directory / 'report.json'
    if receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('Use a bounded, regular pilot receipt.')
    raw = receipt.read_bytes().decode('utf8')
    document = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite receipt.')))
    if type(document) is not dict or document.get('kind') != 'dvidia.actuator-transition-pilot.v1':
        raise ValueError('Unsupported actuator pilot receipt.')
    embedded = json.dumps(raw, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = Path(__file__).with_name('transition_pilot.html').read_text(encoding='utf8')
    with (directory / 'report.html').open('x', encoding='utf8') as output:
        output.write(template.replace('__DVIDIA_RECEIPT__', embedded))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    corpus = sub.add_parser('corpus', help='Record the authored 12-group native fixture; requires the arm extra.')
    corpus.add_argument('--output', required=True)
    corpus.add_argument('--seed-start', type=int, default=311000)
    run = sub.add_parser('run', help='Fit compact models, freeze them, and score the separate test groups.')
    run.add_argument('manifest')
    run.add_argument('--output', required=True)
    inspect = sub.add_parser('inspect', help='Check retained artifact consistency; this does not verify physical accuracy.')
    inspect.add_argument('directory')
    args = parser.parse_args(argv)
    try:
        from .transition_pilot import generate_corpus, inspect_pilot, run_pilot
        if args.operation == 'corpus':
            result = generate_corpus(args.output, seed_start=args.seed_start)
        elif args.operation == 'run':
            result = run_pilot(args.manifest, args.output)
            write_report_page(args.output)
        else:
            result = inspect_pilot(args.directory)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 1 if result.get('status') in ('failed', 'error') else 0
    except (ValueError, OSError, ImportError) as exc:
        message = str(exc) if isinstance(exc, ValueError) else f'Local operation failed ({type(exc).__name__}).'
        print(json.dumps({'status': 'error', 'error': message[:300]}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
