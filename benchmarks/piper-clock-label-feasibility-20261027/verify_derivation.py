"""Verify a public projection against its separately retained original package.

Original records contain machine-local provenance, so they are not republished.
This offline verifier accepts their directory explicitly; it never writes it.
No native simulator is imported and no scientific computation is repeated.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def record(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=sha256(raw).hexdigest())


def canonical(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)


def pointer(document, key):
    parts = key.split('/')[1:]
    current = document
    for part in parts[:-1]:
        part = part.replace('~1', '/').replace('~0', '~')
        current = current[int(part)] if isinstance(current, list) else current[part]
    final = parts[-1].replace('~1', '/').replace('~0', '~')
    return current, int(final) if isinstance(current, list) else final


def asset_default(raw):
    tree = ast.parse(raw)
    matches = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--assets'):
            matches.extend(kw.value.value for kw in node.keywords
                           if kw.arg == 'default' and isinstance(kw.value, ast.Constant)
                           and isinstance(kw.value.value, str))
    if len(matches) != 1:
        raise AssertionError('Expected one assets default string literal.')
    return matches[0]


def verify(original, public):
    derivation = load(public/'public-derivation.json')
    for name, expected in derivation['original_package']['files'].items():
        assert record(original/name) == expected, ('Original package digest mismatch', name)
    for name, expected in derivation['derived_payloads'].items():
        assert record(public/name) == expected, ('Derived payload digest mismatch', name)
    count = 0
    assert set(derivation['json_path_replacements']) == {'report.json', 'runtime.json', 'preserved-inputs.json'}
    for name, changes in derivation['json_path_replacements'].items():
        expected = deepcopy(load(original/name))
        pointers = set()
        for change in changes:
            assert change['pointer'] not in pointers
            pointers.add(change['pointer'])
            parent, key = pointer(expected, change['pointer'])
            before = parent[key]
            assert key in ('path', 'source')
            assert isinstance(before, str) and before.startswith('/')
            assert sha256(before.encode('utf8')).hexdigest() == change['before_value_sha256']
            assert isinstance(change['after_value'], str) and not change['after_value'].startswith('/')
            parent[key] = change['after_value']
            count += 1
        # Full structural equality also retains every numeric value and hash.
        assert canonical(expected) == canonical(load(public/name)), ('Unreviewed JSON field change', name)
    before = (original/'analysis.py').read_bytes()
    after = (public/'analysis.py').read_bytes()
    original_default = asset_default(before)
    change = derivation['script_change']
    assert sha256(original_default.encode('utf8')).hexdigest() == change['before_literal_value_sha256']
    before_literal = repr(original_default).encode('utf8')
    after_literal = repr(change['after_literal_value']).encode('utf8')
    assert before.count(before_literal) == 1
    assert before.replace(before_literal, after_literal, 1) == after
    assert asset_default(after) == 'runs/clock-label-assets'
    old_check = load(original/'dp-check.json')
    expected_check = deepcopy(old_check)
    expected_check['analysis_sha256'] = sha256(after).hexdigest()
    assert canonical(expected_check) == canonical(load(public/'dp-check.json'))
    assert (original/'verify_dp.py').read_bytes() == (public/'verify_dp.py').read_bytes()
    assert (original/'protocol-proposal.md').read_bytes() == (public/'protocol-proposal.md').read_bytes()
    original_report = load(original/'report.json')
    public_report = load(public/'report.json')
    assert original_report['totals'] == public_report['totals']
    assert original_report['pooled_quantiles'] == public_report['pooled_quantiles']
    assert original_report['pooled_counts'] == public_report['pooled_counts']
    assert original_report['per_episode'] == public_report['per_episode']
    assert original_report['jaw_context_sensitivity'] == public_report['jaw_context_sensitivity']
    seal = load(public/'package-integrity.json')
    assert set(seal['files']) == {p.name for p in public.iterdir() if p.is_file()}-{'package-integrity.json'}
    for name, expected in seal['files'].items():
        assert record(public/name) == expected, ('Public package seal mismatch', name)
    return dict(status='passed', original_files_verified=len(derivation['original_package']['files']),
                path_values_replaced=count, other_json_values_unchanged=True,
                script_change='exactly one assets-default literal; no algorithm edits',
                protocol_bytes_unchanged=True, exhaustive_checker_bytes_unchanged=True,
                scientific_numeric_values_unchanged=True, physics_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-package', type=Path, required=True)
    parser.add_argument('--public-package', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.original_package, args.public_package), indent=2))


if __name__ == '__main__':
    main()
