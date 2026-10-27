"""Independently check the packaged DP objective against exhaustive tiny paths.

Only the `monotone` function AST is loaded; the analysis module's imports and
main routine are never executed. No native simulator is imported or stepped.
This checks recurrence/constraints, not the validity of scientific labels.
"""
from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def enumerate_paths(rows, columns, max_steps):
    """An independent recursive enumeration with initial path index zero."""
    def extend(path):
        if len(path) == rows:
            yield tuple(path)
            return
        for index in range(path[-1], min(columns, path[-1]+max_steps+1)):
            yield from extend(path+[index])
    yield from extend([0])


def check():
    source = Path(__file__).with_name('analysis.py')
    raw = source.read_bytes()
    tree = ast.parse(raw, filename=str(source))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'monotone']
    if len(functions) != 1:
        raise AssertionError('Packaged analysis must contain exactly one monotone function.')
    scope = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), scope)
    dynamic_program = scope['monotone']
    rng = np.random.default_rng(27017)
    cases = [
        (np.zeros((1, 1)), 3),
        (np.zeros((5, 8)), 3),
        (np.array([[0., 8., 8.], [8., 0., 8.], [8., 8., 0.]]), 1),
        (np.array([[0., 8., 8.], [0., 8., 8.], [8., 0., 8.]]), 1),
        (np.array([[0., 9., 9., 9., 9., 9., 9.],
                   [9., 9., 9., 0., 9., 9., 9.],
                   [9., 9., 9., 9., 9., 9., 0.]]), 3),
        (np.array([[0., 0., 0.], [0., 10., 10.], [0., 10., 10.]]), 3),
    ]
    for _ in range(64):
        rows = int(rng.integers(1, 6))
        columns = int(rng.integers(1, 9))
        max_steps = int(rng.integers(0, 4))
        # Integers include exact ties; fractions exercise noninteger objectives.
        cost = rng.integers(0, 9, size=(rows, columns)).astype(float)
        if len(cases) % 2:
            cost += rng.random(cost.shape)*.125
        cases.append((cost, max_steps))
    legal_paths_checked = 0
    largest_objective_error = 0.
    for cost, max_steps in cases:
        path = dynamic_program(cost, max_steps=max_steps)
        assert path.shape == (len(cost),)
        assert path[0] == 0
        assert np.all((path >= 0) & (path < cost.shape[1]))
        assert np.all((np.diff(path) >= 0) & (np.diff(path) <= max_steps))
        enumerated = list(enumerate_paths(*cost.shape, max_steps))
        legal_paths_checked += len(enumerated)
        optimum = min(sum(float(cost[i, j]) for i, j in enumerate(candidate))
                      for candidate in enumerated)
        actual = float(np.sum(cost[np.arange(len(cost)), path]))
        error = abs(actual-optimum)
        largest_objective_error = max(largest_objective_error, error)
        assert np.isclose(actual, optimum, atol=1e-12, rtol=1e-12), (cost, path, actual, optimum)
    assert np.array_equal(dynamic_program(np.zeros((5, 8))), np.zeros(5, dtype=int))
    assert np.array_equal(dynamic_program(cases[4][0]), [0, 3, 6])
    assert np.array_equal(dynamic_program(cases[5][0]), [0, 0, 0])
    return dict(schema='dvidia.clock-feasibility-dp-exhaustive-check/v1',
                status='passed', analysis_sha256=sha256(raw).hexdigest(),
                checker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                deterministic_seed=27017, matrices_checked=len(cases),
                legal_paths_checked=legal_paths_checked,
                largest_objective_error=largest_objective_error,
                checks=['initial index zero', 'bounded monotonic increments',
                        'minimum exhaustive objective', 'earliest terminal exact tie',
                        'maximum-rate unique path', 'terminal index not forced'],
                native_simulator_imported=False, physics_steps=0,
                limitation='Recurrence/constraint check only; not a validation of progress-label semantics.')


if __name__ == '__main__':
    print(json.dumps(check(), indent=2, allow_nan=False))
