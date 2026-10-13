import assert from 'node:assert/strict';
import {test} from 'node:test';
import {boundedTarget, dampedTargets, limitNorm, RollingSamples, solveLinear} from '../../src/dvidia_training/workstation_control.js';

test('workspace inputs reject invalid vectors instead of clipping them silently', () => {
  const bounds = [[.18, .54], [-.24, .24], [.055, .34]];
  assert.deepEqual(boundedTarget([.36, -.06, .27], bounds), [.36, -.06, .27]);
  for (const value of [[NaN, 0, .1], [Infinity, 0, .1], [.3, .9, .1], [.3, 0], ['.3', 0, .1]]) {
    assert.throws(() => boundedTarget(value, bounds));
  }
});

test('regularized solver and norm caps preserve direction', () => {
  assert.deepEqual(solveLinear([[2, 0], [0, 4]], [6, 8]), [3, 2]);
  assert.throws(() => solveLinear([[0, 0], [0, 0]], [1, 1]));
  assert.deepEqual(limitNorm([3, 4, 0], 1), [.6, .8, 0]);
});

test('damped target solve respects native joint margins at singular configurations', () => {
  const zero = new Float64Array(18);
  const config = {translation_error_cap_m: .015, rotation_error_cap_rad: .05, orientation_weight: .25,
    damping: .012, max_joint_delta_rad: .06, joint_margin_rad: .02};
  const targets = dampedTargets({jacp: zero, jacr: zero, nv: 6,
    dofIds: [0, 1, 2, 3, 4, 5], qposIds: [0, 1, 2, 3, 4, 5], qpos: [1, -1, 0, 0, 0, 0],
    target: [.5, .2, .3], current: [0, 0, 0], rotationWorld: [0, .2, 0],
    ctrlRange: Array.from({length: 6}, () => [-1, 1]), config});
  assert.deepEqual(targets, [.98, -.98, 0, 0, 0, 0]);
});

test('timing telemetry uses bounded retention during long sessions', () => {
  const samples = new RollingSamples(8);
  for (let value = 0; value < 10000; value += 1) samples.add(value);
  samples.add(NaN);
  assert.equal(samples.values.length, 8);
  assert.equal(samples.quantile(0), 9992);
  assert.equal(samples.quantile(1), 9999);
});
