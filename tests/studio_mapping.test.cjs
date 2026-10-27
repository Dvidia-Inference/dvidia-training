const { test } = require('node:test');
const assert = require('node:assert/strict');
const mapping = require('../src/dvidia_training/studio_mapping.js');
const camera = () => ({ width: 320, height: 240, center_xy_m: [.36, 0], vertical_span_m: .5, position_bound_m: .005, terminal_dwell_s: .5, maximum_terminal_spread_m: .005 });

test('plain measurement fields preserve units including millimetres', () => {
  const result = mapping.fromFields({ width:'320', height:'240', x:'.36', y:'0', span:'.5', bound:'5', dwell:'.5', spread:'5' });
  assert.equal(result.error, '');
  assert.deepEqual(result.value, camera());
  assert.notEqual(mapping.fromFields({ width:'320' }).error, '');
});
test('client rejects the same numeric limits as the native mapping', () => {
  for (const update of [{width:1920}, {height:480.5}, {center_xy_m:[0,Infinity]}, {vertical_span_m:0}, {position_bound_m:0}, {terminal_dwell_s:.1}, {maximum_terminal_spread_m:.1}, {extra:true}]) {
    assert.notEqual(mapping.problem({...camera(), ...update}), '');
  }
  assert.notEqual(mapping.parse('{bad').error, '');
  assert.equal(mapping.problem(camera()), '');
});
test('mapping edits make an existing proposal stale regardless of JSON order', () => {
  const reordered = Object.fromEntries(Object.entries(camera()).reverse());
  assert.equal(mapping.same(camera(), reordered), true);
  assert.equal(mapping.same(camera(), {...camera(),center_xy_m:[.37,0]}), false);
  assert.equal(mapping.same(camera(), null), false);
});
test('world goal projects into the calibrated image with correct axes', () => {
  assert.deepEqual(mapping.imageGoal([.36,0],camera()), {u:.5,v:.5});
  const rightUp = mapping.imageGoal([.46,.1],camera());
  assert.ok(Math.abs(rightUp.u-.65)<1e-12);
  assert.ok(Math.abs(rightUp.v-.3)<1e-12);
  assert.equal(mapping.imageGoal([1,0],camera()), null);
  assert.equal(mapping.imageGoal([NaN,0],camera()), null);
});
