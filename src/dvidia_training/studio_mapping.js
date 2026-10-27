/* Shared mapping checks for the local UI. No calibration is inferred here. */
(function (root) {
  'use strict';
  const keys = ['width', 'height', 'center_xy_m', 'vertical_span_m', 'position_bound_m', 'terminal_dwell_s', 'maximum_terminal_spread_m'];
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const between = (value, low, high) => finite(value) && value >= low && value <= high;
  function problem(value) {
    if (!value || Array.isArray(value) || typeof value !== 'object') return 'Choose a mapping file or enter your camera measurements.';
    if (Object.keys(value).length !== keys.length || keys.some(key => !Object.hasOwn(value, key))) return 'The mapping needs all seven documented fields and no extra fields.';
    if (!Number.isInteger(value.width) || !between(value.width, 64, 640)) return 'Image width must be a whole number from 64 to 640 pixels.';
    if (!Number.isInteger(value.height) || !between(value.height, 64, 480)) return 'Image height must be a whole number from 64 to 480 pixels.';
    if (!Array.isArray(value.center_xy_m) || value.center_xy_m.length !== 2 || !value.center_xy_m.every(n => between(n, -2, 2))) return 'Enter both image-center coordinates, from −2 to 2 metres.';
    if (!between(value.vertical_span_m, .1, 2)) return 'Visible scene height must be from 0.1 to 2 metres.';
    if (!between(value.position_bound_m, .0001, .05)) return 'Declared mapping error must be from 0.1 to 50 mm.';
    if (!between(value.terminal_dwell_s, .5, 5)) return 'Minimum still time must be from 0.5 to 5 seconds.';
    if (!between(value.maximum_terminal_spread_m, .0001, .05)) return 'Maximum image-position drift must be from 0.1 to 50 mm.';
    return '';
  }
  function parse(text) {
    try { const value = JSON.parse(text); return { value, error: problem(value) }; }
    catch { return { value: null, error: 'The mapping is not valid JSON.' }; }
  }
  function fromFields(values) {
    const names = ['width', 'height', 'x', 'y', 'span', 'bound', 'dwell', 'spread'];
    if (names.some(name => typeof values[name] !== 'string' || !values[name].trim())) return { value: null, error: 'Complete the camera measurements below, or import a saved mapping.' };
    const n = Object.fromEntries(names.map(name => [name, Number(values[name])]));
    const value = { width: n.width, height: n.height, center_xy_m: [n.x, n.y], vertical_span_m: n.span, position_bound_m: n.bound / 1000, terminal_dwell_s: n.dwell, maximum_terminal_spread_m: n.spread / 1000 };
    return { value, error: problem(value) };
  }
  function same(a, b) {
    if (problem(a) || problem(b)) return false;
    return keys.every(key => key === 'center_xy_m' ? a[key].every((n, i) => n === b[key][i]) : a[key] === b[key]);
  }
  function imageGoal(goal, mapping) {
    if (problem(mapping) || !Array.isArray(goal) || goal.length !== 2 || !goal.every(finite)) return null;
    const u = .5 + (goal[0] - mapping.center_xy_m[0]) * mapping.height / (mapping.vertical_span_m * mapping.width);
    const v = .5 - (goal[1] - mapping.center_xy_m[1]) / mapping.vertical_span_m;
    return between(u, 0, 1) && between(v, 0, 1) ? { u, v } : null;
  }
  const api = Object.freeze({ problem, parse, fromFields, same, imageGoal });
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.DvidiaStudioMapping = api;
})(globalThis);
