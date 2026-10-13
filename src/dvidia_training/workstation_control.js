/* DVIDIA live workstation control math. MIT. SI units; quaternions are wxyz. */

export const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

export function finiteVector(value, length, name) {
  if (!Array.isArray(value) || value.length !== length || !value.every(Number.isFinite)) {
    throw new Error(`${name} must contain ${length} finite numbers`);
  }
  return value.slice();
}

export function boundedTarget(value, bounds) {
  const position = finiteVector(value, 3, 'Target');
  if (position.some((coordinate, axis) => coordinate < bounds[axis][0] || coordinate > bounds[axis][1])) {
    throw new Error('Target is outside the workstation workspace');
  }
  return position;
}

export function limitNorm(vector, maximum) {
  const length = Math.hypot(...vector);
  return length > maximum ? vector.map(value => value * maximum / length) : vector.slice();
}

// Partial-pivot Gaussian elimination on the small regularized 6x6 system.
// Throws instead of sending non-finite or ill-conditioned targets to actuators.
export function solveLinear(matrix, rhs) {
  const n = rhs.length;
  const rows = matrix.map((row, index) => [...row, rhs[index]]);
  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    }
    if (!Number.isFinite(rows[pivot][column]) || Math.abs(rows[pivot][column]) < 1e-14) {
      throw new Error('Controller solve is singular');
    }
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const divisor = rows[column][column];
    for (let item = column; item <= n; item += 1) rows[column][item] /= divisor;
    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const scale = rows[row][column];
      for (let item = column; item <= n; item += 1) rows[row][item] -= scale * rows[column][item];
    }
  }
  return rows.map(row => row[n]);
}

export function dampedTargets({jacp, jacr, nv, dofIds, qpos, qposIds, target, current,
  rotationWorld, ctrlRange, config}) {
  const translation = limitNorm(target.map((value, axis) => value - current[axis]), config.translation_error_cap_m);
  const rotation = limitNorm(rotationWorld, config.rotation_error_cap_rad);
  const weight = config.orientation_weight;
  const jacobian = Array.from({length: 6}, (_, row) => dofIds.map(dof => (
    row < 3 ? jacp[row * nv + dof] : weight * jacr[(row - 3) * nv + dof]
  )));
  const error = [...translation, ...rotation.map(value => value * weight)];
  const normal = jacobian.map((row, i) => jacobian.map((other, j) => (
    row.reduce((sum, value, index) => sum + value * other[index], 0)
      + (i === j ? config.damping ** 2 : 0)
  )));
  const solution = solveLinear(normal, error);
  return dofIds.map((_, joint) => {
    const delta = jacobian.reduce((sum, row, index) => sum + row[joint] * solution[index], 0);
    const value = clamp(qpos[qposIds[joint]] + clamp(delta, -config.max_joint_delta_rad, config.max_joint_delta_rad),
      ctrlRange[joint][0] + config.joint_margin_rad, ctrlRange[joint][1] - config.joint_margin_rad);
    if (!Number.isFinite(value)) throw new Error('Controller produced a non-finite target');
    return value;
  });
}

export class RollingSamples {
  constructor(capacity = 2048) { this.capacity = capacity; this.values = []; this.index = 0; }
  add(value) {
    if (!Number.isFinite(value)) return;
    if (this.values.length < this.capacity) this.values.push(value);
    else { this.values[this.index] = value; this.index = (this.index + 1) % this.capacity; }
  }
  quantile(fraction) {
    if (!this.values.length) return 0;
    const ordered = this.values.slice().sort((a, b) => a - b);
    return ordered[Math.min(ordered.length - 1, Math.floor(fraction * (ordered.length - 1)))];
  }
}
