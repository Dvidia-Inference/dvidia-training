"""Small explicit controllers; parameters can be tuned in local simulation."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class FeedbackPolicy:
    force_limit: float = 2.0
    kp: float = 10.0
    kd: float = 0.25
    gravity_bias: float = 0.1

    def __post_init__(self):
        values = (self.force_limit, self.kp, self.kd, self.gravity_bias)
        if not all(np.isfinite(values)) or self.force_limit <= 0 or min(values[1:]) < 0:
            raise ValueError("Policy parameters must be finite and nonnegative; force limit positive.")

    def __call__(self, observation):
        delta = np.asarray(observation['target']) - np.asarray(observation['endpoint_position'])
        force = self.kp * delta - self.kd * np.asarray(observation['endpoint_velocity'])
        force[2] += self.gravity_bias
        magnitude = np.linalg.norm(force)
        if magnitude > self.force_limit:
            force *= self.force_limit / magnitude
        return force.tolist()

    def parameters(self):
        return asdict(self)


class ZeroPolicy:
    def __call__(self, observation):
        return [0.0, 0.0, 0.0]


class ReleasePolicy:
    """Apply feedback briefly, then remove the ideal attachment's applied force."""
    def __init__(self, feedback, release_step=10):
        self.feedback, self.release_step, self.steps = feedback, release_step, 0

    def __call__(self, observation):
        self.steps += 1
        return self.feedback(observation) if self.steps <= self.release_step else [0.0, 0.0, 0.0]
