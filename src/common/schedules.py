"""Exploration schedules for DQN training."""


class LinearEpsilonSchedule:
    """Linear epsilon decay from eps_start to eps_end over decay_steps."""

    def __init__(self, eps_start: float = 1.0, eps_end: float = 0.05, decay_steps: int = 20000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = decay_steps

    def value(self, step: int) -> float:
        """Return epsilon at given step."""
        fraction = min(step / self.decay_steps, 1.0)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)
