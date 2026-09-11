"""Motor and propeller dynamics between a command and a thrust.

A command is a normalised speed request in [0, 1]. The rotor does not jump to
it: a brushless motor and propeller have rotational inertia, so speed follows
the command with a first-order lag, and thrust follows speed *squared*. Both
matter to a controller. The lag is the reason a rate loop cannot be tuned
arbitrarily hot, and the square is why "half throttle" is a quarter of full
thrust -- which a mixer that treats command as thrust gets wrong by a factor of
four at the bottom of the range.
"""

from __future__ import annotations

import numpy as np


class MotorBank:
    """N rotors, each spinning up and down with time constant `tau`."""

    def __init__(self, max_thrust, tau: float = 0.03) -> None:
        self.max_thrust = np.asarray(max_thrust, dtype=float)
        if (self.max_thrust <= 0).any():
            raise ValueError("every rotor needs a positive maximum thrust")
        if tau <= 0:
            raise ValueError("motor time constant must be positive")
        self.tau = float(tau)
        self.speed = np.zeros_like(self.max_thrust)      # normalised, [0, 1]

    def reset(self, speed: float | np.ndarray = 0.0) -> None:
        self.speed = np.broadcast_to(np.asarray(speed, float),
                                     self.max_thrust.shape).copy()

    def step(self, command: np.ndarray, dt: float) -> np.ndarray:
        """Advance by `dt` towards `command`; return thrust per rotor, N.

        The exact discrete solution of the first-order lag rather than an
        Euler step, so the response is right for any dt, including one larger
        than tau -- where Euler overshoots and oscillates.
        """
        u = np.clip(np.asarray(command, dtype=float), 0.0, 1.0)
        alpha = 1.0 - np.exp(-dt / self.tau)
        self.speed += alpha * (u - self.speed)
        return self.max_thrust * self.speed ** 2

    @staticmethod
    def command_for(thrust: np.ndarray, max_thrust: np.ndarray) -> np.ndarray:
        """The steady-state command that produces a thrust: the inverse square."""
        return np.sqrt(np.clip(np.asarray(thrust) / max_thrust, 0.0, 1.0))
