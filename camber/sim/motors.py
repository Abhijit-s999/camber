"""Motor and propeller dynamics between a command and a thrust.

A command is a normalised speed request in [0, 1]. The rotor does not jump to
it: a brushless motor and propeller have rotational inertia, so speed follows
the command with a first-order lag, and thrust follows speed *squared*. Both
matter to a controller. The lag is the reason a rate loop cannot be tuned
arbitrarily hot, and the square is why "half throttle" is a quarter of full
thrust -- which a mixer that treats command as thrust gets wrong by a factor of
four at the bottom of the range.

The spinning mass matters twice more, and both are reported here for the
simulator to apply to the airframe:

* **Reaction to spinning up.** Changing a rotor's speed takes torque, and the
  airframe gets the opposite of it: J dw/dt about the rotor axis. On a quad
  commanded to yaw, every rotor changes speed at once and this term arrives
  *before* the aerodynamic one, because drag torque needs the new speed to
  exist first. It is what makes a snap yaw input bite immediately.
* **Gyroscopic precession.** A spinning rotor resists having its axis turned:
  pitch a quad quickly and its rotors push it in roll. Racers feel this as
  coupling between axes, and on a vehicle whose rotors do not cancel -- three
  rotors, five, or a tilting VTOL -- it does not cancel either.

Rotor speed in rad/s needs a maximum to scale to. Rather than invent one, it
comes from momentum theory: static thrust T = C_T rho n^2 D^4 with n in
revolutions per second, so n_max = sqrt(T / (C_T rho D^4)). C_T near 0.1 is
typical of small propellers; it is a parameter here, not a constant buried in
the code.
"""

from __future__ import annotations

import numpy as np


RHO = 1.225                    # kg/m^3, sea level
#: Static thrust coefficient per blade, T = C_T rho n^2 D^4. Two-bladed small
#: propellers measure about 0.11 and three-bladed racing props about 0.165, so
#: roughly 0.055 a blade. It is used only to turn a rotor's rated thrust into
#: the speed it spins at, which the momentum terms need; thrust itself always
#: comes from the catalogue or the blade-element tier, never from here.
C_T_PER_BLADE = 0.055


class TiltBank:
    """Servo dynamics for independently tilting rotors.

    A vectoring command is an angle, not an instantaneous geometry edit.  The
    exact first-order update matches the motor model and clamps at physical
    hard stops.  An empty bank keeps fixed-rotor aircraft on the same path.
    """

    def __init__(self, lower=(), upper=(), tau=()) -> None:
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        if not (self.lower.shape == self.upper.shape == self.tau.shape):
            raise ValueError("tilt servo arrays must have the same length")
        if (self.lower > self.upper).any() or (self.tau <= 0).any():
            raise ValueError("invalid tilt servo limits or time constant")
        self.angle = np.zeros_like(self.lower)

    def reset(self, angle=0.0) -> None:
        self.angle = np.clip(np.broadcast_to(np.asarray(angle, float), self.lower.shape),
                             self.lower, self.upper).copy()

    def step(self, target, dt: float) -> np.ndarray:
        if not len(self.angle):
            return self.angle
        target = np.clip(np.broadcast_to(np.asarray(target, float), self.angle.shape),
                         self.lower, self.upper)
        alpha = 1.0 - np.exp(-float(dt) / self.tau)
        self.angle += alpha * (target - self.angle)
        return self.angle


class MotorBank:
    """N rotors, each spinning up and down with time constant `tau`.

    `inertia` is each rotor's polar moment about its own axis, kg m^2, and
    `max_rate` its angular speed at full command, rad/s. Given both, the bank
    also reports the angular momentum it is carrying and the rate of change of
    it, which is what the airframe feels.
    """

    def __init__(self, max_thrust, tau: float = 0.03, inertia=None,
                 max_rate=None, spin=None) -> None:
        self.max_thrust = np.asarray(max_thrust, dtype=float)
        if (self.max_thrust <= 0).any():
            raise ValueError("every rotor needs a positive maximum thrust")
        if tau <= 0:
            raise ValueError("motor time constant must be positive")
        self.tau = float(tau)
        n = self.max_thrust.shape
        self.inertia = np.zeros(n) if inertia is None else np.broadcast_to(
            np.asarray(inertia, float), n).copy()
        self.max_rate = np.zeros(n) if max_rate is None else np.broadcast_to(
            np.asarray(max_rate, float), n).copy()
        self.spin = np.ones(n) if spin is None else np.broadcast_to(
            np.asarray(spin, float), n).copy()
        self.speed = np.zeros_like(self.max_thrust)      # normalised, [0, 1]
        self.rate_change = np.zeros_like(self.max_thrust)   # rad/s^2, signed

    @staticmethod
    def rate_at_full(thrust, radius, blades=2, rho: float = RHO):
        """Angular speed at full throttle, rad/s, from static thrust.

        Checked against what these actually turn at: a 5-inch three-blade
        racing propeller rated 12.3 N comes out at 29,000 rpm here, against
        the high twenties measured on a bench. With a single blade coefficient
        it read 37,000, which no 5-inch prop survives.
        """
        d = 2.0 * np.asarray(radius, float)
        c_t = C_T_PER_BLADE * np.asarray(blades, float)
        n = np.sqrt(np.asarray(thrust, float) / (c_t * rho * d ** 4))
        return 2.0 * np.pi * n

    @staticmethod
    def disc_inertia(mass, radius):
        """A propeller as a slender rod of length 2R: J = m (2R)^2 / 12."""
        return np.asarray(mass, float) * (2.0 * np.asarray(radius, float)) ** 2 / 12.0

    @property
    def rate(self) -> np.ndarray:
        """Signed angular speed of each rotor about its thrust axis, rad/s."""
        return self.spin * self.max_rate * self.speed

    @property
    def angular_momentum(self) -> np.ndarray:
        """J w per rotor, about its own axis, signed by rotation sense."""
        return self.inertia * self.rate

    @property
    def momentum_change(self) -> np.ndarray:
        """J dw/dt per rotor: the torque the airframe is paying to spin it."""
        return self.inertia * self.rate_change

    def reset(self, speed: float | np.ndarray = 0.0) -> None:
        self.speed = np.broadcast_to(np.asarray(speed, float),
                                     self.max_thrust.shape).copy()
        self.rate_change = np.zeros_like(self.max_thrust)

    def step(self, command: np.ndarray, dt: float) -> np.ndarray:
        """Advance by `dt` towards `command`; return thrust per rotor, N.

        The exact discrete solution of the first-order lag rather than an
        Euler step, so the response is right for any dt, including one larger
        than tau -- where Euler overshoots and oscillates.
        """
        u = np.clip(np.asarray(command, dtype=float), 0.0, 1.0)
        alpha = 1.0 - np.exp(-dt / self.tau)
        before = self.speed.copy()
        self.speed += alpha * (u - self.speed)
        self.rate_change = self.spin * self.max_rate * (self.speed - before) / dt
        return self.max_thrust * self.speed ** 2

    @staticmethod
    def command_for(thrust: np.ndarray, max_thrust: np.ndarray) -> np.ndarray:
        """The steady-state command that produces a thrust: the inverse square."""
        return np.sqrt(np.clip(np.asarray(thrust) / max_thrust, 0.0, 1.0))
