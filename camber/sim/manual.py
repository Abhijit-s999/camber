"""Flying it yourself: sticks in, motor commands out.

Three modes, the ones a real radio has:

  assist  sticks ask for a velocity, centre sticks hold position. Self
          levelling, and what a keyboard can actually fly.
  angle   sticks ask for a bank and pitch angle, throttle is collective.
          Self levelling: let go and it returns to level.
  acro    sticks ask for body *rates*. Nothing levels it. What FPV racers
          fly, and the only mode in which the vehicle can be flown inverted.

Stick conventions are the pilot's, not the maths': roll right is positive,
pitch forward (nose down) is positive, yaw right is positive, and throttle runs
0 to 1 with the middle meaning hover. That last one matters -- on a vehicle
with a thrust-to-weight of 11.7, hover sits at 8.5% of full throttle, so a
linear stick would spend its first tenth on everything below a hover and the
rest on rocketry. The curve here is two straight segments: 0 to mid gives
nothing up to a hover, mid to full gives hover up to everything the rotors
have. Any vehicle is then flyable from the same stick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flight import Cascade, quat_from_matrix

MODES = ("assist", "angle", "acro")


@dataclass
class Sticks:
    """Where the sticks are. Roll/pitch/yaw in [-1, 1], throttle in [0, 1]."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.5

    def clipped(self) -> "Sticks":
        c = lambda x: float(np.clip(x, -1.0, 1.0))                  # noqa: E731
        return Sticks(c(self.roll), c(self.pitch), c(self.yaw),
                      float(np.clip(self.throttle, 0.0, 1.0)))


def expo(x: float, amount: float = 0.3) -> float:
    """Soften the middle of a stick's travel, leaving its ends alone. Every
    radio does this; without it a keyboard's full deflection is unflyable."""
    return (1.0 - amount) * x + amount * x ** 3


class Manual(Cascade):
    """A pilot flown by hand. Set `sticks` and `mode`; `armed` gates the motors."""

    def __init__(self, mode: str = "assist", *, max_angle=np.radians(35),
                 max_stick_rate=np.radians(400), max_yaw_rate=np.radians(180),
                 max_speed: float = 12.0, max_climb: float = 3.0,
                 expo_amount: float = 0.3, **gains) -> None:
        super().__init__(**gains)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.max_angle, self.max_stick_rate = max_angle, max_stick_rate
        self.max_yaw_rate, self.max_speed, self.max_climb = (
            max_yaw_rate, max_speed, max_climb)
        self.expo_amount = expo_amount
        self.sticks = Sticks()
        self.armed = True
        self._tilt = self.max_tilt
        self._apply_mode()

    def _apply_mode(self) -> None:
        """Assist flies gently -- 35 degrees, like a camera drone's normal
        mode. The stick modes keep the full envelope, because in those the
        pilot is the one choosing the angle."""
        self.max_tilt = np.radians(35) if self.mode == "assist" else self._tilt

    def reset(self, sim) -> None:
        super().reset(sim)
        self.yaw = float(sim.state().euler[2])
        self.max_total = float(np.sum(sim.max_thrust))
        self._hold = self.target.copy()

    # --- sticks ---------------------------------------------------------------
    def collective(self, throttle: float) -> float:
        """Stick to thrust, hover at mid-stick. See the module docstring."""
        if throttle <= 0.5:
            return 2.0 * throttle * self.hover
        return self.hover + (throttle - 0.5) / 0.5 * (self.max_total - self.hover)

    def _shaped(self) -> tuple[float, float, float, float]:
        s = self.sticks.clipped()
        e = lambda x: expo(x, self.expo_amount)                     # noqa: E731
        return e(s.roll), e(s.pitch), e(s.yaw), s.throttle

    # --- the modes ------------------------------------------------------------
    def update(self, state, sim) -> np.ndarray:
        if not self.armed:
            self._hold = state.position.copy()
            self.yaw = float(state.euler[2])
            return np.zeros(len(state.thrust))
        roll, pitch, yaw, throttle = self._shaped()
        if self.mode == "assist":
            return self._assist(state, sim, roll, pitch, yaw, throttle)
        # Yaw is a rate stick in both stick modes; it winds the heading round.
        self.yaw = _wrap(self.yaw - yaw * self.max_yaw_rate * sim.dt)
        fz = self.collective(throttle)
        if self.mode == "angle":
            q_sp = _quat_rpy(roll * self.max_angle, pitch * self.max_angle, self.yaw)
            return self.attitude(state, q_sp, fz, sim.dt)
        rate_sp = self.max_stick_rate * np.array([roll, pitch, 0.0])
        rate_sp[2] = -yaw * self.max_yaw_rate
        return self.rates(state, rate_sp, fz, sim.dt)

    def _assist(self, state, sim, roll, pitch, yaw, throttle) -> np.ndarray:
        """Velocity from the sticks, position hold when they are centred."""
        self.yaw = _wrap(self.yaw - yaw * self.max_yaw_rate * sim.dt)
        body = np.array([pitch * self.max_speed,        # forward
                         -roll * self.max_speed,        # +y is left
                         (throttle - 0.5) * 2.0 * self.max_climb])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.velocity = np.array([c * body[0] - s * body[1],
                                  s * body[0] + c * body[1], body[2]])
        # Per axis: while a stick asks for motion, the position term must not
        # fight it, so the hold point rides along with the vehicle. It also
        # keeps riding while the vehicle is still carrying speed after the
        # stick centres -- latching it at the moment of release instead made
        # the vehicle brake from 12 m/s, overshoot by 11 m, and then fly
        # backwards to where the stick was let go. It holds where it comes to
        # rest, which is what letting go is supposed to mean.
        asked = np.abs(self.velocity) > 1e-3
        carrying = np.abs(state.velocity) > 0.3
        self._hold = np.where(asked | carrying, state.position, self._hold)
        self.target = self._hold
        return super().update(state, sim)


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _quat_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """FLU attitude from roll (right positive), pitch (nose down positive)
    and yaw (left positive), applied Z-Y-X."""
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch),
                              np.sin(pitch), np.cos(yaw), np.sin(yaw))
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return quat_from_matrix(Rz @ Ry @ Rx)
