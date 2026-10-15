"""Which flight conditions to test a vehicle at, derived from the vehicle.

A fixed test matrix -- five speeds, five angles, for everything -- tests a
racing quad at conditions it never flies and misses the one it lives in. The
conditions worth measuring are the ones the vehicle's own physics puts it in,
and those follow from what it is and what it is for:

* A **multirotor** has no angle of attack of its own. It tilts its thrust
  vector to fly, and in steady level flight the tilt fixes the speed: thrust
  holds the weight up (T cos(tilt) = m g) and its horizontal component balances
  drag (T sin(tilt) = D(V)). So the test points are tilts, and each tilt comes
  with the speed at which the vehicle would actually be flying there. A racer
  (MAX_SPEED) is tested out to the tilt its thrust-to-weight allows, 60-odd
  degrees nose down; an endurance vehicle barely tilts at all.
* A **fixed-wing** vehicle flies at an angle of attack, at roughly its cruise
  speed, so it gets an alpha sweep through the linear range and past stall.

Angles use the aircraft convention throughout (see `camber.sim.aero`): a
multirotor flying forward is nose-down, air on its top, alpha negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..control import G
from ..spec.vehicle import Objective, Rotor

RHO = 1.225

#: How far each kind of multirotor mission tilts, degrees nose down.
TILT = {
    Objective.MAX_SPEED: (10.0, 65.0),
    Objective.MAX_ENDURANCE: (3.0, 22.0),
}
DEFAULT_TILT = (5.0, 35.0)


@dataclass(frozen=True)
class TestPoint:
    name: str
    speed: float           # m/s, air-relative
    alpha_deg: float       # aircraft convention
    thrust: float          # total rotor thrust for this condition, N (0 = none)
    why: str

    def relative_wind_ned(self) -> np.ndarray:
        """Air velocity past the vehicle, in NED body axes, for the CFD inlet.

        The vehicle moves at V(cos a, 0, -sin a) in FLU, which is
        V(cos a, 0, sin a) in NED; the air moves the opposite way. The old
        hover-only case wrote the freestream as (+V, 0, 0), which for a vehicle
        flying forward is air arriving from behind -- flying backwards. It never
        mattered at V = 0.
        """
        a = math.radians(self.alpha_deg)
        return -self.speed * np.array([math.cos(a), 0.0, math.sin(a)])


def kind_of(vehicle) -> str:
    return "multirotor" if any(isinstance(t, Rotor) for t in vehicle.thrusters) \
        else "fixed_wing"


def trim_speed(vehicle, tilt_deg: float, drag) -> float:
    """Steady level-flight speed at a tilt, from a body drag model.

    D(V) = 1/2 rho V^2 K(alpha) with K from the per-axis drag areas, and
    T sin(tilt) = D with T cos(tilt) = m g gives V in closed form.
    """
    a = math.radians(-tilt_deg)
    cx, _, cz = drag.cda
    K = cx * abs(math.cos(a)) ** 3 + cz * abs(math.sin(a)) ** 3
    mg = vehicle.mass_properties().mass * G
    return math.sqrt(2.0 * mg * math.tan(math.radians(tilt_deg)) / (RHO * K))


def envelope(vehicle, n: int = 6, *, drag=None, cruise: float | None = None
             ) -> list[TestPoint]:
    """The flight conditions to measure this vehicle at."""
    if kind_of(vehicle) == "fixed_wing":
        v = cruise or 20.0
        return [TestPoint(f"a{a:+03.0f}", v, float(a), 0.0,
                          "alpha sweep through the linear range and past stall")
                for a in np.linspace(-4.0, 16.0, max(n, 2))]

    from ..sim.aero import BodyDrag
    drag = drag or BodyDrag.estimate(vehicle)
    mg = vehicle.mass_properties().mass * G
    available = sum(t.max_thrust() for t in vehicle.thrusters)
    lo, hi = TILT.get(vehicle.mission.objective, DEFAULT_TILT)
    # The steepest tilt the vehicle can hold height at, with a 15% reserve for
    # control -- a rotor at 100% has nothing left to steer with.
    limit = math.degrees(math.acos(min(1.0, mg / (0.85 * available))))
    hi = min(hi, limit)
    points = []
    for tilt in np.linspace(lo, hi, n):
        thrust = mg / math.cos(math.radians(tilt))
        v = trim_speed(vehicle, tilt, drag)
        points.append(TestPoint(
            f"t{tilt:02.0f}", round(v, 2), -float(tilt), round(thrust, 4),
            f"level flight at {tilt:.0f} deg nose down, {v:.1f} m/s"))
    return points
