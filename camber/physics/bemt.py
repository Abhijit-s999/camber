"""Blade element momentum theory for propellers.

L1 in the fidelity ladder (ARCH-01): milliseconds per evaluation, thousands of
evaluations per optimiser pass, and validated against the UIUC wind-tunnel
database rather than against itself.

The method divides the blade into radial elements, computes the local inflow
from momentum conservation and the local force from two-dimensional section
data, and requires the two to agree. The unknown at each station is the inflow
angle phi; a and b, the axial and tangential induction factors, follow from it.

**Why a bracketed root-find rather than fixed-point iteration.** The obvious
approach -- guess a and b, recompute, repeat -- diverges for heavily loaded
stations, which is exactly where a propeller operates at low advance ratio.
Reformulating as a sign-changing residual in phi and bracketing it converges
every time or reports honestly that it could not.

**Coefficient convention** is the propeller one throughout, n in rev/s:

    J = V / (n D)    CT = T / (rho n^2 D^4)    CP = P / (rho n^3 D^5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .airfoil import TabulatedPolar, build_polar

#: Sea level, 15 C, ISA.
RHO_AIR = 1.225
NU_AIR = 1.4607e-5

#: Ceiling on the axial induction factor, purely to keep the root-find bounded.
#:
#: This must be large, and the reason is the parameterisation. `a` is defined
#: against the *freestream* -- induced velocity is a * V_inf -- which suits a
#: wind turbine, where the freestream dominates. A propeller at low advance
#: ratio makes its own flow: a 10 inch propeller at 5000 rpm induces about
#: 9 m/s while flying at 2.4 m/s, which is a = 3.75. An earlier ceiling of 1.5
#: silently truncated that and under-predicted thrust by 20-70 per cent, worst
#: at low J. The value below is a numerical guard, not a physical limit.
A_MAX = 40.0


@dataclass
class BladeDefinition:
    """Radial distributions, non-dimensional in r/R."""

    radius_m: float
    blades: int
    r_over_R: np.ndarray
    chord_over_R: np.ndarray
    twist_rad: np.ndarray

    def __post_init__(self) -> None:
        for name in ("r_over_R", "chord_over_R", "twist_rad"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        if not (len(self.r_over_R) == len(self.chord_over_R) == len(self.twist_rad)):
            raise ValueError("blade station arrays must be the same length")
        if self.blades < 2:
            raise ValueError("a propeller needs at least 2 blades")

    @classmethod
    def from_uiuc(cls, geometry, blades: int = 2) -> "BladeDefinition":
        return cls(
            radius_m=geometry.radius_m,
            blades=blades,
            r_over_R=np.array([s.r_over_R for s in geometry.stations]),
            chord_over_R=np.array([s.chord_over_R for s in geometry.stations]),
            twist_rad=np.array([s.twist_rad for s in geometry.stations]),
        )

    def resample(self, n: int = 40) -> "BladeDefinition":
        """Cosine-spaced stations, clustered where the loading changes fastest.

        Uniform spacing wastes points inboard, where little thrust is produced,
        and under-resolves the tip, where the loading falls to zero over a few
        percent of span and most of the induced loss is generated.
        """
        lo, hi = float(self.r_over_R[0]), float(self.r_over_R[-1])
        beta = np.linspace(0.0, math.pi, n + 2)[1:-1]
        x = lo + (hi - lo) * (1 - np.cos(beta)) / 2
        return BladeDefinition(
            self.radius_m, self.blades, x,
            np.interp(x, self.r_over_R, self.chord_over_R),
            np.interp(x, self.r_over_R, self.twist_rad),
        )


@dataclass
class StationResult:
    r_over_R: float
    phi: float
    alpha: float
    a: float
    b: float
    cl: float
    cd: float
    tip_loss: float
    dT_dr: float
    dQ_dr: float
    converged: bool


@dataclass
class PropellerResult:
    thrust_n: float
    torque_nm: float
    power_w: float
    advance_ratio: float
    ct: float
    cp: float
    efficiency: float
    stations: list[StationResult]

    @property
    def converged(self) -> bool:
        return all(s.converged for s in self.stations)

    def __repr__(self) -> str:
        return (f"PropellerResult(J={self.advance_ratio:.3f}, "
                f"CT={self.ct:.4f}, CP={self.cp:.4f}, eta={self.efficiency:.3f}, "
                f"T={self.thrust_n:.2f} N, P={self.power_w:.1f} W)")


def prandtl_tip_loss(phi: float, r_over_R: float, blades: int,
                     hub_over_R: float = 0.15) -> float:
    """Prandtl's correction for a finite number of blades.

    Momentum theory assumes an actuator disc -- infinitely many blades. A real
    propeller sheds a tip vortex from each blade, unloading the tip. Without
    this factor BEMT over-predicts thrust near the tip, and the error grows as
    blade count falls, so a two-blade propeller is where it matters most.
    """
    s = abs(math.sin(phi))
    if s < 1e-6 or r_over_R >= 1.0:
        return 1e-4

    f_tip = blades / 2.0 * (1.0 - r_over_R) / (r_over_R * s)
    tip = 2.0 / math.pi * math.acos(min(1.0, math.exp(-f_tip)))

    hub = 1.0
    if r_over_R > hub_over_R:
        f_hub = blades / 2.0 * (r_over_R - hub_over_R) / (r_over_R * s)
        hub = 2.0 / math.pi * math.acos(min(1.0, math.exp(-f_hub)))

    return max(tip * hub, 1e-4)


def _induction(phi, twist, sigma, r_over_R, blades, reynolds, polar):
    """Induction factors implied by the blade element at this phi."""
    alpha = twist - phi
    cl, cd = polar(alpha, reynolds)
    cl, cd = float(cl), float(cd)

    cn = cl * math.cos(phi) - cd * math.sin(phi)      # along the axis: thrust
    ct = cl * math.sin(phi) + cd * math.cos(phi)      # tangential: torque

    f = prandtl_tip_loss(phi, r_over_R, blades)
    s, c = math.sin(phi), math.cos(phi)

    den_a = 4.0 * f * s * s - sigma * cn
    # A non-positive denominator is the turbulent-wake state: momentum theory
    # has stopped being valid there. Flagged via the bracket rather than
    # silently trusted.
    a = sigma * cn / den_a if den_a > 1e-9 else A_MAX

    den_b = 4.0 * f * s * c + sigma * ct
    b = sigma * ct / den_b if abs(den_b) > 1e-9 else 0.0

    return (min(max(a, -0.5), A_MAX), min(max(b, -0.5), 0.9), cl, cd, f, alpha)


def solve_station(r_over_R, chord_over_R, twist, radius, blades, v_inf, omega,
                  polar, rho=RHO_AIR, nu=NU_AIR):
    r = r_over_R * radius
    chord = chord_over_R * radius
    sigma = blades * chord / (2.0 * math.pi * r)
    u_t = omega * r

    # Reynolds from the undisturbed relative speed. Induction changes it by a
    # few per cent, and section data is weakly sensitive to Re, so one pass is
    # enough -- iterating it buys nothing measurable.
    reynolds = max(math.hypot(v_inf, u_t) * chord / nu, 1e3)

    def residual(phi: float) -> float:
        a, b, *_ = _induction(phi, twist, sigma, r_over_R, blades, reynolds, polar)
        return math.sin(phi) * u_t * (1.0 - b) - math.cos(phi) * v_inf * (1.0 + a)

    lo, hi = 1e-6, math.pi / 2.0 - 1e-6
    converged = True
    try:
        if residual(lo) * residual(hi) > 0.0:
            # No sign change on the primary bracket: scan for one before giving
            # up, because a heavily loaded station can push the root outside.
            grid = np.linspace(lo, hi, 60)
            vals = [residual(p) for p in grid]
            found = None
            for i in range(len(grid) - 1):
                if vals[i] * vals[i + 1] <= 0.0:
                    found = (grid[i], grid[i + 1])
                    break
            if found is None:
                phi = math.atan2(v_inf, u_t)
                converged = False
            else:
                phi = brentq(residual, *found, xtol=1e-10)
        else:
            phi = brentq(residual, lo, hi, xtol=1e-10)
    except (ValueError, RuntimeError):
        phi = math.atan2(v_inf, u_t)
        converged = False

    a, b, cl, cd, f, alpha = _induction(
        phi, twist, sigma, r_over_R, blades, reynolds, polar)

    va = v_inf * (1.0 + a)
    vt = u_t * (1.0 - b)
    w2 = va * va + vt * vt

    cn = cl * math.cos(phi) - cd * math.sin(phi)
    ct = cl * math.sin(phi) + cd * math.cos(phi)

    dT_dr = 0.5 * rho * blades * chord * w2 * cn
    dQ_dr = 0.5 * rho * blades * chord * w2 * ct * r

    return StationResult(r_over_R, phi, alpha, a, b, cl, cd, f,
                         dT_dr, dQ_dr, converged)


def analyse(blade: BladeDefinition, v_inf: float, rpm: float,
            polar: TabulatedPolar | None = None,
            rho: float = RHO_AIR, nu: float = NU_AIR) -> PropellerResult:
    """Thrust, torque and the usual coefficients at one operating point."""
    if rpm <= 0.0:
        raise ValueError("rpm must be positive")
    polar = polar if polar is not None else build_polar()

    omega = rpm * 2.0 * math.pi / 60.0
    n = rpm / 60.0
    diameter = 2.0 * blade.radius_m

    stations = [
        solve_station(float(x), float(c), float(t), blade.radius_m, blade.blades,
                      v_inf, omega, polar, rho, nu)
        for x, c, t in zip(blade.r_over_R, blade.chord_over_R, blade.twist_rad)
    ]

    r = blade.r_over_R * blade.radius_m
    thrust = float(np.trapezoid([s.dT_dr for s in stations], r))
    torque = float(np.trapezoid([s.dQ_dr for s in stations], r))
    power = torque * omega

    ct = thrust / (rho * n ** 2 * diameter ** 4)
    cp = power / (rho * n ** 3 * diameter ** 5)
    j = v_inf / (n * diameter)
    eta = j * ct / cp if cp > 1e-12 else 0.0

    return PropellerResult(thrust, torque, power, j, ct, cp, eta, stations)
