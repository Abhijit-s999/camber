"""Control allocation: can this arrangement of thrusters actually fly?

docs/DECISIONS.md ARCH-03. An optimiser that is not told about controllability
will happily converge on an aircraft that is aerodynamically superb and
dynamically impossible -- all four rotors clustered near the CG, say, which is
light and efficient and cannot produce a roll moment.

The allocation matrix A maps per-thruster commands u (in newtons) to the wrench
the airframe feels:

    [F; M] = A u,     A is 6 x N

Each column is one thruster's contribution: force along its thrust direction,
moment from its offset arm about the CG, plus for a rotor the reaction torque
of spinning it. Everything is expressed about the *current CG*, which is why
this depends on the mass properties and must be recomputed whenever a payload
moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec.units import Vec3
from .spec.vehicle import Rotor, Thruster, Vehicle

#: Rows of the wrench vector.
AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
G = 9.80665


def thruster_column(t: Thruster, cg: Vec3) -> np.ndarray:
    """One thruster's wrench per newton of thrust, about `cg`."""
    d = np.array(t.direction().as_list())
    arm = np.array((t.position - cg).as_list())
    force = d
    moment = np.cross(arm, d)

    if isinstance(t, Rotor):
        # Reaction torque. The motor drives the rotor against aerodynamic drag;
        # the airframe receives the equal and opposite couple, so it acts
        # *against* the rotor's rotation sense. This is the term that lets a
        # multirotor yaw at all, and the sign is why rotors alternate CW/CCW:
        # matched pairs cancel in hover and differ to command yaw.
        moment = moment - t.spin.value * t.torque_coefficient * d

    return np.concatenate([force, moment])


def allocation_matrix(vehicle: Vehicle) -> np.ndarray:
    """The 6 x N matrix mapping thrust commands to force and moment."""
    if not vehicle.thrusters:
        raise ValueError(f"{vehicle.name!r} has no thrusters")
    cg = vehicle.mass_properties().cg
    return np.column_stack([thruster_column(t, cg) for t in vehicle.thrusters])


@dataclass
class Controllability:
    """What the airframe can and cannot command."""

    rank: int
    singular_values: np.ndarray
    controllable_axes: tuple[str, ...]
    uncontrollable_axes: tuple[str, ...]
    #: Smallest non-negligible singular value: the weakest commandable
    #: direction, and the one that limits authority.
    margin: float

    @property
    def is_fully_controllable(self) -> bool:
        return not self.uncontrollable_axes

    def __repr__(self) -> str:
        state = "full 6-DOF" if self.is_fully_controllable else (
            f"cannot command {', '.join(self.uncontrollable_axes)}")
        return (f"Controllability(rank={self.rank}/6, {state}, "
                f"margin={self.margin:.4g})")


def controllability(vehicle: Vehicle, tol: float = 1e-6) -> Controllability:
    """Which wrench directions the thrusters can produce.

    Uses the SVD of A rather than its rank alone: rank is a yes/no answer that
    flips discontinuously, which an optimiser cannot follow. The smallest
    singular value is a continuous measure of how nearly an axis is lost, so it
    can be used as a penalty that pushes designs away from the cliff before
    they fall off it.
    """
    a = allocation_matrix(vehicle)
    u, s, _ = np.linalg.svd(a, full_matrices=True)

    rank = int(np.sum(s > tol))
    padded = np.zeros(6)
    padded[: len(s)] = s

    # An axis is independently commandable only if its unit wrench lies *in*
    # the range of A -- not merely if it has some projection onto it. The
    # difference matters: canting a quadrotor's rotors gives every axis a
    # non-zero projection while the range stays four-dimensional, so lateral
    # force remains unavailable without also disturbing something else. Testing
    # projection alone would report such a design as fully controllable.
    span = u[:, :rank] if rank else np.zeros((6, 0))
    controllable, uncontrollable = [], []
    for i, axis in enumerate(AXES):
        e = np.zeros(6)
        e[i] = 1.0
        outside = float(np.linalg.norm(e - span @ (span.T @ e))) if rank else 1.0
        (controllable if outside < 1e-6 else uncontrollable).append(axis)

    return Controllability(
        rank=rank,
        singular_values=padded,
        controllable_axes=tuple(controllable),
        uncontrollable_axes=tuple(uncontrollable),
        margin=float(padded[:6][padded[:6] > tol].min()) if rank else 0.0,
    )


@dataclass
class HoverSolution:
    feasible: bool
    thrusts: np.ndarray | None
    weight_n: float
    total_thrust_n: float
    #: Fraction of each thruster's ceiling used. >1 anywhere means infeasible.
    utilisation: np.ndarray | None
    reason: str = ""


def solve_hover(vehicle: Vehicle) -> HoverSolution:
    """Find per-thruster commands that hold a level hover, if any exist.

    Solves A u = [0, 0, -W, 0, 0, 0] in the least-squares sense, then checks
    the residual and the bounds. Thrusts must be non-negative -- a propeller
    cannot pull -- and the least-squares solution ignoring that is a common way
    to conclude a design hovers when it does not.
    """
    mp = vehicle.mass_properties()
    weight = mp.mass * G
    a = allocation_matrix(vehicle)

    target = np.zeros(6)
    target[2] = -weight  # +z is down, so lift is negative

    u, *_ = np.linalg.lstsq(a, target, rcond=None)
    residual = float(np.linalg.norm(a @ u - target))

    if residual > 1e-6 * max(1.0, weight):
        return HoverSolution(False, u, weight, float(np.sum(u)), None,
                             f"no exact trim: residual {residual:.4g} N")
    if np.any(u < -1e-9):
        worst = vehicle.thrusters[int(np.argmin(u))].name
        return HoverSolution(False, u, weight, float(np.sum(u)), None,
                             f"trim needs negative thrust on {worst!r}; a "
                             f"propeller cannot pull")

    ceilings = []
    for t in vehicle.thrusters:
        try:
            ceilings.append(t.max_thrust())
        except ValueError:
            ceilings.append(np.inf)  # not yet selected; bounds unknown
    ceil = np.array(ceilings, dtype=float)
    util = np.where(np.isfinite(ceil), u / np.where(ceil == 0, np.nan, ceil), np.nan)

    over = np.nanmax(util) if np.any(np.isfinite(util)) else 0.0
    if np.isfinite(over) and over > 1.0:
        worst = vehicle.thrusters[int(np.nanargmax(util))].name
        return HoverSolution(False, u, weight, float(np.sum(u)), util,
                             f"{worst!r} needs {over:.0%} of its rated thrust")

    return HoverSolution(True, u, weight, float(np.sum(u)), util, "")
