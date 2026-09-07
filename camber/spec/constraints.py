"""Placement constraints, expressed as data rather than code.

ARCH-02. The optimiser reads these to know what it may move, the validator
rejects an over- or under-constrained spec before anything expensive runs, and
a future GUI draws them. Written as imperative checks instead, they would be
none of those things -- an optimiser cannot introspect an `if` statement to
learn that a payload has two free degrees of freedom.

Each constraint answers two questions: is a candidate layout valid
(`violation`, zero when satisfied), and which bodies does it involve.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .units import Vec3


class Constraint(ABC):
    """A restriction on where bodies may be placed."""

    @abstractmethod
    def bodies(self) -> tuple[str, ...]:
        """Names of the bodies this constrains."""

    @abstractmethod
    def violation(self, positions: dict[str, Vec3], vehicle=None) -> float:
        """0.0 when satisfied; a positive magnitude of breach otherwise.

        Returning a magnitude rather than a boolean is what lets the optimiser
        treat a constraint as a penalty and walk downhill towards feasibility,
        instead of seeing a flat wall it cannot get information from.
        """

    def satisfied(self, positions: dict[str, Vec3], vehicle=None,
                  tol: float = 1e-9) -> bool:
        return self.violation(positions, vehicle) <= tol


@dataclass(frozen=True)
class Fixed(Constraint):
    """This body sits exactly here. The optimiser may not move it."""

    body: str
    position: Vec3

    def bodies(self) -> tuple[str, ...]:
        return (self.body,)

    def violation(self, positions, vehicle=None) -> float:
        return positions[self.body].distance_to(self.position)


@dataclass(frozen=True)
class Relative(Constraint):
    """Body `a` sits at a fixed offset from body `b`.

    The pair may move together; their separation may not change. This is the
    "camera and electronics rigidly related, but free as a unit" case.
    """

    a: str
    b: str
    offset: Vec3

    def bodies(self) -> tuple[str, ...]:
        return (self.a, self.b)

    def violation(self, positions, vehicle=None) -> float:
        actual = positions[self.a] - positions[self.b]
        return actual.distance_to(self.offset)


@dataclass(frozen=True)
class MinDistance(Constraint):
    """Keep two bodies at least `distance` apart, centre to centre."""

    a: str
    b: str
    distance: float

    def bodies(self) -> tuple[str, ...]:
        return (self.a, self.b)

    def violation(self, positions, vehicle=None) -> float:
        gap = positions[self.a].distance_to(positions[self.b])
        return max(0.0, self.distance - gap)


@dataclass(frozen=True)
class CGBox(Constraint):
    """The assembly CG must lie inside this box, in body axes.

    The usual reason a multirotor is uncontrollable is not that a rotor is too
    weak but that the CG has drifted outside the region the mixer can trim.
    """

    lower: Vec3
    upper: Vec3

    def bodies(self) -> tuple[str, ...]:
        return ()

    def violation(self, positions, vehicle=None) -> float:
        if vehicle is None:
            raise ValueError("CGBox needs the vehicle to know where the CG is")
        cg = vehicle.mass_properties().cg
        breach = 0.0
        for axis in ("x", "y", "z"):
            lo, hi, v = (getattr(self.lower, axis), getattr(self.upper, axis),
                         getattr(cg, axis))
            breach += max(0.0, lo - v) ** 2 + max(0.0, v - hi) ** 2
        return math.sqrt(breach)


@dataclass(frozen=True)
class Symmetry(Constraint):
    """Mirror `a` and `b` about a body plane ("xz" is the usual one)."""

    a: str
    b: str
    plane: str = "xz"

    def bodies(self) -> tuple[str, ...]:
        return (self.a, self.b)

    def violation(self, positions, vehicle=None) -> float:
        flip = {"xz": (1.0, -1.0, 1.0), "yz": (-1.0, 1.0, 1.0),
                "xy": (1.0, 1.0, -1.0)}
        if self.plane not in flip:
            raise ValueError(f"unknown symmetry plane {self.plane!r}")
        sx, sy, sz = flip[self.plane]
        pa = positions[self.a]
        mirrored = Vec3(pa.x * sx, pa.y * sy, pa.z * sz)
        return mirrored.distance_to(positions[self.b])


@dataclass(frozen=True)
class ViewCone(Constraint):
    """A camera's field of view must not be blocked.

    Approximate by design: each obstruction is tested as its bounding sphere
    against the cone. Cheap enough for the inner loop and conservative -- it
    can report a blockage that exact geometry would clear, never the reverse.
    Exact occlusion is a raycast against the assembled mesh, which belongs at
    the CFD tier where a candidate is already worth that much effort.
    """

    camera: str
    half_angle: float
    obstructed_by: tuple[str, ...] = field(default_factory=tuple)

    def bodies(self) -> tuple[str, ...]:
        return (self.camera, *self.obstructed_by)

    def violation(self, positions, vehicle=None) -> float:
        if vehicle is None:
            raise ValueError("ViewCone needs the vehicle to size obstructions")
        cam = vehicle.part(self.camera)
        apex = positions[self.camera]
        # Boresight in body axes: the camera's own tilt matters. A 25-degree
        # upward tilt is often exactly what clears the propellers.
        axis = cam.boresight()

        worst = 0.0
        for name in self.obstructed_by:
            centre = positions[name]
            offset = centre - apex
            dist = offset.norm
            if dist == 0.0:
                return math.pi  # occupying the lens is maximally blocking
            radius = vehicle.part(name).bounding_radius
            along = offset.dot(axis)
            if along <= 0.0 and dist > radius:
                continue  # entirely behind the camera
            angle = math.acos(max(-1.0, min(1.0, along / dist)))
            angular_radius = math.asin(min(1.0, radius / dist))
            intrusion = self.half_angle - (angle - angular_radius)
            worst = max(worst, intrusion)
        return max(0.0, worst)
