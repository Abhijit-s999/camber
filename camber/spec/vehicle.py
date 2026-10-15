"""The vehicle: bodies that carry mass, thrusters that make force.

One object graph, from which the mass properties, the control allocation, the
CAD solid and the CFD mesh are all derived. See docs/DECISIONS.md ARCH-01.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .constraints import Constraint
from .rotation import Rotation
from .shapes import Box, PointMass, Shape
from .units import UP, Vec3


@dataclass(frozen=True)
class Placement:
    """Where a part sits and how it is turned, in body axes."""

    position: Vec3 = field(default_factory=Vec3)
    rotation: Rotation = field(default_factory=Rotation.identity)


@dataclass
class Body:
    """Anything with mass and extent: payload, electronics, structure."""

    name: str
    mass: float
    shape: Shape = field(default_factory=lambda: Box(0.05, 0.05, 0.05))
    placement: Placement = field(default_factory=Placement)

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError(f"body {self.name!r} has non-positive mass {self.mass}")

    @property
    def position(self) -> Vec3:
        return self.placement.position

    @property
    def bounding_radius(self) -> float:
        """Radius of the sphere enclosing this body, for cheap clearance tests."""
        return self.shape.half_extents().norm

    @property
    def density(self) -> float | None:
        v = self.shape.volume
        return None if v == 0.0 else self.mass / v


@dataclass
class Battery(Body):
    """A pack. Energy, and the current it can actually deliver.

    `max_c_rate` matters as much as capacity: a pack that stores plenty but
    cannot supply the hover current sags, and a design sized on energy alone
    will brown out on the first aggressive manoeuvre.
    """

    capacity_ah: float = 0.0
    nominal_voltage: float = 0.0
    max_c_rate: float = 0.0

    @property
    def energy_wh(self) -> float:
        return self.capacity_ah * self.nominal_voltage

    @property
    def max_current_a(self) -> float:
        return self.capacity_ah * self.max_c_rate


@dataclass
class Camera(Body):
    """A sensor with a direction it must be able to see along."""

    #: Boresight in the camera's own axes; rotated into body axes on demand.
    view_direction: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.0, 0.0))
    horizontal_fov: float = 0.0

    def boresight(self) -> Vec3:
        return self.placement.rotation.apply(self.view_direction).unit()


class Spin(Enum):
    """Rotation sense seen from above, which sets the yaw torque's sign."""

    CW = -1
    CCW = 1


@dataclass
class Thruster:
    """Base class: something that makes a force at a place, in a direction."""

    name: str
    mass: float
    placement: Placement = field(default_factory=Placement)
    #: Thrust direction in the thruster's own axes. Default is a rotor lifting.
    thrust_axis: Vec3 = field(default_factory=lambda: UP)

    @property
    def position(self) -> Vec3:
        return self.placement.position

    def direction(self) -> Vec3:
        """Unit thrust direction in body axes."""
        return self.placement.rotation.apply(self.thrust_axis).unit()

    def max_thrust(self) -> float:
        raise NotImplementedError

    def shape(self) -> Shape:
        return PointMass()

    @property
    def bounding_radius(self) -> float:
        return self.shape().half_extents().norm


@dataclass
class Rotor(Thruster):
    """An electric motor and propeller.

    Thrust and torque are *not* computed here. This object records geometry and
    the selected hardware; the numbers come from the physics tier, which is
    validated against UIUC wind-tunnel data. A `max_thrust` guess baked into the
    spec is exactly the kind of invented constant AI-05 forbids.
    """

    radius: float = 0.0
    pitch: float = 0.0
    blades: int = 2
    ducted: bool = False
    spin: Spin = Spin.CCW
    #: Set once a catalogue part is chosen; None while still a free variable.
    motor_sku: str | None = None
    prop_sku: str | None = None
    #: Measured static thrust, from the catalogue. None until selection.
    static_thrust_n: float | None = None
    #: Torque-to-thrust ratio, metres. Sets yaw authority.
    torque_coefficient: float = 0.016

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"rotor {self.name!r} needs a positive radius")
        if self.blades < 2:
            raise ValueError(f"rotor {self.name!r} needs at least 2 blades")

    @property
    def disc_area(self) -> float:
        return np.pi * self.radius ** 2

    @property
    def bounding_radius(self) -> float:
        """The swept disc, not the hub. What a camera actually sees."""
        return self.radius

    def max_thrust(self) -> float:
        if self.static_thrust_n is None:
            raise ValueError(
                f"rotor {self.name!r} has no measured thrust yet -- select a "
                f"motor/prop from the catalogue, or run the physics tier. "
                f"Camber does not guess thrust from geometry."
            )
        return self.static_thrust_n


class Objective(Enum):
    """What the aircraft is for. Drives the optimiser's cost function."""

    MAX_ENDURANCE = "max_endurance"
    MAX_SPEED = "max_speed"


@dataclass
class Mission:
    objective: Objective = Objective.MAX_ENDURANCE
    #: Seconds, a floor the design must meet regardless of objective.
    min_endurance: float | None = None
    payload_mass: float = 0.0


@dataclass
class Frame:
    """Structure: arms, booms, plates. Bodies like any other, kept separate
    because the generative phase (P5) will synthesise these rather than read
    them from the spec."""

    members: list[Body] = field(default_factory=list)


@dataclass
class Vehicle:
    name: str
    bodies: list[Body] = field(default_factory=list)
    thrusters: list[Thruster] = field(default_factory=list)
    frame: Frame = field(default_factory=Frame)
    constraints: list[Constraint] = field(default_factory=list)
    mission: Mission = field(default_factory=Mission)

    # -- lookup ------------------------------------------------------------

    def all_bodies(self) -> list[Body]:
        return [*self.bodies, *self.frame.members]

    def body(self, name: str) -> Body:
        for b in self.all_bodies():
            if b.name == name:
                return b
        raise KeyError(
            f"no body named {name!r}; have "
            f"{sorted(b.name for b in self.all_bodies())}"
        )

    def thruster(self, name: str) -> Thruster:
        for t in self.thrusters:
            if t.name == name:
                return t
        raise KeyError(f"no thruster named {name!r}")

    def part(self, name: str):
        """Any named part: body, frame member or thruster.

        Constraints reason about both. A camera is most often obstructed by a
        propeller disc, which is a thruster, so a lookup that searched only
        bodies made the most important view-cone case unexpressible.
        """
        for candidate in (*self.all_bodies(), *self.thrusters):
            if candidate.name == name:
                return candidate
        raise KeyError(
            f"no part named {name!r}; have "
            f"{sorted([b.name for b in self.all_bodies()] + [t.name for t in self.thrusters])}"
        )

    def part_positions(self) -> dict[str, Vec3]:
        """Positions of every named part, which is what constraints index."""
        return {**self.positions(),
                **{t.name: t.position for t in self.thrusters}}

    def positions(self) -> dict[str, Vec3]:
        return {b.name: b.position for b in self.all_bodies()}

    # -- derived quantities -------------------------------------------------

    def mass_properties(self):
        """Total mass, CG and inertia tensor about the CG, in body axes."""
        from ..massprops import Part, combine

        parts: list[Part] = []
        for b in self.all_bodies():
            parts.append(Part(
                mass=b.mass,
                centroid=b.position,
                inertia_local=b.shape.inertia(b.mass),
                rotation=b.placement.rotation.matrix,
            ))
        for t in self.thrusters:
            parts.append(Part(
                mass=t.mass,
                centroid=t.position,
                inertia_local=t.shape().inertia(t.mass),
                rotation=t.placement.rotation.matrix,
            ))
        return combine(parts)

    @property
    def batteries(self) -> list[Battery]:
        return [b for b in self.all_bodies() if isinstance(b, Battery)]

    @property
    def cameras(self) -> list[Camera]:
        return [b for b in self.all_bodies() if isinstance(b, Camera)]

    @property
    def total_energy_wh(self) -> float:
        return sum(b.energy_wh for b in self.batteries)
