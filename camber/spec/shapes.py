"""Primitive solids with exact analytic mass properties.

Deliberately not backed by the CAD kernel. A box's inertia tensor is a closed
form known since Euler; computing it by tessellating a B-rep and summing
tetrahedra is slower, less accurate, and makes the whole product depend on a
200 MB optional dependency for arithmetic a schoolchild could do.

The CAD kernel is for what only it can do -- STEP output, booleans, fillets,
imported geometry. Everything here stays exact and dependency-light, which is
what lets the optimiser evaluate mass properties millions of times (AI-07).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .units import Vec3


class Shape(ABC):
    """A solid of uniform density, centred on its own centroid."""

    @property
    @abstractmethod
    def volume(self) -> float:
        """Volume in m^3."""

    @abstractmethod
    def inertia_unit_mass(self) -> np.ndarray:
        """Inertia tensor about the centroid, for unit mass, in local axes.

        Unit mass rather than actual mass because inertia is linear in mass:
        one formula serves whether the caller knows the mass or the density.
        """

    @abstractmethod
    def half_extents(self) -> Vec3:
        """Half-sizes of the axis-aligned bounding box in local axes."""

    def inertia(self, mass: float) -> np.ndarray:
        if mass < 0.0:
            raise ValueError(f"negative mass: {mass}")
        return mass * self.inertia_unit_mass()

    def planform_area(self) -> float:
        """Plan-view footprint in m^2, in the body's own axes.

        The default is the bounding rectangle, which is exact for a box and is
        the definition the interference model was trained on (CAL-02) -- so
        every existing shape reports exactly what it always did and the
        calibration stays valid. Shapes that are mostly air inside their bounds
        override it, because for them the rectangle is not a footprint at all:
        a topology-optimised X-frame has a 201 x 201 mm bounding box and is
        mostly empty space.
        """
        h = self.half_extents()
        return 4.0 * h.x * h.y


@dataclass(frozen=True)
class Box(Shape):
    """Rectangular cuboid, sized along local x, y, z."""

    lx: float
    ly: float
    lz: float

    def __post_init__(self) -> None:
        for name in ("lx", "ly", "lz"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"Box.{name} must be positive, got {getattr(self, name)}")

    @property
    def volume(self) -> float:
        return self.lx * self.ly * self.lz

    def inertia_unit_mass(self) -> np.ndarray:
        return np.diag([
            (self.ly ** 2 + self.lz ** 2) / 12.0,
            (self.lx ** 2 + self.lz ** 2) / 12.0,
            (self.lx ** 2 + self.ly ** 2) / 12.0,
        ])

    def half_extents(self) -> Vec3:
        return Vec3(self.lx / 2.0, self.ly / 2.0, self.lz / 2.0)


@dataclass(frozen=True)
class Cylinder(Shape):
    """Solid circular cylinder with its axis along local z."""

    radius: float
    length: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"Cylinder.radius must be positive, got {self.radius}")
        if self.length <= 0.0:
            raise ValueError(f"Cylinder.length must be positive, got {self.length}")

    @property
    def volume(self) -> float:
        return math.pi * self.radius ** 2 * self.length

    def inertia_unit_mass(self) -> np.ndarray:
        transverse = (3.0 * self.radius ** 2 + self.length ** 2) / 12.0
        return np.diag([transverse, transverse, self.radius ** 2 / 2.0])

    def half_extents(self) -> Vec3:
        return Vec3(self.radius, self.radius, self.length / 2.0)


@dataclass(frozen=True)
class Sphere(Shape):
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"Sphere.radius must be positive, got {self.radius}")

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius ** 3

    def inertia_unit_mass(self) -> np.ndarray:
        return np.eye(3) * (2.0 / 5.0 * self.radius ** 2)

    def half_extents(self) -> Vec3:
        return Vec3(self.radius, self.radius, self.radius)


@dataclass(frozen=True)
class PointMass(Shape):
    """A mass with no extent.

    Legitimate for wiring, adhesive, fasteners and the like, where the mass is
    real and the inertia contribution about its own centroid is negligible
    beside the parallel-axis term. Not legitimate for a battery.
    """

    @property
    def volume(self) -> float:
        return 0.0

    def inertia_unit_mass(self) -> np.ndarray:
        return np.zeros((3, 3))

    def half_extents(self) -> Vec3:
        return Vec3(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Tabulated(Shape):
    """A solid whose properties were measured once, from a mesh, and stored.

    GEOM-03 keeps mass properties analytic and dependency-free so the optimiser
    can evaluate them millions of times. A generated airframe has no closed
    form, but it does not need one: its volume, inertia tensor, extents and
    footprint are computed from the mesh when it is made and carried as
    numbers from then on. Evaluating it costs what a Box costs, and the core
    still imports nothing but numpy.

    `source` records which mesh the numbers came from. It is a pointer for a
    reader, not a dependency: the shape is complete without the file.
    """

    volume_m3: float
    #: Inertia about the centroid for unit mass, local axes, row-major 3 x 3.
    #: Not called `inertia`: Shape already has an `inertia(mass)` method, which
    #: a field of that name would shadow -- and dataclasses would then read the
    #: inherited method as the field's default.
    inertia_tensor: tuple
    extents: tuple                 # half-extents, m
    planform: float                # plan-view footprint, m^2
    source: str = ""

    def __post_init__(self) -> None:
        if self.volume_m3 <= 0.0:
            raise ValueError(f"Tabulated.volume_m3 must be positive, got {self.volume_m3}")
        I = np.asarray(self.inertia_tensor, dtype=float).reshape(3, 3)
        object.__setattr__(self, "inertia_tensor", tuple(map(tuple, I.tolist())))
        if not np.allclose(I, I.T, atol=1e-12 + 1e-9 * np.abs(I).max()):
            raise ValueError("Tabulated.inertia_tensor must be symmetric")
        if np.linalg.eigvalsh(I).min() < -1e-12:
            raise ValueError("Tabulated.inertia_tensor is not positive semi-definite")
        if self.planform < 0.0:
            raise ValueError("Tabulated.planform cannot be negative")
        object.__setattr__(self, "extents", tuple(float(e) for e in self.extents))

    @property
    def volume(self) -> float:
        return self.volume_m3

    def inertia_unit_mass(self) -> np.ndarray:
        return np.asarray(self.inertia_tensor, dtype=float)

    def half_extents(self) -> Vec3:
        return Vec3(*self.extents)

    def planform_area(self) -> float:
        return self.planform
