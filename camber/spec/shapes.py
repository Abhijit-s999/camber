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
